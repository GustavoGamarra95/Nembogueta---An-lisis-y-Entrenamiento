"""
Preprocesa un dataset de imágenes de alfabeto manual (una carpeta por letra)
a features de 120 dim por imagen, en el formato de librai_alphabet_v4.

La letra es el nombre de la carpeta a --label-depth niveles sobre la imagen
(1 = carpeta inmediata) y el sujeto la carpeta a --subject-depth (0 = sin
sujeto) o, con --subject-regex, el primer grupo del nombre de archivo.
Ejemplos (verificados):
    Massey ASL:      asl_dataset/<letra>/handN_*.jpeg
                     → --subject-regex '^(hand\\d+)_' --pad 0.6
    Spelling It Out: dataset5/<sujeto>/<letra>/color_*.png
                     → --subject-depth 2 --include '^color' --pad 0.6 --no-flip
--pad hace falta en recortes ajustados a la muñeca: sin margen el detector
de palma de MediaPipe falla (Massey 58% → 99.5%, Spelling It Out 45% → 97%).

Salida:
    <output-dir>/<LETRA>/<LETRA>_<sujeto>_<archivo>.npy  (120,) float32
    <output-dir>/index.csv, <output-dir>/stats.json

Usa la misma extracción que preprocess_lsc70.py (mano más alta y
--convention librai = espejar la imagen y slot como librai_alphabet_v4).
Verificar la geometría tras procesar: en una letra de palma al frente (B)
el pitch medio de la palma debe ser ≈ −10° como en LibrAI; si sale ≈ ±170°
el dataset ya venía espejado y hay que usar --no-flip.
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing.feature_engineering import (  # noqa: E402
    engineer_frame_features,
)
from preprocess_lsc70 import extract_raw_126  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
_hands = None
_flip = True
_pad = 0.0


def _init_worker(min_conf: float, flip: bool, pad: float):
    global _hands, _flip, _pad
    _hands = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=2,
        min_detection_confidence=min_conf,
    )
    _flip = flip
    _pad = pad


def _process(task):
    img_path, out_path = task
    img = cv2.imread(str(img_path))
    if img is None:
        return img_path, None
    if _flip:
        img = cv2.flip(img, 1)
    if _pad > 0:
        # Recortes ajustados: el detector de palma necesita contexto
        b = int(_pad * max(img.shape[:2]))
        img = cv2.copyMakeBorder(img, b, b, b, b, cv2.BORDER_REPLICATE)
    res = _hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks:
        return img_path, None
    feat = engineer_frame_features(extract_raw_126(res, "top", "librai"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, feat)
    return img_path, out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--label-depth", type=int, default=1)
    p.add_argument("--subject-depth", type=int, default=0)
    p.add_argument("--subject-regex", default=None,
                   help="Sujeto desde el nombre de archivo, primer grupo "
                        "(p. ej. '^(hand\\d+)_' en Massey)")
    p.add_argument("--include", default=".*",
                   help="Regex sobre el nombre de archivo (p. ej. '^color')")
    p.add_argument("--letters-only", action="store_true",
                   help="Descartar clases que no sean una sola letra")
    p.add_argument("--no-flip", action="store_true",
                   help="No espejar (dataset ya espejado, como LibrAI)")
    p.add_argument("--pad", type=float, default=0.0,
                   help="Margen añadido por lado, como fracción del lado "
                        "mayor (útil en recortes ajustados de la mano)")
    p.add_argument("--min-detection-confidence", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    inc = re.compile(args.include)
    tasks, meta = [], {}
    for img in sorted(args.input_dir.rglob("*")):
        if img.suffix.lower() not in IMG_EXT or not inc.search(img.name):
            continue
        label = img.parents[args.label_depth - 1].name.upper()
        if args.letters_only and not re.fullmatch(r"[A-ZÑÇ]", label):
            continue
        if args.subject_regex:
            m = re.search(args.subject_regex, img.name)
            subject = m.group(1) if m else "na"
        else:
            subject = (img.parents[args.subject_depth - 1].name
                       if args.subject_depth else "na")
        out = args.output_dir / label / f"{label}_{subject}_{img.stem}.npy"
        tasks.append((img, out))
        meta[img] = (label, subject)
    if not tasks:
        sys.exit(f"Sin imágenes en {args.input_dir}")
    print(f"{len(tasks)} imágenes, {len({m[0] for m in meta.values()})} clases")

    per_label = defaultdict(lambda: {"saved": 0, "skipped": 0})
    rows = []
    with Pool(args.workers, _init_worker,
              (args.min_detection_confidence, not args.no_flip,
               args.pad)) as pool:
        for i, (img, out) in enumerate(
            pool.imap_unordered(_process, tasks, chunksize=32), 1
        ):
            label, subject = meta[img]
            if out is None:
                per_label[label]["skipped"] += 1
            else:
                per_label[label]["saved"] += 1
                rows.append((out.relative_to(args.output_dir).as_posix(),
                             label, subject, img.name))
            if i % 5000 == 0:
                print(f"  {i}/{len(tasks)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "subject", "frame"])
        w.writerows(sorted(rows))
    saved = sum(v["saved"] for v in per_label.values())
    total = saved + sum(v["skipped"] for v in per_label.values())
    stats = {"source": str(args.input_dir), "flip": not args.no_flip,
             "pad": args.pad,
             "total_saved": saved, "total_skipped": total - saved,
             "detection_rate_pct": round(100 * saved / max(total, 1), 2),
             "per_letter": dict(sorted(per_label.items()))}
    with open(args.output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Guardados {saved}/{total} ({stats['detection_rate_pct']}%) "
          f"→ {args.output_dir}")


if __name__ == "__main__":
    main()

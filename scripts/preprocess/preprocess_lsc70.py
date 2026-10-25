"""
Preprocesa LSC70 (Lengua de Señas Colombiana) a features de 120 dim por frame,
en el mismo formato que data/processed/librai_alphabet_v4.

LSC70 trae 6 frames JPG por seña y sujeto (70 sujetos):
    <input-dir>/PerXX/<SEÑA>/PerXX_<SEÑA>_<k>.jpg

Salida (compatible con train_alphabet.py, la letra va antes del primer "_"):
    <output-dir>/<LETRA>/<LETRA>_PerXX_<k>.npy   (120,) float32
    <output-dir>/index.csv                      archivo, letra, sujeto, frame
    <output-dir>/stats.json                     tasa de detección por letra

Convención (--convention librai, por defecto): la misma geometría y slot
que data/processed/librai_alphabet_v4. Las fotos de LibrAI tienen geometría
espejada (como una cámara selfie, igual que evaluate_realworld.py con
cv2.flip) y su mano dominante quedó en el slot [63:126]. LSC70 está grabado
sin espejar, así que se espeja la imagen y se asigna el slot al revés de la
etiqueta de MediaPipe Hands. Sin esto cambian de signo la orientación de la
palma y los senos de los ángulos de los dedos.
Con --convention legacy: imagen sin espejar y slot por etiqueta
("Right" en [0:63]), como extract_raw_126 de evaluate_realworld.py.

Frames sin mano detectada se descartan.

LSC70 es de cuerpo completo: la mano que no deletrea queda en reposo a la
altura de la cintura y MediaPipe la detecta en ~60% de los frames. Por
defecto (--hand-select top) solo se conserva la mano más alta, que es la
que hace la letra, para que el vector sea comparable con LibrAI (una mano).

Uso:
    python scripts/preprocess/preprocess_lsc70.py \\
        --input-dir data/raw/LSC70/LSC70/LSC70AN \\
        --output-dir data/processed/lsc70_alphabet
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from preprocessing.feature_engineering import (  # noqa: E402
    engineer_frame_features,
)

# Nombres de carpeta de LSC70 que no son la letra tal cual
LABEL_MAP = {"NN": "Ñ"}
NUMBER_SIGNS = {"1", "4", "5", "6", "7", "8", "9", "10", "MIL", "MILLON"}

_hands = None
_hand_select = "top"
_convention = "librai"


def _init_worker(max_hands: int, min_conf: float, hand_select: str,
                 convention: str):
    global _hands, _hand_select, _convention
    _hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=max_hands,
        min_detection_confidence=min_conf,
    )
    _hand_select = hand_select
    _convention = convention


def select_hands(results, hand_select: str):
    """Pares (landmarks, handedness); con "top" solo la mano más alta."""
    pairs = list(zip(results.multi_hand_landmarks, results.multi_handedness))
    if hand_select == "top" and len(pairs) > 1:
        pairs = [min(pairs, key=lambda p: np.mean(
            [lm.y for lm in p[0].landmark]))]
    return pairs


def extract_raw_126(results, hand_select: str = "all",
                    convention: str = "legacy") -> np.ndarray:
    raw = np.zeros(126, dtype=np.float32)
    if not results.multi_hand_landmarks:
        return raw
    for hand_lm, handedness in select_hands(results, hand_select):
        label = handedness.classification[0].label
        offset = 0 if label == "Right" else 63
        if convention == "librai":
            offset = 63 - offset
        for i, lm in enumerate(hand_lm.landmark):
            raw[offset + i * 3 + 0] = lm.x
            raw[offset + i * 3 + 1] = lm.y
            raw[offset + i * 3 + 2] = lm.z
    return raw


def _process(task):
    img_path, out_path = task
    img = cv2.imread(str(img_path))
    if img is None:
        return img_path, None
    if _convention == "librai":
        img = cv2.flip(img, 1)
    results = _hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        return img_path, None
    feat = engineer_frame_features(
        extract_raw_126(results, _hand_select, _convention))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, feat)
    return img_path, out_path


def collect_tasks(input_dir: Path, output_dir: Path, include_numbers: bool):
    tasks, meta = [], {}
    for img in sorted(input_dir.glob("Per*/*/*.jpg")):
        subject, sign = img.parent.parent.name, img.parent.name
        if sign in NUMBER_SIGNS and not include_numbers:
            continue
        label = LABEL_MAP.get(sign, sign)
        frame = img.stem.rsplit("_", 1)[-1]
        out = output_dir / label / f"{label}_{subject}_{frame}.npy"
        tasks.append((img, out))
        meta[img] = (label, subject, frame)
    return tasks, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path,
                   default=Path("data/raw/LSC70/LSC70/LSC70AN"),
                   help="LSC70AN (cuerpo completo) o LSC70ANH (mano)")
    p.add_argument("--output-dir", type=Path,
                   default=Path("data/processed/lsc70_alphabet"))
    p.add_argument("--include-numbers", action="store_true",
                   help="Incluir 1, 4-10, MIL y MILLON además de las letras")
    p.add_argument("--max-hands", type=int, default=2)
    p.add_argument("--hand-select", choices=["top", "all"], default="top",
                   help="top: solo la mano más alta (la que deletrea); "
                        "all: todas las detectadas")
    p.add_argument("--convention", choices=["librai", "legacy"],
                   default="librai",
                   help="librai: geometría espejada y slot como "
                        "librai_alphabet_v4; legacy: sin espejar")
    p.add_argument("--min-detection-confidence", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    tasks, meta = collect_tasks(
        args.input_dir, args.output_dir, args.include_numbers
    )
    if not tasks:
        sys.exit(f"No se encontraron imágenes en {args.input_dir}")
    print(f"{len(tasks)} imágenes de {args.input_dir}")

    per_label = defaultdict(lambda: {"saved": 0, "skipped": 0})
    rows = []
    with Pool(args.workers, _init_worker,
              (args.max_hands, args.min_detection_confidence,
               args.hand_select, args.convention)) as pool:
        for i, (img, out) in enumerate(
            pool.imap_unordered(_process, tasks, chunksize=32), 1
        ):
            label, subject, frame = meta[img]
            if out is None:
                per_label[label]["skipped"] += 1
            else:
                per_label[label]["saved"] += 1
                rows.append((out.relative_to(args.output_dir).as_posix(),
                             label, subject, frame))
            if i % 2000 == 0:
                print(f"  {i}/{len(tasks)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "subject", "frame"])
        w.writerows(sorted(rows))

    saved = sum(v["saved"] for v in per_label.values())
    skipped = sum(v["skipped"] for v in per_label.values())
    stats = {
        "source": str(args.input_dir),
        "hand_select": args.hand_select,
        "convention": args.convention,
        "total_saved": saved,
        "total_skipped": skipped,
        "detection_rate_pct": round(100 * saved / max(saved + skipped, 1), 2),
        "per_letter": dict(sorted(per_label.items())),
    }
    with open(args.output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nGuardados: {saved}  descartados: {skipped}  "
          f"({stats['detection_rate_pct']}% detección)")
    for label, s in stats["per_letter"].items():
        print(f"  {label:7s} {s['saved']:5d} / {s['saved'] + s['skipped']}")
    print(f"Salida: {args.output_dir}")


if __name__ == "__main__":
    main()

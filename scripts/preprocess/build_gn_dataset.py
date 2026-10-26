"""
Arma el alfabeto guaraní (achegety, 33 letras) para Alicia (lengua "GN").

  - Letras compartidas con el alfabeto LSPy (A, E, G, H, …): los mismos
    datos que LSPY en data/processed/lspy_proxy (proxy + 9 videos de 8 canales
    reales) más los frames del video guaraní.
  - Letras propias del guaraní (Ã, Ẽ, Ĩ, Õ, Ũ, Ỹ, G̃, CH, MB, ND, NG, NT,
    PUSO): solo los frames del video guaraní, alineados con
    align_alphabet_videos.py --alphabet guarani. Sus límites salen de una
    emisión constante entre anclas: quedan marcadas como no verificadas
    hasta revisar a mano los tiempos (--review-csv).
  - Las anclas con confianza < --min-anchor-conf se descartan del video.

Salida: <output-dir>/index.csv (split por sujeto; el video va siempre a
train), frames/<LETRA>/reel__<k>.npy, revision.csv con los tiempos de cada
letra para revisar y gn_info.json.

Uso:
    python scripts/preprocess/build_gn_dataset.py \\
        --align-json data/processed/gn_alineado/fb_3777160885909384.json
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align_alphabet_videos import ORDER_GN  # noqa: E402
from extract_letter_sequences import iter_frames, process_frames  # noqa: E402

GN_ONLY = {"Ã", "Ẽ", "Ĩ", "Õ", "Ũ", "Ỹ", "G̃", "CH", "MB", "ND", "NG", "NT",
           "PUSO"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--align-json", type=Path, required=True)
    ap.add_argument("--video-dir", type=Path,
                    default=Path("data/raw/LSPy/alfabetos_youtube"))
    ap.add_argument("--lspy-dir", type=Path,
                    default=Path("data/processed/lspy_proxy"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("data/processed/gn_dataset"))
    ap.add_argument("--review-csv", type=Path, default=None,
                    help="revision.csv corregido (letra, inicio_s, fin_s): "
                         "reemplaza los tiempos del alineamiento")
    ap.add_argument("--min-anchor-conf", type=float, default=0.10)
    ap.add_argument("--real-min", type=int, default=300)
    args = ap.parse_args()

    al = json.load(open(args.align_json))
    video = args.video_dir / al["video"]
    fps = al["fps"]
    segs = {s["letra"]: s for s in al["segmentos"]}
    verified = False
    if args.review_csv:
        with open(args.review_csv) as f:
            for r in csv.DictReader(f):
                segs[r["letra"]] = {"letra": r["letra"], "conf": 1.0,
                                    "frames": [round(float(r["inicio_s"]) * fps),
                                               round(float(r["fin_s"]) * fps)]}
        verified = True

    out = args.output_dir.resolve()
    (out / "frames").mkdir(parents=True, exist_ok=True)
    rows, info = [], {}
    for letter in ORDER_GN:
        s = segs.get(letter)
        n_real = 0
        keep = s is not None and (verified or letter in GN_ONLY
                                  or s["conf"] >= args.min_anchor_conf)
        if keep:
            hands = mp.solutions.hands.Hands(static_image_mode=False,
                                             max_num_hands=2)
            a, b = s["frames"]
            recs = process_frames(iter_frames(video, a, b), hands, al["espejado"])
            hands.close()
            d = out / "frames" / letter
            d.mkdir(exist_ok=True)
            files = []
            for k, r in enumerate(recs):
                if r is not None:
                    f = d / f"reel__{k:04d}.npy"
                    np.save(f, r[0])
                    files.append(f)
            reps = max(1, -(-args.real_min // max(len(files), 1)))
            rows += [[os.path.relpath(f, out), letter, "train_reel", "real"]
                     for f in files * reps]
            n_real = len(files)
        n_lspy = 0
        if letter not in GN_ONLY:
            with open(args.lspy_dir / "index.csv") as f:
                for r in csv.DictReader(f):
                    if r["label"] == letter:
                        src = (args.lspy_dir / r["file"]).resolve()
                        rows.append([os.path.relpath(src, out), letter,
                                     r["subject"], r["frame"]])
                        n_lspy += 1
        info[letter] = {
            "propia_guarani": letter in GN_ONLY,
            "frames_video": n_real, "filas_lspy": n_lspy,
            "tiempos_s": ([round(x / fps, 2) for x in s["frames"]]
                          if s else None),
            "conf_alineamiento": s["conf"] if s else None,
            "verificada": verified or (letter not in GN_ONLY and n_lspy > 0),
            "descartada_del_video": s is not None and not keep,
        }

    with open(out / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "subject", "frame"])
        w.writerows(rows)
    with open(out / "revision.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["letra", "inicio_s", "fin_s", "conf", "propia_guarani"])
        for letter in ORDER_GN:
            t = info[letter]["tiempos_s"] or ["", ""]
            w.writerow([letter, t[0], t[1], info[letter]["conf_alineamiento"],
                        info[letter]["propia_guarani"]])
    json.dump({"fuente": al["video"], "verificado_a_mano": verified,
               "letras": info}, open(out / "gn_info.json", "w"), indent=2,
              ensure_ascii=False)

    print(f"{'letra':6}{'propia':>7}{'video':>7}{'lspy':>8}  nota")
    for letter, i in info.items():
        nota = ("descartada (conf. baja)" if i["descartada_del_video"]
                else "no verificada" if not i["verificada"] else "")
        print(f"{letter:6}{'sí' if i['propia_guarani'] else '':>7}"
              f"{i['frames_video']:>7}{i['filas_lspy']:>8}  {nota}")
    print(f"\n{len(rows)} filas → {out / 'index.csv'}")


if __name__ == "__main__":
    main()

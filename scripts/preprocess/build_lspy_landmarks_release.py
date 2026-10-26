"""
Arma LSPy-Landmarks v1: landmarks crudos de MediaPipe Hands del alfabeto
manual LSPy (castellano y achegety guaraní) y de las señas léxicas del
signario SLS, para publicar con licencia CC BY 4.0 (sin videos).

Subconjuntos (--subset):
  alfabeto  video "Alfabeto Manual" (canal Lengua de Señas LSPy), 7 videos
            del alfabeto alineados (align_alphabet_videos.py) y las letras
            sueltas del signario SLS
  achegety  alfabeto guaraní, tiempos verificados frame a frame
  lexico    señas léxicas del signario SLS (un video por seña)

Por muestra: <out>/<subset>/<id>.npz con
  landmarks  (T, 2, 21, 3) float32  x, y en [0,1] de la imagen original sin
                                    espejar, z relativo a la muñeca; NaN si
                                    la mano no se detectó
  world      (T, 2, 21, 3) float32  coordenadas 3D en metros (MediaPipe)
  handedness (T, 2) str             "Left"/"Right"/"" según MediaPipe (que
                                    asume imagen espejada)
  score      (T, 2) float32         confianza de la lateralidad
  frames     (T,) int               índice de frame en el video original
  fps        float
y una fila en <out>/<subset>/index_<shard>.csv. Los videos se leen frame a
frame (sin cargarlos en memoria) y --shard i/n reparte el trabajo en partes.

Uso:
    python scripts/preprocess/build_lspy_landmarks_release.py --subset alfabeto
    python scripts/preprocess/build_lspy_landmarks_release.py --subset lexico --shard 0/4
    python scripts/preprocess/build_lspy_landmarks_release.py --merge
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_letter_sequences import CANAL_SEGS, iter_frames  # noqa: E402

RAW = ROOT / "data/raw/LSPy"
OUT = ROOT / "data/release/lspy_landmarks_v1"
FIELDS = ["archivo", "subconjunto", "etiqueta", "interprete", "fuente",
          "url", "inicio_frame", "fin_frame", "n_frames", "fps",
          "frames_con_mano", "verificacion", "conf_alineamiento"]

# Fuentes autorizadas por sus titulares (ver README del dataset)
YT_SOURCES = {
    "8KUz7d0h7M8": "Proyecto LSPY (Dra. Mirta Barreto), modelo Luis Jara",
    "cSTAtA3NB6Q": "Tico Rodríguez (Intérprete de LSPY)",
    "ynsmyBTHbQs": "Ariel Torres",
    "cktzWVzatAo": "Información de Sordos del Oficial",
    "_XFs9Sgp29w": "Milcar",
    "fg7FLlG_7co": "CODA DOS LENGUAS",
    "11lDa33jb50": "CODA DOS LENGUAS",
}


def extract(video, a=0, b=None):
    """Landmarks de las dos manos, frame a frame, del video sin espejar."""
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)
    L, W, H, S, F = [], [], [], [], []
    for k, fr in enumerate(iter_frames(video, a, b)):
        res = hands.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        lm = np.full((2, 21, 3), np.nan, np.float32)
        wl = np.full((2, 21, 3), np.nan, np.float32)
        hd, sc = ["", ""], np.zeros(2, np.float32)
        if res.multi_hand_landmarks:
            # En modo video MediaPipe a veces devuelve 3 manos pese a
            # max_num_hands=2: se guardan las 2 de mayor confianza
            order = sorted(range(len(res.multi_hand_landmarks)),
                           key=lambda i: -res.multi_handedness[i]
                           .classification[0].score)[:2]
            for j, i in enumerate(order):
                h, c = res.multi_hand_landmarks[i], res.multi_handedness[i]
                lm[j] = [(p.x, p.y, p.z) for p in h.landmark]
                if res.multi_hand_world_landmarks:
                    wl[j] = [(p.x, p.y, p.z) for p in
                             res.multi_hand_world_landmarks[i].landmark]
                hd[j] = c.classification[0].label
                sc[j] = c.classification[0].score
        L.append(lm), W.append(wl), H.append(hd), S.append(sc), F.append(a + k)
    hands.close()
    return (np.stack(L), np.stack(W), np.array(H), np.stack(S),
            np.array(F, np.int32))


def items_alfabeto():
    canal = RAW / "referencias_alfabeto/alfabeto_manual_canal_LSPy.mp4"
    for L, a, b in CANAL_SEGS:
        yield dict(etiqueta=L, video=canal, a=a, b=b,
                   interprete="Canal Lengua de Señas LSPy",
                   fuente="Lengua de Señas LSPy, 'Alfabeto Manual' (YouTube)",
                   url="https://www.youtube.com/watch?v=ftuy_wIIc2c",
                   verificacion="manual (rótulo en pantalla)", conf="")
    for j in sorted((ROOT / "data/processed/lspy_youtube_alineado").glob("*.json")):
        if j.stem == "config":
            continue
        al = json.load(open(j))
        for s in al["segmentos"]:
            yield dict(etiqueta=s["letra"], video=RAW / "alfabetos_youtube" / al["video"],
                       a=s["frames"][0], b=s["frames"][1],
                       interprete=YT_SOURCES.get(j.stem, j.stem),
                       fuente="YouTube", url=f"https://www.youtube.com/watch?v={j.stem}",
                       verificacion="alineamiento automático (no verificado a mano)",
                       conf=round(s["conf"], 3))
    for p in sorted((RAW / "sls_signario/letras").glob("*.mp4")):
        yield dict(etiqueta=p.stem, video=p, a=0, b=None,
                   interprete="SLS Centro de Relevo del Paraguay",
                   fuente="Signario SLS (YouTube)", url="",
                   verificacion="catálogo del signario", conf="")


def items_achegety():
    v = RAW / "alfabetos_youtube/fb_3777160885909384.mp4"
    fps = cv2.VideoCapture(str(v)).get(5)
    rows = csv.DictReader(open(ROOT / "data/analysis/achegety/revision_verificada.csv"))
    for r in rows:
        yield dict(etiqueta=r["letra"], video=v,
                   a=int(round(float(r["inicio_s"]) * fps)),
                   b=int(round(float(r["fin_s"]) * fps)),
                   interprete="Lengua de Señas Py - LSPy",
                   fuente="'Achegety Guaraní Po Ñe'ẽ rehegua' (Facebook)",
                   url="https://www.facebook.com/reel/3777160885909384",
                   verificacion="manual (frame a frame, puntero sobre la lámina)",
                   conf="")


def items_lexico():
    for info in sorted((RAW / "sls_signario/videos").glob("*.info.json")):
        d = json.load(open(info))
        mp4 = info.with_suffix("").with_suffix(".mp4")
        if not mp4.exists() or mp4.stat().st_size == 0:
            continue
        yield dict(etiqueta=d["title"].strip(), video=mp4, a=0, b=None,
                   interprete="SLS Centro de Relevo del Paraguay",
                   fuente="Signario SLS (YouTube)",
                   url=f"https://www.youtube.com/watch?v={d['id']}",
                   verificacion="catálogo del signario", conf="", vid=d["id"])


def slug(s):
    return re.sub(r"[^\w]+", "_", s, flags=re.UNICODE).strip("_")[:60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["alfabeto", "achegety", "lexico"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Omite muestras ya extraídas (para completar faltantes)")
    ap.add_argument("--merge", action="store_true",
                    help="Une los index_*.csv de cada subconjunto en index.csv")
    args = ap.parse_args()

    if args.merge:
        for sub in sorted(p for p in OUT.iterdir() if p.is_dir()):
            rows = []
            for part in sorted(sub.glob("index_*.csv")):
                rows += list(csv.DictReader(open(part)))
            with open(sub / "index.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(sorted(rows, key=lambda r: r["archivo"]))
            print(f"{sub.name}: {len(rows)} muestras")
        return

    k, n = map(int, args.shard.split("/"))
    gen = {"alfabeto": items_alfabeto, "achegety": items_achegety,
           "lexico": items_lexico}[args.subset]
    out = OUT / args.subset
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, it in enumerate(list(gen())):
        if i % n != k:
            continue
        sid = it.get("vid") or f"{i:04d}"
        name = f"{args.subset}_{sid}_{slug(it['etiqueta'])}.npz"
        if args.skip_existing and (out / name).exists():
            continue
        lm, wl, hd, sc, fr = extract(it["video"], it["a"], it["b"])
        fps = cv2.VideoCapture(str(it["video"])).get(5)
        np.savez_compressed(out / name, landmarks=lm, world=wl, handedness=hd,
                            score=sc, frames=fr, fps=fps)
        rows.append(dict(archivo=f"{args.subset}/{name}", subconjunto=args.subset,
                         etiqueta=it["etiqueta"], interprete=it["interprete"],
                         fuente=it["fuente"], url=it["url"],
                         inicio_frame=int(fr[0]) if len(fr) else "",
                         fin_frame=int(fr[-1]) if len(fr) else "",
                         n_frames=len(fr), fps=round(fps, 3),
                         frames_con_mano=int((~np.isnan(lm[:, :, 0, 0])).any(1).sum()),
                         verificacion=it["verificacion"],
                         conf_alineamiento=it["conf"]))
    suffix = "_faltantes" if args.skip_existing else ""
    with open(out / f"index_{k}of{n}{suffix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{args.subset} {args.shard}: {len(rows)} muestras → {out}")


if __name__ == "__main__":
    main()

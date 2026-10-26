"""
Extrae secuencias de letras con landmarks crudos para modelar el movimiento
(J, Z, Ñ, X, LL, RR, H): el pipeline estático descarta las coordenadas y las
features de movimiento.

Por frame guarda:
  feats  (120,)  features estáticas (feature_engineering, convención librai)
  xy     (3, 2)  posición 2D en la imagen de muñeca (0), punta del índice (8)
                 y punta del meñique (20) de la mano más alta
  size   ()      tamaño 2D de la mano |p9 − p0| para normalizar

Las posiciones 2D no cambian con la ambigüedad palma/dorso de MediaPipe (la
mano ocupa el mismo lugar en ambas interpretaciones). Todas las fuentes
quedan en geometría de selfie (la de LibrAI y la cámara en vivo).

Fuentes (--source):
  lsc70    data/raw/LSC70/LSC70/LSC70AN/PerXX/<SEÑA>/PerXX_<SEÑA>_<k>.jpg
           (6 frames de transición por seña y sujeto; se espeja)
  canal    video "Alfabeto Manual" del canal LSPy (ya espejado), segmentado
           por el rótulo de cada letra
  signario videos de letras del signario SLS (se espejan)

Salida: <output>.npz con listas de secuencias, etiquetas y sujetos.
"""
import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]
from preprocessing.feature_engineering import engineer_frame_features  # noqa: E402
from preprocess_lsc70 import (  # noqa: E402
    LABEL_MAP, NUMBER_SIGNS, extract_raw_126, select_hands,
)

TRACK = (0, 8, 20)  # muñeca, punta índice, punta meñique

# Rótulos del video del canal (frame inicio, frame fin), verificados a mano
CANAL_SEGS = [("A", 723, 798), ("B", 799, 870), ("C", 883, 944),
              ("D", 957, 1018), ("E", 1032, 1093), ("F", 1105, 1168),
              ("G", 1180, 1240), ("H", 1252, 1314), ("I", 1330, 1408),
              ("J", 1420, 1491), ("K", 1492, 1566), ("L", 1567, 1629),
              ("LL", 1641, 1702), ("M", 1716, 1776), ("N", 1788, 1843),
              ("Ñ", 1854, 1907), ("O", 1919, 1980), ("P", 1993, 2047),
              ("Q", 2059, 2120), ("R", 2133, 2194), ("RR", 2208, 2267),
              ("S", 2281, 2342), ("T", 2355, 2417), ("U", 2427, 2472),
              ("V", 2482, 2536), ("W", 2546, 2594), ("X", 2605, 2656),
              ("Y", 2668, 2721), ("Z", 2733, 2794)]


class _OneHand:
    """Resultado de MediaPipe reducido a una sola mano (para extract_raw_126)."""
    def __init__(self, hand_lm, handedness):
        self.multi_hand_landmarks = [hand_lm]
        self.multi_handedness = [handedness]


MAX_JUMP = 3.0  # en tamaños de mano por frame


def _hand_record(hand_lm, hd):
    pts = np.array([(lm.x, lm.y) for lm in hand_lm.landmark], np.float32)
    feats = engineer_frame_features(
        extract_raw_126(_OneHand(hand_lm, hd), "all", "librai"))
    size = float(np.linalg.norm(pts[9] - pts[0])) + 1e-6
    return feats, pts[list(TRACK)], size


def process_frames(frames, hands, flip):
    """
    (feats, xy, size) por frame de la mano que deletrea (None si no está).

    Dos pasadas: se detectan todas las manos y se ancla el seguimiento en el
    frame donde alguna mano está más alta (la que deletrea siempre sube; al
    inicio del signario las dos están en reposo y "la más alta" puede ser la
    otra). Desde el ancla se sigue esa mano hacia adelante y hacia atrás por
    la muñeca más cercana; si la más cercana está a más de MAX_JUMP tamaños
    de mano, la mano que deletrea se perdió y el frame se descarta.
    """
    dets = []
    for fr in frames:
        if flip:
            fr = cv2.flip(fr, 1)
        res = hands.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        dets.append(select_hands(res, "all") if res.multi_hand_landmarks else [])
    cands = [(p[0].landmark[0].y, i, j) for i, ps in enumerate(dets)
             for j, p in enumerate(ps)]
    if not cands:
        return [None] * len(frames)
    _, a, j = min(cands)
    recs = [None] * len(frames)
    recs[a] = _hand_record(*dets[a][j])
    for order in (range(a + 1, len(frames)), range(a - 1, -1, -1)):
        prev = recs[a]
        for i in order:
            if not dets[i]:
                continue
            w, s = prev[1][0], prev[2]
            dist = lambda p: np.hypot(p[0].landmark[0].x - w[0],  # noqa: E731
                                      p[0].landmark[0].y - w[1])
            best = min(dets[i], key=dist)
            if dist(best) <= MAX_JUMP * s:
                recs[i] = _hand_record(*best)
                prev = recs[i]
    return recs


def pack(recs):
    keep = [r for r in recs if r is not None]
    if len(keep) < 3:
        return None
    return {"feats": np.stack([r[0] for r in keep]),
            "xy": np.stack([r[1] for r in keep]),
            "size": np.array([r[2] for r in keep], np.float32)}


def video_frames(path, a=0, b=None):
    cap = cv2.VideoCapture(str(path))
    frames, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok or (b is not None and i > b):
            break
        if i >= a:
            frames.append(fr)
        i += 1
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["lsc70", "canal", "signario"],
                    required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    seqs, labels, subjects = [], [], []
    if args.source == "lsc70":
        hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2)
        for sign_dir in sorted(args.input.glob("Per*/*")):
            sign = sign_dir.name
            if sign in NUMBER_SIGNS:
                continue
            imgs = sorted(sign_dir.glob("*.jpg"),
                          key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
            s = pack(process_frames([cv2.imread(str(p)) for p in imgs],
                                    hands, flip=True))
            if s:
                seqs.append(s)
                labels.append(LABEL_MAP.get(sign, sign))
                subjects.append(sign_dir.parent.name)
    else:
        if args.source == "canal":
            frames = video_frames(args.input)
            items = [(L, frames[a:b + 1], "canal") for L, a, b in CANAL_SEGS]
            flip = False
        else:
            items = [(p.stem, video_frames(p), "signario")
                     for p in sorted(args.input.glob("*.mp4"))]
            flip = True
        for L, frs, subj in items:
            # Rastreador nuevo por secuencia: en modo video MediaPipe arrastra
            # el seguimiento (y la interpretación de la mano) a la siguiente
            hands = mp.solutions.hands.Hands(static_image_mode=False,
                                             max_num_hands=2)
            s = pack(process_frames(frs, hands, flip))
            hands.close()
            if s:
                seqs.append(s)
                labels.append(L)
                subjects.append(subj)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        feats=np.array([s["feats"] for s in seqs], dtype=object),
        xy=np.array([s["xy"] for s in seqs], dtype=object),
        size=np.array([s["size"] for s in seqs], dtype=object),
        labels=np.array(labels), subjects=np.array(subjects))
    print(f"{len(seqs)} secuencias → {args.output}")


if __name__ == "__main__":
    main()

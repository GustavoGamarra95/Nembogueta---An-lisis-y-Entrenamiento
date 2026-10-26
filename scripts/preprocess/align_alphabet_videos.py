"""
Segmenta videos del alfabeto manual LSPy (una persona deletreando A→Z) por
alineamiento forzado: el orden de las letras es conocido, así que se busca
con Viterbi la segmentación monótona que mejor explica las probabilidades
del modelo estático LSPy frame a frame.

Estados: reposo ⇄ letra₁ → reposo ⇄ letra₂ → …; LL, RR y CH son opcionales
(se pueden saltar). Cada letra dura al menos --min-frames frames. Emisiones:
  letra con mano visible: log p(LSPY:letra | frame), renormalizado a LSPy
  letra sin mano:         log(1e-3)
  reposo:                 log(0.9) sin mano, log(--rest-prob) con mano
Cada video se alinea con y sin espejo horizontal y se queda la geometría
con mejor puntaje medio por frame (LibrAI y la cámara en vivo son selfie).

Salida por video: <out>/<video>.json con letra, frames [inicio, fin],
confianza media y si se espejó; y una hoja de contactos para revisar.

Uso:
    python scripts/preprocess/align_alphabet_videos.py \\
        --videos data/raw/LSPy/alfabetos_youtube/*.mp4 \\
        --model data/models/alicia-alphabet-lspy-proxy/run_X
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/train"), str(Path(__file__).resolve().parent)]
from extract_letter_sequences import iter_frames, process_frames  # noqa: E402
from train_alphabet_robust import canonicalize_by_majority  # noqa: E402

ORDER = ["A", "B", "C", "CH", "D", "E", "F", "G", "H", "I", "J", "K", "L",
         "LL", "M", "N", "Ñ", "O", "P", "Q", "R", "RR", "S", "T", "U", "V",
         "W", "X", "Y", "Z"]
OPTIONAL = {"CH", "LL", "RR"}


def emissions(recs, model, mu, sd, classes, rest_prob):
    letters = [l for l in ORDER if f"LSPY:{l}" in classes or l in OPTIONAL]
    idx = {l: classes.index(f"LSPY:{l}") for l in letters
           if f"LSPY:{l}" in classes}
    has = np.array([r is not None for r in recs])
    E = np.full((len(recs), len(letters)), np.log(1e-3))
    if has.any():
        X = canonicalize_by_majority(np.stack([r[0] for r in recs if r]))
        p = model.predict((X - mu) / sd, verbose=0, batch_size=512)
        cols = [idx[l] for l in letters if l in idx]
        q = p[:, cols] / p[:, cols].sum(1, keepdims=True)
        E_has = np.full((has.sum(), len(letters)), np.log(1e-3))
        for k, l in enumerate([l for l in letters if l in idx]):
            E_has[:, letters.index(l)] = np.log(q[:, k] + 1e-6)
        E[has] = E_has
    rest = np.where(has, np.log(rest_prob), np.log(0.9))
    return letters, E, rest


def viterbi(letters, E, rest, min_frames, optional=OPTIONAL):
    """Estados: _0, L1×m, _1, L2×m, ... ; devuelve [(letra, ini, fin)]."""
    REST = "_reposo"  # no puede llamarse "R": choca con la letra R
    states = [(REST, None)]
    for l in letters:
        states += [(l, k) for k in range(min_frames)] + [(REST, l)]
    S, T = len(states), len(E)
    col = {l: i for i, l in enumerate(letters)}
    emis = np.array([[rest[t] if s[0] == REST else E[t, col[s[0]]]
                      for s in states] for t in range(T)])
    NEG = -1e18
    dp = np.full((T, S), NEG)
    bp = np.zeros((T, S), dtype=np.int32)
    dp[0, 0] = emis[0, 0]
    if min_frames == 1:
        dp[0, 1] = emis[0, 1]
    # predecesores de cada estado
    preds = [[] for _ in range(S)]
    for s, (name, k) in enumerate(states):
        preds[s].append(s if (name == REST or k == min_frames - 1) else None)
        if s > 0:
            preds[s].append(s - 1)
        if name != REST and k == 0:
            # se puede llegar saltando letras opcionales previas
            j = s - 1  # reposo anterior
            while j > 0:
                prev_letter = states[j][1]
                if prev_letter in optional:
                    j -= min_frames + 1
                    preds[s].append(j)
                else:
                    break
        preds[s] = [p for p in preds[s] if p is not None and p >= 0]
    for t in range(1, T):
        for s in range(S):
            best, arg = NEG, 0
            for p in preds[s]:
                if dp[t - 1, p] > best:
                    best, arg = dp[t - 1, p], p
            dp[t, s] = best + emis[t, s]
            bp[t, s] = arg
    # final: último reposo o última letra completa
    end = max([S - 1, S - 2], key=lambda s: dp[T - 1, s])
    path = [end]
    for t in range(T - 1, 0, -1):
        path.append(bp[t, path[-1]])
    path = path[::-1]
    segs, cur = [], None
    for t, s in enumerate(path):
        name = states[s][0]
        if name != REST:
            if cur and cur[0] == name:
                cur[2] = t
            else:
                cur = [name, t, t]
                segs.append(cur)
        else:
            cur = None
    return [tuple(x) for x in segs], float(dp[T - 1, end] / T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, nargs="+", required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path,
                    default=Path("data/processed/lspy_youtube_alineado"))
    ap.add_argument("--config", type=Path, default=None,
                    help="JSON {video_stem: {inicio_s, fin_s, incluir, "
                         "excluir}}: tramo útil y letras opcionales (LL, RR, "
                         "CH) confirmadas o ausentes en ese video")
    ap.add_argument("--min-frames", type=int, default=4)
    ap.add_argument("--rest-prob", type=float, default=0.05)
    args = ap.parse_args()

    model = tf.keras.models.load_model(str(args.model / "best_model.keras"))
    mu, sd = np.load(args.model / "norm_mean.npy"), np.load(args.model / "norm_std.npy")
    classes = [str(c) for c in np.load(args.model / "label_classes.npy")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = json.load(open(args.config)) if args.config else {}
    for v in args.videos:
        cfg = config.get(v.stem, {})
        fps = cv2.VideoCapture(str(v)).get(5)
        a0 = int(cfg.get("inicio_s", 0) * fps)
        b0 = int(cfg["fin_s"] * fps) if "fin_s" in cfg else None
        excl = set(cfg.get("excluir", [])) | {"CH"}
        optional = OPTIONAL - set(cfg.get("incluir", [])) - excl
        best = None
        for flip in (False, True):
            hands = mp.solutions.hands.Hands(static_image_mode=False,
                                             max_num_hands=2)
            # streaming: el video se relee en cada pasada, sin cargarlo
            recs = process_frames(iter_frames(v, a0, b0), hands, flip)
            hands.close()
            letters, E, rest = emissions(recs, model, mu, sd, classes,
                                         args.rest_prob)
            keep = [i for i, l in enumerate(letters) if l not in excl]
            letters, E = [letters[i] for i in keep], E[:, keep]
            segs, score = viterbi(letters, E, rest, args.min_frames, optional)
            if best is None or score > best[1]:
                best = (segs, score, flip, E, letters)
        segs, score, flip, E, letters = best
        out = [{"letra": l, "frames": [a + a0, b + a0],
                "conf": float(np.exp(E[a:b + 1, letters.index(l)]).mean())}
               for l, a, b in segs]
        json.dump({"video": v.name, "espejado": flip, "score": score,
                   "fps": cv2.VideoCapture(str(v)).get(5), "segmentos": out},
                  open(args.output_dir / f"{v.stem}.json", "w"),
                  indent=2, ensure_ascii=False)
        # hoja de contactos: frame central de cada letra
        ims = []
        for s in out:
            mid = (s["frames"][0] + s["frames"][1]) // 2
            fr = next(iter_frames(v, mid, mid))
            fr = cv2.resize(fr, (200, int(200 * fr.shape[0] / fr.shape[1])))
            cv2.putText(fr, f"{s['letra']} {s['conf']:.2f}", (4, 20), 0, 0.6,
                        (0, 0, 255), 2)
            ims.append(fr)
        while len(ims) % 8:
            ims.append(np.zeros_like(ims[0]))
        cv2.imwrite(str(args.output_dir / f"{v.stem}.jpg"),
                    np.vstack([np.hstack(ims[i:i + 8])
                               for i in range(0, len(ims), 8)]))
        print(f"{v.stem}: espejo={flip} score={score:.2f} "
              f"{len(out)} letras: {' '.join(s['letra'] for s in out)}",
              flush=True)


if __name__ == "__main__":
    main()

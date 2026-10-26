"""
Modelo de secuencia para el alfabeto LSPy: distingue letras con movimiento
(J, Z, Ñ, X, LL, RR, H) que el modelo de frames sueltos no puede separar.

Entrada por secuencia: T pasos × (60 features estáticas de la mano activa +
8 de movimiento). Movimiento por paso, normalizado por tamaño de mano:
desplazamiento de muñeca (2), del índice relativo a la muñeca (2), del
meñique relativo a la muñeca (2) y posición de la muñeca respecto del
inicio (2).

Datos:
  - LSC70 (extract_letter_sequences.py --source lsc70): letras LSPy cuyo
    origen en el mapeo resuelto es LSC, más J, Z y Ñ por nombre (supuesto:
    mismo movimiento que en LSC; el signario lo pone a prueba).
  - Pseudo-secuencias estáticas (frame repetido, movimiento ~0) para letras
    con origen LIBRAS/ASL, desde data/processed/lspy_proxy.
  - Secuencias reales del canal LSPy, aumentadas (recorte temporal, escala
    del movimiento, jitter).
Validación externa: secuencias del signario.

Uso:
    python scripts/train/train_letter_sequences.py
"""
import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_alphabet_robust import canonicalize_by_majority, mirror_features  # noqa: E402

T = 8


def canonical_hand(feats: np.ndarray) -> np.ndarray:
    """(N,120) → (N,60) de la mano activa en geometría habitual (slot Left)."""
    X = canonicalize_by_majority(feats)
    if (np.abs(X[:, 60:]).sum(1) > 0).mean() <= 0.5:
        X = mirror_features(X)  # todo el video quedó como "otra mano"
    left = np.abs(X[:, 60:]).sum(1) > 0
    return np.where(left[:, None], X[:, 60:], X[:, :60])


def motion(xy: np.ndarray, size: np.ndarray) -> np.ndarray:
    """(N,3,2) muñeca/índice/meñique + (N,) tamaño → (N,8)."""
    s = np.median(size)
    w = xy[:, 0] / s
    rel_i, rel_p = (xy[:, 1] - xy[:, 0]) / s, (xy[:, 2] - xy[:, 0]) / s
    d = lambda a: np.vstack([np.zeros((1, 2)), np.diff(a, axis=0)])  # noqa: E731
    return np.hstack([d(w), d(rel_i), d(rel_p), w - w[0]]).astype(np.float32)


def resample(X: np.ndarray, t: int = T) -> np.ndarray:
    src = np.linspace(0, 1, len(X))
    dst = np.linspace(0, 1, t)
    return np.stack([np.interp(dst, src, X[:, j]) for j in range(X.shape[1])], 1)


def working_span(xy, size, radius: float = 2.5):
    """
    Descarta la llegada de la mano (en el signario sube desde la cintura):
    tramo entre el primer y el último frame con la muñeca a menos de
    `radius` tamaños de mano de su posición de trabajo (mediana de la
    segunda mitad).
    """
    w = xy[:, 0]
    home = np.median(w[len(w) // 2:], axis=0)
    near = np.flatnonzero(np.linalg.norm(w - home, axis=1) < radius * np.median(size))
    if len(near) < 3:
        return 0, len(w)
    return int(near[0]), int(near[-1]) + 1


def held_segment(feats: np.ndarray, min_len: int = 5) -> np.ndarray:
    """
    Índices del tramo más largo con la forma de la mano estable (velocidad
    de la configuración bajo la mediana): la letra sostenida, sin la
    llegada ni el reposo. Es el tramo con el que se validó el modelo
    estático en el signario.
    """
    hand = canonical_hand(feats)[:, :56]
    vel = np.r_[0, np.linalg.norm(np.diff(hand, axis=0), axis=1)]
    still = vel <= np.median(vel)
    best, cur = [], []
    for i, ok in enumerate(still):
        if ok:
            cur.append(i)
        else:
            best, cur = max(best, cur, key=len), []
    best = max(best, cur, key=len)
    return np.array(best if len(best) >= min_len else range(len(feats)))


TRIM_ARRIVAL = False  # --trim-arrival; empeoró el signario (huecos de seguimiento)


def sequence_vector(feats, xy, size, crop=(0.0, 1.0)) -> np.ndarray:
    if TRIM_ARRIVAL:
        s0, s1 = working_span(xy, size)
        feats, xy, size = feats[s0:s1], xy[s0:s1], size[s0:s1]
    n = len(feats)
    a, b = int(crop[0] * n), max(int(crop[1] * n), int(crop[0] * n) + 2)
    hand = canonical_hand(feats)[a:b]
    mov = motion(xy[a:b], size[a:b])
    mov[:, :6] *= (len(mov) - 1) / (T - 1)  # desplazamiento por paso de T
    return resample(np.hstack([hand, mov])).astype(np.float32)


def load_seqs(path: Path):
    d = np.load(path, allow_pickle=True)
    return list(zip(d["feats"], d["xy"], d["size"], d["labels"], d["subjects"]))


def build(args, rng):
    mapping = json.load(open(args.mapping))
    classes = sorted(mapping)
    lsc_to_lspy = {m["origen"][1]: l for l, m in mapping.items()
                   if m["origen"] and m["origen"][0] == "LSC"}
    for name in ("J", "Z", "Ñ"):  # supuesto por nombre
        lsc_to_lspy.setdefault(name, name)

    X, y, subj = [], [], []
    for f, xy, s, lab, sub in load_seqs(args.lsc70):
        if lab in lsc_to_lspy:
            X.append(sequence_vector(f, xy, s))
            y.append(lsc_to_lspy[lab])
            subj.append(sub)

    # Pseudo-secuencias estáticas para letras con origen LIBRAS/ASL
    with open(args.proxy / "index.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if r["frame"] == "proxy"]
    by_letter = {}
    for r in rows:
        if mapping[r["label"]]["origen"][0] != "LSC":
            by_letter.setdefault(r["label"], []).append(r)
    for letter, rs in by_letter.items():
        for r in rng.choice(rs, min(len(rs), args.static_per_letter),
                            replace=False):
            v = np.load(args.proxy / r["file"])[None].repeat(T, 0)
            hand = canonical_hand(v)
            mov = rng.normal(0, 0.02, (T, 8)).astype(np.float32)
            X.append(np.hstack([hand, mov]))
            y.append(letter)
            subj.append(r["subject"])

    # LSM dinámico: J, K, Ñ, Q, X, Z por nombre (supuesto: mismo movimiento
    # que en LSPy; el signario lo pone a prueba)
    if args.lsm and args.lsm.exists():
        for f, xy, s, lab, sub in load_seqs(args.lsm):
            if lab in classes:
                X.append(sequence_vector(f, xy, s))
                y.append(lab)
                subj.append(f"LSM_{sub}")

    # LSPy real (canal y videos de YouTube alineados): aumentados
    real = [(args.canal, args.canal_aug, lambda _: "train_canal")]
    if args.youtube and args.youtube.exists():
        real.append((args.youtube, args.youtube_aug, lambda s: f"yt_{s}"))
    for path, n_aug, subject in real:
        for f, xy, s, lab, sub in load_seqs(path):
            if lab not in classes:
                continue
            for _ in range(n_aug):
                a = rng.uniform(0.0, 0.25)
                b = rng.uniform(0.75, 1.0)
                v = sequence_vector(f, xy, s, (a, b))
                v[:, 60:] *= rng.uniform(0.7, 1.3)
                v[:, :60] += rng.normal(0, 0.02, v[:, :60].shape)
                X.append(v)
                y.append(lab)
                subj.append(subject(sub))
    return np.stack(X), np.array(y), np.array(subj), classes


def build_model(n_classes, n_feat):
    inp = tf.keras.Input((T, n_feat))
    x = tf.keras.layers.Conv1D(96, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(96, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Concatenate()([
        tf.keras.layers.GlobalAveragePooling1D()(x),
        tf.keras.layers.GlobalMaxPooling1D()(x)])
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(96, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="Alicia_LSPy_Secuencias")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lsc70", type=Path,
                    default=Path("data/processed/sequences/lsc70.npz"))
    ap.add_argument("--canal", type=Path,
                    default=Path("data/processed/sequences/canal.npz"))
    ap.add_argument("--signario", type=Path,
                    default=Path("data/processed/sequences/signario.npz"))
    ap.add_argument("--proxy", type=Path,
                    default=Path("data/processed/lspy_proxy"))
    ap.add_argument("--mapping", type=Path,
                    default=Path("data/processed/lspy_proxy/mapeo_resuelto.json"))
    ap.add_argument("--lsm", type=Path,
                    default=Path("data/processed/sequences/lsm.npz"))
    ap.add_argument("--youtube", type=Path,
                    default=Path("data/processed/sequences/youtube.npz"))
    ap.add_argument("--youtube-aug", type=int, default=40)
    ap.add_argument("--static-model", type=Path, default=None,
                    help="Modelo multilengua con clases LSPY:* para fusionar")
    ap.add_argument("--trim-arrival", action="store_true",
                    help="Recortar la llegada de la mano (ver working_span)")
    ap.add_argument("--static-per-letter", type=int, default=400)
    ap.add_argument("--canal-aug", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-root", type=Path,
                    default=Path("data/models/alicia-lspy-sequences"))
    args = ap.parse_args()
    global TRIM_ARRIVAL
    TRIM_ARRIVAL = args.trim_arrival

    rng = np.random.default_rng(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    X, y_lab, subj, classes = build(args, rng)
    y = np.array([classes.index(l) for l in y_lab])
    print(f"{len(X)} secuencias, {len(classes)} clases")
    print(dict(sorted(Counter(y_lab).items())))

    # Validación interna: 10 sujetos de LSC70 fuera del entrenamiento
    lsc_subj = sorted({s for s in subj if s.startswith("Per")})
    rng.shuffle(lsc_subj)
    val_s = set(lsc_subj[:10])
    va = np.isin(subj, list(val_s))
    mu = X[~va].reshape(-1, X.shape[2]).mean(0)
    sd = X[~va].reshape(-1, X.shape[2]).std(0) + 1e-6
    norm = lambda Z: ((Z - mu) / sd).astype(np.float32)  # noqa: E731
    cw = Counter(y[~va])
    w = np.array([len(y[~va]) / (len(cw) * cw[c]) for c in y[~va]], np.float32)

    model = build_model(len(classes), X.shape[2])
    model.compile(tf.keras.optimizers.Adam(1e-3),
                  "sparse_categorical_crossentropy", weighted_metrics=["accuracy"])
    out = args.output_root / f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    model.fit(norm(X[~va]), y[~va], sample_weight=w, batch_size=128,
              epochs=args.epochs, validation_data=(norm(X[va]), y[va]),
              verbose=2, callbacks=[tf.keras.callbacks.EarlyStopping(
                  "val_accuracy", patience=15, restore_best_weights=True)])
    pv = model.predict(norm(X[va]), verbose=0).argmax(1)
    per_letter = {classes[c]: float((pv[y[va] == c] == c).mean())
                  for c in np.unique(y[va])}
    print("\nLSC70, 10 sujetos no vistos, por letra LSPy:")
    print("  " + "  ".join(f"{k} {v:.0%}" for k, v in per_letter.items()))
    model.save(out / "best_model.keras")
    np.save(out / "norm_mean.npy", mu)
    np.save(out / "norm_std.npy", sd)
    np.save(out / "label_classes.npy", np.array(classes))

    # Validación externa: signario, secuencia sola y fusionada con el
    # modelo estático (media geométrica, peso fijo 0.5: no se ajusta con
    # el propio signario)
    static = None
    if args.static_model:
        sm = tf.keras.models.load_model(str(args.static_model / "best_model.keras"))
        s_mu = np.load(args.static_model / "norm_mean.npy")
        s_sd = np.load(args.static_model / "norm_std.npy")
        s_cls = [str(c) for c in np.load(args.static_model / "label_classes.npy")]
        s_idx = [s_cls.index(f"LSPY:{c}") for c in classes]
        static = (sm, s_mu, s_sd, s_idx)

    results = {"secuencia": [], "estatico": [], "fusion": []}
    for f, xy, s, lab, _ in load_seqs(args.signario):
        p_seq = model.predict(norm(sequence_vector(f, xy, s)[None]), verbose=0)[0]
        preds = {"secuencia": p_seq}
        if static:
            sm, s_mu, s_sd, s_idx = static
            seg = held_segment(f)
            ps = sm.predict((canonicalize_by_majority(f[seg]) - s_mu) / s_sd,
                            verbose=0)[:, s_idx]
            p_st = ps.mean(0) / ps.mean(0).sum()
            preds["estatico"] = p_st
            preds["fusion"] = np.sqrt(p_seq * p_st)
        for k, p in preds.items():
            results[k].append((lab, classes[int(p.argmax())]))
    report = {}
    print("\nSignario (validación externa):")
    for k, rows in results.items():
        if not rows:
            continue
        hits = sum(l == t for l, t in rows)
        report[k] = {"aciertos": hits, "n": len(rows),
                     "pred": {l: t for l, t in rows}}
        print(f"  {k:10} {hits}/{len(rows)}  fallos: "
              + " ".join(f"{l}→{t}" for l, t in rows if l != t))
    json.dump({"model_name": "Alicia", "model_type": "conv1d_sequence_lspy",
               "T": T, "classes": classes, "signario": report,
               "val_subjects_lsc70": sorted(val_s),
               "val_lsc70_per_letter": per_letter,
               "trim_arrival": TRIM_ARRIVAL},
              open(out / "model_info.json", "w"), indent=2, ensure_ascii=False)
    print(f"Artefactos: {out}")


if __name__ == "__main__":
    main()

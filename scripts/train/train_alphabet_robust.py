"""
Entrena un modelo base robusto del alfabeto LIBRAS/LSP.

Combina:
  A) Data augmentation en training (palm rotation, jitter, handedness flip)
  B) Mezcla LibrAI + datos del operador con class/source balancing

Objetivo: cerrar el gap entre LibrAI holdout (≈99%) y cámara real (≈24%)
sin caer en catastrophic forgetting.

Uso (dentro del contenedor):
    python scripts/train/train_alphabet_robust.py \\
        --librai-dir /data/processed/librai_alphabet_v4 \\
        --session-dir /data/realworld_eval/sessions/20260614_011801_S01 \\
        --output-root /data/models/librai-alphabet-robust
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Layout de features (por mano, 60 features):
#   0..20  → 21 distancias
#   21..50 → 30 ángulos (15 triplets × sin+cos)
#   51..55 → 5 finger_curl
#   56..59 → 4 palm angles (sin_p, cos_p, sin_y, cos_y)
HAND_OFFSETS = [0, 60]  # Right, Left
PALM_SIN_COS_PAIRS = [(56, 57), (58, 59)]  # (pitch sin/cos), (yaw sin/cos)


def load_librai(root: Path, classes: list, seed: int = 42):
    """Carga todos los .npy de LibrAI y splittea 70/15/15."""
    X, y = [], []
    for i, letter in enumerate(classes):
        d = root / letter
        if not d.exists():
            continue
        for f in sorted(d.glob("*.npy")):
            X.append(np.load(f))
            y.append(i)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # 70/15/15 estratificado (mismo seed que el v4 original)
    X_tv, X_te, y_tv, y_te = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=seed,
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tv, y_tv, test_size=0.15 / 0.85, stratify=y_tv, random_state=seed,
    )
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


def load_session(session_dir: Path, classes: list):
    """Rep 0-2 → train, rep 3 → val, rep 4 → test."""
    splits = {"train": ([], []), "val": ([], []), "test": ([], [])}
    for f in sorted((session_dir / "letters").glob("*.npz")):
        letter, rep_str = f.stem.split("_rep")
        if letter not in classes:
            continue
        rep = int(rep_str)
        d = np.load(f)
        feats = d["features"]
        probs = d["probs"]
        mask = probs.sum(axis=1) > 0
        feats_d = feats[mask]
        if len(feats_d) == 0:
            continue
        y_idx = classes.index(letter)
        split = "train" if rep <= 2 else ("val" if rep == 3 else "test")
        splits[split][0].append(feats_d)
        splits[split][1].extend([y_idx] * len(feats_d))

    out = {}
    for k, (xs, ys) in splits.items():
        if xs:
            out[k] = (np.vstack(xs), np.array(ys, dtype=np.int64))
        else:
            out[k] = (np.empty((0, 120)), np.empty(0, dtype=np.int64))
    return out


def augment_batch(
    X: tf.Tensor,
    feat_std: tf.Tensor,
    palm_rot_deg: float = 20.0,
    jitter_sigma: float = 0.5,
    flip_prob: float = 0.5,
) -> tf.Tensor:
    """
    Aplica augmentation per-sample sobre features (B, 120).
    - palm angles: rotación 2D de cada par (sin, cos) por θ ∈ U(-rot, rot)
    - jitter gaussiano en TODAS las features escalado por feat_std
    - handedness flip: swap (0..60) ↔ (60..120) con probabilidad flip_prob
    """
    B = tf.shape(X)[0]
    F = 120

    # 1) Jitter
    noise = tf.random.normal((B, F)) * jitter_sigma * feat_std[None, :]
    X = X + noise

    # 2) Palm rotation por mano
    rot_max = palm_rot_deg * np.pi / 180.0
    for hand_off in HAND_OFFSETS:
        theta = tf.random.uniform((B,), -rot_max, rot_max)
        cos_t = tf.cos(theta)
        sin_t = tf.sin(theta)
        for s_idx, c_idx in PALM_SIN_COS_PAIRS:
            si = hand_off + s_idx
            ci = hand_off + c_idx
            s = X[:, si]
            c = X[:, ci]
            new_s = s * cos_t + c * sin_t
            new_c = c * cos_t - s * sin_t
            # actualizar (TF no permite assignment directo)
            mask_s = tf.one_hot(si, F)
            mask_c = tf.one_hot(ci, F)
            X = X + (new_s - s)[:, None] * mask_s[None, :]
            X = X + (new_c - c)[:, None] * mask_c[None, :]

    # 3) Handedness flip
    flip_mask = tf.random.uniform((B,)) < flip_prob
    X_flipped = tf.concat([X[:, 60:120], X[:, 0:60]], axis=1)
    X = tf.where(flip_mask[:, None], X_flipped, X)
    return X


def build_model(n_features: int, n_classes: int, dropout: float):
    inp = tf.keras.Input(shape=(n_features,))
    x = tf.keras.layers.Dense(512, activation="relu")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="DNN_Alphabet_Robust")


def make_dataset(
    X: np.ndarray, y: np.ndarray, sample_w: np.ndarray,
    feat_std: tf.Tensor, augment: bool, batch_size: int, shuffle: bool,
    palm_rot: float, jitter: float, flip_p: float,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((X, y, sample_w))
    if shuffle:
        ds = ds.shuffle(min(len(X), 20000), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    if augment:
        ds = ds.map(
            lambda x, yy, w: (
                augment_batch(x, feat_std, palm_rot, jitter, flip_p),
                yy, w,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return ds.prefetch(tf.data.AUTOTUNE)


def evaluate(model, X, y, name) -> dict:
    if len(X) == 0:
        return {"name": name, "top1": 0.0, "top5": 0.0, "n": 0}
    p = model.predict(X, verbose=0, batch_size=512)
    top1 = float((p.argmax(axis=1) == y).mean())
    top5 = float(np.any(
        np.argsort(p, axis=1)[:, -5:] == y[:, None], axis=1
    ).mean())
    return {"name": name, "top1": top1, "top5": top5, "n": int(len(y))}


def per_letter_acc(model, X, y, classes) -> dict:
    if len(X) == 0:
        return {}
    preds = model.predict(X, verbose=0, batch_size=512).argmax(axis=1)
    return {
        letter: float((preds[y == i] == i).mean())
        for i, letter in enumerate(classes)
        if (y == i).sum() > 0
    }


def run(args):
    classes = list(np.load(
        Path(args.base_model_dir) / "label_classes.npy"
    ))
    print(f"\nClases: {len(classes)} → {classes}")

    print("\n=== Cargando datos ===")
    (Xl_tr, yl_tr), (Xl_va, yl_va), (Xl_te, yl_te) = load_librai(
        Path(args.librai_dir), classes,
    )
    print(f"LibrAI: train={len(Xl_tr)}  val={len(Xl_va)}  test={len(Xl_te)}")

    sess = load_session(Path(args.session_dir), classes)
    Xo_tr, yo_tr = sess["train"]
    Xo_va, yo_va = sess["val"]
    Xo_te, yo_te = sess["test"]
    print(f"Operador: train={len(Xo_tr)}  val={len(Xo_va)}  test={len(Xo_te)}")

    # Combinar train con sample_weight
    n_lib = len(Xl_tr)
    n_op = len(Xo_tr)
    # peso del operador para igualar contribución por gradiente
    op_weight = n_lib / max(n_op, 1)
    w_lib = np.ones(n_lib, dtype=np.float32)
    w_op = np.full(n_op, op_weight, dtype=np.float32)

    X_tr = np.vstack([Xl_tr, Xo_tr]).astype(np.float32)
    y_tr = np.concatenate([yl_tr, yo_tr]).astype(np.int64)
    w_tr = np.concatenate([w_lib, w_op])
    print(
        f"\nTrain combinado: {len(X_tr)} samples "
        f"(op_weight={op_weight:.2f})"
    )

    # Normalización: stats sobre TRAIN combinado
    norm_mean = X_tr.mean(axis=0).astype(np.float32)
    norm_std = (X_tr.std(axis=0) + 1e-6).astype(np.float32)

    X_tr_n = (X_tr - norm_mean) / norm_std
    Xl_va_n = (Xl_va - norm_mean) / norm_std
    Xo_va_n = (Xo_va - norm_mean) / norm_std
    Xl_te_n = (Xl_te - norm_mean) / norm_std
    Xo_te_n = (Xo_te - norm_mean) / norm_std

    # Val combinado (para early stopping no sesgado)
    X_va_n = np.vstack([Xl_va_n, Xo_va_n])
    y_va = np.concatenate([yl_va, yo_va])
    w_va = np.concatenate([
        np.ones(len(yl_va), dtype=np.float32),
        np.full(len(yo_va), op_weight, dtype=np.float32),
    ])

    feat_std_tf = tf.constant(np.ones(120, dtype=np.float32))  # ya normalizado

    train_ds = make_dataset(
        X_tr_n, y_tr, w_tr, feat_std_tf,
        augment=True, batch_size=args.batch_size, shuffle=True,
        palm_rot=args.palm_rot_deg, jitter=args.jitter_sigma,
        flip_p=args.flip_prob,
    )
    val_ds = make_dataset(
        X_va_n, y_va, w_va, feat_std_tf,
        augment=False, batch_size=512, shuffle=False,
        palm_rot=0, jitter=0, flip_p=0,
    )

    model = build_model(120, len(classes), dropout=args.dropout)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False,
        ),
        metrics=["accuracy",
                 tf.keras.metrics.SparseTopKCategoricalAccuracy(
                     k=5, name="top5"
                 )],
    )
    model.summary()

    out_dir = (
        Path(args.output_root) /
        f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=15,
            restore_best_weights=True, mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=7,
            min_lr=1e-6, mode="max",
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "best_model.keras"),
            monitor="val_accuracy", save_best_only=True, mode="max",
        ),
        tf.keras.callbacks.CSVLogger(str(out_dir / "history.csv")),
    ]

    print(
        f"\nTraining: lr={args.lr} dropout={args.dropout} "
        f"palm_rot=±{args.palm_rot_deg}° jitter={args.jitter_sigma}σ "
        f"flip={args.flip_prob}\n"
    )
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs, callbacks=callbacks, verbose=2,
    )

    # Mejor modelo
    best = tf.keras.models.load_model(str(out_dir / "best_model.keras"))

    # Evaluaciones
    print("\n=== Evaluación final ===")
    base_model = tf.keras.models.load_model(
        str(Path(args.base_model_dir) / "best_model.keras")
    )
    base_mean = np.load(Path(args.base_model_dir) / "norm_mean.npy")
    base_std = np.load(Path(args.base_model_dir) / "norm_std.npy")

    # Para el modelo base, normalizar con SU mean/std
    Xl_te_base = (Xl_te - base_mean) / base_std
    Xo_te_base = (Xo_te - base_mean) / base_std
    Xo_all_base = (
        np.vstack([sess["train"][0], sess["val"][0], sess["test"][0]])
        - base_mean
    ) / base_std
    yo_all = np.concatenate([
        sess["train"][1], sess["val"][1], sess["test"][1],
    ])
    Xo_all_new = (
        np.vstack([sess["train"][0], sess["val"][0], sess["test"][0]])
        - norm_mean
    ) / norm_std

    results = {
        "base_v4": {
            "librai_test": evaluate(base_model, Xl_te_base, yl_te,
                                    "base_librai_test"),
            "operator_test": evaluate(base_model, Xo_te_base, yo_te,
                                      "base_op_test"),
            "operator_all": evaluate(base_model, Xo_all_base, yo_all,
                                     "base_op_all"),
        },
        "robust_new": {
            "librai_test": evaluate(best, Xl_te_n, yl_te,
                                    "robust_librai_test"),
            "operator_test": evaluate(best, Xo_te_n, yo_te,
                                      "robust_op_test"),
            "operator_all": evaluate(best, Xo_all_new, yo_all,
                                     "robust_op_all"),
        },
    }

    def line(d):
        return f"top-1={d['top1'] * 100:6.2f}%  top-5={d['top5'] * 100:6.2f}%  (n={d['n']})"

    print(f"\n{'Métrica':<25}  {'base v4':<40}  →  {'robust nuevo':<40}")
    print("-" * 115)
    for k in ["librai_test", "operator_test", "operator_all"]:
        b = results["base_v4"][k]
        r = results["robust_new"][k]
        print(f"  {k:<23}  {line(b):<40}  →  {line(r):<40}")

    # Per letter en operator_all
    pl_base = per_letter_acc(base_model, Xo_all_base, yo_all, classes)
    pl_new = per_letter_acc(best, Xo_all_new, yo_all, classes)
    print(f"\nPer letra (operator_all):")
    print(f"{'Letra':<5} {'base':>8} {'robust':>8} {'Δ':>8}")
    for letter in classes:
        if letter not in pl_new:
            continue
        b = pl_base.get(letter, 0)
        n = pl_new[letter]
        d = n - b
        marker = "↑↑" if d > 0.3 else ("↑" if d > 0.05 else (
            "↓↓" if d < -0.3 else ("↓" if d < -0.05 else " ")
        ))
        print(f"  {letter:<3} {b * 100:>6.1f}% {n * 100:>6.1f}% {d * 100:>+6.1f}pp  {marker}")

    # Guardar normalización y metadata
    np.save(out_dir / "norm_mean.npy", norm_mean)
    np.save(out_dir / "norm_std.npy", norm_std)
    np.save(out_dir / "label_classes.npy", np.array(classes))

    info = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "num_features": 120,
        "num_classes": len(classes),
        "classes": classes,
        "label_names": classes,
        "model_type": "dnn_robust",
        "training_data": {
            "librai": {"train": int(n_lib), "val": int(len(Xl_va)),
                       "test": int(len(Xl_te))},
            "operator": {"train": int(n_op), "val": int(len(Xo_va)),
                         "test": int(len(Xo_te))},
            "op_weight": float(op_weight),
        },
        "augmentation": {
            "palm_rot_deg": args.palm_rot_deg,
            "jitter_sigma": args.jitter_sigma,
            "flip_prob": args.flip_prob,
            "dropout": args.dropout,
        },
        "hyperparameters": {
            "lr": args.lr, "batch_size": args.batch_size,
            "epochs_max": args.epochs,
            "epochs_trained": len(history.history["loss"]),
        },
        "test_accuracy": results["robust_new"]["librai_test"]["top1"],
        "results": results,
    }
    with open(out_dir / "model_info.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"\nArtefactos: {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--librai-dir", type=Path,
                   default=Path("/data/processed/librai_alphabet_v4"))
    p.add_argument("--session-dir", type=Path, required=True)
    p.add_argument("--base-model-dir", type=Path,
                   default=Path(
                       "/data/models/librai-alphabet-v4/run_20260601_015641"
                   ),
                   help="Solo para comparación final")
    p.add_argument("--output-root", type=Path,
                   default=Path("/data/models/librai-alphabet-robust"))
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--palm-rot-deg", type=float, default=20.0)
    p.add_argument("--jitter-sigma", type=float, default=0.5)
    p.add_argument("--flip-prob", type=float, default=0.5)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()

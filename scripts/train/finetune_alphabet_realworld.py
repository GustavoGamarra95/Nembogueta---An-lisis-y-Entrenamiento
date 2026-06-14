"""
Fine-tuning del modelo de alfabeto sobre datos de una sesión real.

Toma una sesión generada por evaluate_realworld.py, splittea por
repetición (leave-one-rep-out) y re-entrena el modelo base con LR
baja para adaptarlo al estilo del operador.

Mantiene la normalización (norm_mean/std) del modelo base — fine-tuning
asume que las capas internas esperan esa distribución.

Uso (dentro del contenedor):
    python scripts/train/finetune_alphabet_realworld.py \\
        --session-dir /data/realworld_eval/sessions/20260614_011801_S01 \\
        --base-model-dir /data/models/librai-alphabet-v4/run_20260601_015641 \\
        --output-root /data/models/librai-alphabet-v4-ft
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf


def load_session(session_dir: Path, classes: list, val_rep: int):
    """
    Carga features y labels de la sesión.
    Split: rep == val_rep → val, resto → train.
    Solo frames con mano detectada (probs sum > 0).
    """
    X_train, y_train = [], []
    X_val, y_val = [], []
    per_letter_counts = {l: {"train": 0, "val": 0} for l in classes}

    for f in sorted((session_dir / "letters").glob("*.npz")):
        stem = f.stem  # ej "A_rep00"
        letter, rep_str = stem.split("_rep")
        rep = int(rep_str)
        if letter not in classes:
            continue
        y_idx = classes.index(letter)

        d = np.load(f)
        feats = d["features"]
        probs = d["probs"]
        mask = probs.sum(axis=1) > 0
        feats_d = feats[mask]
        if len(feats_d) == 0:
            continue

        bucket = "val" if rep == val_rep else "train"
        if bucket == "val":
            X_val.append(feats_d)
            y_val.extend([y_idx] * len(feats_d))
        else:
            X_train.append(feats_d)
            y_train.extend([y_idx] * len(feats_d))
        per_letter_counts[letter][bucket] += len(feats_d)

    X_train = np.vstack(X_train) if X_train else np.empty((0, 120))
    X_val = np.vstack(X_val) if X_val else np.empty((0, 120))
    y_train = np.array(y_train, dtype=np.int64)
    y_val = np.array(y_val, dtype=np.int64)
    return X_train, y_train, X_val, y_val, per_letter_counts


def baseline_metrics(model, X, y) -> dict:
    """Top-1 y top-5 del modelo base sobre los datos."""
    if len(X) == 0:
        return {"top1": 0.0, "top5": 0.0, "n": 0}
    preds = model.predict(X, verbose=0, batch_size=512)
    top1 = (preds.argmax(axis=1) == y).mean()
    top5_idx = np.argsort(preds, axis=1)[:, -5:]
    top5 = np.any(top5_idx == y[:, None], axis=1).mean()
    return {"top1": float(top1), "top5": float(top5), "n": int(len(y))}


def per_letter_accuracy(model, X, y, classes) -> dict:
    if len(X) == 0:
        return {}
    preds = model.predict(X, verbose=0, batch_size=512).argmax(axis=1)
    out = {}
    for i, letter in enumerate(classes):
        mask = y == i
        if mask.sum() == 0:
            continue
        out[letter] = float((preds[mask] == i).mean())
    return out


def run(args):
    base_dir = Path(args.base_model_dir)
    session_dir = Path(args.session_dir)

    print(f"\nModelo base : {base_dir}")
    print(f"Sesión      : {session_dir}")

    classes = list(np.load(base_dir / "label_classes.npy"))
    norm_mean = np.load(base_dir / "norm_mean.npy")
    norm_std = np.load(base_dir / "norm_std.npy")
    model = tf.keras.models.load_model(str(base_dir / "best_model.keras"))

    X_train_raw, y_train, X_val_raw, y_val, counts = load_session(
        session_dir, classes, args.val_rep,
    )
    print(
        f"\nDatos: train={len(X_train_raw)}  val={len(X_val_raw)}"
        f"  (val_rep={args.val_rep})"
    )

    # Aplicar normalización del modelo base (no recomputamos)
    X_train = (X_train_raw - norm_mean) / norm_std
    X_val = (X_val_raw - norm_mean) / norm_std

    # Métricas pre-fine-tuning
    pre_train = baseline_metrics(model, X_train, y_train)
    pre_val = baseline_metrics(model, X_val, y_val)
    print(
        f"\nPRE-FT:  train top-1={pre_train['top1'] * 100:.2f}%  "
        f"top-5={pre_train['top5'] * 100:.2f}%"
    )
    print(
        f"PRE-FT:  val   top-1={pre_val['top1'] * 100:.2f}%  "
        f"top-5={pre_val['top5'] * 100:.2f}%"
    )

    # Compilar con LR baja
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.SparseTopKCategoricalAccuracy(
                     k=5, name="top5"
                 )],
    )

    # Fine-tuning
    out_dir = (
        Path(args.output_root) /
        f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10,
            restore_best_weights=True, mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=5,
            min_lr=1e-6, mode="max",
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "best_model.keras"),
            monitor="val_accuracy", save_best_only=True, mode="max",
        ),
        tf.keras.callbacks.CSVLogger(str(out_dir / "history.csv")),
    ]

    print(f"\nFine-tuning: lr={args.lr}  epochs<={args.epochs}  "
          f"batch={args.batch_size}\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    # Cargar mejor modelo
    best = tf.keras.models.load_model(str(out_dir / "best_model.keras"))
    post_train = baseline_metrics(best, X_train, y_train)
    post_val = baseline_metrics(best, X_val, y_val)

    # Per-letter en val
    per_letter_pre = per_letter_accuracy(model, X_val, y_val, classes)
    # nota: model ya fue restored a best por EarlyStopping, así que pre/post
    # comparten objeto — usamos la base original para "pre"
    base_model = tf.keras.models.load_model(str(base_dir / "best_model.keras"))
    per_letter_pre = per_letter_accuracy(base_model, X_val, y_val, classes)
    per_letter_post = per_letter_accuracy(best, X_val, y_val, classes)

    print(
        f"\nPOST-FT: train top-1={post_train['top1'] * 100:.2f}%  "
        f"top-5={post_train['top5'] * 100:.2f}%"
    )
    print(
        f"POST-FT: val   top-1={post_val['top1'] * 100:.2f}%  "
        f"top-5={post_val['top5'] * 100:.2f}%"
    )

    # Copiar normalization y classes para que el modelo sea drop-in
    np.save(out_dir / "norm_mean.npy", norm_mean)
    np.save(out_dir / "norm_std.npy", norm_std)
    np.save(out_dir / "label_classes.npy", np.array(classes))

    # model_info.json compatible con evaluate_realworld
    info = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "num_features": 120,
        "num_classes": len(classes),
        "classes": classes,
        "label_names": classes,
        "model_type": "dnn_finetuned",
        "base_model_dir": str(base_dir),
        "session_dir": str(session_dir),
        "val_rep": args.val_rep,
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "epochs_trained": len(history.history["loss"]),
        "lr": args.lr,
        "batch_size": args.batch_size,
        "test_accuracy": post_val["top1"],  # para compat con eval
        "metrics": {
            "pre_finetune": {
                "train": pre_train, "val": pre_val,
            },
            "post_finetune": {
                "train": post_train, "val": post_val,
            },
            "per_letter_val": {
                "pre": per_letter_pre,
                "post": per_letter_post,
            },
        },
        "per_letter_counts": counts,
    }
    with open(out_dir / "model_info.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    # Resumen letra a letra (pre vs post)
    print(
        f"\n{'Letra':<5} {'pre':>8} {'post':>8} {'Δ':>8}  {'n_val':>5}"
    )
    print("-" * 45)
    deltas = []
    for letter in classes:
        if letter not in per_letter_post:
            continue
        pre = per_letter_pre.get(letter, 0)
        post = per_letter_post[letter]
        n = int((y_val == classes.index(letter)).sum())
        delta = post - pre
        deltas.append(delta)
        marker = "↑↑" if delta > 0.3 else ("↑" if delta > 0.05 else (
            "↓" if delta < -0.05 else " "))
        print(
            f"  {letter:<3} {pre * 100:>6.1f}% {post * 100:>6.1f}% "
            f"{delta * 100:>+6.1f}pp  {n:>5}  {marker}"
        )
    print(
        f"\nΔ promedio top-1 val: "
        f"{np.mean(deltas) * 100:+.2f} pp ({len(deltas)} letras)"
    )
    print(f"\nArtefactos: {out_dir}")


def main():
    p = argparse.ArgumentParser(
        description="Fine-tuning del modelo de alfabeto con datos reales"
    )
    p.add_argument("--session-dir", type=Path, required=True)
    p.add_argument("--base-model-dir", type=Path, required=True)
    p.add_argument(
        "--output-root", type=Path,
        default=Path("/data/models/librai-alphabet-v4-ft"),
    )
    p.add_argument("--val-rep", type=int, default=4,
                   help="Rep usada como val (default: 4 = última)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()

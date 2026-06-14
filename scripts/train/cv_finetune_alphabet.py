"""
Cross-validation 5-fold del fine-tuning del modelo de alfabeto.

Itera val_rep de 0..4, entrena un modelo por fold y agrega resultados
para reportar mean ± std por letra y global. Robustifica el número de
accuracy contra la varianza de elegir un solo split.

Uso (dentro del contenedor):
    python scripts/train/cv_finetune_alphabet.py \\
        --session-dir /data/realworld_eval/sessions/20260614_011801_S01 \\
        --base-model-dir /data/models/librai-alphabet-v4/run_20260601_015641
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf


def load_session_grouped_by_rep(session_dir: Path, classes: list):
    """Devuelve {rep: (X, y)} con todas las features de cada rep."""
    grouped = {}
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
        if rep not in grouped:
            grouped[rep] = ([], [])
        grouped[rep][0].append(feats_d)
        grouped[rep][1].extend([y_idx] * len(feats_d))
    return {
        rep: (np.vstack(x), np.array(y, dtype=np.int64))
        for rep, (x, y) in grouped.items()
    }


def evaluate(model, X, y) -> dict:
    if len(X) == 0:
        return {"top1": 0.0, "top5": 0.0, "n": 0}
    preds = model.predict(X, verbose=0, batch_size=512)
    top1 = (preds.argmax(axis=1) == y).mean()
    top5 = np.any(
        np.argsort(preds, axis=1)[:, -5:] == y[:, None], axis=1
    ).mean()
    return {"top1": float(top1), "top5": float(top5), "n": int(len(y))}


def per_letter_acc(model, X, y, classes) -> dict:
    if len(X) == 0:
        return {}
    preds = model.predict(X, verbose=0, batch_size=512).argmax(axis=1)
    return {
        letter: float((preds[y == i] == i).mean())
        for i, letter in enumerate(classes)
        if (y == i).sum() > 0
    }


def train_one_fold(
    base_dir: Path, grouped: dict, classes: list,
    val_rep: int, lr: float, epochs: int, batch_size: int,
    fold_out_dir: Path,
):
    norm_mean = np.load(base_dir / "norm_mean.npy")
    norm_std = np.load(base_dir / "norm_std.npy")
    model = tf.keras.models.load_model(str(base_dir / "best_model.keras"))

    X_train = np.vstack([
        grouped[r][0] for r in grouped if r != val_rep
    ])
    y_train = np.concatenate([
        grouped[r][1] for r in grouped if r != val_rep
    ])
    X_val, y_val = grouped[val_rep]

    X_train = (X_train - norm_mean) / norm_std
    X_val = (X_val - norm_mean) / norm_std

    base_val = evaluate(model, X_val, y_val)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.SparseTopKCategoricalAccuracy(
                     k=5, name="top5"
                 )],
    )

    fold_out_dir.mkdir(parents=True, exist_ok=True)
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
            filepath=str(fold_out_dir / "best_model.keras"),
            monitor="val_accuracy", save_best_only=True, mode="max",
        ),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=0,
    )

    best = tf.keras.models.load_model(str(fold_out_dir / "best_model.keras"))
    post_val = evaluate(best, X_val, y_val)
    post_train = evaluate(best, X_train, y_train)
    pl_pre = per_letter_acc(
        tf.keras.models.load_model(str(base_dir / "best_model.keras")),
        X_val, y_val, classes,
    )
    pl_post = per_letter_acc(best, X_val, y_val, classes)

    np.save(fold_out_dir / "norm_mean.npy", norm_mean)
    np.save(fold_out_dir / "norm_std.npy", norm_std)
    np.save(fold_out_dir / "label_classes.npy", np.array(classes))

    fold_info = {
        "val_rep": val_rep,
        "epochs_trained": len(history.history["loss"]),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "metrics": {
            "base_val": base_val,
            "post_val": post_val,
            "post_train": post_train,
            "per_letter_val_pre": pl_pre,
            "per_letter_val_post": pl_post,
        },
    }
    with open(fold_out_dir / "fold_info.json", "w") as f:
        json.dump(fold_info, f, indent=2, ensure_ascii=False)
    return fold_info


def aggregate(fold_infos, classes):
    """Calcula mean ± std por letra y global a partir de los 5 folds."""
    pre_global = [fi["metrics"]["base_val"]["top1"] for fi in fold_infos]
    post_global = [fi["metrics"]["post_val"]["top1"] for fi in fold_infos]
    post_top5 = [fi["metrics"]["post_val"]["top5"] for fi in fold_infos]
    post_train = [fi["metrics"]["post_train"]["top1"] for fi in fold_infos]

    per_letter_pre = {l: [] for l in classes}
    per_letter_post = {l: [] for l in classes}
    for fi in fold_infos:
        for l, v in fi["metrics"]["per_letter_val_pre"].items():
            per_letter_pre[l].append(v)
        for l, v in fi["metrics"]["per_letter_val_post"].items():
            per_letter_post[l].append(v)

    def ms(xs):
        if not xs:
            return {"mean": 0.0, "std": 0.0, "n": 0}
        return {
            "mean": float(np.mean(xs)),
            "std": float(np.std(xs)),
            "n": len(xs),
        }

    return {
        "global": {
            "pre_val_top1": ms(pre_global),
            "post_val_top1": ms(post_global),
            "post_val_top5": ms(post_top5),
            "post_train_top1": ms(post_train),
            "delta_top1": float(np.mean(post_global) - np.mean(pre_global)),
        },
        "per_letter": {
            l: {
                "pre": ms(per_letter_pre[l]),
                "post": ms(per_letter_post[l]),
            }
            for l in classes
        },
    }


def run(args):
    base_dir = Path(args.base_model_dir)
    session_dir = Path(args.session_dir)
    classes = list(np.load(base_dir / "label_classes.npy"))

    grouped = load_session_grouped_by_rep(session_dir, classes)
    reps = sorted(grouped.keys())
    print(f"Reps disponibles: {reps}")
    print(f"Letras: {len(classes)}\n")

    cv_root = (
        Path(args.output_root) /
        f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    )
    cv_root.mkdir(parents=True, exist_ok=True)

    fold_infos = []
    for val_rep in reps:
        fold_dir = cv_root / f"fold_{val_rep}"
        print(f"=== Fold val_rep={val_rep} ===")
        fi = train_one_fold(
            base_dir, grouped, classes, val_rep,
            args.lr, args.epochs, args.batch_size, fold_dir,
        )
        m = fi["metrics"]
        print(
            f"  base val: {m['base_val']['top1'] * 100:5.2f}%  →  "
            f"post val: {m['post_val']['top1'] * 100:5.2f}%  "
            f"(train: {m['post_train']['top1'] * 100:5.2f}%, "
            f"epochs: {fi['epochs_trained']})"
        )
        fold_infos.append(fi)

    agg = aggregate(fold_infos, classes)
    with open(cv_root / "cv_summary.json", "w") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)

    g = agg["global"]
    print(
        f"\n=== Resumen CV 5-fold ===\n"
        f"  PRE  val top-1: {g['pre_val_top1']['mean'] * 100:5.2f}% "
        f"± {g['pre_val_top1']['std'] * 100:.2f}\n"
        f"  POST val top-1: {g['post_val_top1']['mean'] * 100:5.2f}% "
        f"± {g['post_val_top1']['std'] * 100:.2f}\n"
        f"  POST val top-5: {g['post_val_top5']['mean'] * 100:5.2f}% "
        f"± {g['post_val_top5']['std'] * 100:.2f}\n"
        f"  POST train top-1: {g['post_train_top1']['mean'] * 100:5.2f}% "
        f"± {g['post_train_top1']['std'] * 100:.2f}\n"
        f"  Δ top-1: {g['delta_top1'] * 100:+.2f} pp"
    )

    print(f"\n{'Letra':<5} {'pre (μ±σ)':<18} {'post (μ±σ)':<18} {'n_folds':<7}")
    print("-" * 50)
    for letter in classes:
        d = agg["per_letter"][letter]
        pre = d["pre"]
        post = d["post"]
        print(
            f"  {letter:<3} "
            f"{pre['mean'] * 100:5.1f}% ± {pre['std'] * 100:4.1f}    "
            f"{post['mean'] * 100:5.1f}% ± {post['std'] * 100:4.1f}    "
            f"{post['n']}"
        )

    print(f"\nArtefactos: {cv_root}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session-dir", type=Path, required=True)
    p.add_argument("--base-model-dir", type=Path, required=True)
    p.add_argument(
        "--output-root", type=Path,
        default=Path("/data/models/librai-alphabet-v4-ft-cv"),
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()

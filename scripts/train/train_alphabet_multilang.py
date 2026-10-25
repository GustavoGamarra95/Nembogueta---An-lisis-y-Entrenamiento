"""
Entrena Alicia multilengua: un solo modelo con etiquetas separadas por lengua
("LIBRAS:A", "LSP:A", ...). La misma configuración de mano puede ser letras
distintas según la lengua (ver data/analysis/lsc_vs_libras), así que en
inferencia se elige la lengua y el argmax se restringe a sus clases.

Datos:
  LIBRAS: LibrAI (70/15/15 estratificado, igual que robust) + sesiones del
          operador (rep 0-2 train, 3 val, 4 test; extras enteras a train)
  LSP:    features de LSC70 (Lengua de Señas Colombiana) usadas como
          alfabeto LSP, split por sujeto según index.csv

Cada fuente pesa lo mismo en el gradiente (sample_weight = n_librai / n_fuente).
Augmentation y arquitectura son las de train_alphabet_robust.py.

Uso:
    python scripts/train/train_alphabet_multilang.py \\
        --librai-dir data/processed/librai_alphabet_v4 \\
        --lsp-dir data/processed/lsc70_alphabet \\
        --session-dir data/realworld_eval/sessions/20260614_223425_S02 \\
        --extra-sessions data/realworld_eval/sessions/20260614_011801_S01 \\
        --output-root data/models/alicia-alphabet-multilang
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_alphabet_robust import (  # noqa: E402
    build_model,
    load_librai,
    load_session,
    make_dataset,
)


def load_lsp(root: Path, n_val: int, n_test: int, seed: int):
    """Split por sujeto: ningún sujeto aparece en dos splits."""
    with open(root / "index.csv") as f:
        rows = list(csv.DictReader(f))
    subjects = sorted({r["subject"] for r in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    test_s = set(subjects[:n_test])
    val_s = set(subjects[n_test:n_test + n_val])

    letters = sorted({r["label"] for r in rows})
    splits = {"train": ([], []), "val": ([], []), "test": ([], [])}
    for r in rows:
        split = ("test" if r["subject"] in test_s
                 else "val" if r["subject"] in val_s else "train")
        splits[split][0].append(np.load(root / r["file"]))
        splits[split][1].append(r["label"])
    out = {k: (np.array(xs, dtype=np.float32), np.array(ys))
           for k, (xs, ys) in splits.items()}
    subj = {"train": len(subjects) - n_val - n_test,
            "val": n_val, "test": n_test}
    return out, letters, subj


def to_global(y_letters, lang: str, classes: list) -> np.ndarray:
    return np.array([classes.index(f"{lang}:{l}") for l in y_letters],
                    dtype=np.int64)


def evaluate(model, X, y, lang_mask: np.ndarray) -> dict:
    """top-1 sobre todas las clases y restringido a la lengua."""
    if len(X) == 0:
        return {"n": 0}
    p = model.predict(X, verbose=0, batch_size=512)
    p_lang = np.where(lang_mask[None, :], p, -1.0)
    return {
        "n": int(len(y)),
        "top1_all": float((p.argmax(1) == y).mean()),
        "top1_lang": float((p_lang.argmax(1) == y).mean()),
        "top5_lang": float(np.any(
            np.argsort(p_lang, axis=1)[:, -5:] == y[:, None], axis=1).mean()),
        "pred_other_lang": float((~lang_mask[p.argmax(1)]).mean()),
    }


def per_letter(model, X, y, lang_mask, classes) -> dict:
    p = model.predict(X, verbose=0, batch_size=512)
    pred = np.where(lang_mask[None, :], p, -1.0).argmax(1)
    return {classes[i]: float((pred[y == i] == i).mean())
            for i in np.unique(y)}


def run(args):
    print("\n=== Cargando datos ===")
    libras_letters = [str(c) for c in np.load(
        Path(args.libras_classes_from) / "label_classes.npy")]
    lsp, lsp_letters, lsp_subj = load_lsp(
        args.lsp_dir, args.lsp_val_subjects, args.lsp_test_subjects, args.seed)

    classes = ([f"LIBRAS:{l}" for l in libras_letters]
               + [f"LSP:{l}" for l in lsp_letters])
    masks = {lang: np.array([c.startswith(lang + ":") for c in classes])
             for lang in ("LIBRAS", "LSP")}
    print(f"Clases: {len(classes)} (LIBRAS {len(libras_letters)}, "
          f"LSP {len(lsp_letters)})")

    # LIBRAS: LibrAI + operador (índices locales → globales)
    (Xl_tr, yl_tr), (Xl_va, yl_va), (Xl_te, yl_te) = load_librai(
        args.librai_dir, libras_letters, seed=args.seed)
    sess = load_session(args.session_dir, libras_letters)
    Xo_tr, yo_tr = sess["train"]
    for extra in args.extra_sessions:
        es = load_session(extra, libras_letters)
        for s in ("train", "val", "test"):
            if len(es[s][0]):
                Xo_tr = np.vstack([Xo_tr, es[s][0]])
                yo_tr = np.concatenate([yo_tr, es[s][1]])
    Xo_va, yo_va = sess["val"]
    Xo_te, yo_te = sess["test"]
    # Las clases LIBRAS van primero, así que el índice local ya es el global
    Xp_tr, yp_tr = lsp["train"][0], to_global(lsp["train"][1], "LSP", classes)
    Xp_va, yp_va = lsp["val"][0], to_global(lsp["val"][1], "LSP", classes)
    Xp_te, yp_te = lsp["test"][0], to_global(lsp["test"][1], "LSP", classes)

    sources = {
        "librai": (Xl_tr, yl_tr, Xl_va, yl_va, Xl_te, yl_te),
        "operator": (Xo_tr, yo_tr, Xo_va, yo_va, Xo_te, yo_te),
        "lsp": (Xp_tr, yp_tr, Xp_va, yp_va, Xp_te, yp_te),
    }
    n_ref = len(Xl_tr)
    weights = {k: n_ref / max(len(v[0]), 1) for k, v in sources.items()}
    for k, v in sources.items():
        print(f"{k:9s} train={len(v[0]):6d}  val={len(v[2]):5d}  "
              f"test={len(v[4]):5d}  peso={weights[k]:.2f}")
    print(f"LSP sujetos: {lsp_subj}")

    X_tr = np.vstack([v[0] for v in sources.values()]).astype(np.float32)
    y_tr = np.concatenate([v[1] for v in sources.values()])
    w_tr = np.concatenate([np.full(len(v[0]), weights[k], np.float32)
                           for k, v in sources.items()])
    X_va = np.vstack([v[2] for v in sources.values()]).astype(np.float32)
    y_va = np.concatenate([v[3] for v in sources.values()])
    w_va = np.concatenate([np.full(len(v[2]), weights[k], np.float32)
                           for k, v in sources.items()])

    norm_mean = X_tr.mean(axis=0).astype(np.float32)
    norm_std = (X_tr.std(axis=0) + 1e-6).astype(np.float32)

    def norm(X):
        return (X - norm_mean) / norm_std

    feat_std_tf = tf.constant(np.ones(120, dtype=np.float32))
    train_ds = make_dataset(
        norm(X_tr), y_tr, w_tr, feat_std_tf, augment=True,
        batch_size=args.batch_size, shuffle=True, palm_rot=args.palm_rot_deg,
        jitter=args.jitter_sigma, flip_p=args.flip_prob)
    val_ds = make_dataset(
        norm(X_va), y_va, w_va, feat_std_tf, augment=False, batch_size=512,
        shuffle=False, palm_rot=0, jitter=0, flip_p=0)

    tf.keras.utils.set_random_seed(args.seed)
    model = build_model(120, len(classes), dropout=args.dropout)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        weighted_metrics=["accuracy"],
    )

    out_dir = (args.output_root /
               f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=args.patience,
            restore_best_weights=True, mode="max"),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=7,
            min_lr=1e-6, mode="max"),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "best_model.keras"),
            monitor="val_accuracy", save_best_only=True, mode="max"),
        tf.keras.callbacks.CSVLogger(str(out_dir / "history.csv")),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        callbacks=callbacks, verbose=2)
    best = tf.keras.models.load_model(str(out_dir / "best_model.keras"))

    print("\n=== Evaluación final (test) ===")
    tests = {
        "librai_test": (Xl_te, yl_te, "LIBRAS"),
        "operator_test": (Xo_te, yo_te, "LIBRAS"),
        "lsp_test": (Xp_te, yp_te, "LSP"),
    }
    results = {k: evaluate(best, norm(X), y, masks[lang])
               for k, (X, y, lang) in tests.items()}
    print(f"{'Conjunto':15}{'n':>7}{'top1 todas':>12}{'top1 lengua':>13}"
          f"{'top5 lengua':>13}{'pred. otra lengua':>19}")
    for k, r in results.items():
        print(f"{k:15}{r['n']:>7}{r['top1_all'] * 100:>11.2f}%"
              f"{r['top1_lang'] * 100:>12.2f}%{r['top5_lang'] * 100:>12.2f}%"
              f"{r['pred_other_lang'] * 100:>18.2f}%")

    pl = {
        "LIBRAS_operator": per_letter(best, norm(Xo_te), yo_te,
                                      masks["LIBRAS"], classes),
        "LSP": per_letter(best, norm(Xp_te), yp_te, masks["LSP"], classes),
    }
    for name, d in pl.items():
        print(f"\nPor letra ({name}, restringido a la lengua):")
        print("  " + "  ".join(f"{c.split(':')[1]} {v * 100:.0f}%"
                               for c, v in d.items()))

    np.save(out_dir / "norm_mean.npy", norm_mean)
    np.save(out_dir / "norm_std.npy", norm_std)
    np.save(out_dir / "label_classes.npy", np.array(classes))
    info = {
        "model_name": "Alicia",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "num_features": 120,
        "num_classes": len(classes),
        "classes": classes,
        "label_names": classes,
        "languages": {
            "LIBRAS": [c for c in classes if c.startswith("LIBRAS:")],
            "LSP": [c for c in classes if c.startswith("LSP:")],
        },
        "inference": "elegir lengua y aplicar argmax solo sobre sus clases",
        "model_type": "dnn_robust_multilang",
        "data_provenance": {
            "LIBRAS": ["LibrAI", "sesiones del operador "
                       + ", ".join(p.name for p in
                                   [args.session_dir, *args.extra_sessions])],
            "LSP": [f"LSC70 (Lengua de Señas Colombiana) desde {args.lsp_dir}"],
        },
        "training_data": {
            k: {"train": int(len(v[0])), "val": int(len(v[2])),
                "test": int(len(v[4])), "weight": float(weights[k])}
            for k, v in sources.items()
        },
        "lsp_split_subjects": lsp_subj,
        "augmentation": {
            "palm_rot_deg": args.palm_rot_deg,
            "jitter_sigma": args.jitter_sigma,
            "flip_prob": args.flip_prob,
            "dropout": args.dropout,
        },
        "hyperparameters": {
            "lr": args.lr, "batch_size": args.batch_size,
            "epochs_max": args.epochs, "patience": args.patience,
            "epochs_trained": len(history.history["loss"]),
            "seed": args.seed,
        },
        "results": results,
        "per_letter": pl,
    }
    with open(out_dir / "model_info.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"\nArtefactos: {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--librai-dir", type=Path,
                   default=Path("/data/processed/librai_alphabet_v4"))
    p.add_argument("--lsp-dir", type=Path,
                   default=Path("/data/processed/lsc70_alphabet"))
    p.add_argument("--session-dir", type=Path, required=True,
                   help="Sesión LIBRAS primaria (split por rep)")
    p.add_argument("--extra-sessions", type=Path, nargs="*", default=[],
                   help="Sesiones LIBRAS que van enteras al training")
    p.add_argument("--libras-classes-from", type=Path, default=Path(
        "/data/models/librai-alphabet-robust-v4/run_20260615_001921"),
        help="Modelo del que se toma la lista de letras LIBRAS")
    p.add_argument("--lsp-val-subjects", type=int, default=10)
    p.add_argument("--lsp-test-subjects", type=int, default=10)
    p.add_argument("--output-root", type=Path,
                   default=Path("/data/models/alicia-alphabet-multilang"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--palm-rot-deg", type=float, default=20.0)
    p.add_argument("--jitter-sigma", type=float, default=0.5)
    p.add_argument("--flip-prob", type=float, default=0.5)
    run(p.parse_args())


if __name__ == "__main__":
    main()

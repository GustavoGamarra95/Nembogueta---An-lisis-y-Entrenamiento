"""
Análisis comprehensivo de modelos del alfabeto.

Para cada modelo provisto produce:
  1. Curvas de aprendizaje (loss/accuracy por epoch desde history.csv)
  2. Matriz de confusión sobre la sesión de test
  3. Robustez al ruido (degradación bajo perturbaciones gaussianas)
  4. Sanitización de ruidos (efecto de filtrar por confianza)

Y produce tablas/plots cross-modelo:
  5. Comparación per-letra de top-1 y top-5
  6. Curva consolidada de robustez al ruido

Uso (dentro del contenedor):
    python scripts/evaluate/comprehensive_analysis.py \\
        --models v1:/data/models/librai-alphabet-robust/run_X \\
                 v2:/data/models/librai-alphabet-robust-v2/run_Y \\
                 v3:/data/models/librai-alphabet-robust-v3/run_Z \\
                 v4:/data/models/librai-alphabet-robust-v4/run_W \\
        --test-session /data/realworld_eval/sessions/20260614_223425_S02 \\
        --output-dir /data/analysis/robust_models
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


# ------------------------------------------------------------ #
# Carga
# ------------------------------------------------------------ #

def load_model_bundle(model_dir: Path):
    nm = np.load(model_dir / "norm_mean.npy")
    ns = np.load(model_dir / "norm_std.npy")
    classes = list(np.load(model_dir / "label_classes.npy"))
    model = tf.keras.models.load_model(str(model_dir / "best_model.keras"))
    history_csv = model_dir / "history.csv"
    history = pd.read_csv(history_csv) if history_csv.exists() else None
    with open(model_dir / "model_info.json") as f:
        info = json.load(f)
    return {
        "model": model, "norm_mean": nm, "norm_std": ns,
        "classes": classes, "history": history, "info": info,
        "dir": model_dir,
    }


def load_session_features(session_dir: Path, classes: list):
    """Devuelve (X, y, letter_array) usando solo frames con mano detectada."""
    X, y, L = [], [], []
    for f in sorted((session_dir / "letters").glob("*.npz")):
        letter = f.stem.split("_rep")[0]
        if letter not in classes:
            continue
        d = np.load(f)
        mask = d["probs"].sum(axis=1) > 0
        feats = d["features"][mask]
        if len(feats) == 0:
            continue
        idx = classes.index(letter)
        X.append(feats)
        y.extend([idx] * len(feats))
        L.extend([letter] * len(feats))
    return np.vstack(X), np.array(y), np.array(L)


# ------------------------------------------------------------ #
# 1. Curvas de aprendizaje
# ------------------------------------------------------------ #

def plot_learning_curves(bundles, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for tag, b in bundles.items():
        h = b["history"]
        if h is None or "loss" not in h.columns:
            continue
        epochs = np.arange(1, len(h) + 1)
        axes[0].plot(epochs, h["loss"], label=f"{tag} train", linestyle="-")
        if "val_loss" in h.columns:
            axes[0].plot(
                epochs, h["val_loss"], label=f"{tag} val", linestyle="--"
            )
        if "accuracy" in h.columns:
            axes[1].plot(
                epochs, h["accuracy"], label=f"{tag} train", linestyle="-"
            )
        if "val_accuracy" in h.columns:
            axes[1].plot(
                epochs, h["val_accuracy"], label=f"{tag} val", linestyle="--"
            )
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle("Curvas de aprendizaje — modelos robust")
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curves.png", dpi=120)
    plt.close(fig)
    print(f"  → learning_curves.png")


# ------------------------------------------------------------ #
# 2. Matrices de confusión
# ------------------------------------------------------------ #

def confusion_plot(cm, classes, title, path):
    fig, ax = plt.subplots(figsize=(11, 9))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_n = np.divide(
        cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0
    )
    im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    # números en las celdas significativas
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cm[i, j]
            if v > 0 and (i == j or v > row_sums[i, 0] * 0.05):
                color = "white" if cm_n[i, j] > 0.5 else "black"
                ax.text(j, i, str(int(v)), ha="center", va="center",
                        color=color, fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def compute_predictions(bundle, X, y):
    Xn = (X - bundle["norm_mean"]) / bundle["norm_std"]
    probs = bundle["model"].predict(Xn, verbose=0, batch_size=512)
    preds = probs.argmax(axis=1)
    return probs, preds


def plot_confusion_matrices(bundles, X, y, classes, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for tag, b in bundles.items():
        _, preds = compute_predictions(b, X, y)
        cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
        for t, p in zip(y, preds):
            cm[t, p] += 1
        title = f"{tag} — matriz de confusión (S02 full session)"
        confusion_plot(cm, classes, title, out_dir / f"cm_{tag}.png")
        np.save(out_dir / f"cm_{tag}.npy", cm)
        print(f"  → cm_{tag}.png ({cm.sum()} frames)")


# ------------------------------------------------------------ #
# 3. Robustez al ruido (ofuscación)
# ------------------------------------------------------------ #

def noise_robustness(bundles, X, y, out_dir):
    """
    Agrega ruido gaussiano de magnitud creciente al input normalizado.
    Mide accuracy en cada nivel. Modelos más robustos degradan menos.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sigmas = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5])
    n_runs = 3  # promediar para reducir varianza del ruido

    results = {}
    fig, ax = plt.subplots(figsize=(10, 6))
    for tag, b in bundles.items():
        Xn = (X - b["norm_mean"]) / b["norm_std"]
        accs = []
        for sigma in sigmas:
            run_accs = []
            for _ in range(n_runs):
                if sigma == 0:
                    Xp = Xn
                else:
                    Xp = Xn + np.random.normal(0, sigma, Xn.shape).astype(
                        np.float32
                    )
                preds = b["model"].predict(
                    Xp, verbose=0, batch_size=512
                ).argmax(axis=1)
                run_accs.append((preds == y).mean())
            accs.append(np.mean(run_accs))
        results[tag] = {"sigmas": sigmas.tolist(), "accuracies": accs}
        ax.plot(sigmas, accs, marker="o", label=tag)

    ax.set_xlabel("σ del ruido gaussiano en features normalizadas")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title(
        "Robustez al ruido — degradación bajo perturbación gaussiana\n"
        "Curvas más planas = modelo más robusto"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "noise_robustness.png", dpi=120)
    plt.close(fig)

    with open(out_dir / "noise_robustness.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  → noise_robustness.png + .json")


# ------------------------------------------------------------ #
# 4. Sanitización de ruidos (filtrado por confianza)
# ------------------------------------------------------------ #

def confidence_sanitization(bundles, X, y, out_dir):
    """
    Para cada threshold de confianza, descarta frames donde el modelo
    no está confiado y reporta:
      - Accuracy sobre los frames retenidos
      - Cobertura (% de frames retenidos)
    Trade-off curva accuracy vs cobertura.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = np.linspace(0.0, 0.95, 20)

    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for tag, b in bundles.items():
        probs, preds = compute_predictions(b, X, y)
        conf = probs.max(axis=1)
        accs, covers = [], []
        for t in thresholds:
            mask = conf >= t
            if mask.sum() == 0:
                accs.append(np.nan)
                covers.append(0)
                continue
            accs.append(float((preds[mask] == y[mask]).mean()))
            covers.append(float(mask.mean()))
        results[tag] = {
            "thresholds": thresholds.tolist(),
            "accuracies": accs,
            "coverages": covers,
        }
        axes[0].plot(thresholds, accs, marker="o", label=tag)
        axes[1].plot(covers, accs, marker="o", label=tag)

    axes[0].set_xlabel("Umbral de confianza mínima")
    axes[0].set_ylabel("Top-1 accuracy sobre frames retenidos")
    axes[0].set_title("Accuracy vs umbral de filtrado")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Cobertura (fracción de frames retenidos)")
    axes[1].set_ylabel("Top-1 accuracy")
    axes[1].set_title("Trade-off cobertura vs accuracy")
    axes[1].invert_xaxis()
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle("Sanitización por filtrado de baja confianza")
    fig.tight_layout()
    fig.savefig(out_dir / "confidence_sanitization.png", dpi=120)
    plt.close(fig)

    with open(out_dir / "confidence_sanitization.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  → confidence_sanitization.png + .json")


# ------------------------------------------------------------ #
# 5. Tabla cross-modelo per-letra
# ------------------------------------------------------------ #

def per_letter_table(bundles, X, y, classes, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for letter_idx, letter in enumerate(classes):
        mask = y == letter_idx
        if mask.sum() == 0:
            continue
        row = {"letter": letter, "n": int(mask.sum())}
        for tag, b in bundles.items():
            _, preds = compute_predictions(b, X, y)
            row[tag] = float((preds[mask] == letter_idx).mean())
        rows.append(row)

    # globales
    global_row = {"letter": "GLOBAL", "n": int(len(y))}
    for tag, b in bundles.items():
        _, preds = compute_predictions(b, X, y)
        global_row[tag] = float((preds == y).mean())
    rows.append(global_row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_letter_comparison.csv", index=False)

    # heatmap
    plot_df = df[df["letter"] != "GLOBAL"]
    tags = list(bundles.keys())
    fig, ax = plt.subplots(figsize=(8, max(6, len(plot_df) * 0.3)))
    arr = plot_df[tags].values
    im = ax.imshow(arr, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["letter"].tolist())
    ax.set_title("Top-1 accuracy por letra y modelo (S02 full session)")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(
                j, i, f"{arr[i, j] * 100:.0f}",
                ha="center", va="center",
                color="white" if arr[i, j] < 0.5 else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / "per_letter_heatmap.png", dpi=120)
    plt.close(fig)
    print("  → per_letter_comparison.csv + per_letter_heatmap.png")
    return df


# ------------------------------------------------------------ #
# 6. Tabla resumen global
# ------------------------------------------------------------ #

def summary_table(bundles, X, y, classes, out_dir):
    rows = []
    for tag, b in bundles.items():
        probs, preds = compute_predictions(b, X, y)
        top5 = np.argsort(probs, axis=1)[:, -5:]
        top1 = (preds == y).mean()
        top5_acc = np.any(top5 == y[:, None], axis=1).mean()
        # confianza promedio en target
        target_conf = probs[np.arange(len(y)), y].mean()
        # entropía promedio de las distribuciones (medida de "indecisión")
        entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1).mean()
        rows.append({
            "model": tag,
            "top1_S02": top1,
            "top5_S02": top5_acc,
            "mean_target_confidence": float(target_conf),
            "mean_entropy": float(entropy),
            "test_acc_holdout": b["info"].get("test_accuracy", None),
            "epochs_trained": b["info"].get("hyperparameters", {}).get(
                "epochs_trained", None,
            ) or b["info"].get("epochs_trained"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary_table.csv", index=False)
    print("  → summary_table.csv")
    return df


# ------------------------------------------------------------ #
# Main
# ------------------------------------------------------------ #

def parse_model_spec(specs):
    out = {}
    for s in specs:
        if ":" not in s:
            raise ValueError(f"--models requiere formato tag:path, got {s}")
        tag, p = s.split(":", 1)
        out[tag] = Path(p)
    return out


def run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Cargando modelos ===")
    model_paths = parse_model_spec(args.models)
    bundles = {tag: load_model_bundle(p) for tag, p in model_paths.items()}
    for tag, b in bundles.items():
        print(f"  {tag}: {b['dir'].name} ({b['info']['num_features']} feat)")
    classes = bundles[list(bundles.keys())[0]]["classes"]

    print("\n=== Cargando sesión de test ===")
    X, y, L = load_session_features(Path(args.test_session), classes)
    print(f"  {Path(args.test_session).name}: {len(X)} frames")

    print("\n=== 1. Curvas de aprendizaje ===")
    plot_learning_curves(bundles, out_dir)

    print("\n=== 2. Matrices de confusión ===")
    plot_confusion_matrices(bundles, X, y, classes, out_dir)

    print("\n=== 3. Robustez al ruido ===")
    noise_robustness(bundles, X, y, out_dir)

    print("\n=== 4. Sanitización por confianza ===")
    confidence_sanitization(bundles, X, y, out_dir)

    print("\n=== 5. Per-letra cross-modelo ===")
    per_letter_table(bundles, X, y, classes, out_dir)

    print("\n=== 6. Resumen global ===")
    summary = summary_table(bundles, X, y, classes, out_dir)
    print(summary.to_string(index=False))

    print(f"\nArtefactos en: {out_dir}")


def main():
    p = argparse.ArgumentParser(
        description="Análisis comprehensivo cross-modelo del alfabeto",
    )
    p.add_argument(
        "--models", nargs="+", required=True,
        help="Lista de tag:ruta, ej v1:/data/models/.../run_X",
    )
    p.add_argument("--test-session", type=Path, required=True)
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("/data/analysis/robust_models"),
    )
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()

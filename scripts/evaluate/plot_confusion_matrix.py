"""
Genera gráfico de matriz de confusión para cualquier modelo del proyecto.

Uso:
    python scripts/evaluate/plot_confusion_matrix.py \
        --run-dir /data/models/librai-alphabet-v2/run_20260510_024606

    # Comparar dos modelos (viejo vs nuevo):
    python scripts/evaluate/plot_confusion_matrix.py \
        --run-dir /data/models/librai-alphabet-v2/run_20260510_024606 \
        --compare-dir /data/models/librai-alphabet
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns


def load_run(run_dir: Path):
    cm      = np.load(run_dir / "confusion_matrix.npy")
    classes = np.load(run_dir / "label_classes.npy")
    report  = json.loads((run_dir / "classification_report.json").read_text())
    return cm, classes, report


def plot_confusion_matrix(
    ax,
    cm: np.ndarray,
    classes: np.ndarray,
    title: str,
    show_values: bool = True,
):
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    mask_diag = np.eye(len(classes), dtype=bool)

    # Fondo: diagonal en azul claro, errores en rojo
    sns.heatmap(
        cm_norm,
        ax=ax,
        cmap="Blues",
        annot=False,
        linewidths=0.4,
        linecolor="#cccccc",
        cbar=False,
        xticklabels=classes,
        yticklabels=classes,
        vmin=0,
        vmax=1,
    )

    # Resaltar celdas con errores (fuera de la diagonal)
    error_mask = ~mask_diag & (cm > 0)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if error_mask[i, j]:
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1,
                    fill=True, color="#ff4444", alpha=min(1.0, cm_norm[i, j] * 8 + 0.3),
                    zorder=2,
                ))

    # Anotar solo celdas relevantes
    if show_values:
        for i in range(len(classes)):
            for j in range(len(classes)):
                if i == j:
                    # Diagonal: porcentaje de acierto
                    pct = cm_norm[i, j] * 100
                    color = "white" if pct > 60 else "black"
                    ax.text(j + 0.5, i + 0.5, f"{pct:.0f}%",
                            ha="center", va="center", fontsize=7,
                            color=color, fontweight="bold")
                elif cm[i, j] > 0:
                    # Error: cantidad absoluta
                    ax.text(j + 0.5, i + 0.5, str(int(cm[i, j])),
                            ha="center", va="center", fontsize=7,
                            color="white", fontweight="bold")

    ax.set_title(title, fontsize=13, pad=10, fontweight="bold")
    ax.set_xlabel("Predicho", fontsize=10, labelpad=6)
    ax.set_ylabel("Real", fontsize=10, labelpad=6)
    ax.tick_params(axis="both", labelsize=9)
    plt.setp(ax.get_xticklabels(), rotation=0)
    plt.setp(ax.get_yticklabels(), rotation=0)


def plot_per_class_f1(ax, report: dict, classes: np.ndarray, title: str):
    f1_scores = [report.get(c, {}).get("f1-score", 0) * 100 for c in classes]
    colors = ["#e74c3c" if f < 99 else "#2ecc71" for f in f1_scores]

    bars = ax.bar(classes, f1_scores, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylim(95, 101)
    ax.axhline(100, color="#aaaaaa", linewidth=0.8, linestyle="--")
    ax.axhline(99,  color="#e74c3c", linewidth=0.8, linestyle=":", alpha=0.6)

    for bar, score in zip(bars, f1_scores):
        if score < 100:
            ax.text(bar.get_x() + bar.get_width() / 2, score + 0.05,
                    f"{score:.1f}", ha="center", va="bottom", fontsize=7,
                    color="#e74c3c", fontweight="bold")

    ax.set_title(title, fontsize=11, pad=8, fontweight="bold")
    ax.set_ylabel("F1-Score (%)", fontsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir",     required=True, help="Directorio del run a visualizar")
    parser.add_argument("--compare-dir", default=None,  help="Run anterior para comparar")
    parser.add_argument("--output",      default=None,  help="Ruta de salida del PNG")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cm, classes, report = load_run(run_dir)

    accuracy = report.get("accuracy", 0) * 100
    n_errors = int(cm.sum() - np.trace(cm))

    compare = args.compare_dir is not None
    if compare:
        cmp_dir = Path(args.compare_dir)
        cm_old, classes_old, report_old = load_run(cmp_dir)
        acc_old = report_old.get("accuracy", 0) * 100
        n_errors_old = int(cm_old.sum() - np.trace(cm_old))

    # --- Layout ---
    if compare:
        fig = plt.figure(figsize=(20, 16))
        gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35,
                                top=0.90, bottom=0.06, left=0.06, right=0.97)
        ax_cm_old = fig.add_subplot(gs[0, 0])
        ax_cm_new = fig.add_subplot(gs[0, 1])
        ax_f1_old = fig.add_subplot(gs[1, 0])
        ax_f1_new = fig.add_subplot(gs[1, 1])

        plot_confusion_matrix(ax_cm_old, cm_old, classes_old,
                              f"Modelo anterior — {acc_old:.2f}% acc  ({n_errors_old} errores)")
        plot_confusion_matrix(ax_cm_new, cm,     classes,
                              f"Modelo nuevo (Z features) — {accuracy:.2f}% acc  ({n_errors} errores)")
        plot_per_class_f1(ax_f1_old, report_old, classes_old, "F1 por letra — modelo anterior")
        plot_per_class_f1(ax_f1_new, report,     classes,     "F1 por letra — modelo nuevo")

        fig.suptitle(
            "Comparación de Matrices de Confusión — LIBRAS Alfabeto\n"
            "208 features (xy)  vs  280 features (xyz + finger curl)",
            fontsize=14, fontweight="bold", y=0.96,
        )
    else:
        fig = plt.figure(figsize=(14, 12))
        gs  = gridspec.GridSpec(2, 1, hspace=0.45,
                                top=0.90, bottom=0.06, left=0.06, right=0.97)
        ax_cm = fig.add_subplot(gs[0])
        ax_f1 = fig.add_subplot(gs[1])

        plot_confusion_matrix(ax_cm, cm, classes,
                              f"Matriz de Confusión — {accuracy:.2f}% acc  ({n_errors} errores)")
        plot_per_class_f1(ax_f1, report, classes, "F1-Score por letra")

        feat_count = 280 if "v2" in str(run_dir) else 208
        fig.suptitle(
            f"Modelo LIBRAS Alfabeto — {feat_count} features\n"
            f"{len(classes)} clases  |  {accuracy:.2f}% accuracy  |  {n_errors} errores totales",
            fontsize=14, fontweight="bold", y=0.96,
        )

    # Leyenda de colores
    legend_txt = (
        "Azul: aciertos (% por fila)  |  "
        "Rojo: errores (cantidad)  |  "
        "Barra roja en F1: < 99%"
    )
    fig.text(0.5, 0.01, legend_txt, ha="center", fontsize=9, color="#555555")

    out = args.output or str(run_dir / "confusion_matrix_plot.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Gráfico guardado en: {out}")
    plt.close()


if __name__ == "__main__":
    main()

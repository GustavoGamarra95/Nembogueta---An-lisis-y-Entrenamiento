"""
Evaluación del modelo UFPR LIBRAS-HC-RGBDS (handshape).
Reproduce el mismo split del entrenamiento (StratifiedShuffleSplit, random_state=42)
y calcula Accuracy, Precision, Recall/Sensitivity, Specificity y F1-Score.

Uso:
    python scripts/evaluate/evaluate_ufpr.py \
        --model-dir /data/models/ufpr-handshape \
        --data-dir  /data/processed/ufpr-handshape \
        --output-dir /data/models/ufpr-handshape/evaluation
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ufpr_evaluation.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def load_dataset(data_dir: Path):
    """Mean-pool cada .npy (frames → vector estático) y devuelve X, y, class_names."""
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"No se encontraron subdirectorios de clase en {data_dir}")

    X_list, y_list = [], []
    class_names = [d.name for d in class_dirs]

    for class_idx, class_dir in enumerate(class_dirs):
        npy_files = sorted(class_dir.glob("*.npy"))
        for npy_path in npy_files:
            try:
                seq = np.load(npy_path).astype(np.float32)
                X_list.append(seq.mean(axis=0))
                y_list.append(class_idx)
            except Exception as e:
                logger.warning(f"Error cargando {npy_path}: {e}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    logger.info(f"Dataset cargado: {X.shape[0]} muestras, {len(class_names)} clases, {X.shape[1]} features")
    return X, y, class_names


def get_test_split(X, y, test_size=0.15, val_size=0.15):
    """Reproduce el split de train_ufpr.py (mismo random_state=42)."""
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_val_idx, test_idx = next(sss_test.split(X, y))

    val_fraction = val_size / (1 - test_size)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=42)
    train_idx, val_idx = next(sss_val.split(X[train_val_idx], y[train_val_idx]))

    X_test, y_test = X[test_idx], y[test_idx]
    logger.info(f"Test set: {len(X_test)} muestras")
    return X_test, y_test


def compute_specificity_per_class(cm):
    specificities = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(float(spec))
    return specificities


def evaluate(model_dir: Path, data_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar metadata
    with open(model_dir / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)

    class_names = meta["class_names"]
    logger.info(f"Modelo: {meta['dataset']} — {meta['n_classes']} clases")
    logger.info(f"Test accuracy reportada en entrenamiento: {meta.get('test_accuracy', '?'):.4f}")

    # Cargar normalización (solo si fue guardada durante el entrenamiento)
    norm_mean_path = model_dir / "norm_mean.npy"
    norm_std_path  = model_dir / "norm_std.npy"
    if norm_mean_path.exists() and norm_std_path.exists():
        norm_mean = np.load(norm_mean_path)
        norm_std  = np.load(norm_std_path)
        logger.info("Normalización cargada desde norm_mean.npy / norm_std.npy")
    else:
        norm_mean = None
        norm_std  = None
        logger.info("Sin normalización guardada — se usarán features brutos")

    # Cargar modelo
    model_path = model_dir / "best_model.keras"
    if not model_path.exists():
        model_path = model_dir / "final_model.keras"
    logger.info(f"Cargando modelo desde {model_path}...")
    model = tf.keras.models.load_model(str(model_path), compile=False)

    # Cargar y dividir datos
    X, y, _ = load_dataset(data_dir)

    # Aplicar normalización si está disponible
    if norm_mean is not None:
        X = (X - norm_mean) / (norm_std + 1e-8)

    X_test, y_test = get_test_split(X, y)

    # Predicciones
    logger.info("Realizando predicciones...")
    y_pred_probs = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Métricas
    accuracy         = accuracy_score(y_test, y_pred)
    precision_macro  = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro     = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro         = f1_score(y_test, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    specificities    = compute_specificity_per_class(cm)
    specificity_macro = float(np.mean(specificities))

    # Top-5 accuracy
    top5_correct = sum(
        1 for i in range(len(y_test))
        if y_test[i] in np.argsort(y_pred_probs[i])[-5:]
    )
    top5_acc = top5_correct / len(y_test)

    # Resumen
    logger.info("\n" + "=" * 80)
    logger.info("RESUMEN DE EVALUACIÓN - UFPR LIBRAS-HC-RGBDS (Handshape)")
    logger.info("=" * 80)
    logger.info(f"Test Samples:                  {len(X_test)}")
    logger.info(f"Clases:                        {len(class_names)}")
    logger.info(f"Accuracy:                      {accuracy*100:.2f}%")
    logger.info(f"Top-5 Accuracy:                {top5_acc*100:.2f}%")
    logger.info(f"Precision (macro):             {precision_macro*100:.2f}%")
    logger.info(f"Recall / Sensitivity (macro):  {recall_macro*100:.2f}%")
    logger.info(f"Specificity (macro):           {specificity_macro*100:.2f}%")
    logger.info(f"F1-Score (macro):              {f1_macro*100:.2f}%")
    logger.info("=" * 80 + "\n")

    # Guardar JSON
    summary = {
        "dataset": "UFPR LIBRAS-HC-RGBDS",
        "test_samples": int(len(X_test)),
        "n_classes": len(class_names),
        "accuracy": float(accuracy),
        "top5_accuracy": float(top5_acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "specificity_macro": specificity_macro,
        "f1_macro": float(f1_macro),
    }
    with open(output_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Classification report
    report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    with open(output_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    # Matriz de confusión
    np.save(output_dir / "confusion_matrix.npy", cm)
    _plot_confusion_matrix(cm, class_names, output_dir)

    logger.info(f"Evaluación completa guardada en {output_dir}")


def _plot_confusion_matrix(cm, class_names, output_dir):
    n = len(class_names)
    fig_size = max(14, n // 3)
    cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

    plt.figure(figsize=(fig_size, fig_size - 2))
    sns.heatmap(
        cm_norm, annot=(n <= 30), fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={"label": "Proporción"},
    )
    plt.title("Matriz de Confusión - UFPR Handshape (Normalizada)", fontsize=14, pad=16)
    plt.ylabel("Clase Verdadera", fontsize=11)
    plt.xlabel("Clase Predicha", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix_normalized.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Matriz de confusión guardada")


def main():
    parser = argparse.ArgumentParser(description="Evalúa el modelo UFPR handshape")
    parser.add_argument("--model-dir", type=Path, default=Path("/data/models/ufpr-handshape"))
    parser.add_argument("--data-dir",  type=Path, default=Path("/data/processed/ufpr-handshape"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data/models/ufpr-handshape/evaluation"))
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    if not args.gpu:
        tf.config.set_visible_devices([], "GPU")

    evaluate(args.model_dir, args.data_dir, args.output_dir)
    logger.info("¡Evaluación completada!")


if __name__ == "__main__":
    main()

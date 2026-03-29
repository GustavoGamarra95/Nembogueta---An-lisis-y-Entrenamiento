"""
Script de evaluación para modelos V-LIBRASIL.
Evalúa un modelo entrenado usando el test set y genera métricas detalladas.

Uso:
    python scripts/evaluate_vlibrasil.py --model /models/vlibrasil/run_xxx/best_model.h5 \
                                         --model-info /models/vlibrasil/run_xxx/model_info.json \
                                         --data-dir /app/data/processed/v-librasil-flat \
                                         --output-dir /models/vlibrasil/run_xxx/evaluation
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
    top_k_accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vlibrasil_evaluation.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_processed_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Carga los datos procesados.

    Args:
        data_dir: Directorio con datos procesados

    Returns:
        Tupla (X, y, label_names)
    """
    logger.info(f"Cargando datos desde {data_dir}...")

    sequences = []
    labels = []

    npy_files = list(data_dir.glob("*.npy"))

    if not npy_files:
        raise FileNotFoundError(f"No se encontraron archivos .npy en {data_dir}")

    logger.info(f"Encontrados {len(npy_files)} archivos procesados")

    for npy_file in npy_files:
        try:
            sequence = np.load(npy_file)

            # Extraer etiqueta del nombre
            filename = npy_file.stem
            parts = filename.split('_')

            label_parts = []
            for part in parts:
                if part.startswith('Articulador') or part == 'processed':
                    break
                label_parts.append(part)

            label = '_'.join(label_parts) if label_parts else filename

            sequences.append(sequence)
            labels.append(label)

        except Exception as e:
            logger.warning(f"Error cargando {npy_file}: {e}")
            continue

    if not sequences:
        raise ValueError("No se pudieron cargar secuencias válidas")

    # Convertir a arrays
    X = np.array(sequences, dtype=np.float32)

    # Normalizar
    X_mean = np.mean(X, axis=(0, 1), keepdims=True)
    X_std = np.std(X, axis=(0, 1), keepdims=True) + 1e-8
    X = (X - X_mean) / X_std

    # Codificar labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    label_names = label_encoder.classes_.tolist()

    logger.info(f"Datos cargados: {X.shape[0]} muestras")
    logger.info(f"Shape: {X.shape}")
    logger.info(f"Clases: {len(label_names)}")

    return X, y, label_names


def get_test_set(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.15
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrae el test set usando el mismo split que en entrenamiento.

    Args:
        X: Datos completos
        y: Labels completos
        test_size: Proporción del test set

    Returns:
        Tupla (X_test, y_test)
    """
    # Mismo proceso que en train_vlibrasil.py
    min_samples_per_class = np.min(np.bincount(y))
    use_stratify = min_samples_per_class >= 3

    if use_stratify:
        logger.info("Usando split estratificado")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    else:
        logger.info(f"Split no estratificado (min_samples={min_samples_per_class})")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    logger.info(f"Test set: {len(X_test)} muestras")
    return X_test, y_test


def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: List[str],
    output_dir: Path
):
    """
    Evalúa el modelo y genera métricas detalladas.

    Args:
        model: Modelo a evaluar
        X_test: Datos de test
        y_test: Labels de test
        label_names: Nombres de las clases
        output_dir: Directorio para guardar resultados
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Realizando predicciones en test set...")
    y_pred_probs = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Métricas básicas
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

    # Top-k accuracy
    top_3_acc = top_k_accuracy_score(y_test, y_pred_probs, k=3)
    top_5_acc = top_k_accuracy_score(y_test, y_pred_probs, k=5)

    # Imprimir resumen
    logger.info("\n" + "="*80)
    logger.info("RESUMEN DE EVALUACIÓN")
    logger.info("="*80)
    logger.info(f"Test Samples: {len(X_test)}")
    logger.info(f"Accuracy: {accuracy*100:.2f}%")
    logger.info(f"Top-3 Accuracy: {top_3_acc*100:.2f}%")
    logger.info(f"Top-5 Accuracy: {top_5_acc*100:.2f}%")
    logger.info(f"Precision (macro): {precision_macro*100:.2f}%")
    logger.info(f"Recall (macro): {recall_macro*100:.2f}%")
    logger.info(f"F1-Score (macro): {f1_macro*100:.2f}%")
    logger.info("="*80 + "\n")

    # Guardar métricas resumidas
    summary_metrics = {
        'test_samples': int(len(X_test)),
        'accuracy': float(accuracy),
        'top_3_accuracy': float(top_3_acc),
        'top_5_accuracy': float(top_5_acc),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro)
    }

    with open(output_dir / 'summary_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(summary_metrics, f, indent=2, ensure_ascii=False)

    # Classification report detallado
    logger.info("Generando classification report...")
    report = classification_report(
        y_test, y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0
    )

    with open(output_dir / 'classification_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Guardar también en formato texto
    report_text = classification_report(
        y_test, y_pred,
        target_names=label_names,
        zero_division=0
    )
    with open(output_dir / 'classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    # Matriz de confusión
    logger.info("Generando matriz de confusión...")
    cm = confusion_matrix(y_test, y_pred)
    np.save(output_dir / 'confusion_matrix.npy', cm)

    # Visualizar matriz de confusión (solo primeras 50 clases para legibilidad)
    if len(label_names) <= 50:
        plot_confusion_matrix(cm, label_names, output_dir)
    else:
        # Para muchas clases, visualizar versión simplificada
        plot_confusion_matrix_simplified(cm, output_dir)

    # Análisis de errores
    logger.info("Analizando errores...")
    analyze_errors(y_test, y_pred, y_pred_probs, label_names, output_dir)

    # Distribución de confianza
    plot_confidence_distribution(y_pred_probs, y_test, y_pred, output_dir)

    # Análisis por clase
    analyze_per_class_performance(y_test, y_pred, label_names, output_dir)

    logger.info(f"\nEvaluación completa guardada en {output_dir}")


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], output_dir: Path):
    """Visualiza matriz de confusión completa."""
    plt.figure(figsize=(20, 16))
    sns.heatmap(
        cm, annot=False, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels
    )
    plt.title('Matriz de Confusión', fontsize=16)
    plt.ylabel('Verdadero', fontsize=12)
    plt.xlabel('Predicho', fontsize=12)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Matriz de confusión guardada en {output_dir / 'confusion_matrix.png'}")


def plot_confusion_matrix_simplified(cm: np.ndarray, output_dir: Path):
    """Visualiza matriz de confusión simplificada (agregada)."""
    # Normalizar por filas
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))
    plt.imshow(cm_normalized, cmap='Blues', aspect='auto')
    plt.colorbar(label='Proporción')
    plt.title('Matriz de Confusión Normalizada (todas las clases)', fontsize=14)
    plt.ylabel('Clase Verdadera', fontsize=12)
    plt.xlabel('Clase Predicha', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_normalized.png', dpi=150)
    plt.close()
    logger.info(f"Matriz normalizada guardada en {output_dir / 'confusion_matrix_normalized.png'}")


def analyze_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_probs: np.ndarray,
    labels: List[str],
    output_dir: Path
):
    """Analiza los errores de predicción."""
    errors = []

    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            errors.append({
                'sample_idx': int(i),
                'true_label': labels[y_true[i]],
                'predicted_label': labels[y_pred[i]],
                'confidence': float(y_pred_probs[i][y_pred[i]]),
                'true_label_confidence': float(y_pred_probs[i][y_true[i]])
            })

    # Ordenar por confianza descendente
    errors.sort(key=lambda x: x['confidence'], reverse=True)

    # Guardar errores
    with open(output_dir / 'error_analysis.json', 'w', encoding='utf-8') as f:
        json.dump({
            'total_errors': len(errors),
            'error_rate': len(errors) / len(y_true),
            'errors': errors[:100]  # Top 100 errores
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"Errores totales: {len(errors)} ({len(errors)/len(y_true)*100:.2f}%)")


def plot_confidence_distribution(
    y_pred_probs: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path
):
    """Visualiza distribución de confianza en predicciones."""
    correct_mask = y_true == y_pred
    incorrect_mask = ~correct_mask

    correct_confidences = np.max(y_pred_probs[correct_mask], axis=1)
    incorrect_confidences = np.max(y_pred_probs[incorrect_mask], axis=1)

    plt.figure(figsize=(10, 6))
    plt.hist(correct_confidences, bins=50, alpha=0.5, label='Correctas', color='green')
    plt.hist(incorrect_confidences, bins=50, alpha=0.5, label='Incorrectas', color='red')
    plt.xlabel('Confianza de Predicción', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.title('Distribución de Confianza en Predicciones', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_distribution.png', dpi=150)
    plt.close()
    logger.info(f"Distribución de confianza guardada")


def analyze_per_class_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    output_dir: Path
):
    """Analiza performance por clase."""
    per_class_accuracy = []

    for i, label in enumerate(labels):
        mask = y_true == i
        if mask.sum() > 0:
            accuracy = (y_pred[mask] == i).sum() / mask.sum()
            per_class_accuracy.append({
                'class': label,
                'class_idx': int(i),
                'samples': int(mask.sum()),
                'accuracy': float(accuracy)
            })

    # Ordenar por accuracy
    per_class_accuracy.sort(key=lambda x: x['accuracy'])

    # Guardar
    with open(output_dir / 'per_class_performance.json', 'w', encoding='utf-8') as f:
        json.dump(per_class_accuracy, f, indent=2, ensure_ascii=False)

    # Top 10 peores y mejores clases
    logger.info("\nTop 10 PEORES clases:")
    for item in per_class_accuracy[:10]:
        logger.info(f"  {item['class']}: {item['accuracy']*100:.1f}% ({item['samples']} samples)")

    logger.info("\nTop 10 MEJORES clases:")
    for item in per_class_accuracy[-10:][::-1]:
        logger.info(f"  {item['class']}: {item['accuracy']*100:.1f}% ({item['samples']} samples)")


def main():
    parser = argparse.ArgumentParser(
        description='Evalúa modelo V-LIBRASIL en test set'
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Ruta al modelo .h5'
    )

    parser.add_argument(
        '--model-info',
        type=str,
        required=True,
        help='Ruta al model_info.json'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directorio con datos procesados'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directorio para guardar resultados de evaluación'
    )

    args = parser.parse_args()

    # Cargar modelo
    logger.info(f"Cargando modelo desde {args.model}...")
    model = tf.keras.models.load_model(args.model)
    logger.info("Modelo cargado exitosamente")

    # Cargar datos
    data_dir = Path(args.data_dir)
    X, y, label_names = load_processed_data(data_dir)

    # Obtener test set
    X_test, y_test = get_test_set(X, y)

    # Evaluar
    output_dir = Path(args.output_dir)
    evaluate_model(model, X_test, y_test, label_names, output_dir)

    logger.info("\nEvaluación completada!")


if __name__ == '__main__':
    main()

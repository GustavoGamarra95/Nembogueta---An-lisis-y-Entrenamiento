"""
Script de evaluación para modelos de ALFABETO (Lenguaje de Señas).
Evalúa un modelo entrenado de letras usando el test set y genera métricas detalladas.

Uso:
    python scripts/evaluate_alphabet.py --model /models/alphabet/run_xxx/final_model.h5 \
                                        --model-info /models/alphabet/run_xxx/model_info.json \
                                        --data-dir /data/processed/alphabet-flat \
                                        --output-dir /models/alphabet/run_xxx/evaluation
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
        logging.FileHandler('alphabet_evaluation.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_processed_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Carga los datos procesados del alfabeto.

    Args:
        data_dir: Directorio con datos procesados

    Returns:
        Tupla (X, y, label_names)
    """
    logger.info(f"Cargando datos del alfabeto desde {data_dir}...")

    sequences = []
    labels = []

    npy_files = list(data_dir.glob("*.npy"))

    if not npy_files:
        raise FileNotFoundError(f"No se encontraron archivos .npy en {data_dir}")

    logger.info(f"Encontrados {len(npy_files)} archivos procesados")

    for npy_file in npy_files:
        try:
            sequence = np.load(npy_file)

            # Extraer letra del nombre del archivo
            # Formato esperado: A_000.npy, B_001.npy, etc.
            filename = npy_file.stem
            parts = filename.split('_')
            letter = parts[0]  # Primera parte es la letra

            sequences.append(sequence)
            labels.append(letter)

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

    # Codificar labels (letras)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    label_names = label_encoder.classes_.tolist()

    logger.info(f"Datos cargados: {X.shape[0]} muestras")
    logger.info(f"Shape: {X.shape}")
    logger.info(f"Letras: {len(label_names)} ({', '.join(sorted(label_names))})")

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
    # Mismo proceso que en train_alphabet.py
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
    Evalúa el modelo de alfabeto y genera métricas detalladas.

    Args:
        model: Modelo a evaluar
        X_test: Datos de test
        y_test: Labels de test
        label_names: Nombres de las letras
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

    # Top-k accuracy (útil para alfabeto)
    top_3_acc = top_k_accuracy_score(y_test, y_pred_probs, k=min(3, len(label_names)))
    top_5_acc = top_k_accuracy_score(y_test, y_pred_probs, k=min(5, len(label_names)))

    # Imprimir resumen
    logger.info("\n" + "="*80)
    logger.info("RESUMEN DE EVALUACIÓN - ALFABETO")
    logger.info("="*80)
    logger.info(f"Test Samples: {len(X_test)}")
    logger.info(f"Letras: {len(label_names)}")
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
        'num_letters': len(label_names),
        'letters': sorted(label_names),
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
    logger.info("Generando classification report por letra...")
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

    # Visualizar matriz de confusión (alfabeto suele tener pocas clases)
    plot_confusion_matrix(cm, label_names, output_dir)

    # Análisis de errores (letras que se confunden)
    logger.info("Analizando confusiones entre letras...")
    analyze_letter_confusions(y_test, y_pred, y_pred_probs, label_names, cm, output_dir)

    # Distribución de confianza
    plot_confidence_distribution(y_pred_probs, y_test, y_pred, output_dir)

    # Análisis por letra
    analyze_per_letter_performance(y_test, y_pred, label_names, output_dir)

    logger.info(f"\nEvaluación completa guardada en {output_dir}")


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], output_dir: Path):
    """Visualiza matriz de confusión del alfabeto."""
    plt.figure(figsize=(16, 14))

    # Normalizar por filas para ver proporciones
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    # Crear heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=sorted(labels),
        yticklabels=sorted(labels),
        cbar_kws={'label': 'Proporción'}
    )

    plt.title('Matriz de Confusión - Alfabeto (Normalizada)', fontsize=16, pad=20)
    plt.ylabel('Letra Verdadera', fontsize=12)
    plt.xlabel('Letra Predicha', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_normalized.png', dpi=150, bbox_inches='tight')
    plt.close()

    # También guardar versión con valores absolutos
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=sorted(labels),
        yticklabels=sorted(labels),
        cbar_kws={'label': 'Cantidad'}
    )

    plt.title('Matriz de Confusión - Alfabeto (Valores Absolutos)', fontsize=16, pad=20)
    plt.ylabel('Letra Verdadera', fontsize=12)
    plt.xlabel('Letra Predicha', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_absolute.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Matrices de confusión guardadas en {output_dir}")


def analyze_letter_confusions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_probs: np.ndarray,
    labels: List[str],
    cm: np.ndarray,
    output_dir: Path
):
    """Analiza las confusiones más comunes entre letras."""

    # Encontrar los pares de letras más confundidos
    confusions = []

    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    'true_letter': labels[i],
                    'predicted_letter': labels[j],
                    'count': int(cm[i, j]),
                    'proportion': float(cm[i, j] / (cm[i].sum() + 1e-8))
                })

    # Ordenar por frecuencia
    confusions.sort(key=lambda x: x['count'], reverse=True)

    # Guardar top confusiones
    with open(output_dir / 'letter_confusions.json', 'w', encoding='utf-8') as f:
        json.dump({
            'total_confusions': len([c for c in confusions if c['count'] > 0]),
            'top_confusions': confusions[:20]
        }, f, indent=2, ensure_ascii=False)

    # Imprimir top 10 confusiones
    logger.info("\nTop 10 CONFUSIONES MÁS COMUNES:")
    logger.info("-" * 60)
    for conf in confusions[:10]:
        logger.info(
            f"  {conf['true_letter']} → {conf['predicted_letter']}: "
            f"{conf['count']} veces ({conf['proportion']*100:.1f}%)"
        )


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

    plt.figure(figsize=(12, 6))
    plt.hist(correct_confidences, bins=50, alpha=0.6, label='Correctas', color='green', edgecolor='black')
    plt.hist(incorrect_confidences, bins=50, alpha=0.6, label='Incorrectas', color='red', edgecolor='black')
    plt.xlabel('Confianza de Predicción', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.title('Distribución de Confianza en Predicciones - Alfabeto', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_distribution.png', dpi=150)
    plt.close()
    logger.info(f"Distribución de confianza guardada")


def analyze_per_letter_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    output_dir: Path
):
    """Analiza performance por cada letra."""
    per_letter_metrics = []

    for i, letter in enumerate(labels):
        mask = y_true == i
        if mask.sum() > 0:
            accuracy = (y_pred[mask] == i).sum() / mask.sum()
            per_letter_metrics.append({
                'letter': letter,
                'letter_idx': int(i),
                'samples': int(mask.sum()),
                'accuracy': float(accuracy),
                'correct': int((y_pred[mask] == i).sum()),
                'incorrect': int((y_pred[mask] != i).sum())
            })

    # Ordenar por accuracy
    per_letter_metrics.sort(key=lambda x: x['accuracy'])

    # Guardar
    with open(output_dir / 'per_letter_performance.json', 'w', encoding='utf-8') as f:
        json.dump(per_letter_metrics, f, indent=2, ensure_ascii=False)

    # Top peores y mejores letras
    logger.info("\nLetras con PEOR performance:")
    logger.info("-" * 60)
    for item in per_letter_metrics[:10]:
        logger.info(
            f"  {item['letter']}: {item['accuracy']*100:.1f}% "
            f"({item['correct']}/{item['samples']} correctas)"
        )

    logger.info("\nLetras con MEJOR performance:")
    logger.info("-" * 60)
    for item in per_letter_metrics[-10:][::-1]:
        logger.info(
            f"  {item['letter']}: {item['accuracy']*100:.1f}% "
            f"({item['correct']}/{item['samples']} correctas)"
        )

    # Crear gráfica de performance por letra
    plot_per_letter_performance(per_letter_metrics, output_dir)


def plot_per_letter_performance(metrics: List[Dict], output_dir: Path):
    """Visualiza performance por letra."""
    letters = [m['letter'] for m in metrics]
    accuracies = [m['accuracy'] * 100 for m in metrics]

    plt.figure(figsize=(16, 6))
    bars = plt.bar(range(len(letters)), accuracies, color='steelblue', edgecolor='black')

    # Colorear barras según performance
    for i, bar in enumerate(bars):
        if accuracies[i] >= 90:
            bar.set_color('green')
        elif accuracies[i] >= 70:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    plt.xlabel('Letra', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Performance por Letra', fontsize=14)
    plt.xticks(range(len(letters)), letters, rotation=0)
    plt.ylim(0, 105)
    plt.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% threshold')
    plt.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='70% threshold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'per_letter_performance.png', dpi=150)
    plt.close()
    logger.info("Gráfica de performance por letra guardada")


def main():
    parser = argparse.ArgumentParser(
        description='Evalúa modelo de ALFABETO en test set'
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
        help='Directorio con datos procesados del alfabeto'
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

    logger.info("\n¡Evaluación del alfabeto completada!")


if __name__ == '__main__':
    main()

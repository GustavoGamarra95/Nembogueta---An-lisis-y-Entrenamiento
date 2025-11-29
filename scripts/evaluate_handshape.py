"""
Script para evaluar modelos de formas de mano entrenados.
Permite evaluar un modelo específico o comparar múltiples vistas.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar para que no use toda la GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class HandshapeEvaluator:
    """Evaluador de modelos de formas de mano."""

    def __init__(self, model_dir: Path, view: str):
        """
        Inicializa el evaluador.

        Args:
            model_dir: Directorio del modelo
            view: Vista del modelo (front/back/left/right)
        """
        self.model_dir = model_dir / view
        self.view = view

        # Cargar modelo
        print(f"\nCargando modelo de vista '{view}'...")
        self.model = keras.models.load_model(self.model_dir / "best_model.keras")

        # Cargar metadata
        with open(self.model_dir / "metadata.json", 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        self.class_names = self.metadata['class_names']
        print(f"  ✓ Modelo cargado: {self.metadata['n_classes']} clases")
        print(f"  ✓ Test Accuracy reportada: {self.metadata['final_test_acc']:.4f}")

    def load_test_data(self, data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carga los datos de test.

        Args:
            data_dir: Directorio con datos procesados

        Returns:
            Tupla (X_test, y_test)
        """
        test_dir = data_dir / self.view / "test"

        X_list = []
        y_list = []

        class_dirs = sorted([d for d in test_dir.glob("*") if d.is_dir()])

        for class_idx, class_dir in enumerate(class_dirs):
            npy_files = list(class_dir.glob("*.npy"))

            for npy_file in npy_files:
                try:
                    data = np.load(npy_file)
                    X_list.append(data)
                    y_list.append(class_idx)
                except Exception as e:
                    print(f"Error cargando {npy_file}: {e}")

        X = np.array(X_list)
        y = np.array(y_list)

        print(f"\nDatos de test cargados: {len(X)} muestras")
        return X, y

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evalúa el modelo en datos de test.

        Args:
            X_test: Datos de entrada
            y_test: Etiquetas verdaderas

        Returns:
            Diccionario con métricas
        """
        print("\nEvaluando modelo...")
        results = self.model.evaluate(X_test, y_test, verbose=0)

        # Predicciones
        y_pred_probs = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Top-5 accuracy manual
        top5_correct = 0
        for i in range(len(y_test)):
            top5_preds = np.argsort(y_pred_probs[i])[-5:]
            if y_test[i] in top5_preds:
                top5_correct += 1
        top5_acc = top5_correct / len(y_test)

        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'top5_accuracy': top5_acc,
            'predictions': y_pred,
            'probabilities': y_pred_probs,
            'true_labels': y_test
        }

        return metrics

    def print_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Imprime reporte de clasificación."""
        print("\n" + "=" * 80)
        print(f"REPORTE DE CLASIFICACIÓN - Vista {self.view.upper()}")
        print("=" * 80)

        # Limitar a las 10 clases más frecuentes en el reporte
        unique_classes = np.unique(y_true)

        if len(unique_classes) > 20:
            print("\nMostrando métricas generales (muchas clases)...")
            report = classification_report(
                y_true, y_pred,
                target_names=[self.class_names[i] for i in unique_classes],
                zero_division=0,
                output_dict=True
            )

            print(f"\nAccuracy global: {report['accuracy']:.4f}")
            print(f"Macro avg - Precision: {report['macro avg']['precision']:.4f}")
            print(f"Macro avg - Recall: {report['macro avg']['recall']:.4f}")
            print(f"Macro avg - F1-score: {report['macro avg']['f1-score']:.4f}")
            print(f"Weighted avg - F1-score: {report['weighted avg']['f1-score']:.4f}")
        else:
            print(classification_report(
                y_true, y_pred,
                target_names=[self.class_names[i] for i in unique_classes],
                zero_division=0
            ))

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            output_dir: Path, top_n: int = 20):
        """
        Genera matriz de confusión.

        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones
            output_dir: Directorio para guardar la figura
            top_n: Número de clases más frecuentes a mostrar
        """
        # Encontrar las clases más frecuentes
        unique, counts = np.unique(y_true, return_counts=True)
        top_classes = unique[np.argsort(counts)[-top_n:]]

        # Filtrar datos
        mask = np.isin(y_true, top_classes)
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]

        # Crear matriz de confusión
        cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=top_classes)

        # Normalizar
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Plot
        plt.figure(figsize=(14, 12))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=[self.class_names[i] for i in top_classes],
            yticklabels=[self.class_names[i] for i in top_classes],
            cbar_kws={'label': 'Proporción'}
        )
        plt.title(f'Matriz de Confusión Normalizada - Vista {self.view.upper()}\n'
                 f'(Top {top_n} clases más frecuentes)')
        plt.ylabel('Etiqueta Verdadera')
        plt.xlabel('Predicción')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        output_path = output_dir / f"confusion_matrix_{self.view}_top{top_n}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Matriz de confusión guardada en: {output_path}")
        plt.close()

    def analyze_errors(self, y_true: np.ndarray, y_pred: np.ndarray,
                      y_probs: np.ndarray, top_n: int = 10):
        """
        Analiza los errores más comunes.

        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones
            y_probs: Probabilidades de predicción
            top_n: Número de errores a mostrar
        """
        print("\n" + "=" * 80)
        print(f"ANÁLISIS DE ERRORES - Vista {self.view.upper()}")
        print("=" * 80)

        # Encontrar errores
        errors = y_true != y_pred
        error_indices = np.where(errors)[0]

        if len(error_indices) == 0:
            print("\n¡No hay errores! Accuracy perfecta.")
            return

        # Calcular confianza de los errores
        error_confidences = []
        for idx in error_indices:
            conf = y_probs[idx][y_pred[idx]]
            error_confidences.append({
                'index': idx,
                'true_class': self.class_names[y_true[idx]],
                'pred_class': self.class_names[y_pred[idx]],
                'confidence': conf,
                'true_prob': y_probs[idx][y_true[idx]]
            })

        # Ordenar por confianza (errores con alta confianza son más problemáticos)
        error_confidences.sort(key=lambda x: x['confidence'], reverse=True)

        print(f"\nTotal de errores: {len(error_indices)} / {len(y_true)} "
              f"({100 * len(error_indices) / len(y_true):.2f}%)")

        print(f"\nTop {top_n} errores con mayor confianza (más problemáticos):")
        print("-" * 80)
        for i, err in enumerate(error_confidences[:top_n], 1):
            print(f"{i}. Verdadero: {err['true_class']:20s} -> "
                  f"Predicho: {err['pred_class']:20s} "
                  f"(Confianza: {err['confidence']:.3f}, "
                  f"Prob. verdadera: {err['true_prob']:.3f})")


def compare_views(model_dir: Path, data_dir: Path, views: List[str]):
    """
    Compara el rendimiento de múltiples vistas.

    Args:
        model_dir: Directorio base de modelos
        data_dir: Directorio de datos
        views: Lista de vistas a comparar
    """
    print("\n" + "=" * 80)
    print("COMPARACIÓN DE VISTAS")
    print("=" * 80)

    results = []

    for view in views:
        if not (model_dir / view / "best_model.keras").exists():
            print(f"\n⚠ Vista '{view}' no tiene modelo entrenado, omitiendo...")
            continue

        evaluator = HandshapeEvaluator(model_dir, view)
        X_test, y_test = evaluator.load_test_data(data_dir)
        metrics = evaluator.evaluate(X_test, y_test)

        results.append({
            'view': view,
            'accuracy': metrics['accuracy'],
            'top5_accuracy': metrics['top5_accuracy'],
            'loss': metrics['loss'],
            'n_test': len(X_test)
        })

    if not results:
        print("\n⚠ No se encontraron modelos para comparar")
        return

    # Imprimir tabla comparativa
    print("\n" + "=" * 80)
    print("TABLA COMPARATIVA")
    print("=" * 80)
    print(f"{'Vista':<10} {'Test Samples':<15} {'Accuracy':<12} {'Top-5 Acc':<12} {'Loss':<10}")
    print("-" * 80)

    for res in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        print(f"{res['view']:<10} {res['n_test']:<15} "
              f"{res['accuracy']:<12.4f} {res['top5_accuracy']:<12.4f} "
              f"{res['loss']:<10.4f}")

    # Mejor modelo
    best = max(results, key=lambda x: x['accuracy'])
    print("\n" + "=" * 80)
    print(f"🏆 MEJOR MODELO: Vista '{best['view'].upper()}' con {best['accuracy']:.2%} accuracy")
    print("=" * 80)


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Evalúa modelos de formas de mano entrenados"
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/app/data/models/handshape"),
        help="Directorio base con modelos entrenados"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/app/data/processed/lswh100"),
        help="Directorio con datos procesados"
    )

    parser.add_argument(
        "--view",
        type=str,
        choices=["front", "back", "left", "right", "all"],
        default="all",
        help="Vista a evaluar (o 'all' para comparar todas)"
    )

    parser.add_argument(
        "--confusion-matrix",
        action="store_true",
        help="Generar matriz de confusión"
    )

    parser.add_argument(
        "--top-classes",
        type=int,
        default=20,
        help="Número de clases a mostrar en matriz de confusión"
    )

    parser.add_argument(
        "--analyze-errors",
        action="store_true",
        help="Analizar errores de clasificación"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./evaluation_results"),
        help="Directorio para guardar resultados"
    )

    args = parser.parse_args()

    # Crear directorio de salida
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.view == "all":
        # Comparar todas las vistas
        compare_views(args.model_dir, args.data_dir,
                     ["front", "back", "left", "right"])
    else:
        # Evaluar una vista específica
        if not (args.model_dir / args.view / "best_model.keras").exists():
            print(f"Error: No se encontró modelo para vista '{args.view}'")
            print(f"Buscado en: {args.model_dir / args.view / 'best_model.keras'}")
            return 1

        evaluator = HandshapeEvaluator(args.model_dir, args.view)
        X_test, y_test = evaluator.load_test_data(args.data_dir)
        metrics = evaluator.evaluate(X_test, y_test)

        # Imprimir resultados
        print("\n" + "=" * 80)
        print(f"RESULTADOS DE EVALUACIÓN - Vista {args.view.upper()}")
        print("=" * 80)
        print(f"Loss: {metrics['loss']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f} ({100 * metrics['accuracy']:.2f}%)")
        print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.4f} ({100 * metrics['top5_accuracy']:.2f}%)")

        # Reporte de clasificación
        evaluator.print_classification_report(
            metrics['true_labels'],
            metrics['predictions']
        )

        # Matriz de confusión
        if args.confusion_matrix:
            evaluator.plot_confusion_matrix(
                metrics['true_labels'],
                metrics['predictions'],
                args.output_dir,
                top_n=args.top_classes
            )

        # Análisis de errores
        if args.analyze_errors:
            evaluator.analyze_errors(
                metrics['true_labels'],
                metrics['predictions'],
                metrics['probabilities']
            )

    print("\n✓ Evaluación completada")
    return 0


if __name__ == "__main__":
    sys.exit(main())

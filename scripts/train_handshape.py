"""
Script para entrenar modelo de clasificación de formas de manos (LSWH100).
Utiliza arquitectura de red neuronal profunda para clasificar landmarks de manos.
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('handshape_training.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class HandshapeDataLoader:
    """Cargador de datos para formas de manos."""

    def __init__(self, data_dir: Path, view: str = "front"):
        """
        Inicializa el cargador de datos.

        Args:
            data_dir: Directorio con datos procesados
            view: Vista a cargar (front/back/left/right)
        """
        self.data_dir = data_dir / view
        self.view = view
        logger.info(f"DataLoader inicializado: {self.data_dir}")

    def load_split(self, split: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Carga un split específico.

        Args:
            split: Split a cargar (train/val/test)

        Returns:
            Tupla (X, y, class_names)
        """
        split_dir = self.data_dir / split

        if not split_dir.exists():
            logger.error(f"Split no encontrado: {split_dir}")
            return np.array([]), np.array([]), []

        X_list = []
        y_list = []
        class_names = []

        # Obtener todos los directorios de clases
        class_dirs = sorted([d for d in split_dir.glob("*") if d.is_dir()])

        logger.info(f"Cargando {split} - {len(class_dirs)} clases")

        for class_idx, class_dir in enumerate(tqdm(class_dirs, desc=f"Cargando {split}")):
            class_name = class_dir.name

            if split == "train":
                class_names.append(class_name)

            # Cargar todos los archivos .npy de esta clase
            npy_files = list(class_dir.glob("*.npy"))

            for npy_file in npy_files:
                try:
                    data = np.load(npy_file)
                    X_list.append(data)
                    y_list.append(class_idx)
                except Exception as e:
                    logger.warning(f"Error cargando {npy_file}: {e}")

        X = np.array(X_list)
        y = np.array(y_list)

        logger.info(f"{split}: X shape={X.shape}, y shape={y.shape}")

        return X, y, class_names


def build_model(
    n_features: int,
    n_classes: int,
    hidden_units: List[int] = [256, 128, 64],
    dropout: float = 0.3,
    learning_rate: float = 0.001
) -> keras.Model:
    """
    Construye el modelo DNN para clasificación de formas de manos.

    Args:
        n_features: Número de features de entrada
        n_classes: Número de clases
        hidden_units: Lista con unidades por capa oculta
        dropout: Tasa de dropout
        learning_rate: Tasa de aprendizaje

    Returns:
        Modelo compilado
    """
    model = keras.Sequential(name="handshape_classifier")

    # Capa de entrada
    model.add(layers.Input(shape=(n_features,)))

    # Normalización
    model.add(layers.BatchNormalization(name='input_norm'))

    # Capas ocultas
    for i, units in enumerate(hidden_units):
        model.add(layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.01),
            name=f"dense_{i+1}"
        ))
        model.add(layers.BatchNormalization(name=f"bn_{i+1}"))
        model.add(layers.Dropout(dropout, name=f"dropout_{i+1}"))

    # Capa de salida
    model.add(layers.Dense(n_classes, activation='softmax', name='output'))

    # Compilar modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_acc')]
    )

    return model


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Entrena modelo de clasificación de formas de manos"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/data/processed/lswh100"),
        help="Directorio con datos procesados"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/models/handshape"),
        help="Directorio de salida para el modelo"
    )

    parser.add_argument(
        "--view",
        type=str,
        default="front",
        choices=["front", "back", "left", "right"],
        help="Vista a entrenar"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Número de épocas de entrenamiento"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Tamaño del batch"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Tasa de aprendizaje"
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Tasa de dropout"
    )

    parser.add_argument(
        "--hidden-units",
        nargs='+',
        type=int,
        default=[256, 128, 64],
        help="Unidades por capa oculta"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Patience para early stopping"
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Usar GPU si está disponible"
    )

    args = parser.parse_args()

    # Configurar GPU
    if not args.gpu:
        tf.config.set_visible_devices([], 'GPU')
        logger.info("GPU deshabilitada, usando CPU")
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"GPU disponible: {gpus}")
        else:
            logger.warning("GPU solicitada pero no disponible")

    # Verificar directorio de datos
    if not args.data_dir.exists():
        logger.error(f"Directorio de datos no encontrado: {args.data_dir}")
        return 1

    # Crear directorio de salida
    output_dir = args.output_dir / args.view
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar datos
    logger.info(f"Cargando datos para vista: {args.view}")
    data_loader = HandshapeDataLoader(args.data_dir, view=args.view)

    X_train, y_train, class_names = data_loader.load_split("train")
    X_val, y_val, _ = data_loader.load_split("val")
    X_test, y_test, _ = data_loader.load_split("test")

    if len(X_train) == 0:
        logger.error("No se pudieron cargar los datos de entrenamiento")
        return 1

    # Información del dataset
    n_features = X_train.shape[1]
    n_classes = len(class_names)

    logger.info(f"\nConfiguración del dataset:")
    logger.info(f"  Vista: {args.view}")
    logger.info(f"  Train: {len(X_train)} muestras")
    logger.info(f"  Val: {len(X_val)} muestras")
    logger.info(f"  Test: {len(X_test)} muestras")
    logger.info(f"  Features: {n_features}")
    logger.info(f"  Clases: {n_classes}")

    # Construir modelo
    logger.info("\nConstruyendo modelo...")
    model = build_model(
        n_features=n_features,
        n_classes=n_classes,
        hidden_units=args.hidden_units,
        dropout=args.dropout,
        learning_rate=args.learning_rate
    )

    model.summary(print_fn=logger.info)

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            filename=str(output_dir / "training_history.csv")
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(output_dir / "logs"),
            histogram_freq=1
        )
    ]

    # Entrenar modelo
    logger.info("\nIniciando entrenamiento...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluar modelo
    logger.info("\nEvaluando modelo...")
    train_results = model.evaluate(X_train, y_train, verbose=0)
    val_results = model.evaluate(X_val, y_val, verbose=0)
    test_results = model.evaluate(X_test, y_test, verbose=0)

    logger.info(f"\nResultados finales:")
    logger.info(f"  Train - Loss: {train_results[0]:.4f}, Acc: {train_results[1]:.4f}, Top5: {train_results[2]:.4f}")
    logger.info(f"  Val   - Loss: {val_results[0]:.4f}, Acc: {val_results[1]:.4f}, Top5: {val_results[2]:.4f}")
    logger.info(f"  Test  - Loss: {test_results[0]:.4f}, Acc: {test_results[1]:.4f}, Top5: {test_results[2]:.4f}")

    # Guardar modelo final
    model.save(output_dir / "final_model.keras")
    logger.info(f"\nModelo guardado en: {output_dir}")

    # Guardar metadatos
    metadata = {
        'view': args.view,
        'class_names': class_names,
        'n_classes': n_classes,
        'n_features': n_features,
        'hidden_units': args.hidden_units,
        'dropout': args.dropout,
        'learning_rate': args.learning_rate,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'final_train_acc': float(train_results[1]),
        'final_val_acc': float(val_results[1]),
        'final_test_acc': float(test_results[1]),
        'final_train_top5': float(train_results[2]),
        'final_val_top5': float(val_results[2]),
        'final_test_top5': float(test_results[2])
    }

    with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("\nEntrenamiento completado exitosamente!")

    return 0


if __name__ == "__main__":
    sys.exit(main())

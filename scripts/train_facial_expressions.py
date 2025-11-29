"""
Script para entrenar modelo de clasificación de expresiones faciales en LIBRAS.
Utiliza arquitectura LSTM para procesar secuencias temporales de landmarks faciales.
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
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('facial_expressions_training.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class FacialExpressionDataLoader:
    """Cargador de datos para expresiones faciales."""

    def __init__(self, data_dir: Path):
        """
        Inicializa el cargador de datos.

        Args:
            data_dir: Directorio con datos procesados
        """
        self.data_dir = data_dir
        logger.info(f"DataLoader inicializado: {data_dir}")

    def load_data(self, max_samples_per_class: int = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Carga todos los datos del directorio de forma eficiente.

        Args:
            max_samples_per_class: Limitar muestras por clase (None = todas)

        Returns:
            Tupla (X, y, class_names) donde:
            - X: array de shape (n_samples, sequence_length, n_features)
            - y: array de etiquetas
            - class_names: lista de nombres de clases
        """
        # Primero, contar total de archivos
        class_dirs = sorted([d for d in self.data_dir.glob("*") if d.is_dir()])
        logger.info(f"Encontradas {len(class_dirs)} clases")

        class_names = []
        file_paths = []
        labels = []

        for class_idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            class_names.append(class_name)

            npy_files = list(class_dir.glob("*.npy"))

            # Limitar muestras por clase si se especifica
            if max_samples_per_class:
                npy_files = npy_files[:max_samples_per_class]

            for npy_file in npy_files:
                file_paths.append(npy_file)
                labels.append(class_idx)

        logger.info(f"Total de archivos a cargar: {len(file_paths)}")

        # Cargar primer archivo para obtener dimensiones
        first_data = np.load(file_paths[0])
        sequence_length, n_features = first_data.shape

        # Pre-alocar array (más eficiente en memoria)
        X = np.zeros((len(file_paths), sequence_length, n_features), dtype=np.float32)
        y = np.array(labels, dtype=np.int32)

        # Cargar datos en el array pre-alocado
        for i, file_path in enumerate(tqdm(file_paths, desc="Cargando datos")):
            try:
                X[i] = np.load(file_path).astype(np.float32)
            except Exception as e:
                logger.warning(f"Error cargando {file_path}: {e}")

        logger.info(f"Datos cargados: X shape={X.shape}, y shape={y.shape}")
        logger.info(f"Clases: {class_names}")

        return X, y, class_names


def build_model(
    sequence_length: int,
    n_features: int,
    n_classes: int,
    lstm_units: List[int] = [128, 64],
    dropout: float = 0.3,
    learning_rate: float = 0.001
) -> keras.Model:
    """
    Construye el modelo LSTM para clasificación de expresiones.

    Args:
        sequence_length: Longitud de las secuencias
        n_features: Número de features por timestep
        n_classes: Número de clases
        lstm_units: Lista con unidades LSTM por capa
        dropout: Tasa de dropout
        learning_rate: Tasa de aprendizaje

    Returns:
        Modelo compilado
    """
    model = keras.Sequential(name="facial_expression_classifier")

    # Capa de entrada
    model.add(layers.Input(shape=(sequence_length, n_features)))

    # Capas LSTM
    for i, units in enumerate(lstm_units):
        return_sequences = i < len(lstm_units) - 1
        model.add(layers.LSTM(
            units,
            return_sequences=return_sequences,
            name=f"lstm_{i+1}"
        ))
        model.add(layers.Dropout(dropout, name=f"dropout_{i+1}"))

    # Capas densas
    model.add(layers.Dense(64, activation='relu', name='dense_1'))
    model.add(layers.Dropout(dropout, name='dropout_dense'))
    model.add(layers.Dense(n_classes, activation='softmax', name='output'))

    # Compilar modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Entrena modelo de clasificación de expresiones faciales"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/data/processed/facial_expressions"),
        help="Directorio con datos procesados"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/models/facial_expressions"),
        help="Directorio de salida para el modelo"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Número de épocas de entrenamiento"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
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
        "--lstm-units",
        nargs='+',
        type=int,
        default=[128, 64],
        help="Unidades LSTM por capa"
    )

    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Máximo de muestras por clase (para reducir uso de memoria)"
    )

    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Proporción de datos para validación"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar datos
    logger.info("Cargando datos...")
    data_loader = FacialExpressionDataLoader(args.data_dir)
    X, y, class_names = data_loader.load_data(max_samples_per_class=args.max_samples_per_class)

    # Información del dataset
    sequence_length = X.shape[1]
    n_features = X.shape[2]
    n_classes = len(class_names)

    logger.info(f"\nConfiguración del dataset:")
    logger.info(f"  Muestras totales: {len(X)}")
    logger.info(f"  Longitud de secuencia: {sequence_length}")
    logger.info(f"  Features por timestep: {n_features}")
    logger.info(f"  Número de clases: {n_classes}")

    # Dividir en train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=args.val_split,
        random_state=42,
        stratify=y
    )

    logger.info(f"\nDivisión del dataset:")
    logger.info(f"  Train: {len(X_train)} muestras")
    logger.info(f"  Val: {len(X_val)} muestras")

    # Construir modelo
    logger.info("\nConstruyendo modelo...")
    model = build_model(
        sequence_length=sequence_length,
        n_features=n_features,
        n_classes=n_classes,
        lstm_units=args.lstm_units,
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
            filepath=str(args.output_dir / "best_model.keras"),
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
            filename=str(args.output_dir / "training_history.csv")
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
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    logger.info(f"\nResultados finales:")
    logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Guardar modelo final
    model.save(args.output_dir / "final_model.keras")
    logger.info(f"\nModelo guardado en: {args.output_dir}")

    # Guardar metadatos
    metadata = {
        'class_names': class_names,
        'n_classes': n_classes,
        'sequence_length': sequence_length,
        'n_features': n_features,
        'lstm_units': args.lstm_units,
        'dropout': args.dropout,
        'learning_rate': args.learning_rate,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'final_train_acc': float(train_acc),
        'final_val_acc': float(val_acc),
        'final_train_loss': float(train_loss),
        'final_val_loss': float(val_loss)
    }

    with open(args.output_dir / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("\nEntrenamiento completado exitosamente!")

    return 0


if __name__ == "__main__":
    sys.exit(main())

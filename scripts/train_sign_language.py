"""
Script universal de entrenamiento para lenguaje de señas.
Soporta letras, palabras y frases con arquitectura CNN-LSTM adaptativa.

Uso:
    # Entrenar letras
    python scripts/train_sign_language.py --data-dir /data/processed_letters \
                                          --output-dir /models/letters \
                                          --task-type letters

    # Entrenar palabras
    python scripts/train_sign_language.py --data-dir /data/processed_words \
                                          --output-dir /models/words \
                                          --task-type words

    # Entrenar frases
    python scripts/train_sign_language.py --data-dir /data/processed_phrases \
                                          --output-dir /models/phrases \
                                          --task-type phrases
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sign_language_training.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configurar GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"GPUs detectadas: {len(gpus)}")
    except RuntimeError as e:
        logger.error(f"Error configurando GPU: {e}")


def load_processed_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Carga los datos procesados.

    Args:
        data_dir: Directorio con datos procesados (.npy files)

    Returns:
        Tupla (X, y, label_names)
    """
    logger.info(f"Cargando datos desde {data_dir}...")

    sequences = []
    labels = []

    # Buscar archivos .npy procesados
    npy_files = list(data_dir.glob("*.npy"))

    if not npy_files:
        raise FileNotFoundError(f"No se encontraron archivos .npy en {data_dir}")

    logger.info(f"Encontrados {len(npy_files)} archivos procesados")

    # Cargar cada archivo
    for npy_file in npy_files:
        try:
            sequence = np.load(npy_file)

            # Extraer etiqueta del nombre del archivo
            filename = npy_file.stem
            parts = filename.split('_')

            if filename.endswith('_processed'):
                label = '_'.join(parts[:-1])
            else:
                label = '_'.join(parts)

            sequences.append(sequence)
            labels.append(label)

        except Exception as e:
            logger.warning(f"Error cargando {npy_file}: {e}")
            continue

    if not sequences:
        raise ValueError("No se pudieron cargar secuencias válidas")

    # Convertir a arrays numpy
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
    logger.info(f"Shape de secuencias: {X.shape}")
    logger.info(f"Número de clases: {len(label_names)}")

    return X, y, label_names


def create_model(
    input_shape: Tuple[int, int],
    num_classes: int,
    task_type: str = 'letters',
    learning_rate: float = 0.001
) -> tf.keras.Model:
    """
    Crea el modelo CNN-LSTM adaptado al tipo de tarea.

    Args:
        input_shape: (sequence_length, feature_dim)
        num_classes: Número de clases
        task_type: 'letters', 'words', o 'phrases'
        learning_rate: Tasa de aprendizaje

    Returns:
        Modelo compilado
    """
    # Configuración según tipo de tarea
    configs = {
        'letters': {
            'conv_filters': [64, 128, 256],
            'lstm_units': [256, 128],
            'dropout': 0.3,
            'name': 'CNN_LSTM_Letters'
        },
        'words': {
            'conv_filters': [128, 256, 512],
            'lstm_units': [512, 256],
            'dropout': 0.4,
            'name': 'CNN_LSTM_Words'
        },
        'phrases': {
            'conv_filters': [128, 256, 512],
            'lstm_units': [512, 256],
            'dropout': 0.4,
            'name': 'CNN_LSTM_Phrases'
        }
    }

    config = configs.get(task_type, configs['letters'])

    layers = []

    # Input layer
    layers.append(tf.keras.layers.Input(shape=input_shape))

    # Conv1D layers
    for i, filters in enumerate(config['conv_filters']):
        kernel_size = 5 if i < 2 else 3
        layers.append(tf.keras.layers.Conv1D(
            filters, kernel_size=kernel_size,
            activation='relu', padding='same'
        ))
        layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.Dropout(config['dropout']))

    # LSTM layers
    for i, units in enumerate(config['lstm_units']):
        return_sequences = (i < len(config['lstm_units']) - 1)
        layers.append(tf.keras.layers.LSTM(units, return_sequences=return_sequences))
        layers.append(tf.keras.layers.Dropout(config['dropout'] + 0.1))

    # Dense layers
    layers.append(tf.keras.layers.Dense(128, activation='relu'))
    layers.append(tf.keras.layers.Dropout(config['dropout']))
    layers.append(tf.keras.layers.Dense(num_classes, activation='softmax'))

    model = tf.keras.Sequential(layers, name=config['name'])

    # Compilar
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def plot_training_history(history: Dict, output_dir: Path):
    """Grafica el historial de entrenamiento."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    ax1.plot(history['accuracy'], label='Train Accuracy')
    ax1.plot(history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Loss
    ax2.plot(history['loss'], label='Train Loss')
    ax2.plot(history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
    logger.info(f"Gráfica guardada en {output_dir / 'training_history.png'}")
    plt.close()


def save_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    output_dir: Path
):
    """Guarda métricas de evaluación."""
    report = classification_report(
        y_true, y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0
    )

    report_path = output_dir / 'classification_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Classification report guardado en {report_path}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("RESUMEN DE MÉTRICAS")
    logger.info("="*60)
    logger.info(f"Accuracy: {report['accuracy']:.4f}")
    logger.info(f"Macro Avg Precision: {report['macro avg']['precision']:.4f}")
    logger.info(f"Macro Avg Recall: {report['macro avg']['recall']:.4f}")
    logger.info(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}")
    logger.info("="*60 + "\n")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    np.save(output_dir / 'confusion_matrix.npy', cm)


def train(args):
    """Función principal de entrenamiento."""
    # Crear directorios
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Directorio de salida: {run_dir}")

    # Cargar datos
    data_dir = Path(args.data_dir)
    X, y, label_names = load_processed_data(data_dir)

    if len(X) < 10:
        logger.error(f"Datos insuficientes: solo {len(X)} muestras")
        return

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Crear modelo
    input_shape = (X.shape[1], X.shape[2])
    num_classes = len(label_names)

    logger.info(f"Creando modelo CNN-LSTM para {args.task_type}...")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Número de clases: {num_classes}")

    model = create_model(
        input_shape=input_shape,
        num_classes=num_classes,
        task_type=args.task_type,
        learning_rate=args.learning_rate
    )

    model.summary(print_fn=logger.info)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(run_dir / 'logs'),
            histogram_freq=1
        )
    ]

    # Entrenar
    logger.info("Iniciando entrenamiento...")
    logger.info(f"Epochs: {args.epochs} | Batch size: {args.batch_size}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluar
    logger.info("\nEvaluando en test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")

    # Predicciones
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Guardar métricas
    save_metrics(y_test, y_pred_classes, label_names, run_dir)
    plot_training_history(history.history, run_dir)

    # Guardar modelo
    model.save(run_dir / 'final_model.h5')
    logger.info(f"Modelo final guardado en {run_dir / 'final_model.h5'}")

    # Guardar info
    model_info = {
        'timestamp': timestamp,
        'task_type': args.task_type,
        'input_shape': input_shape,
        'num_classes': num_classes,
        'label_names': label_names,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'epochs_trained': len(history.history['loss'])
    }

    with open(run_dir / 'model_info.json', 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*60}")
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info(f"{'='*60}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Entrena modelo CNN-LSTM para lenguaje de señas'
    )

    parser.add_argument(
        '--data-dir', type=str, required=True,
        help='Directorio con datos procesados'
    )
    parser.add_argument(
        '--output-dir', type=str, default='/models',
        help='Directorio de salida'
    )
    parser.add_argument(
        '--task-type', type=str, default='letters',
        choices=['letters', 'words', 'phrases'],
        help='Tipo de tarea'
    )
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=15)

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

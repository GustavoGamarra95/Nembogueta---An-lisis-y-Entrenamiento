"""
Script de entrenamiento para reconocimiento del alfabeto (A-Z).
Entrena un modelo CNN-LSTM para clasificación de letras usando landmarks.

Uso:
    python scripts/train_alphabet.py --data-dir data/processed/alphabet \
                                     --output-dir models/alphabet \
                                     --epochs 100 \
                                     --batch-size 32
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alphabet_training.log', encoding='utf-8'),
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
    Carga los datos procesados del alfabeto.

    Args:
        data_dir: Directorio con archivos .npy

    Returns:
        Tupla (X, y, label_names)
    """
    logger.info(f"Cargando datos desde {data_dir}...")

    sequences = []
    labels = []

    # Buscar archivos .npy
    npy_files = list(data_dir.glob("*.npy")) if data_dir.is_file() == False else list(data_dir.parent.glob("*.npy"))

    if not npy_files:
        # Intentar buscar en subdirectorios
        npy_files = list(data_dir.rglob("*.npy"))

    if not npy_files:
        raise FileNotFoundError(f"No se encontraron archivos .npy en {data_dir}")

    logger.info(f"Encontrados {len(npy_files)} archivos procesados")

    for npy_file in npy_files:
        try:
            sequence = np.load(npy_file)

            # Extraer etiqueta del nombre del archivo
            # Formato: "A_0001.npy" -> "A"
            filename = npy_file.stem
            letter = filename.split('_')[0]

            sequences.append(sequence)
            labels.append(letter)

        except Exception as e:
            logger.warning(f"Error cargando {npy_file}: {e}")
            continue

    if not sequences:
        raise ValueError("No se pudieron cargar secuencias válidas")

    # Convertir a arrays
    X = np.array(sequences, dtype=np.float32)

    # Codificar labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    label_names = label_encoder.classes_.tolist()

    logger.info(f"Datos cargados: {X.shape[0]} muestras")
    logger.info(f"Shape de secuencias: {X.shape}")
    logger.info(f"Número de clases: {len(label_names)}")
    logger.info(f"Clases: {sorted(label_names)}")

    return X, y, label_names


@tf.keras.utils.register_keras_serializable(package="Custom", name="AttentionLayer")
class AttentionLayer(tf.keras.layers.Layer):
    """Capa de atención para enfocarse en landmarks importantes."""

    def __init__(self, units=128, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.W = None
        self.U = None
        self.V = None

    def build(self, input_shape):
        self.W = tf.keras.layers.Dense(self.units, name='attention_W')
        self.U = tf.keras.layers.Dense(self.units, name='attention_U')
        self.V = tf.keras.layers.Dense(1, name='attention_V')
        super().build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch, timesteps, features)
        score = tf.nn.tanh(self.W(inputs))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
        })
        return config


def augment_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Aplica data augmentation a una secuencia de landmarks.

    Args:
        sequence: Array de shape (timesteps, features)

    Returns:
        Secuencia aumentada
    """
    augmented = sequence.copy()

    # 1. Ruido gaussiano (pequeño)
    if np.random.random() > 0.5:
        noise = np.random.normal(0, 0.01, augmented.shape)
        augmented += noise

    # 2. Escalado temporal (velocidad)
    if np.random.random() > 0.5:
        scale_factor = np.random.uniform(0.9, 1.1)
        augmented *= scale_factor

    # 3. Desplazamiento temporal pequeño
    if np.random.random() > 0.5:
        shift = np.random.randint(-2, 3)
        if shift > 0:
            augmented = np.concatenate([augmented[shift:], augmented[-shift:]], axis=0)
        elif shift < 0:
            augmented = np.concatenate([augmented[:shift], augmented[:-shift]], axis=0)

    # 4. Dropout de features (simula oclusión)
    if np.random.random() > 0.7:
        mask = np.random.random(augmented.shape) > 0.1
        augmented *= mask

    return augmented


def create_alphabet_model(
    input_shape: Tuple[int, int],
    num_classes: int,
    learning_rate: float = 0.001
) -> tf.keras.Model:
    """
    Crea modelo CNN-LSTM mejorado con atención para alfabeto.

    Args:
        input_shape: (sequence_length, feature_dim)
        num_classes: Número de letras (26 para A-Z)
        learning_rate: Tasa de aprendizaje

    Returns:
        Modelo compilado
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Bloque Conv1D 1 - Extracción de características de bajo nivel
    x = tf.keras.layers.Conv1D(
        64, kernel_size=3, activation='relu', padding='same'
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Bloque Conv1D 2 - Características de nivel medio
    x = tf.keras.layers.Conv1D(
        128, kernel_size=3, activation='relu', padding='same'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Bloque Conv1D 3 - Características complejas
    x = tf.keras.layers.Conv1D(
        256, kernel_size=3, activation='relu', padding='same'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Bloque Conv1D 4 - Características de alto nivel
    x = tf.keras.layers.Conv1D(
        256, kernel_size=3, activation='relu', padding='same'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Bloque LSTM Bidireccional 1 - Captura dependencias temporales
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(256, return_sequences=True)
    )(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    # Bloque LSTM Bidireccional 2 - Refinamiento temporal
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True)
    )(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    # Capa de atención - Enfoca en landmarks importantes
    attention_output = AttentionLayer(units=128)(x)

    # Pooling global adicional
    global_max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Concatenar todas las representaciones
    x = tf.keras.layers.Concatenate()([attention_output, global_max_pool, global_avg_pool])

    # Capas densas finales con regularización L2
    x = tf.keras.layers.Dense(
        256, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(
        128, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    # Capa de salida
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='CNN_LSTM_Attention_Alphabet')

    # Compilar
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def plot_training_history(history: dict, output_dir: Path):
    """Grafica historial de entrenamiento."""
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
    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=label_names,
        output_dict=True
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
    logger.info(f"Matriz de confusión guardada en {output_dir / 'confusion_matrix.npy'}")


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

    # Verificar datos
    if len(X) < 10:
        logger.error(f"Datos insuficientes: {len(X)} muestras")
        return

    # Split train/val/test
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

    logger.info(f"Creando modelo CNN-LSTM para alfabeto...")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Número de clases: {num_classes}")

    model = create_alphabet_model(
        input_shape=input_shape,
        num_classes=num_classes,
        learning_rate=args.learning_rate
    )

    # Mostrar resumen
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

    # Crear generador de datos con augmentation
    class DataGenerator(tf.keras.utils.Sequence):
        """Generador de datos con augmentation en tiempo real."""

        def __init__(self, X, y, batch_size=32, augment=True, shuffle=True):
            self.X = X
            self.y = y
            self.batch_size = batch_size
            self.augment = augment
            self.shuffle = shuffle
            self.indices = np.arange(len(self.X))
            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(len(self.X) / self.batch_size))

        def __getitem__(self, index):
            indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
            X_batch = self.X[indices]
            y_batch = self.y[indices]

            if self.augment:
                X_batch = np.array([augment_sequence(x) for x in X_batch])

            return X_batch, y_batch

        def on_epoch_end(self):
            if self.shuffle:
                np.random.shuffle(self.indices)

    # Crear generadores
    train_gen = DataGenerator(X_train, y_train, batch_size=args.batch_size, augment=True)
    val_gen = DataGenerator(X_val, y_val, batch_size=args.batch_size, augment=False, shuffle=False)

    # Calcular pesos de clase para balancear el dataset
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    logger.info("Pesos de clase calculados (balanceo):")
    for class_idx, weight in class_weight_dict.items():
        if class_idx < len(label_names):
            logger.info(f"  {label_names[class_idx]}: {weight:.2f}")

    # Entrenar
    logger.info("Iniciando entrenamiento...")
    logger.info(f"Epochs: {args.epochs} | Batch size: {args.batch_size}")
    logger.info("Data augmentation: ACTIVADO para training set")
    logger.info("Balanceo de clases: ACTIVADO")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Evaluar en test set
    logger.info("\nEvaluando en test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")

    # Predicciones
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Guardar métricas
    save_metrics(y_test, y_pred_classes, label_names, run_dir)

    # Guardar historial
    plot_training_history(history.history, run_dir)

    # Guardar modelo final
    model.save(run_dir / 'final_model.h5')
    logger.info(f"Modelo final guardado en {run_dir / 'final_model.h5'}")

    # Guardar información del modelo
    model_info = {
        'timestamp': timestamp,
        'input_shape': input_shape,
        'num_classes': num_classes,
        'label_names': label_names,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'epochs_trained': len(history.history['loss']),
        'hyperparameters': {
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'patience': args.patience
        }
    }

    with open(run_dir / 'model_info.json', 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*60}")
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info(f"{'='*60}")
    logger.info(f"Directorio de salida: {run_dir}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Entrena modelo CNN-LSTM para alfabeto'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directorio con datos procesados (.npy files)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/alphabet',
        help='Directorio para guardar modelos'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Número máximo de epochs'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Tamaño del batch'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Tasa de aprendizaje inicial'
    )

    parser.add_argument(
        '--patience',
        type=int,
        default=15,
        help='Paciencia para early stopping'
    )

    args = parser.parse_args()

    # Entrenar
    train(args)


if __name__ == '__main__':
    main()

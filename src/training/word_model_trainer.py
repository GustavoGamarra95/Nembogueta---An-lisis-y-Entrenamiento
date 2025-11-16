import logging
import os
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    LSTM,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
    TimeDistributed,
)
from tensorflow.keras.models import Sequential

# Configurar logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Establecer una semilla para reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)

# Cargar variables de entorno
load_dotenv()

# Directorios de entrada y salida desde .env
input_dir = os.getenv("DATA_RAW_DIR", "data/raw/words")
output_dir = os.getenv("DATA_PROCESSED_DIR", "data/processed/words")

# Configurar rutas de directorios
input_dir = (
    os.path.join(input_dir, "words")
    if os.path.isdir(os.path.join(input_dir, "words"))
    else input_dir
)
output_dir = (
    os.path.join(output_dir, "words")
    if os.path.isdir(os.path.join(output_dir, "words"))
    else output_dir
)
os.makedirs(output_dir, exist_ok=True)

# Rutas a los datos preprocesados
X_path = os.path.join(output_dir, "X_lsp_word_sequences.npy")
y_path = os.path.join(output_dir, "y_lsp_word_sequences.npy")

# Cargar los datos si existen
try:
    X = np.load(X_path)  # Forma: (muestras, 15, 200, 200, 3)
    y = np.load(y_path)  # Forma: (muestras,)
    print(f"Forma de X: {X.shape}, Forma de y: {y.shape}")
except FileNotFoundError:
    print(
        "Archivos de datos no encontrados. "
        "Por favor, ejecute el preprocesamiento primero."
    )
    X = np.array([])
    y = np.array([])


def augment_sequence(sequence):
    """
    Aplica aumento de datos con rotación y escalado aleatorios.
    """
    augmented_sequence = np.zeros_like(sequence)
    for i in range(sequence.shape[0]):
        frame = sequence[i]
        # Rotación aleatoria (-15 a 15 grados)
        angle = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D(
            (frame.shape[1] // 2, frame.shape[0] // 2), angle, 1
        )
        rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        # Escalado aleatorio (0.9 a 1.1)
        scale = np.random.uniform(0.9, 1.1)
        scaled = cv2.resize(
            rotated, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        )
        # Asegurar que la forma de salida coincida con la entrada
        scaled = cv2.resize(
            scaled,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        augmented_sequence[i] = scaled
    return augmented_sequence


def train_model(
    config: Dict[str, Any] = None
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """
    Entrena el modelo de reconocimiento de palabras.

    Args:
        config: Configuración opcional para el entrenamiento

    Returns:
        Tupla con el modelo entrenado y el historial de entrenamiento
    """
    if config is None:
        config = {}

    try:
        # Cargar datos
        X_path = os.path.join(output_dir, "X_lsp_word_sequences.npy")
        y_path = os.path.join(output_dir, "y_lsp_word_sequences.npy")

        try:
            X = np.load(X_path)
            y = np.load(y_path)
            logger.info(
                f"Datos cargados: X shape {X.shape}, y shape {y.shape}"
            )
        except FileNotFoundError:
            logger.error("Archivos de datos no encontrados")
            return None, {}

        if len(X) == 0 or len(y) == 0:
            logger.error("No hay datos para entrenar")
            return None, {}

        # Dividir datos
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Construir modelo CNN-LSTM para landmarks
        # Nota: Si usas landmarks de MediaPipe, X.shape será (n_samples, sequence_length, feature_dim)
        # Si usas frames de video, X.shape será (n_samples, 15, 200, 200, 3)

        if len(X.shape) == 3:
            # Modelo para landmarks (sequence_length, feature_dim)
            model = Sequential(
                [
                    # Conv1D layers para extraer características espaciales de landmarks
                    tf.keras.layers.Conv1D(
                        64, kernel_size=5, activation="relu",
                        padding="same", input_shape=(X.shape[1], X.shape[2])
                    ),
                    tf.keras.layers.BatchNormalization(),
                    Dropout(0.3),

                    tf.keras.layers.Conv1D(
                        128, kernel_size=5, activation="relu", padding="same"
                    ),
                    tf.keras.layers.BatchNormalization(),
                    Dropout(0.3),

                    tf.keras.layers.Conv1D(
                        256, kernel_size=3, activation="relu", padding="same"
                    ),
                    tf.keras.layers.BatchNormalization(),
                    Dropout(0.3),

                    # LSTM layers para dependencias temporales
                    LSTM(256, return_sequences=True),
                    Dropout(0.4),
                    LSTM(128),
                    Dropout(0.4),

                    # Dense layers para clasificación
                    Dense(128, activation="relu"),
                    Dropout(0.3),
                    Dense(10, activation="softmax"),  # 10 clases para palabras
                ]
            )
        else:
            # Modelo para video frames (15, 200, 200, 3)
            model = Sequential(
                [
                    # Capas CNN para extraer características espaciales
                    TimeDistributed(
                        Conv2D(64, (3, 3), activation="relu", padding="same"),
                        input_shape=(15, 200, 200, 3),
                    ),
                    TimeDistributed(MaxPooling2D((2, 2))),
                    TimeDistributed(
                        Conv2D(128, (3, 3), activation="relu", padding="same")
                    ),
                    TimeDistributed(MaxPooling2D((2, 2))),
                    TimeDistributed(
                        Conv2D(256, (3, 3), activation="relu", padding="same")
                    ),
                    TimeDistributed(MaxPooling2D((2, 2))),
                    TimeDistributed(Flatten()),
                    # Capas LSTM para dependencias temporales
                    LSTM(256, return_sequences=True),
                    Dropout(0.3),
                    LSTM(128, return_sequences=False),
                    Dropout(0.3),
                    # Capas densas para clasificación
                    Dense(128, activation="relu"),
                    Dropout(0.3),
                    Dense(10, activation="softmax"),  # 10 clases para palabras
                ]
            )

        # Compilar
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=config.get("learning_rate", 0.0005),
            weight_decay=config.get("weight_decay", 0.01),
        )
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Early stopping
        early_stopping = EarlyStopping(
            monitor="val_accuracy",
            patience=config.get("patience", 10),
            restore_best_weights=True,
        )

        # Entrenar
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=config.get("epochs", 50),
            batch_size=config.get("batch_size", 16),
            callbacks=[early_stopping],
            verbose=1,
        )

        # Guardar modelo
        model_path = os.path.join(output_dir, "cnn_lstm_lsp_words_model.h5")
        model.save(model_path)
        logger.info(f"Modelo guardado en {model_path}")

        return model, history.history

    except Exception as e:
        logger.error(f"Error en el entrenamiento: {e}")
        return None, {}

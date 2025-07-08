import logging
import os
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
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
input_dir = os.getenv("DATA_RAW_DIR", "data/raw/phrases")
output_dir = os.getenv("DATA_PROCESSED_DIR", "data/processed/phrases")

# Configurar rutas de directorios
input_dir = (
    os.path.join(input_dir, "phrases")
    if os.path.isdir(os.path.join(input_dir, "phrases"))
    else input_dir
)
output_dir = (
    os.path.join(output_dir, "phrases")
    if os.path.isdir(os.path.join(output_dir, "phrases"))
    else output_dir
)
os.makedirs(output_dir, exist_ok=True)

# Rutas a los datos preprocesados
processed_dir = "data/processed_lsp_phrase_sequences"
X_path = os.path.join(processed_dir, "X_lsp_phrase_sequences.npy")
y_path = os.path.join(processed_dir, "y_lsp_phrase_sequences.npy")

# Crear archivos vacíos para pruebas si no existen
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

# Cargar los datos si existen, sino crear arrays vacíos para pruebas
try:
    X = np.load(X_path)  # Forma: (muestras, 15, 200, 200, 3)
    y = np.load(y_path)  # Forma: (muestras,)
except FileNotFoundError:
    # Crear arrays vacíos para pruebas
    X = np.zeros((0, 15, 200, 200, 3))
    y = np.zeros((0,))
    np.save(X_path, X)
    np.save(y_path, y)

print(f"Forma de X: {X.shape}, Forma de y: {y.shape}")


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
    Entrena el modelo de reconocimiento de frases.

    Args:
        config: Configuración opcional para el entrenamiento

    Returns:
        Tupla con el modelo entrenado y el historial de entrenamiento
    """
    if config is None:
        config = {}

    try:
        # Cargar datos
        X_path = os.path.join(processed_dir, "X_lsp_phrase_sequences.npy")
        y_path = os.path.join(processed_dir, "y_lsp_phrase_sequences.npy")

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

        # Check if we have data before splitting
        if len(X) == 0 or len(y) == 0:
            print("No hay datos para entrenar.")
            X_train = X_val = y_train = y_val = np.array([])
        else:
            # Aplicar aumento de datos
            X_augmented = np.array([augment_sequence(seq) for seq in X])

            # Normalizar los datos
            X_augmented = X_augmented / 255.0

            # Dividir los datos
            X_train, X_val, y_train, y_val = train_test_split(
                X_augmented, y, test_size=0.2, random_state=42
            )

        # Calcular pesos de clase solo si hay datos
        if len(y_train) > 0:
            class_weights = class_weight.compute_class_weight(
                "balanced", classes=np.unique(y_train), y=y_train
            )
            class_weights_dict = dict(enumerate(class_weights))
        else:
            class_weights_dict = {}

        # Create and train model only if we have data
        if len(X_train) > 0 and len(y_train) > 0:
            # Construir modelo
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
                    Dense(3, activation="softmax"),  # 3 clases para frases
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
                class_weight=class_weights_dict,
                callbacks=[early_stopping],
                verbose=1,
            )

            # Guardar modelo
            model_path = os.path.join(
                output_dir,
                "cnn_lstm_lsp_phrases_model.h5",  # Corrected indentation
            )
            model.save(model_path)
            logger.info(f"Modelo guardado en {model_path}")

            return model, history.history
        else:
            print("No hay suficientes datos para entrenar el modelo.")
            return None, {}

    except Exception as e:
        logger.error(f"Error en el entrenamiento: {e}")
        return None, {}  # Corrected comment spacing and ensured single line

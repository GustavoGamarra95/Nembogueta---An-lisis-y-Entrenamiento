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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

tf.random.set_seed(42)
np.random.seed(42)

load_dotenv()

input_dir = os.getenv("DATA_RAW_DIR", "data/raw/phrases")
output_dir = os.getenv("DATA_PROCESSED_DIR", "data/processed/phrases")

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

processed_dir = "data/processed_lsp_phrase_sequences"
X_path = os.path.join(processed_dir, "X_lsp_phrase_sequences.npy")
y_path = os.path.join(processed_dir, "y_lsp_phrase_sequences.npy")

if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

try:
    X = np.load(X_path)  # Forma: (muestras, 15, 200, 200, 3)
    y = np.load(y_path)  # Forma: (muestras,)
except FileNotFoundError:
    X = np.zeros((0, 15, 200, 200, 3))
    y = np.zeros((0,))
    np.save(X_path, X)
    np.save(y_path, y)

print(f"Forma de X: {X.shape}, Forma de y: {y.shape}")


def augment_sequence(sequence):
    augmented_sequence = np.zeros_like(sequence)
    for i in range(sequence.shape[0]):
        frame = sequence[i]
        angle = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D(
            (frame.shape[1] // 2, frame.shape[0] // 2), angle, 1
        )
        rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        scale = np.random.uniform(0.9, 1.1)
        scaled = cv2.resize(
            rotated, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        )
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
    if config is None:
        config = {}

    try:
        X_path = os.path.join(processed_dir, "X_lsp_phrase_sequences.npy")
        y_path = os.path.join(processed_dir, "y_lsp_phrase_sequences.npy")

        try:
            X = np.load(X_path)
            y = np.load(y_path)
            logger.info(
                f"Dados carregados: X shape {X.shape}, y shape {y.shape}"
            )
        except FileNotFoundError:
            logger.error("Archivos de datos no encontrados")
            return None, {}

        if len(X) == 0 or len(y) == 0:
            logger.error("No hay datos para entrenar")
            return None, {}

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if len(X) == 0 or len(y) == 0:
            print("No hay datos para entrenar.")
            X_train = X_val = y_train = y_val = np.array([])
        else:
            X_augmented = np.array([augment_sequence(seq) for seq in X])
            X_augmented = X_augmented / 255.0
            X_train, X_val, y_train, y_val = train_test_split(
                X_augmented, y, test_size=0.2, random_state=42
            )

        if len(y_train) > 0:
            class_weights = class_weight.compute_class_weight(
                "balanced", classes=np.unique(y_train), y=y_train
            )
            class_weights_dict = dict(enumerate(class_weights))
        else:
            class_weights_dict = {}

        if len(X_train) > 0 and len(y_train) > 0:
            # X.shape == 3: landmark sequences; X.shape == 5: raw video frames
            if len(X.shape) == 3:
                model = Sequential(
                    [
                        tf.keras.layers.Conv1D(
                            64, kernel_size=5, activation="relu",
                            padding="same",
                            input_shape=(X.shape[1], X.shape[2])
                        ),
                        tf.keras.layers.BatchNormalization(),
                        Dropout(0.3),

                        tf.keras.layers.Conv1D(
                            128, kernel_size=5,
                            activation="relu", padding="same"
                        ),
                        tf.keras.layers.BatchNormalization(),
                        Dropout(0.3),

                        tf.keras.layers.Conv1D(
                            256, kernel_size=3,
                            activation="relu", padding="same"
                        ),
                        tf.keras.layers.BatchNormalization(),
                        Dropout(0.3),

                        LSTM(256, return_sequences=True),
                        Dropout(0.4),
                        LSTM(128),
                        Dropout(0.4),

                        Dense(128, activation="relu"),
                        Dropout(0.3),
                        Dense(3, activation="softmax"),  # 3 clases para frases
                    ]
                )
            else:
                model = Sequential(
                    [
                        TimeDistributed(
                            Conv2D(64, (3, 3), activation="relu",
                                   padding="same"),
                            input_shape=(15, 200, 200, 3),
                        ),
                        TimeDistributed(MaxPooling2D((2, 2))),
                        TimeDistributed(
                            Conv2D(128, (3, 3), activation="relu",
                                   padding="same")
                        ),
                        TimeDistributed(MaxPooling2D((2, 2))),
                        TimeDistributed(
                            Conv2D(256, (3, 3), activation="relu",
                                   padding="same")
                        ),
                        TimeDistributed(MaxPooling2D((2, 2))),
                        TimeDistributed(Flatten()),
                        LSTM(256, return_sequences=True),
                        Dropout(0.3),
                        LSTM(128, return_sequences=False),
                        Dropout(0.3),
                        Dense(128, activation="relu"),
                        Dropout(0.3),
                        Dense(3, activation="softmax"),  # 3 clases para frases
                    ]
                )

            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=config.get("learning_rate", 0.0005),
                weight_decay=config.get("weight_decay", 0.01),
            )
            model.compile(
                optimizer=optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            early_stopping = EarlyStopping(
                monitor="val_accuracy",
                patience=config.get("patience", 10),
                restore_best_weights=True,
            )

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

            model_path = os.path.join(
                output_dir,
                "cnn_lstm_lsp_phrases_model.h5",
            )
            model.save(model_path)
            logger.info(f"Modelo guardado en {model_path}")

            return model, history.history
        else:
            print("No hay suficientes datos para entrenar el modelo.")
            return None, {}

    except Exception as e:
        logger.error(f"Error en el entrenamiento: {e}")
        return None, {}

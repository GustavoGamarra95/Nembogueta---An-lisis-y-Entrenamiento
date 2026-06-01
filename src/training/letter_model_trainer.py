import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from src.config.config import Config

logger = logging.getLogger(__name__)

load_dotenv()

input_dir = os.getenv(
    "DATA_PROCESSED_DIR", "data/processed_lsp_letter_sequences"
)
output_dir = os.getenv("MODELS_DIR", "models/h5")

input_dir = (
    os.path.join(input_dir, "letters")
    if os.path.isdir(os.path.join(input_dir, "letters"))
    else input_dir
)
output_dir = (
    os.path.join(output_dir, "letters")
    if os.path.isdir(os.path.join(output_dir, "letters"))
    else output_dir
)
os.makedirs(output_dir, exist_ok=True)


class LetterModelTrainer:
    def __init__(self):
        self.config = Config()
        self.model_config = self.config.model_config
        self.model = None
        self.history = None

    def create_model(
        self, input_shape: Tuple[int, ...], num_classes: int
    ) -> tf.keras.Model:
        try:
            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv1D(
                        64, kernel_size=5, activation="relu",
                        padding="same", input_shape=input_shape
                    ),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(0.3),

                    tf.keras.layers.Conv1D(
                        128, kernel_size=5, activation="relu", padding="same"
                    ),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(0.3),

                    tf.keras.layers.Conv1D(
                        256, kernel_size=3, activation="relu", padding="same"
                    ),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(0.3),

                    tf.keras.layers.LSTM(
                        256, return_sequences=True
                    ),
                    tf.keras.layers.Dropout(0.4),
                    tf.keras.layers.LSTM(128),
                    tf.keras.layers.Dropout(0.4),

                    tf.keras.layers.Dense(64, activation="relu"),
                    tf.keras.layers.Dropout(0.3),
                    tf.keras.layers.Dense(num_classes, activation="softmax"),
                ]
            )

            # one-hot encoded labels → categorical_crossentropy
            model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )

            return model

        except Exception as e:
            logger.error(f"Error al crear el modelo: {e}")
            raise

    def load_data(self, data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        try:
            sequences = []
            labels = []

            for file_path in data_dir.glob("*_processed.npy"):
                sequence = np.load(file_path)
                label = file_path.stem.split("_")[1]

                sequences.append(sequence)
                labels.append(label)

            X = np.array(sequences)
            y = np.array(labels)

            return X, y

        except Exception as e:
            logger.error(f"Error al cargar los datos: {e}")
            raise

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=self.model_config.get("validation_split", 0.2),
                random_state=42,
            )

            input_shape = (X.shape[1], X.shape[2])
            num_classes = y.shape[1]  # Use shape from one-hot encoded labels
            self.model = self.create_model(input_shape, num_classes)

            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=10, restore_best_weights=True
            )

            self.history = self.model.fit(
                X_train,
                y_train,
                batch_size=self.model_config.get("batch_size", 32),
                epochs=self.model_config.get("epochs", 100),
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=1,
            )

            test_loss, test_accuracy = self.model.evaluate(
                X_val, y_val, verbose=0
            )

            metrics = {
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "history": self.history.history,
            }

            return metrics

        except Exception as e:
            logger.error(f"Error durante el entrenamiento: {e}")
            raise

    def save_model(self, save_path: Path):
        try:
            if self.model is None:
                raise ValueError("No hay modelo para guardar")

            model_path = save_path / "letter_model.keras"
            self.model.save(str(model_path))
            logger.info(f"Modelo guardado en: {model_path}")

            metrics_path = save_path / "letter_model_metrics.npy"
            np.save(metrics_path, self.history.history)
            logger.info(f"Métricas guardadas en: {metrics_path}")

        except Exception as e:
            logger.error(f"Error al guardar el modelo: {e}")
            raise


def train_model(
    config: Dict[str, Any] = None
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    if config is None:
        config = Config().model_config

    try:
        X_path = os.path.join(input_dir, "X_lsp_letter_sequences.npy")
        y_path = os.path.join(input_dir, "y_lsp_letter_sequences.npy")

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

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    64, kernel_size=5, activation="relu",
                    padding="same", input_shape=(X.shape[1], X.shape[2])
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),

                tf.keras.layers.Conv1D(
                    128, kernel_size=5, activation="relu", padding="same"
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),

                tf.keras.layers.Conv1D(
                    256, kernel_size=3, activation="relu", padding="same"
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),

                tf.keras.layers.LSTM(256, return_sequences=True),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.LSTM(128),
                tf.keras.layers.Dropout(0.4),

                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(27, activation="softmax"),  # a-z + ñ
            ]
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=config.get("epochs", 50),
            batch_size=config.get("batch_size", 32),
            verbose=1,
        )

        output_dir = os.path.join("models", "h5")
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "letter_recognition_model.h5")
        model.save(model_path)
        logger.info(f"Modelo guardado en {model_path}")

        return model, history.history

    except Exception as e:
        logger.error(f"Error en el entrenamiento: {e}")
        return None, {}

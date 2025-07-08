import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from src.config.config import Config

logger = logging.getLogger(__name__)

class LetterModelTrainer:
    def __init__(self):
        self.config = Config()
        self.model_config = self.config.model_config
        self.model = None
        self.history = None

    def create_model(self, input_shape: Tuple[int, ...], num_classes: int) -> tf.keras.Model:
        """
        Crea el modelo CNN-LSTM para clasificación de letras.

        Args:
            input_shape: Forma de los datos de entrada
            num_classes: Número de clases (letras) a clasificar

        Returns:
            Modelo de Keras compilado
        """
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])

            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            return model

        except Exception as e:
            logger.error(f"Error al crear el modelo: {e}")
            raise

    def load_data(self, data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carga y prepara los datos de entrenamiento.

        Args:
            data_dir: Directorio con las secuencias procesadas

        Returns:
            Tupla de (características, etiquetas)
        """
        try:
            sequences = []
            labels = []

            for file_path in data_dir.glob("*_processed.npy"):
                sequence = np.load(file_path)
                label = file_path.stem.split('_')[1]  # Extraer letra del nombre

                sequences.append(sequence)
                labels.append(label)

            # Convertir a arrays numpy
            X = np.array(sequences)
            y = np.array(labels)

            return X, y

        except Exception as e:
            logger.error(f"Error al cargar los datos: {e}")
            raise

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Entrena el modelo con los datos proporcionados.

        Args:
            X: Datos de características
            y: Etiquetas

        Returns:
            Diccionario con métricas del entrenamiento
        """
        try:
            # Dividir datos en entrenamiento y validación
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.model_config.get('validation_split', 0.2),
                random_state=42
            )

            # Crear el modelo
            input_shape = (X.shape[1], X.shape[2])
            num_classes = len(np.unique(y))
            self.model = self.create_model(input_shape, num_classes)

            # Entrenar el modelo
            self.history = self.model.fit(
                X_train, y_train,
                batch_size=self.model_config.get('batch_size', 32),
                epochs=self.model_config.get('epochs', 100),
                validation_data=(X_val, y_val),
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_accuracy',
                        patience=10,
                        restore_best_weights=True
                    )
                ]
            )

            # Evaluar el modelo
            test_loss, test_accuracy = self.model.evaluate(X_val, y_val)

            metrics = {
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'history': self.history.history
            }

            return metrics

        except Exception as e:
            logger.error(f"Error durante el entrenamiento: {e}")
            raise

    def save_model(self, save_path: Path):
        """
        Guarda el modelo entrenado.

        Args:
            save_path: Ruta donde guardar el modelo
        """
        try:
            if self.model is None:
                raise ValueError("No hay modelo para guardar")

            model_path = save_path / "letter_model.h5"
            self.model.save(str(model_path))
            logger.info(f"Modelo guardado en: {model_path}")

            # Guardar métricas y configuración
            metrics_path = save_path / "letter_model_metrics.npy"
            np.save(metrics_path, self.history.history)
            logger.info(f"Métricas guardadas en: {metrics_path}")

        except Exception as e:
            logger.error(f"Error al guardar el modelo: {e}")
            raise
"""Módulo para convertir modelos entre formatos."""
import tensorflow as tf
import logging
from pathlib import Path
from src.config.config import Config
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Directorios de modelos desde .env
MODELS_DIR = os.getenv('MODELS_DIR', 'models/h5')
TFLITE_DIR = os.getenv('TFLITE_DIR', 'models/tflite')


class ModelConverter:
    def __init__(self):
        self.config = Config()
        self.model_config = self.config.model_config

    def convert_to_tflite(
            self,
            model_path: Path,
            output_path: Path
    ) -> bool:
        """
        Convierte un modelo Keras a formato TFLite.

        Args:
            model_path: Ruta al modelo .h5
            output_path: Ruta donde guardar el modelo .tflite

        Returns:
            bool: True si la conversión fue exitosa
        """
        try:
            # Cargar el modelo
            model = tf.keras.models.load_model(str(model_path))

            # Crear el convertidor
            converter = tf.lite.TFLiteConverter.from_keras_model(model)

            # Configurar optimizaciones
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

            # Convertir el modelo
            tflite_model = converter.convert()

            # Guardar el modelo
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(tflite_model)

            logger.info(f"Modelo convertido y guardado en: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error durante la conversión del modelo: {e}")
            return False

    def convert_all_models(self):
        """Convierte todos los modelos encontrados en el directorio."""
        try:
            model_dir = Path(MODELS_DIR)
            tflite_dir = Path(TFLITE_DIR)

            # Convertir modelos de letras, palabras y frases
            for model_type in ['letter', 'word', 'phrase']:
                model_path = model_dir / f"{model_type}_model.h5"
                if model_path.exists():
                    output_path = tflite_dir / f"{model_type}_model.tflite"
                    if self.convert_to_tflite(model_path, output_path):
                        logger.info(
                            f"Modelo {model_type} convertido exitosamente"
                        )
                    else:
                        logger.error(
                            f"Error al convertir modelo {model_type}"
                        )

        except Exception as e:
            logger.error(f"Error al convertir modelos: {e}")

"""
Módulo de configuración para el proyecto Ñemongueta.
Contiene configuraciones para la conexión a PostgreSQL y otras variables.
"""

import os
from dotenv import load_dotenv


class Config:
    def __init__(self):
        # Cargar variables de entorno desde un archivo .env (si existe)
        load_dotenv()

        # Configuración de PostgreSQL
        self.postgres_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'nembogueta_db'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', ''),
        }

        # Configuración para video
        self.video_config = {
            'fps': 30,
            'duration': 10,
            'num_samples': 10
        }

        # Configuración para datos
        self.data_config = {
            'video_path': {
                'letters': os.path.join(
                    self._get_data_base_path(), 'raw', 'letters'
                ),
                'words': os.path.join(
                    self._get_data_base_path(), 'raw', 'words'
                ),
                'phrases': os.path.join(
                    self._get_data_base_path(), 'raw', 'phrases'
                )
            },
            'processed_path': {
                'letters': os.path.join(
                    self._get_data_base_path(), 'processed', 'letters'
                ),
                'words': os.path.join(
                    self._get_data_base_path(), 'processed', 'words'
                ),
                'phrases': os.path.join(
                    self._get_data_base_path(), 'processed', 'phrases'
                )
            }
        }

        # Configuración para modelos
        self.model_config = {
            'save_path': os.path.join(
                self._get_models_base_path(), 'h5'
            ),
            'tflite_path': os.path.join(
                self._get_models_base_path(), 'tflite'
            ),
            'epochs': 100,
            'batch_size': 32,
            'validation_split': 0.2,
            'learning_rate': 0.001
        }

        # Configuración para logging
        self.logging_config = {
            'level': os.getenv('LOG_LEVEL', 'INFO'),
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'log_file': os.path.join(
                self._get_project_root(), 'logs', 'nembogueta.log'
            ),
        }

    def _get_project_root(self):
        """Obtiene la ruta raíz del proyecto."""
        return os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

    def _get_data_base_path(self):
        """Obtiene la ruta base para los datos."""
        return os.path.join(self._get_project_root(), 'data')

    def _get_models_base_path(self):
        """Obtiene la ruta base para los modelos."""
        return os.path.join(self._get_project_root(), 'models')

    def get_database_url(self):
        """Devuelve la URL de conexión para SQLAlchemy."""
        pg = self.postgres_config
        return (
            f"postgresql://{pg['user']}:{pg['password']}"
            f"@{pg['host']}:{pg['port']}/{pg['database']}"
        )

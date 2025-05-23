# -*- coding: utf-8 -*-
"""
Módulo de configuración para el proyecto Ñemongueta.
Contiene configuraciones para la conexión a PostgreSQL y otras variables globales.
"""

import os
from dotenv import load_dotenv

# Cargar variables de entorno desde un archivo .env (si existe)
load_dotenv()

# Configuración de PostgreSQL
POSTGRES_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'nembogueta_db'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),
}

# Función para obtener una cadena de conexión para SQLAlchemy
def get_database_url():
    """
    Devuelve la URL de conexión para SQLAlchemy
    """
    db_config = POSTGRES_CONFIG
    return f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"

# Ruta base para el almacenamiento de datos
DATA_BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# Rutas para datos crudos
RAW_LETTERS_PATH = os.path.join(DATA_BASE_PATH, 'raw', 'letters')
RAW_WORDS_PATH = os.path.join(DATA_BASE_PATH, 'raw', 'words')
RAW_PHRASES_PATH = os.path.join(DATA_BASE_PATH, 'raw', 'phrases')

# Rutas para datos procesados
PROCESSED_LETTERS_PATH = os.path.join(DATA_BASE_PATH, 'processed', 'letters')
PROCESSED_WORDS_PATH = os.path.join(DATA_BASE_PATH, 'processed', 'words')
PROCESSED_PHRASES_PATH = os.path.join(DATA_BASE_PATH, 'processed', 'phrases')

# Rutas para modelos
MODELS_BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
H5_MODELS_PATH = os.path.join(MODELS_BASE_PATH, 'h5')
TFLITE_MODELS_PATH = os.path.join(MODELS_BASE_PATH, 'tflite')

# Parámetros de configuración para el entrenamiento
TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 32,
    'validation_split': 0.2,
    'learning_rate': 0.001,
}

# Configuración para logging
LOGGING_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'nembogueta.log'),
}
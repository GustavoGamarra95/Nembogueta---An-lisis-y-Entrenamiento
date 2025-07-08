"""
Módulo para el preprocesamiento de datos de señas capturados.
Incluye funciones para normalizar, aumentar y preparar datos para
entrenamiento.
"""
from . import letter_preprocessor
from . import word_processor
from . import phrase_processor

# Funciones de conveniencia
process_letter_data = letter_preprocessor.process_data
process_word_data = word_processor.process_data
process_phrase_data = phrase_processor.process_data

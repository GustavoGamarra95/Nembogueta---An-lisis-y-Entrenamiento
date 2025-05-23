# -*- coding: utf-8 -*-
"""
Utilidades generales para el proyecto Ñemongeta.
Incluye herramientas para análisis de secuencias, conversión de modelos y otras funciones auxiliares.
"""
from . import sequence_analyzer
from . import model_converter

# Funciones de conveniencia
analyze_sequence = sequence_analyzer.analyze
convert_to_tflite = model_converter.convert_model_to_tflite# -*- coding: utf-8 -*-

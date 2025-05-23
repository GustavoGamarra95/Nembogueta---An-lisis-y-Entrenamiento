# -*- coding: utf-8 -*-
"""
Proyecto Ñemongeta - Reconocimiento de Lenguaje de Señas Paraguayo (LSPy)
"""
from . import data_collection
from . import preprocessing
from . import training
from . import utils

# Versión del paquete
__version__ = '0.1.0'

# Facilitar importaciones comunes
from .utils.model_converter import convert_model_to_tflite
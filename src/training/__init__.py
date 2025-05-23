# -*- coding: utf-8 -*-
"""
Módulo para el entrenamiento de modelos de reconocimiento de señas.
Contiene implementaciones para entrenar modelos de letras, palabras y frases.
"""
from . import letter_model_trainer
from . import word_model_trainer
from . import phrase_model_trainer

# Funciones de conveniencia
train_letter_model = letter_model_trainer.train_model
train_word_model = word_model_trainer.train_model
train_phrase_model = phrase_model_trainer.train_model
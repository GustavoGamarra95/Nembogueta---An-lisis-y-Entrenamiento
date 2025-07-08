# -*- coding: utf-8 -*-
"""
Módulo para el entrenamiento de modelos de reconocimiento de señas.
Contiene implementaciones para entrenar modelos de letras, palabras y
frases.
"""
from . import letter_model_trainer, phrase_model_trainer, word_model_trainer

# Funciones de conveniencia
train_letter_model = letter_model_trainer.train_model
train_word_model = word_model_trainer.train_model
train_phrase_model = phrase_model_trainer.train_model

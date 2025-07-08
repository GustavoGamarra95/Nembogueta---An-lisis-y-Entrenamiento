import unittest
import numpy as np
import tensorflow as tf
from pathlib import Path
from src.training.letter_model_trainer import LetterModelTrainer


class TestLetterModelTrainer(unittest.TestCase):

    def setUp(self):
        self.trainer = LetterModelTrainer()
        self.test_data_dir = Path("tests/test_data")
        # Crear datos de prueba
        self.X = np.random.random((100, 30, 63))  # 100 muestras, 30 frames, 63 features
        self.y = np.random.randint(0, 27, 100)  # 27 letras (a-z + ñ)

    def test_create_model(self):
        """Test la creación del modelo"""
        input_shape = (30, 63)  # 30 frames, 63 features
        num_classes = 27  # a-z + ñ
        model = self.trainer.create_model(input_shape, num_classes)
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.input_shape[1:], input_shape)
        self.assertEqual(model.output_shape[1], num_classes)

    def test_train_model(self):
        """Test el entrenamiento del modelo"""
        # Convertir etiquetas a one-hot
        y_one_hot = tf.keras.utils.to_categorical(self.y)
        # Entrenar el modelo con datos pequeños
        metrics = self.trainer.train(self.X, y_one_hot)
        self.assertIn('test_accuracy', metrics)
        self.assertIn('test_loss', metrics)
        self.assertIn('history', metrics)

    def test_save_model(self):
        """Test el guardado del modelo"""
        # Crear y entrenar un modelo pequeño
        input_shape = (30, 63)
        num_classes = 27
        self.trainer.model = self.trainer.create_model(input_shape, num_classes)
        # Intentar guardar el modelo
        with self.assertRaises(Exception):
            self.trainer.save_model(Path("nonexistent_dir"))


if __name__ == '__main__':
    unittest.main()


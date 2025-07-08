"""Tests para el preprocesamiento de letras."""
import unittest
import numpy as np
from pathlib import Path
from src.preprocessing.letter_preprocessor import LetterPreprocessor
from src.utils.validators import VideoData, ProcessedSequence


class TestLetterPreprocessor(unittest.TestCase):
    """Tests para el módulo LetterPreprocessor."""

    def setUp(self):
        """Configura el ambiente de prueba."""
        self.preprocessor = LetterPreprocessor()
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.test_video_path = Path("tests/test_data/test_video.mp4")

    def test_extract_landmarks_empty_frame(self):
        """Test que no se extraen landmarks de un frame vacío"""
        landmarks = self.preprocessor.extract_landmarks(self.test_frame)
        self.assertIsNone(landmarks)

    def test_process_video_invalid_data(self):
        """Test que el procesamiento falla con datos inválidos"""
        invalid_video_data = VideoData(
            path=Path("nonexistent.mp4"),
            frames=[],
            label="A",
            duration=0
        )
        result = self.preprocessor.process_video(invalid_video_data)
        self.assertIsNone(result)

    def test_process_video_valid_data(self):
        """Test el procesamiento con datos válidos"""
        # Crear un video de prueba simple
        frames = [
            np.zeros((480, 640, 3), dtype=np.uint8)
            for _ in range(10)
        ]
        video_data = VideoData(
            path=self.test_video_path,
            frames=frames,
            label="A",
            duration=1.0
        )
        result = self.preprocessor.process_video(video_data)
        self.assertIsInstance(result, ProcessedSequence)
        self.assertEqual(result.label, "A")

    def test_validate_processed_sequence(self):
        """Test la validación de secuencias procesadas"""
        # 30 frames, 21 landmarks x 3 coordinates
        sequence = np.random.random((30, 63))
        processed_sequence = ProcessedSequence(
            sequence=sequence,
            label="A",
            metadata={"test": "data"}
        )
        self.assertTrue(processed_sequence.validate())


if __name__ == '__main__':
    unittest.main()

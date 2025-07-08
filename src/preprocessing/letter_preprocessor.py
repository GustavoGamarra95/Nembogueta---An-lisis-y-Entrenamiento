import cv2
import numpy as np
import mediapipe as mp
import logging
from pathlib import Path
from typing import Optional
from src.config.config import Config
from ..utils.validators import VideoData, ProcessedSequence
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Directorios de entrada y salida desde .env
input_dir = os.getenv('DATA_RAW_DIR', 'data/lsp_letter_videos')
output_dir = os.getenv('DATA_PROCESSED_DIR', 'data/processed_lsp_letter_sequences')
input_dir = os.path.join(input_dir, 'letters') if os.path.isdir(os.path.join(input_dir, 'letters')) else input_dir
output_dir = os.path.join(output_dir, 'letters') if os.path.isdir(os.path.join(output_dir, 'letters')) else output_dir
os.makedirs(output_dir, exist_ok=True)

class LetterPreprocessor:
    def __init__(self):
        self.config = Config()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7
        )

    def extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrae los puntos de referencia de las manos de un frame.

        Args:
            frame: Frame de video en formato BGR

        Returns:
            Array numpy con los landmarks o None si no se detectan manos
        """
        try:
            # Convertir BGR a RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if not results.multi_hand_landmarks:
                return None

            # Extraer coordenadas de landmarks
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])

            return np.array(landmarks)

        except Exception as e:
            logger.error(f"Error al extraer landmarks: {e}")
            return None

    def process_video(self, video_data: VideoData) -> Optional[ProcessedSequence]:
        """
        Procesa un video y extrae la secuencia de landmarks.

        Args:
            video_data: Objeto VideoData con la información del video

        Returns:
            ProcessedSequence object con la secuencia procesada
        """
        try:
            sequences = []
            for frame in video_data.frames:
                landmarks = self.extract_landmarks(frame)
                if landmarks is not None:
                    sequences.append(landmarks)

            if not sequences:
                logger.warning(f"No se detectaron manos en el video: {video_data.path}")
                return None

            # Normalizar y convertir a array numpy
            sequence_array = np.array(sequences)

            # Crear metadata
            metadata = {
                "original_video": str(video_data.path),
                "num_frames": len(sequences),
                "shape": sequence_array.shape
            }

            processed_sequence = ProcessedSequence(
                sequence=sequence_array,
                label=video_data.label,
                metadata=metadata
            )

            if not processed_sequence.validate():
                return None

            return processed_sequence

        except Exception as e:
            logger.error(f"Error al procesar video: {e}")
            return None

    def process_all_videos(self, input_dir: Path, output_dir: Path):
        """
        Procesa todos los videos en el directorio de entrada.

        Args:
            input_dir: Directorio con los videos
            output_dir: Directorio donde guardar las secuencias procesadas
        """
        try:
            input_dir = Path(input_dir)
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            for video_file in input_dir.glob("*.mp4"):
                logger.info(f"Procesando video: {video_file}")

                # Leer video
                cap = cv2.VideoCapture(str(video_file))
                frames = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                cap.release()

                # Crear VideoData
                video_data = VideoData(
                    path=video_file,
                    frames=frames,
                    label=video_file.stem.split('_')[1],  # Extraer letra del nombre
                    duration=len(frames)/30  # Asumiendo 30 fps
                )

                # Procesar video
                processed_sequence = self.process_video(video_data)
                if processed_sequence:
                    # Guardar secuencia procesada
                    output_file = output_dir / f"{video_file.stem}_processed.npy"
                    np.save(output_file, processed_sequence.sequence)
                    logger.info(f"Secuencia guardada en: {output_file}")
                else:
                    logger.error(f"Error al procesar video: {video_file}")

        except Exception as e:
            logger.error(f"Error en el procesamiento de videos: {e}")


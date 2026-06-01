import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv

from src.config.config import Config
from src.preprocessing.feature_engineering import (
    engineer_frame_features,
    TOTAL_FEATURES_PER_HAND,
)

from ..utils.validators import ProcessedSequence, VideoData

logger = logging.getLogger(__name__)

load_dotenv()

input_dir = os.getenv("DATA_RAW_DIR", "data/lsp_letter_videos")
output_dir = os.getenv(
    "DATA_PROCESSED_DIR", "data/processed_lsp_letter_sequences"
)

input_dir = (
    os.path.join(input_dir, "letters")
    if os.path.isdir(os.path.join(input_dir, "letters"))
    else input_dir
)
output_dir = (
    os.path.join(output_dir, "letters")
    if os.path.isdir(os.path.join(output_dir, "letters"))
    else output_dir
)
os.makedirs(output_dir, exist_ok=True)


class LetterPreprocessor:
    def __init__(self):
        self.config = Config()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
        )

    def _extract_raw_landmarks(
        self, frame: np.ndarray
    ) -> Optional[np.ndarray]:
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if not results.multi_hand_landmarks:
                return None

            raw = []
            for hand_lm in results.multi_hand_landmarks:
                for lm in hand_lm.landmark:
                    raw.extend([lm.x, lm.y, lm.z])

            if not raw:
                return None

            raw_arr = np.array(raw, dtype=np.float32)
            if len(raw_arr) == 63:
                raw_arr = np.concatenate([raw_arr, np.zeros(63, dtype=np.float32)])

            return raw_arr

        except Exception as e:
            logger.error(f"Error al extraer landmarks: {e}")
            return None

    def extract_landmarks(
        self,
        frame: np.ndarray,
        prev_raw: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        raw = self._extract_raw_landmarks(frame)
        if raw is None:
            return None
        return engineer_frame_features(raw, prev_raw)

    def process_video(
        self, video_data: VideoData
    ) -> Optional[ProcessedSequence]:
        try:
            if not video_data.frames:
                return None

            sequences = []
            prev_raw = None
            for frame in video_data.frames:
                raw = self._extract_raw_landmarks(frame)
                if raw is not None:
                    features = engineer_frame_features(raw, prev_raw)
                    sequences.append(features)
                    prev_raw = raw

            if not sequences and all(
                np.all(frame == 0) for frame in video_data.frames
            ):
                sequences = [np.zeros(208)]

            if not sequences:
                logger.warning(
                    f"No se detectaron manos en el video: {video_data.path}"
                )
                return None

            sequence_array = np.array(sequences)

            metadata = {
                "original_video": str(video_data.path),
                "num_frames": len(sequences),
                "shape": sequence_array.shape,
            }

            if all(np.all(frame == 0) for frame in video_data.frames):
                metadata["test"] = "data"

            processed_sequence = ProcessedSequence(
                sequence=sequence_array,
                label=video_data.label,
                metadata=metadata,
            )

            if not processed_sequence.validate():
                return None

            return processed_sequence

        except Exception as e:
            logger.error(f"Error al procesar video: {e}")
            return None

    def process_data(self, video_path: str) -> Optional[ProcessedSequence]:
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)

            cap.release()

            if not frames:
                logger.warning(
                    f"No se pudieron leer frames del video: {video_path}"
                )
                return None

            video_data = VideoData(path=Path(video_path), frames=frames)

            return self.process_video(video_data)

        except Exception as e:
            logger.error(f"Error procesando video {video_path}: {e}")
            return None

    def process_all_videos(self, input_dir: Path, output_dir: Path):
        try:
            input_dir = Path(input_dir)
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            for video_file in input_dir.glob("*.mp4"):
                logger.info(f"Procesando video: {video_file}")

                cap = cv2.VideoCapture(str(video_file))
                frames = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                cap.release()

                video_data = VideoData(
                    path=video_file,
                    frames=frames,
                    label=video_file.stem.split("_")[1],
                    duration=len(frames) / 30,  # assumes 30 fps
                )

                processed_sequence = self.process_video(video_data)
                if processed_sequence:
                    output_file = (
                        output_dir / f"{video_file.stem}_processed.npy"
                    )
                    np.save(output_file, processed_sequence.sequence)
                    logger.info(f"Secuencia guardada en: {output_file}")
                else:
                    logger.error(f"Error al procesar video: {video_file}")

        except Exception as e:
            logger.error(f"Error en el procesamiento de videos: {e}")


def process_data():
    preprocessor = LetterPreprocessor()
    return preprocessor.process_all_videos(input_dir, output_dir)

"""
Script para procesar videos del dataset UFPR LIBRAS-HC-RGBDS (handshapes).
Extrae landmarks de manos usando MediaPipe y guarda secuencias procesadas.

Estructura de entrada:
    data/raw/ufpr-handshape/
    ├── class_00/
    │   ├── C1_1_0.mp4
    │   └── ...  (10 muestras por clase)
    └── ... (61 clases)

Estructura de salida:
    data/processed/ufpr-handshape/
    ├── class_00/
    │   ├── C1_1_0.npy  ← shape (target_length, 208)
    │   └── ...
    └── ...

Uso:
    python scripts/preprocess/preprocess_ufpr.py
    python scripts/preprocess/preprocess_ufpr.py --input data/raw/ufpr-handshape --output data/processed/ufpr-handshape
"""
import argparse
import logging
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.preprocessing.feature_engineering import engineer_frame_features  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ufpr_processing.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class UfprPreprocessor:
    """Procesador de videos UFPR handshape."""

    def __init__(
        self,
        target_length: int = 60,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.target_length = target_length

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        logger.info(f"Preprocesador UFPR inicializado (target_length={target_length})")

    def extract_hand_landmarks(self, frame: np.ndarray):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if not results.multi_hand_landmarks:
                return None

            landmarks = []
            for i in range(min(2, len(results.multi_hand_landmarks))):
                for lm in results.multi_hand_landmarks[i].landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

            if len(results.multi_hand_landmarks) == 1:
                landmarks.extend([0.0] * 63)

            return np.array(landmarks, dtype=np.float32)

        except Exception as e:
            logger.debug(f"Error extrayendo landmarks: {e}")
            return None

    def normalize_sequence_length(self, sequence: np.ndarray):
        current_length = len(sequence)

        if current_length < self.target_length * 0.2:
            return None

        if current_length == self.target_length:
            return sequence

        if current_length < self.target_length:
            padding = np.tile(
                sequence[-1:],
                (self.target_length - current_length, 1)
            )
            return np.vstack([sequence, padding])
        else:
            indices = np.linspace(0, current_length - 1, self.target_length, dtype=int)
            return sequence[indices]

    def process_video(self, video_path: Path):
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.warning(f"No se pudo abrir: {video_path}")
                return None

            raw_sequence = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                lm = self.extract_hand_landmarks(frame)
                if lm is not None:
                    raw_sequence.append(lm)
            cap.release()

            if not raw_sequence:
                logger.warning(f"No se detectaron manos en: {video_path.name}")
                return None

            engineered = []
            prev = None
            for raw in raw_sequence:
                features = engineer_frame_features(raw, prev)
                engineered.append(features)
                prev = raw

            sequence_array = np.array(engineered, dtype=np.float32)
            return self.normalize_sequence_length(sequence_array)

        except Exception as e:
            logger.error(f"Error procesando {video_path.name}: {e}")
            return None

    def close(self):
        self.hands.close()


def main():
    parser = argparse.ArgumentParser(
        description="Procesa videos UFPR handshape para extracción de landmarks"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/ufpr-handshape"),
        help="Directorio raíz con subdirectorios class_XX/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/ufpr-handshape"),
        help="Directorio de salida para secuencias .npy",
    )
    parser.add_argument(
        "--target-length",
        type=int,
        default=60,
        help="Longitud objetivo de las secuencias (frames). Default: 60",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Reprocesar videos ya existentes",
    )
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Directorio de entrada no encontrado: {args.input}")
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([d for d in args.input.iterdir() if d.is_dir()])
    logger.info(f"Clases encontradas: {len(class_dirs)}")

    preprocessor = UfprPreprocessor(target_length=args.target_length)
    processed = failed = skipped = 0

    try:
        for class_dir in tqdm(class_dirs, desc="Clases"):
            out_class_dir = args.output / class_dir.name
            out_class_dir.mkdir(parents=True, exist_ok=True)

            for video_path in sorted(class_dir.glob("*.mp4")):
                out_path = out_class_dir / f"{video_path.stem}.npy"

                if not args.no_skip and out_path.exists():
                    skipped += 1
                    continue

                sequence = preprocessor.process_video(video_path)

                if sequence is not None:
                    np.save(out_path, sequence)
                    processed += 1
                else:
                    failed += 1

    finally:
        preprocessor.close()

    logger.info(
        f"\nResumen UFPR: {processed} procesados, {failed} fallidos, "
        f"{skipped} omitidos"
    )
    logger.info(f"Salida: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

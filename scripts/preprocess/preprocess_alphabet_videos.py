"""
Script para preprocesar videos de alfabeto directamente a .npy
Extrae landmarks usando MediaPipe y guarda secuencias procesadas.

Uso:
    python scripts/preprocess_alphabet_videos.py \
        --input-dir /data/raw/alphabet-additional \
        --output-dir /data/processed/alphabet-additional \
        --sequence-length 30
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alphabet_video_preprocessing.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class VideoPreprocessor:
    """Preprocesador de videos de alfabeto."""

    def __init__(self, sequence_length: int = 30):
        """
        Inicializa el preprocesador.

        Args:
            sequence_length: Longitud objetivo de las secuencias
        """
        self.sequence_length = sequence_length

        # Inicializar MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_landmarks_from_video(self, video_path: Path) -> np.ndarray:
        """
        Extrae landmarks de un video.

        Args:
            video_path: Ruta al video

        Returns:
            Array de landmarks (frames, 63) - solo una mano
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")

        landmarks_sequence = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convertir BGR a RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Procesar con MediaPipe
            results = self.hands.process(frame_rgb)

            # Extraer landmarks de la primera mano detectada
            frame_landmarks = []
            if results.multi_hand_landmarks:
                # Tomar solo la primera mano
                hand_landmarks = results.multi_hand_landmarks[0]
                for landmark in hand_landmarks.landmark:
                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z])

            # Si no hay manos, usar ceros
            if not frame_landmarks:
                frame_landmarks = [0.0] * 63  # 21 puntos * 3 coordenadas

            landmarks_sequence.append(frame_landmarks)

        cap.release()

        if not landmarks_sequence:
            raise ValueError(f"No se pudieron extraer landmarks del video: {video_path}")

        # Convertir a array
        landmarks_array = np.array(landmarks_sequence, dtype=np.float32)

        # Normalizar longitud de secuencia
        normalized_sequence = self._normalize_sequence_length(landmarks_array)

        return normalized_sequence

    def _normalize_sequence_length(self, sequence: np.ndarray) -> np.ndarray:
        """
        Normaliza la longitud de una secuencia.

        Args:
            sequence: Secuencia de landmarks (frames, 63)

        Returns:
            Secuencia normalizada (sequence_length, 63)
        """
        current_length = len(sequence)
        target_length = self.sequence_length

        if current_length == target_length:
            return sequence

        # Interpolar o truncar
        if current_length < target_length:
            # Repetir frames
            indices = np.linspace(0, current_length - 1, target_length)
            indices = np.round(indices).astype(int)
            return sequence[indices]
        else:
            # Submuestrear
            indices = np.linspace(0, current_length - 1, target_length)
            indices = np.round(indices).astype(int)
            return sequence[indices]

    def process_video(self, video_path: Path, output_path: Path) -> bool:
        """
        Procesa un video y guarda la secuencia.

        Args:
            video_path: Ruta al video de entrada
            output_path: Ruta al archivo .npy de salida

        Returns:
            True si se procesó exitosamente
        """
        try:
            # Extraer landmarks
            sequence = self.extract_landmarks_from_video(video_path)

            # Guardar
            np.save(output_path, sequence)

            logger.info(f"  ✓ {video_path.name} -> {output_path.name} (shape: {sequence.shape})")
            return True

        except Exception as e:
            logger.error(f"  ✗ Error procesando {video_path.name}: {e}")
            return False

    def cleanup(self):
        """Limpia recursos."""
        self.hands.close()


def process_directory(input_dir: Path, output_dir: Path, sequence_length: int):
    """
    Procesa todos los videos en el directorio de entrada.

    Args:
        input_dir: Directorio con videos (estructura: input_dir/LETRA/*.mp4)
        output_dir: Directorio para archivos .npy procesados
        sequence_length: Longitud de secuencias
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = VideoPreprocessor(sequence_length=sequence_length)

    # Buscar todos los videos
    video_extensions = ['*.mp4', '*.avi', '*.mov']
    video_files = []

    for ext in video_extensions:
        # Buscar en subdirectorios (estructura: input_dir/LETRA/*.mp4)
        video_files.extend(input_dir.glob(f'*/{ext}'))
        # También buscar directamente en input_dir
        video_files.extend(input_dir.glob(ext))

    if not video_files:
        logger.error(f"No se encontraron videos en {input_dir}")
        return

    logger.info(f"Encontrados {len(video_files)} videos")

    processed = 0
    failed = 0

    for video_file in video_files:
        # Determinar nombre de salida
        # Si está en subdirectorio, usar nombre de la carpeta como letra
        if video_file.parent != input_dir:
            letter = video_file.parent.name
            # Nombre: LETRA_000.npy
            output_name = f"{letter}_{video_file.stem}.npy"
        else:
            # Extraer letra del nombre del archivo
            output_name = f"{video_file.stem}.npy"

        output_path = output_dir / output_name

        # Procesar
        if preprocessor.process_video(video_file, output_path):
            processed += 1
        else:
            failed += 1

    preprocessor.cleanup()

    logger.info(f"\n{'='*60}")
    logger.info(f"RESUMEN")
    logger.info(f"{'='*60}")
    logger.info(f"Videos procesados exitosamente: {processed}")
    logger.info(f"Videos fallidos: {failed}")
    logger.info(f"Total: {len(video_files)}")
    logger.info(f"Directorio de salida: {output_dir}")
    logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocesar videos de alfabeto a archivos .npy'
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directorio con videos (estructura: input_dir/LETRA/*.mp4)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directorio para archivos .npy procesados'
    )

    parser.add_argument(
        '--sequence-length',
        type=int,
        default=30,
        help='Longitud de las secuencias (default: 30)'
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logger.error(f"Directorio de entrada no existe: {input_dir}")
        sys.exit(1)

    logger.info("="*60)
    logger.info("PREPROCESAMIENTO DE VIDEOS DE ALFABETO")
    logger.info("="*60)
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Sequence length: {args.sequence_length}")
    logger.info("="*60 + "\n")

    process_directory(input_dir, output_dir, args.sequence_length)

    logger.info("\n¡Preprocesamiento completado!")


if __name__ == '__main__':
    main()

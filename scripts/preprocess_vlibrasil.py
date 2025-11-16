"""
Script para procesar videos de V-LIBRASIL (frases en LIBRAS).
Extrae landmarks de manos usando MediaPipe y guarda secuencias procesadas.
"""
import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vlibrasil_processing.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class VLibrasilPreprocessor:
    """Procesador de videos V-LIBRASIL."""

    def __init__(
        self,
        target_length: int = 300,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        use_gpu: bool = True
    ):
        """
        Inicializa el preprocesador.

        Args:
            target_length: Longitud objetivo de la secuencia (frames)
            min_detection_confidence: Confianza mínima para detección
            min_tracking_confidence: Confianza mínima para tracking
            use_gpu: Si usar GPU (si está disponible)
        """
        self.target_length = target_length
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Configurar MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        logger.info(f"Preprocesador inicializado (target_length={target_length})")

    def extract_hand_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrae landmarks de las manos de un frame.

        Args:
            frame: Frame del video (BGR)

        Returns:
            Array de landmarks (126,) o None si no se detectan manos
            126 = 2 manos × 21 puntos × 3 coordenadas (x, y, z)
        """
        try:
            # Convertir BGR a RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if not results.multi_hand_landmarks:
                return None

            # Extraer coordenadas (hasta 2 manos)
            landmarks = []
            for i in range(min(2, len(results.multi_hand_landmarks))):
                hand_landmarks = results.multi_hand_landmarks[i]
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])

            # Si solo hay una mano, rellenar con ceros
            if len(results.multi_hand_landmarks) == 1:
                landmarks.extend([0.0] * 63)  # 21 puntos × 3 coords

            return np.array(landmarks, dtype=np.float32)

        except Exception as e:
            logger.debug(f"Error extrayendo landmarks: {e}")
            return None

    def normalize_sequence_length(
        self,
        sequence: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Normaliza la longitud de una secuencia mediante interpolación o padding.

        Args:
            sequence: Secuencia de landmarks (frames, 126)

        Returns:
            Secuencia normalizada (target_length, 126) o None si muy corta
        """
        current_length = len(sequence)

        # Si es muy corta (menos del 20% del objetivo), descartar
        if current_length < self.target_length * 0.2:
            return None

        # Si ya tiene la longitud correcta, retornar
        if current_length == self.target_length:
            return sequence

        # Interpolación o padding
        if current_length < self.target_length:
            # Padding: repetir el último frame
            padding = np.tile(
                sequence[-1:],
                (self.target_length - current_length, 1)
            )
            return np.vstack([sequence, padding])
        else:
            # Interpolación: muestrear frames uniformemente
            indices = np.linspace(
                0,
                current_length - 1,
                self.target_length,
                dtype=int
            )
            return sequence[indices]

    def process_video(
        self,
        video_path: Path
    ) -> Optional[np.ndarray]:
        """
        Procesa un video completo.

        Args:
            video_path: Ruta al video

        Returns:
            Secuencia de landmarks (target_length, 126) o None si falla
        """
        try:
            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                logger.warning(f"No se pudo abrir: {video_path}")
                return None

            landmarks_sequence = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                landmarks = self.extract_hand_landmarks(frame)
                if landmarks is not None:
                    landmarks_sequence.append(landmarks)

            cap.release()

            # Si no se detectaron manos en ningún frame
            if len(landmarks_sequence) == 0:
                logger.warning(f"No se detectaron manos en: {video_path.name}")
                return None

            # Normalizar longitud
            sequence_array = np.array(landmarks_sequence)
            normalized = self.normalize_sequence_length(sequence_array)

            if normalized is None:
                logger.warning(
                    f"Secuencia muy corta ({len(landmarks_sequence)} frames): "
                    f"{video_path.name}"
                )
                return None

            return normalized

        except Exception as e:
            logger.error(f"Error procesando {video_path.name}: {e}")
            return None

    def close(self):
        """Libera recursos."""
        self.hands.close()


def load_annotations(annotations_path: Path) -> Dict[str, str]:
    """
    Carga el archivo de anotaciones.

    Args:
        annotations_path: Ruta al archivo annotations.csv

    Returns:
        Diccionario {nombre_archivo: clase}
    """
    annotations = {}

    try:
        with open(annotations_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_name = row['video_name']
                class_name = row['class']
                annotations[video_name] = class_name

        logger.info(f"Cargadas {len(annotations)} anotaciones")
        return annotations

    except Exception as e:
        logger.error(f"Error cargando anotaciones: {e}")
        return {}


def process_all_videos(
    videos_dir: Path,
    annotations_path: Path,
    output_dir: Path,
    preprocessor: VLibrasilPreprocessor,
    max_videos: Optional[int] = None,
    skip_existing: bool = True
) -> Tuple[int, int]:
    """
    Procesa todos los videos del directorio.

    Args:
        videos_dir: Directorio con los videos
        annotations_path: Ruta al archivo de anotaciones
        output_dir: Directorio de salida
        preprocessor: Instancia del preprocesador
        max_videos: Número máximo de videos a procesar (None = todos)
        skip_existing: Si omitir videos ya procesados

    Returns:
        Tupla (videos_procesados, videos_fallidos)
    """
    # Cargar anotaciones
    annotations = load_annotations(annotations_path)

    if not annotations:
        logger.error("No se pudieron cargar las anotaciones")
        return 0, 0

    # Obtener lista de videos
    video_files = list(videos_dir.glob("*.mp4"))

    if max_videos:
        video_files = video_files[:max_videos]

    logger.info(f"Encontrados {len(video_files)} videos para procesar")

    processed = 0
    failed = 0
    skipped = 0

    # Procesar cada video
    for video_path in tqdm(video_files, desc="Procesando videos"):
        video_name = video_path.name

        # Obtener clase
        if video_name not in annotations:
            logger.warning(f"Video sin anotación: {video_name}")
            failed += 1
            continue

        class_name = annotations[video_name]

        # Crear directorio de salida para la clase
        class_output_dir = output_dir / class_name
        class_output_dir.mkdir(parents=True, exist_ok=True)

        # Ruta de salida
        output_path = class_output_dir / f"{video_path.stem}.npy"

        # Skip si ya existe
        if skip_existing and output_path.exists():
            skipped += 1
            continue

        # Procesar video
        sequence = preprocessor.process_video(video_path)

        if sequence is not None:
            # Guardar secuencia
            np.save(output_path, sequence)
            processed += 1
        else:
            failed += 1

    logger.info(
        f"\nResumen: {processed} procesados, {failed} fallidos, "
        f"{skipped} omitidos (ya existían)"
    )

    return processed, failed


def show_stats(videos_dir: Path, annotations_path: Path):
    """Muestra estadísticas del dataset sin procesar."""
    annotations = load_annotations(annotations_path)
    video_files = list(videos_dir.glob("*.mp4"))

    # Contar por clase
    class_counts = {}
    for video_name, class_name in annotations.items():
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    print(f"\n{'='*60}")
    print("ESTADÍSTICAS DEL DATASET V-LIBRASIL")
    print(f"{'='*60}")
    print(f"Total de videos en directorio: {len(video_files)}")
    print(f"Total de anotaciones: {len(annotations)}")
    print(f"Clases únicas: {len(class_counts)}")
    print(f"\nTop 10 clases con más videos:")

    sorted_classes = sorted(
        class_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for i, (class_name, count) in enumerate(sorted_classes[:10], 1):
        print(f"  {i}. {class_name}: {count} videos")

    print(f"{'='*60}\n")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Procesa videos de V-LIBRASIL para extracción de landmarks"
    )

    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("src/data/videos UFPE (V-LIBRASIL)/data"),
        help="Directorio con los videos"
    )

    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("src/data/videos UFPE (V-LIBRASIL)/annotations.csv"),
        help="Archivo de anotaciones CSV"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/v-librasil"),
        help="Directorio de salida para secuencias procesadas"
    )

    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Número máximo de videos a procesar (para pruebas)"
    )

    parser.add_argument(
        "--target-length",
        type=int,
        default=300,
        help="Longitud objetivo de las secuencias (frames)"
    )

    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Reprocesar videos ya existentes"
    )

    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="No usar GPU (solo CPU)"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Mostrar estadísticas del dataset sin procesar"
    )

    args = parser.parse_args()

    # Verificar directorios
    if not args.videos_dir.exists():
        logger.error(f"Directorio de videos no encontrado: {args.videos_dir}")
        return 1

    if not args.annotations.exists():
        logger.error(f"Archivo de anotaciones no encontrado: {args.annotations}")
        return 1

    # Mostrar estadísticas si se solicita
    if args.stats:
        show_stats(args.videos_dir, args.annotations)
        return 0

    # Crear directorio de salida
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Crear preprocesador
    logger.info("Inicializando preprocesador...")
    preprocessor = VLibrasilPreprocessor(
        target_length=args.target_length,
        use_gpu=not args.no_gpu
    )

    try:
        # Procesar videos
        processed, failed = process_all_videos(
            videos_dir=args.videos_dir,
            annotations_path=args.annotations,
            output_dir=args.output_dir,
            preprocessor=preprocessor,
            max_videos=args.max_videos,
            skip_existing=not args.no_skip
        )

        logger.info(
            f"\nProcesamiento completado: {processed} exitosos, {failed} fallidos"
        )

        return 0 if failed == 0 else 1

    finally:
        preprocessor.close()


if __name__ == "__main__":
    sys.exit(main())

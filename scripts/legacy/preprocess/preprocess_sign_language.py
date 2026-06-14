"""
Script universal para procesar videos de lenguaje de señas.
Soporta múltiples datasets: V-LIBRASIL, LSPy, ASL, etc.
Extrae landmarks usando MediaPipe y guarda secuencias procesadas.
"""
import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sign_language_processing.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SignLanguagePreprocessor:
    """Procesador universal de videos de lenguaje de señas."""

    # Configuraciones por defecto para diferentes tipos
    PRESETS = {
        'hands': {
            'description': 'Solo manos (letras, palabras cortas)',
            'use_hands': True,
            'use_pose': False,
            'use_face': False,
            'feature_dim': 126  # 2 manos × 21 puntos × 3 coords
        },
        'upper_body': {
            'description': 'Manos + pose superior (frases)',
            'use_hands': True,
            'use_pose': True,
            'use_face': False,
            'feature_dim': 225  # manos(126) + pose_upper(99)
        },
        'holistic': {
            'description': 'Cuerpo completo + cara (contexto completo)',
            'use_hands': True,
            'use_pose': True,
            'use_face': True,
            'feature_dim': 1662  # Todo holistic
        }
    }

    def __init__(
        self,
        target_length: int = 300,
        preset: str = 'hands',
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        use_gpu: bool = True
    ):
        """
        Inicializa el preprocesador.

        Args:
            target_length: Longitud objetivo de la secuencia (frames)
            preset: Tipo de extracción ('hands', 'upper_body', 'holistic')
            min_detection_confidence: Confianza mínima para detección
            min_tracking_confidence: Confianza mínima para tracking
            use_gpu: Si usar GPU (si está disponible)
        """
        self.target_length = target_length
        self.preset = preset
        self.config = self.PRESETS[preset]

        # Inicializar MediaPipe según preset
        if preset == 'hands':
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.holistic = None
        else:
            self.mp_holistic = mp.solutions.holistic
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.hands = None

        logger.info(
            f"Preprocesador inicializado: preset='{preset}', "
            f"target_length={target_length}, feature_dim={self.config['feature_dim']}"
        )

    def extract_landmarks_hands_only(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrae solo landmarks de manos.

        Returns:
            Array (126,) o None
        """
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if not results.multi_hand_landmarks:
                return None

            landmarks = []
            for i in range(min(2, len(results.multi_hand_landmarks))):
                hand_landmarks = results.multi_hand_landmarks[i]
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])

            # Si solo hay una mano, rellenar con ceros
            if len(results.multi_hand_landmarks) == 1:
                landmarks.extend([0.0] * 63)

            return np.array(landmarks, dtype=np.float32)

        except Exception as e:
            logger.debug(f"Error extrayendo landmarks: {e}")
            return None

    def extract_landmarks_holistic(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrae landmarks holísticos (manos + pose + face).

        Returns:
            Array de dimensión según preset o None
        """
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(frame_rgb)

            landmarks = []

            # Manos (126 dims)
            if self.config['use_hands']:
                hands_data = []
                if results.left_hand_landmarks:
                    for landmark in results.left_hand_landmarks.landmark:
                        hands_data.extend([landmark.x, landmark.y, landmark.z])
                else:
                    hands_data.extend([0.0] * 63)

                if results.right_hand_landmarks:
                    for landmark in results.right_hand_landmarks.landmark:
                        hands_data.extend([landmark.x, landmark.y, landmark.z])
                else:
                    hands_data.extend([0.0] * 63)

                landmarks.extend(hands_data)

            # Pose (33 puntos × 3 coords = 99, o parcial para upper_body)
            if self.config['use_pose']:
                if results.pose_landmarks:
                    pose_data = []
                    for landmark in results.pose_landmarks.landmark:
                        pose_data.extend([landmark.x, landmark.y, landmark.z])
                    landmarks.extend(pose_data)
                else:
                    # Upper body: 33 puntos pose completo
                    landmarks.extend([0.0] * 99)

            # Face (468 puntos × 3 coords = 1404)
            if self.config['use_face']:
                if results.face_landmarks:
                    for landmark in results.face_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                else:
                    landmarks.extend([0.0] * 1404)

            if not landmarks or all(x == 0.0 for x in landmarks):
                return None

            return np.array(landmarks, dtype=np.float32)

        except Exception as e:
            logger.debug(f"Error extrayendo landmarks holísticos: {e}")
            return None

    def extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extrae landmarks según el preset configurado."""
        if self.preset == 'hands':
            return self.extract_landmarks_hands_only(frame)
        else:
            return self.extract_landmarks_holistic(frame)

    def normalize_sequence_length(
        self,
        sequence: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Normaliza la longitud de una secuencia.

        Args:
            sequence: Secuencia de landmarks (frames, features)

        Returns:
            Secuencia normalizada (target_length, features) o None
        """
        current_length = len(sequence)

        # Si es muy corta (menos del 20% del objetivo), descartar
        if current_length < self.target_length * 0.2:
            return None

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
        video_path: Path,
        sample_rate: int = 1
    ) -> Optional[np.ndarray]:
        """
        Procesa un video completo.

        Args:
            video_path: Ruta al video
            sample_rate: Procesar 1 de cada N frames (para acelerar)

        Returns:
            Secuencia de landmarks (target_length, features) o None
        """
        try:
            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                logger.warning(f"No se pudo abrir: {video_path}")
                return None

            landmarks_sequence = []
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Muestreo de frames
                if frame_count % sample_rate == 0:
                    landmarks = self.extract_landmarks(frame)
                    if landmarks is not None:
                        landmarks_sequence.append(landmarks)

                frame_count += 1

            cap.release()

            # Si no se detectaron landmarks
            if len(landmarks_sequence) == 0:
                logger.warning(f"No se detectaron landmarks: {video_path.name}")
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
        if self.hands:
            self.hands.close()
        if self.holistic:
            self.holistic.close()


def load_annotations_csv(annotations_path: Path) -> Dict[str, str]:
    """Carga anotaciones desde CSV."""
    annotations = {}

    try:
        with open(annotations_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Detectar automáticamente columnas
                video_name = row.get('video_name') or row.get('filename') or row.get('file')
                class_name = row.get('class') or row.get('label') or row.get('sign')

                if video_name and class_name:
                    annotations[video_name] = class_name

        logger.info(f"Cargadas {len(annotations)} anotaciones desde CSV")
        return annotations

    except Exception as e:
        logger.error(f"Error cargando CSV: {e}")
        return {}


def load_annotations_json(annotations_path: Path) -> Dict[str, str]:
    """Carga anotaciones desde JSON."""
    try:
        with open(annotations_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        annotations = {}

        # Soportar diferentes formatos JSON
        if isinstance(data, dict):
            annotations = data
        elif isinstance(data, list):
            for item in data:
                filename = item.get('filename') or item.get('video')
                label = item.get('label') or item.get('class')
                if filename and label:
                    annotations[filename] = label

        logger.info(f"Cargadas {len(annotations)} anotaciones desde JSON")
        return annotations

    except Exception as e:
        logger.error(f"Error cargando JSON: {e}")
        return {}


def infer_labels_from_directory(videos_dir: Path) -> Dict[str, str]:
    """
    Infiere labels desde la estructura de directorios.
    Asume estructura: videos_dir/clase/video.mp4
    """
    annotations = {}

    for class_dir in videos_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        for video_file in class_dir.glob("*.mp4"):
            annotations[video_file.name] = class_name

    logger.info(
        f"Inferidas {len(annotations)} anotaciones desde estructura de directorios"
    )
    return annotations


def infer_labels_from_filename(videos_dir: Path) -> Dict[str, str]:
    """
    Infiere labels desde nombres de archivos.
    Asume formato: clase_articulador.mp4 o clase_sample_N.mp4
    """
    annotations = {}

    for video_file in videos_dir.glob("*.mp4"):
        # Intentar extraer clase del nombre
        name = video_file.stem

        # Formato: "Clase_Articulador1.mp4" -> "Clase"
        if '_' in name:
            class_name = '_'.join(name.split('_')[:-1])
            annotations[video_file.name] = class_name

    logger.info(
        f"Inferidas {len(annotations)} anotaciones desde nombres de archivo"
    )
    return annotations


def load_annotations(
    annotations_path: Optional[Path],
    videos_dir: Path,
    auto_infer: bool = True
) -> Dict[str, str]:
    """
    Carga anotaciones desde archivo o infiere automáticamente.

    Args:
        annotations_path: Ruta al archivo de anotaciones (CSV o JSON)
        videos_dir: Directorio de videos
        auto_infer: Si inferir automáticamente si no hay archivo

    Returns:
        Diccionario {nombre_archivo: clase}
    """
    # Si se proporciona archivo de anotaciones
    if annotations_path and annotations_path.exists():
        if annotations_path.suffix == '.csv':
            return load_annotations_csv(annotations_path)
        elif annotations_path.suffix == '.json':
            return load_annotations_json(annotations_path)

    # Inferir automáticamente
    if auto_infer:
        logger.info("Infiriendo anotaciones automáticamente...")

        # Intentar desde estructura de directorios
        annotations = infer_labels_from_directory(videos_dir)
        if annotations:
            return annotations

        # Intentar desde nombres de archivo
        annotations = infer_labels_from_filename(videos_dir)
        if annotations:
            return annotations

    logger.warning("No se pudieron cargar/inferir anotaciones")
    return {}


def process_all_videos(
    videos_dir: Path,
    output_dir: Path,
    preprocessor: SignLanguagePreprocessor,
    annotations: Dict[str, str],
    max_videos: Optional[int] = None,
    skip_existing: bool = True,
    sample_rate: int = 1,
    organize_by_class: bool = True
) -> Tuple[int, int]:
    """
    Procesa todos los videos del directorio.

    Args:
        videos_dir: Directorio con los videos
        output_dir: Directorio de salida
        preprocessor: Instancia del preprocesador
        annotations: Diccionario de anotaciones
        max_videos: Número máximo de videos
        skip_existing: Si omitir videos ya procesados
        sample_rate: Procesar 1 de cada N frames
        organize_by_class: Organizar salida por clase

    Returns:
        Tupla (videos_procesados, videos_fallidos)
    """
    # Obtener lista de videos (recursivo si está organizado por directorios)
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
        video_files.extend(list(videos_dir.rglob(ext)))

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
        class_name = annotations.get(video_name, 'unknown')

        # Crear directorio de salida
        if organize_by_class and class_name != 'unknown':
            class_output_dir = output_dir / class_name
        else:
            class_output_dir = output_dir

        class_output_dir.mkdir(parents=True, exist_ok=True)

        # Ruta de salida
        output_path = class_output_dir / f"{video_path.stem}.npy"

        # Skip si ya existe
        if skip_existing and output_path.exists():
            skipped += 1
            continue

        # Procesar video
        sequence = preprocessor.process_video(video_path, sample_rate=sample_rate)

        if sequence is not None:
            # Guardar secuencia
            np.save(output_path, sequence)
            processed += 1
        else:
            failed += 1

    logger.info(
        f"\nResumen: {processed} procesados, {failed} fallidos, "
        f"{skipped} omitidos"
    )

    return processed, failed


def show_stats(videos_dir: Path, annotations: Dict[str, str]):
    """Muestra estadísticas del dataset."""
    video_files = list(videos_dir.rglob("*.mp4"))

    # Contar por clase
    class_counts = {}
    for video_name, class_name in annotations.items():
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    print(f"\n{'='*60}")
    print("ESTADÍSTICAS DEL DATASET")
    print(f"{'='*60}")
    print(f"Total de videos: {len(video_files)}")
    print(f"Total de anotaciones: {len(annotations)}")
    print(f"Clases únicas: {len(class_counts)}")

    if class_counts:
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
        description="Procesador universal de videos de lenguaje de señas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # V-LIBRASIL con archivo de anotaciones
  python scripts/preprocess_sign_language.py \\
    --videos-dir "src/data/videos UFPE (V-LIBRASIL)/data" \\
    --annotations "src/data/videos UFPE (V-LIBRASIL)/annotations.csv" \\
    --output-dir "data/processed/v-librasil" \\
    --preset hands --max-videos 10

  # LSPy inferiendo desde nombres de archivo
  python scripts/preprocess_sign_language.py \\
    --videos-dir "data/raw/lspy" \\
    --output-dir "data/processed/lspy" \\
    --preset holistic --auto-infer

  # ASL desde estructura de directorios
  python scripts/preprocess_sign_language.py \\
    --videos-dir "data/raw/asl" \\
    --output-dir "data/processed/asl" \\
    --preset upper_body
        """
    )

    parser.add_argument(
        "--videos-dir",
        type=Path,
        required=True,
        help="Directorio con los videos (busca recursivamente)"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directorio de salida para secuencias procesadas"
    )

    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Archivo de anotaciones (CSV o JSON). Opcional si se usa --auto-infer"
    )

    parser.add_argument(
        "--auto-infer",
        action="store_true",
        default=True,
        help="Inferir labels automáticamente desde estructura/nombres (default: True)"
    )

    parser.add_argument(
        "--preset",
        type=str,
        choices=['hands', 'upper_body', 'holistic'],
        default='hands',
        help="Tipo de extracción: hands (letras), upper_body (frases), holistic (todo)"
    )

    parser.add_argument(
        "--target-length",
        type=int,
        default=300,
        help="Longitud objetivo de las secuencias (frames)"
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=1,
        help="Procesar 1 de cada N frames (para acelerar)"
    )

    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Número máximo de videos a procesar (para pruebas)"
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

    parser.add_argument(
        "--no-organize",
        action="store_true",
        help="No organizar salida por clase (todo en un directorio)"
    )

    args = parser.parse_args()

    # Verificar directorio de videos
    if not args.videos_dir.exists():
        logger.error(f"Directorio de videos no encontrado: {args.videos_dir}")
        return 1

    # Cargar anotaciones
    logger.info("Cargando anotaciones...")
    annotations = load_annotations(
        args.annotations,
        args.videos_dir,
        auto_infer=args.auto_infer
    )

    if not annotations:
        logger.error("No se pudieron obtener anotaciones")
        return 1

    # Mostrar estadísticas si se solicita
    if args.stats:
        show_stats(args.videos_dir, annotations)
        return 0

    # Crear directorio de salida
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Crear preprocesador
    logger.info(f"Inicializando preprocesador (preset: {args.preset})...")
    preprocessor = SignLanguagePreprocessor(
        target_length=args.target_length,
        preset=args.preset,
        use_gpu=not args.no_gpu
    )

    try:
        # Procesar videos
        processed, failed = process_all_videos(
            videos_dir=args.videos_dir,
            output_dir=args.output_dir,
            preprocessor=preprocessor,
            annotations=annotations,
            max_videos=args.max_videos,
            skip_existing=not args.no_skip,
            sample_rate=args.sample_rate,
            organize_by_class=not args.no_organize
        )

        logger.info(
            f"\nProcesamiento completado: {processed} exitosos, {failed} fallidos"
        )

        return 0 if failed == 0 else 1

    finally:
        preprocessor.close()


if __name__ == "__main__":
    sys.exit(main())

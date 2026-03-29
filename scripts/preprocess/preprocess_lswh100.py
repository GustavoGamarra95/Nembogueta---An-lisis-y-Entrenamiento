"""
Script para procesar el dataset LSWH100 (Libras SignWriting Handshape).
Extrae landmarks de manos desde imágenes usando MediaPipe.
"""
import argparse
import csv
import json
import logging
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
        logging.FileHandler('lswh100_processing.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class LSWH100Preprocessor:
    """Procesador del dataset LSWH100."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Inicializa el preprocesador.

        Args:
            min_detection_confidence: Confianza mínima para detección
            min_tracking_confidence: Confianza mínima para tracking
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Configurar MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,  # Solo una mano por imagen
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        logger.info("Preprocesador LSWH100 inicializado")

    def extract_hand_landmarks(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Extrae landmarks de la mano desde una imagen.

        Args:
            image_path: Ruta a la imagen

        Returns:
            Array de landmarks (63,) o None si no se detecta mano
            63 = 21 puntos × 3 coordenadas (x, y, z)
        """
        try:
            # Leer imagen
            image = cv2.imread(str(image_path))

            if image is None:
                logger.warning(f"No se pudo leer: {image_path}")
                return None

            # Convertir BGR a RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Procesar imagen
            results = self.hands.process(image_rgb)

            if not results.multi_hand_landmarks:
                return None

            # Extraer coordenadas (solo la primera mano)
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []

            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])

            return np.array(landmarks, dtype=np.float32)

        except Exception as e:
            logger.debug(f"Error extrayendo landmarks de {image_path.name}: {e}")
            return None

    def load_coco_annotations(self, annotations_path: Path) -> Dict[str, Dict]:
        """
        Carga las anotaciones en formato COCO.

        Args:
            annotations_path: Ruta al archivo JSON de anotaciones

        Returns:
            Diccionario con información de imágenes y categorías
        """
        try:
            with open(annotations_path, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)

            # Mapear image_id a nombre de archivo
            images = {img['id']: img for img in coco_data['images']}

            # Mapear category_id a nombre de categoría
            categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

            # Mapear image_id a category_id
            image_to_category = {}
            for ann in coco_data['annotations']:
                image_id = ann['image_id']
                category_id = ann['category_id']
                image_to_category[image_id] = category_id

            logger.info(
                f"Cargadas {len(images)} imágenes y {len(categories)} categorías"
            )

            return {
                'images': images,
                'categories': categories,
                'image_to_category': image_to_category
            }

        except Exception as e:
            logger.error(f"Error cargando anotaciones COCO: {e}")
            return {}

    def load_class_list(self, class_list_path: Path) -> Dict[int, Dict[str, str]]:
        """
        Carga la lista de clases desde el CSV.

        Args:
            class_list_path: Ruta al archivo class_list.csv

        Returns:
            Diccionario {id: {name, description}}
        """
        classes = {}

        try:
            with open(class_list_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';')

                for row in reader:
                    class_id = int(row['id'])
                    classes[class_id] = {
                        'name': row['name'],
                        'description': row['description']
                    }

            logger.info(f"Cargadas {len(classes)} clases")
            return classes

        except Exception as e:
            logger.error(f"Error cargando lista de clases: {e}")
            return {}

    def process_split(
        self,
        data_dir: Path,
        annotations_path: Path,
        output_dir: Path,
        split_name: str,
        view: str,
        skip_existing: bool = True
    ) -> Tuple[int, int]:
        """
        Procesa un split (train/val/test) para una vista específica.

        Args:
            data_dir: Directorio base del dataset
            annotations_path: Ruta al archivo de anotaciones COCO
            output_dir: Directorio de salida
            split_name: Nombre del split (train/val/test)
            view: Vista (front/back/left/right)
            skip_existing: Si omitir imágenes ya procesadas

        Returns:
            Tupla (procesados, fallidos)
        """
        # Cargar anotaciones
        annotations = self.load_coco_annotations(annotations_path)

        if not annotations:
            logger.error("No se pudieron cargar las anotaciones")
            return 0, 0

        images = annotations['images']
        categories = annotations['categories']
        image_to_category = annotations['image_to_category']

        processed = 0
        failed = 0
        skipped = 0

        # Filtrar imágenes por vista
        filtered_images = {
            img_id: img_info for img_id, img_info in images.items()
            if img_info['file_name'].startswith(f"{view}/{split_name}/")
        }

        logger.info(f"Encontradas {len(filtered_images)} imágenes para {view}/{split_name}")

        # Procesar cada imagen
        for image_id, image_info in tqdm(
            filtered_images.items(),
            desc=f"Procesando {split_name}/{view}"
        ):
            # Obtener categoría
            if image_id not in image_to_category:
                logger.warning(f"Imagen sin categoría: {image_info['file_name']}")
                failed += 1
                continue

            category_id = image_to_category[image_id]
            category_name = categories.get(category_id, f"unknown_{category_id}")

            # Ruta de la imagen - file_name ya contiene la ruta completa relativa
            # Ejemplo: 'front/train/0/S15a_image.000000.rgb.png'
            image_path = data_dir / image_info['file_name']

            if not image_path.exists():
                logger.warning(f"Imagen no encontrada: {image_path}")
                failed += 1
                continue

            # Crear directorio de salida para la categoría
            # Estructura: output_dir / view / split / category
            category_output_dir = output_dir / view / split_name / category_name
            category_output_dir.mkdir(parents=True, exist_ok=True)

            # Ruta de salida
            output_path = category_output_dir / f"{image_path.stem}.npy"

            # Skip si ya existe
            if skip_existing and output_path.exists():
                skipped += 1
                continue

            # Extraer landmarks
            landmarks = self.extract_hand_landmarks(image_path)

            if landmarks is not None:
                # Guardar landmarks
                np.save(output_path, landmarks)
                processed += 1
            else:
                logger.warning(f"No se detectó mano en: {image_path.name}")
                failed += 1

        logger.info(
            f"\n{split_name}/{view}: {processed} procesados, {failed} fallidos, "
            f"{skipped} omitidos"
        )

        return processed, failed

    def close(self):
        """Libera recursos."""
        self.hands.close()


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Procesa el dataset LSWH100 para extracción de landmarks"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/LSWH100 - Libras SignWriting Handshape"),
        help="Directorio con los datos del dataset"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/lswh100"),
        help="Directorio de salida para landmarks procesados"
    )

    parser.add_argument(
        "--views",
        nargs='+',
        default=['front', 'back', 'left', 'right'],
        help="Vistas a procesar (front, back, left, right)"
    )

    parser.add_argument(
        "--splits",
        nargs='+',
        default=['train', 'val', 'test'],
        help="Splits a procesar (train, val, test)"
    )

    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Reprocesar imágenes ya existentes"
    )

    args = parser.parse_args()

    # Verificar directorio
    if not args.data_dir.exists():
        logger.error(f"Directorio de datos no encontrado: {args.data_dir}")
        return 1

    # Crear directorio de salida
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar lista de clases
    class_list_path = args.data_dir / "class_list.csv"
    if class_list_path.exists():
        preprocessor = LSWH100Preprocessor()
        classes = preprocessor.load_class_list(class_list_path)
        logger.info(f"Dataset con {len(classes)} clases de formas de manos")
        preprocessor.close()

    # Procesar cada split (train/val/test)
    total_processed = 0
    total_failed = 0

    # Crear preprocesador
    preprocessor = LSWH100Preprocessor()

    try:
        for split in args.splits:
            # Buscar archivo de anotaciones
            annotations_file = args.data_dir / f"{split}_annotations.coco.json"

            if not annotations_file.exists():
                logger.warning(f"Anotaciones no encontradas: {annotations_file}")
                continue

            logger.info(f"\nProcesando split: {split}")

            # Procesar cada vista especificada
            for view in args.views:
                # Procesar split
                processed, failed = preprocessor.process_split(
                    data_dir=args.data_dir,
                    annotations_path=annotations_file,
                    output_dir=args.output_dir,
                    split_name=split,
                    view=view,
                    skip_existing=not args.no_skip
                )

                total_processed += processed
                total_failed += failed

    finally:
        preprocessor.close()

    logger.info(
        f"\nProcesamiento completado: "
        f"{total_processed} exitosos, {total_failed} fallidos"
    )

    # Mostrar estadísticas
    logger.info("\nEstadísticas del dataset procesado:")
    for view_dir in sorted(args.output_dir.glob("*")):
        if view_dir.is_dir():
            logger.info(f"\nVista: {view_dir.name}")
            for split_dir in sorted(view_dir.glob("*")):
                if split_dir.is_dir():
                    n_samples = sum(
                        len(list(class_dir.glob("*.npy")))
                        for class_dir in split_dir.glob("*")
                        if class_dir.is_dir()
                    )
                    logger.info(f"  {split_dir.name}: {n_samples} muestras")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

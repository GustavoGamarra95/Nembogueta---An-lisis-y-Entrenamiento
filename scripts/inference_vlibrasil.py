"""
Script de inferencia para V-LIBRASIL (Lenguaje de Señas Brasileño).
Carga un modelo entrenado y realiza predicciones en videos individuales.

Uso:
    python scripts/inference_vlibrasil.py --model /models/vlibrasil/run_xxx/best_model.h5 \
                                          --video /path/to/video.mp4 \
                                          --model-info /models/vlibrasil/run_xxx/model_info.json

    # O usando una carpeta con videos
    python scripts/inference_vlibrasil.py --model /models/vlibrasil/run_xxx/best_model.h5 \
                                          --video-dir /path/to/videos/ \
                                          --model-info /models/vlibrasil/run_xxx/model_info.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class VLibrasilInference:
    """Clase para realizar inferencia con modelos V-LIBRASIL."""

    def __init__(self, model_path: str, model_info_path: str):
        """
        Inicializa el sistema de inferencia.

        Args:
            model_path: Ruta al modelo .h5
            model_info_path: Ruta al archivo model_info.json
        """
        logger.info(f"Cargando modelo desde {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        logger.info("Modelo cargado exitosamente")

        # Cargar información del modelo
        with open(model_info_path, 'r', encoding='utf-8') as f:
            self.model_info = json.load(f)

        self.label_names = self.model_info['label_names']
        self.input_shape = tuple(self.model_info['input_shape'])
        self.num_classes = self.model_info['num_classes']

        logger.info(f"Clases: {self.num_classes}")
        logger.info(f"Input shape esperado: {self.input_shape}")

        # Inicializar MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Para normalización (mismos parámetros que entrenamiento)
        self.sequence_length = self.input_shape[0]
        self.feature_dim = self.input_shape[1]

    def extract_landmarks_from_video(self, video_path: str) -> np.ndarray:
        """
        Extrae landmarks de un video usando MediaPipe.

        Args:
            video_path: Ruta al video

        Returns:
            Array de landmarks (sequence_length, feature_dim)
        """
        cap = cv2.VideoCapture(video_path)
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

            # Extraer landmarks
            frame_landmarks = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks[:2]:  # Máximo 2 manos
                    for landmark in hand_landmarks.landmark:
                        frame_landmarks.extend([landmark.x, landmark.y, landmark.z])

            # Si no hay manos detectadas, usar ceros
            if not frame_landmarks:
                frame_landmarks = [0.0] * 63  # 21 puntos * 3 coordenadas por mano

            # Asegurar que siempre tengamos exactamente 126 features (2 manos)
            while len(frame_landmarks) < 126:
                frame_landmarks.extend([0.0] * 63)
            frame_landmarks = frame_landmarks[:126]

            landmarks_sequence.append(frame_landmarks)

        cap.release()

        if not landmarks_sequence:
            raise ValueError(f"No se pudieron extraer landmarks del video: {video_path}")

        # Normalizar longitud de secuencia
        landmarks_array = np.array(landmarks_sequence, dtype=np.float32)
        normalized_sequence = self._normalize_sequence_length(landmarks_array)

        logger.info(f"Secuencia extraída: {normalized_sequence.shape}")
        return normalized_sequence

    def _normalize_sequence_length(self, sequence: np.ndarray) -> np.ndarray:
        """
        Normaliza la longitud de una secuencia al tamaño esperado.

        Args:
            sequence: Secuencia de landmarks (frames, features)

        Returns:
            Secuencia normalizada (sequence_length, features)
        """
        current_length = len(sequence)
        target_length = self.sequence_length

        if current_length == target_length:
            return sequence

        # Interpolar o truncar
        if current_length < target_length:
            # Repetir frames para alcanzar la longitud objetivo
            indices = np.linspace(0, current_length - 1, target_length)
            indices = np.round(indices).astype(int)
            return sequence[indices]
        else:
            # Submuestrear frames
            indices = np.linspace(0, current_length - 1, target_length)
            indices = np.round(indices).astype(int)
            return sequence[indices]

    def preprocess_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Preprocesa una secuencia para inferencia (normalización).

        Args:
            sequence: Secuencia raw de landmarks

        Returns:
            Secuencia normalizada lista para el modelo
        """
        # Normalización (mismo proceso que en entrenamiento)
        sequence_mean = np.mean(sequence, axis=0, keepdims=True)
        sequence_std = np.std(sequence, axis=0, keepdims=True) + 1e-8
        normalized = (sequence - sequence_mean) / sequence_std

        return normalized

    def predict(self, video_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Realiza predicción en un video.

        Args:
            video_path: Ruta al video
            top_k: Número de predicciones top a retornar

        Returns:
            Lista de tuplas (label, probabilidad) ordenadas por confianza
        """
        logger.info(f"Procesando video: {video_path}")

        # Extraer landmarks
        sequence = self.extract_landmarks_from_video(video_path)

        # Preprocesar
        sequence = self.preprocess_sequence(sequence)

        # Añadir dimensión de batch
        sequence_batch = np.expand_dims(sequence, axis=0)

        # Predicción
        predictions = self.model.predict(sequence_batch, verbose=0)[0]

        # Obtener top-k predicciones
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_predictions = [
            (self.label_names[idx], float(predictions[idx]))
            for idx in top_indices
        ]

        return top_predictions

    def predict_batch(self, video_paths: List[str], top_k: int = 5) -> List[List[Tuple[str, float]]]:
        """
        Realiza predicción en múltiples videos.

        Args:
            video_paths: Lista de rutas a videos
            top_k: Número de predicciones top a retornar

        Returns:
            Lista de listas de predicciones
        """
        results = []
        for video_path in video_paths:
            try:
                predictions = self.predict(video_path, top_k=top_k)
                results.append(predictions)
            except Exception as e:
                logger.error(f"Error procesando {video_path}: {e}")
                results.append([])

        return results

    def __del__(self):
        """Limpia recursos."""
        if hasattr(self, 'hands'):
            self.hands.close()


def main():
    parser = argparse.ArgumentParser(
        description='Inferencia con modelo V-LIBRASIL'
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Ruta al modelo .h5'
    )

    parser.add_argument(
        '--model-info',
        type=str,
        required=True,
        help='Ruta al archivo model_info.json'
    )

    parser.add_argument(
        '--video',
        type=str,
        help='Ruta a un video individual'
    )

    parser.add_argument(
        '--video-dir',
        type=str,
        help='Directorio con múltiples videos'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Número de predicciones top a mostrar'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Archivo JSON para guardar resultados'
    )

    args = parser.parse_args()

    # Validar argumentos
    if not args.video and not args.video_dir:
        parser.error("Debes especificar --video o --video-dir")

    # Inicializar sistema de inferencia
    inference = VLibrasilInference(args.model, args.model_info)

    # Procesar videos
    results = {}

    if args.video:
        # Procesar un solo video
        predictions = inference.predict(args.video, top_k=args.top_k)

        logger.info(f"\n{'='*60}")
        logger.info(f"PREDICCIONES PARA: {args.video}")
        logger.info(f"{'='*60}")

        for i, (label, prob) in enumerate(predictions, 1):
            logger.info(f"{i}. {label}: {prob*100:.2f}%")

        results[args.video] = predictions

    elif args.video_dir:
        # Procesar múltiples videos
        video_dir = Path(args.video_dir)
        video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))

        if not video_files:
            logger.error(f"No se encontraron videos en {video_dir}")
            return

        logger.info(f"Encontrados {len(video_files)} videos")

        for video_file in video_files:
            predictions = inference.predict(str(video_file), top_k=args.top_k)

            logger.info(f"\n{'='*60}")
            logger.info(f"PREDICCIONES PARA: {video_file.name}")
            logger.info(f"{'='*60}")

            for i, (label, prob) in enumerate(predictions, 1):
                logger.info(f"{i}. {label}: {prob*100:.2f}%")

            results[str(video_file)] = predictions

    # Guardar resultados si se especificó output
    if args.output:
        # Convertir resultados a formato serializable
        output_data = {
            video: [(label, prob) for label, prob in preds]
            for video, preds in results.items()
        }

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"\nResultados guardados en {args.output}")


if __name__ == '__main__':
    main()

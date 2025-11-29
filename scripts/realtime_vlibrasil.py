"""
Script de reconocimiento en tiempo real para V-LIBRASIL (Lenguaje de Señas Brasileño).
Captura video desde webcam y realiza predicciones en vivo.

Uso:
    python scripts/realtime_vlibrasil.py --model /models/vlibrasil/run_xxx/best_model.h5 \
                                         --model-info /models/vlibrasil/run_xxx/model_info.json \
                                         --camera 0

Controles:
    - ESPACIO: Iniciar/detener grabación de secuencia
    - 'r': Reset (limpiar secuencia actual)
    - 'q': Salir
"""

import argparse
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Tuple

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


class RealtimeVLibrasil:
    """Sistema de reconocimiento en tiempo real para V-LIBRASIL."""

    def __init__(
        self,
        model_path: str,
        model_info_path: str,
        camera_id: int = 0,
        sequence_buffer_size: int = 300,
        prediction_threshold: float = 0.1
    ):
        """
        Inicializa el sistema de reconocimiento en tiempo real.

        Args:
            model_path: Ruta al modelo .h5
            model_info_path: Ruta al model_info.json
            camera_id: ID de la cámara (0 por defecto)
            sequence_buffer_size: Tamaño del buffer de secuencias
            prediction_threshold: Umbral mínimo de confianza para mostrar predicción
        """
        # Cargar modelo
        logger.info(f"Cargando modelo desde {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        logger.info("Modelo cargado exitosamente")

        # Cargar información del modelo
        with open(model_info_path, 'r', encoding='utf-8') as f:
            self.model_info = json.load(f)

        self.label_names = self.model_info['label_names']
        self.input_shape = tuple(self.model_info['input_shape'])
        self.num_classes = self.model_info['num_classes']
        self.sequence_length = self.input_shape[0]
        self.feature_dim = self.input_shape[1]

        logger.info(f"Clases: {self.num_classes}")
        logger.info(f"Secuencia esperada: {self.sequence_length} frames de {self.feature_dim} features")

        # Configurar webcam
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara {camera_id}")

        # Configurar resolución
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Inicializar MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Buffer para almacenar secuencias
        self.sequence_buffer = deque(maxlen=sequence_buffer_size)
        self.recording = False
        self.prediction_threshold = prediction_threshold

        # Para calcular FPS
        self.fps_time = time.time()
        self.fps = 0

    def extract_landmarks_from_frame(self, frame: np.ndarray) -> Tuple[List[float], bool]:
        """
        Extrae landmarks de un frame.

        Args:
            frame: Frame de video (BGR)

        Returns:
            Tupla (landmarks, hands_detected)
        """
        # Convertir BGR a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar con MediaPipe
        results = self.hands.process(frame_rgb)

        # Extraer landmarks
        frame_landmarks = []
        hands_detected = False

        if results.multi_hand_landmarks:
            hands_detected = True
            for hand_landmarks in results.multi_hand_landmarks[:2]:  # Máximo 2 manos
                for landmark in hand_landmarks.landmark:
                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z])

        # Si no hay manos, usar ceros
        if not frame_landmarks:
            frame_landmarks = [0.0] * 63

        # Asegurar exactamente 126 features (2 manos)
        while len(frame_landmarks) < 126:
            frame_landmarks.extend([0.0] * 63)
        frame_landmarks = frame_landmarks[:126]

        return frame_landmarks, hands_detected

    def draw_landmarks(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Dibuja landmarks en el frame.

        Args:
            frame: Frame de video
            results: Resultados de MediaPipe

        Returns:
            Frame con landmarks dibujados
        """
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        return frame

    def preprocess_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Preprocesa secuencia para inferencia.

        Args:
            sequence: Secuencia de landmarks (frames, features)

        Returns:
            Secuencia normalizada
        """
        # Normalizar longitud
        current_length = len(sequence)
        if current_length < self.sequence_length:
            # Repetir frames
            indices = np.linspace(0, current_length - 1, self.sequence_length)
            indices = np.round(indices).astype(int)
            sequence = sequence[indices]
        elif current_length > self.sequence_length:
            # Submuestrear
            indices = np.linspace(0, current_length - 1, self.sequence_length)
            indices = np.round(indices).astype(int)
            sequence = sequence[indices]

        # Normalización (mismo proceso que entrenamiento)
        sequence_mean = np.mean(sequence, axis=0, keepdims=True)
        sequence_std = np.std(sequence, axis=0, keepdims=True) + 1e-8
        normalized = (sequence - sequence_mean) / sequence_std

        return normalized

    def predict(self, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Realiza predicción en la secuencia actual.

        Args:
            top_k: Número de predicciones top

        Returns:
            Lista de (label, probabilidad)
        """
        if len(self.sequence_buffer) < 10:  # Mínimo de frames
            return []

        # Convertir buffer a array
        sequence = np.array(list(self.sequence_buffer), dtype=np.float32)

        # Preprocesar
        sequence = self.preprocess_sequence(sequence)

        # Añadir dimensión de batch
        sequence_batch = np.expand_dims(sequence, axis=0)

        # Predicción
        predictions = self.model.predict(sequence_batch, verbose=0)[0]

        # Top-k predicciones
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_predictions = [
            (self.label_names[idx], float(predictions[idx]))
            for idx in top_indices
            if predictions[idx] >= self.prediction_threshold
        ]

        return top_predictions

    def draw_ui(self, frame: np.ndarray, predictions: List[Tuple[str, float]]) -> np.ndarray:
        """
        Dibuja la interfaz de usuario en el frame.

        Args:
            frame: Frame de video
            predictions: Lista de predicciones

        Returns:
            Frame con UI dibujada
        """
        h, w, _ = frame.shape

        # Panel de información (fondo semi-transparente)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Estado de grabación
        status_text = "GRABANDO" if self.recording else "ESPERANDO"
        status_color = (0, 255, 0) if self.recording else (0, 165, 255)
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        # Buffer info
        buffer_text = f"Buffer: {len(self.sequence_buffer)}/{self.sequence_buffer.maxlen}"
        cv2.putText(frame, buffer_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # FPS
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Predicciones
        if predictions:
            y_offset = 130
            cv2.putText(frame, "PREDICCIONES:", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 30

            for i, (label, prob) in enumerate(predictions[:3]):
                text = f"{i+1}. {label}: {prob*100:.1f}%"
                color = (0, 255, 0) if i == 0 else (255, 255, 255)
                cv2.putText(frame, text, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 25

        # Instrucciones (abajo)
        instructions = [
            "ESPACIO: Iniciar/Detener grabacion",
            "R: Reset buffer",
            "Q: Salir"
        ]
        y_offset = h - 80
        for instruction in instructions:
            cv2.putText(frame, instruction, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25

        return frame

    def run(self):
        """Ejecuta el loop principal de reconocimiento en tiempo real."""
        logger.info("Iniciando reconocimiento en tiempo real...")
        logger.info("Presiona ESPACIO para iniciar/detener grabación")
        logger.info("Presiona 'r' para resetear el buffer")
        logger.info("Presiona 'q' para salir")

        predictions = []

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Error leyendo frame de la cámara")
                    break

                # Voltear horizontalmente para efecto espejo
                frame = cv2.flip(frame, 1)

                # Extraer landmarks
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)

                # Dibujar landmarks
                frame = self.draw_landmarks(frame, results)

                # Extraer landmarks numéricos
                landmarks, hands_detected = self.extract_landmarks_from_frame(frame)

                # Si estamos grabando, añadir al buffer
                if self.recording:
                    self.sequence_buffer.append(landmarks)

                    # Hacer predicción cada cierto número de frames
                    if len(self.sequence_buffer) >= 30 and len(self.sequence_buffer) % 10 == 0:
                        predictions = self.predict(top_k=3)

                # Dibujar UI
                frame = self.draw_ui(frame, predictions)

                # Calcular FPS
                current_time = time.time()
                self.fps = 1.0 / (current_time - self.fps_time)
                self.fps_time = current_time

                # Mostrar frame
                cv2.imshow('V-LIBRASIL - Reconocimiento en Tiempo Real', frame)

                # Manejar teclas
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    # Salir
                    break
                elif key == ord(' '):
                    # Toggle grabación
                    self.recording = not self.recording
                    if self.recording:
                        logger.info("Iniciando grabación...")
                        self.sequence_buffer.clear()
                        predictions = []
                    else:
                        logger.info("Deteniendo grabación")
                        # Hacer predicción final
                        if len(self.sequence_buffer) >= 10:
                            predictions = self.predict(top_k=5)
                            logger.info("Predicción final:")
                            for i, (label, prob) in enumerate(predictions, 1):
                                logger.info(f"  {i}. {label}: {prob*100:.1f}%")

                elif key == ord('r'):
                    # Reset
                    logger.info("Reseteando buffer...")
                    self.sequence_buffer.clear()
                    predictions = []
                    self.recording = False

        finally:
            self.cleanup()

    def cleanup(self):
        """Limpia recursos."""
        logger.info("Limpiando recursos...")
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()


def main():
    parser = argparse.ArgumentParser(
        description='Reconocimiento en tiempo real de V-LIBRASIL'
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
        '--camera',
        type=int,
        default=0,
        help='ID de la cámara (default: 0)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.1,
        help='Umbral de confianza para mostrar predicciones (default: 0.1)'
    )

    args = parser.parse_args()

    # Inicializar y ejecutar sistema
    realtime_system = RealtimeVLibrasil(
        model_path=args.model,
        model_info_path=args.model_info,
        camera_id=args.camera,
        prediction_threshold=args.threshold
    )

    realtime_system.run()


if __name__ == '__main__':
    main()

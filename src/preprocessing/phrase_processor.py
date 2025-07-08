"""Módulo para procesar videos de frases en LSPy."""
import os

import cv2
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Directorios de entrada y salida desde .env
input_dir = os.getenv("DATA_RAW_DIR", "data/lsp_phrase_videos")
output_dir = os.getenv(
    "DATA_PROCESSED_DIR", "data/processed_lsp_phrase_sequences"
)

# Configurar rutas de directorios
input_dir = (
    os.path.join(input_dir, "phrases")
    if os.path.isdir(os.path.join(input_dir, "phrases"))
    else input_dir
)
output_dir = (
    os.path.join(output_dir, "phrases")
    if os.path.isdir(os.path.join(output_dir, "phrases"))
    else output_dir
)
os.makedirs(output_dir, exist_ok=True)

# Configuración de MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
    static_image_mode=False, min_detection_confidence=0.5
)

# Lista de frases
phrases = [
    "acceso_a_la_justicia",
    "derecho_a_la_defensa",
    "igualdad_ante_la_ley",
]

# Parámetros de la secuencia
sequence_length = 15
frame_size = (200, 200)
frame_skip = 20


def process_frame(frame):
    """
    Procesa un frame para extraer landmarks y dibujar el esqueleto.
    """
    skeleton_image = np.zeros(
        (frame_size[0], frame_size[1], 3), dtype=np.uint8
    )
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    if (
        results.pose_landmarks
        or results.left_hand_landmarks
        or results.right_hand_landmarks
    ):
        mp_drawing.draw_landmarks(
            skeleton_image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
        )
        mp_drawing.draw_landmarks(
            skeleton_image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
        )
        mp_drawing.draw_landmarks(
            skeleton_image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
        )

    return skeleton_image


# Procesar cada video
X_data = []
y_data = []

for phrase_idx, phrase in enumerate(phrases):
    for sample_num in range(1, 11):
        video_path = os.path.join(
            input_dir, f"{phrase}_sample_{sample_num}.avi"
        )
        if not os.path.exists(video_path):
            print(f"Video no encontrado: {video_path}")
            continue

        print(f"Procesando video: {video_path}")
        cap = cv2.VideoCapture(video_path)

        frames = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                skeleton_image = process_frame(frame)
                frames.append(skeleton_image)

            frame_count += 1
            if len(frames) >= sequence_length:
                break

        cap.release()

        if len(frames) == sequence_length:
            X_data.append(frames)
            y_data.append(phrase_idx)
        else:
            print(f"Secuencia incompleta para {video_path}, descartada.")

# Convertir a arrays NumPy y guardar
X_data = np.array(X_data)
y_data = np.array(y_data)

output_x = os.path.join(output_dir, "X_lsp_phrase_sequences.npy")
output_y = os.path.join(output_dir, "y_lsp_phrase_sequences.npy")
np.save(output_x, X_data)
np.save(output_y, y_data)

print(f"Preprocesamiento completo. Datos guardados en {output_dir}")
print(f"Forma de X: {X_data.shape}, Forma de y: {y_data.shape}")

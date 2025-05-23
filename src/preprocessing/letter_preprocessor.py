import cv2
import mediapipe as mp
import numpy as np
import os

# Configuración de MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5)

# Directorios de entrada y salida
input_dir = 'data/lsp_letter_videos'
output_dir = 'data/processed_lsp_letter_sequences'
os.makedirs(output_dir, exist_ok=True)

# Lista de letras
letters = list('abcdefghijklmnopqrstuvwxyz') + ['ñ']

# Parámetros de la secuencia
sequence_length = 15  # 15 frames por secuencia
frame_size = (200, 200)  # Tamaño de la imagen del esqueleto
frame_skip = 20  # Seleccionar 1 frame cada 20 para obtener 15 frames de 300


# Función para procesar un frame y generar una imagen de esqueleto
def process_frame(frame):
    skeleton_image = np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    if results.pose_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
        mp_drawing.draw_landmarks(skeleton_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(skeleton_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(skeleton_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    return skeleton_image


# Procesar cada video
X_data = []
y_data = []

for letter_idx, letter in enumerate(letters):
    for sample_num in range(1, 11):
        video_path = os.path.join(input_dir, f'{letter}_sample_{sample_num}.avi')
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

        # Asegurarse de que la secuencia tenga exactamente 15 frames
        if len(frames) == sequence_length:
            X_data.append(frames)
            y_data.append(letter_idx)
        else:
            print(f"Secuencia incompleta para {video_path}, descartada.")

# Convertir a arrays NumPy y guardar
X_data = np.array(X_data)  # Forma: (muestras, 15, 200, 200, 3)
y_data = np.array(y_data)  # Forma: (muestras,)

np.save(os.path.join(output_dir, 'X_lsp_letter_sequences.npy'), X_data)
np.save(os.path.join(output_dir, 'y_lsp_letter_sequences.npy'), y_data)

print(f"Preprocesamiento completo. Datos guardados en {output_dir}")
print(f"Forma de X: {X_data.shape}, Forma de y: {y_data.shape}")
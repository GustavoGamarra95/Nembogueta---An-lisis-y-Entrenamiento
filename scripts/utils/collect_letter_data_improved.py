"""
Script mejorado para recolectar datos del alfabeto en tiempo real.
Permite capturar muchas muestras de cada letra fácilmente.
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp

# Configurar MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Alfabeto completo
ALPHABET = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')


def extract_hand_landmarks(frame, hands):
    """Extrae landmarks de la mano."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Extraer coordenadas
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])

        return np.array(landmarks, dtype=np.float32), hand_landmarks

    return None, None


def main():
    parser = argparse.ArgumentParser(
        description='Recolectar datos del alfabeto con la cámara'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/processed/alphabet-combined'),
        help='Directorio donde guardar los datos'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='ID de la cámara'
    )
    parser.add_argument(
        '--samples-per-letter',
        type=int,
        default=50,
        help='Muestras a capturar por letra'
    )
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=30,
        help='Longitud de secuencia (frames)'
    )

    args = parser.parse_args()

    # Crear directorio de salida
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RECOLECCIÓN DE DATOS DEL ALFABETO")
    print("=" * 70)
    print(f"\nDirectorio de salida: {args.output_dir}")
    print(f"Muestras por letra: {args.samples_per_letter}")
    print(f"Secuencia: {args.sequence_length} frames")
    print("\n" + "=" * 70)

    # Abrir cámara
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: No se pudo abrir la cámara")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Inicializar MediaPipe
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    # Estado
    current_letter_idx = 0
    current_letter = ALPHABET[current_letter_idx]
    samples_captured = 0
    collecting = False
    frame_buffer = []

    # Contar muestras existentes
    existing_counts = {}
    for letter in ALPHABET:
        existing = len(list(args.output_dir.glob(f"{letter}_*.npy")))
        existing_counts[letter] = existing

    print("\nMuestras existentes:")
    for i, letter in enumerate(ALPHABET):
        if i % 10 == 0:
            print()
        print(f"  {letter}:{existing_counts[letter]:3d}", end="")
    print("\n")

    print("\nCONTROLES:")
    print("  ESPACIO - Capturar muestra")
    print("  N - Siguiente letra")
    print("  P - Letra anterior")
    print("  R - Reiniciar contador de letra actual")
    print("  Q - Salir")
    print("=" * 70)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: No se pudo leer frame")
                break

            # Voltear horizontalmente
            frame = cv2.flip(frame, 1)
            frame_display = frame.copy()

            # Detectar mano
            landmarks_array, hand_landmarks = extract_hand_landmarks(frame, hands)

            if hand_landmarks is not None:
                # Dibujar landmarks
                mp_drawing.draw_landmarks(
                    frame_display,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

                # Si estamos recolectando
                if collecting and landmarks_array is not None:
                    frame_buffer.append(landmarks_array)

                    if len(frame_buffer) >= args.sequence_length:
                        # Guardar secuencia
                        sequence = np.array(frame_buffer[:args.sequence_length])

                        # Generar nombre de archivo
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = f"{current_letter}_{timestamp}.npy"
                        filepath = args.output_dir / filename

                        np.save(filepath, sequence)

                        samples_captured += 1
                        existing_counts[current_letter] += 1

                        print(f"✓ Capturada: {current_letter} ({samples_captured}/{args.samples_per_letter}) -> {filename}")

                        # Reiniciar buffer
                        frame_buffer = []
                        collecting = False

                        # Si alcanzamos el objetivo, siguiente letra
                        if samples_captured >= args.samples_per_letter:
                            print(f"\n✓✓✓ COMPLETADO: {current_letter} ({existing_counts[current_letter]} muestras totales)")

                            if current_letter_idx < len(ALPHABET) - 1:
                                current_letter_idx += 1
                                current_letter = ALPHABET[current_letter_idx]
                                samples_captured = 0
                                print(f"\n→ Siguiente letra: {current_letter}")
                            else:
                                print("\n¡TODAS LAS LETRAS COMPLETADAS!")

            # UI
            h, w = frame_display.shape[:2]

            # Panel superior
            cv2.rectangle(frame_display, (0, 0), (w, 120), (30, 30, 30), -1)

            # Letra actual (grande)
            cv2.putText(frame_display, f"LETRA: {current_letter}",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

            # Progreso
            progress_text = f"{samples_captured}/{args.samples_per_letter} muestras"
            cv2.putText(frame_display, progress_text,
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Total existente
            total_text = f"Total: {existing_counts[current_letter]}"
            cv2.putText(frame_display, total_text,
                        (400, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            # Estado de recolección
            if collecting:
                status_text = f"CAPTURANDO... {len(frame_buffer)}/{args.sequence_length}"
                color = (0, 255, 0)
            elif landmarks_array is not None:
                status_text = "PRESIONA ESPACIO"
                color = (0, 200, 255)
            else:
                status_text = "MANO NO DETECTADA"
                color = (0, 0, 255)

            cv2.putText(frame_display, status_text,
                        (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Alfabeto completo
            alphabet_y = h - 80
            for i, letter in enumerate(ALPHABET):
                x = 20 + (i % 13) * 50
                y = alphabet_y if i < 13 else alphabet_y + 40

                if letter == current_letter:
                    color = (0, 255, 255)
                    thickness = 3
                elif existing_counts[letter] >= args.samples_per_letter:
                    color = (0, 255, 0)
                    thickness = 2
                else:
                    color = (150, 150, 150)
                    thickness = 1

                cv2.putText(frame_display, letter, (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness)

            cv2.imshow("Recolección de Datos - Alfabeto", frame_display)

            # Manejar teclas
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                print("\nSaliendo...")
                break

            elif key == ord(' '):  # ESPACIO
                if landmarks_array is not None and not collecting:
                    collecting = True
                    frame_buffer = []
                    print(f"Capturando {current_letter}...")

            elif key == ord('n') or key == ord('N'):  # Siguiente letra
                if current_letter_idx < len(ALPHABET) - 1:
                    current_letter_idx += 1
                    current_letter = ALPHABET[current_letter_idx]
                    samples_captured = 0
                    collecting = False
                    frame_buffer = []
                    print(f"\n→ Letra: {current_letter}")

            elif key == ord('p') or key == ord('P'):  # Letra anterior
                if current_letter_idx > 0:
                    current_letter_idx -= 1
                    current_letter = ALPHABET[current_letter_idx]
                    samples_captured = 0
                    collecting = False
                    frame_buffer = []
                    print(f"\n← Letra: {current_letter}")

            elif key == ord('r') or key == ord('R'):  # Reiniciar
                samples_captured = 0
                collecting = False
                frame_buffer = []
                print(f"\nReiniciado contador para {current_letter}")

    except KeyboardInterrupt:
        print("\n\nInterrumpido por el usuario")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

        print("\n" + "=" * 70)
        print("RESUMEN FINAL")
        print("=" * 70)
        for letter in ALPHABET:
            count = existing_counts[letter]
            print(f"  {letter}: {count:4d} muestras")
        print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())

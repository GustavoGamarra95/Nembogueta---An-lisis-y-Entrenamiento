
import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# Agregar directorio src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.libras_unified_predictor import LibrasUnifiedPredictor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def draw_text_with_background(frame, text, position, font_scale=0.5,
                               text_color=(255, 255, 255), bg_color=(50, 50, 50),
                               thickness=1, padding=5):

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Obtener tamaño del texto
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = position

    # Dibujar rectángulo de fondo con transparencia
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (x - padding, y - text_height - padding),
                  (x + text_width + padding, y + baseline + padding),
                  bg_color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Dibujar borde
    cv2.rectangle(frame,
                  (x - padding, y - text_height - padding),
                  (x + text_width + padding, y + baseline + padding),
                  (100, 100, 100), 1)

    # Dibujar texto
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    return text_height + baseline + 2 * padding


def draw_confidence_bar(frame, x, y, width, height, confidence, label=""):
    """Dibuja una barra de confianza."""
    # Fondo de la barra
    cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 40), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (80, 80, 80), 1)

    # Determinar color según confianza
    if confidence > 0.8:
        color = (0, 255, 0)  # Verde
    elif confidence > 0.6:
        color = (0, 200, 255)  # Amarillo
    else:
        color = (0, 100, 255)  # Naranja

    # Barra de progreso
    fill_width = int(width * confidence)
    if fill_width > 0:
        cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)

    # Texto de porcentaje
    text = f"{confidence:.0%}"
    cv2.putText(frame, text, (x + width + 5, y + height - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Label si existe
    if label:
        cv2.putText(frame, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)


def draw_ui_improved(frame, predictions, show_landmarks, text_translation=None):

    h, w = frame.shape[:2]

    # Panel superior con título
    y = 15
    cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.putText(frame, "NEMBOGUETA - Sistema Unificado LIBRAS",
                (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2, cv2.LINE_AA)

    y = 60

    # Estado general de detección
    num_hands = predictions['landmarks_detected']['hands']
    face_detected = predictions['landmarks_detected']['face']

    status_text = f"Manos detectadas: {num_hands}"
    hand_color = (0, 255, 0) if num_hands > 0 else (0, 0, 255)
    draw_text_with_background(frame, status_text, (15, y + 15),
                              font_scale=0.5, text_color=hand_color)

    face_text = f"Rostro: {'SI' if face_detected else 'NO'}"
    face_color = (0, 255, 0) if face_detected else (0, 0, 255)
    draw_text_with_background(frame, face_text, (250, y + 15),
                              font_scale=0.5, text_color=face_color)

    y += 50

    # Mostrar información de cada mano detectada
    for idx, hand_data in enumerate(predictions['hands']):
        # Panel para cada mano
        panel_y = y
        panel_height = 140

        # Fondo del panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, panel_y), (w - 10, panel_y + panel_height), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (10, panel_y), (w - 10, panel_y + panel_height), (100, 100, 100), 2)

        # Título del panel (mano izquierda/derecha)
        handedness_text = f"MANO {hand_data['handedness'].upper()}"
        handedness_color = (255, 150, 100) if hand_data['handedness'] == 'Right' else (100, 150, 255)
        cv2.putText(frame, handedness_text, (20, panel_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, handedness_color, 2, cv2.LINE_AA)

        # Orientación
        orient_text = f"Orient: {hand_data['orientation'].upper()}"
        cv2.putText(frame, orient_text, (20, panel_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        current_y = panel_y + 75

        # ALFABETO (letra)
        if hand_data['alphabet']:
            alph = hand_data['alphabet']
            letter = alph['letter']
            confidence = alph['confidence']

            # Letra grande
            letter_color = (0, 255, 0) if confidence > 0.7 else (0, 200, 255)
            cv2.putText(frame, f"Letra: {letter}", (20, current_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, letter_color, 2, cv2.LINE_AA)

            # Barra de confianza para letra
            draw_confidence_bar(frame, w - 250, current_y - 18, 150, 20, confidence, "")
        else:
            cv2.putText(frame, "Letra: N/A", (20, current_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

        current_y += 30

        # HANDSHAPE
        if hand_data['handshape']:
            hs = hand_data['handshape']
            shape_text = f"Handshape: {hs['class']}"
            confidence = hs['confidence']

            shape_color = (0, 255, 0) if confidence > 0.7 else (0, 200, 255)
            cv2.putText(frame, shape_text, (20, current_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, shape_color, 1, cv2.LINE_AA)

            # Barra de confianza
            draw_confidence_bar(frame, w - 250, current_y - 15, 150, 18, confidence, "")
        else:
            cv2.putText(frame, "Handshape: N/A", (20, current_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

        y += panel_height + 15

    # Expresión facial (si hay)
    if predictions['facial_expression']:
        panel_y = y
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, panel_y), (w - 10, panel_y + 60), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (10, panel_y), (w - 10, panel_y + 60), (100, 100, 100), 2)

        fe = predictions['facial_expression']
        cv2.putText(frame, f"Expresion facial: {fe['expression']}", (20, panel_y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2, cv2.LINE_AA)

        draw_confidence_bar(frame, w - 250, panel_y + 20, 150, 20, fe['confidence'], "")

    # Panel de traducción (si existe)
    if text_translation:
        trans_y = h - 90
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, trans_y), (w - 10, h - 50), (20, 20, 60), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (10, trans_y), (w - 10, h - 50), (100, 150, 255), 2)

        cv2.putText(frame, f"PT-BR: {text_translation['text']}", (20, trans_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        gloss_display = " ".join(text_translation['gloss'][:12])
        if len(text_translation['gloss']) > 12:
            gloss_display += "..."
        cv2.putText(frame, f"LIBRAS: {gloss_display}", (20, trans_y + 43),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 255), 1, cv2.LINE_AA)

    # Panel de controles (inferior)
    controls_y = h - 40
    cv2.rectangle(frame, (0, controls_y), (w, h), (20, 20, 20), -1)

    landmarks_status = "[ON]" if show_landmarks else "[OFF]"
    controls_text = f"Q:Salir  |  T:Traducir  |  L:Landmarks {landmarks_status}"
    cv2.putText(frame, controls_text, (15, controls_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Demo mejorado en tiempo real del sistema LIBRAS"
    )

    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("data/models"),
        help="Directorio con los modelos entrenados"
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="ID de la cámara"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Ancho del frame"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Alto del frame"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("NEMBOGUETA - Sistema Unificado de Reconocimiento LIBRAS")
    print("=" * 70)
    print()

    # Cargar modelos
    print("Cargando modelos...")
    try:
        predictor = LibrasUnifiedPredictor(args.models_dir)
    except Exception as e:
        print(f"ERROR: No se pudieron cargar los modelos: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Mostrar información de modelos
    print("\nModelos cargados:")
    model_info = predictor.get_model_info()
    for model_name, info in model_info.items():
        if model_name == 'handshape' and isinstance(info.get('metadata'), dict):
            orientations = list(info['metadata'].keys())
            print(f"  [OK] Handshape ({len(orientations)} orientaciones): {', '.join(orientations)}")
        elif info['loaded']:
            print(f"  [OK] {model_name}")
        else:
            print(f"  [X] {model_name} (no disponible)")

    # Inicializar cámara
    print(f"\nAbriendo cámara {args.camera}...")
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print("ERROR: No se pudo abrir la cámara")
        return 1

    # Configurar cámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Resolución: {int(actual_width)}x{int(actual_height)}")

    print("\n" + "=" * 70)
    print("Controles:")
    print("  Q - Salir")
    print("  T - Traducir texto PT-BR a glosas LIBRAS")
    print("  L - Activar/desactivar visualización de landmarks")
    print("=" * 70)
    print("\nPresiona Q para salir...\n")

    show_landmarks = True
    text_translation = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: No se pudo leer el frame de la cámara")
                break

            # Hacer predicciones
            predictions = predictor.predict_from_frame(frame, draw_landmarks=show_landmarks)

            # Dibujar UI mejorada
            frame = draw_ui_improved(frame, predictions, show_landmarks, text_translation)

            # Mostrar
            cv2.imshow("NEMBOGUETA - LIBRAS", frame)

            # Manejar teclas
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                print("\nCerrando...")
                break

            elif key == ord('l') or key == ord('L'):
                show_landmarks = not show_landmarks
                status = "activados" if show_landmarks else "desactivados"
                print(f"Landmarks {status}")

            elif key == ord('t') or key == ord('T'):
                print("\n" + "=" * 70)
                print("TRADUCCIÓN PT-BR → LIBRAS")
                print("=" * 70)
                text = input("Ingrese texto en portugués brasileño: ").strip()

                if text:
                    print("Traduciendo...")
                    gloss = predictor.translate_text_to_gloss(text)
                    text_translation = {
                        'text': text,
                        'gloss': gloss
                    }
                    print(f"\nPT-BR:  {text}")
                    print(f"LIBRAS: {' '.join(gloss)}")
                    print("=" * 70)
                else:
                    text_translation = None
                    print("Traducción cancelada")

    except KeyboardInterrupt:
        print("\n\nInterrumpido por el usuario")

    finally:
        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()
        print("\n¡Hasta pronto!")

    return 0


if __name__ == "__main__":
    sys.exit(main())

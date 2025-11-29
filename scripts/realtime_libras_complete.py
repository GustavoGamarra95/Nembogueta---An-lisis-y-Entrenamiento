"""
Script de reconocimiento LIBRAS en tiempo real.
Utiliza todos los modelos entrenados de forma centralizada.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Agregar directorio src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.libras_unified_predictor import LibrasUnifiedPredictor


def draw_predictions(frame, predictions, text_input=None):
    """
    Dibuja las predicciones en el frame.

    Args:
        frame: Frame de video
        predictions: Diccionario con predicciones
        text_input: Texto traducido (opcional)
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Panel de información (fondo semi-transparente)
    panel_height = 200 if predictions.get('hand_orientation') else 180
    cv2.rectangle(overlay, (10, 10), (w - 10, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y_offset = 40

    # Título
    cv2.putText(frame, "LIBRAS - Reconocimiento Unificado",
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 30

    # Estado de detección
    hands_status = "SI" if predictions['landmarks_detected']['hands'] else "NO"
    face_status = "SI" if predictions['landmarks_detected']['face'] else "NO"
    hands_color = (0, 255, 0) if predictions['landmarks_detected']['hands'] else (0, 0, 255)
    face_color = (0, 255, 0) if predictions['landmarks_detected']['face'] else (0, 0, 255)

    cv2.putText(frame, f"Manos: ", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, hands_status, (90, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hands_color, 2)
    cv2.putText(frame, " | Rostro: ", (140, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, face_status, (230, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 2)
    y_offset += 30

    # Orientación de mano
    if predictions.get('hand_orientation'):
        orient_text = f"Orientacion: {predictions['hand_orientation'].upper()}"
        cv2.putText(frame, orient_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 200, 100), 1)
        y_offset += 25

    # Forma de mano
    if predictions['handshape']:
        hs = predictions['handshape']
        color = (0, 255, 0) if hs['confidence'] > 0.7 else (0, 165, 255)
        text = f"Handshape: {hs['class']} ({hs['confidence']:.1%})"
        if hs.get('model_used') and hs.get('model_used') != hs.get('orientation'):
            text += f" [usando modelo {hs['model_used']}]"
        cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)
    else:
        cv2.putText(frame, "Handshape: No detectada",
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    y_offset += 30

    # Expresión facial
    if predictions['facial_expression']:
        fe = predictions['facial_expression']
        color = (0, 255, 0) if fe['confidence'] > 0.7 else (0, 165, 255)
        text = f"Expresion: {fe['expression']} ({fe['confidence']:.1%})"
        cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)
    else:
        cv2.putText(frame, "Expresion: No detectada",
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

    # Panel de traducción (si hay texto)
    if text_input:
        y_trans = h - 100
        cv2.rectangle(overlay, (10, y_trans), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, f"Texto PT-BR: {text_input['text']}",
                    (20, y_trans + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        gloss_text = " -> " + " ".join(text_input['gloss'][:10])  # Mostrar primeras 10 glosas
        cv2.putText(frame, f"Glosas LIBRAS: {gloss_text}",
                    (20, y_trans + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Instrucciones
    cv2.putText(frame, "q:salir | t:traducir | l:landmarks on/off",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return frame


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Reconocimiento LIBRAS en tiempo real con todos los modelos"
    )

    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("/app/data/models"),
        help="Directorio con todos los modelos entrenados"
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

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS objetivo"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("LIBRAS - Sistema Unificado de Reconocimiento")
    print("=" * 60)

    # Inicializar predictor
    print("\nCargando modelos...")
    try:
        predictor = LibrasUnifiedPredictor(args.models_dir)
    except Exception as e:
        print(f"Error al cargar modelos: {e}")
        return 1

    # Mostrar información de modelos
    print("\nModelos disponibles:")
    model_info = predictor.get_model_info()
    for model_name, info in model_info.items():
        status = "✓ Cargado" if info['loaded'] else "✗ No disponible"
        print(f"  {model_name}: {status}")

    # Inicializar cámara
    print(f"\nAbriendo cámara {args.camera}...")
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return 1

    # Configurar cámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    print("\nCámara iniciada.")
    print("Controles:")
    print("  q: Salir")
    print("  t: Traducir texto PT-BR a glosas LIBRAS")
    print("  l: Activar/desactivar visualización de landmarks")
    print("=" * 60)

    text_translation = None
    show_landmarks = True  # Mostrar landmarks por defecto

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame")
            break

        # Realizar predicciones
        predictions = predictor.predict_from_frame(frame, draw_landmarks=show_landmarks)

        # Dibujar resultados
        frame = draw_predictions(frame, predictions, text_translation)

        # Mostrar frame
        cv2.imshow("LIBRAS Recognition", frame)

        # Manejar teclas
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\nCerrando aplicación...")
            break

        elif key == ord('l'):
            show_landmarks = not show_landmarks
            status = "activados" if show_landmarks else "desactivados"
            print(f"Landmarks {status}")

        elif key == ord('t'):
            # Traducir texto
            print("\n" + "=" * 60)
            print("TRADUCCIÓN TEXTO → GLOSAS")
            print("=" * 60)
            text = input("Ingrese texto en portugués brasileño: ").strip()

            if text:
                gloss = predictor.translate_text_to_gloss(text)
                text_translation = {
                    'text': text,
                    'gloss': gloss
                }
                print(f"\nTexto PT-BR: {text}")
                print(f"Glosas LIBRAS: {' '.join(gloss)}")
                print("=" * 60)
            else:
                text_translation = None

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

    print("\n¡Hasta pronto!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

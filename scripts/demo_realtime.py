"""
Script de demostración en tiempo real para LIBRAS.
Versión simplificada para ejecutar localmente (sin Docker).
"""
import argparse
import logging
import sys
from pathlib import Path

import cv2

# Agregar directorio src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.libras_unified_predictor import LibrasUnifiedPredictor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def draw_ui(frame, predictions, show_landmarks, text_translation=None):
    """
    Dibuja la interfaz de usuario con las predicciones.

    Args:
        frame: Frame de video
        predictions: Diccionario con predicciones
        show_landmarks: Si los landmarks están activados
        text_translation: Texto traducido (opcional)
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Panel superior con información
    panel_height = 210
    cv2.rectangle(overlay, (10, 10), (w - 10, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    y = 35

    # Título
    cv2.putText(frame, "NEMBOGUETA - Sistema Unificado LIBRAS",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
    y += 35

    # Línea separadora
    cv2.line(frame, (20, y - 10), (w - 20, y - 10), (100, 100, 100), 1)

    # Estado de detección
    hands_detected = predictions['landmarks_detected']['hands']
    face_detected = predictions['landmarks_detected']['face']

    hand_text = "MANO: "
    hand_status = "DETECTADA" if hands_detected else "NO DETECTADA"
    hand_color = (0, 255, 0) if hands_detected else (0, 0, 255)

    cv2.putText(frame, hand_text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, hand_status, (100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 2)

    face_text = "  |  ROSTRO: "
    face_status = "DETECTADO" if face_detected else "NO DETECTADO"
    face_color = (0, 255, 0) if face_detected else (0, 0, 255)

    x_offset = 280
    cv2.putText(frame, face_text, (x_offset, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, face_status, (x_offset + 110, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 2)
    y += 30

    # Orientación de la mano
    if predictions.get('hand_orientation'):
        orient = predictions['hand_orientation'].upper()
        cv2.putText(frame, f"Orientacion: {orient}",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
        y += 30

    # Handshape
    if predictions['handshape']:
        hs = predictions['handshape']
        confidence = hs['confidence']
        color = (0, 255, 0) if confidence > 0.7 else (0, 200, 255) if confidence > 0.5 else (0, 150, 255)

        text = f"Handshape: {hs['class']}"
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Barra de confianza
        bar_width = 200
        bar_height = 15
        bar_x = w - bar_width - 30
        bar_y = y - 15

        # Fondo de la barra
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        # Barra de progreso
        fill_width = int(bar_width * confidence)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
        # Texto de confianza
        cv2.putText(frame, f"{confidence:.1%}", (bar_x + bar_width + 10, bar_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    else:
        cv2.putText(frame, "Handshape: No detectada",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    y += 35

    # Expresión facial
    if predictions['facial_expression']:
        fe = predictions['facial_expression']
        confidence = fe['confidence']
        color = (0, 255, 0) if confidence > 0.7 else (0, 200, 255) if confidence > 0.5 else (0, 150, 255)

        text = f"Expresion facial: {fe['expression']}"
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Barra de confianza
        bar_width = 200
        bar_height = 15
        bar_x = w - bar_width - 30
        bar_y = y - 15

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        fill_width = int(bar_width * confidence)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
        cv2.putText(frame, f"{confidence:.1%}", (bar_x + bar_width + 10, bar_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    else:
        cv2.putText(frame, "Expresion facial: No detectada",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

    # Panel de traducción (si existe)
    if text_translation:
        y_trans = h - 110
        cv2.rectangle(overlay, (10, y_trans), (w - 10, h - 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, f"PT-BR: {text_translation['text']}",
                    (20, y_trans + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        gloss_text = " ".join(text_translation['gloss'][:15])
        if len(text_translation['gloss']) > 15:
            gloss_text += "..."
        cv2.putText(frame, f"LIBRAS: {gloss_text}",
                    (20, y_trans + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 1)

    # Panel de controles (parte inferior)
    y_controls = h - 50
    cv2.rectangle(overlay, (10, y_controls), (w - 10, h - 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    landmarks_status = "[ON]" if show_landmarks else "[OFF]"
    cv2.putText(frame, f"Q:Salir  |  T:Traducir  |  L:Landmarks {landmarks_status}",
                (20, y_controls + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Demo en tiempo real del sistema unificado LIBRAS"
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
        help="ID de la cámara (0 para la cámara principal)"
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
        return 1

    # Mostrar información de modelos
    print("\nModelos cargados:")
    model_info = predictor.get_model_info()
    for model_name, info in model_info.items():
        if model_name == 'handshape' and isinstance(info.get('metadata'), dict):
            # Mostrar detalles de handshape
            orientations = list(info['metadata'].keys())
            print(f"  - Handshape ({len(orientations)} orientaciones): {', '.join(orientations)}")
        else:
            status = "✓" if info['loaded'] else "✗"
            print(f"  {status} {model_name}")

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

            # Dibujar UI
            frame = draw_ui(frame, predictions, show_landmarks, text_translation)

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

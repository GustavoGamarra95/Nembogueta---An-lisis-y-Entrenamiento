"""
Demo en tiempo real para el modelo DNN de alfabeto LIBRAS (LibrAI dataset).

Predice la letra en cada frame — sin buffer de secuencia — usando
engineer_frame_features() → normalización global → DNN.

Uso (dentro del contenedor, con X11 habilitado):
    # En el host primero:
    xhost +local:docker

    python scripts/demo/realtime_librai_alphabet.py \
        --model-dir /data/models/librai-alphabet \
        --camera 0

Controles:
    Q        : Salir
    C        : Limpiar palabra formada
    SPACE    : Agregar letra actual a la palabra
    BACKSPACE: Borrar última letra
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from preprocessing.feature_engineering import engineer_frame_features


def load_model_artifacts(model_dir: Path):
    model = tf.keras.models.load_model(str(model_dir / "best_model.keras"))
    norm_mean = np.load(model_dir / "norm_mean.npy")
    norm_std = np.load(model_dir / "norm_std.npy")
    classes = np.load(model_dir / "label_classes.npy")
    print(f"Modelo cargado — {len(classes)} clases: {list(classes)}")
    return model, norm_mean, norm_std, classes


def extract_raw_126(results) -> np.ndarray:
    """Convierte resultados de MediaPipe a raw_126 (2 manos × 63 coords)."""
    raw = np.zeros(126, dtype=np.float32)
    if not results.multi_hand_landmarks:
        return raw
    for hand_lm, handedness in zip(
        results.multi_hand_landmarks, results.multi_handedness
    ):
        label = handedness.classification[0].label
        offset = 0 if label == "Right" else 63
        for i, lm in enumerate(hand_lm.landmark):
            raw[offset + i * 3 + 0] = lm.x
            raw[offset + i * 3 + 1] = lm.y
            raw[offset + i * 3 + 2] = lm.z
    return raw


def run_demo(model_dir: Path, camera_id: int, threshold: float):
    model, norm_mean, norm_std, classes = load_model_artifacts(model_dir)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Intentar abrir la cámara solicitada; si falla, probar la otra
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        alt = 1 if camera_id == 0 else 0
        print(f"Cámara {camera_id} no disponible, probando cámara {alt}...")
        cap = cv2.VideoCapture(alt)
    if not cap.isOpened():
        print("Error: no se pudo abrir ninguna cámara")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    WIN = "LIBRAS Alfabeto — Nembogueta"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    # Suavizado: últimas N predicciones para estabilizar
    pred_history: deque = deque(maxlen=10)
    word_buffer: list[str] = []
    prev_raw = None
    fps_time = time.time()
    fps = 0.0

    print("Demo iniciado. Controles: SPACE=agregar letra | C=limpiar | BACKSPACE=borrar | Q=salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Dibujar landmarks
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

        # Predicción continua si hay mano
        top_letter, top_prob = "", 0.0
        top3 = []
        if results.multi_hand_landmarks:
            raw_126 = extract_raw_126(results)
            features = engineer_frame_features(raw_126, prev_landmarks=prev_raw)
            prev_raw = raw_126

            feat_norm = (features - norm_mean) / norm_std
            pred = model.predict(feat_norm[np.newaxis], verbose=0)[0]

            top_idx = np.argsort(pred)[::-1][:3]
            top3 = [(classes[i], float(pred[i])) for i in top_idx]
            top_letter, top_prob = top3[0]

            pred_history.append(top_letter)
        else:
            prev_raw = None
            pred_history.clear()

        # Letra estabilizada (moda de los últimos N frames)
        stable_letter = ""
        if pred_history:
            from collections import Counter
            stable_letter = Counter(pred_history).most_common(1)[0][0]

        # FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - fps_time, 1e-6))
        fps_time = now

        # --- UI ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 210), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)

        if stable_letter and top_prob >= threshold:
            cv2.putText(frame, stable_letter, (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 8)
            cv2.putText(frame, f"{top_prob*100:.1f}%", (130, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 0), 2)
            for i, (ltr, prob) in enumerate(top3[1:], 2):
                cv2.putText(frame, f"{i}. {ltr} {prob*100:.0f}%",
                            (130, 100 + i * 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        elif results.multi_hand_landmarks:
            cv2.putText(frame, "...", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 100, 100), 4)
        else:
            cv2.putText(frame, "Sin mano", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

        # Palabra formada
        word_str = "".join(word_buffer)
        cv2.rectangle(frame, (0, h - 80), (w, h), (30, 30, 30), -1)
        cv2.putText(frame, f"Palabra: {word_str}_", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)

        # Instrucciones
        cv2.putText(frame, "SPACE:agregar  C:limpiar  BKSP:borrar  Q:salir",
                    (10, h - 88), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

        cv2.imshow(WIN, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            if stable_letter and top_prob >= threshold:
                word_buffer.append(stable_letter)
                print(f"Palabra: {''.join(word_buffer)}")
        elif key == ord("c"):
            word_buffer.clear()
        elif key in (8, 127):  # BACKSPACE / DEL
            if word_buffer:
                word_buffer.pop()

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    if word_buffer:
        print(f"Palabra final: {''.join(word_buffer)}")


def main():
    parser = argparse.ArgumentParser(
        description="Demo tiempo real — alfabeto LIBRAS (modelo DNN LibrAI)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/data/models/librai-alphabet",
        help="Directorio con best_model.keras, norm_mean.npy, norm_std.npy, label_classes.npy",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Confianza mínima para mostrar predicción (default: 0.5)"
    )
    args = parser.parse_args()

    run_demo(Path(args.model_dir), args.camera, args.threshold)


if __name__ == "__main__":
    main()

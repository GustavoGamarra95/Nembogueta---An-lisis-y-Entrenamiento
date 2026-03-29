"""
Demo visual del alfabeto LIBRAS — Nembogueta.

Layout de dos paneles:
  Izquierda : cámara + landmarks MediaPipe + predicción actual
  Derecha   : grilla del abecedario (21 letras resaltadas) + palabra formada

Uso (dentro del contenedor):
    xhost +local:docker   # en el host primero

    python scripts/demo/demo_alphabet_panel.py \
        --model-dir /data/models/librai-alphabet \
        --camera 0

Controles:
    SPACE     : agregar letra actual a la palabra
    BACKSPACE : borrar última letra
    C         : limpiar palabra
    Q         : salir
"""

import argparse
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from preprocessing.feature_engineering import engineer_frame_features

# ---------------------------------------------------------------------------
# Colores BGR
# ---------------------------------------------------------------------------
BLACK       = (0, 0, 0)
WHITE       = (255, 255, 255)
GRAY_DARK   = (40, 40, 40)
GRAY_MID    = (80, 80, 80)
GRAY_LIGHT  = (180, 180, 180)
GREEN       = (50, 220, 50)
GREEN_DARK  = (30, 140, 30)
YELLOW      = (0, 220, 220)
ORANGE      = (0, 165, 255)
BLUE        = (220, 100, 30)
RED         = (50, 50, 220)
CYAN        = (220, 220, 0)

# Letras soportadas por el modelo (orden de visualización)
GRID_LETTERS = [
    ["A", "B", "C", "D", "E", "F", "G"],
    ["I", "L", "M", "N", "O", "P", "Q"],
    ["R", "S", "T", "U", "V", "W", "Y"],
]

# Palabras de ejemplo que el usuario puede intentar deletrear
SAMPLE_WORDS = ["LIBRAS", "MANO", "BRASIL", "SIGNO", "LINGUA", "VISUAL"]


# ---------------------------------------------------------------------------
# Carga de artefactos
# ---------------------------------------------------------------------------

def load_artifacts(model_dir: Path):
    model    = tf.keras.models.load_model(str(model_dir / "best_model.keras"))
    mean     = np.load(model_dir / "norm_mean.npy")
    std      = np.load(model_dir / "norm_std.npy")
    classes  = list(np.load(model_dir / "label_classes.npy"))
    print(f"Modelo OK — {len(classes)} clases: {classes}")
    return model, mean, std, classes


# ---------------------------------------------------------------------------
# Panel derecho: grilla del abecedario
# ---------------------------------------------------------------------------

def draw_right_panel(canvas, classes, current_letter, top3, word, target_word,
                     panel_x, panel_w, panel_h):
    # Fondo
    cv2.rectangle(canvas, (panel_x, 0), (panel_x + panel_w, panel_h), GRAY_DARK, -1)

    # Título
    cv2.putText(canvas, "NEMBOGUETA", (panel_x + 10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, CYAN, 2)
    cv2.putText(canvas, "Alfabeto LIBRAS", (panel_x + 10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, GRAY_LIGHT, 1)
    cv2.line(canvas, (panel_x, 70), (panel_x + panel_w, 70), GRAY_MID, 1)

    # Grilla de letras
    cell_w = panel_w // 7
    cell_h = 70
    grid_top = 80

    for row_i, row in enumerate(GRID_LETTERS):
        for col_i, letter in enumerate(row):
            x = panel_x + col_i * cell_w
            y = grid_top + row_i * cell_h

            is_current  = (letter == current_letter)
            is_supported = (letter in classes)

            # Color de fondo de celda
            if is_current:
                bg = GREEN_DARK
                border = GREEN
                text_color = WHITE
                thickness = 3
            elif not is_supported:
                bg = (25, 25, 25)
                border = GRAY_MID
                text_color = GRAY_MID
                thickness = 1
            else:
                bg = (55, 55, 55)
                border = GRAY_MID
                text_color = GRAY_LIGHT
                thickness = 1

            cv2.rectangle(canvas, (x + 2, y + 2), (x + cell_w - 2, y + cell_h - 2), bg, -1)
            cv2.rectangle(canvas, (x + 2, y + 2), (x + cell_w - 2, y + cell_h - 2), border, thickness)

            # Letra
            font_scale = 1.4 if is_current else 1.0
            fw = 3 if is_current else 1
            (tw, th), _ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, font_scale, fw)
            tx = x + (cell_w - tw) // 2
            ty = y + (cell_h + th) // 2
            cv2.putText(canvas, letter, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, fw)

    sep_y = grid_top + len(GRID_LETTERS) * cell_h + 10
    cv2.line(canvas, (panel_x, sep_y), (panel_x + panel_w, sep_y), GRAY_MID, 1)

    # Top-3 predicciones
    y_offset = sep_y + 25
    cv2.putText(canvas, "Prediccion:", (panel_x + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, GRAY_LIGHT, 1)
    y_offset += 28

    for i, (ltr, prob) in enumerate(top3[:3]):
        bar_w = int((panel_w - 80) * prob)
        bar_color = GREEN if i == 0 else GRAY_MID
        cv2.rectangle(canvas,
                      (panel_x + 50, y_offset - 16),
                      (panel_x + 50 + bar_w, y_offset), bar_color, -1)
        label = f"{ltr}  {prob*100:.0f}%"
        cv2.putText(canvas, label, (panel_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 if i == 0 else 0.5,
                    WHITE if i == 0 else GRAY_LIGHT, 1)
        y_offset += 28

    sep_y2 = y_offset + 10
    cv2.line(canvas, (panel_x, sep_y2), (panel_x + panel_w, sep_y2), GRAY_MID, 1)

    # Palabra formada
    y_offset = sep_y2 + 28
    cv2.putText(canvas, "Palabra:", (panel_x + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, GRAY_LIGHT, 1)
    y_offset += 40
    word_display = word + "_"
    cv2.putText(canvas, word_display, (panel_x + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, YELLOW, 2)

    # Palabra objetivo (si existe)
    if target_word:
        y_offset += 50
        cv2.putText(canvas, f"Objetivo: {target_word}", (panel_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ORANGE, 1)

        # Comparar letra a letra
        y_offset += 30
        for i, char in enumerate(word):
            color = GREEN if (i < len(target_word) and char == target_word[i]) else RED
            cv2.putText(canvas, char, (panel_x + 10 + i * 22, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Instrucciones al pie
    instructions = [
        "SPACE  : agregar letra",
        "BKSP   : borrar ultima",
        "C      : limpiar",
        "Q      : salir",
    ]
    y_bottom = panel_h - len(instructions) * 22 - 10
    cv2.line(canvas, (panel_x, y_bottom - 8), (panel_x + panel_w, y_bottom - 8), GRAY_MID, 1)
    for instr in instructions:
        cv2.putText(canvas, instr, (panel_x + 10, y_bottom),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY_LIGHT, 1)
        y_bottom += 22


# ---------------------------------------------------------------------------
# Panel izquierdo: cámara + predicción grande
# ---------------------------------------------------------------------------

def draw_left_panel(canvas, frame, cam_w, cam_h, stable_letter, top_prob, fps):
    # Frame de cámara
    frame_resized = cv2.resize(frame, (cam_w, cam_h))
    canvas[:cam_h, :cam_w] = frame_resized

    # Barra inferior con predicción
    bar_h = 90
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, cam_h), (cam_w, cam_h + bar_h), BLACK, -1)
    canvas[cam_h:cam_h + bar_h, :cam_w] = overlay[cam_h:cam_h + bar_h, :cam_w]

    if stable_letter and top_prob >= 0.5:
        # Letra grande
        cv2.putText(canvas, stable_letter, (20, cam_h + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, GREEN, 5)
        # Barra de confianza
        bar_x, bar_y = 120, cam_h + 20
        bar_total = cam_w - 140
        bar_fill = int(bar_total * top_prob)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_total, bar_y + 25), GRAY_MID, -1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_fill, bar_y + 25), GREEN, -1)
        cv2.putText(canvas, f"{top_prob*100:.0f}%", (bar_x + bar_total + 5, bar_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
        cv2.putText(canvas, "confianza", (bar_x, bar_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY_LIGHT, 1)
    else:
        cv2.putText(canvas, "Muestra tu mano...", (20, cam_h + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, GRAY_LIGHT, 1)

    # FPS
    cv2.putText(canvas, f"FPS {fps:.0f}", (cam_w - 90, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, GRAY_LIGHT, 1)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(model_dir: Path, camera_id: int, target_word: str):
    model, norm_mean, norm_std, classes = load_artifacts(model_dir)

    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Abrir cámara (probar 0 y 1)
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        alt = 1 if camera_id == 0 else 0
        print(f"Cámara {camera_id} no disponible, probando {alt}...")
        cap = cv2.VideoCapture(alt, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Error: no se pudo abrir ninguna cámara")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Canvas total: cámara(640x480) + barra(90) altura | panel_derecho(320)
    CAM_W, CAM_H = 640, 480
    BAR_H = 90
    PANEL_W = 320
    TOTAL_W = CAM_W + PANEL_W
    TOTAL_H = CAM_H + BAR_H

    WIN = "LIBRAS Alfabeto — Nembogueta"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, TOTAL_W, TOTAL_H)

    pred_history: deque = deque(maxlen=12)
    word: list[str] = []
    prev_raw = None
    fps_time = time.time()
    fps = 0.0
    top3: list = []
    stable_letter = ""
    top_prob = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error leyendo frame")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Landmarks
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style(),
                )

            # Extracción de features
            raw_126 = np.zeros(126, dtype=np.float32)
            for hand_lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                offset = 0 if label == "Right" else 63
                for i, lm in enumerate(hand_lm.landmark):
                    raw_126[offset + i * 3 + 0] = lm.x
                    raw_126[offset + i * 3 + 1] = lm.y
                    raw_126[offset + i * 3 + 2] = lm.z

            features = engineer_frame_features(raw_126, prev_landmarks=prev_raw)
            prev_raw = raw_126
            feat_norm = (features - norm_mean) / norm_std
            pred = model.predict(feat_norm[np.newaxis], verbose=0)[0]

            top_idx = np.argsort(pred)[::-1][:3]
            top3 = [(classes[i], float(pred[i])) for i in top_idx]
            pred_history.append(top3[0][0])

            if pred_history:
                stable_letter = Counter(pred_history).most_common(1)[0][0]
                top_prob = top3[0][1]
        else:
            prev_raw = None
            pred_history.clear()
            stable_letter = ""
            top_prob = 0.0
            top3 = []

        # FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - fps_time, 1e-6)
        fps_time = now

        # Componer canvas
        canvas = np.zeros((TOTAL_H, TOTAL_W, 3), dtype=np.uint8)
        draw_left_panel(canvas, frame, CAM_W, CAM_H, stable_letter, top_prob, fps)
        draw_right_panel(canvas, classes, stable_letter, top3,
                         "".join(word), target_word,
                         CAM_W, PANEL_W, TOTAL_H)

        cv2.imshow(WIN, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            if stable_letter and top_prob >= 0.5:
                word.append(stable_letter)
                print(f"Palabra: {''.join(word)}")
        elif key in (8, 127):
            if word:
                word.pop()
        elif key == ord("c"):
            word.clear()

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    if word:
        print(f"Palabra final: {''.join(word)}")


def main():
    parser = argparse.ArgumentParser(
        description="Demo visual alfabeto LIBRAS — panel con abecedario"
    )
    parser.add_argument("--model-dir", default="/data/models/librai-alphabet")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--target-word", type=str, default="",
        help=f"Palabra objetivo para practicar (ej: LIBRAS). Disponibles: {SAMPLE_WORDS}"
    )
    args = parser.parse_args()
    run(Path(args.model_dir), args.camera, args.target_word.upper())


if __name__ == "__main__":
    main()

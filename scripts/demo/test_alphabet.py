"""
Prueba guiada del alfabeto LIBRAS — letra por letra.

Muestra cada letra en pantalla. Cuando el modelo la detecta con
confianza >= umbral durante N frames seguidos, pasa automáticamente
a la siguiente. Al final muestra un reporte completo.

Uso (dentro del contenedor):
    python scripts/demo/test_alphabet.py \
        --model-dir /data/models/librai-alphabet-v3/run_20260601_010139

Controles:
    SPACE  — marcar como correcta manualmente y avanzar
    X      — marcar como fallida y avanzar
    Q      — salir
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

LETTERS    = list("ABCDEFGILMNOPQRSTUVWY") + ["Ç"]
WIN_FRAMES = 8     # frames consecutivos con la letra correcta para auto-avanzar


def load_artifacts(model_dir: Path):
    model     = tf.keras.models.load_model(str(model_dir / "best_model.keras"))
    norm_mean = np.load(model_dir / "norm_mean.npy")
    norm_std  = np.load(model_dir / "norm_std.npy")
    classes   = np.load(model_dir / "label_classes.npy")
    print(f"Modelo cargado — {len(classes)} clases")
    return model, norm_mean, norm_std, classes


def extract_raw_126(results) -> np.ndarray:
    raw = np.zeros(126, dtype=np.float32)
    if not results.multi_hand_landmarks:
        return raw
    for hand_lm, handedness in zip(
        results.multi_hand_landmarks, results.multi_handedness
    ):
        label  = handedness.classification[0].label
        offset = 0 if label == "Right" else 63
        for i, lm in enumerate(hand_lm.landmark):
            raw[offset + i * 3 + 0] = lm.x
            raw[offset + i * 3 + 1] = lm.y
            raw[offset + i * 3 + 2] = lm.z
    return raw


def run(model_dir: Path, camera_id: int, threshold: float):
    model, norm_mean, norm_std, classes = load_artifacts(model_dir)

    mp_hands  = mp.solutions.hands
    mp_draw   = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1 if camera_id == 0 else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    WIN = "Prueba Alfabeto — Nembogueta"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    results_log = {}   # letra → 'ok' | 'fail' | 'skip'
    letter_idx  = 0
    prev_raw    = None
    hit_buf: deque = deque(maxlen=WIN_FRAMES)

    print(f"\n=== Prueba del alfabeto — {len(LETTERS)} letras ===")
    print(f"Umbral: {threshold*100:.0f}% | Frames para auto-avanzar: {WIN_FRAMES}\n")

    while letter_idx < len(LETTERS):
        target = LETTERS[letter_idx]

        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        h, w      = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result    = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_lm in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

        # Predicción
        pred_letter, conf_target, top3 = "", 0.0, []
        if result.multi_hand_landmarks:
            raw     = extract_raw_126(result)
            feat    = engineer_frame_features(raw, prev_landmarks=prev_raw)
            prev_raw = raw
            norm    = (feat - norm_mean) / norm_std
            pred    = model.predict(norm[np.newaxis], verbose=0)[0]
            top_idx = np.argsort(pred)[::-1][:3]
            top3    = [(classes[i], float(pred[i])) for i in top_idx]
            pred_letter = top3[0][0]
            t_idx   = list(classes).index(target) if target in classes else -1
            conf_target = float(pred[t_idx]) if t_idx >= 0 else 0.0
            hit_buf.append(pred_letter == target and conf_target >= threshold)
        else:
            prev_raw = None
            hit_buf.clear()

        # Auto-avanzar si WIN_FRAMES consecutivos correctos
        if len(hit_buf) == WIN_FRAMES and all(hit_buf):
            results_log[target] = "ok"
            print(f"  ✓ {target} — reconocida")
            letter_idx += 1
            hit_buf.clear()
            prev_raw = None
            continue

        hits = sum(hit_buf)

        # ── UI ──────────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 230), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Número de letra
        cv2.putText(frame, f"{letter_idx+1}/{len(LETTERS)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160,160,160), 1)

        # Letra objetivo
        cv2.putText(frame, target, (30, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 255, 255), 10)

        # Barra de progreso de detección
        bar_w = w - 300
        bar_fill = int(bar_w * hits / WIN_FRAMES)
        color_bar = (0, 255, 0) if conf_target >= threshold else (60, 60, 200)
        cv2.rectangle(frame, (250, 45), (250 + bar_w, 70), (50,50,50), -1)
        if bar_fill > 0:
            cv2.rectangle(frame, (250, 45), (250 + bar_fill, 70), color_bar, -1)
        cv2.rectangle(frame, (250, 45), (250 + bar_w, 70), (180,180,180), 1)
        cv2.putText(frame, f"{hits}/{WIN_FRAMES}", (250 + bar_w + 8, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)

        # Confianza de la letra target
        conf_color = (0, 255, 0) if conf_target >= threshold else (0, 80, 255)
        cv2.putText(frame, f"Confianza {target}: {conf_target*100:.1f}%",
                    (250, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, conf_color, 2)

        # Top 3 predicciones
        if top3:
            cv2.putText(frame, f"Detecta: {top3[0][0]} ({top3[0][1]*100:.0f}%)",
                        (250, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200,200,200), 1)
            for i, (l, p) in enumerate(top3[1:], 1):
                cv2.putText(frame, f"  {l}: {p*100:.0f}%",
                            (250, 150 + i*28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (140,140,140), 1)

        if not result.multi_hand_landmarks:
            cv2.putText(frame, "Mostra la mano",
                        (30, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,100,255), 3)

        # Historial lateral
        summary_x = w - 140
        cv2.rectangle(frame, (summary_x - 10, 0), (w, h), (20,20,20), -1)
        cv2.putText(frame, "Resultado", (summary_x, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        for i, l in enumerate(LETTERS):
            y_pos = 50 + i * 28
            status = results_log.get(l, "")
            if status == "ok":
                color, sym = (0, 220, 0), f"✓ {l}"
            elif status == "fail":
                color, sym = (0, 60, 220), f"✗ {l}"
            elif status == "skip":
                color, sym = (120,120,120), f"- {l}"
            elif l == target:
                color, sym = (0, 255, 255), f"► {l}"
            else:
                color, sym = (80,80,80), f"  {l}"
            cv2.putText(frame, sym, (summary_x, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        cv2.putText(frame, "SPACE:ok  X:fallo  Q:salir",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140,140,140), 1)

        cv2.imshow(WIN, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            results_log[target] = "skip"
            break
        elif key == ord(" "):
            results_log[target] = "ok"
            print(f"  ✓ {target} — marcada OK manualmente")
            letter_idx += 1
            hit_buf.clear()
            prev_raw = None
        elif key in (ord("x"), ord("X")):
            results_log[target] = "fail"
            print(f"  ✗ {target} — marcada como fallo")
            letter_idx += 1
            hit_buf.clear()
            prev_raw = None

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    # ── Reporte final ──
    ok   = [l for l, s in results_log.items() if s == "ok"]
    fail = [l for l, s in results_log.items() if s == "fail"]
    skip = [l for l, s in results_log.items() if s == "skip"]

    print(f"\n{'='*40}")
    print(f"REPORTE FINAL — {len(results_log)}/{len(LETTERS)} letras probadas")
    print(f"{'='*40}")
    print(f"  Reconocidas : {len(ok):>2}  {ok}")
    print(f"  Fallidas    : {len(fail):>2}  {fail}")
    print(f"  Saltadas    : {len(skip):>2}  {skip}")
    tested = len(ok) + len(fail)
    if tested:
        print(f"  Accuracy    : {len(ok)/tested*100:.1f}%")
    print(f"{'='*40}\n")


def main():
    parser = argparse.ArgumentParser(description="Prueba guiada del alfabeto LIBRAS")
    parser.add_argument("--model-dir", type=Path,
                        default=Path("/data/models/librai-alphabet-v3/run_20260601_010139"))
    parser.add_argument("--camera",    type=int,   default=0)
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="Confianza mínima para contar frame como correcto (default: 0.25)")
    args = parser.parse_args()
    run(args.model_dir, args.camera, args.threshold)


if __name__ == "__main__":
    main()

"""
Calibración personalizada del modelo LIBRAS.

Muestra cada letra una por una. Captura frames automáticamente solo cuando
la confianza del modelo supera el umbral (default 90%). Guarda los datos
en data/personal/{LETRA}/ para reentrenar después.

Uso (dentro del contenedor):
    python scripts/utils/guided_calibration.py \
        --model-dir /data/models/librai-alphabet-v2/run_20260531_231648

Controles:
    SPACE  — saltar letra actual sin capturar
    Q      — salir y guardar lo capturado hasta ese punto
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

LETTERS = list("ABCDEFGILMNOPQRSTUVWY") + ["Ç"]
SAMPLES_PER_LETTER = 60      # frames a capturar por letra
MIN_INTERVAL        = 0.05   # segundos mínimos entre capturas


def load_artifacts(model_dir: Path):
    model      = tf.keras.models.load_model(str(model_dir / "best_model.keras"))
    norm_mean  = np.load(model_dir / "norm_mean.npy")
    norm_std   = np.load(model_dir / "norm_std.npy")
    classes    = np.load(model_dir / "label_classes.npy")
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


def save_frame_features(features: np.ndarray, letter: str, out_root: Path, idx: int):
    out_dir = out_root / letter
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{letter}_{idx:05d}.npy", features)


def draw_progress_bar(frame, x, y, w, h, progress, color=(0, 255, 0)):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), -1)
    filled = int(w * progress)
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x + filled, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 1)


def run(model_dir: Path, camera_id: int, threshold: float, output_dir: Path, samples: int):
    model, norm_mean, norm_std, classes = load_artifacts(model_dir)

    mp_hands   = mp.solutions.hands
    mp_draw    = mp.solutions.drawing_utils
    mp_styles  = mp.solutions.drawing_styles
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        alt = 1 if camera_id == 0 else 0
        cap = cv2.VideoCapture(alt)
    if not cap.isOpened():
        print("ERROR: no se pudo abrir la cámara")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    WIN = "Calibración — LIBRAS Nembogueta"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    output_dir.mkdir(parents=True, exist_ok=True)

    letter_idx   = 0
    captured     = 0
    file_idx     = 0
    last_capture = 0.0
    prev_raw     = None
    pred_history: deque = deque(maxlen=8)

    print(f"\n=== Calibración personalizada — {len(LETTERS)} letras ===")
    print(f"Umbral de confianza : {threshold*100:.0f}%")
    print(f"Frames por letra    : {samples}")
    print(f"Salida              : {output_dir}\n")

    while letter_idx < len(LETTERS):
        target = LETTERS[letter_idx]

        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        h, w      = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

        # Predicción
        confidence   = 0.0
        pred_letter  = ""
        features_now = None
        if results.multi_hand_landmarks:
            raw_126      = extract_raw_126(results)
            features_now = engineer_frame_features(raw_126, prev_landmarks=prev_raw)
            prev_raw     = raw_126
            feat_norm    = (features_now - norm_mean) / norm_std
            pred         = model.predict(feat_norm[np.newaxis], verbose=0)[0]
            idx_top      = int(np.argmax(pred))
            pred_letter  = classes[idx_top]
            confidence   = float(pred[idx_top])
            # confianza específica de la letra target
            target_idx   = list(classes).index(target) if target in classes else -1
            conf_target  = float(pred[target_idx]) if target_idx >= 0 else 0.0
            pred_history.append(pred_letter)
        else:
            prev_raw = None
            pred_history.clear()
            conf_target = 0.0

        # Captura automática si confianza >= umbral
        now = time.time()
        auto_captured = False
        if (conf_target >= threshold
                and features_now is not None
                and now - last_capture >= MIN_INTERVAL):
            save_frame_features(features_now, target, output_dir, file_idx)
            file_idx    += 1
            captured    += 1
            last_capture = now
            auto_captured = True

        # Letra completada
        if captured >= samples:
            print(f"  ✓ {target}: {captured} frames capturados")
            letter_idx += 1
            captured    = 0
            file_idx    = 0
            prev_raw    = None
            pred_history.clear()
            continue

        # ── UI ──────────────────────────────────────────────
        # Fondo superior
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 220), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Letra objetivo (grande)
        cv2.putText(frame, target, (30, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 255, 255), 10)

        # Progreso de letras completadas
        progress_total = letter_idx / len(LETTERS)
        cv2.putText(frame, f"Letra {letter_idx+1}/{len(LETTERS)}",
                    (220, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
        draw_progress_bar(frame, 220, 45, w - 240, 18, progress_total, (0, 180, 255))

        # Progreso de frames de la letra actual
        progress_letter = captured / samples
        color_bar = (0, 255, 0) if conf_target >= threshold else (80, 80, 80)
        cv2.putText(frame, f"Frames: {captured}/{samples}",
                    (220, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
        draw_progress_bar(frame, 220, 95, w - 240, 22, progress_letter, color_bar)

        # Confianza de la letra target
        conf_color = (0, 255, 0) if conf_target >= threshold else (0, 100, 255)
        cv2.putText(frame, f"Confianza {target}: {conf_target*100:.1f}%",
                    (220, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, conf_color, 2)

        # Lo que el modelo está viendo
        if pred_letter:
            cv2.putText(frame, f"Detecta: {pred_letter} ({confidence*100:.0f}%)",
                        (220, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 180, 180), 1)

        # Flash verde cuando captura
        if auto_captured:
            flash = frame.copy()
            cv2.rectangle(flash, (0, 0), (w, h), (0, 255, 0), -1)
            frame = cv2.addWeighted(flash, 0.08, frame, 0.92, 0)

        # Sin mano
        if not results.multi_hand_landmarks:
            cv2.putText(frame, "Mostra la mano",
                        (30, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 100, 255), 3)

        # Instrucciones
        cv2.putText(frame, "SPACE: saltar letra   Q: salir",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (160, 160, 160), 1)

        cv2.imshow(WIN, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print(f"\nSalida manual en letra {target} ({captured} frames)")
            break
        elif key == ord(" "):
            print(f"  → {target} saltada ({captured} frames capturados)")
            letter_idx += 1
            captured    = 0
            file_idx    = 0
            prev_raw    = None
            pred_history.clear()

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    # Resumen
    print("\n=== Calibración finalizada ===")
    total = 0
    for l in LETTERS:
        d = output_dir / l
        if d.exists():
            n = len(list(d.glob("*.npy")))
            if n:
                print(f"  {l}: {n} frames")
                total += n
    print(f"Total: {total} frames en {output_dir}")
    if total:
        print(f"\nPara reentrenar ejecuta:")
        print(f"  python scripts/train/train_alphabet.py \\")
        print(f"      --data-dir /data/processed/librai_alphabet_v2 \\")
        print(f"      --personal-dir {output_dir} \\")
        print(f"      --output-dir /data/models/librai-alphabet-v2 \\")
        print(f"      --epochs 60 --batch-size 32")


def main():
    parser = argparse.ArgumentParser(
        description="Calibración personalizada — captura gestos por letra con confianza mínima"
    )
    parser.add_argument("--model-dir",   type=Path,
                        default=Path("/data/models/librai-alphabet-v2/run_20260531_231648"))
    parser.add_argument("--camera",      type=int,   default=0)
    parser.add_argument("--threshold",   type=float, default=0.90,
                        help="Confianza mínima para capturar (default: 0.90)")
    parser.add_argument("--samples",     type=int,   default=60,
                        help="Frames a capturar por letra (default: 60)")
    parser.add_argument("--output-dir",  type=Path,
                        default=Path("/data/personal"))
    args = parser.parse_args()
    run(args.model_dir, args.camera, args.threshold, args.output_dir, args.samples)


if __name__ == "__main__":
    main()

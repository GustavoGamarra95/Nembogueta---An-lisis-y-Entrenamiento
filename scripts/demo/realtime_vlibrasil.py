"""
Reconocimiento en tiempo real de V-LIBRASIL con el modelo v2.

Pipeline:
    Cámara → MediaPipe (126 coords) → feature_engineering (208 features/frame)
    → buffer de frames → mean+std pooling (416) → normalización → DNN → top-5

Controles:
    ESPACIO : Capturar seña (acumula frames mientras se mantiene presionado
              o toggle si se presiona y suelta rápido)
    R       : Reset / limpiar buffer
    Q       : Salir

Uso (dentro del contenedor con cámara pasada):
    python scripts/demo/realtime_vlibrasil.py
    python scripts/demo/realtime_vlibrasil.py --model-dir /data/models/vlibrasil-v2 --camera 0
"""

import argparse
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.preprocessing.feature_engineering import engineer_frame_features

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Colores UI
GREEN  = (0, 220, 0)
ORANGE = (0, 165, 255)
WHITE  = (255, 255, 255)
YELLOW = (0, 255, 255)
GRAY   = (180, 180, 180)
BLACK  = (0, 0, 0)
RED    = (0, 0, 220)


def pool_sequence(frames: np.ndarray) -> np.ndarray:
    """Mean + Std pooling sobre (T, 208) → (416,)."""
    return np.concatenate([frames.mean(axis=0), frames.std(axis=0)]).astype(np.float32)


class RealtimeVLibrasilV2:

    MIN_FRAMES = 30   # mínimo de frames para predecir
    PRED_EVERY = 10   # actualizar predicción cada N frames nuevos

    def __init__(self, model_dir: Path, camera_id: int = 0,
                 threshold: float = 0.15, buffer_size: int = 300):

        model_dir = Path(model_dir)

        # --- Modelo ---
        model_path = model_dir / "best_model.keras"
        if not model_path.exists():
            model_path = model_dir / "final_model.keras"
        logger.info(f"Cargando modelo: {model_path}")
        self.model = tf.keras.models.load_model(str(model_path), compile=False)

        # --- Metadata ---
        with open(model_dir / "metadata.json", encoding="utf-8") as f:
            meta = json.load(f)
        self.class_names = meta["class_names"]
        self.n_classes   = meta["n_classes"]
        logger.info(f"Clases: {self.n_classes} | "
                    f"Test acc: {meta.get('test_accuracy', '?'):.3f} | "
                    f"Test top-5: {meta.get('test_top5_accuracy', '?'):.3f}")

        # --- Normalización ---
        self.norm_mean = np.load(model_dir / "norm_mean.npy")[0]  # (416,)
        self.norm_std  = np.load(model_dir / "norm_std.npy")[0]   # (416,)

        # --- Cámara ---
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara {camera_id}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # --- MediaPipe ---
        self.mp_hands    = mp.solutions.hands
        self.mp_drawing  = mp.solutions.drawing_utils
        self.mp_styles   = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # --- Estado ---
        self.frame_buffer  = deque(maxlen=buffer_size)  # frames de 208 features
        self.prev_raw      = None                        # landmarks crudos frame anterior
        self.recording     = False
        self.predictions   = []                          # [(label, prob), ...]
        self.threshold     = threshold
        self.frames_since_pred = 0

        # FPS
        self._fps_t = time.time()
        self.fps = 0.0

    # ------------------------------------------------------------------
    # Extracción de features
    # ------------------------------------------------------------------

    def _extract_raw_landmarks(self, results) -> np.ndarray:
        """Devuelve array (126,): raw xyz de 2 manos (ceros si no detectada)."""
        raw = np.zeros(126, dtype=np.float32)
        if results.multi_hand_landmarks:
            for i, hand_lm in enumerate(results.multi_hand_landmarks[:2]):
                start = i * 63
                for j, lm in enumerate(hand_lm.landmark):
                    raw[start + j*3]     = lm.x
                    raw[start + j*3 + 1] = lm.y
                    raw[start + j*3 + 2] = lm.z
        return raw

    # ------------------------------------------------------------------
    # Predicción
    # ------------------------------------------------------------------

    def _predict(self) -> list:
        if len(self.frame_buffer) < self.MIN_FRAMES:
            return []
        frames = np.array(self.frame_buffer, dtype=np.float32)  # (T, 208)
        pooled = pool_sequence(frames)                           # (416,)
        x_norm = (pooled - self.norm_mean) / self.norm_std      # (416,)
        probs  = self.model.predict(x_norm[np.newaxis], verbose=0)[0]
        top5   = np.argsort(probs)[::-1][:5]
        return [(self.class_names[i], float(probs[i]))
                for i in top5 if probs[i] >= self.threshold]

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _draw_ui(self, frame: np.ndarray, hands_detected: bool) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()

        panel_h = 220 if self.predictions else 120
        cv2.rectangle(overlay, (0, 0), (w, panel_h), BLACK, -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

        # Estado
        if self.recording:
            status, color = "● GRABANDO", GREEN
        else:
            status, color = "○ LISTO  [ESPACIO para grabar]", ORANGE
        cv2.putText(frame, status, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # Info
        buf_pct = int(len(self.frame_buffer) / self.frame_buffer.maxlen * 100)
        info = (f"Buffer: {len(self.frame_buffer):>3} frames ({buf_pct}%)  |  "
                f"FPS: {self.fps:4.1f}  |  "
                f"Manos: {'SI' if hands_detected else 'NO'}")
        cv2.putText(frame, info, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GRAY, 1)

        # Predicciones
        if self.predictions:
            cv2.putText(frame, "PREDICCIONES:", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, YELLOW, 2)
            for rank, (label, prob) in enumerate(self.predictions[:5], 1):
                bar_w = int(prob * 300)
                y = 105 + rank * 22
                cv2.rectangle(frame, (200, y - 14), (200 + bar_w, y), GREEN if rank == 1 else GRAY, -1)
                color = GREEN if rank == 1 else WHITE
                text = f"{rank}. {label[:45]:<45} {prob*100:5.1f}%"
                cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)

        # Instrucciones abajo
        for i, txt in enumerate(["ESPACIO: grabar/parar  |  R: reset  |  Q: salir"]):
            cv2.putText(frame, txt, (20, h - 15 - i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)

        # Barra de buffer
        bar_full = int(len(self.frame_buffer) / self.frame_buffer.maxlen * (w - 40))
        cv2.rectangle(frame, (20, h - 8), (20 + bar_full, h - 3), GREEN, -1)

        return frame

    # ------------------------------------------------------------------
    # Loop principal
    # ------------------------------------------------------------------

    def run(self):
        logger.info("Iniciando demo en tiempo real V-LIBRASIL v2...")
        logger.info("  ESPACIO: grabar seña | R: reset | Q: salir")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Error leyendo frame")
                    break

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)

                # Dibujar landmarks
                if results.multi_hand_landmarks:
                    for hand_lm in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame, hand_lm, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_styles.get_default_hand_landmarks_style(),
                            self.mp_styles.get_default_hand_connections_style(),
                        )

                hands_detected = bool(results.multi_hand_landmarks)

                # Acumular features si estamos grabando
                if self.recording:
                    raw = self._extract_raw_landmarks(results)
                    feat = engineer_frame_features(raw, self.prev_raw)
                    self.frame_buffer.append(feat)
                    self.prev_raw = raw
                    self.frames_since_pred += 1

                    # Predecir cada PRED_EVERY frames nuevos
                    if (len(self.frame_buffer) >= self.MIN_FRAMES and
                            self.frames_since_pred >= self.PRED_EVERY):
                        self.predictions = self._predict()
                        self.frames_since_pred = 0

                # UI
                frame = self._draw_ui(frame, hands_detected)

                # FPS
                now = time.time()
                self.fps = 1.0 / max(now - self._fps_t, 1e-6)
                self._fps_t = now

                cv2.imshow("V-LIBRASIL v2 - Tiempo Real", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

                elif key == ord(' '):
                    self.recording = not self.recording
                    if self.recording:
                        logger.info("Grabando...")
                        self.frame_buffer.clear()
                        self.predictions = []
                        self.prev_raw = None
                        self.frames_since_pred = 0
                    else:
                        logger.info(f"Detenido. Frames: {len(self.frame_buffer)}")
                        if len(self.frame_buffer) >= self.MIN_FRAMES:
                            self.predictions = self._predict()
                            logger.info("Predicciones finales:")
                            for r, (lbl, p) in enumerate(self.predictions, 1):
                                logger.info(f"  {r}. {lbl}: {p*100:.1f}%")

                elif key == ord('r'):
                    self.frame_buffer.clear()
                    self.predictions = []
                    self.prev_raw = None
                    self.recording = False
                    self.frames_since_pred = 0
                    logger.info("Reset")

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            logger.info("Listo.")


def main():
    parser = argparse.ArgumentParser(description="Demo tiempo real V-LIBRASIL v2")
    parser.add_argument("--model-dir", type=Path,
                        default=Path("/data/models/vlibrasil-v2"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Confianza mínima para mostrar predicción")
    args = parser.parse_args()

    system = RealtimeVLibrasilV2(
        model_dir=args.model_dir,
        camera_id=args.camera,
        threshold=args.threshold,
    )
    system.run()


if __name__ == "__main__":
    main()

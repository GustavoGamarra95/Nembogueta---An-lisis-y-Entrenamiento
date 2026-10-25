#!/usr/bin/env python3
"""Synthetic demo video with correct MediaPipe Tasks API (matches training pipeline)."""
import os, sys, random, glob
import numpy as np
import cv2

sys.path.insert(0, '/app/src')
from preprocessing.feature_engineering import engineer_frame_features

import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python as mp_py
from mediapipe.tasks.python import vision as mp_vis

MODEL_DIR = '/data/models/librai-alphabet-v4/run_20260601_015641'
IMG_ROOT = '/app/src/LibrAI/test'
TASK_FILE = '/app/hand_landmarker.task'
OUT = '/tmp/supp_demo.mp4'

random.seed(42)

model = tf.keras.models.load_model(f'{MODEL_DIR}/best_model.keras', compile=False)
mean = np.load(f'{MODEL_DIR}/norm_mean.npy')
std = np.load(f'{MODEL_DIR}/norm_std.npy')
labels = [str(l) for l in np.load(f'{MODEL_DIR}/label_classes.npy', allow_pickle=True)]

opts = mp_vis.HandLandmarkerOptions(
    base_options=mp_py.BaseOptions(model_asset_path=TASK_FILE),
    running_mode=mp_vis.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.3,
)
detector = mp_vis.HandLandmarker.create_from_options(opts)

LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'L', 'O', 'P', 'U', 'W', 'Y', 'I', 'N', 'G']
FRAMES_PER_LETTER = 3
CANVAS = (720, 720)
FPS = 1

def draw_landmarks(canvas, result, w, h):
    if not result.hand_landmarks: return
    conns = mp.solutions.hands.HAND_CONNECTIONS
    for hand_lm in result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
        for a, b in conns:
            cv2.line(canvas, pts[a], pts[b], (0, 200, 0), 2)
        for p in pts:
            cv2.circle(canvas, p, 4, (0, 255, 0), -1)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUT, fourcc, FPS, CANVAS)

n_ok = n_wrong = n_nohand = 0

for letter in LETTERS:
    files = sorted(glob.glob(f'{IMG_ROOT}/{letter}/*.png'))
    if not files:
        print(f'skip {letter}'); continue
    picks = random.sample(files, min(FRAMES_PER_LETTER, len(files)))

    for f in picks:
        img = cv2.imread(f)
        if img is None: continue
        canvas = cv2.resize(img, CANVAS, interpolation=cv2.INTER_CUBIC)
        h, w = canvas.shape[:2]

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect(mp_img)

        if not res.hand_landmarks:
            pred_label, conf = '?', 0.0
            n_nohand += 1
        else:
            raw_126 = np.zeros(126, dtype=np.float32)
            for hand_lm, hness in zip(res.hand_landmarks, res.handedness):
                label = hness[0].display_name
                offset = 0 if label == 'Right' else 63
                for i, lm in enumerate(hand_lm):
                    raw_126[offset + i*3 + 0] = lm.x
                    raw_126[offset + i*3 + 1] = lm.y
                    raw_126[offset + i*3 + 2] = lm.z
            feat = engineer_frame_features(raw_126)
            feat_n = (feat - mean) / (std + 1e-8)
            probs = model.predict(feat_n[None, :], verbose=0)[0]
            k = int(np.argmax(probs))
            pred_label, conf = labels[k], float(probs[k])
            if pred_label == letter: n_ok += 1
            else: n_wrong += 1
            draw_landmarks(canvas, res, w, h)

        cv2.rectangle(canvas, (0, 0), (w, 90), (30, 30, 30), -1)
        color = (0, 220, 0) if pred_label == letter else (0, 100, 220)
        cv2.putText(canvas, f'Previsto: {pred_label}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
        cv2.putText(canvas, f'conf: {conf*100:.0f}%', (20, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(canvas, f'GT: {letter}', (w - 180, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.rectangle(canvas, (0, h-40), (w, h), (30, 30, 30), -1)
        cv2.putText(canvas, 'MediaPipe -> 120 features -> DNN',
                    (20, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        writer.write(canvas)

writer.release()
print(f'OK: {OUT}')
print(f'stats: correct={n_ok} wrong={n_wrong} no-hand={n_nohand}')

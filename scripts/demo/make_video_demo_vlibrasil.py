#!/usr/bin/env python3
"""V-LIBRASIL video demo: 8 videos → landmarks → 208 feats → mean+std pool → predict top-3.

Runs in nembogueta-dev-gpu container. Uses vlibrasil-v2 (99.4% test acc, 1361 classes).
"""
import os, sys, random, glob, json
from collections import deque
import numpy as np
import cv2

sys.path.insert(0, '/app/scripts/demo')          # for _fe208 (old 208-feat spec)
from _fe208 import engineer_frame_features       # noqa: E402

import tensorflow as tf
import mediapipe as mp

MODEL_DIR = '/data/models/vlibrasil-v2'
VIDEO_DIR = '/app/src/videos UFPE (V-LIBRASIL)/data'
OUT = os.environ.get('OUT', '/tmp/supp_demo_vlibrasil.mp4')
PHRASES_ONLY = os.environ.get('PHRASES_ONLY', '0') == '1'

random.seed(42)
model = tf.keras.models.load_model(f'{MODEL_DIR}/best_model.keras', compile=False)
with open(f'{MODEL_DIR}/metadata.json') as f:
    meta = json.load(f)
labels = meta['class_names']
norm_mean = np.load(f'{MODEL_DIR}/norm_mean.npy')[0]  # (416,)
norm_std  = np.load(f'{MODEL_DIR}/norm_std.npy')[0]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

def pool_seq(frames):
    return np.concatenate([frames.mean(0), frames.std(0)]).astype(np.float32)

def extract_raw(res):
    raw = np.zeros(126, dtype=np.float32)
    if res.multi_hand_landmarks:
        for i, hlm in enumerate(res.multi_hand_landmarks[:2]):
            s = i * 63
            for j, lm in enumerate(hlm.landmark):
                raw[s + j*3]     = lm.x
                raw[s + j*3 + 1] = lm.y
                raw[s + j*3 + 2] = lm.z
    return raw

def draw_lm(canvas, res, w, h):
    if not res.multi_hand_landmarks: return
    for hlm in res.multi_hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hlm.landmark]
        for a, b in mp_hands.HAND_CONNECTIONS:
            cv2.line(canvas, pts[a], pts[b], (0, 200, 0), 2)
        for p in pts: cv2.circle(canvas, p, 4, (0, 255, 0), -1)

# Pre-screen 40 random videos across all articuladores; keep 8 where model
# is correct with high confidence. Reports full stats too (no cherry-pick lie).
all_vids = sorted(glob.glob(f'{VIDEO_DIR}/*.mp4'))
def label_of(path):
    return os.path.basename(path).rsplit('_Articulador', 1)[0].strip()
valid = [p for p in all_vids if label_of(p) in labels]
if PHRASES_ONLY:
    valid = [p for p in valid if ' ' in label_of(p)]
    print(f'PHRASES_ONLY: {len(valid)} videos de frases (label con espacios)')
CANDIDATES = random.sample(valid, min(40, len(valid)))
print(f'Pre-screening {len(CANDIDATES)} candidates from {len(valid)} valid videos...')

def quick_predict(vpath):
    """Lightweight pre-screen: features only, no frame cache. Returns dict or None."""
    gt = label_of(vpath)
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened(): return None
    buf, prev_raw = [], None
    while True:
        ok, fr = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        r = hands.process(rgb)
        raw = extract_raw(r)
        buf.append(engineer_frame_features(raw, prev_raw))
        prev_raw = raw
    cap.release()
    if len(buf) < 8: return None
    x = (pool_seq(np.array(buf, dtype=np.float32)) - norm_mean) / (norm_std + 1e-8)
    probs = model.predict(x[None], verbose=0)[0]
    t3 = np.argsort(probs)[::-1][:3]
    return dict(vpath=vpath, gt=gt,
                top1=labels[t3[0]], conf=float(probs[t3[0]]),
                t3lbl=[labels[k] for k in t3],
                t3conf=[float(probs[k]) for k in t3])

screened, top1_hits, top3_hits = [], 0, 0
for i, p in enumerate(CANDIDATES):
    r = quick_predict(p)
    if r is None: continue
    if r['top1'] == r['gt']: top1_hits += 1
    if r['gt'] in r['t3lbl']: top3_hits += 1
    screened.append(r)
    if (i+1) % 5 == 0:
        print(f'  screened {i+1}/{len(CANDIDATES)} '
              f'(top1={top1_hits} top3={top3_hits})', flush=True)

correct = sorted([r for r in screened if r['top1'] == r['gt']],
                 key=lambda r: -r['conf'])[:8]
picks_data = correct if len(correct) >= 8 else screened[:8]

def extract_frames_and_lm(vpath):
    """Second pass: get frames + landmarks for rendering."""
    cap = cv2.VideoCapture(vpath)
    frames, lms = [], []
    while True:
        ok, fr = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        lms.append(hands.process(rgb))
        frames.append(fr)
    cap.release()
    return frames, lms
print(f'Pre-screen stats: top-1 = {top1_hits}/{len(screened)} '
      f'({top1_hits/max(1,len(screened))*100:.0f}%), '
      f'top-3 = {top3_hits}/{len(screened)} '
      f'({top3_hits/max(1,len(screened))*100:.0f}%)')
print(f'Selecting {len(picks_data)} for video (correct+high-conf).')

FPS = 25
CANVAS = (960, 720)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUT, fourcc, FPS, CANVAS)

for r in picks_data:
    gt, top1, conf = r['gt'], r['top1'], r['conf']
    t3lbl, t3conf = r['t3lbl'], r['t3conf']
    print(f'  render: GT={gt!r} pred={top1!r} ({conf*100:.1f}%)', flush=True)
    fcache, lcache = extract_frames_and_lm(r['vpath'])
    step = max(1, len(fcache) // (FPS * 3))
    df = fcache[::step]; dl = lcache[::step]
    feat_buf, prev_raw = [], None
    live_lbl, live_p = '...', 0.0
    PREDICT_EVERY = 3  # frames de video → nueva predicción rolling
    cw, ch = CANVAS
    for k, (f, res) in enumerate(zip(df, dl)):
        feat_buf.append(engineer_frame_features(extract_raw(res), prev_raw))
        prev_raw = extract_raw(res)
        if len(feat_buf) >= 4 and k % PREDICT_EVERY == 0:
            x = (pool_seq(np.array(feat_buf, dtype=np.float32)) - norm_mean) / (norm_std + 1e-8)
            p = model.predict(x[None], verbose=0)[0]
            idx = int(np.argmax(p))
            live_lbl, live_p = labels[idx], float(p[idx])
        canvas = cv2.resize(f, CANVAS, interpolation=cv2.INTER_LINEAR)
        draw_lm(canvas, res, cw, ch)
        cv2.rectangle(canvas, (0, 0), (cw, 40), (30, 30, 30), -1)
        cv2.putText(canvas, f'V-LIBRASIL - GT: {gt}', (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        # Overlay rolling: predicción en curso frame a frame
        cv2.rectangle(canvas, (0, ch-56), (cw, ch), (20, 20, 20), -1)
        color = (0, 220, 0) if live_lbl == gt else (255, 255, 255)
        cv2.putText(canvas, f'Predizindo: {live_lbl[:36]}  {live_p*100:5.1f}%',
                    (12, ch-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        writer.write(canvas)
    # Freeze with top-3 (~1.5 s)
    canvas = cv2.resize(fcache[-1], CANVAS, interpolation=cv2.INTER_LINEAR)
    draw_lm(canvas, lcache[-1], cw, ch)
    cv2.rectangle(canvas, (0, 0), (cw, 210), (20, 20, 20), -1)
    cv2.putText(canvas, f'Verdadeiro: {gt}', (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, 'Top-3 previsto:', (12, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 100), 2)
    for i, (lab, p) in enumerate(zip(t3lbl, t3conf)):
        color = (0, 220, 0) if lab == gt else ((255, 255, 255) if i == 0 else (170, 170, 170))
        cv2.putText(canvas, f'{i+1}. {lab[:32]}  {p*100:5.1f}%', (12, 100 + i*32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
    cv2.putText(canvas, f'Amostra de {top1_hits}/{len(screened)} top-1 e '
                        f'{top3_hits}/{len(screened)} top-3 corretas em pre-triagem',
                (12, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (150, 150, 150), 1)
    for _ in range(int(FPS * 1.5)):
        writer.write(canvas)
n_ok = len([r for r in picks_data if r['top1'] == r['gt']])

writer.release()
hands.close()
print(f'OK: {OUT}')
print(f'stats: top-1 correct = {n_ok}/{len(picks_data)}')

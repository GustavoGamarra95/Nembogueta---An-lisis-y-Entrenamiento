"""
Grabación guiada de señas y letras LSPy para evaluar en cámara real.

Muestra cada seña del vocabulario, cuenta regresiva y graba una ventana fija
guardando los landmarks crudos de MediaPipe Hands (mismo formato que
LSPy-Landmarks v1). Por defecto no guarda video (--save-video lo agrega).

Modo autor (--participant AUTOR): el propio investigador se graba, sin
terceros. Para grabar a otras personas haría falta un TCLE firmado
(--tcle-firmado) y un código seudónimo (P01, P02, …).

Uso:
    python scripts/collect/collect_lspy_signs.py \\
        --vocab data/analysis/vocabulario/vocabulario_objetivo.csv \\
        --participant AUTOR --reps 10

Controles:
    ESPACIO  grabar la seña mostrada
    R        repetir la última toma
    S        saltar la seña en esta ronda
    Q        salir (lo grabado queda guardado; se reanuda al relanzar)

Salida: data/collection/<participante>/<sesión>/
    session.json, <seña>__rep<k>.npz, takes.csv
"""
import argparse
import csv
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

WIN = "Ñemongueta — colecta LSPy"


def slug(s):
    return re.sub(r"[^\w]+", "_", s, flags=re.UNICODE).strip("_")


def landmarks_of(res):
    """(2,21,3) imagen, (2,21,3) world, handedness (2,), score (2,)."""
    lm = np.full((2, 21, 3), np.nan, np.float32)
    wl = np.full((2, 21, 3), np.nan, np.float32)
    hd, sc = ["", ""], np.zeros(2, np.float32)
    if res.multi_hand_landmarks:
        order = sorted(range(len(res.multi_hand_landmarks)),
                       key=lambda i: -res.multi_handedness[i]
                       .classification[0].score)[:2]
        for j, i in enumerate(order):
            lm[j] = [(p.x, p.y, p.z) for p in res.multi_hand_landmarks[i].landmark]
            if res.multi_hand_world_landmarks:
                wl[j] = [(p.x, p.y, p.z) for p in
                         res.multi_hand_world_landmarks[i].landmark]
            hd[j] = res.multi_handedness[i].classification[0].label
            sc[j] = res.multi_handedness[i].classification[0].score
    return lm, wl, hd, sc


def banner(frame, lines, color=(255, 255, 255)):
    h, w = frame.shape[:2]
    over = frame.copy()
    cv2.rectangle(over, (0, 0), (w, 40 + 45 * len(lines)), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(over, 0.6, frame, 0.4, 0)
    for k, text in enumerate(lines):
        cv2.putText(frame, text, (20, 50 + 45 * k), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2 if k == 0 else 0.7, color, 2)


def record(cap, hands, seconds, sign, video_path=None):
    """Graba `seconds` segundos; devuelve arrays y fracción con mano."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = None
    L, W, H, S, T = [], [], [], [], []
    t0 = time.time()
    while time.time() - t0 < seconds:
        ok, fr = cap.read()
        if not ok:
            break
        if video_path is not None and writer is None:
            writer = cv2.VideoWriter(str(video_path),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (fr.shape[1], fr.shape[0]))
        if writer:
            writer.write(fr)
        # MediaPipe sobre la imagen original sin espejar (como el dataset)
        lm, wl, hd, sc = landmarks_of(
            hands.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        L.append(lm), W.append(wl), H.append(hd), S.append(sc)
        T.append(time.time() - t0)
        view = cv2.flip(fr, 1)  # vista espejo para quien seña
        banner(view, [f"GRABANDO: {sign}",
                      f"{seconds - (time.time() - t0):.1f} s"], (0, 0, 255))
        cv2.imshow(WIN, view)
        cv2.waitKey(1)
    if writer:
        writer.release()
    L = np.stack(L) if L else np.zeros((0, 2, 21, 3), np.float32)
    has = (~np.isnan(L[:, :, 0, 0])).any(1).mean() if len(L) else 0.0
    return (L, np.stack(W) if W else L, np.array(H), np.stack(S) if S else
            np.zeros((0, 2)), np.array(T, np.float32), fps, float(has))


def wait_key(cap, lines, valid):
    while True:
        ok, fr = cap.read()
        if not ok:
            return "q"
        view = cv2.flip(fr, 1)
        banner(view, lines)
        cv2.imshow(WIN, view)
        k = cv2.waitKey(30) & 0xFF
        for key in valid:
            if k == ord(key) or (key == " " and k == 32):
                return key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=Path, required=True,
                    help="CSV con columna 'sena' (y opcional 'categoria')")
    ap.add_argument("--participant", required=True,
                    help="AUTOR (el investigador) o código seudónimo P01, P02…")
    ap.add_argument("--tcle-firmado", action="store_true",
                    help="Solo para terceros: confirma que firmaron el TCLE")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--countdown", type=int, default=3)
    ap.add_argument("--camera", default="0")
    ap.add_argument("--min-hand", type=float, default=0.5,
                    help="Fracción mínima de frames con mano para aceptar la toma")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--output-root", type=Path, default=Path("data/collection"))
    ap.add_argument("--session", default=None,
                    help="Nombre de sesión a reanudar (por defecto, la fecha)")
    args = ap.parse_args()

    autor = args.participant == "AUTOR"
    if not autor:
        if not args.tcle_firmado:
            raise SystemExit("Para grabar a terceros hace falta el TCLE "
                             "firmado (--tcle-firmado).")
        if not re.fullmatch(r"P\d{2,3}", args.participant):
            raise SystemExit("Usar un código seudónimo tipo P01, no el nombre.")

    signs = [r["sena"] for r in csv.DictReader(open(args.vocab))]
    session = args.session or datetime.now().strftime("%Y%m%d_%H%M")
    out = args.output_root / args.participant / session
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "session.json"
    if not meta_path.exists():
        json.dump({
            "participante": args.participant, "sesion": session,
            "fecha": datetime.now().isoformat(timespec="seconds"),
            "autor": autor, "tcle_firmado": bool(args.tcle_firmado),
            "vocabulario": str(args.vocab),
            "senas": signs, "repeticiones": args.reps,
            "segundos_por_toma": args.seconds,
            "landmarks": "MediaPipe Hands sobre la imagen original sin espejar",
            "video_guardado": args.save_video,
            "mano_dominante": input("Mano dominante (D/I): ").strip().upper(),
            "notas": input("Notas (iluminación, fondo, distancia): ").strip(),
        }, open(meta_path, "w"), indent=2, ensure_ascii=False)
    takes_path = out / "takes.csv"
    new_file = not takes_path.exists()
    takes = open(takes_path, "a", newline="")
    tw = csv.writer(takes)
    if new_file:
        tw.writerow(["sena", "rep", "archivo", "n_frames", "frac_mano",
                     "fps", "hora"])

    cam = int(args.camera) if args.camera.isdigit() else args.camera
    cap = cv2.VideoCapture(cam)
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                     min_detection_confidence=0.5)
    try:
        for rep in range(1, args.reps + 1):
            order = signs[:]
            random.Random(f"{args.participant}-{rep}").shuffle(order)
            for sign in order:
                name = f"{slug(sign)}__rep{rep:02d}.npz"
                if (out / name).exists():
                    continue  # reanudar
                while True:
                    key = wait_key(cap, [sign, f"ronda {rep}/{args.reps}  "
                                         "ESPACIO grabar · S saltar · Q salir"],
                                   " sq")
                    if key == "q":
                        return
                    if key == "s":
                        break
                    for c in range(args.countdown, 0, -1):
                        t_end = time.time() + 1
                        while time.time() < t_end:
                            ok, fr = cap.read()
                            view = cv2.flip(fr, 1)
                            banner(view, [f"{sign}", f"prepárate… {c}"],
                                   (0, 255, 255))
                            cv2.imshow(WIN, view)
                            cv2.waitKey(1)
                    vid = (out / name.replace(".npz", ".mp4")
                           if args.save_video else None)
                    L, W, H, S, T, fps, has = record(
                        cap, hands, args.seconds, sign, vid)
                    if has < args.min_hand:
                        wait_key(cap, [f"Mano detectada solo {has:.0%}",
                                       "ESPACIO para repetir"], " ")
                        continue
                    np.savez_compressed(out / name, landmarks=L, world=W,
                                        handedness=H, score=S, t=T, fps=fps,
                                        sena=sign, rep=rep,
                                        participante=args.participant)
                    tw.writerow([sign, rep, name, len(L), round(has, 3),
                                 round(fps, 2), datetime.now().isoformat(
                                     timespec="seconds")])
                    takes.flush()
                    key = wait_key(cap, [f"OK {sign} ({has:.0%} con mano)",
                                         "ESPACIO seguir · R repetir"], " r")
                    if key == "r":
                        (out / name).unlink(missing_ok=True)
                        continue
                    break
    finally:
        takes.close()
        cap.release()
        cv2.destroyAllWindows()
        n = len(list(out.glob("*.npz")))
        print(f"{n} tomas guardadas en {out}")


if __name__ == "__main__":
    main()

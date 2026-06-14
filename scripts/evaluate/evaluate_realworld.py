"""
Evaluación en cámara real del modelo de alfabeto.

A diferencia de guided_calibration.py (que captura solo cuando la confianza
supera un umbral, para colectar datos limpios), este script captura TODOS
los frames durante una ventana fija por repetición. El objetivo es medir
qué predice el modelo, no qué queremos que prediga.

Uso (dentro del contenedor):
    python scripts/evaluate/evaluate_realworld.py \\
        --model-dir /data/models/librai-alphabet-v4/run_20260601_015641 \\
        --output-dir /data/realworld_eval \\
        --subject-id S01 \\
        --reps 5

Controles durante la sesión:
    SPACE  — pasar a la siguiente repetición
    R      — re-grabar la repetición actual
    S      — saltar la letra (no contar en el reporte)
    Q      — salir y guardar lo capturado hasta ese punto
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from preprocessing.feature_engineering import (  # noqa: E402
    engineer_frame_features,
)

WINDOW_FRAMES_DEFAULT = 60   # ~2s a 30fps
MIN_DETECTED_RATIO = 0.5     # < 50% frames con mano detectada → forzar regrabación
REF_IMAGE_ROOT = (
    Path(__file__).resolve().parents[2] / "src" / "LibrAI" / "test"
)


def load_artifacts(model_dir: Path):
    model = tf.keras.models.load_model(str(model_dir / "best_model.keras"))
    norm_mean = np.load(model_dir / "norm_mean.npy")
    norm_std = np.load(model_dir / "norm_std.npy")
    classes = np.load(model_dir / "label_classes.npy")
    with open(model_dir / "model_info.json") as f:
        info = json.load(f)
    return model, norm_mean, norm_std, list(classes), info


def extract_raw_126(results) -> np.ndarray:
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


def prompt(question: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"{question}{suffix}: ").strip()
    return answer or default


def collect_session_metadata(subject_id: str, model_info: dict) -> dict:
    print("\n=== Metadatos de sesión ===")
    print("(Enter para usar el valor por defecto entre corchetes)\n")
    meta = {
        "subject_id": subject_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "age": prompt("Edad", "n/a"),
        "dominant_hand": prompt("Mano dominante (R/L)", "R"),
        "libras_experience_0_5": prompt(
            "Experiencia con LIBRAS/LSP (0-5)", "0"
        ),
        "lighting_1_5": prompt(
            "Iluminación (1=muy oscura, 5=muy clara)", "3"
        ),
        "background": prompt("Fondo (uniforme/cluttered)", "uniforme"),
        "hand_to_camera_cm": prompt("Distancia mano-cámara (cm)", "50"),
        "camera_height": prompt("Altura cámara (pecho/cara/mesa)", "pecho"),
        "notes": prompt("Notas adicionales", ""),
        "model_dir": model_info.get("_model_dir", ""),
        "model_test_accuracy": model_info.get("test_accuracy"),
        "model_num_features": model_info.get("num_features"),
    }
    return meta


def load_reference_image(letter: str) -> np.ndarray | None:
    letter_dir = REF_IMAGE_ROOT / letter
    if not letter_dir.exists():
        return None
    pngs = sorted(letter_dir.glob("*.png"))
    if not pngs:
        return None
    img = cv2.imread(str(pngs[0]))
    if img is None:
        return None
    return cv2.resize(img, (180, 180))


def draw_countdown(frame, seconds_left: int):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    text = str(seconds_left)
    (tw, th), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 12, 18
    )
    cv2.putText(
        frame, text, ((w - tw) // 2, (h + th) // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 12, (0, 255, 255), 18,
    )


def predict_batch(
    features_batch: np.ndarray,
    model,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
) -> np.ndarray:
    """features_batch (N, 120) → probs (N, num_classes)."""
    normalized = (features_batch - norm_mean) / norm_std
    return model.predict(normalized, verbose=0)


def run_repetition(
    cap, hands, mp_draw, mp_hands,
    model, norm_mean, norm_std, classes,
    target_letter: str, letter_idx: int, rep_idx: int,
    total_letters: int, total_reps: int,
    window_frames: int, ref_img: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, int, str]:
    """
    Devuelve (features (N, 120), probs (N, num_classes), n_detected, action)
    action ∈ {"keep", "redo", "skip", "quit"}
    """
    target_idx = classes.index(target_letter) if target_letter in classes else -1
    WIN = "Evaluación cámara real — Nembogueta"

    # Fase 1: countdown 3s
    countdown_start = time.time()
    while time.time() - countdown_start < 3.0:
        ret, frame = cap.read()
        if not ret:
            return np.empty((0, 120)), np.empty((0,)), 0, "quit"
        frame = cv2.flip(frame, 1)
        seconds_left = 3 - int(time.time() - countdown_start)
        draw_countdown(frame, max(seconds_left, 1))
        cv2.putText(
            frame, f"Prepara: {target_letter}", (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
        )
        if ref_img is not None:
            frame[20:200, frame.shape[1] - 200:frame.shape[1] - 20] = ref_img
        cv2.imshow(WIN, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return np.empty((0, 120)), np.empty((0,)), 0, "quit"

    # Fase 2: captura ventana fija
    features_list = []
    probs_list = []
    detected = 0
    captured = 0
    action = "keep"

    while captured < window_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                )
            raw_126 = extract_raw_126(results)
            feat = engineer_frame_features(raw_126)
            probs = predict_batch(feat[np.newaxis], model, norm_mean, norm_std)[0]
            features_list.append(feat)
            probs_list.append(probs)
            detected += 1
        else:
            features_list.append(np.zeros(120, dtype=np.float32))
            probs_list.append(np.zeros(len(classes), dtype=np.float32))
        captured += 1

        # UI
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 130), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
        cv2.putText(
            frame, target_letter, (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX, 4.5, (0, 255, 255), 8,
        )
        cv2.putText(
            frame,
            f"Letra {letter_idx + 1}/{total_letters}  "
            f"Rep {rep_idx + 1}/{total_reps}",
            (200, 40), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (200, 200, 200), 1,
        )
        cv2.putText(
            frame,
            f"Frames {captured}/{window_frames}  "
            f"con mano: {detected}",
            (200, 70), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (200, 200, 200), 1,
        )

        # predicción actual (solo informativo)
        if probs_list and detected > 0:
            top_idx = int(np.argmax(probs_list[-1]))
            cv2.putText(
                frame,
                f"Detecta: {classes[top_idx]} "
                f"({probs_list[-1][top_idx] * 100:.0f}%)",
                (200, 100), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (180, 220, 180), 1,
            )

        if ref_img is not None:
            frame[20:200, w - 200:w - 20] = ref_img

        cv2.putText(
            frame, "SPACE: siguiente   R: regrabar   "
            "S: saltar letra   Q: salir",
            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (160, 160, 160), 1,
        )
        cv2.imshow(WIN, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            action = "quit"
            break
        if key == ord("r"):
            action = "redo"
            break
        if key == ord("s"):
            action = "skip"
            break

    features = np.array(features_list, dtype=np.float32)
    probs = np.array(probs_list, dtype=np.float32)
    return features, probs, detected, action


def save_repetition(out_dir: Path, letter: str, rep_idx: int,
                    features: np.ndarray, probs: np.ndarray,
                    n_detected: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{letter}_rep{rep_idx:02d}.npz",
        features=features,
        probs=probs,
        n_detected=n_detected,
    )


def build_report(session_dir: Path, classes: list, model_info: dict) -> dict:
    """Lee todos los .npz de la sesión y genera el reporte."""
    rep_files = sorted((session_dir / "letters").glob("*.npz"))
    per_letter = defaultdict(lambda: {
        "reps": 0,
        "frames_total": 0,
        "frames_detected": 0,
        "top1_correct": 0,
        "top5_correct": 0,
        "mean_target_conf": [],
    })
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)

    for f in rep_files:
        # filename: A_rep00.npz  → letter = A
        letter = f.stem.split("_rep")[0]
        if letter not in classes:
            continue
        target_idx = classes.index(letter)

        data = np.load(f)
        probs = data["probs"]
        n_detected = int(data["n_detected"])
        if n_detected == 0:
            continue

        # solo frames con mano detectada
        mask = probs.sum(axis=1) > 0
        probs_d = probs[mask]
        top1 = np.argmax(probs_d, axis=1)
        top5 = np.argsort(probs_d, axis=1)[:, -5:]

        per_letter[letter]["reps"] += 1
        per_letter[letter]["frames_total"] += int(probs.shape[0])
        per_letter[letter]["frames_detected"] += int(mask.sum())
        per_letter[letter]["top1_correct"] += int((top1 == target_idx).sum())
        per_letter[letter]["top5_correct"] += int(
            np.any(top5 == target_idx, axis=1).sum()
        )
        per_letter[letter]["mean_target_conf"].extend(
            probs_d[:, target_idx].tolist()
        )
        for pred_idx in top1:
            confusion[target_idx, int(pred_idx)] += 1

    # consolidar
    report = {
        "model": {
            "test_accuracy_holdout": model_info.get("test_accuracy"),
            "num_features": model_info.get("num_features"),
            "num_classes": len(classes),
        },
        "per_letter": {},
        "global": {},
        "classes": classes,
    }
    total_frames = 0
    total_top1 = 0
    total_top5 = 0
    for letter, d in per_letter.items():
        det = d["frames_detected"]
        report["per_letter"][letter] = {
            "reps": d["reps"],
            "frames_detected": det,
            "frames_total": d["frames_total"],
            "top1_accuracy": d["top1_correct"] / det if det else 0.0,
            "top5_accuracy": d["top5_correct"] / det if det else 0.0,
            "mean_confidence_on_target": (
                float(np.mean(d["mean_target_conf"]))
                if d["mean_target_conf"] else 0.0
            ),
        }
        total_frames += det
        total_top1 += d["top1_correct"]
        total_top5 += d["top5_correct"]

    if total_frames > 0:
        report["global"] = {
            "frames_detected": total_frames,
            "top1_accuracy": total_top1 / total_frames,
            "top5_accuracy": total_top5 / total_frames,
            "letters_evaluated": len(report["per_letter"]),
        }
        if model_info.get("test_accuracy"):
            report["global"]["gap_vs_holdout"] = (
                model_info["test_accuracy"] - total_top1 / total_frames
            )

    np.save(session_dir / "confusion_matrix.npy", confusion)
    return report


def plot_confusion(session_dir: Path, classes: list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.load(session_dir / "confusion_matrix.npy")
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0
    )
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión — cámara real (normalizada por fila)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(session_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def run(args):
    model_dir = Path(args.model_dir)
    model, norm_mean, norm_std, classes, info = load_artifacts(model_dir)
    info["_model_dir"] = str(model_dir)
    print(
        f"\nModelo cargado: {model_dir.name} · "
        f"{info['num_features']} features · {len(classes)} clases · "
        f"test_acc holdout: {info.get('test_accuracy', 0) * 100:.2f}%"
    )

    letters = (
        list(args.letters) if args.letters
        else [c for c in classes]
    )
    missing = [l for l in letters if l not in classes]
    if missing:
        print(f"ADVERTENCIA: letras no soportadas por el modelo: {missing}")
        letters = [l for l in letters if l in classes]

    meta = collect_session_metadata(args.subject_id, info)
    session_dir = (
        Path(args.output_dir) / "sessions" /
        f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{args.subject_id}"
    )
    (session_dir / "letters").mkdir(parents=True, exist_ok=True)
    with open(session_dir / "session.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nSesión: {session_dir}\n")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1 if args.camera == 0 else 0)
    if not cap.isOpened():
        print("ERROR: no se pudo abrir la cámara")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    meta["camera_fps"] = actual_fps

    cv2.namedWindow("Evaluación cámara real — Nembogueta", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Evaluación cámara real — Nembogueta", 1280, 720)

    print(
        f"\nProtocolo: {len(letters)} letras × {args.reps} reps × "
        f"{args.window} frames (~{args.window / actual_fps:.1f}s cada una)\n"
    )

    quit_now = False
    for i, letter in enumerate(letters):
        if quit_now:
            break
        ref_img = load_reference_image(letter)
        rep_idx = 0
        while rep_idx < args.reps:
            features, probs, n_det, action = run_repetition(
                cap, hands, mp_draw, mp_hands,
                model, norm_mean, norm_std, classes,
                letter, i, rep_idx, len(letters), args.reps,
                args.window, ref_img,
            )
            if action == "quit":
                quit_now = True
                break
            if action == "skip":
                print(f"  → {letter} saltada en rep {rep_idx + 1}")
                break
            if action == "redo":
                print(f"  ↻ {letter} rep {rep_idx + 1} regrabando")
                continue

            ratio = n_det / max(features.shape[0], 1)
            if ratio < MIN_DETECTED_RATIO:
                print(
                    f"  ⚠ {letter} rep {rep_idx + 1}: solo {ratio * 100:.0f}% "
                    "frames con mano detectada — regrabando"
                )
                continue

            save_repetition(
                session_dir / "letters", letter, rep_idx,
                features, probs, n_det,
            )
            print(
                f"  ✓ {letter} rep {rep_idx + 1}/{args.reps} guardada "
                f"({n_det}/{features.shape[0]} frames con mano)"
            )
            rep_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    print("\n=== Generando reporte ===")
    report = build_report(session_dir, classes, info)
    with open(session_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    try:
        plot_confusion(session_dir, classes)
    except Exception as e:
        print(f"Aviso: no se pudo generar matriz visual: {e}")

    g = report.get("global", {})
    if g:
        print(f"\nResultados — sujeto {args.subject_id}")
        print(f"  Frames evaluados   : {g['frames_detected']}")
        print(f"  Top-1 cámara real  : {g['top1_accuracy'] * 100:.2f}%")
        print(f"  Top-5 cámara real  : {g['top5_accuracy'] * 100:.2f}%")
        print(
            f"  Test holdout       : "
            f"{info.get('test_accuracy', 0) * 100:.2f}%"
        )
        if "gap_vs_holdout" in g:
            print(
                f"  Gap dominio        : "
                f"{g['gap_vs_holdout'] * 100:.2f} pp"
            )
    print(f"\nArtefactos en: {session_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluación del modelo de alfabeto en cámara real",
    )
    parser.add_argument(
        "--model-dir", type=Path,
        default=Path(
            "/data/models/librai-alphabet-v4/run_20260601_015641"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/data/realworld_eval"),
    )
    parser.add_argument("--subject-id", type=str, required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--reps", type=int, default=5,
        help="Repeticiones por letra (default: 5)",
    )
    parser.add_argument(
        "--window", type=int, default=WINDOW_FRAMES_DEFAULT,
        help=f"Frames por repetición (default: {WINDOW_FRAMES_DEFAULT})",
    )
    parser.add_argument(
        "--letters", type=str, default="",
        help="Subconjunto de letras (ej: ABCDE). Vacío = todas las del modelo",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

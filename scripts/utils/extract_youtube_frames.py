"""
Extrae frames de videos de YouTube para alimentar datos de una letra específica.

Descarga el video con yt-dlp, aplica MediaPipe Hands frame a frame y guarda
las imágenes recortadas de la mano en src/LibrAI_extra/{LETRA}/.

Uso:
    python scripts/utils/extract_youtube_frames.py --letter Ç \
        --urls "https://www.youtube.com/shorts/g7dQU_DNMCg" \
               "https://www.youtube.com/watch?v=ybRbAH17eZU" \
               "https://www.youtube.com/watch?v=6669YqT9uQA"

Opciones:
    --max-frames  Máximo de frames a extraer por video (default: 200)
    --skip        Tomar 1 frame cada N (default: 3, para evitar duplicados)
    --preview     Mostrar ventana cv2 de previsualización (default: off)
    --output-dir  Directorio raíz de salida (default: src/LibrAI_extra)
"""

import argparse
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

DEFAULT_OUTPUT = Path("src/LibrAI_extra")
WINDOW_NAME = "Extracción — preview"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path("hand_landmarker.task")


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        print(f"Descargando modelo MediaPipe ({MODEL_URL})...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Modelo guardado en {MODEL_PATH}")
    return MODEL_PATH


def download_video(url: str, dest_dir: Path) -> Path | None:
    """Descarga el video usando yt-dlp y retorna la ruta del archivo."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--cookies-from-browser", "firefox",
        "--js-runtimes", "node",
        "--remote-components", "ejs:github",
        "-f", "best[ext=mp4]/best",
        "-o", str(dest_dir / "%(id)s.%(ext)s"),
        "--quiet",
        "--no-warnings",
        url,
    ]
    print(f"  Descargando: {url}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR al descargar: {result.stderr.strip()}")
        return None

    # Busca el archivo descargado
    files = list(dest_dir.glob("*"))
    if not files:
        print("  ERROR: no se encontró archivo descargado")
        return None
    return max(files, key=lambda f: f.stat().st_mtime)


def extract_frames(
    video_path: Path,
    out_dir: Path,
    letter: str,
    max_frames: int,
    skip: int,
    preview: bool,
    detector,
    start_idx: int,
) -> int:
    """
    Extrae hasta max_frames frames del video, detecta mano con MediaPipe y
    guarda el recorte. Retorna el número de frames guardados.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: no se pudo abrir {video_path.name}")
        return 0

    saved = 0
    frame_idx = 0
    idx = start_idx

    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % skip != 0:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        if not result.hand_landmarks:
            continue

        # Recorta bounding box de la primera mano detectada
        h, w = frame.shape[:2]
        lm = result.hand_landmarks[0]
        xs = [l.x for l in lm]
        ys = [l.y for l in lm]
        margin = 0.15
        x1 = max(0, int((min(xs) - margin) * w))
        y1 = max(0, int((min(ys) - margin) * h))
        x2 = min(w, int((max(xs) + margin) * w))
        y2 = min(h, int((max(ys) + margin) * h))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_resized = cv2.resize(crop, (64, 64))
        filename = out_dir / f"{letter}_{idx:05d}.png"
        cv2.imwrite(str(filename), crop_resized)
        idx += 1
        saved += 1

        if preview:
            # Una sola ventana reutilizada — no se abre una nueva por video
            display = frame.copy()
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"{letter} [{saved}/{max_frames}]",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
            cv2.imshow(WINDOW_NAME, display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("  Interrumpido por usuario")
                break

    cap.release()
    print(f"  Guardados: {saved} frames ({video_path.name})")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Extrae frames de YouTube para una letra")
    parser.add_argument("--letter",     required=True, help="Letra a extraer (ej: Ç)")
    parser.add_argument("--urls",       nargs="+", required=True, help="URLs de YouTube")
    parser.add_argument("--max-frames", type=int, default=200,
                        help="Máximo de frames por video (default: 200)")
    parser.add_argument("--skip",       type=int, default=3,
                        help="Tomar 1 frame cada N (default: 3)")
    parser.add_argument("--preview",    action="store_true",
                        help="Mostrar ventana de previsualización")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Directorio raíz de salida (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    letter = args.letter.upper()
    out_dir = args.output_dir / letter
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(out_dir.glob("*.png")))
    print(f"\n=== Extracción de frames para letra: {letter} ===")
    print(f"Salida: {out_dir}")
    print(f"Ya existen: {existing} imágenes")
    print(f"Videos a procesar: {len(args.urls)}")
    print(f"Max frames/video: {args.max_frames}  |  skip: {args.skip}")
    print(f"Preview cv2: {'sí' if args.preview else 'no (modo silencioso)'}\n")

    model_path = ensure_model()
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    total_saved = 0
    start_idx = existing

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, url in enumerate(args.urls, 1):
            print(f"[{i}/{len(args.urls)}] {url}")
            video_path = download_video(url, Path(tmpdir) / f"video_{i}")
            if video_path is None:
                continue

            saved = extract_frames(
                video_path=video_path,
                out_dir=out_dir,
                letter=letter,
                max_frames=args.max_frames,
                skip=args.skip,
                preview=args.preview,
                detector=detector,
                start_idx=start_idx,
            )
            total_saved += saved
            start_idx += saved

    detector.close()

    if args.preview:
        cv2.destroyAllWindows()

    total = existing + total_saved
    print(f"\n✓ Extracción completa: {total_saved} nuevas imágenes ({total} total en {out_dir})")


if __name__ == "__main__":
    main()

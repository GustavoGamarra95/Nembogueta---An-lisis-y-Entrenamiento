"""
Preprocesamiento de datasets de imágenes para el alfabeto LIBRAS/BSL.

Soporta dos estructuras de directorio:
  - Plana    : input_dir/{LETRA}/*.png|jpg   (e.g. biankatpas)
  - Anidada  : input_dir/train/{LETRA}/*.png  +  input_dir/test/{LETRA}/*.png
               (e.g. williansoliveira/libras — LibrAI)

Para cada imagen: OpenCV → MediaPipe Hands (static_image_mode) →
engineer_frame_features() → vector (208,) → guarda como .npy.

Uso (dentro del contenedor):

  # Dataset LibrAI (estructura train/test):
  python scripts/preprocess/preprocess_bsl_alphabet.py \
      --input-dir /app/src/LibrAI \
      --output-dir /data/processed/librai_alphabet

  # Dataset biankatpas (estructura plana):
  python scripts/preprocess/preprocess_bsl_alphabet.py \
      --input-dir /data/raw/bsl_alphabet/dataset \
      --output-dir /data/processed/bsl_alphabet
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from preprocessing.feature_engineering import engineer_frame_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG")


def get_images(letter_dir: Path) -> list[Path]:
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(letter_dir.glob(ext))
    return sorted(images)


def extract_features(image_path: Path, hands) -> np.ndarray | None:
    """
    Extrae vector (208,) desde una imagen usando MediaPipe.
    Retorna None si no se detecta ninguna mano.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None

    raw_126 = np.zeros(126, dtype=np.float32)
    for hand_lm, handedness in zip(
        results.multi_hand_landmarks, results.multi_handedness
    ):
        label = handedness.classification[0].label  # "Left" o "Right"
        offset = 0 if label == "Right" else 63
        for i, lm in enumerate(hand_lm.landmark):
            raw_126[offset + i * 3 + 0] = lm.x
            raw_126[offset + i * 3 + 1] = lm.y
            raw_126[offset + i * 3 + 2] = lm.z

    return engineer_frame_features(raw_126, prev_landmarks=None).astype(np.float32)


def collect_letter_dirs(input_dir: Path) -> list[tuple[str, Path]]:
    """
    Detecta si el dataset tiene estructura plana o anidada (train/test)
    y devuelve lista de (letra, directorio).

    Estructura plana  : input_dir/A/, input_dir/B/, ...
    Estructura anidada: input_dir/train/A/, input_dir/test/A/, ...
    """
    subdirs = [d for d in input_dir.iterdir() if d.is_dir()]
    subnames = {d.name.lower() for d in subdirs}

    # Si hay carpetas llamadas "train" o "test" → estructura anidada
    if subnames & {"train", "test"}:
        logger.info("Estructura detectada: anidada (train/test)")
        pairs = []
        for split in ("train", "test"):
            split_dir = input_dir / split
            if not split_dir.exists():
                continue
            for letter_dir in sorted(split_dir.iterdir()):
                if letter_dir.is_dir():
                    pairs.append((letter_dir.name.upper(), letter_dir))
        return pairs
    else:
        logger.info("Estructura detectada: plana (letras directamente)")
        return [
            (d.name.upper(), d) for d in sorted(subdirs) if d.is_dir()
        ]


def preprocess_dataset(
    input_dir: Path,
    output_dir: Path,
    max_per_letter: int = 0,
    skip_existing: bool = True,
):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.3,
    )

    letter_dirs = collect_letter_dirs(input_dir)
    if not letter_dirs:
        logger.error("No se encontraron carpetas de letras en el directorio de entrada.")
        hands.close()
        return

    # Agrupar por letra (puede haber train + test para la misma letra)
    from collections import defaultdict
    grouped: dict[str, list[Path]] = defaultdict(list)
    for letter, d in letter_dirs:
        grouped[letter].append(d)

    total_saved = 0
    total_skipped = 0
    stats = {}

    for letter in sorted(grouped.keys()):
        letter_out = output_dir / letter
        letter_out.mkdir(parents=True, exist_ok=True)

        # Contar índice actual (para no sobrescribir si skip_existing)
        existing = len(list(letter_out.glob("*.npy")))
        idx = existing

        saved = 0
        skipped = 0

        all_images: list[Path] = []
        for d in grouped[letter]:
            all_images.extend(get_images(d))

        if max_per_letter > 0:
            all_images = all_images[:max_per_letter]

        for img_path in all_images:
            features = extract_features(img_path, hands)
            if features is None:
                skipped += 1
                continue

            out_file = letter_out / f"{letter}_{idx:05d}.npy"
            np.save(out_file, features)
            idx += 1
            saved += 1

        logger.info(f"  {letter}: {saved} guardados, {skipped} sin detección")
        stats[letter] = {"saved": saved, "skipped": skipped}
        total_saved += saved
        total_skipped += skipped

    hands.close()

    detection_rate = total_saved / max(1, total_saved + total_skipped) * 100
    logger.info("=" * 50)
    logger.info(f"Total guardados : {total_saved}")
    logger.info(f"Total sin mano  : {total_skipped}")
    logger.info(f"Tasa detección  : {detection_rate:.1f}%")
    logger.info(f"Salida          : {output_dir}")

    with open(output_dir / "stats.json", "w") as f:
        json.dump(
            {
                "total_saved": total_saved,
                "total_skipped": total_skipped,
                "detection_rate_pct": round(detection_rate, 2),
                "per_letter": stats,
            },
            f,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Preprocesa imágenes de alfabeto LIBRAS → 208 features MediaPipe (.npy)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directorio raíz del dataset (estructura plana o anidada train/test)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/processed/librai_alphabet",
        help="Directorio de salida para archivos .npy",
    )
    parser.add_argument(
        "--max-per-letter",
        type=int,
        default=0,
        help="Máximo de imágenes por letra, 0 = todas",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logger.error(f"Directorio de entrada no existe: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 50)
    logger.info("PREPROCESAMIENTO ALFABETO LIBRAS (MediaPipe)")
    logger.info("=" * 50)
    logger.info(f"Input : {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 50)

    preprocess_dataset(input_dir, output_dir, args.max_per_letter)


if __name__ == "__main__":
    main()

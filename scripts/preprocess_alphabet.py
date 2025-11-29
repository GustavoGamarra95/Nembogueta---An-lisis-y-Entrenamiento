"""
Script de preprocesamiento para landmarks del alfabeto (A-Z).
Convierte archivos CSV a formato NPY normalizado para entrenamiento.

Uso:
    python scripts/preprocess_alphabet.py --input-dir data/landmarks \
                                          --output-dir data/processed/alphabet \
                                          --sequence-length 30
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alphabet_preprocessing.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_csv_landmarks(csv_path: Path) -> np.ndarray:
    """
    Carga landmarks desde archivo CSV.

    Args:
        csv_path: Ruta al archivo CSV

    Returns:
        Array de landmarks (n_frames, 63)
    """
    # Leer CSV sin headers
    df = pd.read_csv(csv_path, header=None)
    landmarks = df.values.astype(np.float32)

    logger.debug(f"Cargado {csv_path.name}: shape {landmarks.shape}")
    return landmarks


def normalize_sequence_length(
    sequence: np.ndarray,
    target_length: int
) -> np.ndarray:
    """
    Normaliza la longitud de una secuencia.

    Args:
        sequence: Secuencia original (n_frames, features)
        target_length: Longitud objetivo

    Returns:
        Secuencia normalizada (target_length, features)
    """
    current_length = len(sequence)

    if current_length == target_length:
        return sequence

    # Interpolar
    indices = np.linspace(0, current_length - 1, target_length)
    indices = np.round(indices).astype(int)
    return sequence[indices]


def process_letter_directory(
    letter_csv: Path,
    output_dir: Path,
    sequence_length: int = 30
):
    """
    Procesa un archivo CSV de una letra.

    Args:
        letter_csv: Ruta al archivo CSV de la letra
        output_dir: Directorio de salida
        sequence_length: Longitud de secuencia objetivo
    """
    letter = letter_csv.stem  # A, B, C, etc.
    letter_output_dir = output_dir / letter
    letter_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Procesando letra {letter}...")

    # Cargar landmarks
    landmarks = load_csv_landmarks(letter_csv)

    # landmarks shape: (n_samples, 63) donde cada fila es un frame
    # Cada muestra es un solo frame, necesitamos crear secuencias

    # Estrategia: crear secuencias deslizantes
    # Por ejemplo, cada 30 frames consecutivos forman una muestra

    if len(landmarks) < sequence_length:
        logger.warning(f"Letra {letter} tiene {len(landmarks)} frames, menos que {sequence_length}")
        # Repetir frames para alcanzar sequence_length
        repetitions = (sequence_length // len(landmarks)) + 1
        landmarks = np.tile(landmarks, (repetitions, 1))[:sequence_length]
        sequences = [landmarks]
    else:
        # Crear secuencias deslizantes con stride
        stride = max(1, len(landmarks) // 100)  # Aproximadamente 100 muestras por letra
        sequences = []

        for i in range(0, len(landmarks) - sequence_length + 1, stride):
            sequence = landmarks[i:i + sequence_length]
            sequences.append(sequence)

    logger.info(f"Letra {letter}: {len(sequences)} secuencias creadas")

    # Guardar cada secuencia
    for idx, sequence in enumerate(sequences):
        # Normalizar features (importante para CNN)
        sequence_mean = np.mean(sequence, axis=0, keepdims=True)
        sequence_std = np.std(sequence, axis=0, keepdims=True) + 1e-8
        normalized_sequence = (sequence - sequence_mean) / sequence_std

        # Guardar
        output_file = letter_output_dir / f"{letter}_{idx:04d}.npy"
        np.save(output_file, normalized_sequence)

    logger.info(f"Letra {letter}: {len(sequences)} archivos guardados en {letter_output_dir}")


def create_flat_structure(
    input_dir: Path,
    output_dir: Path,
    sequence_length: int = 30
):
    """
    Crea estructura plana con todos los archivos .npy.

    Args:
        input_dir: Directorio con CSVs
        output_dir: Directorio de salida plano
        sequence_length: Longitud de secuencia
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Buscar archivos CSV
    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos CSV en {input_dir}")

    logger.info(f"Encontrados {len(csv_files)} archivos CSV")

    for csv_file in csv_files:
        letter = csv_file.stem  # A, B, C, etc.
        logger.info(f"Procesando letra {letter}...")

        # Cargar landmarks
        landmarks = load_csv_landmarks(csv_file)

        # Crear secuencias
        if len(landmarks) < sequence_length:
            logger.warning(f"Letra {letter}: {len(landmarks)} frames < {sequence_length}")
            repetitions = (sequence_length // len(landmarks)) + 1
            landmarks = np.tile(landmarks, (repetitions, 1))[:sequence_length]
            sequences = [landmarks]
        else:
            stride = max(1, len(landmarks) // 100)
            sequences = []
            for i in range(0, len(landmarks) - sequence_length + 1, stride):
                sequences.append(landmarks[i:i + sequence_length])

        # Guardar en estructura plana
        for idx, sequence in enumerate(sequences):
            # Normalizar
            sequence_mean = np.mean(sequence, axis=0, keepdims=True)
            sequence_std = np.std(sequence, axis=0, keepdims=True) + 1e-8
            normalized_sequence = (sequence - sequence_mean) / sequence_std

            # Guardar con nombre: Letra_idx.npy
            output_file = output_dir / f"{letter}_{idx:04d}.npy"
            np.save(output_file, normalized_sequence)

        logger.info(f"Letra {letter}: {len(sequences)} secuencias guardadas")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocesa landmarks del alfabeto'
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directorio con archivos CSV'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directorio de salida para archivos .npy'
    )

    parser.add_argument(
        '--sequence-length',
        type=int,
        default=30,
        help='Longitud de secuencia (default: 30 frames)'
    )

    parser.add_argument(
        '--flat',
        action='store_true',
        help='Crear estructura plana (todos los .npy en un directorio)'
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logger.error(f"Directorio de entrada no existe: {input_dir}")
        return

    # Crear directorio de salida
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("PREPROCESAMIENTO DE ALFABETO")
    logger.info("="*60)
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Sequence length: {args.sequence_length}")
    logger.info(f"Estructura: {'Plana' if args.flat else 'Por letra'}")
    logger.info("="*60 + "\n")

    if args.flat:
        # Estructura plana
        create_flat_structure(input_dir, output_dir, args.sequence_length)
    else:
        # Estructura por letra
        csv_files = sorted(input_dir.glob("*.csv"))
        for csv_file in csv_files:
            process_letter_directory(csv_file, output_dir, args.sequence_length)

    logger.info("\n" + "="*60)
    logger.info("PREPROCESAMIENTO COMPLETADO")
    logger.info("="*60)

    # Estadísticas finales
    npy_files = list(output_dir.rglob("*.npy"))
    logger.info(f"Total de archivos .npy creados: {len(npy_files)}")

    if npy_files:
        # Cargar uno de muestra para verificar shape
        sample = np.load(npy_files[0])
        logger.info(f"Shape de secuencias: {sample.shape}")

    logger.info(f"Datos guardados en: {output_dir}")


if __name__ == '__main__':
    main()

"""
Script de Data Augmentation para letras problemáticas del alfabeto.
Aplica transformaciones para aumentar variabilidad sin alterar el gesto base.

Letras objetivo: U, V, G, L, Y (las que tienen peor performance)

Uso:
    python scripts/augment_problematic_letters.py \
        --input-dir /data/processed/alphabet-flat \
        --output-dir /data/processed/alphabet-augmented \
        --target-letters U V G L Y \
        --augmentation-factor 3
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alphabet_augmentation.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class LetterAugmenter:
    """Clase para aplicar data augmentation a secuencias de letras."""

    def __init__(self, seed: int = 42):
        """
        Inicializa el augmentador.

        Args:
            seed: Semilla para reproducibilidad
        """
        np.random.seed(seed)
        self.seed = seed

    def add_noise(self, sequence: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """
        Añade ruido gaussiano a la secuencia.

        Args:
            sequence: Secuencia original (frames, features)
            noise_level: Nivel de ruido (std dev)

        Returns:
            Secuencia con ruido
        """
        noise = np.random.normal(0, noise_level, sequence.shape)
        return sequence + noise

    def scale_sequence(self, sequence: np.ndarray, scale_factor: float = 1.1) -> np.ndarray:
        """
        Escala la secuencia (simula mano más grande/pequeña).

        Args:
            sequence: Secuencia original
            scale_factor: Factor de escala

        Returns:
            Secuencia escalada
        """
        return sequence * scale_factor

    def shift_sequence(self, sequence: np.ndarray, shift_range: float = 0.05) -> np.ndarray:
        """
        Desplaza la secuencia en el espacio (simula diferentes posiciones).

        Args:
            sequence: Secuencia original
            shift_range: Rango de desplazamiento

        Returns:
            Secuencia desplazada
        """
        shift = np.random.uniform(-shift_range, shift_range, sequence.shape[1])
        return sequence + shift

    def time_warp(self, sequence: np.ndarray, warp_factor: float = 0.2) -> np.ndarray:
        """
        Deforma temporalmente la secuencia (diferentes velocidades).

        Args:
            sequence: Secuencia original (frames, features)
            warp_factor: Factor de deformación temporal

        Returns:
            Secuencia deformada temporalmente
        """
        orig_length = len(sequence)

        # Crear índices deformados
        warp = 1.0 + np.random.uniform(-warp_factor, warp_factor, orig_length)
        warp = np.cumsum(warp)
        warp = warp / warp[-1] * (orig_length - 1)

        # Interpolar
        indices = np.clip(warp, 0, orig_length - 1).astype(int)
        return sequence[indices]

    def random_dropout(self, sequence: np.ndarray, dropout_prob: float = 0.05) -> np.ndarray:
        """
        Aplica dropout aleatorio a algunos frames (simula detección intermitente).

        Args:
            sequence: Secuencia original
            dropout_prob: Probabilidad de dropout por frame

        Returns:
            Secuencia con dropout
        """
        sequence = sequence.copy()
        mask = np.random.random(len(sequence)) > dropout_prob

        # Para frames con dropout, interpolar con vecinos
        for i in range(len(sequence)):
            if not mask[i]:
                # Interpolar con frame anterior y siguiente
                if i > 0 and i < len(sequence) - 1:
                    sequence[i] = (sequence[i-1] + sequence[i+1]) / 2
                elif i > 0:
                    sequence[i] = sequence[i-1]
                else:
                    sequence[i] = sequence[i+1]

        return sequence

    def augment_sequence(
        self,
        sequence: np.ndarray,
        augmentation_type: str = 'mixed'
    ) -> np.ndarray:
        """
        Aplica augmentation a una secuencia.

        Args:
            sequence: Secuencia original
            augmentation_type: Tipo de augmentation ('noise', 'scale', 'shift', 'warp', 'dropout', 'mixed')

        Returns:
            Secuencia aumentada
        """
        if augmentation_type == 'noise':
            return self.add_noise(sequence, noise_level=np.random.uniform(0.005, 0.02))

        elif augmentation_type == 'scale':
            scale = np.random.uniform(0.9, 1.1)
            return self.scale_sequence(sequence, scale_factor=scale)

        elif augmentation_type == 'shift':
            return self.shift_sequence(sequence, shift_range=np.random.uniform(0.02, 0.08))

        elif augmentation_type == 'warp':
            return self.time_warp(sequence, warp_factor=np.random.uniform(0.1, 0.3))

        elif augmentation_type == 'dropout':
            return self.random_dropout(sequence, dropout_prob=np.random.uniform(0.03, 0.08))

        elif augmentation_type == 'mixed':
            # Aplicar múltiples transformaciones
            aug_sequence = sequence.copy()

            # Probabilidad de aplicar cada transformación
            if np.random.random() > 0.5:
                aug_sequence = self.add_noise(aug_sequence, noise_level=np.random.uniform(0.005, 0.015))

            if np.random.random() > 0.5:
                scale = np.random.uniform(0.95, 1.05)
                aug_sequence = self.scale_sequence(aug_sequence, scale_factor=scale)

            if np.random.random() > 0.5:
                aug_sequence = self.shift_sequence(aug_sequence, shift_range=np.random.uniform(0.02, 0.05))

            if np.random.random() > 0.3:
                aug_sequence = self.time_warp(aug_sequence, warp_factor=np.random.uniform(0.1, 0.2))

            return aug_sequence

        else:
            raise ValueError(f"Tipo de augmentation desconocido: {augmentation_type}")


def augment_letter_data(
    input_dir: Path,
    output_dir: Path,
    target_letters: List[str],
    augmentation_factor: int = 3,
    augmentation_type: str = 'mixed'
):
    """
    Aumenta los datos para letras específicas.

    Args:
        input_dir: Directorio con datos originales
        output_dir: Directorio para datos aumentados
        target_letters: Lista de letras a aumentar
        augmentation_factor: Factor de aumento (ej. 3 = 3x más datos)
        augmentation_type: Tipo de augmentation
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    augmenter = LetterAugmenter()

    # Primero, copiar todos los datos originales
    logger.info("Copiando datos originales...")
    npy_files = list(input_dir.glob("*.npy"))

    for npy_file in npy_files:
        sequence = np.load(npy_file)
        output_path = output_dir / npy_file.name
        np.save(output_path, sequence)

    logger.info(f"Copiados {len(npy_files)} archivos originales")

    # Luego, aumentar letras problemáticas
    logger.info(f"\nAumentando datos para letras: {', '.join(target_letters)}")
    logger.info(f"Factor de aumento: {augmentation_factor}x")
    logger.info(f"Tipo de augmentation: {augmentation_type}")

    total_augmented = 0

    for letter in target_letters:
        # Encontrar archivos de esta letra
        letter_files = [f for f in npy_files if f.stem.split('_')[0] == letter]

        if not letter_files:
            logger.warning(f"No se encontraron archivos para letra '{letter}'")
            continue

        logger.info(f"\nLetra '{letter}': {len(letter_files)} archivos originales")

        # Crear versiones aumentadas
        for i, npy_file in enumerate(letter_files):
            sequence = np.load(npy_file)

            # Generar augmentation_factor versiones aumentadas
            for aug_idx in range(augmentation_factor):
                aug_sequence = augmenter.augment_sequence(sequence, augmentation_type)

                # Guardar con nuevo nombre
                base_name = npy_file.stem
                aug_name = f"{base_name}_aug{aug_idx}.npy"
                aug_path = output_dir / aug_name

                np.save(aug_path, aug_sequence)
                total_augmented += 1

        logger.info(f"  Generadas {len(letter_files) * augmentation_factor} versiones aumentadas")

    logger.info(f"\n{'='*60}")
    logger.info(f"RESUMEN")
    logger.info(f"{'='*60}")
    logger.info(f"Archivos originales copiados: {len(npy_files)}")
    logger.info(f"Archivos aumentados creados: {total_augmented}")
    logger.info(f"Total de archivos en output: {len(list(output_dir.glob('*.npy')))}")
    logger.info(f"Directorio de salida: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Data augmentation para letras problemáticas del alfabeto'
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directorio con datos procesados originales'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directorio para datos aumentados'
    )

    parser.add_argument(
        '--target-letters',
        type=str,
        nargs='+',
        default=['U', 'V', 'G', 'L', 'Y'],
        help='Letras a aumentar (default: U V G L Y)'
    )

    parser.add_argument(
        '--augmentation-factor',
        type=int,
        default=3,
        help='Factor de aumento (default: 3)'
    )

    parser.add_argument(
        '--augmentation-type',
        type=str,
        default='mixed',
        choices=['noise', 'scale', 'shift', 'warp', 'dropout', 'mixed'],
        help='Tipo de augmentation (default: mixed)'
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logger.error(f"Directorio de entrada no existe: {input_dir}")
        sys.exit(1)

    augment_letter_data(
        input_dir=input_dir,
        output_dir=output_dir,
        target_letters=args.target_letters,
        augmentation_factor=args.augmentation_factor,
        augmentation_type=args.augmentation_type
    )

    logger.info("\n¡Data augmentation completada!")


if __name__ == '__main__':
    main()

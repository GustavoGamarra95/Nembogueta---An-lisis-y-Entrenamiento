"""
Script para procesar el dataset de expresiones faciales de LIBRAS.
Carga landmarks faciales ya extraídos y los prepara para entrenamiento.
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('facial_expressions_processing.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class FacialExpressionsPreprocessor:
    """Procesador de expresiones faciales de LIBRAS."""

    def __init__(self, target_length: int = 100):
        """
        Inicializa el preprocesador.

        Args:
            target_length: Longitud objetivo de las secuencias temporales
        """
        self.target_length = target_length
        self.expression_types = [
            'affirmative', 'conditional', 'doubt_question', 'emphasis',
            'negative', 'relative', 'topics', 'wh_question', 'yn_question'
        ]
        logger.info(f"Preprocesador inicializado (target_length={target_length})")

    def load_datapoints(self, filepath: Path) -> np.ndarray:
        """
        Carga un archivo de datapoints.

        Args:
            filepath: Ruta al archivo de datapoints

        Returns:
            Array de shape (n_samples, n_timesteps, 300) donde:
            - n_samples: número de muestras en el archivo
            - n_timesteps: número de frames por muestra
            - 300: 100 landmarks × 3 coordenadas (x,y,z)
        """
        sequences = []

        with open(filepath, 'r') as f:
            lines = f.readlines()

            # Primera línea es el header
            header = lines[0].strip().split()
            n_features = len(header) - 1  # Excluir el timestamp

            # Procesar cada línea (cada línea es un frame)
            for line in lines[1:]:
                values = line.strip().split()

                # Extraer coordenadas (ignorar timestamp)
                coords = [float(v) for v in values[1:]]

                if len(coords) == n_features:
                    sequences.append(coords)

        # Convertir a array
        sequences_array = np.array(sequences, dtype=np.float32)
        logger.debug(f"Cargadas {len(sequences)} frames de {filepath.name}")

        return sequences_array

    def load_targets(self, filepath: Path) -> List[int]:
        """
        Carga un archivo de targets (etiquetas).

        Args:
            filepath: Ruta al archivo de targets

        Returns:
            Lista de etiquetas (0 o 1)
        """
        targets = []

        with open(filepath, 'r') as f:
            lines = f.readlines()

            for line in lines:
                target = int(line.strip())
                targets.append(target)

        logger.debug(f"Cargados {len(targets)} targets de {filepath.name}")
        return targets

    def split_into_sequences(
        self,
        datapoints: np.ndarray,
        targets: List[int]
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Divide los datapoints en secuencias individuales basándose en los targets.

        Args:
            datapoints: Array de todos los frames
            targets: Lista de etiquetas (un target por secuencia)

        Returns:
            Tupla de (secuencias, etiquetas)
        """
        n_sequences = len(targets)
        n_frames = len(datapoints)

        # Calcular frames por secuencia (aproximado)
        frames_per_seq = n_frames // n_sequences

        sequences = []
        labels = []

        start_idx = 0
        for i, target in enumerate(targets):
            # Determinar rango de frames para esta secuencia
            end_idx = start_idx + frames_per_seq

            # Ajustar el último rango para incluir frames restantes
            if i == n_sequences - 1:
                end_idx = n_frames

            seq = datapoints[start_idx:end_idx]

            if len(seq) > 0:
                sequences.append(seq)
                labels.append(target)

            start_idx = end_idx

        logger.debug(f"Divididos {n_frames} frames en {len(sequences)} secuencias")
        return sequences, labels

    def normalize_sequence_length(
        self,
        sequence: np.ndarray
    ) -> np.ndarray:
        """
        Normaliza la longitud de una secuencia mediante interpolación o padding.

        Args:
            sequence: Secuencia de shape (n_timesteps, 300)

        Returns:
            Secuencia normalizada de shape (target_length, 300)
        """
        current_length = len(sequence)

        if current_length == self.target_length:
            return sequence

        if current_length < self.target_length:
            # Padding: repetir el último frame
            padding = np.tile(
                sequence[-1:],
                (self.target_length - current_length, 1)
            )
            return np.vstack([sequence, padding])
        else:
            # Interpolación: muestrear frames uniformemente
            indices = np.linspace(
                0,
                current_length - 1,
                self.target_length,
                dtype=int
            )
            return sequence[indices]

    def normalize_coordinates(self, sequence: np.ndarray) -> np.ndarray:
        """
        Normaliza las coordenadas de una secuencia.

        Args:
            sequence: Secuencia de shape (n_timesteps, 300)

        Returns:
            Secuencia normalizada
        """
        # Normalización por media y desviación estándar
        mean = np.mean(sequence, axis=0)
        std = np.std(sequence, axis=0)

        # Evitar división por cero
        std[std == 0] = 1.0

        normalized = (sequence - mean) / std
        return normalized

    def process_expression_type(
        self,
        data_dir: Path,
        participant: str,
        expression_type: str,
        output_dir: Path
    ) -> Tuple[int, int]:
        """
        Procesa un tipo de expresión para un participante.

        Args:
            data_dir: Directorio con los datos raw
            participant: 'a' o 'b'
            expression_type: Tipo de expresión
            output_dir: Directorio de salida

        Returns:
            Tupla (procesados, fallidos)
        """
        # Archivos de entrada
        datapoints_file = data_dir / f"{participant}_{expression_type}_datapoints.txt"
        targets_file = data_dir / f"{participant}_{expression_type}_targets.txt"

        if not datapoints_file.exists():
            # Ajustar para nombres con guión bajo adicional
            datapoints_file = data_dir / f"{participant}_{expression_type.replace('_', '_')}_datapoints.txt"
            targets_file = data_dir / f"{participant}_{expression_type.replace('_', '_')}_targets.txt"

        if not datapoints_file.exists() or not targets_file.exists():
            logger.warning(f"Archivos no encontrados para {participant}_{expression_type}")
            return 0, 0

        # Cargar datos
        datapoints = self.load_datapoints(datapoints_file)
        targets = self.load_targets(targets_file)

        # Dividir en secuencias
        sequences, labels = self.split_into_sequences(datapoints, targets)

        # Procesar cada secuencia
        processed = 0
        failed = 0

        for i, (seq, label) in enumerate(zip(sequences, labels)):
            try:
                # Normalizar longitud
                seq_normalized = self.normalize_sequence_length(seq)

                # Normalizar coordenadas
                seq_final = self.normalize_coordinates(seq_normalized)

                # Crear directorio de salida
                class_dir = output_dir / f"{expression_type}_class{label}"
                class_dir.mkdir(parents=True, exist_ok=True)

                # Guardar
                output_file = class_dir / f"{participant}_seq{i:04d}.npy"
                np.save(output_file, seq_final)

                processed += 1

            except Exception as e:
                logger.error(f"Error procesando secuencia {i}: {e}")
                failed += 1

        logger.info(
            f"Procesado {participant}_{expression_type}: "
            f"{processed} exitosos, {failed} fallidos"
        )

        return processed, failed


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Procesa expresiones faciales de LIBRAS"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/Facial Expressions for Brazilian Sign Language"),
        help="Directorio con los datos raw"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/facial_expressions"),
        help="Directorio de salida"
    )

    parser.add_argument(
        "--target-length",
        type=int,
        default=100,
        help="Longitud objetivo de las secuencias temporales"
    )

    args = parser.parse_args()

    # Verificar directorio de entrada
    if not args.data_dir.exists():
        logger.error(f"Directorio de datos no encontrado: {args.data_dir}")
        return 1

    # Crear directorio de salida
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Crear preprocesador
    preprocessor = FacialExpressionsPreprocessor(
        target_length=args.target_length
    )

    # Procesar todos los participantes y tipos de expresión
    total_processed = 0
    total_failed = 0

    participants = ['a', 'b']

    logger.info("Iniciando procesamiento...")

    for participant in tqdm(participants, desc="Participantes"):
        for expr_type in tqdm(
            preprocessor.expression_types,
            desc=f"Expresiones ({participant})",
            leave=False
        ):
            processed, failed = preprocessor.process_expression_type(
                data_dir=args.data_dir,
                participant=participant,
                expression_type=expr_type,
                output_dir=args.output_dir
            )

            total_processed += processed
            total_failed += failed

    logger.info(
        f"\nProcesamiento completado: "
        f"{total_processed} exitosos, {total_failed} fallidos"
    )

    # Mostrar estadísticas
    logger.info("\nEstadísticas del dataset procesado:")
    for class_dir in sorted(args.output_dir.glob("*")):
        if class_dir.is_dir():
            n_samples = len(list(class_dir.glob("*.npy")))
            logger.info(f"  {class_dir.name}: {n_samples} muestras")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

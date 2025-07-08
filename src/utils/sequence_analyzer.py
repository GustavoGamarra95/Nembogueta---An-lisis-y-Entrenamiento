"""Módulo para análisis de secuencias de landmarks."""
import logging
import os
from typing import Any, Dict

import numpy as np
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Directorio de secuencias procesadas desde .env
SEQUENCE_DIR = os.getenv(
    "DATA_PROCESSED_DIR", "data/processed_lsp_letter_sequences"
)


def analyze(sequence: np.ndarray) -> Dict[str, Any]:
    """
    Analiza una secuencia de landmarks para extraer características.

    Args:
        sequence: Array numpy con la secuencia de landmarks

    Returns:
        Dict con estadísticas y características de la secuencia
    """
    try:
        if sequence.size == 0:
            return {}

        stats = {
            "num_frames": len(sequence),
            "mean": np.mean(sequence, axis=0).tolist(),
            "std": np.std(sequence, axis=0).tolist(),
            "max": np.max(sequence, axis=0).tolist(),
            "min": np.min(sequence, axis=0).tolist(),
        }

        return stats
    except Exception as e:
        logger.error(f"Error al analizar secuencia: {e}")
        return {}

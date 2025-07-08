import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def analyze(sequence: np.ndarray) -> Dict[str, Any]:
    """
    Analiza una secuencia de landmarks para extraer características relevantes.

    Args:
        sequence: Array numpy con la secuencia de landmarks

    Returns:
        Dict con estadísticas y características de la secuencia
    """
    try:
        if sequence.size == 0:
            return {}

        stats = {
            'num_frames': len(sequence),
            'mean': np.mean(sequence, axis=0).tolist(),
            'std': np.std(sequence, axis=0).tolist(),
            'max': np.max(sequence, axis=0).tolist(),
            'min': np.min(sequence, axis=0).tolist()
        }

        return stats
    except Exception as e:
        logger.error(f"Error al analizar secuencia: {e}")
        return {}
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ProcessedSequence:
    def __init__(
            self,
            sequence: np.ndarray,
            label: str,
            metadata: Dict[str, Any]
    ):
        """
        Constructor de ProcessedSequence.
        """
        self.sequence = sequence
        self.label = label
        self.metadata = metadata

    def validate(self) -> bool:
        """
        Valida que la secuencia tenga el formato correcto.
        """
        try:
            # Validar secuencia no vacía
            if self.sequence.size == 0:
                logger.warning("Secuencia vacía")
                return False

            # Validar dimensiones
            if len(self.sequence.shape) != 2:
                logger.warning(
                    f"Forma de secuencia incorrecta: {self.sequence.shape}"
                )
                return False

            # Validar etiqueta
            if not self.label or not isinstance(self.label, str):
                logger.warning("Etiqueta inválida")
                return False

            # Validar metadata
            if not self.metadata or not isinstance(self.metadata, dict):
                logger.warning("Metadata inválida")
                return False

            required_fields = {"original_video", "num_frames", "shape"}
            if not all(field in self.metadata for field in required_fields):
                logger.warning("Faltan campos requeridos en metadata")
                return False

            return True

        except Exception as e:
            logger.error(f"Error en validación: {e}")
            return False

    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte la secuencia a diccionario.
        """
        return {
            'sequence': self.sequence.tolist(),
            'label': self.label,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessedSequence':
        """
        Crea una secuencia desde un diccionario.
        """
        return cls(
            sequence=np.array(data['sequence']),
            label=data['label'],
            metadata=data['metadata']
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Calcula estadísticas de la secuencia.
        """
        if not self.validate():
            return {}

        return {
            'num_frames': len(self.sequence),
            'num_features': self.sequence.shape[1],
            'mean': np.mean(self.sequence, axis=0).tolist(),
            'std': np.std(self.sequence, axis=0).tolist(),
            'max': np.max(self.sequence, axis=0).tolist(),
            'min': np.min(self.sequence, axis=0).tolist()
        }

    def __str__(self) -> str:
        """String representation."""
        return (
            f"ProcessedSequence(label={self.label}, "
            f"frames={len(self.sequence)})"
        )

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"ProcessedSequence(label='{self.label}', "
            f"shape={self.sequence.shape}, "
            f"metadata={self.metadata})"
        )

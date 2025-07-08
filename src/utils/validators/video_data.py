from pathlib import Path
from typing import List
import numpy as np

class VideoData:
    def __init__(self, path: Path, frames: List[np.ndarray], label: str, duration: float):
        self.path = path
        self.frames = frames
        self.label = label
        self.duration = duration

    def validate(self) -> bool:
        """
        Valida que los datos del video sean correctos.
        Returns:
            bool: True si los datos son válidos, False en caso contrario
        """
        try:
            if not self.frames:
                return False

            shape = self.frames[0].shape
            if not all(frame.shape == shape for frame in self.frames):
                return False

            if self.duration <= 0:
                return False

            if not self.label:
                return False

            return True
        except Exception:
            return False
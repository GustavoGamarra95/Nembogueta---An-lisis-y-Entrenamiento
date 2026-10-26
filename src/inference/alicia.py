"""Núcleo de inferencia de Alicia para la API.

Recibe landmarks de MediaPipe Hands calculados sobre la imagen ORIGINAL de
la cámara (sin espejar, como LSPy-Landmarks v1) y reproduce la convención
del entrenamiento de Alicia v5/v6:

1. Geometría selfie: se espeja x (x → 1 − x) y se intercambia la etiqueta
   de mano, que es lo que MediaPipe devolvería sobre la imagen espejada.
2. Slot de la mano como librai_alphabet_v4 (convención "librai").
3. 120 features por frame (feature_engineering.engineer_frame_features).
4. Canonicalización palma/dorso por mayoría en la ventana de frames.
5. Probabilidades restringidas a las clases de la lengua elegida.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.preprocessing.feature_engineering import engineer_frame_features

RAIZ = Path(__file__).resolve().parents[2]
MODELO_POR_DEFECTO = RAIZ / "data/models/alicia-v6/estatico"

# Espejo geométrico de las 120 features (ver mirror_features en
# scripts/train/train_alphabet_robust.py, validado contra imágenes reales)
_SIGNOS_MANO = np.ones(60, dtype=np.float32)
_SIGNOS_MANO[21:51:2] = -1.0
_SIGNOS_MANO[[56, 57, 59]] = -1.0
SIGNOS_ESPEJO = np.concatenate([_SIGNOS_MANO, _SIGNOS_MANO])


@dataclass
class Mano:
    """Una mano detectada: 21 puntos (x, y, z) y etiqueta de MediaPipe."""

    puntos: Sequence[Sequence[float]]
    lateralidad: str  # "Left" o "Right", tal como la reporta MediaPipe


def espejar_features(x: np.ndarray) -> np.ndarray:
    x = x * SIGNOS_ESPEJO
    return np.concatenate([x[..., 60:], x[..., :60]], axis=-1)


def canonicalizar(feats: np.ndarray) -> np.ndarray:
    """Espeja los frames que cayeron en el slot minoritario de la ventana."""
    izquierda = np.abs(feats[:, 60:]).sum(1) > 0
    otra = izquierda != (izquierda.mean() > 0.5)
    salida = feats.copy()
    if otra.any():
        salida[otra] = espejar_features(feats[otra])
    return salida


def vector_126(manos: List[Mano]) -> np.ndarray:
    """Mano más alta en geometría selfie y slot de la convención librai."""
    raw = np.zeros(126, dtype=np.float32)
    if not manos:
        return raw
    mano = min(manos, key=lambda m: float(np.mean([p[1] for p in m.puntos])))
    pts = np.asarray(mano.puntos, dtype=np.float32).copy()
    pts[:, 0] = 1.0 - pts[:, 0]
    # En la imagen espejada MediaPipe vería la otra etiqueta; con la
    # convención librai, "Right" en la imagen espejada va al slot [63:126]
    etiqueta_espejada = "Right" if mano.lateralidad == "Left" else "Left"
    offset = 63 if etiqueta_espejada == "Right" else 0
    fin = offset + 63
    raw[offset:fin] = pts.reshape(-1)
    return raw


def tramo_sostenido(feats: np.ndarray, min_frames: int = 5) -> np.ndarray:
    """Tramo más largo con la forma de la mano estable (velocidad de la
    configuración bajo la mediana): la letra sostenida, sin la llegada ni el
    reposo de la mano. Es el tramo con el que se validó Alicia en el
    signario SLS."""
    if len(feats) < min_frames:
        return feats
    izquierda = np.abs(feats[:, 60:]).sum(1) > 0
    activa = np.where(izquierda[:, None], feats[:, 60:], feats[:, :60])
    vel = np.r_[0, np.linalg.norm(np.diff(activa[:, :56], axis=0), axis=1)]
    quieto = vel <= np.median(vel)
    mejor, actual = (0, 0), None
    for i, ok in enumerate(quieto):
        if ok and actual is None:
            actual = i
        if (not ok or i == len(quieto) - 1) and actual is not None:
            fin = i + 1 if ok else i
            if fin - actual > mejor[1] - mejor[0]:
                mejor = (actual, fin)
            actual = None
    if mejor[1] - mejor[0] < min_frames:
        return feats
    ini, fin = mejor
    return feats[ini:fin]


class Alicia:
    """Modelo estático multilengua de Alicia (v6 por defecto)."""

    def __init__(self, ruta: Optional[Path] = None):
        import tensorflow as tf

        ruta = Path(ruta or os.environ.get("ALICIA_MODEL_DIR", ""))
        if not ruta.name.startswith("run_"):
            base = ruta if ruta.name else MODELO_POR_DEFECTO
            ruta = sorted(base.glob("run_*"))[-1]
        self.ruta = ruta
        self.modelo = tf.keras.models.load_model(
            str(ruta / "best_model.keras")
        )
        self.media = np.load(ruta / "norm_mean.npy")
        self.desvio = np.load(ruta / "norm_std.npy")
        self.clases = [str(c) for c in np.load(ruta / "label_classes.npy")]
        self.lenguas: Dict[str, List[int]] = {}
        for i, c in enumerate(self.clases):
            self.lenguas.setdefault(c.split(":")[0], []).append(i)

    def features(self, frames: List[List[Mano]]) -> np.ndarray:
        feats = np.stack(
            [engineer_frame_features(vector_126(f)) for f in frames]
        )
        con_mano = np.abs(feats).sum(1) > 0
        if not con_mano.any():
            return feats[:0]
        return canonicalizar(feats[con_mano])

    def predecir(
        self,
        frames: List[List[Mano]],
        lengua: str,
        top_k: int = 3,
        sostenido: bool = False,
    ) -> Dict:
        """`sostenido=True` usa solo el tramo estable (secuencias completas de
        una letra); en tiempo real se usa la ventana entera."""
        if lengua not in self.lenguas:
            raise ValueError(f"Lengua no disponible: {lengua}")
        feats = self.features(frames)
        if sostenido:
            feats = tramo_sostenido(feats)
        if len(feats) == 0:
            return {
                "letra": None,
                "confianza": 0.0,
                "top": [],
                "frames_con_mano": 0,
            }
        p = self.modelo.predict((feats - self.media) / self.desvio, verbose=0)
        idx = self.lenguas[lengua]
        q = p[:, idx] / p[:, idx].sum(1, keepdims=True)
        media = q.mean(0)
        orden = np.argsort(media)[::-1][:top_k]
        top = [
            {
                "letra": self.clases[idx[j]].split(":")[1],
                "probabilidad": round(float(media[j]), 4),
            }
            for j in orden
        ]
        return {
            "letra": top[0]["letra"],
            "confianza": top[0]["probabilidad"],
            "top": top,
            "frames_con_mano": int(len(feats)),
        }

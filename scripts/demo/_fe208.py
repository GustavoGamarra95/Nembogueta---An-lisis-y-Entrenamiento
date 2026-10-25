"""
Feature engineering para landmarks de MediaPipe.

Reemplaza coordenadas crudas (x,y,z) por features geométricos invariantes
a la posición y escala del signer, inspirado en el proyecto Omdena LIBRAS:
https://github.com/OmdenaAI/SaoPauloBrazilChapter_BrazilianSignLanguage

Features por mano (84 estáticos + 20 de movimiento = 104 por mano):
    - Posiciones normalizadas x,y:  21 landmarks × 2 = 42
    - Distancias entre pares clave: 12 pares         = 12
    - Ángulos como (sin, cos):      15 tripletes × 2  = 30
    - Motion deltas (frame a frame):10 landmarks × 2  = 20

Total para 2 manos: 104 × 2 = 208 features por frame.
"""

import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Índices de landmarks MediaPipe Hands
# ---------------------------------------------------------------------------

# Pares de landmarks para calcular distancias (12 pares)
HAND_DISTANCE_PAIRS: list[Tuple[int, int]] = [
    (4, 8),    # thumb_tip  ↔ index_tip
    (4, 12),   # thumb_tip  ↔ middle_tip
    (4, 16),   # thumb_tip  ↔ ring_tip
    (4, 20),   # thumb_tip  ↔ pinky_tip
    (8, 12),   # index_tip  ↔ middle_tip
    (12, 16),  # middle_tip ↔ ring_tip
    (16, 20),  # ring_tip   ↔ pinky_tip
    (8, 16),   # index_tip  ↔ ring_tip
    (8, 20),   # index_tip  ↔ pinky_tip
    (5, 9),    # index_base ↔ middle_base
    (9, 13),   # middle_base ↔ ring_base
    (13, 17),  # ring_base  ↔ pinky_base
]

# Tripletes de landmarks para calcular ángulos (15 tripletes)
HAND_ANGLE_TRIPLETS: list[Tuple[int, int, int]] = [
    (0, 1, 2),    # thumb_base
    (1, 2, 3),    # thumb_bend
    (2, 3, 4),    # thumb_tip
    (5, 6, 7),    # index_bend
    (6, 7, 8),    # index_tip
    (9, 10, 11),  # middle_bend
    (10, 11, 12), # middle_tip
    (13, 14, 15), # ring_bend
    (14, 15, 16), # ring_tip
    (17, 18, 19), # pinky_bend
    (18, 19, 20), # pinky_tip
    (4, 0, 8),    # thumb_index_spread
    (8, 0, 16),   # index_ring_spread
    (0, 9, 12),   # middle_finger_angle
    (0, 17, 20),  # pinky_finger_angle
]

# Landmarks clave para motion deltas (10 por mano)
HAND_MOTION_LANDMARKS: list[int] = [0, 4, 8, 12, 16, 20, 5, 9, 13, 17]

# Dimensiones resultantes
N_DISTANCE_FEATURES = len(HAND_DISTANCE_PAIRS)       # 12
N_ANGLE_FEATURES = len(HAND_ANGLE_TRIPLETS) * 2      # 30 (sin + cos)
N_POSITION_FEATURES = 21 * 2                         # 42 (x, y)
N_MOTION_FEATURES = len(HAND_MOTION_LANDMARKS) * 2   # 20 (dx, dy)

STATIC_FEATURES_PER_HAND = (
    N_POSITION_FEATURES + N_DISTANCE_FEATURES + N_ANGLE_FEATURES
)  # 84
TOTAL_FEATURES_PER_HAND = STATIC_FEATURES_PER_HAND + N_MOTION_FEATURES  # 104
TOTAL_FEATURES_TWO_HANDS = TOTAL_FEATURES_PER_HAND * 2                  # 208


# ---------------------------------------------------------------------------
# Funciones de cómputo
# ---------------------------------------------------------------------------

def _compute_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Distancia euclídea 2D normalizada a [-1, 1]."""
    d = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    max_d = np.sqrt(2.0)  # distancia máxima posible en espacio [0,1]
    return float((d / max_d) * 2.0 - 1.0)


def _compute_angle_sincos(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> Tuple[float, float]:
    """
    Ángulo en B entre los segmentos A→B→C, devuelto como (-sin, cos).
    Representación continua preferida para redes neuronales.
    """
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float64)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float64)
    cross = ba[0] * bc[1] - ba[1] * bc[0]
    dot = np.dot(ba, bc)
    angle = np.arctan2(cross, dot)
    return (-np.sin(angle), np.cos(angle))


def _compute_motion(
    prev_pts: np.ndarray, curr_pts: np.ndarray
) -> np.ndarray:
    """Delta de movimiento normalizado a [-1, 1]."""
    return ((curr_pts - prev_pts) / 2.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Extracción por mano
# ---------------------------------------------------------------------------

def extract_static_features(hand_xyz: np.ndarray) -> np.ndarray:
    """
    Extrae features estáticos de una mano.

    Args:
        hand_xyz: Array (63,) con coordenadas x,y,z de los 21 landmarks.
                  Si es todo ceros, devuelve vector de ceros.

    Returns:
        Array (84,): posiciones(42) + distancias(12) + ángulos(30)
    """
    if np.all(hand_xyz == 0):
        return np.zeros(STATIC_FEATURES_PER_HAND, dtype=np.float32)

    pts = hand_xyz.reshape(21, 3)[:, :2]  # Solo x, y → (21, 2)

    # --- Posiciones normalizadas (42) ---
    positions = pts.flatten()

    # --- Distancias (12) ---
    distances = np.array(
        [_compute_distance(pts[a], pts[b]) for a, b in HAND_DISTANCE_PAIRS],
        dtype=np.float32,
    )

    # --- Ángulos como (sin, cos) → 30 valores ---
    angles = np.array(
        [v for a, b, c in HAND_ANGLE_TRIPLETS
         for v in _compute_angle_sincos(pts[a], pts[b], pts[c])],
        dtype=np.float32,
    )

    return np.concatenate([positions, distances, angles]).astype(np.float32)


def extract_motion_features(
    prev_hand_xyz: Optional[np.ndarray],
    curr_hand_xyz: np.ndarray,
) -> np.ndarray:
    """
    Extrae features de movimiento entre frames consecutivos.

    Args:
        prev_hand_xyz: Landmarks del frame anterior (63,) o None.
        curr_hand_xyz: Landmarks del frame actual (63,).

    Returns:
        Array (20,): deltas dx,dy para los 10 landmarks clave.
    """
    if prev_hand_xyz is None or np.all(curr_hand_xyz == 0):
        return np.zeros(N_MOTION_FEATURES, dtype=np.float32)

    curr_pts = curr_hand_xyz.reshape(21, 3)[:, :2]
    prev_pts = prev_hand_xyz.reshape(21, 3)[:, :2]

    curr_key = curr_pts[HAND_MOTION_LANDMARKS]
    prev_key = prev_pts[HAND_MOTION_LANDMARKS]

    return _compute_motion(prev_key, curr_key).flatten()


# ---------------------------------------------------------------------------
# API principal
# ---------------------------------------------------------------------------

def engineer_frame_features(
    raw_landmarks: np.ndarray,
    prev_landmarks: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convierte landmarks crudos de MediaPipe en features engineerizados.

    Args:
        raw_landmarks: Array (126,) con ambas manos (2 × 21 × 3).
                       Si solo hay una mano, la segunda mitad debe ser ceros.
        prev_landmarks: Landmarks del frame anterior (126,) para motion.
                        None en el primer frame.

    Returns:
        Array (208,): features engineerizados para 2 manos.
        Orden: [hand1_static(84) | hand1_motion(20) | hand2_static(84) | hand2_motion(20)]
    """
    hand1 = raw_landmarks[:63]
    hand2 = raw_landmarks[63:]

    prev1 = prev_landmarks[:63] if prev_landmarks is not None else None
    prev2 = prev_landmarks[63:] if prev_landmarks is not None else None

    return np.concatenate([
        extract_static_features(hand1),
        extract_motion_features(prev1, hand1),
        extract_static_features(hand2),
        extract_motion_features(prev2, hand2),
    ]).astype(np.float32)

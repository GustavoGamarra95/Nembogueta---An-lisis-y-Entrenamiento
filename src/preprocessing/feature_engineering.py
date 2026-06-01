import numpy as np
from typing import Optional, Tuple


# Features por mano — solo invariantes a posición, escala y rotación:
#   distances:   21  (pares clave entre landmarks)
#   angles:      30  (sin+cos de 15 tripletas, 2D — invariante a rotación)
#   finger_curl:  5  (z_tip - z_mcp normalizado, curvatura de cada dedo)
#   palm_angles:  4  (orientación de la palma: pitch y yaw en sin+cos)
# Total por mano: 60  |  Dos manos: 120

HAND_DISTANCE_PAIRS: list[Tuple[int, int]] = [
    # punta a punta
    (4,  8),   # thumb_tip  ↔ index_tip
    (4,  12),  # thumb_tip  ↔ middle_tip
    (4,  16),  # thumb_tip  ↔ ring_tip
    (4,  20),  # thumb_tip  ↔ pinky_tip
    (8,  12),  # index_tip  ↔ middle_tip
    (12, 16),  # middle_tip ↔ ring_tip
    (16, 20),  # ring_tip   ↔ pinky_tip
    (8,  20),  # index_tip  ↔ pinky_tip
    # muñeca a puntas
    (0,  4),   # wrist ↔ thumb_tip
    (0,  8),   # wrist ↔ index_tip
    (0,  12),  # wrist ↔ middle_tip
    (0,  16),  # wrist ↔ ring_tip
    (0,  20),  # wrist ↔ pinky_tip
    # muñeca a bases
    (0,  5),   # wrist ↔ index_mcp
    (0,  9),   # wrist ↔ middle_mcp
    (0,  13),  # wrist ↔ ring_mcp
    (0,  17),  # wrist ↔ pinky_mcp
    # bases entre sí
    (5,  9),   # index_mcp  ↔ middle_mcp
    (9,  13),  # middle_mcp ↔ ring_mcp
    (13, 17),  # ring_mcp   ↔ pinky_mcp
    # punta a base opuesta (captura flexión)
    (8,  5),   # index_tip  ↔ index_mcp
]

HAND_ANGLE_TRIPLETS: list[Tuple[int, int, int]] = [
    (0,  1,  2),   # thumb_base
    (1,  2,  3),   # thumb_bend
    (2,  3,  4),   # thumb_tip
    (5,  6,  7),   # index_bend
    (6,  7,  8),   # index_tip
    (9,  10, 11),  # middle_bend
    (10, 11, 12),  # middle_tip
    (13, 14, 15),  # ring_bend
    (14, 15, 16),  # ring_tip
    (17, 18, 19),  # pinky_bend
    (18, 19, 20),  # pinky_tip
    (4,  0,  8),   # thumb_index_spread
    (8,  0,  16),  # index_ring_spread
    (0,  9,  12),  # middle_finger_angle
    (0,  17, 20),  # pinky_finger_angle
]

FINGER_TIP_INDICES: list[int] = [4,  8,  12, 16, 20]
FINGER_MCP_INDICES: list[int] = [2,  5,   9, 13, 17]

N_DISTANCE_FEATURES    = len(HAND_DISTANCE_PAIRS)       # 21
N_ANGLE_FEATURES       = len(HAND_ANGLE_TRIPLETS) * 2   # 30
N_FINGER_CURL_FEATURES = len(FINGER_TIP_INDICES)        #  5
N_PALM_ANGLE_FEATURES  = 4                               #  4 (pitch+yaw en sin+cos)

STATIC_FEATURES_PER_HAND = (
    N_DISTANCE_FEATURES + N_ANGLE_FEATURES + N_FINGER_CURL_FEATURES + N_PALM_ANGLE_FEATURES
)  # 60
TOTAL_FEATURES_TWO_HANDS = STATIC_FEATURES_PER_HAND * 2   # 120

# Alias para compatibilidad con código externo que lea N_POSITION_FEATURES
N_POSITION_FEATURES    = 0
N_MOTION_FEATURES      = 0
TOTAL_FEATURES_PER_HAND = STATIC_FEATURES_PER_HAND


def _normalize_hand(hand_xyz: np.ndarray) -> Tuple[np.ndarray, float]:
    """Centra en muñeca y escala por distancia muñeca→middle_mcp."""
    pts = hand_xyz.reshape(21, 3)
    wrist     = pts[0].copy()
    hand_size = float(np.linalg.norm(pts[9] - pts[0])) + 1e-6
    pts_norm  = (pts - wrist) / hand_size
    return pts_norm, hand_size


def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Distancia 3D euclidiana (ya normalizada → escala comparable)."""
    return float(np.linalg.norm(p1[:3] - p2[:3]))


def _angle_sincos(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[float, float]:
    """Ángulo 2D en B para A→B→C, retorna (sin, cos)."""
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float64)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float64)
    angle = np.arctan2(ba[0]*bc[1] - ba[1]*bc[0], np.dot(ba, bc))
    return (float(-np.sin(angle)), float(np.cos(angle)))


def _palm_angles(pts: np.ndarray) -> np.ndarray:
    """
    Calcula el vector normal de la palma (cross product de dos vectores
    de la palma) y retorna [sin_pitch, cos_pitch, sin_yaw, cos_yaw].
    Invariante a traslación y escala, pero captura orientación 3D de la palma.
    """
    v1 = pts[5]  - pts[0]   # muñeca → index_mcp
    v2 = pts[17] - pts[0]   # muñeca → pinky_mcp
    normal = np.cross(v1[:3], v2[:3])
    norm_len = np.linalg.norm(normal) + 1e-6
    normal = normal / norm_len

    pitch = np.arctan2(normal[1], normal[2])
    yaw   = np.arctan2(normal[0], normal[2])
    return np.array([np.sin(pitch), np.cos(pitch), np.sin(yaw), np.cos(yaw)],
                    dtype=np.float32)


def extract_static_features(hand_xyz: np.ndarray) -> np.ndarray:
    if np.all(hand_xyz == 0):
        return np.zeros(STATIC_FEATURES_PER_HAND, dtype=np.float32)

    pts, _ = _normalize_hand(hand_xyz)

    distances = np.array(
        [_distance(pts[a], pts[b]) for a, b in HAND_DISTANCE_PAIRS],
        dtype=np.float32,
    )

    angles = np.array(
        [v for a, b, c in HAND_ANGLE_TRIPLETS
         for v in _angle_sincos(pts[a], pts[b], pts[c])],
        dtype=np.float32,
    )

    # finger_curl normalizado por hand_size (ya normalizado en pts)
    finger_curl = np.array(
        [pts[tip, 2] - pts[mcp, 2]
         for tip, mcp in zip(FINGER_TIP_INDICES, FINGER_MCP_INDICES)],
        dtype=np.float32,
    )

    palm = _palm_angles(pts)

    return np.concatenate([distances, angles, finger_curl, palm]).astype(np.float32)


def extract_motion_features(
    prev_hand_xyz: Optional[np.ndarray],
    curr_hand_xyz: np.ndarray,
) -> np.ndarray:
    # Motion eliminado — era ruidoso en condiciones reales.
    return np.zeros(0, dtype=np.float32)


def engineer_frame_features(
    raw_landmarks: np.ndarray,
    prev_landmarks: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    raw_landmarks: (126,) = 2 manos × 21 pts × 3 coords.
    Retorna vector (120,) = 60 features × 2 manos.
    Invariante a posición, escala y rotación de la mano en el frame.
    """
    hand1 = raw_landmarks[:63]
    hand2 = raw_landmarks[63:]
    return np.concatenate([
        extract_static_features(hand1),
        extract_static_features(hand2),
    ]).astype(np.float32)

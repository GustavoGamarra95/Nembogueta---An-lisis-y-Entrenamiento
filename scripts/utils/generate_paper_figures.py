"""
Gera figuras suplementares para o artigo científico:
  1. Esquema dos 21 landmarks do MediaPipe Hands com índices e nomes
  2. Diagrama do pipeline do MediaPipe (RGB → palm → landmarks → features)
  3. Visualização das 4 categorias de features sobre uma mão sintética
  4. Galeria de letras do alfabeto LIBRAS (referências visuais)

Uso (host ou contêiner):
    python scripts/utils/generate_paper_figures.py \\
        --ref-dir src/LibrAI/test --output-dir data/analysis/paper_figures
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
import numpy as np


# ---------------------------------------------------------------- #
# Layout canônico dos 21 landmarks (mão direita vista de frente)
# Coordenadas relativas em [-1,1] aproximando uma palma aberta natural
# ---------------------------------------------------------------- #
LANDMARK_NAMES = [
    "WRIST",        # 0
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",   # 1-4
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",  # 5-8
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",  # 9-12
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",      # 13-16
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",  # 17-20
]

LANDMARK_POS = np.array([
    [0.0, 0.0],         # 0 WRIST
    [-0.25, 0.15], [-0.45, 0.40], [-0.55, 0.65], [-0.62, 0.90],   # thumb 1-4
    [-0.18, 0.55], [-0.20, 0.95], [-0.22, 1.20], [-0.23, 1.45],   # index 5-8
    [0.00, 0.60], [0.00, 1.05], [0.00, 1.35], [0.00, 1.60],       # middle 9-12
    [0.18, 0.55], [0.20, 0.95], [0.22, 1.20], [0.23, 1.45],       # ring 13-16
    [0.35, 0.50], [0.40, 0.80], [0.44, 1.05], [0.48, 1.30],       # pinky 17-20
])

# Conexões padrão MediaPipe
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),                # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),                # index
    (5, 9), (9, 10), (10, 11), (11, 12),           # middle
    (9, 13), (13, 14), (14, 15), (15, 16),         # ring
    (13, 17), (17, 18), (18, 19), (19, 20),        # pinky
    (0, 17),                                       # palm base
]


# ---------------------------------------------------------------- #
# Fig 1: Esquema dos 21 landmarks com índices e nomes
# ---------------------------------------------------------------- #
def fig_landmarks_schematic(out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 9))
    pts = LANDMARK_POS

    # cores por dedo
    colors = {
        "thumb": "#d62728", "index": "#1f77b4", "middle": "#2ca02c",
        "ring": "#ff7f0e", "pinky": "#9467bd",
    }
    finger_of = {
        0: "wrist",
        1: "thumb", 2: "thumb", 3: "thumb", 4: "thumb",
        5: "index", 6: "index", 7: "index", 8: "index",
        9: "middle", 10: "middle", 11: "middle", 12: "middle",
        13: "ring", 14: "ring", 15: "ring", 16: "ring",
        17: "pinky", 18: "pinky", 19: "pinky", 20: "pinky",
    }

    # conexões
    for a, b in HAND_CONNECTIONS:
        ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]],
                color="grey", linewidth=2, alpha=0.6, zorder=1)

    # pontos numerados
    for i, (x, y) in enumerate(pts):
        c = colors.get(finger_of[i], "black") if i > 0 else "black"
        ax.scatter(x, y, s=380, color=c, edgecolor="white",
                   linewidth=2, zorder=3)
        ax.text(x, y, str(i), ha="center", va="center",
                fontsize=10, color="white", weight="bold", zorder=4)

    # rótulos dos pontos-chave
    key_labels = {
        0: ("WRIST (pulso)", (-0.35, -0.05)),
        4: ("THUMB_TIP", (-0.92, 0.95)),
        8: ("INDEX_TIP", (-0.52, 1.55)),
        12: ("MIDDLE_TIP", (-0.18, 1.70)),
        16: ("RING_TIP", (0.30, 1.55)),
        20: ("PINKY_TIP", (0.65, 1.40)),
        2: ("THUMB_MCP", (-0.80, 0.40)),
        5: ("INDEX_MCP", (-0.45, 0.50)),
        9: ("MIDDLE_MCP", (0.10, 0.45)),
        13: ("RING_MCP", (0.40, 0.50)),
        17: ("PINKY_MCP", (0.55, 0.50)),
    }
    for idx, (label, pos) in key_labels.items():
        ax.text(pos[0], pos[1], label, fontsize=8, color="black",
                ha="left", style="italic")

    # legenda de cores
    handles = [
        mpatches.Patch(color=colors["thumb"], label="Polegar (1–4)"),
        mpatches.Patch(color=colors["index"], label="Indicador (5–8)"),
        mpatches.Patch(color=colors["middle"], label="Médio (9–12)"),
        mpatches.Patch(color=colors["ring"], label="Anular (13–16)"),
        mpatches.Patch(color=colors["pinky"], label="Mínimo (17–20)"),
        mpatches.Patch(color="black", label="Pulso (0)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9,
              frameon=True, fancybox=True)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.25, 1.85)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "21 landmarks da mão direita extraídos pelo MediaPipe Hands\n"
        "(numeração canônica: pulso=0, ponta dos dedos=4,8,12,16,20)",
        fontsize=12,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path.name}")


# ---------------------------------------------------------------- #
# Fig 2: Pipeline do MediaPipe Hands + engenharia de features
# ---------------------------------------------------------------- #
def fig_mediapipe_pipeline(out_path: Path):
    fig, ax = plt.subplots(figsize=(15, 6))

    stages = [
        {
            "x": 0.5, "title": "Frame RGB",
            "body": "Entrada\nCâmara webcam\n(B, G, R)\n720×1280 px",
            "color": "#e0e0e0",
        },
        {
            "x": 3.0, "title": "Palm Detection",
            "body": "BlazePalm SSD\nlocaliza palma\n(bounding box)\n+ orientação",
            "color": "#ffe5b4",
        },
        {
            "x": 5.5, "title": "ROI Crop & Align",
            "body": "Recorte da\nregião + rotação\np/ alinhar pulso↔dedos",
            "color": "#ffe5b4",
        },
        {
            "x": 8.0, "title": "Landmark Regression",
            "body": "MLP convolucional\nprediz 21 pontos\n(x, y, z)",
            "color": "#cce5ff",
        },
        {
            "x": 10.5, "title": "21 Landmarks 3D",
            "body": "Saída MediaPipe\n21 × (x, y, z)\n= 63 valores\npor mão",
            "color": "#cce5ff",
        },
        {
            "x": 13.5, "title": "Feature Engineering",
            "body": "Normalização\n+ distâncias (21)\n+ ângulos (30)\n"
                    "+ curl (5)\n+ palma (4)\n= 60 / mão",
            "color": "#d4edda",
        },
    ]

    w, h = 1.9, 2.4
    for s in stages:
        bbox = FancyBboxPatch(
            (s["x"] - w / 2, 1.0 - h / 2), w, h,
            boxstyle="round,pad=0.05",
            facecolor=s["color"], edgecolor="black", linewidth=1,
        )
        ax.add_patch(bbox)
        ax.text(s["x"], 1.0 + h / 2 - 0.25, s["title"],
                ha="center", va="top", fontsize=10, weight="bold")
        ax.text(s["x"], 1.0 - 0.15, s["body"],
                ha="center", va="center", fontsize=8.5)

    # setas
    for i in range(len(stages) - 1):
        x1 = stages[i]["x"] + w / 2
        x2 = stages[i + 1]["x"] - w / 2
        arrow = FancyArrowPatch(
            (x1, 1.0), (x2, 1.0), arrowstyle="->", mutation_scale=18,
            color="black", linewidth=1.2,
        )
        ax.add_patch(arrow)

    # Bloco final saída
    ax.annotate(
        "Saída: vetor de 120 atributos\n(60 × 2 mãos)\ninvariante a "
        "posição, escala e rotação 2D",
        xy=(stages[-1]["x"], 1.0 - h / 2), xytext=(13.5, -0.6),
        fontsize=9, ha="center",
        bbox=dict(boxstyle="round", facecolor="#fff3cd",
                  edgecolor="black"),
        arrowprops=dict(arrowstyle="->", color="black"),
    )

    # Modelo aprendido
    ax.annotate(
        "MediaPipe (Zhang et al., 2020):\nestágios pré-treinados em grandes\n"
        "datasets de mão, executam em GPU/CPU\ncom latência ~5–15 ms/frame",
        xy=(8.0, 1.0 + h / 2), xytext=(8.0, 3.0),
        fontsize=8.5, ha="center", style="italic", color="#444",
        arrowprops=dict(arrowstyle="-", color="grey", linestyle="--"),
    )

    ax.set_xlim(-0.5, 16)
    ax.set_ylim(-1.2, 3.6)
    ax.axis("off")
    ax.set_title(
        "Pipeline do MediaPipe Hands para extração de landmarks "
        "+ engenharia de características biomecânicas",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path.name}")


# ---------------------------------------------------------------- #
# Fig 3: Visualização das 4 categorias de features
# ---------------------------------------------------------------- #
def fig_feature_categories(out_path: Path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    pts = LANDMARK_POS

    def draw_hand(ax, alpha=0.3):
        for a, b in HAND_CONNECTIONS:
            ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]],
                    color="grey", linewidth=1.5, alpha=alpha, zorder=1)
        ax.scatter(pts[:, 0], pts[:, 1], s=40, color="grey",
                   alpha=alpha + 0.3, zorder=2)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-0.2, 1.85)
        ax.set_aspect("equal")
        ax.axis("off")

    # (a) Distâncias
    ax = axes[0]
    draw_hand(ax)
    distance_pairs = [(4, 8), (4, 12), (4, 16), (4, 20),
                      (8, 12), (12, 16), (16, 20),
                      (0, 4), (0, 8), (0, 12), (0, 16), (0, 20)]
    for a, b in distance_pairs:
        ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]],
                color="#1f77b4", linewidth=1.4, alpha=0.7, zorder=2)
    ax.set_title("(a) 21 distâncias inter-articulares\n"
                 "Ex: polegar↔índex, pulso↔ponta dos dedos",
                 fontsize=10)

    # (b) Ângulos articulares
    ax = axes[1]
    draw_hand(ax)
    angle_triplets = [(1, 2, 3), (2, 3, 4),
                      (5, 6, 7), (6, 7, 8),
                      (9, 10, 11), (10, 11, 12),
                      (13, 14, 15), (14, 15, 16),
                      (17, 18, 19), (18, 19, 20)]
    for a, b, c in angle_triplets:
        ax.plot([pts[a, 0], pts[b, 0], pts[c, 0]],
                [pts[a, 1], pts[b, 1], pts[c, 1]],
                color="#d62728", linewidth=1.3, alpha=0.6, zorder=2)
        ax.scatter(pts[b, 0], pts[b, 1], s=80, color="#d62728",
                   alpha=0.5, zorder=3)
    ax.set_title("(b) 15 ângulos articulares (sin+cos = 30)\n"
                 "Ex: flexão da falange média de cada dedo",
                 fontsize=10)

    # (c) Curvatura dos dedos (z)
    ax = axes[2]
    draw_hand(ax)
    finger_curl = [(4, 2), (8, 5), (12, 9), (16, 13), (20, 17)]
    for tip, mcp in finger_curl:
        ax.annotate(
            "", xy=(pts[tip, 0], pts[tip, 1] + 0.05),
            xytext=(pts[mcp, 0], pts[mcp, 1] - 0.05),
            arrowprops=dict(arrowstyle="->", color="#2ca02c",
                            lw=2, alpha=0.7),
        )
    ax.text(0, -0.10, r"$c_k = \tilde{p}_{tip,z} - \tilde{p}_{mcp,z}$",
            ha="center", fontsize=11, color="#2ca02c")
    ax.set_title("(c) 5 atributos de curvatura\n"
                 "Diferença z entre ponta e base de cada dedo",
                 fontsize=10)

    # (d) Orientação da palma
    ax = axes[3]
    draw_hand(ax)
    # plano da palma
    palm_pts = pts[[0, 5, 17]]
    palm_polygon = plt.Polygon(palm_pts, color="#9467bd",
                               alpha=0.25, zorder=2)
    ax.add_patch(palm_polygon)
    # vetor normal
    centroid = palm_pts.mean(axis=0)
    ax.annotate(
        "", xy=(centroid[0] - 0.35, centroid[1] + 0.55),
        xytext=(centroid[0], centroid[1]),
        arrowprops=dict(arrowstyle="->", color="#9467bd",
                        lw=2.5, alpha=0.9),
    )
    ax.text(centroid[0] - 0.50, centroid[1] + 0.65,
            r"$\vec{n}$", fontsize=14, color="#9467bd")
    ax.text(0, -0.10,
            r"$\phi_{pitch}, \phi_{yaw}$ (sin+cos)",
            ha="center", fontsize=11, color="#9467bd")
    ax.set_title("(d) 4 atributos de orientação da palma\n"
                 "Vetor normal decomposto em pitch e yaw",
                 fontsize=10)

    fig.suptitle(
        "Categorias dos 60 atributos extraídos por mão "
        "(× 2 mãos = 120 atributos invariantes)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path.name}")


# ---------------------------------------------------------------- #
# Fig 4: Galeria de letras do alfabeto LIBRAS
# ---------------------------------------------------------------- #
def fig_alphabet_gallery(ref_dir: Path, out_path: Path):
    """Mostra uma imagem representativa de cada letra do alfabeto."""
    letters = list("ABCDEFGILMNOPQRSTUVWY") + ["Ç"]
    cols = 6
    rows = (len(letters) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, rows * 1.9))
    axes = axes.flatten()

    import cv2

    for i, letter in enumerate(letters):
        ax = axes[i]
        d = ref_dir / letter
        if d.exists():
            pngs = sorted(d.glob("*.png"))
            if pngs:
                img = cv2.imread(str(pngs[0]))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, "—", ha="center", va="center",
                            transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "(sem ref.)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)
        else:
            ax.text(0.5, 0.5, "(n/d)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
        ax.set_title(letter, fontsize=14, weight="bold")
        ax.axis("off")

    # esconder eixos restantes
    for j in range(len(letters), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Vocabulário-alvo da validação técnica — 22 letras do alfabeto "
        "LIBRAS\n(referências visuais do dataset LibrAI)",
        fontsize=12, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path.name}")


# ---------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref-dir", type=Path,
                   default=Path("/app/src/LibrAI/test"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("/data/analysis/paper_figures"))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Gerando figuras suplementares do paper ===")
    fig_landmarks_schematic(args.output_dir / "landmarks_schematic.png")
    fig_mediapipe_pipeline(args.output_dir / "mediapipe_pipeline.png")
    fig_feature_categories(args.output_dir / "feature_categories.png")
    fig_alphabet_gallery(args.ref_dir,
                         args.output_dir / "alphabet_gallery.png")
    print(f"\nArtefatos em: {args.output_dir}")


if __name__ == "__main__":
    main()

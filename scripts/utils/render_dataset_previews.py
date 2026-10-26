"""Imágenes de vista previa de LSPy-Landmarks v1 con la predicción de Alicia.

Dibuja el esqueleto de la mano (solo landmarks: sin caras ni frames de
video) del tramo sostenido de cada muestra y anota la predicción de Alicia
v6 (alfabeto y achegety) o la traducción del diccionario LSP (lexico).

Uso:
    python scripts/utils/render_dataset_previews.py \\
        --dataset data/release/lspy_landmarks_v1 --salida previews/
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inference.alicia import Alicia, Mano  # noqa: E402
from src.inference.diccionario import Diccionario  # noqa: E402

CONEXIONES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]
FONDO, TINTA, TENUE = "#141413", "#f2f0ea", "#a29f96"
MANO_1, MANO_2, BIEN, MAL = "#f97316", "#60a5fa", "#4ade80", "#f87171"


def cargar(dataset, fila):
    z = np.load(dataset / fila["archivo"])
    frames = []
    for t in range(len(z["landmarks"])):
        manos = []
        for k in range(2):
            pts = z["landmarks"][t, k]
            if not np.isnan(pts).any() and z["handedness"][t, k]:
                manos.append(Mano(pts.tolist(), str(z["handedness"][t, k])))
        frames.append(manos)
    return z["landmarks"], frames


def frame_central(landmarks, frames):
    """Índice del frame con mano más cercano al centro del tramo."""
    con_mano = [i for i, f in enumerate(frames) if f]
    if not con_mano:
        return None
    medio = (con_mano[0] + con_mano[-1]) / 2
    return min(con_mano, key=lambda i: abs(i - medio))


def mano_activa(landmarks, t):
    """Slot de la mano más alta del frame (la que forma la letra)."""
    slots = [k for k in range(2) if not np.isnan(landmarks[t, k]).any()]
    return min(slots, key=lambda k: landmarks[t, k, :, 1].mean())


def manos_juntas(landmarks, t):
    """Las dos manos detectadas y cerca (seña a dos manos): dibujar ambas."""
    if np.isnan(landmarks[t]).any():
        return False
    centros = landmarks[t, :, :, :2].mean(1)
    tamano = np.ptp(landmarks[t, mano_activa(landmarks, t), :, :2], 0).max()
    return np.linalg.norm(centros[0] - centros[1]) < 2.5 * tamano


def dibujar_mano(ax, landmarks, t, solo_activa=False):
    activa = mano_activa(landmarks, t)
    for k, color in enumerate((MANO_1, MANO_2)):
        pts = landmarks[t, k]
        if np.isnan(pts).any() or (solo_activa and k != activa):
            continue
        for a, b in CONEXIONES:
            ax.plot(
                pts[[a, b], 0],
                pts[[a, b], 1],
                color=color,
                lw=2.2,
                solid_capstyle="round",
            )
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=11,
            color=TINTA,
            zorder=3,
            edgecolors=color,
            linewidths=0.8,
        )
    validos = (
        landmarks[t, activa]
        if solo_activa
        else landmarks[t][~np.isnan(landmarks[t]).any(-1)]
    )
    cx, cy = validos[:, 0].mean(), validos[:, 1].mean()
    lado = max(np.ptp(validos[:, 0]), np.ptp(validos[:, 1])) * 0.62 + 0.02
    ax.set_xlim(cx - lado, cx + lado)
    ax.set_ylim(cy + lado, cy - lado)
    ax.set_aspect("equal")
    ax.axis("off")


def lamina(muestras, titulo, salida, columnas=6, alto=2.9):
    filas = int(np.ceil(len(muestras) / columnas))
    fig, ejes = plt.subplots(
        filas,
        columnas,
        figsize=(columnas * 2.4, filas * alto + 1.0),
        facecolor=FONDO,
    )
    for ax in np.ravel(ejes):
        ax.set_facecolor(FONDO)
        ax.axis("off")
    for ax, m in zip(np.ravel(ejes), muestras):
        dibujar_mano(ax, m["landmarks"], m["t"], m.get("solo_activa", True))
        ax.set_title(
            m["titulo"].replace("G\u0303", "G̃ (nasal)"),
            color=TINTA,
            fontsize=17,
            fontweight="bold",
            pad=4,
        )
        ax.text(
            0.5,
            -0.04,
            m["pie"],
            color=m.get("color", TENUE),
            fontsize=9.5,
            ha="center",
            va="top",
            transform=ax.transAxes,
            linespacing=1.35,
        )
    fig.suptitle(
        titulo,
        color=TINTA,
        fontsize=14,
        x=0.02,
        ha="left",
        y=0.995,
        fontweight="bold",
        wrap=True,
    )
    fig.text(
        0.98,
        0.005,
        "LSPy-Landmarks v1 · Ñemongueta / Alicia v6",
        color=TENUE,
        fontsize=8,
        ha="right",
    )
    fig.tight_layout(rect=(0, 0.01, 1, 0.97))
    fig.savefig(salida, dpi=130, facecolor=FONDO)
    plt.close(fig)
    print("→", salida)


def formas(dataset, subconjunto):
    """Solo la forma de la mano y la etiqueta, sin predicción."""
    with open(dataset / subconjunto / "index.csv", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    muestras = []
    for fila in filas:
        lm, frames = cargar(dataset, fila)
        t = frame_central(lm, frames)
        if t is not None:
            muestras.append(
                {
                    "landmarks": lm,
                    "t": t,
                    "titulo": fila["etiqueta"],
                    "pie": "",
                }
            )
    return muestras


def letras(dataset, alicia, subconjunto, lengua, filtro, maximo):
    with open(dataset / subconjunto / "index.csv", encoding="utf-8") as f:
        filas = [r for r in csv.DictReader(f) if filtro(r)][:maximo]
    muestras, aciertos = [], 0
    for fila in filas:
        lm, frames = cargar(dataset, fila)
        t = frame_central(lm, frames)
        if t is None:
            continue
        pred = alicia.predecir(frames, lengua, top_k=2, sostenido=True)
        ok = pred["letra"] == fila["etiqueta"]
        aciertos += ok
        top = "  ".join(
            f"{p['letra']} {p['probabilidad']:.0%}" for p in pred["top"]
        )
        muestras.append(
            {
                "landmarks": lm,
                "t": t,
                "titulo": fila["etiqueta"],
                "pie": f"Alicia: {top}",
                "color": BIEN if ok else MAL,
            }
        )
    return muestras, aciertos


def lexico(dataset, dic, glosas):
    with open(dataset / "lexico/index.csv", encoding="utf-8") as f:
        filas = {r["etiqueta"]: r for r in csv.DictReader(f)}
    muestras = []
    for g in glosas:
        lm, frames = cargar(dataset, filas[g])
        t = frame_central(lm, frames)
        e = dic.buscar(g)
        pie = f"es {e['es']}\npt {e['pt']}\ngn {e['gn'] or '—'}"
        muestras.append(
            {
                "landmarks": lm,
                "t": t,
                "titulo": g,
                "pie": pie,
                "solo_activa": not manos_juntas(lm, t),
            }
        )
    return muestras


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--dataset", type=Path, default=ROOT / "data/release/lspy_landmarks_v1"
    )
    ap.add_argument("--salida", type=Path, default=None)
    args = ap.parse_args()
    salida = args.salida or args.dataset / "previews"
    salida.mkdir(parents=True, exist_ok=True)

    alicia, dic = Alicia(), Diccionario()

    muestras, ok = letras(
        args.dataset,
        alicia,
        "alfabeto",
        "LSPY",
        lambda r: r["interprete"].startswith("SLS"),
        24,
    )
    lamina(
        muestras,
        f"Alfabeto manual LSPy (Signario SLS, intérprete "
        f"externo) — Alicia v6: {ok}/{len(muestras)} correctas "
        f"(verde = acierto)",
        salida / "alfabeto_lspy_alicia.png",
    )

    lamina(
        formas(args.dataset, "achegety"),
        "Achegety — alfabeto guaraní en LSPy (forma de la mano en el "
        "tramo sostenido)",
        salida / "achegety.png",
        alto=2.6,
    )

    glosas = [
        "AGUA",
        "MÉDICO",
        "HOSPITAL",
        "FIEBRE",
        "DOLOR",
        "FAMILIA",
        "HOLA",
        "BUENOS DÍAS",
        "TELÉFONO",
        "AMIGO",
        "MAMÁ",
        "ESCUELA",
    ]
    lamina(
        lexico(args.dataset, dic, glosas),
        "Señas léxicas del Signario SLS — traducción automática (IA, sin "
        "revisión humana)",
        salida / "lexico_diccionario.png",
        alto=3.5,
    )


if __name__ == "__main__":
    main()

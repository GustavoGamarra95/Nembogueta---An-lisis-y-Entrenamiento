"""Genera el notebook de ejemplo de LSPy-Landmarks v1 para Kaggle.

Uso:
    python notebooks/kaggle/build_quickstart.py          # escribe el .ipynb
    python notebooks/kaggle/build_quickstart.py --run D  # ejecuta las celdas
                                                         # con el dataset en D
"""

import argparse
import json
from pathlib import Path

AQUI = Path(__file__).parent

CELDAS = [
    ("md", """# LSPy-Landmarks v1 — quickstart

Hand landmarks (MediaPipe Hands) of **Paraguayan Sign Language (LSPy)**:
the Spanish manual alphabet, the Guarani alphabet (*achegety*) and 1810
lexical signs. This notebook shows how to load a sample, draw it and train a
small baseline for the alphabet evaluated **across channels** (each video
channel is held out in turn), which is the honest way to measure
generalization to unseen signers."""),
    ("code", """import glob, os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", "Mean of empty slice")

# Locate the dataset root (works for any Kaggle mount point)
ROOT = os.path.dirname(os.path.dirname(
    glob.glob("/kaggle/input/**/alfabeto/index.csv", recursive=True)[0]))
alfabeto = pd.read_csv(f"{ROOT}/alfabeto/index.csv")
achegety = pd.read_csv(f"{ROOT}/achegety/index.csv")
lexico = pd.read_csv(f"{ROOT}/lexico/index.csv")
print(ROOT)
print({"alfabeto": len(alfabeto), "achegety": len(achegety),
       "lexico": len(lexico)})
alfabeto.head()"""),
    ("md", """## One sample

Each `.npz` holds up to two hands per frame: `landmarks` (T, 2, 21, 3) in
image coordinates of the **unmirrored** video (NaN when a hand is missing),
`world` (metric 3D), `handedness` and `score`."""),
    ("code", """sample = np.load(f"{ROOT}/{alfabeto.archivo[0]}")
for k in sample.files:
    print(k, sample[k].shape, sample[k].dtype)"""),
    ("code", """CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7),
               (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13),
               (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19),
               (19, 20), (0, 17)]


def active_hand(lm):
    \"\"\"(T, 21, 3): per frame, the highest detected hand (the one
    fingerspelling); NaN where no hand was detected.\"\"\"
    y = np.nanmean(lm[..., 1], axis=-1)            # (T, 2)
    y = np.where(np.isnan(y), np.inf, y)
    k = np.argmin(y, axis=1)
    return lm[np.arange(len(lm)), k]


def middle_frame(hand):
    ok = np.where(~np.isnan(hand).any(axis=(1, 2)))[0]
    return ok[len(ok) // 2] if len(ok) else None


def draw(ax, pts, title):
    for a, b in CONNECTIONS:
        ax.plot(pts[[a, b], 0], pts[[a, b], 1], color="tab:orange", lw=2)
    ax.scatter(pts[:, 0], pts[:, 1], s=10, color="k", zorder=3)
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title)


rows = alfabeto[alfabeto.interprete.str.startswith("SLS")].head(8)
fig, axes = plt.subplots(1, len(rows), figsize=(2.2 * len(rows), 2.8))
for ax, (_, r) in zip(axes, rows.iterrows()):
    hand = active_hand(np.load(f"{ROOT}/{r.archivo}")["landmarks"])
    t = middle_frame(hand)
    if t is not None:
        draw(ax, hand[t], r.etiqueta)
plt.tight_layout()"""),
    ("md", """## Baseline: alphabet, leave-one-channel-out

Features: the active hand at the middle frames, translated to the wrist and
scaled by hand size (63 values, averaged over the central third of the
sample). Some source videos are mirrored, so each training sample is also
added mirrored (x → −x). Classifier: logistic regression. Letters with
movement (J, Z, Ñ, LL, RR) are left out: one averaged frame cannot capture
them."""),
    ("code", """from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def features(lm):
    hand = active_hand(lm)
    ok = np.where(~np.isnan(hand).any(axis=(1, 2)))[0]
    if len(ok) == 0:
        return None
    mid = ok[len(ok) // 3: max(len(ok) // 3 + 1, 2 * len(ok) // 3)]
    h = hand[mid] - hand[mid, :1]                  # wrist at the origin
    h = h / np.linalg.norm(h[:, 9], axis=-1)[:, None, None]  # hand size
    return h.mean(0)                               # (21, 3)


def mirror(f):
    g = f.copy(); g[:, 0] *= -1
    return g


X, y, g = [], [], []
for _, r in alfabeto.iterrows():
    f = features(np.load(f"{ROOT}/{r.archivo}")["landmarks"])
    if f is not None:
        X.append(f); y.append(r.etiqueta); g.append(r.interprete)
X, y, g = np.array(X), np.array(y), np.array(g)

dynamic = {"J", "Z", "Ñ", "LL", "RR"}             # letters with movement
keep = ~np.isin(y, list(dynamic))
print(f"{keep.sum()} static samples, {len(set(y[keep]))} letters, "
      f"{len(set(g))} channels")"""),
    ("code", """results = {}
for channel in sorted(set(g)):
    tr, te = keep & (g != channel), keep & (g == channel)
    if te.sum() == 0:
        continue
    Xtr = np.concatenate([X[tr], np.array([mirror(f) for f in X[tr]])])
    ytr = np.concatenate([y[tr], y[tr]])
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, C=0.5))
    clf.fit(Xtr.reshape(len(Xtr), -1), ytr)
    # each test sample: best of original / mirrored (unknown mirroring)
    p1 = clf.predict_proba(X[te].reshape(te.sum(), -1))
    p2 = clf.predict_proba(
        np.array([mirror(f) for f in X[te]]).reshape(te.sum(), -1))
    pred = clf.classes_[np.maximum(p1, p2).argmax(1)]
    results[channel] = ((pred == y[te]).mean(), te.sum())

res = pd.DataFrame(results, index=["accuracy", "samples"]).T
res["accuracy"] = res["accuracy"].round(3)
print(res.sort_values("accuracy"))
print("mean accuracy across held-out channels:",
      round(res.accuracy.mean(), 3))"""),
    ("md", """This is a deliberately simple reference point: one averaged frame and a
linear model, with a single take per letter and channel. Stronger models
(engineered angles, temporal context, palm/back disambiguation) are in the
project repository:
https://github.com/GustavoGamarra95/Nembogueta---An-lisis-y-Entrenamiento

Please credit the original sources listed in the dataset README when using
this data (CC BY 4.0)."""),
]


def notebook():
    celdas = []
    for tipo, src in CELDAS:
        celda = {"cell_type": "markdown" if tipo == "md" else "code",
                 "metadata": {}, "source": src.splitlines(keepends=True)}
        if tipo == "code":
            celda.update(execution_count=None, outputs=[])
        celdas.append(celda)
    return {"cells": celdas, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, help="Ejecutar con el dataset local")
    args = ap.parse_args()
    if args.run:
        import matplotlib

        matplotlib.use("Agg")
        codigo = "\n\n".join(s for t, s in CELDAS if t == "code")
        raiz = str(args.run.resolve())
        codigo = codigo.replace(
            'glob.glob("/kaggle/input/**/alfabeto/index.csv", recursive=True)'
            "[0]", f'"{raiz}/alfabeto/index.csv"')
        exec(compile(codigo, "quickstart", "exec"), {"__name__": "__main__"})
        return
    salida = AQUI / "lspy-landmarks-quickstart.ipynb"
    salida.write_text(json.dumps(notebook(), indent=1, ensure_ascii=False))
    print("→", salida)


if __name__ == "__main__":
    main()

"""
Compara el alfabeto manual de LSC (Colombia, LSC70) con el de LIBRAS (LibrAI)
para decidir qué letras se pueden unificar al momento de la traducción.

Tres análisis:
  1. Cruce de modelos: muestras LSC70 → Alicia LIBRAS (robust-v4).
     Si la "A" de LSC se predice como "A" de LIBRAS, la configuración coincide.
  2. Centroides: distancia entre la letra LSC y la misma letra LIBRAS,
     comparada con la letra LIBRAS más cercana, sobre la mano activa (60 feats).
     También qué grupo de features explica la diferencia.
  3. Galería: imágenes LSC y LIBRAS lado a lado por letra.

Uso:
    python scripts/evaluate/compare_alphabets_lsc_libras.py \\
        --libras-images /ruta/a/src/LibrAI/test
"""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

# Grupos de la mano en feature_engineering.py (60 por mano)
FEATURE_GROUPS = {
    "distancias": slice(0, 21),
    "angulos": slice(21, 51),
    "curl_dedos": slice(51, 56),
    "orientacion_palma": slice(56, 60),
}


def load_features(root: Path):
    X, y = [], []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(d.glob("*.npy")):
            X.append(np.load(f))
            y.append(d.name)
    return np.stack(X).astype(np.float32), np.array(y)


def active_hand(X: np.ndarray) -> np.ndarray:
    """Devuelve las 60 features de la mano detectada (slot Right o Left)."""
    right_on = np.abs(X[:, :60]).sum(1) > 0
    left_on = np.abs(X[:, 60:]).sum(1) > 0
    use_right = right_on & ~left_on
    return np.where(use_right[:, None], X[:, :60], X[:, 60:])


def cross_model(model_dir: Path, X_lsc, y_lsc):
    model = tf.keras.models.load_model(str(model_dir / "best_model.keras"))
    mean = np.load(model_dir / "norm_mean.npy")
    std = np.load(model_dir / "norm_std.npy")
    classes = [str(c) for c in np.load(model_dir / "label_classes.npy")]
    probs = model.predict((X_lsc - mean) / std, verbose=0, batch_size=512)
    pred = np.array(classes)[probs.argmax(1)]

    rows = {}
    for letter in sorted(set(y_lsc)):
        m = y_lsc == letter
        counts = Counter(pred[m])
        top, top_n = counts.most_common(1)[0]
        same = counts.get(letter, 0) / m.sum() if letter in classes else None
        rows[letter] = {
            "n": int(m.sum()),
            "pred_top": top,
            "pred_top_pct": round(100 * top_n / m.sum(), 1),
            "same_letter_pct": None if same is None else round(100 * same, 1),
            "mean_prob_same": (
                None if letter not in classes
                else round(float(probs[m, classes.index(letter)].mean()), 3)
            ),
            "top3": [(k, round(100 * v / m.sum(), 1))
                     for k, v in counts.most_common(3)],
        }

    # Matriz LSC (filas) × LIBRAS (columnas)
    lsc_letters = sorted(set(y_lsc))
    cm = np.zeros((len(lsc_letters), len(classes)))
    for i, letter in enumerate(lsc_letters):
        m = y_lsc == letter
        for j, c in enumerate(classes):
            cm[i, j] = (pred[m] == c).mean()
    return rows, cm, lsc_letters, classes


def centroid_analysis(H_lsc, y_lsc, H_lib, y_lib):
    """
    Distancia sobre las 60 features de la mano, incluida la orientación de
    la palma. Requiere que ambos datasets compartan geometría (LSC70
    procesado con --convention librai); el desplazamiento de cámara
    restante es ~0° pitch / ~10° yaw (analyze_palm_orientation.py).
    """
    pooled_std = np.concatenate([H_lsc, H_lib]).std(0) + 1e-6
    Z_lsc, Z_lib = H_lsc / pooled_std, H_lib / pooled_std
    c_lsc = {k: Z_lsc[y_lsc == k].mean(0) for k in sorted(set(y_lsc))}
    c_lib = {k: Z_lib[y_lib == k].mean(0) for k in sorted(set(y_lib))}

    rows = {}
    for letter, c in c_lsc.items():
        dists = {k: float(np.linalg.norm(c - v)) for k, v in c_lib.items()}
        nearest = min(dists, key=dists.get)
        row = {"nearest_libras": nearest,
               "dist_nearest": round(dists[nearest], 2)}
        if letter in c_lib:
            others = [v for k, v in dists.items() if k != letter]
            diff = np.abs(c - c_lib[letter])
            row.update({
                "dist_same": round(dists[letter], 2),
                "dist_nearest_other": round(min(others), 2),
                "rank_same": sorted(dists, key=dists.get).index(letter) + 1,
                "group_diff": {g: round(float(diff[s].mean()), 2)
                               for g, s in FEATURE_GROUPS.items()},
            })
        rows[letter] = row
    return rows


def classify(letter, cross, cent, libras_classes):
    if letter not in libras_classes:
        return "solo LSC (o no está en LibrAI)"
    # El cruce de modelos manda; el centroide desempata
    same = cross["same_letter_pct"]
    if same >= 75 or (same >= 60 and cent.get("rank_same") == 1):
        return "equivalente"
    if same >= 30:
        return "parecida (revisar)"
    return "distinta"


def crop_hand(img: np.ndarray, hands, margin: float = 0.6) -> np.ndarray:
    """Recorte cuadrado alrededor de la mano detectada (o centrado si no hay)."""
    h, w = img.shape[:2]
    res = hands.process(img)
    if res.multi_hand_landmarks:
        # La mano más alta es la que deletrea (la otra queda en reposo)
        top = min(res.multi_hand_landmarks,
                  key=lambda hl: np.mean([lm.y for lm in hl.landmark]))
        pts = np.array([(lm.x * w, lm.y * h) for lm in top.landmark])
        (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max(x1 - x0, y1 - y0) * (0.5 + margin)
    else:
        cx, cy, half = w / 2, h / 2, min(h, w) / 2
    x0, x1 = int(max(cx - half, 0)), int(min(cx + half, w))
    y0, y1 = int(max(cy - half, 0)), int(min(cy + half, h))
    return img[y0:y1, x0:x1]


def gallery(lsc_dir: Path, lib_dir: Path, letters, out: Path, k=3, size=120):
    import mediapipe as mp

    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2)
    random.seed(0)
    fig, axes = plt.subplots(len(letters), 2 * k,
                             figsize=(2 * k * 1.2, len(letters) * 1.25))
    for r, letter in enumerate(letters):
        sign = "NN" if letter == "Ñ" else letter
        lsc = sorted(lsc_dir.glob(f"Per*/{sign}/*_3.jpg"))
        lib = sorted((lib_dir / letter).glob("*.png")) if lib_dir else []
        picks = (random.sample(lsc, min(k, len(lsc))) + [None] * k)[:k]
        picks += (random.sample(lib, min(k, len(lib))) + [None] * k)[:k]
        for c, path in enumerate(picks):
            ax = axes[r, c]
            ax.axis("off")
            if path is not None:
                img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
                ax.imshow(cv2.resize(crop_hand(img, hands), (size, size)))
            if c == 0:
                ax.set_title(f"{letter}  LSC", fontsize=8, loc="left")
            if c == k:
                ax.set_title("LIBRAS", fontsize=8, loc="left")
    plt.tight_layout()
    plt.savefig(out, dpi=90)
    plt.close()


def plot_cross_matrix(cm, lsc_letters, libras_classes, out: Path):
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(libras_classes)), libras_classes)
    ax.set_yticks(range(len(lsc_letters)), lsc_letters)
    ax.set_xlabel("Predicción Alicia LIBRAS (robust-v4)")
    ax.set_ylabel("Letra LSC (LSC70)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] >= 0.1:
                ax.text(j, i, f"{100 * cm[i, j]:.0f}", ha="center",
                        va="center", fontsize=7,
                        color="white" if cm[i, j] > 0.5 else "black")
    ax.set_title("LSC → LIBRAS: % de muestras por predicción")
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lsc-dir", type=Path,
                   default=Path("data/processed/lsc70_alphabet"))
    p.add_argument("--libras-dir", type=Path,
                   default=Path("data/processed/librai_alphabet_v4"))
    p.add_argument("--model-dir", type=Path, default=Path(
        "data/models/librai-alphabet-robust-v4/run_20260615_001921"))
    p.add_argument("--lsc-images", type=Path,
                   default=Path("data/raw/LSC70/LSC70/LSC70AN"))
    p.add_argument("--libras-images", type=Path, default=None,
                   help="src/LibrAI/test (para la galería)")
    p.add_argument("--output-dir", type=Path,
                   default=Path("data/analysis/lsc_vs_libras"))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    X_lsc, y_lsc = load_features(args.lsc_dir)
    X_lib, y_lib = load_features(args.libras_dir)
    print(f"LSC70: {len(X_lsc)} muestras, {len(set(y_lsc))} letras")
    print(f"LibrAI: {len(X_lib)} muestras, {len(set(y_lib))} letras")

    cross, cm, lsc_letters, libras_classes = cross_model(
        args.model_dir, X_lsc, y_lsc)
    cent = centroid_analysis(active_hand(X_lsc), y_lsc,
                             active_hand(X_lib), y_lib)

    report = {}
    for letter in lsc_letters:
        report[letter] = {
            "veredicto": classify(letter, cross[letter], cent[letter],
                                  libras_classes),
            "cruce_modelo": cross[letter],
            "centroides": cent[letter],
        }
    only_libras = sorted(set(libras_classes) - set(lsc_letters))
    report["_solo_libras"] = only_libras

    with open(args.output_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(args.output_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["letra", "veredicto", "pct_misma_letra",
                    "prediccion_top", "pct_top", "rank_centroide",
                    "grupo_mas_distinto"])
        for letter in lsc_letters:
            r, c = report[letter], cent[letter]
            gd = c.get("group_diff")
            w.writerow([
                letter, r["veredicto"], cross[letter]["same_letter_pct"],
                cross[letter]["pred_top"], cross[letter]["pred_top_pct"],
                c.get("rank_same"), max(gd, key=gd.get) if gd else None,
            ])

    plot_cross_matrix(cm, lsc_letters, libras_classes,
                      args.output_dir / "cross_matrix.png")
    gallery(args.lsc_images, args.libras_images,
            sorted(set(lsc_letters) | set(only_libras)),
            args.output_dir / "gallery.png")

    print(f"\n{'Letra':6}{'Veredicto':32}{'%misma':>8}  {'top LIBRAS':12}"
          f"{'rank':>5}  {'más cercana':16}grupo más distinto")
    for letter in lsc_letters:
        r, c = report[letter], cent[letter]
        gd = c.get("group_diff")
        same = cross[letter]["same_letter_pct"]
        print(f"{letter:6}{r['veredicto']:32}"
              f"{'-' if same is None else same:>8}  "
              f"{cross[letter]['pred_top']} {cross[letter]['pred_top_pct']:>5}%  "
              f"{c.get('rank_same', '-'):>4}  "
              f"{c['nearest_libras']:16}"
              f"{max(gd, key=gd.get) if gd else '-'}")
    print(f"Solo en LIBRAS: {only_libras}")
    print(f"Salida: {args.output_dir}")


if __name__ == "__main__":
    main()

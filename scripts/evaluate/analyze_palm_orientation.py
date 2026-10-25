"""
Orientación de la palma en el alfabeto manual: ¿diferencia entre alfabetos o
efecto de la cámara?, y ¿qué letras se distinguen por la orientación?

Features de palma (feature_engineering._palm_angles): normal de la palma
(muñeca→index_mcp × muñeca→pinky_mcp) en coordenadas de imagen, como
pitch = atan2(ny, nz) y yaw = atan2(nx, nz).

Controles:
  - Solo la mano en el slot "Left" (mano derecha en imagen sin espejar):
    el producto cruz invierte la normal al reflejar, mezclar manos
    fabricaría diferencias de orientación.
  - Un cambio de punto de vista desplaza pitch/yaw de TODAS las letras por
    igual. Se estima ese desplazamiento global (mediana circular sobre las
    letras compartidas) y se reporta el residuo por letra con IC 95%
    (bootstrap sobre sujetos de LSC70).

Análisis:
  A. Desplazamiento global LSC−LIBRAS y residuo por letra.
  B. Contribución de la palma dentro de cada alfabeto: mismo clasificador
     con features de configuración (56) vs configuración + palma (60);
     ganancia por letra y par que se confunde sin palma.

Uso:
    python scripts/evaluate/analyze_palm_orientation.py
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit, train_test_split  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

LEFT = slice(60, 120)
PALM = slice(56, 60)       # dentro de la mano: sin_p, cos_p, sin_y, cos_y
CONFIG = slice(0, 56)      # distancias + ángulos + curl


def wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


def circ_mean(deg, axis=0):
    r = np.deg2rad(deg)
    return np.rad2deg(np.arctan2(np.sin(r).mean(axis), np.cos(r).mean(axis)))


def circ_median(deg):
    """Valor que minimiza la suma de distancias angulares (búsqueda 0.5°)."""
    grid = np.arange(-180, 180, 0.5)
    cost = np.abs(wrap(np.asarray(deg)[None, :] - grid[:, None])).sum(1)
    return float(grid[cost.argmin()])


def circ_sd(deg):
    r = np.deg2rad(deg)
    R = np.hypot(np.sin(r).mean(), np.cos(r).mean())
    return float(np.rad2deg(np.sqrt(-2 * np.log(max(R, 1e-9)))))


def load_left_hand(root: Path, with_subjects: bool):
    X, y, subj = [], [], []
    if with_subjects:
        with open(root / "index.csv") as f:
            items = [(root / r["file"], r["label"], r["subject"])
                     for r in csv.DictReader(f)]
    else:
        items = [(f, d.name, None)
                 for d in sorted(p for p in root.iterdir() if p.is_dir())
                 for f in sorted(d.glob("*.npy"))]
    n_total = len(items)
    for f, label, s in items:
        v = np.load(f)
        if np.abs(v[LEFT]).sum() > 0 and np.abs(v[:60]).sum() == 0:
            X.append(v[LEFT])
            y.append(label)
            subj.append(s)
    return np.array(X), np.array(y), np.array(subj), n_total


def pitch_yaw(H):
    p = H[:, PALM]
    return (np.rad2deg(np.arctan2(p[:, 0], p[:, 1])),
            np.rad2deg(np.arctan2(p[:, 2], p[:, 3])))


def letter_stats(pitch, yaw, y):
    return {l: {"pitch": float(circ_mean(pitch[y == l])),
                "yaw": float(circ_mean(yaw[y == l])),
                "sd_pitch": circ_sd(pitch[y == l]),
                "sd_yaw": circ_sd(yaw[y == l]),
                "n": int((y == l).sum())}
            for l in sorted(set(y))}


def bootstrap_subject_means(pitch, yaw, y, subj, letter, n_boot, rng):
    """Medias por letra re-muestreando sujetos (LSC70)."""
    m = y == letter
    subjects = np.unique(subj[m])
    per_s = np.array([[circ_mean(pitch[m & (subj == s)]),
                       circ_mean(yaw[m & (subj == s)])] for s in subjects])
    idx = rng.integers(0, len(subjects), (n_boot, len(subjects)))
    return np.stack([circ_mean(per_s[idx, 0], axis=1),
                     circ_mean(per_s[idx, 1], axis=1)], axis=1)


def camera_vs_alphabet(lsc, lib, n_boot, seed):
    (Hs, ys, ss), (Hb, yb) = lsc, lib
    ps, ws = pitch_yaw(Hs)
    pb, wb = pitch_yaw(Hb)
    st_s, st_b = letter_stats(ps, ws, ys), letter_stats(pb, wb, yb)
    shared = sorted(set(st_s) & set(st_b))

    d_pitch = np.array([wrap(st_s[l]["pitch"] - st_b[l]["pitch"]) for l in shared])
    d_yaw = np.array([wrap(st_s[l]["yaw"] - st_b[l]["yaw"]) for l in shared])
    off_p, off_y = circ_median(d_pitch), circ_median(d_yaw)

    rng = np.random.default_rng(seed)
    rows = {}
    for i, l in enumerate(shared):
        boot = bootstrap_subject_means(ps, ws, ys, ss, l, n_boot, rng)
        res_p = wrap(boot[:, 0] - st_b[l]["pitch"] - off_p)
        res_y = wrap(boot[:, 1] - st_b[l]["yaw"] - off_y)
        rp, ry = wrap(d_pitch[i] - off_p), wrap(d_yaw[i] - off_y)
        ci_p = np.percentile(res_p, [2.5, 97.5])
        ci_y = np.percentile(res_y, [2.5, 97.5])
        signif = bool((ci_p[0] > 0 or ci_p[1] < 0) and abs(rp) >= 20
                      or (ci_y[0] > 0 or ci_y[1] < 0) and abs(ry) >= 20)
        rows[l] = {
            "lsc": st_s[l], "libras": st_b[l],
            "delta_pitch": float(d_pitch[i]), "delta_yaw": float(d_yaw[i]),
            "residual_pitch": float(rp), "residual_yaw": float(ry),
            "ci95_residual_pitch": [float(c) for c in ci_p],
            "ci95_residual_yaw": [float(c) for c in ci_y],
            "orientacion_distinta": signif,
        }
    return {"offset_pitch": off_p, "offset_yaw": off_y,
            "spread_delta_pitch": float(np.median(np.abs(wrap(d_pitch - off_p)))),
            "spread_delta_yaw": float(np.median(np.abs(wrap(d_yaw - off_y)))),
            "letters": rows, "stats_lsc": st_s, "stats_libras": st_b,
            "only_lsc": sorted(set(st_s) - set(st_b)),
            "only_libras": sorted(set(st_b) - set(st_s))}


def palm_contribution(H, y, groups, seed, max_n=20000):
    """Recall por letra con config sola vs config + palma (mismo split)."""
    rng = np.random.default_rng(seed)
    if len(H) > max_n:
        keep = rng.choice(len(H), max_n, replace=False)
        H, y = H[keep], y[keep]
        groups = None if groups is None else groups[keep]
    if groups is not None:
        tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed)
                      .split(H, y, groups))
    else:
        tr, te = train_test_split(np.arange(len(H)), test_size=0.2,
                                  stratify=y, random_state=seed)

    def fit_predict(cols):
        sc = StandardScaler().fit(H[tr][:, cols])
        clf = MLPClassifier((128, 64), max_iter=300, early_stopping=True,
                            random_state=seed)
        clf.fit(sc.transform(H[tr][:, cols]), y[tr])
        return clf.predict(sc.transform(H[te][:, cols]))

    p_cfg = fit_predict(np.r_[CONFIG])
    p_all = fit_predict(np.r_[CONFIG, PALM])
    yt = y[te]
    rows = {}
    for l in sorted(set(yt)):
        m = yt == l
        wrong = Counter(p_cfg[m][p_cfg[m] != l])
        rows[l] = {
            "recall_config": float((p_cfg[m] == l).mean()),
            "recall_config_palma": float((p_all[m] == l).mean()),
            "ganancia_palma": float((p_all[m] == l).mean()
                                    - (p_cfg[m] == l).mean()),
            "confusion_sin_palma": wrong.most_common(1)[0][0] if wrong else None,
        }
    acc = {"config": float((p_cfg == yt).mean()),
           "config_palma": float((p_all == yt).mean())}
    return acc, rows


def plot_letters(res, out: Path):
    fig, ax = plt.subplots(figsize=(9, 8))
    sl, sb = res["stats_lsc"], res["stats_libras"]
    for l, s in sb.items():
        ax.scatter(s["yaw"], s["pitch"], c="tab:blue", s=30)
        ax.annotate(l, (s["yaw"], s["pitch"]), color="tab:blue", fontsize=9,
                    xytext=(3, 3), textcoords="offset points")
    for l, s in sl.items():
        yw = wrap(s["yaw"] - res["offset_yaw"])
        pt = wrap(s["pitch"] - res["offset_pitch"])
        distinct = res["letters"].get(l, {}).get("orientacion_distinta")
        ax.scatter(yw, pt, c="tab:red" if distinct else "tab:orange",
                   marker="s", s=30)
        ax.annotate(l, (yw, pt), color="tab:red" if distinct else "tab:orange",
                    fontsize=9, xytext=(3, -10), textcoords="offset points")
        if l in sb:
            # camino corto en el círculo (−180° y +170° están a 10°)
            ax.plot([sb[l]["yaw"], sb[l]["yaw"] + wrap(yw - sb[l]["yaw"])],
                    [sb[l]["pitch"], sb[l]["pitch"] + wrap(pt - sb[l]["pitch"])],
                    c="gray", lw=0.6, alpha=0.6)
    ax.set_xlabel("yaw de la palma (°)")
    ax.set_ylabel("pitch de la palma (°)")
    ax.set_title("Orientación media de la palma por letra\n"
                 "azul: LIBRAS · naranja/rojo: LSC corregido por cámara "
                 "(rojo = diferencia significativa)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lsc-dir", type=Path,
                   default=Path("data/processed/lsc70_alphabet"))
    p.add_argument("--libras-dir", type=Path,
                   default=Path("data/processed/librai_alphabet_v4"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("data/analysis/palm_orientation"))
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    Hs, ys, ss, ns = load_left_hand(args.lsc_dir, with_subjects=True)
    Hb, yb, _, nb = load_left_hand(args.libras_dir, with_subjects=False)
    print(f"Mano derecha (slot Left): LSC70 {len(Hs)}/{ns}, "
          f"LibrAI {len(Hb)}/{nb}")

    res = camera_vs_alphabet((Hs, ys, ss), (Hb, yb), args.n_boot, args.seed)
    print(f"\nA. Desplazamiento global LSC−LIBRAS (cámara): "
          f"pitch {res['offset_pitch']:+.1f}°, yaw {res['offset_yaw']:+.1f}°")
    print(f"   Dispersión de Δ entre letras (mediana |Δ−offset|): "
          f"pitch {res['spread_delta_pitch']:.1f}°, "
          f"yaw {res['spread_delta_yaw']:.1f}°")
    print(f"\n{'Letra':6}{'Δpitch':>8}{'Δyaw':>8}{'res.pitch':>11}"
          f"{'IC95':>17}{'res.yaw':>9}{'IC95':>17}  veredicto")
    for l, r in res["letters"].items():
        cp, cy = r["ci95_residual_pitch"], r["ci95_residual_yaw"]
        print(f"{l:6}{r['delta_pitch']:>+8.1f}{r['delta_yaw']:>+8.1f}"
              f"{r['residual_pitch']:>+11.1f}  [{cp[0]:+6.1f},{cp[1]:+6.1f}]"
              f"{r['residual_yaw']:>+9.1f}  [{cy[0]:+6.1f},{cy[1]:+6.1f}]  "
              f"{'ORIENTACIÓN DISTINTA' if r['orientacion_distinta'] else '='}")

    print("\nB. Contribución de la palma dentro de cada alfabeto")
    contrib = {}
    for name, (H, y, g) in {"LSC": (Hs, ys, ss),
                            "LIBRAS": (Hb, yb, None)}.items():
        acc, rows = palm_contribution(H, y, g, args.seed)
        contrib[name] = {"accuracy": acc, "letters": rows}
        print(f"\n  {name}: accuracy config {acc['config'] * 100:.1f}% → "
              f"config+palma {acc['config_palma'] * 100:.1f}%")
        for l, r in sorted(rows.items(), key=lambda kv: -kv[1]["ganancia_palma"]):
            if r["ganancia_palma"] >= 0.05:
                print(f"    {l}: {r['recall_config'] * 100:5.1f}% → "
                      f"{r['recall_config_palma'] * 100:5.1f}% "
                      f"(+{r['ganancia_palma'] * 100:.1f} pp; sin palma se "
                      f"confunde con {r['confusion_sin_palma']})")

    plot_letters(res, args.output_dir / "palm_by_letter.png")
    with open(args.output_dir / "report.json", "w") as f:
        json.dump({"camera_vs_alphabet": res, "palm_contribution": contrib},
                  f, indent=2, ensure_ascii=False)
    with open(args.output_dir / "per_letter.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["letra", "pitch_lsc", "yaw_lsc", "pitch_libras",
                    "yaw_libras", "residuo_pitch", "residuo_yaw",
                    "orientacion_distinta", "ganancia_palma_lsc",
                    "ganancia_palma_libras"])
        letters = sorted(set(res["stats_lsc"]) | set(res["stats_libras"]))
        for l in letters:
            r = res["letters"].get(l, {})
            sl, sb = res["stats_lsc"].get(l, {}), res["stats_libras"].get(l, {})
            gl = contrib["LSC"]["letters"].get(l, {}).get("ganancia_palma")
            gb = contrib["LIBRAS"]["letters"].get(l, {}).get("ganancia_palma")
            w.writerow([l, sl.get("pitch"), sl.get("yaw"), sb.get("pitch"),
                        sb.get("yaw"), r.get("residual_pitch"),
                        r.get("residual_yaw"), r.get("orientacion_distinta"),
                        gl, gb])
    print(f"\nSalida: {args.output_dir}")


if __name__ == "__main__":
    main()

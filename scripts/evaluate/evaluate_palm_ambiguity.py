"""
Ambigüedad palma/dorso de MediaPipe: precisión por interpretación y efecto
de dos estrategias de inferencia.

MediaPipe decide lateralidad y geometría a la vez; en una fracción de frames
devuelve la interpretación "otra mano" (slot opuesto y normal de la palma
invertida, equivalente a mirror_features de la interpretación habitual).
Para cada sesión se toma como interpretación habitual el slot mayoritario.

Estrategias:
  directa    p(x)
  canonica   p(x) si x está en el slot habitual, p(mirror(x)) si no
  simetrica  (p(x) + p(mirror(x))) / 2

Uso:
    python scripts/evaluate/evaluate_palm_ambiguity.py \\
        --models data/models/alicia-alphabet-robust-ctrl-slot/run_X \\
                 data/models/alicia-alphabet-robust-mirror/run_Y
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from train_alphabet_robust import load_session, mirror_features  # noqa: E402

SESSIONS = ["20260614_011801_S01", "20260614_223425_S02",
            "20260614_233737_S02-extra", "20260615_000035_S02-cluster"]


def strategies(model, X, mu, sd, usual_left: bool):
    def p(Z):
        return model.predict((Z - mu) / sd, verbose=0, batch_size=1024)
    left = np.abs(X[:, 60:]).sum(1) > 0
    other = left != usual_left
    p_x, p_m = p(X), p(mirror_features(X))
    return other, {
        "directa": p_x,
        "canonica": np.where(other[:, None], p_m, p_x),
        "simetrica": (p_x + p_m) / 2,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, nargs="+", required=True)
    ap.add_argument("--sessions-root", type=Path,
                    default=Path("data/realworld_eval/sessions"))
    ap.add_argument("--output", type=Path,
                    default=Path("data/analysis/palm_ambiguity/report.json"))
    args = ap.parse_args()

    report = {}
    for run in args.models:
        model = tf.keras.models.load_model(str(run / "best_model.keras"))
        mu, sd = np.load(run / "norm_mean.npy"), np.load(run / "norm_std.npy")
        classes = [str(c).split(":")[-1] for c in np.load(run / "label_classes.npy")]
        libras = [c for c in np.load(run / "label_classes.npy").astype(str)
                  if ":" not in c or c.startswith("LIBRAS:")]
        letters = [c.split(":")[-1] for c in libras]
        idx = [list(np.load(run / "label_classes.npy").astype(str)).index(c)
               for c in libras]
        name = run.parent.name
        report[name] = {}
        print(f"\n== {name}")
        print(f"{'sesión':14}{'otra mano':>10}  {'estrategia':10}"
              f"{'total':>8}{'habitual':>10}{'otra mano':>11}")
        for s in SESSIONS:
            d = load_session(args.sessions_root / s, letters)
            X = np.vstack([d[k][0] for k in ("train", "val", "test")])
            y = np.concatenate([d[k][1] for k in ("train", "val", "test")])
            usual_left = (np.abs(X[:, 60:]).sum(1) > 0).mean() > 0.5
            other, ps = strategies(model, X, mu, sd, usual_left)
            report[name][s] = {"frac_otra_mano": float(other.mean())}
            for k, pr in ps.items():
                pred = np.array(idx)[pr[:, idx].argmax(1)]
                ok = pred == np.array(idx)[y]
                r = {"total": float(ok.mean()),
                     "habitual": float(ok[~other].mean()),
                     "otra_mano": float(ok[other].mean()) if other.any() else None}
                report[name][s][k] = r
                om = "-" if r["otra_mano"] is None else f"{r['otra_mano']:.1%}"
                print(f"{s.split('_')[-1]:14}{other.mean():>10.1%}  {k:10}"
                      f"{r['total']:>8.1%}{r['habitual']:>10.1%}{om:>11}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.output, "w"), indent=2)
    print(f"\nSalida: {args.output}")


if __name__ == "__main__":
    main()

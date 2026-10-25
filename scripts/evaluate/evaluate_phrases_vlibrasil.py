"""
Evalúa el modelo V-LIBRASIL v2 sobre TODAS las frases (clases con espacios
en el nombre) de v-librasil-flat. Reporta top-1/top-5 por frase y global.

Uso:
    python scripts/evaluate/evaluate_phrases_vlibrasil.py
    python scripts/evaluate/evaluate_phrases_vlibrasil.py --output data/analysis/phrases_report.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def pool_sequence(seq: np.ndarray) -> np.ndarray:
    return np.concatenate([seq.mean(axis=0), seq.std(axis=0)]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=Path("/data/models/vlibrasil-v2"))
    ap.add_argument("--data-dir", type=Path, default=Path("/data/processed/v-librasil-flat"))
    ap.add_argument("--output", type=Path, default=Path("/data/analysis/phrases_report.json"))
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if args.cpu:
        tf.config.set_visible_devices([], "GPU")

    meta = json.loads((args.model_dir / "metadata.json").read_text(encoding="utf-8"))
    class_names = meta["class_names"]
    name_to_idx = {n: i for i, n in enumerate(class_names)}

    norm_mean = np.load(args.model_dir / "norm_mean.npy")[0]
    norm_std = np.load(args.model_dir / "norm_std.npy")[0]

    model_path = args.model_dir / "best_model.keras"
    if not model_path.exists():
        model_path = args.model_dir / "final_model.keras"
    print(f"Modelo: {model_path}")
    model = tf.keras.models.load_model(str(model_path), compile=False)

    phrase_dirs = sorted(d for d in args.data_dir.iterdir() if d.is_dir() and " " in d.name)
    print(f"Frases (clases con espacios): {len(phrase_dirs)}")
    unknown = [d.name for d in phrase_dirs if d.name not in name_to_idx]
    if unknown:
        print(f"WARN: {len(unknown)} frases no están en el vocabulario del modelo, se omiten")
    phrase_dirs = [d for d in phrase_dirs if d.name in name_to_idx]

    per_phrase = {}
    all_pooled, all_true = [], []
    file_index = []
    skipped_dim = 0
    expected_dim = norm_mean.shape[0] // 2  # 208

    for d in phrase_dirs:
        for f in sorted(d.glob("*.npy")):
            seq = np.load(f)
            if seq.shape[1] != expected_dim:
                skipped_dim += 1
                continue
            all_pooled.append((pool_sequence(seq) - norm_mean) / norm_std)
            all_true.append(name_to_idx[d.name])
            file_index.append((d.name, f.name))
    if skipped_dim:
        print(f"WARN: {skipped_dim} archivos con feature-dim ≠ {expected_dim} omitidos (preprocesado antiguo)")

    X = np.asarray(all_pooled, dtype=np.float32)
    y_true = np.asarray(all_true)
    print(f"Muestras: {len(y_true)} — corriendo predicción...")
    probs = model.predict(X, batch_size=256, verbose=1)
    top5 = np.argsort(-probs, axis=1)[:, :5]
    top1 = top5[:, 0]

    hit1 = (top1 == y_true)
    hit5 = np.any(top5 == y_true[:, None], axis=1)

    for i, (phrase, fname) in enumerate(file_index):
        r = per_phrase.setdefault(phrase, {"n": 0, "top1": 0, "top5": 0, "samples": []})
        r["n"] += 1
        r["top1"] += int(hit1[i])
        r["top5"] += int(hit5[i])
        r["samples"].append({
            "file": fname,
            "predicted": class_names[int(top1[i])],
            "conf": float(probs[i, int(top1[i])]),
            "correct_top1": bool(hit1[i]),
            "correct_top5": bool(hit5[i]),
        })

    n = len(y_true)
    global_acc = {"n": n, "top1": float(hit1.mean()), "top5": float(hit5.mean())}
    print(f"\n=== GLOBAL FRASES ({n} muestras / {len(per_phrase)} clases) ===")
    print(f"  Top-1: {global_acc['top1']:.2%}")
    print(f"  Top-5: {global_acc['top5']:.2%}")

    worst = sorted(per_phrase.items(), key=lambda kv: kv[1]["top1"] / max(kv[1]["n"], 1))[:15]
    print("\nPeores 15 frases (top-1):")
    for name, r in worst:
        print(f"  {r['top1']}/{r['n']}  top5={r['top5']}/{r['n']}  {name}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"global": global_acc, "per_phrase": per_phrase, "model_dir": str(args.model_dir)},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReporte -> {args.output}")


if __name__ == "__main__":
    main()

"""
Prueba del modelo V-LIBRASIL v2 (augmentación + mean+std pooling).

Uso:
    python scripts/evaluate/test_vlibrasil_v2.py
    python scripts/evaluate/test_vlibrasil_v2.py --model-dir /data/models/vlibrasil-v2
    python scripts/evaluate/test_vlibrasil_v2.py --npy /data/processed/v-librasil-flat/Amigo/Amigo_Articulador1.npy
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


def pool_sequence(seq: np.ndarray) -> np.ndarray:
    """Mean + Std pooling temporal → (416,)."""
    return np.concatenate([seq.mean(axis=0), seq.std(axis=0)]).astype(np.float32)


def predict_top5(model, X_norm: np.ndarray, class_names: list):
    """Devuelve las top-5 predicciones con sus confianzas."""
    probs = model.predict(X_norm[np.newaxis, :], verbose=0)[0]
    top5_idx = np.argsort(probs)[::-1][:5]
    return [(class_names[i], float(probs[i])) for i in top5_idx]


def main():
    parser = argparse.ArgumentParser(description="Prueba modelo V-LIBRASIL v2")
    parser.add_argument("--model-dir", type=Path,
                        default=Path("/data/models/vlibrasil-v2"))
    parser.add_argument("--data-dir", type=Path,
                        default=Path("/data/processed/v-librasil-flat"))
    parser.add_argument("--npy", type=Path, default=None,
                        help="NPY específico para probar (opcional)")
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Cantidad de muestras aleatorias a probar")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    if not args.gpu:
        tf.config.set_visible_devices([], 'GPU')

    model_dir = args.model_dir

    # Verificar que el modelo existe
    model_path = model_dir / "best_model.keras"
    if not model_path.exists():
        model_path = model_dir / "final_model.keras"
    if not model_path.exists():
        print(f"ERROR: No se encontró modelo en {model_dir}")
        print("  Esperando: best_model.keras o final_model.keras")
        return 1

    # Cargar metadata
    with open(model_dir / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)

    class_names = meta["class_names"]
    n_classes = meta["n_classes"]
    print(f"\nModelo: {model_path.name}")
    print(f"Clases: {n_classes}")
    print(f"Val accuracy:  {meta.get('val_accuracy', '?'):.4f}  | Top-5: {meta.get('val_top5_accuracy', '?'):.4f}")
    print(f"Test accuracy: {meta.get('test_accuracy', '?'):.4f}  | Top-5: {meta.get('test_top5_accuracy', '?'):.4f}")
    print(f"Ref. aleatoria top-1: {meta.get('random_baseline_top1', 1/n_classes):.4f}  | top-5: {meta.get('random_baseline_top5', 5/n_classes):.4f}")

    # Cargar normalización
    norm_mean = np.load(model_dir / "norm_mean.npy")  # (1, 416)
    norm_std = np.load(model_dir / "norm_std.npy")    # (1, 416)

    print(f"\nCargando modelo...")
    model = tf.keras.models.load_model(str(model_path), compile=False)
    print("OK")

    # -----------------------------------------------------------------------
    # Modo 1: NPY específico
    # -----------------------------------------------------------------------
    if args.npy:
        npy_path = args.npy
        if not npy_path.exists():
            print(f"ERROR: {npy_path} no existe")
            return 1

        seq = np.load(npy_path)
        pooled = pool_sequence(seq)
        X_norm = (pooled - norm_mean[0]) / norm_std[0]

        label = npy_path.parent.name
        print(f"\n--- Predicción para: {npy_path.name} ---")
        print(f"Etiqueta real: {label}")
        top5 = predict_top5(model, X_norm, class_names)
        for rank, (name, conf) in enumerate(top5, 1):
            marker = " ← CORRECTO" if name == label else ""
            print(f"  Top-{rank}: {name:40s} {conf:.4f}{marker}")
        return 0

    # -----------------------------------------------------------------------
    # Modo 2: Muestras aleatorias del dataset
    # -----------------------------------------------------------------------
    if not args.data_dir.exists():
        print(f"ERROR: {args.data_dir} no existe")
        return 1

    # Recopilar todos los NPY disponibles
    all_files = []
    for class_dir in args.data_dir.iterdir():
        if class_dir.is_dir():
            for f in class_dir.glob("*.npy"):
                all_files.append((f, class_dir.name))

    if not all_files:
        print("No se encontraron archivos NPY")
        return 1

    rng = np.random.default_rng(42)
    indices = rng.choice(len(all_files), size=min(args.n_samples, len(all_files)), replace=False)
    samples = [all_files[i] for i in indices]

    print(f"\n{'='*60}")
    print(f"EVALUANDO {len(samples)} MUESTRAS ALEATORIAS")
    print(f"{'='*60}")

    correct_top1 = 0
    correct_top5 = 0

    for npy_path, true_label in samples:
        try:
            seq = np.load(npy_path)
            pooled = pool_sequence(seq)
            X_norm = (pooled - norm_mean[0]) / norm_std[0]
            top5 = predict_top5(model, X_norm, class_names)

            top1_name = top5[0][0]
            top5_names = [n for n, _ in top5]

            hit1 = top1_name == true_label
            hit5 = true_label in top5_names
            correct_top1 += hit1
            correct_top5 += hit5

            status = "TOP1✓" if hit1 else ("TOP5✓" if hit5 else "✗")
            print(f"\n[{status}] Real: {true_label}")
            for rank, (name, conf) in enumerate(top5, 1):
                marker = " ←" if name == true_label else ""
                print(f"      Top-{rank}: {name:40s} {conf:.3f}{marker}")

        except Exception as e:
            print(f"Error en {npy_path}: {e}")

    print(f"\n{'='*60}")
    print(f"RESUMEN ({len(samples)} muestras)")
    print(f"  Top-1 accuracy: {correct_top1}/{len(samples)} = {correct_top1/len(samples):.2%}")
    print(f"  Top-5 accuracy: {correct_top5}/{len(samples)} = {correct_top5/len(samples):.2%}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

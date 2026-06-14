"""
Entrenamiento con Prototypical Networks para V-LIBRASIL.

Estrategia: few-shot learning episódico.
  - Divide las 1361 clases en train/val/test (no muestras).
  - Cada episodio: N clases × K support + Q query.
  - El encoder aprende a mapear (300, 208) → embedding L2-normalizado.
  - En inferencia: nearest-prototype con la base de datos completa.

Con 3 muestras/clase → K=1 (1 support, 2 query por clase).

Uso:
    python scripts/train/train_vlibrasil_metric.py
    python scripts/train/train_vlibrasil_metric.py \\
        --data-dir /data/processed/v-librasil-flat \\
        --embed-dim 256 --n-way 20 --epochs 100
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("vlibrasil_metric_training.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_dataset(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Carga todas las secuencias NPY desde subdirectorios por clase.

    Returns:
        X: (n_samples, seq_len, features)
        y: (n_samples,) etiquetas enteras
        class_names: lista de nombres de clase (índice == etiqueta)
    """
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"No hay subdirectorios en {data_dir}")

    # Detectar shape más común
    shape_counts: Dict[tuple, int] = {}
    for cd in class_dirs:
        for f in cd.glob("*.npy"):
            try:
                s = np.load(f).shape
                shape_counts[s] = shape_counts.get(s, 0) + 1
            except Exception:
                pass

    expected = max(shape_counts, key=shape_counts.get)
    logger.info(f"Shape esperado: {expected}  | distribución: {shape_counts}")

    X_list, y_list = [], []
    skipped = 0
    for class_idx, cd in enumerate(class_dirs):
        for f in sorted(cd.glob("*.npy")):
            try:
                seq = np.load(f)
                if seq.shape != expected:
                    skipped += 1
                    continue
                X_list.append(seq)
                y_list.append(class_idx)
            except Exception as e:
                logger.warning(f"Error {f}: {e}")

    if skipped:
        logger.warning(f"Omitidos {skipped} archivos con shape incorrecto")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    class_names = [d.name for d in class_dirs]

    logger.info(f"Cargadas {len(X)} muestras, {len(class_names)} clases, shape {X.shape}")
    return X, y, class_names


def split_classes(
    n_classes: int, train_frac: float = 0.8, val_frac: float = 0.1, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide LAS CLASES (no las muestras) en train/val/test.
    Val y test son clases nunca vistas durante el entrenamiento.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_classes)
    n_train = int(n_classes * train_frac)
    n_val = int(n_classes * val_frac)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


# ---------------------------------------------------------------------------
# Encoder (backbone CNN-LSTM)
# ---------------------------------------------------------------------------

def build_encoder(seq_len: int, n_features: int, embed_dim: int) -> keras.Model:
    """
    CNN-LSTM → embedding L2-normalizado.

    Input:  (seq_len, n_features)
    Output: (embed_dim,)  — L2 normalizado, listo para distancias euclídeas
    """
    inp = keras.Input(shape=(seq_len, n_features), name="sequence")

    x = layers.Conv1D(64, 5, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(128, 5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.LSTM(256, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(128)(x)

    x = layers.Dense(embed_dim)(x)
    # L2 normalization: permite usar distancia euclídea ≡ cosine
    x = layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=-1), name="embedding"
    )(x)

    return keras.Model(inp, x, name="encoder")


# ---------------------------------------------------------------------------
# Episodios y loss prototípica
# ---------------------------------------------------------------------------

def sample_episode(
    X: np.ndarray,
    y: np.ndarray,
    class_pool: np.ndarray,
    n_way: int,
    k_shot: int,
    n_query: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Muestrea un episodio N-way K-shot.

    Returns:
        support_x: (n_way * k_shot, seq_len, features)
        support_y: (n_way * k_shot,)  — etiquetas locales 0..n_way-1
        query_x:   (n_way * n_query, seq_len, features)
        query_y:   (n_way * n_query,)
    """
    chosen = rng.choice(class_pool, n_way, replace=False)
    sx, sy, qx, qy = [], [], [], []

    for local_cls, global_cls in enumerate(chosen):
        idxs = np.where(y == global_cls)[0]
        perm = rng.permutation(idxs)
        needed = k_shot + n_query
        # Si hay menos muestras que needed, repetir con replace
        if len(perm) < needed:
            perm = rng.choice(idxs, needed, replace=True)
        sx.append(X[perm[:k_shot]])
        sy.extend([local_cls] * k_shot)
        qx.append(X[perm[k_shot:k_shot + n_query]])
        qy.extend([local_cls] * n_query)

    return (
        np.concatenate(sx).astype(np.float32),
        np.array(sy, dtype=np.int32),
        np.concatenate(qx).astype(np.float32),
        np.array(qy, dtype=np.int32),
    )


@tf.function
def prototypical_step(
    encoder: keras.Model,
    support_x: tf.Tensor,
    support_y: tf.Tensor,
    query_x: tf.Tensor,
    query_y: tf.Tensor,
    n_way: int,
    training: bool,
    optimizer: keras.optimizers.Optimizer = None,
):
    """
    Un paso de entrenamiento/evaluación prototípico.

    Loss: cross-entropy sobre log-softmax(-distancias²).
    """
    with tf.GradientTape() as tape:
        s_emb = encoder(support_x, training=training)   # (n_way*k, D)
        q_emb = encoder(query_x,   training=training)   # (n_way*q, D)

        # Prototipos: media por clase local (vectorizado para @tf.function)
        # one_hot: (n_way*k, n_way)  →  (n_way, D) = one_hot.T @ s_emb / k_shot
        one_hot = tf.one_hot(support_y, n_way)         # (n_way*k, n_way)
        counts  = tf.reduce_sum(one_hot, axis=0, keepdims=True)  # (1, n_way)
        protos  = tf.matmul(
            tf.transpose(one_hot), s_emb               # (n_way, D)
        ) / tf.transpose(counts)                       # (n_way, D)

        # Distancias euclídeas al cuadrado
        # q_emb: (nq, D)  protos: (n_way, D)
        dists = tf.reduce_sum(
            tf.square(
                tf.expand_dims(q_emb, 1) - tf.expand_dims(protos, 0)
            ),
            axis=-1,
        )  # (nq, n_way)

        log_probs = tf.nn.log_softmax(-dists, axis=-1)  # (nq, n_way)
        # Gather log prob de la clase correcta
        indices = tf.stack([
            tf.range(tf.shape(query_y)[0]), query_y
        ], axis=1)
        loss = -tf.reduce_mean(tf.gather_nd(log_probs, indices))

    if training and optimizer is not None:
        grads = tape.gradient(loss, encoder.trainable_variables)
        optimizer.apply_gradients(zip(grads, encoder.trainable_variables))

    preds = tf.argmax(-dists, axis=-1, output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(preds, query_y), tf.float32))
    return loss, acc


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def run_training(args):
    # GPU
    if args.gpu:
        gpus = tf.config.list_physical_devices("GPU")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        logger.info(f"GPU: {gpus}")
    else:
        tf.config.set_visible_devices([], "GPU")
        logger.info("Usando CPU")

    # Cargar datos
    X, y, class_names = load_dataset(Path(args.data_dir))
    n_classes = len(class_names)
    seq_len, n_features = X.shape[1], X.shape[2]

    # Normalización global
    X_mean = X.mean(axis=(0, 1), keepdims=True)
    X_std  = X.std(axis=(0, 1),  keepdims=True) + 1e-8
    X = (X - X_mean) / X_std

    # Split de clases
    train_cls, val_cls, test_cls = split_classes(n_classes)
    logger.info(
        f"Clases — train: {len(train_cls)}, val: {len(val_cls)}, test: {len(test_cls)}"
    )

    # Encoder y optimizer
    encoder = build_encoder(seq_len, n_features, args.embed_dim)
    encoder.summary(print_fn=logger.info)

    optimizer = keras.optimizers.Adam(args.lr)
    scheduler = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=args.lr,
        first_decay_steps=args.episodes_per_epoch * 10,
    )

    rng = np.random.default_rng(42)
    best_val_acc = 0.0
    patience_counter = 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Guardar normalization params para inferencia
    np.save(output_dir / "norm_mean.npy", X_mean)
    np.save(output_dir / "norm_std.npy",  X_std)

    logger.info(
        f"\nIniciando entrenamiento — "
        f"{args.n_way}-way {args.k_shot}-shot | "
        f"{args.episodes_per_epoch} episodios/epoch | "
        f"{args.epochs} epochs\n"
    )

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        train_losses, train_accs = [], []
        for _ in range(args.episodes_per_epoch):
            sx, sy, qx, qy = sample_episode(
                X, y, train_cls,
                args.n_way, args.k_shot, args.n_query, rng,
            )
            # Actualizar lr con scheduler
            optimizer.learning_rate = scheduler(
                optimizer.iterations
            )
            loss, acc = prototypical_step(
                encoder,
                tf.constant(sx), tf.constant(sy),
                tf.constant(qx), tf.constant(qy),
                args.n_way, training=True, optimizer=optimizer,
            )
            train_losses.append(float(loss))
            train_accs.append(float(acc))

        # ---- Validación ----
        val_losses, val_accs = [], []
        for _ in range(args.val_episodes):
            sx, sy, qx, qy = sample_episode(
                X, y, val_cls,
                args.n_way, args.k_shot, args.n_query, rng,
            )
            loss, acc = prototypical_step(
                encoder,
                tf.constant(sx), tf.constant(sy),
                tf.constant(qx), tf.constant(qy),
                args.n_way, training=False, optimizer=None,
            )
            val_losses.append(float(loss))
            val_accs.append(float(acc))

        t_loss = np.mean(train_losses)
        t_acc  = np.mean(train_accs)
        v_loss = np.mean(val_losses)
        v_acc  = np.mean(val_accs)

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} — "
            f"train loss: {t_loss:.4f}  acc: {t_acc:.4f} | "
            f"val loss: {v_loss:.4f}  acc: {v_acc:.4f}"
        )

        # Checkpoint
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            patience_counter = 0
            encoder.save(output_dir / "best_encoder.keras")
            logger.info(f"  ✓ Nuevo mejor modelo (val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping en epoch {epoch}")
                break

    # ---- Test (clases nunca vistas) ----
    logger.info("\nEvaluando en clases de TEST (nunca vistas)...")
    encoder.load_weights(output_dir / "best_encoder.keras")
    test_accs = []
    for _ in range(args.val_episodes):
        sx, sy, qx, qy = sample_episode(
            X, y, test_cls,
            args.n_way, args.k_shot, args.n_query, rng,
        )
        _, acc = prototypical_step(
            encoder,
            tf.constant(sx), tf.constant(sy),
            tf.constant(qx), tf.constant(qy),
            args.n_way, training=False, optimizer=None,
        )
        test_accs.append(float(acc))

    test_acc = np.mean(test_accs)
    test_ci  = 1.96 * np.std(test_accs) / np.sqrt(len(test_accs))
    logger.info(f"Test accuracy: {test_acc:.4f} ± {test_ci:.4f} (95% CI)")

    # ---- Construir base de datos de prototipos (todas las clases) ----
    logger.info("\nConstruyendo base de prototipos para inferencia...")
    prototypes = {}
    for class_idx, class_name in enumerate(class_names):
        class_samples = X[y == class_idx]
        if len(class_samples) == 0:
            continue
        embs = encoder(
            tf.constant(class_samples, dtype=tf.float32), training=False
        ).numpy()
        prototypes[class_name] = embs.mean(axis=0).tolist()

    with open(output_dir / "prototypes.json", "w", encoding="utf-8") as f:
        json.dump(prototypes, f, ensure_ascii=False)

    # Metadata
    metadata = {
        "n_classes": n_classes,
        "class_names": class_names,
        "seq_len": seq_len,
        "n_features": n_features,
        "embed_dim": args.embed_dim,
        "n_way": args.n_way,
        "k_shot": args.k_shot,
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "test_acc_ci": float(test_ci),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"\nModelo guardado en: {output_dir}")
    logger.info(f"Best val acc:  {best_val_acc:.4f}")
    logger.info(f"Test acc:      {test_acc:.4f} ± {test_ci:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Entrena Prototypical Network para V-LIBRASIL (few-shot)"
    )
    parser.add_argument("--data-dir",   default="/data/processed/v-librasil-flat")
    parser.add_argument("--output-dir", default="/data/models/vlibrasil-metric")
    parser.add_argument("--embed-dim",  type=int,   default=256)
    parser.add_argument("--n-way",      type=int,   default=20,
                        help="Clases por episodio")
    parser.add_argument("--k-shot",     type=int,   default=1,
                        help="Muestras de soporte por clase")
    parser.add_argument("--n-query",    type=int,   default=2,
                        help="Muestras de query por clase")
    parser.add_argument("--episodes-per-epoch", type=int, default=500)
    parser.add_argument("--val-episodes",       type=int, default=200)
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--patience",   type=int,   default=20)
    parser.add_argument("--gpu",        action="store_true")
    args = parser.parse_args()

    run_training(args)


if __name__ == "__main__":
    main()

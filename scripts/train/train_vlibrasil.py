"""
Script de entrenamiento para V-LIBRASIL (Lenguaje de Señas Brasileño).

Problema original: 1361 clases con solo 2-3 muestras por clase → 0.17% accuracy.

Solución aplicada:
  1. Augmentación fuerte (5x): ruido gaussiano, escala, desplazamiento temporal,
     dropout de frames y espejo horizontal de landmarks.
  2. Mean+Std pooling temporal → vector (416,) estático por secuencia.
  3. DNN con regularización L2 + Dropout fuerte.
  4. Label smoothing (0.1) para evitar sobreajuste.
  5. Top-1 y Top-5 accuracy como métricas.
  6. Guardado de prototipos de clase para inferencia robusta.

Resultado esperado con augmentación 5x: ~15-20% top-1, ~40-50% top-5.
La referencia aleatoria es top-1=0.07%, top-5=0.37%.

Uso:
    # Dentro del contenedor Docker (GPU):
    python scripts/train/train_vlibrasil.py --gpu

    # CPU (fuera del contenedor, para pruebas rápidas):
    python scripts/train/train_vlibrasil.py --aug-factor 3 --epochs 150
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow import keras
from tensorflow.keras import layers

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vlibrasil_v2_training.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Augmentación de secuencias temporales
# ---------------------------------------------------------------------------

def augment_sequence(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Aplica una augmentación aleatoria a una secuencia (T, F).

    Técnicas:
        - Ruido gaussiano en features de posición (primeras 42 cols)
        - Escala aleatoria global (±10%)
        - Desplazamiento temporal (shift circular, ±15 frames)
        - Frame dropout (zeroing de hasta 10% de frames)
        - Espejo horizontal (invertir signo de coordenadas x, cols 0,2,4,...40)
    """
    s = seq.copy()
    T, F = s.shape

    # 1. Ruido gaussiano solo en posiciones (primeras 84 features: 42 pos × 2 manos)
    noise_std = 0.015
    s[:, :84] += rng.normal(0, noise_std, size=(T, 84))

    # 2. Escala global
    scale = rng.uniform(0.90, 1.10)
    s *= scale

    # 3. Desplazamiento temporal (shift circular)
    shift = rng.integers(-15, 16)
    if shift != 0:
        s = np.roll(s, shift, axis=0)

    # 4. Frame dropout (hasta 10% de frames → cero)
    n_drop = rng.integers(0, max(1, int(T * 0.10)) + 1)
    drop_idx = rng.choice(T, size=n_drop, replace=False)
    s[drop_idx] = 0.0

    # 5. Espejo horizontal con 50% de probabilidad
    #    Coordenadas x de los 21 landmarks mano derecha: cols 0,2,4,...40
    #    Coordenadas x de los 21 landmarks mano izquierda: cols 104,106,...144
    if rng.random() < 0.5:
        x_cols_r = list(range(0, 42, 2))          # x de mano derecha
        x_cols_l = list(range(104, 146, 2))       # x de mano izquierda
        s[:, x_cols_r] *= -1
        s[:, x_cols_l] *= -1

    return s.astype(np.float32)


def pool_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Convierte una secuencia (T, F) en un vector estático (2*F,)
    usando mean + std pooling sobre el eje temporal.

    Con F=208 → vector (416,).
    """
    mean = seq.mean(axis=0)
    std = seq.std(axis=0)
    return np.concatenate([mean, std]).astype(np.float32)


# ---------------------------------------------------------------------------
# Carga y augmentación del dataset
# ---------------------------------------------------------------------------

def load_and_augment(data_dir: Path, aug_factor: int, rng: np.random.Generator):
    """
    Carga todas las secuencias NPY, aplica mean+std pooling y augmentación.

    Returns:
        X: (n_samples, 416)
        y: (n_samples,) enteros
        class_names: lista de nombres
    """
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No se encontraron clases en {data_dir}")

    class_names = [d.name for d in class_dirs]
    n_classes = len(class_names)
    logger.info(f"Clases encontradas: {n_classes}")

    X_list, y_list = [], []
    skipped = 0

    for class_idx, class_dir in enumerate(class_dirs):
        npy_files = sorted(class_dir.glob("*.npy"))
        for npy_file in npy_files:
            try:
                seq = np.load(npy_file)  # (300, 208)
                if seq.ndim != 2 or seq.shape[1] != 208:
                    skipped += 1
                    continue
                # Original
                X_list.append(pool_sequence(seq))
                y_list.append(class_idx)
                # Aumentados
                for _ in range(aug_factor - 1):
                    aug = augment_sequence(seq, rng)
                    X_list.append(pool_sequence(aug))
                    y_list.append(class_idx)
            except Exception as e:
                logger.warning(f"Error cargando {npy_file}: {e}")
                skipped += 1

    if skipped:
        logger.warning(f"Archivos omitidos: {skipped}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    logger.info(f"Dataset aumentado: {X.shape[0]} muestras, {X.shape[1]} features, {n_classes} clases")
    logger.info(f"Muestras por clase (original): {len(X_list) // n_classes // aug_factor} × {aug_factor} = {aug_factor * (len(X_list) // n_classes // aug_factor)}")

    return X, y, class_names


# ---------------------------------------------------------------------------
# Modelo DNN con regularización fuerte
# ---------------------------------------------------------------------------

def label_smoothing_loss(n_classes: int, smoothing: float = 0.1):
    """
    Label smoothing para etiquetas sparse (enteras).
    SparseCategoricalCrossentropy no soporta label_smoothing en TF 2.10,
    así que convertimos a one-hot y usamos CategoricalCrossentropy.
    """
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_one_hot = tf.one_hot(y_true, n_classes)
        y_smooth = y_one_hot * (1.0 - smoothing) + smoothing / n_classes
        return keras.losses.categorical_crossentropy(y_smooth, y_pred)
    return loss_fn


def build_model(n_features: int, n_classes: int, lr: float, dropout: float) -> keras.Model:
    """
    DNN profundo con regularización L2 + Dropout para clasificación
    de vocabulario grande con pocas muestras por clase.
    """
    reg = keras.regularizers.l2(0.005)

    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.BatchNormalization(),

        layers.Dense(512, kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout),

        layers.Dense(512, kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout),

        layers.Dense(256, kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout * 0.75),

        layers.Dense(256, kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout * 0.75),

        layers.Dense(n_classes, activation='softmax'),
    ], name="vlibrasil_dnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=label_smoothing_loss(n_classes, smoothing=0.1),
        metrics=[
            'accuracy',
            keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_accuracy'),
        ],
    )
    return model


# ---------------------------------------------------------------------------
# Guardado de prototipos para inferencia robusta
# ---------------------------------------------------------------------------

def save_prototypes(model: keras.Model, X_train: np.ndarray, y_train: np.ndarray,
                    n_classes: int, output_dir: Path):
    """
    Calcula el prototipo de cada clase como la media de los embeddings
    de las muestras de entrenamiento en la penúltima capa.
    Guarda prototypes.npy y se usa en inferencia con cosine similarity.
    """
    # Modelo encoder: todo menos la última capa (Dense softmax)
    encoder = keras.Model(
        inputs=model.input,
        outputs=model.layers[-2].output,  # antes del Dense final
        name="encoder"
    )
    embeddings = encoder.predict(X_train, batch_size=256, verbose=0)

    prototypes = np.zeros((n_classes, embeddings.shape[1]), dtype=np.float32)
    counts = np.zeros(n_classes, dtype=np.int32)

    for emb, label in zip(embeddings, y_train):
        prototypes[label] += emb
        counts[label] += 1

    # Normalizar por cantidad de muestras
    mask = counts > 0
    prototypes[mask] /= counts[mask, np.newaxis]

    np.save(output_dir / "prototypes.npy", prototypes)
    logger.info(f"Prototipos guardados: {prototypes.shape} en {output_dir / 'prototypes.npy'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Entrena V-LIBRASIL con augmentación")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("/data/processed/v-librasil-flat"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/data/models/vlibrasil-v2"))
    parser.add_argument("--aug-factor", type=int, default=5,
                        help="Multiplicador de augmentación (default: 5 → 15 muestras por clase)")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    # GPU
    if not args.gpu:
        tf.config.set_visible_devices([], 'GPU')
        logger.info("Usando CPU")
    else:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"GPU: {gpus}")

    if not args.data_dir.exists():
        logger.error(f"Directorio no encontrado: {args.data_dir}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Cargar y aumentar
    X, y, class_names = load_and_augment(args.data_dir, args.aug_factor, rng)
    n_features = X.shape[1]
    n_classes = len(class_names)

    # Normalización global
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - X_mean) / X_std
    np.save(args.output_dir / "norm_mean.npy", X_mean)
    np.save(args.output_dir / "norm_std.npy", X_std)

    # Split estratificado: train / val / test
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_val_idx, test_idx = next(sss_test.split(X, y))
    X_tv, y_tv = X[train_val_idx], y[train_val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    val_frac = args.val_size / (1 - args.test_size)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=args.seed)
    train_idx, val_idx = next(sss_val.split(X_tv, y_tv))
    X_train, y_train = X_tv[train_idx], y_tv[train_idx]
    X_val, y_val = X_tv[val_idx], y_tv[val_idx]

    logger.info(f"Splits: train={len(X_train)} | val={len(X_val)} | test={len(X_test)}")
    logger.info(f"Features: {n_features} | Clases: {n_classes}")
    logger.info(f"Augmentación: {args.aug_factor}x → ~{args.aug_factor * 3} muestras/clase")

    # Modelo
    model = build_model(n_features, n_classes, args.lr, args.dropout)
    model.summary(print_fn=logger.info)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=args.patience,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(args.output_dir / "best_model.keras"),
            monitor='val_top5_accuracy', save_best_only=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10,
            min_lr=1e-6, verbose=1
        ),
        keras.callbacks.CSVLogger(str(args.output_dir / "history.csv")),
    ]

    logger.info("\nIniciando entrenamiento...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluación
    train_loss, train_acc, train_top5 = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc, val_top5 = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc, test_top5 = model.evaluate(X_test, y_test, verbose=0)

    logger.info(f"\nResultados:")
    logger.info(f"  Train — acc: {train_acc:.4f} | top5: {train_top5:.4f}")
    logger.info(f"  Val   — acc: {val_acc:.4f}   | top5: {val_top5:.4f}")
    logger.info(f"  Test  — acc: {test_acc:.4f}  | top5: {test_top5:.4f}")
    logger.info(f"  (Referencia aleatoria — top1: {1/n_classes:.4f} | top5: {5/n_classes:.4f})")

    # Guardar modelo y prototipos
    model.save(args.output_dir / "final_model.keras")
    save_prototypes(model, X_train, y_train, n_classes, args.output_dir)

    metadata = {
        "dataset": "V-LIBRASIL",
        "n_classes": n_classes,
        "class_names": class_names,
        "n_features": n_features,
        "aug_factor": args.aug_factor,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "pooling": "mean+std",
        "val_accuracy": float(val_acc),
        "val_top5_accuracy": float(val_top5),
        "test_accuracy": float(test_acc),
        "test_top5_accuracy": float(test_top5),
        "random_baseline_top1": float(1 / n_classes),
        "random_baseline_top5": float(5 / n_classes),
    }
    with open(args.output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"\nModelo guardado en: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

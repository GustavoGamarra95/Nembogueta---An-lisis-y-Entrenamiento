"""
Entrenamiento de modelo DNN para reconocimiento del alfabeto BSL/LIBRAS.

Dataset: biankatpas/Brazilian-Sign-Language-Alphabet-Dataset (15 letras estáticas).
Input: vectores de 208 features MediaPipe (sin dimensión temporal).
Arquitectura: DNN (Dense + BatchNorm + Dropout) — igual que V-LIBRASIL v2.

Uso (dentro del contenedor):
    python scripts/train/train_bsl_alphabet.py \
        --data-dir /data/processed/bsl_alphabet \
        --output-dir /data/models/bsl-alphabet

Salida:
    /data/models/bsl-alphabet/
        best_model.keras
        norm_mean.npy
        norm_std.npy
        label_classes.npy
        training_history.png
        classification_report.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# GPU: habilitar memory growth para evitar OOM
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logger.info(f"GPUs detectadas: {len(gpus)}")
else:
    logger.info("Sin GPU — entrenando en CPU")


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_data(data_dir: Path):
    """
    Carga archivos .npy desde subdirectorios por letra.

    Espera estructura:
        data_dir/A/A_0000.npy
        data_dir/B/B_0000.npy ...

    Returns:
        X: (N, 208), y_encoded: (N,), label_encoder
    """
    X_list, y_list = [], []

    letter_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not letter_dirs:
        raise FileNotFoundError(f"No se encontraron subcarpetas en {data_dir}")

    for letter_dir in letter_dirs:
        letter = letter_dir.name
        npy_files = sorted(letter_dir.glob("*.npy"))

        if not npy_files:
            logger.warning(f"Sin archivos .npy en {letter_dir}")
            continue

        for npy_file in npy_files:
            feat = np.load(npy_file)
            if feat.shape != (208,):
                logger.warning(f"Shape inesperado {feat.shape} en {npy_file} — saltando")
                continue
            X_list.append(feat)
            y_list.append(letter)

        logger.info(f"  {letter}: {len(npy_files)} muestras")

    X = np.array(X_list, dtype=np.float32)
    le = LabelEncoder()
    y = le.fit_transform(y_list)

    logger.info(f"Total: {len(X)} muestras | {len(le.classes_)} clases: {list(le.classes_)}")
    return X, y, le


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

def build_model(num_features: int, num_classes: int, learning_rate: float) -> tf.keras.Model:
    """
    DNN con BatchNorm y Dropout para clasificación de signos estáticos.

    Input: (208,)
    Output: (num_classes,) softmax
    """
    inputs = tf.keras.Input(shape=(num_features,), name="landmarks")

    x = tf.keras.layers.Dense(512, activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="DNN_BSL_Alphabet")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Augmentation en tiempo real
# ---------------------------------------------------------------------------

def augment_batch(X_batch: np.ndarray) -> np.ndarray:
    """Augmentación leve para features estáticos: ruido + escala."""
    aug = X_batch.copy()
    # Ruido gaussiano pequeño
    aug += np.random.normal(0, 0.02, aug.shape).astype(np.float32)
    # Escala aleatoria por muestra
    scales = np.random.uniform(0.95, 1.05, (len(aug), 1)).astype(np.float32)
    aug *= scales
    return aug


class AugmentedGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=64, augment=True, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        X_b = self.X[batch_idx]
        y_b = self.y[batch_idx]
        if self.augment:
            X_b = augment_batch(X_b)
        return X_b, y_b

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# ---------------------------------------------------------------------------
# Entrenamiento principal
# ---------------------------------------------------------------------------

def train(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar datos
    logger.info("Cargando datos...")
    X, y, le = load_data(data_dir)
    num_classes = len(le.classes_)

    # Normalización global
    norm_mean = X.mean(axis=0)
    norm_std = X.std(axis=0) + 1e-8
    X = (X - norm_mean) / norm_std

    # Split 70/15/15
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42
    )
    logger.info(f"Split — train: {len(X_train)} | val: {len(X_val)} | test: {len(X_test)}")

    # Modelo
    model = build_model(
        num_features=X.shape[1],
        num_classes=num_classes,
        learning_rate=args.learning_rate,
    )
    model.summary(print_fn=logger.info)

    # Pesos de clase para balancear clases desiguales
    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(cw))

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    # Generadores
    train_gen = AugmentedGenerator(X_train, y_train, batch_size=args.batch_size, augment=True)
    val_gen = AugmentedGenerator(X_val, y_val, batch_size=args.batch_size, augment=False, shuffle=False)

    logger.info("Iniciando entrenamiento...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
    )

    # Evaluación en test
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"\nTest Loss    : {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    report = classification_report(
        y_test, y_pred, target_names=le.classes_, output_dict=True
    )
    logger.info("\n" + classification_report(y_test, y_pred, target_names=le.classes_))

    # Guardar artefactos
    np.save(output_dir / "norm_mean.npy", norm_mean)
    np.save(output_dir / "norm_std.npy", norm_std)
    np.save(output_dir / "label_classes.npy", le.classes_)
    np.save(output_dir / "confusion_matrix.npy", confusion_matrix(y_test, y_pred))

    with open(output_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(output_dir / "model_info.json", "w") as f:
        json.dump(
            {
                "num_classes": num_classes,
                "classes": list(le.classes_),
                "num_features": int(X.shape[1]),
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
                "test_accuracy": float(test_acc),
                "test_loss": float(test_loss),
                "epochs_trained": len(history.history["loss"]),
            },
            f,
            indent=2,
        )

    # Gráfica
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history["accuracy"], label="Train")
    ax1.plot(history.history["val_accuracy"], label="Val")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True)
    ax2.plot(history.history["loss"], label="Train")
    ax2.plot(history.history["val_loss"], label="Val")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png", dpi=150)
    plt.close()
    logger.info(f"Gráfica guardada en {output_dir / 'training_history.png'}")

    logger.info("=" * 50)
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info(f"Modelo guardado en: {output_dir / 'best_model.keras'}")
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Entrena DNN para reconocimiento del alfabeto BSL/LIBRAS (signos estáticos)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directorio con datos preprocesados (subcarpetas por letra)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/models/bsl-alphabet",
        help="Directorio para guardar modelo y artefactos",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()

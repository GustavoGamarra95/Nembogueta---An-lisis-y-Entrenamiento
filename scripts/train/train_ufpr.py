"""
Script para entrenar modelo de clasificación de handshapes UFPR LIBRAS-HC-RGBDS.

Dataset: 61 clases, ~605 muestras (9-10 por clase), secuencias (60, 208).
Estrategia: mean-pooling sobre frames → (208,) estático + DNN con regularización fuerte.

Uso:
    python scripts/train/train_ufpr.py
    python scripts/train/train_ufpr.py --data-dir /data/processed/ufpr-handshape --epochs 200
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ufpr_training.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def load_dataset(data_dir: Path):
    """
    Carga todas las secuencias NPY y sus etiquetas.

    Returns:
        X: (n_samples, 208) — mean-pooled over frames
        y: (n_samples,) — integer labels
        class_names: lista ordenada de nombres de clase
    """
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No se encontraron clases en {data_dir}")

    X_list, y_list = [], []
    class_names = [d.name for d in class_dirs]

    logger.info(f"Cargando {len(class_dirs)} clases desde {data_dir}")

    for class_idx, class_dir in enumerate(class_dirs):
        npy_files = sorted(class_dir.glob("*.npy"))
        for npy_file in npy_files:
            try:
                seq = np.load(npy_file)  # (n_frames, 208)
                # Mean-pool sobre frames → feature estático (208,)
                features = seq.mean(axis=0)
                X_list.append(features)
                y_list.append(class_idx)
            except Exception as e:
                logger.warning(f"Error cargando {npy_file}: {e}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    logger.info(f"Dataset: X={X.shape}, y={y.shape}, clases={len(class_names)}")
    return X, y, class_names


def build_model(n_features: int, n_classes: int, dropout: float, lr: float) -> keras.Model:
    """
    DNN con regularización fuerte para datasets pequeños.
    """
    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.BatchNormalization(),

        layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout),

        layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout),

        layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout / 2),

        layers.Dense(n_classes, activation='softmax'),
    ], name="ufpr_handshape_classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Entrena clasificador de handshapes UFPR"
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("/data/processed/ufpr-handshape"),
        help="Directorio con los NPY procesados (class_XX/)"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/data/models/ufpr-handshape"),
        help="Directorio de salida para el modelo"
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--val-size", type=float, default=0.15,
                        help="Fracción para validación")
    parser.add_argument("--test-size", type=float, default=0.15,
                        help="Fracción para test")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--gpu", action="store_true", help="Usar GPU")
    args = parser.parse_args()

    # GPU
    if not args.gpu:
        tf.config.set_visible_devices([], 'GPU')
        logger.info("Usando CPU")
    else:
        gpus = tf.config.list_physical_devices('GPU')
        logger.info(f"GPU: {gpus}")

    if not args.data_dir.exists():
        logger.error(f"Directorio no encontrado: {args.data_dir}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar datos
    X, y, class_names = load_dataset(args.data_dir)
    n_features = X.shape[1]
    n_classes = len(class_names)

    # Split estratificado: train / val / test
    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=42
    )
    train_val_idx, test_idx = next(sss_test.split(X, y))
    X_train_val, y_train_val = X[train_val_idx], y[train_val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    val_fraction = args.val_size / (1 - args.test_size)
    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=42
    )
    train_idx, val_idx = next(sss_val.split(X_train_val, y_train_val))
    X_train, y_train = X_train_val[train_idx], y_train_val[train_idx]
    X_val, y_val = X_train_val[val_idx], y_train_val[val_idx]

    logger.info(f"\nSplits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    logger.info(f"Features: {n_features}, Clases: {n_classes}")

    # Modelo
    model = build_model(n_features, n_classes, args.dropout, args.lr)
    model.summary(print_fn=logger.info)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=args.patience,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(args.output_dir / "best_model.keras"),
            monitor='val_accuracy', save_best_only=True, verbose=1
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
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    logger.info(f"\nResultados finales:")
    logger.info(f"  Train — loss: {train_loss:.4f}, acc: {train_acc:.4f}")
    logger.info(f"  Val   — loss: {val_loss:.4f}, acc: {val_acc:.4f}")
    logger.info(f"  Test  — loss: {test_loss:.4f}, acc: {test_acc:.4f}")

    # Guardar modelo y metadata
    model.save(args.output_dir / "final_model.keras")

    metadata = {
        "dataset": "UFPR LIBRAS-HC-RGBDS",
        "n_classes": n_classes,
        "class_names": class_names,
        "n_features": n_features,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "input_strategy": "mean_pooling_over_frames",
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
    }
    with open(args.output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"\nModelo guardado en: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Script para entrenar modelo de traducción portugués-LIBRAS (glosas).
Utiliza arquitectura Transformer para traducción secuencia a secuencia.
"""
import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable


# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('translation_training.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    """Embedding con encoding posicional. Totalmente serializable."""
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable()
class TransformerBlock(layers.Layer):
    """Bloque Transformer completo. Totalmente serializable."""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn_dense1 = layers.Dense(ff_dim, activation="relu")
        self.ffn_dense2 = layers.Dense(embed_dim)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn_dense1(out1)
        ffn_output = self.ffn_dense2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TranslationDataLoader:
    """Cargador de datos para traducción."""
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.vocab_pt = self._load_vocab("vocab_pt_br.json")
        self.vocab_gloss = self._load_vocab("vocab_libras_gloss.json")
        logger.info(f"DataLoader inicializado: {data_dir}")
        logger.info(f"Vocab PT: {len(self.vocab_pt)} tokens")
        logger.info(f"Vocab Gloss: {len(self.vocab_gloss)} tokens")

    def _load_vocab(self, filename: str) -> Dict[str, int]:
        vocab_file = self.data_dir / filename
        with open(vocab_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_split(self, split: str) -> Tuple[List[List[int]], List[List[int]]]:
        split_file = self.data_dir / f"{split}.json"
        with open(split_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        encoder_inputs = []
        decoder_targets = []

        for item in data:
            encoder_inputs.append(item['pt_ids'])
            decoder_targets.append([self.vocab_gloss['<SOS>']] + item['gloss_ids'])

        logger.info(f"Cargado {split}: {len(data)} muestras")
        return encoder_inputs, decoder_targets


def pad_sequences(sequences: List[List[int]], maxlen: int, pad_value: int = 0) -> np.ndarray:
    padded = np.full((len(sequences), maxlen), pad_value, dtype=np.int32)
    for i, seq in enumerate(sequences):
        length = min(len(seq), maxlen)
        padded[i, :length] = seq[:length]
    return padded


def build_transformer_model(
    vocab_size_pt: int,
    vocab_size_gloss: int,
    max_seq_length: int = 100,
    embed_dim: int = 256,
    num_heads: int = 8,
    ff_dim: int = 512,
    num_layers: int = 4,
    dropout: float = 0.1,
    learning_rate: float = 0.0001
) -> keras.Model:
    # Encoder
    encoder_inputs = keras.Input(shape=(None,), dtype="int32", name="encoder_inputs")
    x = PositionalEmbedding(max_seq_length, vocab_size_pt, embed_dim)(encoder_inputs)
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)(x)
    encoder_outputs = x

    # Decoder
    decoder_inputs = keras.Input(shape=(None,), dtype="int32", name="decoder_inputs")
    x = PositionalEmbedding(max_seq_length, vocab_size_gloss, embed_dim)(decoder_inputs)
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)(x)

    x = layers.Dropout(dropout)(x)
    decoder_outputs = layers.Dense(vocab_size_gloss, activation="softmax")(x)

    model = keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs,
                        name="transformer_ptbr_to_libras")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Entrena modelo de traducción PT-BR → LIBRAS glosas")
    parser.add_argument("--data-dir", type=Path, default=Path("/data/processed/pt_br2libras"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data/models/translation"))
    parser.add_argument("--max-seq-length", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    # GPU
    if args.gpu:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"GPU disponible: {gpus}")
        else:
            logger.warning("Flag --gpu activado pero no hay GPU disponible")
    else:
        tf.config.set_visible_devices([], 'GPU')
        logger.info("Entrenando en CPU")

    if not args.data_dir.exists():
        logger.error(f"Directorio de datos no encontrado: {args.data_dir}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar datos
    logger.info("Cargando datos...")
    loader = TranslationDataLoader(args.data_dir)
    train_enc, train_dec = loader.load_split("train")
    val_enc, val_dec = loader.load_split("val")

    # Padding
    logger.info("Aplicando padding...")
    train_enc_pad = pad_sequences(train_enc, args.max_seq_length)
    train_dec_pad = pad_sequences(train_dec, args.max_seq_length)
    val_enc_pad = pad_sequences(val_enc, args.max_seq_length)
    val_dec_pad = pad_sequences(val_dec, args.max_seq_length)

    train_dec_input = train_dec_pad[:, :-1]
    train_dec_target = train_dec_pad[:, 1:]
    val_dec_input = val_dec_pad[:, :-1]
    val_dec_target = val_dec_pad[:, 1:]

    # Construir modelo
    logger.info("Construyendo modelo Transformer...")
    model = build_transformer_model(
        vocab_size_pt=len(loader.vocab_pt),
        vocab_size_gloss=len(loader.vocab_gloss),
        max_seq_length=args.max_seq_length,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate
    )
    model.summary(print_fn=logger.info)

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(
            filepath=str(args.output_dir / "best_model.keras"),
            monitor='val_loss', save_best_only=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                          patience=3, min_lr=1e-7, verbose=1),
        keras.callbacks.CSVLogger(str(args.output_dir / "training_history.csv"))
    ]

    # Entrenamiento
    logger.info("Iniciando entrenamiento...")
    model.fit(
        [train_enc_pad, train_dec_input], train_dec_target,
        validation_data=([val_enc_pad, val_dec_input], val_dec_target),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluación final
    train_res = model.evaluate([train_enc_pad, train_dec_input], train_dec_target, verbose=0)
    val_res = model.evaluate([val_enc_pad, val_dec_input], val_dec_target, verbose=0)

    logger.info(f"Train Loss: {train_res[0]:.4f} - Acc: {train_res[1]:.4f}")
    logger.info(f"Val   Loss: {val_res[0]:.4f} - Acc: {val_res[1]:.4f}")

    # Guardar todo
    model.save(args.output_dir / "final_model.keras")
    shutil.copy(args.data_dir / "vocab_pt_br.json", args.output_dir / "vocab_pt_br.json")
    shutil.copy(args.data_dir / "vocab_libras_gloss.json", args.output_dir / "vocab_libras_gloss.json")

    metadata = {
        "max_seq_length": args.max_seq_length,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "ff_dim": args.ff_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "final_train_loss": float(train_res[0]),
        "final_train_acc": float(train_res[1]),
        "final_val_loss": float(val_res[0]),
        "final_val_acc": float(val_res[1]),
        "train_samples": len(train_enc),
        "val_samples": len(val_enc),
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir)
    }
    with open(args.output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("Entrenamiento completado con éxito! Modelo guardado en:")
    logger.info(f"  {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
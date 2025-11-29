"""
Script para evaluar el modelo de traducción PT-BR → LIBRAS.
Calcula métricas como BLEU, accuracy, y otras métricas de traducción.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable
from tqdm import tqdm


# Registrar capas personalizadas
@register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    """Embedding con encoding posicional."""
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


@register_keras_serializable()
class TransformerBlock(layers.Layer):
    """Bloque Transformer."""
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


class TranslationEvaluator:
    """Evaluador de modelos de traducción."""

    def __init__(self, model_dir: Path, data_dir: Path):
        """
        Inicializa el evaluador.

        Args:
            model_dir: Directorio con el modelo entrenado
            data_dir: Directorio con datos preprocesados
        """
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)

        print("\n" + "=" * 70)
        print("EVALUACIÓN: Modelo de Traducción PT-BR → LIBRAS")
        print("=" * 70)

        # Cargar modelo
        print("\nCargando modelo...")
        self.model = keras.models.load_model(self.model_dir / "best_model.keras")
        print("  ✓ Modelo cargado")

        # Cargar vocabularios
        print("Cargando vocabularios...")
        with open(self.model_dir / "vocab_pt_br.json", 'r', encoding='utf-8') as f:
            self.vocab_pt = json.load(f)

        with open(self.model_dir / "vocab_libras_gloss.json", 'r', encoding='utf-8') as f:
            self.vocab_gloss = json.load(f)

        self.inv_vocab_gloss = {v: k for k, v in self.vocab_gloss.items()}
        print(f"  ✓ Vocabularios cargados")

        # Cargar metadata
        with open(self.model_dir / "metadata.json", 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            self.max_seq_length = self.metadata.get('max_seq_length', 100)

    def load_test_data(self) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Carga datos de test.

        Returns:
            Tupla (encoder_inputs, decoder_targets)
        """
        test_file = self.data_dir / "test.json"

        if not test_file.exists():
            print(f"⚠ Archivo de test no encontrado: {test_file}")
            return [], []

        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        encoder_inputs = [item['pt_ids'] for item in data]
        decoder_targets = [item['gloss_ids'] for item in data]

        print(f"\n✓ Datos de test cargados: {len(data)} muestras")
        return encoder_inputs, decoder_targets

    def pad_sequence(self, sequence: List[int], maxlen: int) -> np.ndarray:
        """Aplica padding a una secuencia."""
        if len(sequence) < maxlen:
            sequence = sequence + [0] * (maxlen - len(sequence))
        else:
            sequence = sequence[:maxlen]
        return np.array(sequence)

    def translate_batch(self, encoder_inputs: np.ndarray, max_length: int = 100) -> List[List[int]]:
        """
        Traduce un batch de secuencias.

        Args:
            encoder_inputs: Inputs del encoder (batch_size, seq_len)
            max_length: Longitud máxima de salida

        Returns:
            Lista de secuencias traducidas
        """
        batch_size = encoder_inputs.shape[0]
        decoder_input = np.full((batch_size, 1), self.vocab_gloss.get('<SOS>', 1))

        translations = [[] for _ in range(batch_size)]
        finished = np.zeros(batch_size, dtype=bool)

        for _ in range(max_length):
            if finished.all():
                break

            predictions = self.model.predict(
                [encoder_inputs, decoder_input],
                verbose=0
            )

            next_tokens = np.argmax(predictions[:, -1, :], axis=1)

            for i in range(batch_size):
                if not finished[i]:
                    if next_tokens[i] == self.vocab_gloss.get('<EOS>', 2):
                        finished[i] = True
                    else:
                        translations[i].append(next_tokens[i])

            decoder_input = np.concatenate([
                decoder_input,
                next_tokens.reshape(-1, 1)
            ], axis=1)

        return translations

    def calculate_accuracy(self, predictions: List[List[int]], targets: List[List[int]]) -> float:
        """
        Calcula accuracy exacta (secuencia completa correcta).

        Args:
            predictions: Secuencias predichas
            targets: Secuencias objetivo

        Returns:
            Accuracy
        """
        correct = 0
        for pred, target in zip(predictions, targets):
            # Remover tokens especiales de target
            target_clean = [t for t in target if t not in [0, 1, 2]]  # <PAD>, <SOS>, <EOS>
            if pred == target_clean:
                correct += 1

        return correct / len(predictions) if predictions else 0.0

    def calculate_token_accuracy(self, predictions: List[List[int]], targets: List[List[int]]) -> float:
        """
        Calcula accuracy a nivel de token.

        Args:
            predictions: Secuencias predichas
            targets: Secuencias objetivo

        Returns:
            Token accuracy
        """
        total_tokens = 0
        correct_tokens = 0

        for pred, target in zip(predictions, targets):
            target_clean = [t for t in target if t not in [0, 1, 2]]

            for i, token in enumerate(target_clean):
                total_tokens += 1
                if i < len(pred) and pred[i] == token:
                    correct_tokens += 1

        return correct_tokens / total_tokens if total_tokens > 0 else 0.0

    def calculate_bleu_simple(self, predictions: List[List[int]], targets: List[List[int]]) -> float:
        """
        Calcula BLEU score simplificado (unigram precision).

        Args:
            predictions: Secuencias predichas
            targets: Secuencias objetivo

        Returns:
            BLEU score simplificado
        """
        total_precision = 0.0

        for pred, target in zip(predictions, targets):
            target_clean = set([t for t in target if t not in [0, 1, 2]])
            pred_set = set(pred)

            if len(pred_set) > 0:
                matches = len(pred_set & target_clean)
                precision = matches / len(pred_set)
                total_precision += precision

        return total_precision / len(predictions) if predictions else 0.0

    def evaluate(self, batch_size: int = 32) -> Dict:
        """
        Evalúa el modelo en datos de test.

        Args:
            batch_size: Tamaño del batch

        Returns:
            Diccionario con métricas
        """
        # Cargar datos
        encoder_inputs, decoder_targets = self.load_test_data()

        if not encoder_inputs:
            print("⚠ No hay datos de test para evaluar")
            return {}

        # Aplicar padding
        print("\nAplicando padding...")
        encoder_padded = np.array([
            self.pad_sequence(seq, self.max_seq_length)
            for seq in encoder_inputs
        ])

        # Traducir en batches
        print(f"Traduciendo {len(encoder_inputs)} muestras...")
        all_predictions = []

        for i in tqdm(range(0, len(encoder_padded), batch_size), desc="Evaluando"):
            batch = encoder_padded[i:i+batch_size]
            batch_preds = self.translate_batch(batch)
            all_predictions.extend(batch_preds)

        # Calcular métricas
        print("\nCalculando métricas...")
        metrics = {
            'sequence_accuracy': self.calculate_accuracy(all_predictions, decoder_targets),
            'token_accuracy': self.calculate_token_accuracy(all_predictions, decoder_targets),
            'bleu_simple': self.calculate_bleu_simple(all_predictions, decoder_targets),
            'total_samples': len(encoder_inputs)
        }

        # Calcular estadísticas de longitud
        pred_lengths = [len(p) for p in all_predictions]
        target_lengths = [len([t for t in tgt if t not in [0, 1, 2]]) for tgt in decoder_targets]

        metrics['avg_pred_length'] = np.mean(pred_lengths)
        metrics['avg_target_length'] = np.mean(target_lengths)
        metrics['length_ratio'] = np.mean(pred_lengths) / np.mean(target_lengths)

        return metrics

    def print_examples(self, n: int = 10):
        """
        Imprime ejemplos de traducción.

        Args:
            n: Número de ejemplos
        """
        # Cargar datos
        encoder_inputs, decoder_targets = self.load_test_data()

        if not encoder_inputs:
            return

        print("\n" + "=" * 70)
        print(f"EJEMPLOS DE TRADUCCIÓN (primeros {n})")
        print("=" * 70)

        # Tomar primeros n ejemplos
        sample_inputs = encoder_inputs[:n]
        sample_targets = decoder_targets[:n]

        # Aplicar padding
        encoder_padded = np.array([
            self.pad_sequence(seq, self.max_seq_length)
            for seq in sample_inputs
        ])

        # Traducir
        predictions = self.translate_batch(encoder_padded)

        # Mostrar
        for i, (inp, pred, target) in enumerate(zip(sample_inputs, predictions, sample_targets), 1):
            # Convertir a palabras
            inp_words = [self.get_pt_word(t) for t in inp if t != 0]
            pred_words = [self.get_gloss_word(t) for t in pred]
            target_words = [self.get_gloss_word(t) for t in target if t not in [0, 1, 2]]

            print(f"\n{i}.")
            print(f"  PT-BR: {' '.join(inp_words)}")
            print(f"  Predicción: {' '.join(pred_words)}")
            print(f"  Target: {' '.join(target_words)}")
            print(f"  Match: {'✓' if pred == [t for t in target if t not in [0, 1, 2]] else '✗'}")

        print("\n" + "=" * 70)

    def get_pt_word(self, token_id: int) -> str:
        """Obtiene palabra PT a partir de ID."""
        inv_vocab_pt = {v: k for k, v in self.vocab_pt.items()}
        return inv_vocab_pt.get(token_id, '<UNK>')

    def get_gloss_word(self, token_id: int) -> str:
        """Obtiene glosa a partir de ID."""
        return self.inv_vocab_gloss.get(token_id, '<UNK>')


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Evalúa modelo de traducción PT-BR → LIBRAS"
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/app/data/models/translation"),
        help="Directorio con el modelo entrenado"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/app/data/processed/pt_br2libras"),
        help="Directorio con datos preprocesados"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamaño del batch para evaluación"
    )

    parser.add_argument(
        "--examples",
        type=int,
        default=10,
        help="Número de ejemplos a mostrar"
    )

    parser.add_argument(
        "--no-examples",
        action="store_true",
        help="No mostrar ejemplos"
    )

    args = parser.parse_args()

    # Verificar que el modelo existe
    if not (args.model_dir / "best_model.keras").exists():
        print(f"Error: No se encontró modelo en {args.model_dir}")
        return 1

    # Inicializar evaluador
    evaluator = TranslationEvaluator(args.model_dir, args.data_dir)

    # Evaluar
    metrics = evaluator.evaluate(batch_size=args.batch_size)

    if not metrics:
        return 1

    # Imprimir resultados
    print("\n" + "=" * 70)
    print("RESULTADOS DE EVALUACIÓN")
    print("=" * 70)
    print(f"\nMuestras evaluadas: {metrics['total_samples']}")
    print(f"\nMétricas:")
    print(f"  Sequence Accuracy: {metrics['sequence_accuracy']:.4f} ({100*metrics['sequence_accuracy']:.2f}%)")
    print(f"  Token Accuracy: {metrics['token_accuracy']:.4f} ({100*metrics['token_accuracy']:.2f}%)")
    print(f"  BLEU (simple): {metrics['bleu_simple']:.4f}")
    print(f"\nEstadísticas de longitud:")
    print(f"  Longitud promedio predicción: {metrics['avg_pred_length']:.2f}")
    print(f"  Longitud promedio target: {metrics['avg_target_length']:.2f}")
    print(f"  Ratio longitud: {metrics['length_ratio']:.2f}")
    print("=" * 70)

    # Mostrar ejemplos
    if not args.no_examples:
        evaluator.print_examples(n=args.examples)

    return 0


if __name__ == "__main__":
    sys.exit(main())

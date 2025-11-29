"""
Script de demostración para el modelo de traducción PT-BR → LIBRAS.
Permite traducir texto interactivamente.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable


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


class TranslationDemo:
    """Demostrador interactivo de traducción PT-BR → LIBRAS."""

    def __init__(self, model_dir: Path):
        """
        Inicializa el demostrador.

        Args:
            model_dir: Directorio con el modelo entrenado
        """
        self.model_dir = Path(model_dir)

        print("\n" + "=" * 70)
        print("DEMO: Traducción Portugués Brasileño → LIBRAS (Glosas)")
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

        # Crear vocabulario inverso
        self.inv_vocab_gloss = {v: k for k, v in self.vocab_gloss.items()}

        print(f"  ✓ Vocabulario PT: {len(self.vocab_pt)} tokens")
        print(f"  ✓ Vocabulario Glosas: {len(self.vocab_gloss)} tokens")

        # Cargar metadata
        try:
            with open(self.model_dir / "metadata.json", 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
                self.max_seq_length = self.metadata.get('max_seq_length', 100)
        except FileNotFoundError:
            print("  ⚠ metadata.json no encontrado, usando max_seq_length=100")
            self.max_seq_length = 100

        print(f"  ✓ Max sequence length: {self.max_seq_length}")
        print("\n" + "=" * 70)

    def tokenize_text(self, text: str) -> List[int]:
        """
        Tokeniza texto en portugués.

        Args:
            text: Texto en portugués

        Returns:
            Lista de IDs de tokens
        """
        tokens = text.lower().split()
        token_ids = [self.vocab_pt.get(t, self.vocab_pt.get('<UNK>', 0)) for t in tokens]
        return token_ids

    def pad_sequence(self, sequence: List[int]) -> np.ndarray:
        """
        Aplica padding a una secuencia.

        Args:
            sequence: Secuencia de tokens

        Returns:
            Array con padding
        """
        if len(sequence) < self.max_seq_length:
            sequence = sequence + [0] * (self.max_seq_length - len(sequence))
        else:
            sequence = sequence[:self.max_seq_length]
        return np.array([sequence])

    def translate(self, text_pt: str, max_output_length: int = 100) -> List[str]:
        """
        Traduce texto de portugués a glosas LIBRAS.

        Args:
            text_pt: Texto en portugués
            max_output_length: Longitud máxima de salida

        Returns:
            Lista de glosas en LIBRAS
        """
        # Tokenizar y hacer padding del input
        token_ids = self.tokenize_text(text_pt)
        encoder_input = self.pad_sequence(token_ids)

        # Inicializar decoder con token <SOS>
        decoder_input = np.array([[self.vocab_gloss.get('<SOS>', 1)]])

        # Generar glosas (greedy decoding)
        generated = []

        for _ in range(max_output_length):
            # Predecir siguiente token
            predictions = self.model.predict(
                [encoder_input, decoder_input],
                verbose=0
            )

            # Obtener token con mayor probabilidad
            next_token = np.argmax(predictions[0, -1, :])

            # Si es token <EOS>, terminar
            if next_token == self.vocab_gloss.get('<EOS>', 2):
                break

            # Obtener palabra correspondiente
            next_word = self.inv_vocab_gloss.get(next_token, '<UNK>')

            # Evitar tokens especiales en la salida
            if next_word not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']:
                generated.append(next_word)

            # Actualizar decoder input
            decoder_input = np.concatenate([
                decoder_input,
                [[next_token]]
            ], axis=1)

        return generated

    def run_interactive(self):
        """Ejecuta el modo interactivo."""
        print("\nModo Interactivo")
        print("-" * 70)
        print("Escribe frases en portugués para traducir a glosas LIBRAS")
        print("Comandos especiales:")
        print("  - 'exit' o 'quit': Salir")
        print("  - 'exemplos': Mostrar ejemplos")
        print("-" * 70)

        while True:
            try:
                print("\n" + "=" * 70)
                text = input("PT-BR → ").strip()

                if not text:
                    continue

                if text.lower() in ['exit', 'quit', 'sair']:
                    print("\n¡Hasta luego!")
                    break

                if text.lower() in ['exemplos', 'examples']:
                    self.show_examples()
                    continue

                # Traducir
                print("\nTraduciendo...")
                gloss = self.translate(text)

                # Mostrar resultado
                print("\n" + "-" * 70)
                print(f"Texto PT-BR: {text}")
                print(f"Glosas LIBRAS: {' '.join(gloss)}")
                print("-" * 70)

                # Análisis
                print(f"\nAnálisis:")
                print(f"  - Palabras PT-BR: {len(text.split())}")
                print(f"  - Glosas LIBRAS: {len(gloss)}")
                print(f"  - Ratio: {len(gloss) / len(text.split()):.2f}")

            except KeyboardInterrupt:
                print("\n\n¡Hasta luego!")
                break
            except Exception as e:
                print(f"\n⚠ Error: {e}")

    def show_examples(self):
        """Muestra ejemplos de traducción."""
        examples = [
            "Olá, como você está?",
            "Eu gosto de estudar",
            "Qual é o seu nome?",
            "Obrigado pela ajuda",
            "Até logo"
        ]

        print("\n" + "=" * 70)
        print("EJEMPLOS DE TRADUCCIÓN")
        print("=" * 70)

        for i, example in enumerate(examples, 1):
            gloss = self.translate(example)
            print(f"\n{i}. PT-BR: {example}")
            print(f"   LIBRAS: {' '.join(gloss)}")

        print("\n" + "=" * 70)

    def batch_translate(self, texts: List[str], output_file: Path = None):
        """
        Traduce múltiples textos en lote.

        Args:
            texts: Lista de textos en portugués
            output_file: Archivo opcional para guardar resultados
        """
        results = []

        print("\n" + "=" * 70)
        print(f"TRADUCCIÓN EN LOTE - {len(texts)} textos")
        print("=" * 70)

        for i, text in enumerate(texts, 1):
            gloss = self.translate(text)
            results.append({
                'id': i,
                'text_pt': text,
                'gloss_libras': ' '.join(gloss),
                'gloss_list': gloss
            })

            print(f"\n{i}/{len(texts)}")
            print(f"  PT-BR: {text}")
            print(f"  LIBRAS: {' '.join(gloss)}")

        # Guardar resultados si se especifica archivo
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Resultados guardados en: {output_file}")

        print("\n" + "=" * 70)

        return results


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Demo de traducción PT-BR → LIBRAS (glosas)"
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/app/data/models/translation"),
        help="Directorio con el modelo entrenado"
    )

    parser.add_argument(
        "--text",
        type=str,
        help="Texto a traducir (modo single)"
    )

    parser.add_argument(
        "--examples",
        action="store_true",
        help="Mostrar ejemplos de traducción"
    )

    parser.add_argument(
        "--batch",
        type=Path,
        help="Archivo con textos para traducción en lote (uno por línea)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Archivo de salida para traducción en lote"
    )

    args = parser.parse_args()

    # Verificar que el modelo existe
    if not (args.model_dir / "best_model.keras").exists():
        print(f"Error: No se encontró modelo en {args.model_dir}")
        return 1

    # Inicializar demo
    demo = TranslationDemo(args.model_dir)

    # Modo single
    if args.text:
        gloss = demo.translate(args.text)
        print("\n" + "=" * 70)
        print(f"Texto PT-BR: {args.text}")
        print(f"Glosas LIBRAS: {' '.join(gloss)}")
        print("=" * 70)
        return 0

    # Mostrar ejemplos
    if args.examples:
        demo.show_examples()
        return 0

    # Modo batch
    if args.batch:
        if not args.batch.exists():
            print(f"Error: Archivo no encontrado: {args.batch}")
            return 1

        with open(args.batch, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

        demo.batch_translate(texts, args.output)
        return 0

    # Modo interactivo (por defecto)
    demo.run_interactive()

    return 0


if __name__ == "__main__":
    sys.exit(main())

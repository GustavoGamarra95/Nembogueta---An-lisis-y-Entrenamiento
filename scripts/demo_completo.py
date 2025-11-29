"""
Demo Completo del Sistema LIBRAS
Integra todos los modelos entrenados en una demostración interactiva.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable
import mediapipe as mp


# ============================================================================
# CAPAS PERSONALIZADAS
# ============================================================================

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


# ============================================================================
# SISTEMA DEMO COMPLETO
# ============================================================================

class LibrasDemoCompleto:
    """Demo completo del sistema LIBRAS."""

    def __init__(self, models_dir: Path):
        """
        Inicializa el demo con todos los modelos.

        Args:
            models_dir: Directorio raíz con todos los modelos
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.metadata = {}

        print("\n" + "=" * 80)
        print("DEMO COMPLETO - SISTEMA LIBRAS")
        print("=" * 80)

        # Cargar modelos
        self._load_handshape_models()
        self._load_translation_model()

        # Inicializar MediaPipe
        print("\nInicializando MediaPipe...")
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Para imágenes estáticas
        self.hands_static = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )

        # Para video en tiempo real
        self.hands_video = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        print("  ✓ MediaPipe inicializado")

        print("\n" + "=" * 80)

    def _load_handshape_models(self):
        """Carga modelos de formas de mano."""
        print("\nCargando modelos de formas de mano...")

        views = ['front', 'right', 'left', 'back']
        for view in views:
            model_path = self.models_dir / "handshape" / view / "best_model.keras"
            metadata_path = self.models_dir / "handshape" / view / "metadata.json"

            if model_path.exists():
                try:
                    self.models[f'handshape_{view}'] = keras.models.load_model(model_path)

                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        self.metadata[f'handshape_{view}'] = json.load(f)

                    acc = self.metadata[f'handshape_{view}']['final_test_acc']
                    print(f"  ✓ Vista {view}: {acc:.2%} accuracy")
                except Exception as e:
                    print(f"  ✗ Error cargando vista {view}: {e}")

    def _load_translation_model(self):
        """Carga modelo de traducción."""
        print("\nCargando modelo de traducción...")

        model_path = self.models_dir / "translation" / "best_model.keras"

        if model_path.exists():
            try:
                self.models['translation'] = keras.models.load_model(model_path)

                # Cargar vocabularios
                with open(self.models_dir / "translation" / "vocab_pt_br.json", 'r') as f:
                    self.vocab_pt = json.load(f)

                with open(self.models_dir / "translation" / "vocab_libras_gloss.json", 'r') as f:
                    self.vocab_gloss = json.load(f)

                self.inv_vocab_gloss = {v: k for k, v in self.vocab_gloss.items()}

                with open(self.models_dir / "translation" / "metadata.json", 'r') as f:
                    self.metadata['translation'] = json.load(f)

                print(f"  ✓ Modelo de traducción cargado")
                print(f"  ✓ Vocab PT: {len(self.vocab_pt)} | Vocab LIBRAS: {len(self.vocab_gloss)}")
            except Exception as e:
                print(f"  ✗ Error cargando traducción: {e}")

    def extract_hand_landmarks(self, image: np.ndarray, static: bool = True) -> Optional[np.ndarray]:
        """
        Extrae landmarks de la mano.

        Args:
            image: Imagen BGR
            static: Si True, usa modelo para imágenes estáticas, si False usa modelo de video

        Returns:
            Array con 63 valores (21 landmarks x 3 coordenadas)
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hands_processor = self.hands_static if static else self.hands_video
        results = hands_processor.process(image_rgb)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            coords = []
            for lm in landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            return np.array(coords, dtype=np.float32)

        return None

    def predict_handshape(self, landmarks: np.ndarray, view: str = 'front') -> Dict:
        """Predice forma de mano."""
        model_key = f'handshape_{view}'

        if model_key not in self.models:
            return {'error': f'Modelo {view} no disponible'}

        landmarks_reshaped = landmarks.reshape(1, -1)
        predictions = self.models[model_key].predict(landmarks_reshaped, verbose=0)

        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]

        class_names = self.metadata[model_key].get('class_names', [])
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class_{class_idx}"

        # Top 3 predicciones
        top3_indices = np.argsort(predictions[0])[-3:][::-1]
        top3 = []
        for idx in top3_indices:
            name = class_names[idx] if idx < len(class_names) else f"Class_{idx}"
            conf = predictions[0][idx]
            top3.append({'class': name, 'confidence': float(conf)})

        return {
            'class': class_name,
            'confidence': float(confidence),
            'top3': top3
        }

    def translate_text(self, text: str) -> Dict:
        """Traduce texto PT-BR a LIBRAS."""
        if 'translation' not in self.models:
            return {'error': 'Modelo de traducción no disponible'}

        # Tokenizar
        tokens = text.lower().split()
        token_ids = [self.vocab_pt.get(t, self.vocab_pt.get('<UNK>', 0)) for t in tokens]

        # Padding
        max_len = self.metadata['translation'].get('max_seq_length', 100)
        if len(token_ids) < max_len:
            token_ids += [0] * (max_len - len(token_ids))
        else:
            token_ids = token_ids[:max_len]

        encoder_input = np.array([token_ids])
        decoder_input = np.array([[self.vocab_gloss.get('<SOS>', 1)]])

        # Generar glosas
        generated = []
        generated_ids = []

        for _ in range(50):  # Límite de 50 tokens
            pred = self.models['translation'].predict(
                [encoder_input, decoder_input],
                verbose=0
            )
            next_token = np.argmax(pred[0, -1, :])

            if next_token == self.vocab_gloss.get('<EOS>', 2):
                break

            word = self.inv_vocab_gloss.get(next_token, '<UNK>')

            # Detectar loops de repetición
            if len(generated) >= 3 and all(w == word for w in generated[-3:]):
                break

            if word not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']:
                generated.append(word)
                generated_ids.append(int(next_token))

            decoder_input = np.concatenate([
                decoder_input,
                [[next_token]]
            ], axis=1)

        return {
            'input_text': text,
            'glosas': generated,
            'glosas_text': ' '.join(generated),
            'n_tokens': len(generated)
        }

    def menu_principal(self):
        """Menú principal interactivo."""
        while True:
            print("\n" + "=" * 80)
            print("MENÚ PRINCIPAL")
            print("=" * 80)
            print("1. Reconocimiento de Formas de Mano (desde imagen)")
            print("2. Traducción PT-BR → LIBRAS")
            print("3. Reconocimiento en Tiempo Real (Cámara) - TODAS LAS VISTAS")
            print("4. Información de Modelos")
            print("5. Probar con frases del dataset")
            print("6. Salir")
            print("=" * 80)

            try:
                opcion = input("\nSeleccione opción: ").strip()

                if opcion == '1':
                    self.demo_handshape()
                elif opcion == '2':
                    self.demo_translation()
                elif opcion == '3':
                    self.demo_realtime_camera()
                elif opcion == '4':
                    self.show_model_info()
                elif opcion == '5':
                    self.test_dataset_samples()
                elif opcion == '6':
                    print("\n¡Hasta luego!")
                    break
                else:
                    print("⚠ Opción inválida")

            except KeyboardInterrupt:
                print("\n\n¡Hasta luego!")
                break
            except Exception as e:
                print(f"⚠ Error: {e}")

    def demo_realtime_camera(self):
        """Demo de reconocimiento en tiempo real con cámara."""
        print("\n" + "=" * 80)
        print("DEMO: Reconocimiento en Tiempo Real - TODAS LAS VISTAS")
        print("=" * 80)
        print("Controles:")
        print("  - Presione 'q' para salir")
        print("  - Presione 'f' para cambiar vista principal (front/right/left/back)")
        print("  - Presione ESPACIO para pausar/reanudar")
        print("=" * 80)
        input("\nPresione ENTER para iniciar la cámara...")

        # Verificar modelos disponibles
        available_views = [v for v in ['front', 'right', 'left', 'back']
                          if f'handshape_{v}' in self.models]

        if not available_views:
            print("⚠ No hay modelos de formas de mano disponibles")
            return

        print(f"\nVistas disponibles: {', '.join(available_views)}")

        # Inicializar cámara
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("⚠ No se pudo abrir la cámara")
            return

        print("\n✓ Cámara iniciada")
        print("Mostrando predicciones de todas las vistas en la ventana...")

        current_view_idx = 0
        paused = False
        frame = None

        try:
            while cap.isOpened():
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("⚠ Error al capturar frame")
                        break

                    # Voltear horizontalmente para efecto espejo
                    frame = cv2.flip(frame, 1)
                    h, w, c = frame.shape

                    # Procesar con MediaPipe (modo video)
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.hands_video.process(image_rgb)

                    # Dibujar landmarks
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                self.mp_drawing_styles.get_default_hand_connections_style()
                            )

                        # Extraer landmarks del primer hand detectado
                        landmarks = []
                        for lm in results.multi_hand_landmarks[0].landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])
                        landmarks = np.array(landmarks, dtype=np.float32)

                        # Predecir con TODAS las vistas
                        y_offset = 30
                        for view in available_views:
                            result = self.predict_handshape(landmarks, view)

                            # Color según vista
                            if view == available_views[current_view_idx]:
                                color = (0, 255, 0)  # Verde para vista principal
                                thickness = 2
                            else:
                                color = (255, 255, 255)  # Blanco para otras
                                thickness = 1

                            # Mostrar predicción
                            text = f"{view.upper()}: {result['class']} ({result['confidence']:.1%})"
                            cv2.putText(frame, text, (10, y_offset),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
                            y_offset += 25

                    else:
                        cv2.putText(frame, "No se detecta mano", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Mostrar info adicional
                    cv2.putText(frame, f"Vista principal: {available_views[current_view_idx].upper()}",
                              (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, "Q:Salir | F:Cambiar vista | ESPACIO:Pausar",
                              (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Mostrar frame
                cv2.imshow('LIBRAS - Reconocimiento Multi-Vista', frame)

                # Procesar teclas
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('f'):
                    current_view_idx = (current_view_idx + 1) % len(available_views)
                    print(f"Vista principal cambiada a: {available_views[current_view_idx].upper()}")
                elif key == ord(' '):
                    paused = not paused
                    print("Pausado" if paused else "Reanudado")

        except KeyboardInterrupt:
            print("\n\nInterrumpido por el usuario")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\n✓ Cámara cerrada")

    def demo_handshape(self):
        """Demo de reconocimiento de formas de mano."""
        print("\n" + "=" * 80)
        print("DEMO: Reconocimiento de Formas de Mano")
        print("=" * 80)
        print("Ingrese la ruta de una imagen o 'back' para volver")

        path = input("Ruta de imagen: ").strip()

        if path.lower() == 'back':
            return

        if not Path(path).exists():
            print("⚠ Archivo no encontrado")
            return

        # Cargar imagen
        image = cv2.imread(path)
        if image is None:
            print("⚠ Error al cargar imagen")
            return

        # Extraer landmarks
        print("\nExtrayendo landmarks...")
        landmarks = self.extract_hand_landmarks(image)

        if landmarks is None:
            print("⚠ No se detectó mano en la imagen")
            return

        print("✓ Mano detectada")

        # Predecir con todas las vistas disponibles
        print("\nPredicciones:")
        print("-" * 80)

        for view in ['front', 'right', 'left', 'back']:
            if f'handshape_{view}' in self.models:
                result = self.predict_handshape(landmarks, view)
                print(f"\nVista {view.upper()}:")
                print(f"  Clase: {result['class']}")
                print(f"  Confianza: {result['confidence']:.2%}")
                print(f"  Top 3:")
                for i, pred in enumerate(result['top3'], 1):
                    print(f"    {i}. {pred['class']}: {pred['confidence']:.2%}")

    def demo_translation(self):
        """Demo de traducción."""
        print("\n" + "=" * 80)
        print("DEMO: Traducción PT-BR → LIBRAS")
        print("=" * 80)
        print("NOTA: Este modelo solo funciona con frases del dataset de entrenamiento")
        print("Ingrese texto en portugués o 'back' para volver")

        text = input("\nPT-BR: ").strip()

        if text.lower() == 'back':
            return

        if not text:
            return

        print("\nTraduciendo...")
        result = self.translate_text(text)

        if 'error' in result:
            print(f"⚠ {result['error']}")
            return

        print("\n" + "-" * 80)
        print(f"Entrada: {result['input_text']}")
        print(f"Glosas: {result['glosas_text']}")
        print(f"Tokens: {result['n_tokens']}")
        print("-" * 80)

    def show_model_info(self):
        """Muestra información de los modelos."""
        print("\n" + "=" * 80)
        print("INFORMACIÓN DE MODELOS CARGADOS")
        print("=" * 80)

        print("\n1. MODELOS DE FORMAS DE MANO:")
        print("-" * 80)
        for view in ['front', 'right', 'left', 'back']:
            key = f'handshape_{view}'
            if key in self.models:
                meta = self.metadata[key]
                print(f"\nVista {view.upper()}:")
                print(f"  Test Accuracy: {meta['final_test_acc']:.2%}")
                print(f"  Top-5 Accuracy: {meta['final_test_top5']:.2%}")
                print(f"  Clases: {meta['n_classes']}")
                print(f"  Muestras train: {meta['train_samples']}")
                print(f"  Muestras test: {meta['test_samples']}")

        if 'translation' in self.models:
            print("\n2. MODELO DE TRADUCCIÓN:")
            print("-" * 80)
            meta = self.metadata['translation']
            print(f"  Train Accuracy: {meta['final_train_acc']:.2%}")
            print(f"  Val Accuracy: {meta['final_val_acc']:.2%}")
            print(f"  Vocab PT-BR: {len(self.vocab_pt)} tokens")
            print(f"  Vocab LIBRAS: {len(self.vocab_gloss)} glosas")
            print(f"  Muestras train: {meta['train_samples']}")
            print(f"  Muestras val: {meta['val_samples']}")

        print("\n" + "=" * 80)

    def test_dataset_samples(self):
        """Prueba con frases del dataset."""
        print("\n" + "=" * 80)
        print("PRUEBA CON FRASES DEL DATASET")
        print("=" * 80)

        # Cargar algunas frases del dataset
        train_file = Path("/app/data/processed/pt_br2libras/train.json")

        if not train_file.exists():
            print("⚠ Dataset no encontrado")
            return

        with open(train_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Tomar 5 frases aleatorias
        import random
        samples = random.sample(data, min(5, len(data)))

        print("\nEjemplos del dataset:")
        print("-" * 80)

        for i, sample in enumerate(samples, 1):
            print(f"\n{i}. PT-BR: {sample['pt_br']}")
            print(f"   Target: {sample['libras_gloss']}")

            result = self.translate_text(sample['pt_br'])
            print(f"   Predicción: {result['glosas_text']}")

            # Comparar
            target_glosas = sample['libras_gloss'].split()
            pred_glosas = result['glosas']
            match = target_glosas == pred_glosas
            print(f"   Match: {'✓' if match else '✗'}")

        print("\n" + "=" * 80)

    def __del__(self):
        """Liberar recursos."""
        if hasattr(self, 'hands'):
            self.hands.close()


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Demo completo del sistema LIBRAS"
    )

    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("/app/data/models"),
        help="Directorio raíz con todos los modelos"
    )

    args = parser.parse_args()

    # Inicializar demo
    demo = LibrasDemoCompleto(args.models_dir)

    # Ejecutar menú principal
    demo.menu_principal()

    return 0


if __name__ == "__main__":
    sys.exit(main())

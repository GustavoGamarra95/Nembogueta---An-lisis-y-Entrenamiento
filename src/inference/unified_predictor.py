import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable

logger = logging.getLogger(__name__)


@register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )

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
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
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


@register_keras_serializable(package="Custom", name="AttentionLayer")
class AttentionLayer(layers.Layer):
    def __init__(self, units=128, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = None
        self.U = None
        self.V = None

    def build(self, input_shape):
        self.W = layers.Dense(self.units, name='attention_W')
        self.U = layers.Dense(self.units, name='attention_U')
        self.V = layers.Dense(1, name='attention_V')
        super().build(input_shape)

    def call(self, inputs):
        score = tf.nn.tanh(self.W(inputs))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
        })
        return config


class LibrasUnifiedPredictor:
    def __init__(self, models_dir: Path, buffer_size: int = 30):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.metadata = {}

        self.buffer_size = buffer_size
        self.frames_buffer = []
        self.alphabet_predictions_buffer = []
        self.max_prediction_buffer = 5

        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        logger.info("Inicializando predictor unificado...")
        self._load_all_models()

    def _load_all_models(self):
        handshape_dir = self.models_dir / "handshape"
        self.models['handshape'] = {}
        self.metadata['handshape'] = {}

        orientations = ['back', 'front', 'left', 'right']
        loaded_orientations = []

        for orientation in orientations:
            orient_dir = handshape_dir / orientation
            if (orient_dir / "best_model.keras").exists():
                try:
                    model_path = orient_dir / "best_model.keras"
                    self.models['handshape'][orientation] = (
                        tf.keras.models.load_model(model_path)
                    )
                    with open(orient_dir / "metadata.json", 'r') as f:
                        self.metadata['handshape'][orientation] = json.load(f)
                    loaded_orientations.append(orientation)
                except Exception as e:
                    logger.error(
                        f"Error cargando handshape {orientation}: {e}"
                    )

        if loaded_orientations:
            logger.info(
                "Modelos de formas de mano cargados: "
                f"{', '.join(loaded_orientations)}"
            )
            total_classes = sum(
                [m['n_classes'] for m in self.metadata['handshape'].values()]
            )
            logger.info(f"  [OK] {total_classes} clases total")
        else:
            del self.models['handshape']
            del self.metadata['handshape']

        facial_dir = self.models_dir / "facial_expressions"
        if (facial_dir / "best_model.keras").exists():
            logger.info("Cargando modelo de expresiones faciales...")
            try:
                self.models['facial'] = tf.keras.models.load_model(
                    facial_dir / "best_model.keras"
                )
                with open(
                    facial_dir / "metadata.json", 'r', encoding='utf-8'
                ) as f:
                    self.metadata['facial'] = json.load(f)
                n_expr = len(self.metadata['facial']['class_names'])
                logger.info(f"  [OK] {n_expr} expresiones")
            except Exception as e:
                logger.error(f"Error cargando facial expressions: {e}")

        translation_dir = self.models_dir / "translation"
        if (translation_dir / "best_model.keras").exists():
            logger.info("Cargando modelo de traduccion PT-BR -> LIBRAS...")
            try:
                self.models['translation'] = tf.keras.models.load_model(
                    translation_dir / "best_model.keras"
                )
                with open(
                    translation_dir / "vocab_pt_br.json",
                    'r', encoding='utf-8'
                ) as f:
                    self.vocab_pt = json.load(f)
                with open(
                    translation_dir / "vocab_libras_gloss.json",
                    'r', encoding='utf-8'
                ) as f:
                    self.vocab_gloss = json.load(f)

                try:
                    with open(
                        translation_dir / "metadata.json",
                        'r', encoding='utf-8'
                    ) as f:
                        self.metadata['translation'] = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    logger.warning(
                        f"No se pudo cargar metadata.json: {e}. "
                        "Usando valores por defecto."
                    )
                    self.metadata['translation'] = {
                        'max_seq_length': 100,  # valor por defecto
                    }

                self.inv_vocab_gloss = {
                    v: k for k, v in self.vocab_gloss.items()
                }
                logger.info(
                    f"  [OK] Vocabulario: {len(self.vocab_pt)} PT, "
                    f"{len(self.vocab_gloss)} glosas"
                )
            except Exception as e:
                logger.error(f"Error al cargar modelo de traducción: {e}")
                # Quitar del diccionario si falla
                if 'translation' in self.models:
                    del self.models['translation']

        alphabet_dir = self.models_dir / "alphabet"

        best_h5 = alphabet_dir / "run_20251119_182114" / "best_model.h5"

        if best_h5.exists():
            logger.info("Cargando modelo de alfabeto (dactilologia)...")
            try:
                custom_objects = {'AttentionLayer': AttentionLayer}
                with tf.keras.utils.custom_object_scope(custom_objects):
                    self.models['alphabet'] = tf.keras.models.load_model(
                        str(best_h5), compile=False
                    )
                # Cargar metadata (puede ser model_info.json o metadata.json)
                metadata_path = (
                    alphabet_dir / "run_20251119_182114" / "model_info.json"
                )
                if not metadata_path.exists():
                    metadata_path = alphabet_dir / "metadata.json"

                with open(
                    metadata_path, 'r', encoding='utf-8'
                ) as f:
                    metadata = json.load(f)

                # label_names → class_names normalization for older artifacts
                if (
                    'label_names' in metadata
                    and 'class_names' not in metadata
                ):
                    metadata['class_names'] = metadata['label_names']

                self.metadata['alphabet'] = metadata
                num_letters = len(
                    self.metadata['alphabet'].get('class_names', [])
                )
                logger.info(f"  [OK] {num_letters} letras del alfabeto")
            except Exception as e:
                logger.error(f"Error cargando alfabeto: {e}")

        vlibrasil_dir = self.models_dir / "vlibrasil"
        if vlibrasil_dir.exists():
            run_dirs = sorted(
                [
                    d for d in vlibrasil_dir.iterdir()
                    if d.is_dir() and d.name.startswith('run_')
                ],
                reverse=True
            )

            for run_dir in run_dirs:
                model_path = run_dir / "best_model.h5"
                if model_path.exists():
                    try:
                        logger.info(
                            f"Cargando modelo v-librasil desde "
                            f"{run_dir.name}..."
                        )
                        self.models['vlibrasil'] = (
                            tf.keras.models.load_model(str(model_path))
                        )

                        metadata_path = run_dir / "metadata.json"
                        if metadata_path.exists():
                            try:
                                with open(metadata_path, 'r') as f:
                                    self.metadata['vlibrasil'] = json.load(f)
                                logger.info("  [OK] Metadata cargada")
                            except Exception as e:
                                logger.error(f"Error cargando metadata: {e}")

                        logger.info("  [OK] Modelo v-librasil cargado")
                        break
                    except Exception as e:
                        logger.error(f"Error cargando v-librasil: {e}")

        logger.info(f"Modelos cargados: {list(self.models.keys())}")

    def detect_hand_orientation(self, landmarks: np.ndarray) -> str:
        coords = landmarks.reshape(21, 3)

        wrist = coords[0]
        thumb_tip = coords[4]

        avg_z = np.mean(coords[:, 2])
        is_back = avg_z > 0  # deeper z = back view

        thumb_x_relative = thumb_tip[0] - wrist[0]

        # threshold 0.1 empirically chosen to distinguish lateral vs frontal
        if abs(thumb_x_relative) > 0.1:
            if thumb_x_relative > 0:
                return 'right'
            else:
                return 'left'
        else:
            if is_back:
                return 'back'
            else:
                return 'front'

    def extract_hand_landmarks(
        self, frame: np.ndarray
    ) -> Optional[List[Dict]]:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return None

        hands_data = []

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])

            landmarks_array = np.array(coords, dtype=np.float32)
            orientation = self.detect_hand_orientation(landmarks_array)

            handedness = "Right"
            if (
                results.multi_handedness
                and idx < len(results.multi_handedness)
            ):
                handedness = (
                    results.multi_handedness[idx]
                    .classification[0].label
                )

            hands_data.append({
                'landmarks': landmarks_array,
                'orientation': orientation,
                'handedness': handedness,
                'raw_landmarks': hand_landmarks
            })

        return hands_data

    def extract_face_landmarks(
        self, frame: np.ndarray
    ) -> Optional[np.ndarray]:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            coords = []
            for lm in landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            return np.array(coords, dtype=np.float32)

        return None

    def predict_handshape(
        self, landmarks: np.ndarray, orientation: str
    ) -> Tuple[str, float, str]:
        if 'handshape' not in self.models:
            return ("N/A", 0.0, "none")

        # Verificar si existe modelo para esta orientación
        if orientation not in self.models['handshape']:
            available = list(self.models['handshape'].keys())
            if not available:
                return ("N/A", 0.0, "none")
            orientation = available[0]  # Usar primera disponible

        landmarks_reshaped = landmarks.reshape(1, -1)

        model = self.models['handshape'][orientation]
        pred = model.predict(landmarks_reshaped, verbose=0)
        class_idx = np.argmax(pred[0])
        confidence = pred[0][class_idx]

        class_names = self.metadata['handshape'][orientation].get(
            'class_names', []
        )
        class_name = (
            class_names[class_idx]
            if class_idx < len(class_names)
            else f"Class_{class_idx}"
        )

        return (class_name, float(confidence), orientation)

    def predict_alphabet(
        self, landmarks: np.ndarray, use_smoothing: bool = False
    ) -> Tuple[str, float]:
        if 'alphabet' not in self.models:
            return ("N/A", 0.0)

        # alphabet model expects temporal sequence (batch, timesteps, feats);
        # replicate single frame to match expected shape
        sequence_length = self.metadata['alphabet'].get(
            'input_shape', [30, 63]
        )[0]

        if landmarks.ndim == 1:
            landmarks = landmarks.reshape(1, -1)

        landmarks_seq = np.repeat(landmarks, sequence_length, axis=0)

        # Apply the same per-sequence normalization used during preprocessing
        # (preprocess_alphabet.py normalizes each sequence before saving)
        seq_mean = np.mean(landmarks_seq, axis=0, keepdims=True)
        seq_std = np.std(landmarks_seq, axis=0, keepdims=True) + 1e-8
        landmarks_seq = (landmarks_seq - seq_mean) / seq_std

        landmarks_reshaped = landmarks_seq.reshape(1, sequence_length, -1)

        pred = self.models['alphabet'].predict(landmarks_reshaped, verbose=0)
        class_idx = np.argmax(pred[0])
        confidence = pred[0][class_idx]

        class_names = self.metadata['alphabet'].get('class_names', [])
        letter = (
            class_names[class_idx]
            if class_idx < len(class_names)
            else f"?{class_idx}"
        )

        return (letter, float(confidence))

    def _smooth_alphabet_prediction(self) -> Tuple[str, float]:
        if not self.alphabet_predictions_buffer:
            return ("N/A", 0.0)

        class_names = self.metadata['alphabet'].get('class_names', [])

        avg_probs = np.mean(
            [p['probs'] for p in self.alphabet_predictions_buffer], axis=0
        )

        class_idx = np.argmax(avg_probs)
        confidence = avg_probs[class_idx]

        letter = (
            class_names[class_idx]
            if class_idx < len(class_names)
            else f"?{class_idx}"
        )

        top2_idx = np.argsort(avg_probs)[-2:][::-1]
        top2_letters = [
            class_names[i] for i in top2_idx if i < len(class_names)
        ]
        top2_confs = [avg_probs[i] for i in top2_idx]

        top2_close = (
            len(top2_confs) >= 2
            and (top2_confs[0] - top2_confs[1]) < 0.1
        )
        if confidence < 0.4 or top2_close:
            # show both candidates when top-2 are too close to distinguish
            if len(top2_confs) >= 2 and (top2_confs[0] - top2_confs[1]) < 0.1:
                letter = f"{top2_letters[0]}/{top2_letters[1]}"

        return (letter, float(confidence))

    def predict_facial_expression(
        self, landmarks: np.ndarray
    ) -> Tuple[str, float]:
        if 'facial' not in self.models:
            return ("N/A", 0.0)

        # repeat single frame to match LSTM temporal input shape
        sequence_length = self.metadata['facial']['sequence_length']
        landmarks_seq = np.tile(landmarks, (sequence_length, 1))
        landmarks_seq = landmarks_seq.reshape(1, sequence_length, -1)

        pred = self.models['facial'].predict(landmarks_seq, verbose=0)
        class_idx = np.argmax(pred[0])
        confidence = pred[0][class_idx]

        class_names = self.metadata['facial']['class_names']
        expression = class_names[class_idx]

        return (expression, float(confidence))

    def translate_text_to_gloss(self, text_pt: str) -> List[str]:
        if 'translation' not in self.models:
            return []

        tokens = text_pt.lower().split()
        unk_id = self.vocab_pt.get('<UNK>', 0)
        token_ids = [self.vocab_pt.get(t, unk_id) for t in tokens]

        max_len = self.metadata['translation']['max_seq_length']
        if len(token_ids) < max_len:
            token_ids += [0] * (max_len - len(token_ids))
        else:
            token_ids = token_ids[:max_len]

        encoder_input = np.array([token_ids])
        decoder_input = np.array([[self.vocab_gloss['<SOS>']]])

        generated = []
        for _ in range(max_len):
            pred = self.models['translation'].predict(
                [encoder_input, decoder_input],
                verbose=0
            )
            next_token = np.argmax(pred[0, -1, :])

            if next_token == self.vocab_gloss.get('<EOS>', 1):
                break

            generated.append(self.inv_vocab_gloss.get(next_token, '<UNK>'))

            decoder_input = np.concatenate([
                decoder_input,
                [[next_token]]
            ], axis=1)

        return generated

    def predict_from_frame(
        self, frame: np.ndarray, draw_landmarks: bool = False
    ) -> Dict[str, any]:
        results = {
            'hands': [],  # Lista de predicciones para cada mano
            'facial_expression': None,
            'landmarks_detected': {
                'hands': 0,  # Número de manos detectadas
                'face': False
            }
        }

        hands_data = self.extract_hand_landmarks(frame)
        if hands_data is not None and len(hands_data) > 0:
            results['landmarks_detected']['hands'] = len(hands_data)

            for hand_info in hands_data:
                landmarks = hand_info['landmarks']
                orientation = hand_info['orientation']
                handedness = hand_info['handedness']

                if draw_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_info['raw_landmarks'],
                        self.mp_hands.HAND_CONNECTIONS
                    )

                hand_predictions = {
                    'handedness': handedness,
                    'orientation': orientation,
                    'handshape': None,
                    'alphabet': None
                }

                if 'handshape' in self.models:
                    class_name, confidence, used_orientation = (
                        self.predict_handshape(landmarks, orientation)
                    )
                    hand_predictions['handshape'] = {
                        'class': class_name,
                        'confidence': confidence,
                        'model_used': used_orientation
                    }

                if 'alphabet' in self.models:
                    letter, confidence = self.predict_alphabet(landmarks)
                    hand_predictions['alphabet'] = {
                        'letter': letter,
                        'confidence': confidence
                    }

                results['hands'].append(hand_predictions)

        face_landmarks = self.extract_face_landmarks(frame)
        if face_landmarks is not None:
            results['landmarks_detected']['face'] = True
            if 'facial' in self.models:
                expression, confidence = (
                    self.predict_facial_expression(face_landmarks)
                )
                results['facial_expression'] = {
                    'expression': expression,
                    'confidence': confidence
                }

        return results

    def get_model_info(self) -> Dict[str, any]:
        info = {}
        for model_name, metadata in self.metadata.items():
            info[model_name] = {
                'loaded': model_name in self.models,
                'metadata': metadata
            }
        return info

    def __del__(self):
        if hasattr(self, 'hands'):
            self.hands.close()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()

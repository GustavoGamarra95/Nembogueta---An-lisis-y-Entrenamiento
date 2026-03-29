"""
Sistema Unificado de Predicción LIBRAS
Centraliza todos los modelos entrenados para reconocimiento de lengua de señas brasileña.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable

logger = logging.getLogger(__name__)


# Registrar capas personalizadas para el modelo de traducción
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


@register_keras_serializable(package="Custom", name="AttentionLayer")
class AttentionLayer(layers.Layer):
    """Capa de atención para enfocarse en landmarks importantes."""

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
        # inputs shape: (batch, timesteps, features)
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
    """
    Predictor unificado que integra todos los modelos de LIBRAS:
    - Formas de mano (handshapes) por orientación
    - Expresiones faciales
    - Traducción texto-glosas
    - Palabras completas (v-librasil)
    """

    def __init__(self, models_dir: Path, buffer_size: int = 30):
        """
        Inicializa el predictor con todos los modelos.

        Args:
            models_dir: Directorio raíz con todos los modelos entrenados
            buffer_size: Tamaño del buffer para suavizado temporal
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.metadata = {}

        # Buffer de frames para suavizado temporal
        self.buffer_size = buffer_size
        self.frames_buffer = []  # Buffer de landmarks recientes

        # Buffers de predicciones para suavizado
        self.alphabet_predictions_buffer = []
        self.max_prediction_buffer = 5  # Últimas 5 predicciones

        # Inicializar MediaPipe
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
        """Carga todos los modelos disponibles."""
        # 1. Cargar modelos de formas de mano por orientación
        handshape_dir = self.models_dir / "handshape"
        self.models['handshape'] = {}
        self.metadata['handshape'] = {}

        orientations = ['back', 'front', 'left', 'right']
        loaded_orientations = []

        for orientation in orientations:
            orient_dir = handshape_dir / orientation
            if (orient_dir / "best_model.keras").exists():
                try:
                    self.models['handshape'][orientation] = tf.keras.models.load_model(
                        orient_dir / "best_model.keras"
                    )
                    with open(orient_dir / "metadata.json", 'r') as f:
                        self.metadata['handshape'][orientation] = json.load(f)
                    loaded_orientations.append(orientation)
                except Exception as e:
                    logger.error(f"Error cargando handshape {orientation}: {e}")

        if loaded_orientations:
            logger.info(f"Modelos de formas de mano cargados: {', '.join(loaded_orientations)}")
            total_classes = sum([m['n_classes'] for m in self.metadata['handshape'].values()])
            logger.info(f"  [OK] {total_classes} clases total")
        else:
            # Limpiar si no se cargó ninguno
            del self.models['handshape']
            del self.metadata['handshape']

        # 2. Cargar modelo de expresiones faciales
        facial_dir = self.models_dir / "facial_expressions"
        if (facial_dir / "best_model.keras").exists():
            logger.info("Cargando modelo de expresiones faciales...")
            try:
                self.models['facial'] = tf.keras.models.load_model(
                    facial_dir / "best_model.keras"
                )
                with open(facial_dir / "metadata.json", 'r', encoding='utf-8') as f:
                    self.metadata['facial'] = json.load(f)
                logger.info(f"  [OK] {len(self.metadata['facial']['class_names'])} expresiones")
            except Exception as e:
                logger.error(f"Error cargando facial expressions: {e}")

        # 3. Cargar modelo de traducción
        translation_dir = self.models_dir / "translation"
        if (translation_dir / "best_model.keras").exists():
            logger.info("Cargando modelo de traduccion PT-BR -> LIBRAS...")
            try:
                self.models['translation'] = tf.keras.models.load_model(
                    translation_dir / "best_model.keras"
                )
                with open(translation_dir / "vocab_pt_br.json", 'r', encoding='utf-8') as f:
                    self.vocab_pt = json.load(f)
                with open(translation_dir / "vocab_libras_gloss.json", 'r', encoding='utf-8') as f:
                    self.vocab_gloss = json.load(f)

                # Cargar metadata si existe y es válido
                try:
                    with open(translation_dir / "metadata.json", 'r', encoding='utf-8') as f:
                        self.metadata['translation'] = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    logger.warning(f"No se pudo cargar metadata.json: {e}. Usando valores por defecto.")
                    self.metadata['translation'] = {
                        'max_seq_length': 100,  # valor por defecto
                    }

                # Crear vocabulario inverso
                self.inv_vocab_gloss = {v: k for k, v in self.vocab_gloss.items()}
                logger.info(f"  [OK] Vocabulario: {len(self.vocab_pt)} PT, {len(self.vocab_gloss)} glosas")
            except Exception as e:
                logger.error(f"Error al cargar modelo de traducción: {e}")
                # Quitar del diccionario si falla
                if 'translation' in self.models:
                    del self.models['translation']

        # 4. Cargar modelo de alfabeto/dactilología
        alphabet_dir = self.models_dir / "alphabet"

        # Intentar cargar desde el mejor run con class balancing
        best_h5 = alphabet_dir / "run_20251119_182114" / "best_model.h5"

        if best_h5.exists():
            logger.info("Cargando modelo de alfabeto (dactilologia)...")
            try:
                with tf.keras.utils.custom_object_scope({'AttentionLayer': AttentionLayer}):
                    self.models['alphabet'] = tf.keras.models.load_model(str(best_h5), compile=False)
                # Cargar metadata (puede ser model_info.json o metadata.json)
                metadata_path = alphabet_dir / "run_20251119_182114" / "model_info.json"
                if not metadata_path.exists():
                    metadata_path = alphabet_dir / "metadata.json"

                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                # Normalizar metadata: label_names -> class_names
                if 'label_names' in metadata and 'class_names' not in metadata:
                    metadata['class_names'] = metadata['label_names']

                self.metadata['alphabet'] = metadata
                num_letters = len(self.metadata['alphabet'].get('class_names', []))
                logger.info(f"  [OK] {num_letters} letras del alfabeto")
            except Exception as e:
                logger.error(f"Error cargando alfabeto: {e}")

        # 5. Cargar modelo de v-librasil (buscar última ejecución)
        vlibrasil_dir = self.models_dir / "vlibrasil"
        if vlibrasil_dir.exists():
            # Buscar el directorio de ejecución más reciente
            run_dirs = sorted([d for d in vlibrasil_dir.iterdir() if d.is_dir() and d.name.startswith('run_')],
                            reverse=True)

            for run_dir in run_dirs:
                model_path = run_dir / "best_model.h5"
                if model_path.exists():
                    try:
                        logger.info(f"Cargando modelo v-librasil desde {run_dir.name}...")
                        self.models['vlibrasil'] = tf.keras.models.load_model(str(model_path))

                        # Implementación para cargar metadata si existe
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
        """
        Detecta la orientación de la mano basándose en landmarks.

        Args:
            landmarks: Array de landmarks (63 features: x, y, z para 21 puntos)

        Returns:
            Orientación: 'back', 'front', 'left', o 'right'
        """
        # Reshape para obtener coordenadas individuales
        coords = landmarks.reshape(21, 3)

        # Puntos clave para determinar orientación
        wrist = coords[0]  # Muñeca
        thumb_tip = coords[4]  # Punta del pulgar
        index_tip = coords[8]  # Punta del índice
        middle_mcp = coords[9]  # Base del dedo medio
        pinky_tip = coords[20]  # Punta del meñique

        # Calcular vectores
        palm_vector = middle_mcp - wrist
        thumb_vector = thumb_tip - wrist

        # Determinar si es vista frontal o trasera basándose en z
        avg_z = np.mean(coords[:, 2])
        is_back = avg_z > 0  # Mayor profundidad = vista trasera

        # Determinar si es izquierda o derecha basándose en posición del pulgar
        thumb_x_relative = thumb_tip[0] - wrist[0]

        # Clasificar orientación
        if abs(thumb_x_relative) > 0.1:  # Vista lateral
            if thumb_x_relative > 0:
                return 'right'
            else:
                return 'left'
        else:  # Vista frontal/trasera
            if is_back:
                return 'back'
            else:
                return 'front'

    def extract_hand_landmarks(self, frame: np.ndarray) -> Optional[List[Dict]]:
        """
        Extrae landmarks de todas las manos detectadas en un frame.

        Args:
            frame: Frame de video (BGR)

        Returns:
            Lista de diccionarios con información de cada mano detectada, o None
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return None

        hands_data = []

        # Procesar cada mano detectada
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Extraer coordenadas
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])

            landmarks_array = np.array(coords, dtype=np.float32)
            orientation = self.detect_hand_orientation(landmarks_array)

            # Determinar si es mano izquierda o derecha
            handedness = "Right"  # Por defecto
            if results.multi_handedness and idx < len(results.multi_handedness):
                handedness = results.multi_handedness[idx].classification[0].label

            hands_data.append({
                'landmarks': landmarks_array,
                'orientation': orientation,
                'handedness': handedness,  # "Left" o "Right"
                'raw_landmarks': hand_landmarks
            })

        return hands_data

    def extract_face_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrae landmarks faciales desde un frame.

        Args:
            frame: Frame de video (BGR)

        Returns:
            Array de landmarks faciales o None si no se detecta rostro
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            # Extraer landmarks relevantes para expresiones faciales
            # (usar los mismos índices que en el preprocesamiento)
            coords = []
            for lm in landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            return np.array(coords, dtype=np.float32)

        return None

    def predict_handshape(self, landmarks: np.ndarray, orientation: str) -> Tuple[str, float, str]:
        """
        Predice la forma de mano a partir de landmarks usando el modelo apropiado.

        Args:
            landmarks: Landmarks de la mano (63 features)
            orientation: Orientación de la mano ('back', 'front', 'left', 'right')

        Returns:
            Tupla (clase_predicha, confianza, orientación_usada)
        """
        if 'handshape' not in self.models:
            return ("N/A", 0.0, "none")

        # Verificar si existe modelo para esta orientación
        if orientation not in self.models['handshape']:
            # Intentar con orientación por defecto
            available = list(self.models['handshape'].keys())
            if not available:
                return ("N/A", 0.0, "none")
            orientation = available[0]  # Usar primera disponible

        # Reshape para el modelo
        landmarks_reshaped = landmarks.reshape(1, -1)

        # Predicción con el modelo de la orientación específica
        model = self.models['handshape'][orientation]
        pred = model.predict(landmarks_reshaped, verbose=0)
        class_idx = np.argmax(pred[0])
        confidence = pred[0][class_idx]

        # Obtener nombre de la clase
        class_names = self.metadata['handshape'][orientation].get('class_names', [])
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class_{class_idx}"

        return (class_name, float(confidence), orientation)

    def predict_alphabet(self, landmarks: np.ndarray, use_smoothing: bool = False) -> Tuple[str, float]:
        """
        Predice la letra del alfabeto (dactilología) a partir de landmarks.

        Args:
            landmarks: Landmarks de la mano (63 features)
            use_smoothing: Si True, usa suavizado temporal de predicciones

        Returns:
            Tupla (letra_predicha, confianza)
        """
        if 'alphabet' not in self.models:
            return ("N/A", 0.0)

        # El modelo de alfabeto espera secuencias temporales (batch, timesteps, features)
        sequence_length = self.metadata['alphabet'].get('input_shape', [30, 63])[0]

        # Asegurar que landmarks tenga la forma correcta (63,)
        if landmarks.ndim == 1:
            landmarks = landmarks.reshape(1, -1)  # (1, 63)

        # Repetir el frame actual para crear la secuencia temporal
        landmarks_seq = np.repeat(landmarks, sequence_length, axis=0)  # (30, 63)
        landmarks_reshaped = landmarks_seq.reshape(1, sequence_length, -1)  # (1, 30, 63)

        # Predicción directa
        pred = self.models['alphabet'].predict(landmarks_reshaped, verbose=0)
        class_idx = np.argmax(pred[0])
        confidence = pred[0][class_idx]

        # Obtener nombre de la letra
        class_names = self.metadata['alphabet'].get('class_names', [])
        letter = class_names[class_idx] if class_idx < len(class_names) else f"?{class_idx}"

        return (letter, float(confidence))

    def _smooth_alphabet_prediction(self) -> Tuple[str, float]:
        """
        Suaviza la predicción del alfabeto usando las últimas predicciones.

        Returns:
            Tupla (letra_suavizada, confianza_promedio)
        """
        if not self.alphabet_predictions_buffer:
            return ("N/A", 0.0)

        class_names = self.metadata['alphabet'].get('class_names', [])

        # Promediar las probabilidades de las últimas predicciones
        avg_probs = np.mean([p['probs'] for p in self.alphabet_predictions_buffer], axis=0)

        # Obtener la letra con mayor probabilidad promedio
        class_idx = np.argmax(avg_probs)
        confidence = avg_probs[class_idx]

        letter = class_names[class_idx] if class_idx < len(class_names) else f"?{class_idx}"

        # Verificar si hay ambigüedad entre top 2 opciones
        top2_idx = np.argsort(avg_probs)[-2:][::-1]
        top2_letters = [class_names[i] for i in top2_idx if i < len(class_names)]
        top2_confs = [avg_probs[i] for i in top2_idx]

        # Si confianza es baja O diferencia entre top 1 y top 2 es pequeña
        if confidence < 0.4 or (len(top2_confs) >= 2 and (top2_confs[0] - top2_confs[1]) < 0.1):
            # Mostrar ambas opciones si son muy similares
            if len(top2_confs) >= 2 and (top2_confs[0] - top2_confs[1]) < 0.1:
                letter = f"{top2_letters[0]}/{top2_letters[1]}"

        return (letter, float(confidence))

    def predict_facial_expression(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """
        Predice la expresión facial a partir de landmarks faciales.

        Args:
            landmarks: Landmarks faciales

        Returns:
            Tupla (expresión_predicha, confianza)
        """
        if 'facial' not in self.models:
            return ("N/A", 0.0)

        # Reshape para el modelo LSTM (necesita secuencia temporal)
        # Por ahora, repetir el frame para simular secuencia
        sequence_length = self.metadata['facial']['sequence_length']
        landmarks_seq = np.tile(landmarks, (sequence_length, 1))
        landmarks_seq = landmarks_seq.reshape(1, sequence_length, -1)

        # Predicción
        pred = self.models['facial'].predict(landmarks_seq, verbose=0)
        class_idx = np.argmax(pred[0])
        confidence = pred[0][class_idx]

        # Obtener nombre de la expresión
        class_names = self.metadata['facial']['class_names']
        expression = class_names[class_idx]

        return (expression, float(confidence))

    def translate_text_to_gloss(self, text_pt: str) -> List[str]:
        """
        Traduce texto en portugués a glosas de LIBRAS.

        Args:
            text_pt: Texto en portugués

        Returns:
            Lista de glosas en LIBRAS
        """
        if 'translation' not in self.models:
            return []

        # Tokenizar
        tokens = text_pt.lower().split()
        token_ids = [self.vocab_pt.get(t, self.vocab_pt.get('<UNK>', 0)) for t in tokens]

        # Padding
        max_len = self.metadata['translation']['max_seq_length']
        if len(token_ids) < max_len:
            token_ids += [0] * (max_len - len(token_ids))
        else:
            token_ids = token_ids[:max_len]

        # Preparar input para el modelo
        encoder_input = np.array([token_ids])
        decoder_input = np.array([[self.vocab_gloss['<SOS>']]])

        # Generar glosas (simple greedy decoding)
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

            # Actualizar decoder input
            decoder_input = np.concatenate([
                decoder_input,
                [[next_token]]
            ], axis=1)

        return generated

    def predict_from_frame(self, frame: np.ndarray, draw_landmarks: bool = False) -> Dict[str, any]:
        """
        Realiza todas las predicciones posibles desde un frame.

        Args:
            frame: Frame de video (BGR)
            draw_landmarks: Si True, dibuja los landmarks en el frame

        Returns:
            Diccionario con todas las predicciones
        """
        results = {
            'hands': [],  # Lista de predicciones para cada mano
            'facial_expression': None,
            'landmarks_detected': {
                'hands': 0,  # Número de manos detectadas
                'face': False
            }
        }

        # Detectar y predecir para cada mano
        hands_data = self.extract_hand_landmarks(frame)
        if hands_data is not None and len(hands_data) > 0:
            results['landmarks_detected']['hands'] = len(hands_data)

            for hand_info in hands_data:
                landmarks = hand_info['landmarks']
                orientation = hand_info['orientation']
                handedness = hand_info['handedness']

                # Dibujar landmarks si se solicita
                if draw_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_info['raw_landmarks'],
                        self.mp_hands.HAND_CONNECTIONS
                    )

                hand_predictions = {
                    'handedness': handedness,  # "Left" o "Right"
                    'orientation': orientation,
                    'handshape': None,
                    'alphabet': None
                }

                # Predecir handshape
                if 'handshape' in self.models:
                    class_name, confidence, used_orientation = self.predict_handshape(landmarks, orientation)
                    hand_predictions['handshape'] = {
                        'class': class_name,
                        'confidence': confidence,
                        'model_used': used_orientation
                    }

                # Predecir letra del alfabeto
                if 'alphabet' in self.models:
                    letter, confidence = self.predict_alphabet(landmarks)
                    hand_predictions['alphabet'] = {
                        'letter': letter,
                        'confidence': confidence
                    }

                results['hands'].append(hand_predictions)

        # Detectar y predecir expresión facial
        face_landmarks = self.extract_face_landmarks(frame)
        if face_landmarks is not None:
            results['landmarks_detected']['face'] = True
            if 'facial' in self.models:
                expression, confidence = self.predict_facial_expression(face_landmarks)
                results['facial_expression'] = {
                    'expression': expression,
                    'confidence': confidence
                }

        return results

    def get_model_info(self) -> Dict[str, any]:
        """
        Retorna información sobre los modelos cargados.

        Returns:
            Diccionario con información de cada modelo
        """
        info = {}
        for model_name, metadata in self.metadata.items():
            info[model_name] = {
                'loaded': model_name in self.models,
                'metadata': metadata
            }
        return info

    def __del__(self):
        """Liberar recursos al destruir el objeto."""
        if hasattr(self, 'hands'):
            self.hands.close()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()

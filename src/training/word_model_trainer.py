import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    LSTM,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
    TimeDistributed,
)
from tensorflow.keras.models import Sequential

# Establecer una semilla para reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)

# Cargar variables de entorno
load_dotenv()

# Directorios de entrada y salida desde .env
input_dir = os.getenv("DATA_RAW_DIR", "data/lsp_word_videos")
output_dir = os.getenv(
    "DATA_PROCESSED_DIR", "data/processed_lsp_word_sequences"
)

# Configurar rutas de directorios
input_dir = (
    os.path.join(input_dir, "words")
    if os.path.isdir(os.path.join(input_dir, "words"))
    else input_dir
)
output_dir = (
    os.path.join(output_dir, "words")
    if os.path.isdir(os.path.join(output_dir, "words"))
    else output_dir
)
os.makedirs(output_dir, exist_ok=True)

# Rutas a los datos preprocesados
X_path = os.path.join(input_dir, "X_lsp_word_sequences.npy")
y_path = os.path.join(input_dir, "y_lsp_word_sequences.npy")

# Cargar los datos
X = np.load(X_path)  # Forma: (muestras, 15, 200, 200, 3)
y = np.load(y_path)  # Forma: (muestras,)


def augment_sequence(sequence):
    """
    Aplica aumento de datos con rotación y escalado aleatorios.
    """
    augmented_sequence = np.zeros_like(sequence)
    for i in range(sequence.shape[0]):
        frame = sequence[i]
        # Rotación aleatoria (-15 a 15 grados)
        angle = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D(
            (frame.shape[1] // 2, frame.shape[0] // 2), angle, 1
        )
        rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        # Escalado aleatorio (0.9 a 1.1)
        scale = np.random.uniform(0.9, 1.1)
        scaled = cv2.resize(
            rotated, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        )
        # Asegurar que la forma de salida coincida con la entrada
        scaled = cv2.resize(
            scaled,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        augmented_sequence[i] = scaled
    return augmented_sequence


# Aplicar aumento de datos
X_augmented = np.array([augment_sequence(seq) for seq in X])

# Normalizar los datos
X_augmented = X_augmented / 255.0

# Dividir los datos
X_train, X_val, y_train, y_val = train_test_split(
    X_augmented, y, test_size=0.2, random_state=42
)

# Calcular pesos de clase
class_weights = class_weight.compute_class_weight(
    "balanced", classes=np.unique(y_train), y=y_train
)
class_weights_dict = dict(enumerate(class_weights))

# Construir el modelo CNN-LSTM mejorado
model = Sequential(
    [
        # Capas CNN para extraer características espaciales
        TimeDistributed(
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            input_shape=(15, 200, 200, 3),
        ),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(
            Conv2D(128, (3, 3), activation="relu", padding="same")
        ),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(
            Conv2D(256, (3, 3), activation="relu", padding="same")
        ),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Flatten()),
        # Capas LSTM para dependencias temporales
        LSTM(256, return_sequences=True),
        Dropout(0.3),
        LSTM(128, return_sequences=False),
        Dropout(0.3),
        # Capas densas para clasificación
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(10, activation="softmax"),  # 10 clases para palabras
    ]
)

# Compilar el modelo con AdamW
optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0005, weight_decay=0.01)
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Definir parada temprana
early_stopping = EarlyStopping(
    monitor="val_accuracy", patience=10, restore_best_weights=True
)

# Entrenar el modelo
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16,
    class_weight=class_weights_dict,
    callbacks=[early_stopping],
    verbose=1,
)

# Evaluar el modelo
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Precisión final de validación: {val_accuracy * 100:.2f}%")

# Guardar el modelo
model_path = os.path.join(output_dir, "cnn_lstm_lsp_words_model.h5")
model.save(model_path)

# Graficar el historial de entrenamiento
plt.figure(figsize=(10, 6))
plt.plot(
    history.history["accuracy"],
    label="Precisión de Entrenamiento",
    color="#1f77b4",
)
plt.plot(
    history.history["val_accuracy"],
    label="Precisión de Validación",
    color="#ff7f0e",
)
plt.axhline(y=0.95, color="red", linestyle="--", label="Objetivo (95%)")
plt.title("Precisión de Entrenamiento y Validación (Palabras)")
plt.xlabel("Época")
plt.ylabel("Precisión")
plt.legend()
plt.grid(True)
plt.savefig("training_accuracy_words.png", dpi=300, bbox_inches="tight")
plt.show()

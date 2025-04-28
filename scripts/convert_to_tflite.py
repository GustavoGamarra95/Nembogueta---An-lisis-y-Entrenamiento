import tensorflow as tf
import os

# Rutas a los modelos entrenados
models_dir = 'models/'
tflite_dir = 'models/tflite/'
os.makedirs(tflite_dir, exist_ok=True)

# Lista de modelos a convertir
model_names = [
    'cnn_lstm_lsp_letters_model.h5',
    'cnn_lstm_lsp_words_model.h5',
    'cnn_lstm_lsp_phrases_model.h5'
]

# Convertir cada modelo a TensorFlow Lite
for model_name in model_names:
    # Cargar el modelo
    model_path = os.path.join(models_dir, model_name)
    model = tf.keras.models.load_model(model_path)

    # Convertir a TensorFlow Lite con cuantización
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Guardar el modelo TFLite
    tflite_path = os.path.join(tflite_dir, model_name.replace('.h5', '_quant.tflite'))
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Modelo {model_name} convertido a TFLite y guardado en {tflite_path}")
"""
Script para convertir el modelo de alfabeto H5 a formato Keras.
"""
import sys
from pathlib import Path
import tensorflow as tf

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar la capa de atención desde train_alphabet
from scripts.train_alphabet import AttentionLayer


def convert_model(h5_path: str, keras_path: str):
    """Convierte modelo H5 a formato Keras."""
    print(f"Cargando modelo desde {h5_path}...")

    # Cargar con custom objects
    with tf.keras.utils.custom_object_scope({'AttentionLayer': AttentionLayer}):
        model = tf.keras.models.load_model(h5_path)

    print(f"Modelo cargado: {model.name}")
    print(f"Parámetros: {model.count_params():,}")

    # Guardar en formato .keras
    print(f"\nGuardando en formato .keras: {keras_path}")
    model.save(keras_path)

    print("✓ Conversión completada exitosamente")

    # Verificar
    print("\nVerificando modelo convertido...")
    with tf.keras.utils.custom_object_scope({'AttentionLayer': AttentionLayer}):
        test_model = tf.keras.models.load_model(keras_path)
    print(f"✓ Modelo verificado: {test_model.name}")


if __name__ == '__main__':
    h5_path = 'data/models/alphabet/run_20251119_153810/best_model.h5'
    keras_path = 'data/models/alphabet/alphabet_model.keras'

    convert_model(h5_path, keras_path)

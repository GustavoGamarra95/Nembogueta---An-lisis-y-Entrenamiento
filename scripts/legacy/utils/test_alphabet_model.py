"""
Script para probar el modelo de alfabeto con datos de prueba
"""
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import Counter

# Cargar modelo
print("Cargando modelo...")
model = tf.keras.models.load_model('data/models/alphabet/alphabet_model.keras')
print(f"Modelo cargado: {model.name}")
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")

# Cargar metadata
with open('data/models/alphabet/metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

label_names = metadata['label_names']
print(f"\nClases: {label_names}")
print(f"Total clases: {len(label_names)}")

# Cargar algunos datos de test
test_dir = Path('data/processed/alphabet-combined')
npy_files = list(test_dir.glob("*.npy"))[:100]  # Probar con 100 archivos

print(f"\nProbando con {len(npy_files)} archivos...")

predictions_counter = Counter()
correct = 0
total = 0

for npy_file in npy_files:
    # Cargar secuencia
    sequence = np.load(npy_file)

    # Extraer label real
    filename = npy_file.stem
    true_letter = filename.split('_')[0]

    # Hacer predicción
    sequence_reshaped = sequence.reshape(1, *sequence.shape)
    pred = model.predict(sequence_reshaped, verbose=0)
    pred_idx = np.argmax(pred[0])
    pred_letter = label_names[pred_idx]
    pred_conf = pred[0][pred_idx]

    predictions_counter[pred_letter] += 1

    if pred_letter == true_letter:
        correct += 1
    else:
        if total < 10:  # Mostrar primeros 10 errores
            print(f"  Error: {npy_file.name} -> Real: {true_letter}, Pred: {pred_letter} ({pred_conf:.2%})")

    total += 1

print(f"\n{'='*60}")
print(f"Accuracy en muestra: {correct}/{total} = {correct/total:.2%}")
print(f"\nDistribución de predicciones:")
for letter, count in predictions_counter.most_common():
    print(f"  {letter}: {count} veces ({count/total:.1%})")
print(f"{'='*60}")

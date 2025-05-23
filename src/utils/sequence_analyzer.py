import numpy as np
import matplotlib.pyplot as plt
import os

# Directorios de las secuencias preprocesadas
dirs = [
    'data/processed_lsp_letter_sequences',
    'data/processed_lsp_word_sequences',
    'data/processed_lsp_phrase_sequences'
]

# Nombres de las categorías
categories = ['Letras', 'Palabras', 'Frases']
num_classes = [27, 10, 3]  # Número de clases por categoría

# Analizar cada categoría
for dir_path, category, num_class in zip(dirs, categories, num_classes):
    X_path = os.path.join(dir_path, f'X_lsp_{category.lower()}_sequences.npy')
    y_path = os.path.join(dir_path, f'y_lsp_{category.lower()}_sequences.npy')

    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print(f"Datos no encontrados para {category}")
        continue

    # Cargar datos
    X = np.load(X_path)
    y = np.load(y_path)

    print(f"\nAnálisis para {category}:")
    print(f"Forma de X: {X.shape}")
    print(f"Forma de y: {y.shape}")
    print(f"Número de muestras: {X.shape[0]}")

    # Distribución de clases
    class_counts = np.bincount(y, minlength=num_class)
    print("Distribución de clases:")
    for cls, count in enumerate(class_counts):
        print(f"Clase {cls}: {count} muestras")

    # Visualizar la distribución de clases
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_class), class_counts, color='#1f77b4')
    plt.title(f'Distribución de Clases - {category}')
    plt.xlabel('Clase')
    plt.ylabel('Número de Muestras')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'distribution_{category.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()

print("Análisis completo.")
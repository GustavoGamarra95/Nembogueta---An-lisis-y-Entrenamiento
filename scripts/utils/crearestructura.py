#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import sys


def crear_estructura_proyecto():
    """
    Crea la estructura de directorios recomendada para el proyecto Ñemongeta
    y mueve los archivos existentes a sus ubicaciones correctas.
    """
    # Definir la estructura de directorios
    directorios = [
        "data/raw/letters",
        "data/raw/words",
        "data/raw/phrases",
        "data/processed/letters",
        "data/processed/words",
        "data/processed/phrases",
        "models/h5",
        "models/tflite",
        "src/data_collection",
        "src/preprocessing",
        "src/training",
        "src/utils",
        "notebooks",
        "tests",
    ]

    # Crear directorios
    print("Creando estructura de directorios...")
    for directorio in directorios:
        os.makedirs(directorio, exist_ok=True)
        print(f"✓ Creado: {directorio}")

    # Crear archivos __init__.py en directorios de código
    init_dirs = [
        "src",
        "src/data_collection",
        "src/preprocessing",
        "src/training",
        "src/utils",
        "tests",
    ]

    for directorio in init_dirs:
        init_file = os.path.join(directorio, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write("# -*- coding: utf-8 -*-\n")
            print(f"✓ Creado: {init_file}")

    # Definir mapeo de archivos a mover
    archivos_a_mover = {
        # Scripts de recolección de datos
        "letter_collection.py": "src/data_collection/letter_collection.py",
        "word_collection.py": "src/data_collection/word_collection.py",
        "phrase_collection.py": "src/data_collection/phrase_collection.py",
        # Scripts de preprocesamiento
        "letter_preprocessor.py": "src/preprocessing/letter_processor.py",
        "word_processor.py": "src/preprocessing/word_processor.py",
        "phrase_processor.py": "src/preprocessing/phrase_processor.py",
        # Scripts de entrenamiento
        "letter_model_trainer.py": "src/training/letter_model_trainer.py",
        "word_model_trainer.py": "src/training/word_model_trainer.py",
        "phrase_model_trainer.py": "src/training/phrase_model_trainer.py",
        # Utilidades
        "sequence_analyzer.py": "src/utils/sequence_analyzer.py",
        "model_converter.py": "src/utils/model_converter.py",
    }

    # Mover archivos si existen
    print("\nMoviendo archivos...")
    for origen, destino in archivos_a_mover.items():
        if os.path.exists(origen):
            # Asegurarnos que el directorio destino existe
            os.makedirs(os.path.dirname(destino), exist_ok=True)

            # Verificar si el archivo destino ya existe
            if os.path.exists(destino):
                print(f"! El archivo destino ya existe: {destino}")
                respuesta = input(f"¿Deseas sobrescribir {destino}? (s/n): ")
                if respuesta.lower() != "s":
                    print(f"✗ Omitiendo: {origen}")
                    continue

            try:
                # Primero copiamos el archivo
                shutil.copy2(origen, destino)
                print(f"✓ Copiado: {origen} → {destino}")

                # Preguntamos si quiere eliminar el original
                respuesta = input(
                    f"¿Deseas eliminar el archivo original {origen}? (s/n): "
                )
                if respuesta.lower() == "s":
                    os.remove(origen)
                    print(f"✓ Eliminado original: {origen}")
            except Exception as e:
                print(f"✗ Error al mover {origen}: {str(e)}")
        else:
            print(f"? No encontrado: {origen}")

    # Crear requirements.txt si no existe
    if not os.path.exists("requirements.txt"):
        with open("requirements.txt", "w") as f:
            f.write(
                """# Dependencias del proyecto Ñemongeta
opencv-python>=4.5.0
mediapipe>=0.8.9
numpy>=1.20.0
tensorflow>=2.6.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
"""
            )
        print("✓ Creado: requirements.txt")

    # Crear setup.py básico si no existe
    if not os.path.exists("setup.py"):
        with open("setup.py", "w") as f:
            f.write(
                """# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name="nembogueta",
    version="0.1.0",
    description=(
    "Módulo Python para reconocimiento de "
    "Lenguaje de Señas Paraguayo (LSPy)"
    ),
    author="Tu Nombre",
    author_email="tu.email@ejemplo.com",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.5.0",
        "mediapipe>=0.8.9",
        "numpy>=1.20.0",
        "tensorflow>=2.6.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=0.24.0",
    ],
    python_requires=">=3.8",
)
"""
            )
        print("✓ Creado: setup.py")

    # Imprimir resumen
    print("\n✓ Estructura del proyecto creada exitosamente!")
    print("\nLa nueva estructura del proyecto es:")
    print("nembogueta-python/")
    for directorio in sorted(directorios):
        print(f"├── {directorio}")
    print("├── requirements.txt")
    print("├── setup.py")
    print("└── README.md (existente)")


if __name__ == "__main__":
    print("=== Creando estructura para el proyecto Ñemongeta ===\n")

    # Verificar si estamos en el directorio raíz del proyecto
    if not os.path.exists("README.md"):
        print(
            "⚠️ ADVERTENCIA: No se encontró README.md en el directorio actual."
        )
        respuesta = input(
            "¿Estás seguro de que estás en el directorio raíz "
            "del proyecto? (s/n): "
        )
        if respuesta.lower() != "s":
            print("Operación cancelada.")
            sys.exit(1)

    # Confirmar antes de proceder
    print(
        "Este script creará la estructura de directorios recomendada "
        "y moverá los archivos existentes."
    )
    respuesta = input("¿Deseas continuar? (s/n): ")

    if respuesta.lower() == "s":
        crear_estructura_proyecto()
    else:
        print("Operación cancelada.")

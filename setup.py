from setuptools import find_packages, setup

from src.config.config import Config


def setup_project_structure():
    """Inicializa la estructura del proyecto"""
    Config()


setup(
    name="nembogueta",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "numpy",
        "tensorflow",
        "mediapipe",
        "python-dotenv",
        "pillow",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
    ],
    python_requires=">=3.8",
    description=(
        "Sistema de reconocimiento de lenguaje de señas paraguayo"
        "Módulo de análisis y entrenamiento"
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
)

if __name__ == "__main__":
    setup_project_structure()

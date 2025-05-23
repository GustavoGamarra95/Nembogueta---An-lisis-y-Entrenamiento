
from setuptools import setup, find_packages

setup(
    name="nembogueta",
    version="0.1.0",
    description="Modulo Python para reconocimiento de Lenguaje de Se�as Paraguayo (LSPy)",
    author="Gustavo Ariel Gamarra Rojas",
    author_email="guaro8@gmail.com",
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

from setuptools import setup, find_packages

setup(
    name="nembogueta",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'opencv-python>=4.8.1.78',
        'mediapipe>=0.10.8',
        'numpy>=1.24.3',
        'tensorflow>=2.14.0',
        'matplotlib>=3.8.2',
        'scikit-learn>=1.3.2',
        'pyyaml>=6.0.1',
    ],
    extras_require={
        'dev': [
            'pytest>=7.4.3',
            'pytest-cov>=4.1.0',
            'black>=23.11.0',
            'flake8>=6.1.0',
            'isort>=5.12.0',
        ],
    },
    author="Tu Nombre",
    author_email="tu@email.com",
    description="Sistema de reconocimiento de Lenguaje de Señas Paraguayo",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tuusuario/nembogueta",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
# Ñemongeta - Python Module

**Sistema de Reconocimiento de Lenguaje de Señas Paraguayo  
Módulo de Análisis y Entrenamiento**

## Descripción

El módulo `Ñemongeta - Python` contiene scripts para la recolección, preprocesamiento, análisis, entrenamiento y conversión de modelos CNN-LSTM para el reconocimiento de gestos en Lenguaje de Señas Paraguayo (LSPy). Los modelos están optimizados para alcanzar una precisión del 95% en las categorías de letras (a-z, ñ), palabras (ej. juicio, abogado) y frases (ej. acceso a la justicia).

## Requisitos

- **Python**: 3.8 o superior (recomendado: 3.10.12)
- **Hardware**: Cámara web para recolección de videos; GPU recomendada para entrenamiento (opcional)
- **Dependencias** (instaladas con `pip install -r requirements.txt`):
  - tensorflow
  - mediapipe
  - numpy
  - opencv-python
  - python-dotenv
  - flake8
  - black
  - isort
  - pre-commit (opcional, para hooks de calidad)
  - pytest
  - coverage

## Instrucciones de Configuración

1. **Clonar el Repositorio**
   ```bash
   git clone <repository-url>
   cd nembogueta-python
   ```

2. **Configurar el Entorno de Python**
   ```bash
   pyenv install 3.10.12  # Si no está instalado
   pyenv local 3.10.12
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instalar Dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Crear Directorios de Datos**
   ```bash
   mkdir -p data/lsp_letter_videos data/lsp_word_videos data/lsp_phrase_videos
   mkdir -p data/processed_lsp_letter_sequences data/processed_lsp_word_sequences data/processed_lsp_phrase_sequences
   mkdir -p models/h5 models/tflite
   ```

5. **Configurar Variables de Entorno**
   Copia el archivo de ejemplo y ajusta según sea necesario:
   ```bash
   cp .env.example .env
   ```
   Ejemplo de `.env`:
   ```
   DATA_RAW_DIR=data/raw
   DATA_PROCESSED_DIR=data/processed
   MODELS_DIR=models/h5
   TFLITE_DIR=models/tflite
   FRAME_RATE=30
   FRAME_COUNT=300
   SEED=42
   ```

## Calidad de Código

El proyecto utiliza herramientas para garantizar la calidad del código:

- **Black**: Formateo automático
  ```bash
  black src/
  ```
- **isort**: Ordenamiento de imports
  ```bash
  isort src/
  ```
- **Flake8**: Linting de código
  ```bash
  flake8 src/
  ```

Instala estas herramientas:
```bash
pip install flake8 black isort
```

Opcionalmente, configura **pre-commit** para ejecutar estas herramientas automáticamente:
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Estructura del Directorio

```
nembogueta-python/
├── data/
│   ├── lsp_letter_videos/               # Videos crudos de letras LSPy
│   ├── lsp_word_videos/                 # Videos crudos de palabras LSPy
│   ├── lsp_phrase_videos/               # Videos crudos de frases LSPy
│   ├── processed_lsp_letter_sequences/  # Secuencias preprocesadas de letras
│   ├── processed_lsp_word_sequences/    # Secuencias preprocesadas de palabras
│   └── processed_lsp_phrase_sequences/  # Secuencias preprocesadas de frases
├── models/
│   ├── h5/                              # Modelos en formato .h5
│   └── tflite/                          # Modelos convertidos a TensorFlow Lite
├── scripts/
│   ├── lsp_letter_video_collection.py   # Recolección de videos de letras
│   ├── lsp_word_video_collection.py     # Recolección de videos de palabras
│   ├── lsp_phrase_video_collection.py   # Recolección de videos de frases
│   ├── preprocess_lsp_letter_videos.py  # Preprocesamiento de videos de letras
│   ├── preprocess_lsp_word_videos.py    # Preprocesamiento de videos de palabras
│   ├── preprocess_lsp_phrase_videos.py  # Preprocesamiento de videos de frases
│   ├── train_cnn_lstm_lsp_letters.py    # Entrenamiento del modelo para letras
│   ├── train_cnn_lstm_lsp_words.py      # Entrenamiento del modelo para palabras
│   ├── train_cnn_lstm_lsp_phrases.py    # Entrenamiento del modelo para frases
│   ├── convert_to_tflite.py             # Conversión de modelos a TensorFlow Lite
│   └── analyze_sequences.py             # Análisis de secuencias preprocesadas
├── src/
│   ├── config/                          # Configuraciones del proyecto
│   ├── data_collection/                 # Módulos de recolección de datos
│   ├── preprocessing/                   # Módulos de preprocesamiento
│   ├── training/                        # Módulos de entrenamiento
│   └── utils/                           # Funciones utilitarias
├── tests/                               # Pruebas unitarias
├── notebooks/                           # Notebooks de Jupyter
├── .env.example                         # Ejemplo de archivo de entorno
├── docker-compose.yml                   # Configuración de Docker
├── requirements.txt                     # Dependencias del proyecto
└── README.md                            # Este archivo
```

## Flujo de Trabajo

### 1. Recolección de Datos
Graba 10 videos por gesto (10 segundos, 300 frames a 30 fps) para cada categoría:
```bash
python scripts/lsp_letter_video_collection.py  # Letras (a-z, ñ)
python scripts/lsp_word_video_collection.py   # Palabras (ej. juicio, abogado)
python scripts/lsp_phrase_video_collection.py # Frases (ej. acceso a la justicia)
```

### 2. Preprocesamiento
Convierte videos en secuencias de esqueletos usando MediaPipe, generando arrays NumPy:
```bash
python scripts/preprocess_lsp_letter_videos.py
python scripts/preprocess_lsp_word_videos.py
python scripts/preprocess_lsp_phrase_videos.py
```

### 3. Análisis de Datos
Verifica la calidad de las secuencias preprocesadas:
```bash
python scripts/analyze_sequences.py
```

### 4. Entrenamiento de Modelos
Entrena modelos CNN-LSTM para cada categoría, guardándolos en `models/h5/`:
```bash
python scripts/train_cnn_lstm_lsp_letters.py  # 27 letras
python scripts/train_cnn_lstm_lsp_words.py    # 10 palabras
python scripts/train_cnn_lstm_lsp_phrases.py  # 3 frases
```

### 5. Conversión a TensorFlow Lite
Convierte los modelos a formato `.tflite` para uso en API y app Android:
```bash
python scripts/convert_to_tflite.py
```

## Ejecución en Docker

Construye y ejecuta el proyecto con Docker:
```bash
docker-compose up --build
```
Los volúmenes de datos y modelos se mapean automáticamente, y las variables de entorno se leen desde `.env`.

## Pruebas

Ejecuta las pruebas unitarias:
```bash
pytest tests/
```

Genera un informe de cobertura:
```bash
coverage run -m pytest
coverage report
```

## Resultados Esperados

- **Precisión de Entrenamiento**:
  - Letras: 95%
  - Palabras: 95%
  - Frases: 95%

## Solución de Problemas

Si no se alcanza la precisión del 95%:
1. Verifica la calidad de los datos en el preprocesamiento.
2. Ajusta hiperparámetros en los scripts de entrenamiento.
3. Recolecta más videos de entrenamiento.
4. Asegura condiciones consistentes de iluminación y posición de la cámara.

## Contribución

1. Sigue las directrices de calidad de código (Black, isort, Flake8).
2. Agrega pruebas unitarias para nueva funcionalidad.
3. Actualiza la documentación según sea necesario.
4. Envía pull requests para revisión.

## Notas

- Copia los modelos `.tflite` al módulo de la API (`nembogueta-api/src/main/resources/models/`) tras la conversión.
- Asegúrate de agregar `.env` a `.gitignore` para proteger configuraciones sensibles.
- Los scripts leen variables de entorno automáticamente usando `python-dotenv` para una configuración flexible.
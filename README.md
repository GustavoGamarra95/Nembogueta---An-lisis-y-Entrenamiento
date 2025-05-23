
# Ñemongeta - Módulo Python

## Descripción

Este módulo contiene los scripts necesarios para la recolección, preprocesamiento, análisis, entrenamiento y conversión de modelos CNN-LSTM para el reconocimiento de gestos en Lenguaje de Señas Paraguayo (LSPy). Los modelos han sido optimizados para alcanzar una precisión del 95% en las categorías de letras, palabras y frases.

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
│   ├── tflite/                          # Modelos convertidos a TensorFlow Lite
│   └── ...                              # Modelos en formato .h5
├── scripts/
│   ├── lsp_letter_video_collection.py   # Recolección de videos de letras
│   ├── lsp_word_video_collection.py     # Recolección de videos de palabras
│   ├── lsp_phrase_video_collection.py   # Recolección de videos de frases
│   ├── preprocess_lsp_letter_videos.py  # Preprocesamiento de videos de letras
│   ├── preprocess_lsp_word_videos.py    # Preprocesamiento de videos de palabras
│   ├── preprocess_lsp_phrase_videos.py  # Preprocesamiento de videos de frases
│   ├── train_cnn_lstm_lsp_letters.py     # Entrenamiento del modelo para letras
│   ├── train_cnn_lstm_lsp_words.py       # Entrenamiento del modelo para palabras
│   ├── train_cnn_lstm_lsp_phrases.py     # Entrenamiento del modelo para frases
│   ├── convert_to_tflite.py              # Conversión de modelos a TensorFlow Lite
│   └── analyze_sequences.py              # Análisis de las secuencias preprocesadas
└── README.md                             # Este archivo
```

## Requisitos

- Python 3.8 o superior
- Bibliotecas necesarias:

```bash
pip install opencv-python mediapipe numpy tensorflow matplotlib scikit-learn
```

- Cámara web (para recolección de videos).
- GPU recomendada para acelerar el entrenamiento (opcional).

## Instrucciones de Configuración y Ejecución

### 1. Preparar el Entorno

Crea y activa un entorno virtual (opcional pero recomendado):

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

Instala las dependencias:

```bash
pip install -r requirements.txt
```

> Nota: Si no tienes un archivo `requirements.txt`, usa el comando `pip install` mencionado en Requisitos.

Crea los directorios necesarios para los datos:

```bash
mkdir -p data/lsp_letter_videos data/lsp_word_videos data/lsp_phrase_videos
mkdir -p data/processed_lsp_letter_sequences data/processed_lsp_word_sequences data/processed_lsp_phrase_sequences
mkdir -p models/tflite
```

### 2. Recolección de Datos

Graba videos de gestos LSPy para letras, palabras y frases.

**Letras:**

```bash
python scripts/letter_collection.py
```

- Graba 10 videos por letra (a-z, ñ), cada uno de 10 segundos (300 frames a 30 fps).
- Los videos se guardan en `data/lsp_letter_videos/`.

**Palabras:**

```bash
python scripts/word_collection.py
```

- Graba 10 videos por palabra (ejemplo: juicio, abogado).
- Los videos se guardan en `data/lsp_word_videos/`.

**Frases:**

```bash
python scripts/phrase_collection.py
```

- Graba 10 videos por frase (ejemplo: acceso a la justicia).
- Los videos se guardan en `data/lsp_phrase_videos/`.

### 3. Preprocesamiento de Datos

Convierte los videos en secuencias de esqueletos usando MediaPipe.

**Letras:**

```bash
python scripts/letter_preprocessor.py
```

- Genera arrays NumPy en `data/processed_lsp_letter_sequences/`.

**Palabras:**

```bash
python scripts/word_processor.py
```

- Genera arrays NumPy en `data/processed_lsp_word_sequences/`.

**Frases:**

```bash
python scripts/phrase_processor.py
```

- Genera arrays NumPy en `data/processed_lsp_phrase_sequences/`.

### 4. Análisis de Datos

Analiza las secuencias preprocesadas para verificar su calidad:

```bash
python scripts/sequence_analyzer.py
```

### 5. Entrenamiento de Modelos

Entrena los modelos CNN-LSTM para cada categoría.

**Letras:**

```bash
python scripts/letter_model_trainer.py
```

- Entrena un modelo para las 27 letras (a-z, ñ).
- El modelo se guarda en `models/cnn_lstm_lsp_letters_model.h5`.

**Palabras:**

```bash
python scripts/word_model_trainer.py
```

- Entrena un modelo para las 10 palabras.
- El modelo se guarda en `models/cnn_lstm_lsp_words_model.h5`.

**Frases:**

```bash
python scripts/phrase_model_trainer.py
```

- Entrena un modelo para las 3 frases.
- El modelo se guarda en `models/cnn_lstm_lsp_phrases_model.h5`.

### 6. Conversión a TensorFlow Lite

Convierte los modelos a TensorFlow Lite para su uso en la API y la app Android.

```bash
python scripts/model_converter.py
```

- Los modelos `.tflite` se guardan en `models/tflite/`.

## Resultados Esperados

- **Precisión de Entrenamiento:**
  - Letras: 95%
  - Palabras: 95%
  - Frases: 95%



## Notas

- Si no alcanzas la precisión del 95%, considera recolectar más videos o ajustar los hiperparámetros en los scripts de entrenamiento.
- Asegúrate de copiar los modelos `.tflite` al módulo de la API (`nembogueta-api/src/main/resources/models/`) después de la conversión.

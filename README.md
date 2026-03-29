# Nembogueta — Análisis y Entrenamiento

**Sistema de Reconocimiento de Lengua de Señas Paraguaya (LSP)**
**Módulo de Análisis, Preprocesamiento y Entrenamiento**

> "Nembogueta" deriva del guaraní *ñemongeta* (conversar, fazer falar) — GUASCH, 1948.

Desarrollado en **UniAmérica**, Foz do Iguaçu/PR — directamente en la frontera con Paraguay.
**Gustavo Ariel Gamarra Rojas** | TCC I — Março 2026

---

## Descripción

Este módulo contiene los pipelines completos de preprocesamiento, entrenamiento y evaluación de modelos de reconocimiento de Lengua de Señas Paraguaya (LSP), con salida **trilíngue** según selección del usuario:

- **Español paraguayo**
- **Guaraní**
- **Português (Brasil)**

El proyecto está orientado a la región de **tríplice fronteira**:
Ciudad del Este (PY) / Foz do Iguaçu (BR) / Puerto Iguazú (AR).

### Trabajo con LIBRAS

Como base de desarrollo se utilizan dos datasets de LIBRAS (Língua Brasileira de Sinais):
- **V-LIBRASIL** (UFPE) — frases completas, 1361 clases
- **LIBRAS-HC-RGBDS** (UFPR) — formas de mano (handshapes), 61 clases

Estos modelos base serán adaptados a LSP mediante transfer learning y nuevos datos de colecta.

---

## Estado Actual del Sistema

| Modelo | Dataset | Clases | Test Acc | Estado |
|--------|---------|--------|----------|--------|
| V-LIBRASIL v2 (frases) | V-LIBRASIL (UFPE) | 1361 | **99.4%** | ✅ Entrenado |
| UFPR Handshapes | LIBRAS-HC-RGBDS | 61 | **70.3%** | ✅ Entrenado |
| LibrAI Alphabet | LIBRAS alphabet | 21 letras | **~100%** | ✅ Entrenado |
| LSP letters | Pendente | — | — | ❌ Aguardando dados |

---

## Arquitectura de Modelos

### Feature Engineering — 208 features por frame

Todos los modelos usan el mismo pipeline de extracción de features sobre los 21 landmarks de MediaPipe Hands:

```
MediaPipe Hands → 126 coords brutas (21 lm × 2 manos × 3 xyz)
    ↓  src/preprocessing/feature_engineering.py
208 features por frame:
  - 42 posición (21 lm × 2 xy)
  - 12 distancias entre landmarks clave
  - 30 ángulos (sin/cos)
  - 20 features de movimiento
  = 104 features × 2 manos = 208
```

### V-LIBRASIL v2 — DNN (mean+std pooling)

```
Secuencia (T, 208)
    → mean pooling + std pooling → (416,)
    → normalizar con norm_mean.npy / norm_std.npy
    → Dense(512) → Dense(512) → Dense(256) → Dense(256) → Dense(1361, softmax)
```

### UFPR Handshapes — DNN (mean pooling)

```
Secuencia (T, 208)
    → mean pooling → (208,)
    → Dense layers → Dense(61, softmax)
```

### LibrAI Alphabet — CNN-LSTM

```
Secuencia (T, 208)
    → Conv1D(64) → BatchNorm → Dropout(0.3)
    → Conv1D(128) → BatchNorm → Dropout(0.3)
    → Conv1D(256) → BatchNorm → Dropout(0.3)
    → LSTM(256, return_sequences=True) → Dropout(0.4)
    → LSTM(128) → Dropout(0.4)
    → Dense(64, ReLU) → Dropout(0.3)
    → Dense(21, softmax)
```

---

## Estructura del Proyecto

```
Nembogueta---An-lisis-y-Entrenamiento/
│
├── src/                              # Librería principal (importable)
│   ├── config/config.py              # Parámetros globales y rutas
│   ├── preprocessing/
│   │   ├── feature_engineering.py   # 208 features por frame (núcleo del pipeline)
│   │   ├── letter_preprocessor.py   # LetterPreprocessor (alfabeto)
│   │   ├── phrase_processor.py
│   │   └── word_processor.py
│   ├── training/
│   │   ├── letter_model_trainer.py  # CNN-LSTM trainer
│   │   ├── phrase_model_trainer.py
│   │   └── word_model_trainer.py
│   ├── inference/
│   │   └── unified_predictor.py     # LibrasUnifiedPredictor
│   ├── data_collection/             # Colecta de datos en tiempo real
│   └── utils/                       # Validadores y conversores
│
├── scripts/                         # Entry points ejecutables
│   ├── preprocess/                  # Extracción de features: video → .npy
│   ├── train/                       # Entrenamiento de modelos
│   ├── evaluate/                    # Evaluación y métricas
│   ├── demo/                        # Demos en tiempo real (cámara)
│   ├── inference/                   # Inferencia sobre archivos
│   ├── docker/                      # Helpers de Docker
│   └── utils/                       # Utilidades varias
│
├── data/
│   ├── models/
│   │   ├── vlibrasil-v2/            # best_model.keras + norm_*.npy + metadata.json
│   │   ├── ufpr-handshape/          # best_model.keras + metadata.json
│   │   └── librai-alphabet/         # best_model.keras + norm_*.npy + model_info.json
│   └── processed/
│       └── librai_alphabet/         # Features .npy por letra (A/, B/, ...)
│
├── src/                             # Datasets (en host, no en git)
│   ├── "videos UFPE (V-LIBRASIL)/data/"
│   └── LIBRAS-HC-RGBDS-2011/
│
├── tests/
├── docs/source/                     # Documentación Sphinx
├── Dockerfile
├── docker-compose.yml               # Container GPU: nembogueta-dev-gpu
├── requirements.txt
└── pyproject.toml
```

---

## Inicio Rápido

### 1. Instalación (desarrollo local)

```bash
git clone <repository-url>
cd Nembogueta---An-lisis-y-Entrenamiento

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Entorno Docker (recomendado para entrenamiento con GPU)

```bash
# Build del container
docker compose --profile gpu build

# Entrar al container interactivo
docker compose --profile gpu run nembogueta-dev-gpu bash

# Permitir display para demos con cámara (en el host, antes del demo)
xhost +local:docker
```

---

## Flujo Completo: Preprocesamiento → Entrenamiento → Evaluación

### V-LIBRASIL (frases, 1361 clases)

```bash
# 1. Preprocesar videos → features .npy
python scripts/preprocess/preprocess_vlibrasil.py \
  --data-dir "/app/src/videos UFPE (V-LIBRASIL)/data" \
  --output-dir /data/processed/v-librasil-flat

# 2. Entrenar modelo v2 (DNN + mean+std pooling)
python scripts/train/train_vlibrasil.py \
  --data-dir /data/processed/v-librasil-flat \
  --output-dir /data/models/vlibrasil-v2

# 3. Evaluar
python scripts/evaluate/test_vlibrasil_v2.py \
  --model-dir /data/models/vlibrasil-v2 \
  --data-dir /data/processed/v-librasil-flat \
  --n-samples 50
```

### UFPR Handshapes (61 clases)

```bash
# 1. Convertir archivos .oni a .mp4 (si es necesario)
python scripts/utils/convert_oni_to_mp4.py \
  --input-dir /app/src/LIBRAS-HC-RGBDS-2011

# 2. Preprocesar
python scripts/preprocess/preprocess_ufpr.py \
  --data-dir /app/src/LIBRAS-HC-RGBDS-2011 \
  --output-dir /data/processed/ufpr-handshape

# 3. Entrenar
python scripts/train/train_ufpr.py \
  --data-dir /data/processed/ufpr-handshape \
  --output-dir /data/models/ufpr-handshape

# 4. Evaluar
python scripts/evaluate/evaluate_handshape.py \
  --model-dir /data/models/ufpr-handshape \
  --data-dir /data/processed/ufpr-handshape \
  --view all
```

### LibrAI Alphabet (21 letras)

```bash
# 1. Preprocesar videos de letras
python scripts/preprocess/preprocess_alphabet_videos.py \
  --data-dir /data/lsp_letter_videos \
  --output-dir /data/processed/librai_alphabet

# 2. Entrenar
python scripts/train/train_alphabet.py \
  --data-dir /data/processed/librai_alphabet \
  --output-dir /data/models/librai-alphabet

# 3. Evaluar
python scripts/evaluate/evaluate_alphabet.py \
  --model /data/models/librai-alphabet/best_model.keras \
  --model-info /data/models/librai-alphabet/model_info.json \
  --data-dir /data/processed/librai_alphabet \
  --output-dir /data/models/librai-alphabet/evaluation
```

---

## Scripts de Evaluación — Flujo Detallado

### `test_vlibrasil_v2.py` — V-LIBRASIL v2 (modelo actual)

Pipeline interno:
```
NPY (T, 208) → mean+std pooling → (416,) → normalizar → predict → top-5
```

```bash
# Modo 1: probar un NPY específico
python scripts/evaluate/test_vlibrasil_v2.py \
  --npy /data/processed/v-librasil-flat/Amigo/Amigo_Articulador1.npy

# Modo 2: N muestras aleatorias del dataset
python scripts/evaluate/test_vlibrasil_v2.py \
  --model-dir /data/models/vlibrasil-v2 \
  --data-dir /data/processed/v-librasil-flat \
  --n-samples 50
```

Salida: top-1 / top-5 accuracy por muestra y resumen final.

---

### `evaluate_vlibrasil.py` — V-LIBRASIL (runs anteriores / legacy)

Pipeline interno:
```
NPYs planos (data_dir/*.npy) → normalización interna → train_test_split(seed=42) → predict
```

```bash
python scripts/evaluate/evaluate_vlibrasil.py \
  --model /data/models/vlibrasil/run_20260226_001428/best_model.keras \
  --model-info /data/models/vlibrasil/run_20260226_001428/model_info.json \
  --data-dir /data/processed/v-librasil-flat \
  --output-dir /data/models/vlibrasil/run_20260226_001428/evaluation
```

Archivos generados en `--output-dir`:
- `summary_metrics.json` — accuracy, top-3, top-5, precision/recall/F1 macro
- `classification_report.json` / `.txt` — métricas por clase
- `confusion_matrix.npy` + `confusion_matrix.png` / `_normalized.png`
- `error_analysis.json` — top 100 errores con mayor confianza
- `confidence_distribution.png` — histograma predicciones correctas vs incorrectas
- `per_class_performance.json` — accuracy por clase (top 10 mejores/peores)

---

### `evaluate_alphabet.py` — Alfabeto LibrAI

Pipeline interno:
```
NPYs (data_dir/A_000.npy, ...) → extrae letra del nombre → normalización → split(seed=42) → predict
```

```bash
python scripts/evaluate/evaluate_alphabet.py \
  --model /data/models/librai-alphabet/best_model.keras \
  --model-info /data/models/librai-alphabet/model_info.json \
  --data-dir /data/processed/librai_alphabet \
  --output-dir /data/models/librai-alphabet/evaluation
```

Archivos generados (adicionales al estándar):
- `letter_confusions.json` — pares de letras confundidas (ej. U→V)
- `per_letter_performance.json` — accuracy por letra
- `per_letter_performance.png` — barras: verde ≥90%, naranja ≥70%, rojo <70%
- `confusion_matrix_normalized.png` + `_absolute.png`

---

### `evaluate_handshape.py` — UFPR Handshapes

Pipeline interno:
```
NPYs en test_dir/view/class_XX/*.npy → model.evaluate() + predict → top-5 manual
```

```bash
# Evaluar una vista específica con análisis completo
python scripts/evaluate/evaluate_handshape.py \
  --model-dir /data/models/ufpr-handshape \
  --data-dir /data/processed/ufpr-handshape \
  --view front \
  --confusion-matrix \
  --analyze-errors \
  --output-dir /data/models/ufpr-handshape/evaluation

# Comparar todas las vistas (front/back/left/right)
python scripts/evaluate/evaluate_handshape.py \
  --model-dir /data/models/ufpr-handshape \
  --data-dir /data/processed/ufpr-handshape \
  --view all
```

Salida: classification report (macro avg), `confusion_matrix_{view}_top20.png`, análisis de errores con mayor confianza.

---

### `evaluate_translation.py` — Traducción PT-BR → LIBRAS

Pipeline interno:
```
test.json (token IDs) → padding → decodificación autoregresiva token a token (hasta <EOS>) → métricas
```

```bash
python scripts/evaluate/evaluate_translation.py \
  --model-dir /data/models/translation \
  --data-dir /data/processed/pt_br2libras \
  --examples 10
```

Métricas calculadas:
- `sequence_accuracy` — secuencia completa exactamente correcta
- `token_accuracy` — accuracy por token individual
- `bleu_simple` — BLEU unigrama (precisión de tokens en común)
- Estadísticas de longitud promedio predicción vs target

> Nota: el modelo de traducción aún no ha sido entrenado — requiere dataset PT-BR→LIBRAS.

---

## Demos en Tiempo Real

```bash
# Habilitar display en el host
xhost +local:docker

# Demo V-LIBRASIL v2 (frases)
docker compose --profile gpu run nembogueta-dev-gpu \
  python scripts/demo/realtime_vlibrasil.py

# Demo alfabeto LibrAI
docker compose --profile gpu run nembogueta-dev-gpu \
  python scripts/demo/realtime_librai_alphabet.py
```

---

## Predictor Unificado

```python
from src.inference.unified_predictor import LibrasUnifiedPredictor

predictor = LibrasUnifiedPredictor(models_dir="data/models")

# Predicción desde un frame de video (numpy array BGR)
predictions = predictor.predict_from_frame(frame, draw_landmarks=True)
# predictions incluye: letra, handshape, frase, confianzas
```

---

## Ejecución en Docker — Referencia Rápida

```bash
# Build
docker compose --profile gpu build

# Entrar al container
docker compose --profile gpu run nembogueta-dev-gpu bash

# Verificar GPU disponible
docker compose --profile gpu run nembogueta-dev-gpu python scripts/utils/verify_gpu.py

# Ejecutar script directamente
docker compose --profile gpu run nembogueta-dev-gpu \
  python scripts/evaluate/test_vlibrasil_v2.py
```

Rutas dentro del container:
- Código: `/app/`
- Datos/modelos: `/data/`
- Dataset V-LIBRASIL: `/app/src/videos UFPE (V-LIBRASIL)/data/`
- Dataset UFPR: `/app/src/LIBRAS-HC-RGBDS-2011/`

---

## Dependencias Principales

| Componente | Tecnología |
|---|---|
| Extracción de landmarks | MediaPipe Hands |
| Procesamiento de video | OpenCV 4.8 |
| Lenguaje | Python 3.10+ |
| Framework de DL | TensorFlow 2.10 / Keras |
| Métricas | scikit-learn |
| Manipulación de datos | NumPy / Pandas |
| Visualización | Matplotlib / Seaborn |
| API de inferencia | FastAPI |
| Entorno de entrenamiento | Docker + NVIDIA GPU |

```bash
pip install -r requirements.txt
```

---

## Solución de Problemas

### GPU no detectada en Docker

```bash
# Verificar NVIDIA Container Toolkit instalado
nvidia-smi
docker compose --profile gpu run nembogueta-dev-gpu python scripts/utils/verify_gpu.py
```

### Error con archivos .oni (dataset UFPR)

OpenNI2 puede crashear después de ~200 archivos. Usar `--skip-existing` para retomar:
```bash
python scripts/utils/convert_oni_to_mp4.py \
  --input-dir /app/src/LIBRAS-HC-RGBDS-2011 \
  --skip-existing
```

### Cámara no detectada en el demo

```bash
# Verificar dispositivos disponibles en el host
ls /dev/video*

# Asegurarse que docker-compose.yml tenga los devices montados
# devices: ["/dev/video0:/dev/video0", "/dev/video1:/dev/video1"]
```

---

## Hoja de Ruta

### Fase 1 — Infraestructura y modelos base (completada)
- [x] Pipeline de feature engineering (208 features/frame)
- [x] Modelo V-LIBRASIL v2 (99.4% test, 1361 clases)
- [x] Modelo UFPR handshapes (70.3% test, 61 clases)
- [x] Modelo alfabeto LibrAI (~100% test, 21 letras)
- [x] Predictor unificado multi-modelo
- [x] Entorno Docker con GPU

### Fase 2 — Datos y modelos LSP
- [ ] Colecta de videos de letras LSP (A-Z, Ñ)
- [ ] Entrenamiento de modelo de letras LSP
- [ ] Colecta de palabras y frases LSP (tríplice fronteira)
- [ ] Entrenamiento con datos propios LSP

### Fase 3 — Salida trilíngue
- [ ] Integración de traducción Español paraguayo → LSP
- [ ] Integración de traducción Guaraní → LSP
- [ ] Integración de traducción Português BR → LSP
- [ ] Sistema de selección de idioma de salida

### Fase 4 — Despliegue
- [ ] API REST de inferencia (FastAPI)
- [ ] Optimización para edge devices / TFLite
- [ ] Interface de usuario final

---

## Contribución

1. Sigue las directrices de calidad (Black, isort, Flake8 — ver `.pre-commit-config.yaml`)
2. Agrega pruebas unitarias en `tests/` para nueva funcionalidad
3. Actualiza este README al cambiar el estado de modelos o pipelines

---

## Agradecimientos

- Dataset **V-LIBRASIL** (UFPE) — base para desarrollo de frases
- Dataset **LIBRAS-HC-RGBDS** (UFPR) — base para handshapes
- **MediaPipe** (Google) — extracción de landmarks de mano
- Comunidad sorda de la tríplice fronteira

---

*Nembogueta — fazer falar, conversar. Desenvolvido na tríplice fronteira.*

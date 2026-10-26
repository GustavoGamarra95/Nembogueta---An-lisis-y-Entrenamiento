# Nembogueta — Análisis y Entrenamiento

[![CI](https://github.com/GustavoGamarra95/Nembogueta---An-lisis-y-Entrenamiento/actions/workflows/ci.yml/badge.svg)](https://github.com/GustavoGamarra95/Nembogueta---An-lisis-y-Entrenamiento/actions/workflows/ci.yml)

**Sistema de Reconocimiento de Lengua de Señas Paraguaya (LSP)**
**Módulo de Análisis, Preprocesamiento y Entrenamiento**

> "Nembogueta" deriva del guaraní *ñemongeta* (conversar, fazer falar) — GUASCH, 1948.

Desarrollado en **UniAmérica**, Foz do Iguaçu/PR — directamente en la frontera con Paraguay.
**Gustavo Ariel Gamarra Rojas** | TCC

---

## Descripción

Pipelines de preprocesamiento, entrenamiento y evaluación de modelos de
reconocimiento de Lengua de Señas Paraguaya (LSP), con visión a salida
**trilíngue** según selección del usuario:

- **Español paraguayo**
- **Guaraní**
- **Português (Brasil)**

El proyecto está orientado a la región de **tríplice fronteira**:
Ciudad del Este (PY) / Foz do Iguaçu (BR) / Puerto Iguazú (AR).

### Trabajo con LIBRAS

Como base de desarrollo se utilizan tres datasets de LIBRAS (Língua
Brasileira de Sinais):

- **V-LIBRASIL** (UFPE) — frases completas, 1361 clases
- **LIBRAS-HC-RGBDS** (UFPR) — formas de mano (handshapes), 61 clases
- **LibrAI alphabet** — alfabeto manual LIBRAS, 22 clases (A–Y + Ç)

Estos modelos base serán adaptados a LSP mediante transfer learning y
nuevos datos de colecta. Desde Alicia v5 el alfabeto se entrena en un solo
modelo multilengua (LIBRAS, LSC, ASL y LSPy) y Alicia v6 suma el alfabeto
guaraní (achegety).

La tesis es una **prueba de concepto** basada en LibrAI: no incluye colecta
con terceros (y por lo tanto no pasa por comité de ética); los datos LSPy
provienen de material público autorizado por sus titulares y de grabaciones
del propio autor.

---

## Estado Actual del Sistema

### Modelos del alfabeto (pipeline actual, 120 features)

| Modelo | LibrAI test | S01 holdout | S02 (sujeto nuevo) | Recomendación |
|---|---|---|---|---|
| `librai-alphabet-v4` (base) | **99.78%** | 21% | — | Histórico, NO usar en cámara |
| `librai-alphabet-v4-ft` | 88.51% | 88% | — | Sobreajustado a S01 |
| `librai-alphabet-robust` ⭐ | **97.74%** | **79%** | **80%** | Recomendado — generaliza |
| `librai-alphabet-robust-ft` | 96.41% | 84% | — | Personalizado al operador S01 |
| `librai-alphabet-robust-v2` 🧪 | **99.04%** | — | **83.66%** | Experimental — S01 + S02 parcial |
| `librai-alphabet-robust-v3` 🧪 | **99.35%** | — | **84.21%** | Experimental — S01 + S02-extra + S02-cluster |
| `librai-alphabet-robust-v4` ⭐ | **99.33%** | — | **91.81%** | **Best in class** — multi-sujeto balanceado completo |

**Validación inter-sujeto del `robust`** (sesión S02 grabada con sujeto no
visto durante training):

- Top-1 global S01 holdout: 79% · S02: **80%** → modelo generaliza
- Gap dominio consistente: ~19 pp en ambos sujetos
- 9 letras estables entre sujetos con >90%: `B, C, F, L, M, O, P, Q, U`
- Letras con fuerte variabilidad inter-sujeto: `G, Ç, A, D, R, S` —
  sugieren necesidad de dataset multi-sujeto para letras con
  configuración fina de pulgar.

**Experimento multi-sujeto parcial (`robust-v2`)** — S02 capturó 7 letras
inestables × 10 reps (sesión `20260614_233737_S02-extra`), añadidas al
training. Evaluado sobre S02 full session sin tocar:

| Grupo | Δ top-1 promedio |
|---|---|
| 7 letras CON extra training | **+22.7 pp** (G: 21→80, Ç: 8→44, A: 60→98) |
| 15 letras SIN extra training | -4.9 pp (caso aislado: U colapsó -90 pp) |
| Global S02 | +3.6 pp (80.03% → 83.66%) |

**Hallazgo académico**: agregar data multi-sujeto selectivamente para
letras "difíciles" puede romper letras vecinas en feature space — U se
colapsó porque su frontera con R se desplazó al ver más ejemplos de R
del estilo de S02. Para producción, multi-sujeto debe ser balanceado
por cluster semántico (R-V-U-W se mueven juntas), no solo por
dificultad individual.

**Iteración del experimento (`robust-v3`)** — se agregó al training una
segunda sesión parcial de S02 con las 4 letras vecinas del cluster
(U, T, V, W × 10 reps cada una). Resultados sobre S02 full session:

| Letra | v1 | v2 | v3 | Comentario |
|---|---|---|---|---|
| U | 97% | 8% | **97%** | Recuperada (+89.7 pp vs v2) |
| T | 89% | 76% | 88% | Recuperada |
| V | 81% | 96% | **57%** | **Nueva inestabilidad** (-39 pp vs v2) |
| W | 90% | 96% | 97% | Estable |
| C | 96% | 95% | **68%** | Daño colateral en letra estable |
| F | 98% | 98% | **64%** | Daño colateral en letra estable |
| Global S02 | 80.03% | 83.66% | **84.21%** | +0.55 pp marginal |

**Hallazgo refinado**: el cluster R-V-U-W **no es un set fijo**. Cada
iteración parcial mueve las inestabilidades a letras distintas — v1→v2
rompió U, v2→v3 rompió V (y degradó C, F, N). Es un sistema de
equilibrios competitivos: agregar data para A genera presión sobre B
que comparte feature space.

**Conclusión metodológica para colecta LSP** (Fase 3 del roadmap): el
dataset multi-sujeto efectivo debe ser **completo y balanceado** —
todas las letras del alfabeto × todos los sujetos × mismo número de
reps. La colecta parcial selectiva es intrínsecamente inestable y
genera resultados de mejora marginal con costo de daños colaterales
impredecibles.

**Validación experimental del principio (`robust-v4`)** — se entrenó con
S02 completa (todas las letras × 5 reps) + S01 + S02-extra + S02-cluster.
Resultado: las letras que oscilaban entre v2 y v3 se estabilizan todas
en v4, y top-1 global sobre S02 sube a **91.81%**:

| Métrica | v1 | v2 | v3 | **v4** |
|---|---|---|---|---|
| Top-1 S02 | 80.0% | 83.7% | 84.2% | **91.8%** |
| Top-5 S02 | 95.4% | 98.4% | 98.4% | **99.3%** |
| Confianza en target | 0.766 | 0.813 | 0.821 | **0.903** |
| Entropía promedio | 0.226 | 0.271 | 0.227 | **0.134** |
| Robustez ruido σ=0.5 | 74.8% | 79.9% | 80.6% | **89.6%** |
| LibrAI test | 97.74% | 99.04% | 99.35% | 99.33% |

v4 es más preciso (+11.8 pp vs v1), más decisivo (entropía -41%) y
~3× más robusto a perturbaciones de input. Solo S (62%) y Ç (56%) se
mantienen como puntos débiles estructurales, indicando handshapes con
variabilidad inter-sujeto irreducible sin más datos.

Análisis completo en `data/analysis/robust_models/` (curvas de
aprendizaje, matrices de confusión, curvas de robustez al ruido,
sanitización por confianza, heatmaps per-letra).

Lineage:
```
v4 (base LibrAI puro, 99.78% holdout, 24% real → overfit)
 ├─ v4-ft (fine-tune sobre v4 con datos del operador)
 │     ✗ catastrophic forgetting: -11pp en LibrAI
 │
 └─ robust = v4 reentrenado con:
       · LibrAI + datos operador (sample weighted)
       · augmentation: palm rotation ±20°, jitter 0.5σ, permutación de slots
         (v1–v4; desde alicia-alphabet-robust-mirror: espejo geométrico)
       · dropout 0.4
    └─ robust-ft (fine-tune sobre robust)
          + recupera V (única letra débil del robust): 0% → 98%
```

### Alicia v5 y v6 — alfabeto multilengua con LSPy y guaraní

Fichas completas en `data/models/alicia-v5/README.md` y
`data/models/alicia-v6/README.md`. Ambas versiones combinan un modelo
**estático** (frame a frame, 120 features; `train_alphabet_multilang.py`) y
uno de **movimiento** para las letras dinámicas de LSPy (secuencia de 8
pasos; `train_letter_sequences.py`). En inferencia se elige la lengua y el
argmax se restringe a sus clases (`LIBRAS:*`, `LSP:*` = LSC, `ASL:*`,
`LSPY:*` y, en v6, `GN:*`).

| Conjunto | v5 | v6 |
|---|---|---|
| Signario SLS (LSPy, 1 intérprete externo), estático | 15/20 | **16/20** |
| Signario SLS, fusión estático + movimiento | 16/20 | 16/20 |
| Operador en cámara real (LIBRAS) | 85.6% | **92.6%** |
| LibrAI (LIBRAS) | 98.8% | 97.6% |
| LSC / ASL, sujetos no vistos | 81.0% / 84.6% | 79.6% / 85.1% |
| GN, letras compartidas, sujetos no vistos | — | 69.4% |

- **v5**: LIBRAS (LibrAI + S01/S02), LSC (LSC70), ASL (Massey + Spelling It
  Out) y LSPy (proxy letra por letra + frames reales de 8 videos de 7 canales
  públicos). Movimiento: LSC70, LSM dinámico y secuencias LSPy reales.
- **v6**: v5 + alfabeto guaraní (achegety, lengua `GN`, 33 letras). Las 13
  letras propias del guaraní (ã, ẽ, ĩ, õ, ũ, ỹ, g̃, ch, mb, nd, ng, nt,
  puso) salen de un único video leído frame a frame
  (`data/analysis/achegety/`): **prototipo sin validación externa**. El
  modelo de movimiento todavía no tiene clases GN.
- La validación LSPy es un solo intérprete con una toma por letra (el
  signario no tiene B, C, E, F, G, H ni W); la fusión de v5 falla en N, U, X
  y Ñ.

### Alicia — nombre de los modelos entrenados

A partir de ahora, los modelos que producen los scripts de `scripts/train/`
se llaman **Alicia**: se guardan en carpetas `alicia-*` y su
`model_info.json` / `metadata.json` incluye `"model_name": "Alicia"`.

| Script | Salida por defecto |
|---|---|
| `train_alphabet.py` | `models/alicia-alphabet` |
| `train_alphabet_robust.py` | `/data/models/alicia-alphabet-robust` |
| `finetune_alphabet_realworld.py` | `/data/models/alicia-alphabet-ft` |
| `cv_finetune_alphabet.py` | `/data/models/alicia-alphabet-ft-cv` |
| `train_vlibrasil.py` | `/data/models/alicia-phrases` |
| `train_ufpr.py` | `/data/models/alicia-handshape` |
| `train_translation.py` | `/data/models/alicia-translation` |
| `train_alphabet_multilang.py` | `/data/models/alicia-alphabet-multilang` |
| `train_letter_sequences.py` | `data/models/alicia-lspy-sequences` |

Los modelos ya entrenados (`librai-alphabet-*`, `vlibrasil-v2`,
`ufpr-handshape`) conservan su nombre porque el paper y las sesiones de
evaluación los referencian.

### Modelos de frases y handshapes (pipeline legacy, 208 features)

| Modelo | Dataset | Clases | Test Acc | Estado |
|---|---|---|---|---|
| V-LIBRASIL v2 | V-LIBRASIL (UFPE) | 1361 | 99.4% (frases: 98.2%) | ⚠️ Pipeline legacy — incompatible con `feature_engineering.py` actual |
| UFPR Handshapes | LIBRAS-HC-RGBDS | 61 | 70.3% | ⚠️ Pipeline legacy — incompatible con `feature_engineering.py` actual |
| LSP letters | — | — | — | ➡️ Cubiertas por Alicia v5/v6 (alfabeto multilengua, ver arriba) |

⚠️ **Importante:** V-LIBRASIL v2 y UFPR Handshapes fueron entrenados con
un extractor de features anterior (208 features con posición absoluta +
motion). El pipeline actual (`src/preprocessing/feature_engineering.py`)
produce 120 features invariantes. Para usar estos modelos hay que
restaurar el extractor viejo o retrainear con el nuevo. Pendiente
decisión: ver `docs/source/protocols/`.

**Evaluación de V-LIBRASIL v2 sobre frases** — de las 1361 clases, las que
son frases (nombre con espacios) se evaluaron por separado con
`scripts/evaluate/evaluate_phrases_vlibrasil.py`: **795 muestras,
top-1 98.2%, top-5 100%**. Reporte por frase en
`data/analysis/phrases_report.json` y videos demo en
`data/analysis/videos/`.

---

## Feature Engineering — 120 features invariantes por frame

```
MediaPipe Hands → 126 coords brutas (21 lm × 2 manos × 3 xyz)
    ↓  src/preprocessing/feature_engineering.py
120 features por frame:
  Por mano (60 features):
    - 21 distancias entre landmarks clave (normalizadas por hand_size)
    - 30 ángulos articulares (15 triplets × sin+cos, 2D invariante)
    - 5 finger curl (z_tip - z_mcp por dedo, normalizado)
    - 4 palm angles (sin/cos de pitch y yaw del vector normal a la palma)
  Total: 60 × 2 manos = 120
```

**Invariancias garantizadas matemáticamente:**
- Traslación de la mano en el frame
- Escala (tamaño de mano del operador)
- Rotación 2D dentro del plano de la cámara

**Sensibilidades residuales** (medidas empíricamente, ver `docs/source/protocols/realworld_evaluation.md`):
- Orientación 3D de la palma (pitch/yaw) — abordada por augmentation en `robust`
- Estilo individual de signantes (curl exacto, pose del pulgar)
- Ruido en coordenada z de MediaPipe

### Arquitecturas

**Modelos del alfabeto (DNN sobre frame único, 22 clases):**
```
Input (120,)
  → Dense(512) + BN + Dropout(0.3-0.4)
  → Dense(256) + BN + Dropout
  → Dense(128) + BN + Dropout
  → Dense(64) + Dropout
  → Dense(22, softmax)
```
`robust` agrega `tf.data.map(augment_batch)` con rotación de palm angles,
jitter gaussiano y, con probabilidad 0.5, un flip de la mano.

**Flip: permutación de slots (v1–v4) vs. espejo geométrico (actual).**
MediaPipe decide la lateralidad y la geometría 3D a la vez: en manos planas
la palma de una mano y el dorso de la otra tienen la misma silueta, y entre
el 6% y el 36% de los frames reciben la interpretación "otra mano" (slot
opuesto y normal de la palma invertida ~180° en pitch). Los modelos
robust v1–v4 solo permutaban los slots (`x[60:120], x[0:60]`), lo que no
espeja la geometría y genera combinaciones que no ocurren en datos reales.
Desde `--flip-mode mirror` (por defecto) el flip es un espejo geométrico:
cambia el signo de los senos de los 15 ángulos y de
`(sin_pitch, cos_pitch, cos_yaw)` y permuta los slots
(`mirror_features` en `train_alphabet_robust.py`). Validado contra
imágenes reales espejadas: error medio 0.020 vs. 0.658 sin transformar.
La rotación de palma ahora se aplica sobre los sin/cos reales
(`--palm-rot-space raw`); v1–v4 la aplicaban sobre features normalizadas.

| Modelo (seed 42) | Operador test | Operador test espejado | LibrAI test | LibrAI espejado |
|---|---|---|---|---|
| Control: permutación de slots (método v1–v4) | 90.05% | 73.19% | 99.29% | 92.59% |
| Espejo geométrico | 90.21% | 90.37% | 99.22% | 99.29% |
| Espejo + rotación de palma corregida ⭐ | **90.52%** | **89.82%** | 99.33% | 99.17% |

El espejo recupera ~17 pp ante la interpretación "otra mano" sin perder
precisión con la mano dominante. Modelos en
`data/models/alicia-alphabet-robust-{ctrl-slot,mirror-normrot,mirror}/`.

**V-LIBRASIL v2 (legacy, 208 features, mean+std pooling sobre secuencia):**
```
Secuencia (T, 208) → mean+std pooling → (416,) → Dense layers → Dense(1361, softmax)
```

**UFPR Handshapes (legacy, 208 features, mean pooling):**
```
Secuencia (T, 208) → mean pooling → (208,) → Dense layers → Dense(61, softmax)
```

---

## Estructura del Proyecto

```
Nembogueta---An-lisis-y-Entrenamiento/
│
├── .github/workflows/ci.yml          # CI: lint (flake8, black, isort) + tests
│
├── src/                              # Librería principal (importable)
│   ├── api/                          # API FastAPI de inferencia (Alicia v6)
│   │   ├── app.py                    # endpoints REST + WebSocket
│   │   └── static/index.html         # demo web local (cámara, captura PNG)
│   ├── config/config.py
│   ├── preprocessing/
│   │   ├── feature_engineering.py   # 120 features invariantes (núcleo)
│   │   ├── letter_preprocessor.py
│   │   ├── phrase_processor.py
│   │   └── word_processor.py
│   ├── training/
│   ├── inference/
│   │   ├── alicia.py                 # carga de Alicia y predicción por lengua
│   │   ├── deletreo.py               # letras → palabras → idioma de salida
│   │   ├── diccionario.py            # diccionario LSP → es / gn / pt
│   │   └── unified_predictor.py      # predictor legacy + output_lang
│   ├── data_collection/
│   └── utils/
│
├── scripts/                          # Entry points ejecutables
│   ├── preprocess/                   # Extracción de features:
│   │                                 # preprocess_alphabet_images, preprocess_lsc70,
│   │                                 # align_alphabet_videos (alineamiento forzado LSPy),
│   │                                 # extract_letter_sequences, build_lspy_proxy,
│   │                                 # build_gn_dataset, build_lspy_landmarks_release...
│   ├── train/                        # train_alphabet, train_alphabet_robust,
│   │                                 # train_alphabet_multilang, train_letter_sequences,
│   │                                 # finetune_alphabet_realworld,
│   │                                 # cv_finetune_alphabet, train_vlibrasil, train_ufpr...
│   ├── collect/                      # collect_lspy_signs (grabación guiada, modo AUTOR)
│   ├── evaluate/                     # Evaluación offline + evaluate_realworld.py
│   ├── demo/                         # realtime_librai_alphabet, realtime_vlibrasil,
│   │                                 # make_video_demo_vlibrasil, make_synth_video...
│   ├── inference/
│   ├── utils/                        # guided_calibration, verify_gpu, capture_letter_images...
│   └── legacy/                       # Scripts deprecados (ver scripts/legacy/README.md)
│
├── data/
│   ├── models/
│   │   ├── librai-alphabet-v4/      # base
│   │   ├── librai-alphabet-v4-ft/
│   │   ├── librai-alphabet-robust/  # v1
│   │   ├── librai-alphabet-robust-ft/
│   │   ├── librai-alphabet-robust-v2/ … -v3/  # experimentales
│   │   ├── librai-alphabet-robust-v4/  # ⭐ recomendado
│   │   ├── librai-alphabet-v4-ft-cv/  # 5-fold CV runs
│   │   ├── vlibrasil-v2/            # legacy (208 features)
│   │   ├── ufpr-handshape/          # legacy (208 features)
│   │   ├── alicia-v5/               # multilengua + LSPy (estático + movimiento)
│   │   ├── alicia-v6/               # v5 + achegety (GN) — usado por la API
│   │   └── alicia-*/                # modelos que se entrenen de ahora en adelante
│   ├── processed/
│   │   ├── librai_alphabet_v4/      # 22 letras × ~2090 frames (120 features)
│   │   ├── v-librasil-flat/         # legacy
│   │   └── ufpr-handshape/          # legacy
│   ├── analysis/
│   │   ├── robust_models/           # análisis comparativo robust v1–v4
│   │   ├── achegety/                # lectura frame a frame del alfabeto guaraní
│   │   ├── vocabulario/             # vocabulario-objetivo (30 señas), alfabetos
│   │   ├── paper_figures/
│   │   ├── videos/                  # demos de frases V-LIBRASIL (.mp4)
│   │   └── phrases_report.json      # evaluación de frases V-LIBRASIL v2
│   ├── release/                     # LSPy-Landmarks v1 generado (no versionado)
│   └── realworld_eval/
│       └── sessions/                # sesiones de evaluación en cámara real
│
├── docs/source/protocols/           # Protocolos académicos (TCC)
│   └── realworld_evaluation.md
├── tests/
├── Dockerfile                       # Python 3.10 + CUDA 11.8
├── docker-compose.yml               # Container GPU: nembogueta-dev-gpu
├── requirements.txt                 # TF 2.10, MediaPipe 0.10.8, OpenCV 4.8 (+ contrib 4.11)
├── requirements-api.txt             # FastAPI, uvicorn, httpx (API)
├── pyproject.toml                   # target Python 3.10
└── LICENSE                          # MIT
```

En git solo se versionan `alicia-v5/`, `alicia-v6/` y los modelos
`*-alphabet-robust*`; el resto de `data/models/` queda local.

Datasets externos (no versionados, viven en `src/`):
- `src/LibrAI/` — alfabeto LIBRAS (referencia visual + frames)
- `src/videos UFPE (V-LIBRASIL)/`
- `src/LIBRAS-HC-RGBDS-2011/`

---

## Inicio Rápido

### Docker GPU (recomendado)

```bash
# Build
docker compose --profile gpu build

# Entrar al container interactivo
docker compose --profile gpu run nembogueta-gpu bash

# X11 para demos con cámara (host, antes del demo)
xhost +local:docker

# Verificar GPU
docker compose --profile gpu run nembogueta-gpu python scripts/utils/verify_gpu.py
```

Rutas dentro del container:
- Código: `/app/`
- Datos/modelos: `/data/`
- Dataset V-LIBRASIL: `/app/src/videos UFPE (V-LIBRASIL)/data/`
- Dataset UFPR: `/app/src/LIBRAS-HC-RGBDS-2011/`

### Entorno local sin Docker (Linux)

TF 2.10 requiere Python 3.10. Con [uv](https://docs.astral.sh/uv/):

```bash
uv python install 3.10
uv venv --seed --python 3.10 .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # ajustar rutas
```

Para entrenar en GPU sin CUDA instalado en el sistema, las librerías de
CUDA 11 y cuDNN 8 se pueden instalar dentro del venv:

```bash
.venv/bin/pip install nvidia-cuda-runtime-cu11==11.8.89 nvidia-cublas-cu11==11.11.3.6 \
  nvidia-cufft-cu11==10.9.0.58 nvidia-curand-cu11==10.3.0.86 \
  nvidia-cusolver-cu11==11.4.1.48 nvidia-cusparse-cu11==11.7.5.86 \
  nvidia-cudnn-cu11==8.6.0.163 nvidia-cuda-nvcc-cu11==11.8.89

# Antes de correr scripts (bash):
NV=.venv/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$(printf '%s:' "$PWD/$NV"/*/lib)$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$PWD/$NV/cuda_nvcc
```

Fuera del container, pasar rutas locales a los scripts (los defaults
apuntan a `/data/` y `/app/`), p. ej. `--librai-dir data/processed/librai_alphabet_v4`.

---

## Flujos principales

### Alfabeto LIBRAS — train, evaluate, fine-tune

```bash
# 1. Preprocesar videos de letras → features .npy (120 dim)
python scripts/preprocess/preprocess_alphabet_videos.py \
  --data-dir /app/src/LibrAI \
  --output-dir /data/processed/librai_alphabet_v4

# 2. Entrenar modelo base (DNN simple)
python scripts/train/train_alphabet.py \
  --data-dir /data/processed/librai_alphabet_v4 \
  --output-dir /data/models/alicia-alphabet

# 3. Entrenar modelo robusto (augmentation + datos operador)
python scripts/train/train_alphabet_robust.py \
  --librai-dir /data/processed/librai_alphabet_v4 \
  --session-dir /data/realworld_eval/sessions/<SESSION_ID> \
  --output-root /data/models/alicia-alphabet-robust

# 4. Evaluar offline (LibrAI holdout)
python scripts/evaluate/evaluate_alphabet.py \
  --model /data/models/alicia-alphabet-robust/<run>/best_model.keras \
  --model-info /data/models/alicia-alphabet-robust/<run>/model_info.json \
  --data-dir /data/processed/librai_alphabet_v4 \
  --output-dir /data/models/alicia-alphabet-robust/<run>/evaluation

# 5. Evaluar en cámara real (sesión interactiva ~18 min)
python scripts/evaluate/evaluate_realworld.py \
  --model-dir /data/models/alicia-alphabet-robust/<run> \
  --subject-id S02 --reps 5

# 6. Fine-tune sobre sesión real
python scripts/train/finetune_alphabet_realworld.py \
  --session-dir /data/realworld_eval/sessions/<SESSION_ID> \
  --base-model-dir /data/models/alicia-alphabet-robust/<run>

# 7. Cross-validation 5-fold del fine-tune
python scripts/train/cv_finetune_alphabet.py \
  --session-dir /data/realworld_eval/sessions/<SESSION_ID> \
  --base-model-dir /data/models/alicia-alphabet-robust/<run>
```

### V-LIBRASIL (frases) y UFPR Handshapes — pipeline legacy

⚠️ Estos modelos siguen funcionando solo si se conserva el feature extractor
viejo (208 features). Ver `docs/source/protocols/` para el plan de migración.

```bash
# V-LIBRASIL
python scripts/preprocess/preprocess_vlibrasil.py \
  --data-dir "/app/src/videos UFPE (V-LIBRASIL)/data" \
  --output-dir /data/processed/v-librasil-flat
python scripts/train/train_vlibrasil.py \
  --data-dir /data/processed/v-librasil-flat \
  --output-dir /data/models/alicia-phrases
python scripts/evaluate/evaluate_phrases_vlibrasil.py \
  --model-dir /data/models/vlibrasil-v2 \
  --output /data/analysis/phrases_report.json

# UFPR
python scripts/utils/convert_oni_to_mp4.py --input-dir /app/src/LIBRAS-HC-RGBDS-2011
python scripts/preprocess/preprocess_ufpr.py \
  --data-dir /app/src/LIBRAS-HC-RGBDS-2011 \
  --output-dir /data/processed/ufpr-handshape
python scripts/train/train_ufpr.py \
  --data-dir /data/processed/ufpr-handshape \
  --output-dir /data/models/alicia-handshape
```

---

## Evaluación en cámara real

Protocolo completo: [`docs/source/protocols/realworld_evaluation.md`](docs/source/protocols/realworld_evaluation.md)

```bash
xhost +local:docker
docker exec -it -e DISPLAY=$DISPLAY nembogueta-dev-gpu \
  python /app/scripts/evaluate/evaluate_realworld.py \
    --model-dir /data/models/librai-alphabet-robust/<run> \
    --subject-id S01 --reps 5
```

Captura ventana fija de 60 frames (~2s) por repetición, 5 reps × 22 letras
≈ 18 min. Genera por sesión:

- `session.json` — metadatos del sujeto y condiciones
- `report.json` — top-1 / top-5 / per-letter / gap_vs_holdout
- `confusion_matrix.png` y `.npy`
- `letters/<L>_rep<N>.npz` — features, probs y top-5 por frame
  (permite re-evaluar con futuros modelos sin re-grabar)

---

## Demos en tiempo real

```bash
xhost +local:docker

# Alfabeto LibrAI
docker compose --profile gpu run nembogueta-gpu \
  python scripts/demo/realtime_librai_alphabet.py

# V-LIBRASIL (legacy)
docker compose --profile gpu run nembogueta-gpu \
  python scripts/demo/realtime_vlibrasil.py
```

### Videos demo (offline)

```bash
# V-LIBRASIL: videos → landmarks → 208 features → top-3 (OUT = archivo de salida,
# PHRASES_ONLY=1 limita a frases)
docker compose --profile gpu run -e OUT=/data/analysis/videos/phrases_demo.mp4 \
  -e PHRASES_ONLY=1 nembogueta-gpu python scripts/demo/make_video_demo_vlibrasil.py

# Alfabeto LibrAI con MediaPipe Tasks API (requiere hand_landmarker.task en /app)
docker compose --profile gpu run nembogueta-gpu python scripts/demo/make_synth_video.py
```

`scripts/demo/_fe208.py` contiene el extractor legacy de 208 features que
usan estos demos de V-LIBRASIL.

---

## API de inferencia (Alicia v6)

API FastAPI en `src/api/` con el modelo `data/models/alicia-v6`
(override con `ALICIA_MODEL_DIR`). Reconoce el alfabeto manual de LSPy,
guaraní (achegety), LIBRAS, LSC y ASL, y traduce lo deletreado al español
paraguayo, guaraní o portugués con el diccionario LSP.

```bash
pip install -r requirements-api.txt
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

**Demo web local:** http://localhost:8000 — cámara con el esqueleto de la
mano superpuesto, letra y top-3, texto deletreado con su traducción a los
tres idiomas, buscador del diccionario, modo "subir imagen" y botón
**📸 Capturar**, que descarga un PNG con la predicción incrustada. Bajar la
mano un momento separa palabras (o barra espaciadora; Retroceso borra una
letra). Documentación interactiva de la API en `/docs`.

| Endpoint | Uso |
|---|---|
| `GET /salud`, `/lenguas`, `/idiomas` | Estado, lenguas (con sus letras) e idiomas de salida |
| `POST /predecir/landmarks` | Letra a partir de landmarks de MediaPipe calculados en el cliente sobre la imagen **sin espejar** (`sostenido=true` usa el tramo estable) |
| `POST /predecir/imagen` | Letra a partir de una imagen JPG/PNG |
| `POST /traducir` | Palabra o texto (español o glosa LSP) → `es` / `gn` / `pt`; agrupa frases de hasta 4 palabras ("buenos días") |
| `GET /diccionario/{palabra}` | Entrada del diccionario LSP en los tres idiomas |
| `WS /ws?lengua=LSPY&idioma=gn` | Tiempo real: frames (imagen base64 o landmarks) → letra, texto deletreado y traducción; acciones `espacio`, `retroceder`, `borrar` y cambio de idioma o lengua |

La API espeja los landmarks matemáticamente (x → 1 − x) para llevarlos a la
geometría de entrenamiento; en el signario SLS da 14/20 frente a 16/20 de la
validación offline (que espeja la imagen antes de MediaPipe). Las
traducciones del diccionario son de una IA y están sin revisar
(`/diccionario` y `/traducir` informan el estado).

Es un **prototipo**: sin autenticación ni despliegue. Tests en
`tests/test_api/` (corren en el CI):

```bash
pytest tests/test_api
```

## Dataset LSPy-Landmarks v1

Landmarks crudos de MediaPipe Hands (sin videos), publicados en Kaggle con
licencia **CC BY 4.0**: https://www.kaggle.com/datasets/gustavogamarra/lspy-landmarks-v1
(con el notebook de ejemplo
[LSPy-Landmarks quickstart](https://www.kaggle.com/code/gustavogamarra/lspy-landmarks-quickstart),
fuente en `notebooks/kaggle/`). El uso de las fuentes fue autorizado por sus
titulares; los videos no se redistribuyen.

Línea base del notebook (regresión logística sobre la mano normalizada,
dejando un canal afuera): **75 %** de acierto medio en 24 letras estáticas y
8 canales (62–88 % según el canal).

| Subconjunto | Muestras | Contenido |
|---|---|---|
| `alfabeto` | 243 | Alfabeto manual castellano (A–Z, Ñ, LL, RR), 9 videos de 8 canales |
| `achegety` | 33 | Alfabeto guaraní, 1 intérprete, tiempos verificados frame a frame |
| `lexico` | 1810 | Señas léxicas del Signario SLS, 1 intérprete |

```bash
python scripts/preprocess/build_lspy_landmarks_release.py --subset alfabeto
python scripts/preprocess/build_lspy_landmarks_release.py --subset lexico --shard 0/4
python scripts/preprocess/build_lspy_landmarks_release.py --merge
```

Sale en `data/release/lspy_landmarks_v1/` (ignorado en git), con su propio
README, licencia e `index.csv` por subconjunto.

## Diccionario LSP → {ES paraguayo, guaraní, PT-BR}

`data/analysis/diccionario/diccionario_lsp.csv`: las **1810 glosas del
Signario SLS** (1744 conceptos; las variantes "X 2", "X 3" y las acepciones
son filas propias) con su video de referencia y traducción a español
paraguayo, portugués de Brasil y guaraní (746 entradas con guaraní).

> ⚠ **Las traducciones las generó una IA (Claude) y nadie las revisó
> todavía.** Una relectura ya encontró errores (p. ej. TALÓN → `pytã`, que
> significa "rojo"; POR LA NOCHE → `pyharekue`, "anoche"), que se
> corrigieron dejando la celda vacía en lugar de reemplazarlos por otra
> suposición. Cada idioma lleva su estado (`es_estado`, `pt_estado`,
> `gn_estado`): `IA sin revisar`, `pendiente` (sin traducción) o `validado`.
> No usar fuera del prototipo hasta que un hablante de guaraní y uno de
> portugués las revisen.

Las correcciones humanas van en `data/analysis/diccionario/revisiones.tsv`
(`clave`, `idioma` es/pt/gn, `traduccion`, `revisor`, `fecha`, `nota`):
pisan a la IA y marcan la celda como `validado`.

```bash
# traducciones + revisiones + índice del signario → diccionario
python scripts/preprocess/build_diccionario_lsp.py
```

```python
from src.inference.diccionario import Diccionario
d = Diccionario()
d.traducir("médico", "gn")                              # 'pohanohára'
d.traducir_texto("buenos dias doctor", "gn")["texto"]   # frase + palabra libre
```

## Vocabulario-objetivo y grabación del autor

`data/analysis/vocabulario/vocabulario_objetivo.csv` define 30 señas
prioritarias del Signario SLS (salud, servicios, cotidiano); su guaraní
(también propuesto por la IA, pendiente de validación) tiene prioridad en
el diccionario. Para grabarlas en cámara real:

```bash
python scripts/collect/collect_lspy_signs.py \
  --vocab data/analysis/vocabulario/vocabulario_objetivo.csv \
  --participant AUTOR --reps 10
```

En modo `AUTOR` el propio investigador se graba a sí mismo (sin terceros).
Guarda landmarks crudos en el mismo formato que LSPy-Landmarks v1 en
`data/collection/` (no versionado).

## Predictor unificado

```python
from src.inference.unified_predictor import LibrasUnifiedPredictor

predictor = LibrasUnifiedPredictor(
    models_dir="data/models", output_lang="gn", sign_language="LSPY"
)
predictions = predictor.predict_from_frame(frame)   # frame sin espejar
predictions["spelling"]   # letra, top-3, texto deletreado y traducción (gn)
predictor.set_output_lang("pt")
predictor.translate("buenos días")                   # 'Bom dia'
```

Si existe `data/models/alicia-v6`, el predictor carga Alicia y deletrea en
`output_lang` (`es`, `gn`, `pt`); los modelos legacy de LIBRAS se cargan
igual que antes cuando están presentes.

---

## Dependencias

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.10 |
| Framework de DL | TensorFlow 2.10 / Keras |
| Extracción de landmarks | MediaPipe Hands 0.10.8 (Docker) / 0.10.35 (host Tasks API) |
| Procesamiento de video | OpenCV 4.8 (`opencv-python`) + `opencv-contrib-python` 4.11 |
| Métricas | scikit-learn |
| Manipulación de datos | NumPy / Pandas |
| Visualización | Matplotlib / Seaborn |
| Entorno de entrenamiento | Docker + NVIDIA GPU (CUDA 11.8) |
| API de inferencia (prototipo) | FastAPI 0.115 + Uvicorn (`requirements-api.txt`) |
| Calidad / CI | GitHub Actions: Flake8, Black, isort + pytest |

```bash
pip install -r requirements.txt
pip install -r requirements-api.txt   # solo para la API
```

Nota sobre TF 2.10: EOL desde 2023 pero estable y compatible con la GPU
usada en el desarrollo (RTX 3050). Migración a TF 2.15+ planeada
post-defensa de TCC.

---

## Solución de Problemas

### GPU no detectada en Docker
```bash
nvidia-smi
docker compose --profile gpu run nembogueta-gpu python scripts/utils/verify_gpu.py
```

### Error con archivos .oni (UFPR)
OpenNI2 puede crashear después de ~200 archivos:
```bash
python scripts/utils/convert_oni_to_mp4.py \
  --input-dir /app/src/LIBRAS-HC-RGBDS-2011 \
  --skip-existing
```

### Cámara no detectada
```bash
ls /dev/video*
# Verificar que docker-compose.yml tenga devices: ["/dev/video0:/dev/video0", ...]
```

### Ventana cv2 no aparece
```bash
xhost +local:docker  # en el host antes del demo
echo $DISPLAY        # debe ser :0 o similar
```

---

## Hoja de Ruta

### Fase 1 — Infraestructura y modelos base ✅
- [x] Pipeline de feature engineering (120 features invariantes)
- [x] Modelo V-LIBRASIL v2 (99.4% test, 1361 clases) — pipeline legacy
- [x] Modelo UFPR handshapes (70.3% test, 61 clases) — pipeline legacy
- [x] Modelo alfabeto base v4 (99.78% holdout, 22 clases)
- [x] Predictor unificado multi-modelo
- [x] Entorno Docker con GPU

### Fase 2 — Robustez y evaluación en cámara real ✅
- [x] Protocolo de evaluación en cámara real
- [x] Diagnóstico de domain gap (gap 76 pp medido)
- [x] Modelo `robust` con augmentation + datos operador (gap 19 pp)
- [x] Fine-tuning + cross-validation 5-fold para operador específico
- [x] Análisis de catastrophic forgetting
- [x] Validación inter-sujeto del `robust` con sujeto no visto (80% top-1)

### Fase 3 — Datos y modelos LSP
- [x] Recopilación pública de letras LSPy (A–Z, Ñ, LL, RR): video "Alfabeto
      Manual" del canal Lengua de Señas LSPy, 7 videos del alfabeto en
      YouTube (Proyecto LSPY / Dra. Mirta Barreto, Información de Sordos,
      Tico Rodríguez, Ariel Torres, Milcar, CODA DOS LENGUAS) segmentados por
      alineamiento forzado, y 20 letras del signario SLS — 9 videos de 8 canales,
      una toma por letra; uso autorizado por los titulares, se publican solo
      landmarks y modelos (CC BY 4.0), los videos no se redistribuyen
- [ ] Colecta propia multi-sujeto de letras LSPy (protocolo de cámara real,
      con landmarks crudos para las letras con movimiento) — Sprint 4
- [x] Validación del modelo `robust` con segundo sujeto (S02 — `robust-v4`: 91.8% top-1)
- [x] Alicia v5 (`data/models/alicia-v5/`): estático multilengua con LSPy
      proxy + frames reales de 8 videos de 7 canales, y modelo de movimiento (LSC70,
      LSM dinámico, LSPy real). Signario SLS (1 intérprete externo): fusión
      16/20, acierta J y Z; faltan N/Ñ, U y X
- [x] Dataset LSPy-Landmarks v1 (alfabeto 243, achegety 33, léxico 1810;
      solo landmarks, CC BY 4.0) — `build_lspy_landmarks_release.py`
- [x] Vocabulario-objetivo (30 señas del Signario SLS) y herramienta de
      grabación del autor (`scripts/collect/collect_lspy_signs.py`)
- [ ] Colecta de palabras y frases LSP (tríplice fronteira)

### Fase 4 — Salida trilíngue
- [x] Alfabeto guaraní (achegety, 33 letras) en Alicia v6
      (`data/models/alicia-v6/`): las 13 letras propias del guaraní (nasales,
      g̃, dígrafos, puso) vienen de un único video leído frame a frame —
      prototipo sin validación externa
- [x] Diccionario LSP → {ES paraguayo, Guaraní, PT-BR}: 1810 glosas del
      Signario SLS (`data/analysis/diccionario/`); traducciones **generadas
      por IA, sin revisión humana** (es/pt/gn) — pendiente: validación por
      hablantes vía `revisiones.tsv`
- [x] Integración en `unified_predictor.py` con `output_lang`
- [x] Demo end-to-end multi-idioma: web local en `/` (cámara → letras →
      palabras → es / gn / pt, con captura PNG)

### Fase 5 — Migración del pipeline legacy
- [ ] Reentrenar V-LIBRASIL con feature extractor de 120 features
- [ ] Reentrenar UFPR con feature extractor de 120 features
- [ ] Deprecar pipeline de 208 features

### Fase 6 — Despliegue
- [x] API REST de inferencia (FastAPI) — **prototipo** en `src/api/`
      (Alicia v6, REST + WebSocket + demo web); sin despliegue ni autenticación
- [x] CI en GitHub Actions (lint + tests)
- [ ] Optimización para edge devices / TFLite
- [ ] Interface de usuario final
- [ ] Migración a TF 2.15+

---

## Contribución

1. Seguir las directrices de calidad (Black, isort, Flake8 — ver `.pre-commit-config.yaml`);
   el CI (`.github/workflows/ci.yml`) corre `flake8`, `black --check` e
   `isort --check-only` sobre `src` y `tests`, y luego `pytest`
2. Agregar pruebas unitarias en `tests/` para nueva funcionalidad
3. Actualizar este README al cambiar el estado de modelos o pipelines
4. Documentar protocolos académicos en `docs/source/protocols/`

---

## Agradecimientos

- Dataset **V-LIBRASIL** (UFPE) — base para desarrollo de frases
- Dataset **LIBRAS-HC-RGBDS** (UFPR) — base para handshapes
- Dataset **LibrAI** — base para alfabeto
- **MediaPipe** (Google) — extracción de landmarks de mano
- Comunidad sorda de la tríplice fronteira

---

*Nembogueta — fazer falar, conversar. Desenvolvido na tríplice fronteira.*

# Nembogueta — Análisis y Entrenamiento

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
nuevos datos de colecta.

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
       · augmentation: palm rotation ±20°, jitter 0.5σ, handedness flip
       · dropout 0.4
    └─ robust-ft (fine-tune sobre robust)
          + recupera V (única letra débil del robust): 0% → 98%
```

### Modelos de frases y handshapes (pipeline legacy, 208 features)

| Modelo | Dataset | Clases | Test Acc | Estado |
|---|---|---|---|---|
| V-LIBRASIL v2 | V-LIBRASIL (UFPE) | 1361 | 99.4% | ⚠️ Pipeline legacy — incompatible con `feature_engineering.py` actual |
| UFPR Handshapes | LIBRAS-HC-RGBDS | 61 | 70.3% | ⚠️ Pipeline legacy — incompatible con `feature_engineering.py` actual |
| LSP letters | Pendente | — | — | ❌ Aguardando colecta de datos |

⚠️ **Importante:** V-LIBRASIL v2 y UFPR Handshapes fueron entrenados con
un extractor de features anterior (208 features con posición absoluta +
motion). El pipeline actual (`src/preprocessing/feature_engineering.py`)
produce 120 features invariantes. Para usar estos modelos hay que
restaurar el extractor viejo o retrainear con el nuevo. Pendiente
decisión: ver `docs/source/protocols/`.

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
jitter gaussiano, y flip de handedness durante training.

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
├── src/                              # Librería principal (importable)
│   ├── config/config.py
│   ├── preprocessing/
│   │   ├── feature_engineering.py   # 120 features invariantes (núcleo)
│   │   ├── letter_preprocessor.py
│   │   ├── phrase_processor.py
│   │   └── word_processor.py
│   ├── training/
│   ├── inference/
│   │   └── unified_predictor.py
│   ├── data_collection/
│   └── utils/
│
├── scripts/                          # Entry points ejecutables
│   ├── preprocess/                   # Extracción de features
│   ├── train/                        # train_alphabet, train_alphabet_robust,
│   │                                 # finetune_alphabet_realworld,
│   │                                 # cv_finetune_alphabet, train_vlibrasil, train_ufpr...
│   ├── evaluate/                     # Evaluación offline + evaluate_realworld.py
│   ├── demo/                         # realtime_librai_alphabet, realtime_vlibrasil...
│   ├── inference/
│   ├── utils/                        # guided_calibration, verify_gpu, capture_letter_images...
│   └── legacy/                       # Scripts deprecados (ver scripts/legacy/README.md)
│
├── data/
│   ├── models/
│   │   ├── librai-alphabet-v4/      # base
│   │   ├── librai-alphabet-v4-ft/
│   │   ├── librai-alphabet-robust/  # ⭐ recomendado
│   │   ├── librai-alphabet-robust-ft/
│   │   ├── librai-alphabet-v4-ft-cv/  # 5-fold CV runs
│   │   ├── vlibrasil-v2/            # legacy (208 features)
│   │   └── ufpr-handshape/          # legacy (208 features)
│   ├── processed/
│   │   ├── librai_alphabet_v4/      # 22 letras × ~2090 frames (120 features)
│   │   ├── v-librasil-flat/         # legacy
│   │   └── ufpr-handshape/          # legacy
│   └── realworld_eval/
│       └── sessions/                # sesiones de evaluación en cámara real
│
├── docs/source/protocols/           # Protocolos académicos (TCC)
│   └── realworld_evaluation.md
├── tests/
├── Dockerfile                       # Python 3.10 + CUDA 11.8
├── docker-compose.yml               # Container GPU: nembogueta-dev-gpu
├── requirements.txt                 # TF 2.10, MediaPipe 0.10.8, OpenCV 4.8
├── pyproject.toml                   # target Python 3.10
└── LICENSE                          # MIT
```

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
  --output-dir /data/models/librai-alphabet-v4

# 3. Entrenar modelo robusto (augmentation + datos operador)
python scripts/train/train_alphabet_robust.py \
  --librai-dir /data/processed/librai_alphabet_v4 \
  --session-dir /data/realworld_eval/sessions/<SESSION_ID> \
  --output-root /data/models/librai-alphabet-robust

# 4. Evaluar offline (LibrAI holdout)
python scripts/evaluate/evaluate_alphabet.py \
  --model /data/models/librai-alphabet-robust/<run>/best_model.keras \
  --model-info /data/models/librai-alphabet-robust/<run>/model_info.json \
  --data-dir /data/processed/librai_alphabet_v4 \
  --output-dir /data/models/librai-alphabet-robust/<run>/evaluation

# 5. Evaluar en cámara real (sesión interactiva ~18 min)
python scripts/evaluate/evaluate_realworld.py \
  --model-dir /data/models/librai-alphabet-robust/<run> \
  --subject-id S02 --reps 5

# 6. Fine-tune sobre sesión real
python scripts/train/finetune_alphabet_realworld.py \
  --session-dir /data/realworld_eval/sessions/<SESSION_ID> \
  --base-model-dir /data/models/librai-alphabet-robust/<run>

# 7. Cross-validation 5-fold del fine-tune
python scripts/train/cv_finetune_alphabet.py \
  --session-dir /data/realworld_eval/sessions/<SESSION_ID> \
  --base-model-dir /data/models/librai-alphabet-robust/<run>
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
  --output-dir /data/models/vlibrasil-v2

# UFPR
python scripts/utils/convert_oni_to_mp4.py --input-dir /app/src/LIBRAS-HC-RGBDS-2011
python scripts/preprocess/preprocess_ufpr.py \
  --data-dir /app/src/LIBRAS-HC-RGBDS-2011 \
  --output-dir /data/processed/ufpr-handshape
python scripts/train/train_ufpr.py \
  --data-dir /data/processed/ufpr-handshape \
  --output-dir /data/models/ufpr-handshape
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

---

## Predictor unificado

```python
from src.inference.unified_predictor import LibrasUnifiedPredictor

predictor = LibrasUnifiedPredictor(models_dir="data/models")
predictions = predictor.predict_from_frame(frame, draw_landmarks=True)
```

---

## Dependencias

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.10 |
| Framework de DL | TensorFlow 2.10 / Keras |
| Extracción de landmarks | MediaPipe Hands 0.10.8 (Docker) / 0.10.35 (host Tasks API) |
| Procesamiento de video | OpenCV 4.8 |
| Métricas | scikit-learn |
| Manipulación de datos | NumPy / Pandas |
| Visualización | Matplotlib / Seaborn |
| Entorno de entrenamiento | Docker + NVIDIA GPU (CUDA 11.8) |
| API de inferencia (planeada) | FastAPI |

```bash
pip install -r requirements.txt
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
- [ ] Colecta de videos de letras LSP (A–Z, Ñ) — Sprint 4 del roadmap interno
- [ ] Validación del modelo `robust` con segundo sujeto
- [ ] Entrenamiento de modelo de letras LSP
- [ ] Colecta de palabras y frases LSP (tríplice fronteira)

### Fase 4 — Salida trilíngue
- [ ] Diccionario LSP → {ES paraguayo, Guaraní, PT-BR}
- [ ] Integración en `unified_predictor.py` con `output_lang`
- [ ] Demo end-to-end multi-idioma

### Fase 5 — Migración del pipeline legacy
- [ ] Reentrenar V-LIBRASIL con feature extractor de 120 features
- [ ] Reentrenar UFPR con feature extractor de 120 features
- [ ] Deprecar pipeline de 208 features

### Fase 6 — Despliegue
- [ ] API REST de inferencia (FastAPI)
- [ ] Optimización para edge devices / TFLite
- [ ] Interface de usuario final
- [ ] Migración a TF 2.15+

---

## Contribución

1. Seguir las directrices de calidad (Black, isort, Flake8 — ver `.pre-commit-config.yaml`)
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

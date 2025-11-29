# Ñemongeta - Python Module

**Sistema de Reconocimiento de Lenguaje de Señas Paraguayo (LSPy)**
**Módulo de Análisis y Entrenamiento**

## Descripción

El módulo `Ñemongeta - Python` contiene scripts para la recolección, preprocesamiento, análisis, entrenamiento y conversión de modelos CNN-LSTM para el reconocimiento de gestos en **Lenguaje de Señas Paraguayo (LSPy)**. Los modelos están optimizados para alcanzar una precisión del 95% en las categorías de letras (a-z, ñ), palabras (ej. juicio, abogado) y frases (ej. acceso a la justicia).

### Enfoque Principal del Proyecto

Este proyecto está enfocado en el desarrollo de reconocimiento de lenguaje de señas para **Paraguay**, con soporte bilingüe para:
- **Español paraguayo**
- **Guaraní**

El sistema utiliza técnicas de deep learning con arquitecturas CNN-LSTM para el reconocimiento en tiempo real de:

- **Alfabeto dactilológico** (A-Z, Ñ): Reconocimiento de letras individuales
- **Handshapes (Formas de mano)**: Clasificación de configuraciones manuales por orientación
- **Palabras completas**: Reconocimiento de señas completas en LSPy
- **Traducción bilingüe**: Conversión de texto Español/Guaraní a glosas LSPy
- **Expresiones faciales**: Análisis de componentes no manuales

### Trabajo con LIBRAS

Como parte del desarrollo y entrenamiento del sistema, se utiliza el dataset **V-LIBRASIL** (Lenguaje de Señas Brasileño) para:
- Desarrollo y prueba de arquitecturas de modelos
- Entrenamiento de modelos base que serán adaptados a LSPy
- Validación de técnicas de preprocesamiento y extracción de features
- Transfer learning para acelerar el entrenamiento de modelos LSPy

El sistema utiliza **MediaPipe** para extracción de landmarks y modelos **CNN-LSTM** optimizados para alcanzar alta precisión en tiempo real.

## 🎯 Características Principales

### Sistema Unificado de Reconocimiento en Tiempo Real

**Estado Actual (usando dataset LIBRAS para desarrollo):**

- ✅ **Reconocimiento de Alfabeto**: 26 letras (A-Z) con 45.6% de precisión
- ✅ **Handshapes por Orientación**: 4 modelos especializados (back, front, left, right) con 100 clases cada uno
- ✅ **Detección de Ambas Manos**: Soporte simultáneo para mano izquierda y derecha
- ✅ **Traducción multilingüe**: Modelo transformer para conversión texto-glosas
  - Actualmente: PT-BR → LIBRAS (modelo base)
  - Objetivo: Español/Guaraní → LSPy
- ✅ **UI Optimizada**: Interfaz mejorada con mejor contraste y visualización clara
- ✅ **Barras de Confianza**: Visualización en tiempo real de la confianza de predicciones

**Próximos Pasos para LSPy:**
- 🔄 Recolección de dataset LSPy (letras, palabras, frases)
- 🔄 Entrenamiento de modelos específicos para LSPy
- 🔄 Implementación de traducción Español → LSPy
- 🔄 Implementación de traducción Guaraní → LSPy
- 🔄 Letra Ñ para alfabeto paraguayo

### Demo en Tiempo Real

```bash
# Ejecutar sistema completo con cámara (actualmente con modelos LIBRAS)
python scripts/demo_realtime_improved.py

# Controles:
# Q - Salir
# T - Traducir texto (PT-BR → LIBRAS, futuro: ES/GN → LSPy)
# L - Activar/desactivar visualización de landmarks
```

## Dependencias

Este proyecto utiliza las siguientes dependencias principales:
- Python 3.8 o superior
- TensorFlow
- MediaPipe
- NumPy
- OpenCV

Para instalar todas las dependencias, ejecute:
```bash
pip install -r requirements.txt
```

## Estructura del Proyecto

La estructura principal del proyecto es la siguiente:

```
Nembogueta---An-lisis-y-Entrenamiento/
├── data/                # Datos crudos y procesados
├── docs/                # Documentación del proyecto
├── models/              # Modelos entrenados
├── notebooks/           # Jupyter notebooks para experimentación
├── scripts/             # Scripts principales para procesamiento y demos
├── src/                 # Código fuente principal
├── tests/               # Pruebas unitarias
└── README.md            # Documentación principal
```

## 🚀 Inicio Rápido

### 1. Instalación

```bash
# Clonar el repositorio
git clone <repository-url>
cd Nembogueta---An-lisis-y-Entrenamiento

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Ejecutar Demo en Tiempo Real

```bash
# Demo mejorado con todas las funcionalidades
python scripts/demo_realtime_improved.py

# Opciones disponibles:
python scripts/demo_realtime_improved.py --help

# Especificar cámara y resolución
python scripts/demo_realtime_improved.py --camera 0 --width 1280 --height 720
```

### 3. Entrenar Modelos

```bash
# Entrenar modelo de alfabeto (A-Z)
python scripts/train_alphabet.py \
  --data-dir data/processed/alphabet-combined \
  --output-dir data/models/alphabet \
  --epochs 50 --batch-size 64

# Entrenar modelo de handshapes
python scripts/train_handshape.py \
  --data-dir data/processed/lswh100 \
  --output-dir data/models/handshape \
  --epochs 100 --batch-size 32

# Entrenar modelo de traducción
python scripts/train_translation.py \
  --data-dir data/processed/pt_br2libras \
  --output-dir data/models/translation \
  --epochs 30 --batch-size 32

# Entrenar modelo de V-LIBRASIL
python scripts/train_vlibrasil.py \
  --data-dir data/processed/v-librasil-flat \
  --output-dir data/models/vlibrasil \
  --epochs 100 --batch-size 32
```

## 📊 Modelos y Rendimiento

### Alfabeto (Dactilología)

- **Arquitectura**: CNN-LSTM
- **Clases**: 26 letras (A-Z)
- **Muestras**: 2,748
- **Precisión**: 45.6% (validación)
- **Features**: 63 (21 landmarks × 3 coordenadas)

### Handshapes

- **Arquitectura**: Dense Neural Network
- **Modelos**: 4 (por orientación: back, front, left, right)
- **Clases por modelo**: 100
- **Precisión**: ~74% (por orientación)
- **Features**: 63 (21 landmarks × 3 coordenadas)

### Traducción PT-BR → LIBRAS

- **Arquitectura**: Transformer (Encoder-Decoder)
- **Vocabulario PT-BR**: Variable
- **Vocabulario LIBRAS**: Glosas
- **Precisión**: >99.9% (validación)
- **Max sequence length**: 100 tokens

### V-LIBRASIL

- **Dataset**: Videos de LIBRAS
- **Arquitectura**: LSTM
- **Estado**: Modelo base entrenado

## 🛠️ Scripts Disponibles

### Preprocesamiento

```bash
# Alfabeto
python scripts/preprocess_alphabet.py \
  --data-dir data/raw/alphabet \
  --output-dir data/processed/alphabet

# Handshapes
python scripts/preprocess_lswh100.py \
  --data-dir data/raw/lswh100 \
  --output-dir data/processed/lswh100

# V-LIBRASIL
python scripts/preprocess_vlibrasil.py \
  --data-dir "data/raw/videos UFPE (V-LIBRASIL)/data" \
  --output-dir data/processed/v-librasil-flat
```

### Evaluación

```bash
# Evaluar modelo de alfabeto
python scripts/evaluate_alphabet.py \
  --model-path data/models/alphabet/best_model.keras \
  --test-data data/processed/alphabet-combined

# Evaluar handshapes
python scripts/evaluate_handshape.py \
  --model-dir data/models/handshape \
  --test-data data/processed/lswh100

# Evaluar traducción
python scripts/evaluate_translation.py \
  --model-path data/models/translation/best_model.keras
```

### Inferencia

```bash
# Inferencia en video individual
python scripts/inference_alphabet.py \
  --model-path data/models/alphabet/best_model.keras \
  --video-path path/to/video.mp4

# Tiempo real con cámara
python scripts/realtime_alphabet.py \
  --model-path data/models/alphabet/best_model.keras \
  --camera 0
```

## 🎨 Sistema Unificado de Predicción

### Clase `LibrasUnifiedPredictor`

Predictor centralizado que carga y gestiona todos los modelos:

```python
from src.libras_unified_predictor import LibrasUnifiedPredictor

# Inicializar predictor
predictor = LibrasUnifiedPredictor(models_dir="data/models")

# Obtener predicciones desde un frame
predictions = predictor.predict_from_frame(frame, draw_landmarks=True)

# Resultados incluyen:
# - hands: Lista de predicciones por cada mano detectada
#   - handedness: "Left" o "Right"
#   - orientation: "back", "front", "left", "right"
#   - alphabet: Letra predicha con confianza
#   - handshape: Forma de mano predicha con confianza
# - facial_expression: Expresión facial (si disponible)
# - landmarks_detected: Estado de detección

# Traducir texto PT-BR a glosas LIBRAS
glosas = predictor.translate_text_to_gloss("olá mundo")
# Resultado: ['OLA', 'MUNDO']
```

### Características del Predictor

- **Detección automática de orientación**: Clasifica la orientación de la mano
- **Múltiples manos**: Soporta detección de mano izquierda y derecha simultáneamente
- **Modelos especializados**: Usa el modelo de handshape apropiado según orientación
- **MediaPipe integrado**: Extracción automática de landmarks
- **Visualización opcional**: Dibuja landmarks sobre el frame

## 📸 Demo en Tiempo Real - Características

### UI Mejorada

- **Fondos semi-transparentes**: Mejor legibilidad sin ocultar el video
- **Paneles por mano**: Información separada para cada mano detectada
- **Colores distintivos**: Naranja (mano derecha), Azul (mano izquierda)
- **Barras de confianza**: Visualización gráfica de certeza de predicciones
- **Controles claros**: Instrucciones siempre visibles

### Información Mostrada

Para cada mano detectada:
- Tipo de mano (Izquierda/Derecha)
- Orientación (back/front/left/right)
- Letra del alfabeto con barra de confianza
- Handshape con barra de confianza

Adicional:
- Expresión facial (si disponible)
- Traducción PT-BR → LIBRAS (al presionar T)
- FPS y rendimiento

## 🐳 Ejecución en Docker

### Iniciar Contenedor

```bash
# Iniciar con GPU
docker compose --profile gpu up -d nembogueta-gpu

# Verificar estado
docker ps
```

### Ejecutar Scripts en Contenedor

```bash
# Entrenar modelo de alfabeto
docker exec nembogueta-dev-gpu python scripts/train_alphabet.py \
  --data-dir /app/data/processed/alphabet-combined \
  --output-dir /app/data/models/alphabet \
  --epochs 50

# Demo en tiempo real (requiere X11 forwarding)
docker exec nembogueta-dev-gpu python scripts/demo_realtime_improved.py
```

## 🔧 Solución de Problemas

### Error: "No module named sklearn"

```bash
pip install scikit-learn
```

### Error: "No se detecta la cámara"

```bash
# Verificar cámaras disponibles
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"

# Probar con ID diferente
python scripts/demo_realtime_improved.py --camera 1
```

### Predicciones con baja confianza

- Asegúrate de tener buena iluminación
- Mantén las manos visibles y dentro del frame
- Evita fondos complejos o con movimiento
- Ajusta la posición para que MediaPipe detecte correctamente

### Rendimiento lento

- Usa `--width 640 --height 480` para menor resolución
- Desactiva landmarks con `L` durante la ejecución
- Considera usar GPU si está disponible

## 📈 Hoja de Ruta - LSPy (Lenguaje de Señas Paraguayo)

### Fase 1: Infraestructura y Modelos Base (Actual)
- [x] Sistema de preprocesamiento universal
- [x] Arquitectura CNN-LSTM para reconocimiento
- [x] Predictor unificado multi-modelo
- [x] UI en tiempo real con detección de múltiples manos
- [x] Modelos base entrenados con LIBRAS

### Fase 2: Recolección de Datos LSPy
- [ ] **Alfabeto LSPy** (A-Z, Ñ)
  - [ ] Recolección de videos para 27 letras
  - [ ] 10 videos por letra mínimo
  - [ ] Múltiples personas para diversidad
- [ ] **Palabras legales** (jurídicas)
  - [ ] Juicio, abogado, fiscal, defensor, etc.
  - [ ] Términos específicos del sistema judicial paraguayo
- [ ] **Frases completas**
  - [ ] "Acceso a la justicia"
  - [ ] Frases comunes en contexto legal
  - [ ] Frases en español y guaraní

### Fase 3: Entrenamiento LSPy
- [ ] Transfer learning desde modelos LIBRAS a LSPy
- [ ] Entrenamiento de alfabeto LSPy (incluyendo Ñ)
- [ ] Entrenamiento de palabras jurídicas
- [ ] Entrenamiento de frases completas
- [ ] Fine-tuning para español y guaraní

### Fase 4: Traducción Bilingüe
- [ ] **Español → LSPy**
  - [ ] Dataset de traducción Español-Glosas LSPy
  - [ ] Modelo transformer Español → LSPy
- [ ] **Guaraní → LSPy**
  - [ ] Dataset de traducción Guaraní-Glosas LSPy
  - [ ] Modelo transformer Guaraní → LSPy
- [ ] Sistema unificado bilingüe

### Fase 5: Optimización y Despliegue
- [ ] Optimización de modelos para edge devices
- [ ] Conversión a TensorFlow Lite
- [ ] API REST para integración
- [ ] App móvil Android/iOS
- [ ] Integración con sistema judicial paraguayo

### Fase 6: Expansión
- [ ] Entrenamiento de modelo de expresiones faciales
- [ ] Reconocimiento de contexto y gramática LSPy
- [ ] Soporte para más dominios (educación, salud, etc.)
- [ ] Sistema de retroalimentación y mejora continua

## 🎯 Objetivos del Proyecto

Este proyecto busca:

1. **Democratizar el acceso a la justicia** en Paraguay mediante tecnología de reconocimiento de señas
2. **Preservar y promover** el Lenguaje de Señas Paraguayo (LSPy)
3. **Facilitar la comunicación** entre personas sordas y el sistema judicial
4. **Apoyar el bilingüismo** paraguayo (Español y Guaraní) en el contexto de LSPy
5. **Crear herramientas de código abierto** para la comunidad sorda paraguaya

## 🤝 Contribución

1. Sigue las directrices de calidad de código (Black, isort, Flake8)
2. Agrega pruebas unitarias para nueva funcionalidad
3. Actualiza la documentación según sea necesario
4. Envía pull requests para revisión

### Cómo Contribuir con Datos LSPy

Si eres hablante de LSPy y quieres contribuir:
- Contacta al equipo para participar en recolección de videos
- Ayuda a validar las señas reconocidas
- Proporciona feedback sobre la precisión del sistema

## 📝 Licencia

[Especificar licencia]

## 📧 Contacto

[Especificar información de contacto]

## 🙏 Agradecimientos

- Comunidad sorda paraguaya
- Dataset V-LIBRASIL por proporcionar data base para desarrollo
- Proyecto MediaPipe de Google por la tecnología de landmarks
- Comunidad de código abierto

---

**Desarrollado con ❤️ para la comunidad sorda paraguaya**
**Ñemongeta - Hablemos en señas**

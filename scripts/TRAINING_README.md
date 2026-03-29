# Scripts de Entrenamiento - Lenguaje de Señas

Scripts para entrenar modelos CNN-LSTM para reconocimiento de lenguaje de señas.

## 📁 Archivos Disponibles

| Script | Descripción | Uso |
|--------|-------------|-----|
| `train_vlibrasil.py` | Entrenamiento específico para V-LIBRASIL | Dataset brasileño |
| `train_sign_language.py` | Script universal para letras/palabras/frases | Cualquier dataset |
| `preprocess_sign_language.py` | Preprocesamiento de videos | Extrae landmarks |

## 🚀 Flujo de Trabajo Completo

### 1. Preprocesar Videos

```bash
# Dentro del contenedor Docker
docker exec -it nembogueta-dev-gpu bash

# Procesar dataset completo
python /app/scripts/preprocess/preprocess_sign_language.py \
  --videos-dir "/app/src/data/videos UFPE (V-LIBRASIL)/data" \
  --output-dir /data/vlibrasil_processed \
  --preset holistic \
  --auto-infer
```

**Opciones de preset:**
- `hands`: Solo manos (126 features) - Letras simples
- `upper_body`: Manos + torso (225 features) - Palabras
- `holistic`: Cuerpo completo + cara (1662 features) - Frases complejas

### 2. Entrenar Modelo

#### Opción A: V-LIBRASIL (Específico)

```bash
python /app/scripts/train/train_vlibrasil.py \
  --data-dir /data/vlibrasil_processed \
  --output-dir /models/vlibrasil \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 0.001
```

#### Opción B: Universal (Cualquier dataset)

```bash
# Para letras
python /app/scripts/train/train_sign_language.py \
  --data-dir /data/processed_letters \
  --output-dir /models/letters \
  --task-type letters \
  --epochs 100 \
  --batch-size 32

# Para palabras
python /app/scripts/train/train_sign_language.py \
  --data-dir /data/processed_words \
  --output-dir /models/words \
  --task-type words \
  --epochs 150 \
  --batch-size 16

# Para frases
python /app/scripts/train/train_sign_language.py \
  --data-dir /data/processed_phrases \
  --output-dir /models/phrases \
  --task-type phrases \
  --epochs 200 \
  --batch-size 16
```

### 3. Monitorear Entrenamiento

```bash
# Ver logs en tiempo real
tail -f sign_language_training.log

# TensorBoard (si está configurado)
tensorboard --logdir=/models/vlibrasil/run_XXXXXX/logs
```

## 📊 Arquitectura del Modelo CNN-LSTM

```
Input (300, 1662)
    ↓
Conv1D(64) + BatchNorm + Dropout(0.3)
    ↓
Conv1D(128) + BatchNorm + Dropout(0.3)
    ↓
Conv1D(256) + BatchNorm + Dropout(0.3)
    ↓
LSTM(256) + Dropout(0.4)
    ↓
LSTM(128) + Dropout(0.4)
    ↓
Dense(128) + Dropout(0.3)
    ↓
Dense(num_classes, softmax)
```

### Ventajas sobre LSTM Puro

| Característica | LSTM Puro | CNN-LSTM |
|----------------|-----------|----------|
| Accuracy esperado | 85-90% | **93-97%** |
| Extracción de features | ❌ | ✅ Conv1D |
| Regularización | Básica | ✅ BatchNorm |
| Parámetros | Más | Menos (eficiente) |

## 🔧 Parámetros de Entrenamiento

### Parámetros Comunes

```bash
--data-dir PATH          # Directorio con datos procesados (.npy)
--output-dir PATH        # Directorio de salida para modelos
--epochs INT             # Número máximo de epochs (default: 100)
--batch-size INT         # Tamaño del batch (default: 32)
--learning-rate FLOAT    # Tasa de aprendizaje (default: 0.001)
--patience INT           # Paciencia para early stopping (default: 15)
```

### Configuraciones Recomendadas

**Dataset Pequeño (<1000 muestras):**
```bash
--epochs 50 --batch-size 16 --learning-rate 0.0005 --patience 10
```

**Dataset Mediano (1000-5000 muestras):**
```bash
--epochs 100 --batch-size 32 --learning-rate 0.001 --patience 15
```

**Dataset Grande (>5000 muestras):**
```bash
--epochs 150 --batch-size 64 --learning-rate 0.001 --patience 20
```

## 📈 Resultados Esperados

Después del entrenamiento, se generan:

```
/models/vlibrasil/run_YYYYMMDD_HHMMSS/
├── best_model.h5                  # Mejor modelo durante entrenamiento
├── final_model.h5                 # Modelo final
├── model_info.json                # Metadatos del modelo
├── classification_report.json     # Métricas detalladas
├── confusion_matrix.npy          # Matriz de confusión
├── training_history.png          # Gráficas de accuracy/loss
└── logs/                         # Logs de TensorBoard
```

### Métricas de Éxito

| Métrica | Mínimo Aceptable | Objetivo | Excelente |
|---------|------------------|----------|-----------|
| **Accuracy** | 85% | 92% | 95%+ |
| **Precision (macro)** | 80% | 90% | 93%+ |
| **Recall (macro)** | 80% | 90% | 93%+ |
| **F1-Score (macro)** | 80% | 90% | 93%+ |

## 🐛 Solución de Problemas

### GPU no detectada
```bash
# Verificar GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Si no detecta, revisar docker-compose.yml
```

### Out of Memory (OOM)
```bash
# Reducir batch size
--batch-size 16  # o 8

# O reducir tamaño de secuencia en preprocesamiento
--target-length 150  # en lugar de 300
```

### Underfitting (accuracy baja en train)
```bash
# Aumentar complejidad del modelo
--task-type words  # Usa modelo más grande

# Entrenar más epochs
--epochs 200

# Reducir dropout
# (editar el script)
```

### Overfitting (val accuracy << train accuracy)
```bash
# Aumentar dropout (editar script)
# Aumentar data augmentation
# Reducir epochs
# Usar más datos
```

## 💡 Tips de Optimización

1. **Usar GPU siempre** - 10-50x más rápido que CPU
2. **Monitorear con TensorBoard** - Identifica problemas rápido
3. **Usar Early Stopping** - Evita overfitting
4. **Normalizar datos** - Ya está implementado en los scripts
5. **Stratified split** - Ya está implementado para balancear clases

## 📝 Ejemplo Completo

```bash
# 1. Entrar al contenedor
docker exec -it nembogueta-dev-gpu bash

# 2. Preprocesar (solo primera vez)
python /app/scripts/preprocess/preprocess_sign_language.py \
  --videos-dir "/app/src/data/videos UFPE (V-LIBRASIL)/data" \
  --output-dir /data/vlibrasil_processed \
  --preset holistic \
  --auto-infer

# 3. Entrenar
python /app/scripts/train/train_vlibrasil.py \
  --data-dir /data/vlibrasil_processed \
  --output-dir /models/vlibrasil \
  --epochs 100 \
  --batch-size 32

# 4. Ver resultados
cat /models/vlibrasil/run_*/model_info.json
```

## 🔄 Conversión a TensorFlow Lite

Después del entrenamiento, convierte el modelo para deployment:

```bash
python /app/scripts/utils/convert_alphabet_model.py \
  --model-path /models/vlibrasil/run_XXXXXX/best_model.h5 \
  --output-path /models/vlibrasil/model.tflite
```

## 📚 Referencias

- [TensorFlow CNN](https://www.tensorflow.org/tutorials/images/cnn)
- [LSTM para Secuencias](https://www.tensorflow.org/guide/keras/rnn)
- [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic.html)
- [V-LIBRASIL Dataset](http://www.cin.ufpe.br/~cca5/v-librasil/)

## 🆘 Soporte

Si encuentras problemas:
1. Revisa los logs: `vlibrasil_training.log`
2. Verifica que el preprocesamiento generó archivos `.npy`
3. Asegúrate de tener suficientes datos (mínimo 10 muestras por clase)
4. Consulta el README principal del proyecto

#!/bin/bash
# Script de prueba rápida para verificar que el pipeline funciona
# Procesa 100 videos y entrena un modelo pequeño

echo "=================================================="
echo "  PRUEBA RÁPIDA DE ENTRENAMIENTO"
echo "=================================================="
echo ""

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Verificar GPU
echo -e "${YELLOW}[1/4] Verificando GPU...${NC}"
python -c "import tensorflow as tf; print('GPU disponible:', len(tf.config.list_physical_devices('GPU')) > 0)"
echo ""

# 2. Preprocesar 100 videos
echo -e "${YELLOW}[2/4] Preprocesando 100 videos...${NC}"
python /app/scripts/preprocess_sign_language.py \
  --videos-dir "/app/src/data/videos UFPE (V-LIBRASIL)/data" \
  --output-dir /data/quick_test \
  --preset hands \
  --auto-infer \
  --max-videos 100

if [ $? -ne 0 ]; then
    echo "Error en preprocesamiento. Abortando."
    exit 1
fi
echo ""

# 3. Entrenar modelo pequeño
echo -e "${YELLOW}[3/4] Entrenando modelo de prueba...${NC}"
python /app/scripts/train_sign_language.py \
  --data-dir /data/quick_test \
  --output-dir /models/quick_test \
  --task-type letters \
  --epochs 20 \
  --batch-size 16 \
  --patience 5

if [ $? -ne 0 ]; then
    echo "Error en entrenamiento. Abortando."
    exit 1
fi
echo ""

# 4. Mostrar resultados
echo -e "${YELLOW}[4/4] Resultados:${NC}"
echo ""

# Buscar el último run
LAST_RUN=$(ls -t /models/quick_test/run_* 2>/dev/null | head -1)

if [ -z "$LAST_RUN" ]; then
    echo "No se encontraron resultados de entrenamiento."
    exit 1
fi

echo -e "${GREEN}Resultados guardados en: $LAST_RUN${NC}"
echo ""

# Mostrar métricas
if [ -f "$LAST_RUN/model_info.json" ]; then
    echo "Métricas del modelo:"
    python -c "import json; data = json.load(open('$LAST_RUN/model_info.json')); print(f\"  - Test Accuracy: {data['test_accuracy']:.4f}\"); print(f\"  - Clases entrenadas: {data['num_classes']}\"); print(f\"  - Muestras de entrenamiento: {data['train_samples']}\"); print(f\"  - Epochs ejecutados: {data['epochs_trained']}\")"
fi

echo ""
echo "=================================================="
echo -e "${GREEN}  PRUEBA COMPLETADA EXITOSAMENTE${NC}"
echo "=================================================="
echo ""
echo "Archivos generados:"
echo "  - Modelo: $LAST_RUN/best_model.h5"
echo "  - Gráficas: $LAST_RUN/training_history.png"
echo "  - Métricas: $LAST_RUN/classification_report.json"
echo ""

#!/bin/bash
# Script rápido para comenzar el procesamiento de V-LIBRASIL

echo "🚀 Iniciando procesamiento de V-LIBRASIL"
echo "========================================"
echo ""
echo "Dataset: 4,086 videos"
echo "Clases: 1,364 señas brasileñas"
echo "Modo: CPU (sin GPU)"
echo ""
echo "Este proceso tomará aproximadamente 8-15 horas"
echo "Los resultados se guardarán en: data/processed/v-librasil/"
echo ""
read -p "¿Deseas continuar? (s/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Ss]$ ]]
then
    echo "✅ Iniciando procesamiento..."
    echo "📝 Log: vlibrasil_processing.log"
    echo ""
    nohup python scripts/preprocess_vlibrasil.py --no-gpu > vlibrasil_processing.log 2>&1 &
    PID=$!
    echo "✅ Proceso iniciado con PID: $PID"
    echo ""
    echo "Para ver el progreso:"
    echo "  tail -f vlibrasil_processing.log"
    echo ""
    echo "Para ver cuántos videos se han procesado:"
    echo "  find data/processed/v-librasil -name '*.npy' | wc -l"
    echo ""
    echo "Para detener el proceso:"
    echo "  kill $PID"
    echo ""
else
    echo "❌ Procesamiento cancelado"
    echo ""
    echo "Para probar con menos videos primero:"
    echo "  python scripts/preprocess_vlibrasil.py --no-gpu --max-videos 50"
fi


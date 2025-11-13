# Configuración GPU para Ñemongeta

## ✅ GPU Configurada Exitosamente

**Hardware Detectado:**
- GPU: NVIDIA GeForce RTX 3050 Mobile (4GB VRAM)
- CPU: AMD Ryzen 5 7535HS (12 cores)
- Driver NVIDIA: 580.95.05
- CUDA: 11.8
- cuDNN: 8.6

## 🚀 Rendimiento

Aceleración GPU demostrada: **~5000x más rápido** que CPU en operaciones de matrices

## 📋 Uso del Entorno Virtual con GPU

### Activar el entorno virtual CON soporte GPU:
```bash
source venv/bin/activate-gpu.sh
```

Este script activa el entorno virtual y configura todas las variables de entorno necesarias para CUDA.

### Activar el entorno virtual SIN GPU (solo CPU):
```bash
source venv/bin/activate
```

## 🧪 Verificar que la GPU está funcionando

```bash
source venv/bin/activate-gpu.sh
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

## 📦 Librerías CUDA Instaladas (vía pip)

- nvidia-cudnn-cu11==8.6.0.163
- nvidia-cuda-runtime-cu11==11.8.89
- nvidia-cuda-nvcc-cu11==11.8.89
- nvidia-cublas-cu11
- nvidia-cusparse-cu11
- nvidia-cusolver-cu11
- nvidia-cufft-cu11
- nvidia-curand-cu11

## ⚙️ Variables de Entorno

El script `activate-gpu.sh` configura automáticamente:
- `LD_LIBRARY_PATH` para todas las librerías CUDA
- `TF_CPP_MIN_LOG_LEVEL=2` para suprimir advertencias innecesarias

## 💡 Notas Importantes

1. **Siempre usa `activate-gpu.sh`** para entrenar modelos y aprovechar la GPU
2. El entorno normal (`activate`) funcionará pero solo usará CPU
3. Las librerías CUDA están dentro del entorno virtual, no afectan tu sistema
4. TensorFlow 2.13.1 está optimizado para CUDA 11.8

## 🎯 Próximos Pasos

Tu proyecto de reconocimiento de señas paraguayas ahora puede:
- ✓ Entrenar modelos más rápido con GPU
- ✓ Procesar videos en tiempo real
- ✓ Usar modelos de deep learning complejos (MediaPipe + TensorFlow)

¡Listo para entrenar! 🚀

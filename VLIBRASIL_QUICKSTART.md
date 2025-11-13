# 🎯 Guía Rápida: Procesamiento de V-LIBRASIL con GPU

## 📊 Información del Dataset

- **Total de videos**: 4,086
- **Clases (señas)**: 1,364 únicas
- **Articuladores**: 3 (cada uno graba ~1,360 señas)
- **Videos por clase**: 2-3 videos
- **Resolución**: ~1920x1080 promedio
- **Ubicación**: `data/raw/v-librasil/`

## 🚀 Inicio Rápido (Usar CPU Ahora)

```bash
# 1. Explorar el dataset (ya ejecutado ✓)
python scripts/explore_vlibrasil.py

# 2. Procesar 10 videos de prueba (2-3 minutos)
python scripts/preprocess_vlibrasil.py --no-gpu --max-videos 10

# 3. Si funciona bien, procesar 100 videos (~30-40 minutos)
python scripts/preprocess_vlibrasil.py --no-gpu --max-videos 100

# 4. Procesar todos los 4,086 videos (~8-15 horas, dejar corriendo)
python scripts/preprocess_vlibrasil.py --no-gpu
```

## 🎮 Estado de GPU

Tu sistema:
- ✅ GPU: NVIDIA GeForce RTX 3050 (4GB VRAM)
- ✅ CUDA: 13.0
- ✅ Drivers: 580.95.05
- ⚠️ TensorFlow: No detecta GPU (faltan librerías CUDA runtime)

**Solución**: Ver `GPU_SETUP.md` para configurar GPU (opcional, no necesario ahora)

## 📝 Comandos Útiles

```bash
# Ver estadísticas sin procesar
python scripts/preprocess_vlibrasil.py --stats

# Verificar configuración GPU
python scripts/check_gpu.py

# Reprocesar videos existentes
python scripts/preprocess_vlibrasil.py --no-gpu --no-skip

# Procesar con longitud de secuencia diferente
python scripts/preprocess_vlibrasil.py --no-gpu --target-length 200
```

## 📂 Estructura de Salida

```
data/processed/v-librasil/
├── À noite toda/
│   ├── 20210411080131_6072d70b74896.npy  # Secuencia de landmarks
│   ├── 20210929042018_6154bc720abf7.npy
│   └── 20210126072453_601096b5ed907.npy
├── Abacaxi/
│   ├── 20210127091036_6011583c87073.npy
│   └── ...
└── ... (1,364 carpetas más)
```

Cada archivo `.npy` contiene:
- Shape: `(300, 126)` - 300 frames, 126 coordenadas (2 manos × 21 puntos × 3 coords)
- Tipo: `float32`
- Tamaño: ~150 KB por video

## ⏱️ Tiempos Estimados

| Videos | CPU (sin GPU) | GPU (configurada) |
|--------|---------------|-------------------|
| 10     | 2-3 min       | 15-30 seg        |
| 100    | 30-40 min     | 3-5 min          |
| 1,000  | 5-7 horas     | 30-45 min        |
| 4,086  | 8-15 horas    | 1.5-2.5 horas    |

## 🔄 Próximos Pasos Después del Procesamiento

1. **Verificar datos procesados**:
   ```bash
   ls -lh data/processed/v-librasil/ | head -20
   find data/processed/v-librasil/ -name "*.npy" | wc -l
   ```

2. **Analizar secuencias**:
   ```bash
   python scripts/analyze_sequences.py
   ```

3. **Entrenar modelo con V-LIBRASIL**:
   - Adaptar `src/training/letter_model_trainer.py` para usar V-LIBRASIL
   - Combinar con dataset LSPy para transfer learning

## 🐛 Solución de Problemas

**Error: "No se pudo abrir video"**
- Verifica que los videos estén en `data/raw/v-librasil/videos/`
- Algunos videos pueden estar corruptos (3 reportados en error.csv)

**Error: "No se detectaron manos"**
- Normal en algunos videos (~5-10%)
- El script los marca como fallidos y continúa

**Procesamiento muy lento**
- Usa `--max-videos` para procesar en lotes
- Considera configurar GPU (ver GPU_SETUP.md)
- Cierra otros programas que usen recursos

## 📌 Notas Importantes

1. ✅ El procesamiento solo se hace **una vez**
2. ✅ Los videos ya procesados se **omiten automáticamente**
3. ✅ Puedes **interrumpir** (Ctrl+C) y continuar después
4. ✅ Los archivos `.npy` están **listos para entrenamiento**

## 🎯 Recomendación

**Para empezar ahora mismo**:
```bash
# Procesar primeros 50 videos (prueba de ~15-20 min)
python scripts/preprocess_vlibrasil.py --no-gpu --max-videos 50

# Si todo va bien, procesar el resto
python scripts/preprocess_vlibrasil.py --no-gpu
```

**Para máximo rendimiento** (después):
1. Configurar Docker con GPU (ver GPU_SETUP.md)
2. O instalar CUDA Toolkit nativo
3. Reducirá tiempo de 15h → 2h

---

**¿Dudas?** Revisa:
- `GPU_SETUP.md` - Configuración de GPU
- `README.md` - Documentación completa del proyecto
- `scripts/preprocess_vlibrasil.py --help` - Opciones del script


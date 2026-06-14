# Protocolo de Evaluación en Cámara Real — Modelo de Alfabeto

## 1. Objetivo

Medir el desempeño del modelo `librai-alphabet-v4` en condiciones de
inferencia real (webcam, signantes distintos a los del dataset de
entrenamiento), y cuantificar el **gap de dominio** entre la métrica
reportada sobre el holdout del dataset LibrAI (test accuracy = 99.78 %)
y la métrica empírica sobre sesiones nuevas.

## 2. Hipótesis

- **H0** (nula): el accuracy en cámara real es igual al del holdout
  (diferencia ≤ 2 pp), es decir, el rediseño a 120 features
  invariantes eliminó el domain gap.
- **H1**: el accuracy en cámara real es significativamente menor que
  el del holdout (diferencia > 2 pp), lo que indica que el gap persiste
  por causas no cubiertas por las features actuales (iluminación,
  ruido de MediaPipe, variabilidad inter-sujeto, etc.).

El umbral de 2 pp es arbitrario; se reporta también la diferencia
absoluta y el intervalo de confianza.

## 3. Sujetos

### Criterios de inclusión
- Mayor de edad.
- Mano dominante visible (no se exige bilateralidad).
- Consentimiento informado para grabación de landmarks (no se
  guarda video, solo coordenadas x/y/z normalizadas).

### Tamaño mínimo
- **Piloto (TCC II Sprint 3):** 1 sujeto = el desarrollador.
  Suficiente para detectar regresiones gruesas; no generaliza.
- **Validación (Sprint 5+):** ≥ 3 sujetos con perfiles distintos
  (con/sin experiencia en lengua de señas, distintas tallas de mano).

### Estratificación deseada
| Variable | Categorías |
|---|---|
| Experiencia con LSP/LIBRAS | 0 (ninguna) · 1–2 (básica) · 3+ (fluente) |
| Mano dominante | Derecha · Izquierda |
| Tono de piel | Fitzpatrick I–III · IV–VI (relevante para MediaPipe) |

## 4. Condiciones controladas y registradas

Cada sesión registra al inicio (campos guardados en `session.json`):

| Campo | Tipo | Notas |
|---|---|---|
| `subject_id` | str | Anonimizado (S01, S02, …) |
| `age` | str | Auto-declarado |
| `dominant_hand` | R/L | |
| `libras_experience_0_5` | int | Auto-evaluado |
| `lighting_1_5` | int | 1 = muy oscura, 5 = muy clara |
| `background` | str | uniforme / cluttered |
| `hand_to_camera_cm` | int | Distancia aproximada |
| `camera_height` | str | pecho / cara / mesa |
| `camera_fps` | float | Reportado por OpenCV |
| `model_dir` | str | Modelo evaluado |
| `model_test_accuracy` | float | Métrica de holdout reportada |

Condiciones recomendadas (no forzadas) para el piloto:
- Iluminación frontal, sin contraluz.
- Fondo uniforme (pared lisa).
- Distancia mano-cámara: 40–60 cm.
- Webcam 720p mínimo, 30 fps.

## 5. Procedimiento por sesión

1. El operador ejecuta:
   ```bash
   python scripts/evaluate/evaluate_realworld.py \
     --model-dir /data/models/librai-alphabet-v4/run_20260601_015641 \
     --subject-id S01 --reps 5
   ```
2. El script solicita metadatos (~30 s) y los guarda.
3. Para cada letra del alfabeto (22 clases: A–Y + Ç):
   1. Pantalla muestra la letra grande + imagen de referencia.
   2. Cuenta regresiva de 3 segundos.
   3. Captura ventana fija de **60 frames (~2 s)**: se graban
      todos los frames, incluyendo aquellos sin detección de mano.
   4. Repetir 5 veces. El operador puede:
      - `SPACE`: aceptar la repetición y pasar a la siguiente.
      - `R`: descartar y regrabar (si hubo error obvio: mano
        fuera de cuadro, oclusión accidental).
      - `S`: saltar la letra completa.
      - `Q`: terminar la sesión.
   5. Si menos del 50 % de los frames de la ventana tuvieron mano
      detectada, el script fuerza regrabación automática.

**Tiempo total estimado:** 22 letras × 5 reps × (3 s countdown + 2 s
captura + ~3 s setup) ≈ 15–20 min por sesión.

## 6. Datos guardados

```
data/realworld_eval/sessions/{timestamp}_{subject_id}/
├── session.json              # metadatos de la sesión
├── report.json               # métricas calculadas
├── confusion_matrix.npy      # matriz 22×22 cruda
├── confusion_matrix.png      # heatmap normalizado por fila
└── letters/
    ├── A_rep00.npz           # features (60,120), probs (60,22), n_detected
    ├── A_rep01.npz
    ├── ...
    └── Ç_rep04.npz
```

**Nada de video crudo.** Solo coordenadas normalizadas y predicciones
del modelo. Permite reanálisis con modelos futuros sin re-grabar.

## 7. Métricas reportadas

Por letra y globales:

| Métrica | Definición |
|---|---|
| `top1_accuracy` | % de frames donde la letra real es la de mayor probabilidad |
| `top5_accuracy` | % de frames donde la letra real está en el top-5 |
| `mean_confidence_on_target` | probabilidad media asignada a la letra correcta |
| `frames_detected / frames_total` | calidad de detección de MediaPipe en la sesión |
| `gap_vs_holdout` | `test_accuracy_holdout - top1_real` (pp) |

Visualización: matriz de confusión normalizada por fila →
diagonal fuerte = modelo robusto; bandas claras = confusiones
sistemáticas (esperables U↔V, M↔N, R↔U).

## 8. Análisis a posteriori

Con los `.npz` guardados se puede:
- Re-evaluar con otros modelos sin re-grabar (`probs` se regenera
  pasando `features` por el nuevo modelo).
- Identificar frames con confianza baja para inspección manual.
- Estimar varianza inter-sujeto: combinar reportes de múltiples
  sesiones y reportar mean ± std del accuracy por letra.

## 9. Limitaciones declaradas

- **Sesgo del operador:** quien graba conoce el modelo y puede,
  conscientemente o no, posicionarse para maximizar reconocimiento.
  Mitigación: incluir al menos 1 sujeto naïve.
- **Sin ground-truth de gesto correcto:** asumimos que el sujeto
  ejecuta la letra "bien". No hay validador externo (intérprete).
  Mitigación a futuro: revisión post-hoc por intérprete certificado
  para una muestra del 10 %.
- **Variabilidad de MediaPipe** entre versiones (host 0.10.35 vs
  Docker 0.10.8) no está medida — usar la misma versión en
  entrenamiento e inferencia es supuesto.
- **No mide rendimiento en flujo continuo** (signing de palabras),
  solo letras aisladas estáticas.

## 10. Definición de "hecho"

El protocolo está completo cuando:
- [ ] Script ejecutado en al menos 1 sujeto piloto (puede ser el
      desarrollador).
- [ ] `report.json` y `confusion_matrix.png` generados.
- [ ] Resultados anotados en `experiments.csv` (Sprint 2) junto a
      los modelos comparados.
- [ ] Decisión documentada: ¿v4 (120 features) es defendible para
      cámara real, o hay que volver a v3 / rediseñar features?

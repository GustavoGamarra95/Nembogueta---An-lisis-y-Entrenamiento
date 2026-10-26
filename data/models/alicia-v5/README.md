# Alicia v5

Reconocedor del alfabeto manual multilengua (LIBRAS, LSC, ASL y LSPy) con un
componente de movimiento para las letras dinámicas de LSPy.

| Componente | Carpeta | Script |
|---|---|---|
| Estático (frame a frame, 120 features) | `estatico/run_20261026_093418` | `scripts/train/train_alphabet_multilang.py` |
| Movimiento (secuencia de 8 pasos, forma + trayectoria) | `secuencia/run_20261026_094532` | `scripts/train/train_letter_sequences.py` |

Semilla 42 en ambos.

## Datos

| Lengua | Fuentes | Personas |
|---|---|---|
| LIBRAS | LibrAI + sesiones de cámara real S01/S02 | LibrAI + 2 |
| LSC | LSC70 (CC BY 4.0) | 70 |
| ASL | Massey (CC0) + Spelling It Out | 10 |
| LSPy | Proxy letra por letra desde LSC/LIBRAS/ASL (mapeo derivado del video "Alfabeto Manual" del canal Lengua de Señas LSPy) + frames reales de 8 videos públicos de 7 canales (canal + 7 videos de YouTube alineados) | 8 reales |
| Movimiento | LSC70 (J, Z, Ñ), LSM dinámico (J, K, Ñ, Q, X, Z; Zenodo 14689869, CC BY 4.0, 20 personas), secuencias LSPy reales | — |

El uso de los videos LSPy fue autorizado por sus titulares; se publican solo
landmarks y modelos derivados (CC BY 4.0), los videos no se redistribuyen.

## Resultados

Test interno, restringido a la lengua elegida:

| Conjunto | top-1 |
|---|---|
| LibrAI (LIBRAS) | 98.75% |
| Operador en cámara real (LIBRAS) | 85.63% |
| LSC, 10 sujetos no vistos | 80.98% |
| ASL, 2 sujetos no vistos | 84.57% |

Validación externa LSPy: signario SLS, 20 letras de **un** intérprete no
usado en entrenamiento ni en el mapeo:

| Modo | Aciertos | Fallos |
|---|---|---|
| Estático | 15/20 | I, J, N, X, Ñ |
| Movimiento | 11/20 | — |
| **Fusión (recomendado)** | **16/20** | N→Ñ, U→R, X→A, Ñ→H |

## Uso

1. Elegir la lengua y aplicar el argmax solo sobre sus clases (`LSPY:*`,
   `LIBRAS:*`, `LSP:*` = LSC, `ASL:*`).
2. Canonicalizar la secuencia del signante con `canonicalize_by_majority`
   (`train_alphabet_robust.py`) por la ambigüedad palma/dorso de MediaPipe.
3. LSPy con movimiento: fusión = media geométrica de la probabilidad
   estática (promedio del tramo sostenido) y la de secuencia, peso 0.5.

## Limitaciones

- La validación LSPy es un solo intérprete con una toma por letra; el
  signario no tiene B, C, E, F, G, H ni W.
- Falta separar N/Ñ (la Ñ es N con vaivén) y X.
- Las etiquetas LSPy de YouTube vienen de un alineamiento forzado con el
  modelo estático (errores filtrados por confianza, no revisión manual
  letra por letra).
- Los datos LSPy son una recopilación pública; la colecta propia
  multi-sujeto con landmarks crudos sigue pendiente.

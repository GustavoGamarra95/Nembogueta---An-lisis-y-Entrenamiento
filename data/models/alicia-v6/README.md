# Alicia v6

Alicia v5 + **alfabeto guaraní (achegety)** como lengua "GN": 33 letras, de
las cuales 13 son propias del guaraní (ã, ẽ, ĩ, õ, ũ, ỹ, g̃, ch, mb, nd,
ng, nt, puso). Primer paso de la salida trilingüe (Fase 4).

| Componente | Carpeta | Script |
|---|---|---|
| Estático (LIBRAS, LSC, ASL, LSPy, GN) | `estatico/run_20261026_101456` | `scripts/train/train_alphabet_multilang.py` |
| Movimiento (letras dinámicas LSPy) | `secuencia/run_20261026_103155` | `scripts/train/train_letter_sequences.py` |

Semilla 42 en ambos.

## Datos del guaraní

- Video "Achegety Guaraní Po Ñe'ẽ rehegua — Alfabeto Guaraní en Lengua de
  Señas" (página Lengua de Señas Py - LSPy, Facebook), un intérprete.
- Tiempos de cada letra leídos frame a frame combinando la forma de la mano
  con el puntero sobre la lámina del achegety (sincronizado con la seña):
  `data/analysis/achegety/lectura_manual.csv` (con la evidencia de cada
  letra) y `data/analysis/achegety/revision_verificada.csv` (fronteras ajustadas al frame de mayor
  velocidad de la mano, ±0.4 s, sin dejar letras de menos de 0.4 s).
- Patrones observados: las vocales nasales y g̃ son la letra base junto a
  la boca o la nariz; los dígrafos encadenan sus dos letras (CH = C→H,
  NG = N→G, ND = N→D, NT = N→T, MB = M→mano plana a la boca).
- Las letras compartidas con LSPy usan además todos los datos LSPy de v5
  (proxy + 9 intérpretes); las 13 propias salen solo de este video.

## Resultados

| Conjunto | v5 | v6 |
|---|---|---|
| Signario SLS (LSPy, 1 intérprete externo), estático | 15/20 | **16/20** |
| Signario SLS, fusión estático + movimiento | 16/20 | 16/20 |
| Operador en cámara real (LIBRAS) | 85.6% | **92.6%** |
| LibrAI (LIBRAS) | 98.8% | 97.6% |
| LSC / ASL, sujetos no vistos | 81.0% / 84.6% | 79.6% / 85.1% |
| GN, letras compartidas, sujetos no vistos | — | 69.4% |

En GN las letras compartidas bajan donde compiten con su variante guaraní
(A ↔ Ã, N ↔ Ñ/NT, H ↔ CH, R ↔ RR).

## Limitaciones

- **Las 13 letras propias del guaraní no tienen validación**: un solo
  intérprete, una toma, todo en entrenamiento. Son un prototipo funcional,
  no un resultado medido.
- Los datos LSPy y GN son una recopilación pública de investigación (licencia
  estándar de YouTube / Facebook): no se redistribuyen, solo landmarks y
  modelos derivados. No constituyen un dataset publicable.
- Las nasales y los dígrafos llevan movimiento; el modelo de secuencia aún
  no tiene clases GN (las cubre solo el estático).

## Uso

Igual que v5: elegir la lengua (`GN:*` para guaraní), argmax restringido a
sus clases, canonicalizar con `canonicalize_by_majority`.

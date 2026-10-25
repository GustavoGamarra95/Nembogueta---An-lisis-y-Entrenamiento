# LSC vs LIBRAS — comparación de alfabetos manuales

Generado con `scripts/evaluate/compare_alphabets_lsc_libras.py` sobre
`data/processed/lsc70_alphabet` (`preprocess_lsc70.py --convention librai`) y
`data/processed/librai_alphabet_v4`.

- `summary.csv` / `report.json`: veredicto por letra (cruce con Alicia
  robust-v4 + distancia de centroides).
- `cross_matrix.png`: letra LSC (filas) → predicción LIBRAS (columnas).
- `gallery.png`: 3 ejemplos LSC70 y 3 LibrAI por letra, recortados a la mano.

## Atribución

`gallery.png` contiene recortes de imágenes de **LSC70**:
"Speak in Colombian Sign Language: A Dynamic LSC70 Database", Universidad del
Cauca, Mendeley Data, DOI [10.17632/9ssyn8tff5.2](https://doi.org/10.17632/9ssyn8tff5.2),
licencia [CC BY 4.0](http://creativecommons.org/licenses/by/4.0).
Cambios: recorte alrededor de la mano detectada por MediaPipe y redimensionado
a 120×120 px.

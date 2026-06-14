# Scripts legacy

Estos scripts fueron parte de exploración temprana y no están en el flujo
activo del proyecto. Se conservan para referencia histórica y para no
perder código si alguna idea resulta útil más adelante.

**No documentados, no testeados, posiblemente rotos** con la versión actual
de `feature_engineering.py` (120 features) — la mayoría fueron escritos
para versiones anteriores del pipeline (208 / 280 features).

## Por qué cada uno está acá

| Script | Razón |
|---|---|
| `demo/demo_completo.py` | Demo experimental superada por `realtime_librai_alphabet.py` |
| `demo/demo_realtime.py` | Versión inicial del demo, superada |
| `demo/demo_realtime_improved.py` | Iteración intermedia, superada |
| `demo/realtime_alphabet.py` | Demo de alfabeto v1, superado por `realtime_librai_alphabet.py` |
| `demo/realtime_libras_complete.py` | WIP sin documentación, fuera de scope |
| `preprocess/preprocess_bsl_alphabet.py` | British Sign Language, fuera de scope LSP |
| `preprocess/preprocess_facial_expressions.py` | Pipeline de expresiones faciales no integrado |
| `preprocess/preprocess_lswh100.py` | Dataset LSWH100, no usado en el TCC |
| `preprocess/preprocess_sign_language.py` | Pipeline genérico exploratorio |
| `train/train_bsl_alphabet.py` | Modelo BSL, fuera de scope |
| `train/train_facial_expressions.py` | Modelo facial no integrado |
| `train/train_handshape.py` | Trainer LSWH100, reemplazado por `train_ufpr.py` |
| `train/train_sign_language.py` | Trainer genérico exploratorio |
| `train/train_vlibrasil_metric.py` | Prototypical networks (few-shot), experimental |
| `utils/collect_letter_data.py` | Colector v1, reemplazado por `guided_calibration.py` |
| `utils/collect_letter_data_improved.py` | Colector v2, reemplazado por `evaluate_realworld.py` + `guided_calibration.py` |
| `utils/crearestructura.py` | Script one-shot para crear directorios |
| `utils/reorganize_processed_data.py` | Script one-shot de migración |
| `utils/test_alphabet_model.py` | Reemplazado por `scripts/demo/test_alphabet.py` |

Si encontrás algo útil acá, copialo al directorio activo correspondiente y
actualizalo a la API actual.

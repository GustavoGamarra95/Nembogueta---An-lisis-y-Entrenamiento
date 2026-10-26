"""Diccionario trilingüe (español, guaraní, portugués) del vocabulario."""

import csv
import unicodedata
from pathlib import Path
from typing import Dict, Optional

IDIOMAS = {"es": "español", "gn": "guaraní", "pt": "português"}

RAIZ = Path(__file__).resolve().parents[2]
VOCABULARIO = RAIZ / "data/analysis/vocabulario/vocabulario_objetivo.csv"


def normalizar(texto: str) -> str:
    """Mayúsculas sin tildes (salvo Ñ) ni espacios repetidos."""
    texto = texto.upper().replace("Ñ", "\0")
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return " ".join(texto.replace("\0", "Ñ").split())


class Diccionario:
    """Entradas del vocabulario-objetivo indexadas por su forma normalizada.

    La columna `gn_borrador` es una traducción provisional que debe validar
    un hablante de guaraní antes de usarse fuera del prototipo.
    """

    def __init__(self, ruta: Path = VOCABULARIO):
        self.entradas: Dict[str, Dict[str, str]] = {}
        if not ruta.exists():
            return
        with open(ruta, encoding="utf-8") as f:
            for fila in csv.DictReader(f):
                self.entradas[normalizar(fila["es"])] = {
                    "es": fila["es"],
                    "gn": fila["gn_borrador"],
                    "pt": fila["pt"],
                    "categoria": fila["categoria"],
                }

    def traducir(self, palabra: str, idioma: str) -> Optional[str]:
        """Traducción de `palabra` (en español) a `idioma`, o None."""
        if idioma not in IDIOMAS:
            raise ValueError(f"Idioma no soportado: {idioma}")
        entrada = self.entradas.get(normalizar(palabra))
        return entrada[idioma] if entrada else None

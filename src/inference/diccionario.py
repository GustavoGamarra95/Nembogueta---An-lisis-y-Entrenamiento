"""Diccionario LSP → {español paraguayo, guaraní, portugués de Brasil}.

Fuente: data/analysis/diccionario/diccionario_lsp.csv (1810 glosas del
Signario SLS; ver scripts/preprocess/build_diccionario_lsp.py). Las
traducciones las generó una IA y están sin revisar salvo las marcadas como
"validado" en <idioma>_estado.
"""

import csv
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

IDIOMAS = {"es": "español (Paraguay)", "gn": "guaraní", "pt": "português"}
COLUMNA = {"es": "es_py", "gn": "gn", "pt": "pt_br"}

RAIZ = Path(__file__).resolve().parents[2]
DICCIONARIO = RAIZ / "data/analysis/diccionario/diccionario_lsp.csv"
MAX_PALABRAS = 4


def normalizar(texto: str) -> str:
    """Mayúsculas sin tildes (salvo Ñ) ni espacios repetidos."""
    texto = texto.upper().replace("Ñ", "\0")
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return " ".join(texto.replace("\0", "Ñ").split())


class Diccionario:
    """Entradas indexadas por la glosa LSP y por la palabra en español.

    Ante varias acepciones de una misma glosa gana la primera variante sin
    acepción (la forma más general).
    """

    def __init__(self, ruta: Path = DICCIONARIO):
        self.entradas: Dict[str, Dict[str, str]] = {}
        if not ruta.exists():
            return
        with open(ruta, encoding="utf-8") as f:
            filas = list(csv.DictReader(f))
        filas.sort(key=lambda r: (bool(r["acepcion"]), int(r["variante"])))
        for fila in filas:
            entrada = {i: fila[c] for i, c in COLUMNA.items()}
            entrada["estado"] = {i: fila[f"{i}_estado"] for i in COLUMNA}
            entrada["sena"] = fila["clave"]
            base = fila["clave"].split("|")[0]
            for clave in (base, fila["es_py"]):
                self.entradas.setdefault(normalizar(clave), entrada)

    def __len__(self) -> int:
        return len({e["sena"] for e in self.entradas.values()})

    def buscar(self, palabra: str) -> Optional[Dict[str, str]]:
        return self.entradas.get(normalizar(palabra))

    def traducir(self, palabra: str, idioma: str) -> Optional[str]:
        """Traducción de `palabra` (glosa o español) a `idioma`, o None si
        no está en el diccionario o falta la traducción a ese idioma."""
        if idioma not in IDIOMAS:
            raise ValueError(f"Idioma no soportado: {idioma}")
        entrada = self.buscar(palabra)
        return (entrada[idioma] or None) if entrada else None

    def traducir_texto(self, texto: str, idioma: str) -> Dict:
        """Traduce un texto agrupando hasta MAX_PALABRAS palabras por entrada
        (p. ej. "buenos días"); las palabras desconocidas quedan igual."""
        palabras = texto.split()
        partes: List[Dict] = []
        i = 0
        while i < len(palabras):
            for n in range(min(MAX_PALABRAS, len(palabras) - i), 0, -1):
                trozo = " ".join(palabras[i : i + n])  # noqa: E203
                traduccion = self.traducir(trozo, idioma)
                if traduccion or n == 1:
                    partes.append({"origen": trozo, "traduccion": traduccion})
                    i += n
                    break
        texto_traducido = " ".join(
            p["traduccion"] or p["origen"] for p in partes
        )
        return {
            "texto": texto_traducido,
            "partes": partes,
            "completo": all(p["traduccion"] for p in partes),
            "validado": bool(partes)
            and all(
                (self.buscar(p["origen"]) or {}).get("estado", {}).get(idioma)
                == "validado"
                for p in partes
            ),
        }

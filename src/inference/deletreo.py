"""Deletreo en tiempo real: letras de Alicia → palabras → idioma de salida."""

from collections import deque
from typing import Dict, List, Optional

from src.inference.alicia import Alicia, Mano
from src.inference.diccionario import IDIOMAS, Diccionario


class Deletreador:
    """Acumula las letras reconocidas frame a frame en palabras.

    Una letra se confirma cuando domina la ventana de frames con confianza
    suficiente durante `estable` frames seguidos (descarta las transiciones
    entre letras); repetir la misma letra exige bajar la mano o cambiar de
    letra. Si no hay mano durante una ventana entera, la palabra en curso se
    cierra (la mano en reposo funciona como espacio).
    """

    def __init__(
        self,
        alicia: Alicia,
        diccionario: Diccionario,
        lengua: str = "LSPY",
        idioma: str = "es",
        ventana: int = 15,
        umbral: float = 0.6,
        estable: int = 10,
    ):
        if idioma not in IDIOMAS:
            raise ValueError(f"Idioma no soportado: {idioma}")
        self.alicia = alicia
        self.diccionario = diccionario
        self.lengua = lengua
        self.idioma = idioma
        self.ventana = ventana
        self.umbral = umbral
        self.estable = estable
        self.borrar()

    def borrar(self):
        self.buffer: deque = deque(maxlen=self.ventana)
        self.palabras: List[str] = []
        self.actual: List[str] = []
        self.ultima: Optional[str] = None
        self.sin_mano = 0
        self.candidato: Optional[str] = None
        self.racha = 0

    def espacio(self):
        if self.actual:
            self.palabras.append("".join(self.actual))
        self.actual = []
        self.ultima = None

    def retroceder(self):
        if self.actual:
            self.actual.pop()
        elif self.palabras:
            self.actual = list(self.palabras.pop())
        self.ultima = None

    @property
    def texto(self) -> str:
        return " ".join(self.palabras + ["".join(self.actual)]).strip()

    def agregar(self, manos: List[Mano]) -> Dict:
        """Procesa un frame y devuelve la predicción, el texto deletreado y
        su traducción al idioma de salida."""
        self.sin_mano = 0 if manos else self.sin_mano + 1
        if self.sin_mano >= self.ventana:
            self.espacio()
            self.buffer.clear()
        self.buffer.append(manos)
        pred = self.alicia.predecir(list(self.buffer), self.lengua)
        letra = pred["letra"]
        valida = (
            letra
            and pred["confianza"] >= self.umbral
            and pred["frames_con_mano"] >= self.ventana // 2
        )
        if valida and letra == self.candidato:
            self.racha += 1
        else:
            self.candidato = letra if valida else None
            self.racha = 1 if valida else 0
        if (
            self.candidato
            and self.racha >= self.estable
            and self.candidato != self.ultima
        ):
            self.actual.append(letra)
            self.ultima = letra
        return {**pred, **self.resumen()}

    def resumen(self) -> Dict:
        texto = self.texto
        traduccion = (
            self.diccionario.traducir_texto(texto, self.idioma)
            if texto
            else None
        )
        return {
            "palabra": "".join(self.actual),
            "texto": texto,
            "idioma": self.idioma,
            "traduccion": traduccion["texto"] if traduccion else None,
            "en_diccionario": bool(traduccion and traduccion["completo"]),
            "validado": bool(traduccion and traduccion["validado"]),
        }

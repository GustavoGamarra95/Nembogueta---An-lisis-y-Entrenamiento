"""API de inferencia en tiempo real de Ñemongueta (Alicia v6).

Lanzar desde la raíz del repositorio:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000

Documentación interactiva en http://localhost:8000/docs
"""

import base64
from collections import deque
from functools import lru_cache
from typing import List, Literal, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.websockets import WebSocketDisconnect
from pydantic import BaseModel, Field

from src.api.alicia import Alicia, Mano
from src.api.diccionario import IDIOMAS, Diccionario

app = FastAPI(
    title="Ñemongueta — API de Alicia",
    description=(
        "Reconocimiento del alfabeto manual (LSPy, guaraní, LIBRAS, LSC, "
        "ASL) con salida en español, guaraní o portugués."
    ),
    version="0.6.0",
)

Lengua = Literal["LSPY", "GN", "LIBRAS", "LSP", "ASL"]
Idioma = Literal["es", "gn", "pt"]
NOMBRES_LENGUA = {
    "LSPY": "Lengua de Señas Paraguaya (alfabeto)",
    "GN": "Alfabeto guaraní (achegety) en LSPy",
    "LIBRAS": "Língua Brasileira de Sinais (alfabeto)",
    "LSP": "Lengua de Señas Colombiana (LSC70)",
    "ASL": "American Sign Language (alfabeto)",
}


@lru_cache(maxsize=1)
def alicia() -> Alicia:
    return Alicia()


@lru_cache(maxsize=1)
def diccionario() -> Diccionario:
    return Diccionario()


class ManoIn(BaseModel):
    puntos: List[List[float]] = Field(
        ...,
        min_length=21,
        max_length=21,
        description="21 landmarks (x, y, z) de MediaPipe sobre la imagen "
        "original de la cámara, sin espejar",
    )
    lateralidad: Literal["Left", "Right"]


class PrediccionIn(BaseModel):
    lengua: Lengua = "LSPY"
    frames: List[List[ManoIn]] = Field(
        ..., min_length=1, description="Frames; cada uno con 0 a 2 manos"
    )
    top_k: int = Field(3, ge=1, le=10)
    sostenido: bool = Field(
        True,
        description="Usar solo el tramo con la mano estable (para la "
        "grabación completa de una letra)",
    )


class TraduccionIn(BaseModel):
    palabra: str = Field(..., min_length=1, description="Palabra en español")
    idioma: Idioma = "gn"


def _frames(frames: List[List[ManoIn]]) -> List[List[Mano]]:
    return [[Mano(m.puntos, m.lateralidad) for m in f] for f in frames]


def _manos_de_imagen(imagen: np.ndarray) -> List[Mano]:
    import cv2
    import mediapipe as mp

    with mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=2
    ) as hands:
        res = hands.process(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks:
        return []
    return [
        Mano([(p.x, p.y, p.z) for p in h.landmark], c.classification[0].label)
        for h, c in zip(res.multi_hand_landmarks, res.multi_handedness)
    ]


def _decodificar(datos: bytes) -> np.ndarray:
    import cv2

    imagen = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)
    if imagen is None:
        raise HTTPException(400, "No se pudo leer la imagen")
    return imagen


@app.get("/salud")
def salud():
    a = alicia()
    return {
        "estado": "ok",
        "modelo": a.ruta.parent.parent.name,
        "run": a.ruta.name,
    }


@app.get("/lenguas")
def lenguas():
    a = alicia()
    return [
        {
            "codigo": c,
            "nombre": NOMBRES_LENGUA.get(c, c),
            "letras": [a.clases[i].split(":")[1] for i in idx],
        }
        for c, idx in a.lenguas.items()
    ]


@app.get("/idiomas")
def idiomas():
    return [{"codigo": c, "nombre": n} for c, n in IDIOMAS.items()]


@app.post("/predecir/landmarks")
def predecir_landmarks(datos: PrediccionIn):
    """Letra a partir de landmarks calculados en el cliente."""
    return alicia().predecir(
        _frames(datos.frames), datos.lengua, datos.top_k, datos.sostenido
    )


@app.post("/predecir/imagen")
async def predecir_imagen(
    imagen: UploadFile = File(...), lengua: Lengua = "LSPY", top_k: int = 3
):
    """Letra a partir de una imagen (JPG/PNG) de la cámara, sin espejar."""
    manos = _manos_de_imagen(_decodificar(await imagen.read()))
    return alicia().predecir([manos], lengua, top_k)


@app.post("/traducir")
def traducir(datos: TraduccionIn):
    """Palabra deletreada (en español) → idioma elegido.

    Las traducciones al guaraní son un borrador pendiente de validación.
    """
    traduccion = diccionario().traducir(datos.palabra, datos.idioma)
    return {
        "palabra": datos.palabra,
        "idioma": datos.idioma,
        "traduccion": traduccion,
        "en_vocabulario": traduccion is not None,
    }


@app.websocket("/ws")
async def tiempo_real(
    ws: WebSocket,
    lengua: Lengua = "LSPY",
    idioma: Idioma = "es",
    ventana: int = 15,
):
    """Tiempo real: el cliente envía {"imagen": <JPG en base64>} o
    {"manos": [...]} por frame y {"accion": "borrar"} para reiniciar la
    palabra. Una letra se confirma cuando domina la ventana de frames."""
    await ws.accept()
    buffer: deque = deque(maxlen=ventana)
    palabra: List[str] = []
    ultima: Optional[str] = None
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("accion") == "borrar":
                buffer.clear(), palabra.clear()
                ultima = None
                continue
            if "imagen" in msg:
                manos = _manos_de_imagen(
                    _decodificar(base64.b64decode(msg["imagen"]))
                )
            else:
                manos = [
                    Mano(m["puntos"], m["lateralidad"])
                    for m in msg.get("manos", [])
                ]
            buffer.append(manos)
            pred = alicia().predecir(list(buffer), lengua)
            letra = pred["letra"]
            if (
                letra
                and letra != ultima
                and pred["confianza"] >= 0.6
                and pred["frames_con_mano"] >= ventana // 2
            ):
                palabra.append(letra)
                ultima = letra
            texto = "".join(palabra)
            await ws.send_json(
                {
                    **pred,
                    "palabra": texto,
                    "traduccion": diccionario().traducir(texto, idioma)
                    if texto
                    else None,
                }
            )
    except WebSocketDisconnect:
        return

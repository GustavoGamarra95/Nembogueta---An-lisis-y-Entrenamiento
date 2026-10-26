"""API de inferencia en tiempo real de Ñemongueta (Alicia v6).

Lanzar desde la raíz del repositorio:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000

Demo web en http://localhost:8000 y documentación en /docs
"""

import base64
from functools import lru_cache
from pathlib import Path
from typing import List, Literal

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.responses import FileResponse
from fastapi.websockets import WebSocketDisconnect
from pydantic import BaseModel, Field

from src.inference.alicia import Alicia, Mano
from src.inference.deletreo import Deletreador
from src.inference.diccionario import IDIOMAS, Diccionario

ESTATICOS = Path(__file__).parent / "static"

app = FastAPI(
    title="Ñemongueta — API de Alicia",
    description=(
        "Reconocimiento del alfabeto manual (LSPy, guaraní, LIBRAS, LSC, "
        "ASL) con salida en español, guaraní o portugués."
    ),
    version="0.7.0",
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
    palabra: str = Field(
        ..., min_length=1, description="Palabra o texto en español o glosa"
    )
    idioma: Idioma = "gn"


def _frames(frames: List[List[ManoIn]]) -> List[List[Mano]]:
    return [[Mano(m.puntos, m.lateralidad) for m in f] for f in frames]


def _detector(video: bool):
    import mediapipe as mp

    return mp.solutions.hands.Hands(
        static_image_mode=not video,
        max_num_hands=2,
        min_detection_confidence=0.5,
    )


def _manos_de_imagen(imagen: np.ndarray, hands=None) -> List[Mano]:
    import cv2

    propio = hands is None
    hands = hands or _detector(video=False)
    try:
        res = hands.process(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    finally:
        if propio:
            hands.close()
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


def _puntos(manos: List[Mano]) -> List[List[List[float]]]:
    return [
        [[round(p[0], 4), round(p[1], 4)] for p in m.puntos] for m in manos
    ]


@app.get("/", include_in_schema=False)
def demo():
    return FileResponse(ESTATICOS / "index.html")


@app.get("/salud")
def salud():
    a = alicia()
    return {
        "estado": "ok",
        "modelo": a.ruta.parent.parent.name,
        "run": a.ruta.name,
        "diccionario": len(diccionario()),
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


@app.get("/diccionario/{palabra}")
def buscar(palabra: str):
    """Entrada del diccionario LSP (glosa o palabra en español) en los tres
    idiomas con el estado de cada traducción ("validado", "IA sin revisar" o
    "pendiente"), o 404."""
    entrada = diccionario().buscar(palabra)
    if not entrada:
        raise HTTPException(404, "No está en el diccionario")
    return entrada


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
    return {
        **alicia().predecir([manos], lengua, top_k),
        "manos": _puntos(manos),
    }


@app.post("/traducir")
def traducir(datos: TraduccionIn):
    """Palabra o texto deletreado → idioma elegido (diccionario LSP).

    `validado` es falso mientras alguna parte sea una traducción de la IA sin
    revisión humana.
    """
    resultado = diccionario().traducir_texto(datos.palabra, datos.idioma)
    traduccion = diccionario().traducir(datos.palabra, datos.idioma)
    return {
        "palabra": datos.palabra,
        "idioma": datos.idioma,
        "traduccion": traduccion or resultado["texto"],
        "en_vocabulario": resultado["completo"],
        "validado": resultado["validado"],
        "partes": resultado["partes"],
    }


@app.websocket("/ws")
async def tiempo_real(
    ws: WebSocket,
    lengua: Lengua = "LSPY",
    idioma: Idioma = "es",
    ventana: int = 15,
):
    """Tiempo real. Mensajes del cliente:
    {"imagen": <JPG en base64>} o {"manos": [...]} por frame;
    {"accion": "espacio" | "retroceder" | "borrar"};
    {"idioma": "es" | "gn" | "pt"} o {"lengua": "LSPY" | ...}.
    Responde con la letra, el texto deletreado y su traducción."""
    await ws.accept()
    deletreo = Deletreador(alicia(), diccionario(), lengua, idioma, ventana)
    hands = None
    try:
        while True:
            msg = await ws.receive_json()
            accion = msg.get("accion")
            if accion in ("espacio", "retroceder", "borrar"):
                getattr(deletreo, accion)()
                await ws.send_json(deletreo.resumen())
                continue
            if msg.get("idioma") in IDIOMAS:
                deletreo.idioma = msg["idioma"]
            if msg.get("lengua") in alicia().lenguas:
                deletreo.lengua = msg["lengua"]
                deletreo.buffer.clear()
            if "imagen" in msg:
                hands = hands or _detector(video=True)
                imagen = _decodificar(base64.b64decode(msg["imagen"]))
                manos = _manos_de_imagen(imagen, hands)
            elif "manos" in msg:
                manos = [
                    Mano(m["puntos"], m["lateralidad"]) for m in msg["manos"]
                ]
            else:
                await ws.send_json(deletreo.resumen())
                continue
            await ws.send_json(
                {**deletreo.agregar(manos), "manos": _puntos(manos)}
            )
    except WebSocketDisconnect:
        return
    finally:
        if hands is not None:
            hands.close()

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from src.api.alicia import Mano, canonicalizar, vector_126  # noqa: E402
from src.api.app import app  # noqa: E402
from src.api.diccionario import Diccionario, normalizar  # noqa: E402

EJEMPLO_A = Path(__file__).parent / "ejemplo_A.json"


@pytest.fixture(scope="module")
def cliente():
    return TestClient(app)


def test_salud(cliente):
    r = cliente.get("/salud")
    assert r.status_code == 200
    assert r.json()["estado"] == "ok"


def test_idiomas(cliente):
    codigos = {i["codigo"] for i in cliente.get("/idiomas").json()}
    assert codigos == {"es", "gn", "pt"}


def test_lenguas_incluye_lspy_y_guarani(cliente):
    lenguas = {
        x["codigo"]: x["letras"] for x in cliente.get("/lenguas").json()
    }
    assert len(lenguas["LSPY"]) == 29
    assert len(lenguas["GN"]) == 33
    assert {"Ã", "MB", "PUSO"} <= set(lenguas["GN"])


def test_traducir_vocabulario(cliente):
    r = cliente.post("/traducir", json={"palabra": "agua", "idioma": "pt"})
    assert r.json()["traduccion"] == "Água"
    r = cliente.post("/traducir", json={"palabra": "AGUA", "idioma": "gn"})
    assert r.json()["traduccion"] == "y"


def test_traducir_palabra_fuera_de_vocabulario(cliente):
    r = cliente.post("/traducir", json={"palabra": "xyz", "idioma": "es"})
    assert r.json()["en_vocabulario"] is False


def test_predecir_letra_a_del_signario(cliente):
    datos = json.loads(EJEMPLO_A.read_text())
    r = cliente.post("/predecir/landmarks", json=datos)
    assert r.status_code == 200
    assert r.json()["letra"] == "A"


def test_predecir_sin_manos(cliente):
    r = cliente.post(
        "/predecir/landmarks", json={"lengua": "LSPY", "frames": [[]]}
    )
    assert r.json()["letra"] is None


def test_lengua_invalida(cliente):
    r = cliente.post(
        "/predecir/landmarks", json={"lengua": "XX", "frames": [[]]}
    )
    assert r.status_code == 422


def test_vector_126_espeja_y_usa_slot_librai():
    puntos = [[0.2, 0.5, 0.0]] * 21
    raw = vector_126([Mano(puntos, "Left")])
    # "Left" en la imagen original es "Right" en la espejada → slot [63:126]
    assert np.allclose(raw[63], 0.8)
    assert np.all(raw[:63] == 0)


def test_canonicalizar_lleva_minoria_al_slot_mayoritario():
    feats = np.zeros((5, 120), dtype=np.float32)
    feats[:4, 60:] = 1.0
    feats[4, :60] = 1.0
    salida = canonicalizar(feats)
    assert np.all(np.abs(salida[4, 60:]) > 0)
    assert np.all(salida[4, :60] == 0)


def test_normalizar_conserva_enie():
    assert normalizar("  dónde  ") == "DONDE"
    assert normalizar("año") == "AÑO"
    assert Diccionario().traducir("Dónde", "pt") == "Onde"


def test_tiempo_real_deletrea_la_letra(cliente):
    frames = json.loads(EJEMPLO_A.read_text())["frames"]
    with cliente.websocket_connect("/ws?lengua=LSPY&ventana=8") as ws:
        for manos in frames * 2:  # la letra debe sostenerse 10 frames
            ws.send_json({"manos": manos})
            respuesta = ws.receive_json()
    assert respuesta["palabra"] == "A"


def test_diccionario_cubre_el_signario():
    d = Diccionario()
    assert len(d) >= 1744
    assert d.traducir("MÉDICO", "gn") == "pohanohára"
    assert d.traducir("ABURRIDO", "gn") is None  # sin guaraní todavía


def test_traducir_texto_agrupa_frases():
    r = Diccionario().traducir_texto("buenos dias doctor", "gn")
    assert r["texto"] == "mba'éichapa nde pyhareve doctor"
    assert r["completo"] is False


def test_buscar_en_diccionario(cliente):
    r = cliente.get("/diccionario/agua")
    assert r.json()["gn"] == "y"
    assert cliente.get("/diccionario/zzz").status_code == 404


def test_frontend(cliente):
    r = cliente.get("/")
    assert r.status_code == 200
    assert "Alicia" in r.text


def test_tiempo_real_espacio_y_traduccion(cliente):
    frames = json.loads(EJEMPLO_A.read_text())["frames"]
    with cliente.websocket_connect(
        "/ws?lengua=LSPY&idioma=pt&ventana=8"
    ) as ws:
        for manos in frames * 2:  # la letra debe sostenerse 10 frames
            ws.send_json({"manos": manos})
            ws.receive_json()
        ws.send_json({"accion": "espacio"})
        assert ws.receive_json()["texto"] == "A"
        ws.send_json({"accion": "retroceder"})
        ws.send_json({"accion": "retroceder"})
        ws.receive_json()
        assert ws.receive_json()["texto"] == ""


def test_estado_de_revision_visible(cliente):
    entrada = cliente.get("/diccionario/agua").json()
    assert entrada["estado"]["gn"] == "IA sin revisar"
    r = cliente.post("/traducir", json={"palabra": "agua", "idioma": "gn"})
    assert r.json()["validado"] is False
    assert Diccionario().traducir("talón", "gn") is None  # error corregido

"""Diccionario LSP → {español paraguayo, portugués de Brasil, guaraní}.

Une cuatro fuentes:
  - data/analysis/diccionario/traducciones.tsv: traducción de cada glosa base
    del Signario SLS (clave = glosa sin número de variante, con la acepción
    tras "|" cuando el signario la indica). La escribió una IA (Claude) y
    NADIE la revisó todavía: es un borrador en los tres idiomas.
  - índice del subconjunto lexico de LSPy-Landmarks v1: las 1810 glosas con
    su video de referencia (cada variante "X 2", "X 3" es una fila propia).
  - data/analysis/vocabulario/vocabulario_objetivo.csv: las 30 señas
    prioritarias; aporta su categoría y su guaraní (también propuesto por la
    IA y pendiente de validación).
  - data/analysis/diccionario/revisiones.tsv: correcciones de personas
    (clave, idioma es/pt/gn, traduccion, revisor, fecha, nota). Pisan a la IA
    y marcan esa celda como "validado".

Cada idioma lleva su estado (es_estado, pt_estado, gn_estado):
"validado", "IA sin revisar" o "pendiente" (celda vacía: sin traducción).

Uso:
    python scripts/preprocess/build_diccionario_lsp.py
"""

import argparse
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIC = ROOT / "data/analysis/diccionario"
LEXICO = ROOT / "data/release/lspy_landmarks_v1/lexico/index.csv"
VOCAB = ROOT / "data/analysis/vocabulario/vocabulario_objetivo.csv"

IDIOMAS = {"es": "es_py", "pt": "pt_br", "gn": "gn"}
COLUMNAS = [
    "sena",
    "clave",
    "variante",
    "acepcion",
    "es_py",
    "es_estado",
    "pt_br",
    "pt_estado",
    "gn",
    "gn_estado",
    "categoria",
    "video_signario",
]


def separar(glosa):
    """'ATROPELLAR   PERSONA' → ('ATROPELLAR', 'PERSONA', 1);
    'ABURRIDO 2' → ('ABURRIDO', '', 2)."""
    partes = re.split(r"\s{2,}", glosa.strip())
    base, acepcion = partes[0].strip(), " ".join(p.strip() for p in partes[1:])
    variante = 1
    m = re.match(r"^(.*?)\s+(\d+)$", base)
    if m:
        base, variante = m.group(1), int(m.group(2))
    return base, acepcion, variante


def cargar_revisiones():
    ruta = DIC / "revisiones.tsv"
    if not ruta.exists():
        return {}
    with open(ruta, encoding="utf-8") as f:
        filas = list(csv.DictReader(f, delimiter="\t"))
    for r in filas:
        if r["idioma"] not in IDIOMAS:
            raise SystemExit(f"revisiones.tsv: idioma inválido {r['idioma']}")
    return {(r["clave"], r["idioma"]): r["traduccion"] for r in filas}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--salida", type=Path, default=DIC / "diccionario_lsp.csv")
    args = ap.parse_args()

    with open(DIC / "traducciones.tsv", encoding="utf-8") as f:
        trad = {r["clave"]: r for r in csv.DictReader(f, delimiter="\t")}
    with open(VOCAB, encoding="utf-8") as f:
        vocab = {r["sena"]: r for r in csv.DictReader(f)}
    revisiones = cargar_revisiones()

    if not LEXICO.exists():
        raise SystemExit(
            f"Falta {LEXICO}: generar antes LSPy-Landmarks v1 "
            "(build_lspy_landmarks_release.py)"
        )
    with open(LEXICO, encoding="utf-8") as f:
        glosas = [(r["etiqueta"], r["url"]) for r in csv.DictReader(f)]

    filas = []
    for glosa, url in glosas:
        base, acepcion, variante = separar(glosa)
        clave = base + (f"|{acepcion}" if acepcion else "")
        t = trad[clave]
        fila = {
            "sena": glosa.strip(),
            "clave": clave,
            "variante": variante,
            "acepcion": acepcion.lower(),
            "es_py": t["es_py"],
            "pt_br": t["pt_br"],
            "gn": t["gn_borrador"],
            "categoria": "",
            "video_signario": url,
        }
        v = vocab.get(base)
        if v and not acepcion:
            fila.update(gn=v["gn_borrador"], categoria=v["categoria"])
        filas.append(fila)

    # Señas prioritarias sin video en el signario
    en_signario = {f["clave"] for f in filas}
    for sena, v in vocab.items():
        if sena not in en_signario:
            filas.append(
                {
                    "sena": sena,
                    "clave": sena,
                    "variante": 1,
                    "acepcion": "",
                    "es_py": v["es"],
                    "pt_br": v["pt"].capitalize(),
                    "gn": v["gn_borrador"],
                    "categoria": v["categoria"],
                    "video_signario": "",
                }
            )

    for f in filas:
        for idioma, col in IDIOMAS.items():
            revisada = revisiones.get((f["clave"], idioma))
            if revisada is not None:
                f[col], estado = revisada, "validado"
            else:
                estado = "IA sin revisar" if f[col] else "pendiente"
            f[f"{idioma}_estado"] = estado
    filas.sort(key=lambda f: (f["clave"], f["variante"]))

    with open(args.salida, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        w.writeheader()
        w.writerows(filas)
    resumen = ", ".join(
        f"{i}: {sum(1 for f in filas if f[i + '_estado'] == 'validado')} "
        f"validadas / {sum(1 for f in filas if f[c])} con texto"
        for i, c in IDIOMAS.items()
    )
    print(
        f"{len(filas)} entradas ({len({f['clave'] for f in filas})} "
        f"conceptos) → {args.salida}\n{resumen}"
    )


if __name__ == "__main__":
    main()

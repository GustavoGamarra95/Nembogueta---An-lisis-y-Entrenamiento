"""
Arma el alfabeto LSPy proxy: para cada letra LSPy toma los datos de la clase
(lengua, letra) que mejor la representa según un mapeo derivado de una fuente
externa (video "Alfabeto Manual" del canal Lengua de Señas LSPy), y agrega
los frames reales de ese video.

Reglas de resolución del mapeo (sobre los candidatos de cada lengua):
  1. Si alguna lengua lee la letra con su propio nombre con probabilidad
     media ≥ --name-threshold (0.35), se usa esa (la de mayor probabilidad).
  2. Si no, la clase más probable aunque tenga otro nombre (A LSPy ← G LSC).
  3. Cada clase de origen se asigna a una sola letra LSPy (primero las que
     coinciden por nombre, luego por probabilidad); las demás quedan solo
     con los frames reales.
Los frames reales van con sujeto "train_canal" (siempre a train) y se
replican hasta --real-min por letra.

Salida: <output-dir>/index.csv (rutas relativas) y mapeo_resuelto.json.

Uso:
    python scripts/preprocess/build_lspy_proxy.py \\
        --mapping data/analysis/lspy_alfabeto/mapeo_desde_canal.json
"""
import argparse
import csv
import json
import os
from pathlib import Path

LANG_DIRS = {
    "LSC": "data/processed/lsc70_alphabet",
    "ASL": "data/processed/asl_combined",
    "LIBRAS": "data/processed/librai_alphabet_v4",
}


def resolve(mapping: dict, name_threshold: float) -> dict:
    wanted = {}
    for letter, m in mapping.items():
        cands = [(p, lang, src) for lang, (src, p) in m["candidatos"].items()]
        same = [c for c in cands if c[2] == letter and c[0] >= name_threshold]
        p, lang, src = max(same) if same else max(cands)
        wanted[letter] = {"lengua": lang, "letra": src, "prob": p,
                          "por_nombre": bool(same)}
    # Asignación única de cada clase de origen
    order = sorted(wanted, key=lambda l: (not wanted[l]["por_nombre"],
                                          -wanted[l]["prob"]))
    taken, final = {}, {}
    for letter in order:
        key = (wanted[letter]["lengua"], wanted[letter]["letra"])
        if key in taken:
            final[letter] = {**wanted[letter], "origen": None,
                             "nota": f"colisión con LSPy {taken[key]}"}
        else:
            taken[key] = letter
            final[letter] = {**wanted[letter], "origen": list(key)}
    return dict(sorted(final.items()))


def source_rows(root: Path, lang: str, letter: str):
    """(ruta, sujeto) de una clase de origen."""
    idx = root / "index.csv"
    if idx.exists():
        with open(idx) as f:
            for r in csv.DictReader(f):
                if r["label"] == letter:
                    yield (root / r["file"]).resolve(), f"{lang}_{r['subject']}"
    else:  # LibrAI sin sujetos: 10 pseudo-sujetos estables
        for i, f in enumerate(sorted((root / letter).glob("*.npy"))):
            yield f.resolve(), f"{lang}_pseudo{i % 10}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mapping", type=Path, required=True)
    p.add_argument("--real-dir", type=Path,
                   default=Path("data/processed/lspy_canal_alfabeto"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("data/processed/lspy_proxy"))
    p.add_argument("--name-threshold", type=float, default=0.35)
    p.add_argument("--real-min", type=int, default=300)
    args = p.parse_args()

    final = resolve(json.load(open(args.mapping)), args.name_threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir.resolve()
    rows = []
    for letter, m in final.items():
        n_src = 0
        if m["origen"]:
            lang, src = m["origen"]
            for path, subj in source_rows(Path(LANG_DIRS[lang]), lang, src):
                rows.append([os.path.relpath(path, out), letter, subj, "proxy"])
                n_src += 1
        real = sorted((args.real_dir / letter).glob("*.npy"))
        reps = max(1, -(-args.real_min // max(len(real), 1)))
        for f in real * reps:
            rows.append([os.path.relpath(f.resolve(), out), letter,
                         "train_canal", "real"])
        m["n_origen"], m["n_real"] = n_src, len(real)

    with open(out / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "subject", "frame"])
        w.writerows(rows)
    json.dump(final, open(out / "mapeo_resuelto.json", "w"), indent=2,
              ensure_ascii=False)

    print(f"{'LSPy':5}{'origen':16}{'prob':>6}  {'n origen':>8}  {'n real':>6}  nota")
    for letter, m in final.items():
        src = ":".join(m["origen"]) if m["origen"] else "— (solo real)"
        print(f"{letter:5}{src:16}{m['prob']:6.2f}  {m['n_origen']:8d}  "
              f"{m['n_real']:6d}  {m.get('nota', '')}")
    print(f"\n{len(rows)} filas → {out / 'index.csv'}")


if __name__ == "__main__":
    main()

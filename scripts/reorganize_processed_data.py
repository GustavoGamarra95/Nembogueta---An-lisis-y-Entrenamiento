"""
Script para reorganizar datos procesados de carpetas por clase a un directorio plano.

Convierte:
    data/processed/v-librasil/
    ├── Clase1/
    │   ├── file1.npy
    │   └── file2.npy
    └── Clase2/
        └── file3.npy

En:
    data/processed/v-librasil-flat/
    ├── file1.npy
    ├── file2.npy
    └── file3.npy

Uso:
    python scripts/reorganize_processed_data.py \
        --input-dir data/processed/v-librasil \
        --output-dir data/processed/v-librasil-flat
"""

import argparse
import shutil
import sys
from pathlib import Path

from tqdm import tqdm


def reorganize_data(input_dir: Path, output_dir: Path, copy: bool = True):
    """
    Reorganiza archivos .npy de subdirectorios a un directorio plano.

    Args:
        input_dir: Directorio con subdirectorios por clase
        output_dir: Directorio de salida plano
        copy: Si True, copia archivos. Si False, los mueve
    """
    print(f"Buscando archivos .npy en {input_dir}...")

    # Buscar todos los .npy recursivamente
    npy_files = list(input_dir.rglob("*.npy"))

    if not npy_files:
        print(f"❌ No se encontraron archivos .npy en {input_dir}")
        return

    print(f"✅ Encontrados {len(npy_files)} archivos")

    # Crear directorio de salida
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copiar o mover archivos
    action = "Copiando" if copy else "Moviendo"
    print(f"\n{action} archivos a {output_dir}...")

    for npy_file in tqdm(npy_files, desc=action):
        dest = output_dir / npy_file.name

        # Si ya existe, agregar sufijo
        if dest.exists():
            counter = 1
            while dest.exists():
                stem = npy_file.stem
                dest = output_dir / f"{stem}_{counter}.npy"
                counter += 1

        if copy:
            shutil.copy2(npy_file, dest)
        else:
            shutil.move(str(npy_file), dest)

    print(f"\n✅ {action} completado!")
    print(f"📁 Archivos en: {output_dir}")

    # Mostrar estadísticas
    final_count = len(list(output_dir.glob("*.npy")))
    print(f"📊 Total de archivos en destino: {final_count}")


def main():
    parser = argparse.ArgumentParser(
        description='Reorganiza archivos .npy de subdirectorios a un directorio plano'
    )

    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Directorio con subdirectorios por clase'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Directorio de salida plano'
    )

    parser.add_argument(
        '--move',
        action='store_true',
        help='Mover archivos en lugar de copiarlos'
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"❌ Error: {args.input_dir} no existe")
        sys.exit(1)

    reorganize_data(args.input_dir, args.output_dir, copy=not args.move)


if __name__ == '__main__':
    main()

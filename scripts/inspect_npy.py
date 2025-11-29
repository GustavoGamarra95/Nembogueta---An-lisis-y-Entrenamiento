"""
Script para inspeccionar archivos .npy de secuencias procesadas.

Uso:
    python scripts/inspect_npy.py <archivo.npy>

Ejemplo:
    python scripts/inspect_npy.py /data/vlibrasil_processed/Abacaxi/Abacaxi_Articulador1.npy
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def inspect_npy(filepath: str):
    """Inspecciona un archivo .npy y muestra información detallada."""

    # Cargar el archivo
    try:
        data = np.load(filepath)
    except Exception as e:
        print(f"Error cargando archivo: {e}")
        return

    # Información básica
    print("=" * 70)
    print(f"ARCHIVO: {Path(filepath).name}")
    print("=" * 70)
    print(f"\n📊 INFORMACIÓN GENERAL:")
    print(f"  - Shape: {data.shape}")
    print(f"  - Dtype: {data.dtype}")
    print(f"  - Tamaño en memoria: {data.nbytes / 1024:.2f} KB")
    print(f"  - Número de elementos: {data.size}")

    if len(data.shape) == 2:
        print(f"\n📹 ESTRUCTURA DE SECUENCIA:")
        print(f"  - Frames: {data.shape[0]}")
        print(f"  - Features por frame: {data.shape[1]}")

        # Determinar tipo de preset según número de features
        feature_dim = data.shape[1]
        if feature_dim == 126:
            preset = "hands (solo manos)"
        elif feature_dim == 225:
            preset = "upper_body (manos + torso)"
        elif feature_dim == 1662:
            preset = "holistic (cuerpo completo + cara)"
        else:
            preset = f"desconocido ({feature_dim} features)"

        print(f"  - Tipo de preset: {preset}")

    # Estadísticas
    print(f"\n📈 ESTADÍSTICAS:")
    print(f"  - Mínimo: {data.min():.6f}")
    print(f"  - Máximo: {data.max():.6f}")
    print(f"  - Media: {data.mean():.6f}")
    print(f"  - Desviación estándar: {data.std():.6f}")

    # Verificar valores nulos o infinitos
    has_nan = np.isnan(data).any()
    has_inf = np.isinf(data).any()

    print(f"\n🔍 VALIDACIÓN:")
    print(f"  - Contiene NaN: {'❌ SÍ' if has_nan else '✅ NO'}")
    print(f"  - Contiene Inf: {'❌ SÍ' if has_inf else '✅ NO'}")

    if has_nan:
        nan_count = np.isnan(data).sum()
        print(f"    ⚠️  {nan_count} valores NaN encontrados")

    if has_inf:
        inf_count = np.isinf(data).sum()
        print(f"    ⚠️  {inf_count} valores Inf encontrados")

    # Mostrar muestra de datos
    print(f"\n🔬 MUESTRA DE DATOS:")
    print(f"  - Primeros 3 frames:")
    if len(data) >= 3:
        for i in range(3):
            print(f"\n    Frame {i}:")
            print(f"      Primeros 10 valores: {data[i][:10]}")
    else:
        print(f"    {data}")

    # Últimos valores
    print(f"\n  - Últimos 3 frames:")
    if len(data) >= 3:
        for i in range(-3, 0):
            print(f"\n    Frame {i}:")
            print(f"      Primeros 10 valores: {data[i][:10]}")

    print("\n" + "=" * 70)
    print("✅ Inspección completada")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Inspecciona archivos .npy de secuencias procesadas'
    )

    parser.add_argument(
        'filepath',
        type=str,
        help='Ruta al archivo .npy'
    )

    parser.add_argument(
        '--show-all',
        action='store_true',
        help='Mostrar todos los datos (no solo muestra)'
    )

    args = parser.parse_args()

    # Verificar que el archivo existe
    if not Path(args.filepath).exists():
        print(f"❌ Error: El archivo no existe: {args.filepath}")
        sys.exit(1)

    # Inspeccionar
    inspect_npy(args.filepath)

    # Si se solicita mostrar todos los datos
    if args.show_all:
        data = np.load(args.filepath)
        print("\n📋 DATOS COMPLETOS:")
        print(data)


if __name__ == '__main__':
    main()

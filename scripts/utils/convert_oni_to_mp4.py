
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

OPENNI2_LIB = "/usr/lib/x86_64-linux-gnu/"


def convert_oni(oni_path: Path, out_path: Path, fps: int = 30) -> bool:

    try:
        from openni import openni2

        dev = openni2.Device.open_file(str(oni_path).encode())
        stream = dev.create_color_stream()
        stream.start()

        playback = dev.playback
        playback.set_speed(-1)  # Lo más rápido posible (sin sincronía real)
        n_frames = playback.get_number_of_frames(stream)

        if n_frames == 0:
            stream.stop()
            dev.close()
            return False

        # Leer primer frame para obtener dimensiones
        first = stream.read_frame()
        h, w = first.height, first.width

        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )

        def _frame_to_bgr(frame) -> np.ndarray:
            data = np.frombuffer(frame.get_buffer_as_uint8(), dtype=np.uint8)
            img = data.reshape(frame.height, frame.width, 3)
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        writer.write(_frame_to_bgr(first))

        for _ in range(n_frames - 1):
            frame = stream.read_frame()
            writer.write(_frame_to_bgr(frame))

        writer.release()
        stream.stop()
        dev.close()
        return True

    except Exception as e:
        print(f"    Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convierte .oni del dataset UFPR a .mp4"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("src/LIBRAS-HC-RGBDS-2011"),
        help="Directorio raíz del dataset .oni",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/ufpr-handshape"),
        help="Directorio de salida para los .mp4",
    )
    parser.add_argument(
        "--openni2-lib",
        default=OPENNI2_LIB,
        help="Directorio que contiene libOpenNI2.so (ej: /usr/lib/x86_64-linux-gnu/)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS del video de salida",
    )
    args = parser.parse_args()

    # Inicializar OpenNI2
    try:
        from openni import openni2
        openni2.initialize(args.openni2_lib)
        print(f"OpenNI2 inicializado desde: {args.openni2_lib}")
    except Exception as e:
        print(f"Error inicializando OpenNI2: {e}")
        print(f"Asegurate de tener instalado: sudo apt install libopenni2-0")
        sys.exit(1)

    if not args.input.exists():
        print(f"Directorio de entrada no encontrado: {args.input}")
        sys.exit(1)

    session_dirs = sorted([d for d in args.input.iterdir() if d.is_dir()])
    print(f"Sesiones encontradas: {len(session_dirs)}")

    total = converted = failed = skipped = 0

    for session_dir in session_dirs:
        oni_files = sorted(session_dir.glob("*.oni"))
        print(f"\n{session_dir.name}: {len(oni_files)} archivos")

        for oni_path in oni_files:
            # Extraer clase del nombre: C1_1_5.oni → clase 5
            try:
                class_idx = int(oni_path.stem.split("_")[-1])
            except ValueError:
                print(f"  Nombre inesperado: {oni_path.name}")
                failed += 1
                continue

            class_dir = args.output / f"class_{class_idx:02d}"
            class_dir.mkdir(parents=True, exist_ok=True)
            out_path = class_dir / f"{oni_path.stem}.mp4"

            total += 1

            if out_path.exists():
                skipped += 1
                continue

            print(f"  Convirtiendo {oni_path.name} → class_{class_idx:02d}/ ...", end=" ")
            ok = convert_oni(oni_path, out_path, fps=args.fps)
            if ok:
                converted += 1
                print("OK")
            else:
                failed += 1
                print("FAIL")

    openni2.unload()

    print(f"\n{'='*50}")
    print(f"Total   : {total}")
    print(f"Convertidos: {converted}")
    print(f"Omitidos   : {skipped}")
    print(f"Fallidos   : {failed}")
    print(f"Salida     : {args.output}")


if __name__ == "__main__":
    main()

"""
Captura imagens PNG de uma letra específica via webcam.
As imagens ficam salvas em: src/LibrAI_extra/{LETRA}/

Após capturar, rodar preprocess_bsl_alphabet.py sobre o novo diretório
para gerar os vetores .npy, depois retreinar.

Uso (dentro do contêiner):
    python scripts/utils/capture_letter_images.py --letter Ç
    python scripts/utils/capture_letter_images.py --letter Ç --samples 200 --camera 1

Controles durante a captura:
    ESPAÇO  — captura o frame atual
    q       — encerra
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

DEFAULT_OUTPUT = Path("/app/src/LibrAI_extra")


def capture(letter: str, n_samples: int, camera_index: int, output_dir: Path):
    letter = letter.upper()
    out_dir = output_dir / letter
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(out_dir.glob("*.png")))
    print(f"\n=== Captura de imagens para letra: {letter} ===")
    print(f"Saída: {out_dir}")
    print(f"Já existem: {existing} imagens")
    print(f"Meta: {n_samples} novas imagens")
    print("\nControles:")
    print("  ESPAÇO → captura o frame atual")
    print("  q      → encerra\n")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERRO: câmera {camera_index} não encontrada")
        sys.exit(1)

    captured = 0
    idx = existing
    last_capture_time = 0
    MIN_INTERVAL = 0.1  # segundos mínimos entre capturas para evitar duplicatas

    while captured < n_samples:
        ret, frame = cap.read()
        if not ret:
            print("ERRO ao ler frame da câmera")
            break

        display = frame.copy()
        progress = f"{captured}/{n_samples}"
        cv2.putText(display, f"Letra: {letter}  [{progress}]",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
        cv2.putText(display, "ESPACO: capturar  |  q: sair",
                    (10, display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1)

        cv2.imshow(f"Captura — {letter}", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            now = time.time()
            if now - last_capture_time >= MIN_INTERVAL:
                filename = out_dir / f"{letter}_{idx:05d}.png"
                cv2.imwrite(str(filename), frame)
                idx += 1
                captured += 1
                last_capture_time = now
                print(f"  [{captured}/{n_samples}] {filename.name}")

    cap.release()
    cv2.destroyAllWindows()

    total = existing + captured
    print(f"\n✓ Captura encerrada: {captured} novas imagens ({total} total em {out_dir})")
    if captured < n_samples:
        print(f"  Atenção: meta era {n_samples}, somente {captured} capturadas")


def main():
    parser = argparse.ArgumentParser(description="Captura PNG de uma letra via webcam")
    parser.add_argument("--letter",  required=True, help="Letra a capturar (ex: Ç)")
    parser.add_argument("--samples", type=int, default=150,
                        help="Número de imagens a capturar (default: 150)")
    parser.add_argument("--camera",  type=int, default=0,
                        help="Índice da câmera (default: 0)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Directorio raíz de salida (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    capture(args.letter, args.samples, args.camera, args.output_dir)


if __name__ == "__main__":
    main()

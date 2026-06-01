import os
import time

import cv2

output_dir = "data/lsp_phrase_videos"
os.makedirs(output_dir, exist_ok=True)

phrases = [
    "acceso a la justicia",
    "derecho a la defensa",
    "igualdad ante la ley",
]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se puede abrir la cámara.")
    exit()

fps = 30
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = 300
video_duration = 10


def record_video(phrase: str, sample_num: int) -> None:
    phrase_filename = phrase.replace(" ", "_")
    output_path = os.path.join(
        output_dir, f"{phrase_filename}_sample_{sample_num}.avi"
    )
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(
        output_path, fourcc, fps, (frame_width, frame_height)
    )

    print(f"Grabando video para la frase '{phrase}' (muestra {sample_num})...")
    print("Prepárate, grabación comienza en 3 segundos...")
    time.sleep(3)

    start_time = time.time()
    frame_counter = 0

    while frame_counter < frame_count:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se puede leer el frame.")
            break

        cv2.imshow("Grabando...", frame)
        out.write(frame)
        frame_counter += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    elapsed_time = time.time() - start_time
    print(
        f"Video grabado: {output_path} "
        f"(Duración: {elapsed_time:.2f} segundos)"
    )

    out.release()
    cv2.destroyAllWindows()


def main():
    try:
        for phrase in phrases:
            for sample_num in range(1, 11):
                print(f"\nPróxima frase: '{phrase}', muestra {sample_num}")
                input("Presiona Enter para comenzar la grabación...")
                record_video(phrase, sample_num)

    finally:
        cap.release()
        print("Recolección de videos completa.")


if __name__ == "__main__":
    main()

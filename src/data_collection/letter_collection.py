import cv2
import os
import time

# Directorio donde se guardarán los videos de letras
output_dir = 'data/lsp_letter_videos'
os.makedirs(output_dir, exist_ok=True)

# Lista de letras (a-z, ñ)
letters = list('abcdefghijklmnopqrstuvwxyz') + ['ñ']

# Configuración de la cámara
cap = cv2.VideoCapture(0)  # 0 para la cámara predeterminada
if not cap.isOpened():
    print("Error: No se puede abrir la cámara.")
    exit()

# Configuración del video
fps = 30
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = 300  # 10 segundos a 30 fps
video_duration = 10  # segundos


# Función para grabar un video
def record_video(letter, sample_num):
    output_path = os.path.join(output_dir, f'{letter}_sample_{sample_num}.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"Grabando video para la letra '{letter}' (muestra {sample_num})...")
    print("Prepárate, grabación comienza en 3 segundos...")
    time.sleep(3)

    start_time = time.time()
    frame_counter = 0

    while frame_counter < frame_count:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se puede leer el frame.")
            break

        # Mostrar el frame en una ventana
        cv2.imshow('Grabando...', frame)
        out.write(frame)
        frame_counter += 1

        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    elapsed_time = time.time() - start_time
    print(f"Video grabado: {output_path} (Duración: {elapsed_time:.2f} segundos)")

    out.release()
    cv2.destroyAllWindows()


# Grabar 10 muestras por letra
for letter in letters:
    for sample_num in range(1, 11):  # 10 muestras por letra
        print(f"\nPróxima letra: '{letter}', muestra {sample_num}")
        input("Presiona Enter para comenzar la grabación...")
        record_video(letter, sample_num)

# Liberar la cámara
cap.release()
print("Recolección de videos completa.")
"""
Script de diagnóstico para probar acceso a la cámara.
"""
import cv2
import sys

print("=" * 60)
print("DIAGNÓSTICO DE CÁMARA")
print("=" * 60)

# Probar diferentes índices de cámara
for camera_idx in range(5):
    print(f"\nProbando cámara {camera_idx}...")
    cap = cv2.VideoCapture(camera_idx)

    if cap.isOpened():
        print(f"  ✓ Cámara {camera_idx} se abrió correctamente")

        # Intentar leer un frame
        ret, frame = cap.read()

        if ret:
            print(f"  ✓ Frame leído exitosamente")
            print(f"    - Dimensiones: {frame.shape}")
            print(f"    - Tipo: {frame.dtype}")

            # Mostrar propiedades de la cámara
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)

            print(f"    - Resolución: {int(width)}x{int(height)}")
            print(f"    - FPS: {fps}")

            # Intentar mostrar el frame
            try:
                cv2.imshow(f"Cámara {camera_idx} - Presiona cualquier tecla", frame)
                print("  ✓ Mostrando frame. Presiona cualquier tecla para continuar...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"  ✗ Error al mostrar frame: {e}")
        else:
            print(f"  ✗ No se pudo leer frame")
            print(f"    - ret = {ret}")

        cap.release()
    else:
        print(f"  ✗ No se pudo abrir cámara {camera_idx}")

print("\n" + "=" * 60)
print("Información de OpenCV:")
print(f"  Versión: {cv2.__version__}")
print(f"  Backends disponibles:")

backends = [
    ('DSHOW', cv2.CAP_DSHOW),
    ('MSMF', cv2.CAP_MSMF),
    ('ANY', cv2.CAP_ANY),
]

for name, backend in backends:
    print(f"    - {name}: {backend}")

print("\nProbando con diferentes backends...")

for name, backend in backends:
    print(f"\nProbando backend {name} (índice 0)...")
    cap = cv2.VideoCapture(0, backend)

    if cap.isOpened():
        print(f"  ✓ Cámara se abrió con backend {name}")
        ret, frame = cap.read()

        if ret:
            print(f"  ✓ Frame leído exitosamente con {name}")
            print(f"    - Dimensiones: {frame.shape}")
        else:
            print(f"  ✗ No se pudo leer frame con {name}")

        cap.release()
    else:
        print(f"  ✗ No se pudo abrir cámara con backend {name}")

print("\n" + "=" * 60)
print("Diagnóstico completado")
print("=" * 60)

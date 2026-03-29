#!/usr/bin/env python
"""Script para verificar la configuración de GPU en Docker."""

import sys


def verify_tensorflow_gpu():
    """Verifica que TensorFlow detecte la GPU."""
    print("=" * 60)
    print("VERIFICACIÓN DE GPU - TENSORFLOW")
    print("=" * 60)

    try:
        import tensorflow as tf
        print(f"✓ TensorFlow versión: {tf.__version__}")

        # Verificar GPUs físicas
        gpus = tf.config.list_physical_devices('GPU')
        print(f"\nGPUs físicas detectadas: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  [{i}] {gpu}")

        # Verificar GPUs lógicas
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"\nGPUs lógicas disponibles: {len(logical_gpus)}")
        for i, gpu in enumerate(logical_gpus):
            print(f"  [{i}] {gpu}")

        # Test simple de operación en GPU
        if len(gpus) > 0:
            print("\n" + "-" * 60)
            print("Ejecutando operación de prueba en GPU...")
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                c = tf.matmul(a, b)
                print(f"✓ Operación completada exitosamente")
                print(f"  Resultado shape: {c.shape}")

        # Verificar configuración de memoria
        print("\n" + "-" * 60)
        print("Configuración de memoria GPU:")
        for gpu in gpus:
            try:
                memory_info = tf.config.experimental.get_memory_info(gpu.name)
                print(f"  {gpu.name}:")
                print(f"    Memoria actual: {memory_info['current'] / 1e9:.2f} GB")
                print(f"    Pico de memoria: {memory_info['peak'] / 1e9:.2f} GB")
            except Exception as e:
                print(f"  No se pudo obtener info de memoria: {e}")

        return len(gpus) > 0

    except ImportError:
        print("✗ TensorFlow no está instalado")
        return False
    except Exception as e:
        print(f"✗ Error verificando TensorFlow: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_cuda():
    """Verifica la configuración de CUDA."""
    print("\n" + "=" * 60)
    print("VERIFICACIÓN DE CUDA")
    print("=" * 60)

    try:
        import tensorflow as tf

        # Verificar si TensorFlow fue compilado con CUDA
        print(f"TensorFlow compilado con CUDA: {tf.test.is_built_with_cuda()}")

        # Obtener información de CUDA
        print(f"CUDA disponible: {tf.test.is_gpu_available(cuda_only=True)}")

        return True
    except Exception as e:
        print(f"✗ Error verificando CUDA: {e}")
        return False


def verify_environment():
    """Verifica las variables de entorno."""
    print("\n" + "=" * 60)
    print("VARIABLES DE ENTORNO")
    print("=" * 60)

    import os

    env_vars = [
        'CUDA_VISIBLE_DEVICES',
        'TF_FORCE_GPU_ALLOW_GROWTH',
        'NVIDIA_VISIBLE_DEVICES',
        'NVIDIA_DRIVER_CAPABILITIES',
    ]

    for var in env_vars:
        value = os.environ.get(var, 'No configurada')
        print(f"  {var}: {value}")


def verify_opencv():
    """Verifica OpenCV."""
    print("\n" + "=" * 60)
    print("VERIFICACIÓN DE OPENCV")
    print("=" * 60)

    try:
        import cv2
        print(f"✓ OpenCV versión: {cv2.__version__}")

        # Verificar soporte CUDA en OpenCV
        cuda_enabled = cv2.cuda.getCudaEnabledDeviceCount() > 0
        print(f"CUDA habilitado en OpenCV: {cuda_enabled}")
        if cuda_enabled:
            print(f"Dispositivos CUDA: {cv2.cuda.getCudaEnabledDeviceCount()}")

        return True
    except ImportError:
        print("✗ OpenCV no está instalado")
        return False
    except Exception as e:
        print(f"⚠ OpenCV instalado pero sin soporte CUDA: {e}")
        return True


def main():
    """Función principal."""
    print("\n")
    print("*" * 60)
    print("VERIFICACIÓN DE CONFIGURACIÓN GPU EN DOCKER")
    print("*" * 60)
    print("\n")

    # Verificaciones
    tf_ok = verify_tensorflow_gpu()
    cuda_ok = verify_cuda()
    verify_environment()
    opencv_ok = verify_opencv()

    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)

    status = []
    status.append(("TensorFlow GPU", "✓" if tf_ok else "✗"))
    status.append(("CUDA", "✓" if cuda_ok else "✗"))
    status.append(("OpenCV", "✓" if opencv_ok else "✗"))

    for name, result in status:
        print(f"  {name}: {result}")

    print("\n" + "=" * 60)

    if tf_ok:
        print("✓ GPU configurada correctamente")
        print("  Puedes comenzar a entrenar modelos con aceleración GPU")
        return 0
    else:
        print("✗ GPU no detectada o configuración incorrecta")

        return 1


if __name__ == "__main__":
    sys.exit(main())

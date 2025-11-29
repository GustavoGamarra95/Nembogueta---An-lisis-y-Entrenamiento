"""
Script para recolectar datos de letras específicas con guía visual.
Útil para aumentar datos de letras problemáticas: U, V, G, L, Y

Uso:
    python scripts/collect_letter_data.py \
        --letter U \
        --output-dir data/raw/alphabet/U \
        --samples 50 \
        --duration 3
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# Descripciones de cómo hacer cada letra
LETTER_DESCRIPTIONS = {
    'U': 'Dedos índice y medio juntos hacia arriba, los demás cerrados',
    'V': 'Dedos índice y medio separados en forma de V, los demás cerrados',
    'G': 'Mano horizontal, dedo índice apuntando, pulgar hacia arriba',
    'L': 'Pulgar e índice en forma de L, los demás cerrados',
    'Y': 'Pulgar y meñique extendidos, los demás cerrados',
    'A': 'Puño cerrado con pulgar al lado',
    'B': 'Mano abierta con dedos juntos, pulgar cruzado',
    'C': 'Mano en forma de C',
    'D': 'Dedo índice arriba, pulgar toca dedos medios',
    'E': 'Dedos doblados sobre pulgar',
    'F': 'Tres dedos arriba, índice y pulgar unidos',
    'H': 'Dedos índice y medio horizontales juntos',
    'I': 'Solo meñique extendido',
    'K': 'Índice arriba, medio horizontal, pulgar toca el medio',
    'M': 'Pulgar bajo tres dedos cerrados',
    'N': 'Pulgar bajo dos dedos cerrados',
    'O': 'Todos los dedos formando un círculo',
    'P': 'Como K pero apuntando hacia abajo',
    'Q': 'Pulgar e índice apuntando hacia abajo',
    'R': 'Índice y medio cruzados',
    'S': 'Puño cerrado con pulgar sobre los dedos',
    'T': 'Pulgar entre índice y medio',
    'W': 'Tres dedos arriba (índice, medio, anular)',
    'X': 'Dedo índice doblado en forma de gancho',
    'Z': 'Dedo índice dibuja una Z en el aire'
}


class LetterDataCollector:
    """Recolector de datos para letras del alfabeto."""

    def __init__(self, letter: str, output_dir: Path, duration: int = 3):
        """
        Inicializa el recolector.

        Args:
            letter: Letra a recolectar
            output_dir: Directorio para guardar videos
            duration: Duración de cada muestra en segundos
        """
        self.letter = letter.upper()
        self.output_dir = output_dir
        self.duration = duration
        self.fps = 30

        # Crear directorio de salida
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Inicializar cámara
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara")

        # Configurar resolución
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Inicializar MediaPipe para visualización
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        logger.info(f"Recolector inicializado para letra: {self.letter}")
        logger.info(f"Directorio de salida: {self.output_dir}")
        logger.info(f"Duración por muestra: {self.duration}s")

    def get_next_filename(self) -> Path:
        """Obtiene el siguiente nombre de archivo disponible."""
        existing_files = list(self.output_dir.glob(f"{self.letter}_*.mp4"))
        if not existing_files:
            idx = 0
        else:
            # Extraer índices existentes
            indices = []
            for f in existing_files:
                try:
                    idx = int(f.stem.split('_')[1])
                    indices.append(idx)
                except (IndexError, ValueError):
                    continue
            idx = max(indices) + 1 if indices else 0

        return self.output_dir / f"{self.letter}_{idx:03d}.mp4"

    def draw_ui(self, frame, state: str, countdown: int = None, sample_num: int = None):
        """
        Dibuja interfaz de usuario en el frame.

        Args:
            frame: Frame de video
            state: Estado actual ('waiting', 'countdown', 'recording')
            countdown: Contador si está en countdown
            sample_num: Número de muestra actual
        """
        h, w, _ = frame.shape

        # Fondo semi-transparente superior
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Título con letra
        title = f"Letra: {self.letter}"
        cv2.putText(frame, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        # Descripción del gesto
        description = LETTER_DESCRIPTIONS.get(self.letter, "Descripción no disponible")
        cv2.putText(frame, description, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Estado
        if state == 'waiting':
            status = "Presiona ESPACIO para comenzar"
            color = (0, 165, 255)
        elif state == 'countdown':
            status = f"Preparate... {countdown}"
            color = (0, 255, 255)
        elif state == 'recording':
            status = "GRABANDO!"
            color = (0, 255, 0)
        else:
            status = state
            color = (255, 255, 255)

        cv2.putText(frame, status, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

        # Número de muestra
        if sample_num is not None:
            sample_text = f"Muestra #{sample_num + 1}"
            cv2.putText(frame, sample_text, (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        # Instrucciones abajo
        instructions = [
            "ESPACIO: Iniciar grabacion",
            "Q: Salir",
            f"Duracion: {self.duration}s por muestra"
        ]

        y_offset = h - 80
        for instruction in instructions:
            cv2.putText(frame, instruction, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25

        return frame

    def record_sample(self, sample_num: int) -> bool:
        """
        Graba una muestra.

        Args:
            sample_num: Número de muestra

        Returns:
            True si se grabó exitosamente, False si se canceló
        """
        # Countdown de 3 segundos
        for countdown in range(3, 0, -1):
            start_time = time.time()
            while time.time() - start_time < 1:
                ret, frame = self.cap.read()
                if not ret:
                    return False

                frame = cv2.flip(frame, 1)

                # Procesar con MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)

                # Dibujar landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )

                # Dibujar UI
                frame = self.draw_ui(frame, 'countdown', countdown, sample_num)

                cv2.imshow('Recolectar Letra', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return False

        # Iniciar grabación
        output_path = self.get_next_filename()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (1280, 720))

        logger.info(f"Grabando muestra {sample_num + 1} en {output_path.name}...")

        start_time = time.time()
        frames_recorded = 0

        while time.time() - start_time < self.duration:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # Procesar con MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            # Dibujar landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )

            # Dibujar UI
            frame = self.draw_ui(frame, 'recording', sample_num=sample_num)

            # Barra de progreso
            progress = (time.time() - start_time) / self.duration
            bar_width = int(1200 * progress)
            cv2.rectangle(frame, (40, 650), (40 + bar_width, 680), (0, 255, 0), -1)
            cv2.rectangle(frame, (40, 650), (1240, 680), (255, 255, 255), 2)

            out.write(frame)
            frames_recorded += 1

            cv2.imshow('Recolectar Letra', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                out.release()
                return False

        out.release()
        logger.info(f"Muestra guardada: {output_path.name} ({frames_recorded} frames)")
        return True

    def collect(self, num_samples: int):
        """
        Recolecta múltiples muestras.

        Args:
            num_samples: Número de muestras a recolectar
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"RECOLECCIÓN DE DATOS - Letra '{self.letter}'")
        logger.info(f"{'='*60}")
        logger.info(f"Descripción: {LETTER_DESCRIPTIONS.get(self.letter, 'N/A')}")
        logger.info(f"Muestras a recolectar: {num_samples}")
        logger.info(f"Presiona ESPACIO para cada muestra, Q para salir")
        logger.info(f"{'='*60}\n")

        samples_collected = 0

        try:
            while samples_collected < num_samples:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)

                # Procesar con MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)

                # Dibujar landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )

                # Dibujar UI
                frame = self.draw_ui(frame, 'waiting', sample_num=samples_collected)

                cv2.imshow('Recolectar Letra', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    # Grabar muestra
                    if self.record_sample(samples_collected):
                        samples_collected += 1
                        logger.info(f"Progreso: {samples_collected}/{num_samples} muestras\n")
                    else:
                        break

                elif key == ord('q'):
                    break

        finally:
            self.cleanup()

        logger.info(f"\n{'='*60}")
        logger.info(f"RECOLECCIÓN COMPLETADA")
        logger.info(f"{'='*60}")
        logger.info(f"Muestras recolectadas: {samples_collected}/{num_samples}")
        logger.info(f"Directorio: {self.output_dir}")

    def cleanup(self):
        """Limpia recursos."""
        logger.info("Limpiando recursos...")
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()


def main():
    parser = argparse.ArgumentParser(
        description='Recolectar datos de letras del alfabeto'
    )

    parser.add_argument(
        '--letter',
        type=str,
        required=True,
        help='Letra a recolectar (A-Z)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directorio para guardar videos'
    )

    parser.add_argument(
        '--samples',
        type=int,
        default=50,
        help='Número de muestras a recolectar (default: 50)'
    )

    parser.add_argument(
        '--duration',
        type=int,
        default=3,
        help='Duración de cada muestra en segundos (default: 3)'
    )

    args = parser.parse_args()

    letter = args.letter.upper()
    if len(letter) != 1 or not letter.isalpha():
        logger.error("La letra debe ser un solo carácter alfabético")
        sys.exit(1)

    output_dir = Path(args.output_dir)

    collector = LetterDataCollector(letter, output_dir, args.duration)
    collector.collect(args.samples)


if __name__ == '__main__':
    main()

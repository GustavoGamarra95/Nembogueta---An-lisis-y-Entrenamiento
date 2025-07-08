import logging
from pathlib import Path
from typing import Optional

import cv2

from src.config.config import Config
from src.utils.validators import VideoData

logger = logging.getLogger(__name__)


class LetterDataCollector:
    def __init__(self):
        self.config = Config()
        self.video_config = self.config.video_config
        self.data_config = self.config.data_config
        self.output_path = Path(self.data_config["video_path"]["letters"])
        self.output_path.mkdir(parents=True, exist_ok=True)

    def collect_video(
        self, letter: str, sample_num: int
    ) -> Optional[VideoData]:
        """
        Recolecta un video para una letra específica.
        """
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logger.error("No se pudo acceder a la cámara")
                return None

            filename = f"letter_{letter}_sample_{sample_num}.mp4"
            output_file = self.output_path / filename

            # Configurar el escritor de video
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = self.video_config.get("fps", 30)
            writer = cv2.VideoWriter(
                str(output_file),
                fourcc,
                fps,
                (int(cap.get(3)), int(cap.get(4))),
            )

            frames = []
            frame_count = 0
            total_frames = fps * self.video_config.get("duration", 10)

            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Error al leer frame de la cámara")
                    break

                frames.append(frame)
                writer.write(frame)
                frame_count += 1

                # Mostrar el frame con información
                cv2.putText(
                    frame,
                    f"Recording letter {letter}: "
                    f"{frame_count}/{total_frames}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Recording", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            cap.release()
            writer.release()
            cv2.destroyAllWindows()

            if frame_count < total_frames:
                logger.warning(f"Grabación incompleta para letra {letter}")
                return None

            return VideoData(
                path=output_file,
                frames=frames,
                label=letter,
                duration=frame_count / fps,
            )

        except Exception as e:
            logger.error(f"Error durante la grabación: {e}")
            return None

    def collect_all_letters(self):
        """Recolecta videos para todas las letras del alfabeto con 'ñ'"""
        letters = list("abcdefghijklmnñopqrstuvwxyz")
        samples_per_letter = self.video_config.get("num_samples", 10)

        for letter in letters:
            logger.info(f"Iniciando recolección para letra: {letter}")
            for sample in range(samples_per_letter):
                logger.info(
                    f"Grabando muestra {sample + 1}/{samples_per_letter}"
                )

                video_data = self.collect_video(letter, sample)
                if video_data and video_data.validate():
                    logger.info(f"Video guardado: {video_data.path}")
                else:
                    logger.error(
                        f"Error en la grabación de letra {letter}, "
                        f"muestra {sample}"
                    )

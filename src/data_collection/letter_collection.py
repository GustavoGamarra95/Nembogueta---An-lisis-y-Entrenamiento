import logging
import os
import sys
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

        os.environ[
            "OPENCV_VIDEOIO_PRIORITY_BACKEND"
        ] = "2"  # V4L2 en Linux

    def _init_camera(self):
        logger.info("Intentando inicializar la cámara...")

        for camera_index in range(2):
            logger.info(f"Probando cámara índice {camera_index}")
            cap = cv2.VideoCapture(camera_index)

            if cap.isOpened():
                logger.info(f"Cámara {camera_index} abierta exitosamente")
                ret, test_frame = cap.read()
                if ret:
                    logger.info(f"Resolución de cámara: {test_frame.shape}")
                    return cap
                else:
                    logger.warning(
                        f"No se pudo leer frame de cámara {camera_index}"
                    )
                    cap.release()
            else:
                logger.warning(f"No se pudo abrir cámara {camera_index}")

        logger.error("No se encontró ninguna cámara funcional")
        return None

    def collect_video(
        self, letter: str, sample_num: int
    ) -> Optional[VideoData]:
        try:
            cap = self._init_camera()
            if cap is None:
                logger.error("No se pudo inicializar la cámara")
                return None

            filename = f"letter_{letter}_sample_{sample_num}.mp4"
            output_file = self.output_path / filename
            logger.info(f"Guardando video en: {output_file}")

            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            fps = self.video_config.get("fps", 30)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(
                f"Configuración de video: "
                f"{frame_width}x{frame_height} @ {fps}fps"
            )
            writer = cv2.VideoWriter(
                str(output_file),
                fourcc,
                fps,
                (frame_width, frame_height),
            )
            frames = []
            frame_count = 0
            total_frames = fps * self.video_config.get("duration", 10)

            logger.info("Iniciando grabación...")
            logger.info(
                f"Presiona 'q' para salir, grabando {total_frames} frames..."
            )

            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Error al leer frame de la cámara")
                    break

                frames.append(frame)
                writer.write(frame)
                frame_count += 1

                info_text = (
                    f"Recording letter {letter}: {frame_count}/{total_frames}"
                )
                cv2.putText(
                    frame,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                try:
                    cv2.imshow("Recording", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("Grabación interrumpida por usuario")
                        break
                except Exception as e:
                    logger.error(f"Error al mostrar frame: {e}")
                    break

            cap.release()
            writer.release()
            cv2.destroyAllWindows()

            if frame_count < total_frames:
                logger.warning(f"Grabación incompleta para letra {letter}")
                return None

            logger.info(
                f"Grabación completada: {frame_count} frames guardados"
            )
            return VideoData(
                path=output_file,
                frames=frames,
                label=letter,
                duration=frame_count / fps,
            )

        except Exception as e:
            logger.error(f"Error durante la grabación: {e}")
            if "cap" in locals():
                cap.release()
            if "writer" in locals():
                writer.release()
            cv2.destroyAllWindows()
            return None

    # Letras estáticas do alfabeto LIBRAS (handshapes sem movimento).
    # H J K X Z são dinâmicas (requerem modelo temporal) — excluídas aqui.
    # Ñ pertence ao espanhol, não ao português/LIBRAS.
    STATIC_LETTERS = list("ABCDEFGILMNOPQRSTUVWY") + ["Ç"]

    def collect_all_letters(self):
        self._collect_letters(self.STATIC_LETTERS)

    def collect_specific_letters(self, letters: list):
        """Coleta apenas as letras informadas."""
        unknown = [ltr for ltr in letters if ltr not in self.STATIC_LETTERS]
        if unknown:
            logger.warning(
                f"Letras fora do alfabeto estático LIBRAS: {unknown}"
            )
        self._collect_letters(letters)

    def _collect_letters(self, letters: list):
        samples_per_letter = self.video_config.get("num_samples", 10)

        logger.info("Iniciando recolección de letras")
        logger.info(f"Letras a grabar: {letters}")
        logger.info(f"Muestras por letra: {samples_per_letter}")

        for letter in letters:
            logger.info(f"=== Iniciando recolección para letra: {letter} ===")
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


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    os.environ["QT_QPA_PLATFORM"] = "xcb"

    parser = argparse.ArgumentParser(
        description="Coleta vídeos do alfabeto LIBRAS"
    )
    parser.add_argument(
        "--letters",
        nargs="+",
        default=None,
        help="Letras específicas a coletar (ex: --letters Ç). "
             "Se omitido, coleta todo o alfabeto estático.",
    )
    args = parser.parse_args()

    logger.info("=== Iniciando programa de recolección de letras ===")
    logger.info(f"OpenCV version: {cv2.__version__}")

    try:
        collector = LetterDataCollector()
        if args.letters:
            collector.collect_specific_letters(args.letters)
        else:
            collector.collect_all_letters()
    except KeyboardInterrupt:
        logger.info("Programa interrumpido por el usuario")
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
    finally:
        cv2.destroyAllWindows()
        logger.info("=== Programa finalizado ===")

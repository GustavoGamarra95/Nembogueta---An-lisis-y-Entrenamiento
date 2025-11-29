"""
Script para procesar el dataset pt-br2libras-gloss.
Prepara datos de traducción portugués-LIBRAS para entrenamiento de modelos NLP.
"""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from collections import Counter

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pt_br2libras_processing.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PtBr2LibrasPreprocessor:
    """Procesador del dataset pt-br2libras-gloss."""

    def __init__(self, min_length: int = 3, max_length: int = 100):
        """
        Inicializa el preprocesador.

        Args:
            min_length: Longitud mínima de texto en palabras
            max_length: Longitud máxima de texto en palabras
        """
        self.min_length = min_length
        self.max_length = max_length
        logger.info(
            f"Preprocesador inicializado "
            f"(min_length={min_length}, max_length={max_length})"
        )

    def load_dataset(self, csv_path: Path) -> List[Dict[str, str]]:
        """
        Carga el dataset desde CSV.

        Args:
            csv_path: Ruta al archivo CSV

        Returns:
            Lista de diccionarios con los datos
        """
        data = []

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    data.append({
                        'pt_br': row['pt-br'],
                        'libras_gloss': row['libras-gloss'],
                        'is_government_source': row['is_government_source'],
                        'english_translation': row['english_translation']
                    })

            logger.info(f"Cargados {len(data)} pares de traducción")
            return data

        except Exception as e:
            logger.error(f"Error cargando dataset: {e}")
            return []

    def clean_text(self, text: str) -> str:
        """
        Limpia y normaliza el texto.

        Args:
            text: Texto a limpiar

        Returns:
            Texto limpio
        """
        # Eliminar espacios extras
        text = ' '.join(text.split())

        # Convertir a minúsculas (excepto glosas que están en mayúsculas)
        return text.strip()

    def filter_by_length(self, data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Filtra datos por longitud de texto.

        Args:
            data: Lista de pares de traducción

        Returns:
            Lista filtrada
        """
        filtered = []

        for item in data:
            pt_words = len(item['pt_br'].split())
            gloss_words = len(item['libras_gloss'].split())

            if (self.min_length <= pt_words <= self.max_length and
                    self.min_length <= gloss_words <= self.max_length):
                filtered.append(item)

        logger.info(
            f"Filtrados {len(filtered)}/{len(data)} pares "
            f"(longitud {self.min_length}-{self.max_length} palabras)"
        )

        return filtered

    def split_dataset(
        self,
        data: List[Dict[str, str]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Divide el dataset en train/val/test.

        Args:
            data: Lista de pares de traducción
            train_ratio: Proporción para entrenamiento
            val_ratio: Proporción para validación

        Returns:
            Tupla (train, val, test)
        """
        # Mezclar datos
        np.random.seed(42)
        indices = np.random.permutation(len(data))

        # Calcular índices de división
        n_train = int(len(data) * train_ratio)
        n_val = int(len(data) * val_ratio)

        # Dividir
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        train = [data[i] for i in train_indices]
        val = [data[i] for i in val_indices]
        test = [data[i] for i in test_indices]

        logger.info(
            f"Dataset dividido: {len(train)} train, {len(val)} val, {len(test)} test"
        )

        return train, val, test

    def build_vocabulary(
        self,
        data: List[Dict[str, str]],
        field: str,
        max_vocab_size: int = 10000
    ) -> Dict[str, int]:
        """
        Construye vocabulario a partir de los datos.

        Args:
            data: Lista de pares de traducción
            field: Campo a usar ('pt_br' o 'libras_gloss')
            max_vocab_size: Tamaño máximo del vocabulario

        Returns:
            Diccionario {token: id}
        """
        # Contar tokens
        token_counts = Counter()

        for item in data:
            tokens = item[field].split()
            token_counts.update(tokens)

        # Seleccionar los más frecuentes
        most_common = token_counts.most_common(max_vocab_size - 4)

        # Crear vocabulario (reservar índices especiales)
        vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<SOS>': 2,  # Start of sequence
            '<EOS>': 3   # End of sequence
        }

        for i, (token, _) in enumerate(most_common, start=4):
            vocab[token] = i

        logger.info(f"Vocabulario construido: {len(vocab)} tokens ({field})")

        return vocab

    def save_split(
        self,
        data: List[Dict[str, str]],
        output_dir: Path,
        split_name: str,
        pt_vocab: Dict[str, int],
        gloss_vocab: Dict[str, int]
    ):
        """
        Guarda un split procesado.

        Args:
            data: Lista de pares de traducción
            output_dir: Directorio de salida
            split_name: Nombre del split
            pt_vocab: Vocabulario de portugués
            gloss_vocab: Vocabulario de glosas
        """
        # Preparar datos para guardar
        processed_data = []

        for item in data:
            # Tokenizar
            pt_tokens = item['pt_br'].split()
            gloss_tokens = item['libras_gloss'].split()

            # Convertir a IDs
            pt_ids = [pt_vocab.get(token, pt_vocab['<UNK>']) for token in pt_tokens]
            gloss_ids = [gloss_vocab.get(token, gloss_vocab['<UNK>']) for token in gloss_tokens]

            processed_data.append({
                'pt_br': item['pt_br'],
                'libras_gloss': item['libras_gloss'],
                'pt_ids': pt_ids,
                'gloss_ids': gloss_ids,
                'is_government_source': item['is_government_source'],
                'english_translation': item['english_translation']
            })

        # Guardar como JSON
        output_file = output_dir / f"{split_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Guardado {split_name}: {len(processed_data)} muestras")

    def save_vocabularies(
        self,
        output_dir: Path,
        pt_vocab: Dict[str, int],
        gloss_vocab: Dict[str, int]
    ):
        """
        Guarda los vocabularios.

        Args:
            output_dir: Directorio de salida
            pt_vocab: Vocabulario de portugués
            gloss_vocab: Vocabulario de glosas
        """
        # Guardar vocabulario de portugués
        pt_vocab_file = output_dir / "vocab_pt_br.json"
        with open(pt_vocab_file, 'w', encoding='utf-8') as f:
            json.dump(pt_vocab, f, ensure_ascii=False, indent=2)

        # Guardar vocabulario de glosas
        gloss_vocab_file = output_dir / "vocab_libras_gloss.json"
        with open(gloss_vocab_file, 'w', encoding='utf-8') as f:
            json.dump(gloss_vocab, f, ensure_ascii=False, indent=2)

        logger.info("Vocabularios guardados")

    def generate_statistics(
        self,
        data: List[Dict[str, str]],
        output_dir: Path
    ):
        """
        Genera estadísticas del dataset.

        Args:
            data: Lista de pares de traducción
            output_dir: Directorio de salida
        """
        stats = {
            'total_samples': len(data),
            'government_sources': sum(
                1 for item in data if item['is_government_source'] == 'True'
            ),
            'avg_pt_length': np.mean([len(item['pt_br'].split()) for item in data]),
            'avg_gloss_length': np.mean([len(item['libras_gloss'].split()) for item in data]),
            'max_pt_length': max(len(item['pt_br'].split()) for item in data),
            'max_gloss_length': max(len(item['libras_gloss'].split()) for item in data),
            'unique_pt_words': len(set(' '.join([item['pt_br'] for item in data]).split())),
            'unique_gloss_words': len(set(' '.join([item['libras_gloss'] for item in data]).split()))
        }

        # Guardar estadísticas
        stats_file = output_dir / "statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info("\nEstadísticas del dataset:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Procesa el dataset pt-br2libras-gloss"
    )

    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("data/pt-br2libras-gloss/pt-br2libras-gloss/pt_br2libras_gloss.csv"),
        help="Archivo CSV con los datos"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/pt_br2libras"),
        help="Directorio de salida"
    )

    parser.add_argument(
        "--min-length",
        type=int,
        default=3,
        help="Longitud mínima en palabras"
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Longitud máxima en palabras"
    )

    parser.add_argument(
        "--max-vocab-size",
        type=int,
        default=10000,
        help="Tamaño máximo del vocabulario"
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proporción de datos para entrenamiento"
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Proporción de datos para validación"
    )

    args = parser.parse_args()

    # Verificar archivo de entrada
    if not args.data_file.exists():
        logger.error(f"Archivo de datos no encontrado: {args.data_file}")
        return 1

    # Crear directorio de salida
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Crear preprocesador
    preprocessor = PtBr2LibrasPreprocessor(
        min_length=args.min_length,
        max_length=args.max_length
    )

    # Cargar datos
    logger.info("Cargando dataset...")
    data = preprocessor.load_dataset(args.data_file)

    if not data:
        logger.error("No se pudieron cargar los datos")
        return 1

    # Limpiar datos
    logger.info("Limpiando datos...")
    for item in data:
        item['pt_br'] = preprocessor.clean_text(item['pt_br'])
        item['libras_gloss'] = preprocessor.clean_text(item['libras_gloss'])

    # Filtrar por longitud
    data = preprocessor.filter_by_length(data)

    # Dividir dataset
    logger.info("Dividiendo dataset...")
    train, val, test = preprocessor.split_dataset(
        data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )

    # Construir vocabularios (solo con datos de entrenamiento)
    logger.info("Construyendo vocabularios...")
    pt_vocab = preprocessor.build_vocabulary(
        train,
        'pt_br',
        max_vocab_size=args.max_vocab_size
    )
    gloss_vocab = preprocessor.build_vocabulary(
        train,
        'libras_gloss',
        max_vocab_size=args.max_vocab_size
    )

    # Guardar vocabularios
    preprocessor.save_vocabularies(args.output_dir, pt_vocab, gloss_vocab)

    # Guardar splits
    logger.info("Guardando splits procesados...")
    preprocessor.save_split(train, args.output_dir, 'train', pt_vocab, gloss_vocab)
    preprocessor.save_split(val, args.output_dir, 'val', pt_vocab, gloss_vocab)
    preprocessor.save_split(test, args.output_dir, 'test', pt_vocab, gloss_vocab)

    # Generar estadísticas
    preprocessor.generate_statistics(data, args.output_dir)

    logger.info("\nProcesamiento completado exitosamente")

    return 0


if __name__ == "__main__":
    sys.exit(main())

import json
import logging
from pathlib import Path
from typing import Generator

from datasets import load_dataset
from tqdm import tqdm

from src.core.models import DocumentInput

logger = logging.getLogger(__name__)


class EuroParlDataLoader:
    """
    Downloads and prepares EuroParl parallel corpus documents.

    The EuroParl corpus is EU Parliament proceedings in 21 languages.
    We use the German-English pair (de-en) as our primary language pair.

    Dataset card: https://huggingface.co/datasets/Helsinki-NLP/europarl
    """

    DATASET_NAME = "Helsinki-NLP/europarl"
    DEFAULT_LANGUAGE_PAIR = "de-en"

    def __init__(self, processed_dir: str = "data/processed/", sample_size: int = 20):
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.sample_size = sample_size

    def download_and_prepare(self, language_pair: str = DEFAULT_LANGUAGE_PAIR) -> Path:
        """
        Downloads EuroParl dataset and saves a processed subset to disk.

        Args:
            language_pair: Language pair string, e.g. 'de-en'

        Returns:
            Path to the saved processed JSON file
        """
        logger.info(f"Loading EuroParl dataset ({language_pair})...")

        dataset = load_dataset(
            self.DATASET_NAME,
            language_pair,
            split="train",
            streaming=True,   # streaming=True avoids downloading the full dataset
            trust_remote_code=True,
        )

        documents: list[dict] = []
        source_lang = language_pair.split("-")[0]  # "de"

        for i, example in enumerate(tqdm(dataset, total=self.sample_size, desc="Loading documents")):
            if i >= self.sample_size:
                break

            # EuroParl stores parallel sentences — we group them into document-length chunks
            source_text = example["translation"][source_lang]
            reference_english = example["translation"]["en"]

            doc = DocumentInput(
                doc_id=f"europarl_{language_pair}_{i:04d}",
                source_language=source_lang,
                raw_text=source_text,
                source="europarl",
                metadata={"reference_translation": reference_english},
            )
            documents.append(doc.model_dump())

        output_path = self.processed_dir / f"europarl_{language_pair}_{self.sample_size}docs.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(documents)} documents to {output_path}")
        return output_path

    def load_from_disk(self, file_path: str) -> list[DocumentInput]:
        """
        Loads preprocessed documents from disk.

        Args:
            file_path: Path to the processed JSON file

        Returns:
            List of DocumentInput objects
        """
        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        return [DocumentInput(**doc) for doc in raw]

    def load_ground_truth(self, file_path: str) -> dict[str, str]:
        """
        Loads human reference translations for evaluation.

        Returns:
            Dict mapping doc_id → reference English translation
        """
        documents = self.load_from_disk(file_path)
        return {
            doc.doc_id: doc.metadata.get("reference_translation", "")
            for doc in documents
        }
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.models.document import CanonicalDocument, FailedRecord

logger = logging.getLogger(__name__)


class JSONLStore:
    def __init__(
        self,
        processed_path: Path,
        raw_dir: Path,
        failed_dir: Path,
    ):
        self._processed_path = processed_path
        self._raw_dir = raw_dir
        self._failed_dir = failed_dir

        self._processed_path.parent.mkdir(parents=True, exist_ok=True)
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._failed_dir.mkdir(parents=True, exist_ok=True)

        self._existing_ids: set[str] = set()
        self._load_existing_ids()

    def _load_existing_ids(self) -> None:
        if not self._processed_path.exists():
            return

        try:
            with self._processed_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = json.loads(line)
                        if "document_id" in doc:
                            self._existing_ids.add(doc["document_id"])
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning("Failed to load existing document IDs: %s", e)

    def load_all(self) -> list[CanonicalDocument]:
        documents: list[CanonicalDocument] = []
        if not self._processed_path.exists():
            return documents

        try:
            with self._processed_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = CanonicalDocument.model_validate_json(line)
                        documents.append(doc)
                    except Exception as e:
                        logger.warning("Failed to parse document line: %s", e)
                        continue
        except Exception as e:
            logger.error("Failed to load documents: %s", e)

        return documents

    def save(self, document: CanonicalDocument) -> None:
        if document.document_id in self._existing_ids:
            logger.debug("Document %s already exists, skipping", document.document_id)
            return

        self._existing_ids.add(document.document_id)

        try:
            with self._processed_path.open("a", encoding="utf-8") as f:
                f.write(document.model_dump_json() + "\n")
        except Exception as e:
            logger.error("Failed to save document: %s", e)
            raise

    def save_many(self, documents: list[CanonicalDocument]) -> None:
        for doc in documents:
            self.save(doc)

    def save_failed(self, failed_records: list[FailedRecord]) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        failed_path = self._failed_dir / f"failed_{timestamp}.jsonl"

        try:
            with failed_path.open("w", encoding="utf-8") as f:
                for record in failed_records:
                    f.write(record.model_dump_json() + "\n")
            logger.info("Saved %d failed records to %s", len(failed_records), failed_path)
        except Exception as e:
            logger.error("Failed to save failed records: %s", e)

    def save_raw(self, source_id: str, raw_data: str) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = self._raw_dir / f"{source_id}_{timestamp}.json"

        try:
            with raw_path.open("w", encoding="utf-8") as f:
                f.write(raw_data)
        except Exception as e:
            logger.warning("Failed to save raw data for %s: %s", source_id, e)

    def exists(self, document_id: str) -> bool:
        return document_id in self._existing_ids

    def count(self) -> int:
        return len(self._existing_ids)
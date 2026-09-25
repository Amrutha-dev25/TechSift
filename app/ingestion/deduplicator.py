import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Optional

from app.models.document import CanonicalDocument, RawDocument

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationResult:
    is_duplicate: bool
    existing_doc_id: Optional[str] = None
    match_type: Optional[str] = None


class Deduplicator:
    def __init__(
        self,
        near_duplicate_threshold: float = 0.9,
        min_text_length: int = 100,
    ):
        self._seen_hashes: set[str] = set()
        self._seen_canonical_urls: set[str] = set()
        self._near_duplicate_fingerprints: dict[str, str] = {}
        self._near_duplicate_threshold = near_duplicate_threshold
        self._min_text_length = min_text_length

    def compute_document_id(self, canonical_url: str) -> str:
        return hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:16]

    def compute_content_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _normalize_for_fingerprint(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _generate_fingerprint(self, title: str, content: str) -> str:
        combined = f"{title} {content}"
        normalized = self._normalize_for_fingerprint(combined)
        if len(normalized) < self._min_text_length:
            return ""
        words = normalized.split()
        shingles = set()
        for i in range(len(words) - 4):
            shingle = " ".join(words[i:i+5])
            shingles.add(shingle)
        return hashlib.md5("|".join(sorted(shingles)).encode()).hexdigest()

    def _jaccard_similarity(self, fp1: str, fp2: str) -> float:
        if not fp1 or not fp2:
            return 0.0
        return 1.0 if fp1 == fp2 else 0.0

    def check_duplicate(
        self,
        doc: RawDocument,
        canonical_url: str,
        content: str,
    ) -> DeduplicationResult:
        doc_id = self.compute_document_id(canonical_url)
        content_hash = self.compute_content_hash(content)

        if doc_id in self._seen_canonical_urls:
            logger.debug("Exact duplicate detected (canonical URL): %s", canonical_url)
            return DeduplicationResult(
                is_duplicate=True,
                existing_doc_id=doc_id,
                match_type="exact_url",
            )

        if content_hash in self._seen_hashes:
            logger.debug("Exact duplicate detected (content hash): %s", canonical_url)
            return DeduplicationResult(
                is_duplicate=True,
                existing_doc_id=doc_id,
                match_type="exact_content",
            )

        fingerprint = self._generate_fingerprint(doc.title, content)
        if fingerprint:
            for existing_fp, existing_id in self._near_duplicate_fingerprints.items():
                similarity = self._jaccard_similarity(fingerprint, existing_fp)
                if similarity >= self._near_duplicate_threshold:
                    logger.debug(
                        "Near duplicate detected: %s (similarity=%.2f with %s)",
                        canonical_url,
                        similarity,
                        existing_id,
                    )
                    return DeduplicationResult(
                        is_duplicate=True,
                        existing_doc_id=existing_id,
                        match_type="near_duplicate",
                    )

        self._seen_canonical_urls.add(doc_id)
        self._seen_hashes.add(content_hash)
        if fingerprint:
            self._near_duplicate_fingerprints[fingerprint] = doc_id

        return DeduplicationResult(is_duplicate=False)

    def mark_seen(self, doc: CanonicalDocument) -> None:
        self._seen_canonical_urls.add(doc.document_id)
        self._seen_hashes.add(doc.content_hash)
        fingerprint = self._generate_fingerprint(doc.title, doc.content)
        if fingerprint:
            self._near_duplicate_fingerprints[fingerprint] = doc.document_id

    def load_existing(self, documents: list[CanonicalDocument]) -> None:
        for doc in documents:
            self.mark_seen(doc)
        logger.info("Loaded %d existing documents into deduplicator", len(documents))
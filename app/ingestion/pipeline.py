import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from app.config.settings import RSSSourceConfig, settings
from app.ingestion.base import DataSource
from app.ingestion.cleaner import Cleaner
from app.ingestion.deduplicator import Deduplicator, DeduplicationResult
from app.ingestion.normalizer import Normalizer
from app.ingestion.rss import RSSSource
from app.models.document import CanonicalDocument, FailedRecord, RawDocument, SourceType
from app.storage.jsonl_store import JSONLStore

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    feeds_attempted: int = 0
    feeds_succeeded: int = 0
    feeds_failed: int = 0
    entries_seen: int = 0
    documents_accepted: int = 0
    documents_rejected: int = 0
    duplicates: int = 0
    failed_feeds: list[dict] = field(default_factory=list)
    failed_records: list[FailedRecord] = field(default_factory=list)


class IngestionPipeline:
    def __init__(
        self,
        max_articles_per_feed: Optional[int] = None,
        min_content_length: int = 50,
    ):
        self._max_articles_per_feed = max_articles_per_feed or settings.max_articles_per_feed
        self._normalizer = Normalizer(min_content_length=min_content_length)
        self._cleaner = Cleaner()
        self._deduplicator = Deduplicator()
        self._store = JSONLStore(
            processed_path=settings.processed_data_dir / "documents.jsonl",
            raw_dir=settings.raw_data_dir,
            failed_dir=settings.failed_data_dir,
        )

    def _create_source(self, config: RSSSourceConfig) -> DataSource[RawDocument]:
        return RSSSource(
            source_id=config.source_id,
            source_name=config.source_name,
            feed_url=config.feed_url,
            max_articles=self._max_articles_per_feed,
        )

    def _process_raw_document(self, raw_doc: RawDocument) -> Optional[CanonicalDocument]:
        normalized = self._normalizer.normalize(raw_doc)

        is_valid, reason = self._normalizer.is_valid(normalized)
        if not is_valid:
            logger.warning(
                "Document rejected: %s (source=%s, url=%s)",
                reason,
                normalized.source_id,
                normalized.url,
            )
            return None

        canonical_url = self._normalizer.get_canonical_url(normalized.url)
        content = normalized.content or normalized.summary or ""
        cleaned_content = self._cleaner.clean(content)

        if not cleaned_content or len(cleaned_content.strip()) < self._normalizer.min_content_length:
            logger.warning(
                "Document rejected after cleaning: content_too_short (source=%s, url=%s)",
                normalized.source_id,
                normalized.url,
            )
            return None

        content_hash = self._deduplicator.compute_content_hash(cleaned_content)
        document_id = self._deduplicator.compute_document_id(canonical_url)

        canonical_doc = CanonicalDocument(
            document_id=document_id,
            source_id=normalized.source_id,
            source_name=normalized.source_name,
            source_type=normalized.source_type,
            title=normalized.title,
            content=cleaned_content,
            url=normalized.url,
            author=normalized.author,
            published_at=normalized.published_at,
            retrieved_at=normalized.retrieved_at,
            content_hash=content_hash,
            canonical_url=canonical_url,
        )

        return canonical_doc

    def run(self, source_configs: Optional[list[RSSSourceConfig]] = None) -> IngestionResult:
        if source_configs is None:
            source_configs = [s for s in settings.rss_sources if s.enabled]

        existing_docs = self._store.load_all()
        self._deduplicator.load_existing(existing_docs)

        result = IngestionResult()
        result.feeds_attempted = len(source_configs)

        for config in source_configs:
            source = self._create_source(config)
            try:
                logger.info("Processing source: %s", config.source_id)
                raw_documents = source.fetch()
                result.entries_seen += len(raw_documents)

                for raw_doc in raw_documents:
                    canonical_doc = self._process_raw_document(raw_doc)
                    if canonical_doc is None:
                        result.documents_rejected += 1
                        failed_record = FailedRecord(
                            timestamp=datetime.now(timezone.utc),
                            source_id=raw_doc.source_id,
                            source_name=raw_doc.source_name,
                            source_type=raw_doc.source_type,
                            url=raw_doc.url,
                            reason="validation_failed",
                            error_type="ValidationError",
                        )
                        result.failed_records.append(failed_record)
                        continue

                    dup_result = self._deduplicator.check_duplicate(
                        raw_doc, canonical_doc.canonical_url, canonical_doc.content
                    )

                    if dup_result.is_duplicate:
                        result.duplicates += 1
                        logger.debug(
                            "Duplicate skipped: %s (type=%s)",
                            canonical_doc.canonical_url,
                            dup_result.match_type,
                        )
                        continue

                    self._deduplicator.mark_seen(canonical_doc)
                    self._store.save(canonical_doc)
                    result.documents_accepted += 1

                result.feeds_succeeded += 1
                logger.info("Source %s completed successfully", config.source_id)

            except Exception as e:
                result.feeds_failed += 1
                error_info = {
                    "source_id": config.source_id,
                    "source_name": config.source_name,
                    "feed_url": config.feed_url,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
                result.failed_feeds.append(error_info)
                logger.error("Source %s failed: %s", config.source_id, e)

                failed_record = FailedRecord(
                    timestamp=datetime.now(timezone.utc),
                    source_id=config.source_id,
                    source_name=config.source_name,
                    source_type=SourceType.RSS,
                    url=config.feed_url,
                    reason=str(e),
                    error_type=type(e).__name__,
                )
                result.failed_records.append(failed_record)

        if result.failed_records:
            self._store.save_failed(result.failed_records)

        logger.info(
            "Ingestion complete: feeds=%d/%d, accepted=%d, rejected=%d, duplicates=%d",
            result.feeds_succeeded,
            result.feeds_attempted,
            result.documents_accepted,
            result.documents_rejected,
            result.duplicates,
        )

        return result
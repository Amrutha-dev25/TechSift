import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from app.config.settings import RSSSourceConfig
from app.ingestion.pipeline import IngestionPipeline
from app.models.document import CanonicalDocument, RawDocument, SourceType
from app.storage.jsonl_store import JSONLStore


class TestIngestionPipeline:
    def setup_method(self):
        self.pipeline = IngestionPipeline(max_articles_per_feed=5, min_content_length=10)
        self.test_config = RSSSourceConfig(
            source_id="test_source",
            source_name="Test Source",
            feed_url="https://example.com/feed.xml",
            enabled=True,
        )

    def test_process_raw_document_valid(self):
        raw_doc = RawDocument(
            source_id="test",
            source_name="Test",
            source_type=SourceType.RSS,
            title="Test Title",
            url="https://example.com/article",
            content="This is sufficient content for the document to be valid",
            retrieved_at=datetime.now(timezone.utc),
        )

        canonical = self.pipeline._process_raw_document(raw_doc)

        assert canonical is not None
        assert isinstance(canonical, CanonicalDocument)
        assert canonical.title == "Test Title"
        assert canonical.url == "https://example.com/article"

    def test_process_raw_document_short_content_rejected(self):
        raw_doc = RawDocument(
            source_id="test",
            source_name="Test",
            source_type=SourceType.RSS,
            title="Test",
            url="https://example.com/article",
            content="Short",
            retrieved_at=datetime.now(timezone.utc),
        )

        canonical = self.pipeline._process_raw_document(raw_doc)
        assert canonical is None

    def test_process_raw_document_html_cleaned(self):
        raw_doc = RawDocument(
            source_id="test",
            source_name="Test",
            source_type=SourceType.RSS,
            title="Test",
            url="https://example.com/article",
            content="<p>Content with <b>HTML</b> tags that is long enough to pass validation</p>",
            retrieved_at=datetime.now(timezone.utc),
        )

        canonical = self.pipeline._process_raw_document(raw_doc)

        assert canonical is not None
        assert "<p>" not in canonical.content
        assert "<b>" not in canonical.content
        assert "Content with HTML tags" in canonical.content

    @patch("app.ingestion.pipeline.JSONLStore")
    def test_run_single_feed_success(self, mock_store_class):
        mock_store = Mock(spec=JSONLStore)
        mock_store.load_all.return_value = []
        mock_store_class.return_value = mock_store

        pipeline = IngestionPipeline(max_articles_per_feed=5, min_content_length=10)

        mock_source = Mock()
        mock_source.fetch.return_value = [
            RawDocument(
                source_id="test",
                source_name="Test",
                source_type=SourceType.RSS,
                title="Test Article",
                url="https://example.com/1",
                content="Valid content that is long enough to pass validation",
                retrieved_at=datetime.now(timezone.utc),
            )
        ]

        with patch.object(pipeline, "_create_source", return_value=mock_source):
            result = pipeline.run([self.test_config])

        assert result.feeds_attempted == 1
        assert result.feeds_succeeded == 1
        assert result.feeds_failed == 0
        assert result.entries_seen == 1
        assert result.documents_accepted == 1
        assert result.documents_rejected == 0
        assert result.duplicates == 0

    @patch("app.ingestion.pipeline.JSONLStore")
    def test_run_duplicate_documents(self, mock_store_class):
        mock_store = Mock(spec=JSONLStore)

        canonical_url = "https://example.com/existing"
        content = "Existing content that is long enough"
        doc_id = self.pipeline._deduplicator.compute_document_id(canonical_url)
        content_hash = self.pipeline._deduplicator.compute_content_hash(content)

        existing_doc = CanonicalDocument(
            document_id=doc_id,
            source_id="test",
            source_name="Test",
            source_type=SourceType.RSS,
            title="Existing",
            content=content,
            url=canonical_url,
            canonical_url=canonical_url,
            content_hash=content_hash,
            retrieved_at=datetime.now(timezone.utc),
        )
        mock_store.load_all.return_value = [existing_doc]
        mock_store_class.return_value = mock_store

        pipeline = IngestionPipeline(max_articles_per_feed=5, min_content_length=10)

        mock_source = Mock()
        mock_source.fetch.return_value = [
            RawDocument(
                source_id="test",
                source_name="Test",
                source_type=SourceType.RSS,
                title="Existing",
                url=canonical_url,
                content=content,
                retrieved_at=datetime.now(timezone.utc),
            )
        ]

        with patch.object(pipeline, "_create_source", return_value=mock_source):
            result = pipeline.run([self.test_config])

        assert result.documents_accepted == 0
        assert result.duplicates == 1

    @patch("app.ingestion.pipeline.JSONLStore")
    def test_run_feed_failure_isolated(self, mock_store_class):
        mock_store = Mock(spec=JSONLStore)
        mock_store.load_all.return_value = []
        mock_store_class.return_value = mock_store

        pipeline = IngestionPipeline(max_articles_per_feed=5, min_content_length=10)

        mock_source = Mock()
        mock_source.fetch.side_effect = Exception("Feed error")

        with patch.object(pipeline, "_create_source", return_value=mock_source):
            result = pipeline.run([self.test_config])

        assert result.feeds_attempted == 1
        assert result.feeds_succeeded == 0
        assert result.feeds_failed == 1
        assert len(result.failed_feeds) == 1

    @patch("app.ingestion.pipeline.JSONLStore")
    def test_run_multiple_feeds_one_fails(self, mock_store_class):
        mock_store = Mock(spec=JSONLStore)
        mock_store.load_all.return_value = []
        mock_store_class.return_value = mock_store

        pipeline = IngestionPipeline(max_articles_per_feed=5, min_content_length=10)

        configs = [
            RSSSourceConfig(source_id="feed1", source_name="Feed 1", feed_url="url1", enabled=True),
            RSSSourceConfig(source_id="feed2", source_name="Feed 2", feed_url="url2", enabled=True),
            RSSSourceConfig(source_id="feed3", source_name="Feed 3", feed_url="url3", enabled=True),
        ]

        def create_source(config):
            source = Mock()
            if config.source_id == "feed2":
                source.fetch.side_effect = Exception("Feed error")
            else:
                source.fetch.return_value = [
                    RawDocument(
                        source_id=config.source_id,
                        source_name=config.source_name,
                        source_type=SourceType.RSS,
                        title=f"Test {config.source_id}",
                        url=f"https://example.com/{config.source_id}",
                        content=f"Valid content for {config.source_id} that is long enough to pass validation",
                        retrieved_at=datetime.now(timezone.utc),
                    )
                ]
            return source

        with patch.object(pipeline, "_create_source", side_effect=create_source):
            result = pipeline.run(configs)

        assert result.feeds_attempted == 3
        assert result.feeds_succeeded == 2
        assert result.feeds_failed == 1
        assert result.documents_accepted == 2

    @patch("app.ingestion.pipeline.JSONLStore")
    def test_run_idempotent(self, mock_store_class):
        mock_store = Mock(spec=JSONLStore)
        mock_store.load_all.return_value = []
        mock_store_class.return_value = mock_store

        pipeline = IngestionPipeline(max_articles_per_feed=5, min_content_length=10)

        mock_source = Mock()
        mock_source.fetch.return_value = [
            RawDocument(
                source_id="test",
                source_name="Test",
                source_type=SourceType.RSS,
                title="Test Article",
                url="https://example.com/1",
                content="Valid content that is long enough to pass validation",
                retrieved_at=datetime.now(timezone.utc),
            )
        ]

        with patch.object(pipeline, "_create_source", return_value=mock_source):
            result1 = pipeline.run([self.test_config])
            result2 = pipeline.run([self.test_config])

        assert result1.documents_accepted == 1
        assert result2.documents_accepted == 0
        assert result2.duplicates == 1

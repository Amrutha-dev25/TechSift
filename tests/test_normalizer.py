import pytest
from datetime import datetime, timezone

from app.ingestion.normalizer import Normalizer
from app.models.document import RawDocument, SourceType


class TestNormalizer:
    def setup_method(self):
        self.normalizer = Normalizer(min_content_length=10)

    def test_normalize_title_whitespace(self):
        doc = RawDocument(
            source_id="test",
            source_name="Test",
            source_type=SourceType.RSS,
            title="  Test   Title  ",
            url="https://example.com",
            retrieved_at=datetime.now(timezone.utc),
        )
        normalized = self.normalizer.normalize(doc)
        assert normalized.title == "Test Title"

    def test_normalize_url_trailing_slash(self):
        doc = RawDocument(
            source_id="test",
            source_name="Test",
            source_type=SourceType.RSS,
            title="Test",
            url="https://example.com/article/",
            retrieved_at=datetime.now(timezone.utc),
        )
        normalized = self.normalizer.normalize(doc)
        assert normalized.url == "https://example.com/article"

    def test_normalize_url_no_trailing_slash(self):
        doc = RawDocument(
            source_id="test",
            source_name="Test",
            source_type=SourceType.RSS,
            title="Test",
            url="https://example.com/article",
            retrieved_at=datetime.now(timezone.utc),
        )
        normalized = self.normalizer.normalize(doc)
        assert normalized.url == "https://example.com/article"

    def test_normalize_url_case_insensitive(self):
        doc = RawDocument(
            source_id="test",
            source_name="Test",
            source_type=SourceType.RSS,
            title="Test",
            url="HTTPS://EXAMPLE.COM/Article",
            retrieved_at=datetime.now(timezone.utc),
        )
        normalized = self.normalizer.normalize(doc)
        assert normalized.url == "https://example.com/Article"

    def test_aware_datetime_preserved(self):
        aware_dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        doc = RawDocument(
            source_id="test",
            source_name="Test",
            source_type=SourceType.RSS,
            title="Test",
            url="https://example.com",
            published_at=aware_dt,
            retrieved_at=aware_dt,
        )
        normalized = self.normalizer.normalize(doc)
        assert normalized.published_at == aware_dt

    def test_canonical_url_normalization(self):
        canonical = self.normalizer.get_canonical_url("https://example.com/article/")
        assert canonical == "https://example.com/article"

    def test_valid_document(self):
        doc = RawDocument(
            source_id="test",
            source_name="Test",
            source_type=SourceType.RSS,
            title="Test Title",
            url="https://example.com",
            content="This is sufficient content for the document",
            retrieved_at=datetime.now(timezone.utc),
        )
        is_valid, reason = self.normalizer.is_valid(doc)
        assert is_valid is True
        assert reason is None

    def test_short_content_invalid(self):
        doc = RawDocument(
            source_id="test",
            source_name="Test",
            source_type=SourceType.RSS,
            title="Title",
            url="https://example.com",
            content="Short",
            retrieved_at=datetime.now(timezone.utc),
        )
        is_valid, reason = self.normalizer.is_valid(doc)
        assert is_valid is False
        assert reason == "content_too_short"

    def test_summary_fallback_for_content(self):
        doc = RawDocument(
            source_id="test",
            source_name="Test",
            source_type=SourceType.RSS,
            title="Title",
            url="https://example.com",
            summary="This is a long enough summary to pass validation",
            retrieved_at=datetime.now(timezone.utc),
        )
        is_valid, reason = self.normalizer.is_valid(doc)
        assert is_valid is True

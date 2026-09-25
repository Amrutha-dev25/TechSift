import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from app.models.document import CanonicalDocument, RawDocument, SourceType


class TestRawDocument:
    def test_valid_raw_document(self):
        doc = RawDocument(
            source_id="test_source",
            source_name="Test Source",
            source_type=SourceType.RSS,
            title="Test Title",
            url="https://example.com/article",
            summary="Test summary",
            published_at=datetime.now(timezone.utc),
            retrieved_at=datetime.now(timezone.utc),
        )
        assert doc.title == "Test Title"
        assert doc.url == "https://example.com/article"

    def test_empty_title_raises(self):
        with pytest.raises(ValidationError):
            RawDocument(
                source_id="test",
                source_name="Test",
                source_type=SourceType.RSS,
                title="",
                url="https://example.com",
                retrieved_at=datetime.now(timezone.utc),
            )

    def test_whitespace_title_normalized(self):
        doc = RawDocument(
            source_id="test",
            source_name="Test",
            source_type=SourceType.RSS,
            title="  Test   Title  ",
            url="https://example.com",
            retrieved_at=datetime.now(timezone.utc),
        )
        assert doc.title == "Test Title"

    def test_invalid_url_raises(self):
        with pytest.raises(ValidationError):
            RawDocument(
                source_id="test",
                source_name="Test",
                source_type=SourceType.RSS,
                title="Test",
                url="not-a-url",
                retrieved_at=datetime.now(timezone.utc),
            )

    def test_empty_url_raises(self):
        with pytest.raises(ValidationError):
            RawDocument(
                source_id="test",
                source_name="Test",
                source_type=SourceType.RSS,
                title="Test",
                url="",
                retrieved_at=datetime.now(timezone.utc),
            )

    def test_naive_datetime_raises(self):
        with pytest.raises(ValidationError):
            RawDocument(
                source_id="test",
                source_name="Test",
                source_type=SourceType.RSS,
                title="Test",
                url="https://example.com",
                published_at=datetime(2024, 1, 1),
                retrieved_at=datetime.now(timezone.utc),
            )

    def test_optional_fields_none(self):
        doc = RawDocument(
            source_id="test",
            source_name="Test",
            source_type=SourceType.RSS,
            title="Test",
            url="https://example.com",
            retrieved_at=datetime.now(timezone.utc),
        )
        assert doc.author is None
        assert doc.content is None
        assert doc.summary == ""


class TestCanonicalDocument:
    def test_valid_canonical_document(self):
        doc = CanonicalDocument(
            document_id="abc123",
            source_id="test",
            source_name="Test",
            source_type=SourceType.RSS,
            title="Test Title",
            content="Test content",
            url="https://example.com",
            canonical_url="https://example.com",
            content_hash="a" * 64,
            retrieved_at=datetime.now(timezone.utc),
        )
        assert doc.document_id == "abc123"

    def test_content_hash_validation(self):
        with pytest.raises(ValidationError):
            CanonicalDocument(
                document_id="abc123",
                source_id="test",
                source_name="Test",
                source_type=SourceType.RSS,
                title="Test",
                content="Content",
                url="https://example.com",
                canonical_url="https://example.com",
                content_hash="short",
                retrieved_at=datetime.now(timezone.utc),
            )

    def test_empty_content_raises(self):
        with pytest.raises(ValidationError):
            CanonicalDocument(
                document_id="abc123",
                source_id="test",
                source_name="Test",
                source_type=SourceType.RSS,
                title="Test",
                content="",
                url="https://example.com",
                canonical_url="https://example.com",
                content_hash="a" * 64,
                retrieved_at=datetime.now(timezone.utc),
            )

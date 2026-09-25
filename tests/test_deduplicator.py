import pytest

from app.ingestion.deduplicator import Deduplicator
from app.models.document import CanonicalDocument, RawDocument, SourceType
from datetime import datetime, timezone


class TestDeduplicator:
    def setup_method(self):
        self.dedup = Deduplicator()

    def test_compute_document_id_deterministic(self):
        url = "https://example.com/article"
        id1 = self.dedup.compute_document_id(url)
        id2 = self.dedup.compute_document_id(url)
        assert id1 == id2
        assert len(id1) == 16

    def test_compute_content_hash_deterministic(self):
        content = "Test content"
        hash1 = self.dedup.compute_content_hash(content)
        hash2 = self.dedup.compute_content_hash(content)
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_exact_duplicate_by_url(self):
        raw_doc = RawDocument(
            source_id="test",
            source_name="Test",
            source_type=SourceType.RSS,
            title="Test",
            url="https://example.com/article",
            content="Content",
            retrieved_at=datetime.now(timezone.utc),
        )
        canonical_url = "https://example.com/article"
        content = "Content"

        result1 = self.dedup.check_duplicate(raw_doc, canonical_url, content)
        assert result1.is_duplicate is False

        result2 = self.dedup.check_duplicate(raw_doc, canonical_url, content)
        assert result2.is_duplicate is True
        assert result2.match_type == "exact_url"

    def test_exact_duplicate_by_content(self):
        raw_doc = RawDocument(
            source_id="test",
            source_name="Test",
            source_type=SourceType.RSS,
            title="Test",
            url="https://example.com/article",
            content="Content",
            retrieved_at=datetime.now(timezone.utc),
        )
        canonical_url = "https://example.com/article"
        content = "Content"

        self.dedup.check_duplicate(raw_doc, canonical_url, content)

        raw_doc2 = RawDocument(
            source_id="test",
            source_name="Test",
            source_type=SourceType.RSS,
            title="Test 2",
            url="https://example.com/article-2",
            content="Content",
            retrieved_at=datetime.now(timezone.utc),
        )
        result = self.dedup.check_duplicate(raw_doc2, "https://example.com/article-2", content)
        assert result.is_duplicate is True
        assert result.match_type == "exact_content"

    def test_different_articles_not_duplicates(self):
        raw_doc1 = RawDocument(
            source_id="test",
            source_name="Test",
            source_type=SourceType.RSS,
            title="Article 1",
            url="https://example.com/1",
            content="Content about topic A",
            retrieved_at=datetime.now(timezone.utc),
        )
        raw_doc2 = RawDocument(
            source_id="test",
            source_name="Test",
            source_type=SourceType.RSS,
            title="Article 2",
            url="https://example.com/2",
            content="Content about topic B",
            retrieved_at=datetime.now(timezone.utc),
        )

        result1 = self.dedup.check_duplicate(raw_doc1, "https://example.com/1", "Content about topic A")
        assert result1.is_duplicate is False

        result2 = self.dedup.check_duplicate(raw_doc2, "https://example.com/2", "Content about topic B")
        assert result2.is_duplicate is False

    def test_load_existing_documents(self):
        canonical_url = "https://example.com/article"
        doc_id = self.dedup.compute_document_id(canonical_url)
        content = "Content"
        content_hash = self.dedup.compute_content_hash(content)

        docs = [
            CanonicalDocument(
                document_id=doc_id,
                source_id="test",
                source_name="Test",
                source_type=SourceType.RSS,
                title="Test",
                content=content,
                url=canonical_url,
                canonical_url=canonical_url,
                content_hash=content_hash,
                retrieved_at=datetime.now(timezone.utc),
            )
        ]
        self.dedup.load_existing(docs)

        raw_doc = RawDocument(
            source_id="test",
            source_name="Test",
            source_type=SourceType.RSS,
            title="Test",
            url=canonical_url,
            content=content,
            retrieved_at=datetime.now(timezone.utc),
        )
        result = self.dedup.check_duplicate(raw_doc, canonical_url, content)
        assert result.is_duplicate is True

    def test_mark_seen(self):
        canonical_url = "https://example.com/article"
        doc_id = self.dedup.compute_document_id(canonical_url)
        content = "Content"
        content_hash = self.dedup.compute_content_hash(content)

        doc = CanonicalDocument(
            document_id=doc_id,
            source_id="test",
            source_name="Test",
            source_type=SourceType.RSS,
            title="Test",
            content=content,
            url=canonical_url,
            canonical_url=canonical_url,
            content_hash=content_hash,
            retrieved_at=datetime.now(timezone.utc),
        )
        self.dedup.mark_seen(doc)

        raw_doc = RawDocument(
            source_id="test",
            source_name="Test",
            source_type=SourceType.RSS,
            title="Test",
            url=canonical_url,
            content=content,
            retrieved_at=datetime.now(timezone.utc),
        )
        result = self.dedup.check_duplicate(raw_doc, canonical_url, content)
        assert result.is_duplicate is True

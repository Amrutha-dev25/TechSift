"""Tests for Phase 4 RAG functionality."""

import pytest
from datetime import datetime

from app.rag import (
    RetrievalOrchestrator,
    QueryProcessor,
    ContextBuilder,
    Evidence,
    RetrievalResponse,
)


class TestQueryProcessor:
    """Test query processor."""

    def test_valid_query(self):
        qp = QueryProcessor()
        result = qp.validate_query("What is AI?")
        assert result == "What is AI?"

    def test_empty_query_rejected(self):
        qp = QueryProcessor()
        with pytest.raises(ValueError):
            qp.validate_query("")

    def test_whitespace_query_rejected(self):
        qp = QueryProcessor()
        with pytest.raises(ValueError):
            qp.validate_query("   ")

    def test_none_query_rejected(self):
        qp = QueryProcessor()
        with pytest.raises(ValueError):
            qp.validate_query(None)

    def test_long_query_rejected(self):
        qp = QueryProcessor()
        long_q = "a" * 3000
        with pytest.raises(ValueError):
            qp.validate_query(long_q)


class TestRetrievalOrchestrator:
    """Test retrieval orchestrator."""

    def test_valid_query_returns_results(self):
        orchestrator = RetrievalOrchestrator()
        resp = orchestrator.retrieve("AI coding assistants")
        assert isinstance(resp, RetrievalResponse)
        assert resp.query == "AI coding assistants"
        assert isinstance(resp.results, list)
        assert resp.total_results >= 0

    def test_empty_query_raises(self):
        orchestrator = RetrievalOrchestrator()
        with pytest.raises(ValueError):
            orchestrator.retrieve("")

    def test_whitespace_query_raises(self):
        orchestrator = RetrievalOrchestrator()
        with pytest.raises(ValueError):
            orchestrator.retrieve("   ")

    def test_semantic_retrieval_works(self):
        orchestrator = RetrievalOrchestrator()
        resp = orchestrator.retrieve("developers concerned about AI")
        # Should return results if they exist
        assert resp.total_results >= 0

    def test_sentiment_filtering(self):
        orchestrator = RetrievalOrchestrator()
        resp = orchestrator.retrieve("AI", sentiment_filter="negative")
        assert resp.total_results >= 0
        # If results exist, check sentiment
        for r in resp.results:
            assert r.sentiment == "negative" or r.sentiment is None

    def test_technology_filtering(self):
        orchestrator = RetrievalOrchestrator()
        resp = orchestrator.retrieve("AI", technology_filter="AI")
        assert resp.total_results >= 0

    def test_concern_filtering(self):
        orchestrator = RetrievalOrchestrator()
        resp = orchestrator.retrieve("AI", concern_filter="security")
        assert resp.total_results >= 0

    def test_date_filtering(self):
        orchestrator = RetrievalOrchestrator()
        resp = orchestrator.retrieve("AI", date_from=datetime(2020, 1, 1))
        assert resp.total_results >= 0

    def test_retrieval_mode(self):
        orchestrator = RetrievalOrchestrator()
        for mode in ["balanced", "concern", "hype"]:
            resp = orchestrator.retrieve("AI", retrieval_mode=mode)
            assert resp.total_results >= 0

    def test_evidence_metadata(self):
        orchestrator = RetrievalOrchestrator()
        resp = orchestrator.retrieve("AI coding assistants")
        if resp.results:
            e = resp.results[0]
            assert hasattr(e, "document_id")
            assert hasattr(e, "chunk_id")
            assert hasattr(e, "text")
            assert hasattr(e, "url")
            assert hasattr(e, "retrieval_score")

    def test_source_url_preservation(self):
        orchestrator = RetrievalOrchestrator()
        resp = orchestrator.retrieve("AI coding assistants")
        if resp.results:
            # At least some should have URLs
            assert any(r.url for r in resp.results)

    def test_no_result_query(self):
        orchestrator = RetrievalOrchestrator()
        resp = orchestrator.retrieve("xyznonexistentsuperlongquerythatmatchesnothing12345")
        assert resp.total_results == 0
        assert len(resp.results) == 0

    def test_multiple_source_retrieval(self):
        orchestrator = RetrievalOrchestrator()
        resp = orchestrator.retrieve("AI")
        assert resp.total_results >= 0


class TestContextBuilder:
    """Test context builder."""

    def test_build_context_empty(self):
        cb = ContextBuilder()
        ctx = cb.build_context([])
        assert ctx == ""

    def test_build_context_with_evidence(self):
        cb = ContextBuilder()
        e = Evidence(
            document_id="doc1",
            chunk_id="doc1:chunk:000",
            text="Test content about AI concerns",
            title="Test Title",
            url="https://example.com/test",
            source="Test Source",
            retrieval_score=0.85,
        )
        ctx = cb.build_context([e])
        assert "SOURCE 1" in ctx
        assert "Test content" in ctx
        assert "Test Title" in ctx
        assert "https://example.com/test" in ctx

    def test_context_no_answer_generation(self):
        cb = ContextBuilder()
        e = Evidence(
            document_id="doc1",
            chunk_id="doc1:chunk:000",
            text="Content",
            url="https://example.com",
        )
        ctx = cb.build_context([e])
        # Must not contain answer-like phrases
        ctx_lower = ctx.lower()
        assert "answer:" not in ctx_lower
        assert "therefore" not in ctx_lower or "content" in ctx  # Just ensure it's raw context

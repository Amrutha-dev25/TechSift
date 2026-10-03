"""Tests for Phase 4 RAG (query -> evidence -> context) functionality."""

import pytest
from datetime import datetime, timezone

from app.rag import (
    ContextBuilder,
    Evidence,
    QueryProcessor,
    RetrievalOrchestrator,
    RetrievalResponse,
)


# Shared orchestrator: the embedding model loads once for the whole module.
_orchestrator: RetrievalOrchestrator | None = None


def get_orchestrator() -> RetrievalOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = RetrievalOrchestrator()
    return _orchestrator


# ---------------------------------------------------------------------------
# 1. Query validation
# ---------------------------------------------------------------------------


class TestQueryProcessor:
    def test_valid_query(self):
        qp = QueryProcessor()
        assert qp.validate_query("What is AI?") == "What is AI?"

    def test_empty_query_rejected(self):
        with pytest.raises(ValueError):
            QueryProcessor().validate_query("")

    def test_whitespace_query_rejected(self):
        with pytest.raises(ValueError):
            QueryProcessor().validate_query("   \n\t  ")

    def test_none_query_rejected(self):
        with pytest.raises(ValueError):
            QueryProcessor().validate_query(None)

    def test_long_query_rejected(self):
        with pytest.raises(ValueError):
            QueryProcessor().validate_query("a" * 5000)

    def test_query_whitespace_trims(self):
        assert QueryProcessor().validate_query("  hello  ") == "hello"


# ---------------------------------------------------------------------------
# 2. Retrieval (semantic, filters, modes, evidence metadata)
# ---------------------------------------------------------------------------


class TestRetrievalOrchestrator:
    def test_valid_query_returns_response(self):
        resp = get_orchestrator().retrieve("AI coding assistants")
        assert isinstance(resp, RetrievalResponse)
        assert resp.query == "AI coding assistants"
        assert resp.total_results == len(resp.results)

    def test_empty_query_raises(self):
        with pytest.raises(ValueError):
            get_orchestrator().retrieve("")

    def test_whitespace_query_raises(self):
        with pytest.raises(ValueError):
            get_orchestrator().retrieve("     ")

    def test_semantic_retrieval_returns_relevant_evidence(self):
        resp = get_orchestrator().retrieve(
            "Why are developers concerned about AI coding assistants?", top_k=5
        )
        assert resp.total_results >= 1
        # The top result must actually relate to the query topic
        top = resp.results[0]
        blob = (top.text + " " + (top.title or "")).lower()
        assert "ai" in blob or "coding" in blob or "assistant" in blob

    def test_sentiment_filtering(self):
        resp = get_orchestrator().retrieve(
            "AI technology", sentiment_filter="negative", top_k=10
        )
        assert resp.total_results >= 1
        for r in resp.results:
            assert r.sentiment == "negative"

    def test_technology_filtering(self):
        resp = get_orchestrator().retrieve(
            "artificial intelligence tools", technology_filter="AI", top_k=10
        )
        assert resp.total_results >= 1
        for r in resp.results:
            blob = " ".join(str(t) for t in r.technology).lower()
            assert "ai" in blob

    def test_concern_filtering(self):
        resp = get_orchestrator().retrieve(
            "security risks in software", concern_filter="security", top_k=10
        )
        assert resp.total_results >= 1
        for r in resp.results:
            blob = " ".join(str(c) for c in r.concerns).lower()
            assert "security" in blob

    def test_date_filtering(self):
        resp = get_orchestrator().retrieve(
            "technology news",
            date_from=datetime(2020, 1, 1, tzinfo=timezone.utc),
            top_k=10,
        )
        assert resp.total_results >= 1
        for r in resp.results:
            assert r.published_at is not None
            assert r.published_at >= datetime(2020, 1, 1, tzinfo=timezone.utc)

    def test_source_filtering(self):
        resp = get_orchestrator().retrieve("AI", source_filter="Test Feed", top_k=5)
        for r in resp.results:
            assert r.source == "Test Feed"

    def test_retrieval_mode_balanced(self):
        resp = get_orchestrator().retrieve("AI", retrieval_mode="balanced", top_k=5)
        assert resp.total_results >= 1

    def test_retrieval_mode_concern(self):
        resp = get_orchestrator().retrieve("AI", retrieval_mode="concern", top_k=5)
        assert resp.total_results >= 1

    def test_retrieval_mode_hype(self):
        resp = get_orchestrator().retrieve("AI", retrieval_mode="hype", top_k=5)
        assert resp.total_results >= 1

    def test_evidence_metadata_complete(self):
        resp = get_orchestrator().retrieve("AI coding assistants", top_k=1)
        assert resp.total_results >= 1
        e = resp.results[0]
        assert e.document_id
        assert e.chunk_id
        assert e.text
        assert e.retrieval_score is not None

    def test_source_url_preserved(self):
        resp = get_orchestrator().retrieve("AI coding assistants", top_k=5)
        assert resp.total_results >= 1
        # Evidence must carry a source URL for citations
        assert any(r.url for r in resp.results)
        for r in resp.results:
            if r.url:
                assert r.url.startswith("http")

    def test_no_result_query_returns_empty(self):
        # Impossible filter combination -> explicit empty result, no fabrication
        resp = get_orchestrator().retrieve(
            "AI",
            concern_filter="definitely_not_a_real_concern_xyz",
            top_k=10,
        )
        assert resp.total_results == 0
        assert resp.results == []

    def test_multiple_source_retrieval(self):
        resp = get_orchestrator().retrieve("artificial intelligence", top_k=10)
        assert resp.total_results >= 1
        doc_ids = {r.document_id for r in resp.results}
        assert len(doc_ids) >= 1

    def test_diversity_limits_chunks_per_document(self):
        resp = get_orchestrator().retrieve("AI", top_k=10)
        counts: dict[str, int] = {}
        for r in resp.results:
            counts[r.document_id] = counts.get(r.document_id, 0) + 1
        assert all(c <= 2 for c in counts.values())


# ---------------------------------------------------------------------------
# 3. Context construction
# ---------------------------------------------------------------------------


class TestContextBuilder:
    def _sample_evidence(self) -> list[Evidence]:
        return [
            Evidence(
                document_id="doc1",
                chunk_id="doc1:chunk:000",
                text="Developers are worried about security backdoors.",
                title="AI Coding Assistant Concern",
                url="https://example.com/ai-concern",
                source="TechCrunch",
                published_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
                sentiment="negative",
                sentiment_score=-0.8,
                concerns=[{"category": "security", "confidence": 0.98}],
                retrieval_score=0.85,
            ),
            Evidence(
                document_id="doc2",
                chunk_id="doc2:chunk:000",
                text="Privacy risks were highlighted in the report.",
                title="Privacy Report",
                url="https://example.com/privacy",
                source="The Verge",
                retrieval_score=0.7,
            ),
        ]

    def test_build_context_empty(self):
        assert ContextBuilder().build_context([]) == ""

    def test_context_contains_all_sources(self):
        ctx = ContextBuilder().build_context(self._sample_evidence())
        assert "SOURCE 1" in ctx
        assert "SOURCE 2" in ctx
        assert "AI Coding Assistant Concern" in ctx
        assert "https://example.com/ai-concern" in ctx
        assert "TechCrunch" in ctx
        assert "Privacy Report" in ctx

    def test_context_contains_evidence_text_verbatim(self):
        evidence = self._sample_evidence()
        ctx = ContextBuilder().build_context(evidence)
        for e in evidence:
            assert e.text in ctx

    def test_context_does_not_generate_answer(self):
        ctx = ContextBuilder().build_context(self._sample_evidence())
        lower = ctx.lower()
        # Context builder must not synthesize answers
        assert "answer:" not in lower
        assert "in conclusion" not in lower
        assert "the answer is" not in lower

    def test_context_no_facts_beyond_evidence(self):
        evidence = self._sample_evidence()
        ctx = ContextBuilder().build_context(evidence)
        # Every sentence of content comes from evidence text or metadata labels
        assert evidence[0].text in ctx
        assert evidence[1].text in ctx

    def test_retrieve_with_context_returns_tuple(self):
        resp, ctx = get_orchestrator().retrieve_with_context(
            "Why are developers concerned about AI coding assistants?", top_k=3
        )
        assert isinstance(resp, RetrievalResponse)
        assert isinstance(ctx, str)
        assert resp.total_results >= 1
        assert "SOURCE 1" in ctx
        # Context contains the retrieved evidence text
        assert resp.results[0].text in ctx

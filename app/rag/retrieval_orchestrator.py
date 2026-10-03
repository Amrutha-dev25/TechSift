from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple

from app.rag.context_builder import ContextBuilder
from app.rag.models import Evidence, RetrievalResponse
from app.rag.query_processor import QueryProcessor
from app.retrieval.models import RetrievalFilters, RetrievalMode, RetrievalResult
from app.retrieval.retriever import Retriever

# When post-filters (technology/concern) are active, fetch extra candidates
# so that filtering does not starve the result set.
POST_FILTER_FETCH_MULTIPLIER = 3
# Default diversity: max chunks from a single document (Phase 3 mechanism)
DEFAULT_MAX_CHUNKS_PER_DOCUMENT = 2


class RetrievalOrchestrator:
    """Orchestrates query processing, retrieval, filtering, ranking, diversity.

    Flow: query -> Phase 3 semantic search -> metadata filtering ->
    ranking -> diversity -> final evidence.
    """

    def __init__(self, retriever: Optional[Retriever] = None):
        self.query_processor = QueryProcessor()
        self.retriever = retriever or Retriever()
        self.context_builder = ContextBuilder()

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        sentiment_filter: Optional[str] = None,
        sentiment_min: Optional[float] = None,
        sentiment_max: Optional[float] = None,
        technology_filter: Optional[str] = None,
        concern_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        source_type_filter: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        retrieval_mode: str = "balanced",
    ) -> RetrievalResponse:
        """Retrieve evidence for a query.

        Args:
            query: User query (validated; empty/whitespace rejected)
            top_k: Number of results to return
            sentiment_filter: Sentiment label (positive/negative/neutral)
            sentiment_min: Min sentiment score (-1..1)
            sentiment_max: Max sentiment score (-1..1)
            technology_filter: Substring match on technology entities
            concern_filter: Match on concern category
            source_filter: Source name (exact)
            source_type_filter: Source type (exact)
            date_from: Published on/after this datetime
            date_to: Published on/before this datetime
            retrieval_mode: balanced | concern | hype

        Returns:
            RetrievalResponse - empty results when nothing matches.
        """
        # 1. Query processing (original query always preserved)
        normalized_query = self.query_processor.validate_query(query)

        mode = self._parse_mode(retrieval_mode)

        # 2. Chroma-side filters only for fields stored as primitives
        #    (technology/concern are serialized strings -> post-filtered)
        filters = RetrievalFilters(
            sentiment_label=sentiment_filter,
            min_sentiment_score=sentiment_min,
            max_sentiment_score=sentiment_max,
            source_name=source_filter,
            source_type=source_type_filter,
            published_after=date_from,
            published_before=date_to,
            max_chunks_per_document=DEFAULT_MAX_CHUNKS_PER_DOCUMENT,
        )
        filters.validate()

        # Fetch extra candidates when post-filtering will remove some
        fetch_k = top_k
        if technology_filter or concern_filter:
            fetch_k = top_k * POST_FILTER_FETCH_MULTIPLIER

        results = self.retriever.search(
            query=normalized_query,
            top_k=fetch_k,
            filters=filters,
        )

        # 3. Post-filter on serialized technology/concern entities
        if technology_filter:
            results = [
                r
                for r in results
                if self._matches_technology(r, technology_filter)
            ]
        if concern_filter:
            results = [
                r for r in results if self._matches_concern(r, concern_filter)
            ]

        # 4. Ranking (mode-aware score adjustment), then 5. diversity/trim
        results = self._rank(results, mode)
        results = self._apply_diversity(results, DEFAULT_MAX_CHUNKS_PER_DOCUMENT)
        results = results[:top_k]

        evidence_list = [self._to_evidence(r) for r in results]

        return RetrievalResponse(
            query=normalized_query,
            results=evidence_list,
            total_results=len(evidence_list),
        )

    def retrieve_with_context(
        self,
        query: str,
        **kwargs,
    ) -> Tuple[RetrievalResponse, str]:
        """Retrieve evidence and build LLM-ready context.

        Returns (response, context). Context contains only retrieved
        evidence - no generated answers.
        """
        response = self.retrieve(query, **kwargs)
        context = self.context_builder.build_context(response.results)
        return response, context

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_mode(retrieval_mode: str) -> RetrievalMode:
        try:
            return RetrievalMode((retrieval_mode or "balanced").lower())
        except ValueError:
            return RetrievalMode.BALANCED

    @staticmethod
    def _entity_blob(items: List[str]) -> str:
        """Flattened lowercase string of serialized entities for matching."""
        return " ".join(str(i) for i in items).lower()

    @classmethod
    def _matches_technology(cls, result: RetrievalResult, tech: str) -> bool:
        return tech.lower() in cls._entity_blob(result.technology_entities)

    @classmethod
    def _matches_concern(cls, result: RetrievalResult, concern: str) -> bool:
        return concern.lower() in cls._entity_blob(result.concerns)

    @staticmethod
    def _rank(
        results: List[RetrievalResult], mode: RetrievalMode
    ) -> List[RetrievalResult]:
        """Deterministic mode-aware re-ranking of retrieval scores."""
        if mode == RetrievalMode.BALANCED:
            ranked = list(results)
        else:
            ranked = []
            for r in results:
                adjusted = r.score
                if mode == RetrievalMode.CONCERN:
                    # Prefer chunks that carry explicit concerns,
                    # especially negative-sentiment evidence.
                    if r.concerns:
                        adjusted += 0.03 * min(len(r.concerns), 3)
                    if r.sentiment_label == "negative":
                        adjusted += 0.03
                elif mode == RetrievalMode.HYPE:
                    # Prefer positive/excited sentiment evidence
                    if r.sentiment_label == "positive":
                        adjusted += 0.04
                    elif r.sentiment_label == "negative":
                        adjusted -= 0.02
                ranked.append(r.model_copy(update={"score": round(adjusted, 4)}))

        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked

    @staticmethod
    def _apply_diversity(
        results: List[RetrievalResult], max_per_doc: int
    ) -> List[RetrievalResult]:
        """Limit chunks per document, keeping the best-scoring chunks."""
        by_doc: dict[str, List[RetrievalResult]] = {}
        for r in results:
            by_doc.setdefault(r.document_id, []).append(r)

        trimmed: List[RetrievalResult] = []
        for chunks in by_doc.values():
            chunks.sort(key=lambda r: r.score, reverse=True)
            trimmed.extend(chunks[:max_per_doc])

        trimmed.sort(key=lambda r: r.score, reverse=True)
        return trimmed

    @staticmethod
    def _to_evidence(r: RetrievalResult) -> Evidence:
        return Evidence(
            document_id=r.document_id,
            chunk_id=r.chunk_id,
            text=r.text,
            title=r.title or None,
            url=r.url or None,
            source=r.source_name or None,
            published_at=r.published_at,
            sentiment=r.sentiment_label,
            sentiment_score=r.sentiment_score,
            technology=r.technology_entities,
            concerns=r.concerns,
            retrieval_score=r.score,
        )

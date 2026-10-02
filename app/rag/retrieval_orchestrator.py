from __future__ import annotations

from typing import List, Optional, Any, Dict
from datetime import datetime

from app.rag.models import Evidence, RetrievalResponse
from app.rag.query_processor import QueryProcessor
from app.retrieval.retriever import Retriever
from app.retrieval.models import RetrievalFilters, RetrievalMode


class RetrievalOrchestrator:
    """Orchestrates query processing and retrieval."""

    def __init__(self, retriever: Optional[Retriever] = None):
        self.query_processor = QueryProcessor()
        self.retriever = retriever or Retriever()
        self.context_builder = None  # Lazy import to avoid circular deps

    def _get_context_builder(self):
        if self.context_builder is None:
            from app.rag.context_builder import ContextBuilder
            self.context_builder = ContextBuilder()
        return self.context_builder

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
        """
        Retrieve evidence for a query.

        Args:
            query: User query
            top_k: Number of results
            sentiment_filter: Sentiment label (positive/negative/neutral)
            sentiment_min: Min sentiment score
            sentiment_max: Max sentiment score
            technology_filter: Filter by technology
            concern_filter: Filter by concern category
            source_filter: Filter by source name
            source_type_filter: Filter by source type
            date_from: Start date
            date_to: End date
            retrieval_mode: Retrieval mode (balanced/concern/hype)

        Returns:
            RetrievalResponse with evidence
        """
        # Validate query
        normalized_query = self.query_processor.validate_query(query)

        # Build filters
        filters = RetrievalFilters(
            sentiment_label=sentiment_filter,
            sentiment_min=sentiment_min,
            sentiment_max=sentiment_max,
            technology_contains=technology_filter,
            concern_category=concern_filter,
            source_name=source_filter,
            source_type=source_type_filter,
            date_from=date_from,
            date_to=date_to,
        )

        # Map retrieval mode
        mode_map = {
            "balanced": RetrievalMode.BALANCED,
            "concern": RetrievalMode.CONCERN,
            "hype": RetrievalMode.HYPE,
        }
        mode = mode_map.get(retrieval_mode.lower(), RetrievalMode.BALANCED)

        # Retrieve from Phase 3 retriever
        results = self.retriever.search(
            query=normalized_query,
            top_k=top_k,
            filters=filters,
            mode=mode,
        )

        # Convert to Evidence objects
        evidence_list = []
        for r in results:
            evidence = Evidence(
                document_id=r.document_id,
                chunk_id=r.chunk_id,
                text=r.text,
                title=r.title,
                url=r.url,
                source=r.source_name,
                published_at=r.published_at,
                sentiment=r.sentiment_label,
                sentiment_score=r.sentiment_score,
                technology=r.technology_entities,
                concerns=r.concerns,
                retrieval_score=r.score,
            )
            evidence_list.append(evidence)

        return RetrievalResponse(
            query=normalized_query,
            results=evidence_list,
            total_results=len(evidence_list),
        )

    def retrieve_with_context(
        self,
        query: str,
        **kwargs,
    ) -> tuple[RetrievalResponse, str]:
        """
        Retrieve evidence and build context.

        Returns:
            Tuple of (RetrievalResponse, context_string)
        """
        response = self.retrieve(query, **kwargs)
        context_builder = self._get_context_builder()
        context = context_builder.build_context(response.results)
        return response, context

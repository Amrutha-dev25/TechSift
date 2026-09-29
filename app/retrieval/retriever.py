from __future__ import annotations

import logging
from typing import List, Optional

from app.retrieval.models import RetrievalResult, RetrievalFilters
from app.vectorstore.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Primary retrieval interface for the knowledge base.

    Searches the ChromaDB vector store using semantic similarity
    with optional metadata filtering.
    """

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        default_top_k: int = 10,
    ):
        self._vector_store = vector_store or ChromaVectorStore()
        self._default_top_k = default_top_k

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[RetrievalFilters] = None,
    ) -> list[RetrievalResult]:
        """Search for chunks similar to the query.

        Args:
            query: Natural language search query
            top_k: Number of results to return (defaults to self._default_top_k)
            filters: Optional RetrievalFilters for metadata constraints

        Returns:
            list[RetrievalResult] - ranked results with source traceability
        """
        if not query or not query.strip():
            logger.warning("Empty query provided to retriever.search()")
            return []

        # Validate query length
        if (
            self._vector_store
            and self._vector_store._embedding_model
            and hasattr(self._vector_store._embedding_model, "model_name")
        ):
            pass

        # Use provided top_k or default
        k = top_k or self._default_top_k

        # Build ChromaDB filters dict from RetrievalFilters
        chroma_filters: Optional[Dict[str, Any]] = None
        if filters:
            chroma_filters = self._filters_to_chroma(filters)

        # Perform semantic search
        raw_results = self._vector_store.search(
            query=query,
            top_k=k,
            filters=chroma_filters,
        )

        # Convert to RetrievalResult objects
        results: list[RetrievalResult] = []
        for raw in raw_results:
            try:
                result = RetrievalResult(
                    chunk_id=raw.get("id", ""),
                    document_id=raw.get("document_id", ""),
                    text=raw.get("text", ""),
                    score=round(raw.get("score", 0.0), 4),
                    title=raw.get("title", ""),
                    source_name=raw.get("source_name", ""),
                    source_type=raw.get("source_type", ""),
                    url=raw.get("url", ""),
                    published_at=raw.get("published_at"),
                    sentiment_label=raw.get("sentiment_label"),
                    sentiment_score=raw.get("sentiment_score"),
                    technology_entities=raw.get("technology_entities", []),
                    concerns=raw.get("concerns", []),
                    emotions=raw.get("emotions", []),
                )
                results.append(result)
            except Exception as e:
                logger.warning("Failed to construct RetrievalResult: %s", e)
                continue

        # Apply diversity limit (max chunks per document)
        if filters and filters.max_chunks_per_document:
            results = self._apply_diversity(results, filters.max_chunks_per_document)

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        logger.info(
            "Retrieved %d results for query: %s (top_k=%d)",
            len(results),
            query[:50],
            k,
        )
        return results

    def _apply_diversity(
        self, results: list[RetrievalResult], max_per_doc: int
    ) -> list[RetrievalResult]:
        """Limit chunks per document to improve result diversity.

        Retains the highest-scoring chunks from each document.
        """
        doc_chunks: dict[str, list[RetrievalResult]] = {}

        for result in results:
            doc_id = result.document_id
            if doc_id not in doc_chunks:
                doc_chunks[doc_id] = []
            doc_chunks[doc_id].append(result)

        # Trim each document's chunks
        trimmed: list[RetrievalResult] = []
        for doc_id, chunks in doc_chunks.items():
            chunks.sort(key=lambda r: r.score, reverse=True)
            trimmed.extend(chunks[:max_per_doc])

        # Re-sort by score
        trimmed.sort(key=lambda r: r.score, reverse=True)
        return trimmed

    def _filters_to_chroma(self, filters: RetrievalFilters) -> Dict[str, Any]:
        """Convert RetrievalFilters to ChromaDB where clause.

        Priority: build dict with Chroma-compatible operators.
        """
        where: Dict[str, Any] = {}

        # Sentiment label
        if filters.sentiment_label:
            where["sentiment_label"] = filters.sentiment_label

        # Sentiment score range
        if filters.min_sentiment_score is not None:
            if "sentiment_score" in where:
                where["sentiment_score"]["$gte"] = filters.min_sentiment_score
            else:
                where["sentiment_score"] = {"$gte": filters.min_sentiment_score}

        if filters.max_sentiment_score is not None:
            if "sentiment_score" in where:
                where["sentiment_score"]["$lte"] = filters.max_sentiment_score
            else:
                where["sentiment_score"] = {"$lte": filters.max_sentiment_score}

        # Concern filter
        if filters.concern:
            where["concerns"] = filters.concern

        # Technology entity
        if filters.technology:
            where["technology_entities"] = filters.technology

        # Source type
        if filters.source_type:
            where["source_type"] = filters.source_type

        # Source name
        if filters.source_name:
            where["source_name"] = filters.source_name

        # Date range
        if filters.published_after:
            where["published_timestamp"] = {"$gte": filters.published_after.timestamp()}
        if filters.published_before:
            if "published_timestamp" in where:
                where["published_timestamp"]["$lte"] = filters.published_before.timestamp()
            else:
                where["published_timestamp"] = {"$lte": filters.published_before.timestamp()}

        # Mode (balanced/concern/hype) - handled by ranking boost, not filter
        # Diversity
        # max_chunks_per_document - handled post-retrieval

        return where if where else None
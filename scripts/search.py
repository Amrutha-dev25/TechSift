#!/usr/bin/env python
"""Search the Phase 3 ChromaDB knowledge base.

Returns structured retrieval results with source traceability.
Does NOT generate LLM answers - belongs to Phase 5.
"""

import argparse
import json
import sys
import logging
from datetime import datetime

from app.retrieval.models import RetrievalResult, RetrievalFilters
from app.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
logger.addHandler(handler)


def run_search(
    query: str,
    top_k: int = 10,
    sentiment: str = None,
    technology: str = None,
    concern: str = None,
    source_type: str = None,
    source_name: str = None,
    mode: str = None,
    published_after: str = None,
    published_before: str = None,
    max_chunks_per_doc: int = 2,
) -> list[RetrievalResult]:
    """Run a semantic search query.

    Args:
        query: Search query text
        top_k: Number of results to return
        sentiment: Filter by sentiment (positive/neutral/negative)
        technology: Filter by technology entity
        concern: Filter by concern category
        source_type: Filter by source type (e.g. rss)
        source_name: Filter by source name
        mode: Retrieval mode (balanced, concern, hype)
        published_after: ISO timestamp or Unix timestamp
        published_before: ISO timestamp or Unix timestamp
        max_chunks_per_doc: Max chunks per document for diversity

    Returns:
        list[RetrievalResult] - structured search results
    """
    # Build filters
    filters = RetrievalFilters()

    if sentiment:
        filters.sentiment_label = sentiment
    if technology:
        filters.technology = technology
    if concern:
        filters.concern = concern
    if source_type:
        filters.source_type = source_type
    if source_name:
        filters.source_name = source_name
    if mode:
        filters.mode = mode

    # Handle date filters
    if published_after:
        try:
            # Try ISO format first
            if "/" in published_after or "-" in published_after:
                dt = datetime.fromisoformat(published_after)
            else:
                # Unix timestamp
                dt = datetime.fromtimestamp(float(published_after))
            filters.published_after = dt
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not parse published_after: {e}")

    if published_before:
        try:
            if "/" in published_before or "-" in published_before:
                dt = datetime.fromisoformat(published_before)
            else:
                dt = datetime.fromtimestamp(float(published_before))
            filters.published_before = dt
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not parse published_before: {e}")

    # Set max query length constraint
    filters.max_query_length = 2000

    # Set max chunks per document for diversity
    if max_chunks_per_doc:
        filters.max_chunks_per_document = max_chunks_per_doc

    # Initialize retriever and search
    retriever = Retriever(default_top_k=top_k)

    results = retriever.search(
        query=query,
        top_k=top_k,
        filters=filters,
    )

    return results


def format_result(result: RetrievalResult, index: int) -> str:
    """Format a single RetrievalResult for CLI output."""
    lines = [
        "=" * 60,
        f"Result {index + 1}",
        "-" * 60,
        f"Title: {result.title}",
        f"Source: {result.source_name}",
        f"Published: {result.published_at}",
        f"Similarity: {result.score:.2f}",
        f"Sentiment: {result.sentiment_label or 'N/A'}",
        f"Concerns: {', '.join(result.concerns) if result.concerns else 'N/A'}",
        f"Technology: {', '.join(result.technology_entities) if result.technology_entities else 'N/A'}",
        "",
        f"Text: \"{result.text}\"",
        f"URL: {result.url}",
        "-" * 60,
    ]
    return "\n".join(lines)


def main():
    """CLI entry point for search."""
    parser = argparse.ArgumentParser(
        description="Phase 3: Semantic search the technology reaction knowledge base"
    )

    parser.add_argument(
        "query",
        type=str,
        help="Search query text",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to return (default: 10)",
    )
    parser.add_argument(
        "--sentiment",
        type=str,
        choices=["positive", "neutral", "negative"],
        help="Filter by sentiment",
    )
    parser.add_argument(
        "--technology",
        type=str,
        help="Filter by technology entity",
    )
    parser.add_argument(
        "--concern",
        type=str,
        help="Filter by concern category",
    )
    parser.add_argument(
        "--source-type",
        type=str,
        help="Filter by source type (e.g. rss)",
    )
    parser.add_argument(
        "--source-name",
        type=str,
        help="Filter by source name (e.g. TechCrunch)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["balanced", "concern", "hype"],
        help="Retrieval mode",
    )
    parser.add_argument(
        "--published-after",
        type=str,
        help="Filter documents published after this (ISO timestamp or Unix)",
    )
    parser.add_argument(
        "--published-before",
        type=str,
        help="Filter documents published before this (ISO timestamp or Unix)",
    )
    parser.add_argument(
        "--max-chunks-per-doc",
        type=int,
        default=2,
        help="Max chunks per document for diversity (default: 2)",
    )

    args = parser.parse_args()

    # Run search
    results = run_search(
        query=args.query,
        top_k=args.top_k,
        sentiment=args.sentiment,
        technology=args.technology,
        concern=args.concern,
        source_type=args.source_type,
        source_name=args.source_name,
        mode=args.mode,
        published_after=args.published_after,
        published_before=args.published_before,
        max_chunks_per_doc=args.max_chunks_per_doc,
    )

    # Display results
    print("=" * 60)
    print("Retrieval Results")
    print("=" * 60)
    print(f"Query: {args.query}")
    print(f"Results: {len(results)} found")
    print()

    if not results:
        print("No results matched the specified filters.")
        print("=" * 60)
        return

    for i, result in enumerate(results):
        print(format_result(result, i))

    print()
    print("=" * 60)
    print("End of results")
    print("=" * 60)


if __name__ == "__main__":
    main()
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    """Result from a semantic retrieval query.

    Contains enough metadata for Phase 5 citations.
    """

    chunk_id: str = Field(..., description="Deterministic chunk ID (e.g. doc_id:chunk:000)")
    document_id: str = Field(..., description="Phase 1 document ID")
    text: str = Field(..., description="Chunk text content")
    score: float = Field(..., description="Similarity score (0-1, higher is better)")

    title: str = Field(default="", description="Article title")
    source_name: str = Field(default="", description="Source name (e.g. TechCrunch)")
    source_type: str = Field(default="", description="Source type (e.g. rss)")
    url: str = Field(default="", description="Original article URL")

    published_at: Optional[datetime] = Field(default=None, description="ISO publication timestamp")

    sentiment_label: Optional[str] = Field(
        default=None, description="Sentiment: positive/neutral/negative"
    )
    sentiment_score: Optional[float] = Field(
        default=None, description="Sentiment score in [-1, 1]"
    )

    technology_entities: List[str] = Field(
        default_factory=list,
        description="Technology entities extracted from Phase 2",
    )
    concerns: List[str] = Field(
        default_factory=list,
        description="Concern categories from Phase 2",
    )
    emotions: List[str] = Field(
        default_factory=list,
        description="Emotion labels from Phase 2",
    )


class RetrievalFilters(BaseModel):
    """Filters to apply during semantic retrieval.

    All fields are optional - unset fields are not applied.
    """

    # Sentiment filters
    sentiment_label: Optional[str] = Field(
        default=None,
        description="Filter by sentiment: positive, neutral, negative",
    )
    min_sentiment_score: Optional[float] = Field(
        default=None,
        description="Minimum sentiment score (inclusive, -1 to 1)",
        ge=-1,
        le=1,
    )
    max_sentiment_score: Optional[float] = Field(
        default=None,
        description="Maximum sentiment score (inclusive, -1 to 1)",
        ge=-1,
        le=1,
    )

    # Technology/entity filter
    technology: Optional[str] = Field(
        default=None,
        description="Filter by technology entity name",
    )

    # Concern filter
    concern: Optional[str] = Field(
        default=None,
        description="Filter by concern category",
    )

    # Source filters
    source_type: Optional[str] = Field(
        default=None,
        description="Filter by source type (e.g. rss)",
    )
    source_name: Optional[str] = Field(
        default=None,
        description="Filter by source name (e.g. TechCrunch)",
    )

    # Date filters
    published_after: Optional[datetime] = Field(
        default=None,
        description="Filter documents published after this timestamp",
    )
    published_before: Optional[datetime] = Field(
        default=None,
        description="Filter documents published before this timestamp",
    )

    # Mode filters
    mode: Optional[str] = Field(
        default=None,
        description="Retrieval mode: balanced, concern, hype",
    )

    # Diversity control
    max_chunks_per_document: Optional[int] = Field(
        default=None,
        description="Maximum chunks to return per document",
    )

    # Query constraints
    max_query_length: Optional[int] = Field(
        default=None,
        description="Maximum query length in characters",
    )

    model_config = {"validate_by_name": True, "str": True}

    def validate(self) -> None:
        """Validate filter constraints.

        Raises ValueError if constraints are invalid.
        """
        # Validate date range
        if self.published_after and self.published_before:
            if self.published_after > self.published_before:
                raise ValueError(
                    "published_after must be <= published_before"
                )

        # Validate query length if set
        if self.max_query_length and self.max_query_length < 1:
            raise ValueError("max_query_length must be >= 1")

        # Validate sentiment scores
        if self.min_sentiment_score is not None:
            if self.min_sentiment_score < -1 or self.min_sentiment_score > 1:
                raise ValueError("min_sentiment_score must be in [-1, 1]")

        if self.max_sentiment_score is not None:
            if self.max_sentiment_score < -1 or self.max_sentiment_score > 1:
                raise ValueError("max_sentiment_score must be in [-1, 1]")
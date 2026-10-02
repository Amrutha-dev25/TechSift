from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Any, Dict

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """Individual piece of retrieved evidence."""

    document_id: str = Field(..., description="Document identifier")
    chunk_id: str = Field(..., description="Chunk identifier")
    text: str = Field(..., description="Evidence text content")
    title: Optional[str] = Field(None, description="Document title")
    url: Optional[str] = Field(None, description="Source URL")
    source: Optional[str] = Field(None, description="Source name")
    published_at: Optional[datetime] = Field(None, description="Publication timestamp")
    sentiment: Optional[str] = Field(None, description="Sentiment label")
    sentiment_score: Optional[float] = Field(None, description="Sentiment score")
    technology: List[Dict[str, Any]] = Field(default_factory=list, description="Technology entities")
    concerns: List[Dict[str, Any]] = Field(default_factory=list, description="Concern entities")
    retrieval_score: Optional[float] = Field(None, description="Retrieval similarity score")

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


class RetrievalResponse(BaseModel):
    """Response from retrieval operation."""

    query: str = Field(..., description="Original query")
    results: List[Evidence] = Field(default_factory=list, description="Retrieved evidence")
    total_results: int = Field(default=0, description="Total number of results")

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True

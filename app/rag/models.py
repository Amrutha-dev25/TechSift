from __future__ import annotations

import ast
import json
from datetime import datetime
from typing import List, Optional, Any, Dict, Union

from pydantic import BaseModel, Field, field_validator


def _parse_json_like(value: Union[str, Dict, List]) -> Any:
    """Parse string representations of dicts/lists."""
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        if not value.strip():
            return value
        try:
            return ast.literal_eval(value)
        except Exception:
            pass
        try:
            return json.loads(value.replace("'", '"'))
        except Exception:
            pass
    return value


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
    technology: List[Union[Dict[str, Any], str]] = Field(default_factory=list, description="Technology entities")
    concerns: List[Union[Dict[str, Any], str]] = Field(default_factory=list, description="Concern entities")
    retrieval_score: Optional[float] = Field(None, description="Retrieval similarity score")

    @field_validator("technology", "concerns", mode="before")
    @classmethod
    def _parse_complex_fields(cls, v):
        if not v:
            return []
        if isinstance(v, list):
            return [_parse_json_like(item) for item in v]
        return [_parse_json_like(v)]

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

from datetime import datetime
from enum import Enum
from typing import Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator


class SourceType(str, Enum):
    RSS = "rss"
    NEWS_API = "news_api"
    REDDIT = "reddit"
    OFFICIAL = "official"


class RawDocument(BaseModel):
    source_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    source_type: SourceType
    title: str
    url: str
    summary: str = ""
    published_at: Optional[datetime] = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now().astimezone())
    author: Optional[str] = None
    content: Optional[str] = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("title cannot be empty")
        return " ".join(stripped.split())

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("url cannot be empty")
        parsed = urlparse(stripped)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("url must be a valid URL with scheme and netloc")
        return stripped

    @field_validator("summary", "author", "content", mode="before")
    @classmethod
    def normalize_whitespace(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return " ".join(v.strip().split()) if v.strip() else None

    @model_validator(mode="after")
    def ensure_timezone_aware(self) -> "RawDocument":
        for field_name in ("published_at", "retrieved_at"):
            dt = getattr(self, field_name)
            if dt is not None and dt.tzinfo is None:
                raise ValueError(f"{field_name} must be timezone-aware")
        return self


class CanonicalDocument(BaseModel):
    document_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    source_type: SourceType
    title: str
    content: str
    url: str
    author: Optional[str] = None
    published_at: Optional[datetime] = None
    retrieved_at: datetime
    content_hash: str
    canonical_url: str

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("title cannot be empty")
        return " ".join(stripped.split())

    @field_validator("url", "canonical_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("url cannot be empty")
        parsed = urlparse(stripped)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("url must be a valid URL with scheme and netloc")
        return stripped

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("content cannot be empty")
        return stripped

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, v: str) -> str:
        if not v or len(v) != 64:
            raise ValueError("content_hash must be a 64-character hex string")
        return v

    @model_validator(mode="after")
    def ensure_timezone_aware(self) -> "CanonicalDocument":
        for field_name in ("published_at", "retrieved_at"):
            dt = getattr(self, field_name)
            if dt is not None and dt.tzinfo is None:
                raise ValueError(f"{field_name} must be timezone-aware")
        return self


class FailedRecord(BaseModel):
    timestamp: datetime
    source_id: str
    source_name: str
    source_type: SourceType
    url: Optional[str] = None
    reason: str
    error_type: str
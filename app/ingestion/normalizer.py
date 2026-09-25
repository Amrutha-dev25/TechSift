import logging
import re
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse, urlunparse

from app.models.document import RawDocument

logger = logging.getLogger(__name__)


class Normalizer:
    def __init__(self, min_content_length: int = 50):
        self.min_content_length = min_content_length

    def normalize(self, doc: RawDocument) -> RawDocument:
        normalized = doc.model_copy(deep=True)

        normalized.title = self._normalize_title(normalized.title)
        normalized.url = self._normalize_url(normalized.url)
        normalized.summary = self._normalize_text(normalized.summary) if normalized.summary else ""
        normalized.author = self._normalize_text(normalized.author) if normalized.author else None
        normalized.content = self._normalize_text(normalized.content) if normalized.content else None

        if normalized.published_at is not None:
            normalized.published_at = self._ensure_utc(normalized.published_at)
        normalized.retrieved_at = self._ensure_utc(normalized.retrieved_at)

        return normalized

    def _normalize_title(self, title: str) -> str:
        normalized = " ".join(title.strip().split())
        return normalized

    def _normalize_url(self, url: str) -> str:
        url = url.strip()
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return url

        normalized = urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/") if parsed.path != "/" else "/",
            parsed.params,
            parsed.query,
            "",
        ))
        return normalized

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.strip().split())

    def _ensure_utc(self, dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def get_canonical_url(self, url: str) -> str:
        return self._normalize_url(url)

    def is_valid(self, doc: RawDocument) -> tuple[bool, Optional[str]]:
        if not doc.title or not doc.title.strip():
            return False, "empty_title"

        if not doc.url or not doc.url.strip():
            return False, "empty_url"

        content = doc.content or doc.summary or ""
        if not content or len(content.strip()) < self.min_content_length:
            return False, "content_too_short"

        return True, None
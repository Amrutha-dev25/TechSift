import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import feedparser
import requests

from app.config.settings import settings
from app.ingestion.base import DataSource
from app.models.document import RawDocument, SourceType

logger = logging.getLogger(__name__)


class RSSSource(DataSource[RawDocument]):
    def __init__(
        self,
        source_id: str,
        source_name: str,
        feed_url: str,
        max_articles: Optional[int] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ):
        self._source_id = source_id
        self._source_name = source_name
        self._feed_url = feed_url
        self._max_articles = max_articles or settings.max_articles_per_feed
        self._timeout = timeout or settings.request_timeout_seconds
        self._max_retries = max_retries or settings.max_retries

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def source_name(self) -> str:
        return self._source_name

    @property
    def source_type(self) -> str:
        return SourceType.RSS.value

    def _fetch_with_retry(self) -> feedparser.FeedParserDict:
        last_exception: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                response = requests.get(
                    self._feed_url,
                    timeout=self._timeout,
                    headers={"User-Agent": "TechnologyReactionIntelligence/0.1"},
                )
                response.raise_for_status()
                return feedparser.parse(response.content)
            except requests.Timeout as e:
                last_exception = e
                logger.warning(
                    "Feed %s timeout on attempt %d/%d",
                    self._source_id,
                    attempt + 1,
                    self._max_retries + 1,
                )
            except requests.ConnectionError as e:
                last_exception = e
                logger.warning(
                    "Feed %s connection error on attempt %d/%d",
                    self._source_id,
                    attempt + 1,
                    self._max_retries + 1,
                )
            except requests.HTTPError as e:
                if e.response.status_code >= 500:
                    last_exception = e
                    logger.warning(
                        "Feed %s HTTP %d on attempt %d/%d",
                        self._source_id,
                        e.response.status_code,
                        attempt + 1,
                        self._max_retries + 1,
                    )
                else:
                    logger.error(
                        "Feed %s HTTP %d (non-retryable)",
                        self._source_id,
                        e.response.status_code,
                    )
                    raise
            except Exception as e:
                logger.error("Feed %s unexpected error: %s", self._source_id, e)
                raise

            if attempt < self._max_retries:
                backoff = 2**attempt
                logger.info("Retrying feed %s in %d seconds", self._source_id, backoff)
                time.sleep(backoff)

        raise last_exception or Exception(f"Failed to fetch feed {self._source_id}")

    def _parse_date(self, entry: feedparser.FeedParserDict) -> Optional[datetime]:
        for field in ("published_parsed", "updated_parsed", "created_parsed"):
            if entry.get(field):
                try:
                    return datetime(*entry[field][:6], tzinfo=timezone.utc)
                except Exception:
                    continue
        return None

    def _extract_content(self, entry: feedparser.FeedParserDict) -> Optional[str]:
        if entry.get("content"):
            for content_obj in entry.content:
                if isinstance(content_obj, dict):
                    if content_obj.get("type", "").startswith("text/"):
                        return content_obj.get("value")
                    if content_obj.get("value"):
                        return content_obj.get("value")
                else:
                    if getattr(content_obj, "type", "").startswith("text/"):
                        return getattr(content_obj, "value", None)
                    if getattr(content_obj, "value", None):
                        return content_obj.value
        if entry.get("summary"):
            return entry.summary
        if entry.get("description"):
            return entry.description
        return None

    def _extract_author(self, entry: feedparser.FeedParserDict) -> Optional[str]:
        if entry.get("author"):
            return entry.author
        if entry.get("authors"):
            for author in entry.authors:
                if isinstance(author, dict):
                    if author.get("name"):
                        return author.get("name")
                else:
                    if getattr(author, "name", None):
                        return author.name
        return None

    def fetch(self) -> list[RawDocument]:
        logger.info("Fetching feed: %s (%s)", self._source_name, self._feed_url)

        try:
            feed = self._fetch_with_retry()
        except Exception as e:
            logger.error("Failed to fetch feed %s: %s", self._source_id, e)
            raise

        if feed.bozo and feed.bozo_exception:
            logger.warning(
                "Feed %s has parsing issues: %s", self._source_id, feed.bozo_exception
            )

        entries = feed.entries[: self._max_articles]
        logger.info("Feed %s: %d entries to process", self._source_id, len(entries))

        documents: list[RawDocument] = []
        for entry in entries:
            try:
                url = entry.get("link", "").strip()
                title = entry.get("title", "").strip()

                if not url or not title:
                    logger.warning("Skipping entry from %s: missing url or title", self._source_id)
                    continue

                published_at = self._parse_date(entry)
                if published_at is None:
                    logger.warning(
                        "Feed %s entry has no valid publication date: %s", self._source_id, url
                    )

                content = self._extract_content(entry)
                author = self._extract_author(entry)

                doc = RawDocument(
                    source_id=self._source_id,
                    source_name=self._source_name,
                    source_type=SourceType.RSS,
                    title=title,
                    url=url,
                    summary=entry.get("summary", "").strip() or "",
                    published_at=published_at,
                    retrieved_at=datetime.now(timezone.utc),
                    author=author,
                    content=content,
                )
                documents.append(doc)
            except Exception as e:
                logger.warning("Failed to parse entry from %s: %s", self._source_id, e)
                continue

        logger.info("Feed %s: successfully parsed %d documents", self._source_id, len(documents))
        return documents
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import feedparser

from app.ingestion.rss import RSSSource
from app.models.document import RawDocument, SourceType


class TestRSSSource:
    def setup_method(self):
        self.source = RSSSource(
            source_id="test_feed",
            source_name="Test Feed",
            feed_url="https://example.com/feed.xml",
            max_articles=10,
            timeout=5,
            max_retries=1,
        )

    @patch("app.ingestion.rss.requests.get")
    def test_fetch_valid_feed(self, mock_get):
        mock_response = Mock()
        mock_response.content = b"""<?xml version="1.0"?>
<rss version="2.0">
<channel>
<title>Test Feed</title>
<item>
<title>Article 1</title>
<link>https://example.com/1</link>
<description>Description 1</description>
<pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
</item>
</channel>
</rss>"""
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        documents = self.source.fetch()

        assert len(documents) == 1
        assert isinstance(documents[0], RawDocument)
        assert documents[0].title == "Article 1"
        assert documents[0].url == "https://example.com/1"
        assert documents[0].source_id == "test_feed"

    @patch("app.ingestion.rss.requests.get")
    def test_fetch_multiple_entries(self, mock_get):
        mock_response = Mock()
        mock_response.content = b"""<?xml version="1.0"?>
<rss version="2.0">
<channel>
<title>Test Feed</title>
<item>
<title>Article 1</title>
<link>https://example.com/1</link>
<description>Description 1</description>
<pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
</item>
<item>
<title>Article 2</title>
<link>https://example.com/2</link>
<description>Description 2</description>
<pubDate>Tue, 02 Jan 2024 12:00:00 GMT</pubDate>
</item>
</channel>
</rss>"""
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        documents = self.source.fetch()

        assert len(documents) == 2

    @patch("app.ingestion.rss.requests.get")
    def test_fetch_missing_fields(self, mock_get):
        mock_response = Mock()
        mock_response.content = b"""<?xml version="1.0"?>
<rss version="2.0">
<channel>
<title>Test Feed</title>
<item>
<title></title>
<link></link>
</item>
<item>
<title>Valid Article</title>
<link>https://example.com/valid</link>
<description>Valid description</description>
</item>
</channel>
</rss>"""
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        documents = self.source.fetch()

        assert len(documents) == 1
        assert documents[0].title == "Valid Article"

    @patch("app.ingestion.rss.requests.get")
    def test_fetch_empty_feed(self, mock_get):
        mock_response = Mock()
        mock_response.content = b"""<?xml version="1.0"?>
<rss version="2.0">
<channel>
<title>Test Feed</title>
</channel>
</rss>"""
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        documents = self.source.fetch()

        assert len(documents) == 0

    @patch("app.ingestion.rss.requests.get")
    def test_fetch_timeout_retries(self, mock_get):
        import requests
        mock_get.side_effect = requests.Timeout("Timeout")

        with pytest.raises(requests.Timeout):
            self.source.fetch()

        assert mock_get.call_count == 2  # max_retries + 1

    @patch("app.ingestion.rss.requests.get")
    def test_fetch_http_5xx_retries(self, mock_get):
        import requests
        mock_response = Mock()
        mock_response.status_code = 503
        mock_response.raise_for_status.side_effect = requests.HTTPError("503", response=mock_response)
        mock_get.return_value = mock_response

        with pytest.raises(requests.HTTPError):
            self.source.fetch()

        assert mock_get.call_count == 2

    @patch("app.ingestion.rss.requests.get")
    def test_fetch_http_4xx_no_retry(self, mock_get):
        import requests
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.HTTPError("404", response=mock_response)
        mock_get.return_value = mock_response

        with pytest.raises(requests.HTTPError):
            self.source.fetch()

        assert mock_get.call_count == 1

    @patch("app.ingestion.rss.requests.get")
    def test_fetch_atom_feed(self, mock_get):
        mock_response = Mock()
        mock_response.content = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
<title>Test Feed</title>
<entry>
<title>Atom Article</title>
<link href="https://example.com/atom"/>
<summary>Atom summary</summary>
<updated>2024-01-01T12:00:00Z</updated>
</entry>
</feed>"""
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        documents = self.source.fetch()

        assert len(documents) == 1
        assert documents[0].title == "Atom Article"

    def test_parse_date_from_published_parsed(self):
        entry = feedparser.FeedParserDict()
        entry['published_parsed'] = (2024, 1, 1, 12, 0, 0, 0, 1, -1)
        parsed = self.source._parse_date(entry)
        assert parsed == datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_parse_date_from_updated_parsed(self):
        entry = feedparser.FeedParserDict()
        entry['updated_parsed'] = (2024, 1, 1, 12, 0, 0, 0, 1, -1)
        parsed = self.source._parse_date(entry)
        assert parsed == datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_parse_date_missing_returns_none(self):
        entry = feedparser.FeedParserDict()
        parsed = self.source._parse_date(entry)
        assert parsed is None

    def test_extract_content_prefers_full_content(self):
        entry = feedparser.FeedParserDict()
        entry['content'] = [
            {"type": "text/html", "value": "Full content"},
            {"type": "text/plain", "value": "Plain content"},
        ]
        entry['summary'] = "Summary"
        content = self.source._extract_content(entry)
        assert content == "Full content"

    def test_extract_content_fallback_to_summary(self):
        entry = feedparser.FeedParserDict()
        entry['summary'] = "Summary content"
        content = self.source._extract_content(entry)
        assert content == "Summary content"

    def test_extract_content_fallback_to_description(self):
        entry = feedparser.FeedParserDict()
        entry['description'] = "Description content"
        content = self.source._extract_content(entry)
        assert content == "Description content"

    def test_extract_author(self):
        entry = feedparser.FeedParserDict()
        entry['author'] = "Test Author"
        author = self.source._extract_author(entry)
        assert author == "Test Author"

    def test_extract_author_from_authors_list(self):
        entry = feedparser.FeedParserDict()
        entry['authors'] = [{"name": "Author 1"}, {"name": "Author 2"}]
        author = self.source._extract_author(entry)
        assert author == "Author 1"

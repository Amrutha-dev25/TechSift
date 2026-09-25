import logging
import re
from html import unescape
from typing import Optional

logger = logging.getLogger(__name__)


class Cleaner:
    def __init__(self):
        self._html_tag_pattern = re.compile(r"<[^>]+>")
        self._script_style_pattern = re.compile(
            r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL
        )
        self._whitespace_pattern = re.compile(r"\s+")
        self._tracking_pattern = re.compile(
            r"(utm_source|utm_medium|utm_campaign|utm_content|utm_term|fbclid|gclid)=[^&\s]+",
            re.IGNORECASE,
        )

    def clean(self, text: Optional[str]) -> str:
        if not text:
            return ""

        cleaned = unescape(text)

        cleaned = self._script_style_pattern.sub("", cleaned)

        cleaned = self._html_tag_pattern.sub(" ", cleaned)

        cleaned = self._tracking_pattern.sub("", cleaned)

        cleaned = self._whitespace_pattern.sub(" ", cleaned)

        cleaned = cleaned.strip()

        return cleaned

    def clean_preserve_paragraphs(self, text: Optional[str]) -> str:
        if not text:
            return ""

        cleaned = unescape(text)

        cleaned = self._script_style_pattern.sub("", cleaned)

        cleaned = re.sub(r"</p\s*>", "\n\n", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<br\s*/?>", "\n", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"</div\s*>", "\n", cleaned, flags=re.IGNORECASE)

        cleaned = self._html_tag_pattern.sub(" ", cleaned)

        cleaned = self._tracking_pattern.sub("", cleaned)

        cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)

        cleaned = re.sub(r"[ \t]+", " ", cleaned)

        cleaned = cleaned.strip()

        return cleaned
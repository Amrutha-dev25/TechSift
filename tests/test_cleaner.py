import pytest

from app.ingestion.cleaner import Cleaner


class TestCleaner:
    def setup_method(self):
        self.cleaner = Cleaner()

    def test_remove_html_tags(self):
        text = "<p>Hello <b>world</b></p>"
        cleaned = self.cleaner.clean(text)
        assert cleaned == "Hello world"

    def test_remove_scripts(self):
        text = '<script>alert("xss")</script><p>Content</p>'
        cleaned = self.cleaner.clean(text)
        assert "alert" not in cleaned
        assert "Content" in cleaned

    def test_remove_styles(self):
        text = '<style>body { color: red; }</style><p>Content</p>'
        cleaned = self.cleaner.clean(text)
        assert "color: red" not in cleaned
        assert "Content" in cleaned

    def test_remove_tracking_params(self):
        text = "Check out https://example.com?utm_source=feed&utm_medium=rss"
        cleaned = self.cleaner.clean(text)
        assert "utm_source" not in cleaned
        assert "utm_medium" not in cleaned

    def test_normalize_whitespace(self):
        text = "Hello    world\n\n\ntest"
        cleaned = self.cleaner.clean(text)
        assert cleaned == "Hello world test"

    def test_unescape_html_entities(self):
        text = "<p>Hello & world</p>"
        cleaned = self.cleaner.clean(text)
        assert cleaned == "Hello & world"

    def test_empty_string(self):
        assert self.cleaner.clean("") == ""
        assert self.cleaner.clean(None) == ""

    def test_preserve_paragraphs(self):
        text = "<p>First paragraph</p><p>Second paragraph</p>"
        cleaned = self.cleaner.clean_preserve_paragraphs(text)
        assert "First paragraph" in cleaned
        assert "Second paragraph" in cleaned
        assert "\n\n" in cleaned

    def test_br_tags_to_newlines(self):
        text = "Line 1<br>Line 2<br/>Line 3"
        cleaned = self.cleaner.clean_preserve_paragraphs(text)
        assert "Line 1\nLine 2\nLine 3" in cleaned

    def test_div_tags_to_newlines(self):
        text = "<div>Block 1</div><div>Block 2</div>"
        cleaned = self.cleaner.clean_preserve_paragraphs(text)
        assert "Block 1" in cleaned
        assert "Block 2" in cleaned
        assert "\n" in cleaned

    def test_multiple_newlines_collapsed(self):
        text = "<p>Para 1</p>\n\n\n\n<p>Para 2</p>"
        cleaned = self.cleaner.clean_preserve_paragraphs(text)
        assert cleaned.count("\n\n") == 1

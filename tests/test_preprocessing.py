import pytest

from app.nlp.preprocessing import (
    prepare_text_for_analysis,
    truncate_text,
    clean_for_nlp,
    split_into_chunks,
    extract_evidence_span,
    normalize_entity_text,
    canonicalize_tech_entity,
    is_technology_entity,
)
from app.nlp.taxonomy import EntityType


class TestPrepareTextForAnalysis:
    def test_both_title_and_content(self):
        result = prepare_text_for_analysis("Title here", "Content here")
        assert result == "Title here\n\nContent here"

    def test_only_title(self):
        result = prepare_text_for_analysis("Title only", "")
        assert result == "Title only"

    def test_only_content(self):
        result = prepare_text_for_analysis("", "Content only")
        assert result == "Content only"

    def test_none_values(self):
        result = prepare_text_for_analysis(None, None)
        assert result == ""

    def test_whitespace_handling(self):
        result = prepare_text_for_analysis("  Title  ", "  Content  ")
        assert result == "Title\n\nContent"


class TestTruncateText:
    def test_no_truncation_needed(self):
        text = "Short text"
        result = truncate_text(text, max_length=100)
        assert result == text

    def test_truncation(self):
        text = "x" * 5000
        result = truncate_text(text, max_length=100)
        assert len(result) == 100
        assert result == "x" * 100


class TestCleanForNLP:
    def test_normalizes_whitespace(self):
        text = "Multiple    spaces\n\nand\t\ttabs"
        result = clean_for_nlp(text)
        assert result == "Multiple spaces and tabs"

    def test_strips_edges(self):
        text = "  leading and trailing  "
        result = clean_for_nlp(text)
        assert result == "leading and trailing"


class TestSplitIntoChunks:
    def test_short_text_single_chunk(self):
        text = "Short text"
        chunks = split_into_chunks(text, max_chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_multiple_chunks(self):
        text = "x" * 1200
        chunks = split_into_chunks(text, max_chunk_size=512, overlap=50)
        assert len(chunks) == 3
        assert len(chunks[0]) == 512
        assert len(chunks[1]) == 512
        assert len(chunks[2]) == 276

    def test_overlap(self):
        text = "ABCDEFGHIJ"
        chunks = split_into_chunks(text, max_chunk_size=5, overlap=2)
        assert chunks[0] == "ABCDE"
        assert chunks[1] == "DEFGH"
        assert chunks[2] == "GHIJ"


class TestExtractEvidenceSpan:
    def test_finds_keyword(self):
        text = "The system collects user data without proper consent from users."
        keywords = ["user data", "consent", "privacy"]
        result = extract_evidence_span(text, keywords, window=20)
        assert result is not None
        assert "user data" in result.lower()

    def test_returns_none_for_no_match(self):
        text = "This is a completely unrelated text about cooking."
        keywords = ["privacy", "security", "data"]
        result = extract_evidence_span(text, keywords)
        assert result is None

    def test_case_insensitive(self):
        text = "PRIVACY is important."
        keywords = ["privacy"]
        result = extract_evidence_span(text, keywords)
        assert result is not None
        assert "privacy" in result.lower()


class TestNormalizeEntityText:
    def test_normalizes_whitespace(self):
        result = normalize_entity_text("  GitHub   Copilot  ")
        assert result == "GitHub Copilot"

    def test_handles_newlines(self):
        result = normalize_entity_text("GitHub\nCopilot")
        assert result == "GitHub Copilot"


class TestCanonicalizeTechEntity:
    def test_known_entities(self):
        assert canonicalize_tech_entity("copilot") == "GitHub Copilot"
        assert canonicalize_tech_entity("GITHUB COPILOT") == "GitHub Copilot"
        assert canonicalize_tech_entity("chatgpt") == "ChatGPT"
        assert canonicalize_tech_entity("claude") == "Claude"
        assert canonicalize_tech_entity("gemini") == "Gemini"
        assert canonicalize_tech_entity("windows recall") == "Windows Recall"
        assert canonicalize_tech_entity("openai") == "OpenAI"
        assert canonicalize_tech_entity("microsoft") == "Microsoft"

    def test_unknown_entity(self):
        assert canonicalize_tech_entity("unknown product xyz") is None

    def test_case_insensitive(self):
        assert canonicalize_tech_entity("CoPiLoT") == "GitHub Copilot"


class TestIsTechnologyEntity:
    def test_technology_type(self):
        assert is_technology_entity("Some Tech", EntityType.TECHNOLOGY) is True

    def test_product_type(self):
        assert is_technology_entity("Some Product", EntityType.PRODUCT) is True

    def test_org_with_tech_indicators(self):
        assert is_technology_entity("OpenAI", EntityType.ORG) is True
        assert is_technology_entity("Microsoft AI", EntityType.ORG) is True

    def test_person_not_tech(self):
        assert is_technology_entity("Satya Nadella", EntityType.PERSON) is False

    def test_other_not_tech(self):
        assert is_technology_entity("Random Company", EntityType.OTHER) is False
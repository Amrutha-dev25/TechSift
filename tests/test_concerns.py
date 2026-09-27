import pytest
from unittest.mock import Mock, patch

from app.nlp.concerns import ConcernExtractor
from app.nlp.taxonomy import ConcernCategory
from app.nlp.models import ConcernResult


class TestConcernExtractor:
    @pytest.fixture
    def mock_pipeline(self):
        with patch("app.nlp.concerns.pipeline") as mock:
            mock_instance = Mock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def extractor(self, mock_pipeline):
        with patch("app.nlp.concerns.AutoTokenizer.from_pretrained"), \
             patch("app.nlp.concerns.AutoModelForSequenceClassification.from_pretrained"):
            extractor = ConcernExtractor(model_name="test-model")
            extractor._is_loaded = True
            extractor._pipeline = mock_pipeline
            return extractor

    def test_extract_concerns(self, extractor, mock_pipeline):
        mock_pipeline.return_value = {
            "labels": ["privacy", "security", "cost"],
            "scores": [0.85, 0.72, 0.45],
        }

        concerns = extractor.extract("The system collects user data without consent.")
        assert len(concerns) == 2

        assert concerns[0].category == ConcernCategory.PRIVACY
        assert concerns[0].confidence == 0.85
        assert concerns[0].evidence is not None

        assert concerns[1].category == ConcernCategory.SECURITY
        assert concerns[1].confidence == 0.72

    def test_extract_filters_by_threshold(self, extractor, mock_pipeline):
        mock_pipeline.return_value = {
            "labels": ["privacy", "security", "cost"],
            "scores": [0.85, 0.45, 0.30],
        }

        concerns = extractor.extract("Text", threshold=0.5)
        assert len(concerns) == 1
        assert concerns[0].category == ConcernCategory.PRIVACY

    def test_extract_empty_text(self, extractor):
        concerns = extractor.extract("")
        assert concerns == []

    def test_extract_whitespace_only(self, extractor):
        concerns = extractor.extract("   ")
        assert concerns == []

    def test_extract_sorts_by_confidence(self, extractor, mock_pipeline):
        mock_pipeline.return_value = {
            "labels": ["cost", "privacy", "security"],
            "scores": [0.6, 0.9, 0.75],
        }

        concerns = extractor.extract("Text", threshold=0.5)
        assert concerns[0].category == ConcernCategory.PRIVACY
        assert concerns[1].category == ConcernCategory.SECURITY
        assert concerns[2].category == ConcernCategory.COST

    def test_extract_handles_unknown_categories(self, extractor, mock_pipeline):
        mock_pipeline.return_value = {
            "labels": ["unknown_category", "privacy"],
            "scores": [0.8, 0.7],
        }

        concerns = extractor.extract("Text", threshold=0.5)
        assert len(concerns) == 1
        assert concerns[0].category == ConcernCategory.PRIVACY

    def test_extract_ignores_other_category(self, extractor, mock_pipeline):
        mock_pipeline.return_value = {
            "labels": ["privacy", "security"],
            "scores": [0.8, 0.7],
        }

        concerns = extractor.extract("Text", threshold=0.5)
        assert len(concerns) == 2
        categories = {c.category for c in concerns}
        assert ConcernCategory.PRIVACY in categories
        assert ConcernCategory.SECURITY in categories
        assert ConcernCategory.OTHER not in categories

    def test_truncates_long_text(self, extractor, mock_pipeline):
        mock_pipeline.return_value = {
            "labels": ["privacy"],
            "scores": [0.9],
        }

        long_text = "x" * 5000
        concerns = extractor.extract(long_text)
        call_args = mock_pipeline.call_args[0][0]
        assert len(call_args) <= 1024


class TestGetConcernExtractor:
    def test_returns_instance(self):
        with patch("app.nlp.concerns.ConcernExtractor") as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            from app.nlp.concerns import get_concern_extractor
            extractor = get_concern_extractor()
            assert extractor == mock_instance
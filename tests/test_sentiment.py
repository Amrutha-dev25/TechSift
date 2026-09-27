import pytest
from unittest.mock import Mock, patch, MagicMock

from app.nlp.sentiment import SentimentAnalyzer
from app.nlp.taxonomy import SentimentLabel
from app.nlp.models import SentimentResult


class TestSentimentAnalyzer:
    @pytest.fixture
    def mock_pipeline(self):
        with patch("app.nlp.sentiment.pipeline") as mock:
            mock_instance = Mock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def analyzer(self, mock_pipeline):
        with patch("app.nlp.sentiment.AutoTokenizer.from_pretrained"), \
             patch("app.nlp.sentiment.AutoModelForSequenceClassification.from_pretrained"):
            analyzer = SentimentAnalyzer(model_name="test-model")
            analyzer._is_loaded = True
            analyzer._pipeline = mock_pipeline
            analyzer._label_to_sentiment = {
                "positive": SentimentLabel.POSITIVE,
                "neutral": SentimentLabel.NEUTRAL,
                "negative": SentimentLabel.NEGATIVE,
            }
            return analyzer

    def test_analyze_positive(self, analyzer, mock_pipeline):
        mock_pipeline.return_value = [[
            {"label": "positive", "score": 0.8},
            {"label": "neutral", "score": 0.1},
            {"label": "negative", "score": 0.1},
        ]]

        result = analyzer.analyze("This is amazing!")
        assert result.label == SentimentLabel.POSITIVE
        assert result.score == pytest.approx(0.7)
        assert result.confidence == 0.8

    def test_analyze_negative(self, analyzer, mock_pipeline):
        mock_pipeline.return_value = [[
            {"label": "negative", "score": 0.85},
            {"label": "neutral", "score": 0.1},
            {"label": "positive", "score": 0.05},
        ]]

        result = analyzer.analyze("This is terrible!")
        assert result.label == SentimentLabel.NEGATIVE
        assert result.score == pytest.approx(-0.8)
        assert result.confidence == 0.85

    def test_analyze_neutral(self, analyzer, mock_pipeline):
        mock_pipeline.return_value = [[
            {"label": "neutral", "score": 0.7},
            {"label": "positive", "score": 0.15},
            {"label": "negative", "score": 0.15},
        ]]

        result = analyzer.analyze("The company released an update.")
        assert result.label == SentimentLabel.NEUTRAL
        assert result.score == pytest.approx(0.0)
        assert result.confidence == 0.7

    def test_empty_text(self, analyzer):
        result = analyzer.analyze("")
        assert result.label == SentimentLabel.NEUTRAL
        assert result.score == 0.0
        assert result.confidence == 0.0

    def test_whitespace_only(self, analyzer):
        result = analyzer.analyze("   \n\t  ")
        assert result.label == SentimentLabel.NEUTRAL

    def test_truncates_long_text(self, analyzer, mock_pipeline):
        mock_pipeline.return_value = [[
            {"label": "positive", "score": 0.9},
            {"label": "neutral", "score": 0.05},
            {"label": "negative", "score": 0.05},
        ]]

        long_text = "x" * 10000
        result = analyzer.analyze(long_text)
        assert result.label == SentimentLabel.POSITIVE
        call_args = mock_pipeline.call_args[0][0]
        assert len(call_args) <= 4096


class TestGetSentimentAnalyzer:
    def test_returns_instance(self):
        with patch("app.nlp.sentiment.SentimentAnalyzer") as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            from app.nlp.sentiment import get_sentiment_analyzer
            analyzer = get_sentiment_analyzer()
            assert analyzer == mock_instance
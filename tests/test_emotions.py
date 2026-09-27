import pytest
from unittest.mock import Mock, patch

from app.nlp.emotions import EmotionAnalyzer
from app.nlp.taxonomy import EmotionLabel
from app.nlp.models import EmotionResult


class TestEmotionAnalyzer:
    @pytest.fixture
    def mock_pipeline(self):
        with patch("app.nlp.emotions.pipeline") as mock:
            mock_instance = Mock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def analyzer(self, mock_pipeline):
        with patch("app.nlp.emotions.AutoTokenizer.from_pretrained"), \
             patch("app.nlp.emotions.AutoModelForSequenceClassification.from_pretrained"):
            analyzer = EmotionAnalyzer(model_name="test-model")
            analyzer._is_loaded = True
            analyzer._pipeline = mock_pipeline
            return analyzer

    def test_analyze_returns_emotions(self, analyzer, mock_pipeline):
        mock_pipeline.return_value = [[
            {"label": "anger", "score": 0.8},
            {"label": "fear", "score": 0.6},
            {"label": "neutral", "score": 0.1},
        ]]

        emotions = analyzer.analyze("This is terrible and scary!")
        assert len(emotions) == 2
        assert emotions[0].label == EmotionLabel.ANGER
        assert emotions[0].confidence == 0.8
        assert emotions[1].label == EmotionLabel.FEAR
        assert emotions[1].confidence == 0.6

    def test_analyze_filters_by_threshold(self, analyzer, mock_pipeline):
        mock_pipeline.return_value = [[
            {"label": "anger", "score": 0.8},
            {"label": "fear", "score": 0.2},
            {"label": "neutral", "score": 0.1},
        ]]

        emotions = analyzer.analyze("Text", threshold=0.3)
        assert len(emotions) == 1
        assert emotions[0].label == EmotionLabel.ANGER

    def test_analyze_empty_text(self, analyzer):
        emotions = analyzer.analyze("")
        assert len(emotions) == 1
        assert emotions[0].label == EmotionLabel.NEUTRAL
        assert emotions[0].confidence == 1.0

    def test_analyze_whitespace_only(self, analyzer):
        emotions = analyzer.analyze("   ")
        assert len(emotions) == 1
        assert emotions[0].label == EmotionLabel.NEUTRAL

    def test_analyze_sorts_by_confidence(self, analyzer, mock_pipeline):
        mock_pipeline.return_value = [[
            {"label": "fear", "score": 0.6},
            {"label": "anger", "score": 0.8},
            {"label": "joy", "score": 0.4},
        ]]

        emotions = analyzer.analyze("Text", threshold=0.3)
        assert emotions[0].label == EmotionLabel.ANGER
        assert emotions[1].label == EmotionLabel.FEAR
        assert emotions[2].label == EmotionLabel.EXCITEMENT

    def test_analyze_handles_unknown_labels(self, analyzer, mock_pipeline):
        mock_pipeline.return_value = [[
            {"label": "sadness", "score": 0.7},
            {"label": "joy", "score": 0.6},
        ]]

        emotions = analyzer.analyze("Text")
        labels = [e.label for e in emotions]
        assert EmotionLabel.DISAPPOINTMENT in labels
        assert EmotionLabel.EXCITEMENT in labels

    def test_truncates_long_text(self, analyzer, mock_pipeline):
        mock_pipeline.return_value = [[
            {"label": "anger", "score": 0.9},
        ]]

        long_text = "x" * 2000
        emotions = analyzer.analyze(long_text)
        call_args = mock_pipeline.call_args[0][0]
        assert len(call_args) <= 512


class TestGetEmotionAnalyzer:
    def test_returns_instance(self):
        with patch("app.nlp.emotions.EmotionAnalyzer") as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            from app.nlp.emotions import get_emotion_analyzer
            analyzer = get_emotion_analyzer()
            assert analyzer == mock_instance
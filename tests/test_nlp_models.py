import pytest
from pydantic import ValidationError

from app.nlp.models import (
    SentimentResult,
    Entity,
    EmotionResult,
    ConcernResult,
    DocumentAnalysis,
    AnalyzedDocument,
    ModelMetadata,
    NLPAnalysisFailure,
)
from app.nlp.taxonomy import (
    SentimentLabel,
    EmotionLabel,
    ConcernCategory,
    EntityType,
)


class TestSentimentResult:
    def test_valid_positive(self):
        result = SentimentResult(label="positive", score=0.8, confidence=0.9)
        assert result.label == SentimentLabel.POSITIVE
        assert result.score == 0.8
        assert result.confidence == 0.9

    def test_valid_negative(self):
        result = SentimentResult(label="negative", score=-0.7, confidence=0.85)
        assert result.label == SentimentLabel.NEGATIVE
        assert result.score == -0.7

    def test_valid_neutral(self):
        result = SentimentResult(label="neutral", score=0.0, confidence=0.6)
        assert result.label == SentimentLabel.NEUTRAL
        assert result.score == 0.0

    def test_score_bounds(self):
        with pytest.raises(ValidationError):
            SentimentResult(label="positive", score=1.5, confidence=0.9)
        with pytest.raises(ValidationError):
            SentimentResult(label="positive", score=-1.5, confidence=0.9)

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            SentimentResult(label="positive", score=0.5, confidence=1.5)
        with pytest.raises(ValidationError):
            SentimentResult(label="positive", score=0.5, confidence=-0.1)


class TestEntity:
    def test_valid_entity(self):
        entity = Entity(text="Microsoft", entity_type="ORG", confidence=0.97)
        assert entity.text == "Microsoft"
        assert entity.entity_type == EntityType.ORG
        assert entity.confidence == 0.97

    def test_entity_with_canonical_form(self):
        entity = Entity(
            text="Copilot",
            entity_type="PRODUCT",
            surface_form="Copilot",
            canonical_form="GitHub Copilot",
        )
        assert entity.canonical_form == "GitHub Copilot"

    def test_entity_type_case_insensitive(self):
        entity = Entity(text="test", entity_type="org")
        assert entity.entity_type == EntityType.ORG

    def test_invalid_entity_type(self):
        with pytest.raises(ValidationError):
            Entity(text="test", entity_type="INVALID_TYPE")


class TestEmotionResult:
    def test_valid_emotion(self):
        emotion = EmotionResult(label="concern", confidence=0.78)
        assert emotion.label == EmotionLabel.CONCERN
        assert emotion.confidence == 0.78

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            EmotionResult(label="anger", confidence=1.5)
        with pytest.raises(ValidationError):
            EmotionResult(label="anger", confidence=-0.1)


class TestConcernResult:
    def test_valid_concern(self):
        concern = ConcernResult(category="privacy", confidence=0.84, evidence="user data collected")
        assert concern.category == ConcernCategory.PRIVACY
        assert concern.confidence == 0.84
        assert concern.evidence == "user data collected"

    def test_concern_without_evidence(self):
        concern = ConcernResult(category="security", confidence=0.75)
        assert concern.evidence is None

    def test_invalid_category(self):
        with pytest.raises(ValidationError):
            ConcernResult(category="invalid_category", confidence=0.5)


class TestDocumentAnalysis:
    def test_minimal_analysis(self):
        sentiment = SentimentResult(label="neutral", score=0.0, confidence=0.5)
        analysis = DocumentAnalysis(sentiment=sentiment)
        assert analysis.analysis_version == "phase2-v1"
        assert analysis.entities == []
        assert analysis.technology_entities == []
        assert analysis.emotions == []
        assert analysis.concerns == []

    def test_full_analysis(self):
        sentiment = SentimentResult(label="negative", score=-0.5, confidence=0.8)
        entities = [Entity(text="Microsoft", entity_type="ORG", confidence=0.9)]
        tech_entities = [Entity(text="Copilot", entity_type="PRODUCT", confidence=0.95, canonical_form="GitHub Copilot")]
        emotions = [EmotionResult(label="concern", confidence=0.7)]
        concerns = [ConcernResult(category="privacy", confidence=0.8, evidence="data collection")]

        analysis = DocumentAnalysis(
            entities=entities,
            technology_entities=tech_entities,
            sentiment=sentiment,
            emotions=emotions,
            concerns=concerns,
            models=ModelMetadata(
                sentiment="test-model",
                ner="test-ner",
                emotion="test-emotion",
                concern="test-concern",
            ),
        )
        assert len(analysis.entities) == 1
        assert len(analysis.technology_entities) == 1
        assert len(analysis.emotions) == 1
        assert len(analysis.concerns) == 1
        assert analysis.models is not None

    def test_sentiment_score_range(self):
        sentiment = SentimentResult(label="positive", score=1.0, confidence=1.0)
        analysis = DocumentAnalysis(sentiment=sentiment)
        assert -1.0 <= analysis.sentiment.score <= 1.0

    def test_confidence_range(self):
        sentiment = SentimentResult(label="positive", score=0.5, confidence=0.9)
        analysis = DocumentAnalysis(sentiment=sentiment)
        assert 0.0 <= analysis.sentiment.confidence <= 1.0


class TestAnalyzedDocument:
    def test_from_canonical_plus_analysis(self):
        sentiment = SentimentResult(label="positive", score=0.6, confidence=0.8)
        analysis = DocumentAnalysis(sentiment=sentiment)

        doc = AnalyzedDocument(
            document_id="test123",
            source_id="test",
            source_name="Test",
            source_type="rss",
            title="Test Title",
            content="Test content",
            url="https://example.com",
            retrieved_at="2026-01-15T10:00:00Z",
            content_hash="abc123",
            canonical_url="https://example.com",
            analysis=analysis,
        )
        assert doc.document_id == "test123"
        assert doc.analysis.sentiment.label == SentimentLabel.POSITIVE


class TestNLPAnalysisFailure:
    def test_failure_record(self):
        failure = NLPAnalysisFailure(
            document_id="test123",
            timestamp="2026-01-15T10:00:00Z",
            stage="sentiment",
            error_type="ValueError",
            error_message="Model output invalid",
        )
        assert failure.document_id == "test123"
        assert failure.stage == "sentiment"
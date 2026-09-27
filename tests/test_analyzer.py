import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from pathlib import Path

from app.models.document import CanonicalDocument, SourceType
from app.nlp.analyzer import DocumentAnalyzer, AnalysisPipeline, create_pipeline
from app.nlp.models import (
    DocumentAnalysis,
    AnalyzedDocument,
    SentimentResult,
    Entity,
    EmotionResult,
    ConcernResult,
    ModelMetadata,
)
from app.nlp.taxonomy import SentimentLabel, EntityType, EmotionLabel, ConcernCategory


class TestDocumentAnalyzer:
    @pytest.fixture
    def mock_analyzers(self):
        with patch("app.nlp.analyzer.get_sentiment_analyzer") as mock_sentiment, \
             patch("app.nlp.analyzer.get_entity_extractor") as mock_entity, \
             patch("app.nlp.analyzer.get_emotion_analyzer") as mock_emotion, \
             patch("app.nlp.analyzer.get_concern_extractor") as mock_concern:

            sentiment_analyzer = Mock()
            sentiment_analyzer.analyze.return_value = SentimentResult(
                label=SentimentLabel.NEGATIVE,
                score=-0.7,
                confidence=0.85,
            )
            mock_sentiment.return_value = sentiment_analyzer

            entity_extractor = Mock()
            entity_extractor.extract.return_value = [
                Entity(text="Microsoft", entity_type=EntityType.ORG, confidence=0.9),
                Entity(text="Copilot", entity_type=EntityType.PRODUCT, confidence=0.95),
            ]
            entity_extractor.extract_technology_entities.return_value = [
                Entity(text="Copilot", entity_type=EntityType.PRODUCT, confidence=0.95),
            ]
            mock_entity.return_value = entity_extractor

            emotion_analyzer = Mock()
            emotion_analyzer.analyze.return_value = [
                EmotionResult(label=EmotionLabel.CONCERN, confidence=0.78),
            ]
            mock_emotion.return_value = emotion_analyzer

            concern_extractor = Mock()
            concern_extractor.extract.return_value = [
                ConcernResult(category=ConcernCategory.PRIVACY, confidence=0.84, evidence="user data"),
            ]
            mock_concern.return_value = concern_extractor

            yield {
                "sentiment": sentiment_analyzer,
                "entity": entity_extractor,
                "emotion": emotion_analyzer,
                "concern": concern_extractor,
            }

    @pytest.fixture
    def analyzer(self, mock_analyzers):
        return DocumentAnalyzer()

    @pytest.fixture
    def sample_document(self):
        return CanonicalDocument(
            document_id="test123",
            source_id="test",
            source_name="Test Source",
            source_type=SourceType.RSS,
            title="Privacy concerns over AI data collection",
            content="Researchers warned that the system stores sensitive user information without proper consent.",
            url="https://example.com/test",
            author="Test Author",
            published_at=datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            retrieved_at=datetime(2026, 1, 15, 11, 0, 0, tzinfo=timezone.utc),
            content_hash="a" * 64,
            canonical_url="https://example.com/test",
        )

    def test_analyze_returns_document_analysis(self, analyzer, sample_document, mock_analyzers):
        analysis = analyzer.analyze(sample_document)

        assert isinstance(analysis, DocumentAnalysis)
        assert analysis.sentiment.label == SentimentLabel.NEGATIVE
        assert analysis.sentiment.score == -0.7
        assert len(analysis.entities) == 2
        assert len(analysis.technology_entities) == 1
        assert len(analysis.emotions) == 1
        assert analysis.emotions[0].label == EmotionLabel.CONCERN
        assert len(analysis.concerns) == 1
        assert analysis.concerns[0].category == ConcernCategory.PRIVACY
        assert analysis.analysis_version == "phase2-v2"
        assert analysis.models is not None

    def test_analyze_calls_all_components(self, analyzer, sample_document, mock_analyzers):
        analyzer.analyze(sample_document)

        mock_analyzers["sentiment"].analyze.assert_called_once()
        mock_analyzers["entity"].extract.assert_called_once()
        mock_analyzers["entity"].extract_technology_entities.assert_called_once()
        mock_analyzers["emotion"].analyze.assert_called_once()
        mock_analyzers["concern"].extract.assert_called_once()

    def test_analyze_prepares_text_correctly(self, analyzer, sample_document, mock_analyzers):
        analyzer.analyze(sample_document)

        call_args = mock_analyzers["sentiment"].analyze.call_args[0][0]
        assert "Privacy concerns over AI data collection" in call_args
        assert "Researchers warned" in call_args


class TestAnalysisPipeline:
    @pytest.fixture
    def mock_store(self):
        with patch("app.nlp.analyzer.JSONLStore") as mock:
            mock_instance = Mock()
            mock_instance.load_all.return_value = []
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def pipeline(self, mock_store, tmp_path):
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"
        failure_path = tmp_path / "failures.jsonl"
        input_path.write_text("")  # Create empty input file

        pipeline = AnalysisPipeline(
            input_path=input_path,
            output_path=output_path,
            failure_path=failure_path,
        )
        pipeline._store = mock_store
        return pipeline

    @pytest.fixture
    def sample_documents(self):
        return [
            CanonicalDocument(
                document_id=f"doc{i}",
                source_id="test",
                source_name="Test",
                source_type=SourceType.RSS,
                title=f"Title {i}",
                content=f"Content {i}",
                url=f"https://example.com/{i}",
                retrieved_at=datetime.now(timezone.utc),
                content_hash=f"{i:064x}",
                canonical_url=f"https://example.com/{i}",
            )
            for i in range(3)
        ]

    def test_run_processes_documents(self, pipeline, sample_documents):
        pipeline._store.load_all.return_value = sample_documents

        with patch.object(pipeline, "process_document") as mock_process:
            mock_process.return_value = (
                AnalyzedDocument(
                    document_id="doc0",
                    source_id="test",
                    source_name="Test",
                    source_type="rss",
                    title="Title 0",
                    content="Content 0",
                    url="https://example.com/0",
                    retrieved_at=datetime.now(timezone.utc),
                    content_hash="hash0",
                    canonical_url="https://example.com/0",
                    analysis=DocumentAnalysis(
                        sentiment=SentimentResult(label=SentimentLabel.NEUTRAL, score=0.0, confidence=0.5),
                    ),
                ),
                None,
            )

            stats = pipeline.run(limit=2)

        assert stats["attempted"] == 2
        assert stats["success"] == 2

    def test_run_respects_limit(self, pipeline, sample_documents):
        pipeline._store.load_all.return_value = sample_documents

        with patch.object(pipeline, "process_document") as mock_process:
            mock_process.return_value = (None, None)

            stats = pipeline.run(limit=1)

        assert stats["attempted"] == 1

    def test_run_skips_existing(self, pipeline, sample_documents):
        pipeline._store.load_all.return_value = sample_documents
        pipeline._existing_analyses.add("doc0:phase2-v1")

        def mock_process_document(doc, force=False):
            if doc.document_id == "doc0":
                return None, None  # skipped
            return (
                AnalyzedDocument(
                    document_id=doc.document_id,
                    source_id="test",
                    source_name="Test",
                    source_type="rss",
                    title=doc.title,
                    content=doc.content,
                    url=doc.url,
                    retrieved_at=doc.retrieved_at,
                    content_hash=doc.content_hash,
                    canonical_url=doc.canonical_url,
                    analysis=DocumentAnalysis(
                        sentiment=SentimentResult(label=SentimentLabel.NEUTRAL, score=0.0, confidence=0.5),
                    ),
                ),
                None,
            )

        with patch.object(pipeline, "process_document", side_effect=mock_process_document):
            stats = pipeline.run(limit=3)

        assert stats["skipped"] == 1
        assert stats["success"] == 2
        assert stats["attempted"] == 3


class TestCreatePipeline:
    def test_creates_with_defaults(self):
        with patch("app.nlp.analyzer.AnalysisPipeline") as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance

            pipeline = create_pipeline()

            mock_class.assert_called_once()
            args, kwargs = mock_class.call_args
            assert args[0].name == "documents.jsonl"
            assert args[1].name == "analyzed_documents.jsonl"
            assert args[2].name == "nlp_analysis_failures.jsonl"

    def test_creates_with_custom_paths(self, tmp_path):
        input_path = tmp_path / "custom_in.jsonl"
        output_path = tmp_path / "custom_out.jsonl"
        failure_path = tmp_path / "custom_fail.jsonl"

        with patch("app.nlp.analyzer.AnalysisPipeline") as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance

            pipeline = create_pipeline(input_path, output_path, failure_path)

            mock_class.assert_called_once()
            args, kwargs = mock_class.call_args
            assert args[0] == input_path
            assert args[1] == output_path
            assert args[2] == failure_path
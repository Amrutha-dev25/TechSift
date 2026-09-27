import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config.settings import settings
from app.models.document import CanonicalDocument
from app.nlp.models import (
    AnalyzedDocument,
    DocumentAnalysis,
    ModelMetadata,
    NLPAnalysisFailure,
)
from app.nlp.preprocessing import prepare_text_for_analysis
from app.nlp.sentiment import SentimentAnalyzer, get_sentiment_analyzer
from app.nlp.entities import EntityExtractor, get_entity_extractor
from app.nlp.emotions import EmotionAnalyzer, get_emotion_analyzer
from app.nlp.concerns import ConcernExtractor, get_concern_extractor
from app.storage.jsonl_store import JSONLStore

logger = logging.getLogger(__name__)


class DocumentAnalyzer:
    def __init__(
        self,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None,
        entity_extractor: Optional[EntityExtractor] = None,
        emotion_analyzer: Optional[EmotionAnalyzer] = None,
        concern_extractor: Optional[ConcernExtractor] = None,
    ):
        self.sentiment_analyzer = sentiment_analyzer or get_sentiment_analyzer()
        self.entity_extractor = entity_extractor or get_entity_extractor()
        self.emotion_analyzer = emotion_analyzer or get_emotion_analyzer()
        self.concern_extractor = concern_extractor or get_concern_extractor()

        self._models_metadata = ModelMetadata(
            sentiment=settings.nlp_models.sentiment_model_name,
            ner=settings.nlp_models.ner_model_name,
            emotion=settings.nlp_models.emotion_model_name,
            concern=settings.nlp_models.concern_model_name,
        )

    def analyze(self, document: CanonicalDocument) -> DocumentAnalysis:
        text = prepare_text_for_analysis(document.title, document.content)

        logger.debug("Analyzing document %s", document.document_id)

        sentiment = self.sentiment_analyzer.analyze(text)
        logger.debug("Sentiment: %s (score=%.2f, conf=%.2f)", sentiment.label, sentiment.score, sentiment.confidence)

        entities = self.entity_extractor.extract(text)
        logger.debug("Extracted %d entities", len(entities))

        technology_entities = self.entity_extractor.extract_technology_entities(entities)
        logger.debug("Identified %d technology entities", len(technology_entities))

        emotions = self.emotion_analyzer.analyze(text)
        logger.debug("Extracted %d emotions", len(emotions))

        concerns = self.concern_extractor.extract(text)
        logger.debug("Extracted %d concerns", len(concerns))

        analysis = DocumentAnalysis(
            entities=entities,
            technology_entities=technology_entities,
            sentiment=sentiment,
            emotions=emotions,
            concerns=concerns,
            analysis_version="phase2-v1",
            models=self._models_metadata,
        )

        return analysis


class AnalysisPipeline:
    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        failure_path: Path,
        analyzer: Optional[DocumentAnalyzer] = None,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.failure_path = failure_path
        self.analyzer = analyzer or DocumentAnalyzer()
        self._store = JSONLStore(
            processed_path=input_path,
            raw_dir=settings.raw_data_dir,
            failed_dir=failure_path.parent,
        )
        self._existing_analyses: set[str] = set()
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.output_path.exists():
            return

        try:
            with self.output_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        import json
                        doc = json.loads(line)
                        if "document_id" in doc and "analysis" in doc:
                            key = f"{doc['document_id']}:{doc['analysis'].get('analysis_version', 'phase2-v1')}"
                            self._existing_analyses.add(key)
                    except Exception:
                        continue
        except Exception as e:
            logger.warning("Failed to load existing analyses: %s", e)

    def _save_failure(self, failure: NLPAnalysisFailure) -> None:
        try:
            self.failure_path.parent.mkdir(parents=True, exist_ok=True)
            with self.failure_path.open("a", encoding="utf-8") as f:
                f.write(failure.model_dump_json() + "\n")
        except Exception as e:
            logger.error("Failed to save failure record: %s", e)

    def process_document(self, document: CanonicalDocument, force: bool = False) -> tuple[Optional[AnalyzedDocument], Optional[NLPAnalysisFailure]]:
        analysis_key = f"{document.document_id}:phase2-v1"

        if not force and analysis_key in self._existing_analyses:
            logger.debug("Document %s already analyzed, skipping", document.document_id)
            return None, None

        try:
            analysis = self.analyzer.analyze(document)

            analyzed_doc = AnalyzedDocument(
                document_id=document.document_id,
                source_id=document.source_id,
                source_name=document.source_name,
                source_type=document.source_type.value,
                title=document.title,
                content=document.content,
                url=document.url,
                author=document.author,
                published_at=document.published_at,
                retrieved_at=document.retrieved_at,
                content_hash=document.content_hash,
                canonical_url=document.canonical_url,
                analysis=analysis,
            )

            self._existing_analyses.add(analysis_key)
            return analyzed_doc, None

        except Exception as e:
            logger.error("Failed to analyze document %s: %s", document.document_id, e)
            failure = NLPAnalysisFailure(
                document_id=document.document_id,
                timestamp=datetime.now().astimezone(),
                stage="analysis",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            self._save_failure(failure)
            return None, failure

    def run(
        self,
        limit: Optional[int] = None,
        force: bool = False,
    ) -> dict:
        if not self.input_path.exists():
            logger.error("Input file not found: %s", self.input_path)
            return {
                "attempted": 0,
                "success": 0,
                "failed": 0,
                "skipped": 0,
            }

        logger.info("Starting analysis pipeline")
        logger.info("Input: %s", self.input_path)
        logger.info("Output: %s", self.output_path)
        logger.info("Failures: %s", self.failure_path)

        documents = self._store.load_all()
        logger.info("Loaded %d documents from %s", len(documents), self.input_path)

        if limit:
            documents = documents[:limit]
            logger.info("Limited to %d documents", limit)

        stats = {
            "attempted": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "sentiment_counts": {"positive": 0, "neutral": 0, "negative": 0},
            "entities_count": 0,
            "tech_entities_count": 0,
            "concerns_count": 0,
            "emotions_count": 0,
        }

        for doc in documents:
            stats["attempted"] += 1

            analyzed, failure = self.process_document(doc, force=force)

            if failure:
                stats["failed"] += 1
                continue

            if analyzed is None:
                stats["skipped"] += 1
                continue

            try:
                with self.output_path.open("a", encoding="utf-8") as f:
                    f.write(analyzed.model_dump_json() + "\n")

                stats["success"] += 1
                stats["sentiment_counts"][analyzed.analysis.sentiment.label.value] += 1
                stats["entities_count"] += 1 if analyzed.analysis.entities else 0
                stats["tech_entities_count"] += 1 if analyzed.analysis.technology_entities else 0
                stats["concerns_count"] += 1 if analyzed.analysis.concerns else 0
                stats["emotions_count"] += 1 if analyzed.analysis.emotions else 0

            except Exception as e:
                logger.error("Failed to save analyzed document: %s", e)
                stats["failed"] += 1

        logger.info("Analysis complete: %s", stats)
        return stats


def create_analyzer() -> DocumentAnalyzer:
    return DocumentAnalyzer()


def create_pipeline(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    failure_path: Optional[Path] = None,
) -> AnalysisPipeline:
    input_path = input_path or (settings.processed_data_dir / "documents.jsonl")
    output_path = output_path or (settings.processed_data_dir / "analyzed_documents.jsonl")
    failure_path = failure_path or (settings.failed_data_dir / "nlp_analysis_failures.jsonl")

    return AnalysisPipeline(input_path, output_path, failure_path)
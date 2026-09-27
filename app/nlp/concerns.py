import logging
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from app.config.settings import settings
from app.nlp.models import ConcernResult
from app.nlp.taxonomy import ConcernCategory, get_concern_keywords
from app.nlp.preprocessing import extract_evidence_span

logger = logging.getLogger(__name__)


class ConcernExtractor:
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name or settings.nlp_models.concern_model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipeline: Optional[pipeline] = None
        self._is_loaded = False
        self._hypothesis_template = "This text is about {}."

    def _load_model(self) -> None:
        if self._is_loaded:
            return

        logger.info("Loading concern model: %s on %s", self.model_name, self.device)

        try:
            self._pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if self.device == "cuda" else -1,
            )
            self._is_loaded = True
            logger.info("Concern model loaded successfully")
        except Exception as e:
            logger.error("Failed to load concern model: %s", e)
            raise

    def extract(self, text: str, threshold: float = 0.5) -> list[ConcernResult]:
        if not text or not text.strip():
            return []

        self._load_model()

        try:
            candidate_labels = [cat.value for cat in ConcernCategory if cat != ConcernCategory.OTHER]

            truncated_text = text[:1024]

            result = self._pipeline(
                truncated_text,
                candidate_labels=candidate_labels,
                hypothesis_template=self._hypothesis_template,
                multi_label=True,
            )

            concerns = []
            labels = result["labels"]
            scores = result["scores"]

            for label, score in zip(labels, scores):
                if score < threshold:
                    continue

                try:
                    category = ConcernCategory(label)
                except ValueError:
                    continue

                keywords = get_concern_keywords(category)
                evidence = extract_evidence_span(text, keywords)

                concerns.append(ConcernResult(
                    category=category,
                    confidence=score,
                    evidence=evidence,
                ))

            concerns.sort(key=lambda x: x.confidence, reverse=True)
            return concerns

        except Exception as e:
            logger.error("Concern extraction failed: %s", e)
            return []


def get_concern_extractor() -> ConcernExtractor:
    return ConcernExtractor()
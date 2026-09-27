import logging
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from app.config.settings import settings
from app.nlp.models import SentimentResult
from app.nlp.taxonomy import SentimentLabel

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name or settings.nlp_models.sentiment_model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipeline: Optional[pipeline] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForSequenceClassification] = None
        self._label_mapping: dict[int, SentimentLabel] = {}
        self._is_loaded = False

    def _load_model(self) -> None:
        if self._is_loaded:
            return

        logger.info("Loading sentiment model: %s on %s", self.model_name, self.device)

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()

            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self._model,
                tokenizer=self._tokenizer,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True,
            )

            config = self._model.config
            if hasattr(config, "id2label") and config.id2label:
                for idx, label in config.id2label.items():
                    label_lower = label.lower()
                    if "positive" in label_lower or label_lower == "pos":
                        self._label_mapping[idx] = SentimentLabel.POSITIVE
                    elif "negative" in label_lower or label_lower == "neg":
                        self._label_mapping[idx] = SentimentLabel.NEGATIVE
                    else:
                        self._label_mapping[idx] = SentimentLabel.NEUTRAL
            else:
                self._label_mapping = {
                    0: SentimentLabel.NEGATIVE,
                    1: SentimentLabel.NEUTRAL,
                    2: SentimentLabel.POSITIVE,
                }

            self._is_loaded = True
            logger.info("Sentiment model loaded successfully")

        except Exception as e:
            logger.error("Failed to load sentiment model: %s", e)
            raise

    def analyze(self, text: str) -> SentimentResult:
        if not text or not text.strip():
            return SentimentResult(
                label=SentimentLabel.NEUTRAL,
                score=0.0,
                confidence=0.0,
            )

        self._load_model()

        try:
            truncated_text = text[:4096]
            results = self._pipeline(truncated_text)
            
            scores = {}
            if results and isinstance(results[0], dict):
                results = [results]
                
            for result in results[0]:
                label_idx = int(result["label"].split("_")[-1]) if "_" in result["label"] else 0
                if "label" in result and result["label"].isdigit():
                    label_idx = int(result["label"])
                elif "label" in result and "LABEL_" in result["label"]:
                    label_idx = int(result["label"].replace("LABEL_", ""))

                label = self._label_mapping.get(label_idx, SentimentLabel.NEUTRAL)
                scores[label] = result["score"]

            positive_prob = scores.get(SentimentLabel.POSITIVE, 0.0)
            negative_prob = scores.get(SentimentLabel.NEGATIVE, 0.0)
            neutral_prob = scores.get(SentimentLabel.NEUTRAL, 0.0)

            sentiment_score = positive_prob - negative_prob

            max_label = max(scores, key=scores.get) if scores else SentimentLabel.NEUTRAL
            confidence = scores.get(max_label, 0.0)

            return SentimentResult(
                label=max_label,
                score=sentiment_score,
                confidence=confidence,
            )

        except Exception as e:
            logger.error("Sentiment analysis failed: %s", e)
            return SentimentResult(
                label=SentimentLabel.NEUTRAL,
                score=0.0,
                confidence=0.0,
            )


def get_sentiment_analyzer() -> SentimentAnalyzer:
    return SentimentAnalyzer()
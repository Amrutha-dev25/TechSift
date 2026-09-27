import logging
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from app.config.settings import settings
from app.nlp.models import EmotionResult
from app.nlp.taxonomy import EmotionLabel

logger = logging.getLogger(__name__)


class EmotionAnalyzer:
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name or settings.nlp_models.emotion_model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipeline: Optional[pipeline] = None
        self._is_loaded = False

    def _load_model(self) -> None:
        if self._is_loaded:
            return

        logger.info("Loading emotion model: %s on %s", self.model_name, self.device)

        try:
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True,
            )
            self._is_loaded = True
            logger.info("Emotion model loaded successfully")
        except Exception as e:
            logger.error("Failed to load emotion model: %s", e)
            raise

    def analyze(self, text: str, threshold: float = 0.3) -> list[EmotionResult]:
        if not text or not text.strip():
            return [EmotionResult(label=EmotionLabel.NEUTRAL, confidence=1.0)]

        self._load_model()

        try:
            truncated_text = text[:512]
            results = self._pipeline(truncated_text)

            emotions = []
            if results and isinstance(results[0], dict):
                results = [results]
                
            for result in results[0]:
                label_str = result["label"].lower()
                score = result["score"]

                if score < threshold:
                    continue

                try:
                    emotion_label = EmotionLabel(label_str)
                except ValueError:
                    if "anger" in label_str:
                        emotion_label = EmotionLabel.ANGER
                    elif "fear" in label_str:
                        emotion_label = EmotionLabel.FEAR
                    elif "sadness" in label_str or "disappointment" in label_str:
                        emotion_label = EmotionLabel.DISAPPOINTMENT
                    elif "joy" in label_str or "happy" in label_str or "excitement" in label_str:
                        emotion_label = EmotionLabel.EXCITEMENT
                    elif "surprise" in label_str:
                        emotion_label = EmotionLabel.CONFUSION
                    elif "neutral" in label_str:
                        emotion_label = EmotionLabel.NEUTRAL
                    else:
                        continue

                emotions.append(EmotionResult(label=emotion_label, confidence=score))

            if not emotions:
                emotions.append(EmotionResult(label=EmotionLabel.NEUTRAL, confidence=1.0))

            emotions.sort(key=lambda x: x.confidence, reverse=True)
            return emotions

        except Exception as e:
            logger.error("Emotion analysis failed: %s", e)
            return [EmotionResult(label=EmotionLabel.NEUTRAL, confidence=1.0)]


def get_emotion_analyzer() -> EmotionAnalyzer:
    return EmotionAnalyzer()
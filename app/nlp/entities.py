import logging
from typing import Optional

import spacy
from spacy.tokens import Doc

from app.config.settings import settings
from app.nlp.models import Entity
from app.nlp.taxonomy import EntityType
from app.nlp.preprocessing import (
    canonicalize_tech_entity,
    is_technology_entity,
    normalize_entity_text,
)

logger = logging.getLogger(__name__)

SPACY_LABEL_MAP = {
    "ORG": EntityType.ORG,
    "PERSON": EntityType.PERSON,
    "PRODUCT": EntityType.PRODUCT,
    "EVENT": EntityType.EVENT,
    "GPE": EntityType.OTHER,
    "LOC": EntityType.OTHER,
    "NORP": EntityType.OTHER,
    "FAC": EntityType.OTHER,
    "LAW": EntityType.OTHER,
    "LANGUAGE": EntityType.OTHER,
    "WORK_OF_ART": EntityType.OTHER,
    "DATE": EntityType.OTHER,
    "TIME": EntityType.OTHER,
    "PERCENT": EntityType.OTHER,
    "MONEY": EntityType.OTHER,
    "QUANTITY": EntityType.OTHER,
    "ORDINAL": EntityType.OTHER,
    "CARDINAL": EntityType.OTHER,
}

NOISY_ENTITY_LABELS = {
    "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"
}

NOISY_ENTITY_TEXTS = {
    "ml", "llm", "api", "sdk", "ui", "ux", "os", "db", "http", "https",
    "www", "com", "org", "net", "io", "co", "inc", "ltd", "llc", "corp",
}


class EntityExtractor:
    def __init__(
        self,
        model_name: Optional[str] = None,
    ):
        self.model_name = model_name or settings.nlp_models.ner_model_name
        self._nlp: Optional[spacy.Language] = None
        self._is_loaded = False

    def _load_model(self) -> None:
        if self._is_loaded:
            return

        logger.info("Loading NER model: %s", self.model_name)

        try:
            self._nlp = spacy.load(self.model_name)
            self._is_loaded = True
            logger.info("NER model loaded successfully")
        except OSError:
            logger.warning("spaCy model %s not found, downloading...", self.model_name)
            try:
                spacy.cli.download(self.model_name)
                self._nlp = spacy.load(self.model_name)
                self._is_loaded = True
                logger.info("NER model downloaded and loaded successfully")
            except Exception as e:
                logger.error("Failed to load NER model: %s", e)
                raise
        except Exception as e:
            logger.error("Failed to load NER model: %s", e)
            raise

    def _is_noisy_entity(self, text: str, label: str) -> bool:
        normalized = normalize_entity_text(text).lower()
        if label in NOISY_ENTITY_LABELS:
            return True
        if normalized in NOISY_ENTITY_TEXTS:
            return True
        if len(normalized) < 2:
            return True
        return False

    def extract(self, text: str) -> list[Entity]:
        if not text or not text.strip():
            return []

        self._load_model()

        try:
            doc: Doc = self._nlp(text[:100000])

            entities = []
            seen_texts = set()

            for ent in doc.ents:
                if self._is_noisy_entity(ent.text, ent.label_):
                    continue

                entity_type = SPACY_LABEL_MAP.get(ent.label_, EntityType.OTHER)
                normalized_text = normalize_entity_text(ent.text)

                text_key = normalized_text.lower()
                if text_key in seen_texts:
                    continue
                seen_texts.add(text_key)

                canonical = canonicalize_tech_entity(normalized_text)

                entity = Entity(
                    text=normalized_text,
                    entity_type=entity_type,
                    confidence=None,
                    surface_form=ent.text,
                    canonical_form=canonical,
                )
                entities.append(entity)

            return entities

        except Exception as e:
            logger.error("Entity extraction failed: %s", e)
            return []

    def extract_technology_entities(self, entities: list[Entity]) -> list[Entity]:
        tech_entities = []
        seen_texts = set()
        for entity in entities:
            if is_technology_entity(entity.text, entity.entity_type):
                text_key = entity.text.lower()
                if text_key in seen_texts:
                    continue
                seen_texts.add(text_key)
                # Prefer PRODUCT/TECHNOLOGY type over ORG/OTHER for tech entities
                tech_entities.append(entity)
        return tech_entities


def get_entity_extractor() -> EntityExtractor:
    return EntityExtractor()
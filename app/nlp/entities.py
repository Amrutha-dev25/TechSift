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

    def extract(self, text: str) -> list[Entity]:
        if not text or not text.strip():
            return []

        self._load_model()

        try:
            doc: Doc = self._nlp(text[:100000])

            entities = []
            seen_spans = set()

            for ent in doc.ents:
                span_key = (ent.start_char, ent.end_char)
                if span_key in seen_spans:
                    continue
                seen_spans.add(span_key)

                entity_type = SPACY_LABEL_MAP.get(ent.label_, EntityType.OTHER)
                normalized_text = normalize_entity_text(ent.text)

                if len(normalized_text) < 2:
                    continue

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
        for entity in entities:
            if is_technology_entity(entity.text, entity.entity_type):
                tech_entities.append(entity)
        return tech_entities


def get_entity_extractor() -> EntityExtractor:
    return EntityExtractor()
import pytest
from unittest.mock import Mock, patch, MagicMock

from app.nlp.entities import EntityExtractor
from app.nlp.taxonomy import EntityType
from app.nlp.models import Entity


class TestEntityExtractor:
    @pytest.fixture
    def mock_nlp(self):
        mock_doc = Mock()
        mock_ent1 = Mock()
        mock_ent1.text = "Microsoft"
        mock_ent1.label_ = "ORG"
        mock_ent1.start_char = 0
        mock_ent1.end_char = 9

        mock_ent2 = Mock()
        mock_ent2.text = "Copilot"
        mock_ent2.label_ = "PRODUCT"
        mock_ent2.start_char = 20
        mock_ent2.end_char = 27

        mock_doc.ents = [mock_ent1, mock_ent2]

        mock_nlp = Mock()
        mock_nlp.return_value = mock_doc

        return mock_nlp

    @pytest.fixture
    def extractor(self, mock_nlp):
        with patch("app.nlp.entities.spacy.load", return_value=mock_nlp):
            extractor = EntityExtractor(model_name="test-model")
            extractor._is_loaded = True
            extractor._nlp = mock_nlp
            return extractor

    def test_extract_entities(self, extractor):
        entities = extractor.extract("Microsoft announced Copilot")
        assert len(entities) == 2

        assert entities[0].text == "Microsoft"
        assert entities[0].entity_type == EntityType.ORG
        assert entities[0].surface_form == "Microsoft"

        assert entities[1].text == "Copilot"
        assert entities[1].entity_type == EntityType.PRODUCT
        assert entities[1].surface_form == "Copilot"

    def test_extract_empty_text(self, extractor):
        entities = extractor.extract("")
        assert entities == []

    def test_extract_whitespace_only(self, extractor):
        entities = extractor.extract("   ")
        assert entities == []

    def test_deduplicates_overlapping_spans(self, extractor):
        mock_doc = Mock()
        mock_ent1 = Mock()
        mock_ent1.text = "Microsoft"
        mock_ent1.label_ = "ORG"
        mock_ent1.start_char = 0
        mock_ent1.end_char = 9

        mock_ent2 = Mock()
        mock_ent2.text = "Microsoft"
        mock_ent2.label_ = "ORG"
        mock_ent2.start_char = 0
        mock_ent2.end_char = 9

        mock_doc.ents = [mock_ent1, mock_ent2]
        extractor._nlp.return_value = mock_doc

        entities = extractor.extract("Microsoft")
        assert len(entities) == 1

    def test_skips_short_entities(self, extractor):
        mock_doc = Mock()
        mock_ent = Mock()
        mock_ent.text = "a"
        mock_ent.label_ = "ORG"
        mock_ent.start_char = 0
        mock_ent.end_char = 1
        mock_doc.ents = [mock_ent]
        extractor._nlp.return_value = mock_doc

        entities = extractor.extract("a")
        assert len(entities) == 0

    def test_extract_technology_entities(self, extractor):
        entities = [
            Entity(text="Microsoft", entity_type=EntityType.ORG, confidence=0.9),
            Entity(text="Copilot", entity_type=EntityType.PRODUCT, confidence=0.95),
            Entity(text="Satya Nadella", entity_type=EntityType.PERSON, confidence=0.9),
            Entity(text="AI coding assistant", entity_type=EntityType.TECHNOLOGY, confidence=0.8),
        ]

        tech_entities = extractor.extract_technology_entities(entities)
        assert len(tech_entities) == 2
        texts = [e.text for e in tech_entities]
        assert "Copilot" in texts
        assert "AI coding assistant" in texts
        assert "Microsoft" not in texts
        assert "Satya Nadella" not in texts


class TestGetEntityExtractor:
    def test_returns_instance(self):
        with patch("app.nlp.entities.EntityExtractor") as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            from app.nlp.entities import get_entity_extractor
            extractor = get_entity_extractor()
            assert extractor == mock_instance
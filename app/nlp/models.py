from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from app.nlp.taxonomy import (
    ConcernCategory,
    EmotionLabel,
    EntityType,
    SentimentLabel,
)


class SentimentResult(BaseModel):
    label: SentimentLabel
    score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("label", mode="before")
    @classmethod
    def validate_label(cls, v):
        if isinstance(v, str):
            return SentimentLabel(v.lower())
        return v


class Entity(BaseModel):
    text: str = Field(min_length=1)
    entity_type: EntityType
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    surface_form: Optional[str] = None
    canonical_form: Optional[str] = None

    @field_validator("entity_type", mode="before")
    @classmethod
    def validate_entity_type(cls, v):
        if isinstance(v, str):
            return EntityType(v.upper())
        return v


class EmotionResult(BaseModel):
    label: EmotionLabel
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("label", mode="before")
    @classmethod
    def validate_label(cls, v):
        if isinstance(v, str):
            return EmotionLabel(v.lower())
        return v


class ConcernResult(BaseModel):
    category: ConcernCategory
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: Optional[str] = None

    @field_validator("category", mode="before")
    @classmethod
    def validate_category(cls, v):
        if isinstance(v, str):
            return ConcernCategory(v.lower())
        return v


class ModelMetadata(BaseModel):
    sentiment: str
    ner: str
    emotion: str
    concern: str


class DocumentAnalysis(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    technology_entities: list[Entity] = Field(default_factory=list)
    sentiment: SentimentResult
    emotions: list[EmotionResult] = Field(default_factory=list)
    concerns: list[ConcernResult] = Field(default_factory=list)
    analysis_version: str = "phase2-v1"
    models: Optional[ModelMetadata] = None
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now().astimezone())

    @field_validator("sentiment")
    @classmethod
    def validate_sentiment(cls, v):
        if isinstance(v, dict):
            return SentimentResult(**v)
        return v


class AnalyzedDocument(BaseModel):
    document_id: str
    source_id: str
    source_name: str
    source_type: str
    title: str
    content: str
    url: str
    author: Optional[str] = None
    published_at: Optional[datetime] = None
    retrieved_at: datetime
    content_hash: str
    canonical_url: str
    analysis: DocumentAnalysis

    @field_validator("analysis")
    @classmethod
    def validate_analysis(cls, v):
        if isinstance(v, dict):
            return DocumentAnalysis(**v)
        return v


class NLPAnalysisFailure(BaseModel):
    document_id: str
    timestamp: datetime
    stage: str
    error_type: str
    error_message: str
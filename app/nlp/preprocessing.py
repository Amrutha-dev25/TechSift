import re
from typing import Optional

from app.nlp.taxonomy import EntityType


def prepare_text_for_analysis(title: str, content: str) -> str:
    parts = []
    if title and title.strip():
        parts.append(title.strip())
    if content and content.strip():
        parts.append(content.strip())
    return " ".join(parts)


def truncate_text(text: str, max_length: int = 4096) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length]


def clean_for_nlp(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def split_into_chunks(text: str, max_chunk_size: int = 512, overlap: int = 50) -> list[str]:
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def extract_evidence_span(text: str, keywords: list[str], window: int = 100) -> Optional[str]:
    text_lower = text.lower()
    best_match = None
    best_idx = -1
    for keyword in keywords:
        keyword_lower = keyword.lower()
        idx = text_lower.find(keyword_lower)
        if idx != -1 and (best_idx == -1 or idx < best_idx):
            best_idx = idx
            best_match = keyword
    if best_match is not None:
        idx = text_lower.find(best_match.lower())
        start = max(0, idx - window)
        end = min(len(text), idx + len(best_match) + window)
        return text[start:end].strip()
    return None


def normalize_entity_text(text: str) -> str:
    return " ".join(text.strip().split())


def canonicalize_tech_entity(text: str) -> Optional[str]:
    known_canonical = {
        "copilot": "GitHub Copilot",
        "github copilot": "GitHub Copilot",
        "chatgpt": "ChatGPT",
        "gpt": "GPT",
        "gpt-4": "GPT-4",
        "gpt-3.5": "GPT-3.5",
        "claude": "Claude",
        "gemini": "Gemini",
        "bard": "Bard",
        "llama": "Llama",
        "llama 2": "Llama 2",
        "llama 3": "Llama 3",
        "mistral": "Mistral",
        "mixtral": "Mixtral",
        "windows recall": "Windows Recall",
        "recall": "Windows Recall",
        "copilot+": "Copilot+",
        "ai coding assistant": "AI coding assistants",
        "ai coding assistants": "AI coding assistants",
        "coding assistant": "AI coding assistants",
        "code assistant": "AI coding assistants",
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "google": "Google",
        "microsoft": "Microsoft",
        "meta": "Meta",
        "facebook": "Meta",
        "amazon": "Amazon",
        "aws": "AWS",
        "azure": "Azure",
        "nvidia": "NVIDIA",
        "supabase": "Supabase",
        "muse": "Muse",
        "recursive intelligence": "Recursive Intelligence",
        "techcrunch": "TechCrunch",
        "techcrunch disrupt": "TechCrunch Disrupt",
        "lightspeed": "Lightspeed Venture Partners",
    }

    normalized = normalize_entity_text(text).lower()
    return known_canonical.get(normalized)


def is_technology_entity(entity_text: str, entity_type: EntityType) -> bool:
    if entity_type in (EntityType.TECHNOLOGY, EntityType.PRODUCT):
        return True

    tech_indicators = [
        "ai", "ml", "llm", "model", "api", "sdk", "framework",
        "platform", "tool", "assistant", "agent", "copilot",
        "gpt", "claude", "gemini", "llama", "mistral",
    ]

    text_lower = entity_text.lower()
    return any(indicator in text_lower for indicator in tech_indicators)
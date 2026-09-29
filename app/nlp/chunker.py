from __future__ import annotations

import re
from typing import List

try:
    import spacy

    _nlp = None
    _nlp_name = "en_core_web_sm"


    def _get_nlp():
        global _nlp
        if _nlp is None:
            try:
                _nlp = spacy.load(_nlp_name, disable=["parser", "tagger", "ner"])
            except Exception:
                _nlp = spacy.blank("en")
        return _nlp

except ImportError:

    def _get_nlp:  # type: ignore
        return None


CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 75


def _count_tokens(text: str) -> int:
    if _get_nlp() is not None:
        return len(_get_nlp()(text))
    return len(re.findall(r"\S+", text))


def _split_into_sentences(text: str) -> List[str]:
    if _get_nlp() is not None:
        doc = _get_nlp()(text)
        return [sent.text for sent in doc.sents]
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def chunk_text(
    text: str,
    document_id: str,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
) -> List[dict]:
    if not text or not text.strip():
        return []

    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    chunks: List[dict] = []
    current_chunk_sentences: List[str] = []
    current_chunk_tokens = 0
    sentence_token_counts: List[int] = [_count_tokens(s) for s in sentences]

    sentence_idx = 0
    while sentence_idx < len(sentences):
        sent = sentences[sentence_idx]
        sent_tokens = sentence_token_counts[sentence_idx]

        if current_chunk_tokens + sent_tokens <= chunk_size:
            current_chunk_sentences.append(sent)
            current_chunk_tokens += sent_tokens
            sentence_idx += 1
        else:
            if current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                chunk_id = f"{document_id}:chunk:{len(chunks):03d}"
                chunks.append(
                    {
                        "document_id": document_id,
                        "chunk_id": chunk_id,
                        "chunk_index": len(chunks),
                        "text": chunk_text,
                    }
                )
                start_overlap = max(0, len(current_chunk_sentences) - chunk_overlap)
                current_chunk_sentences = current_chunk_sentences[start_overlap:]
                current_chunk_tokens = sum(sentence_token_counts[
                    sentence_idx - len(current_chunk_sentences) : sentence_idx
                ]) if start_overlap > 0 else 0
            else:
                sentence_idx += 1

    if current_chunk_sentences:
        chunk_text = " ".join(current_chunk_sentences)
        chunk_id = f"{document_id}:chunk:{len(chunks):03d}"
        chunks.append(
            {
                "document_id": document_id,
                "chunk_id": chunk_id,
                "chunk_index": len(chunks),
                "text": chunk_text,
            }
        )

    return chunks
from __future__ import annotations

import logging
from typing import List, Optional

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wrapper for Sentence Transformers embedding model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ("cuda" if False else "cpu")
        self._model: Optional[SentenceTransformer] = None

        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name, device=self.device)

        # Get embedding dimension (model API changed versions)
        try:
            self.dimension = self._model.get_sentence_embedding_dimension()
        except AttributeError:
            self.dimension = self._model.get_embedding_dimension()

        logger.info("Embedding model loaded - dimension: %d", self.dimension)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Returns list of embedding vectors (list of floats).
        """
        if not texts:
            return []

        # Filter out empty/whitespace-only texts
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return [[] for _ in texts]

        # Generate embeddings in batches
        embeddings = self._model.encode(
            valid_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # Convert numpy arrays to lists
        result_lists = [emb.tolist() for emb in embeddings]

        # Preserve position for filtered-out texts
        final_result: List[List[float]] = []
        valid_idx = 0
        for original_text in texts:
            if original_text and original_text.strip():
                final_result.append(result_lists[valid_idx])
                valid_idx += 1
            else:
                final_result.append([0.0] * self.dimension)

        return final_result

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query string."""
        if not query or not query.strip():
            return [0.0] * self.dimension
        emb = self._model.encode(query.strip(), convert_to_numpy=True)
        return emb.tolist()

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of document texts."""
        return self.embed(documents)
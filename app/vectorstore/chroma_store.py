from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import chromadb

from app.nlp.embedding import EmbeddingModel

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Persistent ChromaDB vector store for technology reaction documents."""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "technology_reaction_documents",
        embedding_model: Optional[EmbeddingModel] = None,
        distance_metric: str = "cosine",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.distance_metric = distance_metric

        # Embedding model (must be set before _get_or_create_collection)
        self._embedding_model = embedding_model or EmbeddingModel()
        self.dimension = self._embedding_model.dimension

        # Initialize Chroma client
        self._client = chromadb.PersistentClient(path=persist_directory)

        # Initialize or get collection
        self._collection = self._get_or_create_collection()

        logger.info(
            "ChromaVectorStore initialized - collection: %s, dimension: %d, "
            "distance: %s, persist_dir: %s",
            collection_name,
            self.dimension,
            distance_metric,
            persist_directory,
        )

    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get existing collection or create new one with deterministic name."""
        try:
            collection = self._client.get_collection(
                name=self.collection_name
            )
            logger.info("Retrieved existing collection: %s", self.collection_name)
            return collection
        except Exception:
            logger.info("Creating new collection: %s", self.collection_name)
            collection = self._client.create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": self.distance_metric,
                    "embedding_model_name": self._embedding_model.model_name,
                    "embedding_dimension": self.dimension,
                },
            )
            return collection

    def add_chunks(
        self,
        chunks: List[dict],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add document chunks to the vector store.

        Each chunk dict must contain at minimum:
        - document_id
        - chunk_id
        - chunk_index
        - text
        - Plus all Phase 2 metadata (sentiment, concerns, entities, etc.)
        """
        if not chunks:
            return []

        # Generate IDs if not provided
        if ids is None:
            ids = [c.get("chunk_id", f"{c['document_id']}:chunk:{i:03d}") for i, c in enumerate(chunks)]

        # Extract texts and metadata
        texts = [c["text"] for c in chunks]

        # Generate embeddings
        embeddings = self._embedding_model.embed(texts)

        # Prepare metadata - flatten to supported primitive types
        metadatas = self._prepare_metadata(chunks)

        # Add to collection
        self._collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )

        logger.info("Added %d chunks to collection %s", len(chunks), self.collection_name)
        return ids

    def _prepare_metadata(self, chunks: List[dict]) -> List[Dict[str, Any]]:
        """Prepare chunk metadata for ChromaDB (primitive types only).

        Flatten complex fields to deterministic serialized strings.
        """
        metadatas: List[Dict[str, Any]] = []

        for chunk in chunks:
            metadata: Dict[str, Any] = {
                "document_id": chunk.get("document_id", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                "text": chunk.get("text", ""),
            }

            # Phase 2 metadata - flatten to strings
            # Sentiment
            if chunk.get("sentiment_label"):
                metadata["sentiment_label"] = str(chunk["sentiment_label"])
            if chunk.get("sentiment_score") is not None:
                metadata["sentiment_score"] = float(chunk["sentiment_score"])
            if chunk.get("sentiment_confidence") is not None:
                metadata["sentiment_confidence"] = float(chunk["sentiment_confidence"])

            # Technology entities - serialized as pipe-separated string
            tech_entities = chunk.get("technology_entities", [])
            if tech_entities:
                metadata["technology_entities"] = "|".join(
                    str(e) for e in tech_entities if e
                )

            # Concerns - serialized as pipe-separated string
            concerns = chunk.get("concerns", [])
            if concerns:
                metadata["concerns"] = "|".join(
                    str(c) for c in concerns if c
                )

            # Emotions - serialized as pipe-separated string
            emotions = chunk.get("emotions", [])
            if emotions:
                metadata["emotions"] = "|".join(
                    str(e) for e in emotions if e
                )

            # Source metadata
            if chunk.get("title"):
                metadata["title"] = str(chunk["title"])
            if chunk.get("source_id"):
                metadata["source_id"] = str(chunk["source_id"])
            if chunk.get("source_name"):
                metadata["source_name"] = str(chunk["source_name"])
            if chunk.get("source_type"):
                metadata["source_type"] = str(chunk["source_type"])
            if chunk.get("url"):
                metadata["url"] = str(chunk["url"])

            # Publication timestamp
            if chunk.get("published_timestamp") is not None:
                metadata["published_timestamp"] = int(chunk["published_timestamp"])

            # Analysis version
            if chunk.get("analysis_version"):
                metadata["analysis_version"] = str(chunk["analysis_version"])

            metadatas.append(metadata)

        return metadatas

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks.

        Returns list of result dicts with:
        - id, text, metadata, score (similarity/distance)
        """
        if not query or not query.strip():
            return []

        # Generate query embedding
        query_embedding = self._embedding_model.embed_query(query.strip())

        # Build where clause for filters
        where_clause = None
        if filters:
            where_clause = self._build_filter(filters)

        # Query collection
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted = self._format_results(results)
        return formatted

    def _build_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build ChromaDB where clause from filters dict."""
        where: Dict[str, Any] = {}

        # Sentiment filters
        if "sentiment_label" in filters:
            where["sentiment_label"] = filters["sentiment_label"]

        if "min_sentiment_score" in filters:
            where["sentiment_score"] = {
                "$gte": float(filters["min_sentiment_score"])
            }

        if "max_sentiment_score" in filters:
            if "sentiment_score" in where:
                where["sentiment_score"]["$lte"] = float(filters["max_sentiment_score"])
            else:
                where["sentiment_score"] = {
                    "$lte": float(filters["max_sentiment_score"])
                }

        # Concern filter
        if "concern" in filters:
            # Match if concerns field contains the specified concern
            # Chroma uses substring matching on strings
            where["concerns"] = filters["concern"]

        # Technology entity filter
        if "technology" in filters:
            where["technology_entities"] = filters["technology"]

        # Source filters
        if "source_type" in filters:
            where["source_type"] = filters["source_type"]
        if "source_name" in filters:
            where["source_name"] = filters["source_name"]

        # Date filters
        if "published_after" in filters or "published_before" in filters:
            date_where: Dict[str, Any] = {}
            if "published_after" in filters:
                date_where["$gte"] = float(filters["published_after"])
            if "published_before" in filters:
                date_where["$lte"] = float(filters["published_before"])
            where.update(date_where)

        return where if where else None

    def _format_results(
        self, results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Format ChromaDB query results into standard format."""
        formatted: List[Dict[str, Any]] = []

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        distances = results.get("distances", [])

        for i, (doc_list, meta_list, distance_list) in enumerate(
            zip(documents, metadatas, distances)
        ):
            # Chroma returns lists: documents=[['text']], metadatas=[{dict}], distances=[[float]]
            # Extract the single elements
            doc = doc_list[0] if isinstance(doc_list, list) and doc_list else ""
            meta = meta_list[0] if isinstance(meta_list, list) and len(meta_list) > 0 else {}
            dist = distance_list[0] if isinstance(distance_list, list) and len(distance_list) > 0 else None

            # Convert distance to similarity: similarity = 1 - distance (cosine)
            similarity = 1.0 - float(dist) if dist is not None else 0.0

            # Convert published_timestamp (unix seconds) to datetime
            published_at = meta.get("published_at")
            if published_at is None and meta.get("published_timestamp") is not None:
                try:
                    from datetime import datetime as _dt, timezone as _tz

                    published_at = _dt.fromtimestamp(
                        float(meta["published_timestamp"]), tz=_tz.utc
                    )
                except Exception:
                    published_at = None

            result = {
                "id": meta.get("chunk_id", ""),
                "document_id": meta.get("document_id", ""),
                "text": doc,
                "score": round(similarity, 4),
                "title": meta.get("title", ""),
                "source_name": meta.get("source_name", ""),
                "source_type": meta.get("source_type", ""),
                "url": meta.get("url", ""),
                "published_at": published_at,
                "sentiment_label": meta.get("sentiment_label"),
                "sentiment_score": meta.get("sentiment_score"),
                "technology_entities": (
                    meta.get("technology_entities", "").split("|")
                    if meta.get("technology_entities")
                    else []
                ),
                "concerns": (
                    meta.get("concerns", "").split("|")
                    if meta.get("concerns")
                    else []
                ),
                "emotions": (
                    meta.get("emotions", "").split("|")
                    if meta.get("emotions")
                    else []
                ),
                "chunk_index": meta.get("chunk_index", 0),
                "analysis_version": meta.get("analysis_version"),
            }
            formatted.append(result)

        return formatted

    def count(self) -> int:
        """Return total number of chunks in collection."""
        return self._collection.count()

    def get_stats(self) -> Dict[str, Any]:
        """Return collection statistics."""
        count = self._collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,  # Approximate - Chroma doesn't store doc count separately
            "chunk_count": count,
            "embedding_model": self._embedding_model.model_name,
            "embedding_dimension": self.dimension,
            "distance_metric": self.distance_metric,
            "persist_directory": self.persist_directory,
        }

    def delete_chunks(self, chunk_ids: List[str]) -> None:
        """Delete specific chunks by ID."""
        if chunk_ids:
            self._collection.delete(ids=chunk_ids)
            logger.info("Deleted %d chunks from collection %s", len(chunk_ids), self.collection_name)
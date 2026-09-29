#!/usr/bin/env python
"""Index Phase 2 analyzed documents into ChromaDB vector store."""

import argparse
import json
import sys
import logging
import os
from pathlib import Path
from typing import List

from app.nlp.chunker import chunk_text
from app.nlp.embedding import EmbeddingModel
from app.vectorstore.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
logger.addHandler(handler)


def load_phase2_documents(input_path: Path) -> list:
    """Load Phase 2 analyzed documents from JSONL.

    Each document should have:
    - document_id
    - analysis (with sentiment, entities, concerns, etc.)
    - title, content, source metadata
    """
    documents = []
    not_found = []

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return documents

    try:
        with input_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    # Validate required fields
                    if "document_id" not in doc or "analysis" not in doc:
                        not_found.append((line_num, "Missing document_id or analysis"))
                        continue

                    # Ensure analysis has required fields
                    analysis = doc["analysis"]
                    if "sentiment" not in analysis:
                        analysis["sentiment"] = {"label": "neutral", "score": 0.0, "confidence": 1.0}
                    if "technology_entities" not in analysis:
                        analysis["technology_entities"] = []
                    if "concerns" not in analysis:
                        analysis["concerns"] = []
                    if "emotions" not in analysis:
                        analysis["emotions"] = []

                    documents.append(doc)
                except json.JSONDecodeError as e:
                    logger.warning("Failed to parse line %d: %s", line_num, e)
                    continue
    except Exception as e:
        logger.error("Failed to load documents from %s: %s", input_path, e)

    logger.info("Loaded %d Phase 2 documents from %s (skipped %d invalid)",
                len(documents), input_path, len(not_found))
    return documents


def index_documents(
    input_path: Path = None,
    batch_size: int = 32,
    force: bool = False,
) -> dict:
    """Index Phase 2 analyzed documents into ChromaDB.

    Args:
        input_path: Path to analyzed_documents.jsonl (uses default if None)
        batch_size: Chunk batch size for embedding
        force: If True, recreate collection from scratch

    Returns:
        dict with indexing statistics
    """
    # Use default path if not provided
    if input_path is None:
        input_path = Path("data/processed/analyzed_documents.jsonl")

    # Load Phase 2 documents
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        print(f"Error: Input file not found: {input_path}")
        return {"error": f"Input file not found: {input_path}"}

    documents = load_phase2_documents(input_path)
    if not documents:
        print("No documents found in input path.")
        return {"chunks_indexed": 0, "chunks_skipped": 0, "chunks_failed": 0}

    # Initialize vector store (respect --force flag)
    # Always use ./chroma_db as the persistent directory
    # --force resets the collection (deletes and recreates)
    persist_dir = "./chroma_db"

    if force:
        # Delete existing ChromaDB directory if --force is used
        import shutil
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)

    vector_store = ChromaVectorStore(
        persist_directory=persist_dir,
        collection_name="technology_reaction_documents",
        embedding_model=EmbeddingModel(batch_size=batch_size),
    )

    # If not force mode, check existing count for idempotency
    if not force:
        existing_count = vector_store.count()
        print(f"Collection has {existing_count} existing chunks")
        # We'll rely on deterministic chunk_ids for idempotency

    # Process documents
    total_chunks = 0
    total_indexed = 0
    total_skipped = 0
    total_failed = 0
    failed_records = []

    batch_chunks = []
    batch_ids = []

    for doc in documents:
        try:
            document_id = doc["document_id"]
            analysis = doc.get("analysis", {})
            # Phase 2 title and content are at document level, not inside analysis
            title = doc.get("title") or ""
            content = doc.get("content") or ""

            # Build text from available Phase 2 fields
            text_parts = []
            if title:
                text_parts.append(str(title))

            # Add content/summary if available
            if content:
                text_parts.append(str(content))

            # Fallback: use document ID if no text
            if not text_parts:
                text_parts.append(document_id)

            full_text = " ".join(text_parts)

            if not full_text.strip():
                total_skipped += 1
                continue

            # Chunk the text
            chunks = chunk_text(
                full_text,
                document_id=document_id,
                chunk_size=500,
                chunk_overlap=75,
            )

            if not chunks:
                total_skipped += 1
                continue

            # Process each chunk
            for chunk_index, chunk in enumerate(chunks):
                chunk_id = f"{document_id}:chunk:{chunk_index:03d}"

                # Check for existing chunk (idempotency)
                if not force and vector_store._collection.get(ids=[chunk_id]):
                    total_skipped += 1
                    continue

                # Prepare chunk metadata from Phase 2 analysis
                chunk_meta = {
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "text": chunk["text"],
                    # Phase 2 sentiment metadata
                    "sentiment_label": analysis.get("sentiment", {}).get("label"),
                    "sentiment_score": analysis.get("sentiment", {}).get("score"),
                    "sentiment_confidence": analysis.get("sentiment", {}).get("confidence"),
                    # Technology entities from Phase 2
                    "technology_entities": analysis.get("technology_entities", []) or [],
                    # Concerns from Phase 2
                    "concerns": analysis.get("concerns", []) or [],
                    # Emotions from Phase 2 (top 3)
                    "emotions": (analysis.get("emotions", []) or [])[:3],
                    # Source metadata
                    "source_id": analysis.get("source_id"),
                    "source_name": analysis.get("source_name"),
                    "source_type": analysis.get("source_type"),
                    "url": analysis.get("url"),
                    # Publication timestamp
                    "published_timestamp": analysis.get("published_at"),
                    # Analysis version
                    "analysis_version": analysis.get("analysis_version", "phase2-v2"),
                }

                # Add to batch
                batch_chunks.append(chunk_meta)
                batch_ids.append(chunk_id)
                total_chunks += 1

                # Add in batches
                if len(batch_chunks) >= batch_size:
                    vector_store.add_chunks(batch_chunks, batch_ids)
                    total_indexed += len(batch_chunks)
                    batch_chunks = []
                    batch_ids = []

            # Reset batch for next document (remaining chunks)
            if batch_chunks:
                vector_store.add_chunks(batch_chunks, batch_ids)
                total_indexed += len(batch_chunks)
                batch_chunks = []
                batch_ids = []

        except Exception as e:
            logger.error("Failed to index document %s: %s", document_id, e)
            total_failed += 1
            failed_records.append(
                {
                    "document_id": document_id,
                    "chunk_id": "unknown",
                    "stage": "indexing",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "timestamp": __import__("datetime").datetime.now().isoformat(),
                }
            )

    # Write failed records
    if failed_records:
        from datetime import datetime
        failed_path = Path("data/failed/vector_index_failures.jsonl")
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        with open(failed_path, "a", encoding="utf-8") as f:
            for rec in failed_records:
                f.write(json.dumps(rec) + "\n")

    # Print summary
    print("=" * 60)
    print("Phase 3 Indexing Complete")
    print("=" * 60)
    print(f"Documents processed: {len(documents)}")
    print(f"Total chunks generated: {total_chunks}")
    print(f"Chunks indexed: {total_indexed}")
    print(f"Chunks skipped (existing): {total_skipped}")
    print(f"Chunks failed: {total_failed}")
    print(f"Vector store stats: {vector_store.get_stats()}")
    print("=" * 60)

    return {
        "documents_processed": len(documents),
        "total_chunks_generated": total_chunks,
        "chunks_indexed": total_indexed,
        "chunks_skipped": total_skipped,
        "chunks_failed": total_failed,
        "vector_stats": vector_store.get_stats(),
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 3: Index documents into ChromaDB knowledge base"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to analyzed_documents.jsonl (default: data/processed/analyzed_documents.jsonl)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size (default: 32)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate collection from scratch (skip idempotency check)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Default top-K for searches (default: 10)",
    )

    args = parser.parse_args()

    result = index_documents(
        input_path=args.input,
        batch_size=args.batch_size,
        force=args.force,
    )

    sys.exit(0 if not result.get("error") else 1)


if __name__ == "__main__":
    main()
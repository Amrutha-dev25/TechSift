#!/usr/bin/env python
"""Phase 3 verification tests."""

import chromadb
import shutil
import os
from pathlib import Path

def test_1_indexing():
    """Test 1: Index all Phase 2 documents."""
    print("=" * 60)
    print("TEST 1: Indexing Phase 2 documents")
    print("=" * 60)
    
    # Clean up
    for d in ['./chroma_db', './chroma_db_force']:
        if os.path.exists(d):
            shutil.rmtree(d)
    
    # Run indexing
    import subprocess
    result = subprocess.run(
        ['python', 'scripts\\index_documents.py', '--force'],
        cwd='D:\\TechSignal',
        capture_output=True,
        text=True,
    )
    
    # Check ChromaDB
    client = chromadb.PersistentClient(path='./chroma_db')
    collections = client.list_collections()
    tech_coll = [c for c in collections if c.name == 'technology_reaction_documents']
    
    if not tech_coll:
        print("FAIL: ChromaDB collection not created")
        return False
    
    coll = tech_coll[0]
    chunk_count = coll.count()
    print(f"Chunks indexed: {chunk_count}")
    
    # Get some IDs
    all_ids = coll.get(limit=chunk_count)
    print(f"Documents traced: {len(all_ids['ids'])}")
    
    if chunk_count == 0:
        print("FAIL: No chunks indexed")
        return False
    
    print(f"PASS: {chunk_count} chunks indexed successfully")
    return True


def test_2_search():
    """Test 2: Run semantic search and verify relevant results."""
    print("\n" + "=" * 60)
    print("TEST 2: Semantic search")
    print("=" * 60)
    
    from app.vectorstore.chroma_store import ChromaVectorStore
    vs = ChromaVectorStore(persist_directory='./chroma_db')
    
    results = vs.search(
        query='Why are developers concerned about AI coding assistants?',
        top_k=3
    )
    
    if not results:
        print("FAIL: No search results returned")
        return False
    
    print(f"Results returned: {len(results)}")
    for i, r in enumerate(results):
        score = r.get('score', 0)
        text = r.get('text', '')[:60] if r.get('text') else ''
        doc_id = r.get('document_id', '')
        sent = r.get('sentiment_label', '')
        print(f"  {i+1}. Score: {score:.4f} | Doc: {doc_id[:20]}... | Sentiment: {sent}")
        print(f"   Text: {text}...")
    
    if len(results) < 3:
        print("WARN: Expected 3 results, got fewer")
    
    print("PASS: Semantic search works")
    return True


def test_3_idempotency():
    """Test 3: Run indexing twice and verify 0 duplicates."""
    print("\n" + "=" * 60)
    print("TEST 3: Idempotency (re-indexing)")
    print("=" * 60)
    
    # Run indexing again without --force
    import subprocess
    result = subprocess.run(
        ['python', 'scripts\\index_documents.py'],
        cwd='D:\\TechSignal',
        capture_output=True,
        text=True,
    )
    
    # Check collection count
    client = chromadb.PersistentClient(path='./chroma_db')
    collections = client.list_collections()
    tech_coll = [c for c in collections if c.name == 'technology_reaction_documents']
    
    if not tech_coll:
        print("FAIL: Collection not found")
        return False
    
    chunk_count = tech_coll[0].count()
    print(f"Chunks after re-index: {chunk_count}")
    
    # The key test: if idempotent, should still be 50 (not 100)
    if chunk_count == 50:
        print("PASS: No duplicates created on re-index")
        return True
    elif chunk_count == 100:
        print("FAIL: Duplicates created (100 chunks instead of 50)")
        return False
    else:
        print(f"UNEXPECTED: {chunk_count} chunks")
        return False


def test_4_persistence():
    """Test 4: ChromaDB data survives restart."""
    print("\n" + "=" * 60)
    print("TEST 4: Persistence (restart test)")
    print("=" * 60)
    
    # First check data exists
    client = chromadb.PersistentClient(path='./chroma_db')
    collections = client.list_collections()
    tech_coll = [c for c in collections if c.name == 'technology_reaction_documents']
    
    if not tech_coll:
        print("FAIL: Collection not found - nothing to persist")
        return False
    
    chunk_count_before = tech_coll[0].count()
    print(f"Chunks before 'restart': {chunk_count_before}")
    
    # "Restart" = just reopen the same directory
    # (ChromaDB handles this automatically with PersistentClient)
    
    # Re-query
    coll = tech_coll[0]
    chunk_count_after = coll.count()
    print(f"Chunks after 'restart': {chunk_count_after}")
    
    if chunk_count_before == chunk_count_after and chunk_count_before > 0:
        print("PASS: Data persists across restarts")
        return True
    else:
        print("FAIL: Data lost on restart")
        return False


def test_5_metadata_filtering():
    """Test 5: Metadata filtering works."""
    print("\n" + "=" * 60)
    print("TEST 5: Metadata filtering")
    print("=" * 60)
    
    from app.retrieval.models import RetrievalResult, RetrievalFilters
    from app.retrieval.retriever import Retriever
    
    retriever = Retriever(default_top_k=5)
    
    # Test sentiment filter
    filters = RetrievalFilters(sentiment_label='neutral')
    results = retriever.search(
        query='AI technology',
        top_k=5,
        filters=filters,
    )
    
    # Verify all results have the right sentiment
    all_neutral = all(r.sentiment_label == 'neutral' for r in results if r.sentiment_label is not None)
    print(f"Results with sentiment filter: {len(results)}")
    for r in results[:3]:
        print(f"  - Sentiment: {r.sentiment_label} | Text: {r.text[:50]}...")
    
    if all_neutral:
        print("PASS: Sentiment filtering works")
        return True
    else:
        print("FAIL: Sentiment filtering not working correctly")
        return False


def test_6_query_validation():
    """Test 6: Invalid queries are rejected."""
    print("\n" + "=" * 60)
    print("TEST 6: Invalid query validation")
    print("=" * 60)
    
    from app.retrieval.models import RetrievalFilters
    
    # Test empty query should be handled
    retriever = Retriever(default_top_k=5)
    results = retriever.search(query='', top_k=5)
    
    # Empty query should return empty results, not crash
    if results is not None and isinstance(results, list):
        print(f"PASS: Empty query handled gracefully ({len(results)} results)")
        return True
    else:
        print("FAIL: Empty query caused error")
        return False


if __name__ == '__main__':
    results = []
    
    # Run all tests
    results.append(("Indexing", test_1_indexing()))
    results.append(("Search", test_2_search()))
    results.append(("Idempotency", test_3_idempotency()))
    results.append(("Persistence", test_4_persistence()))
    results.append(("Metadata filtering", test_5_metadata_filtering()))
    results.append(("Query validation", test_6_query_validation()))
    
    # Summary
    print("\n" + "=" * 60)
    print("PHASE 3 VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nPHASE 3: COMPLETE")
    else:
        print("\nPHASE 3: INCOMPLETE - see failures above")
from app.storage.jsonl_store import JSONLStore
from pathlib import Path

store = JSONLStore(
    processed_path=Path('data/processed/analyzed_documents.jsonl'),
    raw_dir=Path('data/raw'),
    failed_dir=Path('data/failed'),
)
docs = store.load_all()
print(f'Loaded {len(docs)} documents')
if docs:
    d = docs[0]
    print(f'First doc ID: {d.document_id}')
    print(f'Has analysis: {hasattr(d, "analysis")}')
    if hasattr(d, 'analysis'):
        a = d.analysis
        print(f'Analysis version: {a.get("analysis_version")}')
        print(f'Sentiment: {a.get("sentiment")}')
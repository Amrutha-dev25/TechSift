from app.storage.jsonl_store import JSONLStore
from pathlib import Path
import json

store = JSONLStore(
    processed_path=Path('data/processed/analyzed_documents.jsonl'),
    raw_dir=Path('data/raw'),
    failed_dir=Path('data/failed'),
)
docs = store.load_all()
print(f'Type of first doc: {type(docs[0])}')
print(f'Fields: {list(docs[0].model_dump().keys())}')
d = docs[0]
print(f'Has analysis attr: {hasattr(d, "analysis")}')
if hasattr(d, 'analysis'):
    print(f'Analysis: {d.analysis}')
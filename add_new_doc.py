import json
from pathlib import Path

# Create a new Phase 2 style document
new_doc = {
    'document_id': 'test_new_doc_001',
    'source_id': 'test_feed',
    'source_name': 'Test Feed',
    'source_type': 'rss',
    'title': 'Test AI Coding Assistant Concern',
    'content': 'Developers are very concerned about security vulnerabilities in AI coding assistants. The generated code may introduce security risks and backdoors that could be exploited by malicious actors.',
    'url': 'https://example.com/test-ai-concern',
    'author': 'Test Author',
    'published_at': '2024-01-15T10:00:00Z',
    'retrieved_at': '2026-09-28T00:00:00Z',
    'content_hash': 'test_hash_001',
    'canonical_url': 'https://example.com/test-ai-concern',
    'analysis': {
        'sentiment': {'label': 'negative', 'score': -0.8, 'confidence': 0.95},
        'technology_entities': [{'text': 'AI coding assistants', 'entity_type': 'OTHER', 'confidence': None, 'surface_form': 'AI coding assistants', 'canonical_form': None}],
        'concerns': [{'category': 'security', 'confidence': 0.98}, {'category': 'code_quality', 'confidence': 0.87}],
        'emotions': [{'label': 'anger', 'confidence': 0.8}, {'label': 'fear', 'confidence': 0.7}],
        'analysis_version': 'phase2-v2',
        'source_id': 'test_feed',
        'source_name': 'Test Feed',
        'source_type': 'rss',
        'url': 'https://example.com/test-ai-concern',
        'published_at': '2024-01-15T10:00:00Z'
    }
}

# Append to the Phase 2 JSONL file
with Path('data/processed/analyzed_documents.jsonl').open('a', encoding='utf-8') as f:
    f.write(json.dumps(new_doc) + '\n')
    
print('New document appended to analyzed_documents.jsonl')
print(f'Document ID: {new_doc["document_id"]}')
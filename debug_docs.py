import json
from pathlib import Path

input_path = Path('data/processed/analyzed_documents.jsonl')
documents = []
with input_path.open('r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        doc = json.loads(line)
        documents.append(doc)
        if line_num <= 3:
            a = doc.get('analysis', {})
            print(f'Doc {line_num}:')
            print(f'  title: {a.get("title")}')
            print(f'  content: {str(a.get("content", ""))[:50] if a.get("content") else "None"}')
            print(f'  technology_entities: {a.get("technology_entities")}')
            print(f'  concerns: {a.get("concerns")}')
            print(f'  emotions: {a.get("emotions")}')
            print()
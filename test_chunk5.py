from app.nlp.chunker import chunk_text

text = 'Developers are concerned about AI coding assistants.'
result = chunk_text(text, document_id='test_123', chunk_size=500, chunk_overlap=75)
print(f'Chunks: {len(result)}')
for c in result:
    print(f'  {c["chunk_index"]}: {c["text"][:50]}...')
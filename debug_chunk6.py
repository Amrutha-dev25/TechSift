from app.nlp.chunker import chunk_text

# Test with title from Phase 2 doc
text = "Meta's Muse just stole the AI spotlight from OpenAI and Anthropic"
result = chunk_text(text, document_id='test_123', chunk_size=500, chunk_overlap=75)
print(f'Chunks: {len(result)}')
for c in result:
    print(f'  {c["chunk_index"]}: OK - {c["text"][:50]}...')
print('Success!')
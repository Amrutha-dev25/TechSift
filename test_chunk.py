from app.nlp.chunker import chunk_text

text = "Developers are concerned that AI coding assistants may introduce security vulnerabilities in generated code. Many teams worry about the quality and maintenance burden. However, others see productivity gains. The concern about code quality is real. Security teams fear potential exploits."
result = chunk_text(text, document_id="test_article_001")
for c in result:
    print(f'Chunk {c["chunk_index"]}: {c["text"][:80]}...')
print(f'Total chunks: {len(result)}')
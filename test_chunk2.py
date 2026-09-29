from app.nlp.chunker import chunk_text

# Longer text to test multiple chunks
text = """Developers are concerned that AI coding assistants may introduce security vulnerabilities in generated code. Many teams worry about the quality and maintenance burden of AI-generated software. However, others see significant productivity gains and faster development cycles. The concern about code quality is real, with security teams fearing potential exploits and backdoors. On the other hand, proponents argue that AI-assisted development can catch bugs earlier and improve overall software reliability. There are also concerns about job displacement for junior developers and the erosion of fundamental programming skills. Companies must balance the productivity benefits with rigorous testing and code review processes to mitigate risks."""

result = chunk_text(text, document_id="test_article_002")
print(f'Total chunks: {len(result)}')
for c in result:
    print(f'Chunk {c["chunk_index"]}: {c["text"][:80]}...')
    print()
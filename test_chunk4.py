from app.nlp.chunker import chunk_text, _count_tokens

# Generate a long text by repeating sentences
sentences = [
    "Developers are concerned that AI coding assistants may introduce security vulnerabilities in generated code.",
    "Many teams worry about the quality and maintenance burden of AI-generated software.",
    "However, others see significant productivity gains and faster development cycles.",
    "The concern about code quality is real, with security teams fearing potential exploits and backdoors.",
    "On the other hand, proponents argue that AI-assisted development can catch bugs earlier.",
    "There are also concerns about job displacement for junior developers.",
    "Companies must balance the productivity benefits with rigorous testing and code review.",
    "Investors are watching the AI software market growth closely.",
    "Regulators consider guidelines for responsible AI code generation.",
    "Developers must adapt skills to work effectively with AI tools.",
]

# Create a long text (50+ sentences)
long_text = " ".join(sentences * 10)

result = chunk_text(long_text, document_id="test_article_long", chunk_size=500, chunk_overlap=75)
print(f"Total chunks: {len(result)}")
for c in result[:3]:
    tok = _count_tokens(c["text"])
    print(f"Chunk {c['chunk_index']}: {tok} tokens - {c['text'][:80]}...")
with open('README.md', 'r', encoding='utf-8') as f:
    content = f.read()

phase5 = '''## Phase 5 — RAG Answer Generation

The system accepts a user question, retrieves relevant evidence from the Phase 3 knowledge base via the Phase 4 retrieval pipeline, and generates a grounded answer using OpenAI.

### Configuration

Add the following to your `.env`:

```text
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

The `OPENAI_API_KEY` must be set as an environment variable. The `OPENAI_MODEL` defaults to `gpt-4o-mini` if omitted.

### CLI usage

```bash
python scripts\ask.py "--query \"Why are developers concerned about AI coding assistants?\""
```

The CLI displays:
- The generated answer
- Source titles, URLs, and published dates
- Evidence IDs for citation

### Answer format

The generated answer includes:
- The synthesized answer text
- Clickable source citations with title, URL, and publication date
- A statement if evidence was insufficient

### Limitations

- Answer quality depends on the OpenAI model and retrieved evidence
- If no relevant evidence is found, the system states that explicitly
- No conversation memory or multi-turn context is maintained across calls
'''

# Add Phase 5 after the Phase 2 verification commands section
marker = 'After the second run, verify that `data/processed/analyzed_documents.jsonl` has not grown with duplicate analyses.'
if marker in content:
    new_content = content.replace(marker, marker + phase5)
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print('README updated: Phase 5 section inserted')
else:
    # Append at end
    with open('README.md', 'a', encoding='utf-8') as f:
        f.write('\n' + phase5)
    print('README appended: Phase 5 section added at end')
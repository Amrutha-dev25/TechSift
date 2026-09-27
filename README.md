# Technology Reaction Intelligence System

## Phase 1 — Production-Grade Technology News Ingestion Pipeline

This project builds the data ingestion foundation for a system that will eventually analyze public reactions to emerging technologies using NLP, semantic retrieval, and RAG.

**Phase 1 scope is ingestion only.** It collects, validates, normalizes, cleans, deduplicates, and persists technology news documents from RSS/Atom feeds into a JSON Lines store.

---

## What the ingestion pipeline does

The pipeline follows this flow:

```text
RSS feeds
    ↓
Fetch
    ↓
Parse
    ↓
Validate
    ↓
Normalize
    ↓
Clean
    ↓
Deduplicate
    ↓
Quality-check
    ↓
Persist
```

For each enabled source, the pipeline fetches the feed, extracts entries, converts them into a validated internal document format, removes HTML and tracking markup, rejects low-quality records, removes duplicates, and writes accepted documents to `data/processed/documents.jsonl`.

---

## Supported source type

Phase 1 supports **RSS/Atom feeds** via `feedparser`.

The source abstraction in `app/ingestion/base.py` allows additional source types (News API, Reddit, official sources, etc.) to be added in later phases without changing the downstream pipeline.

---

## How to configure feeds

Feeds are configured in `app/config/settings.py` as a list of `RSSSourceConfig` objects.

Each source requires:

```text
source_id  — unique identifier
source_name — human-readable name
source_type — "rss"
feed_url    — publicly accessible RSS/Atom URL
enabled     — whether the source is active
```

Example:

```python
RSSSourceConfig(
    source_id="techcrunch_ai",
    source_name="TechCrunch AI",
    source_type="rss",
    feed_url="https://techcrunch.com/category/artificial-intelligence/feed/",
    enabled=True,
)
```

Environment-based configuration is loaded from `.env`. Copy `.env.example` to `.env` and adjust values as needed:

```bash
copy .env.example .env
```

---

## How to run ingestion

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Copy the example environment file and edit if needed:

```bash
copy .env.example .env
```

3. Run the ingestion script:

```bash
python scripts\ingest_news.py
```

Optionally limit entries per feed:

```bash
python scripts\ingest_news.py --max-per-feed 20
```

---

## Where processed data is stored

| Path | Purpose |
|------|---------|
| `data/processed/documents.jsonl` | Accepted canonical documents |
| `data/raw/` | Optional raw feed snapshots for debugging |
| `data/failed/` | Rejected/failed records with reason |
| `logs/` | Application logs |

---

## How duplicate handling works

Documents receive a deterministic `document_id` derived from the canonical URL using SHA-256. The same article ingested twice resolves to the same logical document.

Exact duplicates are detected by content hash. Near-duplicate detection uses a lightweight normalized-text fingerprint so the component can later be replaced with semantic deduplication.

Re-running ingestion is idempotent: duplicate documents are not added to `documents.jsonl`; they are counted and reported as duplicates.

---

## How failures are handled

A single failed feed does not stop the pipeline. Failures are isolated per source and recorded in `data/failed/` with:

```text
timestamp
source
URL if available
failure reason
error type
```

The final CLI report shows feeds attempted, succeeded, failed, entries fetched, accepted, rejected, and duplicates.

---

## How to run tests

Tests use `pytest` and do not depend on live websites. External requests are mocked.

```bash
pytest
```

---

## Phase 2 — NLP Analysis Layer

Phase 2 transforms each clean Phase 1 document into structured analytical information. The system determines:

1. What technology/product/entity is being discussed
2. What is the sentiment of the text (positive/neutral/negative)
3. What is the strength of that sentiment (-1.0 to +1.0)
4. What emotional/reaction signals are present (anger, fear, concern, excitement, etc.)
5. What concerns or issues are being discussed (privacy, security, cost, etc.)
6. Which parts of the document support those classifications

### NLP Architecture

```
Document
   │
   ▼
Text Preparation
   │
   ├──────────────┐
   ▼              ▼
Entity         Sentiment
Detection      Analysis
   │              │
   ▼              ▼
Entities       Sentiment
   │              │
   └──────┬───────┘
          ▼
     Emotion Analysis
          │
          ▼
     Concern Extraction
          │
          ▼
     Validation
          │
          ▼
     Analysis Result
```

### Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` | 3-class sentiment (positive/neutral/negative) |
| NER | `en_core_web_sm` (spaCy) | Named entity recognition |
| Emotion | `j-hartmann/emotion-english-distilroberta-base` | 7-class emotion classification |
| Concerns | `facebook/bart-large-mnli` | Zero-shot classification against concern taxonomy |

All models run locally via Hugging Face `transformers`. No external LLM API is required.

### Sentiment Methodology

- Analyzes combined `title + content` (title given natural prominence by position)
- Normalizes to `-1.0` (strongly negative) to `+1.0` (strongly positive) using: `score = positive_prob - negative_prob`
- Confidence = model probability of the predicted class
- For long documents: controlled truncation to 4096 tokens

### Entity Extraction

- Uses spaCy NER with label mapping to application taxonomy: `ORG`, `PRODUCT`, `PERSON`, `TECHNOLOGY`, `EVENT`, `OTHER`
- Technology/product entities separated via keyword heuristics
- Conservative canonicalization (e.g., "Copilot" → "GitHub Copilot") for known entities
- Stores both `surface_form` and `canonical_form` when mapped

### Emotion Methodology

- Separate from sentiment; multi-label classification
- Taxonomy: `anger`, `fear`, `concern`, `disappointment`, `excitement`, `approval`, `confusion`, `skepticism`, `frustration`, `neutral`
- Threshold-based filtering (default 0.3)
- Returns ranked emotions by confidence

### Concern Taxonomy

Controlled vocabulary of 20 categories:

```
privacy, security, data_collection, copyright, bias, safety,
accuracy, reliability, cost, performance, job_displacement,
skill_erosion, user_control, transparency, vendor_lock_in,
code_quality, misinformation, environmental_impact, accessibility, other
```

- Zero-shot classification via BART-MNLI
- Evidence spans extracted from document text
- Conservative: only extracts concerns with textual evidence

### Output Schema

Each analyzed document preserves all original Phase 1 fields plus an `analysis` object:

```json
{
  "document_id": "...",
  "title": "...",
  "content": "...",
  "url": "...",
  "source_name": "...",
  "published_at": "...",
  "analysis": {
    "entities": [
      {"text": "Microsoft", "entity_type": "ORG", "confidence": 0.97}
    ],
    "technology_entities": [
      {"text": "Copilot", "entity_type": "PRODUCT", "canonical_form": "GitHub Copilot"}
    ],
    "sentiment": {"label": "negative", "score": -0.72, "confidence": 0.84},
    "emotions": [{"label": "concern", "confidence": 0.78}],
    "concerns": [
      {"category": "privacy", "confidence": 0.84, "evidence": "stores sensitive user information"}
    ],
    "analysis_version": "phase2-v1",
    "models": {
      "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
      "ner": "en_core_web_sm",
      "emotion": "j-hartmann/emotion-english-distilroberta-base",
      "concern": "facebook/bart-large-mnli"
    }
  }
}
```

### Important Design Principle

The NLP system distinguishes between **negative sentiment** and **public backlash**. A negative article ≠ public anger. The system records what the text actually indicates using precise terminology: `sentiment`, `emotion`, `concern`, `criticism`, `reaction` — not "backlash."

---

## How to run analysis

1. Ensure Phase 1 documents exist:

```bash
python scripts\ingest_news.py
```

2. Install Phase 2 dependencies (includes Phase 1 deps):

```bash
pip install -r requirements.txt
```

3. Download spaCy model (first run only):

```bash
python -m spacy download en_core_web_sm
```

4. Run the analysis script:

```bash
python scripts\analyze_documents.py
```

Optional arguments:

```bash
python scripts\analyze_documents.py --input data/processed/documents.jsonl
python scripts\analyze_documents.py --output data/processed/analyzed_documents.jsonl
python scripts\analyze_documents.py --limit 100
python scripts\analyze_documents.py --force
```

- `--force` re-processes already analyzed documents
- Without `--force`, existing analyses are skipped (idempotent)

---

## Where analysis data is stored

| Path | Purpose |
|------|---------|
| `data/processed/analyzed_documents.jsonl` | Analyzed documents with NLP results |
| `data/failed/nlp_analysis_failures.jsonl` | Documents that failed analysis |
| `logs/analysis.log` | Analysis logs |

---

## How to run tests

```bash
pytest
```

Tests use mocked model outputs and do not download models. Unit tests cover:
- Pydantic model validation (`test_nlp_models.py`)
- Text preprocessing (`test_preprocessing.py`)
- Sentiment logic (`test_sentiment.py`)
- Entity extraction (`test_entities.py`)
- Emotion classification (`test_emotions.py`)
- Concern extraction (`test_concerns.py`)
- Pipeline orchestration (`test_analyzer.py`)

Regression fixture: `tests/fixtures/sample_documents.jsonl` (10 documents covering positive/negative/neutral, privacy, security, cost, copyright, job displacement, multiple technologies, no clear concern)

---

## Project structure (Phases 1-2)

```text
technology-reaction-intelligence/
├── app/
│   ├── config/
│   │   └── settings.py
│   ├── models/
│   │   └── document.py
│   ├── ingestion/
│   │   ├── base.py
│   │   ├── rss.py
│   │   ├── normalizer.py
│   │   ├── cleaner.py
│   │   ├── deduplicator.py
│   │   └── pipeline.py
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── preprocessing.py
│   │   ├── taxonomy.py
│   │   ├── sentiment.py
│   │   ├── entities.py
│   │   ├── emotions.py
│   │   ├── concerns.py
│   │   └── analyzer.py
│   └── storage/
│       └── jsonl_store.py
├── scripts/
│   ├── ingest_news.py
│   └── analyze_documents.py
├── tests/
│   ├── test_models.py
│   ├── test_normalizer.py
│   ├── test_cleaner.py
│   ├── test_deduplicator.py
│   ├── test_rss.py
│   ├── test_pipeline.py
│   ├── test_nlp_models.py
│   ├── test_preprocessing.py
│   ├── test_sentiment.py
│   ├── test_entities.py
│   ├── test_emotions.py
│   ├── test_concerns.py
│   ├── test_analyzer.py
│   └── fixtures/
│       └── sample_documents.jsonl
├── data/
│   ├── raw/
│   ├── processed/
│   └── failed/
├── logs/
├── .env.example
├── .gitignore
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Limitations

- **Document-level analysis**: Analyzes language in collected documents, not population opinions. Does NOT establish "X% of developers believe..." or "the public is angry..."
- **No LLM synthesis**: Phase 2 uses deterministic classifiers. LLM-based explanation belongs to Phase 5.
- **Conservative concern extraction**: Only extracts concerns with direct textual evidence
- **English only**: Current models support English text
- **Model versions**: Pinned in `.env.example`; `analysis_version` tracks pipeline version for reproducibility

---

## Phase 1 verification commands

```bash
pytest
python scripts\ingest_news.py
python scripts\ingest_news.py
```

After the second run, verify that `data/processed/documents.jsonl` has not grown with duplicate documents.

---

## Phase 2 verification commands

```bash
pytest
python scripts\analyze_documents.py
python scripts\analyze_documents.py  # verify idempotency
```

After the second run, verify that `data/processed/analyzed_documents.jsonl` has not grown with duplicate analyses.

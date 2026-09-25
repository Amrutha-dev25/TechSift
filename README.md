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

## Project structure

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
│   └── storage/
│       └── jsonl_store.py
├── scripts/
│   └── ingest_news.py
├── tests/
│   ├── test_models.py
│   ├── test_normalizer.py
│   ├── test_cleaner.py
│   ├── test_deduplicator.py
│   └── test_rss.py
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

## Phase 1 verification commands

```bash
pytest
python scripts\ingest_news.py
python scripts\ingest_news.py
```

After the second run, verify that `data/processed/documents.jsonl` has not grown with duplicate documents.

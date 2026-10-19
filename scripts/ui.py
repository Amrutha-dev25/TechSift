#!/usr/bin/env python3
"""Streamlit UI for AI Backlash & Sentiment Co-Pilot (Phases 1-5)."""

import streamlit as st
import plotly.express as px
from pathlib import Path
import json
import sys
from datetime import datetime, timedelta
from collections import Counter, defaultdict

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.rag import answer_from_query, RetrievalOrchestrator
from app.rag.context_builder import ContextBuilder
from app.nlp.sentiment import get_sentiment_analyzer
from app.nlp.concerns import get_concern_extractor
from app.nlp.emotions import get_emotion_analyzer
from app.nlp.entities import get_entity_extractor

# ---------------------------------------------------------------------------
# Page config & constants
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Backlash & Sentiment Co-Pilot",
    page_icon="📊",
    layout="wide",
)

st.title("AI Backlash & Sentiment Co-Pilot")
st.caption("Phases 1–5 complete · Retrieval-augmented answers with real news citations")

# ---------------------------------------------------------------------------
# Helpers / data loading (cached)
# ---------------------------------------------------------------------------

SENTIMENT_LABELS = ["positive", "neutral", "negative"]

@st.cache_data(ttl=600)
def _load_analyzed_data():
    """Load all analyzed documents from the JSONL file."""
    docs = []
    with open("data/processed/analyzed_documents.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return docs

@st.cache_data(ttl=600)
def _sentiment_counts(docs):
    """Return dict of sentiment label -> count."""
    counts = {lbl: 0 for lbl in SENTIMENT_LABELS}
    for d in docs:
        sent = d.get("analysis", {}).get("sentiment", {})
        label = sent.get("label", "neutral")
        if label in counts:
            counts[label] += 1
    return counts

@st.cache_data(ttl=600)
def _technology_counter(docs):
    """Return Counter of technology entity texts."""
    c = Counter()
    for d in docs:
        for ent in d.get("analysis", {}).get("technology_entities", []):
            if isinstance(ent, dict):
                text = ent.get("text") or ent.get("surface_form") or str(ent)
            else:
                text = str(ent)
            if text:
                c[text] += 1
    return c

@st.cache_data(ttl=600)
def _concern_counter(docs):
    """Return Counter of concern categories."""
    c = Counter()
    for d in docs:
        for c_item in d.get("analysis", {}).get("concerns", []):
            if isinstance(c_item, dict) and "category" in c_item:
                c[c_item["category"]] += 1
    return c

@st.cache_data(ttl=600)
def _emotion_counter(docs):
    """Return Counter of emotion labels."""
    c = Counter()
    for d in docs:
        for e in d.get("analysis", {}).get("emotions", []):
            if isinstance(e, dict) and "label" in e:
                c[e["label"]] += 1
    return c

@st.cache_data(ttl=600)
def _date_distribution(docs):
    """Return dict of year -> count for documents with published dates."""
    from datetime import datetime
    counts = Counter()
    for d in docs:
        pub = d.get("published_at")
        if pub:
            try:
                dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                counts[dt.year] += 1
            except Exception:
                pass
    return counts

@st.cache_data(ttl=600)
def _load_sentiment_analyzer():
    return get_sentiment_analyzer()

@st.cache_data(ttl=600)
def _load_concern_extractor():
    return get_concern_extractor()

@st.cache_data(ttl=600)
def _load_emotion_analyzer():
    return get_emotion_analyzer()

@st.cache_data(ttl=600)
def _load_entity_extractor():
    return get_entity_extractor()


# ---------------------------------------------------------------------------
# Sidebar / filters
# ---------------------------------------------------------------------------

st.sidebar.header("📊 Filters")

docs = _load_analyzed_data()
sentiment_counts = _sentiment_counts(docs)
tech_counter = _technology_counter(docs)
concern_counter = _concern_counter(docs)
emotion_counter = _emotion_counter(docs)
date_dist = _date_distribution(docs)

sentiment_option = st.sidebar.selectbox(
    "Sentiment filter",
    options=["All", "Positive", "Neutral", "Negative"],
    index=["All", "Positive", "Neutral", "Negative"].index(
        st.session_state.get("filters_sentiment", "All")
    )
    if st.session_state.get("filters_sentiment") is not None
    else 0,
)
st.session_state.filters_sentiment = sentiment_option

date_from = st.sidebar.date_input("From", value=None)
st.session_state.filters_date_from = str(date_from) if date_from else None
st.session_state.filters_date_to = st.session_state.get("filters_date_to")

# ---------------------------------------------------------------------------
# Header / metric cards
# ---------------------------------------------------------------------------

st.title("AI Backlash & Sentiment Co-Pilot")
st.caption("Phases 1–5 complete · Retrieval-augmented answers with real news citations")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Analyzed Articles", len(docs))
with col2:
    st.metric("Positive", sentiment_counts.get("positive", 0))
with col3:
    st.metric("Neutral", sentiment_counts.get("neutral", 0))
with col4:
    st.metric("Negative", sentiment_counts.get("negative", 0))

# Trend chart (year-based)
if date_dist:
    fig_year = px.bar(
        x=list(date_dist.keys()),
        y=list(date_dist.values()),
        labels={"x": "Year", "y": "Articles"},
        title="Articles Published Over Time",
    )
    st.plotly_chart(fig_year, use_container_width=True)

# Top concerns chart
if concern_counter:
    top_concerns = dict(concern_counter.most_common(10))
    fig_concerns = px.bar(
        x=list(top_concerns.keys()),
        y=list(top_concerns.values()),
        labels={"x": "Concern", "y": "Mentions"},
        title="Top Concern Categories",
    )
    st.plotly_chart(fig_concerns, use_container_width=True)

# Emotions chart
if emotion_counter:
    top_emotions = dict(emotion_counter.most_common(10))
    fig_emotions = px.bar(
        x=list(top_emotions.keys()),
        y=list(top_emotions.values()),
        labels={"x": "Emotion", "y": "Mentions"},
        title="Emotion Distribution",
    )
    st.plotly_chart(fig_emotions, use_container_width=True)

# Technology topics
if tech_counter:
    top_tech = dict(tech_counter.most_common(10))
    fig_tech = px.bar(
        x=list(top_tech.keys()),
        y=list(top_tech.values()),
        labels={"x": "Technology", "y": "Mentions"},
        title="Top Technology Topics",
    )
    st.plotly_chart(fig_tech, use_container_width=True)

# ---------------------------------------------------------------------------
# Article explorer
# ---------------------------------------------------------------------------

st.header("📋 Article Explorer")

# Apply simple filters
filtered = docs
if sentiment_option != "All":
    filtered = [d for d in filtered if d.get("analysis", {}).get("sentiment", {}).get("label") == sentiment_option]

# Search box
search_query = st.text_input("Search articles (title/content)", "")

if search_query:
    q = search_query.lower()
    filtered = [d for d in filtered if q in (d.get("title", "") or "").lower() or q in (d.get("content", "") or "").lower()]

# Table of articles
st.dataframe(
    filtered,
    column_config={
        "title": "Title",
        "source_name": "Source",
        "published_at": "Published",
        "sentiment": "Sentiment",
        "concerns": "Concerns",
    },
    hide_index=True,
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# Ask AI panel
# ---------------------------------------------------------------------------

st.header("💬 Ask AI")

# Display chat history (simple)
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("View evidence citations"):
                for citation in msg["citations"]:
                    st.markdown(f"- **{citation.get('title', 'Untitled')}** · {citation.get('source', 'Unknown')} · {citation.get('date', 'N/A')} · [URL]({citation.get('url', '#')})")

# Question input
query = st.chat_input("What would you like to know about AI technology & society?")
if query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query, "citations": []})

    with st.chat_message("user"):
        st.markdown(query)

    # Generate answer via Gemini‑augmented RAG
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer_text = answer_from_query(query, top_k=5)

            if answer_text is None:
                st.info(
                    "Could not generate an answer – no Gemini API key or no relevant evidence. "
                    "Check that GEMINI_API_KEY is set."
                )
                citations = []
            else:
                # Parse citations from the answer text (the answer format includes SOURCE lines)
                citations = []
                for line in answer_text.split("\n"):
                    if line.startswith("SOURCE:"):
                        parts = line.replace("SOURCE:", "").strip().split("(")
                        title = parts[0].strip()
                        rest = parts[1].rstrip(")") if len(parts) > 1 else ""
                        date_match = ...  # We'll parse simply
                        # For now, also do a fresh retrieval to get proper citations
                        from app.rag import RetrievalOrchestrator
                        orch = RetrievalOrchestrator()
                        retr = orch.retrieve(query, top_k=5)
                        citations = []
                        for ev in retr.results:
                            if ev.url or ev.title:
                                citations.append(
                                    {
                                        "title": ev.title or "Untitled",
                                        "source": ev.source_name or "Unknown",
                                        "date": (
                                            datetime.fromisoformat(ev.published_at.replace("Z", "+00:00")).strftime("%b %d, %Y")
                                            if ev.published_at
                                            else "N/A"
                                        ),
                                        "url": ev.url or "#",
                                    }
                                )

            # Display the answer
            st.markdown(answer_text if answer_text else "No answer generated.")

            # Display citations expander
            if citations:
                with st.expander("View evidence citations", expanded=False):
                    for c in citations:
                        st.markdown(f"- **{c['title']}** · {c['source']} · {c['date']} · [URL]({c['url']})")
            else:
                st.info("No source citations available for this answer.")

            # Store in chat history
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer_text if answer_text else "",
                    "citations": citations,
                }
            )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.caption(
    "AI Backlash & Sentiment Co-Pilot · Phases 1–5 · Data from analyzed_documents.jsonl"
)

# ---------------------------------------------------------------------------
# Run the app
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    st.run()
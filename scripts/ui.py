#!/usr/bin/env python3
"""Streamlit UI for AI Backlash & Sentiment Co-Pilot (Phases 1-5)."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from collections import Counter, defaultdict

# Add project root to path
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.rag import answer_from_query, RetrievalOrchestrator
from app.rag.context_builder import ContextBuilder
from app.nlp.sentiment import create_pipeline as create_sentiment_pipeline
from app.nlp.concerns import create_pipeline as create_concern_pipeline
from app.nlp.emotions import create_pipeline as create_emotion_pipeline
from app.nlp.entities import create_pipeline as create_entity_pipeline

# ---------------------------------------------------------------------------
# Page config & helpers
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Backlash Co-Pilot",
    page_icon="📊",
    layout="wide",
)

st.title("🤖 AI Backlash & Sentiment Co-Pilot")
st.caption("Phases 1–5 complete · Retrieval‑augmented answers with real news citations")

# ---------------------------------------------------------------------------
# State initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"|"assistant", "content": str, "answer": str, "citations": list}

if "filters_applied" not in st.session_state:
    st.session_state.filters_applied = {
        "sentiment": None,  # "positive" | "negative" | "neutral"
        "date_from": None,
        "date_to": None,
    }

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def _load_sentiment_distribution():
    """Return sentiment counts from the Phase 2 analyzed documents."""
    from app.sentiment import create_pipeline as create_sentiment_pipeline
    pipe = create_sentiment_pipeline()
    # The pipeline expects raw documents; we read the JSONL and count.
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    try:
        with open("data/processed/analyzed_documents.jsonl", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line.strip())
                sent = d.get("analysis", {}).get("sentiment", {})
                label = sent.get("label", "neutral")
                if label in counts:
                    counts[label] += 1
    except Exception as e:
        st.error(f"Failed to load sentiment data: {e}")
    return counts


@st.cache_data(ttl=600)
def _load_technology_counter():
    """Return a Counter of technology entity texts from Phase 2 analysis."""
    from app.nlp.entities import create_pipeline as create_entity_pipeline
    pipe = create_entity_pipeline()
    counter = Counter()
    try:
        with open("data/processed/analyzed_documents.jsonl", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line.strip())
                for ent in d.get("analysis", {}).get("technology_entities", []):
                    if isinstance(ent, dict):
                        text = ent.get("text") or ent.get("surface_form") or str(ent)
                    else:
                        text = str(ent)
                    if text:
                        counter[text] += 1
    except Exception as e:
        st.error(f"Failed to load technology data: {e}")
    return counter


@st.cache_data(ttl=600)
def _load_concern_counter():
    """Return a Counter of concern categories from Phase 2 analysis."""
    from app.nlp.concerns import create_pipeline as create_concern_pipeline
    counter = Counter()
    try:
        with open("data/processed/analyzed_documents.jsonl", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line.strip())
                for c in d.get("analysis", {}).get("concerns", []):
                    if isinstance(c, dict) and "category" in c:
                        counter[c["category"]] += 1
    except Exception as e:
        st.error(f"Failed to load concern data: {e}")
    return counter


@st.cache_data(ttl=600)
def _load_emotion_counter():
    """Return a Counter of emotion labels from Phase 2 analysis."""
    from app.nlp.emotions import create_pipeline as create_emotion_pipeline
    counter = Counter()
    try:
        with open("data/processed/analyzed_documents.jsonl", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line.strip())
                for e in d.get("analysis", {}).get("emotions", []):
                    if isinstance(e, dict) and "label" in e:
                        counter[e["label"]] += 1
    except Exception as e:
        st.error(f"Failed to load emotion data: {e}")
    return counter


def _format_published(dt_str: str) -> str:
    """Turn an ISO timestamp into a nicer date, if possible."""
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %Y")
    except Exception:
        return dt_str


# ---------------------------------------------------------------------------
# Sidebar: filters
# ---------------------------------------------------------------------------

st.sidebar.header("🔎 Filters")

sentiment_filter = st.sidebar.radio(
    "Sentiment",
    options=["All", "Positive", "Neutral", "Negative"],
    index=0,
    help="Filter documents by overall sentiment label.",
)
date_from = st.sidebar.date_input("From", value=None)
date_to = st.sidebar.date_input("To", value=None)

# Apply filters to the session state
st.session_state.filters_applied = {
    "sentiment": sentiment_filter,
    "date_from": str(date_from) if date_from else None,
    "date_to": str(date_to) if date_to else None,
}

# ---------------------------------------------------------------------------
# Chat / Ask area
# ---------------------------------------------------------------------------

st.header("💬 Ask AI")

# Display chat history
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
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": query, "citations": []})

    with st.chat_message("user"):
        st.markdown(query)

    # Show a spinner while generating
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            # Build a targeted query using the user's question and any active filters
            base_query = query

            # Generate answer via the Phase 5 pipeline
            answer_text = answer_from_query(base_query, top_k=5)

            if answer_text is None:
                answer_text = "⚠️ Could not generate an answer – no OpenAI API key or no relevant evidence found. Please check that `OPENAI_API_KEY` is set."
                citations = []
            else:
                # Parse the answer and extract citations from the context.
                # The answer_generator already appends citation lines; we pull them out.
                # For a robust approach, re-run retrieval and capture the RetrievalResponse.
                orch = RetrievalOrchestrator()
                retr = orch.retrieve(base_query, top_k=5)

                citations = []
                for ev in retr.results:
                    if ev.url or ev.title or ev.published_at:
                        citations.append(
                            {
                                "title": ev.title or "Untitled",
                                "source": ev.source or "Unknown",
                                "date": format(ev.published_at, "%b %d, %Y") if ev.published_at else "N/A",
                                "url": ev.url or "#",
                            }
                        )

            # Display the answer
            st.markdown(answer_text)

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
                    "content": answer_text,
                    "citations": citations,
                }
            )

# ---------------------------------------------------------------------------
# Sentiment Dashboard
# ---------------------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.header("📊 Sentiment Dashboard")

sentiment_counts = _load_sentiment_distribution()
fig_pie = px.pie(
    values=list(sentiment_counts.values()),
    names=list(sentiment_counts.keys()),
    title="Sentiment Distribution across Documents",
    color_discrete_map={"positive": "green", "neutral": "gray", "negative": "red"},
)
st.sidebar.plotly_chart(fig_pie, use_container_width=True)

# Technology topics
tech_counter = _load_technology_counter()
if tech_counter:
    top_tech = tech_counter.most_common(10)
    tech_df = px.bar(
        x=[t[0] for t in top_tech],
        y=[t[1] for t in top_tech],
        labels={"x": "Technology", "y": "Mentions"},
        title="Top Technology Topics",
    )
    st.sidebar.plotly_chart(tech_df, use_container_width=True)

# Concerns
concern_counter = _load_concern_counter()
if concern_counter:
    top_concerns = concern_counter.most_common(10)
    concern_df = px.bar(
        x=[c[0] for c in top_concerns],
        y=[c[1] for c in top_concerns],
        labels={"x": "Concern", "y": "Mentions"},
        title="Common Concerns",
    )
    st.sidebar.plotly_chart(concern_df, use_container_width=True)

# Emotions
emotion_counter = _load_emotion_counter()
if emotion_counter:
    top_emotions = emotion_counter.most_common(10)
    emotion_df = px.bar(
        x=[e[0] for e in top_emotions],
        y=[e[1] for e in top_emotions],
        labels={"x": "Emotion", "y": "Mentions"},
        title="Emotion Distribution",
    )
    st.sidebar.plotly_chart(emotion_df, use_container_width=True)

# ---------------------------------------------------------------------------
# Footer / credits
# ---------------------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.caption(
    "Built with Streamlit · Phases 1–5 complete · "
    "Data from analyzed_documents.jsonl · OpenAI key required for answer generation"
)

# ---------------------------------------------------------------------------
# Run the app
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    st.run()
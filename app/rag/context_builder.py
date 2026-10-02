from __future__ import annotations

from typing import List
from datetime import datetime

from app.rag.models import Evidence


class ContextBuilder:
    """Builds context from retrieved evidence for LLM consumption.

    Note: This does NOT generate answers - only prepares evidence.
    """

    def build_context(self, evidence_list: List[Evidence]) -> str:
        """
        Build a formatted context string from evidence.

        Args:
            evidence_list: List of evidence items

        Returns:
            Formatted context string
        """
        if not evidence_list:
            return ""

        parts = []
        for i, evidence in enumerate(evidence_list, start=1):
            parts.append(self._format_evidence(i, evidence))

        return "\n\n---\n\n".join(parts)

    def _format_evidence(self, index: int, evidence: Evidence) -> str:
        """Format a single evidence item."""
        lines = [f"SOURCE {index}"]

        if evidence.title:
            lines.append(f"Title: {evidence.title}")
        if evidence.source:
            lines.append(f"Source: {evidence.source}")
        if evidence.published_at:
            # Format datetime
            pub_date = evidence.published_at
            if isinstance(pub_date, datetime):
                lines.append(f"Published: {pub_date.strftime('%Y-%m-%d')}")
            else:
                lines.append(f"Published: {pub_date}")
        if evidence.url:
            lines.append(f"URL: {evidence.url}")
        if evidence.retrieval_score is not None:
            lines.append(f"Retrieval score: {evidence.retrieval_score:.3f}")
        if evidence.sentiment:
            lines.append(f"Sentiment: {evidence.sentiment}")
        if evidence.concerns:
            # Format concerns
            concern_names = []
            for c in evidence.concerns[:3]:  # Limit to top 3
                if isinstance(c, dict) and "category" in c:
                    concern_names.append(c["category"])
                elif isinstance(c, dict) and "label" in c:
                    concern_names.append(c["label"])
            if concern_names:
                lines.append(f"Concerns: {', '.join(concern_names)}")

        lines.append("\nContent:")
        lines.append(evidence.text)

        return "\n".join(lines)

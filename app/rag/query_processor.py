from __future__ import annotations

from typing import Optional, Any, Dict


class QueryProcessor:
    """Processes and validates user queries."""

    MAX_QUERY_LENGTH = 2000

    def validate_query(self, query: str) -> str:
        """
        Validate and normalize a user query.

        Args:
            query: The input query

        Returns:
            Normalized query string

        Raises:
            ValueError: If query is empty, whitespace-only, or too long
        """
        if query is None:
            raise ValueError("Query cannot be None")

        # Normalize
        normalized = query.strip()

        if not normalized:
            raise ValueError("Query cannot be empty or whitespace-only")

        if len(normalized) > self.MAX_QUERY_LENGTH:
            raise ValueError(f"Query too long (max {self.MAX_QUERY_LENGTH} chars)")

        return normalized

    def extract_intent(self, query: str) -> Dict[str, Any]:
        """
        Extract basic intent from query (simple heuristic-based).

        Args:
            query: The input query

        Returns:
            Dict with intent information
        """
        query_lower = query.lower()
        intent = {
            "has_why": "why" in query_lower,
            "has_what": "what" in query_lower,
            "has_how": "how" in query_lower,
            "has_concern": "concern" in query_lower or "worry" in query_lower or "risk" in query_lower,
            "has_security": "security" in query_lower,
            "has_privacy": "privacy" in query_lower,
            "has_bias": "bias" in query_lower,
        }
        return intent

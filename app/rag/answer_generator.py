from __future__ import annotations

import os
from typing import List, Optional

from openai import OpenAI, APIError, APITimeoutError

from app.rag.models import Evidence, RetrievalResponse
from app.rag.context_builder import ContextBuilder


class AnswerGenerator:
    """Generates grounded answers from retrieved evidence using OpenAI.

    Rules:
    - Never invent citations or facts
    - If evidence is insufficient, say so clearly
    - Return answer, source title, publisher/date, URL, evidence IDs
    - Handle missing keys, API failures, timeouts gracefully
    """

    def __init__(self) -> None:
        self._api_key = os.environ.get("OPENAI_API_KEY")
        self._model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._client: Optional[OpenAI] = (
            OpenAI(api_key=self._api_key) if self._api_key else None
        )
        self._context_builder = ContextBuilder()

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(self, retrieval_response: RetrievalResponse) -> Optional[str]:
        """Generate a grounded answer from a RetrievalResponse.

        Args:
            retrieval_response: The retrieval results from Phase 4

        Returns:
            Generated answer string, or None if cannot generate
        """
        # No API key -> cannot generate; return None (caller handles)
        if not self._client:
            return None

        # No evidence -> cannot generate
        if not retrieval_response.results:
            return None

        # Build context from evidence
        context = self._context_builder.build_context(retrieval_response.results)

        # Build user message: query + context
        user_message = _build_user_message(
            retrieval_response.query, context, retrieval_response.results
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": user_message}],
                temperature=0.3,
            )
            return response.choices[0].message.content

        except APITimeoutError:
            # Timeout – don't crash; just report limitation
            return None

        except APIError:
            # Other API error – don't crash
            return None

        except Exception:
            # Any other unexpected error – be safe
            return None


# ----------------------------------------------------------------------
# Helper: build the prompt the LLM sees
# ----------------------------------------------------------------------


def _build_user_message(query: str, context: str, results: List[Evidence]) -> str:
    """Construct the prompt for the LLM.

    The LLM receives:
    1. The user's original question
    2. The formatted evidence context (SOURCE 1/2/3 … with title, source,
       published date, URL, content)
    3. An explicit instruction to answer from the evidence only
    """
    lines = [
        f"USER QUESTION: {query}",
        "",
        "EVIDENCE",
        context,
        "",
        "INSTRUCTION:",
        "- Answer the user question using ONLY the evidence above.",
        "- Do NOT add facts that were not retrieved.",
        "- Provide a concise answer (2-4 sentences).",
        "- At the end, list each source as:",
        "-   SOURCE: <title> (<published>), URL: <url>",
        "- If a source has no title or date, state that honestly.",
        "- If the evidence is insufficient to answer, say so clearly.",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Convenience wrapper used by the CLI / orchestrator
# ----------------------------------------------------------------------


def answer_from_query(query: str, top_k: int = 10) -> Optional[str]:
    """End-to-end: retrieve → generate → return answer string.

    Returns None if no OpenAI key, no evidence, or generation fails.
    """
    from app.rag import RetrievalOrchestrator

    orchestrator = RetrievalOrchestrator()
    retrieval = orchestrator.retrieve(query, top_k=top_k)

    generator = AnswerGenerator()
    return generator.generate(retrieval)
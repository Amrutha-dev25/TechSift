#!/usr/bin/env python3
"""CLI for Phase 5 - Ask a question and get a grounded RAG answer."""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.rag.answer_generator import answer_from_query


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4+5 RAG: retrieve evidence + generate answer")
    parser.add_argument("--query", required=True, help="Natural language query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of evidence results")
    args = parser.parse_args()

    answer = answer_from_query(args.query, top_k=args.top_k)

    if answer is None:
        print("Could not generate answer (no OpenAI key or no evidence).", file=sys.stderr)
        return 1

    print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
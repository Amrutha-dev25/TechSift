#!/usr/bin/env python3
"""Query CLI for Phase 4 - Retrieve evidence from knowledge base."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.rag import RetrievalOrchestrator


def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def main() -> int:
    parser = argparse.ArgumentParser(description="Query knowledge base for evidence (Phase 4)")
    parser.add_argument("--query", required=True, help="Natural language query")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return")
    parser.add_argument("--sentiment", choices=["positive", "negative", "neutral"], help="Filter by sentiment")
    parser.add_argument("--sentiment-min", type=float, help="Min sentiment score (-1.0 to 1.0)")
    parser.add_argument("--sentiment-max", type=float, help="Max sentiment score (-1.0 to 1.0)")
    parser.add_argument("--technology", help="Filter by technology")
    parser.add_argument("--concern", help="Filter by concern category")
    parser.add_argument("--source", help="Filter by source name")
    parser.add_argument("--source-type", help="Filter by source type")
    parser.add_argument("--date-from", type=parse_date, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--date-to", type=parse_date, help="End date (YYYY-MM-DD)")
    parser.add_argument("--mode", choices=["balanced", "concern", "hype"], default="balanced", help="Retrieval mode")

    args = parser.parse_args()

    try:
        orchestrator = RetrievalOrchestrator()
        response = orchestrator.retrieve(
            query=args.query,
            top_k=args.top_k,
            sentiment_filter=args.sentiment,
            sentiment_min=args.sentiment_min,
            sentiment_max=args.sentiment_max,
            technology_filter=args.technology,
            concern_filter=args.concern,
            source_filter=args.source,
            source_type_filter=args.source_type,
            date_from=args.date_from,
            date_to=args.date_to,
            retrieval_mode=args.mode,
        )

        # Display results
        print("=" * 60)
        print("QUERY")
        print("=" * 60)
        print(response.query)
        print(f"\nNumber of results: {response.total_results}")
        print("=" * 60)

        if not response.results:
            print("No relevant evidence found.")
            return 0

        for i, evidence in enumerate(response.results, 1):
            print(f"\nRESULT {i}")
            print("-" * 60)
            if evidence.title:
                print(f"Title: {evidence.title}")
            if evidence.source:
                print(f"Source: {evidence.source}")
            if evidence.published_at:
                pub = evidence.published_at
                if isinstance(pub, datetime):
                    print(f"Published: {pub.strftime('%Y-%m-%d')}")
                else:
                    print(f"Published: {pub}")
            if evidence.retrieval_score is not None:
                print(f"Retrieval score: {evidence.retrieval_score:.3f}")
            if evidence.url:
                print(f"URL: {evidence.url}")
            if evidence.sentiment:
                print(f"Sentiment: {evidence.sentiment}")
            if evidence.concerns:
                concern_names = []
                for c in evidence.concerns[:3]:
                    if isinstance(c, dict):
                        concern_names.append(c.get("category", c.get("label", "unknown")))
                if concern_names:
                    print(f"Concerns: {', '.join(concern_names)}")

            print(f"\nEvidence text:")
            # Print truncated text for readability
            text = evidence.text
            if len(text) > 400:
                text = text[:397] + "..."
            print(text)
            print("-" * 60)

        print("\n" + "=" * 60)
        print("END OF RESULTS")
        print("=" * 60)

        return 0

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

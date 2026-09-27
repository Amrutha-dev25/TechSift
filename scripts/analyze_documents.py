#!/usr/bin/env python3
import argparse
import logging
import sys
import time
from pathlib import Path

from app.config.settings import settings
from app.nlp.analyzer import create_pipeline

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(log_level: str, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "analysis.log"

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )

    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("spacy").setLevel(logging.WARNING)


def print_report(stats: dict, duration: float) -> None:
    print("\n" + "=" * 40)
    print("Technology Reaction Intelligence")
    print("Phase 2 NLP Analysis Report")
    print("=" * 40)
    print()
    print(f"Documents attempted : {stats['attempted']}")
    print(f"Successfully analyzed: {stats['success']}")
    print(f"Failed               : {stats['failed']}")
    print(f"Skipped (existing)   : {stats['skipped']}")
    print()
    print("Sentiment:")
    print(f"  Positive             : {stats['sentiment_counts']['positive']}")
    print(f"  Neutral              : {stats['sentiment_counts']['neutral']}")
    print(f"  Negative             : {stats['sentiment_counts']['negative']}")
    print()
    print("Documents with:")
    print(f"  Entities             : {stats['entities_count']}")
    print(f"  Technology entities  : {stats['tech_entities_count']}")
    print(f"  Concerns             : {stats['concerns_count']}")
    print(f"  Emotion signals      : {stats['emotions_count']}")
    print()
    print(f"Processing time: {duration:.2f}s ({stats['attempted']/duration:.1f} docs/sec)" if duration > 0 else "Processing time: N/A")
    print()
    print("Output:")
    print(f"  {settings.processed_data_dir / 'analyzed_documents.jsonl'}")
    print()
    print("Failures:")
    print(f"  {settings.failed_data_dir / 'nlp_analysis_failures.jsonl'}")
    print("=" * 40)


def main() -> int:
    parser = argparse.ArgumentParser(description="Technology News NLP Analysis Pipeline (Phase 2)")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input JSONL file (default: data/processed/documents.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL file (default: data/processed/analyzed_documents.jsonl)",
    )
    parser.add_argument(
        "--failures",
        type=Path,
        default=None,
        help="Failures JSONL file (default: data/failed/nlp_analysis_failures.jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to process",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-analysis of already processed documents",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default from config)",
    )

    args = parser.parse_args()

    log_level = args.log_level or settings.log_level
    setup_logging(log_level, settings.log_dir)

    logger = logging.getLogger(__name__)
    logger.info("Starting Phase 2 NLP analysis pipeline")

    pipeline = create_pipeline(
        input_path=args.input,
        output_path=args.output,
        failure_path=args.failures,
    )

    start_time = time.time()
    stats = pipeline.run(limit=args.limit, force=args.force)
    duration = time.time() - start_time

    print_report(stats, duration)

    if stats["success"] == 0 and stats["attempted"] > 0:
        logger.error("All documents failed analysis")
        return 1

    if stats["failed"] > 0:
        logger.warning("Some documents failed: %d/%d", stats["failed"], stats["attempted"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
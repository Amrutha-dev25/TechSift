#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

from app.config.settings import settings
from app.ingestion.pipeline import IngestionPipeline

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(log_level: str, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "ingestion.log"

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("feedparser").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def print_report(result) -> None:
    print("\n" + "=" * 40)
    print("Technology Reaction Intelligence")
    print("Phase 1 Ingestion Report")
    print("=" * 40)
    print()
    print(f"Feeds attempted : {result.feeds_attempted}")
    print(f"Feeds succeeded : {result.feeds_succeeded}")
    print(f"Feeds failed    : {result.feeds_failed}")
    print()
    print(f"Entries fetched : {result.entries_seen}")
    print(f"Accepted        : {result.documents_accepted}")
    print(f"Rejected        : {result.documents_rejected}")
    print(f"Duplicates      : {result.duplicates}")
    print()
    print(f"Output: {settings.processed_data_dir / 'documents.jsonl'}")
    print("=" * 40)


def main() -> int:
    parser = argparse.ArgumentParser(description="Technology News Ingestion Pipeline")
    parser.add_argument(
        "--max-per-feed",
        type=int,
        default=None,
        help="Maximum articles per feed (default from config)",
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
    logger.info("Starting Phase 1 ingestion pipeline")

    pipeline = IngestionPipeline(max_articles_per_feed=args.max_per_feed)
    result = pipeline.run()

    print_report(result)

    if result.feeds_succeeded == 0 and result.feeds_attempted > 0:
        logger.error("All feeds failed")
        return 1

    if result.feeds_failed > 0:
        logger.warning("Some feeds failed: %d/%d", result.feeds_failed, result.feeds_attempted)

    return 0


if __name__ == "__main__":
    sys.exit(main())
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RSSSourceConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    source_id: str
    source_name: str
    source_type: Literal["rss"] = "rss"
    feed_url: str
    enabled: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    request_timeout_seconds: int = Field(default=15, alias="REQUEST_TIMEOUT_SECONDS")
    max_retries: int = Field(default=3, alias="MAX_RETRIES")

    max_articles_per_feed: int = Field(default=50, alias="MAX_ARTICLES_PER_FEED")

    raw_data_dir: Path = Field(default=Path("data/raw"), alias="RAW_DATA_DIR")
    processed_data_dir: Path = Field(default=Path("data/processed"), alias="PROCESSED_DATA_DIR")
    failed_data_dir: Path = Field(default=Path("data/failed"), alias="FAILED_DATA_DIR")
    log_dir: Path = Field(default=Path("logs"), alias="LOG_DIR")

    @property
    def rss_sources(self) -> list[RSSSourceConfig]:
        return [
            RSSSourceConfig(
                source_id="techcrunch_ai",
                source_name="TechCrunch AI",
                feed_url="https://techcrunch.com/category/artificial-intelligence/feed/",
                enabled=True,
            ),
            RSSSourceConfig(
                source_id="theverge_ai",
                source_name="The Verge AI",
                feed_url="https://www.theverge.com/ai-artificial-intelligence/rss/index.xml",
                enabled=True,
            ),
            RSSSourceConfig(
                source_id="ars_technica_ai",
                source_name="Ars Technica AI",
                feed_url="https://feeds.arstechnica.com/arstechnica/technology-lab",
                enabled=True,
            ),
            RSSSourceConfig(
                source_id="venturebeat_ai",
                source_name="VentureBeat AI",
                feed_url="https://venturebeat.com/category/ai/feed/",
                enabled=True,
            ),
            RSSSourceConfig(
                source_id="wired_ai",
                source_name="Wired AI",
                feed_url="https://www.wired.com/feed/tag/ai/latest/rss",
                enabled=True,
            ),
        ]


settings = Settings()
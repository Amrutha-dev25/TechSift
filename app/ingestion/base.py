from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from app.models.document import RawDocument

T = TypeVar("T", bound=RawDocument)


class DataSource(ABC, Generic[T]):
    @abstractmethod
    def fetch(self) -> list[T]:
        """Fetch raw documents from the source."""
        ...

    @property
    @abstractmethod
    def source_id(self) -> str:
        """Unique identifier for this source."""
        ...

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable name for this source."""
        ...

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Type of source (e.g., 'rss', 'news_api')."""
        ...
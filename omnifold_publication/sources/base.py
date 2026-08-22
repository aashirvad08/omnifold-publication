"""Abstract data-source interface for event data.

Every backend (local disk / NERSC GPFS, Zenodo, user upload, future
S3/URL) implements the same three operations — column-selective event
reading, metadata, and checksumming — so downstream code never needs to
know where a file physically lives. Swapping one source for another
requires zero changes to calling code.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from ..exceptions import OmniFoldPublicationError


class DataSourceError(OmniFoldPublicationError):
    """Raised when a data source cannot provide what was asked of it."""


class DataSource(ABC):
    """Common interface over every event-data backend.

    Contract:

    - :meth:`open_events` returns a DataFrame, reading only the requested
      columns where the storage format allows it; implementations must
      never buffer an entire multi-GB file when a cheaper read exists.
    - :meth:`metadata` returns a JSON-compatible description of the
      source.
    - :meth:`checksum` returns a SHA-256 hex digest of the underlying
      file when one is (or can be made) available locally, else None.
    """

    kind: str = "base"

    @abstractmethod
    def open_events(
        self, columns: list[str] | None = None
    ) -> pd.DataFrame:
        """Load event data, restricted to ``columns`` when given."""

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Describe this source (provenance, files, identifiers)."""

    @abstractmethod
    def checksum(self) -> str | None:
        """SHA-256 hex digest of the underlying data, if obtainable."""

    def health_check(self) -> bool:
        """Whether the source is currently usable; subclasses override."""

        return True

    @staticmethod
    def _sha256_of_file(
        path: str | Path, chunk_size: int = 8 * 1024 * 1024
    ) -> str:
        """Streaming SHA-256 of a file, never loading it whole."""

        sha256 = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(chunk_size), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} kind={self.kind!r}>"

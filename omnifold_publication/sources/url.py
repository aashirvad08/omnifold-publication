"""Generic URL data source (future work).

Stub with a realistic constructor. When built, this would generalise the
chunked-download path of
:class:`~omnifold_publication.sources.zenodo.ZenodoDataSource` to any
HTTP(S) URL (with an ``fsspec`` Parquet range-read optimization), minus
Zenodo's record-metadata API. Until then, download the file and use
:class:`~omnifold_publication.sources.local.LocalDataSource`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..reader import DEFAULT_HDF_KEY
from .base import DataSource


_NOT_IMPLEMENTED = (
    "UrlDataSource is not implemented yet. For a Zenodo record use "
    "ZenodoDataSource; otherwise download the file and use "
    "LocalDataSource."
)


class UrlDataSource(DataSource):
    """Event data at an arbitrary HTTP(S) URL — not yet implemented."""

    kind = "url"

    def __init__(
        self,
        url: str,
        cache_dir: str | Path | None = None,
        hdf5_key: str = DEFAULT_HDF_KEY,
        max_retries: int = 3,
    ):
        self.url = url
        self.cache_dir = cache_dir
        self.hdf5_key = hdf5_key
        self.max_retries = max_retries

    def open_events(self, columns: list[str] | None = None):
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def checksum(self) -> str | None:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def health_check(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"<{type(self).__name__} kind={self.kind!r} url={self.url!r} (stub)>"

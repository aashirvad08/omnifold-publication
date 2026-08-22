"""S3 data source (future work).

Stub with a realistic constructor so calling code and the factory can be
wired up now; the read paths are not implemented yet. When built, this
would stream column-selective Parquet reads directly from S3 via
``pyarrow`` + ``s3fs`` and fall back to a chunked download for HDF5,
mirroring :class:`~omnifold_publication.sources.zenodo.ZenodoDataSource`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..reader import DEFAULT_HDF_KEY
from .base import DataSource


_NOT_IMPLEMENTED = (
    "S3DataSource is not implemented yet. Use LocalDataSource for a file "
    "already synced from S3, or open an issue to prioritise native S3 "
    "streaming (planned via pyarrow + s3fs)."
)


class S3DataSource(DataSource):
    """Event data in an S3 (or S3-compatible) bucket — not yet implemented."""

    kind = "s3"

    def __init__(
        self,
        uri: str,
        cache_dir: str | Path | None = None,
        hdf5_key: str = DEFAULT_HDF_KEY,
        endpoint_url: str | None = None,
        anonymous: bool = False,
    ):
        self.uri = uri
        self.cache_dir = cache_dir
        self.hdf5_key = hdf5_key
        self.endpoint_url = endpoint_url
        self.anonymous = anonymous

    def open_events(self, columns: list[str] | None = None):
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def checksum(self) -> str | None:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def health_check(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"<{type(self).__name__} kind={self.kind!r} uri={self.uri!r} (stub)>"

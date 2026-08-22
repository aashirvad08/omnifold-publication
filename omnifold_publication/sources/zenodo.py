"""Zenodo data source: stream event files from a public Zenodo record.

Two read paths:

- **Chunked download (primary, always available):** the file is streamed
  to ``cache_dir`` in chunks and read from disk. Works for any format and
  needs only the standard library. This is the path the real ATLAS Z+jets
  release uses, because that record ships HDF5 (and, in fact, a single
  ``files.zip``), which cannot be column-projected over HTTP.
- **Parquet range reads (optional optimization):** when the target is a
  Parquet file *and* ``fsspec`` is importable, only the requested columns
  are fetched via HTTP range requests, without downloading the whole
  file. If ``fsspec`` is missing or the range read fails, the source
  falls back to the chunked-download path transparently.

Network calls retry with exponential backoff and raise
:class:`DataSourceError` once ``max_retries`` is exhausted. Reuses the
standard-library ``urllib`` fetch style of :mod:`omnifold_publication.hepdata`
rather than adding a hard ``requests``/``fsspec`` dependency.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..reader import DEFAULT_HDF_KEY
from .base import DataSource, DataSourceError
from .local import HDF5_SUFFIXES, PARQUET_SUFFIXES, LocalDataSource


ZENODO_API_BASE = "https://zenodo.org/api"
_TIMEOUT_SECONDS = 30.0
_USER_AGENT = "omnifold-publication/0.2"
DEFAULT_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 0.5
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024


def _backoff_sleep(attempt: int) -> None:
    """Exponential backoff between retries (its own seam for tests)."""

    time.sleep(_BACKOFF_BASE_SECONDS * (2 ** attempt))


class ZenodoDataSource(DataSource):
    """Event data hosted in a public Zenodo record.

    Provide either ``record_id`` (with an optional ``filename`` when the
    record holds more than one file) or a direct ``file_url``.
    """

    kind = "zenodo"

    def __init__(
        self,
        record_id: str | int | None = None,
        file_url: str | None = None,
        filename: str | None = None,
        cache_dir: str | Path | None = None,
        hdf5_key: str = DEFAULT_HDF_KEY,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        if record_id is None and file_url is None:
            raise DataSourceError(
                "ZenodoDataSource requires either record_id or file_url."
            )
        self.record_id = str(record_id) if record_id is not None else None
        self.file_url = file_url
        self.filename = filename
        self.hdf5_key = hdf5_key
        self.max_retries = max(1, int(max_retries))
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else Path(__import__("tempfile").gettempdir()) / "omnifold_zenodo_cache"
        )
        self._record: dict[str, Any] | None = None

    # -- network primitives (single urlopen seam for mocking) ------------

    def _with_retries(self, func, description: str):
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return func()
            except (urllib.error.URLError, TimeoutError) as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    _backoff_sleep(attempt)
        raise DataSourceError(
            f"{description} failed after {self.max_retries} attempts: "
            f"{last_exc}"
        )

    def _fetch_json(self, url: str) -> dict[str, Any]:
        def _once() -> dict[str, Any]:
            request = urllib.request.Request(
                url, headers={"User-Agent": _USER_AGENT}
            )
            with urllib.request.urlopen(
                request, timeout=_TIMEOUT_SECONDS
            ) as response:
                return json.loads(response.read().decode("utf-8"))

        return self._with_retries(_once, f"Fetching {url}")

    def _download(self, url: str, dest: Path) -> Path:
        def _once() -> Path:
            request = urllib.request.Request(
                url, headers={"User-Agent": _USER_AGENT}
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            partial = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(
                request, timeout=_TIMEOUT_SECONDS
            ) as response, partial.open("wb") as sink:
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    sink.write(chunk)
            partial.replace(dest)
            return dest

        return self._with_retries(_once, f"Downloading {url}")

    # -- record metadata -------------------------------------------------

    def _record_json(self) -> dict[str, Any]:
        if self._record is None:
            if self.record_id is None:
                raise DataSourceError(
                    "Record metadata is unavailable for a direct file_url; "
                    "construct with record_id to read Zenodo metadata."
                )
            self._record = self._fetch_json(
                f"{ZENODO_API_BASE}/records/{self.record_id}"
            )
        return self._record

    def _file_entry(self) -> dict[str, Any]:
        files = self._record_json().get("files") or []
        if not files:
            raise DataSourceError(
                f"Zenodo record {self.record_id} lists no files."
            )
        if self.filename is not None:
            for entry in files:
                if entry.get("key") == self.filename:
                    return entry
            available = ", ".join(str(f.get("key")) for f in files)
            raise DataSourceError(
                f"File {self.filename!r} not found in Zenodo record "
                f"{self.record_id}; available: {available}."
            )
        if len(files) > 1:
            available = ", ".join(str(f.get("key")) for f in files)
            raise DataSourceError(
                f"Zenodo record {self.record_id} has multiple files; pass "
                f"filename= to choose one. Available: {available}."
            )
        return files[0]

    def _resolve_file_url(self) -> str:
        if self.file_url is not None:
            return self.file_url
        entry = self._file_entry()
        link = (entry.get("links") or {}).get("self")
        if not link:
            raise DataSourceError(
                f"Zenodo file entry for record {self.record_id} has no "
                "download link."
            )
        return link

    def _cache_path(self) -> Path:
        if self.filename is not None:
            name = self.filename
        elif self.file_url is not None:
            name = Path(urlparse(self.file_url).path).name or "download"
        else:
            name = str(self._file_entry().get("key") or "download")
        return self.cache_dir / name

    def _ensure_cached(self) -> Path:
        url = self._resolve_file_url()
        dest = self._cache_path()
        if not dest.is_file():
            self._download(url, dest)
        return dest

    # -- DataSource interface --------------------------------------------

    def open_events(self, columns: list[str] | None = None):
        # optional optimization: column-selective Parquet over HTTP range
        # requests, only when fsspec is present and we have a file URL
        suffix = self._cache_path().suffix.lower()
        if suffix in PARQUET_SUFFIXES:
            projected = self._try_parquet_range_read(columns)
            if projected is not None:
                return projected
        cached = self._ensure_cached()
        return LocalDataSource(cached, hdf5_key=self.hdf5_key).open_events(
            columns=columns
        )

    def _try_parquet_range_read(self, columns: list[str] | None):
        try:
            import fsspec  # noqa: F401
            import pyarrow.parquet as pq
        except ImportError:
            return None
        try:
            url = self._resolve_file_url()
            with fsspec.open(url, "rb") as handle:
                return pq.read_table(handle, columns=columns).to_pandas()
        except Exception:
            # any failure (range unsupported, network, parse) falls back
            # to the guaranteed chunked-download path
            return None

    def metadata(self) -> dict[str, Any]:
        record = self._record_json()
        meta = record.get("metadata") or {}
        return {
            "kind": self.kind,
            "record_id": self.record_id,
            "doi": record.get("doi"),
            "title": record.get("title") or meta.get("title"),
            "files": [
                {
                    "key": entry.get("key"),
                    "size": entry.get("size"),
                    "checksum": entry.get("checksum"),
                }
                for entry in (record.get("files") or [])
            ],
        }

    def checksum(self) -> str | None:
        """The provider's checksum, self-labeled with its algorithm.

        Zenodo publishes an MD5 digest (e.g. ``"md5:9b83..."``); it is
        returned verbatim so the algorithm is never misrepresented as
        SHA-256. Use :meth:`sha256_of_cache` for a SHA-256 of the
        downloaded file.
        """

        if self.file_url is not None and self.record_id is None:
            return None
        return self._file_entry().get("checksum")

    def sha256_of_cache(self) -> str | None:
        """SHA-256 of the downloaded file, or None if not yet cached."""

        dest = self._cache_path()
        return self._sha256_of_file(dest) if dest.is_file() else None

    def health_check(self) -> bool:
        try:
            if self.record_id is not None:
                self._record_json()
            else:
                request = urllib.request.Request(
                    self.file_url, method="HEAD",
                    headers={"User-Agent": _USER_AGENT},
                )
                with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS):
                    pass
            return True
        except (urllib.error.URLError, TimeoutError, DataSourceError):
            return False

    def __repr__(self) -> str:
        target = self.file_url or f"record {self.record_id}"
        return f"<{type(self).__name__} kind={self.kind!r} {target}>"

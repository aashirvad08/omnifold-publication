"""User-upload data source: chunked receive, format sniffing, cleanup.

Uploaded files are private inputs, not publication artifacts: they must
not outlive the job that used them. Two disposal paths are supported —
explicit :meth:`UploadDataSource.cleanup` (also via context manager) when
the job finishes, and :func:`sweep_expired_uploads` for abandoned files
older than a TTL window.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, BinaryIO

from ..reader import DEFAULT_HDF_KEY
from .base import DataSource, DataSourceError
from .local import HDF5_SUFFIXES, PARQUET_SUFFIXES, LocalDataSource


DEFAULT_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
DEFAULT_TTL_SECONDS = 24 * 60 * 60
RECEIVE_CHUNK_BYTES = 8 * 1024 * 1024

# real file signatures, not extensions: Parquet files open with the
# "PAR1" magic; HDF5 superblocks open with \x89HDF\r\n\x1a\n at offset 0
PARQUET_MAGIC = b"PAR1"
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def _expected_magic(suffix: str) -> bytes:
    if suffix in PARQUET_SUFFIXES:
        return PARQUET_MAGIC
    if suffix in HDF5_SUFFIXES:
        return HDF5_MAGIC
    raise DataSourceError(
        f"Unsupported upload suffix {suffix!r}; expected one of: "
        f"{', '.join(sorted(PARQUET_SUFFIXES | HDF5_SUFFIXES))}."
    )


def _validate_magic(path: Path) -> None:
    expected = _expected_magic(path.suffix.lower())
    with path.open("rb") as stream:
        head = stream.read(len(expected))
    if head != expected:
        raise DataSourceError(
            f"Upload {path.name!r} does not contain a valid "
            f"{path.suffix.lower()} file (magic bytes mismatch)."
        )


def sweep_expired_uploads(
    upload_dir: str | Path,
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
    now: float | None = None,
) -> list[Path]:
    """Delete files in ``upload_dir`` older than the TTL; return them."""

    upload_dir = Path(upload_dir)
    if not upload_dir.is_dir():
        return []
    cutoff = (now if now is not None else time.time()) - ttl_seconds
    removed: list[Path] = []
    for path in sorted(upload_dir.iterdir()):
        if path.is_file() and path.stat().st_mtime < cutoff:
            path.unlink()
            removed.append(path)
    return removed


class UploadDataSource(DataSource):
    """A user-uploaded event file staged on local disk.

    Create with :meth:`receive`, which streams the incoming bytes to disk
    in chunks (never buffering the file in memory), enforces a size
    limit, and verifies the real file format from its magic bytes before
    accepting it. Reading then behaves exactly like
    :class:`LocalDataSource`.
    """

    kind = "upload"

    def __init__(
        self,
        stored_path: str | Path,
        hdf5_key: str = DEFAULT_HDF_KEY,
    ):
        self.path = Path(stored_path)
        self._local = LocalDataSource(self.path, hdf5_key=hdf5_key)
        self.received_at = (
            self.path.stat().st_mtime if self.path.is_file() else None
        )

    @classmethod
    def receive(
        cls,
        stream: BinaryIO,
        upload_dir: str | Path,
        filename: str,
        max_bytes: int = DEFAULT_MAX_BYTES,
        hdf5_key: str = DEFAULT_HDF_KEY,
    ) -> "UploadDataSource":
        """Stream an incoming file to disk, validate it, and wrap it."""

        upload_dir = Path(upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        # never trust a client-supplied path; keep the basename only
        target = upload_dir / Path(filename).name
        _expected_magic(target.suffix.lower())  # reject bad suffix early

        written = 0
        try:
            with target.open("wb") as sink:
                while True:
                    chunk = stream.read(RECEIVE_CHUNK_BYTES)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        raise DataSourceError(
                            f"Upload {target.name!r} exceeds the size "
                            f"limit of {max_bytes} bytes."
                        )
                    sink.write(chunk)
            _validate_magic(target)
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return cls(target, hdf5_key=hdf5_key)

    def cleanup(self) -> None:
        """Delete the uploaded file (and its sidecar metadata, if any)."""

        self.path.unlink(missing_ok=True)
        self._local.metadata_path.unlink(missing_ok=True)

    def __enter__(self) -> "UploadDataSource":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.cleanup()

    def health_check(self) -> bool:
        return self._local.health_check()

    def open_events(self, columns: list[str] | None = None):
        return self._local.open_events(columns=columns)

    def metadata(self) -> dict[str, Any]:
        info = self._local.metadata()
        info["kind"] = self.kind
        if self.received_at is not None:
            info["received_at"] = self.received_at
        return info

    def checksum(self) -> str | None:
        return self._local.checksum()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} kind={self.kind!r} path={str(self.path)!r}>"

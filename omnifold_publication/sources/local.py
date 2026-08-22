"""Local-disk / NERSC-GPFS data source."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from ..reader import DEFAULT_HDF_KEY
from .base import DataSource, DataSourceError


PARQUET_SUFFIXES = {".parquet"}
HDF5_SUFFIXES = {".h5", ".hdf5"}


class LocalDataSource(DataSource):
    """Event data on a locally mounted filesystem (including GPFS).

    ``hdf5_key`` defaults to the key this package's own writer and reader
    use ("df"). A sidecar ``<file>.metadata.json`` next to the data file,
    when present, is returned by :meth:`metadata`.
    """

    kind = "local"

    def __init__(
        self,
        path: str | Path,
        hdf5_key: str = DEFAULT_HDF_KEY,
        metadata_path: str | Path | None = None,
    ):
        self.path = Path(path)
        self.hdf5_key = hdf5_key
        self.metadata_path = (
            Path(metadata_path)
            if metadata_path is not None
            else Path(f"{self.path}.metadata.json")
        )

    def health_check(self) -> bool:
        return self.path.is_file() and os.access(self.path, os.R_OK)

    def _require_file(self) -> None:
        if not self.path.is_file():
            raise DataSourceError(f"Data file not found: {self.path}")

    def open_events(
        self, columns: list[str] | None = None
    ) -> pd.DataFrame:
        self._require_file()
        suffix = self.path.suffix.lower()
        try:
            if suffix in PARQUET_SUFFIXES:
                return pd.read_parquet(self.path, columns=columns)
            if suffix in HDF5_SUFFIXES:
                try:
                    return pd.read_hdf(
                        self.path, key=self.hdf5_key, columns=columns
                    )
                except (TypeError, ValueError):
                    # fixed-format HDF5 stores do not support column
                    # projection; fall back to full load + slice
                    frame = pd.read_hdf(self.path, key=self.hdf5_key)
                    return frame if columns is None else frame[columns]
        except DataSourceError:
            raise
        except Exception as exc:
            raise DataSourceError(
                f"Failed to read {self.path}: {exc}"
            ) from exc
        raise DataSourceError(
            f"Unsupported data file suffix {suffix!r} for {self.path}; "
            f"expected one of: "
            f"{', '.join(sorted(PARQUET_SUFFIXES | HDF5_SUFFIXES))}."
        )

    def metadata(self) -> dict[str, Any]:
        if self.metadata_path.is_file():
            with self.metadata_path.open("r", encoding="utf-8") as stream:
                loaded = json.load(stream)
            if isinstance(loaded, dict):
                return loaded
            raise DataSourceError(
                f"Sidecar metadata must be a JSON object: {self.metadata_path}"
            )
        info: dict[str, Any] = {
            "kind": self.kind,
            "path": str(self.path),
        }
        if self.path.is_file():
            info["size_bytes"] = self.path.stat().st_size
        return info

    def checksum(self) -> str | None:
        if not self.path.is_file():
            return None
        return self._sha256_of_file(self.path)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} kind={self.kind!r} path={str(self.path)!r}>"

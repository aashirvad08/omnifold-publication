"""Data-access layer: one interface, many backends.

Every source implements :class:`DataSource`; calling code reads events
through it without knowing where the file lives. Use :func:`get_source`
to build one from a prefixed spec string.
"""

from __future__ import annotations

from pathlib import Path

from .base import DataSource, DataSourceError
from .local import LocalDataSource
from .s3 import S3DataSource
from .upload import UploadDataSource, sweep_expired_uploads
from .url import UrlDataSource
from .zenodo import ZenodoDataSource


_PREFIX_SOURCES: dict[str, type[DataSource]] = {
    "local": LocalDataSource,
    "upload": UploadDataSource,
    "zenodo": ZenodoDataSource,
    "s3": S3DataSource,
    "url": UrlDataSource,
}
# sources whose spec is itself a scheme-prefixed URI ("s3://bucket/key"):
# the prefix is part of the value, so pass the whole spec, not the remainder
_URI_SCHEME_SOURCES = {"s3"}


def get_source(spec: str, **kwargs) -> DataSource:
    """Build a data source from a prefixed spec string.

    Dispatches on a ``<prefix>:`` head — ``local:``, ``upload:``,
    ``zenodo:``, ``s3:``, ``url:``. A bare path to an existing file
    defaults to :class:`LocalDataSource`. Scheme-style specs like
    ``s3://bucket/key`` keep their scheme; ``zenodo:11507450`` and
    ``url:https://...`` pass everything after the first colon.
    """

    prefix, sep, remainder = spec.partition(":")
    if sep and remainder and prefix in _PREFIX_SOURCES:
        argument = spec if prefix in _URI_SCHEME_SOURCES else remainder
        return _PREFIX_SOURCES[prefix](argument, **kwargs)

    if Path(spec).is_file():
        return LocalDataSource(spec, **kwargs)

    known = ", ".join(sorted(_PREFIX_SOURCES))
    raise DataSourceError(
        f"Cannot resolve data source spec {spec!r}: not a known prefix "
        f"({known}) and not an existing file."
    )


__all__ = [
    "DataSource",
    "DataSourceError",
    "LocalDataSource",
    "UploadDataSource",
    "ZenodoDataSource",
    "S3DataSource",
    "UrlDataSource",
    "get_source",
    "sweep_expired_uploads",
]

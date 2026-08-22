"""Tests for the abstract DataSource interface."""

from __future__ import annotations

import hashlib

import pytest

from omnifold_publication.exceptions import OmniFoldPublicationError
from omnifold_publication.sources import DataSource, DataSourceError


def test_abstract_class_cannot_be_instantiated():
    with pytest.raises(TypeError):
        DataSource()


def test_data_source_error_is_a_package_error():
    assert issubclass(DataSourceError, OmniFoldPublicationError)
    assert issubclass(DataSourceError, Exception)


def test_sha256_helper_matches_hashlib(tmp_path):
    payload = b"omnifold" * 5000
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)

    assert (
        DataSource._sha256_of_file(path)
        == hashlib.sha256(payload).hexdigest()
    )
    # chunked streaming gives the same digest with a tiny chunk size
    assert (
        DataSource._sha256_of_file(path, chunk_size=7)
        == hashlib.sha256(payload).hexdigest()
    )


def test_default_health_check_and_repr(tmp_path):
    class MinimalSource(DataSource):
        kind = "minimal"

        def open_events(self, columns=None):
            raise NotImplementedError

        def metadata(self):
            return {}

        def checksum(self):
            return None

    source = MinimalSource()
    assert source.health_check() is True
    assert "minimal" in repr(source)

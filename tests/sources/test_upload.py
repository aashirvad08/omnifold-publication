"""Tests for the user-upload data source."""

from __future__ import annotations

import io
import os
import time

import numpy as np
import pandas as pd
import pytest

from omnifold_publication.sources import (
    DataSourceError,
    UploadDataSource,
    sweep_expired_uploads,
)


@pytest.fixture
def parquet_bytes():
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "pT_ll": rng.uniform(10.0, 200.0, 15),
            "weights_nominal": rng.uniform(0.5, 1.5, 15),
        }
    )
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return frame, buffer.getvalue()


def test_receive_round_trip_and_metadata(tmp_path, parquet_bytes):
    frame, payload = parquet_bytes
    source = UploadDataSource.receive(
        io.BytesIO(payload), tmp_path / "uploads", "events.parquet"
    )

    assert source.kind == "upload"
    assert source.health_check()
    pd.testing.assert_frame_equal(source.open_events(), frame)
    selected = source.open_events(columns=["pT_ll"])
    assert list(selected.columns) == ["pT_ll"]

    info = source.metadata()
    assert info["kind"] == "upload"
    assert "received_at" in info
    assert len(source.checksum()) == 64


def test_size_limit_enforced_and_partial_removed(tmp_path, parquet_bytes):
    _, payload = parquet_bytes
    upload_dir = tmp_path / "uploads"
    with pytest.raises(DataSourceError, match="size limit"):
        UploadDataSource.receive(
            io.BytesIO(payload),
            upload_dir,
            "events.parquet",
            max_bytes=100,
        )
    assert not (upload_dir / "events.parquet").exists()


def test_magic_bytes_reject_disguised_fake(tmp_path, parquet_bytes):
    upload_dir = tmp_path / "uploads"
    fake = b"#!/bin/sh\necho not a parquet file\n"
    with pytest.raises(DataSourceError, match="magic bytes"):
        UploadDataSource.receive(
            io.BytesIO(fake), upload_dir, "disguised.parquet"
        )
    assert not (upload_dir / "disguised.parquet").exists()

    # a real HDF5 payload with the right suffix is accepted
    frame = pd.DataFrame({"pT_ll": [1.0, 2.0], "weights_nominal": [1.0, 1.0]})
    hdf_path = tmp_path / "staging.h5"
    frame.to_hdf(hdf_path, key="df", mode="w")
    source = UploadDataSource.receive(
        io.BytesIO(hdf_path.read_bytes()), upload_dir, "events.h5"
    )
    pd.testing.assert_frame_equal(source.open_events(), frame)

    # ...but the same HDF5 bytes disguised as parquet are rejected
    with pytest.raises(DataSourceError, match="magic bytes"):
        UploadDataSource.receive(
            io.BytesIO(hdf_path.read_bytes()), upload_dir, "disguised2.parquet"
        )


def test_unsupported_suffix_rejected_before_writing(tmp_path):
    with pytest.raises(DataSourceError, match="Unsupported upload suffix"):
        UploadDataSource.receive(
            io.BytesIO(b"anything"), tmp_path / "uploads", "notes.txt"
        )
    assert not (tmp_path / "uploads" / "notes.txt").exists()


def test_client_path_is_reduced_to_basename(tmp_path, parquet_bytes):
    _, payload = parquet_bytes
    source = UploadDataSource.receive(
        io.BytesIO(payload),
        tmp_path / "uploads",
        "../../escape/events.parquet",
    )
    assert source.path.parent == tmp_path / "uploads"
    assert source.path.name == "events.parquet"
    source.cleanup()


def test_explicit_cleanup_and_context_manager(tmp_path, parquet_bytes):
    _, payload = parquet_bytes
    source = UploadDataSource.receive(
        io.BytesIO(payload), tmp_path / "uploads", "events.parquet"
    )
    assert source.path.exists()
    source.cleanup()
    assert not source.path.exists()
    assert source.health_check() is False

    with UploadDataSource.receive(
        io.BytesIO(payload), tmp_path / "uploads", "scoped.parquet"
    ) as scoped:
        assert scoped.path.exists()
    assert not scoped.path.exists()


def test_ttl_sweep_removes_only_expired(tmp_path, parquet_bytes):
    _, payload = parquet_bytes
    upload_dir = tmp_path / "uploads"
    old = UploadDataSource.receive(
        io.BytesIO(payload), upload_dir, "old.parquet"
    )
    fresh = UploadDataSource.receive(
        io.BytesIO(payload), upload_dir, "fresh.parquet"
    )
    expired_mtime = time.time() - 7200
    os.utime(old.path, (expired_mtime, expired_mtime))

    removed = sweep_expired_uploads(upload_dir, ttl_seconds=3600)
    assert removed == [old.path]
    assert not old.path.exists()
    assert fresh.path.exists()

    # sweeping a nonexistent directory is a no-op
    assert sweep_expired_uploads(tmp_path / "nowhere") == []

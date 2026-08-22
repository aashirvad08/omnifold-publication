"""Tests for the local-disk data source."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from omnifold_publication.sources import (
    DataSourceError,
    LocalDataSource,
    get_source,
)


@pytest.fixture
def events_frame():
    rng = np.random.default_rng(42)
    n = 20
    return pd.DataFrame(
        {
            "pT_ll": rng.uniform(10.0, 200.0, n),
            "pT_l1": rng.uniform(25.0, 150.0, n),
            "weight_mc": rng.uniform(0.5, 1.5, n),
            "weights_nominal": rng.uniform(0.5, 1.5, n),
        }
    )


@pytest.fixture
def parquet_file(tmp_path, events_frame):
    path = tmp_path / "events.parquet"
    events_frame.to_parquet(path, index=False)
    return path


@pytest.fixture
def hdf5_file(tmp_path, events_frame):
    path = tmp_path / "events.h5"
    events_frame.to_hdf(path, key="df", mode="w")
    return path


def test_parquet_round_trip_and_column_selection(parquet_file, events_frame):
    source = LocalDataSource(parquet_file)

    full = source.open_events()
    pd.testing.assert_frame_equal(full, events_frame)

    selected = source.open_events(columns=["pT_ll", "weights_nominal"])
    assert list(selected.columns) == ["pT_ll", "weights_nominal"]
    np.testing.assert_allclose(
        selected["pT_ll"], events_frame["pT_ll"]
    )


def test_hdf5_round_trip_and_column_selection(hdf5_file, events_frame):
    source = LocalDataSource(hdf5_file)

    full = source.open_events()
    pd.testing.assert_frame_equal(full, events_frame)

    # fixed-format HDF5 does not support column projection natively;
    # the source must fall back to full-load + slice transparently
    selected = source.open_events(columns=["pT_l1"])
    assert list(selected.columns) == ["pT_l1"]
    np.testing.assert_allclose(selected["pT_l1"], events_frame["pT_l1"])


def test_checksum_is_stable_and_none_when_missing(parquet_file, tmp_path):
    source = LocalDataSource(parquet_file)
    first = source.checksum()
    assert first == source.checksum()
    assert len(first) == 64

    missing = LocalDataSource(tmp_path / "nope.parquet")
    assert missing.checksum() is None


def test_missing_file_and_unsupported_suffix_raise(tmp_path):
    with pytest.raises(DataSourceError, match="not found"):
        LocalDataSource(tmp_path / "nope.parquet").open_events()

    bad = tmp_path / "events.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(DataSourceError, match="Unsupported data file suffix"):
        LocalDataSource(bad).open_events()


def test_health_check_reflects_real_state(parquet_file, tmp_path):
    assert LocalDataSource(parquet_file).health_check() is True
    assert LocalDataSource(tmp_path / "nope.parquet").health_check() is False


def test_sidecar_metadata_is_preferred(parquet_file):
    source = LocalDataSource(parquet_file)
    minimal = source.metadata()
    assert minimal["kind"] == "local"
    assert minimal["size_bytes"] > 0

    sidecar = {"dataset": "zjets", "release": "2024"}
    sidecar_path = parquet_file.parent / f"{parquet_file.name}.metadata.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    assert LocalDataSource(parquet_file).metadata() == sidecar


def test_corrupt_file_raises_data_source_error(tmp_path):
    fake = tmp_path / "fake.parquet"
    fake.write_bytes(b"this is not parquet")
    with pytest.raises(DataSourceError, match="Failed to read"):
        LocalDataSource(fake).open_events()


def test_factory_dispatches_local(parquet_file):
    prefixed = get_source(f"local:{parquet_file}")
    bare = get_source(str(parquet_file))
    assert prefixed.kind == bare.kind == "local"
    pd.testing.assert_frame_equal(
        prefixed.open_events(), bare.open_events()
    )

    with pytest.raises(DataSourceError, match="Cannot resolve"):
        get_source("nonsense:whatever")

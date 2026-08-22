"""Tests for the Zenodo data source. The HTTP layer is fully mocked."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest

from omnifold_publication.sources import (
    DataSourceError,
    ZenodoDataSource,
    get_source,
)
from omnifold_publication.sources import zenodo as zenodo_module


RECORD_ID = "11507450"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
CONTENT_URL = f"{API_URL}/files/events.h5/content"
PARQUET_CONTENT_URL = f"{API_URL}/files/events.parquet/content"


def _record_json(key="events.h5", content_url=CONTENT_URL, extra_files=()):
    files = [
        {
            "key": key,
            "size": 1234,
            "checksum": "md5:9b838510f41e1d5a8b26d7f3c5b2ca42",
            "links": {"self": content_url},
        }
    ]
    for name in extra_files:
        files.append(
            {"key": name, "size": 10, "checksum": "md5:0",
             "links": {"self": f"{API_URL}/files/{name}/content"}}
        )
    return {
        "doi": "10.5281/zenodo.11507450",
        "title": "ATLAS OmniFold 24-Dimensional Z+jets Open Data",
        "metadata": {"title": "ATLAS OmniFold Z+jets"},
        "files": files,
    }


@pytest.fixture
def events_frame():
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        {
            "pT_ll": rng.uniform(10.0, 200.0, 12),
            "weights_nominal": rng.uniform(0.5, 1.5, 12),
        }
    )


class _FakeResponse:
    def __init__(self, data: bytes):
        self._buffer = io.BytesIO(data)

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read() if size is None or size < 0 else self._buffer.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeUrlopen:
    """Routes URLs to byte payloads; can simulate leading transient failures."""

    def __init__(self, routes, fail_times=0, fail_exc=None):
        self.routes = routes
        self.fail_times = fail_times
        self.fail_exc = fail_exc or urllib.error.URLError("transient reset")
        self.calls: list[str] = []

    def __call__(self, request, timeout=None):
        url = getattr(request, "full_url", request)
        self.calls.append(url)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise self.fail_exc
        # exact match first: the API URL is a substring of the content URL,
        # so substring routing alone would misroute downloads
        if url in self.routes:
            return _FakeResponse(self.routes[url])
        for fragment, data in self.routes.items():
            if fragment in url:
                return _FakeResponse(data)
        raise urllib.error.HTTPError(url, 404, "not found", {}, None)

    def count(self, fragment: str) -> int:
        return sum(1 for url in self.calls if fragment in url)


@pytest.fixture
def no_backoff(monkeypatch):
    """Replace the backoff sleep with a call recorder (no real sleeping)."""

    attempts: list[int] = []
    monkeypatch.setattr(zenodo_module, "_backoff_sleep", attempts.append)
    return attempts


def _install(monkeypatch, fake):
    monkeypatch.setattr(urllib.request, "urlopen", fake)


def test_constructor_requires_record_or_url():
    with pytest.raises(DataSourceError, match="record_id or file_url"):
        ZenodoDataSource()


def test_metadata_extracts_doi_title_and_files(monkeypatch, no_backoff):
    fake = FakeUrlopen({API_URL: json.dumps(_record_json()).encode()})
    _install(monkeypatch, fake)

    source = ZenodoDataSource(RECORD_ID)
    meta = source.metadata()
    assert meta["doi"] == "10.5281/zenodo.11507450"
    assert "Z+jets" in meta["title"]
    assert meta["files"][0]["key"] == "events.h5"
    assert meta["files"][0]["checksum"].startswith("md5:")


def test_checksum_is_provider_labeled_and_sha256_is_local(
    monkeypatch, no_backoff
):
    fake = FakeUrlopen({API_URL: json.dumps(_record_json()).encode()})
    _install(monkeypatch, fake)

    source = ZenodoDataSource(RECORD_ID)
    # provider checksum is returned verbatim, algorithm-labeled
    assert source.checksum() == "md5:9b838510f41e1d5a8b26d7f3c5b2ca42"
    # nothing downloaded yet, so no local SHA-256
    assert source.sha256_of_cache() is None


def test_retry_succeeds_after_transient_failures(monkeypatch, no_backoff):
    fake = FakeUrlopen(
        {API_URL: json.dumps(_record_json()).encode()}, fail_times=2
    )
    _install(monkeypatch, fake)

    source = ZenodoDataSource(RECORD_ID, max_retries=3)
    assert source.metadata()["doi"].endswith("11507450")
    # two transient failures -> two backoff sleeps before the success
    assert no_backoff == [0, 1]


def test_retry_exhausts_and_raises(monkeypatch, no_backoff):
    fake = FakeUrlopen(
        {API_URL: json.dumps(_record_json()).encode()}, fail_times=99
    )
    _install(monkeypatch, fake)

    source = ZenodoDataSource(RECORD_ID, max_retries=3)
    with pytest.raises(DataSourceError, match="failed after 3 attempts"):
        source.metadata()
    # backoff between attempts: max_retries - 1
    assert no_backoff == [0, 1]


def test_open_events_downloads_and_caches_hdf5(
    tmp_path, monkeypatch, no_backoff, events_frame
):
    hdf_path = tmp_path / "src.h5"
    events_frame.to_hdf(hdf_path, key="df", mode="w")
    fake = FakeUrlopen(
        {
            API_URL: json.dumps(_record_json()).encode(),
            CONTENT_URL: hdf_path.read_bytes(),
        }
    )
    _install(monkeypatch, fake)

    source = ZenodoDataSource(RECORD_ID, cache_dir=tmp_path / "cache")
    loaded = source.open_events()
    pd.testing.assert_frame_equal(loaded, events_frame)

    # the file is now cached and its SHA-256 is available locally
    assert (tmp_path / "cache" / "events.h5").is_file()
    assert len(source.sha256_of_cache()) == 64

    # a second read must not hit the network again
    downloads_before = fake.count("/content")
    again = source.open_events(columns=["pT_ll"])
    assert list(again.columns) == ["pT_ll"]
    assert fake.count("/content") == downloads_before


def test_parquet_range_read_used_when_available(
    tmp_path, monkeypatch, no_backoff, events_frame
):
    # the range-read path is an optional optimization gated on fsspec
    fsspec = pytest.importorskip("fsspec")
    parquet_path = tmp_path / "src.parquet"
    events_frame.to_parquet(parquet_path, index=False)
    fake = FakeUrlopen(
        {API_URL: json.dumps(
            _record_json("events.parquet", PARQUET_CONTENT_URL)
        ).encode()}
    )
    _install(monkeypatch, fake)

    monkeypatch.setattr(
        fsspec, "open", lambda url, mode="rb": open(parquet_path, mode)
    )
    source = ZenodoDataSource(
        RECORD_ID, filename="events.parquet", cache_dir=tmp_path / "cache"
    )
    selected = source.open_events(columns=["pT_ll"])
    assert list(selected.columns) == ["pT_ll"]
    # range read means no file download and nothing written to the cache
    assert fake.count("/content") == 0
    assert not (tmp_path / "cache").exists()


def test_parquet_range_read_falls_back_to_download(
    tmp_path, monkeypatch, no_backoff, events_frame
):
    fsspec = pytest.importorskip("fsspec")
    parquet_path = tmp_path / "src.parquet"
    events_frame.to_parquet(parquet_path, index=False)
    fake = FakeUrlopen(
        {
            API_URL: json.dumps(
                _record_json("events.parquet", PARQUET_CONTENT_URL)
            ).encode(),
            PARQUET_CONTENT_URL: parquet_path.read_bytes(),
        }
    )
    _install(monkeypatch, fake)

    def _boom(url, mode="rb"):
        raise OSError("range requests unsupported")

    monkeypatch.setattr(fsspec, "open", _boom)
    source = ZenodoDataSource(
        RECORD_ID, filename="events.parquet", cache_dir=tmp_path / "cache"
    )
    loaded = source.open_events()
    pd.testing.assert_frame_equal(loaded, events_frame)
    # fell back to a real download
    assert fake.count("/content") == 1
    assert (tmp_path / "cache" / "events.parquet").is_file()


def test_multiple_files_requires_filename(monkeypatch, no_backoff):
    fake = FakeUrlopen(
        {API_URL: json.dumps(
            _record_json(extra_files=("other.h5",))
        ).encode()}
    )
    _install(monkeypatch, fake)

    ambiguous = ZenodoDataSource(RECORD_ID)
    with pytest.raises(DataSourceError, match="multiple files"):
        ambiguous.checksum()

    chosen = ZenodoDataSource(RECORD_ID, filename="other.h5")
    assert chosen.checksum() == "md5:0"


def test_health_check_reflects_reachability(monkeypatch, no_backoff):
    ok = FakeUrlopen({API_URL: json.dumps(_record_json()).encode()})
    _install(monkeypatch, ok)
    assert ZenodoDataSource(RECORD_ID).health_check() is True

    down = FakeUrlopen({}, fail_times=99)
    _install(monkeypatch, down)
    assert ZenodoDataSource(RECORD_ID).health_check() is False


def test_direct_file_url_has_no_record_metadata(
    tmp_path, monkeypatch, no_backoff, events_frame
):
    hdf_path = tmp_path / "src.h5"
    events_frame.to_hdf(hdf_path, key="df", mode="w")
    url = "https://example.org/data/events.h5"
    fake = FakeUrlopen({url: hdf_path.read_bytes()})
    _install(monkeypatch, fake)

    source = ZenodoDataSource(file_url=url, cache_dir=tmp_path / "cache")
    # no record id -> provider metadata/checksum are unavailable, not faked
    assert source.checksum() is None
    with pytest.raises(DataSourceError, match="Record metadata is unavailable"):
        source.metadata()
    pd.testing.assert_frame_equal(source.open_events(), events_frame)


def test_factory_dispatches_zenodo():
    source = get_source(f"zenodo:{RECORD_ID}")
    assert isinstance(source, ZenodoDataSource)
    assert source.kind == "zenodo"
    assert source.record_id == RECORD_ID

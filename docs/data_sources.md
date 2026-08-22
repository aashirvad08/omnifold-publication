# Data-Source Layer

Event data can live anywhere — local disk, NERSC GPFS, Zenodo, a user
upload, and (future) S3/URL — but everything downstream reads it through
one interface, `DataSource`. Swapping backends requires no change to
calling code.

```python
from omnifold_publication.sources import get_source

src = get_source("local:data/multifold.h5")     # local / GPFS
src = get_source("zenodo:11507450")              # Zenodo record
events = src.open_events(columns=["pT_ll", "weights_nominal"])
```

## The interface

Every backend implements three methods:

- `open_events(columns=None)` → DataFrame, reading only the requested
  columns where the format allows; large files are never buffered whole.
- `metadata()` → a JSON-compatible description of the source.
- `checksum()` → SHA-256 of the underlying file when obtainable, else
  None (see the Zenodo note below for the one deliberate exception).

Plus `health_check()` and a shared streaming `_sha256_of_file` helper
(8 MiB chunks). `DataSourceError` (a subclass of the package's
`OmniFoldPublicationError`) reports every failure.

## The factory

`get_source(spec, **kwargs)` dispatches on a `<prefix>:` head:

| Spec | Backend | Notes |
|---|---|---|
| `local:<path>` or a bare existing path | `LocalDataSource` | Parquet + HDF5 |
| `upload:<stored path>` | `UploadDataSource` | private, auto-expiring |
| `zenodo:<record_id>` | `ZenodoDataSource` | streams from a record |
| `s3://bucket/key` | `S3DataSource` | stub (NotImplementedError) |
| `url:https://...` | `UrlDataSource` | stub (NotImplementedError) |

`s3://` keeps its scheme (the prefix is part of the URI); the others pass
everything after the first colon.

## Backends

**LocalDataSource** — local disk / GPFS. Column-selective Parquet reads;
for fixed-format HDF5 (which cannot be column-projected) it transparently
falls back to a full load and column slice. A sidecar
`<file>.metadata.json`, if present, is returned by `metadata()`.

**ZenodoDataSource** — streams a file from a public Zenodo record. The
guaranteed path downloads the file to `cache_dir` in chunks and reads it
from disk (this is the path the real ATLAS release uses — that record
ships HDF5). When the target is Parquet *and* `fsspec` is installed, only
the requested columns are fetched via HTTP range requests; any failure
falls back to the chunked download. Network calls retry with exponential
backoff and raise `DataSourceError` once `max_retries` is exhausted.

*Checksum note:* Zenodo publishes an **MD5** digest, so
`ZenodoDataSource.checksum()` returns it verbatim and algorithm-labeled
(`"md5:..."`) rather than misrepresenting it as SHA-256;
`sha256_of_cache()` computes a real SHA-256 once the file is downloaded.

**UploadDataSource** — a user-supplied file staged on disk. `receive()`
streams the incoming bytes in chunks, enforces a size limit *during*
streaming, and validates the real file format from its magic bytes
(`PAR1` for Parquet, `\x89HDF...` for HDF5) — a fake with the wrong
signature is rejected even if its extension looks right, and
client-supplied paths are reduced to basenames. Uploads are private and
must not outlive the job: dispose of them with `cleanup()` (also via
`with`), or sweep abandoned files with `sweep_expired_uploads(dir, ttl)`.

**S3DataSource / UrlDataSource** — stubs with realistic constructors that
raise `NotImplementedError` with a clear message, so the factory and
calling code can be wired up ahead of the implementation.

## Optional dependencies

The layer needs nothing beyond the base requirements. The Parquet
range-read optimization uses `fsspec` (and `pyarrow`) if present and is
skipped otherwise; native S3 will use `s3fs`. Uploading via
`scripts/publish_to_zenodo.py` uses `requests`, imported only on the
upload path so `--dry-run` needs no network dependency.

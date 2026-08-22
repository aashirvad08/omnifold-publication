#!/usr/bin/env python3
"""Convert raw OmniFold input files into publication packages and upload
them to Zenodo.

Each input file is turned into a full publication package with this
package's own :func:`omnifold_publication.write_package` (replicas +
systematic weight families included) — no packaging logic is
reimplemented here. The script then assembles a README, a citation block,
a license, and Zenodo deposition metadata, and (unless ``--dry-run``)
creates a Zenodo deposition, uploads every package as a zip, and
publishes it, printing the resulting DOI.

Auth: the Zenodo personal access token is read from the ``ZENODO_TOKEN``
environment variable. ``--sandbox`` targets https://sandbox.zenodo.org.

``--dry-run`` performs the conversion and writes all artifacts locally
without importing any network library or making any request, so it works
in a minimal environment.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omnifold_publication import write_package  # noqa: E402


ZENODO_LIVE = "https://zenodo.org"
ZENODO_SANDBOX = "https://sandbox.zenodo.org"
TOKEN_ENV = "ZENODO_TOKEN"
LICENSE_ID = "cc-by-4.0"
LICENSE_TEXT = (
    "This dataset is released under the Creative Commons Attribution 4.0 "
    "International License (CC-BY-4.0). https://creativecommons.org/licenses/by/4.0/\n"
)


def build_packages(
    inputs: list[Path], output_dir: Path, event_count: int | None
) -> list[Path]:
    """Convert each raw input into a full publication package."""

    package_dirs: list[Path] = []
    for input_path in inputs:
        package_dir = output_dir / f"{input_path.stem}_package"
        write_package(
            input_path=input_path,
            output_dir=package_dir,
            event_count=event_count,
            include_all_replicas=True,  # full-featured: replicas + families
        )
        package_dirs.append(package_dir)
    return package_dirs


def build_readme(title: str, package_dirs: list[Path]) -> str:
    lines = [
        f"# {title}",
        "",
        "OmniFold publication packages generated with `omnifold_publication`.",
        "Each package holds an event table (`events.parquet`), its",
        "`metadata.yaml`, and a SHA-256 checksum, and is readable with:",
        "",
        "```python",
        "from omnifold_publication import load_package",
        'pkg = load_package("<package-directory>")',
        'w = pkg.get_weights("final")',
        "```",
        "",
        "## Packages",
        "",
    ]
    for package_dir in package_dirs:
        lines.append(f"- `{package_dir.name}/`")
    lines.append("")
    return "\n".join(lines)


def build_citation(title: str) -> str:
    return (
        "To cite this dataset, please cite the ATLAS Z+jets OmniFold\n"
        "measurement (arXiv:2405.20041) and this Zenodo deposition.\n\n"
        "@dataset{omnifold_publication,\n"
        f"  title  = {{{title}}},\n"
        "  note   = {Generated with omnifold_publication},\n"
        "}\n"
    )


def build_zenodo_metadata(
    title: str, description: str, creators: list[str]
) -> dict:
    """The deposition metadata dict Zenodo's API expects under 'metadata'."""

    return {
        "metadata": {
            "title": title,
            "upload_type": "dataset",
            "description": description,
            "creators": [{"name": name} for name in creators],
            "license": LICENSE_ID,
            "keywords": ["OmniFold", "ATLAS", "Z+jets", "unfolding"],
        }
    }


def write_artifacts(
    output_dir: Path,
    title: str,
    description: str,
    creators: list[str],
    package_dirs: list[Path],
) -> dict:
    """Write README, LICENSE, citation, and metadata; return the metadata."""

    import json

    (output_dir / "README.md").write_text(
        build_readme(title, package_dirs), encoding="utf-8"
    )
    (output_dir / "LICENSE.txt").write_text(LICENSE_TEXT, encoding="utf-8")
    (output_dir / "CITATION.txt").write_text(
        build_citation(title), encoding="utf-8"
    )
    metadata = build_zenodo_metadata(title, description, creators)
    (output_dir / "zenodo_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def zip_packages(package_dirs: list[Path], output_dir: Path) -> list[Path]:
    """Zip each package directory for upload; return the archive paths."""

    archives: list[Path] = []
    for package_dir in package_dirs:
        archive = shutil.make_archive(
            str(output_dir / package_dir.name), "zip", root_dir=package_dir
        )
        archives.append(Path(archive))
    return archives


def upload_to_zenodo(
    metadata: dict,
    archives: list[Path],
    base_url: str,
    token: str,
) -> str:
    """Create a deposition, upload archives, publish; return the DOI.

    Uses ``requests`` (imported lazily so --dry-run needs no network
    dependency). Follows the classic Zenodo deposition flow: create,
    upload to the deposition bucket, set metadata, publish.
    """

    try:
        import requests
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Uploading requires the 'requests' package "
            "(`pip install requests`). Use --dry-run to skip uploads."
        ) from exc

    auth = {"params": {"access_token": token}}
    deposition = requests.post(
        f"{base_url}/api/deposit/depositions", json={}, **auth, timeout=60
    )
    deposition.raise_for_status()
    deposition_json = deposition.json()
    bucket_url = deposition_json["links"]["bucket"]
    deposition_id = deposition_json["id"]

    for archive in archives:
        with archive.open("rb") as stream:
            put = requests.put(
                f"{bucket_url}/{archive.name}",
                data=stream,  # streamed, not buffered
                **auth,
                timeout=None,
            )
            put.raise_for_status()

    meta_response = requests.put(
        f"{base_url}/api/deposit/depositions/{deposition_id}",
        json=metadata,
        **auth,
        timeout=60,
    )
    meta_response.raise_for_status()

    publish = requests.post(
        f"{base_url}/api/deposit/depositions/{deposition_id}/actions/publish",
        **auth,
        timeout=60,
    )
    publish.raise_for_status()
    return publish.json()["doi"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="raw input file(s)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("zenodo_submission"),
        help="where packages and artifacts are written",
    )
    parser.add_argument(
        "--title",
        default="ATLAS Z+jets OmniFold publication packages",
        help="Zenodo deposition title",
    )
    parser.add_argument(
        "--description",
        default="OmniFold event-level publication packages with weight "
        "families and systematic variations.",
    )
    parser.add_argument(
        "--creator",
        action="append",
        dest="creators",
        default=None,
        help="creator name (repeatable); defaults to a placeholder",
    )
    parser.add_argument(
        "--event-count",
        type=int,
        default=None,
        help="events per package (default: all events)",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="target sandbox.zenodo.org instead of the live site",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="convert and write artifacts locally; no network calls",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    missing = [str(path) for path in args.inputs if not path.is_file()]
    if missing:
        raise SystemExit(f"Input file(s) not found: {', '.join(missing)}")

    creators = args.creators or ["OmniFold Publication Contributors"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {len(args.inputs)} input file(s) into packages ...")
    package_dirs = build_packages(
        args.inputs, args.output_dir, args.event_count
    )
    metadata = write_artifacts(
        args.output_dir,
        args.title,
        args.description,
        creators,
        package_dirs,
    )
    for package_dir in package_dirs:
        print(f"  package: {package_dir}")
    print(f"  artifacts: README.md, LICENSE.txt, CITATION.txt, "
          f"zenodo_metadata.json in {args.output_dir}")

    if args.dry_run:
        print("\n--dry-run: conversion complete, no upload performed.")
        return 0

    token = os.environ.get(TOKEN_ENV)
    if not token:
        raise SystemExit(
            f"Set {TOKEN_ENV} to a Zenodo access token to upload "
            "(or use --dry-run)."
        )
    base_url = ZENODO_SANDBOX if args.sandbox else ZENODO_LIVE
    print(f"\nUploading to {base_url} ...")
    archives = zip_packages(package_dirs, args.output_dir)
    doi = upload_to_zenodo(metadata, archives, base_url, token)
    print(f"Published. DOI: {doi}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

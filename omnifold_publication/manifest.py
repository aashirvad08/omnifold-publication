"""Read and write multi-sample OmniFold analysis manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .exceptions import ManifestNotFoundError


MANIFEST_FILENAME = "manifest.yaml"
MANIFEST_FORMAT_VERSION = "0.1"


def write_manifest(
    output_dir: str | Path,
    nominal_path: str | Path,
    variations: dict[str, dict],
    analysis_name: str = "unnamed",
    target_path: str | Path | None = None,
) -> Path:
    """Write a manifest.yaml linking multiple OmniFold packages.

    ``target_path`` declares a truth/target companion sample (the known
    distribution pseudo-data was reweighted toward, cf. target.h5 in
    2_pseudo_results.ipynb) under the role "target"; it is not listed as a
    variation.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples: dict[str, dict[str, Any]] = {
        "nominal": {
            "path": Path(nominal_path).as_posix(),
            "role": "nominal",
        }
    }
    if target_path is not None:
        samples["target"] = {
            "path": Path(target_path).as_posix(),
            "role": "target",
        }
    for name, variation in variations.items():
        samples[name] = {
            "path": Path(variation["path"]).as_posix(),
            "role": "variation",
            "type": variation.get("type"),
            "combination": variation.get("combination"),
        }

    manifest = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "analysis": analysis_name,
        "samples": samples,
    }

    manifest_path = output_dir / MANIFEST_FILENAME
    with manifest_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False)
    return manifest_path


def load_manifest(manifest_dir: str | Path) -> dict:
    """Load manifest.yaml from a directory."""

    manifest_path = Path(manifest_dir) / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise ManifestNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream) or {}
    if not isinstance(manifest, dict):
        raise ManifestNotFoundError(f"Manifest must be a mapping: {manifest_path}")
    return manifest


def list_manifest_variations(manifest: dict) -> list[str]:
    """Return list of variation names declared in a manifest."""

    samples = manifest.get("samples", {})
    if not isinstance(samples, dict):
        return []
    return sorted(
        name
        for name, sample in samples.items()
        if name != "nominal"
        and isinstance(sample, dict)
        and sample.get("role") == "variation"
    )

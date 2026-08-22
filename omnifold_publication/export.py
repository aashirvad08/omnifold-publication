"""JSON export helpers for OmniFold publication results."""

from __future__ import annotations

import json
from pathlib import Path

from .analysis import load_analysis
from .reader import load_package


def export_histogram_json(
    package_dir: str | Path,
    observable: str,
    output_path: str | Path,
    bins: list[float] | None = None,
    variation: str = "nominal",
) -> None:
    """Compute a package histogram and write it as JSON."""

    package = load_package(package_dir)
    result = package.histogram(
        observable=observable,
        variation=variation,
        bins=bins,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(result.to_dict(), stream, indent=2)


def export_comparison_json(
    manifest_dir: str | Path,
    observable: str,
    output_path: str | Path,
    bins: list[float] | None = None,
) -> None:
    """Compute a multi-sample comparison and write it as JSON."""

    analysis = load_analysis(manifest_dir)
    comparison = analysis.compare(observable=observable, bins=bins)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    comparison.export_json(destination)


__all__ = ["export_comparison_json", "export_histogram_json"]

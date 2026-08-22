"""Tests for JSON result exports."""

from __future__ import annotations

import json

from omnifold_publication import (
    export_comparison_json,
    export_histogram_json,
    write_manifest,
    write_package,
)
from omnifold_publication.cli import main


def _write_analysis(tmp_path, source_hdf):
    nominal = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "nominal",
        event_count=6,
    )
    variation = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "variation",
        event_count=6,
    )
    manifest_dir = tmp_path / "analysis"
    write_manifest(
        output_dir=manifest_dir,
        nominal_path=nominal,
        variations={
            "generator_choice": {
                "path": variation,
                "type": "alternative_generator",
                "combination": "envelope",
            }
        },
        analysis_name="export-test",
    )
    return nominal, manifest_dir


def test_export_histogram_json_writes_expected_keys(tmp_path, source_hdf):
    package_dir, _ = _write_analysis(tmp_path, source_hdf)
    output_path = tmp_path / "exports" / "histogram.json"

    export_histogram_json(
        package_dir,
        "pT_ll",
        output_path,
        bins=[0.0, 40.0, 80.0, 130.0],
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert {"hist", "edges", "centers", "stat_uncertainty"} <= payload.keys()


def test_export_comparison_json_writes_nominal_and_variations(tmp_path, source_hdf):
    _, manifest_dir = _write_analysis(tmp_path, source_hdf)
    output_path = tmp_path / "comparison.json"

    export_comparison_json(
        manifest_dir,
        "pT_ll",
        output_path,
        bins=[0.0, 40.0, 80.0, 130.0],
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert "nominal" in payload["histograms"]
    assert "generator_choice" in payload["histograms"]


def test_exported_json_round_trips_through_json_loads(tmp_path, source_hdf):
    package_dir, _ = _write_analysis(tmp_path, source_hdf)
    output_path = tmp_path / "histogram.json"

    export_histogram_json(package_dir, "pT_ll", output_path, bins=[0.0, 130.0])

    encoded = output_path.read_text(encoding="utf-8")
    assert json.loads(json.dumps(json.loads(encoded))) == json.loads(encoded)


def test_cli_export_commands_write_json(tmp_path, source_hdf):
    package_dir, manifest_dir = _write_analysis(tmp_path, source_hdf)
    histogram_path = tmp_path / "cli-histogram.json"
    comparison_path = tmp_path / "cli-comparison.json"
    metadata_path = tmp_path / "cli-metadata.json"

    assert main(
        [
            "export-histogram",
            str(package_dir),
            "--observable",
            "pT_ll",
            "--output",
            str(histogram_path),
            "--bins",
            "0",
            "40",
            "80",
            "130",
        ]
    ) == 0
    assert main(
        [
            "export-comparison",
            str(manifest_dir),
            "--observable",
            "pT_ll",
            "--output",
            str(comparison_path),
            "--bins",
            "0",
            "40",
            "80",
            "130",
        ]
    ) == 0
    assert main(
        [
            "export-metadata",
            str(manifest_dir),
            "--output",
            str(metadata_path),
        ]
    ) == 0

    assert "hist" in json.loads(histogram_path.read_text(encoding="utf-8"))
    assert "histograms" in json.loads(comparison_path.read_text(encoding="utf-8"))
    assert "analysis" in json.loads(metadata_path.read_text(encoding="utf-8"))

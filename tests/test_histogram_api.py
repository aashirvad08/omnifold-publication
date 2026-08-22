"""Tests for package and analysis histogram APIs."""

from __future__ import annotations

import numpy as np
import yaml

from omnifold_publication import (
    HistogramResult,
    load_analysis,
    load_package,
    write_manifest,
    write_package,
)


def _write_analysis(tmp_path, source_hdf):
    nominal = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "nominal",
        event_count=6,
        include_all_replicas=True,
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
        analysis_name="histogram-test",
    )
    return manifest_dir


def test_package_histogram_returns_structured_result(tmp_path, source_hdf):
    package_dir = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "package",
        event_count=6,
    )
    result = load_package(package_dir).histogram("pT_ll", bins=3)

    assert isinstance(result, HistogramResult)
    assert len(result.hist) == 3
    assert len(result.edges) == 4
    assert len(result.centers) == 3
    assert len(result.stat_uncertainty) == 3


def test_package_histogram_uses_observable_default_bins(tmp_path, source_hdf):
    package_dir = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "package",
        event_count=6,
    )
    metadata_path = package_dir / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    metadata["observables"][0]["suggested_bins"] = [0.0, 40.0, 80.0, 130.0]
    metadata_path.write_text(
        yaml.safe_dump(metadata, sort_keys=False),
        encoding="utf-8",
    )
    package = load_package(package_dir)
    expected_bins = package.observable_bins("pT_ll")

    result = package.histogram("pT_ll")

    assert expected_bins is not None
    np.testing.assert_allclose(result.edges, expected_bins)


def test_analysis_histogram_fills_systematic_uncertainty(tmp_path, source_hdf):
    analysis = load_analysis(_write_analysis(tmp_path, source_hdf))

    result = analysis.histogram(
        "pT_ll",
        systematic_variations=["generator_choice"],
        bins=[0.0, 40.0, 80.0, 130.0],
    )

    assert result.sys_uncertainty is not None
    assert result.sys_uncertainty.shape == result.hist.shape


def test_analysis_histogram_fills_replica_uncertainty(tmp_path, source_hdf):
    analysis = load_analysis(_write_analysis(tmp_path, source_hdf))

    result = analysis.histogram("pT_ll", bins=3)

    assert result.replica_uncertainty is not None
    assert result.replica_uncertainty.shape == result.hist.shape


def test_histogram_result_to_dict_omits_none_and_serializes_arrays():
    result = HistogramResult(
        hist=np.array([1.0, 2.0]),
        edges=np.array([0.0, 1.0, 2.0]),
        centers=np.array([0.5, 1.5]),
        stat_uncertainty=np.array([0.1, 0.2]),
    )

    payload = result.to_dict()

    assert payload["hist"] == [1.0, 2.0]
    assert isinstance(payload["edges"], list)
    assert "sys_uncertainty" not in payload
    assert "replica_uncertainty" not in payload

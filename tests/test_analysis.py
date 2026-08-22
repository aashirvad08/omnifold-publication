"""Tests for multi-sample analysis loading."""

import numpy as np
import pytest

from omnifold_publication import load_analysis, write_manifest, write_package
from omnifold_publication.analysis import OmniFoldAnalysis
from omnifold_publication.exceptions import ManifestNotFoundError


def _write_analysis(tmp_path, source_hdf, include_all_replicas=False):
    nominal = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "zjets_nominal",
        event_count=6,
        include_all_replicas=include_all_replicas,
    )
    variation = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "zjets_sherpa",
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
        analysis_name="zjets",
    )
    return manifest_dir


def test_load_analysis_returns_omnifold_analysis(source_hdf, tmp_path):
    manifest_dir = _write_analysis(tmp_path, source_hdf)
    analysis = load_analysis(manifest_dir)
    assert isinstance(analysis, OmniFoldAnalysis)


def test_list_variations_returns_nominal(source_hdf, tmp_path):
    manifest_dir = _write_analysis(tmp_path, source_hdf)
    analysis = load_analysis(manifest_dir)
    assert "nominal" in analysis.list_variations()


def test_get_weights_nominal_returns_numpy_array(source_hdf, tmp_path):
    manifest_dir = _write_analysis(tmp_path, source_hdf)
    analysis = load_analysis(manifest_dir)
    weights = analysis.get_weights("nominal")
    assert isinstance(weights, np.ndarray)


def test_get_weights_nominal_length_matches_event_count(source_hdf, tmp_path):
    manifest_dir = _write_analysis(tmp_path, source_hdf)
    analysis = load_analysis(manifest_dir)
    weights = analysis.get_weights("nominal")
    assert len(weights) == 6


def test_get_weights_cross_package_variation_returns_numpy_array(source_hdf, tmp_path):
    manifest_dir = _write_analysis(tmp_path, source_hdf)
    analysis = load_analysis(manifest_dir)
    weights = analysis.get_weights("generator_choice")
    assert isinstance(weights, np.ndarray)
    assert len(weights) == 6


def test_validate_all_passes_on_valid_multi_package_setup(source_hdf, tmp_path):
    manifest_dir = _write_analysis(tmp_path, source_hdf)
    analysis = load_analysis(manifest_dir)
    analysis.validate_all()


def test_summary_contains_analysis_nominal_and_variations(source_hdf, tmp_path):
    manifest_dir = _write_analysis(tmp_path, source_hdf)
    analysis = load_analysis(manifest_dir)
    summary = analysis.summary()
    assert summary["analysis"] == "zjets"
    assert "nominal" in summary
    assert "variations" in summary
    assert "generator_choice" in summary["variations"]


def test_get_replica_weights_returns_2d_array_when_replicas_present(source_hdf, tmp_path):
    manifest_dir = _write_analysis(tmp_path, source_hdf, include_all_replicas=True)
    analysis = load_analysis(manifest_dir)
    replicas = analysis.get_replica_weights()
    assert replicas.ndim == 2


def test_get_replica_weights_shape_is_replicas_by_events(source_hdf, tmp_path):
    manifest_dir = _write_analysis(tmp_path, source_hdf, include_all_replicas=True)
    analysis = load_analysis(manifest_dir)
    replicas = analysis.get_replica_weights()
    assert replicas.shape == (1, 6)


def test_load_analysis_raises_manifest_not_found(tmp_path):
    with pytest.raises(ManifestNotFoundError):
        load_analysis(tmp_path / "missing")

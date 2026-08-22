"""Tests for multi-sample manifest helpers."""

from pathlib import Path

import pytest

from omnifold_publication.exceptions import ManifestNotFoundError
from omnifold_publication.manifest import (
    list_manifest_variations,
    load_manifest,
    write_manifest,
)


def test_write_manifest_creates_manifest_yaml(tmp_path):
    manifest_path = write_manifest(
        output_dir=tmp_path / "analysis",
        nominal_path=tmp_path / "zjets_nominal",
        variations={},
        analysis_name="zjets",
    )
    assert manifest_path == tmp_path / "analysis" / "manifest.yaml"
    assert manifest_path.exists()


def test_write_manifest_contains_nominal_path(tmp_path):
    nominal_path = tmp_path / "zjets_nominal"
    manifest_path = write_manifest(
        output_dir=tmp_path / "analysis",
        nominal_path=nominal_path,
        variations={},
        analysis_name="zjets",
    )
    manifest = load_manifest(manifest_path.parent)
    assert manifest["samples"]["nominal"]["path"] == nominal_path.as_posix()


def test_write_manifest_contains_all_variation_paths(tmp_path):
    variations = {
        "generator_choice": {
            "path": tmp_path / "zjets_sherpa",
            "type": "alternative_generator",
            "combination": "envelope",
        },
        "non_dy": {
            "path": tmp_path / "zjets_nonDY",
            "type": "alternative_sample",
            "combination": "difference",
        },
    }
    write_manifest(
        output_dir=tmp_path / "analysis",
        nominal_path=tmp_path / "zjets_nominal",
        variations=variations,
        analysis_name="zjets",
    )
    manifest = load_manifest(tmp_path / "analysis")
    assert manifest["samples"]["generator_choice"]["path"] == (
        tmp_path / "zjets_sherpa"
    ).as_posix()
    assert manifest["samples"]["non_dy"]["path"] == (
        tmp_path / "zjets_nonDY"
    ).as_posix()


def test_load_manifest_loads_written_manifest(tmp_path):
    write_manifest(
        output_dir=tmp_path / "analysis",
        nominal_path=tmp_path / "zjets_nominal",
        variations={},
        analysis_name="zjets",
    )
    manifest = load_manifest(tmp_path / "analysis")
    assert manifest["analysis"] == "zjets"


def test_load_manifest_raises_if_missing(tmp_path):
    with pytest.raises(ManifestNotFoundError):
        load_manifest(tmp_path / "missing")


def test_list_manifest_variations_returns_declared_names(tmp_path):
    write_manifest(
        output_dir=tmp_path / "analysis",
        nominal_path=tmp_path / "zjets_nominal",
        variations={
            "generator_choice": {
                "path": tmp_path / "zjets_sherpa",
                "type": "alternative_generator",
                "combination": "envelope",
            }
        },
        analysis_name="zjets",
    )
    manifest = load_manifest(tmp_path / "analysis")
    assert list_manifest_variations(manifest) == ["generator_choice"]


def test_manifest_format_version_is_0_1(tmp_path):
    write_manifest(
        output_dir=tmp_path / "analysis",
        nominal_path=tmp_path / "zjets_nominal",
        variations={},
        analysis_name="zjets",
    )
    manifest = load_manifest(tmp_path / "analysis")
    assert manifest["format_version"] == "0.1"

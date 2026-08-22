"""Tests for the high-level Publication builder."""

import pytest

from omnifold_publication import OmniFoldAnalysis, Publication
from omnifold_publication.exceptions import PackageWriteError


def test_publication_build_returns_analysis(source_hdf, tmp_path):
    pub = Publication(name="zjets_analysis")
    pub.add_dataset(source_hdf, role="nominal", label="Pythia8")
    pub.add_dataset(
        source_hdf,
        role="variation",
        label="Sherpa",
        variation_type="alternative_generator",
    )

    analysis = pub.build(output_dir=tmp_path / "publication", event_count=6)
    assert isinstance(analysis, OmniFoldAnalysis)


def test_publication_build_writes_manifest(source_hdf, tmp_path):
    pub = Publication(name="zjets_analysis")
    pub.add_dataset(source_hdf, role="nominal", label="Pythia8")
    analysis = pub.build(output_dir=tmp_path / "publication", event_count=6)

    assert (tmp_path / "publication" / "manifest.yaml").exists()
    assert analysis.summary()["analysis"] == "zjets_analysis"


def test_publication_inspect_prints_dataset_summary(source_hdf, tmp_path, capsys):
    pub = Publication(name="zjets_analysis")
    pub.add_dataset(source_hdf, role="nominal", label="Pythia8")

    pub.inspect()
    output = capsys.readouterr().out
    assert "Publication: zjets_analysis" in output
    assert "Dataset 1: Pythia8" in output
    assert "Events:" in output
    assert "Weights:" in output


def test_publication_requires_single_nominal(source_hdf, tmp_path):
    pub = Publication(name="zjets_analysis")
    pub.add_dataset(source_hdf, role="variation", label="Sherpa", variation_type="alt")

    with pytest.raises(PackageWriteError, match="exactly one nominal"):
        pub.build(output_dir=tmp_path / "publication", event_count=6)


def test_publication_rejects_invalid_role(source_hdf):
    pub = Publication(name="zjets_analysis")
    with pytest.raises(PackageWriteError, match="role"):
        pub.add_dataset(source_hdf, role="central", label="Pythia8")


def test_publication_build_includes_all_replicas(source_hdf, tmp_path):
    pub = Publication(name="zjets_analysis")
    pub.add_dataset(
        source_hdf,
        role="nominal",
        label="Pythia8",
        include_all_replicas=True,
    )

    analysis = pub.build(output_dir=tmp_path / "publication", event_count=6)
    assert analysis.get_replica_weights().shape == (1, 6)

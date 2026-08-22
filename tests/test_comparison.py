"""Tests for histogram comparisons."""

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")

from omnifold_publication import HistogramComparison, Publication


def _build_analysis(source_hdf, tmp_path, include_all_replicas=True):
    pub = Publication(name="zjets_analysis")
    pub.add_dataset(
        source_hdf,
        role="nominal",
        label="Pythia8",
        include_all_replicas=include_all_replicas,
    )
    pub.add_dataset(
        source_hdf,
        role="variation",
        label="Sherpa",
        variation_type="alternative_generator",
    )
    return pub.build(output_dir=tmp_path / "publication", event_count=6)


def test_analysis_compare_returns_histogram_comparison(source_hdf, tmp_path):
    analysis = _build_analysis(source_hdf, tmp_path)
    comparison = analysis.compare("pT_ll", bins=[0.0, 50.0, 100.0, 150.0])
    assert isinstance(comparison, HistogramComparison)


def test_comparison_contains_nominal_and_variation_histograms(source_hdf, tmp_path):
    analysis = _build_analysis(source_hdf, tmp_path)
    comparison = analysis.compare("pT_ll", bins=[0.0, 50.0, 100.0, 150.0])
    assert "nominal" in comparison.histograms
    assert "sherpa" in comparison.histograms


def test_comparison_plot_returns_figure(source_hdf, tmp_path):
    analysis = _build_analysis(source_hdf, tmp_path)
    comparison = analysis.compare("pT_ll", bins=[0.0, 50.0, 100.0, 150.0])
    fig = comparison.plot()
    assert fig.axes
    fig.clf()


def test_comparison_plot_writes_output_file(source_hdf, tmp_path):
    analysis = _build_analysis(source_hdf, tmp_path)
    comparison = analysis.compare("pT_ll", bins=[0.0, 50.0, 100.0, 150.0])
    output = tmp_path / "comparison.png"
    fig = comparison.plot(output_path=output)
    assert output.exists()
    fig.clf()


def test_comparison_print_table_outputs_bins(source_hdf, tmp_path, capsys):
    analysis = _build_analysis(source_hdf, tmp_path)
    comparison = analysis.compare("pT_ll", bins=[0.0, 50.0, 100.0, 150.0])
    comparison.print_table()
    output = capsys.readouterr().out
    assert "low | high" in output
    assert "nominal" in output


def test_comparison_export_json(source_hdf, tmp_path):
    analysis = _build_analysis(source_hdf, tmp_path)
    comparison = analysis.compare("pT_ll", bins=[0.0, 50.0, 100.0, 150.0])
    output = tmp_path / "comparison.json"
    comparison.export_json(output)
    payload = json.loads(output.read_text())
    assert payload["observable"] == "pT_ll"
    assert "nominal" in payload["histograms"]
    assert "sherpa" in payload["histograms"]


def test_comparison_replica_uncertainty_present(source_hdf, tmp_path):
    analysis = _build_analysis(source_hdf, tmp_path, include_all_replicas=True)
    comparison = analysis.compare("pT_ll", bins=[0.0, 50.0, 100.0, 150.0])
    assert comparison.replica_uncertainty is not None
    assert np.all(comparison.replica_uncertainty >= 0.0)


def test_compare_default_bins(source_hdf, tmp_path):
    analysis = _build_analysis(source_hdf, tmp_path)
    comparison = analysis.compare("pT_ll")
    assert len(comparison.bins) == 31

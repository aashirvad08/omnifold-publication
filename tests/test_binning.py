"""Tests for binning helpers and validity checks (README recs 2-3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from omnifold_publication import (
    equal_effective_events_bins,
    load_package,
    n_eff,
    validate_binning,
    write_package,
)
from omnifold_publication.exceptions import PackageReadError


def _reference_equal_effective_events_bins(values, weights, target):
    """Literal port of equal_effective_events_bins (multifold_util.py:89-120),
    operating on arrays instead of a DataFrame (tqdm removed)."""

    order = np.argsort(values, kind="stable")
    values = np.asarray(values, dtype=float)[order]
    weights = np.asarray(weights, dtype=float)[order]

    total_weight = n_eff(weights)
    n_bins = int(total_weight / target)

    bin_edges = [values[0]]
    bin_index = 0
    cumulative_sum_weights = 0.0
    cumulative_sum_weights_squared = 0.0
    for value, weight in zip(values, weights):
        cumulative_sum_weights += weight
        cumulative_sum_weights_squared += weight**2
        current = cumulative_sum_weights**2 / cumulative_sum_weights_squared
        if current >= (bin_index + 1) * target and bin_index < n_bins - 1:
            bin_edges.append(value)
            bin_index += 1
    bin_edges.append(values.max())
    return bin_edges


def test_n_eff_hand_examples():
    assert n_eff(np.ones(100)) == pytest.approx(100.0)
    # (1+3)^2 / (1+9) = 1.6
    assert n_eff(np.array([1.0, 3.0])) == pytest.approx(1.6)
    assert n_eff(np.zeros(5)) == 0.0


def test_equal_effective_events_bins_matches_literal_port():
    rng = np.random.default_rng(5)
    values = rng.exponential(50.0, 5000)
    weights = rng.uniform(0.5, 1.5, 5000)

    for target in (200.0, 500.0, 1600.0):
        ours = equal_effective_events_bins(values, weights, target)
        reference = _reference_equal_effective_events_bins(
            values, weights, target
        )
        np.testing.assert_allclose(ours, reference, err_msg=f"target={target}")


def test_equal_effective_events_bins_rejects_tiny_samples():
    with pytest.raises(PackageReadError, match="effective events"):
        equal_effective_events_bins(np.array([1.0, 2.0]), np.ones(2), 100.0)


def test_validate_binning_thresholds(tmp_path, atlas_like_hdf):
    package_dir = write_package(
        input_path=atlas_like_hdf,
        output_dir=tmp_path / "package",
        include_all_replicas=True,
    )
    package = load_package(package_dir)
    bins = [0.0, 100.0, 200.0]

    report = validate_binning(
        package, "pT_ll", bins=bins, n_eff_min=1.0, data_stat_max=10.0
    )
    assert report["errors"] == []
    assert report["data_stat_source"] == "bootstrap_data"

    events = package.load_events()
    weights = events["weights_nominal"].to_numpy()
    values = events["pT_ll"].to_numpy()
    for index in range(2):
        in_bin = (values >= bins[index]) & (values < bins[index + 1])
        np.testing.assert_allclose(
            report["n_eff"][index], n_eff(weights[in_bin])
        )

    strict = validate_binning(
        package, "pT_ll", bins=bins, n_eff_min=5000.0, data_stat_max=0.0001
    )
    assert any("n_eff" in error for error in strict["errors"])
    assert any("data statistical" in error for error in strict["errors"])


def test_validate_binning_falls_back_without_bootstrap_family(
    tmp_path, atlas_like_hdf
):
    package_dir = write_package(
        input_path=atlas_like_hdf,
        output_dir=tmp_path / "package",
        include_all_replicas=False,
    )
    package = load_package(package_dir)
    report = validate_binning(
        package, "pT_ll", bins=[0.0, 200.0], n_eff_min=1.0, data_stat_max=10.0
    )
    assert report["data_stat_source"] == "sample_stat"


def test_official_binning_resolution_order(tmp_path):
    input_path = Path("data/multifold.h5")
    if not input_path.exists():
        pytest.skip(f"real data not available: {input_path}")

    package_dir = write_package(
        input_path=input_path,
        output_dir=tmp_path / "package",
        event_count=200,
    )
    package = load_package(package_dir)

    # spec metadata declares the ibu_bins official binning for pT_ll
    assert package.observable_bins("pT_ll") == [
        200.0, 230.0, 300.0, 450.0, 600.0, 1000.0,
    ]
    assert "IBU comparison" in package.observable_binning_provenance("pT_ll")

    # an explicit 'bins' key wins over the official binning
    import yaml

    metadata_path = package_dir / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    for observable in metadata["observables"]:
        if observable["name"] == "pT_ll":
            observable["bins"] = [0.0, 500.0, 1000.0]
    metadata_path.write_text(
        yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8"
    )
    assert load_package(package_dir).observable_bins("pT_ll") == [
        0.0, 500.0, 1000.0,
    ]


def test_real_data_official_binning_validity(tmp_path):
    """README recs 2-3 evaluated on the real file with the official bins."""

    input_path = Path("data/multifold.h5")
    if not input_path.exists():
        pytest.skip(f"real data not available: {input_path}")

    package_dir = write_package(
        input_path=input_path,
        output_dir=tmp_path / "zjets",
        event_count=418014,
        include_all_replicas=True,
    )
    package = load_package(package_dir)
    report = validate_binning(package, "pT_ll")

    np.testing.assert_allclose(
        report["edges"], [200.0, 230.0, 300.0, 450.0, 600.0, 1000.0]
    )
    assert report["data_stat_source"] == "bootstrap_data"
    # internal consistency: errors reflect exactly the per-bin numbers
    expected_errors = int(np.sum(report["n_eff"] < 5000.0)) + int(
        np.sum(report["data_stat_fraction"] >= 0.15)
    )
    assert len(report["errors"]) == expected_errors
    # the official binning of the release must satisfy its own criteria
    assert report["errors"] == []

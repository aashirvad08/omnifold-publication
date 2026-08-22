"""Tests for the closure-grid plotting utility."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.patches import StepPatch

from omnifold_publication import (
    PackageReadError,
    load_package,
    plot_closure_grid,
    write_package,
)

from conftest import set_official_bins as _set_official_bins

BINS = [0.0, 50.0, 100.0, 150.0, 200.0]
BINS_L1 = [0.0, 50.0, 100.0, 150.0]


@pytest.fixture
def closure_pair(tmp_path, atlas_like_hdf):
    """Measurement package plus a slightly shifted target package."""

    measurement = load_package(
        write_package(
            input_path=atlas_like_hdf,
            output_dir=tmp_path / "measurement",
            include_all_replicas=True,
        )
    )
    df = pd.read_hdf(atlas_like_hdf, "df").copy()
    df["weights_nominal"] = df["weights_nominal"] * 1.02
    target_hdf = tmp_path / "target.h5"
    df.to_hdf(target_hdf, key="df", mode="w")
    target = load_package(
        write_package(input_path=target_hdf, output_dir=tmp_path / "target")
    )
    return measurement, target


def test_grid_has_two_axes_per_observable(closure_pair):
    measurement, target = closure_pair
    observables = ["pT_ll", "pT_l1"]
    fig, grid = plot_closure_grid(
        measurement,
        target_package=target,
        observables=observables,
        ncols=2,
        bins_map={"pT_ll": BINS, "pT_l1": BINS_L1},
    )

    # one main + one ratio axes per observable
    assert len(fig.get_axes()) == 2 * len(observables)
    assert set(grid["results"]) == set(observables)
    assert grid["skipped"] == []


def test_ratio_panel_matches_chi2_test_numbers(closure_pair):
    """The drawn ratio must be exactly result/target from chi2_test."""

    measurement, target = closure_pair
    fig, grid = plot_closure_grid(
        measurement,
        target_package=target,
        observables=["pT_ll"],
        bins_map={"pT_ll": BINS},
    )
    outcome = grid["results"]["pT_ll"]

    main_ax, ratio_ax = fig.get_axes()
    ratio_steps = [
        patch for patch in ratio_ax.patches if isinstance(patch, StepPatch)
    ]
    drawn = [np.asarray(patch.get_data().values) for patch in ratio_steps]
    expected = outcome["result"] / outcome["target"]
    assert any(
        values.shape == expected.shape and np.allclose(values, expected)
        for values in drawn
    )
    # main panel carries the result density (result / bin widths)
    widths = np.diff(np.asarray(outcome["edges"]))
    main_steps = [
        np.asarray(patch.get_data().values)
        for patch in main_ax.patches
        if isinstance(patch, StepPatch)
    ]
    assert any(
        values.shape == widths.shape
        and np.allclose(values, outcome["result"] / widths)
        for values in main_steps
    )


def test_default_grid_skips_derived_without_official_binning(
    tmp_path, atlas_like_hdf
):
    package_dir = write_package(
        input_path=atlas_like_hdf,
        output_dir=tmp_path / "package",
        observables=["pT_ll", "pT_l1", "tau21"],
    )
    # fixture data does not populate the real official binning ranges
    _set_official_bins(package_dir, {"pT_ll": BINS, "pT_l1": BINS_L1})
    package = load_package(package_dir)

    df = pd.read_hdf(atlas_like_hdf, "df").copy()
    df["weights_nominal"] = df["weights_nominal"] * 1.02
    target_hdf = tmp_path / "target.h5"
    df.to_hdf(target_hdf, key="df", mode="w")
    target = load_package(
        write_package(
            input_path=target_hdf,
            output_dir=tmp_path / "target",
            observables=["pT_ll", "pT_l1", "tau21"],
        )
    )

    with pytest.warns(UserWarning, match="tau21"):
        fig, grid = plot_closure_grid(package, target_package=target)

    assert grid["skipped"] == ["tau21"]
    assert set(grid["results"]) == {"pT_ll", "pT_l1"}
    # tau21 CAN be included when bins are provided explicitly
    fig2, grid2 = plot_closure_grid(
        package,
        target_package=target,
        observables=["tau21"],
        bins_map={"tau21": [0.0, 0.5, 1.0, 10.0]},
    )
    assert set(grid2["results"]) == {"tau21"}


def test_output_path_saves_figure(tmp_path, closure_pair):
    measurement, target = closure_pair
    output = tmp_path / "grid.png"
    plot_closure_grid(
        measurement,
        target_package=target,
        observables=["pT_ll"],
        bins_map={"pT_ll": BINS},
        output_path=output,
    )
    assert output.exists() and output.stat().st_size > 0


def test_empty_observable_list_raises(closure_pair):
    measurement, target = closure_pair
    with pytest.raises(PackageReadError, match="No observables"):
        plot_closure_grid(measurement, target_package=target, observables=[])

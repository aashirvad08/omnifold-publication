"""Tests for weight-family packaging and ATLAS uncertainty recipes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from omnifold_publication import (
    PackageReadError,
    load_analysis,
    load_package,
    write_manifest,
    write_package,
)
from omnifold_publication.uncertainty import (
    MEDIAN_STD_ERROR_FACTOR,
    ensemble_median_standard_error,
    paired_relative_difference,
    quadrature_difference_from_nominal,
)
from omnifold_publication.validation import validate_package


def _write(tmp_path, source, **kwargs):
    return write_package(
        input_path=source,
        output_dir=tmp_path / "package",
        **kwargs,
    )


def test_writer_discovers_systematic_families(tmp_path, atlas_like_hdf):
    package = load_package(_write(tmp_path, atlas_like_hdf))
    families = {
        name: package.weight_family(name)
        for name in package.list_weight_families()
    }

    assert families["syst_event"]["columns"] == [
        "weights_muEffReco",
        "weights_pileup",
    ]
    assert families["syst_theory"]["columns"] == [
        "weights_theoryPDF",
        "weights_theoryQCD",
    ]
    assert families["syst_track"]["columns"] == ["weights_trackFake"]
    assert families["syst_muon"]["columns"] == ["weights_muCalID"]
    # lumi and topBackground are individual families (release binned code
    # treats them separately; the closure chi2 includes top but not lumi)
    assert families["syst_lumi"]["columns"] == ["weights_lumi"]
    assert families["syst_background"]["columns"] == ["weights_topBackground"]
    assert "syst_other" not in families
    for name in ("syst_event", "syst_theory", "syst_track", "syst_muon"):
        assert families[name]["type"] == "systematic"
        assert (
            families[name]["combination"]
            == "quadrature_difference_from_nominal"
        )

    dd = families["dd_unfolding"]
    assert dd["type"] == "paired"
    assert dd["columns"] == ["weights_dd"]
    assert dd["reference_column"] == "target_dd"


def test_writer_discovers_replica_families(tmp_path, atlas_like_hdf):
    package = load_package(
        _write(tmp_path, atlas_like_hdf, include_all_replicas=True)
    )
    families = {
        name: package.weight_family(name)
        for name in package.list_weight_families()
    }

    assert families["ensemble"]["type"] == "ensemble"
    assert families["ensemble"]["combination"] == "median_standard_error"
    assert families["ensemble"]["columns"] == [
        f"weights_ensemble_{i}" for i in range(4)
    ]
    assert families["bootstrap_mc"]["combination"] == "standard_deviation"
    assert len(families["bootstrap_mc"]["columns"]) == 3
    assert len(families["bootstrap_data"]["columns"]) == 3


def test_include_systematics_false_keeps_old_behavior(
    tmp_path, atlas_like_hdf
):
    package = load_package(
        _write(tmp_path, atlas_like_hdf, include_systematics=False)
    )

    assert package.list_weight_families() == []
    events = package.load_events()
    assert "weights_pileup" not in events.columns
    assert "target_dd" not in events.columns


def test_family_package_validates(tmp_path, atlas_like_hdf):
    package_dir = _write(tmp_path, atlas_like_hdf, include_all_replicas=True)
    assert validate_package(package_dir) == []


def test_get_family_weights_matrix(tmp_path, atlas_like_hdf):
    package = load_package(_write(tmp_path, atlas_like_hdf))
    matrix = package.get_family_weights("syst_event")

    events = package.load_events(
        columns=["weights_muEffReco", "weights_pileup"]
    )
    assert matrix.shape == (2, len(events))
    np.testing.assert_allclose(
        matrix[0], events["weights_muEffReco"].to_numpy()
    )
    np.testing.assert_allclose(matrix[1], events["weights_pileup"].to_numpy())


def test_family_columns_resolve_as_weights(tmp_path, atlas_like_hdf):
    package = load_package(_write(tmp_path, atlas_like_hdf))

    events = package.load_events(columns=["weights_pileup", "target_dd"])
    np.testing.assert_allclose(
        package.get_weights("weights_pileup"),
        events["weights_pileup"].to_numpy(),
    )
    np.testing.assert_allclose(
        package.get_weights("target_dd"),
        events["target_dd"].to_numpy(),
    )
    with pytest.raises(PackageReadError):
        package.get_weights("weights_not_there")


def test_unknown_family_raises(tmp_path, atlas_like_hdf):
    package = load_package(_write(tmp_path, atlas_like_hdf))
    with pytest.raises(PackageReadError, match="Unknown weight family"):
        package.get_family_weights("nope")


def test_breakdown_matches_atlas_recipes(tmp_path, atlas_like_hdf):
    """Components reproduce the multifold_util.py formulas exactly."""

    package = load_package(
        _write(tmp_path, atlas_like_hdf, include_all_replicas=True)
    )
    bins = [0.0, 50.0, 100.0, 150.0, 200.0]
    breakdown = package.uncertainty_breakdown("pT_ll", bins=bins)

    events = package.load_events()
    values = events["pT_ll"].to_numpy(dtype=float)
    nominal_w = events["weights_nominal"].to_numpy(dtype=float)
    nom, _ = np.histogram(values, bins=bins, weights=nominal_w)

    # calculate_uncertainty: quadrature of (syst - nom) over the family
    expected_event = np.zeros(len(bins) - 1)
    for column in ("weights_muEffReco", "weights_pileup"):
        syst, _ = np.histogram(
            values, bins=bins, weights=events[column].to_numpy()
        )
        expected_event += (syst - nom) ** 2
    np.testing.assert_allclose(
        breakdown["components"]["syst_event"], np.sqrt(expected_event)
    )

    # calculate_stat_uncertainty: std across bootstrap replica histograms
    boot_hists = [
        np.histogram(
            values,
            bins=bins,
            weights=events[f"weights_bootstrap_mc_{i}"].to_numpy(),
        )[0]
        for i in range(3)
    ]
    np.testing.assert_allclose(
        breakdown["components"]["bootstrap_mc"],
        np.std(boot_hists, axis=0),
    )

    # ensemble: 1.253 * std / sqrt(N)
    ensemble_hists = [
        np.histogram(
            values,
            bins=bins,
            weights=events[f"weights_ensemble_{i}"].to_numpy(),
        )[0]
        for i in range(4)
    ]
    np.testing.assert_allclose(
        breakdown["components"]["ensemble"],
        MEDIAN_STD_ERROR_FACTOR * np.std(ensemble_hists, axis=0) / 2.0,
    )

    # dd pair: |(dd - target) * nom / target|
    dd, _ = np.histogram(
        values, bins=bins, weights=events["weights_dd"].to_numpy()
    )
    target, _ = np.histogram(
        values, bins=bins, weights=events["target_dd"].to_numpy()
    )
    np.testing.assert_allclose(
        breakdown["components"]["dd_unfolding"],
        np.abs((dd - target) * nom / target),
    )

    # sample stat: sqrt(hist of w^2)
    np.testing.assert_allclose(
        breakdown["components"]["sample_stat"],
        np.sqrt(np.histogram(values, bins=bins, weights=nominal_w**2)[0]),
    )

    # total: quadrature of all components
    stacked = np.vstack(list(breakdown["components"].values()))
    np.testing.assert_allclose(
        breakdown["total"], np.sqrt((stacked**2).sum(axis=0))
    )
    np.testing.assert_allclose(breakdown["nominal"], nom)


def test_recipe_functions_directly():
    nominal = np.array([10.0, 20.0])
    variations = np.array([[11.0, 19.0], [9.0, 22.0]])

    np.testing.assert_allclose(
        quadrature_difference_from_nominal(nominal, variations),
        np.sqrt(np.array([1.0 + 1.0, 1.0 + 4.0])),
    )
    np.testing.assert_allclose(
        ensemble_median_standard_error(variations),
        MEDIAN_STD_ERROR_FACTOR
        * np.std(variations, axis=0)
        / np.sqrt(2.0),
    )
    # zero reference bins contribute zero, not inf
    np.testing.assert_allclose(
        paired_relative_difference(
            nominal, np.array([12.0, 5.0]), np.array([10.0, 0.0])
        ),
        np.array([2.0, 0.0]),
    )


def test_normalization_metadata_is_absolute(tmp_path, atlas_like_hdf):
    package = load_package(_write(tmp_path, atlas_like_hdf))
    normalization = package.metadata()["normalization"]

    assert normalization["mode"] == "absolute"
    assert normalization["weight_units"] == "fb"
    assert normalization["sum_weights_equals"] == "fiducial_cross_section"
    assert package.weight_units() == "fb"

    events = package.load_events(columns=["weights_nominal"])
    assert package.cross_section() == pytest.approx(
        float(events["weights_nominal"].sum())
    )


def test_analysis_breakdown_adds_two_point_component(
    tmp_path, atlas_like_hdf
):
    nominal_dir = write_package(
        input_path=atlas_like_hdf,
        output_dir=tmp_path / "nominal",
    )
    alternative_dir = write_package(
        input_path=atlas_like_hdf,
        output_dir=tmp_path / "alternative",
    )
    manifest_dir = tmp_path / "analysis"
    write_manifest(
        output_dir=manifest_dir,
        nominal_path=nominal_dir,
        variations={
            "generator_choice": {
                "path": alternative_dir,
                "type": "alternative_generator",
                "combination": "two_point_difference",
            }
        },
        analysis_name="family-breakdown-test",
    )
    analysis = load_analysis(manifest_dir)
    bins = [0.0, 100.0, 200.0]
    breakdown = analysis.uncertainty_breakdown("pT_ll", bins=bins)

    assert "two_point_generator_choice" in breakdown["components"]
    # identical samples: the two-point difference must vanish
    np.testing.assert_allclose(
        breakdown["components"]["two_point_generator_choice"],
        np.zeros(len(bins) - 1),
        atol=1e-9,
    )
    stacked = np.vstack(list(breakdown["components"].values()))
    np.testing.assert_allclose(
        breakdown["total"], np.sqrt((stacked**2).sum(axis=0))
    )


def test_real_data_families_and_breakdown(tmp_path):
    """Family capture and breakdown against the real ATLAS release file."""

    input_path = Path("data/multifold.h5")
    if not input_path.exists():
        pytest.skip(f"real data not available: {input_path}")

    package_dir = write_package(
        input_path=input_path,
        output_dir=tmp_path / "zjets",
        event_count=2000,
        include_all_replicas=True,
    )
    package = load_package(package_dir)
    families = {
        name: package.weight_family(name)
        for name in package.list_weight_families()
    }

    systematic_columns = [
        column
        for name, family in families.items()
        if family["type"] == "systematic"
        for column in family["columns"]
    ]
    assert len(systematic_columns) == 22
    assert families["ensemble"]["columns"] == sorted(
        f"weights_ensemble_{i}" for i in range(100)
    )
    assert len(families["bootstrap_mc"]["columns"]) == 25
    assert len(families["bootstrap_data"]["columns"]) == 25
    assert families["dd_unfolding"]["reference_column"] == "target_dd"

    breakdown = package.uncertainty_breakdown("pT_ll", bins=6)
    assert np.isfinite(breakdown["total"]).all()
    assert (breakdown["total"] >= 0).all()
    expected_components = {
        "sample_stat",
        "syst_event",
        "syst_theory",
        "syst_track",
        "syst_muon",
        "syst_lumi",
        "syst_background",
        "dd_unfolding",
        "bootstrap_mc",
        "bootstrap_data",
        "ensemble",
    }
    assert expected_components <= set(breakdown["components"])

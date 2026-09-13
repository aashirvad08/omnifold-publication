"""Tests for the declaration-based writer.

The point of this path is that **nothing is inferred**. These tests pin
that: only declared columns are packaged, an undeclared weight column
never becomes a systematic, a sample with no base weight is publishable,
and a bad declaration is reported in full before anything is written.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from omnifold_publication import (
    load_package,
    write_package_from_declaration,
)
from omnifold_publication.exceptions import PackageReadError, PackageWriteError

BINS = [0.0, 50.0, 100.0, 200.0]


@pytest.fixture
def foreign(tmp_path):
    """An unfolding output named nothing like the reference release."""

    rng = np.random.default_rng(7)
    n = 200
    nominal = rng.uniform(0.5, 1.5, n)
    frame = pd.DataFrame(
        {
            "evt": np.arange(n),
            "dilepton_pt": rng.uniform(0.0, 200.0, n),
            "jet_pt": rng.uniform(0.0, 100.0, n),
            "w_prior": rng.uniform(0.5, 1.5, n),
            "w_unfolded": nominal,
            "w_jes": nominal * rng.normal(1.0, 0.05, n),
            "w_jer": nominal * rng.normal(1.0, 0.05, n),
            "w_alt": nominal * rng.normal(1.0, 0.05, n),
            "w_alt_ref": nominal * rng.normal(1.0, 0.05, n),
            **{f"w_bs_{i}": nominal * rng.normal(1.0, 0.05, n) for i in range(4)},
        }
    )
    path = tmp_path / "foreign.parquet"
    frame.to_parquet(path, index=False)
    return path, frame


def _declare(tmp_path, foreign_path, **overrides):
    kwargs = dict(
        events=foreign_path,
        output_dir=tmp_path / "pkg",
        observables={
            "dilepton_pt": {
                "units": "GeV",
                "description": "Dilepton transverse momentum",
                "bins": BINS,
                "binning_provenance": "analysis note table 3",
            }
        },
        weights={"nominal": "w_unfolded", "base_mc_weight": "w_prior"},
        families={
            "detector": {
                "type": "systematic",
                "combination": "quadrature_difference_from_nominal",
                "columns": ["w_jes", "w_jer"],
            },
            "mc_stat": {
                "type": "bootstrap",
                "combination": "standard_deviation",
                "columns": [f"w_bs_{i}" for i in range(4)],
            },
        },
    )
    kwargs.update(overrides)
    return write_package_from_declaration(**kwargs)


def test_declared_package_round_trips(tmp_path, foreign):
    path, frame = foreign
    package = load_package(_declare(tmp_path, path))
    package.validate()

    assert package.list_observables() == ["dilepton_pt"]
    assert package.list_weight_families() == ["detector", "mc_stat"]
    assert package.observable_bins("dilepton_pt") == BINS
    assert package.cross_section() == pytest.approx(frame["w_unfolded"].sum())

    breakdown = package.uncertainty_breakdown("dilepton_pt")
    assert set(breakdown["components"]) == {"sample_stat", "detector", "mc_stat"}
    assert package.covariance_matrix("dilepton_pt")["total"].shape == (3, 3)


def test_nothing_is_inferred_from_column_names(tmp_path, foreign):
    """Undeclared weight columns are neither packaged nor made systematics.

    This is the whole reason the path exists: ``write_package`` would sweep
    the undeclared ``w_alt``/``w_alt_ref`` columns into ``syst_other`` and
    silently inflate the total.
    """

    path, _ = foreign
    package = load_package(_declare(tmp_path, path))

    columns = set(package.load_events().columns)
    assert "w_alt" not in columns
    assert "w_alt_ref" not in columns

    declared = set()
    for name in package.list_weight_families():
        declared.update(package.weight_family(name)["columns"])
    assert "w_alt" not in declared
    assert not any("other" in name for name in package.list_weight_families())


def test_sample_with_no_base_weight_is_publishable(tmp_path, foreign):
    """A single-weight alternative sample needs no invented base weight."""

    path, frame = foreign
    package_dir = write_package_from_declaration(
        events=path,
        output_dir=tmp_path / "single",
        observables={"dilepton_pt": {"units": "GeV", "bins": BINS}},
        weights={"nominal": "w_unfolded", "base_mc_weight": None},
    )
    package = load_package(package_dir)
    package.validate()

    metadata = yaml.safe_load(
        (package_dir / "metadata.yaml").read_text(encoding="utf-8")
    )
    assert "base_mc_weight" not in metadata["weights"]
    assert package.list_weight_families() == []
    # the default convention never needs it, so final weights still resolve
    np.testing.assert_allclose(
        package.get_weights("final"), package.get_weights("nominal")
    )


def test_reweighting_factor_requires_a_base_weight(tmp_path, foreign):
    path, _ = foreign
    with pytest.raises(PackageWriteError, match="base_mc_weight"):
        write_package_from_declaration(
            events=path,
            output_dir=tmp_path / "bad",
            observables={"dilepton_pt": {"units": "GeV", "bins": BINS}},
            weights={
                "nominal": "w_unfolded",
                "base_mc_weight": None,
                "nominal_convention": "reweighting_factor",
            },
        )


def test_reader_reports_a_missing_base_weight_for_that_convention(
    tmp_path, foreign
):
    path, _ = foreign
    package_dir = write_package_from_declaration(
        events=path,
        output_dir=tmp_path / "single",
        observables={"dilepton_pt": {"units": "GeV", "bins": BINS}},
        weights={"nominal": "w_unfolded", "base_mc_weight": None},
    )
    metadata_path = package_dir / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    metadata["weights"]["nominal_convention"] = "reweighting_factor"
    metadata_path.write_text(yaml.safe_dump(metadata), encoding="utf-8")

    with pytest.raises(PackageReadError, match="base_mc_weight"):
        load_package(package_dir).get_weights("final")


def test_every_problem_is_reported_at_once_and_nothing_is_written(
    tmp_path, foreign
):
    path, _ = foreign
    output = tmp_path / "never"

    with pytest.raises(PackageWriteError) as info:
        write_package_from_declaration(
            events=path,
            output_dir=output,
            observables={"ghost": {"units": "GeV", "selection": "phantom > 5"}},
            weights={"nominal": "nope", "base_mc_weight": "also_nope"},
            families={
                "typo": {
                    "type": "systematic",
                    "combination": "quadrature_difference",
                    "columns": ["missing"],
                }
            },
        )

    message = str(info.value)
    for expected in ("nope", "also_nope", "ghost", "phantom", "typo", "missing"):
        assert expected in message
    assert not output.exists()


def test_unknown_combination_is_rejected(tmp_path, foreign):
    path, _ = foreign
    with pytest.raises(PackageWriteError, match="unknown combination"):
        _declare(
            tmp_path,
            path,
            families={
                "x": {
                    "type": "systematic",
                    "combination": "average_of_everything",
                    "columns": ["w_jes"],
                }
            },
        )


def test_paired_family_is_validated_at_write_time(tmp_path, foreign):
    path, _ = foreign
    with pytest.raises(PackageWriteError, match="reference_column"):
        _declare(
            tmp_path,
            path,
            families={
                "pair": {
                    "type": "paired",
                    "combination": "paired_relative_difference",
                    "columns": ["w_alt"],
                }
            },
        )

    with pytest.raises(PackageWriteError, match="exactly one column"):
        _declare(
            tmp_path,
            path,
            families={
                "pair": {
                    "type": "paired",
                    "combination": "paired_relative_difference",
                    "columns": ["w_alt", "w_jes"],
                    "reference_column": "w_alt_ref",
                }
            },
        )


def test_paired_family_round_trips(tmp_path, foreign):
    path, _ = foreign
    package = load_package(
        _declare(
            tmp_path,
            path,
            families={
                "unfolding": {
                    "type": "paired",
                    "combination": "paired_relative_difference",
                    "columns": ["w_alt"],
                    "reference_column": "w_alt_ref",
                }
            },
        )
    )
    assert package.weight_family("unfolding")["reference_column"] == "w_alt_ref"
    breakdown = package.uncertainty_breakdown("dilepton_pt")
    assert "unfolding" in breakdown["components"]


def test_declared_bins_are_official_so_closure_and_hepdata_accept_them(
    tmp_path, foreign
):
    """Declared edges must land in binning.official, not a soft suggestion."""

    path, _ = foreign
    package_dir = _declare(tmp_path, path)
    metadata = yaml.safe_load(
        (package_dir / "metadata.yaml").read_text(encoding="utf-8")
    )
    binning = metadata["observables"][0]["binning"]
    assert binning["official"] == BINS
    assert binning["provenance"] == "analysis note table 3"

    submission = load_package(package_dir).export_hepdata(tmp_path / "hepdata")
    assert (submission / "submission.yaml").exists()


def test_selection_columns_are_validated_and_packaged(tmp_path, foreign):
    path, _ = foreign
    package_dir = _declare(
        tmp_path,
        path,
        observables={
            "dilepton_pt": {"units": "GeV", "bins": BINS},
            "jet_pt": {"units": "GeV", "bins": BINS, "selection": "jet_pt > 5"},
        },
    )
    package = load_package(package_dir)
    values, mask = package.observable_values("jet_pt")
    assert mask.sum() < len(mask)
    assert float(values.min()) > 5.0


def test_event_id_column_declares_column_alignment(tmp_path, foreign):
    path, _ = foreign
    package_dir = _declare(tmp_path, path, event_id_column="evt")
    load_package(package_dir).validate()

    metadata = yaml.safe_load(
        (package_dir / "metadata.yaml").read_text(encoding="utf-8")
    )
    alignment = metadata["publication"]["event_alignment"]
    assert alignment == {"method": "column", "column": "evt"}


def test_subsetting_records_the_assumption(tmp_path, foreign):
    path, _ = foreign
    package_dir = _declare(tmp_path, path, event_count=50)
    metadata = yaml.safe_load(
        (package_dir / "metadata.yaml").read_text(encoding="utf-8")
    )
    assert metadata["publication"]["event_count"] == 50
    assert any(
        "sum(w) is not the full-sample normalisation" in note
        for note in metadata["dataset"]["assumptions"]
    )


def test_accepts_a_dataframe_directly(tmp_path, foreign):
    path, frame = foreign
    package = load_package(_declare(tmp_path, frame))
    assert package.summary()["event_count"] == len(frame)


# --- the convention-based writer's two sharp edges ------------------------


def test_write_package_reports_missing_columns_typed(tmp_path, foreign):
    """A foreign file must not meet a raw pandas KeyError."""

    from omnifold_publication import write_package

    path, _ = foreign
    with pytest.raises(PackageWriteError) as info:
        write_package(
            input_path=path,
            output_dir=tmp_path / "pkg",
            observables=["dilepton_pt"],
        )
    message = str(info.value)
    assert "weight_mc" in message
    assert "column_rename" in message
    assert "write_package_from_declaration" in message


def test_write_package_announces_its_syst_other_guess(tmp_path, atlas_like_hdf):
    """Grouping unmatched columns as systematics is a guess; it must be said."""

    from omnifold_publication import write_package

    with pytest.warns(UserWarning, match="syst_other"):
        write_package(
            input_path=atlas_like_hdf,
            output_dir=tmp_path / "pkg",
            column_rename={"weights_pileup": "weights_mystery"},
        )

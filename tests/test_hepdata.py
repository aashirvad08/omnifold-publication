"""Tests for HEPData export and the binned submission loader."""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from conftest import set_official_bins

from omnifold_publication import (
    PackageReadError,
    load_hepdata_submission,
    load_package,
    write_package,
)

BINS = [0.0, 50.0, 100.0, 150.0, 200.0]
BINS_L1 = [0.0, 50.0, 100.0, 150.0]


@pytest.fixture
def exported(tmp_path, atlas_like_hdf):
    """A package with fixture-ranged official bins, exported to HEPData."""

    package_dir = write_package(
        input_path=atlas_like_hdf,
        output_dir=tmp_path / "package",
        include_all_replicas=True,
    )
    set_official_bins(package_dir, {"pT_ll": BINS, "pT_l1": BINS_L1})
    package = load_package(package_dir)
    submission_dir = package.export_hepdata(tmp_path / "hepdata")
    return package, submission_dir


def test_export_passes_official_hepdata_validators(exported):
    """The generated files must satisfy HEPData's own schema validators."""

    from hepdata_validator.data_file_validator import DataFileValidator
    from hepdata_validator.submission_file_validator import (
        SubmissionFileValidator,
    )

    package, submission_dir = exported
    submission_path = submission_dir / "submission.yaml"

    submission_validator = SubmissionFileValidator()
    assert submission_validator.validate(file_path=str(submission_path)), (
        submission_validator.get_messages()
    )

    data_validator = DataFileValidator()
    for table_file in ("data_pt_ll.yaml", "data_pt_l1.yaml"):
        path = submission_dir / table_file
        assert path.exists()
        assert data_validator.validate(file_path=str(path)), (
            data_validator.get_messages()
        )


def test_export_values_match_uncertainty_breakdown(exported):
    package, submission_dir = exported
    table = yaml.safe_load(
        (submission_dir / "data_pt_ll.yaml").read_text(encoding="utf-8")
    )

    breakdown = package.uncertainty_breakdown("pT_ll", bins=BINS)
    widths = np.diff(np.asarray(BINS))

    rows = table["dependent_variables"][0]["values"]
    np.testing.assert_allclose(
        [row["value"] for row in rows], breakdown["nominal"] / widths
    )
    np.testing.assert_allclose(
        [row["errors"][0]["symerror"] for row in rows],
        breakdown["total"] / widths,
    )
    assert all(row["errors"][0]["label"] == "total" for row in rows)

    edges = table["independent_variables"][0]["values"]
    assert [entry["low"] for entry in edges] == BINS[:-1]
    assert [entry["high"] for entry in edges] == BINS[1:]

    qualifiers = table["dependent_variables"][0]["qualifiers"]
    names = {qualifier["name"] for qualifier in qualifiers}
    assert {"SQRT(S)", "LUMINOSITY"} <= names
    assert table["dependent_variables"][0]["header"]["units"] == "fb/GeV"


def test_export_error_breakdown_labels_components(tmp_path, exported):
    package, _ = exported
    submission_dir = package.export_hepdata(
        tmp_path / "hepdata_breakdown", error_breakdown=True
    )
    table = yaml.safe_load(
        (submission_dir / "data_pt_ll.yaml").read_text(encoding="utf-8")
    )

    labels = {
        error["label"]
        for error in table["dependent_variables"][0]["values"][0]["errors"]
    }
    assert {"sample_stat", "syst_event", "dd_unfolding", "ensemble"} <= labels

    from hepdata_validator.data_file_validator import DataFileValidator

    validator = DataFileValidator()
    assert validator.validate(
        file_path=str(submission_dir / "data_pt_ll.yaml")
    ), validator.get_messages()


def test_round_trip_load_matches_export(exported):
    package, submission_dir = exported
    binned = load_hepdata_submission(submission_dir)

    assert set(binned.list_observables()) == {"pT_ll", "pT_l1"}
    assert binned.observable_bins("pT_ll") == BINS

    breakdown = package.uncertainty_breakdown("pT_ll", bins=BINS)
    widths = np.diff(np.asarray(BINS))
    result = binned.histogram("pT_ll")
    np.testing.assert_allclose(result.hist, breakdown["nominal"] / widths)
    np.testing.assert_allclose(
        result.total_uncertainty, breakdown["total"] / widths
    )
    # integrating the density recovers the binned cross-section
    assert binned.cross_section("pT_ll") == pytest.approx(
        float(breakdown["nominal"].sum())
    )


def test_load_package_auto_detects_submission_dir(exported):
    _, submission_dir = exported
    binned = load_package(submission_dir)
    assert binned.summary()["binned_only"] is True
    assert "pT_ll" in binned.list_observables()


def test_labelled_errors_reconstruct_total_in_quadrature(
    tmp_path, exported
):
    package, _ = exported
    submission_dir = package.export_hepdata(
        tmp_path / "hepdata_breakdown", error_breakdown=True
    )
    binned = load_hepdata_submission(submission_dir)
    result = binned.histogram("pT_ll")

    breakdown = package.uncertainty_breakdown("pT_ll", bins=BINS)
    widths = np.diff(np.asarray(BINS))
    np.testing.assert_allclose(
        result.total_uncertainty, breakdown["total"] / widths
    )
    np.testing.assert_allclose(
        result.stat_uncertainty,
        breakdown["components"]["sample_stat"] / widths,
    )


def test_event_level_api_raises_clear_errors(exported):
    _, submission_dir = exported
    binned = load_hepdata_submission(submission_dir)

    for operation in (
        lambda: binned.load_events(),
        lambda: binned.get_weights("nominal"),
        lambda: binned.list_weight_families(),
        lambda: binned.get_family_weights("ensemble"),
        lambda: binned.get_uncertainty("replica"),
        lambda: binned.observable_values("pT_ll"),
        lambda: binned.uncertainty_breakdown("pT_ll"),
        lambda: binned.covariance_matrix("pT_ll"),
    ):
        with pytest.raises(PackageReadError, match="binned"):
            operation()


def test_binned_validate_passes_and_catches_corruption(exported):
    from omnifold_publication import PackageValidationError

    _, submission_dir = exported
    binned = load_hepdata_submission(submission_dir)
    binned.validate()  # a freshly exported submission is valid

    binned.table("pT_ll")["values"][0] = np.nan
    binned.table("pT_ll")["errors"]["total"][1] = -1.0
    with pytest.raises(PackageValidationError) as excinfo:
        binned.validate()
    message = str(excinfo.value)
    assert "non-finite bin values" in message
    assert "negative magnitudes" in message


def test_export_without_official_binning_raises(tmp_path, atlas_like_hdf):
    package_dir = write_package(
        input_path=atlas_like_hdf,
        output_dir=tmp_path / "package",
        observables=["pT_ll", "tau21"],
    )
    set_official_bins(package_dir, {"pT_ll": BINS})
    package = load_package(package_dir)

    with pytest.raises(PackageReadError, match="No official binning"):
        package.export_hepdata(tmp_path / "hepdata", observables=["tau21"])

    # but bins_map makes derived observables exportable, selection applied
    submission_dir = package.export_hepdata(
        tmp_path / "hepdata2",
        observables=["tau21"],
        bins_map={"tau21": [0.0, 0.5, 1.0, 10.0]},
    )
    binned = load_hepdata_submission(submission_dir)
    assert binned.observable_bins("tau21") == [0.0, 0.5, 1.0, 10.0]


def test_internal_format_conversion_offline():
    """The network path's x/y-format converter, on a Table 1-like stub."""

    from omnifold_publication.hepdata import _normalize_internal_table

    stub = {
        "headers": [
            {"colspan": 1, "name": "Dilepton $p_{\\text{T}}$ [GeV]"},
            {"colspan": 3, "name": "Differential cross-section [fb/GeV]"},
        ],
        "qualifiers": {},
        "values": [
            {
                "x": [{"low": "200.0", "high": "230.0"}],
                "y": [
                    {"value": "23.4681", "errors": [{"symerror": "0.5332"}]},
                    {"value": "22.9586", "errors": [{"symerror": "1.608"}]},
                ],
            },
            {
                "x": [{"low": "230.0", "high": "300.0"}],
                "y": [
                    {
                        "value": "10.0673",
                        "errors": [
                            {"asymerror": {"plus": "0.3", "minus": "-0.2"}}
                        ],
                    },
                    {"value": "9.9", "errors": []},
                ],
            },
        ],
    }
    table = _normalize_internal_table("Table 1", "dimuon pT", stub)

    np.testing.assert_allclose(table["edges"], [200.0, 230.0, 300.0])
    np.testing.assert_allclose(table["values"], [23.4681, 10.0673])
    np.testing.assert_allclose(table["errors"]["total"], [0.5332, 0.3])
    assert table["units"] == "GeV"
    assert table["value_units"] == "fb/GeV"

    # point data (no low/high) is not a binned distribution -> skipped
    point_stub = {
        "headers": [{"name": "x"}],
        "values": [{"x": [{"value": "1.0"}], "y": [{"value": "2.0"}]}],
    }
    assert _normalize_internal_table("Table 2", "", point_stub) is None


def test_live_record_round_trip():
    """Network-gated: load the real record and check published values.

    The reference numbers are the published Table 1 (dimuon pT) Data
    column of https://www.hepdata.net/record/ins2791852, which is frozen
    at version 1.
    """

    try:
        binned = load_package("hepdata:ins2791852")
    except PackageReadError as exc:
        pytest.skip(f"hepdata.net not reachable: {exc}")

    assert len(binned.list_observables()) >= 24
    binned.validate()

    result = binned.histogram("Table 1")
    np.testing.assert_allclose(
        result.edges, [200.0, 230.0, 300.0, 450.0, 600.0, 1000.0]
    )
    assert result.hist[0] == pytest.approx(23.4681)
    assert result.total_uncertainty[0] == pytest.approx(0.5332)
    # integral of the differential cross-section ~ fiducial cross-section
    assert 1700.0 < binned.cross_section("Table 1") < 1900.0

    # compare against our own export from the local (reduced-version)
    # release files, within the documented data-version tolerance
    from pathlib import Path

    if not Path("data/multifold.h5").exists():
        return
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        package = load_package(
            write_package(
                input_path="data/multifold.h5",
                output_dir=Path(tmp) / "package",
                event_count=418014,
                include_all_replicas=True,
            )
        )
        ours = load_hepdata_submission(
            package.export_hepdata(Path(tmp) / "hepdata", observables=["pT_ll"])
        ).histogram("pT_ll")
    np.testing.assert_allclose(ours.hist, result.hist, rtol=0.05)


def test_malformed_submission_raises(tmp_path):
    (tmp_path / "submission.yaml").write_text(
        "name: broken\ndata_file: missing.yaml\n", encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError):
        load_hepdata_submission(tmp_path)
    with pytest.raises(PackageReadError, match="No submission.yaml"):
        load_hepdata_submission(tmp_path / "nowhere")

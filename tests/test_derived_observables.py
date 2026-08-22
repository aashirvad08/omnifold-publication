"""Tests for derived observables (port of calculate_vars) and selections."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from omnifold_publication import (
    PackageReadError,
    compute_derived_observables,
    load_package,
    write_package,
)
from omnifold_publication.exceptions import PackageWriteError
from omnifold_publication.selection import (
    parse_selection,
    selection_columns,
    selection_mask,
)


def _reference_calculate_vars(df):
    """Literal port of calculate_vars (multifold_util.py:257-304)."""

    import vector

    df = df.copy()
    df["tau21"] = df.tau2_trackj1 / df.tau1_trackj1

    def dR(v1, v2):
        dy = v1.rapidity - v2.rapidity
        dphi = v1.deltaphi(v2)
        return np.sqrt(dy**2 + dphi**2)

    def dR_ll_trackj1(pT_l1, eta_l1, phi_l1, pT_l2, eta_l2, phi_l2,
                      pT_trackj1, y_trackj1, phi_trackj1):
        l1 = vector.array(
            {"pt": pT_l1, "phi": phi_l1, "eta": eta_l1,
             "m": np.zeros_like(pT_l1)}
        )
        l2 = vector.array(
            {"pt": pT_l2, "phi": phi_l2, "eta": eta_l2,
             "m": np.zeros_like(pT_l2)}
        )
        track_j1 = vector.array(
            {"pt": pT_trackj1, "phi": phi_trackj1, "eta": y_trackj1,
             "m": np.zeros_like(pT_trackj1)}
        )
        return dR(l1 + l2, track_j1)

    def phi_ll(pt_l1, eta_l1, phi_l1, pt_l2, eta_l2, phi_l2):
        l1 = vector.array({"pt": pt_l1, "eta": eta_l1, "phi": phi_l1,
                           "m": np.zeros(len(pt_l1))})
        l2 = vector.array({"pt": pt_l2, "eta": eta_l2, "phi": phi_l2,
                           "m": np.zeros(len(pt_l2))})
        return l1.add(l2).phi

    df["phi_ll"] = phi_ll(
        np.array(df.pT_l1), np.array(df.eta_l1), np.array(df.phi_l1),
        np.array(df.pT_l2), np.array(df.eta_l2), np.array(df.phi_l2))
    df["dR_ll"] = dR_ll_trackj1(
        np.array(df.pT_l1), np.array(df.eta_l1), np.array(df.phi_l1),
        np.array(df.pT_l2), np.array(df.eta_l2), np.array(df.phi_l2),
        np.array(df.pT_trackj1), np.array(df.y_trackj1),
        np.array(df.phi_trackj1))
    return df


def _kinematics_frame(n=200, seed=3):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "pT_l1": rng.uniform(25.0, 300.0, n),
            "eta_l1": rng.uniform(-2.5, 2.5, n),
            "phi_l1": rng.uniform(-np.pi, np.pi, n),
            "pT_l2": rng.uniform(25.0, 200.0, n),
            "eta_l2": rng.uniform(-2.5, 2.5, n),
            "phi_l2": rng.uniform(-np.pi, np.pi, n),
            "pT_trackj1": rng.uniform(6.0, 500.0, n),
            "y_trackj1": rng.uniform(-2.5, 2.5, n),
            "phi_trackj1": rng.uniform(-np.pi, np.pi, n),
            "tau1_trackj1": rng.uniform(0.01, 0.9, n),
            "tau2_trackj1": rng.uniform(0.0, 0.5, n),
        }
    )


def test_matches_literal_calculate_vars_port():
    df = _kinematics_frame()
    ours = compute_derived_observables(df)
    reference = _reference_calculate_vars(df)

    for name in ("tau21", "phi_ll", "dR_ll"):
        np.testing.assert_allclose(
            ours[name].to_numpy(), reference[name].to_numpy(), err_msg=name
        )


def test_jet_vector_uses_rapidity_not_pseudorapidity():
    """dR must place the jet at y_trackj1 and measure dy in rapidity.

    Verified against a from-scratch numpy computation: massless leptons
    summed into the (massive) dilepton system whose rapidity is
    0.5*ln((E+pz)/(E-pz)); the jet contributes its y directly.
    """

    df = _kinematics_frame(n=50, seed=11)
    ours = compute_derived_observables(df, ["dR_ll"])["dR_ll"].to_numpy()

    def momenta(pt, eta, phi):
        return (
            pt * np.cos(phi),
            pt * np.sin(phi),
            pt * np.sinh(eta),
            pt * np.cosh(eta),  # massless: E = |p|
        )

    px1, py1, pz1, e1 = momenta(
        df["pT_l1"].to_numpy(), df["eta_l1"].to_numpy(), df["phi_l1"].to_numpy()
    )
    px2, py2, pz2, e2 = momenta(
        df["pT_l2"].to_numpy(), df["eta_l2"].to_numpy(), df["phi_l2"].to_numpy()
    )
    px, py, pz, e = px1 + px2, py1 + py2, pz1 + pz2, e1 + e2
    y_ll = 0.5 * np.log((e + pz) / (e - pz))
    phi_ll = np.arctan2(py, px)

    dphi = np.mod(phi_ll - df["phi_trackj1"].to_numpy() + np.pi, 2 * np.pi) - np.pi
    expected = np.sqrt(
        (y_ll - df["y_trackj1"].to_numpy()) ** 2 + dphi**2
    )
    np.testing.assert_allclose(ours, expected, rtol=1e-10)


def test_tau21_keeps_non_finite_for_selection_to_remove():
    df = _kinematics_frame(n=4)
    df.loc[0, "tau1_trackj1"] = 0.0
    tau21 = compute_derived_observables(df, ["tau21"])["tau21"].to_numpy()
    assert not np.isfinite(tau21[0])
    assert np.isfinite(tau21[1:]).all()


def test_missing_input_columns_raise():
    with pytest.raises(PackageWriteError, match="requires missing columns"):
        compute_derived_observables(pd.DataFrame({"pT_l1": [1.0]}), ["dR_ll"])
    with pytest.raises(PackageWriteError, match="Unknown derived"):
        compute_derived_observables(_kinematics_frame(), ["mjj"])


def test_selection_parser_accepts_conjunctions_and_rejects_code():
    assert parse_selection("pT_trackj1 > 5") == [("pT_trackj1", ">", 5.0)]
    assert parse_selection("a >= 1.5e2 and b != -2") == [
        ("a", ">=", 150.0),
        ("b", "!=", -2.0),
    ]
    assert selection_columns("a > 1 and a < 9 and b == 0") == ["a", "b"]

    for bad in ("__import__('os')", "a > b", "a >> 3", "a > 5; b > 2", ""):
        with pytest.raises(PackageReadError, match="Cannot parse"):
            parse_selection(bad)


def test_selection_mask_applies_all_clauses():
    df = pd.DataFrame({"a": [1.0, 6.0, 8.0, 11.0], "b": [0.0, 1.0, 0.0, 1.0]})
    mask = selection_mask(df, "a > 5 and b == 0")
    np.testing.assert_array_equal(mask, [False, False, True, False])
    with pytest.raises(PackageReadError, match="not in"):
        selection_mask(df, "missing > 1")


def test_written_package_derives_and_declares_observables(
    tmp_path, atlas_like_hdf
):
    package_dir = write_package(
        input_path=atlas_like_hdf,
        output_dir=tmp_path / "package",
        observables=["pT_ll", "tau21", "dR_ll"],
    )
    package = load_package(package_dir)

    events = package.load_events()
    assert {"tau21", "dR_ll", "pT_trackj1"} <= set(events.columns)
    assert package.observable_selection("tau21") == "pT_trackj1 > 5"
    assert package.observable_selection("dR_ll") == "pT_trackj1 > 5"
    assert package.observable_selection("pT_ll") is None

    entry = next(
        observable
        for observable in package.metadata()["observables"]
        if observable["name"] == "tau21"
    )
    assert entry["derived_from"] == ["tau1_trackj1", "tau2_trackj1"]


def test_histogram_applies_selection(tmp_path, atlas_like_hdf):
    package_dir = write_package(
        input_path=atlas_like_hdf,
        output_dir=tmp_path / "package",
        observables=["pT_ll", "tau21"],
    )
    package = load_package(package_dir)
    events = package.load_events()
    mask = events["pT_trackj1"].to_numpy() > 5
    assert 0 < mask.sum() < len(events)  # fixture has both populations

    values, returned_mask = package.observable_values("tau21")
    np.testing.assert_array_equal(returned_mask, mask)
    assert np.isfinite(values).all()  # tau1 == 0 events are masked out

    result = package.histogram("tau21", bins=[0.0, 0.5, 1.0, 10.0])
    expected, _ = np.histogram(
        events["tau21"].to_numpy()[mask],
        bins=[0.0, 0.5, 1.0, 10.0],
        weights=events["weights_nominal"].to_numpy()[mask],
    )
    np.testing.assert_allclose(result.hist, expected)

    breakdown = package.uncertainty_breakdown("tau21", bins=[0.0, 0.5, 1.0, 10.0])
    np.testing.assert_allclose(breakdown["nominal"], expected)


def test_real_data_derived_observables(tmp_path):
    input_path = Path("data/multifold.h5")
    if not input_path.exists():
        pytest.skip(f"real data not available: {input_path}")

    # promote to float64 once so ours and the literal port see identical
    # inputs (the file stores float32; our reader always computes in f64)
    df = pd.read_hdf(input_path, "df").iloc[:20000].astype(np.float64)
    ours = compute_derived_observables(df)
    reference = _reference_calculate_vars(df)
    for name in ("tau21", "phi_ll", "dR_ll"):
        np.testing.assert_allclose(
            ours[name].to_numpy(),
            reference[name].to_numpy(),
            equal_nan=True,
            err_msg=name,
        )

    selected = ours[df["pT_trackj1"].to_numpy() > 5]
    assert (np.abs(selected["phi_ll"]) <= np.pi + 1e-9).all()
    assert (selected["dR_ll"] >= 0).all() and np.isfinite(selected["dR_ll"]).all()
    # tau21 stays undefined for single-track jets (tau1 == 0 by definition)
    # even under the jet-pT selection — 0.07% of the real file; every
    # non-finite entry must be exactly a tau1 == 0 event
    non_finite = ~np.isfinite(selected["tau21"].to_numpy())
    assert non_finite.mean() < 0.005
    np.testing.assert_array_equal(
        non_finite, selected["tau1_trackj1"].to_numpy() == 0.0
    )

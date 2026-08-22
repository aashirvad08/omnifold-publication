"""Tests for the covariance/correlation builders (port of fill_cov_matrix)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from omnifold_publication import load_analysis, load_package, write_manifest, write_package
from omnifold_publication.exceptions import PackageReadError
from omnifold_publication.uncertainty import (
    correlation_matrix,
    family_covariance,
    fill_cov_matrix,
)


def _reference_fill_cov_matrix(n, uncerts_list, uncerts_mean=None):
    """Literal port of multifold_util.py:49-58 (the original triple loop)."""

    v = np.zeros((n, n), float)
    for i in range(n):
        for j in range(n):
            for k in range(len(uncerts_list)):
                if uncerts_mean is not None:
                    v[i, j] += (
                        1
                        / (len(uncerts_list) - 1)
                        * (uncerts_list[k][i] - uncerts_mean[i])
                        * (uncerts_list[k][j] - uncerts_mean[j])
                    )
                else:
                    v[i, j] += uncerts_list[k][i] * uncerts_list[k][j]
    return v


def test_hessian_mode_hand_example():
    # two NPs, two bins: d1 = (1, 2), d2 = (3, -1)
    # V_ij = sum_k d_k,i d_k,j:
    #   V_00 = 1 + 9 = 10; V_11 = 4 + 1 = 5; V_01 = 1*2 + 3*(-1) = -1
    deltas = np.array([[1.0, 2.0], [3.0, -1.0]])
    np.testing.assert_allclose(
        fill_cov_matrix(deltas),
        np.array([[10.0, -1.0], [-1.0, 5.0]]),
    )


def test_bootstrap_mode_hand_example():
    # replicas (1,2), (3,6), (2,4): mean (2,4); centered (-1,-2),(1,2),(0,0)
    # V = 1/(3-1) * [[2, 4], [4, 8]] = [[1, 2], [2, 4]]
    replicas = np.array([[1.0, 2.0], [3.0, 6.0], [2.0, 4.0]])
    result = fill_cov_matrix(replicas, replicas.mean(axis=0))
    np.testing.assert_allclose(result, np.array([[1.0, 2.0], [2.0, 4.0]]))
    # and it is exactly numpy's ddof=1 sample covariance
    np.testing.assert_allclose(result, np.cov(replicas.T, ddof=1))


def test_matches_literal_triple_loop_port():
    rng = np.random.default_rng(7)
    hists = rng.normal(10.0, 2.0, size=(5, 4))

    np.testing.assert_allclose(
        fill_cov_matrix(hists),
        _reference_fill_cov_matrix(4, hists),
    )
    np.testing.assert_allclose(
        fill_cov_matrix(hists, hists.mean(axis=0)),
        _reference_fill_cov_matrix(4, hists, hists.mean(axis=0)),
    )


def test_bootstrap_mode_requires_two_replicas():
    with pytest.raises(PackageReadError, match="at least two"):
        fill_cov_matrix(np.array([[1.0, 2.0]]), np.array([1.0, 2.0]))


def test_single_hessian_component_is_fully_correlated():
    # one NP/two-point delta: corr must be +-1 everywhere (nonzero bins)
    delta = np.array([[2.0, -1.0, 3.0]])
    corr = correlation_matrix(fill_cov_matrix(delta))
    np.testing.assert_allclose(np.abs(corr), np.ones((3, 3)))
    assert corr[0, 1] == pytest.approx(-1.0)
    assert corr[0, 2] == pytest.approx(1.0)


def test_correlation_matrix_properties_and_zero_variance_guard():
    cov = np.array(
        [
            [4.0, 1.2, 0.0],
            [1.2, 1.0, 0.0],
            [0.0, 0.0, 0.0],  # empty bin: zero variance
        ]
    )
    corr = correlation_matrix(cov)
    assert corr[0, 0] == 1.0 and corr[1, 1] == 1.0
    assert corr[2, 2] == 1.0  # unit diagonal preserved on the guard path
    assert corr[0, 2] == 0.0 and corr[2, 1] == 0.0
    assert corr[0, 1] == pytest.approx(1.2 / 2.0)
    np.testing.assert_allclose(corr, corr.T)
    assert (np.abs(corr) <= 1.0 + 1e-12).all()


def test_ensemble_covariance_is_sample_cov_over_n():
    hists = np.array([[10.0, 5.0], [12.0, 4.0], [11.0, 6.0], [9.0, 5.0]])
    expected = np.cov(hists.T, ddof=1) / 4.0
    np.testing.assert_allclose(
        family_covariance("median_standard_error", np.zeros(2), hists),
        expected,
    )


def test_paired_covariance_uses_signed_delta():
    nominal = np.array([10.0, 20.0])
    varied = np.array([[12.0, 18.0]])
    reference = np.array([8.0, 24.0])
    # signed delta: (12-8)*10/8 = +5 ; (18-24)*20/24 = -5
    delta = np.array([5.0, -5.0])
    cov = family_covariance(
        "paired_relative_difference", nominal, varied, reference
    )
    np.testing.assert_allclose(cov, np.outer(delta, delta))
    # off-diagonal must be negative: sign information kept (band uses |.|)
    assert cov[0, 1] < 0


def test_package_covariance_diagonals_consistent_with_breakdown(
    tmp_path, atlas_like_hdf
):
    """Known exact relations between sqrt(diag(V_family)) and the band."""

    package_dir = write_package(
        input_path=atlas_like_hdf,
        output_dir=tmp_path / "package",
        include_all_replicas=True,
    )
    package = load_package(package_dir)
    bins = [0.0, 50.0, 100.0, 150.0, 200.0]
    band = package.uncertainty_breakdown("pT_ll", bins=bins)
    cov = package.covariance_matrix("pT_ll", bins=bins)

    np.testing.assert_allclose(cov["nominal"], band["nominal"])
    assert set(cov["components"]) == set(band["components"])

    # Hessian families and sample_stat: sqrt(diag) == band exactly
    for name in (
        "syst_event",
        "syst_theory",
        "syst_track",
        "syst_muon",
        "syst_lumi",
        "syst_background",
        "dd_unfolding",
        "sample_stat",
    ):
        np.testing.assert_allclose(
            np.sqrt(np.diag(cov["components"][name])),
            band["components"][name],
            err_msg=name,
        )

    # bootstrap: cov uses ddof=1, band uses np.std (ddof=0) -> factor K/(K-1)
    k = 3
    np.testing.assert_allclose(
        np.diag(cov["components"]["bootstrap_mc"]),
        band["components"]["bootstrap_mc"] ** 2 * k / (k - 1),
    )

    # ensemble: cov = sample_cov/K (no 1.253); band = 1.253*popstd/sqrt(K)
    k = 4
    np.testing.assert_allclose(
        np.diag(cov["components"]["ensemble"]),
        (band["components"]["ensemble"] / 1.253) ** 2 * k / (k - 1),
    )

    # total is the sum of the components, correlation has unit diagonal
    np.testing.assert_allclose(
        cov["total"], np.sum(list(cov["components"].values()), axis=0)
    )
    np.testing.assert_allclose(np.diag(cov["correlation"]), np.ones(len(bins) - 1))


def test_analysis_covariance_adds_two_point_matrix(tmp_path, atlas_like_hdf):
    nominal_dir = write_package(
        input_path=atlas_like_hdf, output_dir=tmp_path / "nominal"
    )
    alternative_dir = write_package(
        input_path=atlas_like_hdf, output_dir=tmp_path / "alternative"
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
        analysis_name="covariance-test",
    )
    analysis = load_analysis(manifest_dir)
    bins = [0.0, 100.0, 200.0]
    cov = analysis.covariance_matrix("pT_ll", bins=bins)

    assert "two_point_generator_choice" in cov["components"]
    # identical samples: the two-point matrix vanishes
    np.testing.assert_allclose(
        cov["components"]["two_point_generator_choice"],
        np.zeros((2, 2)),
        atol=1e-18,
    )
    without = analysis.covariance_matrix(
        "pT_ll", bins=bins, include_two_point=False
    )
    assert "two_point_generator_choice" not in without["components"]


def test_real_data_covariance_is_symmetric_positive_semidefinite(tmp_path):
    input_path = Path("data/multifold.h5")
    if not input_path.exists():
        pytest.skip(f"real data not available: {input_path}")

    package_dir = write_package(
        input_path=input_path,
        output_dir=tmp_path / "zjets",
        event_count=5000,
        include_all_replicas=True,
    )
    package = load_package(package_dir)
    cov = package.covariance_matrix("pT_ll", bins=6)

    total = cov["total"]
    np.testing.assert_allclose(total, total.T)
    eigenvalues = np.linalg.eigvalsh(total)
    assert eigenvalues.min() >= -1e-9 * max(1.0, eigenvalues.max())
    corr = cov["correlation"]
    np.testing.assert_allclose(np.diag(corr), np.ones(total.shape[0]))
    assert (np.abs(corr) <= 1.0 + 1e-9).all()

"""Tests for the physics closure test (port of 2_pseudo_results cell 26).

The pseudodata files behind the notebook's printed chi2 table are not in
the local dataset, so no printed reference can be reproduced; correctness
is instead established against an independently coded literal
implementation of the cell 26 recipe (fill_cov_matrix triple loop,
scipy-norm kernel smoothing, inv-based chi2) run on the same packages.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import norm

from omnifold_publication import (
    PackageReadError,
    chi2_test,
    load_analysis,
    load_package,
    write_manifest,
    write_package,
)

BINS = [0.0, 50.0, 100.0, 150.0, 200.0]


def _variant_hdf(tmp_path, source, name, nominal_scale):
    """Copy of the source file with weights_nominal scaled per event."""

    df = pd.read_hdf(source, "df").copy()
    df["weights_nominal"] = df["weights_nominal"] * nominal_scale(df)
    path = tmp_path / f"{name}.h5"
    df.to_hdf(path, key="df", mode="w")
    return path


@pytest.fixture
def closure_analysis(tmp_path, atlas_like_hdf):
    """Analysis with nominal, hv/nonDY variations, and a target sample."""

    hv_hdf = _variant_hdf(
        tmp_path, atlas_like_hdf, "hv",
        lambda df: 1.0 + 0.04 * np.sin(df["pT_ll"] / 40.0),
    )
    nondy_hdf = _variant_hdf(
        tmp_path, atlas_like_hdf, "nondy",
        lambda df: 1.0 + 0.02 * (df["pT_ll"] > 100.0),
    )
    target_hdf = _variant_hdf(
        tmp_path, atlas_like_hdf, "target",
        lambda df: np.full(len(df), 1.03),
    )

    nominal_dir = write_package(
        input_path=atlas_like_hdf,
        output_dir=tmp_path / "nominal",
        include_all_replicas=True,
    )
    hv_dir = write_package(input_path=hv_hdf, output_dir=tmp_path / "hv")
    nondy_dir = write_package(
        input_path=nondy_hdf, output_dir=tmp_path / "nondy"
    )
    target_dir = write_package(
        input_path=target_hdf, output_dir=tmp_path / "target"
    )

    manifest_dir = tmp_path / "analysis"
    write_manifest(
        output_dir=manifest_dir,
        nominal_path=nominal_dir,
        target_path=target_dir,
        variations={
            "hv": {"path": hv_dir, "type": "alternative_generator",
                   "combination": "two_point_difference"},
            "nonDY": {"path": nondy_dir, "type": "alternative_sample",
                      "combination": "two_point_difference"},
        },
        analysis_name="closure-test",
    )
    return load_analysis(manifest_dir)


def _literal_cell26(analysis, target_scale, bins, decorrelate_threshold=0.01):
    """Independently coded literal implementation of cell 26.

    Loads events straight from the packages and reproduces the notebook
    formulas (triple-loop fill_cov_matrix, scipy-norm smoothing kernel,
    inv-based chi2) with no calls into omnifold_publication.
    """

    bins = np.asarray(bins, dtype=float)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    n = len(bins) - 1

    def fill_cov_matrix(n, uncerts_list, uncerts_mean=None):
        v = np.zeros((n, n), float)
        for i in range(n):
            for j in range(n):
                for k in range(len(uncerts_list)):
                    if uncerts_mean is not None:
                        v[i, j] += (
                            1 / (len(uncerts_list) - 1)
                            * (uncerts_list[k][i] - uncerts_mean[i])
                            * (uncerts_list[k][j] - uncerts_mean[j])
                        )
                    else:
                        v[i, j] += uncerts_list[k][i] * uncerts_list[k][j]
        return v

    def smooth_uncertainty(uncert, bin_centers):
        n_sig = 10
        xrange = bin_centers[-1] - bin_centers[0]
        log_scale = bin_centers[0] > 0
        if log_scale:
            xrange = np.log(bin_centers[-1]) - np.log(bin_centers[0])
        kernel_width = xrange / n_sig
        smooth = uncert.copy()
        for i in range(len(bin_centers)):
            x_i = np.log(bin_centers[i]) if log_scale else bin_centers[i]
            sumw = sumwy = 0.0
            for j in range(len(bin_centers)):
                x_j = np.log(bin_centers[j]) if log_scale else bin_centers[j]
                w = norm.pdf((x_j - x_i) / kernel_width)
                sumw += w
                sumwy += w * uncert[j]
            smooth[i] = sumwy / sumw
        return smooth

    df = analysis.nominal_package.load_events()
    # fixture manifests store absolute package paths
    df_hv = load_package(analysis.manifest["samples"]["hv"]["path"]).load_events()
    df_nondy = load_package(
        analysis.manifest["samples"]["nonDY"]["path"]
    ).load_events()
    df_target = load_package(
        analysis.manifest["samples"]["target"]["path"]
    ).load_events()

    var = "pT_ll"
    target_counts, _ = np.histogram(
        df_target[var], bins=bins, weights=df_target["weights_nominal"]
    )
    nom, _ = np.histogram(df[var], bins=bins, weights=df["weights_nominal"])
    mc_stat_cov, _ = np.histogram(
        df[var], bins=bins, weights=df["weights_nominal"] ** 2
    )

    def hists(columns):
        return [
            np.histogram(df[var], bins=bins, weights=df[c])[0]
            for c in columns
        ]

    mc_stat_bs = hists([f"weights_bootstrap_mc_{i}" for i in range(3)])
    data_stat_bs = hists([f"weights_bootstrap_data_{i}" for i in range(3)])
    ensembles = hists([f"weights_ensemble_{i}" for i in range(4)])
    theory_bin_uncerts = [
        h - nom for h in hists(["weights_theoryPDF", "weights_theoryQCD"])
    ]

    counts_dd, _ = np.histogram(df[var], bins=bins, weights=df["weights_dd"])
    target_dd, _ = np.histogram(df[var], bins=bins, weights=df["target_dd"])
    unfolding_dd = [(counts_dd - target_dd) * (nom / target_dd)]

    hv_counts, _ = np.histogram(
        df_hv[var], bins=bins, weights=df_hv["weights_nominal"]
    )
    unfolding_hv = [smooth_uncertainty(hv_counts - nom, bin_centers)]

    bkg_bin_uncerts = [
        hists(["weights_topBackground"])[0] - nom,
        np.histogram(
            df_nondy[var], bins=bins, weights=df_nondy["weights_nominal"]
        )[0]
        - nom,
    ]

    v_mc = np.zeros((n, n), float)
    np.fill_diagonal(v_mc, mc_stat_cov)
    v_bs_mc = fill_cov_matrix(n, mc_stat_bs, np.mean(mc_stat_bs, axis=0))
    v_bs_data = fill_cov_matrix(n, data_stat_bs, np.mean(data_stat_bs, axis=0))
    v_nn = fill_cov_matrix(n, ensembles, np.mean(ensembles, axis=0)) / len(
        ensembles
    )
    v_theory = fill_cov_matrix(n, theory_bin_uncerts)
    v_bkg = fill_cov_matrix(n, bkg_bin_uncerts)
    v_unfolding_dd = fill_cov_matrix(n, unfolding_dd)
    v_unfolding_hv = fill_cov_matrix(n, unfolding_hv)

    v_total = (
        v_mc + v_bs_mc + v_bs_data + v_nn + v_theory
        + v_unfolding_dd + v_unfolding_hv + v_bkg
    )

    dof = len(bins) - 1
    D = nom - target_counts
    chi_2 = D.dot(np.linalg.inv(v_total)).dot(D.T)
    p_value = 1 - stats.chi2.cdf(chi_2, dof)
    decorrelated = False

    if p_value < decorrelate_threshold:
        v_hv_dec = np.diag(np.diag(v_unfolding_hv))
        v_dd_dec = np.diag(np.diag(v_unfolding_dd))
        v_total_dec = (
            v_mc + v_bs_mc + v_bs_data + v_nn + v_theory
            + v_dd_dec + v_hv_dec + v_bkg
        )
        chi_2 = D.dot(np.linalg.inv(v_total_dec)).dot(D.T)
        p_value = 1 - stats.chi2.cdf(chi_2, dof)
        decorrelated = True

    return {"chi2": float(chi_2), "p_value": float(p_value),
            "dof": dof, "decorrelated": decorrelated}


def test_release_exact_matches_literal_cell26(closure_analysis):
    result = chi2_test(
        closure_analysis,
        observable="pT_ll",
        bins=BINS,
        mode="release-exact",
        smooth_two_point=["hv"],
        exclude_components=[],  # literal cell 26 keeps v_unfolding_dd
    )
    reference = _literal_cell26(closure_analysis, 1.03, BINS)

    assert result["dof"] == reference["dof"] == len(BINS) - 1
    assert result["decorrelated"] == reference["decorrelated"]
    np.testing.assert_allclose(result["chi2"], reference["chi2"])
    np.testing.assert_allclose(result["p_value"], reference["p_value"])


def test_decorrelation_fallback_matches_literal_cell26(
    tmp_path, closure_analysis
):
    """A grossly wrong target must trigger the p < 0.01 fallback in both."""

    # a huge threshold forces the fallback branch deterministically in both
    # implementations regardless of the initial p-value
    result = chi2_test(
        closure_analysis,
        observable="pT_ll",
        bins=BINS,
        mode="release-exact",
        smooth_two_point=["hv"],
        decorrelate_threshold=1.1,
        exclude_components=[],  # literal cell 26 keeps v_unfolding_dd
    )
    reference = _literal_cell26(
        closure_analysis, 1.03, BINS, decorrelate_threshold=1.1
    )

    assert result["decorrelated"] and reference["decorrelated"]
    assert result["decorrelated_components"] == [
        "dd_unfolding",
        "two_point_hv",
    ]
    assert "initial_chi2" in result and "initial_p_value" in result
    np.testing.assert_allclose(result["chi2"], reference["chi2"])
    np.testing.assert_allclose(result["p_value"], reference["p_value"])
    # decorrelation changes the statistic
    assert result["chi2"] != pytest.approx(result["initial_chi2"])


def test_full_mode_includes_detector_families(closure_analysis):
    full = chi2_test(
        closure_analysis, observable="pT_ll", bins=BINS, mode="full"
    )
    exact = chi2_test(
        closure_analysis, observable="pT_ll", bins=BINS, mode="release-exact"
    )

    assert {"syst_event", "syst_track", "syst_muon", "syst_lumi"} <= set(
        full["components"]
    )
    assert (
        set(exact["components"])
        & {"syst_event", "syst_track", "syst_muon", "syst_lumi"}
        == set()
    )
    assert {"two_point_hv", "two_point_nonDY", "syst_background"} <= set(
        exact["components"]
    )
    # more covariance -> smaller chi2 in full mode for the same difference
    assert full["chi2"] < exact["chi2"]


def test_manifest_target_role_is_used(closure_analysis):
    assert closure_analysis.target_package is not None

    implicit = chi2_test(closure_analysis, observable="pT_ll", bins=BINS)
    explicit = chi2_test(
        closure_analysis,
        target_package=closure_analysis.target_package,
        observable="pT_ll",
        bins=BINS,
    )
    np.testing.assert_allclose(implicit["chi2"], explicit["chi2"])
    # ~3% flat target offset must be visible in the difference
    assert np.all(implicit["target"] > implicit["result"])


def test_in_package_dd_pair_closure(tmp_path, atlas_like_hdf):
    package = load_package(
        write_package(
            input_path=atlas_like_hdf,
            output_dir=tmp_path / "package",
            include_all_replicas=True,
        )
    )
    result = chi2_test(
        package,
        observable="pT_ll",
        bins=BINS,
        variation="weights_dd",
        target_variation="target_dd",
    )

    events = package.load_events()
    expected_diff = (
        np.histogram(events["pT_ll"], bins=BINS,
                     weights=events["weights_dd"])[0]
        - np.histogram(events["pT_ll"], bins=BINS,
                       weights=events["target_dd"])[0]
    )
    np.testing.assert_allclose(result["difference"], expected_diff)
    assert 0.0 <= result["p_value"] <= 1.0
    # the circular dd component must be excluded from the test covariance
    assert result["excluded_components"] == ["dd_unfolding"]
    assert "dd_unfolding" not in result["components"]


def test_self_target_same_variation_raises(tmp_path, atlas_like_hdf):
    package = load_package(
        write_package(input_path=atlas_like_hdf, output_dir=tmp_path / "p")
    )
    with pytest.raises(PackageReadError, match="same"):
        chi2_test(package, observable="pT_ll", bins=BINS)


def test_derived_observable_requires_explicit_bins(tmp_path, atlas_like_hdf):
    package = load_package(
        write_package(
            input_path=atlas_like_hdf,
            output_dir=tmp_path / "package",
            observables=["pT_ll", "tau21"],
        )
    )
    with pytest.raises(PackageReadError, match="No official binning"):
        chi2_test(
            package,
            observable="tau21",
            variation="weights_dd",
            target_variation="target_dd",
        )


def test_invalid_inputs_raise(closure_analysis, tmp_path, atlas_like_hdf):
    with pytest.raises(PackageReadError, match="Unknown chi2_test mode"):
        chi2_test(closure_analysis, observable="pT_ll", bins=BINS,
                  mode="banana")
    with pytest.raises(PackageReadError, match="not declared as variations"):
        chi2_test(closure_analysis, observable="pT_ll", bins=BINS,
                  smooth_two_point=["nope"])

    package = load_package(
        write_package(input_path=atlas_like_hdf, output_dir=tmp_path / "p2")
    )
    with pytest.raises(PackageReadError, match="requires an OmniFoldAnalysis"):
        chi2_test(package, observable="pT_ll", bins=BINS,
                  variation="weights_dd", target_variation="target_dd",
                  smooth_two_point=["hv"])


def test_circularity_guard_rejects_injected_wrong_target(
    tmp_path, atlas_like_hdf
):
    """Regression test for the circular-covariance problem.

    A target_dd deliberately shifted by +30% must be rejected. Without
    the guard (exclude_components=[]), the dd_unfolding component absorbs
    the injected discrepancy and chi2 stays bounded near 1
    (Sherman-Morrison, docs/closure_test_design.md) — the legacy branch
    is asserted too so the failure mode itself stays documented.
    """

    from conftest import make_atlas_like_frame

    # higher statistics than the shared fixture so the injected shift is
    # well outside the statistical uncertainty (~5% per bin at n=2000)
    df = make_atlas_like_frame(n=2000, seed=7)
    df["target_dd"] = df["target_dd"] * 1.3
    injected = tmp_path / "injected.h5"
    df.to_hdf(injected, key="df", mode="w")

    package = load_package(
        write_package(
            input_path=injected,
            output_dir=tmp_path / "package",
            include_all_replicas=True,
        )
    )

    fixed = chi2_test(
        package,
        observable="pT_ll",
        bins=BINS,
        variation="weights_dd",
        target_variation="target_dd",
    )
    assert fixed["excluded_components"] == ["dd_unfolding"]
    assert fixed["p_value"] < 0.01  # the wrong target is rejected

    legacy = chi2_test(
        package,
        observable="pT_ll",
        bins=BINS,
        variation="weights_dd",
        target_variation="target_dd",
        exclude_components=[],
    )
    # circular covariance: chi2 bounded near (target scale)^2, test blind
    assert legacy["chi2"] < 2.0
    assert legacy["p_value"] > 0.5


def test_two_point_component_of_target_sample_is_excluded(closure_analysis):
    """Using a variation sample as the target must drop its two-point term."""

    hv_package = load_package(
        closure_analysis.manifest["samples"]["hv"]["path"]
    )
    result = chi2_test(
        closure_analysis,
        target_package=hv_package,
        observable="pT_ll",
        bins=BINS,
    )
    assert "two_point_hv" in result["excluded_components"]
    assert "two_point_hv" not in result["components"]
    assert "two_point_nonDY" in result["components"]


def test_p_value_consistent_with_scipy(closure_analysis):
    result = chi2_test(closure_analysis, observable="pT_ll", bins=BINS)
    assert result["p_value"] == pytest.approx(
        1.0 - stats.chi2.cdf(result["chi2"], result["dof"])
    )

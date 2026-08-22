"""Histogram-level uncertainty recipes from the ATLAS Z+jets OmniFold release.

Each combination rule reproduces the corresponding recipe in the release
code (``multifold_util.py`` and the release notebooks):

- ``quadrature_difference_from_nominal``: per-NP differences to the nominal
  histogram summed in quadrature (``calculate_uncertainty``,
  multifold_util.py:61-68).
- ``standard_deviation``: spread across bootstrap replica histograms
  (``calculate_stat_uncertainty``, multifold_util.py:71-82).
- ``median_standard_error``: ensemble/NN-initialization spread scaled by
  1.253 (standard error of the median) and divided by sqrt(N_replicas)
  (multifold_util.py:78-82 plus the /sqrt(N) at the call site in
  2_pseudo_results.ipynb cell 11).
- ``paired_relative_difference``: data-driven unfolding uncertainty from
  the (weights_dd, target_dd) pair, (h_var - h_ref) * h_nom / h_ref
  (multifold_util.py, "Unfolding (DD)" block).

See spec/uncertainty_recipes.md for the full evidence trail.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .exceptions import PackageReadError


MEDIAN_STD_ERROR_FACTOR = 1.253


def histogram_matrix(
    values: np.ndarray,
    weight_matrix: np.ndarray,
    bins: np.ndarray,
) -> np.ndarray:
    """Histogram one observable under many weight columns.

    ``weight_matrix`` has shape (n_variations, n_events); the result has
    shape (n_variations, n_bins).
    """

    return np.vstack(
        [np.histogram(values, bins=bins, weights=row)[0] for row in weight_matrix]
    )


def quadrature_difference_from_nominal(
    nominal_hist: np.ndarray,
    variation_hists: np.ndarray,
) -> np.ndarray:
    """Quadrature sum of per-variation differences to the nominal histogram."""

    deltas = np.asarray(variation_hists, dtype=float) - np.asarray(
        nominal_hist, dtype=float
    )
    return np.sqrt((deltas**2).sum(axis=0))


def bootstrap_standard_deviation(variation_hists: np.ndarray) -> np.ndarray:
    """Per-bin standard deviation across bootstrap replica histograms."""

    return np.std(np.asarray(variation_hists, dtype=float), axis=0)


def ensemble_median_standard_error(variation_hists: np.ndarray) -> np.ndarray:
    """Ensemble (NN initialization) uncertainty: 1.253 * std / sqrt(N)."""

    hists = np.asarray(variation_hists, dtype=float)
    return (
        MEDIAN_STD_ERROR_FACTOR
        * np.std(hists, axis=0)
        / np.sqrt(hists.shape[0])
    )


def paired_relative_difference(
    nominal_hist: np.ndarray,
    varied_hist: np.ndarray,
    reference_hist: np.ndarray,
) -> np.ndarray:
    """Paired-variation uncertainty: |h_var - h_ref| * h_nom / h_ref.

    Bins where the reference histogram is zero contribute zero uncertainty.
    """

    nominal = np.asarray(nominal_hist, dtype=float)
    varied = np.asarray(varied_hist, dtype=float)
    reference = np.asarray(reference_hist, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(
            reference != 0.0,
            (varied - reference) * nominal / reference,
            0.0,
        )
    return np.abs(delta)


def combine_family(
    combination: str,
    nominal_hist: np.ndarray,
    variation_hists: np.ndarray,
    reference_hist: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a named combination recipe to a family of variation histograms."""

    if combination == "quadrature_difference_from_nominal":
        return quadrature_difference_from_nominal(nominal_hist, variation_hists)
    if combination == "standard_deviation":
        return bootstrap_standard_deviation(variation_hists)
    if combination == "median_standard_error":
        return ensemble_median_standard_error(variation_hists)
    if combination == "paired_relative_difference":
        if reference_hist is None:
            raise PackageReadError(
                "Combination 'paired_relative_difference' requires a "
                "reference histogram."
            )
        if len(variation_hists) != 1:
            raise PackageReadError(
                "Combination 'paired_relative_difference' expects exactly "
                "one variation column."
            )
        return paired_relative_difference(
            nominal_hist, variation_hists[0], reference_hist
        )
    raise PackageReadError(f"Unknown weight-family combination: {combination!r}")


def total_in_quadrature(components: dict[str, np.ndarray]) -> np.ndarray:
    """Combine independent uncertainty components in quadrature."""

    stacked = np.vstack([np.asarray(c, dtype=float) for c in components.values()])
    return np.sqrt((stacked**2).sum(axis=0))


def smooth_uncertainty(
    uncert: np.ndarray,
    bin_centers: np.ndarray,
) -> np.ndarray:
    """Gaussian-kernel smoothing of a binned uncertainty amplitude.

    Port of ``smooth_uncertainty`` (multifold_util.py:621-644), applied by
    the release to the hidden-variable (Sherpa) two-point delta before it
    enters the closure covariance (2_pseudo_results.ipynb cell 26) and in
    ``corr_matrix`` when ``smooth=True``. The kernel width is 1/10 of the
    bin-center range; when all bin centers are positive the smoothing runs
    in log space. The Gaussian normalization cancels in the weighted mean,
    so results are identical to the original ``norm.pdf`` loop.
    """

    uncert = np.asarray(uncert, dtype=float)
    centers = np.asarray(bin_centers, dtype=float)
    n_sig = 10
    log_scale = centers[0] > 0
    x = np.log(centers) if log_scale else centers
    x_range = x[-1] - x[0]
    kernel_width = x_range / n_sig

    z = (x[np.newaxis, :] - x[:, np.newaxis]) / kernel_width
    kernel = np.exp(-0.5 * z**2)
    return kernel @ uncert / kernel.sum(axis=1)


def fill_cov_matrix(
    uncerts_list: np.ndarray,
    uncerts_mean: np.ndarray | None = None,
) -> np.ndarray:
    """Port of ``fill_cov_matrix`` (multifold_util.py:49-58), vectorized.

    Two modes, selected exactly as in the release code:

    - **Hessian mode** (``uncerts_mean is None``): the inputs are per-NP
      difference vectors (h_k - h_nom); each NP is fully correlated across
      bins and NPs are uncorrelated with each other:
      V_ij = sum_k d_k,i * d_k,j.
    - **Bootstrap mode** (``uncerts_mean`` given): the inputs are replica
      histograms; V is the sample covariance with 1/(K-1):
      V_ij = 1/(K-1) * sum_k (x_k,i - mean_i) * (x_k,j - mean_j).

    The original takes the dimension ``n`` as its first argument; it is
    inferred here. Results are identical to the release's triple loop.
    """

    arr = np.asarray(uncerts_list, dtype=float)
    if arr.ndim != 2:
        raise PackageReadError(
            "fill_cov_matrix expects a 2D (n_variations, n_bins) input."
        )
    if uncerts_mean is None:
        return arr.T @ arr
    if arr.shape[0] < 2:
        raise PackageReadError(
            "Bootstrap covariance requires at least two replicas."
        )
    centered = arr - np.asarray(uncerts_mean, dtype=float)
    return centered.T @ centered / (arr.shape[0] - 1)


def correlation_matrix(covariance: np.ndarray) -> np.ndarray:
    """Correlation from covariance: corr_ij = V_ij / (sigma_i * sigma_j).

    The derivation at the end of ``corr_matrix`` (multifold_util.py,
    sigmas = sqrt(diag(v_total)) loop). Bins with zero variance — where the
    release code would divide by zero — are set to zero correlation (unit
    diagonal preserved).
    """

    cov = np.asarray(covariance, dtype=float)
    sigmas = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(sigmas, sigmas)
    corr[~np.isfinite(corr)] = 0.0
    np.fill_diagonal(corr, np.where(sigmas > 0.0, np.diag(corr), 1.0))
    return corr


def family_covariance(
    combination: str,
    nominal_hist: np.ndarray,
    variation_hists: np.ndarray,
    reference_hist: np.ndarray | None = None,
) -> np.ndarray:
    """Per-family covariance matrix, mode chosen as in corr_matrix.

    - systematic families (``quadrature_difference_from_nominal``):
      Hessian mode on the (h_k - h_nom) differences (v_theory, v_lumi).
    - bootstrap families (``standard_deviation``): bootstrap mode on the
      replica histograms (v_bs_mc, v_bs_data).
    - ensemble (``median_standard_error``): bootstrap mode divided by the
      replica count (v_nn = fill_cov_matrix(...)/100). Note the release
      applies the 1.253 median factor in the band but NOT in the
      covariance; this inconsistency is ported faithfully, so
      sqrt(diag) here is the ensemble band without the 1.253 factor
      (and with ddof=1 instead of the band's ddof=0).
    - paired (``paired_relative_difference``): Hessian mode on the single
      signed delta (h_var - h_ref) * h_nom / h_ref (v_unfolding_dd).
    """

    nominal = np.asarray(nominal_hist, dtype=float)
    hists = np.asarray(variation_hists, dtype=float)

    if combination == "quadrature_difference_from_nominal":
        return fill_cov_matrix(hists - nominal)
    if combination == "standard_deviation":
        return fill_cov_matrix(hists, np.mean(hists, axis=0))
    if combination == "median_standard_error":
        return fill_cov_matrix(hists, np.mean(hists, axis=0)) / hists.shape[0]
    if combination == "paired_relative_difference":
        if reference_hist is None:
            raise PackageReadError(
                "Combination 'paired_relative_difference' requires a "
                "reference histogram."
            )
        if len(hists) != 1:
            raise PackageReadError(
                "Combination 'paired_relative_difference' expects exactly "
                "one variation column."
            )
        reference = np.asarray(reference_hist, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = np.where(
                reference != 0.0,
                (hists[0] - reference) * nominal / reference,
                0.0,
            )
        return fill_cov_matrix(delta[np.newaxis, :])
    raise PackageReadError(f"Unknown weight-family combination: {combination!r}")


def covariance_breakdown(
    package: Any,
    observable: str,
    bins: list[float] | int | None = None,
) -> dict[str, Any]:
    """Per-family covariance matrices and their total, plus correlation.

    Follows the assembly in ``corr_matrix`` (multifold_util.py:402-427):
    a diagonal sample-statistics matrix (v_mc, diag of hist(w_nominal**2)),
    one matrix per family in its mode, summed into the total, with the
    correlation matrix derived from the total. One deliberate
    generalization: the release's ``corr_matrix`` omits the detector NP
    groups (event/track/muon) from its total even though its uncertainty
    bands include them; here every packaged family contributes, which is
    consistent with the band treatment. Reproduce the release selection by
    summing a subset of ``components`` if needed.
    """

    values, mask, edges, nominal_hist, stat_variance = _nominal_context(
        package, observable, bins
    )

    components: dict[str, np.ndarray] = {"sample_stat": np.diag(stat_variance)}
    for name in package.list_weight_families():
        combination, variation_hists, reference_hist = _family_inputs(
            package, name, values, mask, edges
        )
        components[name] = family_covariance(
            combination,
            nominal_hist=nominal_hist,
            variation_hists=variation_hists,
            reference_hist=reference_hist,
        )

    total = np.sum(list(components.values()), axis=0)
    return {
        "edges": edges,
        "nominal": nominal_hist,
        "components": components,
        "total": total,
        "correlation": correlation_matrix(total),
    }


def _nominal_context(
    package: Any,
    observable: str,
    bins: list[float] | int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load observable values and build the nominal histogram context.

    Applies the observable's declared selection (e.g. the pT > 5 GeV
    trackjet masks of multifold_util.py corr_matrix). Returns
    (values, mask, edges, nominal_hist, sample_stat_variance) where the
    last entry is the per-bin sum of squared nominal weights — the
    evaluation-sample statistical variance (the hist(weights_nominal**2)
    line in 3_results.ipynb / multifold_util.py corr_matrix).
    """

    selected_bins = bins
    if selected_bins is None:
        selected_bins = package.observable_bins(observable) or 30
    values, mask = package.observable_values(observable)
    nominal_weights = np.asarray(package.get_weights("nominal"), dtype=float)[
        mask
    ]

    nominal_hist, edges = np.histogram(
        values, bins=selected_bins, weights=nominal_weights
    )
    sample_stat_variance = np.histogram(
        values, bins=edges, weights=nominal_weights**2
    )[0]
    return values, mask, edges, nominal_hist.astype(float), sample_stat_variance


def _family_inputs(
    package: Any,
    name: str,
    values: np.ndarray,
    mask: np.ndarray,
    edges: np.ndarray,
) -> tuple[str, np.ndarray, np.ndarray | None]:
    """Histogram a declared family: (combination, variation_hists, reference)."""

    family = package.weight_family(name)
    variation_hists = histogram_matrix(
        values, package.get_family_weights(name)[:, mask], edges
    )
    reference_hist = None
    reference_column = family.get("reference_column")
    if isinstance(reference_column, str):
        reference_weights = np.asarray(
            package.get_weights(reference_column), dtype=float
        )[mask]
        reference_hist = np.histogram(
            values, bins=edges, weights=reference_weights
        )[0]
    return family.get("combination", ""), variation_hists, reference_hist


def uncertainty_breakdown(
    package: Any,
    observable: str,
    bins: list[float] | int | None = None,
) -> dict[str, Any]:
    """Per-family uncertainty breakdown for one observable of a package.

    Mirrors the per-observable assembly in 2_pseudo_results.ipynb cell 11:
    the nominal histogram, the evaluation-sample statistical uncertainty
    sqrt(sum w^2) ("sample_stat", cf. the hist(w_nominal**2) covariance in
    3_results.ipynb), one component per declared weight family, and the
    quadrature total.
    """

    values, mask, edges, nominal_hist, stat_variance = _nominal_context(
        package, observable, bins
    )

    components: dict[str, np.ndarray] = {"sample_stat": np.sqrt(stat_variance)}
    for name in package.list_weight_families():
        combination, variation_hists, reference_hist = _family_inputs(
            package, name, values, mask, edges
        )
        components[name] = combine_family(
            combination,
            nominal_hist=nominal_hist,
            variation_hists=variation_hists,
            reference_hist=reference_hist,
        )

    return {
        "edges": edges,
        "nominal": nominal_hist,
        "components": components,
        "total": total_in_quadrature(components),
    }

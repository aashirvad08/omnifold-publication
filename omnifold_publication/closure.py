"""Physics closure test: chi2 of a result against a known target.

Port of the pseudo-data closure recipe in 2_pseudo_results.ipynb cell 26:

    D = h_result - h_target
    chi2 = D . V^-1 . D,   dof = n_bins,   p = 1 - chi2_cdf(chi2, dof)

with the covariance V assembled from the Stage 4 builders and, when the
initial p-value falls below 0.01, the release's decorrelation fallback:
cell 26 reduces exactly v_unfolding_dd and v_unfolding_hv to their
diagonals and repeats the test — here the dd_unfolding component plus the
two-points named in ``smooth_two_point`` (the HV-type unfolding terms),
overridable via ``decorrelate_components``. The non-DY two-point stays
fully correlated, as in the release.

Covariance modes:

- ``mode="full"`` (default): every packaged component. Physically
  preferred for actual closure testing — omitting real detector
  systematics deflates the uncertainty and inflates chi2.
- ``mode="release-exact"``: exactly the components cell 26 sums:
  v_mc (sample_stat), v_bs_mc (bootstrap_mc), v_bs_data (bootstrap_data),
  v_nn (ensemble), v_theory (syst_theory), v_unfolding_dd (dd_unfolding),
  v_bkg (syst_background i.e. topBackground, plus the non-DY two-point)
  and the smoothed HV two-point. NOTE: this list was re-derived from the
  cell 26 source, not from ``corr_matrix`` — cell 26 *excludes* v_lumi,
  which ``corr_matrix`` includes, in addition to excluding the
  syst_event/syst_track/syst_muon detector groups. Two-point components
  require an OmniFoldAnalysis with variation samples; on a plain package
  the mode uses the package-level subset only.

**Circularity guard**: by default the internal covariance excludes the
``dd_unfolding`` component and any two-point component built from the
sample serving as the target. Those components are constructed from the
very discrepancy a closure test measures; keeping them bounds chi2 near 1
regardless of how wrong the target is (Sherman-Morrison:
chi2 = q/(1+q) with q = D V_other^-1 D), so the test could never fail.
This applies ONLY here — dd_unfolding remains a legitimate systematic in
uncertainty_breakdown()/covariance_matrix() for published measurements.
Pass ``exclude_components=[]`` to reproduce the literal cell 26 recipe.
Full derivation and rationale: docs/closure_test_design.md.

No printed chi2/p-value reference from the release notebook could be
reproduced numerically: cell 26's saved table derives from the pseudodata
files (files/pseudodata/*.h5), which are not part of the local dataset.
The implementation is instead verified against an independently coded
literal implementation of the cell 26 recipe (tests/test_closure.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from .analysis import OmniFoldAnalysis
from .exceptions import PackageReadError
from .reader import OmniFoldPackage


# Components of cell 26's v_total, in package-family terms (see module
# docstring; two_point_* components are additionally always included).
RELEASE_EXACT_COMPONENTS = (
    "sample_stat",      # v_mc
    "bootstrap_mc",     # v_bs_mc
    "bootstrap_data",   # v_bs_data
    "ensemble",         # v_nn
    "syst_theory",      # v_theory
    "dd_unfolding",     # v_unfolding_dd
    "syst_background",  # topBackground half of v_bkg
)
CHI2_MODES = ("full", "release-exact")


def _resolve_bins(
    package: OmniFoldPackage,
    observable: str,
    bins: list[float] | None,
) -> list[float]:
    """Explicit bins, or the observable's declared/official binning.

    Never falls back to suggested_bins silently: closure statistics are
    only meaningful on a validated binning, so observables without an
    official binning (e.g. the derived tau21/dR_ll) must be given bins
    explicitly.
    """

    if bins is not None:
        return [float(edge) for edge in bins]
    for entry in package.metadata().get("observables", []):
        if isinstance(entry, dict) and entry.get("name") == observable:
            declared = entry.get("bins")
            if declared is None:
                binning = entry.get("binning")
                if isinstance(binning, dict):
                    declared = binning.get("official")
            if isinstance(declared, list):
                return [float(edge) for edge in declared]
            suffix = (
                " (suggested_bins exist but are not an official binning "
                "and are not used silently for closure)"
                if entry.get("suggested_bins")
                else ""
            )
            raise PackageReadError(
                f"No official binning declared for {observable!r}; pass "
                f"bins= explicitly{suffix}."
            )
    raise PackageReadError(f"Unknown observable: {observable}")


def _weighted_histogram(
    package: OmniFoldPackage,
    observable: str,
    variation: str,
    edges: np.ndarray,
) -> np.ndarray:
    values, mask = package.observable_values(observable)
    weights = np.asarray(package.get_weights(variation), dtype=float)[mask]
    return np.histogram(values, bins=edges, weights=weights)[0]


def _select_components(
    components: dict[str, np.ndarray],
    mode: str,
) -> dict[str, np.ndarray]:
    if mode == "full":
        return dict(components)
    selected = {
        name: matrix
        for name, matrix in components.items()
        if name in RELEASE_EXACT_COMPONENTS or name.startswith("two_point_")
    }
    if not selected:
        raise PackageReadError(
            "release-exact mode selected no covariance components; the "
            "package declares none of the cell 26 families."
        )
    return selected


def _circular_components(
    components: dict[str, np.ndarray],
    analysis: OmniFoldAnalysis | None,
    resolved_target: OmniFoldPackage,
) -> list[str]:
    """Components whose construction can absorb the tested discrepancy.

    See docs/closure_test_design.md for the full Sherman-Morrison
    argument. Excluded by default from chi2_test's internal covariance:

    - ``dd_unfolding``: built from the (weights_dd - target_dd)
      discrepancy vector — when the tested pair involves those columns
      (or a target derived from them) the covariance grows as ~D*D^T and
      chi2 is bounded near 1 no matter how wrong the target is. Since
      chi2_test cannot know an arbitrary target's provenance, the
      component is categorically excluded here (and only here).
    - ``two_point_<name>`` where the variation sample IS the resolved
      target package: the two-point delta is then exactly -D.

    ``syst_background`` (topBackground) is NOT circular: its delta
    h_top - h_nominal never involves the target.
    """

    circular = [name for name in components if name == "dd_unfolding"]
    if analysis is not None:
        target_dir = Path(resolved_target.package_dir).resolve()
        for name in components:
            if not name.startswith("two_point_"):
                continue
            sample = analysis._packages.get(name.removeprefix("two_point_"))
            if sample is not None and (
                Path(sample.package_dir).resolve() == target_dir
            ):
                circular.append(name)
    return circular


def _decorrelate(
    components: dict[str, np.ndarray],
    names: set[str],
) -> dict[str, np.ndarray]:
    """Reduce the named components to their diagonals.

    Cell 26's fallback decorrelates exactly v_unfolding_hv and
    v_unfolding_dd — the two unfolding two-point systematics whose
    bin-to-bin correlation is unknown (README recommendation 5). The
    non-DY two-point (inside v_bkg) stays fully correlated.
    """

    return {
        name: np.diag(np.diag(matrix)) if name in names else matrix
        for name, matrix in components.items()
    }


def _chi2(difference: np.ndarray, covariance: np.ndarray) -> float:
    try:
        return float(difference @ np.linalg.solve(covariance, difference))
    except np.linalg.LinAlgError as exc:
        raise PackageReadError(
            "Closure covariance matrix is singular (empty bins or no "
            "uncertainty components cover a bin); adjust the binning."
        ) from exc


def chi2_test(
    package: OmniFoldPackage | OmniFoldAnalysis,
    target_package: OmniFoldPackage | None = None,
    observable: str = "pT_ll",
    bins: list[float] | None = None,
    mode: str = "full",
    variation: str = "nominal",
    target_variation: str = "nominal",
    decorrelate_threshold: float = 0.01,
    smooth_two_point: list[str] | None = None,
    decorrelate_components: list[str] | None = None,
    exclude_components: list[str] | None = None,
) -> dict[str, Any]:
    """Closure chi2 of a result histogram against a target histogram.

    ``package`` may be a single package or an OmniFoldAnalysis (whose
    covariance then includes two-point terms; name variations in
    ``smooth_two_point`` to kernel-smooth their deltas as cell 26 does for
    the hidden-variable term). The target resolves to, in order:
    ``target_package``, the analysis's declared role-"target" sample, or
    the result package itself (for in-package pairs such as
    weights_dd vs target_dd via ``variation``/``target_variation``).

    The covariance is always built around the package's nominal result,
    matching cell 26 (which tests weights_nominal); ``variation`` selects
    the histogram being tested.
    """

    if mode not in CHI2_MODES:
        allowed = ", ".join(CHI2_MODES)
        raise PackageReadError(
            f"Unknown chi2_test mode {mode!r}; expected one of: {allowed}."
        )

    if isinstance(package, OmniFoldAnalysis):
        analysis: OmniFoldAnalysis | None = package
        result_package = package.nominal_package
    else:
        analysis = None
        result_package = package

    resolved_target = target_package
    if resolved_target is None and analysis is not None:
        resolved_target = analysis.target_package
    if resolved_target is None:
        resolved_target = result_package
    if resolved_target is result_package and variation == target_variation:
        raise PackageReadError(
            "Target resolves to the result package with the same "
            "variation; nothing to test. Provide a target package or "
            "distinct target_variation."
        )

    edges = np.asarray(
        _resolve_bins(result_package, observable, bins), dtype=float
    )

    if analysis is not None:
        covariance = analysis.covariance_matrix(
            observable, bins=list(edges), smooth_two_point=smooth_two_point
        )
    else:
        if smooth_two_point:
            raise PackageReadError(
                "smooth_two_point requires an OmniFoldAnalysis with "
                "variation samples."
            )
        covariance = result_package.covariance_matrix(
            observable, bins=list(edges)
        )

    result_hist = _weighted_histogram(
        result_package, observable, variation, edges
    )
    target_hist = _weighted_histogram(
        resolved_target, observable, target_variation, edges
    )
    difference = result_hist - target_hist

    components = _select_components(covariance["components"], mode)
    if exclude_components is None:
        exclude_components = _circular_components(
            components,
            analysis=analysis,
            resolved_target=resolved_target,
        )
    excluded = sorted(set(exclude_components) & set(components))
    components = {
        name: matrix
        for name, matrix in components.items()
        if name not in excluded
    }
    if not components:
        raise PackageReadError(
            "No covariance components remain after circularity exclusion."
        )
    total = np.sum(list(components.values()), axis=0)

    dof = len(edges) - 1  # cell 26: dof = len(bins) - 1 = number of bins
    chi2 = _chi2(difference, total)
    p_value = float(1.0 - stats.chi2.cdf(chi2, dof))

    result: dict[str, Any] = {
        "observable": observable,
        "edges": edges,
        "result": result_hist.astype(float),
        "target": target_hist.astype(float),
        "difference": difference.astype(float),
        "mode": mode,
        "components": sorted(components),
        "excluded_components": excluded,
        "chi2": chi2,
        "dof": dof,
        "p_value": p_value,
        "decorrelated": False,
    }

    if decorrelate_components is None:
        # cell 26 default: the DD term plus the smoothed (HV-type)
        # unfolding two-points; other components stay correlated
        decorrelate_components = ["dd_unfolding"] + [
            f"two_point_{name}" for name in smooth_two_point or []
        ]
    decorrelatable = sorted(set(decorrelate_components) & set(components))
    if p_value < decorrelate_threshold and decorrelatable:
        result["decorrelated_components"] = decorrelatable
        decorrelated = _decorrelate(
            components, set(decorrelate_components)
        )
        total = np.sum(list(decorrelated.values()), axis=0)
        result["initial_chi2"] = chi2
        result["initial_p_value"] = p_value
        result["chi2"] = _chi2(difference, total)
        result["p_value"] = float(
            1.0 - stats.chi2.cdf(result["chi2"], dof)
        )
        result["decorrelated"] = True

    result["covariance"] = total
    return result

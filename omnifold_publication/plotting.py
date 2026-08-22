"""Multi-panel closure plots (Start / Target / Result with ratio panels).

Reproduces the layout of 2_pseudo_results.ipynb cell 14: a grid of
subfigures, one per observable, each holding a main panel (height ratio 3)
with the Start distribution (the MC prediction the unfolding started
from), the Target (pseudodata truth, drawn as points), and the Result
with its total-uncertainty band — plus a "Ratio to Target" panel (height
ratio 1) beneath.

All histogram and target-comparison numbers come from ``chi2_test`` — the
same code path used for the closure statistics — so the plots can never
disagree with the test.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import OmniFoldAnalysis
from .closure import _resolve_bins, _weighted_histogram, chi2_test
from .exceptions import PackageReadError
from .reader import OmniFoldPackage


def _default_observables(
    package: OmniFoldPackage,
) -> tuple[list[str], list[str]]:
    """Observables with an official/declared binning, and those skipped.

    Derived observables without an official binning (e.g. tau21, dR_ll)
    are excluded from the default grid and reported explicitly rather
    than silently falling back to suggested bins.
    """

    selected: list[str] = []
    skipped: list[str] = []
    for name in package.list_observables():
        try:
            _resolve_bins(package, name, None)
        except PackageReadError:
            skipped.append(name)
        else:
            selected.append(name)
    return selected, skipped


def plot_closure_grid(
    package: OmniFoldPackage | OmniFoldAnalysis,
    target_package: OmniFoldPackage | None = None,
    observables: list[str] | None = None,
    ncols: int = 4,
    bins_map: dict[str, list[float]] | None = None,
    mode: str = "full",
    variation: str = "nominal",
    target_variation: str = "nominal",
    start_variation: str = "base_mc_weight",
    output_path: str | Path | None = None,
    **chi2_kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Render a closure grid; returns (figure, {"results", "skipped"}).

    Each panel runs ``chi2_test`` for its observable (all keyword options
    are forwarded), draws Start/Target/Result as differential densities,
    annotates chi2/dof and the p-value, and shows Result/Target and
    Start/Target in the ratio panel with the closure-covariance band.
    """

    import matplotlib.pyplot as plt

    result_package = (
        package.nominal_package
        if isinstance(package, OmniFoldAnalysis)
        else package
    )

    skipped: list[str] = []
    if observables is None:
        observables, skipped = _default_observables(result_package)
        if skipped:
            warnings.warn(
                "Skipping observables without an official binning: "
                + ", ".join(skipped)
                + ". Pass them via observables=/bins_map= to include them.",
                stacklevel=2,
            )
    if not observables:
        raise PackageReadError("No observables with an official binning to plot.")

    ncols = max(1, min(ncols, len(observables)))
    nrows = -(-len(observables) // ncols)
    fig = plt.figure(
        figsize=(3.6 * ncols, 3.4 * nrows), constrained_layout=True
    )
    subfigs = np.atleast_1d(fig.subfigures(nrows, ncols)).ravel()

    results: dict[str, dict[str, Any]] = {}
    for index, observable in enumerate(observables):
        bins = (bins_map or {}).get(observable)
        outcome = chi2_test(
            package,
            target_package=target_package,
            observable=observable,
            bins=bins,
            mode=mode,
            variation=variation,
            target_variation=target_variation,
            **chi2_kwargs,
        )
        results[observable] = outcome

        edges = np.asarray(outcome["edges"], dtype=float)
        widths = np.diff(edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        target_density = outcome["target"] / widths
        result_density = outcome["result"] / widths
        band_density = np.sqrt(np.diag(outcome["covariance"])) / widths
        start_density = (
            _weighted_histogram(result_package, observable, start_variation, edges)
            / widths
        )

        axs = subfigs[index].subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        main, ratio = axs

        main.stairs(
            start_density, edges, color="dodgerblue", linestyle="--",
            label="Start",
        )
        main.stairs(
            result_density, edges, color="deeppink", linewidth=1.8,
            label="Result",
        )
        main.fill_between(
            centers,
            result_density - band_density,
            result_density + band_density,
            step=None, alpha=0.25, color="deeppink", linewidth=0,
        )
        main.errorbar(
            centers, target_density, xerr=widths / 2, fmt=".", color="black",
            markersize=4, linewidth=1, label="Target",
        )
        main.set_ylabel("dσ/dx")
        main.set_title(observable, fontsize=10)
        main.legend(fontsize=7, frameon=False, loc="center right")
        main.annotate(
            f"$\\chi^2$/dof = {outcome['chi2']:.1f}/{outcome['dof']}\n"
            f"p = {outcome['p_value']:.3f}"
            + (" (decorrelated)" if outcome["decorrelated"] else ""),
            xy=(0.97, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=7,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            result_ratio = np.where(
                outcome["target"] != 0.0,
                outcome["result"] / outcome["target"],
                np.nan,
            )
            start_ratio = np.where(
                outcome["target"] != 0.0,
                start_density * widths / outcome["target"],
                np.nan,
            )
            band_ratio = np.where(
                outcome["target"] != 0.0,
                np.sqrt(np.diag(outcome["covariance"])) / outcome["target"],
                np.nan,
            )
        ratio.axhline(1.0, color="black", linewidth=0.7)
        ratio.stairs(
            start_ratio, edges, baseline=None, color="dodgerblue",
            linestyle="--",
        )
        ratio.stairs(
            result_ratio, edges, baseline=None, color="deeppink",
            linewidth=1.8,
        )
        ratio.fill_between(
            centers, result_ratio - band_ratio, result_ratio + band_ratio,
            alpha=0.25, color="deeppink", linewidth=0,
        )
        finite = np.concatenate(
            [
                result_ratio - band_ratio,
                result_ratio + band_ratio,
                start_ratio,
            ]
        )
        finite = finite[np.isfinite(finite)]
        if finite.size:
            low = min(0.9, float(finite.min()))
            high = max(1.1, float(finite.max()))
            pad = 0.05 * (high - low)
            ratio.set_ylim(low - pad, high + pad)
        ratio.set_ylabel("Ratio to Target", fontsize=7)
        ratio.set_xlabel(observable, fontsize=8)

    for subfig in subfigs[len(observables):]:
        subfig.set_visible(False)

    if output_path is not None:
        fig.savefig(output_path, dpi=200)
    return fig, {"results": results, "skipped": skipped}

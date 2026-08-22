"""Robust weighted histogram utilities for OmniFold event-level analyses.

This module is intended for advanced analysis use cases where weighted
uncertainties, NaN/Inf filtering, density normalization, or plotting helpers
are needed. Simple proposal examples may use ``numpy.histogram`` directly for
clarity, but this file is the richer utility for realistic downstream studies.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike


class HistogramResult(dict[str, np.ndarray]):
    """Dictionary-like histogram container.

    Keys:
    - ``hist``: weighted bin contents
    - ``edges``: bin edges
    - ``centers``: bin centers
    - ``uncertainty``: per-bin statistical uncertainty (sqrt(sum w^2))

    Notes
    -----
    For backward compatibility, iterating over this object yields
    ``(hist, edges, uncertainty)`` so existing tuple-unpacking code continues
    to work.
    """

    def __iter__(self):  # type: ignore[override]
        return iter((self["hist"], self["edges"], self["uncertainty"]))


def compute_weighted_histogram(
    values: ArrayLike,
    weights: ArrayLike | None = None,
    bins: int | Sequence[float] | str = 50,
    hist_range: tuple[float, float] | None = None,
    density: bool = False,
) -> HistogramResult:
    """Compute a weighted histogram of an observable using per-event OmniFold weights.

    This is the main analysis function. It handles bin validation, NaN/Inf filtering,
    uncertainty estimation, and optional density normalization.

    Plotting is available separately via plot_weighted_histogram.

    Parameters
    ----------
    values:
        Observable values for each event. Input is flattened to 1D.
    weights:
        Event weights. Must match ``values`` shape. If ``None``, all weights are 1.
    bins:
        Histogram binning specification:
        - positive integer (fixed number of bins),
        - explicit edge sequence,
        - ``"auto"`` for automatic bin-edge selection from data.
    hist_range:
        Optional ``(min, max)`` range used for integer/auto binning.
    density:
        If ``True``, return density-normalized histogram and uncertainties.

    Returns
    -------
    HistogramResult
        Dictionary-like object with keys:
        - ``hist``: weighted bin contents,
        - ``edges``: bin edges,
        - ``centers``: bin centers,
        - ``uncertainty``: per-bin uncertainty ``sqrt(sum(w^2))``.

    Raises
    ------
    ValueError
        If inputs are empty, shapes mismatch, no finite entries remain, or bin
        specification is invalid.
    """

    values_arr = np.asarray(values, dtype=float).ravel()
    if values_arr.size == 0:
        raise ValueError("`values` is empty.")

    if weights is None:
        weights_arr = np.ones_like(values_arr, dtype=float)
    else:
        weights_arr = np.asarray(weights, dtype=float).ravel()
        if values_arr.shape != weights_arr.shape:
            raise ValueError("Observable and weights must have the same shape.")

    finite_mask = np.isfinite(values_arr) & np.isfinite(weights_arr)
    values_arr = values_arr[finite_mask]
    weights_arr = weights_arr[finite_mask]

    if values_arr.size == 0:
        raise ValueError("No finite entries remain after filtering NaN/Inf values.")

    # Bin validation is inlined here to keep the computation path in one place,
    # making the function easier to read and audit.
    if isinstance(bins, str):
        if bins != "auto":
            raise ValueError("`bins` as string is only supported for bins='auto'.")
        edges = np.asarray(
            np.histogram_bin_edges(values_arr, bins="auto", range=hist_range),
            dtype=float,
        )
    elif np.isscalar(bins):
        n_bins = int(bins)
        if n_bins <= 0:
            raise ValueError("`bins` must be a positive integer.")
        edges = np.asarray(
            np.histogram_bin_edges(values_arr, bins=n_bins, range=hist_range),
            dtype=float,
        )
    else:
        edges = np.asarray(bins, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("Explicit `bins` must be a 1D array with at least 2 edges.")
        if not np.all(np.isfinite(edges)):
            raise ValueError("Explicit `bins` contain non-finite values.")
        if not np.all(np.diff(edges) > 0.0):
            raise ValueError("Explicit `bins` must be strictly increasing.")

    hist, _ = np.histogram(values_arr, bins=edges, weights=weights_arr, density=False)
    # Statistical uncertainty for weighted events: sigma = sqrt(sum(w^2)) per bin
    sumw2, _ = np.histogram(
        values_arr,
        bins=edges,
        weights=weights_arr * weights_arr,
        density=False,
    )
    uncertainty = np.sqrt(sumw2)
    centers = 0.5 * (edges[1:] + edges[:-1])

    hist = hist.astype(float)
    uncertainty = uncertainty.astype(float)
    centers = centers.astype(float)
    edges = edges.astype(float)

    if density:
        widths = np.diff(edges)
        if not np.all(widths > 0):
            raise ValueError("Bin widths must be positive for density normalization.")
        total_weight = float(np.sum(hist))
        if total_weight <= 0.0:
            raise ValueError(
                "Cannot build a density histogram with non-positive total weight."
            )
        scale = total_weight * widths
        hist = hist / scale
        uncertainty = uncertainty / scale

    return HistogramResult(
        hist=hist,
        edges=edges,
        centers=centers,
        uncertainty=uncertainty,
    )


def plot_weighted_histogram(
    values: ArrayLike,
    weights: ArrayLike | None = None,
    bins: int | Sequence[float] | str = 50,
    hist_range: tuple[float, float] | None = None,
    density: bool = False,
    ax: Axes | None = None,
    label: str | None = None,
    color: str = "C0",
    show_errors: bool = True,
    xlabel: str = "Observable",
) -> tuple[Figure, Axes, np.ndarray, np.ndarray, np.ndarray]:
    """Optional convenience wrapper around ``compute_weighted_histogram``.

    This function preserves the legacy return format:
    ``(fig, ax, hist, edges, uncertainty)``.
    """

    result = compute_weighted_histogram(
        values=values,
        weights=weights,
        bins=bins,
        hist_range=hist_range,
        density=density,
    )

    hist = result["hist"]
    edges = result["edges"]
    uncertainty = result["uncertainty"]
    centers = result["centers"]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.stairs(hist, edges, color=color, label=label, linewidth=1.8)

    if show_errors:
        ax.errorbar(
            centers,
            hist,
            yerr=uncertainty,
            fmt="none",
            ecolor=color,
            elinewidth=1.0,
            capsize=2.0,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density" if density else "Weighted events")
    if label:
        ax.legend(frameon=False)

    return fig, ax, hist, edges, uncertainty

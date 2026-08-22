"""Public histogram result types and weighted histogram utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


def compute_weighted_histogram(
    values: ArrayLike,
    weights: ArrayLike | None = None,
    bins: int | Sequence[float] | str = 50,
    hist_range: tuple[float, float] | None = None,
    density: bool = False,
) -> dict[str, np.ndarray]:
    """Compute weighted bin contents and ``sqrt(sum(w^2))`` uncertainty."""

    values_array = np.asarray(values, dtype=float).ravel()
    if values_array.size == 0:
        raise ValueError("`values` is empty.")

    if weights is None:
        weights_array = np.ones_like(values_array, dtype=float)
    else:
        weights_array = np.asarray(weights, dtype=float).ravel()
        if values_array.shape != weights_array.shape:
            raise ValueError("Observable and weights must have the same shape.")

    finite = np.isfinite(values_array) & np.isfinite(weights_array)
    values_array = values_array[finite]
    weights_array = weights_array[finite]
    if values_array.size == 0:
        raise ValueError("No finite entries remain after filtering NaN/Inf values.")

    if isinstance(bins, str):
        if bins != "auto":
            raise ValueError("`bins` as string is only supported for bins='auto'.")
        edges = np.histogram_bin_edges(
            values_array,
            bins="auto",
            range=hist_range,
        )
    elif np.isscalar(bins):
        bin_count = int(bins)
        if bin_count <= 0:
            raise ValueError("`bins` must be a positive integer.")
        edges = np.histogram_bin_edges(
            values_array,
            bins=bin_count,
            range=hist_range,
        )
    else:
        edges = np.asarray(bins, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("Explicit `bins` must be a 1D array with at least 2 edges.")
        if not np.all(np.isfinite(edges)):
            raise ValueError("Explicit `bins` contain non-finite values.")
        if not np.all(np.diff(edges) > 0.0):
            raise ValueError("Explicit `bins` must be strictly increasing.")

    edges = np.asarray(edges, dtype=float)
    hist, _ = np.histogram(values_array, bins=edges, weights=weights_array)
    sumw2, _ = np.histogram(
        values_array,
        bins=edges,
        weights=weights_array * weights_array,
    )
    uncertainty = np.sqrt(sumw2).astype(float)
    hist = hist.astype(float)

    if density:
        total_weight = float(np.sum(hist))
        if total_weight <= 0.0:
            raise ValueError(
                "Cannot build a density histogram with non-positive total weight."
            )
        scale = total_weight * np.diff(edges)
        hist = hist / scale
        uncertainty = uncertainty / scale

    return {
        "hist": hist,
        "edges": edges,
        "centers": 0.5 * (edges[1:] + edges[:-1]),
        "uncertainty": uncertainty,
    }


@dataclass
class HistogramResult:
    """Structured weighted histogram and its uncertainty components."""

    hist: np.ndarray
    edges: np.ndarray
    centers: np.ndarray
    stat_uncertainty: np.ndarray
    sys_uncertainty: np.ndarray | None = None
    replica_uncertainty: np.ndarray | None = None
    # combined uncertainty, for sources (e.g. HEPData tables) that publish
    # a single total error rather than a component breakdown
    total_uncertainty: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation, omitting absent fields."""

        result: dict[str, Any] = {
            "hist": self.hist.tolist(),
            "edges": self.edges.tolist(),
            "centers": self.centers.tolist(),
            "stat_uncertainty": self.stat_uncertainty.tolist(),
        }
        if self.sys_uncertainty is not None:
            result["sys_uncertainty"] = self.sys_uncertainty.tolist()
        if self.replica_uncertainty is not None:
            result["replica_uncertainty"] = self.replica_uncertainty.tolist()
        if self.total_uncertainty is not None:
            result["total_uncertainty"] = self.total_uncertainty.tolist()
        return result


__all__ = ["HistogramResult", "compute_weighted_histogram"]

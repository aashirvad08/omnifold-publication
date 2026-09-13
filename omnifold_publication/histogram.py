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


# Optional per-source uncertainty components, in the order they are
# combined into the total. Which ones a given result carries depends on
# what its source could compute; ``total_uncertainty`` and
# ``stat_uncertainty`` are always present, whatever the source.
OPTIONAL_COMPONENTS = ("sys_uncertainty", "replica_uncertainty")


@dataclass
class HistogramResult:
    """Structured weighted histogram and its uncertainty components.

    **Guaranteed fields.** ``hist``, ``edges``, ``centers``,
    ``stat_uncertainty`` and ``total_uncertainty`` are always arrays, for
    every source. Code that needs one number per bin can read
    ``total_uncertainty`` without checking where the result came from.

    **Optional components.** ``sys_uncertainty`` and
    ``replica_uncertainty`` are filled only when the source can compute
    them, so they differ by construction path:

    - :meth:`OmniFoldPackage.histogram` — one sample, so no cross-sample
      systematic and no replica band: both are ``None`` and
      ``total_uncertainty`` equals ``stat_uncertainty``.
    - :meth:`OmniFoldAnalysis.histogram` — fills ``sys_uncertainty`` when
      ``systematic_variations`` are requested and ``replica_uncertainty``
      when the nominal package carries replicas.
    - :meth:`HEPDataPackage.histogram` — a published record carries a
      total error, so ``total_uncertainty`` is the published total and
      ``stat_uncertainty`` is filled only when the table publishes a
      labelled breakdown.

    Rather than testing each field, ask :meth:`components` for the ones
    this result actually carries.
    """

    hist: np.ndarray
    edges: np.ndarray
    centers: np.ndarray
    stat_uncertainty: np.ndarray
    sys_uncertainty: np.ndarray | None = None
    replica_uncertainty: np.ndarray | None = None
    # Always populated: passed in by sources that publish a single total
    # (e.g. HEPData tables), otherwise derived in __post_init__ from the
    # components present.
    total_uncertainty: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Derive ``total_uncertainty`` when the source did not supply one."""

        if self.total_uncertainty is not None:
            self.total_uncertainty = np.asarray(
                self.total_uncertainty, dtype=float
            )
            return
        squared = np.asarray(self.stat_uncertainty, dtype=float) ** 2
        for name in OPTIONAL_COMPONENTS:
            component = getattr(self, name)
            if component is not None:
                squared = squared + np.asarray(component, dtype=float) ** 2
        self.total_uncertainty = np.sqrt(squared)

    def components(self) -> dict[str, np.ndarray]:
        """The uncertainty components this result actually carries.

        Always includes ``stat_uncertainty``; includes the optional
        components only when populated. ``total_uncertainty`` is excluded
        — it is the quadrature sum of these, not a peer of them.
        """

        present: dict[str, np.ndarray] = {
            "stat_uncertainty": np.asarray(self.stat_uncertainty, dtype=float)
        }
        for name in OPTIONAL_COMPONENTS:
            component = getattr(self, name)
            if component is not None:
                present[name] = np.asarray(component, dtype=float)
        return present

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation.

        The guaranteed fields are always present; the optional components
        appear only when this result carries them.
        """

        result: dict[str, Any] = {
            "hist": self.hist.tolist(),
            "edges": self.edges.tolist(),
            "centers": self.centers.tolist(),
            "stat_uncertainty": np.asarray(
                self.stat_uncertainty, dtype=float
            ).tolist(),
        }
        if self.sys_uncertainty is not None:
            result["sys_uncertainty"] = np.asarray(
                self.sys_uncertainty, dtype=float
            ).tolist()
        if self.replica_uncertainty is not None:
            result["replica_uncertainty"] = np.asarray(
                self.replica_uncertainty, dtype=float
            ).tolist()
        result["total_uncertainty"] = np.asarray(
            self.total_uncertainty, dtype=float
        ).tolist()
        return result


__all__ = [
    "OPTIONAL_COMPONENTS",
    "HistogramResult",
    "compute_weighted_histogram",
]

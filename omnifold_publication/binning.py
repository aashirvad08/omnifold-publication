"""Binning helpers and validity checks from the ATLAS Z+jets release.

- ``n_eff``: effective statistics of a weighted sample,
  (sum w)^2 / sum(w^2) (multifold_util.py:86-87).
- ``equal_effective_events_bins``: bin-edge generator targeting a fixed
  number of effective events per bin (multifold_util.py:89-120), here
  vectorized with cumulative sums instead of the original row loop.
- ``validate_binning``: the README "usage recommendations" 2 and 3 as
  per-bin checks — every bin should hold at least 5,000 effective events,
  and the data statistical uncertainty should stay below 15%.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .exceptions import PackageReadError


N_EFF_MIN = 5000.0  # README usage recommendation 2
DATA_STAT_MAX = 0.15  # README usage recommendation 3


def n_eff(weights: np.ndarray) -> float:
    """Effective number of events: (sum w)^2 / sum(w^2)."""

    w = np.asarray(weights, dtype=float)
    sumw2 = float((w**2).sum())
    if sumw2 == 0.0:
        return 0.0
    return float(w.sum()) ** 2 / sumw2


def equal_effective_events_bins(
    values: np.ndarray,
    weights: np.ndarray,
    target_n_eff: float,
) -> list[float]:
    """Bin edges giving ~``target_n_eff`` effective events per bin.

    Vectorized port of ``equal_effective_events_bins``
    (multifold_util.py:89-120): events are sorted in the observable,
    cumulative n_eff(t) = (cumsum w)^2 / (cumsum w^2) is tracked, and a
    new edge is placed at the first event where the cumulative n_eff
    crosses (bin_index + 1) * target, up to n_bins - 1 internal edges with
    n_bins = int(total_n_eff / target). The final edge is the maximum
    value. Semantics match the original row loop exactly.
    """

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.shape != weights.shape:
        raise PackageReadError(
            "equal_effective_events_bins requires values and weights of "
            "equal length."
        )

    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]

    total = n_eff(sorted_weights)
    n_bins = int(total / target_n_eff)
    if n_bins < 1:
        raise PackageReadError(
            f"Sample holds only {total:.1f} effective events; cannot form "
            f"a single bin of {target_n_eff:.0f}."
        )

    cumulative_w = np.cumsum(sorted_weights)
    cumulative_w2 = np.cumsum(sorted_weights**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        cumulative_n_eff = np.where(
            cumulative_w2 > 0.0, cumulative_w**2 / cumulative_w2, 0.0
        )

    edges = [float(sorted_values[0])]
    bin_index = 0
    position = 0
    while bin_index < n_bins - 1 and position < len(sorted_values):
        threshold = (bin_index + 1) * target_n_eff
        crossings = np.nonzero(cumulative_n_eff[position:] >= threshold)[0]
        if crossings.size == 0:
            break
        position += int(crossings[0])
        edges.append(float(sorted_values[position]))
        bin_index += 1
        # the original row loop consumes one row per placed edge: the next
        # threshold is checked from the following row onward
        position += 1
    edges.append(float(sorted_values[-1]))
    return edges


def validate_binning(
    package: Any,
    observable: str,
    bins: list[float] | int | None = None,
    n_eff_min: float = N_EFF_MIN,
    data_stat_max: float = DATA_STAT_MAX,
) -> dict[str, Any]:
    """Check a binning against the release's per-bin validity criteria.

    Returns a report with per-bin effective statistics and data statistical
    uncertainty, plus an ``errors`` list (empty when the binning is valid):

    - ``n_eff`` >= ``n_eff_min`` per bin (README recommendation 2),
      computed from the nominal weights in each bin.
    - data statistical uncertainty < ``data_stat_max`` per bin (README
      recommendation 3), from the bootstrap_data family spread when the
      package declares one, otherwise from sqrt(sum w^2)/sum w with the
      fallback recorded in the report as ``data_stat_source``.

    The observable's declared selection, if any, is applied first.
    """

    from .uncertainty import bootstrap_standard_deviation, histogram_matrix

    selected_bins = bins
    if selected_bins is None:
        selected_bins = package.observable_bins(observable) or 30

    values, mask = package.observable_values(observable)
    nominal = np.asarray(package.get_weights("nominal"), dtype=float)[mask]

    nominal_hist, edges = np.histogram(
        values, bins=selected_bins, weights=nominal
    )
    sumw2 = np.histogram(values, bins=edges, weights=nominal**2)[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        bin_n_eff = np.where(sumw2 > 0.0, nominal_hist**2 / sumw2, 0.0)

    if "bootstrap_data" in package.list_weight_families():
        replica_hists = histogram_matrix(
            values, package.get_family_weights("bootstrap_data")[:, mask], edges
        )
        data_stat = bootstrap_standard_deviation(replica_hists)
        data_stat_source = "bootstrap_data"
    else:
        data_stat = np.sqrt(sumw2)
        data_stat_source = "sample_stat"
    with np.errstate(divide="ignore", invalid="ignore"):
        data_stat_fraction = np.where(
            nominal_hist != 0.0, data_stat / np.abs(nominal_hist), np.inf
        )

    errors: list[str] = []
    for index in range(len(edges) - 1):
        label = f"[{edges[index]:g}, {edges[index + 1]:g})"
        if bin_n_eff[index] < n_eff_min:
            errors.append(
                f"{observable} bin {label}: n_eff {bin_n_eff[index]:.0f} "
                f"below the minimum {n_eff_min:.0f}."
            )
        if data_stat_fraction[index] >= data_stat_max:
            errors.append(
                f"{observable} bin {label}: data statistical uncertainty "
                f"{100 * data_stat_fraction[index]:.1f}% exceeds "
                f"{100 * data_stat_max:.0f}%."
            )

    return {
        "edges": edges,
        "nominal": nominal_hist.astype(float),
        "n_eff": bin_n_eff,
        "data_stat_fraction": data_stat_fraction,
        "data_stat_source": data_stat_source,
        "errors": errors,
    }

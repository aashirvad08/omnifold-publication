"""Compare two *independently published* results against each other.

Unlike :class:`~omnifold_publication.comparison.HistogramComparison`, which
compares variations within a single publication against that publication's
own nominal, this compares two separate packages (different provenance —
different checksums, sources, and possibly unfolding methods) on a shared
observable and shared binning.

Two principles:

- **Each side's uncertainty is computed independently, from its own data.**
  There is no shared nominal across two publications, so each package's
  full uncertainty breakdown (statistical + replica/ensemble + bootstrap +
  systematic families it declares) is computed by that package's own
  machinery and attributed to that side by label. This is the gap the
  cross-publication view fills.
- **Rebinning is an explicit re-histogram of raw events, never
  interpolation.** Both sides are re-filled from their raw event tables
  with the same requested bin edges; nothing is resampled or curve-fit
  after the fact.

The ratio panel compares the two results to each other (side B / side A),
not to a nominal within one of them. By default each side's own relative
uncertainty band is shown *separately* on the ratio — no correlation model
between the two publications is assumed. A combined ratio band would
require knowing whether the two unfold the same underlying events
(correlated) or independent data; that is exposed as an explicit
``correlation`` option rather than silently baked in either way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import PackageReadError
from .reader import OmniFoldPackage

# Correlation models for the ratio band. Only "none" is implemented; the
# others are named so the choice is explicit and discoverable, but must not
# be used until the physical relationship between the two results is known.
CORRELATION_MODELS = ("none", "independent", "same_events")


def _describe(package: OmniFoldPackage, fallback_label: str) -> dict[str, Any]:
    metadata = package.metadata()
    dataset = metadata.get("dataset", {})
    dataset = dataset if isinstance(dataset, dict) else {}
    publication = metadata.get("publication", {})
    publication = publication if isinstance(publication, dict) else {}
    method = dataset.get("method")
    return {
        "label": method or dataset.get("name") or fallback_label,
        "method": method,
        "dataset": dataset.get("name"),
        "checksum_sha256": publication.get("checksum_sha256"),
        "checksum_kind": "package",
        "n_events": publication.get("event_count"),
        "assumptions": dataset.get("assumptions"),
    }


def _step(values: np.ndarray) -> np.ndarray:
    """Pad per-bin values to bin-edge length for a ``step="post"`` fill."""

    values = np.asarray(values, dtype=float)
    return np.append(values, values[-1])


def _relative(total: np.ndarray, hist: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(hist != 0.0, total / np.abs(hist), np.nan)


# Coarse grouping for the per-component uncertainty comparison view, derived
# from each package's own declared weight-family "type" — never inferred
# from the component name string. "sample_stat" is the one component with no
# declared family: it is the baseline sqrt(sum w^2) statistical term computed
# directly in uncertainty.py, so it is treated as statistical by convention.
_FAMILY_GROUP = {
    "bootstrap": "statistical",
    "ensemble": "statistical",
    "systematic": "systematic",
    "paired": "data_driven",
}
_GROUP_COLOR = {
    "statistical": "#2c7fb8",
    "systematic": "#d6604d",
    "data_driven": "#7570b3",
    "other": "#888888",
}


def _component_group(package: OmniFoldPackage, name: str) -> str:
    if name == "sample_stat":
        return "statistical"
    try:
        declared = package.weight_family(name).get("type")
    except PackageReadError:
        # A breakdown component with no declared family: group it as "other"
        # rather than failing the whole plot. Grouping is presentation only,
        # so an unrecognised component must never hide the numbers.
        return "other"
    return _FAMILY_GROUP.get(declared, "other")


class CrossPublicationComparison:
    """Side-by-side comparison of two published results with independent
    uncertainty bands and a between-results ratio."""

    def __init__(
        self,
        package_a: OmniFoldPackage,
        package_b: OmniFoldPackage,
        observable: str,
        bins: list[float] | None = None,
        labels: tuple[str, str] | None = None,
        correlation: str = "none",
    ):
        if correlation not in CORRELATION_MODELS:
            allowed = ", ".join(CORRELATION_MODELS)
            raise PackageReadError(
                f"Unknown correlation model {correlation!r}; one of: {allowed}."
            )
        if correlation != "none":
            raise NotImplementedError(
                f"correlation={correlation!r} is not implemented. Only "
                "'none' (each side's band shown separately) is supported; a "
                "combined ratio band requires a confirmed correlation model "
                "between the two publications."
            )

        self.observable = observable
        self.correlation = correlation
        self._package_a = package_a
        self._package_b = package_b
        fallback_a, fallback_b = labels or ("A", "B")

        # shared bins: explicit, else side A's declared/official binning,
        # applied to BOTH by re-histogramming raw events (no interpolation).
        resolved = bins if bins is not None else package_a.observable_bins(observable)
        if resolved is None:
            raise PackageReadError(
                f"No shared bins given and package A declares none for "
                f"{observable!r}; pass explicit bin edges."
            )
        self.bins = [float(edge) for edge in resolved]

        self._side_a = self._build_side(package_a, fallback_a)
        self._side_b = self._build_side(package_b, fallback_b)

        a_hist = np.asarray(self._side_a["hist"], dtype=float)
        b_hist = np.asarray(self._side_b["hist"], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(a_hist != 0.0, b_hist / a_hist, np.nan)
        self.ratio = {
            "reference": self._side_a["label"],
            "comparand": self._side_b["label"],
            "values": ratio.tolist(),
            # each side's own relative band, kept separate (see class doc)
            "reference_rel_band": _relative(
                np.asarray(self._side_a["uncertainty"]["total"]), a_hist
            ).tolist(),
            "comparand_rel_band": _relative(
                np.asarray(self._side_b["uncertainty"]["total"]), b_hist
            ).tolist(),
            "correlation": correlation,
        }

    def _build_side(
        self, package: OmniFoldPackage, fallback_label: str
    ) -> dict[str, Any]:
        breakdown = package.uncertainty_breakdown(self.observable, bins=self.bins)
        info = _describe(package, fallback_label)
        info["hist"] = np.asarray(breakdown["nominal"], dtype=float).tolist()
        info["uncertainty"] = {
            "total": np.asarray(breakdown["total"], dtype=float).tolist(),
            "components": {
                name: np.asarray(values, dtype=float).tolist()
                for name, values in breakdown["components"].items()
            },
        }
        return info

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "cross_publication_comparison",
            "observable": self.observable,
            "bins": self.bins,
            "sides": [self._side_a, self._side_b],
            "ratio": self.ratio,
            "provenance": {
                "correlation_model": self.correlation,
                "correlation_note": (
                    "each publication's uncertainty band is shown "
                    "independently; no correlation between the two results is "
                    "assumed for a combined ratio band"
                ),
                "rebinning": (
                    "both sides re-histogrammed from raw events with the "
                    "shared bin edges; no interpolation"
                ),
            },
        }

    def export_json(self, output_path: str | Path) -> None:
        with Path(output_path).open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2)

    def plot(self, output_path: str | Path | None = None) -> Any:
        """Main panel (both results with their own uncertainty bands) and a
        ratio panel (B/A, each band shown separately)."""

        import matplotlib.pyplot as plt

        edges = np.asarray(self.bins, dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        fig, (ax, rax) = plt.subplots(
            2, 1, sharex=True, figsize=(7, 6.5),
            gridspec_kw={"height_ratios": [3, 1]},
        )

        for side, color in ((self._side_a, "#1f77b4"), (self._side_b, "#d6336c")):
            hist = np.asarray(side["hist"], dtype=float)
            total = np.asarray(side["uncertainty"]["total"], dtype=float)
            ax.stairs(hist, edges, color=color, linewidth=2, label=side["label"])
            # step band, not a fill across bin centres: the band must follow
            # the histogram it belongs to rather than interpolate between bins
            ax.fill_between(
                edges, _step(hist - total), _step(hist + total), step="post",
                color=color, alpha=0.25, linewidth=0,
            )
        # falling spectra span decades; a linear axis hides the tail bins and
        # their bands entirely, so switch to log once the range warrants it
        positive = np.concatenate([
            np.asarray(s["hist"], dtype=float) for s in (self._side_a, self._side_b)
        ])
        positive = positive[positive > 0.0]
        if positive.size and positive.max() / positive.min() > 20.0:
            ax.set_yscale("log")

        ax.set_ylabel("weighted events (fb)")
        ax.set_title(f"Cross-publication comparison — {self.observable}")
        ax.legend(frameon=False)

        values = np.asarray(self.ratio["values"], dtype=float)
        a_band = np.asarray(self.ratio["reference_rel_band"], dtype=float)
        b_band = np.asarray(self.ratio["comparand_rel_band"], dtype=float)
        rax.axhline(1.0, color="black", linewidth=0.8)
        # reference (A) band around 1, as a step for the same reason
        rax.fill_between(
            edges, _step(1 - a_band), _step(1 + a_band), step="post",
            color="#1f77b4", alpha=0.2, linewidth=0,
            label=f"{self._side_a['label']} unc.",
        )
        # comparand (B) points with its own band
        rax.errorbar(
            centers, values, yerr=np.abs(values) * b_band, fmt="o",
            color="#d6336c", markersize=4,
            label=f"{self._side_b['label']}/{self._side_a['label']}",
        )
        rax.set_ylabel(f"{self._side_b['label']} / {self._side_a['label']}")
        rax.set_xlabel(self.observable)
        rax.legend(frameon=False, fontsize=8)

        fig.tight_layout()
        if output_path is not None:
            fig.savefig(output_path, dpi=160)
        return fig

    def uncertainty_comparison_to_dict(self) -> dict[str, Any]:
        """Per-side relative-uncertainty numbers behind
        :meth:`plot_uncertainty_comparison`: per-component and type-grouped
        relative uncertainty (%), derived from each side's own
        ``uncertainty_breakdown()`` output — no uncertainty is recomputed
        here, only expressed as a percentage of that side's own histogram."""

        sides_out = []
        for side, package in (
            (self._side_a, self._package_a),
            (self._side_b, self._package_b),
        ):
            hist = np.asarray(side["hist"], dtype=float)
            total = np.asarray(side["uncertainty"]["total"], dtype=float)
            components = side["uncertainty"]["components"]

            component_group: dict[str, str] = {}
            components_relative_pct: dict[str, list[float]] = {}
            group_sq: dict[str, np.ndarray] = {}
            for name, values in components.items():
                values_arr = np.asarray(values, dtype=float)
                components_relative_pct[name] = (
                    100.0 * _relative(values_arr, hist)
                ).tolist()
                group = _component_group(package, name)
                component_group[name] = group
                group_sq[group] = group_sq.get(
                    group, np.zeros_like(values_arr)
                ) + values_arr**2

            group_relative_pct = {
                group: (100.0 * _relative(np.sqrt(sq), hist)).tolist()
                for group, sq in group_sq.items()
            }

            sides_out.append(
                {
                    "label": side["label"],
                    "total_relative_pct": (100.0 * _relative(total, hist)).tolist(),
                    "components_relative_pct": components_relative_pct,
                    "component_group": component_group,
                    "group_relative_pct": group_relative_pct,
                }
            )

        return {
            "kind": "cross_publication_uncertainty_comparison",
            "observable": self.observable,
            "bins": self.bins,
            "sides": sides_out,
            "provenance": {
                "correlation_model": self.correlation,
                "correlation_note": (
                    "each side's uncertainty is computed independently from "
                    "its own data; this view never implies a combined or "
                    "correlated uncertainty between the two publications"
                ),
                "derivation_note": (
                    "components_relative_pct[name] = 100 * "
                    "components[name] / abs(nominal), both taken as-is from "
                    "that side's uncertainty_breakdown(); group_relative_pct "
                    "is the quadrature sum (sqrt of sum of squares) of the "
                    "named components sharing a declared weight-family "
                    "'type' (bootstrap/ensemble -> statistical, systematic "
                    "-> systematic, paired -> data_driven; sample_stat is "
                    "the undeclared baseline statistical term). No "
                    "uncertainty value is recomputed from raw events."
                ),
            },
        }

    def export_uncertainty_comparison_json(self, output_path: str | Path) -> None:
        with Path(output_path).open("w", encoding="utf-8") as stream:
            json.dump(self.uncertainty_comparison_to_dict(), stream, indent=2)

    def plot_uncertainty_comparison(self, output_path: str | Path | None = None) -> Any:
        """Compare the two sides' *own* uncertainties directly: total
        relative size per bin (top), and per-component composition with both
        publications overlaid on one canvas (bottom), coloured by declared
        family type and styled by publication. Both panels use equal-width
        bin slots so every bin carries the same visual weight. Each side's
        uncertainty is independent — this never implies a combined band,
        consistent with ``correlation="none"``."""

        import matplotlib.pyplot as plt

        edges = np.asarray(self.bins, dtype=float)
        # The grouped bars sit on evenly spaced categorical slots, one per
        # bin, rather than at physical bin centres. Physical placement scales
        # bar width with bin width, so a wide trailing bin renders as a huge
        # offset pair that misstates the binning and squeezes narrow bins to
        # invisibility. The bin range is carried by the tick label instead.
        slots = np.arange(len(edges) - 1, dtype=float)
        bar_width = 0.38
        slot_labels = [
            f"{edges[i]:g}–{edges[i + 1]:g}" for i in range(len(edges) - 1)
        ]

        sides = (
            (self._side_a, self._package_a, "#1f77b4"),
            (self._side_b, self._package_b, "#d6336c"),
        )

        # Uniform slot edges: one equal-width slot per bin. Both sides are
        # drawn on these same edges so every bin carries equal visual weight
        # regardless of its physical width (200-230 GeV is 30 GeV wide,
        # 600-1000 GeV is 400) - otherwise the wide tail bin dominates the
        # canvas and the narrow bins are unreadable.
        slot_edges = np.arange(len(edges), dtype=float)

        fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.3])
        ax_total = fig.add_subplot(gs[0])
        ax_comp = fig.add_subplot(gs[1])

        # --- top: total relative uncertainty, grouped bars ---
        rel_totals = []
        for (side, _, color), sign in zip(sides, (-1, 1)):
            hist = np.asarray(side["hist"], dtype=float)
            total = np.asarray(side["uncertainty"]["total"], dtype=float)
            rel = 100.0 * _relative(total, hist)
            rel_totals.append(rel)
            ax_total.bar(
                slots + sign * bar_width / 2, rel, width=bar_width,
                color=color, label=side["label"],
            )
        ax_total.set_ylabel("total relative uncertainty [%]")
        ax_total.set_xticks(slots)
        ax_total.set_xticklabels(slot_labels)
        units = self._package_a.observable_units(self.observable)
        ax_total.set_xlabel(
            f"{self.observable} [{units}]" if units else self.observable
        )
        ax_total.set_title(
            f"Uncertainty comparison — {self.observable}\n"
            "each side computed independently; no cross-publication "
            "correlation assumed (correlation=\"none\")",
            fontsize=10,
        )
        ax_total.legend(frameon=False)
        finite_totals = np.concatenate(
            [rel[np.isfinite(rel) & (rel > 0)] for rel in rel_totals]
        )
        if finite_totals.size and finite_totals.max() / finite_totals.min() > 20.0:
            ax_total.set_yscale("log")

        # --- bottom: both sides' components overlaid on one canvas ---
        # Two encodings are needed at once, so they are kept orthogonal:
        # colour carries the declared family group, line style carries the
        # publication. Component names are deliberately not in the legend -
        # every systematic shares one colour, so naming them individually
        # would imply a distinction the colours do not actually make.
        all_component_vals = []
        seen_groups: set[str] = set()
        side_styles = ("-", "--")
        for (side, package, _), style in zip(sides, side_styles):
            hist = np.asarray(side["hist"], dtype=float)
            total = np.asarray(side["uncertainty"]["total"], dtype=float)
            for name, values in sorted(
                side["uncertainty"]["components"].items(),
                key=lambda kv: -np.sum(kv[1]),
            ):
                group = _component_group(package, name)
                rel = 100.0 * _relative(np.asarray(values, dtype=float), hist)
                all_component_vals.append(rel)
                ax_comp.stairs(
                    rel, slot_edges, color=_GROUP_COLOR[group], linewidth=1.1,
                    linestyle=style, alpha=0.75, baseline=None,
                )
                seen_groups.add(group)
            rel_total = 100.0 * _relative(total, hist)
            all_component_vals.append(rel_total)
            ax_comp.stairs(
                rel_total, slot_edges, color="black", linewidth=2.2,
                linestyle=style, baseline=None,
            )

        ax_comp.set_ylabel("relative uncertainty [%]")
        ax_comp.set_xticks(0.5 * (slot_edges[:-1] + slot_edges[1:]))
        ax_comp.set_xticklabels(slot_labels)
        ax_comp.set_xlabel(
            f"{self.observable} [{units}]" if units else self.observable
        )
        ax_comp.set_title(
            "per-component composition, both publications overlaid "
            "(equal-width bins)",
            fontsize=9,
        )

        finite_components = np.concatenate(
            [v[np.isfinite(v) & (v > 0)] for v in all_component_vals]
        )
        if finite_components.size and (
            finite_components.max() / finite_components.min() > 20.0
        ):
            ax_comp.set_yscale("log")

        group_handles = [
            plt.Line2D([0], [0], color=_GROUP_COLOR[g], lw=2, label=g.replace("_", " "))
            for g in ("statistical", "systematic", "data_driven", "other")
            if g in seen_groups
        ]
        group_handles.append(
            plt.Line2D([0], [0], color="black", lw=2.2, label="total")
        )
        side_handles = [
            plt.Line2D([0], [0], color="#444444", lw=1.6, linestyle=style,
                       label=side["label"])
            for (side, _, _), style in zip(sides, side_styles)
        ]
        legend_groups = ax_comp.legend(
            handles=group_handles, title="uncertainty type", fontsize=7,
            title_fontsize=7, frameon=False, ncol=len(group_handles),
            loc="upper left",
        )
        ax_comp.add_artist(legend_groups)
        ax_comp.legend(
            handles=side_handles, title="publication", fontsize=7,
            title_fontsize=7, frameon=False, loc="lower right",
        )

        if output_path is not None:
            fig.savefig(output_path, dpi=160)
        return fig


def compare_publications(
    package_a: OmniFoldPackage,
    package_b: OmniFoldPackage,
    observable: str,
    bins: list[float] | None = None,
    labels: tuple[str, str] | None = None,
    correlation: str = "none",
) -> CrossPublicationComparison:
    """Convenience constructor for :class:`CrossPublicationComparison`."""

    return CrossPublicationComparison(
        package_a, package_b, observable, bins=bins, labels=labels,
        correlation=correlation,
    )

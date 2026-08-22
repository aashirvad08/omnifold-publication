"""Histogram comparison utilities for multi-sample OmniFold analyses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from .analysis import OmniFoldAnalysis


class HistogramComparison:
    """Side-by-side histogram comparison across multiple datasets."""

    def __init__(
        self,
        analysis: OmniFoldAnalysis,
        observable: str,
        bins: list[float] | np.ndarray,
    ):
        """Build histogram comparison data for an analysis observable."""

        self.analysis = analysis
        self.observable = observable
        self.bins = np.asarray(bins, dtype=float)
        self.histograms = self._compute_histograms()
        self.replica_uncertainty = self._compute_replica_uncertainty()

    def plot(
        self,
        show_ratio: bool = True,
        show_uncertainty: bool = True,
        output_path: str | Path | None = None,
    ):
        """Draw comparison histogram with optional ratio panel."""

        if show_ratio:
            fig, (ax, ratio_ax) = plt.subplots(
                2,
                1,
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
                figsize=(7, 6),
            )
        else:
            fig, ax = plt.subplots(figsize=(7, 4))
            ratio_ax = None

        centers = 0.5 * (self.bins[1:] + self.bins[:-1])
        nominal = self.histograms["nominal"]
        ax.step(
            centers,
            nominal,
            where="mid",
            color="blue",
            label="nominal",
            linewidth=2.0,
        )

        if show_uncertainty and self.replica_uncertainty is not None:
            ax.fill_between(
                centers,
                nominal - self.replica_uncertainty,
                nominal + self.replica_uncertainty,
                step="mid",
                alpha=0.25,
                label="replica uncertainty",
            )

        for name, hist in self.histograms.items():
            if name == "nominal":
                continue
            color = "green" if name.lower() == "nondy" else None
            ax.step(
                centers,
                hist,
                where="mid",
                color=color,
                label=name,
                linewidth=1.6,
            )
            if ratio_ax is not None:
                ratio_ax.plot(
                    centers,
                    self._ratio(hist, nominal),
                    color=color,
                    marker="o",
                    label=name,
                )

        ax.set_ylabel("Weighted events")
        ax.legend(frameon=False)

        if ratio_ax is not None:
            ratio_ax.axhline(1.0, color="black", linewidth=1.0, alpha=0.5)
            ratio_ax.set_ylabel("ratio")
            ratio_ax.set_xlabel(self.observable)
            ratio_ax.legend(frameon=False)
        else:
            ax.set_xlabel(self.observable)

        fig.tight_layout()
        if output_path is not None:
            fig.savefig(output_path)
        return fig

    def print_table(self) -> None:
        """Print bin-by-bin comparison table to terminal."""

        headers = ["low", "high", *self.histograms]
        print(" | ".join(headers))
        print(" | ".join(["---"] * len(headers)))
        for index, (low, high) in enumerate(zip(self.bins[:-1], self.bins[1:])):
            values = [
                f"{low:.6g}",
                f"{high:.6g}",
                *[f"{hist[index]:.6g}" for hist in self.histograms.values()],
            ]
            print(" | ".join(values))

    def export_json(self, output_path: str | Path) -> None:
        """Export comparison data as JSON for visualization."""

        payload: dict[str, Any] = {
            "observable": self.observable,
            "bins": self.bins.tolist(),
            "histograms": {
                name: hist.tolist() for name, hist in self.histograms.items()
            },
            "ratios": {
                name: self._ratio(hist, self.histograms["nominal"]).tolist()
                for name, hist in self.histograms.items()
                if name != "nominal"
            },
            "replica_uncertainty": None
            if self.replica_uncertainty is None
            else self.replica_uncertainty.tolist(),
        }
        with Path(output_path).open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)

    def _compute_histograms(self) -> dict[str, np.ndarray]:
        histograms: dict[str, np.ndarray] = {}
        for variation in self.analysis.list_variations():
            if "ensemble" in variation:
                continue
            df = self.analysis.load_events(columns=[self.observable], variation=variation)
            weights = self.analysis.get_weights(variation)
            hist, _ = np.histogram(
                df[self.observable].to_numpy(dtype=float),
                bins=self.bins,
                weights=weights,
            )
            histograms[variation] = hist.astype(float)
        return histograms

    def _compute_replica_uncertainty(self) -> np.ndarray | None:
        replicas = self.analysis.get_replica_weights()
        if replicas.size == 0:
            return None

        nominal_events = self.analysis.load_events(columns=[self.observable])
        values = nominal_events[self.observable].to_numpy(dtype=float)
        if replicas.shape[1] != values.shape[0]:
            return None

        replica_hists = []
        for weights in replicas:
            hist, _ = np.histogram(values, bins=self.bins, weights=weights)
            replica_hists.append(hist.astype(float))
        return np.std(np.vstack(replica_hists), axis=0)

    @staticmethod
    def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        return np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=float),
            where=denominator != 0,
        )

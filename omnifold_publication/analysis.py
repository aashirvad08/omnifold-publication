"""Multi-sample analysis wrapper for OmniFold publication packages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import PackageReadError, PackageValidationError
from .histogram import HistogramResult, compute_weighted_histogram
from .manifest import list_manifest_variations, load_manifest
from .reader import OmniFoldPackage, load_package
from .uncertainty import (
    correlation_matrix,
    ensemble_median_standard_error,
    fill_cov_matrix,
    quadrature_difference_from_nominal,
    smooth_uncertainty,
    total_in_quadrature,
)


class OmniFoldAnalysis:
    """Multi-sample analysis wrapper backed by a manifest.yaml."""

    def __init__(self, manifest_dir: str | Path):
        """Load manifest and initialize all package references."""

        self.manifest_dir = Path(manifest_dir)
        self.manifest = load_manifest(self.manifest_dir)
        self.analysis_name = self.manifest.get("analysis", "unnamed")
        self._packages: dict[str, OmniFoldPackage] = {}
        self._load_packages()

    def _resolve_package_path(self, package_path: str | Path) -> Path:
        path = Path(package_path)
        if path.is_absolute():
            return path
        return self.manifest_dir / path

    def _load_packages(self) -> None:
        samples = self.manifest.get("samples", {})
        if not isinstance(samples, dict) or "nominal" not in samples:
            raise PackageReadError("Manifest must declare samples.nominal.")

        for name, sample in samples.items():
            if not isinstance(sample, dict) or "path" not in sample:
                raise PackageReadError(f"Manifest sample {name!r} must define path.")
            self._packages[name] = load_package(self._resolve_package_path(sample["path"]))

    @property
    def nominal_package(self) -> OmniFoldPackage:
        """Return the nominal package."""

        return self._packages["nominal"]

    @property
    def target_package(self) -> OmniFoldPackage | None:
        """The declared truth/target companion package, if any.

        The target sample (role "target" in the manifest) is the known
        distribution pseudo-data was reweighted toward — the reference for
        closure tests. It is not a variation and contributes no
        uncertainty component.
        """

        samples = self.manifest.get("samples", {})
        target = samples.get("target") if isinstance(samples, dict) else None
        if isinstance(target, dict) and target.get("role") == "target":
            return self._packages.get("target")
        return None

    def load_events(
        self,
        columns: list[str] | None = None,
        variation: str = "nominal",
    ) -> pd.DataFrame:
        """Load events from the appropriate package for a variation."""

        package = self._package_for_variation(variation)
        return package.load_events(columns=columns)

    def get_weights(self, variation: str = "nominal") -> np.ndarray:
        """Get weights for any declared variation."""

        if variation in {"nominal", "final"}:
            return self.nominal_package.get_weights(variation)

        if variation in self._packages:
            return self._packages[variation].get_weights("nominal")

        nominal_weights = self.nominal_package.list_weights()
        if variation in nominal_weights:
            return self.nominal_package.get_weights(variation)

        column_name = f"weights_{variation}"
        if column_name in nominal_weights:
            return self.nominal_package.get_weights(column_name)

        try:
            return self.nominal_package.get_weights(variation)
        except PackageReadError:
            raise PackageReadError(f"Unknown analysis variation: {variation}")

    def list_variations(self) -> list[str]:
        """List all available variations."""

        variations = ["nominal"]
        variations.extend(list_manifest_variations(self.manifest))
        variations.extend(
            weight
            for weight in self.nominal_package.list_weights()
            if "ensemble" in weight and weight not in variations
        )
        return variations

    def get_replica_weights(self) -> np.ndarray:
        """Return all ensemble replica weights as a 2D array."""

        replica_names = [
            weight
            for weight in self.nominal_package.list_weights()
            if "ensemble" in weight
        ]
        if not replica_names:
            return np.empty((0, 0))
        return np.vstack(
            [self.nominal_package.get_weights(replica) for replica in replica_names]
        )

    def validate_all(self) -> None:
        """Validate all packages declared in manifest."""

        errors: list[str] = []
        for name, package in self._packages.items():
            try:
                package.validate()
            except PackageValidationError as exc:
                errors.append(f"{name}: {exc}")
        if errors:
            raise PackageValidationError("\n".join(errors))

    def summary(self) -> dict[str, Any]:
        """Return summary of the full analysis."""

        return {
            "analysis": self.analysis_name,
            "nominal": self.nominal_package.summary(),
            "variations": self.list_variations(),
            "replica_count": len(
                [
                    weight
                    for weight in self.nominal_package.list_weights()
                    if "ensemble" in weight
                ]
            ),
        }

    def histogram(
        self,
        observable: str,
        nominal_variation: str = "nominal",
        systematic_variations: list[str] | None = None,
        bins: list[float] | int | None = None,
    ) -> HistogramResult:
        """Compute a nominal histogram with statistical and uncertainty bands."""

        selected_bins = bins
        if selected_bins is None:
            selected_bins = self.nominal_package.observable_bins(observable) or 30
        nominal_values, nominal_mask = self._package_for_variation(
            nominal_variation
        ).observable_values(observable)
        nominal_weights = np.asarray(self.get_weights(nominal_variation))[
            nominal_mask
        ]
        nominal_result = compute_weighted_histogram(
            nominal_values,
            nominal_weights,
            bins=selected_bins,
        )
        common_edges = np.asarray(nominal_result["edges"], dtype=float)

        systematic_differences: list[np.ndarray] = []
        for variation in systematic_variations or []:
            varied_values, varied_mask = self._package_for_variation(
                variation
            ).observable_values(observable)
            varied_result = compute_weighted_histogram(
                varied_values,
                np.asarray(self.get_weights(variation))[varied_mask],
                bins=common_edges,
            )
            systematic_differences.append(
                np.asarray(varied_result["hist"], dtype=float)
            )
        # ATLAS combines systematic variations in quadrature, never as an
        # envelope (multifold_util.py calculate_uncertainty).
        sys_uncertainty = (
            quadrature_difference_from_nominal(
                np.asarray(nominal_result["hist"], dtype=float),
                np.vstack(systematic_differences),
            )
            if systematic_differences
            else None
        )

        replica_uncertainty = None
        replicas = self.get_replica_weights()
        if replicas.size and replicas.shape[1] == nominal_mask.shape[0]:
            replica_histograms = [
                np.asarray(
                    compute_weighted_histogram(
                        nominal_values,
                        replica[nominal_mask],
                        bins=common_edges,
                    )["hist"],
                    dtype=float,
                )
                for replica in replicas
            ]
            # NN-ensemble replicas follow the median-standard-error recipe
            # (1.253 * std / sqrt(N), 2_pseudo_results.ipynb cell 11).
            replica_uncertainty = ensemble_median_standard_error(
                np.vstack(replica_histograms)
            )

        return HistogramResult(
            hist=np.asarray(nominal_result["hist"], dtype=float),
            edges=common_edges,
            centers=np.asarray(nominal_result["centers"], dtype=float),
            stat_uncertainty=np.asarray(
                nominal_result["uncertainty"],
                dtype=float,
            ),
            sys_uncertainty=sys_uncertainty,
            replica_uncertainty=replica_uncertainty,
        )

    def uncertainty_breakdown(
        self,
        observable: str,
        bins: list[float] | int | None = None,
        include_two_point: bool = True,
    ) -> dict[str, Any]:
        """Full uncertainty breakdown: package families plus two-point terms.

        Extends the nominal package's per-family breakdown with one
        component per manifest variation, taken as the difference between
        the variation sample's nominal histogram and the nominal one — the
        two-point recipe the release uses for alternative-sample
        systematics (multifold_util.py, "Unfolding (HV)" and
        "Non-Strong Background" blocks). Components combine in quadrature.
        """

        breakdown = self.nominal_package.uncertainty_breakdown(
            observable, bins=bins
        )
        if not include_two_point:
            return breakdown

        components: dict[str, np.ndarray] = breakdown["components"]
        deltas = self._two_point_deltas(
            observable, breakdown["edges"], breakdown["nominal"]
        )
        for name, delta in deltas.items():
            components[f"two_point_{name}"] = np.abs(delta)

        breakdown["total"] = total_in_quadrature(components)
        return breakdown

    def _two_point_deltas(
        self,
        observable: str,
        edges: np.ndarray,
        nominal_hist: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Signed (h_variation - h_nominal) per manifest variation sample."""

        deltas: dict[str, np.ndarray] = {}
        for name in list_manifest_variations(self.manifest):
            package = self._packages[name]
            varied_values, varied_mask = package.observable_values(observable)
            varied_hist, _ = np.histogram(
                varied_values,
                bins=edges,
                weights=np.asarray(package.get_weights("nominal"))[varied_mask],
            )
            deltas[name] = varied_hist - nominal_hist
        return deltas

    def covariance_matrix(
        self,
        observable: str,
        bins: list[float] | int | None = None,
        include_two_point: bool = True,
        smooth_two_point: list[str] | None = None,
    ) -> dict[str, Any]:
        """Full covariance: package families plus two-point sample terms.

        Two-point terms enter in Hessian mode — one fully-bin-correlated
        matrix per alternative sample, as in corr_matrix's v_unfolding_hv
        and v_bkg (multifold_util.py:402-427). Variations named in
        ``smooth_two_point`` have their delta Gaussian-kernel smoothed
        first, as the release does for the hidden-variable (Sherpa) term
        in the closure covariance (2_pseudo_results.ipynb cell 26).
        """

        result = self.nominal_package.covariance_matrix(observable, bins=bins)
        if not include_two_point:
            return result

        components: dict[str, np.ndarray] = result["components"]
        edges = np.asarray(result["edges"], dtype=float)
        centers = 0.5 * (edges[1:] + edges[:-1])
        deltas = self._two_point_deltas(
            observable, result["edges"], result["nominal"]
        )
        smooth_names = set(smooth_two_point or [])
        unknown = smooth_names - set(deltas)
        if unknown:
            raise PackageReadError(
                "smooth_two_point names not declared as variations: "
                f"{', '.join(sorted(unknown))}."
            )
        for name, delta in deltas.items():
            if name in smooth_names:
                delta = smooth_uncertainty(delta, centers)
            components[f"two_point_{name}"] = fill_cov_matrix(
                delta[np.newaxis, :]
            )

        result["total"] = np.sum(list(components.values()), axis=0)
        result["correlation"] = correlation_matrix(result["total"])
        return result

    def compare(
        self,
        observable: str,
        bins: list[float] | None = None,
    ):
        """Return a comparison object for the given observable."""

        from .comparison import HistogramComparison

        if bins is None:
            events = self.load_events(columns=[observable])
            bins = np.histogram_bin_edges(
                events[observable].to_numpy(dtype=float),
                bins=30,
            ).tolist()
        return HistogramComparison(self, observable=observable, bins=bins)

    def _package_for_variation(self, variation: str) -> OmniFoldPackage:
        if variation in {"nominal", "final"}:
            return self.nominal_package
        if variation in self._packages:
            return self._packages[variation]
        if variation in self.nominal_package.list_weights():
            return self.nominal_package
        if f"weights_{variation}" in self.nominal_package.list_weights():
            return self.nominal_package
        from .reader import resolve_weight_column

        try:
            resolve_weight_column(
                self.nominal_package.metadata(), variation=variation
            )
        except PackageReadError:
            raise PackageReadError(f"Unknown analysis variation: {variation}")
        return self.nominal_package


def load_analysis(manifest_dir: str | Path) -> OmniFoldAnalysis:
    """Load a multi-sample OmniFold analysis from a manifest directory."""

    return OmniFoldAnalysis(manifest_dir)

"""OmniFold publication package helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .analysis import OmniFoldAnalysis, load_analysis
from .binning import equal_effective_events_bins, n_eff, validate_binning
from .closure import chi2_test
from .hepdata import HEPDataPackage, export_hepdata, load_hepdata_submission
from .plotting import plot_closure_grid
from .derived_observables import (
    DERIVED_OBSERVABLES,
    compute_derived_observables,
)
from .exceptions import (
    ManifestNotFoundError,
    OmniFoldPublicationError,
    PackageReadError,
    PackageValidationError,
    PackageWriteError,
    UnsupportedFormatVersion,
)
from .export import export_comparison_json, export_histogram_json
from .histogram import HistogramResult
from .manifest import load_manifest, write_manifest
from .publication import Publication
from .reader import (
    OmniFoldPackage,
    get_uncertainty,
    get_weights,
    list_systematics,
    load_events,
    load_metadata,
    load_package,
)
from .validation import (
    closure_test,
    ensure_valid_package,
    validate_normalization,
    validate_package,
)
from .cross_comparison import (
    CrossPublicationComparison,
    compare_publications,
)
from .writer import write_package

if TYPE_CHECKING:
    from .comparison import HistogramComparison

__all__ = [
    "OmniFoldAnalysis",
    "OmniFoldPackage",
    "CrossPublicationComparison",
    "compare_publications",
    "HistogramComparison",
    "HistogramResult",
    "ManifestNotFoundError",
    "OmniFoldPublicationError",
    "Publication",
    "PackageReadError",
    "PackageValidationError",
    "PackageWriteError",
    "UnsupportedFormatVersion",
    "closure_test",
    "ensure_valid_package",
    "export_comparison_json",
    "export_histogram_json",
    "get_uncertainty",
    "get_weights",
    "list_systematics",
    "load_analysis",
    "load_events",
    "load_manifest",
    "load_metadata",
    "load_package",
    "validate_normalization",
    "validate_package",
    "write_manifest",
    "write_package",
]


def __getattr__(name: str) -> Any:
    """Lazily import plotting functionality from the optional plot extra."""

    if name == "HistogramComparison":
        from .comparison import HistogramComparison

        return HistogramComparison
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

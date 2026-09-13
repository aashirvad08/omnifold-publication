"""Read the OmniFold publication package."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .exceptions import PackageReadError, UnsupportedFormatVersion
from .histogram import HistogramResult, compute_weighted_histogram
from .schema import Observable, PackagedWeightFamily, parse_metadata


SUPPORTED_FORMAT_VERSIONS = {"0.1", "0.2"}
DEFAULT_HDF_KEY = "df"
NOMINAL_CONVENTIONS = {"includes_mc_weight", "reweighting_factor"}
# Packages written before the convention field existed follow the ATLAS
# release layout, where weights_nominal already contains the MC weight.
DEFAULT_NOMINAL_CONVENTION = "includes_mc_weight"
# Keys inside metadata['weights'] that are not weight variations.
_NON_VARIATION_KEYS = {"iterations", "nominal_convention", "families"}


def _resolve_metadata_path(path: str | Path) -> Path:
    path = Path(path)
    return path / "metadata.yaml" if path.is_dir() else path


def _column_from_spec(spec: Any) -> str:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict) and isinstance(spec.get("column"), str):
        return spec["column"]
    raise PackageReadError(f"Weight specification does not define a column: {spec!r}")


def ensure_supported_format_version(metadata: dict[str, Any]) -> None:
    """Raise a clear error when package metadata uses an unsupported version."""

    version = metadata.get("format_version")
    if version not in SUPPORTED_FORMAT_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_FORMAT_VERSIONS))
        raise UnsupportedFormatVersion(
            f"Unsupported format_version {version!r}; supported versions: {supported}."
        )


def load_metadata(path: str | Path, enforce_version: bool = False) -> dict[str, Any]:
    """Load package metadata from a package directory or metadata file path."""

    metadata_path = _resolve_metadata_path(path)
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream) or {}
    if not isinstance(metadata, dict):
        raise PackageReadError("Package metadata must be a mapping.")
    if enforce_version:
        ensure_supported_format_version(metadata)
    return metadata


def _resolve_data_path(base_path: Path, data_path: str) -> Path:
    data = Path(data_path)
    if data.is_absolute():
        return data
    return base_path / data


def load_events(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Load event data declared by metadata or directly from an event file."""

    path = Path(path)
    if path.is_dir():
        metadata = load_metadata(path)
        files = metadata.get("files", {})
        nominal_file = files.get("nominal", {}) if isinstance(files, dict) else {}
        if isinstance(nominal_file, dict) and "path" in nominal_file:
            events_path = _resolve_data_path(path, nominal_file["path"])
            return pd.read_hdf(events_path, key=DEFAULT_HDF_KEY, columns=columns)

        publication = metadata.get("publication")
        if not isinstance(publication, dict) or "events_file" not in publication:
            raise PackageReadError(
                "Metadata must define either files.nominal.path or "
                "publication.events_file."
            )
        events_file = publication["events_file"]
        events_path = path / events_file
        return pd.read_parquet(events_path, columns=columns)
    else:
        events_path = path
        if events_path.suffix in {".h5", ".hdf5"}:
            return pd.read_hdf(events_path, key=DEFAULT_HDF_KEY, columns=columns)
        return pd.read_parquet(events_path, columns=columns)


def list_systematics(metadata: dict[str, Any]) -> list[str]:
    """Return systematic variation names declared in package metadata."""

    file_block = metadata.get("files", {})
    files = file_block.get("systematics", []) if isinstance(file_block, dict) else []
    if isinstance(files, list) and files:
        return sorted(
            Path(systematic["path"]).stem
            for systematic in files
            if isinstance(systematic, dict) and "path" in systematic
        )

    systematics = metadata.get("systematics", {})
    if isinstance(systematics, dict) and systematics:
        return sorted(systematics)

    weights = metadata.get("weights", {})
    if isinstance(weights, dict) and "replica" in weights:
        return ["replica"]
    return []


def resolve_weight_column(
    metadata: dict[str, Any],
    variation: str = "nominal",
    iteration: int | None = None,
    step: str | None = None,
) -> str:
    """Resolve a metadata-declared weight selection to a concrete column name."""

    weights = metadata.get("weights", {})
    if not isinstance(weights, dict):
        raise PackageReadError("Metadata key 'weights' must be a mapping.")

    if iteration is not None or step is not None:
        if iteration is None or step is None:
            raise PackageReadError(
                "Both iteration and step are required for iteration weights."
            )
        if step not in {"step1", "step2"}:
            raise PackageReadError("Iteration step must be 'step1' or 'step2'.")
        for entry in weights.get("iterations", []):
            if entry.get("iteration") == iteration and step in entry:
                return _column_from_spec(entry[step])
        raise PackageReadError(
            f"No weight column declared for iteration={iteration}, step={step!r}."
        )

    if variation == "nominal":
        if "nominal" not in weights:
            raise PackageReadError("Metadata does not declare a nominal weight.")
        return _column_from_spec(weights["nominal"])

    if (
        variation in weights
        and variation not in _NON_VARIATION_KEYS
        and isinstance(weights[variation], str)
    ):
        return weights[variation]

    families = weights.get("families")
    if isinstance(families, dict):
        for family in families.values():
            if not isinstance(family, dict):
                continue
            if variation in family.get("columns", []):
                return variation
            if variation == family.get("reference_column"):
                return variation

    raise PackageReadError(f"Unknown metadata-declared weight variation: {variation}")


def get_weights(
    df: pd.DataFrame,
    metadata: dict[str, Any],
    variation: str = "nominal",
    iteration: int | None = None,
    step: str | None = None,
):
    """Return the requested weight array from the loaded event table."""

    column = resolve_weight_column(
        metadata,
        variation=variation,
        iteration=iteration,
        step=step,
    )
    if column not in df.columns:
        raise PackageReadError(
            f"Weight column {column!r} is not present in the event table."
        )
    return df[column].to_numpy()


def get_uncertainty(
    df: pd.DataFrame,
    metadata: dict[str, Any],
    variation: str,
) -> np.ndarray:
    """Return per-event absolute difference between a variation and nominal weights."""

    nominal = get_weights(df, metadata, variation="nominal")
    varied = get_weights(df, metadata, variation=variation)
    return np.abs(varied - nominal)


class OmniFoldPackage:
    """Thin wrapper matching the proposal-facing package API.

    Metadata is parsed once, at construction, into the typed
    :class:`~omnifold_publication.schema.Metadata` model; every accessor
    below navigates that model. Malformed metadata is therefore reported
    once here, naming the offending field, instead of each accessor
    re-checking the shape of the block it happens to read.
    """

    def __init__(self, package_dir: str | Path):
        self.package_dir = Path(package_dir)
        self._metadata = load_metadata(self.package_dir, enforce_version=True)
        try:
            self._model = parse_metadata(self._metadata)
        except ValueError as exc:
            raise PackageReadError(
                f"Package metadata at {self.package_dir} does not match the "
                f"package schema.\n{exc}"
            ) from exc

    def load_events(self, columns: list[str] | None = None) -> pd.DataFrame:
        """Load event columns from this package."""

        return load_events(self.package_dir, columns=columns)

    def list_systematics(self) -> list[str]:
        """Return systematic names declared by this package."""

        return list_systematics(self._metadata)

    def list_weights(self) -> list[str]:
        """Return all declared weight variation names.

        ``exclude_unset`` keeps this to what the file actually declares —
        the model's unset defaults are not weights — and preserves
        declaration order, which ``get_replica_weights`` relies on to
        build a stable replica matrix.
        """

        declared = self._model.weights.model_dump(exclude_unset=True)
        return [key for key in declared if key not in _NON_VARIATION_KEYS]

    def list_observables(self) -> list[str]:
        """Return all declared observable names."""

        return [observable.name for observable in self._model.observables]

    def observable_units(self, name: str) -> str:
        """Return the declared units for an observable, or an empty string."""

        return self._observable_entry(name).units

    def _observable_entry(self, name: str) -> Observable:
        for observable in self._model.observables:
            if observable.name == name:
                return observable
        raise PackageReadError(f"Unknown observable: {name}")

    def observable_bins(self, name: str) -> list[float] | None:
        """Return bin edges: explicit bins, then official binning, then
        suggested bins."""

        observable = self._observable_entry(name)
        bins = observable.bins
        if bins is None and observable.binning is not None:
            bins = observable.binning.official
        if bins is None:
            bins = observable.suggested_bins
        if bins is None:
            return None
        return [float(edge) for edge in bins]

    def observable_binning_provenance(self, name: str) -> str | None:
        """Provenance of the official binning, if declared."""

        binning = self._observable_entry(name).binning
        return binning.provenance if binning is not None else None

    def observable_selection(self, name: str) -> str | None:
        """Declared event selection for an observable (e.g. 'pT_trackj1 > 5')."""

        return self._observable_entry(name).selection

    def observable_values(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Observable values after its declared selection, plus the mask.

        Returns ``(values, mask)`` where ``mask`` is a boolean array over
        all events (all True when no selection is declared) and ``values``
        are the observable's values with the mask applied. Weight arrays
        must be masked with the same ``mask`` before histogramming —
        the release applies exactly this pattern for trackjet observables
        (pT > 5 GeV masks in multifold_util.py corr_matrix).
        """

        from .selection import selection_mask

        expression = self.observable_selection(name)
        columns = [name]
        if expression is not None:
            from .selection import selection_columns

            columns.extend(
                column
                for column in selection_columns(expression)
                if column != name
            )
        events = self.load_events(columns=columns)
        if expression is None:
            mask = np.ones(len(events), dtype=bool)
        else:
            mask = selection_mask(events, expression)
        return events[name].to_numpy(dtype=float)[mask], mask

    def summary(self) -> dict[str, Any]:
        """Return a concise summary of the package contents."""

        publication = self._model.publication
        checksum = publication.checksum_sha256 if publication is not None else None
        return {
            "format_version": self._model.format_version,
            "event_count": publication.event_count if publication is not None else None,
            "observables": self.list_observables(),
            "weights": self.list_weights(),
            "systematics": self.list_systematics(),
            "checksum_sha256": checksum if checksum is not None else "not recorded",
        }

    def get_weights(
        self,
        variation: str = "nominal",
        iteration: int | None = None,
        step: str | None = None,
        kind: str | None = None,
    ) -> np.ndarray:
        """Return a declared weight array or the derived final event weights.

        ``variation`` names a metadata-declared weight — "nominal", a
        declared variation, or any column of a declared weight family —
        or "final" for the convention-aware measurement weight (see
        :meth:`nominal_convention`). Pass ``iteration`` and ``step``
        together to select a declared iteration weight.

        ``kind`` is a deprecated alias for ``variation``, kept so existing
        callers keep working; it will be removed in a future release.
        """

        if kind is not None:
            if variation != "nominal":
                raise PackageReadError(
                    "Pass either `variation` or the deprecated `kind`, not "
                    "both; they name the same thing."
                )
            warnings.warn(
                "OmniFoldPackage.get_weights(kind=...) is deprecated; use "
                "variation=... instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            variation = kind

        if variation == "final" and iteration is None and step is None:
            return self._final_weights()

        column = resolve_weight_column(
            self._metadata,
            variation=variation,
            iteration=iteration,
            step=step,
        )
        df = self.load_events(columns=[column])
        return get_weights(
            df,
            self._metadata,
            variation=variation,
            iteration=iteration,
            step=step,
        )

    def nominal_convention(self) -> str:
        """Return the declared relation between weights_nominal and weight_mc."""

        convention = (
            self._model.weights.nominal_convention or DEFAULT_NOMINAL_CONVENTION
        )
        if convention not in NOMINAL_CONVENTIONS:
            allowed = ", ".join(sorted(NOMINAL_CONVENTIONS))
            raise PackageReadError(
                f"Unknown weights.nominal_convention {convention!r}; "
                f"expected one of: {allowed}."
            )
        return convention

    def _families(self) -> dict[str, PackagedWeightFamily]:
        return self._model.weights.families or {}

    def list_weight_families(self) -> list[str]:
        """Return the names of declared weight families."""

        return list(self._families())

    def weight_family(self, name: str) -> dict[str, Any]:
        """Return the metadata block of one declared weight family.

        Returned as a plain mapping (``type``, ``combination``,
        ``columns``, ``reference_column``) so callers keep using ``.get()``
        against a stable public shape; it is produced from the validated
        model, so every key is guaranteed present and correctly typed.
        """

        families = self._families()
        if name not in families:
            raise PackageReadError(f"Unknown weight family: {name!r}")
        return families[name].model_dump()

    def get_family_weights(self, name: str) -> np.ndarray:
        """Return a family's weight columns as a (n_columns, n_events) matrix."""

        family = self.weight_family(name)
        columns = [
            column
            for column in family.get("columns", [])
            if isinstance(column, str)
        ]
        if not columns:
            raise PackageReadError(f"Weight family {name!r} declares no columns.")
        df = self.load_events(columns=columns)
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise PackageReadError(
                f"Weight family {name!r} columns missing from event table: "
                f"{', '.join(missing)}."
            )
        return df[columns].to_numpy(dtype=float).T

    def uncertainty_breakdown(
        self,
        observable: str,
        bins: list[float] | int | None = None,
    ) -> dict[str, Any]:
        """Per-family uncertainty breakdown following the ATLAS recipes."""

        from .uncertainty import uncertainty_breakdown

        return uncertainty_breakdown(self, observable, bins=bins)

    def covariance_matrix(
        self,
        observable: str,
        bins: list[float] | int | None = None,
    ) -> dict[str, Any]:
        """Per-family covariance matrices, total, and correlation matrix."""

        from .uncertainty import covariance_breakdown

        return covariance_breakdown(self, observable, bins=bins)

    def cross_section(self) -> float:
        """Sum of final weights: the measured fiducial cross-section.

        Meaningful when normalization.mode is "absolute" (the ATLAS release
        semantics, 1_basics.ipynb cells 10-11); units via weight_units().
        """

        return float(np.sum(self.get_weights("final")))

    def weight_units(self) -> str | None:
        """Return declared weight units (e.g. "fb"), if recorded."""

        return self._model.normalization.weight_units

    def _load_final_weight_columns(self, columns: list[str]) -> pd.DataFrame:
        required = ", ".join(repr(column) for column in columns)
        try:
            df = self.load_events(columns=columns)
        except Exception as exc:
            raise PackageReadError(
                f"Cannot compute final weights; required columns {required} "
                "must be present."
            ) from exc
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise PackageReadError(
                "Cannot compute final weights; missing event columns: "
                f"{', '.join(missing)}."
            )
        return df

    def _final_weights(self) -> np.ndarray:
        """Return the canonical per-event weight for central-value observables.

        Under the 'includes_mc_weight' convention (the ATLAS Z+jets release
        layout) weights_nominal is the complete measurement weight, so the
        final weight is weights_nominal unchanged. Under 'reweighting_factor'
        the nominal column stores only the learned OmniFold factor and must
        be multiplied by the base MC weight. See spec/weight_formula.md.
        """

        # The schema requires weights.nominal and weights.base_mc_weight, so
        # a package that reached this point declares both: the missing-field
        # cases are reported at load time by parse_metadata, with the field
        # named, rather than being re-checked here.
        weights = self._model.weights
        convention = self.nominal_convention()
        nominal_column = weights.nominal

        if convention == "includes_mc_weight":
            df = self._load_final_weight_columns([nominal_column])
            return df[nominal_column].to_numpy(dtype=float)

        base_column = weights.base_mc_weight
        if base_column is None:
            raise PackageReadError(
                "Cannot compute final weights: nominal_convention is "
                "'reweighting_factor', which multiplies the nominal column by "
                "the base MC weight, but this package declares no "
                "weights.base_mc_weight."
            )
        df = self._load_final_weight_columns([base_column, nominal_column])
        return (
            df[base_column].to_numpy(dtype=float)
            * df[nominal_column].to_numpy(dtype=float)
        )

    def histogram(
        self,
        observable: str,
        variation: str = "nominal",
        bins: list[float] | int | None = None,
    ) -> HistogramResult:
        """Compute a weighted histogram for one observable and variation."""

        selected_bins = bins
        if selected_bins is None:
            selected_bins = self.observable_bins(observable) or 30
        values, mask = self.observable_values(observable)
        weights = np.asarray(self.get_weights(variation))[mask]
        result = compute_weighted_histogram(
            values,
            weights,
            bins=selected_bins,
        )
        return HistogramResult(
            hist=np.asarray(result["hist"], dtype=float),
            edges=np.asarray(result["edges"], dtype=float),
            centers=np.asarray(result["centers"], dtype=float),
            stat_uncertainty=np.asarray(result["uncertainty"], dtype=float),
        )

    def get_uncertainty(self, variation: str) -> np.ndarray:
        """Return per-event absolute differences from nominal weights."""

        nominal_column = resolve_weight_column(self._metadata, variation="nominal")
        variation_column = resolve_weight_column(self._metadata, variation=variation)
        columns = list(dict.fromkeys([nominal_column, variation_column]))
        df = self.load_events(columns=columns)
        return get_uncertainty(df, self._metadata, variation=variation)

    def metadata(self) -> dict[str, Any]:
        """Return this package's loaded metadata mapping."""

        return self._metadata

    def validate(self) -> None:
        """Validate package metadata, contents, and integrity."""

        from .validation import ensure_valid_package

        ensure_valid_package(self.package_dir)

    def export_hepdata(self, output_dir: str | Path, **kwargs: Any) -> Path:
        """Write a HEPData submission (submission.yaml + table YAMLs)."""

        from .hepdata import export_hepdata

        return export_hepdata(self, output_dir, **kwargs)


def load_package(package_dir: str | Path):
    """Load a publication package and return the proposal-style wrapper.

    Accepts a package directory (metadata.yaml + events.parquet), a
    HEPData submission directory (submission.yaml + table YAMLs), or a
    remote HEPData record reference ("hepdata:ins2791852"); the latter
    two load as a binned
    :class:`~omnifold_publication.hepdata.HEPDataPackage`.
    """

    if isinstance(package_dir, str) and package_dir.startswith("hepdata:"):
        from .hepdata import load_hepdata_record

        return load_hepdata_record(package_dir.removeprefix("hepdata:"))

    path = Path(package_dir)
    if path.is_dir() and not (path / "metadata.yaml").exists() and (
        path / "submission.yaml"
    ).exists():
        from .hepdata import load_hepdata_submission

        return load_hepdata_submission(path)
    return OmniFoldPackage(package_dir)

"""Write the OmniFold publication package to Parquet."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any

import pandas as pd
import yaml

from .derived_observables import DERIVED_OBSERVABLES, compute_derived_observables
from .exceptions import PackageWriteError
from .selection import selection_columns


DEFAULT_INPUT_PATH = Path("data/multifold.h5")
DEFAULT_METADATA_SOURCE = Path("spec/metadata.yaml")
DEFAULT_OUTPUT_DIR = Path("artifacts/demo_nominal")
DEFAULT_EVENT_COUNT = 10_000
PRIMARY_OBSERVABLE = "pT_ll"
EXTRA_OBSERVABLE = "pT_l1"
EVENT_ID_COLUMN = "event_id"
BASE_WEIGHT_COLUMN = "weight_mc"
NOMINAL_WEIGHT_COLUMN = "weights_nominal"
# How weights_nominal relates to weight_mc. The ATLAS Z+jets release ships
# weights_nominal with the MC weight already folded in ("includes_mc_weight");
# "reweighting_factor" covers files that store only the learned factor.
NOMINAL_CONVENTIONS = ("includes_mc_weight", "reweighting_factor")
DEFAULT_NOMINAL_CONVENTION = "includes_mc_weight"

# Systematic NP grouping follows multifold_util.py:335-341 (event = muEff* +
# pileup, theory*, track*, muCal*). Luminosity and top-background get their
# own single-column families because the release's binned code treats them
# individually (v_lumi and the topBackground term of v_bkg in corr_matrix;
# the closure chi2 of 2_pseudo_results cell 26 includes topBackground but
# NOT lumi). Anything unmatched falls into "syst_other". All groups combine
# in quadrature, so the split only affects labels and component selection,
# never the full total.
SYSTEMATIC_GROUP_PREFIXES = (
    ("syst_event", ("weights_muEff", "weights_pileup")),
    ("syst_theory", ("weights_theory",)),
    ("syst_track", ("weights_track",)),
    ("syst_muon", ("weights_muCal",)),
    ("syst_lumi", ("weights_lumi",)),
    ("syst_background", ("weights_topBackground",)),
)
REPLICA_FAMILY_RULES = (
    ("bootstrap_mc", "weights_bootstrap_mc_", "bootstrap", "standard_deviation"),
    ("bootstrap_data", "weights_bootstrap_data_", "bootstrap", "standard_deviation"),
    ("ensemble", "weights_ensemble_", "ensemble", "median_standard_error"),
)
PAIRED_DD_COLUMN = "weights_dd"
PAIRED_DD_REFERENCE = "target_dd"
NORMALIZATION_MODES = ("absolute", "shape")
# The ATLAS release weights are absolute cross-sections in femtobarns:
# sum(weights_nominal) over any selection is the measured fiducial
# cross-section of that region (1_basics.ipynb cells 10-11).
DEFAULT_NORMALIZATION_MODE = "absolute"
DEFAULT_WEIGHT_UNITS = "fb"
ITERATION_PATTERNS = (
    re.compile(r"^weights_(step[12])_(?:iter|iteration)_?(\d+)$"),
    re.compile(r"^weights_(?:iter|iteration)_?(\d+)_(step[12])$"),
)
REPLICA_PREFIXES = ("weights_ensemble_", "weights_bootstrap_mc_")
ALL_REPLICA_PREFIXES = (
    "weights_ensemble_",
    "weights_bootstrap_mc_",
    "weights_bootstrap_data_",
)
FORMAT_VERSION = "0.2"


def _compute_checksum(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""

    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _read_events(input_path: Path) -> pd.DataFrame:
    """Load event rows from an HDF5 or Parquet source.

    HDF5 is read exactly as before (key ``"df"``). Parquet is read with
    only its scalar columns: variable-length / particle-level array columns
    (e.g. ``truth_pT_particles``) are skipped, since binned-observable
    publication never uses them and they dominate the file size. This keeps
    the existing HDF5 path byte-for-byte unchanged.
    """

    if input_path.suffix.lower() == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = pq.ParquetFile(input_path).schema_arrow
        scalar = [
            name
            for name, dtype in zip(schema.names, schema.types, strict=True)
            if not (pa.types.is_list(dtype) or pa.types.is_large_list(dtype))
        ]
        return pd.read_parquet(input_path, columns=scalar)
    return pd.read_hdf(input_path, "df")


def _load_source_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise PackageWriteError("Source metadata must be a mapping.")
    return data


def _find_replica_column(columns: list[str]) -> str | None:
    for prefix in REPLICA_PREFIXES:
        for column in columns:
            if column.startswith(prefix):
                return column
    return None


def _find_replica_columns(columns: list[str]) -> list[str]:
    return [
        column
        for column in columns
        if any(column.startswith(prefix) for prefix in ALL_REPLICA_PREFIXES)
    ]


def _discover_iteration_weights(columns: list[str]) -> list[dict[str, Any]]:
    """Find step1/step2 iteration weights when the source file provides them.

    Optional feature: the ATLAS Z+jets release files publish only final
    (nominal + variation) weights and contain no iteration columns, so this
    discovery yields an empty list for the actual analysis data. It is kept
    for source files that do store intermediate OmniFold iterations.
    """

    by_iteration: dict[int, dict[str, dict[str, str]]] = {}

    for column in columns:
        for pattern in ITERATION_PATTERNS:
            match = pattern.match(column)
            if match is None:
                continue
            first, second = match.groups()
            if first.startswith("step"):
                step = first
                iteration = int(second)
            else:
                iteration = int(first)
                step = second
            by_iteration.setdefault(iteration, {})[step] = {"column": column}
            break

    return [
        {"iteration": iteration, **steps}
        for iteration, steps in sorted(by_iteration.items())
    ]


def _discover_systematic_families(columns: list[str]) -> dict[str, dict[str, Any]]:
    """Group NP systematic weight columns into ATLAS-style families.

    A systematic column is any ``weights_*`` column that is not the nominal,
    a replica (bootstrap/ensemble), an iteration weight, or the paired
    ``weights_dd`` column.
    """

    excluded = {NOMINAL_WEIGHT_COLUMN, BASE_WEIGHT_COLUMN, PAIRED_DD_COLUMN}
    systematic_columns = [
        column
        for column in columns
        if column.startswith("weights_")
        and column not in excluded
        and not any(column.startswith(prefix) for prefix in ALL_REPLICA_PREFIXES)
        and not any(pattern.match(column) for pattern in ITERATION_PATTERNS)
    ]

    families: dict[str, dict[str, Any]] = {}
    remaining = list(systematic_columns)
    for family_name, prefixes in SYSTEMATIC_GROUP_PREFIXES:
        members = sorted(
            column
            for column in remaining
            if any(column.startswith(prefix) for prefix in prefixes)
        )
        if members:
            families[family_name] = {
                "type": "systematic",
                "combination": "quadrature_difference_from_nominal",
                "columns": members,
            }
            remaining = [column for column in remaining if column not in members]
    if remaining:
        families["syst_other"] = {
            "type": "systematic",
            "combination": "quadrature_difference_from_nominal",
            "columns": sorted(remaining),
        }

    if PAIRED_DD_COLUMN in columns and PAIRED_DD_REFERENCE in columns:
        families["dd_unfolding"] = {
            "type": "paired",
            "combination": "paired_relative_difference",
            "columns": [PAIRED_DD_COLUMN],
            "reference_column": PAIRED_DD_REFERENCE,
        }
    return families


def _discover_replica_families(columns: list[str]) -> dict[str, dict[str, Any]]:
    families: dict[str, dict[str, Any]] = {}
    for family_name, prefix, family_type, combination in REPLICA_FAMILY_RULES:
        members = sorted(column for column in columns if column.startswith(prefix))
        if members:
            families[family_name] = {
                "type": family_type,
                "combination": combination,
                "columns": members,
            }
    return families


def _family_columns(families: dict[str, dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    for family in families.values():
        columns.extend(family.get("columns", []))
        reference = family.get("reference_column")
        if isinstance(reference, str):
            columns.append(reference)
    return columns


def _build_systematics(
    replica_column: str | None,
    families: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, str]]:
    systematics: dict[str, dict[str, str]] = {}
    if replica_column is not None:
        systematics["replica"] = {
            "column": replica_column,
            "type": "ensemble",
            "combination": "absolute_difference_from_nominal",
        }
    for name, family in (families or {}).items():
        if family.get("type") in {"systematic", "paired"}:
            systematics[name] = {
                "family": name,
                "type": family["type"],
                "combination": family["combination"],
            }
    return systematics


def _build_normalization(
    nominal_sumw: float,
    mode: str = DEFAULT_NORMALIZATION_MODE,
    weight_units: str | None = DEFAULT_WEIGHT_UNITS,
) -> dict[str, Any]:
    """Normalization semantics of the packaged weights.

    Under "absolute" mode the nominal weights are cross-sections: summing
    them over any selection yields the measured fiducial cross-section of
    that region (1_basics.ipynb cells 10-11). "shape" is kept for sources
    whose weights carry no absolute scale.
    """

    normalization: dict[str, Any] = {
        "mode": mode,
        "base_weight_column": BASE_WEIGHT_COLUMN,
        "nominal_weight_column": NOMINAL_WEIGHT_COLUMN,
        "expected_nominal_sumw": nominal_sumw,
        "tolerance": 1.0e-8,
    }
    if weight_units is not None:
        normalization["weight_units"] = weight_units
    if mode == "absolute":
        normalization["sum_weights_equals"] = "fiducial_cross_section"
    return normalization


def _filter_observables(
    source_metadata: dict[str, Any],
    selected_names: list[str],
) -> list[dict[str, Any]]:
    observables = source_metadata.get("observables", [])
    if not isinstance(observables, list):
        return [{"name": name} for name in selected_names]

    by_name = {
        observable["name"]: observable
        for observable in observables
        if isinstance(observable, dict) and "name" in observable
    }
    entries: list[dict[str, Any]] = []
    for name in selected_names:
        entry = dict(by_name.get(name, {"name": name}))
        # derived observables keep their registry selection/inputs even when
        # the source metadata does not describe them
        if name in DERIVED_OBSERVABLES:
            registry = DERIVED_OBSERVABLES[name]
            entry.setdefault("derived_from", list(registry["inputs"]))
            if registry["selection"] is not None:
                entry.setdefault("selection", registry["selection"])
            entry.setdefault("description", registry["description"])
            entry.setdefault("units", registry["units"])
        entries.append(entry)
    return entries


def _build_package_metadata(
    source_metadata: dict[str, Any],
    observable_entries: list[dict[str, Any]],
    selected_columns: list[str],
    replica_column: str | None,
    iteration_weights: list[dict[str, Any]],
    event_count: int,
    nominal_sumw: float,
    input_path: Path,
    has_event_id: bool,
    replica_columns: list[str] | None = None,
    nominal_convention: str = DEFAULT_NOMINAL_CONVENTION,
    families: dict[str, dict[str, Any]] | None = None,
    normalization_mode: str = DEFAULT_NORMALIZATION_MODE,
    weight_units: str | None = DEFAULT_WEIGHT_UNITS,
    method: str | None = None,
    assumptions: list[str] | None = None,
) -> dict[str, Any]:
    weights: dict[str, Any] = {
        "nominal": NOMINAL_WEIGHT_COLUMN,
        "base_mc_weight": BASE_WEIGHT_COLUMN,
        "nominal_convention": nominal_convention,
    }
    if replica_column is not None:
        weights["replica"] = replica_column
    for column in replica_columns or []:
        weights[column] = column
    if families:
        weights["families"] = families
    if iteration_weights:
        weights["iterations"] = iteration_weights

    dataset = dict(source_metadata.get("dataset", {}))
    if method is not None:
        dataset["method"] = method
    if assumptions:
        dataset["assumptions"] = list(assumptions)

    metadata: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "dataset": dataset,
        "observables": observable_entries,
        "weights": weights,
        "systematics": _build_systematics(replica_column, families),
        "normalization": _build_normalization(
            nominal_sumw,
            mode=normalization_mode,
            weight_units=weight_units,
        ),
        "publication": {
            "format": "parquet",
            "events_file": "events.parquet",
            "event_count": event_count,
            "columns": selected_columns,
            "source_file": input_path.as_posix(),
            "event_alignment": {
                "method": "column" if has_event_id else "row_order",
                "column": EVENT_ID_COLUMN if has_event_id else None,
            },
        },
    }
    return metadata


def write_package(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    metadata_source: str | Path = DEFAULT_METADATA_SOURCE,
    event_count: int | None = DEFAULT_EVENT_COUNT,
    observables: list[str] | None = None,
    include_all_replicas: bool = False,
    include_systematics: bool = True,
    nominal_convention: str = DEFAULT_NOMINAL_CONVENTION,
    normalization_mode: str = DEFAULT_NORMALIZATION_MODE,
    weight_units: str | None = DEFAULT_WEIGHT_UNITS,
    column_rename: dict[str, str] | None = None,
    method: str | None = None,
    assumptions: list[str] | None = None,
) -> Path:
    """Create a minimal Parquet-backed publication package.

    ``include_systematics`` packages every discovered NP systematic weight
    column plus the paired (weights_dd, target_dd) columns as declared
    weight families. ``include_all_replicas`` additionally packages the
    full bootstrap/ensemble replica sets as families; by default only the
    single first replica column is kept, preserving small demo packages.

    The remaining parameters exist to publish an independently produced
    result (e.g. an OmniFold parquet) through the same single path, and all
    default to the pre-existing behaviour:

    - ``input_path`` may be ``.parquet`` as well as ``.h5``.
    - ``column_rename`` renames source columns before packaging, e.g.
      ``{"truth_pT_ll": "pT_ll", "weights_prior": "weight_mc"}`` to align a
      differently-named schema to the package conventions.
    - ``method`` records the unfolding method ("MultiFold"/"OmniFold") and
      ``assumptions`` records publication-time caveats; both travel in the
      package metadata's ``dataset`` block.
    - ``event_count=None`` publishes the whole sample and additionally records
      the resulting luminosity assumption in ``assumptions`` automatically.
    """

    if nominal_convention not in NOMINAL_CONVENTIONS:
        allowed = ", ".join(NOMINAL_CONVENTIONS)
        raise PackageWriteError(
            f"Unknown nominal_convention {nominal_convention!r}; "
            f"expected one of: {allowed}."
        )
    if normalization_mode not in NORMALIZATION_MODES:
        allowed = ", ".join(NORMALIZATION_MODES)
        raise PackageWriteError(
            f"Unknown normalization_mode {normalization_mode!r}; "
            f"expected one of: {allowed}."
        )

    input_path = Path(input_path)
    if not input_path.exists():
        raise PackageWriteError(
            f"Input file not found: {input_path}. "
            f"Make sure data/multifold.h5 is present locally."
        )

    output_dir = Path(output_dir)
    metadata_source = Path(metadata_source)

    df = _read_events(input_path).iloc[:event_count].copy()
    if column_rename:
        df = df.rename(columns=column_rename)
    source_columns = list(df.columns)
    replica_column = _find_replica_column(source_columns)
    replica_columns = _find_replica_columns(source_columns) if include_all_replicas else []
    iteration_weights = _discover_iteration_weights(source_columns)

    families: dict[str, dict[str, Any]] = {}
    if include_systematics:
        families.update(_discover_systematic_families(source_columns))
    if include_all_replicas:
        families.update(_discover_replica_families(source_columns))

    observable_names = observables or [PRIMARY_OBSERVABLE, EXTRA_OBSERVABLE]
    derived_requested = [
        name
        for name in observable_names
        if name not in df.columns and name in DERIVED_OBSERVABLES
    ]
    if derived_requested:
        df = compute_derived_observables(df, derived_requested)

    source_metadata = _load_source_metadata(metadata_source)
    observable_entries = _filter_observables(source_metadata, observable_names)

    selected_columns = [
        *observable_names,
        BASE_WEIGHT_COLUMN,
        NOMINAL_WEIGHT_COLUMN,
    ]
    for entry in observable_entries:
        expression = entry.get("selection")
        if not isinstance(expression, str):
            continue
        for column in selection_columns(expression):
            if column not in df.columns:
                raise PackageWriteError(
                    f"Selection for observable {entry.get('name')!r} "
                    f"references missing column {column!r}."
                )
            selected_columns.append(column)
    if EVENT_ID_COLUMN in df.columns:
        selected_columns.append(EVENT_ID_COLUMN)
    if replica_column is not None:
        selected_columns.append(replica_column)
    selected_columns.extend(replica_columns)
    selected_columns.extend(_family_columns(families))
    for iteration in iteration_weights:
        for step in ("step1", "step2"):
            step_spec = iteration.get(step)
            if isinstance(step_spec, dict):
                selected_columns.append(step_spec["column"])

    selected_columns = list(dict.fromkeys(selected_columns))

    package_df = df.loc[:, selected_columns]
    package_event_count = int(len(package_df))

    # Publishing the whole sample means the weighted sum is the expected yield
    # at the measurement luminosity. That is an assumption a downstream
    # comparison depends on, so record it in the package rather than relying on
    # the caller to have passed it.
    if event_count is None:
        full_sample_note = (
            "all events used; weighted sum is the expected yield at the "
            "measurement luminosity (no event-count subsetting)"
        )
        assumptions = list(assumptions or [])
        if full_sample_note not in assumptions:
            assumptions.append(full_sample_note)
    nominal_sumw = float(package_df[NOMINAL_WEIGHT_COLUMN].to_numpy(dtype=float).sum())
    package_metadata = _build_package_metadata(
        source_metadata=source_metadata,
        observable_entries=observable_entries,
        selected_columns=selected_columns,
        replica_column=replica_column,
        iteration_weights=iteration_weights,
        event_count=package_event_count,
        nominal_sumw=nominal_sumw,
        input_path=input_path,
        has_event_id=EVENT_ID_COLUMN in package_df.columns,
        replica_columns=replica_columns,
        nominal_convention=nominal_convention,
        families=families,
        normalization_mode=normalization_mode,
        weight_units=weight_units,
        method=method,
        assumptions=assumptions,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.parquet"
    package_df.to_parquet(events_path, index=False)
    package_metadata["publication"]["checksum_sha256"] = _compute_checksum(events_path)
    with (output_dir / "metadata.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(package_metadata, stream, sort_keys=False)

    return output_dir

"""Write a publication package from an explicit declaration.

:func:`~omnifold_publication.writer.write_package` discovers weight
families from column-name conventions, which are the ATLAS Z+jets
release's conventions. That is convenient for files shaped like that
release and wrong for everything else: a column the rules do not
recognise is assumed to be a systematic and folded into ``syst_other``,
so a prior/target/scale-factor column under an unfamiliar name silently
becomes a 1-sigma nuisance parameter and inflates the published
uncertainty.

This module is the other entry point: **nothing is inferred**. You state
the observables, the weights and the families; columns you do not declare
are not packaged, and a family whose recipe or columns do not make sense
is rejected before anything is written. It is the supported path for an
unfolding output whose columns are named nothing like the reference
release.

    write_package_from_declaration(
        events="my_unfolding.parquet",
        output_dir="artifacts/mine",
        observables={
            "pt_dilepton": {
                "units": "GeV",
                "description": "Dilepton transverse momentum",
                "bins": [200, 300, 450, 1000],
                "binning_provenance": "analysis note table 3",
            },
        },
        weights={"nominal": "w_unfolded", "base_mc_weight": None},
        families={
            "detector": {
                "type": "systematic",
                "combination": "quadrature_difference_from_nominal",
                "columns": ["w_jes_up", "w_jer_up"],
            },
        },
    )

``base_mc_weight`` may be ``None``: a sample carrying a single weight
column (an alternative generator, a background sample) has no separate
prior weight, and inventing one to satisfy the layout is worse than
declaring its absence. Under the default ``includes_mc_weight``
convention the base weight is never used numerically; it is required only
by ``reweighting_factor``, which is checked here rather than failing
later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .exceptions import PackageWriteError
from .schema import parse_metadata
from .selection import selection_columns
from .uncertainty import COMBINATIONS
from .writer import (
    FORMAT_VERSION,
    NOMINAL_CONVENTIONS,
    NORMALIZATION_MODES,
    _compute_checksum,
    _read_events,
)

PAIRED_COMBINATION = "paired_relative_difference"


def _events_frame(events: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(events, pd.DataFrame):
        return events
    path = Path(events)
    if not path.exists():
        raise PackageWriteError(f"Events file not found: {path}")
    return _read_events(path)


def _observable_entries(
    observables: dict[str, dict[str, Any]] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalise the declared observables into schema entries."""

    if isinstance(observables, dict):
        items = [(name, spec or {}) for name, spec in observables.items()]
    elif isinstance(observables, list):
        items = []
        for spec in observables:
            if not isinstance(spec, dict) or "name" not in spec:
                raise PackageWriteError(
                    "Each observable in a list must be a mapping with a "
                    f"'name' key; got: {spec!r}"
                )
            items = [*items, (spec["name"], {k: v for k, v in spec.items() if k != "name"})]
    else:
        raise PackageWriteError(
            "`observables` must be a {name: spec} mapping or a list of "
            "mappings with a 'name' key."
        )
    if not items:
        raise PackageWriteError("`observables` declares no observables.")

    entries: list[dict[str, Any]] = []
    for name, spec in items:
        entry: dict[str, Any] = {
            "name": name,
            "description": spec.get("description", ""),
            "units": spec.get("units", ""),
        }
        bins = spec.get("bins")
        if bins is not None:
            # declared edges become the *official* binning, so closure tests
            # and HEPData export -- which refuse to guess -- can use them
            binning: dict[str, Any] = {"official": [float(e) for e in bins]}
            provenance = spec.get("binning_provenance")
            if provenance is not None:
                binning["provenance"] = provenance
            entry["binning"] = binning
        for optional in ("selection", "suggested_bins", "derived_from", "bins_note"):
            if spec.get(optional) is not None:
                entry[optional] = spec[optional]
        entries.append(entry)
    return entries


def _check_families(
    families: dict[str, dict[str, Any]],
    columns: set[str],
) -> list[str]:
    """Validate declared families; return every problem found, not the first."""

    problems: list[str] = []
    for name, family in families.items():
        if not isinstance(family, dict):
            problems.append(f"family {name!r}: must be a mapping.")
            continue
        if not isinstance(family.get("type"), str):
            problems.append(f"family {name!r}: missing a string 'type'.")

        combination = family.get("combination")
        if combination not in COMBINATIONS:
            known = ", ".join(sorted(COMBINATIONS))
            problems.append(
                f"family {name!r}: unknown combination {combination!r}; "
                f"expected one of: {known}."
            )

        declared = family.get("columns") or []
        if not declared:
            problems.append(f"family {name!r}: declares no columns.")
        missing = [column for column in declared if column not in columns]
        if missing:
            problems.append(
                f"family {name!r}: columns not in the event table: "
                f"{', '.join(missing)}."
            )

        reference = family.get("reference_column")
        if combination == PAIRED_COMBINATION:
            # checked here rather than at histogram time, where the same
            # mistake surfaces only once someone computes an uncertainty
            if not isinstance(reference, str):
                problems.append(
                    f"family {name!r}: combination {PAIRED_COMBINATION!r} "
                    "requires a 'reference_column'."
                )
            elif reference not in columns:
                problems.append(
                    f"family {name!r}: reference_column {reference!r} is not "
                    "in the event table."
                )
            if len(declared) != 1:
                problems.append(
                    f"family {name!r}: combination {PAIRED_COMBINATION!r} "
                    f"expects exactly one column, got {len(declared)}."
                )
        elif reference is not None and reference not in columns:
            problems.append(
                f"family {name!r}: reference_column {reference!r} is not in "
                "the event table."
            )
    return problems


def write_package_from_declaration(
    events: pd.DataFrame | str | Path,
    output_dir: str | Path,
    observables: dict[str, dict[str, Any]] | list[dict[str, Any]],
    weights: dict[str, Any],
    families: dict[str, dict[str, Any]] | None = None,
    dataset: dict[str, Any] | None = None,
    normalization_mode: str = "absolute",
    weight_units: str | None = "fb",
    event_count: int | None = None,
    event_id_column: str | None = None,
    usage_notes: list[str] | None = None,
) -> Path:
    """Write a publication package from declared observables and weights.

    Nothing is discovered: only declared columns are packaged, and every
    declared column must exist. Problems are collected and reported
    together, so one run tells you everything that is wrong.

    Args:
        events: a DataFrame, or a path to a ``.parquet`` / ``.h5`` file.
        output_dir: destination package directory.
        observables: ``{name: spec}`` (or a list of specs carrying
            ``name``). Recognised spec keys: ``description``, ``units``,
            ``bins`` (written as the *official* binning),
            ``binning_provenance``, ``selection``, ``suggested_bins``,
            ``derived_from``, ``bins_note``.
        weights: must carry ``nominal``; may carry ``base_mc_weight``
            (``None`` when the sample has no separate prior weight) and
            ``nominal_convention``.
        families: ``{name: {type, combination, columns,
            reference_column?}}``. ``combination`` must be one of the
            implemented recipes; ``type`` is free text used for grouping.
        dataset: free-form identity block (``name``, ``method``,
            ``assumptions``, ...).
        event_id_column: declares column-based event alignment.

    Returns:
        The package directory.
    """

    if normalization_mode not in NORMALIZATION_MODES:
        allowed = ", ".join(NORMALIZATION_MODES)
        raise PackageWriteError(
            f"Unknown normalization_mode {normalization_mode!r}; expected "
            f"one of: {allowed}."
        )
    if not isinstance(weights, dict) or "nominal" not in weights:
        raise PackageWriteError(
            "`weights` must be a mapping declaring at least 'nominal'."
        )

    convention = weights.get("nominal_convention") or "includes_mc_weight"
    if convention not in NOMINAL_CONVENTIONS:
        allowed = ", ".join(NOMINAL_CONVENTIONS)
        raise PackageWriteError(
            f"Unknown nominal_convention {convention!r}; expected one of: "
            f"{allowed}."
        )

    nominal_column = weights["nominal"]
    base_column = weights.get("base_mc_weight")
    if convention == "reweighting_factor" and not isinstance(base_column, str):
        raise PackageWriteError(
            "nominal_convention 'reweighting_factor' means the nominal "
            "column holds only the learned factor, so weights.base_mc_weight "
            "must name the base MC weight column."
        )

    frame = _events_frame(events)
    if event_count is not None:
        frame = frame.iloc[:event_count]
    available = set(frame.columns)
    families = dict(families or {})
    entries = _observable_entries(observables)

    # --- collect every problem before touching the filesystem ------------
    problems: list[str] = []
    if nominal_column not in available:
        problems.append(
            f"weights.nominal column {nominal_column!r} is not in the event table."
        )
    if base_column is not None:
        if not isinstance(base_column, str):
            problems.append("weights.base_mc_weight must be a column name or None.")
        elif base_column not in available:
            problems.append(
                f"weights.base_mc_weight column {base_column!r} is not in the "
                "event table."
            )
    if event_id_column is not None and event_id_column not in available:
        problems.append(
            f"event_id_column {event_id_column!r} is not in the event table."
        )

    selection_needed: list[str] = []
    for entry in entries:
        if entry["name"] not in available:
            problems.append(
                f"observable {entry['name']!r} is not in the event table."
            )
        expression = entry.get("selection")
        if isinstance(expression, str):
            try:
                referenced = selection_columns(expression)
            except Exception as exc:
                problems.append(
                    f"observable {entry['name']!r}: cannot parse selection "
                    f"{expression!r}: {exc}"
                )
                continue
            for column in referenced:
                if column not in available:
                    problems.append(
                        f"observable {entry['name']!r}: selection references "
                        f"column {column!r}, which is not in the event table."
                    )
                else:
                    selection_needed.append(column)

    problems.extend(_check_families(families, available))

    if problems:
        raise PackageWriteError(
            "Cannot write the package; the declaration does not match the "
            "event table:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )

    # --- assemble ---------------------------------------------------------
    selected: list[str] = [entry["name"] for entry in entries]
    selected.extend(selection_needed)
    if event_id_column is not None:
        selected.append(event_id_column)
    selected.append(nominal_column)
    if isinstance(base_column, str):
        selected.append(base_column)
    for family in families.values():
        selected.extend(family.get("columns", []))
        reference = family.get("reference_column")
        if isinstance(reference, str):
            selected.append(reference)
    selected = list(dict.fromkeys(selected))

    package_df = frame.loc[:, selected]
    packaged_count = int(len(package_df))
    nominal_sumw = float(
        package_df[nominal_column].to_numpy(dtype=float).sum()
    )

    weights_block: dict[str, Any] = {
        "nominal": nominal_column,
        "nominal_convention": convention,
    }
    if isinstance(base_column, str):
        weights_block["base_mc_weight"] = base_column
    if families:
        weights_block["families"] = families

    normalization: dict[str, Any] = {
        "mode": normalization_mode,
        "nominal_weight_column": nominal_column,
        "expected_nominal_sumw": nominal_sumw,
        "tolerance": 1.0e-8,
    }
    if isinstance(base_column, str):
        normalization["base_weight_column"] = base_column
    if weight_units is not None:
        normalization["weight_units"] = weight_units
    if normalization_mode == "absolute":
        normalization["sum_weights_equals"] = "fiducial_cross_section"

    dataset_block = dict(dataset or {})
    dataset_block.setdefault("name", Path(output_dir).name)
    if event_count is not None:
        # a subset rescales sum(w): record it rather than let a downstream
        # cross-section comparison silently rest on it
        assumptions = list(dataset_block.get("assumptions") or [])
        note = (
            f"packaged the first {packaged_count} events of the source; "
            "sum(w) is not the full-sample normalisation"
        )
        if note not in assumptions:
            assumptions.append(note)
        dataset_block["assumptions"] = assumptions

    metadata: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "dataset": dataset_block,
        "observables": entries,
        "weights": weights_block,
        "normalization": normalization,
        "publication": {
            "format": "parquet",
            "events_file": "events.parquet",
            "event_count": packaged_count,
            "columns": selected,
            "event_alignment": (
                {"method": "column", "column": event_id_column}
                if event_id_column is not None
                else {"method": "row_order", "column": None}
            ),
        },
    }
    if usage_notes:
        metadata["usage_notes"] = list(usage_notes)

    # fail before writing anything, not after
    try:
        parse_metadata(metadata)
    except ValueError as exc:
        raise PackageWriteError(
            f"The declaration produced metadata that fails the package "
            f"schema.\n{exc}"
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.parquet"
    package_df.to_parquet(events_path, index=False)
    metadata["publication"]["checksum_sha256"] = _compute_checksum(events_path)

    import yaml

    with (output_dir / "metadata.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)
    return output_dir


__all__ = ["write_package_from_declaration"]

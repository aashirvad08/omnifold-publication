"""HEPData export and import for OmniFold publication packages.

Export writes a HEPData submission (submission.yaml plus one data-table
YAML per observable) in the format hepdata.net validates against
(independent_variables / dependent_variables / qualifiers / per-bin
errors), modelled on the record of this measurement itself
(https://www.hepdata.net/record/ins2791852, Table 1: differential
cross-section in fb/GeV with a total symmetric error per bin).

Import parses a submission back into a :class:`HEPDataPackage` — the
binned counterpart of :class:`OmniFoldPackage`. A HEPData record holds
histograms, not per-event weights, so the event-level API surface
(load_events, get_weights, weight families, covariance) is physically
unavailable and raises a clear error; the binned subset (list_observables,
observable_bins, histogram, cross_section, summary, validate) matches the
package API.

Remote loading (``load_package("hepdata:ins2791852")``) uses hepdata.net's
JSON API endpoints (``record/<id>?format=json`` plus
``record/data/<recid>/<table_id>/<version>/``) as the **primary** path,
not a fallback: the bulk ``/download/`` endpoints sit behind Cloudflare
bot protection that challenges non-browser clients (curl, urllib,
requests), so they are unreliable for programmatic access in general.
The JSON endpoints serve HEPData's internal x/y table representation,
which is converted here into the same normalized tables produced by the
submission-YAML parser.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .exceptions import PackageReadError
from .histogram import HistogramResult


SUBMISSION_FILENAME = "submission.yaml"
DEFAULT_SQRT_S_GEV = 13000.0
DEFAULT_LUMINOSITY = "139 fb$^{-1}$"


def _table_filename(observable: str) -> str:
    return f"data_{observable.lower()}.yaml"


def _cross_section_units(observable_units: str) -> str:
    if observable_units and observable_units not in {"1", ""}:
        return f"fb/{observable_units}"
    return "fb"


def export_hepdata(
    package: Any,
    output_dir: str | Path,
    observables: list[str] | None = None,
    bins_map: dict[str, list[float]] | None = None,
    error_breakdown: bool = False,
    sqrt_s_gev: float = DEFAULT_SQRT_S_GEV,
    luminosity: str = DEFAULT_LUMINOSITY,
) -> Path:
    """Write a HEPData submission from a package's binned cross-sections.

    One table per observable: the differential cross-section (final
    weights, declared selection applied, official binning) with per-bin
    errors — a single ``total`` symmetric error by default (matching the
    published record), or one labelled error per uncertainty component
    when ``error_breakdown`` is set. Observables default to those with an
    official/declared binning; requesting one without it raises unless
    ``bins_map`` provides edges.
    """

    from .closure import _resolve_bins

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if observables is None:
        observables = []
        for name in package.list_observables():
            try:
                _resolve_bins(package, name, (bins_map or {}).get(name))
            except PackageReadError:
                continue
            observables.append(name)
        if not observables:
            raise PackageReadError(
                "No observables with an official binning to export."
            )

    submission_documents: list[dict[str, Any]] = [
        {
            "comment": (
                "Binned differential cross-sections exported from an "
                "OmniFold publication package (omnifold_publication)."
            ),
        }
    ]

    for observable in observables:
        bins = _resolve_bins(package, observable, (bins_map or {}).get(observable))
        breakdown = package.uncertainty_breakdown(observable, bins=bins)
        edges = np.asarray(breakdown["edges"], dtype=float)
        widths = np.diff(edges)
        density = breakdown["nominal"] / widths

        units = package.observable_units(observable)
        value_units = _cross_section_units(units)

        values: list[dict[str, Any]] = []
        for index in range(len(widths)):
            if error_breakdown:
                errors = [
                    {
                        "symerror": float(component[index] / widths[index]),
                        "label": name,
                    }
                    for name, component in breakdown["components"].items()
                ]
            else:
                errors = [
                    {
                        "symerror": float(
                            breakdown["total"][index] / widths[index]
                        ),
                        "label": "total",
                    }
                ]
            values.append(
                {"value": float(density[index]), "errors": errors}
            )

        independent_header: dict[str, Any] = {"name": observable}
        if units and units != "1":
            independent_header["units"] = units
        table = {
            "independent_variables": [
                {
                    "header": independent_header,
                    "values": [
                        {
                            "low": float(edges[index]),
                            "high": float(edges[index + 1]),
                        }
                        for index in range(len(widths))
                    ],
                }
            ],
            "dependent_variables": [
                {
                    "header": {
                        "name": "Differential cross-section",
                        "units": value_units,
                    },
                    "qualifiers": [
                        {"name": "SQRT(S)", "units": "GeV",
                         "value": float(sqrt_s_gev)},
                        {"name": "LUMINOSITY", "value": luminosity},
                    ],
                    "values": values,
                }
            ],
        }

        table_file = _table_filename(observable)
        with (output_dir / table_file).open("w", encoding="utf-8") as stream:
            yaml.safe_dump(table, stream, sort_keys=False)

        entry = next(
            (
                item
                for item in package.metadata().get("observables", [])
                if isinstance(item, dict) and item.get("name") == observable
            ),
            {},
        )
        submission_documents.append(
            {
                "name": observable,
                "description": entry.get(
                    "description",
                    f"Differential cross-section in bins of {observable}.",
                ),
                "keywords": [
                    {"name": "observables", "values": [observable]},
                    {"name": "cmenergies", "values": [float(sqrt_s_gev)]},
                ],
                "data_file": table_file,
            }
        )

    with (output_dir / SUBMISSION_FILENAME).open("w", encoding="utf-8") as stream:
        yaml.safe_dump_all(submission_documents, stream, sort_keys=False)
    return output_dir


def _bin_edges_from_values(values: list[dict[str, Any]]) -> np.ndarray:
    edges: list[float] = []
    for index, value in enumerate(values):
        if "low" not in value or "high" not in value:
            raise PackageReadError(
                "HEPData table rows must define low/high bin edges."
            )
        low, high = float(value["low"]), float(value["high"])
        if index == 0:
            edges.append(low)
        elif not np.isclose(edges[-1], low):
            raise PackageReadError(
                "HEPData table bins are not contiguous; cannot form edges."
            )
        edges.append(high)
    return np.asarray(edges, dtype=float)


def _normalize_table(
    name: str,
    description: str,
    table: dict[str, Any],
) -> dict[str, Any]:
    """Convert a submission-format table into the normalized structure."""

    independent = table.get("independent_variables") or []
    dependent = table.get("dependent_variables") or []
    if len(independent) != 1 or not dependent:
        raise PackageReadError(
            f"HEPData table {name!r} must have exactly one independent "
            "variable and at least one dependent variable."
        )

    edges = _bin_edges_from_values(independent[0].get("values") or [])
    primary = dependent[0]
    rows = primary.get("values") or []
    if len(rows) != len(edges) - 1:
        raise PackageReadError(
            f"HEPData table {name!r} has {len(rows)} values for "
            f"{len(edges) - 1} bins."
        )

    values = np.array([float(row["value"]) for row in rows], dtype=float)
    errors: dict[str, np.ndarray] = {}
    for index, row in enumerate(rows):
        for error in row.get("errors") or []:
            label = error.get("label", "total")
            if "symerror" in error:
                magnitude = abs(float(error["symerror"]))
            elif "asymerror" in error:
                plus = abs(float(error["asymerror"].get("plus", 0.0)))
                minus = abs(float(error["asymerror"].get("minus", 0.0)))
                magnitude = max(plus, minus)
            else:
                continue
            errors.setdefault(label, np.zeros(len(rows)))[index] = magnitude

    return {
        "name": name,
        "description": description,
        "independent_name": (independent[0].get("header") or {}).get("name", name),
        "units": (independent[0].get("header") or {}).get("units", ""),
        "value_units": (primary.get("header") or {}).get("units", ""),
        "qualifiers": primary.get("qualifiers") or [],
        "edges": edges,
        "values": values,
        "errors": errors,
    }


class HEPDataPackage:
    """Binned counterpart of OmniFoldPackage, backed by HEPData tables.

    Implements the binned subset of the package interface. Event-level
    operations are physically unavailable for a binned record and raise
    PackageReadError.
    """

    def __init__(self, tables: dict[str, dict[str, Any]], source: str):
        if not tables:
            raise PackageReadError("HEPData source contains no data tables.")
        self._tables = tables
        self.source = source

    def list_observables(self) -> list[str]:
        return list(self._tables)

    def table(self, observable: str) -> dict[str, Any]:
        if observable not in self._tables:
            raise PackageReadError(f"Unknown observable: {observable}")
        return self._tables[observable]

    def observable_bins(self, observable: str) -> list[float]:
        return [float(edge) for edge in self.table(observable)["edges"]]

    def observable_units(self, observable: str) -> str:
        return self.table(observable)["units"]

    def histogram(self, observable: str) -> HistogramResult:
        """Differential cross-section as published, with its errors.

        ``total_uncertainty`` carries the table's total error;
        ``stat_uncertainty`` is filled only when the table publishes a
        labelled per-component breakdown, otherwise zero.
        """

        table = self.table(observable)
        edges = table["edges"]
        errors = table["errors"]
        total = errors.get("total")
        if total is None and errors:
            total = np.sqrt(
                np.sum([error**2 for error in errors.values()], axis=0)
            )
        return HistogramResult(
            hist=table["values"].copy(),
            edges=edges.copy(),
            centers=0.5 * (edges[:-1] + edges[1:]),
            stat_uncertainty=errors.get(
                "sample_stat", np.zeros_like(table["values"])
            ),
            total_uncertainty=total,
        )

    def cross_section(self, observable: str | None = None) -> float:
        """Integral of a table (density times bin widths), in fb."""

        if observable is None:
            observable = next(iter(self._tables))
        table = self.table(observable)
        return float(np.sum(table["values"] * np.diff(table["edges"])))

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "observables": self.list_observables(),
            "binned_only": True,
        }

    def _binned_only(self, operation: str) -> None:
        raise PackageReadError(
            f"{operation} is unavailable: a HEPData record contains binned "
            "tables, not per-event weights. Use the event-level package "
            "(Zenodo release) for unbinned analysis."
        )

    def load_events(self, *args: Any, **kwargs: Any) -> None:
        self._binned_only("load_events")

    def get_weights(self, *args: Any, **kwargs: Any) -> None:
        self._binned_only("get_weights")

    def list_weight_families(self) -> None:
        self._binned_only("list_weight_families")

    def get_family_weights(self, *args: Any, **kwargs: Any) -> None:
        self._binned_only("get_family_weights")

    def get_uncertainty(self, *args: Any, **kwargs: Any) -> None:
        self._binned_only("get_uncertainty")

    def observable_values(self, *args: Any, **kwargs: Any) -> None:
        self._binned_only("observable_values")

    def uncertainty_breakdown(self, *args: Any, **kwargs: Any) -> None:
        self._binned_only("uncertainty_breakdown")

    def covariance_matrix(self, *args: Any, **kwargs: Any) -> None:
        self._binned_only("covariance_matrix")

    def validate(self) -> None:
        """Validate the binned tables themselves.

        Unlike the other event-level methods, validation is meaningful
        for a binned record — it just checks different invariants:
        strictly increasing bin edges, value/error arrays matching the
        bin count, finite values, and non-negative finite error
        magnitudes. Raises PackageValidationError listing every problem.
        """

        from .exceptions import PackageValidationError

        errors: list[str] = []
        for name, table in self._tables.items():
            edges = np.asarray(table["edges"], dtype=float)
            values = np.asarray(table["values"], dtype=float)
            n_bins = len(edges) - 1
            if not np.all(np.diff(edges) > 0):
                errors.append(f"{name}: bin edges are not strictly increasing.")
            if len(values) != n_bins:
                errors.append(
                    f"{name}: {len(values)} values for {n_bins} bins."
                )
            if not np.isfinite(values).all():
                errors.append(f"{name}: non-finite bin values.")
            for label, magnitudes in table["errors"].items():
                magnitudes = np.asarray(magnitudes, dtype=float)
                if len(magnitudes) != n_bins:
                    errors.append(
                        f"{name}: error {label!r} has {len(magnitudes)} "
                        f"entries for {n_bins} bins."
                    )
                if not np.isfinite(magnitudes).all() or (magnitudes < 0).any():
                    errors.append(
                        f"{name}: error {label!r} has non-finite or "
                        "negative magnitudes."
                    )
        if errors:
            raise PackageValidationError("\n".join(errors))


HEPDATA_BASE_URL = "https://www.hepdata.net"
_FETCH_TIMEOUT_SECONDS = 30.0


def _fetch_json(url: str) -> dict[str, Any]:
    import json
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        url, headers={"User-Agent": "omnifold-publication/0.2"}
    )
    try:
        with urllib.request.urlopen(
            request, timeout=_FETCH_TIMEOUT_SECONDS
        ) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as exc:
        raise PackageReadError(
            f"Could not fetch HEPData resource {url}: {exc}"
        ) from exc


def _parse_header_units(name: str) -> tuple[str, str]:
    """Split an internal header like 'Dilepton $p_T$ [GeV]' into name, units."""

    stripped = name.strip()
    if stripped.endswith("]") and "[" in stripped:
        base, _, units = stripped.rpartition("[")
        return base.strip(), units[:-1].strip()
    return stripped, ""


def _normalize_internal_table(
    name: str,
    description: str,
    table: dict[str, Any],
) -> dict[str, Any] | None:
    """Convert HEPData's internal x/y table format to a normalized table.

    Returns None for tables that are not one-dimensional binned
    distributions (no low/high on the first independent variable).
    """

    rows = table.get("values") or []
    if not rows:
        return None
    first_x = (rows[0].get("x") or [{}])[0]
    if "low" not in first_x or "high" not in first_x:
        return None

    edges = _bin_edges_from_values([row["x"][0] for row in rows])
    values = np.array(
        [float(row["y"][0]["value"]) for row in rows], dtype=float
    )
    errors: dict[str, np.ndarray] = {}
    for index, row in enumerate(rows):
        for error in row["y"][0].get("errors") or []:
            label = error.get("label", "total")
            if "symerror" in error:
                magnitude = abs(float(error["symerror"]))
            elif "asymerror" in error:
                plus = abs(float(error["asymerror"].get("plus", 0.0)))
                minus = abs(float(error["asymerror"].get("minus", 0.0)))
                magnitude = max(plus, minus)
            else:
                continue
            errors.setdefault(label, np.zeros(len(rows)))[index] = magnitude

    headers = table.get("headers") or []
    x_name, x_units = _parse_header_units(
        headers[0]["name"] if headers else name
    )
    _, y_units = _parse_header_units(
        headers[1]["name"] if len(headers) > 1 else ""
    )
    return {
        "name": name,
        "description": description,
        "independent_name": x_name,
        "units": x_units,
        "value_units": y_units,
        "qualifiers": table.get("qualifiers") or {},
        "edges": edges,
        "values": values,
        "errors": errors,
    }


def load_hepdata_record(record_id: str) -> HEPDataPackage:
    """Load a HEPData record (e.g. "ins2791852") into a HEPDataPackage.

    Fetches the record summary and every one-dimensional binned table via
    the JSON API (see module docstring for why the bulk download
    endpoints are not used). The first dependent column of each table —
    the measured Data column in the Z+jets record — becomes the table's
    values; tables that are not 1D binned distributions are skipped.
    """

    record = _fetch_json(
        f"{HEPDATA_BASE_URL}/record/{record_id}?format=json"
    )
    recid = (record.get("record") or {}).get("recid")
    version = record.get("version", 1)
    data_tables = record.get("data_tables") or []
    if recid is None or not data_tables:
        raise PackageReadError(
            f"HEPData record {record_id!r} has no data tables."
        )

    tables: dict[str, dict[str, Any]] = {}
    for entry in data_tables:
        table_json = _fetch_json(
            f"{HEPDATA_BASE_URL}/record/data/{recid}/{entry['id']}/{version}/"
        )
        normalized = _normalize_internal_table(
            entry.get("name", str(entry["id"])),
            entry.get("description", ""),
            table_json,
        )
        if normalized is not None:
            tables[normalized["name"]] = normalized
    return HEPDataPackage(tables, source=f"hepdata:{record_id}")


def load_hepdata_submission(path: str | Path) -> HEPDataPackage:
    """Load a HEPData submission directory into a HEPDataPackage."""

    path = Path(path)
    submission_path = (
        path / SUBMISSION_FILENAME if path.is_dir() else path
    )
    if not submission_path.exists():
        raise PackageReadError(
            f"No {SUBMISSION_FILENAME} found under {path}."
        )

    base = submission_path.parent
    tables: dict[str, dict[str, Any]] = {}
    with submission_path.open("r", encoding="utf-8") as stream:
        for document in yaml.safe_load_all(stream):
            if not isinstance(document, dict) or "data_file" not in document:
                continue
            name = document.get("name", document["data_file"])
            with (base / document["data_file"]).open(
                "r", encoding="utf-8"
            ) as table_stream:
                table = yaml.safe_load(table_stream)
            tables[name] = _normalize_table(
                name, document.get("description", ""), table
            )
    return HEPDataPackage(tables, source=str(submission_path))

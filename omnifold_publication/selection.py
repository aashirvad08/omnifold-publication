"""Per-observable event selections (e.g. ``"pT_trackj1 > 5"``).

Jet observables in the ATLAS Z+jets release require a jet-pT threshold
(README usage recommendation 1; the release code masks
``pT_trackj1 > 5`` / ``pT_trackj2 > 5`` before histogramming trackjet
variables, multifold_util.py corr_matrix). Without the mask, events with
no real jet — whose trackjet columns hold soft/degenerate values (down to
pT 0.59 GeV, tau1 exactly 0 in the release files) — silently enter the
histograms.

Selections are simple comparisons joined by ``and``:
``"pT_trackj1 > 5"``, ``"pT_trackj1 > 5 and pT_trackj2 > 5"``.
They are parsed with a strict grammar — never evaluated as code.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .exceptions import PackageReadError


_CLAUSE = re.compile(
    r"^\s*(?P<column>[A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(?P<op>>=|<=|==|!=|>|<)\s*"
    r"(?P<value>[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s*$"
)

_OPERATORS = {
    ">": np.greater,
    ">=": np.greater_equal,
    "<": np.less,
    "<=": np.less_equal,
    "==": np.equal,
    "!=": np.not_equal,
}


def parse_selection(expression: str) -> list[tuple[str, str, float]]:
    """Parse a selection into (column, operator, value) clauses."""

    clauses: list[tuple[str, str, float]] = []
    for part in expression.split(" and "):
        match = _CLAUSE.match(part)
        if match is None:
            raise PackageReadError(
                f"Cannot parse selection clause {part.strip()!r}; expected "
                "'<column> <op> <number>' joined by ' and '."
            )
        clauses.append(
            (match["column"], match["op"], float(match["value"]))
        )
    return clauses


def selection_columns(expression: str) -> list[str]:
    """Column names referenced by a selection expression."""

    return list(dict.fromkeys(column for column, _, _ in parse_selection(expression)))


def selection_mask(df: pd.DataFrame, expression: str) -> np.ndarray:
    """Boolean event mask for a selection expression."""

    mask = np.ones(len(df), dtype=bool)
    for column, op, value in parse_selection(expression):
        if column not in df.columns:
            raise PackageReadError(
                f"Selection references column {column!r} which is not in "
                "the event table."
            )
        mask &= _OPERATORS[op](df[column].to_numpy(dtype=float), value)
    return mask

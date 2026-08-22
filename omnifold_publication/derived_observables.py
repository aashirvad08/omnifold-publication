"""Derived observables from the ATLAS Z+jets release.

Port of ``calculate_vars`` (multifold_util.py:257-304; an identical
duplicate lives in 2_pseudo_results.ipynb cell 17), built with the
scikit-hep ``vector`` library like the original.

Convention notes, pinned deliberately rather than assumed:

- **The jet four-vector is built with rapidity in the eta slot**:
  ``vector.array({"pt": pT_trackj1, "phi": phi_trackj1,
  "eta": y_trackj1, "m": 0})``. Because the vector is massless,
  eta == rapidity, so this places the jet at its rapidity y — the release
  files provide ``y_trackj1`` (no jet eta column exists). This is the
  release's own convention, reproduced exactly.
- **dR is a (rapidity, phi) distance**: sqrt((y1-y2)^2 + dphi^2), using
  ``.rapidity`` of both vectors. For the massive dilepton system
  (l1 + l2), rapidity != eta, and the release uses rapidity.
- **tau21 = tau2/tau1 divides blindly** in the release; events without a
  real jet can carry tau1 == 0, producing non-finite tau21. The port
  suppresses the warning but keeps the non-finite values so that the
  declared selection (``pT_trackj1 > 5``, README recommendation 1) — not a
  silent value rewrite — removes those events. Note that the selection
  does not remove them all: single-track jets have tau1 == 0 by
  definition (0.07% of the real file above the pT threshold), so tau21
  remains undefined for them; np.histogram drops NaN entries silently,
  which is also what the release's own plots do.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .exceptions import PackageWriteError


# name -> (source columns required, selection recommended by the release)
DERIVED_OBSERVABLES: dict[str, dict[str, Any]] = {
    "tau21": {
        "inputs": ["tau1_trackj1", "tau2_trackj1"],
        "selection": "pT_trackj1 > 5",
        "description": "N-subjettiness ratio tau2/tau1 of the leading track jet",
        "units": "",
    },
    "phi_ll": {
        "inputs": ["pT_l1", "eta_l1", "phi_l1", "pT_l2", "eta_l2", "phi_l2"],
        "selection": None,
        "description": "Azimuthal angle of the dilepton system",
        "units": "rad",
    },
    "dR_ll": {
        "inputs": [
            "pT_l1", "eta_l1", "phi_l1",
            "pT_l2", "eta_l2", "phi_l2",
            "pT_trackj1", "y_trackj1", "phi_trackj1",
        ],
        "selection": "pT_trackj1 > 5",
        "description": "Rapidity-phi distance between the dilepton system "
        "and the leading track jet",
        "units": "",
    },
}


def _massless(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray):
    import vector

    return vector.array(
        {"pt": pt, "eta": eta, "phi": phi, "m": np.zeros_like(pt)}
    )


def _rapidity_phi_distance(v1, v2) -> np.ndarray:
    """dR in (rapidity, phi): the release's dR helper."""

    dy = v1.rapidity - v2.rapidity
    dphi = v1.deltaphi(v2)
    return np.sqrt(dy**2 + dphi**2)


def compute_tau21(df: pd.DataFrame) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (
            df["tau2_trackj1"].to_numpy(dtype=float)
            / df["tau1_trackj1"].to_numpy(dtype=float)
        )


def compute_phi_ll(df: pd.DataFrame) -> np.ndarray:
    l1 = _massless(
        df["pT_l1"].to_numpy(dtype=float),
        df["eta_l1"].to_numpy(dtype=float),
        df["phi_l1"].to_numpy(dtype=float),
    )
    l2 = _massless(
        df["pT_l2"].to_numpy(dtype=float),
        df["eta_l2"].to_numpy(dtype=float),
        df["phi_l2"].to_numpy(dtype=float),
    )
    return l1.add(l2).phi


def compute_dR_ll(df: pd.DataFrame) -> np.ndarray:
    l1 = _massless(
        df["pT_l1"].to_numpy(dtype=float),
        df["eta_l1"].to_numpy(dtype=float),
        df["phi_l1"].to_numpy(dtype=float),
    )
    l2 = _massless(
        df["pT_l2"].to_numpy(dtype=float),
        df["eta_l2"].to_numpy(dtype=float),
        df["phi_l2"].to_numpy(dtype=float),
    )
    # rapidity passed in the eta slot of a massless vector (see module note)
    track_j1 = _massless(
        df["pT_trackj1"].to_numpy(dtype=float),
        df["y_trackj1"].to_numpy(dtype=float),
        df["phi_trackj1"].to_numpy(dtype=float),
    )
    return _rapidity_phi_distance(l1 + l2, track_j1)


_COMPUTERS = {
    "tau21": compute_tau21,
    "phi_ll": compute_phi_ll,
    "dR_ll": compute_dR_ll,
}


def compute_derived_observables(
    df: pd.DataFrame,
    names: list[str] | None = None,
) -> pd.DataFrame:
    """Add derived observable columns to a copy of ``df``.

    ``names`` defaults to all of tau21, phi_ll, dR_ll (the calculate_vars
    set). Raises if a required input column is missing.
    """

    selected = names if names is not None else list(DERIVED_OBSERVABLES)
    result = df.copy()
    for name in selected:
        if name not in DERIVED_OBSERVABLES:
            known = ", ".join(DERIVED_OBSERVABLES)
            raise PackageWriteError(
                f"Unknown derived observable {name!r}; known: {known}."
            )
        missing = [
            column
            for column in DERIVED_OBSERVABLES[name]["inputs"]
            if column not in df.columns
        ]
        if missing:
            raise PackageWriteError(
                f"Derived observable {name!r} requires missing columns: "
                f"{', '.join(missing)}."
            )
        result[name] = _COMPUTERS[name](df)
    return result

"""End-to-end pseudodata closure workflow (cf. 2_pseudo_results.ipynb).

Runs the full validation chain on the real release file:

1. Package a (pseudo-)measurement with all weight families.
2. Build a *known-target* companion package — the truth distribution the
   measurement should close against — and link it in a manifest under the
   role "target" (the release ships this as target.h5; here the target is
   constructed from the nominal weights with a known smooth reweighting,
   since the public pseudodata files are not part of the local dataset).
3. Run chi2_test per observable (dual-mode covariance, circularity guard,
   p < 0.01 decorrelation fallback) and print a cell 26-style table.
4. Demonstrate that a deliberately wrong target is rejected.
5. Run the release's own embedded pseudodata closure: weights_dd vs
   target_dd.
6. Render the multi-panel closure grid (Start / Target / Result + ratio).

Usage:  python examples/pseudodata_closure.py  (from the repo root;
requires data/multifold.h5)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omnifold_publication import (  # noqa: E402
    chi2_test,
    load_analysis,
    plot_closure_grid,
    write_manifest,
    write_package,
)

INPUT = ROOT / "data" / "multifold.h5"
EVENT_COUNT = 50_000
OBSERVABLES = ["pT_ll", "pT_l1", "pT_l2", "y_ll", "m_trackj1", "pT_trackj1"]
OUTPUT_DIR = ROOT / "examples" / "output"


def build_target_source(workdir: Path) -> Path:
    """A truth sample: nominal weights reweighted by a known smooth factor.

    The factor (0.5% tilt across the pT_ll range) is small compared to
    the total uncertainty, so a correct closure test must PASS against
    this target.
    """

    df = pd.read_hdf(INPUT, "df").iloc[:EVENT_COUNT].copy()
    tilt = 1.0 + 0.005 * (df["pT_ll"] - 400.0) / 800.0
    df["weights_nominal"] = df["weights_nominal"] * tilt
    path = workdir / "target_source.h5"
    df.to_hdf(path, key="df", mode="w")
    return path


def build_wrong_target_source(workdir: Path) -> Path:
    """A deliberately wrong truth sample (+20% flat) that must FAIL."""

    df = pd.read_hdf(INPUT, "df").iloc[:EVENT_COUNT].copy()
    df["weights_nominal"] = df["weights_nominal"] * 1.2
    path = workdir / "wrong_target_source.h5"
    df.to_hdf(path, key="df", mode="w")
    return path


def print_table(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    for row in rows:
        flag = "  [decorrelated]" if row["decorrelated"] else ""
        print(
            f"  {row['observable']:<14} dof: {row['dof']:<3} "
            f"chi2: {row['chi2']:9.4f}   p value: {row['p_value']:.4f}{flag}"
        )


def main() -> int:
    if not INPUT.exists():
        print(f"real data not available: {INPUT}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)

        print("1) Packaging the (pseudo-)measurement ...")
        measurement = write_package(
            input_path=INPUT,
            output_dir=workdir / "measurement",
            event_count=EVENT_COUNT,
            observables=OBSERVABLES,
            include_all_replicas=True,
        )

        print("2) Packaging the known-target companion sample ...")
        target = write_package(
            input_path=build_target_source(workdir),
            output_dir=workdir / "target",
            event_count=EVENT_COUNT,
            observables=OBSERVABLES,
        )
        manifest_dir = workdir / "analysis"
        write_manifest(
            output_dir=manifest_dir,
            nominal_path=measurement,
            target_path=target,
            variations={},
            analysis_name="pseudodata-closure-example",
        )
        analysis = load_analysis(manifest_dir)

        print("3) chi2 closure test against the known target ...")
        rows = [
            chi2_test(analysis, observable=name)
            for name in OBSERVABLES
        ]
        print_table("Closure vs known target (must pass):", rows)

        print("\n4) Deliberately wrong target (+20% flat, must fail) ...")
        wrong_target = write_package(
            input_path=build_wrong_target_source(workdir),
            output_dir=workdir / "wrong_target",
            event_count=EVENT_COUNT,
            observables=OBSERVABLES,
        )
        from omnifold_publication import load_package

        wrong = chi2_test(
            analysis,
            target_package=load_package(wrong_target),
            observable="pT_ll",
        )
        print_table("Closure vs wrong target:", [wrong])
        assert wrong["p_value"] < 0.01, "wrong target must be rejected"

        print("\n5) Release's embedded dd closure (weights_dd vs target_dd) ...")
        dd_rows = [
            chi2_test(
                analysis.nominal_package,
                observable=name,
                variation="weights_dd",
                target_variation="target_dd",
            )
            for name in OBSERVABLES
        ]
        print_table("dd-pair closure (circularity guard active):", dd_rows)

        print("\n6) Rendering the closure grid ...")
        figure_path = OUTPUT_DIR / "closure_grid.png"
        fig, grid = plot_closure_grid(
            analysis,
            observables=OBSERVABLES,
            ncols=3,
        )
        fig.savefig(figure_path, dpi=200)
        print(f"   saved {figure_path}")
        if grid["skipped"]:
            print(f"   skipped (no official binning): {grid['skipped']}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

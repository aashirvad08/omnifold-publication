"""Command-line interface for OmniFold publication packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import load_analysis
from .export import export_comparison_json, export_histogram_json
from .reader import list_systematics, load_events, load_metadata
from .validation import validate_package


def _print_metadata_summary(path: Path, include_columns: bool = False) -> None:
    metadata = load_metadata(path, enforce_version=True)
    publication = metadata.get("publication", {})
    observables = metadata.get("observables", [])

    print(f"Package: {path}")
    print(f"Format version: {metadata.get('format_version')}")
    print(f"Events: {publication.get('event_count', 'unknown')}")
    print(f"Observables: {len(observables)}")
    print(f"Systematics: {', '.join(list_systematics(metadata)) or 'none'}")

    if include_columns:
        df = load_events(path)
        print(f"Columns: {len(df.columns)}")
        for column in df.columns:
            print(f"  - {column}")


def inspect_command(args: argparse.Namespace) -> int:
    """Print detailed package metadata and columns."""

    _print_metadata_summary(Path(args.path), include_columns=True)
    return 0


def summary_command(args: argparse.Namespace) -> int:
    """Print a compact package summary."""

    _print_metadata_summary(Path(args.path), include_columns=False)
    return 0


def validate_command(args: argparse.Namespace) -> int:
    """Validate a package and return a process exit code."""

    errors = validate_package(args.path)
    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("Validation passed.")
    return 0


def export_comparison_command(args: argparse.Namespace) -> int:
    """Export a multi-sample histogram comparison as JSON."""

    export_comparison_json(
        manifest_dir=args.manifest_dir,
        observable=args.observable,
        output_path=args.output,
        bins=args.bins,
    )
    return 0


def export_metadata_command(args: argparse.Namespace) -> int:
    """Export an analysis description or summary as JSON."""

    analysis = load_analysis(args.manifest_dir)
    describe = getattr(analysis, "describe", None)
    payload = describe() if callable(describe) else analysis.summary()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    return 0


def export_histogram_command(args: argparse.Namespace) -> int:
    """Export a package histogram as JSON."""

    export_histogram_json(
        package_dir=args.package_dir,
        observable=args.observable,
        output_path=args.output,
        bins=args.bins,
        variation=args.variation,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="omnifold-publication",
        description="Inspect and validate OmniFold publication packages.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="show package metadata and columns",
    )
    inspect_parser.add_argument("path", help="package directory")
    inspect_parser.set_defaults(func=inspect_command)

    validate_parser = subparsers.add_parser("validate", help="validate a package")
    validate_parser.add_argument("path", help="package directory")
    validate_parser.set_defaults(func=validate_command)

    summary_parser = subparsers.add_parser(
        "summary",
        help="show a compact package summary",
    )
    summary_parser.add_argument("path", help="package directory")
    summary_parser.set_defaults(func=summary_command)

    comparison_parser = subparsers.add_parser(
        "export-comparison",
        help="export a multi-sample histogram comparison as JSON",
    )
    comparison_parser.add_argument("manifest_dir", help="manifest directory")
    comparison_parser.add_argument(
        "--observable",
        required=True,
        help="observable column name",
    )
    comparison_parser.add_argument("--output", required=True, help="output JSON path")
    comparison_parser.add_argument("--bins", nargs="+", type=float, help="bin edges")
    comparison_parser.set_defaults(func=export_comparison_command)

    metadata_parser = subparsers.add_parser(
        "export-metadata",
        help="export an analysis summary as JSON",
    )
    metadata_parser.add_argument("manifest_dir", help="manifest directory")
    metadata_parser.add_argument("--output", required=True, help="output JSON path")
    metadata_parser.set_defaults(func=export_metadata_command)

    histogram_parser = subparsers.add_parser(
        "export-histogram",
        help="export a package histogram as JSON",
    )
    histogram_parser.add_argument("package_dir", help="package directory")
    histogram_parser.add_argument(
        "--observable",
        required=True,
        help="observable column name",
    )
    histogram_parser.add_argument("--output", required=True, help="output JSON path")
    histogram_parser.add_argument(
        "--variation",
        default="nominal",
        help="weight variation name",
    )
    histogram_parser.add_argument("--bins", nargs="+", type=float, help="bin edges")
    histogram_parser.set_defaults(func=export_histogram_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the OmniFold publication command-line interface."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""High-level builder for multi-dataset OmniFold publications."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd

from .analysis import OmniFoldAnalysis, load_analysis
from .exceptions import PackageWriteError
from .manifest import write_manifest
from .writer import write_package


@dataclass
class _DatasetSpec:
    path: Path
    role: str
    label: str
    variation_type: str | None
    include_all_replicas: bool


class Publication:
    """High-level interface for building a multi-dataset publication."""

    def __init__(self, name: str):
        """Create an empty publication builder."""

        self.name = name
        self._datasets: list[_DatasetSpec] = []

    def add_dataset(
        self,
        path: str | Path,
        role: str,
        label: str,
        variation_type: str | None = None,
        include_all_replicas: bool = False,
    ) -> None:
        """Add one HDF5 dataset to the publication."""

        if role not in {"nominal", "variation"}:
            raise PackageWriteError("Dataset role must be 'nominal' or 'variation'.")
        if role == "variation" and variation_type is None:
            raise PackageWriteError("Variation datasets must define variation_type.")
        self._datasets.append(
            _DatasetSpec(
                path=Path(path),
                role=role,
                label=label,
                variation_type=variation_type,
                include_all_replicas=include_all_replicas,
            )
        )

    def inspect(self) -> None:
        """Print summary of all added datasets before building."""

        print(f"Publication: {self.name}")
        print("-" * 45)
        for index, dataset in enumerate(self._datasets, start=1):
            info = self._inspect_dataset(dataset.path)
            role = dataset.role
            if dataset.variation_type:
                role = f"{role}: {dataset.variation_type}"
            print(f"Dataset {index}: {dataset.label} [{role}]")
            print(f"  File:        {dataset.path}")
            print(f"  Events:      {info['events']:,}")
            print(
                "  Observables: "
                f"{', '.join(info['observables'][:4])}"
                f"{' ...' if len(info['observables']) > 4 else ''} "
                f"({len(info['observables'])} total)"
            )
            print(
                "  Weights:     "
                f"{', '.join(info['weights'][:4])}"
                f"{' ...' if len(info['weights']) > 4 else ''} "
                f"({len(info['weights'])} total)"
            )
            replicas = info["ensemble_replicas"]
            replica_summary = (
                f"{replicas} ensemble columns found" if replicas else "None"
            )
            print(f"  Replicas:    {replica_summary}")
            print()
        print("-" * 45)
        print("Status: Ready to build" if self._datasets else "Status: No datasets added")

    def build(
        self,
        output_dir: str | Path,
        event_count: int | None = None,
        observables: list[str] | None = None,
    ) -> OmniFoldAnalysis:
        """Write all packages and manifest, then return a loaded analysis."""

        output_dir = Path(output_dir)
        nominal = self._nominal_dataset()
        variations = [dataset for dataset in self._datasets if dataset.role == "variation"]

        nominal_package = output_dir / self._package_name(nominal)
        nominal_rows = event_count or self._event_count(nominal.path)
        write_package(
            input_path=nominal.path,
            output_dir=nominal_package,
            event_count=nominal_rows,
            observables=observables,
            include_all_replicas=nominal.include_all_replicas,
        )

        manifest_variations: dict[str, dict] = {}
        for variation in variations:
            package_dir = output_dir / self._package_name(variation)
            rows = event_count or self._event_count(variation.path)
            write_package(
                input_path=variation.path,
                output_dir=package_dir,
                event_count=rows,
                observables=observables,
                include_all_replicas=variation.include_all_replicas,
            )
            manifest_variations[self._variation_name(variation)] = {
                "path": package_dir.name,
                "type": variation.variation_type,
                "combination": "envelope",
            }

        write_manifest(
            output_dir=output_dir,
            nominal_path=nominal_package.name,
            variations=manifest_variations,
            analysis_name=self.name,
        )
        return load_analysis(output_dir)

    def _nominal_dataset(self) -> _DatasetSpec:
        nominal = [dataset for dataset in self._datasets if dataset.role == "nominal"]
        if len(nominal) != 1:
            raise PackageWriteError("Publication must contain exactly one nominal dataset.")
        return nominal[0]

    @staticmethod
    def _safe_name(value: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
        return safe or "sample"

    def _package_name(self, dataset: _DatasetSpec) -> str:
        base = self._safe_name(self.name)
        if dataset.role == "nominal":
            return f"{base}_nominal"
        return f"{base}_{self._safe_name(dataset.label)}"

    def _variation_name(self, dataset: _DatasetSpec) -> str:
        return self._safe_name(dataset.label)

    @staticmethod
    def _event_count(path: Path) -> int:
        with pd.HDFStore(path, mode="r") as store:
            return int(len(store["df"]))

    @staticmethod
    def _inspect_dataset(path: Path) -> dict:
        if not path.exists():
            raise PackageWriteError(f"Input file not found: {path}")
        df = pd.read_hdf(path, "df")
        weights = [
            column
            for column in df.columns
            if column == "weight_mc" or column.startswith("weights_")
        ]
        observables = [column for column in df.columns if column not in weights]
        return {
            "events": int(len(df)),
            "observables": observables,
            "weights": weights,
            "ensemble_replicas": len(
                [column for column in weights if column.startswith("weights_ensemble_")]
            ),
        }

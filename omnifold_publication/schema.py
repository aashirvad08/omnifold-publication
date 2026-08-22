"""Pydantic v2 schema for OmniFold ``metadata.yaml`` files.

Example:
    metadata = yaml.safe_load(open("spec/metadata.yaml"))
    Metadata(**metadata)
"""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError


class PhaseSpace(BaseModel):
    lepton_pt_min: str | float = Field(...)
    lepton_eta_max: str | float = Field(...)
    mll_min: str | float = Field(...)
    mll_max: str | float = Field(...)
    min_track_jets: str | int = Field(...)
    note: str | None = Field(default=None)


class Binning(BaseModel):
    """Official binning with its provenance, distinct from suggestions."""

    official: list[int | float] | None = Field(default=None)
    provenance: str | None = Field(default=None)


class Observable(BaseModel):
    name: str = Field(...)
    description: str = Field(...)
    units: str = Field(...)
    suggested_bins: list[int | float] | None = Field(default=None)
    bins_note: str | None = Field(default=None)
    binning: Binning | None = Field(default=None)
    selection: str | None = Field(default=None)
    derived_from: list[str] | None = Field(default=None)


class WeightFamily(BaseModel):
    name: str = Field(...)
    pattern: str = Field(...)
    type: str = Field(...)
    combination: str = Field(...)


class PackagedWeightFamily(BaseModel):
    """A packaged family of weight columns with its combination recipe."""

    type: str = Field(...)
    combination: str = Field(...)
    columns: list[str] = Field(default_factory=list)
    reference_column: str | None = Field(default=None)


class IterationStep(BaseModel):
    column: str = Field(...)


class IterationEntry(BaseModel):
    iteration: int = Field(...)
    step1: IterationStep | None = Field(default=None)
    step2: IterationStep | None = Field(default=None)


class Weights(BaseModel):
    nominal: str = Field(...)
    base_mc_weight: str = Field(...)
    nominal_convention: str | None = Field(default=None)
    replica: str | None = Field(default=None)
    families: dict[str, PackagedWeightFamily] | None = Field(default=None)
    iterations: list[IterationEntry] | None = Field(default=None)
    bootstrap_prefix: str | None = Field(default=None)
    ensemble_prefix: str | None = Field(default=None)
    data_bootstrap_prefix: str | None = Field(default=None)
    additional_families: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class Systematics(BaseModel):
    expected_families: list[WeightFamily] = Field(default_factory=list)


class Normalization(BaseModel):
    luminosity: str | float | None = Field(default=None)
    cross_section: str | float | None = Field(default=None)
    event_weights_normalized: bool | None = Field(default=None)
    mode: str | None = Field(default=None)
    base_weight_column: str | None = Field(default=None)
    nominal_weight_column: str | None = Field(default=None)
    expected_nominal_sumw: float | None = Field(default=None)
    tolerance: float | None = Field(default=None)
    weight_units: str | None = Field(default=None)
    sum_weights_equals: str | None = Field(default=None)
    note: str | None = Field(default=None)


class EventSelection(BaseModel):
    description: str = Field(...)
    inferred_phase_space: str | None = Field(default=None)
    phase_space: PhaseSpace | None = Field(default=None)


class Iterations(BaseModel):
    supported_steps: list[str] = Field(default_factory=list)
    note: str | None = Field(default=None)


class Training(BaseModel):
    algorithm: str = Field(...)
    iterations: str | int = Field(...)
    architecture: str = Field(...)
    note: str | None = Field(default=None)


class Dataset(BaseModel):
    name: str = Field(...)
    experiment: str | None = Field(default=None)
    description: str | None = Field(default=None)
    schema_version: str | None = Field(default=None)
    provenance_note: str | None = Field(default=None)
    # Unfolding method that produced this result (e.g. "MultiFold",
    # "OmniFold"); lets a cross-publication comparison label each side.
    method: str | None = Field(default=None)
    # Free-text assumptions recorded at publication time (e.g. "all events
    # used", undocumented columns) so they travel with the package.
    assumptions: list[str] | None = Field(default=None)


class Generation(BaseModel):
    nominal_generator: str = Field(...)
    alternative_generators: list[str] = Field(default_factory=list)
    alternative_samples: list[str] = Field(default_factory=list)


class FileEntry(BaseModel):
    filename: str = Field(...)
    path: str = Field(...)
    event_count: int = Field(...)
    note: str | None = Field(default=None)


class Files(BaseModel):
    nominal: FileEntry = Field(...)
    systematics: list[FileEntry] = Field(default_factory=list)


class EventAlignment(BaseModel):
    method: str = Field(default="row_order")
    column: str | None = Field(default=None)


class Publication(BaseModel):
    format: str = Field(...)
    events_file: str = Field(...)
    event_count: int = Field(...)
    columns: list[str] = Field(default_factory=list)
    source_file: str | None = Field(default=None)
    event_alignment: EventAlignment | None = Field(default=None)
    checksum_sha256: str | None = Field(default=None)


class Metadata(BaseModel):
    format_version: str = Field(...)
    dataset: Dataset | None = Field(default=None)
    generation: Generation | None = Field(default=None)
    files: Files | None = Field(default=None)
    observables: list[Observable] = Field(...)
    weights: Weights = Field(...)
    systematics: Systematics | None = Field(default=None)
    iterations: Iterations | None = Field(default=None)
    normalization: Normalization = Field(...)
    event_selection: EventSelection | None = Field(default=None)
    training: Training | None = Field(default=None)
    publication: Publication | None = Field(default=None)
    usage_notes: list[str] = Field(default_factory=list)


def validate_metadata(metadata: dict) -> None:
    """Validate metadata and raise a clear ``ValueError`` on schema errors."""

    try:
        Metadata(**metadata)
    except ValidationError as exc:
        errors = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"])
            errors.append(f"{location}: {error['msg']}")
        message = "Invalid OmniFold metadata:\n" + "\n".join(
            f"- {error}" for error in errors
        )
        raise ValueError(message) from exc

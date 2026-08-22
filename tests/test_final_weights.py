"""Tests for canonical derived final event weights.

The default convention is 'includes_mc_weight': the ATLAS Z+jets release
ships weights_nominal with weight_mc already folded in, so the final weight
is weights_nominal unchanged. The 'reweighting_factor' convention covers
files whose nominal column stores only the learned factor, where the final
weight is weight_mc * weights_nominal. See spec/weight_formula.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from omnifold_publication import (
    PackageReadError,
    load_analysis,
    load_package,
    write_manifest,
    write_package,
)
from omnifold_publication.exceptions import PackageWriteError


def _write_analysis(tmp_path, source_hdf):
    nominal = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "nominal",
        event_count=6,
    )
    manifest_dir = tmp_path / "analysis"
    write_manifest(
        output_dir=manifest_dir,
        nominal_path=nominal,
        variations={},
        analysis_name="final-weight-test",
    )
    return manifest_dir


def _set_nominal_convention(package_dir, convention):
    metadata_path = package_dir / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    if convention is None:
        metadata["weights"].pop("nominal_convention", None)
    else:
        metadata["weights"]["nominal_convention"] = convention
    metadata_path.write_text(
        yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8"
    )


def test_final_weights_default_to_nominal(tmp_path, source_hdf):
    package_dir = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "package",
        event_count=6,
    )
    package = load_package(package_dir)

    assert package.nominal_convention() == "includes_mc_weight"
    expected = package.load_events(columns=["weights_nominal"])[
        "weights_nominal"
    ].to_numpy()
    np.testing.assert_allclose(package.get_weights("final"), expected)


def test_final_weights_ignore_missing_base_column_by_default(
    tmp_path, source_hdf
):
    package_dir = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "package",
        event_count=6,
    )
    events_path = package_dir / "events.parquet"
    events = pd.read_parquet(events_path).drop(columns=["weight_mc"])
    events.to_parquet(events_path, index=False)

    package = load_package(package_dir)
    np.testing.assert_allclose(
        package.get_weights("final"),
        events["weights_nominal"].to_numpy(),
    )


def test_final_weights_reweighting_factor_convention(tmp_path, source_hdf):
    package_dir = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "package",
        event_count=6,
        nominal_convention="reweighting_factor",
    )
    package = load_package(package_dir)

    assert package.nominal_convention() == "reweighting_factor"
    events = package.load_events(columns=["weight_mc", "weights_nominal"])
    expected = (
        events["weight_mc"].to_numpy()
        * events["weights_nominal"].to_numpy()
    )
    np.testing.assert_allclose(package.get_weights("final"), expected)


def test_final_weights_factor_convention_missing_base_column_raises(
    tmp_path, source_hdf
):
    package_dir = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "package",
        event_count=6,
        nominal_convention="reweighting_factor",
    )
    events_path = package_dir / "events.parquet"
    events = pd.read_parquet(events_path).drop(columns=["weight_mc"])
    events.to_parquet(events_path, index=False)

    package = load_package(package_dir)
    with pytest.raises(PackageReadError, match="required columns"):
        package.get_weights("final")


def test_final_weights_legacy_metadata_defaults_to_includes_mc(
    tmp_path, source_hdf
):
    package_dir = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "package",
        event_count=6,
    )
    _set_nominal_convention(package_dir, None)

    package = load_package(package_dir)
    assert package.nominal_convention() == "includes_mc_weight"
    np.testing.assert_allclose(
        package.get_weights("final"),
        package.get_weights("nominal"),
    )


def test_final_weights_invalid_convention_raises(tmp_path, source_hdf):
    package_dir = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "package",
        event_count=6,
    )
    _set_nominal_convention(package_dir, "not-a-convention")

    package = load_package(package_dir)
    with pytest.raises(PackageReadError, match="nominal_convention"):
        package.get_weights("final")


def test_write_package_rejects_unknown_convention(tmp_path, source_hdf):
    with pytest.raises(PackageWriteError, match="nominal_convention"):
        write_package(
            input_path=source_hdf,
            output_dir=tmp_path / "package",
            event_count=6,
            nominal_convention="not-a-convention",
        )


def test_nominal_convention_is_not_a_weight_variation(tmp_path, source_hdf):
    package_dir = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "package",
        event_count=6,
    )
    package = load_package(package_dir)

    assert "nominal_convention" not in package.list_weights()
    with pytest.raises(PackageReadError, match="Unknown metadata-declared"):
        package.get_weights("nominal_convention")


def test_nominal_weights_are_unchanged(tmp_path, source_hdf):
    package_dir = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "package",
        event_count=6,
    )
    package = load_package(package_dir)

    expected = package.load_events(columns=["weights_nominal"])[
        "weights_nominal"
    ].to_numpy()
    np.testing.assert_allclose(package.get_weights("nominal"), expected)


def test_analysis_final_weights_work(tmp_path, source_hdf):
    analysis = load_analysis(_write_analysis(tmp_path, source_hdf))

    np.testing.assert_allclose(
        analysis.get_weights("final"),
        analysis.nominal_package.get_weights("final"),
    )

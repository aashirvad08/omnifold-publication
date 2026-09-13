"""Regression tests for the package's API contracts.

Each test here pins a property the codebase claims about itself:

1. metadata is parsed once into the typed model, and the reader navigates
   that model rather than re-walking raw mappings;
2. variation names resolve only through what the metadata *declares* —
   never through a guessed column spelling;
3. uncertainty-component grouping is public API, not a private helper;
4. ``get_weights`` has one parameter for the thing it selects;
5. ``HistogramResult`` has the same guaranteed fields whatever built it.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from conftest import set_official_bins

from omnifold_publication import (
    HistogramResult,
    PackageReadError,
    component_group,
    group_components,
    load_analysis,
    load_hepdata_submission,
    load_package,
    write_manifest,
    write_package,
)

BINS = [0.0, 50.0, 100.0, 150.0, 200.0]


def _corrupt(package_dir, mutate):
    path = package_dir / "metadata.yaml"
    metadata = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(metadata)
    path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")


def _analysis(tmp_path, source_hdf):
    nominal = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "nominal",
        event_count=6,
        include_all_replicas=True,
    )
    variation = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "variation",
        event_count=6,
    )
    manifest_dir = tmp_path / "analysis"
    write_manifest(
        output_dir=manifest_dir,
        nominal_path=nominal,
        variations={
            "generator_choice": {
                "path": variation,
                "type": "alternative_generator",
            }
        },
        analysis_name="contract-test",
    )
    return load_analysis(manifest_dir)


# --- 1. the typed model is the reader's source of truth -------------------


def test_malformed_metadata_is_rejected_at_load_naming_the_field(
    tmp_path, source_hdf
):
    """A schema violation surfaces once, at load, with the field named."""

    package_dir = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "pkg",
        event_count=6,
    )
    _corrupt(package_dir, lambda m: m["weights"].update(nominal=42))

    with pytest.raises(PackageReadError, match="weights.nominal"):
        load_package(package_dir)


def test_observables_declared_as_a_mapping_are_rejected(tmp_path, source_hdf):
    package_dir = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "pkg",
        event_count=6,
    )
    _corrupt(package_dir, lambda m: m.update(observables={"not": "a list"}))

    with pytest.raises(PackageReadError, match="observables"):
        load_package(package_dir)


def test_model_preserves_every_declared_weight_key_in_order(
    tmp_path, source_hdf
):
    """The writer declares one key per replica column; none may be dropped.

    Guards the schema's ``extra="allow"``: pydantic's default would parse
    these away, leaving the model disagreeing with the file it came from.
    """

    package_dir = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "pkg",
        event_count=6,
        include_all_replicas=True,
    )
    package = load_package(package_dir)
    raw = yaml.safe_load(
        (package_dir / "metadata.yaml").read_text(encoding="utf-8")
    )
    skipped = {"iterations", "nominal_convention", "families"}
    expected = [key for key in raw["weights"] if key not in skipped]

    assert package.list_weights() == expected
    assert any(name.startswith("weights_ensemble_") for name in expected)


def test_weight_family_shape_is_uniform(tmp_path, atlas_like_hdf):
    """Every family exposes the same keys, whether or not it has a reference."""

    package = load_package(
        write_package(
            input_path=atlas_like_hdf,
            output_dir=tmp_path / "pkg",
            include_all_replicas=True,
        )
    )
    expected = {"type", "combination", "columns", "reference_column"}
    for name in package.list_weight_families():
        assert set(package.weight_family(name)) == expected

    assert package.weight_family("dd_unfolding")["reference_column"] == "target_dd"
    assert package.weight_family("ensemble")["reference_column"] is None


def test_legacy_explicit_bins_key_still_resolves(tmp_path, source_hdf):
    """The model must keep honouring the explicit per-observable `bins` key."""

    package_dir = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "pkg",
        event_count=6,
    )

    def set_bins(metadata):
        metadata["observables"][0]["bins"] = [0.0, 60.0, 120.0]

    _corrupt(package_dir, set_bins)
    assert load_package(package_dir).observable_bins("pT_ll") == [0.0, 60.0, 120.0]


# --- 2. declared names only, never a guessed spelling ---------------------


def test_analysis_resolves_the_declared_weight_name(tmp_path, source_hdf):
    analysis = _analysis(tmp_path, source_hdf)

    weights = analysis.get_weights("weights_ensemble_0")
    assert len(weights) == 6


def test_analysis_refuses_an_undeclared_prefix_stripped_name(
    tmp_path, source_hdf
):
    """``ensemble_0`` is not declared; only ``weights_ensemble_0`` is.

    The previous cascade prepended ``weights_`` and resolved it anyway,
    which is exactly the name-guessing the uncertainty stack refuses to do.
    """

    analysis = _analysis(tmp_path, source_hdf)

    with pytest.raises(PackageReadError, match="Unknown analysis variation"):
        analysis.get_weights("ensemble_0")


def test_analysis_unknown_variation_names_what_it_looked_for(
    tmp_path, source_hdf
):
    analysis = _analysis(tmp_path, source_hdf)

    with pytest.raises(PackageReadError) as info:
        analysis.get_weights("no_such_thing")
    message = str(info.value)
    assert "manifest sample" in message
    assert "generator_choice" in message


def test_analysis_still_resolves_manifest_samples_and_nominal(
    tmp_path, source_hdf
):
    analysis = _analysis(tmp_path, source_hdf)

    assert len(analysis.get_weights("nominal")) == 6
    assert len(analysis.get_weights("final")) == 6
    assert len(analysis.get_weights("generator_choice")) == 6


# --- 3. component grouping is public --------------------------------------


def test_component_group_is_public_and_uses_declared_types(
    tmp_path, atlas_like_hdf
):
    package = load_package(
        write_package(
            input_path=atlas_like_hdf,
            output_dir=tmp_path / "pkg",
            include_all_replicas=True,
        )
    )

    assert component_group(package, "sample_stat") == "statistical"
    assert component_group(package, "ensemble") == "statistical"
    assert component_group(package, "bootstrap_mc") == "statistical"
    assert component_group(package, "syst_theory") == "systematic"
    assert component_group(package, "dd_unfolding") == "data_driven"
    # no declared family: grouped, never raised
    assert component_group(package, "two_point_sherpa") == "other"


def test_group_components_matches_manual_quadrature(tmp_path, atlas_like_hdf):
    package_dir = write_package(
        input_path=atlas_like_hdf,
        output_dir=tmp_path / "pkg",
        include_all_replicas=True,
    )
    set_official_bins(package_dir, {"pT_ll": BINS})
    package = load_package(package_dir)
    breakdown = package.uncertainty_breakdown("pT_ll")

    grouped = group_components(package, breakdown["components"])

    expected: dict[str, np.ndarray] = {}
    for name, values in breakdown["components"].items():
        group = component_group(package, name)
        expected[group] = expected.get(group, 0.0) + np.asarray(values) ** 2

    assert set(grouped) == set(expected)
    for group, squared in expected.items():
        np.testing.assert_allclose(grouped[group], np.sqrt(squared))

    # grouping must conserve the total
    total = np.sqrt(sum(values**2 for values in grouped.values()))
    np.testing.assert_allclose(total, breakdown["total"])


# --- 4. one parameter for the selected weight -----------------------------


def test_get_weights_takes_variation_positionally(tmp_path, source_hdf):
    package = load_package(
        write_package(
            input_path=source_hdf,
            output_dir=tmp_path / "pkg",
            event_count=6,
            include_all_replicas=True,
        )
    )

    np.testing.assert_allclose(
        package.get_weights("weights_ensemble_0"),
        package.get_weights(variation="weights_ensemble_0"),
    )


def test_get_weights_kind_is_deprecated_but_works(tmp_path, source_hdf):
    package = load_package(
        write_package(
            input_path=source_hdf,
            output_dir=tmp_path / "pkg",
            event_count=6,
        )
    )

    with pytest.warns(DeprecationWarning, match="kind"):
        legacy = package.get_weights(kind="nominal")
    np.testing.assert_allclose(legacy, package.get_weights("nominal"))


def test_get_weights_rejects_both_spellings_at_once(tmp_path, source_hdf):
    package = load_package(
        write_package(
            input_path=source_hdf,
            output_dir=tmp_path / "pkg",
            event_count=6,
        )
    )

    with pytest.raises(PackageReadError, match="not both"):
        package.get_weights(variation="weights_dd", kind="nominal")


# --- 5. HistogramResult keeps the same guaranteed shape -------------------


def test_total_uncertainty_is_present_from_a_package(tmp_path, atlas_like_hdf):
    package_dir = write_package(
        input_path=atlas_like_hdf,
        output_dir=tmp_path / "pkg",
        include_all_replicas=True,
    )
    set_official_bins(package_dir, {"pT_ll": BINS})
    result = load_package(package_dir).histogram("pT_ll")

    assert result.total_uncertainty is not None
    # one sample: no systematic or replica band, so the total is the stat term
    np.testing.assert_allclose(result.total_uncertainty, result.stat_uncertainty)
    assert set(result.components()) == {"stat_uncertainty"}


def test_total_uncertainty_is_present_from_an_analysis(tmp_path, source_hdf):
    analysis = _analysis(tmp_path, source_hdf)

    result = analysis.histogram(
        "pT_ll", systematic_variations=["generator_choice"], bins=3
    )

    assert result.total_uncertainty is not None
    expected = np.sqrt(
        result.stat_uncertainty**2
        + result.sys_uncertainty**2
        + result.replica_uncertainty**2
    )
    np.testing.assert_allclose(result.total_uncertainty, expected)
    assert set(result.components()) == {
        "stat_uncertainty",
        "sys_uncertainty",
        "replica_uncertainty",
    }


def test_total_uncertainty_is_present_from_hepdata(tmp_path, atlas_like_hdf):
    package_dir = write_package(
        input_path=atlas_like_hdf,
        output_dir=tmp_path / "pkg",
        include_all_replicas=True,
    )
    set_official_bins(package_dir, {"pT_ll": BINS})
    submission = load_package(package_dir).export_hepdata(tmp_path / "hepdata")

    result = load_hepdata_submission(submission).histogram("pT_ll")

    assert result.total_uncertainty is not None
    assert result.total_uncertainty.shape == result.hist.shape


def test_total_uncertainty_survives_a_json_round_trip():
    result = HistogramResult(
        hist=np.array([1.0, 2.0]),
        edges=np.array([0.0, 1.0, 2.0]),
        centers=np.array([0.5, 1.5]),
        stat_uncertainty=np.array([0.1, 0.2]),
    )

    payload = result.to_dict()

    assert payload["total_uncertainty"] == [0.1, 0.2]
    assert "sys_uncertainty" not in payload

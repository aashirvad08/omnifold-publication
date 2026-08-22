"""Cross-publication comparison + the write_package parquet/rename/method
extension. Fixtures go through the real publication + loading path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import make_atlas_like_frame

from omnifold_publication import (
    CrossPublicationComparison,
    compare_publications,
    load_package,
    write_package,
)
from omnifold_publication.exceptions import PackageReadError

BINS = [0.0, 50.0, 100.0, 150.0, 200.0]


def _hdf5_source(path: Path, seed: int) -> Path:
    make_atlas_like_frame(n=400, seed=seed).to_hdf(path, key="df", mode="w")
    return path


def _omnifold_like_parquet(path: Path, seed: int) -> Path:
    """A differently-named parquet: truth_-prefixed observable and a
    'weights_prior' base column, mimicking an independent publication."""

    df = make_atlas_like_frame(n=400, seed=seed).rename(
        columns={"pT_ll": "truth_pT_ll", "weight_mc": "weights_prior"}
    )
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def multifold_pkg(tmp_path):
    src = _hdf5_source(tmp_path / "mf.h5", seed=1)
    return load_package(
        write_package(
            input_path=src, output_dir=tmp_path / "mf_pkg",
            observables=["pT_ll"], include_all_replicas=True, method="MultiFold",
        )
    )


@pytest.fixture
def omnifold_pkg(tmp_path):
    src = _omnifold_like_parquet(tmp_path / "of.parquet", seed=2)
    return load_package(
        write_package(
            input_path=src, output_dir=tmp_path / "of_pkg",
            observables=["pT_ll"], include_all_replicas=True,
            column_rename={"truth_pT_ll": "pT_ll", "weights_prior": "weight_mc"},
            method="OmniFold",
            assumptions=["all events used", "hv provenance undocumented"],
        )
    )


# --- writer extension ----------------------------------------------------


def test_writer_ingests_parquet_with_rename_and_method(omnifold_pkg):
    assert omnifold_pkg.list_observables() == ["pT_ll"]
    meta = omnifold_pkg.metadata()
    assert meta["dataset"]["method"] == "OmniFold"
    assert meta["dataset"]["assumptions"] == [
        "all events used", "hv provenance undocumented"
    ]
    # renamed base column is packaged and the observable histograms
    hist = omnifold_pkg.histogram("pT_ll", bins=BINS)
    assert len(hist.hist) == len(BINS) - 1
    omnifold_pkg.validate()


def test_writer_defaults_are_backward_compatible(tmp_path):
    """New params default to no-ops: an HDF5 package built the old way gains
    no method/assumptions and reads exactly as before."""

    src = _hdf5_source(tmp_path / "s.h5", seed=3)
    package_dir = write_package(
        input_path=src, output_dir=tmp_path / "pkg", observables=["pT_ll"],
    )
    meta = yaml.safe_load((package_dir / "metadata.yaml").read_text())
    assert "method" not in meta["dataset"]
    assert "assumptions" not in meta["dataset"]
    # and the normal load/histogram path is unaffected
    assert load_package(package_dir).list_observables() == ["pT_ll"]


def test_full_sample_records_luminosity_assumption(tmp_path):
    """Publishing every event records the luminosity assumption in the package
    itself, so it stays visible to anyone using the comparison later."""

    src = _hdf5_source(tmp_path / "full.h5", seed=7)
    package_dir = write_package(
        input_path=src, output_dir=tmp_path / "pkg", observables=["pT_ll"],
        event_count=None,
    )
    assumptions = yaml.safe_load(
        (package_dir / "metadata.yaml").read_text()
    )["dataset"]["assumptions"]
    assert any("all events used" in note for note in assumptions)
    # caller-supplied assumptions are preserved alongside it, not replaced
    package_dir = write_package(
        input_path=src, output_dir=tmp_path / "pkg2", observables=["pT_ll"],
        event_count=None, assumptions=["custom caveat"],
    )
    assumptions = yaml.safe_load(
        (package_dir / "metadata.yaml").read_text()
    )["dataset"]["assumptions"]
    assert "custom caveat" in assumptions and len(assumptions) == 2


def test_parquet_scalar_only_skips_array_columns(tmp_path):
    df = make_atlas_like_frame(n=50, seed=4)
    df["truth_pT_particles"] = [np.arange(3, dtype=float) for _ in range(len(df))]
    src = tmp_path / "arr.parquet"
    df.to_parquet(src, index=False)
    # must not raise on the variable-length array column; it is simply skipped
    package = load_package(
        write_package(
            input_path=src, output_dir=tmp_path / "pkg", observables=["pT_ll"],
        )
    )
    assert "truth_pT_particles" not in package.load_events().columns


# --- cross-publication comparison ---------------------------------------


def test_two_packages_each_get_their_own_uncertainty(
    multifold_pkg, omnifold_pkg
):
    cmp = compare_publications(
        multifold_pkg, omnifold_pkg, "pT_ll", bins=BINS,
        labels=("MultiFold", "OmniFold"),
    )
    d = cmp.to_dict()
    assert d["kind"] == "cross_publication_comparison"
    assert d["bins"] == BINS
    a, b = d["sides"]
    assert a["method"] == "MultiFold" and b["method"] == "OmniFold"
    # distinct provenance: two independent publications
    assert a["checksum_sha256"] != b["checksum_sha256"]
    # each side carries its own hist + full uncertainty breakdown
    for side in (a, b):
        assert len(side["hist"]) == len(BINS) - 1
        assert len(side["uncertainty"]["total"]) == len(BINS) - 1
        assert {"sample_stat", "ensemble", "bootstrap_mc"} <= set(
            side["uncertainty"]["components"]
        )
    # each side's uncertainty matches a direct breakdown on that package
    direct = omnifold_pkg.uncertainty_breakdown("pT_ll", bins=BINS)
    np.testing.assert_allclose(b["uncertainty"]["total"], direct["total"])
    np.testing.assert_allclose(b["hist"], direct["nominal"])


def test_ratio_is_between_the_two_results(multifold_pkg, omnifold_pkg):
    cmp = compare_publications(multifold_pkg, omnifold_pkg, "pT_ll", bins=BINS)
    ratio = cmp.to_dict()["ratio"]
    a_hist = np.asarray(cmp.to_dict()["sides"][0]["hist"])
    b_hist = np.asarray(cmp.to_dict()["sides"][1]["hist"])
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.where(a_hist != 0, b_hist / a_hist, np.nan)
    np.testing.assert_allclose(ratio["values"], expected, equal_nan=True)
    # each side's band is kept separate; no combined/correlated band
    assert ratio["correlation"] == "none"
    assert len(ratio["reference_rel_band"]) == len(BINS) - 1
    assert len(ratio["comparand_rel_band"]) == len(BINS) - 1


def test_rebinning_refills_from_raw_events(multifold_pkg, omnifold_pkg):
    # two different bin requests re-fill from raw events -> different results
    coarse = compare_publications(
        multifold_pkg, omnifold_pkg, "pT_ll", bins=[0.0, 100.0, 200.0]
    )
    fine = compare_publications(
        multifold_pkg, omnifold_pkg, "pT_ll", bins=BINS
    )
    assert coarse.bins == [0.0, 100.0, 200.0]
    assert len(coarse.to_dict()["sides"][0]["hist"]) == 2
    assert len(fine.to_dict()["sides"][0]["hist"]) == 4
    assert "no interpolation" in fine.to_dict()["provenance"]["rebinning"]


def test_default_bins_use_package_a_official_binning(
    multifold_pkg, omnifold_pkg
):
    # pT_ll declares the official IBU binning in the shared metadata
    cmp = compare_publications(multifold_pkg, omnifold_pkg, "pT_ll")
    assert cmp.bins == [200.0, 230.0, 300.0, 450.0, 600.0, 1000.0]


def test_correlation_model_is_explicit_not_baked(multifold_pkg, omnifold_pkg):
    with pytest.raises(NotImplementedError, match="correlation"):
        compare_publications(
            multifold_pkg, omnifold_pkg, "pT_ll", bins=BINS,
            correlation="independent",
        )
    with pytest.raises(PackageReadError, match="correlation"):
        CrossPublicationComparison(
            multifold_pkg, omnifold_pkg, "pT_ll", bins=BINS,
            correlation="bogus",
        )


def test_export_json_round_trips(tmp_path, multifold_pkg, omnifold_pkg):
    import json

    cmp = compare_publications(multifold_pkg, omnifold_pkg, "pT_ll", bins=BINS)
    out = tmp_path / "cmp.json"
    cmp.export_json(out)
    loaded = json.loads(out.read_text())
    assert loaded["observable"] == "pT_ll"
    assert [s["method"] for s in loaded["sides"]] == ["MultiFold", "OmniFold"]


def test_uncertainty_comparison_traces_directly_to_breakdown(
    multifold_pkg, omnifold_pkg
):
    """Every number in the uncertainty-comparison view must come straight
    from a fresh uncertainty_breakdown() call on each package, never a
    recomputation from raw events."""

    cmp = compare_publications(multifold_pkg, omnifold_pkg, "pT_ll", bins=BINS)
    d = cmp.uncertainty_comparison_to_dict()
    assert d["kind"] == "cross_publication_uncertainty_comparison"
    assert d["provenance"]["correlation_model"] == "none"

    a_direct = multifold_pkg.uncertainty_breakdown("pT_ll", bins=BINS)
    a_side, b_side = d["sides"]
    expected_total_pct = 100.0 * np.asarray(a_direct["total"]) / np.abs(
        np.asarray(a_direct["nominal"])
    )
    np.testing.assert_allclose(a_side["total_relative_pct"], expected_total_pct)

    # the type-grouped rollup reconstructs the same total in quadrature
    for side in (a_side, b_side):
        groups = np.stack(
            [np.asarray(v) for v in side["group_relative_pct"].values()]
        )
        recon = np.sqrt((groups**2).sum(axis=0))
        np.testing.assert_allclose(recon, side["total_relative_pct"], atol=1e-8)
        assert set(side["component_group"].values()) <= {
            "statistical", "systematic", "data_driven"
        }


def test_export_uncertainty_comparison_json_round_trips(
    tmp_path, multifold_pkg, omnifold_pkg
):
    import json

    cmp = compare_publications(multifold_pkg, omnifold_pkg, "pT_ll", bins=BINS)
    out = tmp_path / "unc_cmp.json"
    cmp.export_uncertainty_comparison_json(out)
    loaded = json.loads(out.read_text())
    assert loaded["kind"] == "cross_publication_uncertainty_comparison"
    assert len(loaded["sides"]) == 2


def test_plot_uncertainty_comparison_saves_figure(
    tmp_path, multifold_pkg, omnifold_pkg
):
    import matplotlib.pyplot as plt

    cmp = compare_publications(multifold_pkg, omnifold_pkg, "pT_ll", bins=BINS)
    out = tmp_path / "unc_cmp.png"
    fig = cmp.plot_uncertainty_comparison(out)
    assert out.exists() and out.stat().st_size > 0
    plt.close(fig)


def test_real_multifold_vs_omnifold(tmp_path):
    """The real deliverable, gated on the actual data files."""

    mf_src = Path("data/multifold.h5")
    of_src = Path("data/omnifold_data/data.parquet")
    if not (mf_src.exists() and of_src.exists()):
        pytest.skip("real MultiFold/OmniFold data not available")

    # full data: weights_nominal only sums to the measured cross-section over
    # all events, so a normalized ratio requires the complete sample, not a
    # subset (a subset would just scale each side by its own event fraction).
    mf = load_package(
        write_package(
            input_path=mf_src, output_dir=tmp_path / "mf",
            event_count=None, observables=["pT_ll"],
            include_all_replicas=True, method="MultiFold",
        )
    )
    of = load_package(
        write_package(
            input_path=of_src, output_dir=tmp_path / "of",
            event_count=None, observables=["pT_ll"],
            include_all_replicas=True, method="OmniFold",
            column_rename={"truth_pT_ll": "pT_ll", "weights_prior": "weight_mc"},
        )
    )
    cmp = compare_publications(mf, of, "pT_ll", labels=("MultiFold", "OmniFold"))
    d = cmp.to_dict()

    a, b = d["sides"]
    assert a["checksum_sha256"] != b["checksum_sha256"]
    # both are boosted-Z cross-sections on the official pT_ll binning
    assert d["bins"][0] == 200.0
    assert sum(a["hist"]) > 0 and sum(b["hist"]) > 0
    # both bands are real and positive
    assert all(u > 0 for u in a["uncertainty"]["total"])
    assert all(u > 0 for u in b["uncertainty"]["total"])
    # two independent unfoldings of the same measurement agree within ~10%
    ratio = np.asarray(d["ratio"]["values"])
    assert np.all(np.abs(ratio - 1.0) < 0.1)

    # both publications are systematic-dominated in every bin, not
    # statistics-dominated -- a real finding from this data, pinned here as
    # a regression check rather than assumed
    unc = cmp.uncertainty_comparison_to_dict()
    for side in unc["sides"]:
        stat = np.asarray(side["group_relative_pct"]["statistical"])
        syst = np.asarray(side["group_relative_pct"]["systematic"])
        assert np.all(syst > stat)

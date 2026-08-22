from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_atlas_like_frame(n=60, seed=42):
    """Synthetic events with NP systematics, bootstraps, ensemble, dd pair."""

    rng = np.random.default_rng(seed)
    base = rng.uniform(0.5, 1.5, n)
    nominal = base * rng.normal(1.0, 0.1, n)

    def varied(scale=0.05):
        return nominal * rng.normal(1.0, scale, n)

    # ~1/5 of events carry a below-threshold "jet" (pT <= 5) with tau1 = 0,
    # mimicking the release's jetless events that the pT_trackj1 > 5
    # selection must remove
    has_jet = rng.uniform(0.0, 1.0, n) > 0.2
    pT_trackj1 = np.where(has_jet, rng.uniform(6.0, 300.0, n), rng.uniform(0.5, 5.0, n))
    tau1 = np.where(has_jet, rng.uniform(0.05, 0.9, n), 0.0)

    data = {
        "event_id": np.arange(n),
        "pT_ll": rng.uniform(10.0, 200.0, n),
        "pT_l1": rng.uniform(25.0, 150.0, n),
        "eta_l1": rng.uniform(-2.5, 2.5, n),
        "phi_l1": rng.uniform(-np.pi, np.pi, n),
        "pT_l2": rng.uniform(25.0, 100.0, n),
        "eta_l2": rng.uniform(-2.5, 2.5, n),
        "phi_l2": rng.uniform(-np.pi, np.pi, n),
        "pT_trackj1": pT_trackj1,
        "y_trackj1": rng.uniform(-2.5, 2.5, n),
        "phi_trackj1": rng.uniform(-np.pi, np.pi, n),
        "tau1_trackj1": tau1,
        "tau2_trackj1": rng.uniform(0.0, 0.5, n),
        "weight_mc": base,
        "weights_nominal": nominal,
        "weights_pileup": varied(),
        "weights_muEffReco": varied(),
        "weights_theoryPDF": varied(),
        "weights_theoryQCD": varied(),
        "weights_trackFake": varied(),
        "weights_muCalID": varied(),
        "weights_lumi": varied(0.017),
        "weights_topBackground": varied(),
        "weights_dd": varied(),
        "target_dd": varied(),
    }
    for i in range(3):
        data[f"weights_bootstrap_mc_{i}"] = varied()
        data[f"weights_bootstrap_data_{i}"] = varied()
    for i in range(4):
        data[f"weights_ensemble_{i}"] = varied()

    return pd.DataFrame(data)


@pytest.fixture
def atlas_like_hdf(tmp_path):
    """Small ATLAS-like source file shared across family/covariance tests."""

    path = tmp_path / "atlas_like.h5"
    make_atlas_like_frame().to_hdf(path, key="df", mode="w")
    return path


def set_official_bins(package_dir, mapping):
    """Patch a package's official binning to fixture-appropriate ranges."""

    import yaml

    metadata_path = package_dir / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    for observable in metadata["observables"]:
        if observable["name"] in mapping:
            observable["binning"] = {
                "official": mapping[observable["name"]],
                "provenance": "test fixture",
            }
    metadata_path.write_text(
        yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8"
    )


@pytest.fixture
def source_hdf(tmp_path):
    """Small HDF5 source file with nominal, systematic, and iteration weights."""

    path = tmp_path / "source_multifold.h5"
    df = pd.DataFrame(
        {
            "event_id": np.arange(6),
            "pT_ll": np.array([10.0, 20.0, 35.0, 50.0, 80.0, 120.0]),
            "pT_l1": np.array([25.0, 28.0, 32.0, 40.0, 45.0, 60.0]),
            "weight_mc": np.array([1.0, 1.1, 0.9, 1.2, 1.0, 0.8]),
            "weights_nominal": np.array([1.0, 1.2, 0.8, 1.1, 0.9, 1.3]),
            "weights_ensemble_0": np.array([1.1, 1.1, 0.9, 1.0, 1.0, 1.2]),
            "weights_iter0_step1": np.array([0.9, 1.0, 1.1, 1.0, 0.95, 1.05]),
            "weights_iter0_step2": np.array([1.0, 1.1, 1.0, 1.2, 1.0, 0.9]),
        }
    )
    df.to_hdf(path, key="df", mode="w")
    return path

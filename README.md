# omnifold-publication

Turn event-level OmniFold/MultiFold unfolding output into a self-describing, validated, publishable data product 
and then use it for physics analysis.
omnifold-publication addresses a key reproducibility gap: OmniFold results are typically shared as event tables with many weight columns
but without machine-readable information about what those weights represent  bootstrap replicas, systematic uncertainties, nominal normalization, or binning.
This makes it possible to apply the weights, but difficult to reproduce published results.
The package solves this by pairing each event table with a metadata contract and implementing analysis recipes based on that contract rather than column-name conventions:

- weighted histograms with `sqrt(sum w^2)` errors
- per-family uncertainty breakdowns and a quadrature total
- full bin-to-bin covariance and correlation matrices
- chi-squared closure tests with a circularity guard
- effective-statistics binning helpers and per-bin validity checks
- HEPData export **and** import
- cross-publication comparison of two independent unfoldings

The recipes are ports of the ATLAS Z+jets OmniFold release
(`sm-z-jets-omnifold-2024`, [arXiv:2405.20041](https://arxiv.org/abs/2405.20041),
HEPData record `ins2791852`), with the source of each formula recorded in
`spec/uncertainty_recipes.md` and in the module docstrings.

---

## Table of contents

- [What a publication package is](#what-a-publication-package-is)
- [Install](#install)
- [Get the data](#get-the-data)
- [Quickstart](#quickstart)
- [Command-line interface](#command-line-interface)
- [Python API by task](#python-api-by-task)
  - [1. Write a package](#1-write-a-package)
  - [2. Load, inspect, validate](#2-load-inspect-validate)
  - [3. Weighted histograms and plots](#3-weighted-histograms-and-plots)
  - [4. Uncertainty breakdown](#4-uncertainty-breakdown)
  - [5. Covariance and correlation](#5-covariance-and-correlation)
  - [6. Binning: effective statistics and validity](#6-binning-effective-statistics-and-validity)
  - [7. Multi-sample analyses](#7-multi-sample-analyses)
  - [8. Closure tests](#8-closure-tests)
  - [9. Cross-publication comparison](#9-cross-publication-comparison)
  - [10. HEPData export and import](#10-hepdata-export-and-import)
  - [11. Data sources](#11-data-sources)
  - [12. Derived observables](#12-derived-observables)
  - [13. JSON export helpers](#13-json-export-helpers)
- [Using your own dataset](#using-your-own-dataset)
- [The metadata contract](#the-metadata-contract)
- [Uncertainty recipes](#uncertainty-recipes)
- [Run the tests](#run-the-tests)
- [Repository layout](#repository-layout)
- [Publishing your result](#publishing-your-result)
- [Gotchas](#gotchas)
- [Further documentation](#further-documentation)
- [Citing](#citing)
- [License](#license)

---

## What a publication package is

A **package** is a directory:

```
my_result/
├── events.parquet     # one row per event: observables + every weight column
└── metadata.yaml      # the contract: observables, binning, weight families, normalisation, checksum
```

`metadata.yaml` declares, among other things:

- **observables** — name, description, units, official binning (with provenance),
  and any per-observable event selection
- **weights** — the nominal column, the base MC weight, the convention relating
  them, and **weight families**: named groups of columns, each with a `type`
  (`systematic` / `bootstrap` / `ensemble` / `paired`) and a `combination` recipe
- **normalization** — absolute vs shape, weight units, the expected `sum(w)`
- **publication** — event count, column list, event-alignment method, and a
  SHA-256 of `events.parquet`

Because the uncertainty machinery dispatches on the *declared* `combination`,
never on column-name patterns, any naming scheme works as long as the metadata
describes it.

An **analysis** is a directory with a `manifest.yaml` linking several packages
(a nominal, optional variation samples, and an optional truth target). That is
what unlocks two-point systematics from alternative generators.

---

## Install

Requires Python **3.10+** (tested on 3.11).

```bash
git clone https://github.com/aashirvad08/omnifold-publication.git
cd omnifold-publication

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1

pip install -e .                   # the package + its dependencies
```

Or install the pinned development set (adds `pytest` and `hepdata-validator`):

```bash
pip install -r requirements.txt
pip install -e .
```

Optional extras:

```bash
pip install -e ".[dev]"            # pytest, hepdata-validator, ruff, mypy
pip install -e ".[sources]"        # fsspec[http] + requests: remote data sources
```

Verify:

```bash
python -c "import omnifold_publication as o; print(len(o.__all__), 'public names')"
omnifold-publication --help
```

Runtime dependencies are `numpy`, `pandas`, `pyarrow`, `pydantic>=2`, `PyYAML`,
`tables` (HDF5), `scipy`, `matplotlib`, and `vector`. All are hard requirements —
importing the top-level package pulls in the closure (scipy), plotting
(matplotlib), and derived-observable (vector) code paths.

---

## Get the data

**No data ships with this repository.** The event files are hundreds of MB to
several GB, and `.gitignore` blocks `data/*`, `artifacts/`, `*.h5`, and
`*.parquet` so they can never be committed by accident.

The reference inputs are the event-level weight files of the ATLAS Z+jets
OmniFold release (`sm-z-jets-omnifold-2024`). Put them — or copy/symlink them —
under `data/`:

```
data/
├── multifold.h5            # nominal OmniFold output, full weight content
├── multifold_sherpa.h5     # alternative-generator variation
└── multifold_nonDY.h5      # alternative sample (nominal weights only)
```

```bash
mkdir -p data
ln -s /path/to/multifold.h5 data/multifold.h5      # symlink is fine; nothing is copied
```

Everything works with any HDF5 (key `df`) or Parquet file that carries an
observable column and weight columns — see
[Using your own dataset](#using-your-own-dataset).

The tests that need real data **skip themselves cleanly** when it is absent, so
a bare clone still runs a full green suite.

`docs/data_provenance.md` records exactly how the local files relate to the
published release (they are a reduced, regenerated version — same event count,
fewer replica columns, ~7% different `n_eff`), and every package written by
`write_package` carries that caveat forward in
`metadata.dataset.provenance_note`.

---

## Quickstart

From raw file to a validated package to a plot, in one script:

```python
from omnifold_publication import write_package, load_package

# 1. package a raw file (auto-discovers weight families)
pkg_dir = write_package(
    input_path="data/multifold.h5",
    output_dir="artifacts/my_result",
    event_count=None,               # None = all events
    observables=["pT_ll"],
    include_all_replicas=True,      # package the bootstrap/ensemble replica sets
    method="MultiFold",
)

# 2. load and check it
pkg = load_package(pkg_dir)
pkg.validate()                       # checksum + row alignment + normalisation
print(pkg.summary())
print(f"fiducial cross-section: {pkg.cross_section():.2f} {pkg.weight_units()}")

# 3. the physics
hist = pkg.histogram("pT_ll")                    # official binning by default
brk  = pkg.uncertainty_breakdown("pT_ll")        # per-family components + total
cov  = pkg.covariance_matrix("pT_ll")            # matrices + correlation

# 4. a plot
import matplotlib.pyplot as plt
import numpy as np

widths = np.diff(brk["edges"])
fig, ax = plt.subplots()
ax.stairs(brk["nominal"] / widths, brk["edges"], baseline=None, label="MultiFold")
lo = (brk["nominal"] - brk["total"]) / widths
hi = (brk["nominal"] + brk["total"]) / widths
ax.fill_between(
    brk["edges"], np.append(lo, lo[-1]), np.append(hi, hi[-1]),
    step="post", alpha=0.25, linewidth=0,
)
ax.set_yscale("log")
ax.set_xlabel(r"$p_T^{\ell\ell}$ [GeV]")
ax.set_ylabel(r"$d\sigma/dp_T^{\ell\ell}$ [fb/GeV]")
ax.legend()
fig.savefig("artifacts/pT_ll.png", dpi=160)
```

Ready-made end-to-end scripts live in `examples/`:

```bash
python examples/package_roundtrip.py      # write a package, validate, verify a histogram roundtrip
python examples/reproduce_histogram.py    # reproduce a histogram from a package
python examples/example_plot.py           # weighted-histogram preview plot
python examples/pseudodata_closure.py     # the full closure chain + closure-grid figure
```

`examples/pseudodata_closure.py` is the most complete demonstration: it packages
a pseudo-measurement, builds a known target, links them in a manifest, runs
`chi2_test` per observable, shows that a deliberately wrong target is
**rejected**, runs the release's embedded `weights_dd` vs `target_dd` closure,
and renders the multi-panel closure grid.

---

## Command-line interface

Installed as `omnifold-publication` (equivalently `python -m omnifold_publication`).

```bash
omnifold-publication --help
```

| Command | Purpose |
|---|---|
| `summary <package>` | compact package summary |
| `inspect <package>` | summary plus the full column list |
| `validate <package>` | validate; exit code 1 on failure |
| `export-histogram <package>` | write one histogram as JSON |
| `export-comparison <analysis>` | write a multi-sample comparison as JSON |
| `export-metadata <analysis>` | write an analysis description/summary as JSON |

### Examples

```bash
# compact summary
omnifold-publication summary artifacts/my_result
# Package: artifacts/my_result
# Format version: 0.2
# Events: 418014
# Observables: 1
# Systematics: dd_unfolding, replica, syst_background, syst_event, ...

# summary + every column in events.parquet
omnifold-publication inspect artifacts/my_result

# validate: prints "Validation passed." or lists every error, exit 1
omnifold-publication validate artifacts/my_result
echo $?

# one histogram -> JSON  {hist, edges, centers, stat_uncertainty}
omnifold-publication export-histogram artifacts/my_result \
    --observable pT_ll --output out/hist.json

# a non-nominal weight variation, explicit bin edges
omnifold-publication export-histogram artifacts/my_result \
    --observable pT_ll --variation weights_dd \
    --bins 200 230 300 450 600 1000 --output out/hist_dd.json

# multi-sample comparison (needs a manifest directory, see section 7)
omnifold-publication export-comparison artifacts/analysis \
    --observable pT_ll --output out/comparison.json

# analysis description
omnifold-publication export-metadata artifacts/analysis --output out/analysis.json
```

The CLI is deliberately thin. Closure tests, binning helpers, covariance, and
HEPData export are Python-API only.

---

## Python API by task

### 1. Write a package

```python
from omnifold_publication import write_package

pkg_dir = write_package(
    input_path="data/multifold.h5",        # .h5 (key "df") or .parquet
    output_dir="artifacts/my_result",
    metadata_source="spec/metadata.yaml",  # observable descriptions / official binning
    event_count=None,                      # int = first N rows; None = all events
    observables=["pT_ll", "pT_l1"],
    include_systematics=True,              # package NP systematic families
    include_all_replicas=True,             # package full bootstrap/ensemble sets
    nominal_convention="includes_mc_weight",   # or "reweighting_factor"
    normalization_mode="absolute",             # or "shape"
    weight_units="fb",
    column_rename=None,                    # {"src_col": "expected_col"}
    method="MultiFold",                    # recorded in metadata.dataset.method
    assumptions=["..."],                   # free-text caveats travelling with the package
)
```

What it does: reads the source, renames columns if asked, computes any requested
derived observables, groups weight columns into families, applies each
observable's declared selection columns, writes `events.parquet`, then writes
`metadata.yaml` including the SHA-256 of the events file.

Notes:

- `include_all_replicas=False` (the default) keeps only a single representative
  replica column, which keeps demo packages small. Full uncertainty work needs
  `True`.
- `event_count=None` also records the resulting luminosity assumption in
  `dataset.assumptions` automatically — a subset scales `sum(w)` and quietly
  breaks any absolute cross-section comparison, so the package states it.

**Multi-dataset builder.** For a nominal plus variation samples in one call:

```python
from omnifold_publication import Publication

pub = Publication("zjets")
pub.add_dataset("data/multifold.h5", role="nominal", label="nominal",
                include_all_replicas=True)
pub.add_dataset("data/multifold_sherpa.h5", role="variation", label="sherpa",
                variation_type="generator")
pub.inspect()                       # print what will be built, before building
analysis = pub.build("artifacts/zjets", event_count=None, observables=["pT_ll"])
```

`inspect()` prints per-dataset event counts, observable and weight counts, and
the replica column count, so you see what you are about to publish.

### 2. Load, inspect, validate

```python
from omnifold_publication import load_package

pkg = load_package("artifacts/my_result")   # also accepts a HEPData submission dir
                                            # or "hepdata:ins2791852"

pkg.summary()                  # format version, event count, observables, weights, checksum
pkg.metadata()                 # the raw metadata mapping
pkg.load_events(columns=["pT_ll"])          # column-selective DataFrame

pkg.list_observables()
pkg.observable_units("pT_ll")               # "GeV"
pkg.observable_bins("pT_ll")                # [200.0, 230.0, 300.0, 450.0, 600.0, 1000.0]
pkg.observable_binning_provenance("pT_ll")  # where that binning came from
pkg.observable_selection("pT_trackj1")      # "pT_trackj1 > 5"
values, mask = pkg.observable_values("pT_ll")   # selection already applied

pkg.list_weights()
pkg.list_systematics()
pkg.list_weight_families()     # ['syst_event', 'syst_theory', ..., 'bootstrap_mc', 'ensemble']
pkg.weight_family("ensemble")  # {'type': 'ensemble', 'combination': 'median_standard_error', ...}
pkg.get_family_weights("ensemble")          # (n_columns, n_events) matrix

pkg.get_weights()                           # nominal
pkg.get_weights("final")                    # the canonical per-event measurement weight
pkg.get_weights("weights_dd")               # any declared variation or family column
pkg.get_weights(iteration=0, step="step1")  # a declared iteration weight
pkg.get_uncertainty("replica")              # per-event |variation - nominal|
                                            # (single weight variations only, not family names)
pkg.nominal_convention()                    # "includes_mc_weight" | "reweighting_factor"

pkg.cross_section()                         # sum of final weights
pkg.weight_units()                          # "fb"
```

Validation:

```python
from omnifold_publication import validate_package, ensure_valid_package

errors = validate_package("artifacts/my_result")   # -> list[str], empty when valid
ensure_valid_package("artifacts/my_result")        # raises PackageValidationError
pkg.validate()                                     # same, as a method
```

Checks performed: supported `format_version`, required columns present, weight
columns finite and row-aligned, the declared event-alignment contract
(`row_order` or an `event_id` column), normalisation against
`expected_nominal_sumw` within tolerance, and the recorded SHA-256 of
`events.parquet`.

Exceptions all derive from `OmniFoldPublicationError`:
`PackageReadError`, `PackageWriteError`, `PackageValidationError`,
`ManifestNotFoundError`, `UnsupportedFormatVersion`.

### 3. Weighted histograms and plots

```python
result = pkg.histogram("pT_ll")                    # official binning
result = pkg.histogram("pT_ll", bins=[200, 300, 600, 1000])
result = pkg.histogram("pT_ll", variation="weights_dd", bins=30)

result.hist, result.edges, result.centers, result.stat_uncertainty
result.total_uncertainty                           # always present, whatever the source
result.components()                                # only the components this result carries
result.to_dict()                                   # JSON-ready
```

`hist`, `edges`, `centers`, `stat_uncertainty` and `total_uncertainty` are
**always** arrays, for every source — so code that needs one error per bin can
read `total_uncertainty` without knowing which call produced the result.
`sys_uncertainty` and `replica_uncertainty` are filled only when the source can
compute them (an analysis with variation samples or replicas; see section 7),
and `components()` returns exactly the ones present rather than making you test
each field for `None`.

The standalone primitive, if you want it without a package:

```python
from omnifold_publication.histogram import compute_weighted_histogram

out = compute_weighted_histogram(
    values, weights,
    bins=50,                  # int, explicit edges, or "auto"
    hist_range=(200.0, 1000.0),
    density=False,
)
out["hist"], out["edges"], out["centers"], out["uncertainty"]   # sqrt(sum w^2)
```

Non-finite values and weights are filtered, and explicit edges are checked for
monotonicity — errors are raised, never silently absorbed.

### 4. Uncertainty breakdown

```python
brk = pkg.uncertainty_breakdown("pT_ll")           # or bins=[...] / bins=30

brk["edges"]        # bin edges
brk["nominal"]      # nominal histogram
brk["components"]   # {family_name: per-bin uncertainty}
brk["total"]        # quadrature sum of all components
```

`components` always contains `sample_stat` — the evaluation-sample statistical
term `sqrt(sum w^2)` — plus one entry per declared weight family. Each family is
combined with its own declared recipe (see
[Uncertainty recipes](#uncertainty-recipes)).

Relative composition, grouped by declared family type — the readable way to plot
a dozen families:

```python
import numpy as np
from omnifold_publication import group_components

for group, values in group_components(pkg, brk["components"]).items():
    print(group, np.round(100 * values / brk["nominal"], 2), "%")
    # statistical / systematic / data_driven / other
```

`group_components` sums each group in quadrature; `component_group(pkg, name)`
classifies one component on its own. Both resolve the group from the package's
**declared** family `type`, never from the component name.

### 5. Covariance and correlation

```python
cov = pkg.covariance_matrix("pT_ll")

cov["components"]     # {family_name: (n_bins, n_bins) matrix}
cov["total"]          # summed covariance
cov["correlation"]    # correlation matrix derived from the total
cov["nominal"], cov["edges"]
```

Each family enters in the mode its recipe implies: systematic families in
Hessian mode on the per-NP differences, bootstrap families as a sample
covariance, the ensemble family as the bootstrap covariance divided by the
replica count, and paired families as a single fully-correlated signed delta.
`sample_stat` enters as a diagonal matrix.

Lower-level building blocks, if you want to assemble a covariance yourself:

```python
from omnifold_publication.uncertainty import (
    histogram_matrix,                    # histogram many weight columns at once
    quadrature_difference_from_nominal,
    bootstrap_standard_deviation,
    ensemble_median_standard_error,      # 1.253 * std / sqrt(N)
    paired_relative_difference,
    combine_family,
    total_in_quadrature,
    fill_cov_matrix,                     # Hessian and bootstrap modes
    family_covariance,
    correlation_matrix,
    smooth_uncertainty,                  # Gaussian-kernel smoothing of a binned amplitude
)
```

### 6. Binning: effective statistics and validity

```python
import numpy as np
from omnifold_publication import n_eff, equal_effective_events_bins, validate_binning

values, mask = pkg.observable_values("pT_ll")
weights = np.asarray(pkg.get_weights("nominal"))[mask]

n_eff(weights)                                     # (sum w)^2 / sum(w^2)

edges = equal_effective_events_bins(values, weights, target_n_eff=20_000.0)

report = validate_binning(pkg, "pT_ll")            # official binning by default
report = validate_binning(pkg, "pT_ll", bins=edges, n_eff_min=5000.0, data_stat_max=0.15)

report["n_eff"]                # effective events per bin
report["data_stat_fraction"]   # data statistical uncertainty per bin
report["data_stat_source"]     # "bootstrap_data", or "sample_stat" when no such family
report["errors"]               # [] when the binning is valid
```

The two thresholds are the release's own usage recommendations: at least 5,000
effective events per bin, and data statistical uncertainty below 15%. The data
statistical term is taken from the `bootstrap_data` family when the package
declares one, and the fallback is reported rather than hidden.

### 7. Multi-sample analyses

Link packages with a manifest, then work through `OmniFoldAnalysis`:

```python
from omnifold_publication import write_manifest, load_analysis

write_manifest(
    output_dir="artifacts/analysis",
    nominal_path="../nominal",                      # relative to the manifest directory
    variations={
        "sherpa": {"path": "../sherpa", "type": "generator"},
        "nonDY":  {"path": "../nonDY",  "type": "background"},
    },
    target_path="../target",                        # optional truth/target sample
    analysis_name="zjets",
)

ana = load_analysis("artifacts/analysis")

ana.summary()
ana.list_variations()
ana.nominal_package, ana.target_package
ana.validate_all()
ana.get_replica_weights()                           # (n_replicas, n_events)
```

Histograms with bands, and breakdowns extended by two-point terms:

```python
h = ana.histogram("pT_ll", systematic_variations=["sherpa", "nonDY"])
h.hist, h.stat_uncertainty, h.sys_uncertainty, h.replica_uncertainty
h.total_uncertainty                                 # the three, in quadrature

brk = ana.uncertainty_breakdown("pT_ll")                     # adds two_point_<sample>
brk = ana.uncertainty_breakdown("pT_ll", include_two_point=False)

cov = ana.covariance_matrix("pT_ll", smooth_two_point=["sherpa"])
cmp = ana.compare("pT_ll")                                   # HistogramComparison
cmp.print_table()
cmp.plot("out/comparison.png")
cmp.export_json("out/comparison.json")
```

A **two-point** component is `h_variation - h_nominal` for an alternative sample,
entering the covariance as one fully bin-correlated matrix. `smooth_two_point`
Gaussian-kernel smooths the named variation's delta first, as the release does
for its hidden-variable term. Systematic variations are always combined in
quadrature, never as an envelope.

### 8. Closure tests

```python
from omnifold_publication import chi2_test

# in-package pair: the release's embedded data-driven closure
r = chi2_test(pkg, observable="pT_ll",
              variation="weights_dd", target_variation="target_dd")

# against a known target sample declared in a manifest
r = chi2_test(ana, observable="pT_ll", mode="full")

# every option
r = chi2_test(
    ana,
    target_package=None,           # else the manifest's role="target", else the package itself
    observable="pT_ll",
    bins=None,                     # None = the observable's official binning
    mode="full",                   # or "release-exact"
    variation="nominal",
    target_variation="nominal",
    decorrelate_threshold=0.01,    # p below this triggers the decorrelation fallback
    smooth_two_point=["sherpa"],
    decorrelate_components=None,   # default: dd_unfolding + the smoothed two-points
    exclude_components=None,       # default: the circularity guard's choice
)

r["chi2"], r["dof"], r["p_value"]
r["components"], r["excluded_components"], r["decorrelated"]
r["result"], r["target"], r["difference"], r["covariance"]
```

Three behaviours worth knowing:

- **`mode`** — `"full"` uses every packaged component (physically preferred:
  dropping real systematics deflates the covariance and inflates chi-squared).
  `"release-exact"` restricts to exactly the component list the release's closure
  cell sums.
- **Circularity guard** — by default the internal covariance excludes
  `dd_unfolding` and any two-point component built from the sample serving as the
  target. Those are constructed from the very discrepancy the test measures, so
  keeping them pins chi-squared near 1 no matter how wrong the target is. Pass
  `exclude_components=[]` to reproduce the literal release recipe. Full
  derivation in `docs/closure_test_design.md`.
- **Decorrelation fallback** — if the initial p-value falls below
  `decorrelate_threshold`, the unfolding terms are reduced to their diagonals and
  the test repeats; both results are reported (`initial_chi2`, `initial_p_value`).

Closure grid figure — one panel per observable, Start / Target / Result with
ratio panels and annotated chi-squared:

```python
from omnifold_publication import plot_closure_grid

fig, info = plot_closure_grid(
    ana,
    observables=None,          # default: every observable with an official binning
    ncols=4,
    bins_map={"tau21": [0.0, 0.2, 0.4, 0.6, 1.0]},   # for observables without one
    mode="full",
    output_path="out/closure_grid.png",
)
info["results"], info["skipped"]
```

### 9. Cross-publication comparison

Compare two *independent* unfoldings of the same measurement — e.g. MultiFold vs
OmniFold — on shared bins, re-histogrammed from raw events with no interpolation:

```python
from omnifold_publication import compare_publications

cmp = compare_publications(
    mf_pkg, of_pkg, "pT_ll",
    bins=None,                       # None = side A's official binning
    labels=("MultiFold", "OmniFold"),
    correlation="none",              # only "none" is implemented, deliberately
)

cmp.bins, cmp.ratio
cmp.to_dict()
cmp.export_json("out/comparison.json")
cmp.plot("out/comparison.png")                        # both results + ratio panel

cmp.uncertainty_comparison_to_dict()
cmp.export_uncertainty_comparison_json("out/unc.json")
cmp.plot_uncertainty_comparison("out/unc.png")        # totals + per-component composition
```

`correlation="none"` means each side's uncertainty band is shown separately and no
combined ratio band is implied. Any other value raises `NotImplementedError`
rather than inventing a correlation model between two publications.

### 10. HEPData export and import

```python
# export: submission.yaml + one data_<observable>.yaml per observable
out = pkg.export_hepdata("out/hepdata")
out = pkg.export_hepdata(
    "out/hepdata",
    observables=["pT_ll"],                      # default: all with an official binning
    bins_map={"tau21": [0.0, 0.5, 1.0]},
    error_breakdown=True,                       # one labelled error per component
    sqrt_s_gev=13000.0,
    luminosity="139 fb$^{-1}$",
)
```

Tables carry the differential cross-section (final weights, declared selection
applied, official binning) with per-bin symmetric errors — a single `total` by
default, matching the published record's style. Export values come from
`uncertainty_breakdown`, so there is exactly one computation path, and the
generated files are validated against HEPData's own schemas in the test suite.

```python
from omnifold_publication import load_package, load_hepdata_submission

binned = load_hepdata_submission("out/hepdata")     # local submission directory
binned = load_package("out/hepdata")                # auto-detected
binned = load_package("hepdata:ins2791852")         # remote record (needs network)

binned.list_observables()
binned.observable_bins("pT_ll")
binned.histogram("pT_ll").total_uncertainty
binned.cross_section("pT_ll")
binned.summary()
binned.validate()
```

A HEPData record holds histograms, not per-event weights, so `load_events`,
`get_weights`, weight families, `uncertainty_breakdown`, `covariance_matrix`,
`observable_values`, and `get_uncertainty` raise a clear `PackageReadError`
pointing you at the event-level release instead of returning something
misleading.

### 11. Data sources

One interface over local disk, uploads, and remote records:

```python
from omnifold_publication.sources import get_source

src = get_source("local:data/multifold.h5")     # or a bare existing path
src = get_source("zenodo:11507450")             # streams from a Zenodo record
src = get_source("upload:<stored path>")        # private, auto-expiring

df = src.open_events(columns=["pT_ll", "weights_nominal"])
src.metadata()
src.checksum()          # SHA-256, streamed in chunks
src.health_check()
```

`s3://bucket/key` and `url:https://...` are wired into the factory but raise
`NotImplementedError` with a clear message — declared stubs, not silent
failures. Details in `docs/data_sources.md`.

### 12. Derived observables

Observables that are computed rather than stored:

```python
from omnifold_publication import DERIVED_OBSERVABLES, compute_derived_observables

list(DERIVED_OBSERVABLES)                       # ['tau21', 'phi_ll', 'dR_ll']
df = compute_derived_observables(df, ["tau21", "dR_ll"])
```

`write_package(observables=[...])` computes any of these automatically when the
name is requested and absent from the source, carrying the registry's selection,
units, and description into the package metadata.

### 13. JSON export helpers

```python
from omnifold_publication import export_histogram_json, export_comparison_json

export_histogram_json("artifacts/my_result", "pT_ll", "out/hist.json",
                      bins=[200, 300, 600, 1000], variation="nominal")
export_comparison_json("artifacts/analysis", "pT_ll", "out/comparison.json")
```

---

## Using your own dataset

**Short answer: yes, your data has to end up in this schema — but you do not
hand-write it, and there are two ways in.** The distinction that matters:

- The **reader and the uncertainty stack are fully generic.** Nothing there
  infers meaning from column names; they dispatch on the *declared* family
  `combination`. Any column naming works.
- The **writer's auto-discovery is convention-based**, keyed to the ATLAS
  release's column names. That is the only place a foreign dataset meets
  friction.

### Path A — convert with `write_package` (recommended)

Use `column_rename` to align names and `metadata_source` to supply your own
observable descriptions and official binning:

```python
write_package(
    input_path="my_unfolding_output.parquet",
    output_dir="artifacts/mine",
    metadata_source="my_metadata.yaml",
    column_rename={
        "truth_pt_dilepton": "pT_ll",       # observable -> the name your metadata declares
        "w_unfolded":        "weights_nominal",
        "w_prior":           "weight_mc",
    },
    nominal_convention="reweighting_factor",   # if the nominal column is only the learned factor
    normalization_mode="shape",                # if the weights carry no absolute scale
    weight_units=None,
    event_count=None,
    observables=["pT_ll"],
    method="MyMethod",
    assumptions=["unfolded with 5 iterations; detector systematics not included"],
)
```

What the writer expects:

| Requirement | Escape hatch |
|---|---|
| Nominal weight column named `weights_nominal`, base MC weight `weight_mc` | `column_rename` |
| Each requested observable described in the `metadata_source` YAML (units, description, `binning.official`) | supply your own YAML |
| Systematic families discovered from prefixes `weights_muEff*`, `weights_pileup*`, `weights_theory*`, `weights_track*`, `weights_muCal*`, `weights_lumi*`, `weights_topBackground*` | anything unmatched lands in `syst_other` — still combined in quadrature, just coarser labels |
| Bootstrap/ensemble families from prefixes `weights_bootstrap_mc_`, `weights_bootstrap_data_`, `weights_ensemble_` | rename to match, or declare families by hand (Path B) |
| Paired data-driven term from exactly `weights_dd` + `target_dd` | Path B |
| Iteration weights matched as `weights_step1_iter_3` / `weights_iter_3_step1` | optional; absent in the release files |

### Path B — hand-author the metadata

For an arbitrary schema, write `metadata.yaml` yourself against `schema.py` and
put `events.parquet` next to it. Declare families with your own column names and
one of the four combinations, and everything downstream works unchanged:

```yaml
format_version: '0.2'
observables:
  - name: pT_ll
    description: Transverse momentum of the dilepton system
    units: GeV
    binning:
      official: [200.0, 230.0, 300.0, 450.0, 600.0, 1000.0]
      provenance: my analysis note, table 3
weights:
  nominal: w_unfolded
  base_mc_weight: w_prior
  nominal_convention: reweighting_factor
  families:
    my_detector_np:
      type: systematic
      combination: quadrature_difference_from_nominal
      columns: [w_jes_up, w_jer_up, w_muon_up]
    my_bootstrap:
      type: bootstrap
      combination: standard_deviation
      columns: [w_bs_000, w_bs_001, w_bs_002]
    my_unfolding_pair:
      type: paired
      combination: paired_relative_difference
      columns: [w_alt]
      reference_column: w_ref
normalization:
  mode: absolute
  weight_units: fb
  nominal_weight_column: w_unfolded
  base_weight_column: w_prior
publication:
  format: parquet
  events_file: events.parquet
  event_count: 123456
```

Validate the metadata before you rely on it:

```python
import yaml
from omnifold_publication.schema import validate_metadata

validate_metadata(yaml.safe_load(open("metadata.yaml")))   # raises with field-level errors
```

### Path C — come in already binned

If all you have is a published HEPData record, `load_package("hepdata:<id>")`
gives you the binned API (histograms, bins, cross-sections) with event-level
operations cleanly refused.

---

## The metadata contract

`spec/metadata.yaml` is the worked reference instance; `omnifold_publication/schema.py`
is the pydantic model. Top-level blocks:

| Block | Contents |
|---|---|
| `format_version` | `"0.1"` or `"0.2"` (both readable; the writer emits `0.2`) |
| `dataset` | name, experiment, description, `method`, `assumptions`, `provenance_note` |
| `generation` | nominal generator, alternative generators and samples |
| `files` | nominal/systematic source files and event counts |
| `observables` | name, description, units, `binning.official` + provenance, `suggested_bins`, `selection`, `derived_from` |
| `weights` | `nominal`, `base_mc_weight`, `nominal_convention`, `families`, `iterations`, replica prefixes, notes |
| `systematics` | declared families with type and combination |
| `normalization` | mode, weight units, expected `sum(w)`, tolerance, column names |
| `event_selection` | phase-space description (lepton pT/eta, m_ll window, jet requirements) |
| `training` | algorithm, iterations, architecture |
| `publication` | format, events file, event count, columns, event alignment, SHA-256 |
| `usage_notes` | analysis guidance carried with the data |

Two fields deserve emphasis:

- **`weights.nominal_convention`** — `includes_mc_weight` means the nominal
  column is already the complete measurement weight (the ATLAS release layout),
  so `get_weights("final")` returns it unchanged. `reweighting_factor` means it
  holds only the learned OmniFold factor and must be multiplied by the base MC
  weight. Getting this wrong silently rescales every result, which is why it is
  declared rather than guessed. See `spec/weight_formula.md`.
- **`publication.event_alignment`** — `row_order` (rows line up positionally
  across files) or `column` with an `event_id`. Validation enforces whichever is
  declared. See `spec/alignment_contract.md`.

---

## Uncertainty recipes

Each weight family declares one combination; `uncertainty.py` implements them and
`spec/uncertainty_recipes.md` records where each formula comes from.

| `combination` | Used for | Band | Covariance mode |
|---|---|---|---|
| `quadrature_difference_from_nominal` | systematic NP families | quadrature sum of per-NP differences from nominal | Hessian on the differences |
| `standard_deviation` | bootstrap replicas | per-bin std across replica histograms | sample covariance, `1/(K-1)` |
| `median_standard_error` | NN-ensemble replicas | `1.253 * std / sqrt(N)` | sample covariance `/ N` |
| `paired_relative_difference` | data-driven pair | `abs((h_var - h_ref) * h_nom / h_ref)` | Hessian on the single signed delta |

Plus `sample_stat` — always present, `sqrt(sum w^2)` from the nominal weights —
and, for an analysis, `two_point_<sample>` per alternative sample. All components
combine in quadrature for the total band, and additively for the total
covariance.

---

## Run the tests

```bash
pip install -e ".[dev]"
pytest -q
```

Expected on a bare clone, with no data present:

```
201 passed, 11 skipped
```

The 11 skips are 9 real-data-gated tests and 2 needing `fsspec`; each prints its
reason with `-rs`. With the data files under `data/` and the `sources` extra
installed:

```
212 passed
```

Useful invocations:

```bash
pytest -q -rs                        # show skip reasons
pytest tests/test_closure.py -v      # one module
pytest -k covariance                 # by keyword
pytest -q tests/test_hepdata.py      # HEPData files checked against HEPData's own schemas
```

Linting, if you installed the dev extra:

```bash
ruff check .
```

---

## Repository layout

```
omnifold_publication/        the package
├── reader.py                OmniFoldPackage, load_package
├── writer.py                write_package + weight-family discovery
├── schema.py                pydantic metadata model
├── validation.py            validate_package, checksum/alignment/normalisation checks
├── histogram.py             compute_weighted_histogram, HistogramResult
├── uncertainty.py           recipes, covariance, correlation, smoothing
├── binning.py               n_eff, equal-n_eff bins, per-bin validity
├── closure.py               chi2_test
├── analysis.py              OmniFoldAnalysis (multi-sample, two-point terms)
├── manifest.py              write_manifest / load_manifest
├── comparison.py            HistogramComparison (within one analysis)
├── cross_comparison.py      CrossPublicationComparison (two publications)
├── plotting.py              plot_closure_grid
├── hepdata.py               export_hepdata, HEPDataPackage, remote record loading
├── derived_observables.py   tau21, phi_ll, dR_ll
├── selection.py             per-observable selection parser (strict grammar, never eval)
├── publication.py           Publication builder
├── export.py                JSON export helpers
├── cli.py                   command-line interface
└── sources/                 local, upload, zenodo, url, s3 data sources

spec/                        the contract: metadata.yaml, weight_formula.md,
                             uncertainty_recipes.md, alignment_contract.md
docs/                        design notes (schema, closure, HEPData, sources, provenance)
examples/                    runnable end-to-end scripts
scripts/publish_to_zenodo.py package -> Zenodo deposition -> DOI
tests/                       the test suite
data/                        put your event files here (gitignored)
artifacts/                   generated output (gitignored)
```

---

## Publishing your result

**HEPData** — the binned tables that accompany a paper:

```python
pkg.export_hepdata("submission", error_breakdown=True)
```

Upload the resulting directory to hepdata.net. See `docs/hepdata_integration.md`.

**Zenodo** — the event-level dataset, with a DOI:

```bash
# safe rehearsal: converts and writes every artifact, makes no network call
python scripts/publish_to_zenodo.py data/multifold.h5 --dry-run

# rehearse against the sandbox, then drop --sandbox for the real thing
export ZENODO_TOKEN=<your token>
python scripts/publish_to_zenodo.py data/multifold.h5 data/multifold_sherpa.h5 \
    --sandbox --creator "Lastname, Firstname" \
    --title "My OmniFold publication packages" \
    --output-dir zenodo_submission
```

The script builds full packages (all replicas), writes `README.md`,
`LICENSE.txt`, `CITATION.txt`, and `zenodo_metadata.json`, zips each package, and
runs the Zenodo deposition flow, printing the DOI.

Two cautions: **publishing on Zenodo is irreversible** — a published record
cannot be deleted and its DOI is permanent, so always rehearse with `--sandbox`
first. And the defaults produce large deposits: a full-replica package of the
nominal release file is several hundred MB.

---

## Gotchas

- **Official binning is required, never guessed.** Closure tests, HEPData export,
  and cross-publication comparison refuse to fall back to `suggested_bins` — pass
  `bins=` / `bins_map=` explicitly for observables without an official binning
  (`tau21` and `dR_ll` in the reference metadata).
- **`pT_ll` starts at 200 GeV** in this measurement (boosted-Z phase space).
  A histogram range of `(0, 200)` catches zero events and looks like a
  catastrophic failure. Use `pkg.observable_bins("pT_ll")`.
- **Track-jet observables carry a selection** (`pT_trackj1 > 5`). Use
  `observable_values()`, which applies it, rather than reading the raw column —
  events with no real jet hold soft/degenerate values that would otherwise enter
  the histogram silently.
- **Subsetting breaks absolute normalisation.** `sum(weights_nominal)` is the
  fiducial cross-section only over the whole sample. For cross-section or ratio
  work use `event_count=None`.
- **`include_all_replicas` defaults to `False`.** Statistical uncertainties from
  bootstrap/ensemble families need `True`.
- **Replica "envelopes" are spreads, not min/max.** The release combines
  variations in quadrature and replicas by standard deviation; nothing here uses
  an envelope.
- **The ensemble covariance omits the 1.253 factor** that the band applies. That
  inconsistency exists in the release and is ported faithfully and deliberately —
  documented in `uncertainty.py`.
- **HEPData packages are binned only.** Event-level calls raise
  `PackageReadError` by design.

---

## Further documentation

| File | Contents |
|---|---|
| `docs/schema_design.md` | why the metadata schema looks like this; what was included and excluded |
| `docs/closure_test_design.md` | the circularity argument behind the closure guard |
| `docs/hepdata_integration.md` | export/import details, remote-record access |
| `docs/data_sources.md` | the `DataSource` layer and its backends |
| `docs/data_provenance.md` | how the local files relate to the published release, with verification numbers |
| `docs/gap_analysis.md` | what the raw weight files contain and what metadata was missing |
| `spec/uncertainty_recipes.md` | every uncertainty formula and its source |
| `spec/weight_formula.md` | the nominal-weight convention |
| `spec/alignment_contract.md` | the event-alignment contract |

---

## Citing

If you use the packaged ATLAS Z+jets OmniFold data, cite the measurement
([arXiv:2405.20041](https://arxiv.org/abs/2405.20041), HEPData record
`ins2791852`) alongside this software.

## License

MIT, as declared in `pyproject.toml`.

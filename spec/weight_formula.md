# Canonical Weight Formula

## Final Weight

The canonical per-event weight for central-value observables depends on the
declared **nominal-weight convention** of the package:

| `weights.nominal_convention` | Final weight | When it applies |
|---|---|---|
| `includes_mc_weight` (default) | `w_final = w_nominal` | ATLAS-style releases where `weights_nominal` already contains the MC weight |
| `reweighting_factor` | `w_final = w_mc * w_nominal` | Files whose nominal column stores only the learned OmniFold factor |

The convention is recorded in package metadata under
`weights.nominal_convention`. Packages written before this field existed are
read as `includes_mc_weight`, matching the ATLAS release layout.

## What Each Term Means

| Term | Column | Description |
|---|---|---|
| `w_mc` | `weight_mc` | MC generator/base event weight. In the ATLAS release this is the **MC prediction** weight: `sum(weight_mc)` is the MC-predicted cross-section. |
| `w_nominal` | `weights_nominal` | Nominal measurement weight. In the ATLAS release this is the **complete** measured weight (MC weight included): `sum(weights_nominal)` is the measured fiducial cross-section in fb. |
| `w_final` | derived | Canonical downstream event weight for central-value observables. |

## Why `includes_mc_weight` Is the Default

The ATLAS Z+jets OmniFold release states and demonstrates that
`weights_nominal` already includes `weight_mc`:

- `multifold_util.py` (ATLAS release) says verbatim:
  *"note that weight_mc is already included in weights_nominal"*, and every
  histogram in the release code fills with `weights=df.weights_nominal`
  alone — the product never appears.
- The release notebooks use `sum(weights_nominal)` as the measured
  cross-section and `sum(weight_mc)` as the MC prediction: the two columns
  are parallel alternatives, not factors.

This was verified empirically against the local data files (2026-07-18):

| File | `sum(w_nominal)` | `sum(w_mc)` | `sum(w_mc * w_nominal)` |
|---|---|---|---|
| `multifold.h5` (418,014 events) | 1809.46 fb | 1701.13 fb | 11.29 |
| `multifold_sherpa.h5` (326,430 events) | 1816.17 fb | 1601.73 fb | 13.22 |
| `multifold_nonDY.h5` (433,397 events) | 1810.59 fb | 1780.72 fb | 11.54 |

Both weight columns sit on the same absolute scale (per-event mean
~0.004 fb; a standalone factor would average ~1 and sum to ~N). The ratio
`w_nominal / w_mc` is the quantity that behaves like the learned factor
(median 0.87–1.06, >99.7% of events within [0.2, 5]). The product
`w_mc * w_nominal` is dimensionally weight-squared and its sum (~11 fb·fb)
is physically meaningless. Multiplying would therefore double-count the MC
weight for these files.

The `reweighting_factor` convention is retained for OmniFold outputs that
follow the original paper's storage convention (a pure learned factor
`nu(x)` with `w_final = w_mc * nu`), and must be declared explicitly when
writing such a package.

## Nominal vs Iteration Weights

- **Nominal**: the final truth-level OmniFold weight after training is
  complete. This is the required publication object for central values.
- **Iteration weights**: *optional and unused for the actual analysis
  data.* The ATLAS release files store no intermediate iteration or step
  columns, so this machinery exists only for source files that do (columns
  such as `weights_iter{N}_step{1,2}`). If present, they must declare both
  the iteration number and the step.

## Downstream Usage

The reader exposes the derived final weight directly and resolves the
convention from metadata:

```python
from omnifold_publication import load_package

pkg = load_package("artifacts/zjets/")
w_final = pkg.get_weights("final")   # convention-aware
pkg.nominal_convention()             # "includes_mc_weight" | "reweighting_factor"
```

Variation weights are organized into declared families with per-family
combination recipes; see `spec/uncertainty_recipes.md`.

For manual work with ATLAS-style files, use `weights_nominal` as-is:

```python
import numpy as np

df = pkg.load_events(columns=["pT_ll", "weights_nominal"])
hist, edges = np.histogram(df["pT_ll"], bins=30, weights=df["weights_nominal"])
```

# Data Provenance

Status of the local `data/*.h5` files relative to the public ATLAS Z+jets
OmniFold release (`sm-z-jets-omnifold-2024`), established 2026-07-19 by
comparing against the saved cell outputs of `1_basics.ipynb`.

## Finding

The local files are a **reduced, regenerated release version** — not the
exact files the notebook's saved outputs were produced from. Any
implementation, no matter how correct, therefore cannot reproduce the
notebook's saved numbers exactly; close agreement plus exact formula
equality on the local files is the strongest achievable verification.

| Quantity | Local `data/multifold.h5` | Notebook saved run |
|---|---|---|
| Events (nominal) | 418,014 | 418,014 (match) |
| Columns | 200 | 276 (per `1_basics.ipynb` cell 7) |
| Stored replica weights | 150 (100 ensemble + 25 + 25 bootstrap) | ~226 |
| nEff(weights_nominal) | 213,024 | 197,876 (printed) |
| sum(weights_nominal) | 1809.46 fb | 1808.34 fb (printed) |

The event counts match but nEff differs by ~7%, so the weight columns were
regenerated (retrained unfolding), not just truncated.

## Verification against the saved outputs

Two-layer check of `omnifold_publication`'s uncertainty machinery
(recipes in `spec/uncertainty_recipes.md`):

1. **Formula equality (same file).** Each component of
   `uncertainty_breakdown()` compared per bin against the release formulas
   (`calculate_uncertainty`, `calculate_stat_uncertainty`, the paired-dd
   block) re-coded independently in numpy on the local file: **maximum
   absolute difference 0.0** for every component and the nominal histogram.
2. **Agreement with the saved notebook outputs** (`printXsecUnc`,
   cells 15-16): all 22 NP variations within **0.02 percentage points** in
   the full fiducial region (e.g. lumi 1.70% vs 1.70%); within 0.26
   percentage points in the pT(ll) > 500 GeV tail; cross-sections
   1809.46 vs 1808.34 fb (full) and 41.48 vs 41.18 fb (tail). The residual
   differences are attributed to the file-version difference above, which
   affects tails more than inclusive quantities.

## Where this is recorded

Every package written by `write_package` carries this caveat in its
metadata under `dataset.provenance_note` (sourced from
`spec/metadata.yaml`; field modeled in `schema.py` `Dataset`).

# Uncertainty Recipes

How packaged weight families combine into per-bin uncertainties. Every
recipe reproduces the corresponding rule in the ATLAS Z+jets OmniFold
release (`multifold_util.py` and the release notebooks); nothing here is an
invented convention.

## Weight Families

The writer discovers families from the source columns and declares them in
package metadata under `weights.families`:

```yaml
weights:
  families:
    syst_event:
      type: systematic
      combination: quadrature_difference_from_nominal
      columns: [weights_muEffReco, weights_muEffIso, ..., weights_pileup]
    dd_unfolding:
      type: paired
      combination: paired_relative_difference
      columns: [weights_dd]
      reference_column: target_dd
    ensemble:
      type: ensemble
      combination: median_standard_error
      columns: [weights_ensemble_0, ...]
```

Grouping of the NP systematics follows `multifold_util.py:335-341`:
`syst_event` (muEff\* + pileup), `syst_theory` (theory\*), `syst_track`
(track\*), `syst_muon` (muCal\*); anything unmatched (lumi, topBackground,
non-ATLAS names) goes to `syst_other`. Since all systematic families combine
in quadrature, grouping affects only breakdown labels, never the total.

Bootstrap and ensemble families are packaged when
`write_package(include_all_replicas=True)`; NP and paired families are
packaged by default (`include_systematics=True`).

## Combination Recipes

| Recipe | Formula (per bin) | ATLAS evidence |
|---|---|---|
| `quadrature_difference_from_nominal` | sqrt(Σ_k (h_k − h_nom)²) | `calculate_uncertainty`, multifold_util.py:61-68 |
| `standard_deviation` | std over replica histograms | `calculate_stat_uncertainty`, multifold_util.py:71-82 |
| `median_standard_error` | 1.253 · std / √N | multifold_util.py:78-82 (1.253, std-err of the median) plus the /√N at the call site, 2_pseudo_results.ipynb cell 11 |
| `paired_relative_difference` | \|(h_var − h_ref) · h_nom / h_ref\| | "Unfolding (DD)" block, multifold_util.py (weights_dd vs target_dd) |

Two additional components are computed outside the family system:

- **`sample_stat`** — the statistical uncertainty of the released
  evaluation sample: sqrt(Σ w²) per bin (the `hist(weights_nominal**2)`
  covariance diagonal in 3_results.ipynb; `Vstat_meas` in 1_basics.ipynb
  cell 14).
- **`two_point_<variation>`** — alternative-sample systematics at the
  analysis level: |h_alt − h_nom| where h_alt is the variation package's
  nominal histogram (the "Unfolding (HV)" / "Non-Strong Background" blocks
  in multifold_util.py, using the Sherpa and non-DY samples). The release
  optionally smooths the HV term with a Gaussian kernel; smoothing is a
  plotting refinement and is not implemented here.

## Total

All components are treated as independent and combined in quadrature,
matching `printXsecUnc` (1_basics.ipynb cell 14), which accumulates
fractional variances from every source and reports sqrt of the sum.

## Covariance and Correlation

`fill_cov_matrix` (port of multifold_util.py:49-58) provides the two modes
the release uses to build bin-to-bin covariance matrices, assembled per
family exactly as in `corr_matrix` (multifold_util.py:307, assembly at
402-427):

| Family type | Mode | Formula | Release term |
|---|---|---|---|
| systematic | Hessian | V = Σ_k outer(h_k − h_nom) — each NP fully correlated across bins, NPs uncorrelated | v_theory, v_lumi |
| bootstrap | Bootstrap | sample covariance, 1/(K−1) | v_bs_mc, v_bs_data |
| ensemble | Bootstrap / N | sample covariance divided by replica count | v_nn (`/100`) |
| paired | Hessian | outer of the **signed** delta (h_dd − h_ref)·h_nom/h_ref | v_unfolding_dd |
| sample_stat | diagonal | diag of hist(w_nominal²) | v_mc |
| two-point (analysis level) | Hessian | outer of signed (h_alt − h_nom) | v_unfolding_hv, v_bkg |

Total = sum of component matrices; correlation_ij = V_ij/(σ_i·σ_j) with
σ = sqrt(diag(V_total)) (zero-variance bins get zero correlation, unit
diagonal preserved).

Faithfully ported release inconsistencies (documented, not "fixed"):

- The band's bootstrap std uses ddof=0 while the covariance uses 1/(K−1),
  so diag(V_bootstrap) = K/(K−1) · band².
- The ensemble band carries the 1.253 median factor; the ensemble
  covariance (v_nn) does not, so sqrt(diag(V_ensemble)) is the band
  without 1.253 (and with the ddof difference).
- The release's `corr_matrix` omits the detector NP groups
  (event/track/muon) from its total even though its bands include them;
  our builder includes every packaged family — sum a subset of
  `components` to reproduce the release selection exactly.

```python
cov = pkg.covariance_matrix("pT_ll")        # components, total, correlation
cov = analysis.covariance_matrix("pT_ll")   # adds two_point_* matrices
```

## Closure Testing

`chi2_test` ports the pseudo-data closure recipe of
2_pseudo_results.ipynb cell 26: `D = h_result − h_target`,
`chi2 = D·V⁻¹·D` with **dof = n_bins**, and the p < 0.01 fallback that
decorrelates the DD and smoothed-HV terms (only those — the non-DY
two-point stays correlated, as in the release).

- **Target resolution**: explicit `target_package` → the manifest's
  role-`target` sample (`write_manifest(target_path=...)`,
  `analysis.target_package`) → the result package itself (in-package
  pairs such as `variation="weights_dd"` vs `target_variation="target_dd"`).
- **Covariance modes**: `mode="full"` (default; every packaged component —
  physically preferred) or `mode="release-exact"` (exactly cell 26's sum,
  which *excludes* v_lumi and the detector NP groups; the list was
  re-derived from the cell source, not from `corr_matrix`, which differs
  by including v_lumi).
- **Circularity guard** (on by default): `dd_unfolding` and any two-point
  component built from the sample serving as target are excluded from the
  *internal* covariance — they are constructed from the discrepancy being
  tested and would bound chi2 near 1 regardless of how wrong the target
  is. Full derivation and empirical before/after numbers:
  docs/closure_test_design.md. `exclude_components=[]` restores the
  literal cell 26 recipe.
- **Binning strictness**: observables without an official binning (e.g.
  the derived tau21/dR_ll) raise instead of silently using
  suggested_bins; pass `bins=` explicitly.
- No printed chi2/p reference could be reproduced: cell 26's saved table
  derives from the pseudodata files, which are not in the local dataset.
  Verification is an exact match against an independently coded literal
  implementation of the recipe (tests/test_closure.py).

`plot_closure_grid` renders the cell 14 layout (Start / Target / Result
with total-uncertainty band, plus a Ratio-to-Target panel per
observable); every number it draws comes from `chi2_test`, so plots and
statistics cannot diverge. End-to-end workflow:
`examples/pseudodata_closure.py`.

```python
result = chi2_test(analysis, observable="pT_ll")          # target from manifest
fig, grid = plot_closure_grid(analysis, observables=[...])
```

## API

```python
pkg = load_package("artifacts/zjets/")
pkg.list_weight_families()                  # ["syst_event", ...]
pkg.get_family_weights("ensemble")          # (n_replicas, n_events)
result = pkg.uncertainty_breakdown("pT_ll") # per-family + sample_stat + total

analysis = load_analysis("artifacts/analysis/")
analysis.uncertainty_breakdown("pT_ll")     # adds two_point_* components
```

`uncertainty_breakdown` returns `{"edges", "nominal", "components", "total"}`
with one entry per family in `components`, all in the (absolute) units of
the weights — femtobarns for the ATLAS release. Percent uncertainties, as
printed by the release code, are `100 * component / nominal`.

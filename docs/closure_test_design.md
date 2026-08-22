# Closure-Test Design: the Circular-Covariance Problem

Why `chi2_test` excludes the `dd_unfolding` component (and any
target-coincident two-point component) from its **internal** covariance by
default, while `uncertainty_breakdown()` / `covariance_matrix()` keep them
for published measurements.

## The problem

The dd_unfolding covariance component is built from the discrepancy
between the data-driven unfolding result and its known target
(2_pseudo_results.ipynb cell 26; multifold_util.py "Unfolding (DD)"):

```
Delta_i = (h_dd,i - h_target_dd,i) * h_nominal,i / h_target_dd,i
V_dd    = Delta Delta^T          (Hessian mode, fully bin-correlated)
```

A closure test of the same pair measures `D = h_dd - h_target_dd`. Since
`h_nominal ≈ h_target_dd`, `Delta ≈ D`, so the test's covariance contains
(approximately) `D D^T` — it grows in exact proportion to the discrepancy
being tested.

## Sherman–Morrison bound

With `V = V_other + u u^T` and `u = alpha * D` (alpha ≈ 1, more precisely
`alpha ≈ 1/s` elementwise for a target scaled by `s`):

```
chi2 = D^T V^-1 D
     = D^T V_other^-1 D  -  alpha^2 (D^T V_other^-1 D)^2 / (1 + alpha^2 D^T V_other^-1 D)
     = q / (1 + alpha^2 q)        with q = D^T V_other^-1 D
```

As the discrepancy grows (q → ∞), chi2 → 1/alpha² ≈ s². The statistic is
**bounded** no matter how wrong the target is; with dof = 5, chi2 ≤ ~1.4
means p ≥ ~0.92. The test structurally cannot fail.

## Empirical confirmation (full 418k-event dataset, official pT_ll bins)

Injected wrong targets (`target_dd` scaled), result = `weights_dd`:

| Injected target | Before fix (dd included) | After fix (dd excluded) |
|---|---|---|
| A: flat +5% | chi2 = 0.879, p = 0.972 | chi2 = 4.67, p = 0.458 |
| B: flat +20% | chi2 = 1.396, p = 0.925 | chi2 = 78.4, p = 1.8e-15 |
| C: slope +10% → −10% | chi2 = 1.162, p = 0.949 | chi2 = 22.3, p = 4.5e-4 |

Before the fix every case "passes" with near-constant p; case B sits just
under its Sherman–Morrison bound s² = 1.44. After the fix, the 20% shift
and the slope are decisively rejected, while the 5% shift remains within
the genuine (correlated-systematics dominated) uncertainty — a defensible
outcome, not a blind pass. Regression test:
`tests/test_closure.py::test_circularity_guard_rejects_injected_wrong_target`.

## Which components are circular

- **`dd_unfolding`** — circular whenever the tested pair involves the dd
  columns or any target derived from them. Since `chi2_test` cannot know
  the provenance of an arbitrary target histogram, the component is
  excluded from the internal covariance **categorically** (safe default;
  it under-covers slightly in the cell 26 setting where the dd pair is
  independent of the pseudodata target — pass `exclude_components=[]` to
  reproduce that literal recipe, as the literal-equivalence tests do).
- **`two_point_<name>`** — circular iff the variation sample *is* the
  resolved target package: the two-point delta is then exactly −D.
  Detected by package-path identity and excluded automatically.
- **`syst_background`** (topBackground) — **not** circular: its delta
  `h_top − h_nominal` never involves the target. Verified against the
  cell 26 source; it stays in the covariance.
- All other families (detector/theory NPs, bootstraps, ensemble,
  sample_stat) are built from the nominal result's own variations and are
  independent of any target.

## Scope of the exclusion

The exclusion applies **only** inside `chi2_test`'s covariance
construction. For published measurements, the data-driven unfolding
uncertainty is a legitimate systematic: `uncertainty_breakdown()` and
`covariance_matrix()` include `dd_unfolding` unchanged. The distinction is
purpose: in a measurement, `Delta_dd` estimates an uncertainty from an
auxiliary closure exercise; in a closure test, letting the measured
discrepancy enlarge its own error band is circular validation.

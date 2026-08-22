# HEPData Integration

Binned export and import connecting `omnifold_publication` to
https://www.hepdata.net. Reference record: this measurement's own entry,
[ins2791852](https://www.hepdata.net/record/ins2791852) (26 binned
differential cross-section tables; Table 1 is dimuon pT in exactly the
official `ibu_bins` binning).

## Export

```python
pkg.export_hepdata(output_dir)                       # submission.yaml + data_<obs>.yaml
pkg.export_hepdata(output_dir, error_breakdown=True) # labelled per-component errors
```

One table per observable: the differential cross-section (final weights,
declared selection applied, official binning — suggested_bins are never
used silently; `bins_map=` provides explicit edges for derived
observables). Per-bin errors are a single `total` symmetric error by
default, matching the published record's style. Every generated file is
validated in the test suite against HEPData's own schemas
(`hepdata-validator`), and export values are pinned to
`uncertainty_breakdown` — one computation path.

## Import

```python
binned = load_package("hepdata:ins2791852")   # remote record
binned = load_package("path/to/submission")   # local submission directory
```

Both return a `HEPDataPackage`: the **binned** counterpart of
`OmniFoldPackage`. A HEPData record holds histograms, not per-event
weights, so event-level operations (`load_events`, `get_weights`, weight
families, `uncertainty_breakdown`, `covariance_matrix`,
`observable_values`, `get_uncertainty`) raise a clear `PackageReadError`
pointing to the event-level Zenodo release. The binned subset matches the
package API: `list_observables`, `observable_bins`, `observable_units`,
`histogram` (with `total_uncertainty`), `cross_section`, `summary`, and a
binned-appropriate `validate()` (strictly increasing edges, array-length
consistency, finite values, non-negative finite errors).

## Documented constraint: JSON API is the primary remote path

The bulk `/download/...` endpoints on hepdata.net sit behind Cloudflare
bot protection that challenges non-browser clients (curl, urllib,
requests) — this is a general property of programmatic access, not a
quirk of one environment. The loader therefore uses the JSON API as its
primary path, by design:

- `record/<id>?format=json` — record summary and table listing;
- `record/data/<recid>/<table_id>/<version>/` — per-table data in
  HEPData's internal x/y format, converted here into the same normalized
  tables the submission-YAML parser produces.

Tables that are not one-dimensional binned distributions are skipped.

## Verification

- Offline: official-schema validation of exports; exact round-trip
  (export → load → histograms/uncertainties/cross-sections equal);
  internal-format converter tested on a Table 1-shaped stub.
- Live (network-gated, `test_live_record_round_trip`): the real record
  loads, validates, Table 1 reproduces the frozen published values
  (23.4681 ± 0.5332 fb/GeV in [200, 230] GeV), and our own export from
  the local files agrees per bin within 5% — consistent with the
  documented data-version difference (docs/data_provenance.md); measured
  per-bin deviations are 0.05-3.1%.

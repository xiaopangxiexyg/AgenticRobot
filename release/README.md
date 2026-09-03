# Release handoffs

`release/` contains reviewed handoff definitions and small, independently
maintained release sources. Generated source projections are not versioned in
this repository. Dataset examples remain read-only.

`annotation_projection.toml` is the canonical refinement-annotation projection
manifest. It lives here so packaging metadata does not become part of the main
runtime layout. `scripts/export_annotation_projection.py` creates a disposable,
self-verifying bundle from the current canonical sources. Exported bundles
carry the same manifest at `release/annotation_projection.toml` and can
validate themselves without access to the original checkout.

Current handoff source:

- [`phase_dataset_loader`](./phase_dataset_loader/README.md): standalone loader
  for the documented 2026-08-04 writing dataset layout.

Refinement annotation handoffs are generated on demand under the ignored
`dist/` root or another disposable output directory:

```bash
python scripts/export_annotation_projection.py \
  --output dist/refinement_annotation_$(date +%Y%m%d)
python scripts/export_annotation_projection.py \
  --verify dist/refinement_annotation_$(date +%Y%m%d)
```

The output path must be new. The exporter refuses to overwrite an existing
directory.

Retired snapshots:

- The last tracked 2026-08-08 projection was removed after the canonical
  exporter was shown to recreate and verify a current standalone handoff. Its
  exact tree and hashes remain in the
  [dated-bundle retirement history](../docs/legacy/reports/refinement_annotation_20260807_bundle_retirement_20260829.md#2026-08-31-后续最后版本化投影退役).

- The 2026-08-07 rollback projection was removed from the current source tree
  after its 2026-08-08 successor was verified. Its exact tree, hashes, and Git
  recovery path remain recorded in the
  [retirement report](../docs/legacy/reports/refinement_annotation_20260807_bundle_retirement_20260829.md).

The former undated `refinement_annotation/` snapshot was retired because its
tracked contents no longer matched its receipt. See the
[retirement record](../docs/legacy/reports/refinement_annotation_undated_bundle_retirement_20260827.md).
Do not add generated annotation projections back to the versioned source tree.

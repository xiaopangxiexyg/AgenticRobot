# Release bundles

`release/` contains portable handoff bundles generated from canonical DexTeleop
sources. Dataset examples remain read-only. The refinement-annotation bundle is
a source projection of the existing loopback GUI application; it must not
import hardware, teleoperation, collection frontends, deployment, simulation,
or training runtimes.

`annotation_projection.toml` is the canonical refinement-annotation projection
manifest. It lives beside the release bundles so packaging metadata does not
become part of the main runtime layout. Exported bundles carry the same
manifest at `release/annotation_projection.toml` and can validate themselves
without access to the original checkout.

Current bundle:

- [`phase_dataset_loader`](./phase_dataset_loader/README.md): standalone loader
  for the documented 2026-08-04 writing dataset layout.
- [`refinement_annotation_20260808_c959d1d38695`](./refinement_annotation_20260808_c959d1d38695/ANNOTATION_RELEASE.md):
  current generated post-capture annotation source projection. Its directory
  suffix is the first 12 characters of the source projection hash; verify the
  complete source and release hashes in `annotation_projection_receipt.json`.

Retired snapshots:

- The 2026-08-07 rollback projection was removed from the current source tree
  after its 2026-08-08 successor was verified. Its exact tree, hashes, and Git
  recovery path remain recorded in the
  [retirement report](../docs/legacy/reports/refinement_annotation_20260807_bundle_retirement_20260829.md).

The former undated `refinement_annotation/` snapshot was retired because its
tracked contents no longer matched its receipt. See the
[retirement record](../docs/legacy/reports/refinement_annotation_undated_bundle_retirement_20260827.md).
Generate disposable handoff candidates under the ignored `dist/` root, and use
a dated, source-hash-qualified name for any newly reviewed tracked snapshot.

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

Retained snapshots:

- [`refinement_annotation_20260807_39895a67ef52`](./refinement_annotation_20260807_39895a67ef52/ANNOTATION_RELEASE.md):
  prior generated source projection retained for rollback and receipt
  verification.
- [`refinement_annotation`](./refinement_annotation/ANNOTATION_RELEASE.md):
  immutable 2026-08-07 15:57 snapshot retained for rollback and receipt
  verification. It is not the current bundle and must not be overwritten.

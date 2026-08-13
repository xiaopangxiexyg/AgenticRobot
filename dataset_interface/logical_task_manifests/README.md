# Logical task datasets

These manifests group immutable capture datasets under one canonical task
identity without rewriting source metadata. `source_task_id` and
`source_dataset_fingerprint` are capture provenance; `canonical_task_id` is the
shared semantic identity used to organize later training snapshots.

Create a new versioned manifest when sources are added. Do not edit an old
manifest after a training profile or report has bound its
`semantic_content_hash`.

Audit a manifest and every pinned source read-only with:

```bash
python -m dataset_interface.logical_task_dataset audit \
  --manifest dataset_interface/logical_task_manifests/writing_stage1_scenarios_through_2026_08_11_v1.json
```

The audit intentionally has no source-normalization or apply mode. Physical
task metadata migration changes the source fingerprint and therefore requires
a separate offline migration of every dependent curation, modality, Prepared
view, and training binding.

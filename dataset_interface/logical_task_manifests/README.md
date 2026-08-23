# Logical task datasets

These manifests group immutable capture datasets under one canonical task
identity without rewriting source metadata. `source_task_id` and
`source_dataset_fingerprint` are capture provenance; `canonical_task_id` is the
shared semantic identity used to organize later training snapshots.

An immutable historical source whose `meta/tasks.jsonl` contains more than one
task identity may additionally pin `source_task_id_aliases`. The primary
`source_task_id` must still match `meta/info.json`, and the audit requires the
set of task rows to match the primary ID plus the declared aliases exactly.
This is a fail-closed compatibility binding for existing captures, not a
license for new collection or physical-merge outputs to emit mixed task IDs;
those outputs must continue to use one canonical task ID.

Create a new versioned manifest when sources are added. Do not edit an old
manifest after a training profile or report has bound its
`semantic_content_hash`.

The all-valid Stage1 snapshot through 2026-08-21 is pinned by
`writing_stage1_all_valid_through_2026_08_21_v1.json`. It replaces the two raw
2026-08-20 detail shards with their verified 400-Episode physical merge and
adds the single 2026-08-21 Stage1 source. The latter preserves its historical
secondary task row as an explicit alias; no source metadata is rewritten.

Audit a manifest and every pinned source read-only with:

```bash
python -m dataset_interface.logical_task_dataset audit \
  --manifest dataset_interface/logical_task_manifests/writing_stage1_scenarios_through_2026_08_11_v1.json
```

The audit intentionally has no source-normalization or apply mode. Physical
task metadata migration changes the source fingerprint and therefore requires
a separate offline migration of every dependent curation, modality, Prepared
view, and training binding.

The nominal full-task sources used by the stage-3 OpenPI profiles are pinned
separately in
`writing_stage3_fulltask_sources_2026_08_04_through_08_07_v1.json`. Its first
five sources reproduce the 879-eligible 2026-08-04 through 2026-08-06 baseline;
all seven reproduce the 1200-eligible extension through 2026-08-07. Raw episode
counts in this manifest are provenance inventory, not OpenPI eligibility
counts. Eligibility remains owned by the prepared-training profiles and their
annotation selections.

`writing_stage3_0804_0806_front_h16_absolute_replay_v2.json` preserves the
historical 879-Episode Stage3 training input by binding the immutable prepared
binding `28c9d7693324fe738f635e6da11ab8875dd793302af6e4dcf3081687b838f39e`.
It still re-audits the explicitly selected annotations
and requires the resulting Episode identities and order to match that binding;
it does not rebuild the historical view from mutable annotation directories.
After the logical symlink view is present, reproduce that exact input with:

```bash
python -m deployments.openpi.training.prepared_skill_launch prepare \
  --profile deployments/openpi/configs/training/\
writing_stage3_0804_0806_front_h16_absolute_replay_v2.json \
  --no-motion
```

Use the existing non-`replay` profiles when intentionally building a new
immutable snapshot from the currently selected annotations.

Remote folder organization may expose these sources through a read-only
`LogicalCollections` symlink view. The physical dataset directories remain
authoritative, and neither this manifest nor that view authorizes a move,
merge, overwrite, or deletion.

The current OpenPI profiles reference the stable view at
`/home/user/zc/datasets/RealmanDMTAC/LogicalCollections/`
`writing_four_stage_fulltask_2026_08_04_through_08_07_v1/sources/<source_id>`.
Each symlink resolves to the same authoritative directory pinned above, so
existing immutable Prepared bindings still compare against the same resolved
source root. The view changes path organization only: source order, episode
selection, annotation selection, mixing, windows, and task semantics remain
owned by each training profile. Stage1-only and detail-stage1 captures are not
members of this view.

One historical Episode later acquired a second annotation set. Profiles that
contain that Episode explicitly select
`stage-1786011149701-e78dad71`, which is the set used when the existing
bindings were first built. This converts the old implicit unique selection
into an explicit one without changing the eligible Episode set.

# Managed task progress profiles

This directory is the application-neutral catalog for deployable collection
task profiles. Pure schemas remain in `dataset_contracts`; this package loads
editable profile sources and derives their canonical identities at runtime.

Organization:

```text
profiles/
  linear/       # currently admitted by RealTeleop
  tree/         # reserved catalog namespace; runtime admission stays gated
  catalog.py
  common_semantics.py
```

Each task is self-contained:

- `<name>.progress.json` owns the task ID, unique coarse progress-step IDs,
  display names, ordering, legal transitions, terminals, and retry policy;
- optional `<name>.stages.json` owns the strict ordered execution/failure
  stages inside each coarse step and is rebound to the co-located progress
  profile when the editable source bundle is loaded;
- optional `<name>.refinement.json` owns post-capture boundaries inside those
  coarse steps and is rebound to the co-located progress profile; this is the
  compatibility salvage-boundary model rather than the new occurrence model;
- an optional pair of `<name>.progress_detailed.json` and
  `<name>.stages_detailed.json` is an explicit collection variant. The
  detailed progress source must preserve the coarse progress hash; only the
  ordered stage profile may split one coarse step into fine stages. The
  catalog exposes this variant with the selection ID `<progress_id>_detailed`;
  it is not included in the compatibility coarse-only listing;
- a stage/refinement file without a same-name progress file is invalid.

`semantic_content_hash` and companion `task_progress_profile_hash` values in
these editable JSON sources are treated as generated hints and are not
operator-maintained admission checks. The loader removes/recomputes them after
each edit. Constructed profile objects and immutable dataset artifacts still
carry exact canonical hashes, so one dataset cannot mix different profile
contents.

A coarse step may additionally bind `arm_control.left/right` to `teleop` or
`fixed_hold`. In RealTeleop, `fixed_hold` has one exact scope:
`arm_and_end_effector`; a profile cannot use it to fix only the robot arm while
leaving that side's configured end effector under live control. Profiles that
omit `arm_control` retain legacy behavior in the default/ignore modes and are
rejected by strict-profile enforcement.

`semantic_type` may reuse the optional vocabulary in `common_semantics.py` to
support cross-task search such as `ACQUIRE_OBJECT`. It is not an inherited
template and is not required: task-specific semantic types remain legal.
Exact task meaning, step identity, and transition semantics always live in the
task's own progress profile. This avoids making old profiles depend on a
mutable shared primitive definition.

The bundled coarse profiles include stage profiles for post-capture occurrence
annotation and exact recovery entry. Those fine stages are not online
Advance/Failure controls; current collection operators manage only coarse
progress and Failure. A recovery capture may explicitly start at a noninitial
fine stage, but must record `RECOVERY_FROM_FAILURE`. The explicit
`_detailed` variant is the exception for a new dataset: selecting it binds the
fine stage profile at capture time, so the resulting coarse annotation timeline
already contains those fine stages. It must not be mixed with a dataset that
has already bound the coarse stage profile.

The formal DMTAC task also provides three writing variants:

- `dmtac_rm65_bimanual_position_writing_four_stage_progress`: right-hand
  brush acquisition, left-hand cap removal/placement, right-hand writing, and
  right-hand brush return;
- `dmtac_rm65_bimanual_position_writing_five_stage_progress`: the four-stage
  workflow with a separate ink-dipping step between cap removal and writing;
- `dmtac_rm65_bimanual_position_writing_three_stage_progress`: the same task
  without the complete left-hand cap workflow. Its final acquisition stage
  enters writing preparation directly instead of presenting the brush to the
  left hand.

Both variants own an exact same-name `.stages.json` profile and keep the
formal `dmtac_rm65_bimanual_position_v1` task ID as their own profile metadata.
RealTeleop exposes all admitted LINEAR profiles independently of the formal
dataset task/name, so this ID is not a GUI filtering key. They are different
dataset-wide profile identities and must never be mixed in one dataset.

The four-stage writing profile additionally owns the explicit
`dmtac_rm65_bimanual_position_writing_four_stage_progress_detailed` selection.
It keeps the same four coarse steps but splits `right_write` into
`right_write.pen_down`, `right_write.write`, and `right_write.pen_up`. For a
coarse four-stage Episode, the annotation application discovers this same
pair after the coarse annotation set is sealed and publishes the fine labels
as an additive detail overlay; it never rewrites the original Episode,
Parquet, video, or coarse annotation set.

The catalog also contains
`test_dmtac_rm65_bimanual_position_v1_three_stage_progress`. It deliberately
uses the formal DMTAC task ID so that it appears in the current RealTeleop
profile selector, while every step display name and semantic type is marked as
test-only. Use it only with a new/empty dataset directory when exercising the
three-stage GUI workflow; a dataset already bound to the single-stage profile
must not be mixed or rebound.

External profiles remain supported as explicit JSON paths. Managed profiles
use the unambiguous reference form:

```text
catalog:<profile_id>
```

## Dataset-wide uniqueness

The first progress-enabled Episode binds the exact
`TaskProgressProfileV1` to:

```text
meta/progress/profile.cjson
```

That immutable canonical profile is the dataset-level binding; no second
profile identity schema is maintained. Every later Episode must use the same
semantic content hash. A populated dataset without this capability remains
profile-less and cannot be partially upgraded in place. Older datasets that
already contain progress Episodes may be backfilled only when every
Episode-local `profile.cjson` is exactly identical.

The corresponding exact stage profile, when present, is independently locked
at `meta/progress/stages.cjson`. A dataset cannot mix stage-profiled and
unprofiled progress Episodes or switch a stage semantic hash after binding.

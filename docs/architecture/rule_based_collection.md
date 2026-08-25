# Rule-based collection boundary

The rule-based collector is a small plan-composition seam for collection
workflows that need a deterministic combination of existing operations, for
example:

```text
initialization -> replay frames 10..24 -> custom action x3 -> replay frames 80..90
```

Its implementation is `robot_motion.replay.rule_based`. The collection-facing
bridge is `teleop_runtime.collection.rule_based_adapter`; it owns only:

- ordered declarative steps (`RuleBasedInitialize`, `RuleBasedReplay`, and
  `RuleBasedCustomAction` / `RuleBasedActionSegment`);
- expansion of those steps into `RuleBasedCollectionDirective` values; and
- per-step `RuleBasedCaptureSelection` intent; and
- resolution of each directive into the existing `RobotAction` /
  `TeleopCommand` boundary for a caller that already owns a collection loop.

It deliberately reuses the existing `ReplayTrajectoryGroup`, qpos-init
selection, replay action construction, and canonical `RobotAction` APIs.

## Explicit non-ownership

The rule-based layer does not:

- call hardware, start a device, dispatch a robot command, solve IK, or apply
  safety/admission;
- own a collection loop, worker, queue, episode lifecycle, or dataset writer;
- define or replace a dataset schema; or
- create a second replay implementation.

`RuleBasedCollectionAdapter` consumes each directive and resolves its action;
the caller then applies the resulting command through the normal lower runtime
and submits a sample only when `emission.recordable` is true. An empty
`stream_ids` allowlist means all already-bound streams, while a populated
allowlist exposes the requested stream selection to the caller. The adapter
does not rewrite samples or writer jobs, so the existing sample builder and
writer remain authoritative for causal cuts, validation, persistence, sealing,
and commit.

## Runtime rule-based capture segments

`RuleBasedCustomAction` is appropriate for a fixed canonical action repeated a
known number of ticks. For a rule-based segment whose action depends on the
current runtime observation, use `RuleBasedActionSegment` instead. It reuses
the existing `RobotActionSource.produce_action(observation, context=...)`
contract and adds only two collection-plan facts:

- `RuleBasedCaptureSelection` says which already-bound streams are eligible for
  this segment;
- `RuleBasedSegmentStopCondition` supplies a mandatory hard `max_steps`, an
  optional `min_steps`, and an optional `stop_when` predicate evaluated after
  each produced action.

The runtime caller advances such a segment with
`RuleBasedCollectionCursor.next_runtime(observation, context=...)`. A source
returning `None` is rejected rather than interpreted as an implicit stop. The
terminal directive is returned with `segment_terminal=True` and an explicit
`segment_stop_reason`; only the next cursor request advances to the following
plan step. This gives the caller a precise point to finish the capture, read
the actual TCP state, publish a `RuleBasedBoundaryState`, and then request the
next relative replay.

The action source's `start()`/`close()` lifecycle is intentionally not called
by the plan cursor. The existing runtime that owns the source remains
responsible for lifecycle, observation acquisition, dispatch, safety, and
writer interaction.

## Relative replay at a segment boundary

The pure boundary contract is `RuleBasedBoundaryState` together with
`rebase_replay_action_at_boundary`. It uses the source trajectory's recorded
actual TCP at `source_start_index` as the source anchor, converts each source
pose target into a fixed relative delta, and applies that delta to the
runtime-measured TCP in `RuleBasedBoundaryState`. The result is an absolute
`RobotAction` in the already-validated per-side pose frames.

This first seam is intentionally pose-only and fails closed when the source
actual TCP history or per-side frame identity is missing. Joint-qpos replay is
not treated as a relative pose replay. A runtime action segment publishes its
terminal directive before the cursor resolves the following step, so the
boundary can be measured from the actual post-segment state rather than from a
planned action target.

This module is a pure, default-off composition seam. Its unit tests use only
synthetic trajectory arrays, canonical action objects, and an in-process
source; they are not evidence of robot readiness or production collection
integration. No existing CLI or GUI entrypoint is changed by this adapter;
adoption is an explicit caller choice. The existing collection adapter still
owns only command resolution; production runtime wiring for
`next_runtime(...)`, boundary measurement, and dataset submission remains a
separate integration step.

## MuJoCo composition proof

The simulation-only example
`sim.mujoco.replay.rule_based_multistage_demo` exercises the same boundary
without adding a second control or writer implementation. Its sequence is:

```text
initial alignment (not recorded)
  -> absolute replay of --stage-a
  -> right TCP circle in desk XY at constant desk Z
  -> measured actual-TCP boundary
  -> relative replay of --stage-b
```

Run it with an external canonical UMI dataset root:

```bash
MUJOCO_GL=egl PYTHONPATH=. python -m sim.mujoco.replay.rule_based_multistage_demo \
  --dataset-root /path/to/dataset \
  --episode 0 --stage-a 0:120 --stage-b 250:400 \
  --output /tmp/rule_based_mujoco_demo --record-depth --no-sleep
```

`start:end` spans are half-open and intentionally editable. The output keeps
the normal simulation parquet/video layout and adds
`rule_based_composition_manifest.json`; each row carries its phase in
`sample_metadata_json`. For this composition demo the writer also carries the
source canonical pose-frame contract and writes the action target in
desk-absolute xyz-quat form, so the result can be checked with the existing
UMI loader rather than being only a visualization artifact. The relative phase
fails closed if the source group does not contain actual TCP history or if the
source/boundary pose frames do not match. The optional passive viewer is
MuJoCo-only (`--viewer`); rendered RGB/depth videos are also produced for
headless verification.

```bash
PYTHONPATH=. python - <<'PY'
from sim.mujoco.replay.umi_dataset import load_umi_replay_episode
episode = load_umi_replay_episode("/tmp/rule_based_mujoco_demo", 0)
print(episode.action_dim, len(episode.frames))
PY
```

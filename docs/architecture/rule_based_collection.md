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
  `RuleBasedCustomAction`);
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

## Relative replay at a segment boundary

The pure boundary contract is `RuleBasedBoundaryState` together with
`rebase_replay_action_at_boundary`. It uses the source trajectory's recorded
actual TCP at `source_start_index` as the source anchor, converts each source
pose target into a fixed relative delta, and applies that delta to the
runtime-measured TCP in `RuleBasedBoundaryState`. The result is an absolute
`RobotAction` in the already-validated per-side pose frames.

This first seam is intentionally pose-only and fails closed when the source
actual TCP history or per-side frame identity is missing. Joint-qpos replay is
not treated as a relative pose replay. The current finite cursor still expands
only static steps; a later step will pass a boundary state into the plan
execution path so a collection segment can produce the state used by the next
relative replay segment.

This module is a pure, default-off composition seam. Its unit tests use only
synthetic trajectory arrays and canonical action objects; they are not evidence
of robot readiness or production collection integration. No existing CLI or
GUI entrypoint is changed by this adapter; adoption is an explicit caller
choice.

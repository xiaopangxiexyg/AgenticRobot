# Rule-based collection boundary

The rule-based collector is a small plan-composition seam for collection
workflows that need a deterministic combination of existing operations, for
example:

```text
initialization -> replay frames 10..24 -> custom action x3 -> replay frames 80..90
```

Its implementation is `robot_motion.replay.rule_based`. It owns only:

- ordered declarative steps (`RuleBasedInitialize`, `RuleBasedReplay`, and
  `RuleBasedCustomAction`);
- expansion of those steps into `RuleBasedCollectionDirective` values; and
- per-step `RuleBasedCaptureSelection` intent.

It deliberately reuses the existing `ReplayTrajectoryGroup`, qpos-init
selection, replay action construction, and canonical `RobotAction` APIs.

## Explicit non-ownership

The rule-based layer does not:

- call hardware, start a device, dispatch a robot command, solve IK, or apply
  safety/admission;
- own a collection loop, worker, queue, episode lifecycle, or dataset writer;
- define or replace a dataset schema; or
- create a second replay implementation.

An existing collection/runtime adapter consumes each directive and applies its
action through the normal lower runtime. The directive's capture selection is
also an input to that adapter: an empty `stream_ids` allowlist means all
already-bound streams, while a populated allowlist selects only those stream
IDs. `enabled=False` is useful for initialization or other non-training
segments. The existing sample builder and writer remain authoritative for
causal cuts, validation, persistence, sealing, and commit.

This module is a pure, default-off composition seam. Its unit tests use only
synthetic trajectory arrays and canonical action objects; they are not evidence
of robot readiness or production collection integration.

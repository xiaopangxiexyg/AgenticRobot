# Rule-based real writing collection

This entrypoint composes existing RealTeleop, replay, fixed-trajectory, safety,
and formal writer implementations. It does not introduce another hardware
loop or recorder.

## Executed plan

The fixed plan is:

1. Move both arms through the existing qpos initialization path to the first
   `right_write.pen_down` source frame. This phase is not admitted to the
   dataset.
2. Replay and record `right_write.pen_down`. The right arm follows the source;
   the left arm is pinned to the source episode's first `fixed_hold` target.
3. Generate and record one selected catalog path on the right TCP. The path is
   anchored at the causally measured pen-down endpoint. The left arm keeps the
   same source `fixed_hold` target.
4. Publish the measured writing endpoint and relative-replay and record
   `right_write.pen_up` on the right arm. The left arm receives no new target
   and remains held.

Only steps 2-4 are admitted to the formal dataset. Cameras, tactile capture,
sample synchronization, and persistence remain atomic and are owned by the
existing formal writer.

## Coordinate and admission boundary

The application fails before device opening unless all of these conditions
hold:

- the source is a typed dataset with an explicit pose-frame contract;
- the source and current real deployment pose-frame contract hashes match
  exactly;
- source action and actual-TCP anchors can be converted to `canonical_desk`;
- the detail overlay exactly binds the source sealed progress Episode and its
  base annotation set;
- the ordered detail stages are successful `pen_down`, `write`, and `pen_up`;
- every source row from pen-down through pen-up labels the left arm
  `fixed_hold`;
- both arms, position mode, collision avoidance, active TCP envelopes,
  tool-frame force guarding, and formal collection are enabled;
- all Active admittance overlays are disabled.

The launcher pins
`teleop_runtime/configs/real_teleop/rule_based_writing_bimanual_workspace_v1.json`
and enables it in Active mode. The profile covers both final `tool_tcp` targets
at the authoritative pre-IK boundary and resolves each side's absolute bounds
from the same runtime calibration workspace already selected for that arm. It
does not add a force/admittance modifier or maintain a second workspace owner.

The fixed path is authored in glyph-local XY and rotated only within
`canonical_desk`. Its right-arm start is the causal real TCP endpoint converted
through the current runtime contract. The existing RealTeleop resolver then
performs the canonical-desk to per-arm hardware-frame conversion.

## Offline command inspection

From `/home/user/zc/AgenticRobot`, in the complete `dexteleop310` environment:

```bash
bash scripts/run_rule_based_writing_collection.sh \
  --source-dataset /absolute/path/to/source_dataset \
  --source-episode 0 \
  --detail-annotation-dir /absolute/path/to/detail_annotation_set \
  --dataset-root /home/user/zc/datasets/RealmanDMTAC/FormalCollection \
  --tactile-bundle-root /absolute/path/to/existing/tactile_root \
  --trajectory digit_1_v1 \
  --size-m 0.08 \
  --direction up \
  --dry-run
```

`--dry-run` validates only shell-level paths and prints the resolved command;
it does not import the source episode and does not prove hardware readiness.

## One real collection

Run the existing five-pedal service in one terminal:

```bash
cd /home/user/zc/AgenticRobot
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate dexteleop310
python -m hardware.foot_pad --count 5
```

After the normal device, calibration, workspace-clearance, and operator
preflight, run the collector in a second terminal by removing `--dry-run` from
the inspected command. Start exactly one formal Episode in the existing GUI.
The rule extension remains idle until the formal writer and causal dual-arm
feedback are ready, then runs the bounded plan and enters the existing hardware
hold after completion.

The default fixed-path speed is derived from the right TCP average over the
source pen-down and pen-up spans, with gaps excluded. Use
`--speed-m-s VALUE` only when an explicit reviewed speed is required. Supported
directions are `up`, `right`, `down`, and `left`; `--size-m` is the glyph height
before that rotation.

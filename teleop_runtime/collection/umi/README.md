# UMI v2.3 collection

This package is the formal arm-free UMI collection frontend. It reuses the
existing DexTeleop Vive source, RealSense source and causal frame buffer,
AngleTrigger source, scalar-to-gripper retargeter, DMTac backend,
`HandStateHistoryPoller`, LeRobot-style writer, arm-mode label columns,
`CalibrationWebUi`, and four-pedal adapter.

It never constructs an arm backend. The legacy v2.3 `left.arm.pose` and
`right.arm.pose` layout components carry measured UMI tool-TCP poses, with
dataset metadata declaring `embodiment_has_arm=false`.

## Pose convention

The runtime uses the project convention `p_B = T_A_to_B @ p_A`:

```text
T_tcp_to_vive_world = T_tracker_to_vive_world @ T_tcp_to_tracker
T_tcp_to_desk       = T_vive_world_to_desk @ T_tcp_to_vive_world
```

`T_tcp_to_tracker` is currently allowed to be identity only when the config
explicitly sets `allow_identity_tcp_placeholder=true`. The placeholder status
and matrix are stored in dataset and episode metadata.

## Configuration validation

The default profile contains the current real-collection wrist camera serials:

- left: `261122273924`
- right: `260522278055`

Validate config, policy layout, calibration expiry, and device bindings without
initializing any device:

```bash
python -m teleop_runtime.collection umi \
  --config teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json \
  --validate-only
```

The calibration file must exist and still be within its configured one-hour
maximum lifetime. A changed or renewed calibration cannot be appended to an
existing populated dataset because its pose-frame contract/hash differs.
Tactile data uses the existing external-stream writer; the default compression
is `none`, while `lz4` and `zstd` remain available through the config when the
corresponding existing optional package is installed.

## Formal run

After filling the DMTac assignment and producing a fresh UMI calibration:

```bash
python -m teleop_runtime.collection umi \
  --config teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json
```

Useful staging options:

```bash
# Collect input/state while preventing physical gripper command transmission.
python -m teleop_runtime.collection umi --no-send-gripper

# Require the existing F13-F16 four-pedal virtual keyboard.
python -m teleop_runtime.collection umi --foot-pad
```

No command starts an episode automatically. Recording begins and ends through
the reused formal Web GUI. Device startup and real gripper output are hardware
operations and must follow the project safety procedure.

## Operator labels

The existing GUI fix/init controls, keys (`c`/`g`), or foot pedals generate
latched per-side annotations. They do not control a robot:

- normal: `teleop`
- fix: `fixed_hold`
- init: `init_pose`

The labels remain in the existing `teleop.arm_mode.*` columns for v2.3 reader
compatibility. JSON detail records `annotation_only=true`,
`control_effect=false`, and `embodiment_has_arm=false`. Init arrival audit is
invalid with reason `annotation_only_no_arm`.

## Collection rejection policy

A row is written only when both sides have causal, fresh Vive pose, wrist image,
measured gripper state, and—when configured as required—tactile data. Invalid or
stale AngleTrigger input freezes the last physical target by sending no new
command and pauses row submission. Calibration expiry stops the active episode
and terminates the run so a new dataset/session can be created after calibration.

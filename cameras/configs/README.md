# Camera Process Configs

This directory is the canonical home for shared camera configuration used by
teleop runtime, deployment, real-to-sim replay, and offline camera processing.

- `rm65/devices/realsense.json` stores RealSense serials and requested stream
  profiles for RM65 logical camera keys. The default color transport is native
  `yuyv`; runtime sources convert it back to the public RGB/BGR image contract.
- `rm65/cameras/realsense_registry.json` stores RM65 logical camera metadata,
  profile defaults, and URDF frame placements/extrinsics. Runtime RealSense
  intrinsics are still captured from device profiles and written into dataset
  snapshots.

The legacy `teleop_runtime/configs/...` JSON files are kept as compatibility
mirrors for older commands that pass those paths explicitly.

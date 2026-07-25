# Teleop camera device configs

This directory owns workstation-side camera transport assignments. The
`realsense.json` file maps logical keys (`front`, `left_wrist`, and
`right_wrist`) to RealSense serials and requested stream profiles used by
teleop, collection, and policy deployment.

`realsense_front.json` is retained as the single-front station variant for
explicit callers; it is not the multi-camera default.

The checked-in RM65 station profiles use `bgr8` for color transport.  Real-teleop
also defaults to `--camera-color-format bgr8` and overrides stale workstation
profiles unless another format is explicitly requested.  RGB/depth resolution
is kept as configured; startup first-frame waiting is longer than the runtime
freshness timeout so a multi-camera negotiation does not trigger a pipeline
restart loop.

Shared RM65/Rig camera geometry and URDF placements live in
`robot_motion/configs/rm65_camera_registry.json`. Runtime intrinsics may be
overlaid into dataset snapshots; do not put those snapshots or new device
assignments into the shared robot profile.

Historical `cameras/configs/rm65/devices/...` references are resolved only by
the offline metadata path migration helper. New callers must use this
directory or the exported path constants in
`teleop_runtime.configs.camera_paths`.

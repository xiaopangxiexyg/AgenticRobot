# Teleop camera device configs

This directory owns workstation-side camera transport assignments. The
`realsense.json` file maps logical keys (`front`, `left_wrist`, and
`right_wrist`) to RealSense serials and requested stream profiles used by
teleop, collection, and policy deployment.

`realsense_front.json` is retained as the single-front station variant for
explicit callers; it is not the multi-camera default.

Shared RM65/Rig camera geometry and URDF placements live in
`robot_motion/configs/rm65_camera_registry.json`. Runtime intrinsics may be
overlaid into dataset snapshots; do not put those snapshots or new device
assignments into the shared robot profile.

Historical `cameras/configs/rm65/devices/...` references are resolved only by
the offline metadata path migration helper. New callers must use this
directory or the exported path constants in
`teleop_runtime.configs.camera_paths`.

# Teleop camera device configs

This directory owns workstation-side camera transport assignments. The
`realsense.json` file maps logical keys (`front`, `left_wrist`, and
`right_wrist`) to RealSense serials and requested stream profiles used by
teleop, collection, and policy deployment.

The current RM65 physical wrist binding is authoritative and must not be
inferred from USB enumeration order:

```text
left_wrist  -> RealSense 260522278055
right_wrist -> RealSense 261122273924
```

Logical camera roles are also independent of hardware startup order.  The
optional numeric `metadata.startup_priority` is lifecycle-only: lower values
start first, while a missing value preserves JSON declaration order.  The
current station starts `front`, then serial `261122273924`, then serial
`260522278055`, matching the order recorded by the stable August 4/5 captures.

An already running process keeps the configuration it loaded at startup. The
corrected binding applies after the next complete collection/deployment
restart. Historical datasets remain immutable; training resolves their stored
wrist keys from each dataset's recorded RealSense serial.

`realsense_front.json` is retained as the single-front station variant for
explicit callers; it is not the multi-camera default.

The shared station profiles use bounded software reopen attempts for transient
startup failures. Only the reviewed front serial opts into one targeted device
re-enumeration when a pipeline opens but produces no first frame; wrist cameras
never request a startup hardware reset. This applies to real teleop, collection,
and policy deployment that load these profiles. A native SDK call that remains
blocked is not safe to retry or close immediately; the formal multi-camera
startup owner reports a terminal timeout instead. Other camera inventories
must opt in separately after device-specific validation.

`agent_gateway_realsense.json` is the independent Agent Gateway inventory.
It preserves the reviewed three serial assignments at 640x480/15 Hz and
enables aligned depth only on `front`. The front camera alone permits one
targeted reset after a first-frame warmup failure; ordinary connection errors
and both wrist cameras do not request a reset. All three cameras may serve RGB.
Wrist depth is intentionally disabled:
the wrist mounts move with link 6, while the shared registry currently provides
no numeric activation-time camera pose for those roles. Do not enable it by
copying the front camera's static arm-base transform; add and review the
joint/FK-backed pose binding first.

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

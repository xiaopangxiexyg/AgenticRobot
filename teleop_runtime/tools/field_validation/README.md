# Teleoperation field validation tools

These scripts exercise live device runtimes and are not pytest tests.  Use
them only on an explicitly authorized host after the device-specific safety
preflight.  The default pytest suite must never import or execute them.

## Camera and tactile health probe

`camera_tactile_health_probe.py` opens the configured RealSense streams and
DMTAC tactile workers, observes freshness, and closes owned resources in a
`finally` block.  It creates no arm or gripper transport and commands no robot
motion.  Recovery exercises may stop and restart an owned camera pipeline or
tactile child process; camera hardware reset remains separately opt-in.

`--help` and `py_compile` are the only device-free checks.  They do not prove
that cameras, USB assignments, tactile calibration, or runtime recovery are
ready on a target host.

## 2026-08-09 path migration

`tests/manual_camera_tactile_health_probe.py` moved to
`teleop_runtime/tools/field_validation/camera_tactile_health_probe.py` because
it is a live runtime probe rather than a pytest test.  The old path has no
compatibility wrapper, and repository-root discovery no longer depends on the
script's directory depth.

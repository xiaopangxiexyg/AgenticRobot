# RM65 field validation tools

These scripts are operator-run hardware diagnostics, not pytest tests.  They
must never be imported or executed by the default test suite.  Run them only
with an explicitly authorized RM65 target, the device-specific safety
preflight, and a clear stop procedure.

## Tools and gates

- `rm65_safe_real_robot_smoke.py` connects to the selected arm and reads state
  and force by default.  Motion and force-position commands additionally
  require `--execute --confirm MOVE_RM65`; both-arm motion has a separate
  `--allow-both-execute` gate.
- `rm65_force_pose_survey.py` connects to one arm and reads force/state.  It
  does not command motion; any pose change is an operator action in teach mode.
- `rm65_force_standard_setup.py` reads the current configuration by default.
  Install-pose, tool-frame, or force-calibration changes require
  `--execute --confirm CONFIG_RM65_FORCE`.  Vendor force calibration can move
  the robot.

`--help` and `py_compile` are the only hardware-free checks.  A successful
offline check does not establish network reachability, device readiness, safe
workspace clearance, or real-motion correctness.

## 2026-08-09 path migration

The scripts moved out of `tests/` because they perform device I/O and are not
pytest tests:

- `tests/rm65_safe_real_robot_test.py` ->
  `hardware/rm65/tools/field_validation/rm65_safe_real_robot_smoke.py`
- `tests/rm65_force_pose_survey.py` ->
  `hardware/rm65/tools/field_validation/rm65_force_pose_survey.py`
- `tests/rm65_force_standard_setup.py` ->
  `hardware/rm65/tools/field_validation/rm65_force_standard_setup.py`

The old paths have no compatibility wrappers so that hardware tools cannot be
mistaken for tests.  Repository-root discovery is content-based rather than
dependent on directory depth, so direct `python <path> --help` remains valid.

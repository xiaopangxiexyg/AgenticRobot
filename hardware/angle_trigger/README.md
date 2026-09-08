# Angle Trigger

This module discovers one or two compatible USB serial angle sensors. A port is
accepted only when exactly one of servo IDs `1` and `2` responds to the
read-only `PRAD` angle query. The default `--side both` preserves the bilateral
teleop/UMI behavior; `--side left` and `--side right` are explicit single-side
profiles.

Install the serial dependency in the runtime environment when it is not already
provided by the project:

```bash
python -m pip install -r hardware/angle_trigger/requirements.txt
```

The first bilateral launch performs two independent calibrations:

1. Move the left sensor and press Enter to assign it, then repeat for the right
   sensor.  This assignment is saved in `assignment/latest.json`.
2. For left then right, hold the intended `0` position and press Enter to
   capture one second; repeat for the intended `1` position.  This interpolation
   data is saved separately in `angles/latest.json`.

Both files live under `teleop_runtime/configs/hardware/angle_trigger` by
default.  When a USB location changes, the module repeats only step 1 and keeps
the fixed angle interpolation.  A swap between two identical no-serial-number
devices on the same two unchanged ports cannot be inferred automatically; run
`--reassign` in that situation.

Single-side calibration uses the same procedure for only the selected side and
stores `assignment/right.json` plus `angles/right.json` (or the corresponding
left files). It never overwrites the bilateral `latest.json` files. A
single-side reader may also consume the matching side of an existing bilateral
profile, so temporarily unplugging the unused side does not invalidate the
selected side.

If the same physical sensor is moved to another USB port, its saved
location-based identity may no longer match. Reassign only the selected side;
the existing angle endpoints can be retained:

```bash
python -m hardware.angle_trigger --side right --reassign
```

This writes `assignment/right.json` without overwriting the bilateral
`assignment/latest.json`. If the connected sensor is a different physical
sensor, or a former left sensor is being repurposed as right, its old endpoint
calibration must not be reused. Reassign and recalibrate together:

```bash
python -m hardware.angle_trigger \
  --side right --reassign --recalibrate-angles
```

The non-interactive Sim-HIL device stage deliberately does not update an
identity automatically. An error such as `Could not uniquely resolve the
configured right sensor` therefore means that the connected sensor does not
match the saved right identity; run the appropriate command above, then retry
the `device_input` stage.

```bash
python -m hardware.angle_trigger
python -m hardware.angle_trigger --once
python -m hardware.angle_trigger --reassign
python -m hardware.angle_trigger --recalibrate-angles
python -m hardware.angle_trigger --list

# One right-side trigger for single-arm Sim-HIL
python -m hardware.angle_trigger --side right
python -m hardware.angle_trigger --side right --once
python -m hardware.angle_trigger --side right --reassign
python -m hardware.angle_trigger --side right --recalibrate-angles
```

`AngleTriggerReader` can also be used by another Python process after calling
`prepare_reader()` or by constructing it with a resolved assignment and angle
calibration. `reader.read()` returns only its declared active sides. The
reader and calibration retain their raw normalized `0..1` values. At the
real-teleop scalar-to-DMTAC boundary (and its passive Sim-HIL mirror), the
angle-trigger mapping is intentionally reversed: trigger `0` commands closed
and trigger `1` commands open. Q6 mapping remains unchanged.
bilateral default returns `{"left": 0.0..1.0, "right": 0.0..1.0}`; a right-only
reader returns `{"right": 0.0..1.0}`.

GUI applications should use `DeferredAngleTriggerReader`: constructing the GUI
does not discover or open serial devices, and `open()` loads only a previously
calibrated profile. Interactive assignment/calibration remains an explicit CLI
operation instead of blocking a GUI control thread.

# Angle Trigger

This module discovers exactly two compatible USB serial angle sensors.  A port
is accepted only when exactly one of servo IDs `1` and `2` responds to the
read-only `PRAD` angle query.

Install the serial dependency in the runtime environment when it is not already
provided by the project:

```bash
python -m pip install -r hardware/angle_trigger/requirements.txt
```

The first launch performs two independent calibrations:

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

```bash
python -m hardware.angle_trigger
python -m hardware.angle_trigger --once
python -m hardware.angle_trigger --reassign
python -m hardware.angle_trigger --recalibrate-angles
python -m hardware.angle_trigger --list
```

`AngleTriggerReader` can also be used by another Python process after calling
`prepare_reader()` or by constructing it with a resolved assignment and angle
calibration.  `reader.read()` returns `{"left": 0.0..1.0, "right": 0.0..1.0}`.

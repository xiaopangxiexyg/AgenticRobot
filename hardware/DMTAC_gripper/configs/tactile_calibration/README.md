# DM-Tac fixed base-frame calibration

This directory stores one lossless `uint8 [480, 640]` SDK base frame per sensor
serial, plus a JSON sidecar containing provenance and SHA-256. Calibration must
be captured while that sensor is unloaded. Do not copy a calibration between
serial numbers.

Runtime and offline postprocessing use policy `auto` by default: a matching
serial is loaded when present; a missing file preserves the SDK raw solution.
Use policy `required` for production validation or `off` for diagnostics.

Capture directly from unloaded real sensors with:

```bash
python hardware/DMTAC_gripper/examples/calibrate_tactile_base_frames.py
```

The script uses a multi-frame pixelwise median, checks temporal stability,
installs the result into the current SDK session, and verifies the saved digest.
Existing serials require an explicit `--overwrite`.

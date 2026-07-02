# RM65 Control And Calibration

This folder wraps RealMan API2 `Robotic_Arm` version `1.1.5` for the RM65 arms.

- Left arm IP: `192.168.101.21`
- Right arm IP: `192.168.1.20`
- Local adapter: `hardware.rm65.rm_api2_adapter.Arm`
- Calibration profiles: `hardware/rm65/utils/configs/calibrations/`

Activate the conda environment once before running the commands below:

```bash
conda activate <your-rm65-env>
```

All commands are expected to run from the repository root with that environment
active. The examples below assume the mounted tool name is `inspire`; replace
that name when calibrating another end effector.

## Standard Workflow

### 1. Record Four Calibration Poses

The recorder is read-only. It waits for Enter before sampling and requires a
5 second stable window by default.

Left:

```bash
python hardware/rm65/utils/scripts/rm65_auto_record_force_poses.py \
  --arm left \
  --max-records 4 \
  --force-stability-metric std \
  --force-stable-n 0.25 \
  --torque-stable-nm 0.02
```

Right:

```bash
python hardware/rm65/utils/scripts/rm65_auto_record_force_poses.py \
  --arm right \
  --max-records 4 \
  --force-stability-metric std \
  --force-stable-n 0.25 \
  --torque-stable-nm 0.02
```

### 2. Replay Four Poses And Record Force

During execution, Ctrl-C/SIGTERM sends `Arm_Slow_Stop` before closing the
socket. Use `--interrupt-stop-mode fast` only when a hard stop is required.

Left dry-run:

```bash
python hardware/rm65/utils/scripts/rm65_replay_force_pose_records.py \
  --arm left \
  --calib4 \
  --expected-tool inspire \
  --check-current-start
```

Left execute:

```bash
python hardware/rm65/utils/scripts/rm65_replay_force_pose_records.py \
  --arm left \
  --calib4 \
  --expected-tool inspire \
  --check-current-start \
  --execute \
  --confirm REPLAY_RM65_FORCE_POSES
```

Right dry-run:

```bash
python hardware/rm65/utils/scripts/rm65_replay_force_pose_records.py \
  --arm right \
  --calib4 \
  --expected-tool inspire \
  --check-current-start
```

Right execute:

```bash
python hardware/rm65/utils/scripts/rm65_replay_force_pose_records.py \
  --arm right \
  --calib4 \
  --expected-tool inspire \
  --check-current-start \
  --execute \
  --confirm REPLAY_RM65_FORCE_POSES
```

### 3. Fit, Apply, And Validate Tool Payload

The apply step also updates the arm/tool calibration profile with payload/CoM
and an initial TCP. A fitted CoM farther than 0.20 m from the current tool
origin falls back to `hand_base`; `inspire` also starts from `hand_base` by
default and can be adjusted in the GUI. Profiles are arm-specific, so calibrate
left and right independently even when the tool name is identical.

Left fit:

```bash
python hardware/rm65/utils/scripts/rm65_force_payload_calibrate.py \
  --arm left \
  --expected-tool inspire
```

Left apply:

```bash
python hardware/rm65/utils/scripts/rm65_apply_tool_payload_from_fit.py \
  --arm left \
  --expected-tool inspire \
  --execute \
  --confirm APPLY_RM65_TOOL_PAYLOAD
```

Left validate:

```bash
python hardware/rm65/utils/scripts/rm65_validate_tool_payload.py \
  --arm left \
  --expected-tool inspire \
  --checkpoints 4 \
  --samples 20 \
  --interval-sec 0.05 \
  --evaluate-source tool
```

Right fit:

```bash
python hardware/rm65/utils/scripts/rm65_force_payload_calibrate.py \
  --arm right \
  --expected-tool inspire
```

Right apply:

```bash
python hardware/rm65/utils/scripts/rm65_apply_tool_payload_from_fit.py \
  --arm right \
  --expected-tool inspire \
  --execute \
  --confirm APPLY_RM65_TOOL_PAYLOAD
```

Right validate:

```bash
python hardware/rm65/utils/scripts/rm65_validate_tool_payload.py \
  --arm right \
  --expected-tool inspire \
  --checkpoints 4 \
  --samples 20 \
  --interval-sec 0.05 \
  --evaluate-source tool
```

### 4. Confirm Or Adjust Tool TCP

The GUI shows a draggable 3D arm view and the TCP axes. Drag empty space to
orbit, use the mouse wheel to zoom, and drag the TCP point on the selected local
plane. `Save Profile` updates the profile only. RealMan controller TCP writes
are disabled until SDK TCP semantics are reviewed. `--expected-tool` is optional
here; it is inferred from the profile when omitted.

Left:

```bash
python hardware/rm65/utils/scripts/rm65_tool_tcp_adjust.py \
  --profile inspire \
  --arm left
```

Right:

```bash
python hardware/rm65/utils/scripts/rm65_tool_tcp_adjust.py \
  --profile inspire \
  --arm right
```

You can reopen the GUI later to modify the same arm/tool TCP. DexTeleop RM65
backends automatically read the active RealMan tool name and load the matching
`rm65_{arm}_{tool}.json` TCP profile when one exists, so retargeting does not
need a separate tool-profile argument.

### 5. Run Official Force-Sensor Calibration

This writes controller force-sensor calibration data. Run only with a clear
workspace and no external contact.

Large adjacent calibration-pose jumps are smoothed by default with segmented
MoveJ commands before each `Manual_Set_Force` point. Tune with
`--interp-max-joint-step-deg` or `--max-interpolation-segments` if needed.

Left:

```bash
python hardware/rm65/utils/scripts/rm65_official_force_sensor_calibrate.py \
  --arm left \
  --expected-tool inspire \
  --check-current \
  --move-to-first \
  --execute \
  --confirm CALIBRATE_RM65_FORCE_SENSOR
```

Right:

```bash
python hardware/rm65/utils/scripts/rm65_official_force_sensor_calibrate.py \
  --arm right \
  --expected-tool inspire \
  --check-current \
  --move-to-first \
  --execute \
  --confirm CALIBRATE_RM65_FORCE_SENSOR
```

### 6. Create Calibration Profile

Left create:

```bash
python hardware/rm65/utils/scripts/rm65_calibration_profile.py create \
  --name inspire \
  --arm left \
  --expected-tool inspire
```

Left show:

```bash
python hardware/rm65/utils/scripts/rm65_calibration_profile.py show \
  --name inspire \
  --arm left
```

Right create:

```bash
python hardware/rm65/utils/scripts/rm65_calibration_profile.py create \
  --name inspire \
  --arm right \
  --expected-tool inspire
```

Right show:

```bash
python hardware/rm65/utils/scripts/rm65_calibration_profile.py show \
  --name inspire \
  --arm right
```

## Fixed-Point Impedance Tuning

Tune `pose` first, then repeat with `joint` for the IK/QPOS path.
`--save-to-profile` writes the tested parameters back to the profile.

Left pose:

```bash
python hardware/rm65/utils/scripts/rm65_fixed_point_impedance_tune.py \
  --profile inspire \
  --arm left \
  --mode pose \
  --axes all \
  --control-hz 40 \
  --action-hz 20 \
  --duration-sec 40 \
  --linear-compliance-m-per-n 0.004 \
  --angular-axis-signs 1,1,-1 \
  --angular-compliance-rad-per-nm 0.04 \
  --hard-stop-force-n 50.0 \
  --hard-stop-torque-nm 10.0 \
  --save-to-profile \
  --execute \
  --confirm TEST_RM65_FIXED_POINT_IMPEDANCE \
  --output-json hardware/rm65/utils/results/rm65_left_inspire_fixed_pose_impedance_tune.json
```

Left joint:

```bash
python hardware/rm65/utils/scripts/rm65_fixed_point_impedance_tune.py \
  --profile inspire \
  --arm left \
  --mode joint \
  --axes all \
  --control-hz 40 \
  --action-hz 20 \
  --duration-sec 40 \
  --linear-compliance-m-per-n 0.004 \
  --angular-axis-signs 1,1,-1 \
  --angular-compliance-rad-per-nm 0.04 \
  --hard-stop-force-n 50.0 \
  --hard-stop-torque-nm 10.0 \
  --save-to-profile \
  --execute \
  --confirm TEST_RM65_FIXED_POINT_IMPEDANCE \
  --output-json hardware/rm65/utils/results/rm65_left_inspire_fixed_joint_impedance_tune.json
```

Right pose:

```bash
python hardware/rm65/utils/scripts/rm65_fixed_point_impedance_tune.py \
  --profile inspire \
  --arm right \
  --mode pose \
  --axes all \
  --control-hz 40 \
  --action-hz 20 \
  --duration-sec 40 \
  --linear-compliance-m-per-n 0.004 \
  --angular-axis-signs 1,1,-1 \
  --angular-compliance-rad-per-nm 0.04 \
  --hard-stop-force-n 50.0 \
  --hard-stop-torque-nm 10.0 \
  --save-to-profile \
  --execute \
  --confirm TEST_RM65_FIXED_POINT_IMPEDANCE \
  --output-json hardware/rm65/utils/results/rm65_right_inspire_fixed_pose_impedance_tune.json
```

Right joint:

```bash
python hardware/rm65/utils/scripts/rm65_fixed_point_impedance_tune.py \
  --profile inspire \
  --arm right \
  --mode joint \
  --axes all \
  --control-hz 40 \
  --action-hz 20 \
  --duration-sec 40 \
  --linear-compliance-m-per-n 0.004 \
  --angular-axis-signs 1,1,-1 \
  --angular-compliance-rad-per-nm 0.04 \
  --hard-stop-force-n 50.0 \
  --hard-stop-torque-nm 10.0 \
  --save-to-profile \
  --execute \
  --confirm TEST_RM65_FIXED_POINT_IMPEDANCE \
  --output-json hardware/rm65/utils/results/rm65_right_inspire_fixed_joint_impedance_tune.json
```

## Teach Trajectory And Replay

### Record Teach Trajectory

This is read-only. Drag or teach the robot manually during the recording window.

Left:

```bash
python hardware/rm65/utils/scripts/rm65_record_teach_trajectory.py \
  --arm left \
  --expected-tool inspire \
  --hz 40 \
  --duration-sec 30.0 \
  --start-delay-sec 3.0 \
  --record-force \
  --force-every 1 \
  --output-json hardware/rm65/utils/data/trajectories/rm65_left_teach_trajectory.json
```

Right:

```bash
python hardware/rm65/utils/scripts/rm65_record_teach_trajectory.py \
  --arm right \
  --expected-tool inspire \
  --hz 40 \
  --duration-sec 30.0 \
  --start-delay-sec 3.0 \
  --record-force \
  --force-every 1 \
  --output-json hardware/rm65/utils/data/trajectories/rm65_right_teach_trajectory.json
```

### Replay With POSE Impedance

Left:

```bash
python hardware/rm65/utils/scripts/rm65_replay_teach_trajectory_impedance.py \
  --arm left \
  --expected-tool inspire \
  --input-json hardware/rm65/utils/data/trajectories/rm65_left_teach_trajectory.json \
  --move-to-start \
  --start-move-hz 20 \
  --start-move-sec 6.0 \
  --start-move-hold-sec 1.5 \
  --speed-scale 0.25 \
  --control-hz 40 \
  --force-source work \
  --axes all \
  --linear-compliance-m-per-n 0.004 \
  --angular-axis-signs 1,1,-1 \
  --angular-compliance-rad-per-nm 0.04 \
  --max-linear-offset-m 0.05 \
  --max-angular-offset-rad 0.3 \
  --hard-stop-force-n 50.0 \
  --hard-stop-torque-nm 10.0 \
  --execute \
  --confirm TEST_RM65_TRAJECTORY_IMPEDANCE \
  --output-json hardware/rm65/utils/results/rm65_left_teach_trajectory_pose_impedance_replay.json
```

Right:

```bash
python hardware/rm65/utils/scripts/rm65_replay_teach_trajectory_impedance.py \
  --arm right \
  --expected-tool inspire \
  --input-json hardware/rm65/utils/data/trajectories/rm65_right_teach_trajectory.json \
  --move-to-start \
  --start-move-hz 20 \
  --start-move-sec 6.0 \
  --start-move-hold-sec 1.5 \
  --speed-scale 0.25 \
  --control-hz 40 \
  --force-source work \
  --axes all \
  --linear-compliance-m-per-n 0.004 \
  --angular-axis-signs 1,1,-1 \
  --angular-compliance-rad-per-nm 0.04 \
  --max-linear-offset-m 0.05 \
  --max-angular-offset-rad 0.3 \
  --hard-stop-force-n 50.0 \
  --hard-stop-torque-nm 10.0 \
  --execute \
  --confirm TEST_RM65_TRAJECTORY_IMPEDANCE \
  --output-json hardware/rm65/utils/results/rm65_right_teach_trajectory_pose_impedance_replay.json
```

### Replay With QPOS Impedance

Left:

```bash
python hardware/rm65/utils/scripts/rm65_replay_teach_trajectory_qpos_impedance.py \
  --arm left \
  --expected-tool inspire \
  --input-json hardware/rm65/utils/data/trajectories/rm65_left_teach_trajectory.json \
  --move-to-start \
  --start-move-hz 20 \
  --start-move-sec 6.0 \
  --start-move-hold-sec 1.5 \
  --speed-scale 0.25 \
  --control-hz 40 \
  --force-source work \
  --axes all \
  --linear-compliance-m-per-n 0.004 \
  --angular-axis-signs 1,1,-1 \
  --angular-compliance-rad-per-nm 0.04 \
  --max-linear-offset-m 0.05 \
  --max-angular-offset-rad 0.3 \
  --hard-stop-force-n 50.0 \
  --hard-stop-torque-nm 10.0 \
  --max-command-joint-step-deg 5.0 \
  --max-admittance-joint-offset-deg 25.0 \
  --execute \
  --confirm TEST_RM65_QPOS_TRAJECTORY_IMPEDANCE \
  --output-json hardware/rm65/utils/results/rm65_left_teach_trajectory_qpos_impedance_replay.json
```

Right:

```bash
python hardware/rm65/utils/scripts/rm65_replay_teach_trajectory_qpos_impedance.py \
  --arm right \
  --expected-tool inspire \
  --input-json hardware/rm65/utils/data/trajectories/rm65_right_teach_trajectory.json \
  --move-to-start \
  --start-move-hz 20 \
  --start-move-sec 6.0 \
  --start-move-hold-sec 1.5 \
  --speed-scale 0.25 \
  --control-hz 40 \
  --force-source work \
  --axes all \
  --linear-compliance-m-per-n 0.004 \
  --angular-axis-signs 1,1,-1 \
  --angular-compliance-rad-per-nm 0.04 \
  --max-linear-offset-m 0.05 \
  --max-angular-offset-rad 0.3 \
  --hard-stop-force-n 50.0 \
  --hard-stop-torque-nm 10.0 \
  --max-command-joint-step-deg 5.0 \
  --max-admittance-joint-offset-deg 25.0 \
  --execute \
  --confirm TEST_RM65_QPOS_TRAJECTORY_IMPEDANCE \
  --output-json hardware/rm65/utils/results/rm65_right_teach_trajectory_qpos_impedance_replay.json
```

## Runtime API

Profiles can be loaded directly from application code:

```python
from hardware.rm65 import RM65PoseImpedanceController

with RM65PoseImpedanceController.from_profile("inspire", mode="pose", arm="left") as ctrl:
    ctrl.run(10.0)  # no action target: hold current reference with impedance
```

For the right arm, pass `arm="right"`.

## Notes

- Keep `control_hz <= 40` unless the frequency probe proves a higher rate is
  stable on the actual robot/network.
- External wrench sources are `zero`, `work`, and `tool`; use `work` for the
  current impedance tests unless a task explicitly needs another frame.
- RZ currently uses `--angular-axis-signs 1,1,-1`.
- Use `rm65_comm_frequency_probe.py` and `rm65_pose_response_probe.py` only for
  diagnostics; they are not part of the normal calibration path.

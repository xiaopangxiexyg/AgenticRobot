# RM65 Utilities

This folder contains calibration, validation, recording, and impedance test
tools for both RM65 arms.

Activate the conda environment once before running the commands below:

```bash
conda activate <your-rm65-env>
```

All commands are expected to run from the repository root with that environment
active.

## Layout

- `scripts/`: command-line utilities.
- `data/poses/`: recorded safe joint pose sequences.
- `data/measurements/`: force measurements replayed at pose sequences.
- `data/trajectories/`: read-only taught trajectories.
- `configs/calibrations/`: named arm/tool calibration profiles.
- `results/`: calibration fits, reports, validation JSON, and visualizations.

## Data Files

- Pose sequence: `data_type="pose_sequence"`, with `poses` and
  `joint_targets_deg`.
- Force measurement dataset: `data_type="force_measurement_dataset"`, with one
  sampled force summary per pose.
- Calibration profile: `data_type="rm65_calibration_profile"`, binding one
  arm to one tool profile. It stores calibration paths, payload/CoM, the
  confirmed TCP, and tuned pose/joint impedance parameters. Left/right arms use
  separate profiles even when the tool name is the same.

## Main Commands

### Record Four Calibration Poses

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

### Execute Four Poses And Record Force

During execution, Ctrl-C/SIGTERM sends `Arm_Slow_Stop` before closing the
socket. Use `--interrupt-stop-mode fast` only when a hard stop is required.

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

### Fit, Apply, And Validate Tool Payload

`rm65_apply_tool_payload_from_fit.py` also updates the matching calibration
profile. It stores payload mass/CoM and initializes a TCP candidate. The default
candidate is the fitted CoM shifted from the current tool pose; if the CoM is
farther than 0.20 m from the current tool origin, the script warns and falls
back to `hand_base`. The `inspire` tool defaults to `hand_base` first and can be
adjusted later in the TCP GUI.

Left, one-shot chain:

```bash
python hardware/rm65/utils/scripts/rm65_force_payload_calibrate.py \
  --arm left \
  --expected-tool inspire \
  --interp-speed-multiplier 4.0 && \
python hardware/rm65/utils/scripts/rm65_apply_tool_payload_from_fit.py \
  --arm left \
  --expected-tool inspire \
  --execute \
  --confirm APPLY_RM65_TOOL_PAYLOAD
```

`rm65_apply_tool_payload_from_fit.py` will automatically load the latest fit for
the same arm/tool, so the chain above is enough as long as the calibration step
finishes successfully.


Adjust or confirm the saved TCP. The GUI shows a draggable 3D arm view and TCP
axes. Drag empty space to orbit, use the mouse wheel to zoom, and drag the TCP
point on the selected local plane. `Save Profile` saves the profile only;
RealMan controller TCP writes are disabled until SDK TCP semantics are reviewed.
`--expected-tool` is optional here; it is inferred from the profile when omitted.

```bash
python hardware/rm65/utils/scripts/rm65_tool_tcp_adjust.py \
  --profile inspire \
  --arm left
```

Right, one-shot chain:

```bash
python hardware/rm65/utils/scripts/rm65_force_payload_calibrate.py \
  --arm right \
  --expected-tool empty \
  --interp-speed-multiplier 4.0 && \
python hardware/rm65/utils/scripts/rm65_apply_tool_payload_from_fit.py \
  --arm right \
  --expected-tool empty \
  --execute \
  --confirm APPLY_RM65_TOOL_PAYLOAD
```

`rm65_apply_tool_payload_from_fit.py` will automatically load the latest fit for
the same arm/tool, so the chain above is enough as long as the calibration step
finishes successfully.


```bash
python hardware/rm65/utils/scripts/rm65_tool_tcp_adjust.py \
  --profile empty \
  --arm right
```

You can reopen the GUI any time to adjust the same arm/tool TCP. A confirmed
TCP is not overwritten by later payload-apply runs unless `--overwrite-tcp` is
passed. DexTeleop RM65 backends automatically read the current active RealMan
tool name and load the matching `rm65_{arm}_{tool}.json` TCP profile when one
exists, so retargeting does not need a separate tool-profile argument.

### Run Official Force-Sensor Calibration

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
  --expected-tool test \
  --check-current \
  --move-to-first \
  --execute \
  --confirm CALIBRATE_RM65_FORCE_SENSOR
```

### Create And Inspect Profile

Left:

```bash
python hardware/rm65/utils/scripts/rm65_calibration_profile.py create \
  --name inspire \
  --arm left \
  --expected-tool inspire
```

Right:

```bash
python hardware/rm65/utils/scripts/rm65_calibration_profile.py create \
  --name inspire \
  --arm right \
  --expected-tool inspire
```

### Fixed-Point Impedance Tuning

Start fixed-point tuning with conservative compliance. The right-arm pose
example below intentionally uses the same low linear compliance as joint mode;
large values such as `0.2` or `0.3 m/N` saturate a `0.05 m` offset after only
about `0.55 N` of filtered force and can push nearby Cartesian targets outside
the local IK solution region.

Left pose:

```bash
python hardware/rm65/utils/scripts/rm65_fixed_point_impedance_tune.py \
  --profile inspire \
  --arm left \
  --mode pose \
  --axes all \
  --control-hz 40 \
  --action-hz 20 \
  --duration-sec 60.0 \
  --linear-compliance-m-per-n 0.004 \
  --angular-axis-signs 1,1,-1 \
  --angular-compliance-rad-per-nm 0.04 \
  --hard-stop-force-n 50.0 \
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
  --duration-sec 60.0 \
  --linear-compliance-m-per-n 0.004 \
  --angular-axis-signs 1,1,-1 \
  --angular-compliance-rad-per-nm 0.04 \
  --hard-stop-force-n 50.0 \
  --save-to-profile \
  --execute \
  --confirm TEST_RM65_FIXED_POINT_IMPEDANCE \
  --output-json hardware/rm65/utils/results/rm65_left_inspire_fixed_joint_impedance_tune.json
```

Right pose:

```bash
python hardware/rm65/utils/scripts/rm65_fixed_point_impedance_tune.py \
  --profile DMTAC_L \
  --arm right \
  --mode pose \
  --axes all \
  --control-hz 40 \
  --action-hz 20 \
  --duration-sec 60.0 \
  --linear-compliance-m-per-n 0.004 \
  --angular-axis-signs 1,1,-1 \
  --angular-compliance-rad-per-nm 0.04 \
  --hard-stop-force-n 50.0 \
  --save-to-profile \
  --execute \
  --confirm TEST_RM65_FIXED_POINT_IMPEDANCE \
  --output-json hardware/rm65/utils/results/rm65_right_DMTAC_L_fixed_pose_impedance_tune.json
```

Right joint:

```bash
python hardware/rm65/utils/scripts/rm65_fixed_point_impedance_tune.py \
  --profile DMTAC_L \
  --arm right \
  --mode joint \
  --axes all \
  --control-hz 40 \
  --action-hz 20 \
  --duration-sec 60.0 \
  --linear-compliance-m-per-n 0.004 \
  --angular-axis-signs 1,1,-1 \
  --angular-compliance-rad-per-nm 0.04 \
  --hard-stop-force-n 50.0 \
  --save-to-profile \
  --execute \
  --confirm TEST_RM65_FIXED_POINT_IMPEDANCE \
  --output-json hardware/rm65/utils/results/rm65_right_DMTAC_L_fixed_joint_impedance_tune.json
```

If the run stops with `Algo_Inverse_Kinematics failed with return code 1`,
first rerun the same command with `--axes z --ik-offset-backoff-min-scale 0.1`.
The failure log will include singularity diagnostics and, when backoff succeeds,
the applied offset scale. A profile with `tcp_confirmed=false` should also be
opened in `rm65_tool_tcp_adjust.py` and checked against the active RealMan tool
frame before increasing compliance.

For axis-direction debugging, `--debug-arm-endpoint-tcp` temporarily switches the
active RealMan tool frame to a zero TCP pose `[0,0,0,0,0,0]`, so the TCP origin
and orientation are aligned with the arm endpoint. The script restores the
original active tool frame on exit and does not save this debug setting to the
profile.

### Record Teach Trajectory

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

### Replay Teach Trajectory With Impedance

Left POSE:

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

Left QPOS:

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

Right POSE:

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

Right QPOS:

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

## Force Frames

RealMan force arrays use `Fx, Fy, Fz, Mx, My, Mz`.

- `raw`: raw force sensor reading.
- `zero`: compensated external wrench in the force sensor/default force frame.
- `work`: compensated external wrench in the work frame.
- `tool`: compensated external wrench in the tool frame.

Use `work` for the current impedance tests unless a task explicitly needs
another frame.

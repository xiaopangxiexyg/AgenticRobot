# MuJoCo Real2Sim

This package provides a reusable MuJoCo layer for DexTeleop real2sim checks.
The first implementation is intentionally position-controlled only: no force
sensor, gravity compensation, impedance control, or hardware streaming.

## Goals

- Load an explicit URDF path for repeatable camera/debug work.
- Convert URDF links and joints to a temporary or user-specified MJCF.
- Promote URDF camera frames such as `left_camera_rgb`, `right_camera_rgb`, and
  `head_camera_frame` to MuJoCo render cameras.
- Support dual-arm and right-arm-only qpos control.
- Support pose target control through MuJoCo Jacobian IK.
- Support normalized gripper control for parallel gripper modules.
- Expose `MuJoCoArmBackend` as an `EffectorBackend` so existing
  `TeleopSession` pipelines can later replace real RM65 backends with MuJoCo.

## Basic Usage

Run from the repository root in the `dexteleop310` environment:

```bash
conda activate dexteleop310
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
```

Compile the explicit URDF and list arms/cameras:

```bash
python -m sim.mujoco_real2sim.run_control \
  --urdf assets/rm65_inspire_gripper_v2/rm65_dual_arm_wrist_cameras_v2_head_camera.urdf \
  --dump-mjcf-only \
  --mjcf-out sim/outputs/rm65_dual_arm_wrist_cameras_v2_head_camera.xml
```

Right-arm qpos check without rendering:

```bash
python -m sim.mujoco_real2sim.run_control \
  --urdf assets/rm65_inspire_gripper_v2/rm65_dual_arm_wrist_cameras_v2_head_camera.urdf \
  --mode right \
  --control qpos \
  --right-qpos 0 0 0 0 0 0 \
  --no-render
```

Interactive MuJoCo GUI viewer:

```bash
python -m sim.mujoco_real2sim.run_control \
  --urdf assets/rm65_inspire_gripper_v2/rm65_dual_arm_wrist_cameras_v2_head_camera.urdf \
  --mode right \
  --control qpos \
  --right-qpos 0 0 0 0 0 0 \
  --viewer \
  --no-render
```

The GUI stays open until the window is closed. To auto-close after a short
check:

```bash
python -m sim.mujoco_real2sim.run_control \
  --urdf assets/rm65_inspire_gripper_v2/rm65_dual_arm_wrist_cameras_v2_head_camera.urdf \
  --mode right \
  --control qpos \
  --right-qpos 0 0 0 0 0 0 \
  --viewer \
  --viewer-duration 5 \
  --no-render
```

## Gripper Control

Grippers are controlled through a normalized scalar:

```text
gripper_close = 0.0  # fully open
gripper_close = 1.0  # fully closed
```

The scene maps that scalar to the actual MuJoCo joint positions through a
`GripperSpec`. For the v3 single-arm EG2-4B2 model this is inferred
automatically:

```text
eg2_4b2_left_finger_joint:   0.0 -> -0.035
eg2_4b2_right_finger_joint:  0.0 -> -0.035
```

Close the v3 gripper without moving the arm:

```bash
python -m sim.mujoco_real2sim.run_control \
  --urdf assets/rm65_inspire_gripper_v3/rm65_arm_wrist_camera_inspire_gripper_v3.urdf \
  --mode right \
  --control none \
  --gripper-close 1.0 \
  --no-render
```

Open the MuJoCo GUI and animate the gripper:

```bash
python -m sim.mujoco_real2sim.run_control \
  --urdf assets/rm65_inspire_gripper_v3/rm65_arm_wrist_camera_inspire_gripper_v3.urdf \
  --mode right \
  --control none \
  --viewer \
  --animate-gripper \
  --gripper-period 3 \
  --no-render
```

When a scene has multiple grippers, pass a named command:

```bash
--gripper-close eg2_4b2=0.5
```

For a new gripper module, define a `GripperSpec` with the module's joint names,
open qpos, and closed qpos. The rest of the control path can keep using the
same normalized close command.

Right-arm pose IK and camera recording:

```bash
python -m sim.mujoco_real2sim.run_control \
  --urdf assets/rm65_inspire_gripper_v2/rm65_dual_arm_wrist_cameras_v2_head_camera.urdf \
  --mode right \
  --control pose-ik \
  --right-pose 0.30 0.20 0.35 0 0 0 1 \
  --record \
  --steps 1
```

For headless rendering, set one of these before running:

```bash
export MUJOCO_GL=egl
# or
export MUJOCO_GL=osmesa
```

## Dataset QPOS Replay

Replay recorded robot joint positions from a LeRobot-style dataset:

```bash
python -m sim.mujoco_real2sim.replay.runner \
  --dataset-root datasets/2026-07-03_vive_rm65_real_teleop \
  --episode 0 \
  --urdf assets/rm65_inspire_gripper_v2/rm65_dual_arm_wrist_cameras_v2_head_camera.urdf \
  --mode dual \
  --qpos-source auto \
  --no-render
```

During dataset replay, camera intrinsics are loaded from
`<dataset-root>/meta/devices/realsense.json` when available. These values
override empty registry intrinsics, so the MuJoCo camera `fovy` matches the
recorded RealSense profile instead of falling back to `--fovy`.

Dataset replay also replays gripper motion when the URDF exposes a detected
`GripperSpec`. The runner reads the real collected gripper columns in this
priority order:

```text
action.gripper.<side>.target
observation.gripper.<side>.state
observation.gripper.<side>.measured
observation.gripper.<side>.commanded
action.replay.<side>_hand_joint_target[gripper_command_dim]
observation.hand.<side>.joint_pos[gripper_command_dim]
```

Replay the right-arm data onto the v3 EG2-4B2 gripper. This uses the dataset's
right-side gripper scalar and maps it to `eg2_4b2_left/right_finger_joint`:

```bash
python -m sim.mujoco_real2sim.replay.runner \
  --dataset-root datasets/2026-07-05_vive_rm65_real_teleop \
  --episode 0 \
  --urdf assets/rm65_inspire_gripper_v3/rm65_arm_wrist_camera_inspire_gripper_v3.urdf \
  --mode right \
  --qpos-source auto \
  --gripper-source auto \
  --gripper-side right \
  --start-frame 82 \
  --max-frames 120 \
  --viewer \
  --no-render
```

Use `--disable-gripper-replay` to hold the simulated gripper fixed, or
`--require-gripper-replay` to fail fast if a selected frame has no finite
gripper value. When replaying from a 6-wide hand vector, set
`--gripper-command-dim`; the DM-Tac gripper recordings use dimension `1`.

Right-arm-only replay keeps the left arm fixed at the first recorded frame:

```bash
python -m sim.mujoco_real2sim.replay.runner \
  --dataset-root datasets/2026-07-03_vive_rm65_real_teleop \
  --episode 0 \
  --urdf assets/rm65_inspire_gripper_v2/rm65_dual_arm_wrist_cameras_v2_head_camera.urdf \
  --mode right \
  --qpos-source auto \
  --no-render
```

Open the MuJoCo GUI while replaying:

```bash
python -m sim.mujoco_real2sim.replay.runner \
  --dataset-root datasets/2026-07-03_vive_rm65_real_teleop \
  --episode 0 \
  --urdf assets/rm65_inspire_gripper_v2/rm65_dual_arm_wrist_cameras_v2_head_camera.urdf \
  --mode dual \
  --viewer \
  --no-render
```

Inspect a wrist-camera mask/proxy issue at a specific dataset frame. This keeps
the viewer open after loading the selected frame and highlights proxy boxes:

```bash
python -m sim.mujoco_real2sim.replay.runner \
  --dataset-root datasets/2026-07-04_vive_rm65_real_teleop \
  --episode 0 \
  --urdf assets/rm65_inspire_gripper_v2/rm65_dual_arm_wrist_cameras_v2.urdf \
  --mode dual \
  --qpos-source auto \
  --start-frame 75 \
  --max-frames 1 \
  --viewer \
  --no-render \
  --hold-viewer-at-end \
  --debug-proxy-visuals \
  --mjcf-out /tmp/rm65_proxy_debug.xml
```

The generated MJCF keeps the URDF proxy bodies and places highlighted proxy
geoms in MuJoCo geom group 2. Group 2 is visible by default and can still be
toggled from the viewer's right UI. Use `--start-frame <seconds * fps>` to
inspect another timestamp.

Record all discovered URDF camera views during replay:

```bash
export MUJOCO_GL=egl
python -m sim.mujoco_real2sim.replay.runner \
  --dataset-root datasets/2026-07-03_vive_rm65_real_teleop \
  --episode 0 \
  --urdf assets/rm65_inspire_gripper_v2/rm65_dual_arm_wrist_cameras_v2_head_camera.urdf \
  --mode dual \
  --record \
  --record-dir sim/outputs
```

Export replayed sim cameras as a LeRobot-style dataset that the replay GUI can
load directly:

```bash
export MUJOCO_GL=egl
python -m sim.mujoco_real2sim.replay.runner \
  --dataset-root datasets/2026-07-03_vive_rm65_real_teleop \
  --episode 0 \
  --urdf assets/rm65_inspire_gripper_v2/rm65_dual_arm_wrist_cameras_v2_head_camera.urdf \
  --mode dual \
  --qpos-source auto \
  --export-dataset datasets/2026-07-03_mujoco_real2sim_replay
```

The exported dataset has the same viewer-facing layout as collected episodes:

```text
meta/info.json
meta/episodes.jsonl
meta/episodes_stats.jsonl
data/chunk-000/episode_000000.parquet
videos/observation.images.front/chunk-000/episode_000000.mp4
videos/observation.images.left_wrist/chunk-000/episode_000000.mp4
videos/observation.images.right_wrist/chunk-000/episode_000000.mp4
```

Open it with:

```bash
python -m teleop_runtime.replay.web_gui \
  --dataset-root datasets/2026-07-03_mujoco_real2sim_replay \
  --episode 0
```

`--qpos-source auto` tries `observation.arm.<side>.actual_qpos`, then
`current_qpos`, then replay/action target columns. Use `--start-frame` and
`--max-frames` for short debug windows. `--export-dataset` records rendered
camera videos, so do not combine it with `--no-render`.

## Teleop Integration Direction

The reusable path is:

```text
existing ObservationSource
  -> existing Retargeter
  -> TeleopCommand / EffectorCommand
  -> MuJoCoArmBackend
  -> shared MujocoScene
  -> MuJoCo cameras / recorder
```

This keeps the real robot code and the simulation execution code separate while
sharing the high-level teleoperation semantics.

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
  --urdf assets/rm65_inspire_gripper_v1/rm65_dual_arm_gripper_v1_head_camera.urdf \
  --dump-mjcf-only \
  --mjcf-out sim/outputs/rm65_dual_arm_gripper_v1_head_camera.xml
```

Right-arm qpos check without rendering:

```bash
python -m sim.mujoco_real2sim.run_control \
  --urdf assets/rm65_inspire_gripper_v1/rm65_dual_arm_gripper_v1_head_camera.urdf \
  --mode right \
  --control qpos \
  --right-qpos 0 0 0 0 0 0 \
  --no-render
```

Interactive MuJoCo GUI viewer:

```bash
python -m sim.mujoco_real2sim.run_control \
  --urdf assets/rm65_inspire_gripper_v1/rm65_dual_arm_gripper_v1_head_camera.urdf \
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
  --urdf assets/rm65_inspire_gripper_v1/rm65_dual_arm_gripper_v1_head_camera.urdf \
  --mode right \
  --control qpos \
  --right-qpos 0 0 0 0 0 0 \
  --viewer \
  --viewer-duration 5 \
  --no-render
```

Right-arm pose IK and camera recording:

```bash
python -m sim.mujoco_real2sim.run_control \
  --urdf assets/rm65_inspire_gripper_v1/rm65_dual_arm_gripper_v1_head_camera.urdf \
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

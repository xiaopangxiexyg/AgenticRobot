# MuJoCo Real2Sim

MuJoCo is the canonical task-physics backend for this project. SAPIEN remains
the teleop/debug preview. Cross-domain action, observation, timing, and scene
semantics are versioned in `sim/contracts/`, with the first physical profile at
`sim/configs/rm65_bimanual_physics_v1.json`. See `sim/README.md` for ownership
and current limitations.

This package provides a reusable MuJoCo layer for DexTeleop real2sim checks.
The first implementation is intentionally position-controlled only: no force
sensor, gravity compensation, impedance control, or hardware streaming.

The existing dataset replay remains an explicit kinematic diagnostic path. The
physical profile additionally compiles URDF collision geoms, contact
group/masks, gravity, ground, and a freejoint debug object. Full articulated
robot actuator stepping still waits for the 150 Hz target/actuator model; do
not call a passive full-scene `mj_step` result a dynamic robot rollout.

Compile the canonical physical profile:

```bash
python -m sim.mujoco.run_control \
  --contract sim/configs/rm65_bimanual_physics_v1.json \
  --dump-mjcf-only \
  --mjcf-out /tmp/rm65_bimanual_physics_v1.xml
```

## v3 RM65 CAD Colors

The v3 RM65 arm visual meshes are multi-material OBJ assets: their CAD colors
live in adjacent `.mtl` files as `Kd` diffuse colors and are assigned per face
with OBJ `usemtl` groups. During visual conversion, `sim.mujoco` now splits a
complete OBJ/MTL asset into visual-only mesh partitions and writes one MuJoCo
geom per material color. This preserves the blue/yellow/gray/white details of
the arm without changing URDF collision meshes or contact behavior.

The derived partition OBJ files are cached under
`/tmp/dexteleop_mujoco_obj_materials_v1/<source-hash>/`. They are regenerated
when the source OBJ or referenced MTL changes and are safe to delete between
runs. The conversion supports the current v3 diffuse-color assets (`Kd` and
`d`/`Tr` alpha). OBJ/MTL texture maps and PBR properties are not represented;
an asset that cannot be parsed completely falls back to the prior single-geom
URDF color/default-gray behavior rather than producing a partial color model.

For example, compile the latest v3 dual-arm MJCF for inspection:

```bash
python -m sim.mujoco.run_control \
  --urdf assets/rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera.urdf \
  --dump-mjcf-only \
  --mjcf-out /tmp/rm65_v3_colored.xml
```

## Isolated Cube-Drop GUI

Run the physical smoke test without loading either articulated arm. The
viewer model contains only the profile's ground, the `table` body from the
profile URDF, and the profile's `debug_cube` free body:

```bash
python -m sim.mujoco.diagnostics.cube_drop \
  --contract sim/configs/rm65_bimanual_physics_v1.json \
  --repeat-every 3
```

Each cycle holds the initial pose for one second and then advances MuJoCo at
real time. Omit `--repeat-every 3` for a single drop that remains at rest on
the table, or add `--duration 10` for an automatically closing check. The
command deliberately does not load or step the passive RM65 joints.

This is an interactive GLFW viewer, so Linux requires `DISPLAY` or
`WAYLAND_DISPLAY`. `MUJOCO_GL=egl` and `MUJOCO_GL=osmesa` are offscreen render
options and do not provide an interactive desktop window.

## 150 Hz Direct-Qpos Debug Execution

`sim.mujoco.control.DirectQposDebugController` is the temporary execution
bridge for building the first cube task before real 150 Hz actuator data is
available. It consumes an already-planned, complete dual-arm qpos command at
150 Hz, maps normalized gripper close values to finger qpos, and re-applies
the held command at every 1000 Hz physics tick.

It deliberately does not perform pose IK, qpos interpolation, PD control,
torque control, or target tracking. Use the existing `robot_motion` canonical
planner/interpolator upstream to construct the 150 Hz commands. Its manifest
metadata is always `execution_mode=kinematic_direct_qpos_debug` and
`robot_dynamics=not_simulated`.

```python
from sim.contracts import TimingSpec
from sim.mujoco.control import DirectQposCommand, DirectQposDebugController

controller = DirectQposDebugController(
    scene,
    TimingSpec(command_hz=150.0, physics_hz=1000.0),
)
controller.reset(
    DirectQposCommand(
        sequence=0,
        qpos_by_side={"left": left_home_qpos, "right": right_home_qpos},
        gripper_close_by_name={"<detected_gripper_name>": 0.0},
        source="scripted_cube_task",
    )
)
while not done:
    command = next_150hz_command() if controller.command_due else None
    audit_tick = controller.step_physics(command)
```

The direct-qpos bridge is suitable for task reset, observation, timing,
trajectory, media/export and contact diagnostics. It is not evidence of
physical grasp stability, contact force validity, actuator tracking, or
sim2real transfer. A future `debug_attachment` grasp helper must be explicitly
recorded as assisted grasp rather than physical-contact success.

## Cube-Lift RL smoke environment

After installing `gymnasium` from `README_ENV.md`, run the minimal headless
environment with the canonical physical profile's red 3 cm cube at the table
center:

```bash
PYTHONPATH=. python -m sim.mujoco.diagnostics.cube_lift \
  --task-config sim/configs/tasks/rm65_right_cube_lift_rl_debug_v0.json \
  --seed 7 \
  --steps 3
```

The CLI only drives `CubeLiftEnv`; it does not run an independent control loop.
It reports the 31-D privileged observation contract, 15→150→1000 Hz scheduler
state, cube pose/contact, and `kinematic_direct_qpos_debug` provenance. A zero
action is expected to leave the cube on the table; successfully lifting it
still requires a separately verified pure-contact qpos trace before training.

To visualise the same environment and random bounded right-arm/gripper actions
in the MuJoCo GUI (run from a desktop session with `DISPLAY` or Wayland):

```bash
PYTHONPATH=. python -m sim.mujoco.diagnostics.cube_lift \
  --viewer \
  --random-actions \
  --random-action-scale 0.25
```

Close the viewer window to stop. Add `--steps 300` for a finite 20-second-ish
run, or `--viewer-duration 20` for an exact wall-clock limit. The viewer shows
the task's direct-qpos debug dynamics only; its random motion is not valid
real-robot motion and must not be replayed on hardware.

The canonical Python namespace is `sim.mujoco`. The former package namespace
has been removed rather than kept as a second implementation or compatibility
shim. Use the imports and `python -m` entrypoints shown below.

Existing replay exports may contain `mujoco_real2sim_replay` as a task ID or
inside the dataset format marker. Those values are stable on-disk metadata,
not Python import paths. They remain unchanged so existing datasets and
downstream readers keep working across this package rename.

## Goals

- Load an explicit URDF path for repeatable camera/debug work.
- Convert URDF links and joints to a temporary or user-specified MJCF.
- Promote URDF camera frames such as `left_camera_rgb`, `right_camera_rgb`, and
  `head_camera_frame` to MuJoCo render cameras.
- Support dual-arm and right-arm-only qpos control.
- Route pose targets through the coupled `robot_motion.rm65` planner; MuJoCo
  only receives its collision-checked joint targets.
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
python -m sim.mujoco.run_control \
  --urdf assets/rm65_inspire_gripper_v2/rm65_dual_arm_wrist_cameras_v2_head_camera.urdf \
  --dump-mjcf-only \
  --mjcf-out sim/outputs/rm65_dual_arm_wrist_cameras_v2_head_camera.xml
```

Right-arm qpos check without rendering:

```bash
python -m sim.mujoco.run_control \
  --urdf assets/rm65_inspire_gripper_v2/rm65_dual_arm_wrist_cameras_v2_head_camera.urdf \
  --mode right \
  --control qpos \
  --right-qpos 0 0 0 0 0 0 \
  --no-render
```

Interactive MuJoCo GUI viewer:

```bash
python -m sim.mujoco.run_control \
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
python -m sim.mujoco.run_control \
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
python -m sim.mujoco.run_control \
  --urdf assets/rm65_inspire_gripper_v3/rm65_arm_wrist_camera_inspire_gripper_v3.urdf \
  --mode right \
  --control none \
  --gripper-close 1.0 \
  --no-render
```

Open the MuJoCo GUI and animate the gripper:

```bash
python -m sim.mujoco.run_control \
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

One-shot right-arm pose planning and camera recording uses the same coupled
RM65 planner as the real position-control path. Even for a right-only target,
the planner requires both tool profiles and keeps the left arm in the complete
collision model. Poses are `tool_tcp` targets in the corresponding arm-base
frame:

```bash
python -m sim.mujoco.run_control \
  --contract sim/configs/rm65_bimanual_physics_v1.json \
  --mode right \
  --control pose-ik \
  --left-tool-profile position_endpoint \
  --right-tool-profile position_endpoint \
  --right-pose 0.30 0.20 0.35 0 0 0 1 \
  --record \
  --steps 1
```

There is deliberately no implicit TCP profile: selecting one silently would
make sim and real control different frames. This command plans one reference
and writes the accepted pair of qpos atomically. It is not yet the continuous
75 Hz -> 150 Hz actuator loop.

The old MuJoCo-Jacobian DLS implementation lives under
`sim.mujoco.diagnostics.jacobian_ik`. It is a model/Jacobian diagnostic only;
the backend and `pose-ik` command never fall back to it.

For headless rendering, set one of these before running:

```bash
export MUJOCO_GL=egl
# or
export MUJOCO_GL=osmesa
```

## Dataset QPOS Replay

Replay recorded robot joint positions from a LeRobot-style dataset:

```bash
python -m sim.mujoco.replay.runner \
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

For visual comparison against raw RealSense RGB videos, `--distort-rgb-to-real-raw`
uses the source dataset's recorded color distortion model. The runner first
sets the render `fovy` from the selected source pinhole model, then remaps RGB
frames back into the raw RealSense distortion. This is RGB-only for
inspection/export; MuJoCo metric depth export remains in the original render
camera frame. `--rgb-distortion-source-k undistort-alpha0` is the default; use
`undistort-alpha1` for a looser OpenCV undistorted source, or `mujoco` to keep
the original MJCF `fovy` and only apply remapping.

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
python -m sim.mujoco.replay.runner \
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
python -m sim.mujoco.replay.runner \
  --dataset-root datasets/2026-07-03_vive_rm65_real_teleop \
  --episode 0 \
  --urdf assets/rm65_inspire_gripper_v2/rm65_dual_arm_wrist_cameras_v2_head_camera.urdf \
  --mode right \
  --qpos-source auto \
  --no-render
```

Open the MuJoCo GUI while replaying:

```bash
python -m sim.mujoco.replay.runner \
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
python -m sim.mujoco.replay.runner \
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
python -m sim.mujoco.replay.runner \
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
python -m sim.mujoco.replay.runner \
  --dataset-root datasets/2026-07-03_vive_rm65_real_teleop \
  --episode 0 \
  --urdf assets/rm65_inspire_gripper_v2/rm65_dual_arm_wrist_cameras_v2_head_camera.urdf \
  --mode dual \
  --qpos-source auto \
  --export-dataset datasets/2026-07-03_mujoco_replay
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
  --dataset-root datasets/2026-07-03_mujoco_replay \
  --episode 0
```

`--qpos-source auto` tries `observation.arm.<side>.actual_qpos`, then
`current_qpos`, then replay/action target columns. Use `--start-frame` and
`--max-frames` for short debug windows. `--export-dataset` records rendered
camera videos, so do not combine it with `--no-render`.

### Native v2.2 sim export (RM65 dual DM-Tac grippers)

For new real2sim/sim2real training datasets, use the independent native v2.2
writer rather than the legacy `--export-dataset` path. It reads the same
real-collection layout
`teleop_runtime/configs/policy_io_layouts/dual_arm_bimanual_dmtac_gripper_v22.json`,
so both sides are one continuous `gripper_close` channel with `0=open` and
`1=closed`.

```bash
export MUJOCO_GL=egl
python -m sim.mujoco.replay.runner \
  --dataset-root <real_dataset_root> \
  --episode 0 \
  --urdf assets/rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera.urdf \
  --mode dual \
  --qpos-source auto \
  --require-gripper-replay \
  --export-dataset-v22 datasets/<sim_dataset_name>
```

This path writes model-facing `action` / `observation.state` directly from
`PolicyIOLayout` v2.2 (16 continuous dimensions), including the canonical
layout and hash in `meta/info.json`, typed fixed-size Parquet columns, and
per-frame source provenance. It also writes `meta/modality_validity.json` and
episode provenance declaring `force_torque` and `tactile` as unavailable;
they are not represented by fabricated zero-valued rows. Wrist depth and
pointcloud export are intentionally not accepted by this new path yet, rather
than falling back to the legacy writer.

## Teleop Integration Direction

The reusable position-control path is:

```text
RobotAction tool_tcp pose
  -> timestamp pose resampling
  -> robot_motion.rm65 coupled IK / collision / limits / hold
  -> joint TeleopCommand
  -> shared 75 Hz -> 150 Hz joint interpolation (pending sim actuator wiring)
  -> MuJoCoArmBackend joint sink
  -> shared MujocoScene
  -> MuJoCo physics feedback / cameras / recorder
```

`MuJoCoArmBackend` therefore rejects per-arm `cartesian_pose` commands. A
per-effector backend cannot see both arm targets and must not run independent
single-arm IK. Tool-TCP feedback is reported only when its side-specific tool
profile was supplied; `arm_link6` is always reported separately as the arm
endpoint.

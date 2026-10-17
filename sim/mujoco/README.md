# MuJoCo Real2Sim

> 本页维护 MuJoCo 模块参数和局部示例。当前 canonical 仿真命令统一从
> [`docs/sim/README.md`](../../docs/sim/README.md) 复制。

MuJoCo is the canonical task-physics backend for this project. SAPIEN references
describe a frozen retained legacy preview only; they are not a current backend,
development target, or validation source. Cross-domain action, observation,
timing, and scene semantics are versioned in `sim/contracts/`, with the first
physical profile at `sim/configs/rm65_bimanual_physics_v1.json`. See
`sim/README.md` for ownership and current limitations.

This package provides a reusable MuJoCo layer for DexTeleop real2sim checks.
The first implementation is intentionally position-controlled only: no force
sensor, gravity compensation, impedance control, or hardware streaming.

The existing dataset replay remains an explicit kinematic diagnostic path. The
physical profile additionally compiles URDF collision geoms, contact
group/masks, gravity, ground, a freejoint debug object, and opt-in RM65
position actuators. The VIVE position path uses the same 15 Hz input, 60 Hz
qref/IK, and shared 120 Hz trajectory/follow-control semantics as the real
chain. Only its final sender/plant is replaced by MuJoCo.

Compile the canonical physical profile:

```bash
python -m sim.mujoco.run_control \
  --contract sim/configs/rm65_bimanual_physics_v1.json \
  --dump-mjcf-only \
  --mjcf-out /tmp/rm65_bimanual_physics_v1.xml
```

## Real-aligned VIVE position-teleop debug

The shared position path can be exercised without a real RM65 through the
dedicated debug composition root below. It receives VIVE poses, runs the same
collision-aware target-to-qref solver at 60 Hz, plans accepted qref to qcmd at
120 Hz, and holds the accepted command through the 1000 Hz MuJoCo bridge. The
tracker source contract remains 15 Hz. Tool profiles are mandatory so the
solver cannot silently substitute a flange/identity TCP.

```bash
PYTHONPATH=. python -m teleop_runtime.debug.run_vive_mujoco_teleop \
  --left-tool-profile position_endpoint \
  --right-tool-profile position_endpoint \
  --viewer
```

Keep both trackers steady at the desired runtime home and press Enter. Add
`--auto-home` for a non-interactive run. To compare the retained 1.0/1.0 and
1.0/0.2 IK objectives, use the default command above and then add
`--position-ik-orientation-cost 0.2`.

This entry exposes only the current 15/60/120 position-control chain. MuJoCo
owns task physics, while `--viewer` opens a passive GUI that does not create a
second step loop. The final 120 Hz sender replaces the real backend's
`Movej_CANFD` call: SDK acceptance is modeled separately from measured joint
state, the fitted per-controller transport delay is applied before the MuJoCo
position target changes, and the robot moves through torque-limited MuJoCo
actuators. Old 150 Hz, RL, and prescribed-qpos diagnostics are not selectable
from this runtime entry.

## Deployable shared-control fixture

### Committed demonstration export

`sim.mujoco.adapters.demonstration_trace` is a read-only bridge from one
committed ControlTrace decision span into the backend-neutral
`DemonstrationTrajectory` contract. It exports the canonical `safe_action`
pose and explicitly configured hand representation, not the pre-safety
selection, resolved qcmd, or measured qpos. Every selected decision must have
a normal execution directive, executable shared-safety result, fully accepted
arm/hand dispatch receipt, and a later MuJoCo backend measurement. The caller
must also supply hash-bound task-success evidence for the exact first/last
decision IDs; the bridge does not encode Cube-Lift or grasp-phase logic.

The trace execution effector and exported demonstration component are separate
contracts. For the current DM-Tac compatibility carrier, accepted execution is
proved under `right_hand`, while export selects only slot 0 (`gripper_close`)
and names the editable component `right_gripper`; the remaining five padded
slots never become demonstration dimensions. The downstream representation
adapter then projects this scalar through the scene-declared open/closed joint
targets into the native two-joint `right_gripper` payload.

This is an import/export and offline preflight boundary. It does not step
MuJoCo or admit the trajectory back into the authoritative runtime. A passing
zero-edit preflight proves only nominal edit/interpolation identity; physical
replay and collision/task success remain downstream gates.

`sim.mujoco.deploy` is the first backend binding for the shared authoritative
control core. It is a library fixture, not a new production entrypoint. The
single-policy profile samples measured MuJoCo arm qpos into a causal
`ObservationCutRecord/Bundle`, admits an absolute bimanual-qpos proposal, and
runs the common immutable snapshot, ActionArbiter publication, persistence,
shared-safety, resolver, one-shot dispatch permit, position rig, receipt, and
measurement lineage. The ActionArbiter is capped at 60 Hz; reference, command,
and physics clocks remain independent.

Execution observations remain default-off. A plan may retain the historical
`joint_position` active-qref/measured-qpos stream or explicitly require
`cartesian_pose`. The latter is produced by the MuJoCo owner as
`FK(active qref)` and `FK(measured qpos)` using a private `MjData` scratch
buffer. The Rig captures qref, measured qpos, model qpos, and sample time under
one owner lock; FK never writes the live simulator. Pose schema/frame identity,
the raw qpos observation, and the model/kinematics contract stay in lineage.

The optional trajectory-residual publication fixture is likewise default-off.
Its Control Station adapter may admit one future composed proposal, but it does
not call the MuJoCo control loop. The existing runtime remains the only owner
that selects the proposal at its effective time, resolves the component target,
activates qref, steps physics, and publishes the resulting FK pose observations.
Consequently publication success is not reported as target execution, and no
Reference/Measurement exists until the lower loop has actually processed the
proposal.

This scoped M5.5a/b path has no `teleop_runtime` dependency and does not copy
`run_policy.py`. In addition to deterministic single-policy tests, an explicitly
enabled canonical replay source may become sim authority only when its source
contract is verified-compatible with the exact immutable MuJoCo target contract.
Binding eligibility and typed execution authorization are receiver-admitted and
frozen into the arbitration input; missing, denied, or wrong-target evidence is
rejected before backend publication. This does not authorize real replay.
Slow/fast topology, hold/fault cancellation, formal sim episode sealing, and
validated sim2real dynamics are later milestones.

### Realtime trace-consumption rule

Reference receivers and execution rigs expose append-only history snapshots for
inspection and final capture. A realtime caller must treat those snapshots as a
log and keep one monotonic cursor per stream. Each control cut may project only
the newly appended suffix into ControlTrace.

Do not rebuild every historical fact and then deduplicate by `fact_id`: the
projection and hashing cost has already been paid before deduplication, causing
control latency to grow with episode length. A successful dispatch performs one
post-dispatch incremental trace sync; deterministic physics frames requested by
that call are included in the same sync. Background physics frames are consumed
at the next cut or at runtime close. History rewind below a published cursor is
a contract violation and must fail visibly.

Regression tests for a new execution-observation stream must verify both:

- repeated sync with no new records invokes no projector;
- after a long history, appending one record invokes its projector exactly once.

For policy outputs that contain action chunks, call
`MujocoDeployRuntime.submit_policy_action_chunk(...)`. Its
`action_chunk_execution_steps` argument (or the matching
`MujocoDeployConfig` field) admits only the first `X` actions and assigns them
consecutive decision timestamps. If the next chunk is submitted without an
explicit start time, it begins at the previous prefix's end, so an asynchronous
producer can prepare it ahead of execution. `None` keeps the existing full
chunk/single-action behavior.

For model deployment, `MujocoDeploymentObservationAdapter` is the read-only
scene bridge: it samples explicit TCP body poses, arm qpos, gripper closure,
and optional renderer images into the common `DeploymentObservation` contract.
`deployments.openpi.client.OpenPISimAdapter` can consume that observation and
returns a decoded chunk; neither adapter steps physics or submits a target.
`MujocoPolicyRollout` is the model-neutral sim-only composition seam: it
captures one observation cut, calls any decoded policy adapter, adapts every
keyframe through `DeployRobotActionAdapter`, and submits the whole chunk once
to `MujocoDeployRuntime`. A failed keyframe rejects the whole chunk and, when
the runtime exposes the gate transition, requests a fault/hold. It does not
construct an OpenPI client or choose an IK/pose-frame profile; those remain
explicit caller configuration. The existing generic action adapter and this
runtime's explicit target profile therefore remain the only execution path.
The default catalog is arms-only; a policy chunk carrying hand keyframes is
accepted only when the caller explicitly constructs the DMTAC compound target
profile (including its shared safety callback), so gripper commands cannot be
silently dropped or routed through an arms-only runtime.

Formal composer-owned compound motion has an additional default-off gate:
`ConservativeCompoundAuthorityProfile` must bind the exact composition spec,
component rule set, formal strategy, complete component catalog, and every
active selection group. The composer must be the only authorized proposal
producer, while raw policy and replay authority stay disabled. Sim-HIL enables
the generic source-eligibility store: source state and action inputs are read
from one causal cut, and a missing active contributor yields no new proposal
for that period. Station does not mutate the execution gate for that condition;
the lower persistence policy remains the sole owner of keep/expiry/HOLD
behavior. Failures that require operator/lower-runtime repair request latched
domain HOLD.
Canonical safety, pose resolution, receiver admission, and all backend
receipts are fail-closed: any failure cancels every pending/active compound
group, and the receiver activates arms and required hands only as one guarded
generation bundle. This profile currently remains MuJoCo + bilateral DM-Tac
q6 only and does not authorize real hardware.

The RM65-specific accepted-qcmd/delay/filter state machine now lives in
`sim/rm65/actuation`; this package only maps its target to MuJoCo position
actuators and reads measured qpos. The historical v3 free-motion profile used
ten paired command/UDP traces and conservatively refit only left J6. The
canonical v4 profile extends that history to fourteen valid pairs and refits
bilateral J5-J6 phase terms while freezing J1-J4. Position
gains, damping, torque limits, armature, force/contact behavior, and payload
dependence remain provisional, and v4 still needs a new untouched held-out
capture. This path is therefore a physical rollout model, not yet a validated
RM65 servo twin.

The same lower path is available to batched Cube-Lift through the main task
config below; the implementation lives in `sim.mujoco`, not in a training
package:

```text
sim/configs/tasks/rm65_right_cube_lift_rl_actuated_v1.json
sim/configs/tasks/rm65_right_cube_lift_rl_actuated_v2.json
```

Its model artifact cache fingerprints the full actuation profile and compiles
stiff servos with MuJoCo `implicitfast`. The policy observes measured arm qpos,
while compact task info reports mean/max target-tracking error. V1 retains a
prescribed gripper for regression. V2 is the default training task and compiles
four additional DM-Tac finger actuators; its observation and latch use measured
gripper closure after the identified delay and acceleration-limited response.
The state-only normal v2 profile preserves that free-motion response and adds a
separately hashed simulation-only post-settle contact-friction override.

The current temporary PPO default is the state-only
`sim/configs/tasks/rm65_right_cube_lift_rl_normal_v2.json`. Camera-conditioned
policy variants remain explicit compatibility configs while RGB training is
paused:

```text
sim/configs/tasks/rm65_right_cube_lift_rl_normal_rgb_v2.json
sim/configs/tasks/rm65_right_cube_lift_rl_privileged_expert_rgb_v2.json
```

They append only the active `right_wrist` RGB camera as a fixed 3x48x48 CHW
input; the pose action and fitted arm/gripper actuators are unchanged. The
earlier 3x32x32 RGB configs remain available as compatibility baselines.

The `48x48` RGB size in these PPO configs is a compact learning input, not the
OpenPI camera contract. For the current sim-only OpenPI smoke path, the three
model camera slots may also be rendered at `48x48`; OpenPI accepts that payload
and resizes it server-side to `224x224`. This proves transport, model-adapter,
action, and MuJoCo execution compatibility only. It does not prove visual
sim-real alignment with the real RM65 `640x480` camera streams, whose 4:3
content is resized with top/bottom padding. A separate OpenPI observation
profile must use camera-contract-matched source resolution, aspect ratio, view,
and timing before making a visual-alignment claim.

If the latest Q6 assignment is available, the same command starts the
left/right Q6 readers and adds two continuous DM-Tac grippers. Explicit ports
override the assignment:

```bash
PYTHONPATH=. python -m teleop_runtime.debug.run_vive_mujoco_teleop \
  --left-tool-profile position_endpoint \
  --right-tool-profile position_endpoint \
  --left-hand-port /dev/serial/by-id/LEFT_Q6 \
  --right-hand-port /dev/serial/by-id/RIGHT_Q6 \
  --q6-config-required \
  --viewer
```

The Q6 mapping follows the current real profile: source dimension 2,
`input_min=0.1`, `input_max=0.9`, no inversion, and `0=open, 1=closed`. The
gripper transport gate is 20 Hz; this is an endpoint-device gate attached to
the 15/60/120 arm chain, not another arm-control path. The 2026-07-22 V1 profile
uses independent left/right and close/open free-motion fits, with directional
delays of roughly 0.047--0.064 s and maximum normalized rates of
1.377--1.502/s. Combined with MuJoCo servo response, this
gives approximately 0.131 s to visible motion and 0.851 s to measured full
travel from software command acceptance. The symmetric model lies between the
repeat-3 hardware close/open means of approximately 0.829/0.870 s.
Hold-last-target is preserved. The 20 N
per-finger limit, normalized torque, contact material,
short-step and preemption behavior remain explicitly provisional because the
trace had no external force gauge and did not identify those regimes. Pad
friction and `solref`/`solimp` remain explicit unset tuning interfaces in
`sim/mujoco/actuation/gripper_profile.py`.

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
  --urdf assets/rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera_4b2_cad_with_legacy_housing.urdf \
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

## DMTAC CAD 方块接触诊断（仅离线）

```bash
python -m scripts.probes.diagnose_dmtac_cube_contact \
  --output /absolute/path/to/new-contact-run \
  --cube-edge-m 0.070 --cube-mass-kg 0.08
```

该探针复用当前 DMTAC CAD、MJCF 转换和 `MujocoDmtacGripperExecutor`，分别运行
左右手的水平／顶抓隔离夹具。自由法兰通过 weld 跟随独立 mocap 驱动；方块是自由刚体，不使用物体
weld、抓取 latch、qpos 强制回放或摩擦覆盖。80 g 是可改的**仿真假设**，不是实测质量。
`--object-approach-offset-m 0.01` 可对照更浅的抓取位置，不会修改部署抓取标定。

每组保存 `fixture.xml`、启动配置 `configuration.json`、每 10 ms flush 的
`trace.jsonl` 和最终 `result.json`，中断时也可保留配置及已产生的状态。记录实际
接触位置、法向／切向力、六个关节角、联动偏差、穿透和超限；先区分初始放置干涉、
接触动力学失真、未夹住，再判断 3 cm 试抬。诊断有效性阈值为初始穿透 0.1 mm、
运动中穿透 2 mm、联动／限位偏差 0.01 rad；它们只用于该离线实验，不改变生产安全策略。
试抬还要求法兰跟踪误差及手部地面穿透不超过 2 mm，手—物相对滑移不超过 3 mm、
相对转角不超过 2°，并且最终两侧传感器仍有接触力。
明显穿透或关节变形时，即使方块被弹起，也不能当作抓取成功。
法向／切向分量按 [MuJoCo 接触坐标约定](https://mujoco.readthedocs.io/en/stable/computation/#contact)
从 `mj_contactForce` 读取，不将力向量模长误当作法向力。

`--contact-model constraint_candidate` 仅供对照：传感器—方块接触 `solref=0.002`、
`solimp=0.99`，CAD 联动 `solref=0.002`、`solimp=0.9999`，椭圆摩擦锥及
`impratio=10`；对应独立的 `dmtac_bimanual_cad_cube_contact_20261017.v1` 仿真 profile。
传感器—方块显式接触对的滑动摩擦默认从 1.0 提高到 1.2（`--sliding-friction 1.0`
可做原值对照）；不改变扭转／滚动摩擦、驱动力、几何、共享 URDF 或真机 profile。
遵循 [MuJoCo 的 mocap 与防滑建模建议](https://mujoco.readthedocs.io/en/stable/modeling.html#mocap-bodies)：
直接移动 mocap 手部缺少基座速度，不适合作为试抬摩擦响应的依据；自由手部跟踪驱动能保留
物理速度。接触刚度和求解设置仍是仿真候选，不代表实测材料标定。

这不是完整机械臂路径、Gateway Skill 或 Agent 验收。旧版平行夹爪的接触测试通过，
也不能替代当前六关节 CAD／DMTAC 的动力学验收。完整诊断完成但任一试抬不通过时
退出码为 2，不能仅凭程序执行完毕宣称抓取成功。

### 可选的传感器外观覆盖

场景可通过 `scene.visual_rgba_by_link` 显式覆盖 URDF link 的可视材质。
70 mm DMTAC CAD 任务将四个传感器末端设为灰色 `[0.5, 0.5, 0.5, 1]`。
转换只改可视 geom 的 RGBA；碰撞体、质量、摩擦、关节、TCP 和共享资产不变。
未配置覆盖时保留原材质和场景序列化身份；未知 link、非有限值和越界颜色在
构建边界报错。该外观配置不代表新的抓取或携物执行能力。

## 150 Hz Direct-Qpos Debug Execution

`sim.mujoco.control.DirectQposDebugController` is the temporary execution
bridge for building the first cube task before real 150 Hz actuator data is
available. It consumes an already-planned, complete dual-arm qpos command at
150 Hz, maps normalized gripper close values to finger qpos, and re-applies
the held command at every 1000 Hz physics tick.

Physical scene profiles now pass every new prescribed target through the
shared collision-safety layer before it is written. The layer checks forbidden
robot/environment, non-gripper/object and self-collision pairs on a scratch
`MjData`, sweeps the joint-space segment, and projects an RL target back to the
last safe point. A hard penetration detected after `mj_step` is reported as an
environment failure. Collision-disabled kinematic replay remains raw and is
explicitly labelled as unavailable for collision validation.

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
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
```

Compile the explicit URDF and list arms/cameras:

```bash
python -m sim.mujoco.run_control \
  --urdf assets/rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera_4b2_cad_with_legacy_housing.urdf \
  --dump-mjcf-only \
  --mjcf-out sim/outputs/rm65_dual_arm_wrist_cameras_v3_head_camera.xml
```

Right-arm qpos check without rendering:

```bash
python -m sim.mujoco.run_control \
  --urdf assets/rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera_4b2_cad_with_legacy_housing.urdf \
  --mode right \
  --control qpos \
  --right-qpos 0 0 0 0 0 0 \
  --no-render
```

Interactive MuJoCo GUI viewer:

```bash
python -m sim.mujoco.run_control \
  --urdf assets/rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera_4b2_cad_with_legacy_housing.urdf \
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
  --urdf assets/rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera_4b2_cad_with_legacy_housing.urdf \
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

## Canonical Dataset Replay (default)

The normal dataset replay entrypoint consumes canonical `desk_absolute`
left/right tool-TCP actions (UMI 16D or real-teleop 26D), then runs the shared
collision-aware dual-arm retargeter and the physical MuJoCo actuator path:

```bash
python -m sim.mujoco.replay \
  --dataset-root <dataset-root> \
  --episode 0 \
  --viewer
```

It does not derive arm motion from recorded qpos. See
[`replay/README_UMI.md`](replay/README_UMI.md) for the action/frame contracts,
TCP target overlay, and headless options.

## Debug-only Dataset QPOS Replay

For joint-space diagnosis only, replay recorded robot joint positions from a
LeRobot-style dataset with the explicit `.runner` entrypoint:

```bash
python -m sim.mujoco.replay.runner \
  --dataset-root datasets/2026-07-03_vive_rm65_real_teleop \
  --episode 0 \
  --urdf assets/rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera_4b2_cad_with_legacy_housing.urdf \
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
  --urdf assets/rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera_4b2_cad_with_legacy_housing.urdf \
  --mode right \
  --qpos-source auto \
  --no-render
```

Open the MuJoCo GUI while replaying:

```bash
python -m sim.mujoco.replay.runner \
  --dataset-root datasets/2026-07-03_vive_rm65_real_teleop \
  --episode 0 \
  --urdf assets/rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera_4b2_cad_with_legacy_housing.urdf \
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
  --urdf assets/rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera_4b2_cad_with_legacy_housing.urdf \
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
  --urdf assets/rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera_4b2_cad_with_legacy_housing.urdf \
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
  --urdf assets/rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera_4b2_cad_with_legacy_housing.urdf \
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
  --urdf assets/rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera_4b2_cad_with_legacy_housing.urdf \
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

## Current Teleop Integration

The reusable position-control path is:

```text
VIVE target update (15 Hz)
  -> robot_motion.rm65 coupled IK / collision / limits (60 Hz qref)
  -> shared acceleration-limited qref -> qcmd planner (120 Hz)
  -> delayed, force-limited MuJoCo position actuators (1000 Hz physics)

Q6 scalar (source dim 2, 0=open, 1=closed)
  -> shared ScalarToGripperRetargeter
  -> DM-Tac threshold / 40 Hz transport gate / 0.10 s post-command delay
  -> acceleration/braking-limited, provisionally force-limited finger actuators
  -> the same shared MujocoScene and MjData owner
```

The simulation rig receives joint commands only. A per-effector backend cannot
see both arm targets and must not run independent single-arm IK. Tool-TCP
feedback is reported only when its side-specific tool profile was supplied;
`arm_link6` is always reported separately as the arm endpoint.

Canonical desk-TCP UMI data must use the separate physical replay path; it
does not fabricate qpos or write MuJoCo qpos directly. See
[`replay/README_UMI.md`](replay/README_UMI.md) for its frame contracts, control
chain, headless validation, and viewer commands.

# Sim/Real common test laboratory

`sim/sim_real_tests` contains executable acceptance scenarios for motion logic
that must behave the same in simulation and on a real robot, while never
connecting to real hardware. It is a diagnostic/test package, not a production
teleoperation entrypoint.

## Layout

```text
sim/sim_real_tests/
├── core/       # backend-neutral result and future scenario protocols
├── configs/    # versioned deterministic scenario inputs
├── rm65/       # model-only RM65 FK/IK/collision/qpos scenarios
├── mujoco/     # physics, actuator and contact scenarios
└── tests/      # pytest acceptance checks over the reusable scenarios
```

Dependency direction is strict:

```text
sim.sim_real_tests -> sim backend (optional) + robot_motion + pure contracts
```

Do not import `teleop_runtime`, vendor SDKs, USB devices, cameras, or real
network backends. Shared motion behavior missing from `robot_motion` should be
moved there first instead of copied into this directory.

## Current RM65 qpos-init scenario

The first scenario loads the measured collision-safe init qpos, generates a
small deterministic random reachable TCP pose through FK, solves that pose
back to qpos with the shared IK implementation, and plans from the previous
dual-arm qpos to the solved qpos. Every emitted qpos sample is checked against
the same strict collision proxy and hold distance used by real qpos init.

Run the lightweight structure checks in any project environment:

```bash
python -m pytest sim/sim_real_tests/tests/test_core_contracts.py -q
```

The package-local suite uses the repository-wide pytest markers, but remains
outside the default root `testpaths`. Activate the shared project environment
once from the selected worktree root:

```bash
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
```

Run its fast feedback tier while editing:

```bash
python -m pytest sim/sim_real_tests/tests -q -m "not slow_sim"
```

Run only the long physical closed-loop checks on the designated integration
worktree:

```bash
python -m pytest sim/sim_real_tests/tests -q -m slow_sim
```

Run the complete Pinocchio/Pink/MuJoCo acceptance suite before integration:

```bash
python -m pytest sim/sim_real_tests/tests -q
```

`not slow_sim` is a development feedback selector, not a replacement for the
complete suite. The slow marker is attached to exact closed-loop test
functions; it does not reduce their sample counts, physical ticks, thresholds,
or inclusion in the unfiltered command.

Git does not populate ignored robot assets in a linked worktree. Configure the
clone-shared development asset store once and use the standard worktree claim
tool documented in [`CONTRIBUTING.md`](../../CONTRIBUTING.md). For a worktree
created before that tool, run:

```bash
python scripts/dev_asset_store.py link-worktree --worktree "$PWD"
python scripts/dev_environment_doctor.py --require-development-assets
```

The ignored `assets` symlink is a compatibility mapping, not proof that the
target filesystem is read-only. Do not edit the canonical store through it.
Treat a missing or mismatched asset as a preflight failure; do not copy assets
into Git or change scenario semantics to make the test pass. Runtime asset
identity and semantic hashes remain owned by the TargetRuntime contract rather
than this development-path mechanism.

The random stream is fixed by the JSON config. Increase `sample_count` for a
longer offline soak test; do not replace deterministic CI coverage with an
unbounded random loop.

## TCP 包络的最终输出与物理响应

`test_mujoco_tcp_pose_envelope_closed_loop.py` 保留原 8° 限值，并逐个 1000 Hz
physics tick 检查实际 TCP 和最终 120 Hz qcmd 的 FK；平移仍受原 15 mm 包络限制。
IK 前的目标投影不能单独证明这些边界：关节独立整形及不同传输延迟会改变 TCP 路径。
本离线场景显式绑定同一 profile、anchor 和 epoch 的最终命令接纳，使用独立 FK 数据，
复制当前 MuJoCo 完整状态及原执行器的待交付命令、滤波状态，查询接纳命令后保持的响应。
只接纳通过检查且在有界时间内稳定的候选，必要时缩短原碰撞投影已接纳的双臂线段；
无法证明安全保持则失败并锁存，不修改真实物理状态、伺服参数或碰撞门禁。
该查询仅支持显式离线 stepping，其他控制量在预演中保持当前值；它不覆盖未来夹爪输入，
不构成连续扫掠认证或真机制动保证。场景的 `passed` 同样包含实际姿态越界的失败判定。

## Arbitrary grasp-pose reachability probe

`rm65_random_grasp_pose_reachability_v1.json` defines a bounded Cartesian box,
a PCG64 seed, and a Shoemake-uniform SO(3) sampler.  The probe emits exactly one
pose per requested sample without rejection sampling, then records exactly one
typed terminal endpoint result:

The config, generated sample, endpoint receipt, and scenario record all carry
the same `reference_frame_id` and `reference_frame_spec_hash`.  The default is
the single-arm URDF solver's model/arm-base frame.  The target translation and
the conservative workspace origin are interpreted in that exact frame; a frame
ID or spec-hash mismatch is `invalid_target`, never an unreachability proof.
Each sample also has a deterministic `target_pose_content_hash` over the
sampler ID, PCG64 seed, sample index, frame identity, and exact little-endian
float64 pose bytes.  Scenario projection verifies both that hash identity and
the target matrix before accepting an evaluator receipt.

- `reachable`: a residual-bounded endpoint IK candidate was found;
- `kinematically_unreachable`: a conservative workspace exclusion proves the
  target lies outside the configured outer envelope;
- `ik_search_exhausted`: bounded multi-start IK found no accepted candidate,
  which is not a mathematical unreachability claim;
- `collision_blocked` or `collision_search_exhausted`: injected endpoint
  collision checks proved every IK candidate blocked or exhausted their bounded
  search;
- `invalid_target` or `planner_error`: target/adapter failures were converted
  to typed terminal evidence.

The JSON loader accepts only the exact
`rm65.random_grasp_pose_reachability.v1` schema and scenario identifier; a
missing or different version is rejected rather than silently interpreted.

An unreachable or collision-blocked sample is a valid scenario observation,
not a failed scenario.  The receipt contains endpoint candidates, residuals,
rejection reasons, proof scope, and collision metadata only.  It never contains
an executable path or trajectory; the future obstacle/path planner owns that
separate receipt layer.

## Continuous MuJoCo motion to arbitrary random grasp poses

The arbitrary-pose viewer continuously interleaves one-draw random joint-FK
poses with unfiltered Cartesian positions and uniform SO(3) orientations, then
turns each admitted endpoint into a complete physical MuJoCo movement. The
target contract is unambiguous:

```text
selected_arm_base_from_actual_DMTAC_tool_TCP
```

This is not the shorter single-arm solver `tool_tcp`.  The test derives and
records the rigid transform between those frames, converts the target for the
endpoint reachability solver, and still executes and visualizes the actual
DMTAC tool TCP in the full dual-arm model.

With no explicit target, the default mode runs until the viewer is closed. It
does not rejection-sample either stream for reachability: every random pose gets
one terminal receipt, rejected targets emit zero motion, and the next sample is
still tried. By default every third sample is a raw Cartesian/SO(3) probe; set
`--raw-cartesian-probe-interval 1` to run only raw probes.

```bash
python -m sim.sim_real_tests.mujoco.view_arbitrary_grasp_pose_motion \
  --side left
```

The target axes remain visible while a rejected target holds the robot still.
The extra target marker is green for a completed motion, red only for a
conservatively proven kinematic unreachable target, orange for endpoint/path
collision blocking, yellow for bounded search exhaustion, and purple for a
system error. In particular, `ik_search_exhausted` is not mislabeled as proven
unreachable. Yellow receipts now carry a bounded diagnostic reason in the
viewer log: `bounded_position_search_exhausted` means the auxiliary
position-only search also missed the position tolerance, while
`bounded_orientation_conditioned_ik_exhausted` means position was found but
the requested exact orientation still was not. A candidate already within
5 cm and 10 degrees receives one deterministic, joint-limited trust-region
refinement; distant residuals do not receive an indiscriminate time increase.
All refined candidates still have to meet the original 3 mm / 1 degree exact
pose limits and pass the unchanged MuJoCo collision check. These diagnostics
remain bounded-search evidence, never a mathematical unreachability proof.

Use a finite headless batch to exercise the same target generation, endpoint
admission, path preflight, and physics execution without a GUI:

```bash
python -m sim.sim_real_tests.mujoco.view_arbitrary_grasp_pose_motion \
  --side left \
  --headless \
  --random-sample-count 12
```

The former single collision-safe demo remains available explicitly (the
default seed currently produces about 0.50 m translation, 85 degrees rotation,
and 38 degrees maximum joint motion):

```bash
python -m sim.sim_real_tests.mujoco.view_arbitrary_grasp_pose_motion \
  --side left \
  --single-reachable-demo
```

Supply any explicit grasp pose with either a quaternion or XYZ Euler angles.
The following is one known-reachable example in the left arm base frame:

```bash
python -m sim.sim_real_tests.mujoco.view_arbitrary_grasp_pose_motion \
  --side left \
  --target-xyz-m -0.32341645 -0.19260886 0.62870855 \
  --target-rpy-deg 22.893756 37.834623 -140.624559 \
  --linear-speed-m-s 0.12 \
  --angular-speed-deg-s 45
```

For an explicit or single-demo target, use `--headless` for identical
planning/control/physics execution without opening the viewer. A reachable
target follows a zero-endpoint-velocity and zero-endpoint-acceleration quintic
joint path; each 60 Hz Cartesian reference
is generated by full-tool FK from that path, followed only by a bounded local
TCP correction for the endpoint IK residual.  References pass through the
existing collision-aware IK, 120 Hz command planner, and 1000 Hz MuJoCo
physics chain.  The GUI replays the captured physical qpos sequence; it does
not teleport the model between requested poses.  RGB axes show the final DMTAC
TCP target.

Targets outside the conservative workspace, exhausted by bounded IK, or
blocked by the full MuJoCo endpoint collision check return their distinct typed
reachability receipt and emit zero motion frames.  Endpoint admission is not a
path proof: the direct joint path is sampled at no more than 0.1 degree maximum
joint increments against the full MuJoCo collision scene.  A blocked path is
reported as `path_collision_blocked` with the exact progress and colliding
bodies, also with zero emitted frames.  During execution the existing
collision-aware dual-arm retargeter and rig remain authoritative; any
intermediate `ik_target_unmet` remains the whole motion's terminal status even
if a later frame settles, so a broken path cannot be relabeled as success.

Static axis-aligned planning cubes can be added in the shared MuJoCo/solver
world frame. Repeat the option for multiple cubes:

```bash
python -m sim.sim_real_tests.mujoco.view_arbitrary_grasp_pose_motion \
  --side left \
  --single-reachable-demo \
  --obstacle-cube-world-m -0.3358 -0.0975 0.5000 0.12 \
  --obstacle-clearance-m 0.02
```

The translucent orange boxes are display overlays backed by the same immutable
scene hash used for endpoint filtering and planning. The planner checks the
direct joint edge first, then runs deterministic bounded BiRRT-Connect only if
that edge is blocked. A found path is shortened, rechecked against FCL at
most 0.1 degree joint increments, split into zero-velocity/zero-acceleration
quintic segments, and every segment is independently rechecked in the MuJoCo
scene before any reference is sent. `path_start_collision_blocked`,
`path_target_collision_blocked`, `path_search_exhausted`, and
`path_planner_error` remain distinct zero-motion results. Search exhaustion is
yellow evidence that the bounded search found no path, not proof that no path
exists.

The example above is a deterministic regression case: the cube blocks the
single-demo target's direct path, the bounded local search returns a detour,
and the captured MuJoCo playback completes. Search bounds are the
current/target joint corridor plus 45 degrees, intersected with physical joint
limits; this avoids full-range detours that are collision-free geometrically
but unnecessarily large and difficult for the Cartesian execution chain.

This first obstacle mode is a planning-path-guided simulation experiment, not
an execution certificate or real-motion authorization. The admitted joint
waypoints are converted by FK to Cartesian references and the existing lower
retargeter solves IK again; the added display/planning cubes are not yet
inserted into that lower IK collision model. The local terminal Cartesian
correction is therefore disabled in obstacle mode, and receipts explicitly
record `exact_planned_qpos_dispatch=false` and
`runtime_cube_constraints_in_lower_ik=false`. A strict runtime guarantee needs
either an admitted exact-qpos execution interface or the identical immutable
obstacle scene installed in the lower IK/dispatch owner.

Run the focused contract regression with the repository environment wrapper:

```bash
bash scripts/run_pytest.sh \
  sim/sim_real_tests/tests/test_mujoco_arbitrary_grasp_pose_motion.py -q
```

## Simulation-first real shadow trajectory boundary

The real integration path is intentionally split into evidence-producing
stages. It currently performs no real motion and is not published as a real
Skill:

1. the exclusive lower owner supplies one fresh dual-arm hardware qpos cut;
2. independently measured left/right gripper state and evidence complete the
   collision-relevant initial state;
3. the snapshot binds the canonical station frame, tool profiles, frame graph,
   robot, joint layout, gripper, and collision-model hashes;
4. deterministic MuJoCo is cold-started from that full snapshot with zero
   velocity, rebased arm planners, rebased gripper executors, and a fresh
   collision check;
5. one registered joint trajectory binds the exact initial snapshot, obstacle
   scene, target, planner configuration, moving side, and every timed dual-arm
   qpos point;
6. only an exact-qpos simulation execution plus a terminal HOLD receipt can
   pass replay preflight; a new hardware cut must still match identity, scene,
   freshness, arm position, and gripper position thresholds.

The preflight result always records `preflight_only=true` and
`real_motion_authorized=false`. It is not a dispatch permit. Assumed or
disconnected fallback gripper state is rejected because the hand geometry can
change both initial collision and path clearance. Endpoint, start collision,
target collision, path collision, search exhaustion, planner error, IK failure,
and incomplete tracking remain distinct simulation statuses.

The current Cartesian waypoint runner cannot yet satisfy stage 6: it re-solves
IK after FK waypoint generation and has no lower terminal-HOLD receipt. A
visually completed run is therefore translated to `execution_uncertified`, not
`completed`, and real replay preflight rejects it. This is the principal reason
the workflow has not been released as a real Skill. The current public MuJoCo
builder also checks the requested arm state once with its conservative default
open gripper before installing measured gripper geometry; it then performs a
second exact-state collision check. A default-open collision returns
`conservative_open_gripper_initial_collision_blocked` rather than silently
continuing.

Run the no-device contracts and the MuJoCo cold-start check with explicit
checkout, Python 3.10 environment, and local asset root:

```bash
cd /home/zc/HandMotion/Developing/AgenticRobot
DEXTELEOP_PYTHON=/home/zc/anaconda3/envs/dexteleop310/bin/python \
DEXTELEOP_ASSET_ROOT=/home/zc/HandMotion/Developing/AgenticRobot/assets \
bash scripts/run_pytest.sh -q \
  tests/test_real_shadow_grasp_trajectory.py \
  sim/sim_real_tests/tests/test_mujoco_arbitrary_grasp_pose_motion.py
```

## Adding scenarios

1. Put reusable scenario code under `rm65/` or the relevant backend package.
2. Add a versioned JSON input under `configs/`.
3. Return `SimRealScenarioResult`; do not print-only or hide failures in logs.
4. Add a bounded deterministic pytest under `tests/`.
5. Record minimum clearance, convergence errors, seed and executed step count.
6. Keep real-hardware confirmation and dispatch outside this package.

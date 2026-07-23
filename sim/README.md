# Simulation Architecture

> 本页说明仿真架构和模块边界。当前可直接运行的仿真命令统一从
> [`docs/sim/README.md`](../docs/sim/README.md) 复制。

## Backend roles

- **MuJoCo is the canonical task-physics backend** for real2sim, task scenes,
  contact, system identification, policy rollout, and later sim2real checks.
- **SAPIEN remains a teleop/debug preview**; its RM65 renderer lives in
  `sim/rm65`, while entrypoints remain in `teleop_runtime`. It is useful for
  tracker mapping and operator visualization, but it is not the canonical
  task-physics source of truth.
- MuJoCo `replay/` keeps an explicit kinematic diagnostic profile. It may set
  qpos directly for camera/geometry checks and must not be reported as a
  dynamic manipulation rollout.

## Directory ownership

```text
sim/
  contracts/                 # versioned cross-domain semantic contracts
    action.py                 # pins canonical RobotAction/deploy action layout
    observation.py            # state/image layout and timestamp semantics
    timing.py                 # sample/policy/reference/command/physics clocks
    scene/                    # scene composition contract split by concern
      contact.py              # friction and solver/contact parameters
      collision.py            # collision profile and explicit deny pairs
      world.py                # ground specification
      object.py               # free-body shape, mass, inertia and pose
      spec.py                 # composes the versioned SceneSpec
    manifest.py               # one validated contract bundle
  configs/                    # reproducible contract profiles
  rm65/                       # engine-independent RM65 embodiment behavior/evidence
    actuation/                # causal qcmd -> actuator-target profile and core
    real2sim/                 # fit indexes, manifests and fidelity contracts
    sapien_renderer.py        # SAPIEN operator/debug preview adapter
  mujoco/
    actuation/                # thin MuJoCo bindings for RM65/DM-Tac actuators
    deploy/                   # shared-control observation/action/backend binding; no teleop_runtime dependency
    modeling/                 # MJCF construction helpers only
      geometry.py             # shared visual/collision geometry conversion
      collision.py            # group masks and allow/deny body pairs
      world.py                 # ground/world geoms
      objects.py               # explicit-inertia freejoint rigid bodies
    replay/                   # kinematic dataset replay/export
    urdf.py                   # URDF parsing
    mjcf.py                   # thin assembly facade
    scene.py                  # loaded MuJoCo model/data facade
    tasks/cube_lift/          # task-local Gymnasium env, reward and clocks
```

Do not add task rewards, policy adapters, actuator models, randomization, and
dataset writing to `mjcf.py`. New responsibilities should get their own
package at the corresponding layer.

## Canonical physical profile

The first versioned profile is:

```text
sim/configs/rm65_bimanual_physics_v1.json
```

It pins the existing 26D action and 38D observation contracts, the current
15/60/120 Hz position-control clocks, a separate 1000 Hz MuJoCo physics clock,
the v3 RM65 URDF, collision profile, gravity, ground, and a minimal freejoint
debug cube. Values that still require dedicated real 120 Hz identification are
marked as pending rather than presented as identified truth.

Generate/inspect its MJCF without advancing physics:

```bash
python -m sim.mujoco.run_control \
  --contract sim/configs/rm65_bimanual_physics_v1.json \
  --dump-mjcf-only \
  --mjcf-out /tmp/rm65_bimanual_physics_v1.xml
```

Legacy kinematic inspection remains explicit:

```bash
python -m sim.mujoco.run_control \
  --urdf assets/rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera.urdf \
  --control qpos \
  --no-render
```

## Current physical boundary

This stage provides collision geometry, group/mask filtering, gravity, ground,
dynamic rigid objects, and a provisional RM65 position-actuator model. The
actuated path preserves 15 Hz policy targets, the shared acceleration-limited
planner, 120 Hz qcmd transport, fitted per-joint causal delay/high-speed target
lag, and 1000 Hz MuJoCo dynamics. Position gains, torque limits, damping and
friction remain pending dedicated plant identification, so this is a physical
rollout model, not a validated RM65 servo twin.

The accepted-qcmd/delay/filter state machine and current actuator v4 profile live in
`sim/rm65/actuation`; MuJoCo owns only actuator binding, physical stepping, and
measured qpos. The fourteen-trace fit evidence lives in `sim/rm65/real2sim` and
has no untouched v4 held-out capture yet.

The v3 URDF already contains one table collision body. The scene profile adds
a ground plane below it; it does not synthesize a second table.

## Minimal Cube-Lift RL environment

`sim.mujoco.tasks.cube_lift.CubeLiftEnv` is the first headless environment for
RL plumbing. It can use either the retained 150 Hz direct-qpos diagnostic or
the 15/60/120/1000 Hz actuated-position chain, selected by task config, and
lets the right arm control the red 3 cm cube at the physical table center.
The agent action is a 7-D normalized right-arm joint/gripper delta; the initial
31-D observation is explicitly `state_privileged`, not a real/deploy policy
state or a camera observation.

Install `gymnasium` using the dedicated instructions in `README_ENV.md` before
constructing this environment. The task config is:

```text
sim/configs/tasks/rm65_right_cube_lift_rl_debug_v0.json
sim/configs/tasks/rm65_right_cube_lift_rl_actuated_v1.json
sim/configs/tasks/rm65_right_cube_lift_rl_actuated_v2.json
```

The actuated v1 task keeps the gripper prescribed and the grasp helper as
`contact_latch_debug` while validating physical arm execution. Actuated v2 is
the default training environment: both arms and both DM-Tac grippers use
MuJoCo actuators, policy observation reports measured gripper close, and the
identified free-motion profile is bound by name and SHA-256. The latch remains
a temporary grasp helper because contact force is not Newton-calibrated. Success is
deliberately only `height_only_lift_success`: the cube must remain
above its settled center height plus 8 cm and leave table contact for a short
physics-tick hold. This is an environment/API milestone, not evidence of
actuator dynamics, physical grasp stability, or sim2real transfer.

## High-throughput Cube-Lift runtime

The reference `CubeLiftEnv` intentionally preserves detailed info and complete
per-physics-tick audit history. RL sampling must use the explicit fast APIs:

```python
from sim.mujoco.tasks.cube_lift import FastCubeLiftEnv
from sim.mujoco.runtime import CubeLiftVectorEnv, CubeLiftVectorExecutor

single_env = FastCubeLiftEnv()
vector_env = CubeLiftVectorEnv(num_envs=16)
```

Both default constructors select actuated v2. Pass the v0 or v1 task path
explicitly only for regression diagnostics.

The fast kernel preserves the task-selected execution semantics while caching
numeric MuJoCo indices, avoiding rich contact/audit
objects in the hot loop, restoring a settled-state snapshot on repeated resets,
and loading a versioned MJB artifact from `.cache/mujoco/`. Override the cache
with `SIM_MUJOCO_CACHE_DIR` or the `model_cache_dir` constructor argument. MJB
files are reproducible caches and must not be committed.

Reference and fast environments share the same collision policy: prescribed
targets are swept and projected before application, gripper/object and
object/table contacts remain legal, and robot/environment penetration beyond
the hard threshold terminates the episode. Compact info exposes projection
rate, projection alpha, minimum safety distance and hard-penetration status.

Run the debug-machine benchmark without rendering or realtime sleeps:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m sim.mujoco.benchmarks.benchmark_cube_lift \
  --runtime both --vector-envs 1,4,8,16 --steps 500 --warmup-steps 50
```

Add `--random-actions --random-action-scale 1.0` to include collision-oracle
and projection cost; the default zero-action benchmark measures the physics
and multiprocessing floor.

The vector executor transfers actions, observations, rewards, done flags and
terminal observations through fixed shared-memory arrays. Pipes carry only
commands and compact metadata. It supports selective reset, deterministic
`(master_seed, env_id, episode_id)` random streams, same-step auto-reset, worker
error propagation and bounded shutdown. Actuated Viewer rollout deliberately
delegates to this same scalar kernel so deployment cannot silently substitute
the direct-qpos backend.

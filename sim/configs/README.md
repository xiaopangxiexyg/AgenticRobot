# Simulation Profiles

Each JSON file in this directory is a complete, versioned
`SimulationContract`. Entrypoints should load a profile instead of copying
action dimensions, observation layouts, control rates, gravity, collision
masks, or object parameters into CLI defaults.

Current profiles:

- `rm65_bimanual_physics_v1.json`: canonical MuJoCo physical-scene foundation;
  its one `debug_cube` is a red `0.03 × 0.03 × 0.03 m` free body centered on
  the table at `x=-0.472, y=0` (with 1 mm initial settling clearance). Its
  collision section also owns prescribed-motion safety clearance, hard
  penetration, joint-sweep resolution and projection-iteration parameters.

## Scene expansion for manipulation tasks

Scene contracts remain shared by MuJoCo tasks and robot-facing adapters;
task-specific geometry must not be hidden in Agent prompts or renderer code.

The existing `scene_spec.v1` remains box-only and unchanged. New scene
experiments may explicitly select `dexteleop.sim.scene_spec.v2`, whose
`FreeBodyObjectSpec.size_m` always means the object's full x/y/z bounding
extents in meters. The first v2 slice supports box, upright cylinder, sphere,
and ellipsoid free bodies. Cylinder x/y extents must match; sphere extents must
all match. Mass and diagonal inertia are explicit in the contract, with
shape-specific solid-body defaults. These primitive forms are stable and
inexpensive for grasp/contact experiments; they are geometric approximations,
not claims about the exact mass distribution, surface finish, or friction of a
real product.

The v2 primitives are a geometry foundation only. They do not yet certify
generic grasp/contact task semantics, mesh collision quality, placement success,
or transfer to the RealMan hand. Each later scene/task batch must retain an
unchanged baseline and add a bounded, reproducible MuJoCo comparison before it
is used for Agent capability claims.

v2 的形状支持是场景建模能力，不等于所有形状都能进入携物 Skill。当前载荷守卫
仅支持完整 box 与 convex mesh；sphere／cylinder／ellipsoid 仍须明确拒绝，直到
对应的支撑几何与携物检查另行接入并验证。v1 现有 JSON、默认场景及碰撞政策保持不变。

Task configurations live in `tasks/`, rather than pretending to be a second
`SimulationContract`:

- `tasks/rm65_right_cube_lift_rl_debug_v0.json`: the task-local, privileged-
  state Cube-Lift environment. It references the base profile by path and
  SHA-256, freezes the 7-D right-arm delta action and 31-D observation layout,
  and declares `kinematic_direct_qpos_debug` / `contact_latch_debug` provenance.
- `tasks/rm65_right_cube_lift_rl_actuated_v1.json`: the corresponding physical-
  arm validation task. It binds the versioned RM65 actuation-profile hash,
  declares `actuated_position_servo` / `mujoco_position_actuator`, runs the
  shared 15/60/120 Hz planner over 1000 Hz dynamics, and explicitly retains
  `prescribed_qpos_debug` gripper dynamics plus `contact_latch_debug` grasping.
- `tasks/rm65_right_cube_lift_rl_actuated_v2.json`: the retained v2 privileged
  compatibility task.
  It additionally binds the identified bilateral DM-Tac profile by hash, drives
  four physical finger actuators, and exposes measured rather than target
  gripper closure in the privileged observation. Contact force and the grasp
  latch remain explicitly provisional.
- `tasks/rm65_right_cube_lift_rl_normal_v1.json`: the retained original normal
  task. It exposes only the 15-D real-observable proprioceptive policy state,
  uses physical MuJoCo contact, forbids the debug grasp latch, and binds the
  same fitted RM65/DM-Tac profiles as every current expert configuration.
- `tasks/rm65_right_cube_lift_rl_normal_v2.json`: the current state-only
  training default. It uses the 28-D measured motion/tracking profile without
  RGB or an episode clock, smaller pose/gripper increments, bilateral-contact
  reward/success gates, a validated 22 mm closed-grasp target / 70 mm lift
  milestone, failure-aware terminal rewards, and pose-target anti-windup. Its
  versioned DM-Tac variant preserves the identified free-motion actuator and
  applies a simulation-only post-settle contact-friction tuple.
- `tasks/rm65_right_cube_lift_rl_normal_contact_curriculum_v1.json`: lift-first
  diagnostic retained for the completed 200k run. It established bilateral
  contact, but its dense projection penalty made leaving the cube more
  profitable than lifting; do not use that checkpoint for home-reset warm-start.
  The contact criterion uses a 0.55 measured-close floor because bilateral
  cube contact physically stalls closure below 0.60 on some resets; one
  recoverable retry prevents that numerical edge from aborting a vector run.
- `tasks/rm65_right_cube_lift_rl_normal_contact_curriculum_v2.json`: corrected
  lift-first curriculum with the same 28-D input and 7-D output contract. It
  keeps the gripper target closed, limits horizontal/rotation motion, admits
  only non-negative lift motion, terminates sustained bilateral-contact loss,
  removes the contact-reacquisition bonus, reduces dense projection penalty,
  and strengthens lift progress. Its reset follows the validated float32 grasp
  path and executes the gate's final centered stabilization step before policy
  control. It remains simulator-only.
- `tasks/rm65_right_cube_lift_rl_normal_rgb_v2.json`: the retained camera
  baseline. It keeps the normal 15-D state contract and appends the active
  wrist RGB image at 48x48 with the fitted actuator profiles.
- `tasks/rm65_right_cube_lift_rl_normal_rgb_v3.json`: a retained normal-policy
  input/reward candidate. It remains real-observable (29-D measured
  motion/tracking proprioception plus the active 48x48 wrist RGB image), uses
  the same pose action and identified actuator profiles, and records staged
  close-gate reward parameters on the simulator task boundary only.
- `tasks/rm65_right_cube_lift_rl_normal_rgb_v4.json`: the retained no-episode-
  clock RGB profile. New camera training is paused; use it only explicitly for
  checkpoint compatibility.
- `tasks/rm65_right_cube_lift_rl_privileged_expert_v1.json`: the opt-in expert
  task. It exposes the 31-D simulator state and retains the task-local debug
  latch for teacher/curriculum experiments. It does not define a deployment
  policy contract.

The same pair is also the default of `MujocoPositionExecutionConfig`: RM65 arm
profile `rm65_bimanual_udp_feedback_20260721.v4` plus DM-Tac profile
`dmtac_bimanual_position_20260722.v1`. Low-level diagnostics that require
prescribed qpos must opt out explicitly with both actuation profiles set to
`None`; loading a scene or model artifact remains side-effect free and does not
silently add actuators.

The two actuated tasks currently bind the fourteen-trace provisional RM65 arm
profile `rm65_bimanual_udp_feedback_20260721.v4` by canonical metadata hash.
Its engine-independent profile/core and evidence live under `sim/rm65/`; the
MuJoCo executor is a backend adapter.

The v0 direct-qpos and v1 prescribed-gripper files are marked `legacy_debug`;
they remain available for regression diagnostics but are outside the two
current policy classes and must not be used for new policy training.

Changing a base profile intentionally invalidates a task config's stored base
hash. Update that hash only after reviewing the physics/scene change and its
task regression.

The corresponding validation and serialization code lives in
`sim/contracts/`; MuJoCo-specific XML construction lives in
`sim/mujoco/modeling/`.

# Simulation Profiles

Each JSON file in this directory is a complete, versioned
`SimulationContract`. Entrypoints should load a profile instead of copying
action dimensions, observation layouts, control rates, gravity, collision
masks, or object parameters into CLI defaults.

Current profiles:

- `rm65_bimanual_physics_v1.json`: canonical MuJoCo physical-scene foundation;
  its one `debug_cube` is a red `0.03 × 0.03 × 0.03 m` free body centered on
  the table at `x=-0.412, y=0` (with 1 mm initial settling clearance).

Task configurations live in `tasks/`, rather than pretending to be a second
`SimulationContract`:

- `tasks/rm65_right_cube_lift_rl_debug_v0.json`: the task-local, privileged-
  state Cube-Lift environment. It references the base profile by path and
  SHA-256, freezes the 7-D right-arm delta action and 31-D observation layout,
  and declares `kinematic_direct_qpos_debug` / `physics_contact` provenance.

Changing a base profile intentionally invalidates a task config's stored base
hash. Update that hash only after reviewing the physics/scene change and its
task regression.

The corresponding validation and serialization code lives in
`sim/contracts/`; MuJoCo-specific XML construction lives in
`sim/mujoco/modeling/`.

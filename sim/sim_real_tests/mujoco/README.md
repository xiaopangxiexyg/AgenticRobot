# MuJoCo-backed sim-real checks

Put physics, actuator, contact, and swept-collision acceptance scenarios here.
They may depend on `sim.mujoco` and `robot_motion`, but must not import
`teleop_runtime` or real hardware drivers. Keep model-only FK/IK/qpos checks in
`../rm65/` so they remain fast and can run without constructing `mjModel`.

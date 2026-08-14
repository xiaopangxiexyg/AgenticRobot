# MuJoCo-backed sim-real checks

Put physics, actuator, contact, and swept-collision acceptance scenarios here.
They may depend on `sim.mujoco` and `robot_motion`, but must not import
`teleop_runtime` or real hardware drivers. Keep model-only FK/IK/qpos checks in
`../rm65/` so they remain fast and can run without constructing `mjModel`.

## Cardinal TCP displacement primitives

`tcp_displacement_primitives_closed_loop.py` executes the six fixed 3 cm
`robot_motion.primitives` translations through the existing 60 Hz coupled IK,
120 Hz qcmd, and 1000 Hz MuJoCo physics chain. The left arm completes
forward/backward/left/right/up/down first; the right arm then repeats the same
sequence. Every primitive starts from and returns to the measured simulation
anchor, so direction or frame mistakes do not accumulate across cases.

`view_tcp_displacement_primitives.py` replays the physical trace with a target
star at each arm anchor. Red denotes +/-X, green +/-Y, blue +/-Z, magenta the
left actual TCP trail, cyan the right actual TCP trail, and yellow the active
target.

```bash
python -m sim.sim_real_tests.mujoco.view_tcp_displacement_primitives
```

## Active TCP-pose envelope closed loop

`tcp_pose_envelope_closed_loop.py` binds the backend-neutral TCP target
projector to the existing final pre-IK hook and exercises one left-arm relative
translation/rotation envelope through the physical 15 Hz source -> 60 Hz
coupled IK -> 120 Hz qcmd -> 1000 Hz MuJoCo chain. It checks exact emitted
target projection, an unchanged unconfigured arm, nonzero physical tracking
residual, immediate inward release, absence of stored catch-up, IK status, and
the configured 120 Hz joint-step bound.

The result deliberately reports `swept_tcp_certified=false`: it does not claim
that the continuous physical TCP path between two safe emitted targets remains
inside the target envelope.

An interactive viewer first generates that physical closed-loop trace, then
replays the complete result with angular direction cones and three full TCP
position trails:

- orange: requested Pose, including the out-of-range path;
- green: effective pre-IK Pose after exact clipping;
- magenta: actual MuJoCo TCP Pose under actuator dynamics.

Three translucent RGB wireframe cones start from the same anchor TCP and use
the configured angular limit as their half-angle. Orange, green and magenta
direction triads also share that exact apex, so whether the raw, effective and
actual X/Y/Z directions lie inside their cones is directly comparable. The
three basis directions together encode the complete orientation, including
twist. No translation boundary box is rendered in this angular demonstration.

```bash
conda run --name "${DEXTELEOP_ENV_NAME:-dexteleop310}" env PYTHONPATH=. \
  python -m pytest \
  sim/sim_real_tests/tests/test_mujoco_tcp_pose_envelope_closed_loop.py -q
```

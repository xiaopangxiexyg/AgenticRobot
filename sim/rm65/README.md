# RM65 simulation embodiment

`sim/rm65` owns simulation-only behavior and evidence that is specific to the
RM65 embodiment. It is deliberately above any one physics engine and must not
import `teleop_runtime` or initialize real hardware.

`sapien_renderer.py` is an explicit frozen exception retained for historical
debug-preview reproduction. It is not part of the current RM65 simulation
route, must not receive new callers or features, and must not be used as
real2sim, task-physics, release-readiness, or validation evidence.

```text
sim/rm65/
  actuation/
    profile.py        # versioned free-motion response parameters
    core.py           # backend-independent qcmd -> actuator-target state machine
    gripper.py        # fitted normalized DM-Tac position actuator core
  real2sim/
    artifact.py       # evidence/claim/validation contract
    gripper_position_fit.py  # offline DM-Tac free-motion position fit
    artifacts/        # content-addressed fit indexes and manifests
  sapien_renderer.py  # frozen retained legacy preview only

sim/mujoco/actuation/
  executor.py         # writes RM65 core targets to MuJoCo and reads measured qpos
```

The reusable actuator core owns the 120 Hz acceptance deadline, per-joint
causal transport delay, raw target, and velocity-dependent filtered target. A
physics backend owns servo/contact dynamics and measured state. This split
allows another simulation backend to reuse the same fitted command path
without depending on MuJoCo internals.

The current canonical arm profile is
`rm65_bimanual_udp_feedback_20260721.v4`. It is a provisional fourteen-trace
free-motion fit; force, payload, saturation, and contact response are not yet
identified. See `real2sim/README.md` and the v4 manifest before making a
fidelity claim.

The current canonical DM-Tac normalized position profile is
`dmtac_bimanual_position_20260722.v1`. It is a bilateral close/open fit at the
real 20 Hz command gate with separate held-out step validation. Its scope is
free motion only; fast preemption is provisional and contact/force behavior is
not identified. The immutable stage artifact lives at
`real2sim/artifacts/dmtac_gripper_position_20260722_v1.json`.

The former top-level `sim.real2sim` compatibility package has been removed.
RM65 code must use `sim.rm65.actuation` or `sim.rm65.real2sim` directly.

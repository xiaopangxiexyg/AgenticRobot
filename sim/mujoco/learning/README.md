# MuJoCo learning contracts

This package is the additive first layer of the sim/real policy-parity
migration.  It contains no task reward and no hardware setup.

The intended data path is:

```text
sim snapshot / real deployment snapshot
        -> named causal ObservationContract fields
        -> ModelInputBinding (same order, dtype, image layout, normalization)
        -> policy model
        -> PolicyActionBinding (latent model output)
        -> PolicyActionTransform (task-owned, deterministic)
        -> complete real-aligned task action binding
        -> IK / canonical safety / sim or real execution adapter
```

The PPO latent action, complete task action, and admitted/executed action are
separate values. PPO stores only the sampled latent value and its original log
probability. Task conditioning, IK, safety projection, actuator dynamics, and
contact are downstream transition dynamics; their outputs must not be inverted
to manufacture a replacement PPO sample. A transform may reduce a policy to
xyz, xyz+gripper, or z-only while the public Cube-Lift boundary stays on the
seven-value right-TCP pose/gripper layout used by the real-aligned decoder.

`ObservationRole.ACTOR` is the normal deployable-policy allowlist.  Privileged
and reward-only fields can be present in an expert/critic snapshot, but a
normal binding rejects them before encoding.  This is deliberately separate
from task-specific Cube-Lift state: a task adapter owns task progress terms,
while `reward.py` only reports embodiment/execution costs that should remain
valid when the same policy is moved between sim and real.  The Cube-Lift
reward path keeps its legacy 31-D simulator state at the task boundary, but
projects it once into named `CubeLiftRewardState` fields before calculating
terms; fixed indices are no longer scattered through the reward calculation.

The selected first fixture is now available at
`sim.mujoco.tasks.cube_lift.learning`: it names the existing normal/expert
state and wrist-RGB slices, declares both latent and complete task actions,
and provides a lossless flat-vector projection for parity tests. Its
identities are checked at PPO startup and recorded in temporary-PPO
run/checkpoint metadata. The normal v4 fixture is
the long-term policy contract: it uses only measured state and wrist RGB and
does not expose `episode_progress`.  v3 remains loadable as a versioned
compatibility profile.

`PolicyModelManifest` and `dry_run_model_io` add the model-side half of the
boundary.  Before inference they require matching observation, input-binding
and action-binding hashes, exact input/output dimensions, explicit normal vs
privileged policy class, finite values, and matching batch shapes.  They do
not load a model or touch hardware, so a future real observation adapter can
use the same fail-closed check without changing the simulator execution path.

The compatibility fixture names `episode_progress` separately as a
runtime-clock input.  v4 deliberately removes it rather than asking a real
deployment adapter to reconstruct simulator reset-relative time.  A future
real adapter still has to provide the measured-state and camera source
semantics declared by the v4 fields; the manifest check is necessary but does
not replace calibration or paired sim/real validation.

Randomized task profiles that append object pose split it into an explicit
`object_pose` field. Their current source is named
`sim_ground_truth_proxy_for_external_object_pose_estimator.v1`; this is a
simulation oracle standing in for a required real estimator, not evidence
that the real observation adapter exists or has been calibrated.

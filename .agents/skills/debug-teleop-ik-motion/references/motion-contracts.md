# Motion contracts

## Frame and time

- Confirm base, tool, tracker, task, solver endpoint, and controlled frames.
- Compare rotations with SO(3)/SE(3) errors; do not subtract Euler angles.
- Preserve quaternion sign continuity.
- Align streams by authoritative timestamps and investigate clock offsets before
  interpreting lag. Treat negative or impossible age as a clock defect.
- Distinguish raw intent, shaped intent, admitted pose, accepted QPOS/FK, and
  measured pose.
- Advance velocity/acceleration history once per physical control step, not once
  per solver iteration.
- Use lookahead only when it existed inside the online causal boundary.

## Stateful modifiers

- Expand the implemented equation into base target, slew state, anchor,
  integrator, bias, cached reference, and engage/release state.
- Record each term's owner, frame, update clock, finite bound, and reset, HOLD,
  stale, home, engage, and release behavior.
- Feed persistent state only with an explicitly incremental, same-frame,
  same-time modifier effect relative to a nominal or counterfactual baseline.
  Never integrate an absolute target-to-command or target-to-pose residual.
- Treat step, velocity, acceleration, jerk, and jump limits as dynamics bounds,
  not total range bounds.
- Bound the final composed modifier output independently of component bounds.
- Exercise Active-only feedback, rebase, integration, release, and fault paths
  offline in closed loop; a shadow path proves only the edges it executes.

## IK and reference ownership

- Preserve candidate generation, candidate selection, safety admission,
  committed reference, interpolation, and dispatch as distinct stages.
- Anchor committed evolution to an authoritative accepted or realized state,
  never an unaccepted proposal.
- Require explicit branch identity and dynamically feasible transitions for
  rebranching.
- Treat singular metrics as evidence, not automatic authority to hold or relax a
  task.
- Do not duplicate lower tool geometry, frame conversion, collision, or safety
  ownership in Gateway, GUI, replay, or diagnostic code.

# Evidence and metrics

## Locate the boundary

- Smooth source but discontinuous raw IK points to generation or branch
  behavior.
- Smooth raw IK but discontinuous canonical output points to selection,
  normalization, or admission.
- Smooth accepted reference but discontinuous command points to interpolation,
  scheduling, or dispatch.
- Smooth commands but discontinuous feedback points to execution, transport, or
  hardware.
- A slow downstream stage may amplify an upstream defect without causing it.

## Measure separately

- Geometry: joint-limit margin, branch identity, singular values, and
  owner-supplied collision margin.
- Demand: pose error, desired twist, weak-direction projection, and feasible
  command fraction.
- Execution: step, velocity, acceleration, jerk, FK residual, branch switches,
  dwell/HOLD duration, catch-up distance, latency, deadlines, queue age, and
  dispatch receipts.

Compare teacher-forced and closed-loop replay when both are available.
Teacher-forced evidence tests local solving; closed loop exposes anchor drift
and path-lift errors.

For stateful modifiers, include nonzero nominal residual, held/rejected
admission, duplicate/stale samples, repeated engage/release, sign chatter, and
long constant or zero input. Assert every persistent state and the final output
on every tick.

## Validation order

1. Hard safety and validity.
2. Pose accuracy within the declared tolerance.
3. Unintended branch transitions.
4. Velocity, acceleration, jerk, and deadline behavior.
5. Singular and joint-limit robustness.
6. Representative normal-motion non-regression.

Label conclusions as confirmed, likely, bounded, or unknown. Do not present a
heuristic or a successful process exit as proof of realized motion.

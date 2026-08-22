# Quest Future Visualization Architecture

## Goal

The final operator-facing behavior is a Quest WebXR view that overlays the
robot's measured TCP and a policy's selected future trajectory in one reviewed
coordinate frame.  The view is diagnostic and predictive only: it must not own
source advancement, authority, safety, dispatch, or robot motion.

The planned controller-on-TCP calibration solves the two fixed transforms in:

```text
T_questReference_from_controllerGrip(i)
  = T_questReference_from_robotReference
  @ T_robotReference_from_tcp(i)
  @ T_tcp_from_controllerGrip
```

The Quest controller is rigidly mounted to the TCP while an operator moves the
robot through independent stationary poses.  Movement remains owned by the
existing robot/teleoperation path; calibration code only observes immutable
facts that path has already produced.

## Current evidence boundary

As of 2026-08-23, the operator has confirmed on the real Quest that the local
HTTPS page can be opened, a WebXR session can be entered, and the basic reference
geometry is rendered.  That is hardware evidence for the Quest transport and
rendering boundary only.  It is not evidence for robot pose publication,
paired-pose capture, calibration quality, or policy-trajectory projection.

The implemented source contracts now provide:

- authenticated Quest controller samples emitted only after session, reference
  generation, sequence, and rate admission;
- ordered Quest accepted/invalidation events carrying server, client, XR
  session, reference generation, receiver clock, and source sequence identity;
- an atomic Quest subscription cut that contains no pose and prevents a
  read-latest/subscribe reset race;
- canonical robot `ExecutionMeasurementObservation` projection with exact
  source epoch, frame, tool, FK, observation-plan, and receiver-clock lineage;
- a bounded robot accepted/invalidation journal with an atomic cut and explicit
  gap failure, without callbacks or worker threads on the policy/control path;
- fail-closed invalidation on hardware-read loss, same-sequence content reuse,
  source epoch transition, or frame/tool/FK binding drift.

The robot projection remains opt-in.  No current production caller supplies an
`ArmPoseObservationProjection`, and the RM65 backend does not yet emit the typed
`PoseCoordinateBinding` attestation required by that projection.  Therefore the
new robot contract is verified offline with fake backends, not wired real-robot
evidence.

## Ownership and data flow

```text
Quest WebXR gripSpace
  -> controller_pose runtime admission
  -> ordered accepted/invalidation events
                                      \
                                       -> lower paired-pose capture engine
                                      /
real arm owner -> hardware qpos -> bound FK pose
  -> canonical measurement -> bounded passive event journal

paired-pose capture candidate
  -> unchanged R0 dataset parser
  -> diagnostic fit
  -> reviewed acceptance + independent validation (not implemented)
  -> immutable visualization frame binding
  -> measured TCP + selected policy future trajectory
  -> Quest read-only renderer
```

`hardware/quest_dev/features/visualization/` owns reusable static-window
admission, pairing, SE(3) metrics, dataset assembly, and bounded evidence.
`control_station/quest_calibration/` may later own operator intents, application
lifecycle, source selection, and a pose-redacted status projection, but must not
reimplement the capture reducer or call a refreshing robot snapshot API.

`control_composition/` continues to own backend-neutral source/future semantics.
It must not import Quest, hardware, Control Station, or a concrete policy
deployment.  A concrete application leaf will adapt an already-selected future
trajectory into the accepted visualization frame and publish it passively.

## Time and sequence rules

Pairing may compare only receiver-local timestamps proven to use the same exact
`ClockDomainDescriptor`:

- robot: `PoseMeasurementPublication.sample_ref.available_at`;
- Quest: `ControllerPoseSample.received_monotonic_ns / 1e9`.

It must never subtract Quest `xr_frame_time_ms` from a robot-native measurement
timestamp.  Each side keeps its native observed time and clock domain in the R0
dataset; receiver time is used only for window admission and deterministic pair
selection.

The current Quest runtime carries only an optional receiver clock-domain ID;
that ID alone does not prove descriptor equality.  The lower capture binding
must be given the full receiver `ClockDomainDescriptor`, verify that its ID
matches the Quest event and robot publication, and include the descriptor hash
in sealed evidence.  Until that binding exists, cross-source time pairing is
not implemented and must fail closed.

Quest callbacks run outside the runtime lock and may arrive out of wall-clock
order across HTTP threads.  Consumers must order/fence by `event_sequence` and
must not let an older callback resurrect an invalidated session.  Robot journal
events are allocated and retained under one journal lock; a consumer that falls
behind the bounded prefix receives an explicit gap error and must fail closed.

## Calibration acceptance boundary

An admitted stationary window is only a paired-pose candidate.  The existing R0
fit is only a diagnostic fit.  Neither may be installed into a visualization or
execution frame graph without a reviewed acceptance receipt binding at least:

- paired dataset and fit hashes;
- robot source instance/epoch and coordinate-binding hash;
- tool profile and FK contract hashes;
- Quest server/page/XR session/reference generation/controller identity;
- residual and observability checks;
- independent held-out validation poses;
- invalidation rules for reference reset, remount, tool change, or source epoch
  transition.

No acceptance threshold is treated as hardware truth until it is reviewed
against real capture evidence.

## Safety invariants

- All Quest and calibration paths are read-only with respect to robot state.
- No module in this path starts, homes, servos, replays, holds, or commands a
  robot.
- Reading the robot pose journal or latest publication never refreshes a backend.
- A source failure clears the published latest pose before downstream sealing.
- Visualization never grants control authority and never weakens canonical
  target safety or dispatch checks.
- Offline/mock tests prove contracts, not motion safety, real-time performance,
  calibration quality, or hardware readiness.

## Next implementation slices

1. Add the lower-owned manual static-window capture reducer and strict evidence
   bundle parser next to the R0 paired-pose dataset owner.
2. Expose the unique running Quest feature runtime through a programmatic host,
   rather than creating a second runtime or using a global registry.
3. Add an RM65-owned typed coordinate/FK/tool attestation and bind it to the
   existing arm-state owner without creating a second hardware reader.
4. Add the thin Control Station operator lifecycle for begin/complete/cancel and
   pose-redacted status.
5. Collect real stationary-pose evidence, fit, independently validate, and
   review an acceptance profile.
6. Only after acceptance, transform the measured TCP and selected policy future
   trajectory into Quest space and render them as read-only geometry.

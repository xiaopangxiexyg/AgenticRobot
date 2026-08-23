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
- a pose-redacted Quest `atomic_event_cut()` whose exclusive watermark permits
  consumers to drain callbacks created before an end boundary without
  registering another observer;
- a Quest `unsubscribe_with_cut()` close operation that removes one observer
  and fixes its final exclusive watermark under the same runtime lock, returning
  a token-bound `ControllerPoseObserverCloseCut` that a plain event cut cannot
  impersonate;
- canonical robot `ExecutionMeasurementObservation` projection with exact
  source epoch, frame, tool, FK, observation-plan, and receiver-clock lineage;
- a bounded robot accepted/invalidation journal with an atomic cut and explicit
  gap failure, without callbacks or worker threads on the policy/control path;
- fail-closed invalidation on backend readiness or hardware-read loss,
  same-sequence content reuse, source epoch transition, or frame/tool/FK
  binding drift;
- a lower-owned, bounded in-memory manual static-window reducer with exact
  Quest/robot binding, global Quest callback reordering, deterministic
  representative pairing, stationarity evidence, and strict outer evidence
  projection into the unchanged R0 paired-pose dataset.
- a programmatic `QuestDevHost` that owns its HTTP server, shared stop
  event, and enabled feature runtime instances, and exposes an exact-name,
  explicit expected-type borrowing seam only while those runtimes are running.
  The CLI uses this same host; one composition root creates one host for the
  configured endpoint, so no second controller runtime or global registry is
  required.

The robot projection remains opt-in.  No current production caller supplies an
`ArmPoseObservationProjection`, and the RM65 backend does not yet emit the typed
`PoseCoordinateBinding` attestation required by that projection.  Therefore the
robot publication and R2 paired-capture contracts are verified offline with
fake sources, not wired real-robot paired-pose evidence.  R2 does not add a
Control Station lifecycle, persistence, calibration fit or acceptance, or a
frame installation path.

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

For R2, `hardware/quest_dev/features/visualization/` provisionally owns the
Quest-specific static-window reducer, pairing, SE(3) metrics, bounded evidence,
and projection into the pre-existing R0 Quest calibration dataset. This is a
feature-local integration boundary, not permission for `hardware/` to define a
general dataset schema or become the canonical calibration owner. Promotion to
a neutral calibration runtime is an explicit decision before non-Quest reuse.
`control_station/quest_calibration/` may later own operator intents, application
lifecycle, source selection, and a pose-redacted status projection, but must not
reimplement the capture reducer or call a refreshing robot snapshot API.

The detailed R2 source and evidence contract is documented in
`hardware/quest_dev/docs/features/paired-pose-capture.md`.  The reducer is a
passive in-memory component: it consumes immutable source events and owns no
source runtime, background device worker, persistence service, or operator
lifecycle.

`control_composition/` continues to own backend-neutral source/future semantics.
It must not import Quest, hardware, Control Station, or a concrete policy
deployment.  A concrete application leaf will adapt an already-selected future
trajectory into the accepted visualization frame and publish it passively.

## Time and sequence rules

Pairing may compare only receiver-local timestamps proven to use the same exact
`ClockDomainDescriptor`:

- robot: exact `PoseMeasurementPublication.received_monotonic_ns`;
- Quest: exact `ControllerPoseSample.received_monotonic_ns`.

It must never subtract Quest `xr_frame_time_ms` from a robot-native measurement
timestamp.  Each side keeps its native observed time and clock domain in the R0
dataset; receiver time is used only for window admission and deterministic pair
selection.

Quest events and cuts carry a receiver clock-domain ID; that ID alone does not
prove descriptor equality.  The R2 capture binding is given the complete
receiver `ClockDomainDescriptor`; its source adapter must verify subscription
and end-cut IDs before passing their watermarks, while the reducer verifies
every ingested Quest event and the exact descriptor on robot events.  The
complete descriptor and hash remain in sealed evidence.  Construction without
that binding fails closed.

Quest callbacks run outside the runtime lock and may arrive out of wall-clock
order across HTTP threads.  The R2 reducer places accepted and invalidation
callbacks in one bounded global reorder buffer, drains only a contiguous
runtime-global `event_sequence`, and filters the selected identity only after
ordering.  It must not let an older callback resurrect an invalidated session.
Robot journal events are allocated and retained under one journal lock; the
reducer advances one global cursor across all robot streams before filtering to
the selected stream.  Falling behind the bounded prefix is an explicit terminal
gap rather than permission to take a new cut and continue.

Both sources use explicit half-open event intervals.  Quest cuts already expose
the exclusive `next_event_sequence`; a robot cut is normalized to
`event_sequence + 1`.  A window consumes only
`[start_next_event_sequence, end_next_event_sequence)`.  End cuts are
watermarks, not synchronous callback completion: the reducer remains pending
until robot pulls and out-of-lock Quest callbacks have asynchronously drained
through both end watermarks. `unsubscribe()` is not a drain barrier, and a
separate `unsubscribe()` plus `atomic_event_cut()` leaves a race on one side or
the other. Final closure therefore uses `unsubscribe_with_cut()` to remove the
observer and fix the Quest close watermark atomically, plus one typed robot
journal cut. The capture binding and close cut pin the same subscription token.
The first `seal_capture(robot_cut=..., quest_cut=...)` call pins both cuts and
returns pending until their exclusive watermarks are drained.

Stationarity is checked independently on all eligible raw poses from each
source using bounded sample counts, receiver-time span, maximum pairwise
translation distance, and maximum pairwise SO(3) rotation.  One manual window
selects one raw representative pair deterministically by receipt skew and
source/event sequence tie-breakers; it does not average poses or inflate one
stationary hold into multiple calibration poses.

## Calibration acceptance boundary

An admitted stationary window is only a paired-pose candidate.  R2 seals a
strict outer capture-evidence bundle containing the full source bindings,
clock, per-window watermarks, final typed robot/Quest cuts, ordered raw events
and hashes, stationarity metrics, representative selection, and the projected
unchanged R0 dataset and hash. The R0 dataset is
not the complete capture proof because it intentionally omits receiver cuts and
part of the host/source lineage.

R2 capture evidence fixes `calibration_accepted` to `null`.  The existing R0 fit
is diagnostic only and continues to emit `accepted: null`.  Neither may be
installed into a visualization or execution frame graph without a reviewed
acceptance receipt binding at least:

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
- Source callbacks only enter bounded in-memory evidence; this slice does not
  write capture files or own persistence.
- A gap, reorder overflow, identity drift, source invalidation, clock mismatch,
  non-atomic close attempt, or incomplete asynchronous drain prevents sealing.
- Visualization never grants control authority and never weakens canonical
  target safety or dispatch checks.
- Offline/mock tests prove contracts, not motion safety, real-time performance,
  calibration quality, or hardware readiness.

## Next implementation slices

The unique-running-runtime host prerequisite is now implemented. The remaining
ordered slices are:

1. Add an RM65-owned typed coordinate/FK/tool attestation and bind it to the
   existing arm-state owner without creating a second hardware reader.
2. Add the thin Control Station operator lifecycle for begin/complete/cancel and
   pose-redacted status.
3. Add an explicit persistence owner for reviewed capture evidence; the R2
   in-memory reducer deliberately writes nothing.
4. Collect real stationary-pose evidence, fit, independently validate, and
   review an acceptance profile.
5. Only after acceptance, transform the measured TCP and selected policy future
   trajectory into Quest space and render them as read-only geometry.

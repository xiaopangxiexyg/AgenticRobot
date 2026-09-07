# Quest Future Visualization Architecture

## Goal

当前 Quest/RM65 被动标定的现场操作、代码身份与安全交接见
[Quest 被动标定与 RM65 可视化任务交接](quest_passive_calibration_handoff.zh-CN.md)。

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
  required;
- an RM65 position-backend source attestation that is created only after a new
  SDK attachment has produced a trusted hardware-qpos measurement and the
  exact tool-TCP and FK inputs have been resolved.  It binds a fresh source
  epoch, hashed configured endpoint, canonical typed arm-base frame, the
  backend's effective tool transform, construction-time raw-URDF hash, actual
  six-joint solver-model fingerprint/order/frame, construction-time FK callable
  identity, FK implementation version, and the host wall-clock measurement
  domain.  Hardware-slot degrees to solver radians are bound slot-for-joint
  with no offset.  The attestation performs no
  SDK read of its own; startup/close are lifecycle-serialized, and failed
  startup, fresh attachment, or close clears the prior proof.  The endpoint
  passed to the SDK, side, effector identity, and default controlled frame are
  frozen at their owning construction/attachment boundary.  SDK-discovered TCP
  state is backend-runtime state and never mutates a caller-reusable config.
  If mutable compatibility config later diverges from the frozen side,
  effector, or default frame, publication fails closed for the rest of that
  attachment epoch rather than pairing the proof with changed runtime identity.
  Each published proof is revalidated against the exact solver/model/tool
  state; drift removes it rather than publishing a stale hash.  Attestation
  construction, full revalidation, and publication are independently opt-in
  through `include_pose_source_attestation=True`; ordinary diagnostic, deploy,
  and lean snapshots neither pay the model-fingerprint cost nor return an
  unvalidated proof.  A cached proof remains hidden unless explicitly
  requested again.
- a default-off policy-deployment application binding selected by
  `--rm65-pose-observation`.  After the existing RM65 resource lifecycle has
  started, it requires every active position backend to be ready, obtains and
  validates one typed R4a attestation per side, and joins the exact lower source,
  epoch, frame, tool, FK, and measurement-clock facts with side-local
  application-owned component, stream, observation-plan, pose-schema, control
  domain, and receiver-clock facts.  All sides must bind before the single
  existing `ArmStateReader` is constructed with the result.  Its injected state
  reader performs no SDK refresh: after the reader's existing cadence-owned
  refresh it makes one explicit cached-state read with full proof revalidation.
  The generic reader removes any canonical binding supplied inside lower state
  and injects the application binding only when the separately returned typed
  result exactly equals the configured projection.  A missing or drifting proof
  invalidates only the passive pose latest while preserving the legacy arm-state
  snapshot.  Shutdown terminally invalidates the journal before backend close.

The robot projection remains opt-in.  The RM65 lower object is deliberately a
narrow source/tool/FK attestation, not a second copy of the canonical
`PoseCoordinateBinding`; the deploy application is the sole owner of that
join.  A default-off concrete application bridge now injects one complete
receiver-clock descriptor and monotonic clock into the unique Quest host,
subscribes the existing robot journal and Quest event stream to the R2 reducer,
and exposes programmatic begin/complete/cancel/seal actions without adding a
second robot reader.  A separate deploy-owned store provides explicit immutable
capture publication/reload, offline diagnostic fitting, and explicit immutable
diagnostic-report publication/reload.  Reducer construction, ordinary polling,
fit computation, and ordinary close remain non-writing operations.

Those connections have only offline synthetic/contract evidence.  A thin
programmatic Control Station leaf now translates typed select/begin/complete/
seal/cancel intents into the existing deploy application and projects bounded,
pose-redacted operator status.  It has no GUI, Web binding, runtime composition
caller, or device-start authority.  There is still no real Quest + RM65
multi-pose calibration evidence, reviewed acceptance, frame installation, or
Quest future-trajectory renderer.  The lifecycle contract or successful reload
of a candidate report therefore cannot be used as calibration or visualization
readiness.

The app-root option is intentionally unavailable with `--send-actions`: every
projected state explicitly requests the materially expensive full lower proof,
and there is not yet real deadline evidence for putting that work on an active
control loop.  It also requires position mode, enabled arms, a configured IP for
every active side, a positive refresh rate, and no allow-connect-failure mode.
These checks make binding failure atomic; they do not turn the ordinary
policy-deploy backend lifecycle into an observation-only hardware profile.
Depending on its existing configuration, RM65 startup may still perform tool,
singularity, or background-worker setup.  The new option itself sends no arm
command and owns no device lifecycle, but it must not be described or operated
as a zero-write hardware commissioning path.

## Ownership and data flow

```text
Quest WebXR gripSpace
  -> controller_pose runtime admission
  -> ordered accepted/invalidation events
                                      \
                                       -> lower paired-pose capture engine
                                      /
real arm owner -> hardware qpos -> bound FK pose
  -> lower source/tool/FK attestation
  -> application-owned canonical coordinate binding
  -> canonical measurement -> bounded passive event journal

paired-pose capture candidate
  -> unchanged R0 dataset parser
  -> diagnostic fit
  -> explicit immutable diagnostic report (candidate-only)
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
`control_station/quest_calibration/` owns typed operator intents, application
lifecycle admission, source-selection translation, UI revision sequencing, and
a pose-redacted status projection. Its local Web coordinator waits through the
unique factory's public application port and owns only the HTTP server,
attachment thread, and rebuildable presentation state. It does not construct
or close the lower factory/application, or own a worker, reducer, pose stream,
artifact writer, fit, acceptance, frame, or renderer. Its cancel intent is the
only stale-revision exception so cleanup remains reachable while a prior action
is in progress. Admitted attempts consume their client sequence even on lower
failure, preventing ambiguous same-intent retries. It must not call a
refreshing robot snapshot API.

The detailed R2 source and evidence contract is documented in
`hardware/quest_dev/docs/features/paired-pose-capture.md`.  The reducer is a
passive in-memory component: it consumes immutable source events and owns no
source runtime, background device worker, persistence service, or operator
lifecycle.

`control_composition/` continues to own backend-neutral source/future semantics.
It must not import Quest, hardware, Control Station, or a concrete policy
deployment.  A concrete application leaf will adapt an already-selected future
trajectory into the accepted visualization frame and publish it passively.

The R4a attestation is owned beside the concrete RM65 position backend.  It
does not contain `control_domain_id`, component or stream identity, receipt
clock, observation capability/binding, observation plan, or pose schema.  Those
facts are known only by the application composition root and continue to enter
the one canonical `PoseCoordinateBinding`.  The impedance backend has a
different tool/FK lifecycle and is explicitly outside this slice; it must fail
closed for this Quest pose-publication path until it receives equivalent
attestation support.

R4b is owned by `teleop_runtime/deploy/`: the concrete adapter validates the
RM65 proof and returns an immutable side-local projection/state-read bundle;
`run_policy` owns the exact receipt clock and passes that bundle into the one
reader.  `ArmStateReader` remains concrete-backend-neutral and imports neither
RM65 nor Quest.  It owns only refresh cadence, causal history, and the bounded
passive journal.  The callable replaces its existing one state read rather than
adding another read, and no wrapper or second backend/resource facade is
created.

The typed reference frame is the canonical `CoordinateFrameSpec` for
`<side>_arm_base`: fixed-base Pinocchio world and the exact URDF root are equal
under `T_reference_frame_from_pinocchio_world = I`, with right-handed URDF axes,
metres, and radians.  Tool-TCP proof is available only when the actual solver
frame is `hand_base`; an `arm_link6` fallback may continue to serve the existing
diagnostic FK behavior but cannot publish this attestation.  Solver provenance
failure is likewise optional-proof failure and does not replace or break the
legacy FK owner.  The backend's construction-time default controlled frame must
also be `tool_tcp`, so the existing no-argument FK consumer cannot accidentally
pair an arm-endpoint pose with a tool-TCP proof.

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

- complete capture-evidence semantic hash and outer diagnostic-report semantic
  hash;
- derived paired dataset hash and nested fit hash;
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
- The R4b adapter and passive journal do not send robot commands, but the
  surrounding ordinary policy-deploy RM65 startup is not proven observation-only
  and may perform its pre-existing configuration operations.
- No module in this path starts, homes, servos, replays, holds, or commands a
  robot solely on behalf of pose observation; the existing deploy resource owner
  remains the only backend lifecycle owner.
- Reading the robot pose journal or latest publication never refreshes a backend.
- A source failure clears the published latest pose before downstream sealing.
- Source callbacks only enter bounded in-memory evidence.  The reducer,
  callbacks, ordinary polling, fit computation, and ordinary close do not
  write artifacts; only explicit application-level publish calls write the
  immutable capture or candidate diagnostic-report files.
- A gap, reorder overflow, identity drift, source invalidation, clock mismatch,
  non-atomic close attempt, or incomplete asynchronous drain prevents sealing.
- Visualization never grants control authority and never weakens canonical
  target safety or dispatch checks.
- Offline/mock tests prove contracts, not motion safety, real-time performance,
  calibration quality, or hardware readiness.

## Next implementation slices

The unique-running-runtime host, RM65 lower attestation, policy-deployment
robot-side canonical binding, default-off application bridge, explicit capture
store, explicit candidate diagnostic-report store, and thin programmatic
Control Station operator lifecycle are now implemented.  The Station leaf is
now bound to a local operator Web UI and the unique deploy application through
the factory's public wait port, without adding a second device/runtime owner.
An opt-in Control Station launcher now provides a closed-over `hold`, one-arm,
no-action, no-end-effector, no-camera composition plus a thin shell wrapper. It
prints the Quest connection details, opens the local operator page, and offers
an explicit trusted-LAN easy pairing code while retaining protocol
authentication. It is still an operator trial rather than a canonical policy
deployment entrypoint and has not been exercised against real Quest + RM65
hardware. Its stationary-window profile is candidate admission only; `seal`
remains in-memory by default, while explicit launcher publication flags can
write candidate evidence and a diagnostic fit without accepting or installing
a frame. The remaining ordered slices are:

1. Exercise the one-command operator trial under the real-hardware safety
   procedure, while preserving policy deploy as the sole factory/application
   shutdown owner and recording the ordinary backend startup effects.
2. Collect real stationary multi-pose Quest + RM65 evidence, including remount,
   noise, and held-out/leave-one-window-out validation.
3. Review residual and observability policy, then define a content-bound
   acceptance profile/receipt and invalidation lifecycle.  A persisted
   diagnostic report alone must never satisfy this step.
4. Only from a valid reviewed acceptance receipt, install the accepted transform
   into a read-only visualization-frame binding with reset/remount/tool/source
   invalidation.
5. Project the schema-aware selected policy future plus measured TCP through
   that accepted frame, publish a bounded passive protocol, and render it as
   read-only Quest geometry.

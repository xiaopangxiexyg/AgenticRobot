# Control Station architecture boundary

`control_station` is the application leaf that composes existing human,
replay, policy, dataset, sim, and real-target modules. It does not replace the
canonical teleop, replay, collection, or deploy entrypoints and lower-level
packages must never import it.

Reusable multi-source composition semantics are migrating to
`control_composition`. Station remains responsible for application operations,
side-effect coordination, publication barriers, GUI/API state, and binding
concrete source/target adapters. It must not duplicate composition strategies
or canonical motion authority.

## Reuse model

Keep extension work on the narrowest stable seam:

| New capability | Reuse | Add only when required |
|---|---|---|
| Robot arm/backend | RobotMotion authority, gate, epoch, cancellation and Station target lifecycle | Thin backend runtime/dispatch adapter and capability evidence |
| Teleop device | `HumanCommandEmission`/human proposal seam and existing Station source lifecycle | Device-to-emission adapter; no new Arbiter or target composition |
| Simulation environment | RobotMotion authoritative runtime and generic Station target adapter | Sim measurement/execution adapter |
| Model/checkpoint/server | `PolicyActionEmission` or RTC chunk protocol | Observation/action representation adapter only when the model contract differs |
| Replay dataset/profile | Replay emission/proposal source and Station replay binding | Dataset reader/compatibility assessment, never a second cursor in Station |
| Mixed-source handoff | `request_human_takeover` plus the pure handoff receipt reducer | One operation executor binding; never a reducer or coordinator per source pair |

### Operation preparation contract

`SideEffectStationOperationCoordinator.operation_prepare` is called for an
intent whose receiver receipt is `operation_requested`, after the runtime has
created its pending operation and before the request enters the worker queue.
The only non-operation exception is the explicit lower-safety seam for
accepted `hold`/`estop`; lease acquisition/release and `stop_runtime` do not
invoke this hook because they do not own a pending operation.

If preparation for a pending operation raises an ordinary `Exception`, the
coordinator admits a typed failed operation result
(`operation_prepare_exception:<ExceptionType>`), which resolves and removes
the pending operation, releases reserved capacity, and returns
`operation_prepare_failed`. It does not run the adapter or publish a queue
item. Preparation failure on the non-operation safety seam remains an
exception because fabricating a completion for an intent with no pending
operation would break the receiver contract. Coordinator/lifecycle bugs that
prevent result admission also remain exceptions.

The reusable target lifecycle lives in `targets/authoritative.py`. Concrete
MuJoCo validation and manifests remain in `composition/mujoco_*.py` because
they bind a source to one exact target profile. Generic source evidence lives
in `sources/` and must not import `sim` or `hardware`.

Do not create one composition module per checkpoint, teleop device, robot arm,
or task configuration when the typed source/target seam is unchanged. Prefer
data-driven manifests and capability hashes. Introduce a new module only when
it owns a genuinely different lifecycle, authority boundary, or protocol.

## Mainline mixed-action representation

Human, replay, and policy arm components use pose at the mainline mixed-source
boundary. The selected and canonical-safe arm action therefore remains pose. A
target may subsequently derive an arm-qpos execution target through the
verified frame-transform, coupled-IK, reference, and planner pipeline; that
derived qpos is not an arm mixed-source contract. Grippers and q6/q16 hands are
different components: they retain typed qpos/native continuous values and will
use a separate mask-aware interpolation strategy before compound action
assembly. Existing direct arm-qpos human/replay mixed compositions are retained
only as scoped debug/contract fixtures. Direct arm-qpos mixing is deferred until
the planned homogeneous-arm profile.

The numerical/source-mixing rule is a replaceable, pure strategy physically
owned by `control_composition.composition`; it is not a branch embedded in a
device, model client, simulator, or Station reducer. Its input is one immutable
decision-cut snapshot containing admitted arm-pose candidates plus optional
history, future-action, modality, label, and policy-context facts. Every
context carries source identity/hash, receiver availability time, admission
sequence, clock domain, and an immutable semantic payload. A strategy may
return a pose decision and lineage only; it cannot mutate authority/epochs,
consume a queue, run safety/IK, or dispatch.

CS-6b1d currently runs `discrete_authority_owner_pose_v1` as a read-only shadow
beside the existing ActionArbiter and records selection/output-pose parity. It
does not influence execution. This first strategy intentionally ignores all
optional contexts and preserves current discrete handoff behavior. A later
history/future/modality/label strategy is plugged into the same seam, but may
become authoritative only after a separate versioned proposal/admission,
ControlTrace, safety, latency/fallback, and shadow-parity gate. Unversioned
pose averaging or GUI-controlled blend weights remain forbidden.

The shared pose strategy/composer now accepts an explicit non-empty subset of
the current v1 arm components (`left_arm`, `right_arm`). A one-arm profile
projects every candidate to that declared subset before strategy evaluation;
the decision and composed action must contain exactly the same components.
The omitted/default declaration remains the historical coupled bimanual
contract and keeps its strategy/composition spec hash and behavior. This is a
topology-manifest improvement, not permission to make the current CS-6b1i
MuJoCo target single-arm: that concrete profile remains bimanual and must fail
closed until a target binding advertises a matching resolver/safety capability.
Arbitrary named arms require a later component-schema version.

`control_composition.observability` owns the non-authoritative observation over
one exact composition snapshot/result. An explicit `SourceDifferenceSpec`
binds the composition spec hash and declares comparable `(compared, reference)`
role pairs plus source-to-output residual roles; the runtime never infers that
`base` and `residual`, or any other same-shaped payloads, are semantically
comparable. Cartesian pose uses signed `reference - compared` translation plus
an SO(3) rotation vector; qpos uses only jointly finite slots. Action-space,
representation, shape, or pose-frame mismatch fails explicitly.

`ControlCompositionRuntime.evaluate_composition()` now accepts that optional spec,
returns the observation on `CompositeProposalSubmission`, and emits the exact
hash-linked `source_difference_spec` and `source_difference_observation` v2.4
ControlTrace facts through its existing backend-neutral trace sink. Diagnostic
failure is returned separately and cannot rewrite the
composition result. CS-6b1h/1i only forward the opt-in spec and expose the same
observation on their step result. With the default `None`, no observation,
extra trace fact, clock read, or legacy binding-hash field is added.
`guidance_feedback.py` can map one configured pose pair to bounded dimensionless
guidance cues with TTL and lineage. The only adapter currently provided is
deterministic no-I/O validation; commands are typed
`guidance_haptic_non_force`, carry `authoritative=false`, and cannot move a
target or reflect physical force. Real haptics, GUI visualization, automatic
HOLD clearing, qpos-to-guidance mapping, and production profile wiring remain
separate gates.

The same module now contains a pure hand/gripper qpos component seam for
DMTAC-style 1DoF slots, q6 hands, and q16 active-mask slots. It interpolates
only active continuous qpos values, preserves q16 padding semantics, and
returns immutable lineage; missing weighted candidates fail closed by default.
`assemble_compound_component_action` only performs a strict structural join.
`evaluate_component_mixing_shadow` now evaluates the arm pose branch and any
explicit hand branches from the same immutable arbitration cut and emits a
non-authoritative, hash-linked decision. `project_component_mixing_shadow`
records that preview as an explicit `ControlTrace` fact, and a sim-only formal
fixture exercises the existing subordinate `EpisodeTraceRecorderSink` across
HOLD/no-action, abort, restart, and writer-failure cases. Neither the shadow
nor its preview action is admitted to ActionArbiter, canonical safety,
reference scheduling, Station authority, sim dispatch, real dispatch, or
deploy. Enabling hand interpolation later requires a versioned hand
spec/joint-order/units/mask/bounds/safety contract and the same causal cut as
the arm decision; a production compound action still needs a downstream
component admission/receipt gate.

Multi-source operations are bound by source slot IDs and an immutable operation
spec ID. The generic reducer never imports or branches on a concrete replay
reader, teleop device, model family, robot vendor, or simulator. A concrete
composition resolves the operation spec and returns aggregate lower-runtime
evidence through the existing side-effect coordinator.

Canonical human takeover readiness is exposed by `sources/human.py`. It wraps
an already-produced `HumanCommandEmission` and lower deadman/rebase evidence;
it creates no device reader or worker. Long readiness acquisition runs outside
Station admission. The short readiness-to-READY finalization uses a pending
operation CAS, so a prior HOLD/ESTOP prevents lower READY rather than merely
rejecting a completion after the lower runtime has already reopened.

The same module now provides the narrower
`PassiveHumanCommandEmissionIngress` and
`PassiveHumanCommandObserverFanout`. They are application-root wiring aids,
not another teleop source: the fanout preserves the exact lower-owned emission
object and isolates passive subscribers; the ingress uses a latest-preserving
bounded queue and waits only from the Station side-effect worker. The ingress
does **not** implement `HumanTakeoverReadinessProvider`. The current canonical
real-teleop path has no typed post-takeover deadman-rearm or retargeter-rebase
receipt, so production human takeover must remain fail-closed until both
receipts exist and can be correlated with the same operation boundary.

`sources/teleop_pose.py` adds a narrower device-neutral observation boundary.
It composes arbitrary `ObservationSource` and pose `Retargeter` instances into
an existing `TeleopSession(rig=None, recorder=None)`, forbids recording and
hardware-arm eligibility, and accepts exactly the finite pose effectors declared
by `PassiveTeleopPoseSourceSpec` (the default remains the historical bimanual
RM65 tuple). CS-6b1d may bind this source to the human slot while replay remains
the only active authority. A typed Station source-connect operation may start
the supplied tracker source, but no command can be dispatched and no human
proposal is admitted by this passive step. The module deliberately imports no
VIVE, robot, simulator, or model implementation; application profiles remain
responsible for constructing their concrete source and retargeters. The public
`teleop_runtime.configs.build_bimanual_passive_session` profile now provides
that projection for the current real setup: it reuses the canonical
`teleop_runtime.collection.run_collect` source/retargeter/calibration
construction while forcing `rig=None`, `recorder=None`, and no hardware-arm
eligibility. The Control Station must not copy device defaults or import
private `_build_*` helpers. A single-arm or other compatible pose topology uses
`PassiveTeleopPoseSourceSpec.for_arm/for_effectors` with the same lifecycle;
the concrete target must independently advertise matching component and
execution capability evidence. Construction is passive; starting a source
still requires an explicit caller lifecycle operation.

The same module now exposes the opt-in
`PassiveTeleopComponentSessionSource` for CS-6b2a. It reuses that exact
non-dispatching session lifecycle but validates a data-driven tuple of
`PassiveTeleopComponentSpec`: arm components must be typed Cartesian poses and
hand/gripper components must be canonical full-slot qpos with an explicit
`HandActionSpec`. A source may therefore be arm-pose-only, hand-qpos-only, or
pose+hand without adding a device-specific Station module. The bound source
spec supplies arm/hand specs to the later proposal adapter; command metadata,
when present, must match it exactly. Q16 inactive slots must be NaN and
zero-padded gripper slots must remain zero. The historical pose-only class is
unchanged semantically and still rejects every extra hand/qpos component.

Hand-only frames do not infer a whole-action representation from their payload.
`HumanProposalAdapter` binds the output action space to its
`ActionComponentCatalog`, so a hand update in a pose+hand profile remains in
that profile's action space. Incompatible arm-pose/qpos catalog bindings fail
closed. This gate is still passive: it does not run component mixing, proposal
admission, safety, reference scheduling, dispatch, or capture commit.

This source uses the current action-v1 topology (`left/right` arm/hand and
q6/q16 hand slots). Its construction is reusable across devices, model
producers, simulator backends, and compatible embodiments, but it is not a
claim that arbitrary effectors or dimensions are already supported. Those
require the planned generic component-action bridge and schema-visible,
payload-opaque core rather than more v1 shape branches here.

## Opt-in CS-6b1f sim HIL capture

`build_scoped_cs6b1f_mujoco_pose_human_replay_station()` is the first formal
human-in-the-loop capture slice. It is an explicit MuJoCo composition, not a
new entrypoint and not a change to the legacy teleop, replay, deploy, or RTC
paths. Replay remains the initial authority; a connected passive pose source
is only observed until the typed `request_human_takeover` transaction reaches
`human_takeover_ready`. In the sim fixture,
`VirtualSimHumanTakeoverReadinessProvider` supplies virtual deadman/rebase
evidence and waits on a post-boundary immutable emission. The emission is
promoted exactly once through the existing human publication seam, while the
arm remains pose-native until the target's verified pose-to-qpos resolver.

The profile
`deployable_sim_scheduled_pose_human_replay_station_v1` reuses the existing
Station operation coordinator, publication barriers, authoritative MuJoCo
runtime, ControlTrace/WAL, and external `EpisodeTraceRecorderSink`. The
composition owns no action queue, replay cursor, producer/authority/dispatch
epoch, or episode UUID. `hold_before_unbind` is enforced before recorder
release, and the committed reader rebuilds the same pose decisions and
resolved execution lineage.

The virtual provider fails closed unless the takeover is already awaiting
readiness: no post-boundary sample, a stale queued sample, or a late worker
completion leaves the station in HOLD and cannot reopen the human barrier.
These cases are covered by the HIL corner regressions; real readiness remains a
separate provider gate.

## Opt-in CS-6b1i composition session

`control_composition.lifecycle.CompositionSessionRegistry` owns the
backend-independent logical-session spec registry, per-activation epoch/state
transitions, re-arm binding, and trace observers. Station's
`CompositionSessionController` subclasses it only to bind typed Station
operation results, the publication barrier, and lower HOLD/cancel ordering.
`build_scoped_cs6b1i_mujoco_pose_composition_session()` is only the first
MuJoCo binding around the existing CS-6b1h data plane: it supplies concrete
readiness, lower HOLD/cancel, adapters, input admission, and sim stepping.
Human/replay slots are explicit
contributors and never parallel proposal owners; only the composite slot may
become active after `start_composition_session` returns the evidenced
`composition_session_ready` status.

Pause, ESTOP, stop, or a non-publishable composition closes the composer
publication barrier before lower HOLD/cancellation. Fresh input cannot reopen
the held activation. The opt-in MuJoCo re-arm operation keeps the logical
session identity but retires the held activation and creates the next epoch
only after typed cancellation, measurement, producer/authority epoch, READY,
and fresh-input-barrier evidence. Old proposals and contributor inputs cannot
cross that boundary, and re-arm itself admits no action. Trace contracts cover
the re-arm spec/receipt and activation transitions. This remains deterministic
sim only: no GUI, device start, replay cursor, policy/RTC source, real binding,
or production entrypoint is added, and no physical deadman/retarget evidence is
claimed.

The reusable boundary is intentional: a new tracker only supplies the public
passive-session profile, a new target supplies the shared authoritative target
adapter plus capability/manifest evidence, and a new model/replay source
supplies its existing immutable emission. Only a genuinely different
lifecycle or protocol should add a composition module; do not create one per
device, checkpoint, or simulator.

## Ownership rules

- `robot_motion`: backend-neutral authority, admission, execution gate,
  cancellation, safety and epoch semantics.
- `sim`: simulation state measurement and execution implementation; it never
  imports Control Station or real teleop code.
- `control_station/sources`: passive source bindings and source lifecycle; no
  target execution and no concrete sim dependency.
- `control_station/targets`: reusable application-level target lifecycle over
  the shared authoritative runtime; no vendor backend.
- `control_station/composition`: exact source/target/profile wiring only.
- `control_station/preflight`: read-only assessment of already-created source
  or target evidence.

Authority epochs in Station manifests are sourced canonically from the
RobotMotion `AuthorityStore`. `DispatchOwnerRegistry` is the synchronized
permit-enforcement mirror and has its own `owner_registry_id`.

## Periodic architecture gate

Run this review after each mixed-execution milestone, before RTC merges, and
before adding a new real target:

1. Lower packages have zero imports of `control_station`.
2. `sources/` and `targets/` have no concrete `sim` or `hardware` imports.
3. Composition modules do not import private symbols from sibling modules.
4. Direct `runtime.core.<store>`, trace slicing, and backend scene access are
   either removed or explicitly scoped as a temporary fixture.
5. Every queue, cursor, producer epoch, authority epoch, execution epoch and
   dispatch owner has exactly one owner.
6. A new device/model/backend reuses the existing typed seam unless its public
   semantics actually differ.
7. Modules approaching 800 lines receive a cohesion review, but are split only
   when the resulting seam is reusable and independently testable.
8. Run `tests/test_control_station_contracts.py` and all Control Station tests.
9. Adding a source/target type must not add another authority store, operation
   coordinator, publication barrier, queue, cursor, or episode owner.

The current mixed MuJoCo Station wrapper remains beside its existing CS6b0
data plane in `composition/mujoco_human_replay.py`. This large module has been
reviewed for cohesion; it should be split only when a second target proves a
backend-neutral executable seam, not merely to create per-device files.

Compatibility re-exports are kept during migrations so existing entrypoints
and tests do not need synchronized rewrites.

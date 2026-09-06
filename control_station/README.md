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

Formal composition start/re-arm creates a backend-neutral
`CompositionActivationScope` in `control_composition`; Station supplies the
operation/session identity and concrete readiness timestamp, but does not
implement a second input-freshness reducer. Contributor admission, snapshot
selection, result identity, and trace lineage consume the same scope.
The concrete MuJoCo migration profile still has one isolated start-time
barrier for callers that have not created an explicit scope. It is an unscoped
compatibility shim, is not reusable Station semantics, and must be deleted once
all formal callers establish `CompositionActivationScope`.

## GUI/control-plane boundary freeze

Control Station is the GUI-facing application control plane, not a
presentation-only package and not an action data plane. Its stable core owns
typed operator intents, application lifecycle state, operation coordination,
rebuildable UI status/events, declarative run profiles, and thin bindings to
already-constructed source/target ports.

The stable core must not implement or import concrete teleop, replay, policy,
RTC, simulator, hardware, numerical-composition, or dataset-writer behavior.
Those owners expose typed emissions, receipts, status, and lifecycle ports;
Station selects a profile and coordinates operations over those ports.

The concrete `refinement_annotation/` application follows the same rule for
post-capture data. Station owns the loopback HTTP/UI lifecycle and operator
interaction. Finalized-Episode discovery and stage/training-index resolution
remain in `dataset_interface`; draft and immutable annotation-set persistence
remain in `collection_runtime.progress`. The application never starts a
device, owns an Episode, rewrites raw data, or becomes a second schema source.
It starts without dataset arguments via
`python -m control_station refinement-annotation`; dataset absolute path and
annotator identity are selected in the loopback GUI before the lower-owned
annotation workspace and writer lock are opened. CLI values remain optional
GUI prefill only.

Within that leaf, `refinement_annotation/detail_annotation.py` is the reusable
optional-overlay capability: it discovers compatible detail profiles, selects
only immutable overlays bound to the current coarse set, and projects
`PENDING`/`STALE`/`COMPLETE` status. `episode_status.py` converts coarse review,
whole-Episode abnormal, draft, and detail state into a versioned additive UI
projection. The application still owns locks and commit coordination; these
modules do not define schemas or persistence and old scalar response fields
remain available for compatibility.

Current concrete-profile and migration debt is frozen as follows:

| Current Station area | Long-term owner |
|---|---|
| action/context selection, TTL, fallback, difference, handoff/session semantics | `control_composition` |
| human admission, passive teleop session, human emission fanout | `teleop_runtime.action` (implemented; Station paths are compatibility/profile seams) |
| replay binding/fanout/cursor-owned status | `teleop_runtime.replay` / `robot_motion.replay` |
| sync/RTC policy binding and step validation | `teleop_runtime.deploy` / `robot_motion.deploy` |
| OpenPI artifact/protocol assessment | `deployments.openpi` |
| MuJoCo measurement, readiness, resolution and execution evidence | `sim.mujoco` |
| authority, safety, IK, reference and dispatch | `robot_motion` |
| ControlTrace/v2.4 projection | `dataset_interface` |
| ControlTrace WAL/session capture | `teleop_runtime.control_trace` (until the whole capture runtime can move without a forbidden reverse dependency) |

`control_station/composition/` is therefore a transitional name. After the
concrete algorithms above have returned to their owners, the remaining thin,
GUI-selectable application assemblies should become declarative
`control_station/profiles/`; they may import multiple lower packages but may
not recreate their queues, workers, cursors, epochs, safety, or dispatch.

The default-off Sim-HIL application keeps
`control_station.sim_hil` as its stable CLI/facade. Its internal
`control_station/sim_hil_profile/` package is split by application concern:
`config.py` validates profile selection without constructing resources,
`human.py` binds lower-owned passive VIVE/gripper producers to the selected
component contract, `factory.py` assembles the concrete MuJoCo graph once,
`application.py` owns the core lifecycle, `application_human.py` owns the
human-attachment/runtime-Home sub-lifecycle, and `application_status.py` plus
`status.py` contain read-only GUI projection. `validation.py` is intentionally
limited to small profile-local scalar/text validation. `control_cycle.py`
defines the typed application seam between one selected composition profile
and the lower-owned execution result. `allocation_control_cycle.py` provides
the separate activation-scoped port for allocation-aware publication. Its
profile-owned binding must supply the exact roles, anchor, source admission and
allocation basis; the port does not infer them. The fixed-alpha profile remains
an exact compatibility adapter over the existing cycle and does not inherit
the allocation port's lifecycle. Activation priming is an explicit port
capability: the compatibility port retains application-owned legacy priming,
while the allocation port activates its scoped binding first and then primes
only through that binding's injected source publisher.
The CLI runner remains in the stable facade; none of these modules owns device
workers, composition algorithms, canonical safety, IK, dispatch, or dataset
persistence.

### Agent Gateway compatibility boundary

Agent Gateway protocol/runtime/transport, Skill definitions, executors, and
concrete MuJoCo deployments are owned by the top-level `agent_gateway` package.
Control Station retains only its application ports and the historic
`python -m control_station agent-gateway-mujoco` forwarding command. See
[`agent_gateway/README.md`](../agent_gateway/README.md) for the canonical
package layout, `python -m agent_gateway mujoco` entrypoint, published Skill
surface, and lower-owned safety/HOLD boundary.

The standalone `python -m control_station vive-deflection` application is a
read-only VIVE input viewer. It opens one VIVE pose receiver and a loopback
GUI, loads the latest source-compatible Control Station VIVE coordinate
profile by default, auto-captures the first fresh calibrated pose as Viewer
Home, and supports
explicit recenter. Its fixed display axes follow the calibrated retarget-frame
contract: `+X` forward, `+Y` left, and `+Z` up; the six signed endpoints are
labelled in both 3D panels. A profile with an explicit cardinal post-yaw is
labelled by its resulting physical signed axes rather than being falsely
presented as canonical; for example, a signed-sweep profile with `180 deg`
shows `-X` forward and `-Y` left and carries a visible warning. Newly captured
canonical profiles use `0 deg`. `--raw-vive-frame` is an explicit debug fallback
and deliberately makes no front/back/left/right claim. It does not create
MuJoCo, a policy, HIL composition, authority, recording, or any robot execution
target. Its retarget-frame translation and Home-local-rotation projection is
passive observability; Viewer Home is not a teleop Runtime Home or a
composition residual anchor. A finite positive-
determinant raw VIVE rotation may be projected to the nearest SO(3) only for
axis-angle display; this never rewrites the source sample or any control path.
The same GUI also contains a demo-only three-dimensional trajectory panel with
two selectable SI-unit projections. The default `velocity_joystick` mode maps
normalized Home translation to metres per second and integrates an explicit
browser-local offset using elapsed display time; returning Home stops further
integration and preserves the offset, while clear or successful recenter resets
it. The compatible `position_offset` mode applies the current Home translation
directly in metres with a dimensionless gain and carries no accumulator. Both
the nominal and corrected trajectories remain live at 30 Hz, while the demo
parabola uses a 2.4-second bounce period. The panel is an
operator experiment, not a composition result, executable Reference,
coordinate-frame adapter, or command producer.
The demo parabola and its floor grid are authored in physical
`forward/left/up` coordinates and mapped into the active profile's signed-axis
basis before rendering. Human correction remains in that already projected
profile frame and is added exactly once.

`重新标定坐标系` opens the existing Control Station VIVE calibration GUI on a
free loopback port. The calibration workflow shares the viewer's already-open
VIVE receiver, so it neither binds the pose port a second time nor takes over
the receiver lifecycle. It is restricted to the viewer's selected wrist side.
Completed profiles are saved by the existing `save_coordinate_profile()` path
under `teleop_records/control_station_vive_coordinate_profiles`. After a save,
the viewer applies that exact artifact, resets Viewer Home, and clears the
browser-local trajectory accumulator; no control/runtime Home is changed.

The boundary tests keep current migration debt explicit: dataset projection is
confined to `control_trace.py` and the default-off
`composition/mujoco_sim_hil.py` profile, OpenPI knowledge to
`preflight/openpi_rtc.py`, and MuJoCo imports to concrete active profiles.
Device-free synthetic Human production, receiver admission, passive
ingress/fanout, and the observe-only `TeleopSession(rig=None)` source now live
in `teleop_runtime.action`; their Station paths are compatibility/profile
seams only. The former CS-2b/2c/2d, CS-3 passive-shadow, and CS-6 replay/pose
composer fixtures have no current executable entry and were removed from the
versioned tree. Their audit and local recovery copy are recorded in
[`docs/legacy/control_station_profile_retirement_20260906.md`](../docs/legacy/control_station_profile_retirement_20260906.md).
New debt may not spread while the remaining concrete profiles are migrated.

## Reuse model

Keep extension work on the narrowest stable seam:

| New capability | Reuse | Add only when required |
|---|---|---|
| Robot arm/backend | RobotMotion authority, gate, epoch, cancellation and Station target lifecycle | Thin backend runtime/dispatch adapter and capability evidence |
| Teleop device | `HumanCommandEmission`/human proposal seam and the lower-owned source lifecycle | Device-to-emission adapter plus a thin Station profile binding; no new Arbiter or target composition |
| Simulation environment | RobotMotion authoritative runtime and generic Station target adapter | Sim measurement/execution adapter |
| Model/checkpoint/server | `PolicyActionEmission` or RTC chunk protocol | Observation/action representation adapter only when the model contract differs |
| Replay dataset/profile | Replay emission/proposal source and Station replay binding | Dataset reader/compatibility assessment, never a second cursor in Station |
| Exclusive Human/Policy handoff | `control_station.hitl` switch and activation-evidence contracts | One sim/real lifecycle port; never a transaction, reducer, or coordinator per backend/model/robot |

`control_station/hitl/` is the reusable application-lifecycle seam for
exclusive Human/Policy switching. Exclusive control itself is a restricted
profile of the common streaming allocation cycle (`base_only` or
`absolute_overlay_owner` for the complete component set), not a fixed-alpha
or backend-private composition path. The lifecycle seam knows only source roles, HOLD evidence,
Runtime Home evidence, source-selection identity, activation identity, and one
fresh target-source emission. A switch is not reported ready at that point:
the concrete lifecycle port must also return the existing downstream target,
target-admission, completion, command, and dispatch receipts from one formal
cycle. These facts remain lower-owned and are linked through `evidence_refs`;
the HITL seam does not reconstruct them. It imports no simulator, hardware,
model client, teleop runtime, or ControlTrace projection and contains no
component-name or arm-cardinality assumptions. `sim_hil_profile/exclusive_hitl.py`
is the current concrete MuJoCo application adapter; a real deployment must add
its own thin adapter before the existing single safety/IK/reference/dispatch
owner.

Control and recording remain one-way coupled.  `control_station.hitl` may call
a best-effort evidence observer after building a result, while
`control_station.control_trace` owns the passive projection.  A trace sink
failure is reported by capture health and may invalidate dataset finalization,
but cannot synchronously change the switch result or grant/revoke motion
authority.  A capture-required application may separately translate faulted
capture health into an explicit typed HOLD request; that policy remains outside
the trace projector and carries its own visible reason/evidence.

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
they bind a source to one exact target profile. Generic source production,
admission, evidence, and passive observation live below Station;
`control_station/sources/` contains only takeover-readiness and thin
compatibility/profile bindings and must not import `sim` or `hardware`.

Do not create one composition module per checkpoint, teleop device, robot arm,
or task configuration when the typed source/target seam is unchanged. Prefer
data-driven manifests and capability hashes. Introduce a new module only when
it owns a genuinely different lifecycle, authority boundary, or protocol.

### Future-trajectory display ports

Station trajectory-preview ports are backend-neutral, default-off, read-only
demand gates. The intent port may derive a source-intent view from one already
published `ComponentActionTrajectory`. The corrected port accepts one
lower-owned `TrajectoryCorrectedPreviewSnapshot`; the supplier remains
responsible for its coherent non-closing composition cut. Neither port admits
inputs, advances a source cursor, publishes proposals, or sends actions.

Freshness evidence is fetched only when the selected profile declares a
`TrajectoryPreviewFreshnessSpec`. This keeps observation/modality acquisition
demand-driven. The assessment annotates display age only and cannot change the
producer-locked/speculative class or turn a conditional preview into an
executable Reference.

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

The generic pose strategy can run as a read-only shadow beside the existing
ActionArbiter and record selection/output-pose parity. It does not influence
execution. The strategy intentionally ignores optional contexts and preserves
the established discrete handoff behavior. A later
history/future/modality/label strategy must use the same seam and pass a
separate versioned proposal/admission, ControlTrace, safety, latency/fallback,
and shadow-parity gate before it can become authoritative. Unversioned pose
averaging or GUI-controlled blend weights remain forbidden.

An opt-in formal-strategy-v2 contract fixture can inject the strategy into
`MujocoDeployRuntime` rather than a sim configuration. Its binding carries the
strategy-spec hash under a v2 binding domain, and the deterministic migration
adapter preserves pose/action parity on one frozen input snapshot before the
existing single authority, safety, reference, and dispatch chain. With no
injected strategy, the established v1 binding and behavior remain unchanged.
This is not a default operational profile or permission to run a slow model/RPC
in the control loop.

The shared pose strategy/composer accepts an explicit non-empty subset of the
current v1 arm components (`left_arm`, `right_arm`). A one-arm profile projects
every candidate to that declared subset before strategy evaluation; the
decision and composed action must contain exactly the same components. The
omitted/default declaration remains the historical coupled-bimanual contract
and keeps its strategy/composition spec hash and behavior. A concrete target
must still fail closed until its binding advertises a matching resolver/safety
capability. Arbitrary named arms require a later component-schema version.

The default-off `mujoco_sim_hil` application profile adds the first formal
human+policy compound binding for the existing MuJoCo pose execution target.
It accepts a lower-owned `PolicyActionEmission` and a passive
`PassiveTeleopComponentSessionSource`, converts both to non-authoritative
composition inputs, and promotes only one all-required composer proposal.
Policy inference/chunk ownership, teleop calibration/Home/rebase/deadman,
formal fusion, shared safety, RM65 IK, reference scheduling, and dispatch stay
with their existing lower owners. The profile does not alter the direct
`run_vive_mujoco_teleop.py` debug path or `policy_rollout.py`; it has no default
CLI activation and is not a real-arm motion permission.
An explicit readiness provider is required even for a virtual source; recovery
from HOLD uses an explicit `rearm()` activation and never auto-clears the
lower execution gate. Human/policy availability and policy-chunk TTL are
published through the generic source-eligibility store and consumed with the
actions on the same causal cut. A missing active contributor yields the formal
`no_action` disposition; Station neither substitutes another source nor
mutates the gate. The lower execution profile alone decides target persistence
or HOLD after no-action. Calibration, Home/rebase/deadman, and non-finite
measured-state failures request latched HOLD because they require operator or
lower-runtime repair.

`control_composition.observability` owns the non-authoritative observation over
one exact composition snapshot/result. An explicit `SourceDifferenceSpec`
binds the composition spec hash and declares comparable `(compared, reference)`
role pairs plus source-to-output residual roles; the runtime never infers that
`base` and `residual`, or any other same-shaped payloads, are semantically
comparable. Cartesian pose uses signed `reference - compared` translation plus
an SO(3) rotation vector; qpos uses only jointly finite slots. Action-space,
representation, shape, or pose-frame mismatch fails explicitly.

`ControlCompositionRuntime.evaluate_composition()` now accepts that optional
spec, returns the observation on `CompositeProposalSubmission`, and emits
exact protocol-neutral source-difference domain events. The injected dataset
adapter may project them into hash-linked v2.4 ControlTrace facts. Diagnostic
failure is returned separately and cannot rewrite the composition evaluation.
Application adapters only forward the opt-in spec and expose the same
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

The lower `teleop_runtime.action` data plane provides the narrower
`PassiveHumanCommandEmissionIngress` and
`PassiveHumanCommandObserverFanout`; `sources/human.py` only re-exports them
beside the Station-owned takeover-readiness contract. They are passive wiring
aids, not another teleop source: the fanout preserves the exact lower-owned
emission object and isolates passive subscribers; the ingress uses a
latest-preserving bounded queue and waits only from an application side-effect
worker. The ingress does **not** implement
`HumanTakeoverReadinessProvider`. The current canonical real-teleop path has no
typed post-takeover deadman-rearm or retargeter-rebase receipt, so production
human takeover must remain fail-closed until both receipts exist and can be
correlated with the same operation boundary.

`teleop_runtime.action.passive_teleop` owns the narrower device-neutral
observation boundary; `sources/teleop_pose.py` is its compatibility export. It
composes arbitrary `ObservationSource` and pose `Retargeter` instances into an
existing `TeleopSession(rig=None, recorder=None)`, forbids recording and
hardware-arm eligibility, and accepts exactly the finite pose effectors
declared by `PassiveTeleopPoseSourceSpec` (the default remains the historical
bimanual RM65 tuple). A passive pose source may bind to a human slot while
another source remains the only active authority. A typed Station source-connect
operation may start the supplied tracker source, but no command can be
dispatched and no human proposal is admitted by this passive step. The lower
module deliberately imports no VIVE, robot, simulator, or model implementation;
application profiles remain
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
`PassiveTeleopComponentSessionSource` for component-level inputs. It reuses that exact
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

## Retired CS-2/CS-3/CS-6 fixture profiles

The former CS-2b/2c/2d, CS-3 passive-shadow, and CS-6 replay/pose-composer
modules were test/application-only compositions. They introduced no canonical
teleop, replay, policy, RTC, authority, safety, IK, dispatch, or recording
owner, and they had no current production caller. Their versioned source and
fixture tests are therefore retired; no current command or public builder may
refer to those profile names.

Current applications use the lower-owned `PolicyActionEmission`,
`HumanCommandEmission`, `control_composition`, and default-off
`control_station sim-hil` seams described above. This keeps source selection,
HOLD/re-arm, safety, target resolution, dispatch, and capture ownership in one
active path instead of preserving parallel profile adapters. The deletion
audit, caller inventory, replacement mapping, and recoverable local copy are
recorded in
[`docs/legacy/control_station_profile_retirement_20260906.md`](../docs/legacy/control_station_profile_retirement_20260906.md).

Allocation-context publication has a separate backend-neutral, default-off
Station port. `StationAllocationPublicationPort` binds one
`AllocationPublicationRuntime` to the active typed composition-session barrier
and moves only the potentially blocking `CompositionContext` admission onto a
single application-owned worker. The control-facing call performs a lazy,
bounded read of one already-produced `AllocationPublicationBasis`; while one
exact attempt is pending, the reusable runtime retains only its latest newer
basis. Unknown admission outcomes stop rather than spin and require an
explicit exact-retry request.

Closing the port invalidates the allocation activation before joining the
worker, so a late receipt cannot commit reducer state. With the profile off it
does not create a worker or evaluate the basis supplier. The port has no
device/model/cursor, authority, safety, action publication, target, reference,
dispatch, or dataset ownership. Its latest status remains available to UI
observers after the bounded event cache is drained for trace projection.

`MujocoAllocationShadowController` is the concrete default-off D3 binding. It
connects that port to an already-started MuJoCo composition topology, evaluates
only through `evaluate_composition_shadow()`, and verifies that input and
allocation context snapshots share the exact same `ControlCut`. Pending
candidates are invisible; only a receiver-admitted context may select a mode.
The optional corrected-horizon view additionally verifies each allocation
context against the MuJoCo store before exposing it. The cycle is always
non-authoritative/non-promotable and has no proposal, target, Reference, IK,
physics-step, or dispatch method.

`MujocoAllocationProposalPublicationController` is the separate default-off
D4 gate. It never promotes a D3 shadow result: while the same typed Station
composition-session barrier is held, it captures a new receiver `ControlCut`,
performs one formal allocation-aware evaluation, and submits only that exact
ready evaluation through the existing MuJoCo composer proposal port. A missing
or expired context returns typed `no_action`; it does not manufacture HOLD or
fallback. The controller does not call `runtime.step()`, so an accepted
proposal is still neither authoritative nor executed and cannot by itself
create a target, Reference, IK result, physics command, or dispatch receipt.
Component coverage and selection groups are derived and validated from the
typed action/catalog; the Station gate does not assume two arms or one action
representation. Exclusive takeover remains a different typed lifecycle.

The first executable allocation profile, Scheme A, is deliberately
relative-only. Its component policies cap automatic selection at
`relative_overlay`, and its proposal allowlist contains exactly `base_only`
plus `relative_overlay`; it therefore does not manufacture an absolute
candidate and later block it. Scheme B is a separate explicit profile that
allows all three modes and enables the reusable target-transition continuity
policy. Both profiles assemble the same derived-input, allocation-context,
composition, proposal, and lower-cycle ports. The existing fixed-alpha public
factory and numeric behavior are unchanged.

Scheme B corrects only the composition-target seam. Entering absolute mode
aligns the current raw human target to the previous composed target and retains
that alignment while human deltas continue; leaving a mode begins at the
previous composed target and decays the seam correction over the profile-owned
target-time interval. Formal evaluation only prepares the correction; the
continuity state advances after that exact proposal admission is accepted, so a
rejected or expired evaluation cannot alter the next target baseline. The
evaluation records the prepared state/identity and correction receipt in
`strategy_result_data`. This is not a
claim that an execution Reference or measured command is continuous: target
resolution, planner interpolation, limits, IK, canonical safety, and dispatch
remain lower-owned. Read-only projected trajectory preview does not advance the
stateful continuity reducer and remains advisory/raw at this boundary.

Allocation publication outcomes have a compact generic
`allocation_publication_outcome` ControlTrace projection. Ordinary context
admission/context/evaluation facts remain the owners of full payloads. Atomic
human absolute/residual batch admission is also projected as actual-data
lineage without duplicating action vectors. The public factory selects this
path only for an explicit `policy_human_allocation_scheme_a` or
`policy_human_allocation_scheme_b` profile; the default remains fixed-alpha
compatibility.

The default-off MuJoCo policy/human allocation input profile now binds one
already-produced human sample to fixed `human_absolute` and `human_residual`
roles and admits them as one batch. It does not poll a device or infer topology;
component schemas, thresholds, normalization, custom metric providers, and role
names are profile inputs. Only accepted receiver stamps enter the reusable
time-window evidence builder. The explicit Scheme-A and Scheme-B Sim-HIL
profiles wire this adapter to the existing passive VIVE plus gripper source;
the adapter itself remains device-, topology-, and model-neutral.

Requesting a new anchor is now a target-fenced transaction. The Station input
controller admits `pending` before mutating its source adapter, advances the
allocation generation fence, and then waits for the first fresh complete
sample. That sample produces one atomic absolute plus identity-residual input
batch; only its accepted batch may publish the exact `ready` receipt. Rejected
batch or ready admission is retried exactly before a later sample is consumed.
At the declared boundary, missing/pending lifecycle yields typed `no_action`;
Station does not reinterpret it as HOLD or own the lower execution response.

Lifecycle, allocation context, and selected inputs are not sequential Station
objects. They are independently receiver-admitted facts captured under one
immutable cut by `control_composition`. Thus the protocol adds admission work
only when an anchor changes, not one extra control-period delay. Old-generation
allocation attempts are fenced, and their late receipts remain observable but
cannot commit reducer state.

The allocation-aware Sim-HIL cycle now also has a thin MuJoCo binding. A
profile callback publishes already-configured policy/human inputs and an
optional allocation basis; the existing D2 worker admits context asynchronously,
the D4 gate captures a later exact cut and may admit a proposal, and the binding
then calls the one lower `runtime.step()`. A `no_action` proposal status still
calls that lower step so existing proposal persistence and execution-gate
semantics decide the physical response. Source publication carries an explicit
post-admission receiver `observed_at`; basis, proposal and lower execution are
never evaluated against the earlier pre-poll timestamp.

The application distinguishes compatibility-owned and control-cycle-owned
activation priming. The fixed-alpha path preserves its original
prime-before-port-activation order. The allocation path creates the exact
activation-local binding first, resets/steps the required policy source, and
waits within the existing application timeout for one genuinely new human
emission after the activation barrier. It passes that immutable emission into
the binding's source-prime operation without publishing a proposal or running
a lower cycle; it never substitutes `last_emission`. A cycle cannot run until
that prime succeeds; re-arm must create and prime a fresh binding. Profiles
using allocation also reject the legacy fixed-alpha GUI control explicitly. A
cached device pose cannot cross the activation barrier. Allocation shutdown is
likewise explicit: the application closes the passive human source, while the
fixed-alpha compatibility composition retains its historical source-close
ownership.

`MujocoPolicyHumanAllocationSession` now supplies the application-facing
start/HOLD/rearm/close seam without constructing a source or target. Every
source readiness field is required for this first profile. Start and rearm
activate fresh receiver producer epochs and create a new
`CompositionActivationScope`; the shared publication barrier opens only after
an injected upper Station operation-result provider acknowledges the exact
profile, activation, preflight, producer receipts, and (for rearm) lower READY
gate evidence. Operation completion must be receiver-causal: its admission
time cannot precede those same-clock facts. Activation recording is an
injected required observer, so this session does not add a new dependency on
dataset projection or choose a v2.4 persistence implementation.

This session remains default-off. The public CLI selects Scheme A with
`--control-profile policy_human_allocation_scheme_a` or Scheme B with
`--control-profile policy_human_allocation_scheme_b`; omitting the option keeps
the fixed-alpha compatibility profile. Numeric allocation thresholds, evidence
window, TTL, dwell, and Scheme-B target handoff duration remain explicit
profile/integration inputs rather than generic defaults.

`StationCompositionSessionOperationPort` is the backend-neutral upper
completion seam for that lifecycle. One instance binds one already-admitted
Station start/rearm/stop request and completes it through
`ControlStationRuntime.finalize_pending_operation()`; it never admits a result
directly or creates another pending-operation state machine. Successful lower
evidence must expose its operation kind, complete evidence references, and the
latest causal readiness time in the Station receiver clock. Cross-clock
evidence requires a separate conversion receipt instead of timestamp guessing.
If lower preparation fails before such evidence exists, the same port can
resolve the pending request with a typed failed result. A Station HOLD, fault,
stop, or competing transition supersedes the request and the port cannot revive
it.

Sim-HIL profile selection treats `{application session lifecycle,
control-cycle port}` as one atomic application profile. This prevents a new
cycle implementation from accidentally inheriting fixed-alpha
start/HOLD/rearm/close behavior. Profile builders are lazy: registration is
resource-free and only the selected builder is invoked. The built-in/default
selection remains `fixed_alpha_compatibility`; explicit Scheme-A or Scheme-B
selection builds its session and cycle port as one bundle. Programmatic integrations may
inject another binding, but it must supply both the session and cycle port with
a matching declared cycle kind. Allocation thresholds, source/anchor wiring, and
Station intent source remain explicit profile data rather than hidden generic
defaults.

Pause and episode-end use two-phase worker teardown. The control-cycle port
first fences the activation and invalidates pending allocation work without
waiting for the worker, so the application can immediately request the one
lower-owned HOLD. Terminal application cleanup joins all retired bindings only
after that HOLD boundary. A fence error is reported, but it does not suppress
the lower HOLD attempt.

Residual `rebase-and-clear` has a separate default-off Station port. The
central reducer accepts the typed operator intent only for an active composite
source in a running application session. Its admission receipt binds the exact
intent hash, not only the sequence-derived intent ID. The thin
`StationResidualRebaseController` then checks that receipt, the selected
composer slot, profile ID, and current active-anchor binding before invoking a
lazy command supplier and entering the backend-neutral rebase reducer. With
the profile off, the supplier is not evaluated.

The reusable port does not read a button, poll a tracker, request HOLD, or
publish/dispatch an action. In either explicit allocation application, a GUI intent
is translated by the thin Station adapter into the barrier-bound anchor request;
the next fresh complete VIVE/gripper emission atomically supplies the new pose
and qpos anchor receipt. Downstream execution still decides what typed
`no_action` means physically while rebase is pending. Deploy, replay, teleop,
and real-robot execution paths are not changed by this binding.

The current default-off `python -m control_station sim-hil --enable` application
profile is the operational Station leaf for the temporary PPO plus one passive
VIVE tracker and one selected right gripper input (Q6 first, calibrated angle
trigger fallback). Its native composition schema is
`right_arm: cartesian_pose` plus `right_gripper: physical_joint_qpos`; the old
`RobotAction` is materialized only inside the current coupled MuJoCo execution
edge. The selected topology is an application profile, while the lower source,
component schema, and target binding remain cardinality-driven. The
operator commands, including the explicit single-arm allocation angle-trigger
profile, are maintained in [`docs/HIL/README.md`](../docs/HIL/README.md).
The ordinary `--gui` remains a read-only status surface showing readiness, source freshness,
composition input/proposal receipts, activation hashes, and execution
receipts. The explicit `--control-gui --enable` profile additionally exposes
a loopback-only typed control endpoint. Browser requests enter a bounded
intent queue; only the Station control thread starts passive devices, changes
blend parameters, steps MuJoCo, or finalizes ControlTrace. Neither GUI mode
accepts raw actions or owns authority/dispatch. The status API uses the
versioned `dexteleop.control_station.sim_hil_status.v1` projection; a failed
provider is rendered as an explicit GUI error rather than as a false HOLD.
When `--gui` or `--control-gui` starts the local page, the default behavior is
to open `http://127.0.0.1:8787/` (or the configured host/port) in the default
browser. Use `--no-open-browser` on a headless or remote machine.

The ordinary `--gui` intentionally has no mutation buttons. Use
`--control-gui --enable` for formal operator controls. In that page,
`保持输出` requests the lower HOLD, `恢复运行` is an explicit re-arm, and
`采集当前 Runtime Home` re-anchors the live VIVE pose to the current MuJoCo
TCP while held, without closing the viewer or publishing an action.
`结束当前仿真并返回准备页` is the separate destructive rebuild operation:
it closes the current MuJoCo episode and restores the VIVE setup owner. Neither
operation commands a physical robot to home. A composition failure is shown
separately from later input diagnostics, so the operator can diagnose the
first fault before choosing Home capture, re-arm, or a whole-runtime rebuild.
The same control page now exposes one pre-construction `Control workflow`
selector. `--control-profile` supplies its initial value; before MuJoCo is
constructed the operator may select fixed-alpha compatibility, exclusive
takeover, Scheme A, or Scheme B. One Runtime atomically binds the selected
lifecycle/control-cycle pair. Selection is locked after construction and can
change only after the explicit whole-runtime rebuild, so authority, Runtime
Home, source history, and activation state never migrate implicitly between
profiles.

The same pre-construction panel selects the Sim planner's preventive-clearance
guard. It is enabled by default. The planner clearance comes from the selected
MuJoCo scene's `CollisionSpec.safety_minimum_clearance_m`; Sim-HIL does not
inherit the real-side planner's 30 mm default. A lower `collision_margin_low`
rejection means a configured Pinocchio proxy pair fell below that preventive
clearance; it does not assert that MuJoCo has already produced a physical
contact. Separately, Sim-HIL observes MuJoCo's active forbidden contacts and
terminates when penetration exceeds the scene contract's hard-penetration
limit; ordinary cube-table and gripper-object task contacts remain allowed.
The lower resolver or contact observer first establishes evidence, then the
Station terminates the current episode and offers only `Reset simulation`;
re-arm, Runtime Home capture, and source switching cannot continue that
episode. Diagnostics retain the best candidate distance, effective threshold,
closest proxy pair, or actual forbidden contact pair and penetration depth.
With the guard disabled (or `--no-sim-collision-stop` in headless use), the pose
planner installs no collision pairs and observed MuJoCo contacts are not
promoted to terminal lifecycle events. MuJoCo contact physics itself remains
enabled. IK, joint constraints, non-finite checks, and other execution failures
are unchanged. This selection is locked after target construction and applies
only to the Sim target, never to real-robot safety.

With an allocation profile the page replaces fixed-alpha controls with one
`Policy / Human streaming control` panel. Dynamic profiles expose typed
`Auto allocation`, `Policy baseline`, `Human relative correction`, and
`Rebase and clear residual` options; Scheme B additionally exposes `Human
absolute takeover`. The exclusive truncated-HITL profile uses the same panel
and streaming backend, but shows only `Policy`, `Human takeover`, and `Policy
resume` options. It is not a separate GUI or execution path. Scheme A neither
advertises nor accepts absolute takeover.
Its dedicated allocation start
intent starts from the policy baseline and carries no fixed-alpha control mode
or weight. Human devices and coordinate-profile readiness are deferred until
the first Auto/relative/absolute/rebase operation that consumes Human input.
Until then the admitted composition contains only the real policy input; the
optional allocation context is absent and the strategy records an explicit
`base_only` selection rather than synthesizing Human data.

If the enabled collision stop rejects a collision margin, the lower execution
gate first establishes HOLD and the application promotes that exact typed
rejection to a terminal episode. The page projects the structured reason
instead of showing only a generic compound-authority error, and it never
offers a source/Home/re-arm recovery path for that episode. Reset closes and
rebuilds the lower-owned MuJoCo target.

The control page starts before a MuJoCo target or policy runtime is
constructed. One setup controller owns the VIVE receiver and projects live
input plus the selected saved coordinate profile into the same page. The
operator may start/cancel/retry coordinate calibration there; an unsaved
calibration is never used for execution. Human-only and proportional launch in
the fixed-alpha compatibility profile require a finite/fresh selected VIVE
channel plus a valid coordinate profile. PPO-only launch, and policy-baseline
launch in either allocation profile, do not consume or wait for either fact:
Station reserves the
typed human component port without constructing a device source, starts the
policy/target/viewer path, and keeps the setup receiver available only for an
optional later Human attachment. A setup-receiver bind/start failure remains
visible and disables Human modes, but does not prevent PPO-only construction.
Switching from PPO-only to a Human-using mode
first requires live Human readiness, enters HOLD, constructs and attaches the
exact reserved source, validates one passive emission, captures Runtime Home,
then publishes the new blend parameter and re-arms. A failed Human preparation
leaves the effective PPO blend unchanged.

Mode changes activate contributors transactionally. Crossing from zero Human
weight to a Human-using mode always captures a new Runtime Home against the
current MuJoCo pose; returning to PPO from Human-only similarly refreshes the
previously inactive policy source under HOLD. Changes that only adjust weights
between already-active contributors do not add a HOLD/re-arm cycle.

For a Human-using initial mode, Station constructs the application while the
setup receiver remains live, then hands the exact source-role mapping and
profile path to it. The setup receiver is closed only after construction
succeeds, immediately before the new application starts its receiver. Station
checks the passive human emission and samples the current VIVE pose as Runtime
Home before composition starts, then opens the native passive MuJoCo viewer.
The viewer is opened before the realtime physics worker is started,
avoiding a native `MjData` race; missing/stale inputs and failed source
activation remain on the Station page without opening a misleading simulator
window. Because native window construction may take longer than a contributor
TTL, Station acquires and validates a new human/policy emission after the viewer
opens and before composition preflight; it never relabels the pre-viewer frame
as fresh or widens the TTL. The ControlTrace episode is prepared only
after viewer, composition, and blend activation all succeed, so a native startup
crash cannot leave an active episode preparation behind.

The formal ControlTrace WAL grows in bounded filesystem extents rather than
stopping at the initial 64 MiB allocation. Its in-memory admission journal
remains explicitly bounded and its utilization is visible in status. Any
durable-writer or journal-admission failure makes the formal capture
uncommittable, is displayed by the GUI, and causes the Station control thread
to establish lower HOLD; re-arm and mode activation remain blocked until the
runtime is rebuilt.

The viewer handle, synchronization, and teardown are owned by `sim.mujoco`;
Station only coordinates that lower resource. Formal interactive control
requires this viewer because the operator must be able to observe the simulator.
Ordinary non-control runs remain
headless unless `--mujoco-viewer` is supplied.
Closing the native window is observational and does not introduce a new motion
gate. If full PPO/runtime startup fails after resources have been torn down,
the same closed application instance is never reused. The unified control
controller restores a fresh setup receiver and permits an explicit retry that
builds a new application; the first failure remains visible. While a passive
input wait or startup operation is active, `cancel_operation` requests
cancellation without giving the HTTP thread device, simulator, or authority
ownership.

The right-side angle-trigger fallback is side-keyed and lazy: application/GUI
construction performs no angle-sensor discovery or serial open, while the
formal runtime startup loads the configured right-only reader. Failures remain
visible and retryable before full startup unless lower resources were already
closed by a terminal startup failure. Interactive assignment and endpoint
calibration are lower-owned hardware setup operations and never block the
Station HTTP or control thread.

The control GUI also exposes one typed `swap_vive_sources` intent until Human
input is attached, including while a pure PPO runtime is active. The lower
VIVE source swaps only upstream
`left_wrist`/`right_wrist` roles and clears buffered history; when a sender
snapshot for the new role exists, that snapshot is re-seeded into causal
history so the display cache and cutoff query cannot disagree for one control
cycle. Logical output channels remain stable. The same operation toggles one
single-side binding or exchanges both bindings in a bilateral profile. It does
not inspect source identity, validity, or freshness. A mapping generation is
carried on channel metadata so the passive session resets pose baseline and
retargeter smoothing at the boundary. Therefore a single sender can be
switched in either direction and immediately switched back; an unavailable
role remains explicitly invalid. Station rejects swapping after Human
attachment, so it cannot become an untracked live target jump.

The same GUI now projects both upstream sender roles before that mapping,
including packet count/age, valid/stale state, pose summary, sender error, and
the logical output currently selecting each role. This is a passive diagnostic
cache shared with the real-teleop display semantics; it never becomes a second
observation source or composition input. The device-input wait runs outside the
Station status lock, so these facts continue to refresh while acceptance is
pending. A JSON packet with an explicit `tracker_role` is routed strictly by
that role; tracker-name fallback is retained only for legacy JSON without a
role and cannot silently undo a role swap.

The page uses two refresh budgets: the complete `/api/status` projection is
polled at a low rate, while `/api/vive-status` only reads the VIVE receiver's
small lock-protected diagnostic snapshot and refreshes at 4 Hz. This keeps live
pose/role display responsive without repeatedly rebuilding policy,
composition, or execution receipts.

The VIVE status distinguishes transport freshness from source acceptance. In
addition to packet count/age and upstream role validity, it exposes
`worker_alive`, `process_error_count`, `last_process_error`, and the latest
batch/parse diagnostics. Sim-HIL human status separately exposes the latest
observed emission sequence and `last_rejection_reason`; an accepted emission
is required for human readiness, so a fresh UDP packet with a rejected action
contract is not reported as a healthy human command.

An intent POST returns a submission/queue receipt, not completion. After a
source swap the page briefly reads `/api/status` until the same `intent_id`
appears in `acceptance.last_result`, then repaints the current mapping from the
Station-owned status snapshot.

The primary readiness label follows the sources required by the currently
effective blend: PPO-only does not falsely fail because human evidence is
stale, and human-only does not falsely fail because policy evidence is stale.
The same payload retains `all_evidence_ready` and
`all_evidence_failed_checks` for diagnosis, while the GUI marks inactive checks
as “not required for current mix” rather than claiming that their evidence is
healthy. This is a status projection of the existing admission rule, not a
readiness bypass.

The runtime exposes three peer control modes:

- `human_teleop` publishes human alpha one.
- `ppo_only` publishes human alpha zero.
- `proportional_blend` publishes a typed dynamic parameter record with alpha
  strictly between zero and one for proportional pose/qpos composition.

The runtime can start in any mode, and the operator may switch modes in any
order and any number of times. Mode selection occurs before the first Station
control cycle and before composition contract construction/preflight, so no
different mode is silently exercised during startup. HOLD, re-arm, Runtime
Home, rebuild, and trace finalization are
independent lifecycle operations, not mode-unlock gates. Finalization requires
an active recording episode but does not require all three modes to have been
visited. `SimHILAcceptanceTraceCapture` remains a profile-default compatibility
name over the lower-owned `teleop_runtime.control_trace` capture lifecycle:
it binds the lower-owned `EpisodeTraceRecorderSink`, records both immediate
source/composed differences and lower reference/measurement differences,
establishes HOLD/cancellation at episode end, then seal/commits the trace.
The policy-source port is representation-neutral: absolute policies,
observation-relative chunks, and sequential-delta policies all enter through
the same canonical `PolicyActionEmission` boundary. A profile may inject any
source implementing that port; Station does not inspect its checkpoint or
model encoding.

After a cycle, Station may project passive `policy_execution_feedback`. It
matches the policy input selected by composition through its exact source fact,
then accepts Reference/Measurement values only when their receipt chain contains
the same downstream target-admission receipt. The record distinguishes an
unchanged policy output from an output composed with human/other sources.
Feedback consumption is optional and defaults to recording only; it never
reruns IK/interpolation/safety or silently rebases a relative policy target.
Runtime termination stays outside episode membership. This is ControlTrace-core
capture, not a replacement for collection-runtime camera/tactile/native-stream
writers. `--acceptance-gui` remains only a CLI compatibility alias for
`--control-gui`.

The top-level Agent Gateway owns the pull-based visual request contract and
MuJoCo visual binding. Station only exposes the application revision and
existing scene/physics cut used by that profile; the visual request grants no
streaming or motion/control authority. See
[`agent_gateway/README.md`](../agent_gateway/README.md#observation-and-visual-cuts).

The Station loop follows the authoritative scheduler's exposed next deadline.
If synchronous contributor work misses one or more upstream compose cuts,
Station records and skips those cuts without burst catch-up; it does not add a
second HOLD policy. The lower target admission/execution gate remains the
authority for late target acceptance and HOLD. The status page preserves the
first lower control fault separately from any later source-staleness symptom.
Scheduled reference profiles also declare an upstream admission margin
(`--target-admission-margin-sec`, default 50 ms); source validity covers the
next inference interval plus the target lead. These are timing/support
contracts, not extra safety filters. The receiver still rejects late segment
admission and remains the sole reference/activation owner.

`--vive-coordinate-profile` selects a coordinate-only profile from the
independent `control_station_vive_coordinate_profiles` directory. Its default
is the exact versioned repository profile
`vive_rm65_virtual_desk_coordinate_v1.json`; it does not depend on which local
profile was saved most recently. Passing `latest` explicitly selects the newest
valid machine-local record. The checked legacy seed in that directory is a
non-mutating extraction of the former combined profile.
It owns only the VIVE frame, tracker offsets, canonical station-frame contract,
and source artifact identity. `--calibration-profile` is currently retained
only as the compatibility source of the stored human Home. Its robot Home and
base transforms are not trusted for Sim-HIL.

The MuJoCo target derives a `TargetFrameProfile` from the exact loaded URDF and
scene body poses, and a `ToolFrameProfile` from the exact selected-arm tool
file. For a task that declares scripted reset semantics, the simulator target
reuses `CubeLiftKernel.reset` and publishes the resulting initialization facts;
Station and the PPO adapter do not reproduce that trajectory. After task reset
it measures each selected `tool_tcp`, converts it through that target
profile, replaces the compatibility robot Home in an in-memory calibration
copy, and emits a `SessionAnchorReceipt` binding all profile hashes and selected
Home poses. Neither source JSON is modified. Coordinate/Home provenance
mismatch, canonical-frame mismatch, missing side, wrong tool frame, or a
non-closing target transform fails before the passive source is built.

The present receipt deliberately labels human Home as
`stored_human_home_from_selected_profile`; live VIVE recenter is a later
device-lifecycle gate. Real targets must supply their own measured
`TargetFrameProfile`, `ToolFrameProfile`, and activation-local target Home.
They must not reuse the simulated robot Home or treat an URDF-derived frame as
hardware measurement.

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
- `control_station/sources`: takeover-readiness and thin compatibility/profile
  bindings only; generic source production, admission, fanout, and source
  lifecycle remain lower-owned.
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

The retired mixed MuJoCo Station wrapper and pose-composer fixtures are no
longer part of the active source tree. New target/source work must use the
current typed seams and an explicit profile owner; a compatibility re-export
is not a substitute for a live caller or a validated migration. Historical
source is retained only in the local ignored backup and the audit linked above.

## VIVE coordinate calibration

`python -m control_station vive-calibration` opens a local browser GUI for the
VIVE coordinate/tracker calibration workflow. It reuses the existing VIVE
receiver, virtual-desk sweep fitting, tracker-offset calculation, and
`TeleopCalibration` serialization through
`teleop_runtime.real_teleop.vive_calibration_runtime`; Control Station only
binds the source, lifecycle, GUI events, and independent profile store. It
creates no robot/sim target and owns no dispatch or motion authority.

The GUI supports `--sides left`, `--sides right`, and `--sides both`. It fits
the VIVE virtual desk from X/Y/Z sweeps, captures wrist tracker offsets for the
selected side(s), and displays live channels, fit quality, transforms, and
saved profiles. Use `--view-only` to inspect profiles without opening the VIVE
receiver and `--dry-run` for configuration validation.

Before a workflow starts, Station checks every channel needed by the selected
sweep tracker and wrist side(s) for a finite, valid, fresh pose. A missing,
stale, or wrong-side channel keeps the workflow idle so the operator can change
the side, select another sweep tracker, or swap the left/right VIVE source
roles. During any capture stage, **Cancel calibration** stops only the current
workflow and returns to that configurable idle state; the receiver and page
remain open. In the standalone calibration command, **Exit calibration** also
closes the receiver and terminates that standalone session.

The signed sweep directions are explicit: desk `+X` is captured back to front,
desk `+Y` right to left, and desk `+Z` low to high. Move once in the displayed
direction after the preparation countdown and do not reverse before the
capture window ends. These signed sweeps already establish the canonical axes,
so newly captured Control Station profiles default to
`virtual_desk_post_yaw_deg = 0`; `--virtual-desk-post-yaw-deg` is an explicit
override, not a second implicit alignment step. Existing profiles with an
explicit post-yaw remain unchanged, while legacy profiles without that field
retain their load-time compatibility correction. Reset restarts only the
current stage; it is not an exit.

For right-arm Sim-HIL control, the same `--control-gui` contains VIVE
readiness and calibration controls. It first selects `--vive-coordinate-profile`
(`latest` by default). If that profile and the live selected tracker are ready,
the operator may start a Human-using mode immediately; if not, the operator can
still start pure PPO without a coordinate artifact. Calibration may continue
on the same page while PPO runs. Before a later Human attachment, Station
requires the saved artifact and fresh input, holds policy execution, closes
the setup receiver, and starts the bound runtime source on the same port. A
live Human target never hot-swaps coordinate frames. The compatibility flag
`--vive-calibration-gui` no longer opens a second page; its port/browser
options are retained only so older commands fail neither parsing nor startup.

Sim-HIL 的网络容忍使用独立的
`--human-emission-timeout`（默认 5 秒），不改真机 TeleopSession 的丢帧 HOLD。
`--human-timeout-fallback hold` 默认保持下游输出；显式选择 `policy` 时，运行中
人类输出超过窗口未更新会用新 PPO 输出锁存切换，恢复后需人工重新选择人类或混合。
GUI 中该选项与“融合目标不可执行回退”分开显示。

These profiles use the independent
`control_station_vive_coordinate_profiles_dir` path, defaulting to
`teleop_records/control_station_vive_coordinate_profiles`. They are not
discovered by the real-teleop `calibration_profiles_dir` workflow and carry a
distinct `profile_kind=control_station_vive_coordinate` marker. Robot/sim
desk-to-base transforms and session Home anchors remain target/runtime-owned.

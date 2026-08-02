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

Current concrete-profile and migration debt is frozen as follows:

| Current Station area | Long-term owner |
|---|---|
| action/context selection, TTL, fallback, difference, handoff/session semantics | `control_composition` |
| human admission, passive teleop session, human emission fanout | `teleop_runtime.action` |
| replay binding/fanout/cursor-owned status | `teleop_runtime.replay` / `robot_motion.replay` |
| sync/RTC policy binding and step validation | `teleop_runtime.deploy` / `robot_motion.deploy` |
| OpenPI artifact/protocol assessment | `deployments.openpi` |
| MuJoCo measurement, readiness, resolution and execution evidence | `sim.mujoco` |
| authority, safety, IK, reference and dispatch | `robot_motion` |
| ControlTrace/v2.4 projection and capture | `dataset_interface` and the existing capture owner |

`control_station/composition/` is therefore a transitional name. After the
concrete algorithms above have returned to their owners, the remaining thin,
GUI-selectable application assemblies should become declarative
`control_station/profiles/`; they may import multiple lower packages but may
not recreate their queues, workers, cursors, epochs, safety, or dispatch.

The boundary tests keep current migration debt explicit: dataset projection is
confined to `control_trace.py`, the existing
`composition/mujoco_pose_composer.py` profile, and the default-off
`composition/mujoco_sim_hil.py` profile, OpenPI knowledge to
`preflight/openpi_rtc.py`, MuJoCo imports to concrete profiles, and numerical
source code to the two source modules scheduled for extraction. New debt may
not spread while those files are migrated.

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

CS-6b1h now also has an explicit opt-in formal-strategy-v2 contract fixture.
The strategy is injected into `MujocoDeployRuntime` rather than stored in a sim
configuration, and the Station binding includes its strategy-spec hash under a
v2 binding domain. The deterministic migration adapter has exact pose/action
parity with the compatibility composer on the same frozen input snapshot, but
produces a distinct v4 result and proposal lineage before entering the existing
single authority, safety, reference, and dispatch chain. With no injected
strategy, the established v1 binding and behavior remain unchanged. This is
not yet a default operational profile or permission to run a slow model/RPC in
the control loop.

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

`ControlCompositionRuntime.evaluate_composition()` now accepts that optional spec,
returns the observation on `CompositeProposalSubmission`, and emits exact
protocol-neutral source-difference domain events. The injected dataset adapter
may project them into hash-linked v2.4 ControlTrace facts. Diagnostic
failure is returned separately and cannot rewrite the
composition evaluation. CS-6b1h/1i only forward the opt-in spec and expose the same
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

The current default-off `python -m control_station sim-hil --enable` application
profile is the operational Station leaf for the temporary PPO plus one passive
VIVE tracker and one selected right gripper input (Q6 first, calibrated angle
trigger fallback). Its native composition schema is
`right_arm: cartesian_pose` plus `right_gripper: physical_joint_qpos`; the old
`RobotAction` is materialized only inside the current coupled MuJoCo execution
edge. The selected topology is an application profile, while the lower source,
component schema, and target binding remain cardinality-driven. The ordinary
`--gui` remains a read-only status surface showing readiness, source freshness,
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

The control page starts before a MuJoCo target or policy runtime is
constructed. One setup controller owns the VIVE receiver and projects live
input plus the selected saved coordinate profile into the same page. The
operator may start/cancel/retry coordinate calibration there; an unsaved
calibration is never used for execution. Human-only and proportional launch
require a finite/fresh selected VIVE channel plus a valid coordinate profile.
PPO-only launch does not consume or wait for either fact: Station reserves the
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
visited. `SimHILAcceptanceTraceCapture` remains the internal capture adapter:
it binds the lower-owned `EpisodeTraceRecorderSink`, records both immediate
source/composed differences and lower reference/measurement differences,
establishes HOLD/cancellation at episode end, then seal/commits the trace.
Runtime termination stays outside episode membership. This is ControlTrace-core
capture, not a replacement for collection-runtime camera/tactile/native-stream
writers. `--acceptance-gui` remains only a CLI compatibility alias for
`--control-gui`.

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
is `latest`, resolved as the newest valid independent record. The checked
legacy seed in that directory is a non-mutating extraction of the former
combined profile.
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
capture window ends. Reset restarts only the current stage; it is not an exit.

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

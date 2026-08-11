# Control Composition

`control_composition` is the reusable, backend-neutral framework for combining
multiple control sources. It turns immutable contributor facts from human,
replay, synchronous/asynchronous policy, or synthetic producers into one time-bound composition
result with exact causal lineage.

It is a library, not a production entrypoint.

Its scope is intentionally source- and target-agnostic:

```text
source adapters
  -> immutable action/context facts
  -> receiver admission + one causal cut
  -> replaceable strategy
  -> component-keyed composed candidate + differences + complete evaluation record
  -> proposal admission
  -> downstream target/execution owner
```

A strategy can depend on frozen history, task state, future actions, labels,
vision features, or other modalities. The framework never rereads a live
model, sensor, device, simulator, or GUI after the cut.

## Ownership

This package owns:

- contributor roles, immutable inputs, receiver-side causal cuts, TTL,
  missing-input and fallback semantics;
- opt-in source eligibility/liveness facts, leases, status epochs and
  fail-closed re-arm lineage, kept independent from action TTL;
- immutable, payload-contract-bound strategy context for history, state,
  future action, vision and arbitrary new modalities;
- replaceable arm-pose and hand/gripper-qpos composition strategies;
- composed candidates and proposal-promotion eligibility;
- logical composition sessions, activation epochs and source handoff reducers;
- composition-local immediate-difference observations and their
  non-authoritative feedback facts; downstream execution differences remain
  owned by the execution observer;
- a complete evaluation return record containing the exact spec, cut, active
  producer epochs, selected inputs, contexts, result, differences, diagnostics,
  and protocol-neutral domain events emitted for that evaluation;
- an independent `CompositionInformationPlan` and narrow ports for domain-event
  observation and canonical proposal admission.

This package does **not** own:

- canonical `RobotAction` component schemas, authority, arbitration, producer
  or execution epochs, cancellation, safety, IK, reference generation,
  dispatch, or a hardware/sim target;
- a teleoperation device, replay cursor, model worker or action-chunk queue, dataset
  episode, WAL, GUI, or web server.

The output boundary is a `ComposedActionCandidate` plus lineage, never a
motion permit or a safe-action claim:

```text
contributors
  -> control_composition
  -> RobotMotion proposal admission
  -> authority/arbitration
  -> target/execution capability
  -> downstream-private safety/IK/smoothing/limits as applicable
  -> dispatch receipt
```

### Thin downstream handoff

The reusable framework requires a common **handoff protocol**, not a common
execution algorithm.  Concrete sim, robot, velocity, wrench, or hybrid-control
targets remain free to implement their own safety, IK, planning, resampling,
command and backend stages.

The transport-neutral handoff contracts live in
`robot_motion.action.execution_handoff`, rather than this package, because they
begin after proposal admission, arbitration and authoritative publication.
Putting them here would reverse the required
`control_composition -> robot_motion` dependency.  The thin boundary is:

```text
accepted ActionAdaptationResult
  -> TargetExecutionRequest
  -> TargetExecutionSubmissionReceipt       # accepted for processing only
  -> target-private processing
  -> TargetExecutionCompletionReceipt       # resolved/failed/cancelled
  -> ComponentExecutionTarget               # no universal safe-action claim
  -> ComponentExecutionTargetAdmissionReceipt
  -> optional ComponentExecutionTargetActivationReceipt
  -> optional activation/reference/command/dispatch/measurement facts
```

`TargetExecutionPort` owns no worker, queue, authority or fallback policy.  Its
submission receipt is not target admission or motion permission.  Every accepted
submission must eventually publish one terminal completion receipt, including
failed, rejected, or cancelled outcomes when no target exists.  Optional
target-private safety, IK, planner or limiter evidence is carried only as
opaque `TargetProcessingFactReference` values.  A target capability explicitly
declares whether activation is `required` or `omitted`; an activation receipt,
when required, is separate from target admission and dispatch.  `control_composition`
may consume later observations for differences, but cannot create, admit or
activate an execution target.

The support boundary is propagated without transferring execution ownership:

```text
CompositionEvaluation.composition_supported_until
  -> ActionProposal.effective_until
  -> ArbitrationResult.supported_until
  -> AuthoritativeArbitrationDecision.supported_until
  -> ActionAdaptationRequest / TargetExecutionRequest
  -> ComponentExecutionTarget.supported_until
```

Each stage may shorten the interval and none may extend it. Publication,
processing acceptance, target admission, and activation reject an already
expired object; scheduled intervals and persisted execution segments are
capped by the same boundary. Late processing may still produce a terminal
receipt for observability, but cannot restore executability. This is temporal
validity, not canonical safety. The development protocol upgrades the
composition spec/snapshot/evaluation, arbitration result/decision, and
adaptation-request hash domains; downstream code must rebuild these facts
rather than interpreting old hashes as the new contract.

The first production-wired consumer is the explicitly enabled MuJoCo Sim-HIL
profile. It treats the resolved qpos endpoint as a `point_target`; the existing
MuJoCo reference owner alone derives the continuous interpolation segment and
reports its actual activation interval. This wiring does not make
`control_composition` aware of PPO, RTC, VIVE, MuJoCo, or RM65.

## Time and rate semantics

Sources do not need to run at the same frequency. This package performs
**same-target temporal snapshot selection**, not interpolation or resampling.
Every admitted contribution
declares receiver-clock `[effective_from, effective_until)` validity. For one
`target_effective_at`, the receiver captures a single admission cut and:

1. excludes facts that arrived after the cut or belong to an inactive producer
   epoch;
2. excludes facts whose TTL does not cover the target;
3. selects the deterministic newest valid fact per declared role;
4. applies the declared role-dependency closure;
5. evaluates all participating roles and contexts on that same cut;
6. emits one result for downstream proposal admission.

`CompositionSpec.required_roles=None` means every declared role is required.
Optional roles must therefore be explicit. `depends_on_roles_by_role` declares
one-way participation dependencies: if `overlay` depends on `baseline`, a
missing baseline removes both from that evaluation, while a present baseline
may still participate when the optional overlay is missing. The dependency
graph is acyclic, and a required dependent may depend only on required
ancestors. The snapshot retains both the pre-closure selection and exclusion
reasons, so a valid-but-excluded input is not lost from trace lineage.

The exact rule is carried by `CompositionTemporalSelectionSpec` and its hash is
part of every `CompositionInputSnapshot`. Covering inputs are ranked by
`effective_from`, then receiver `control_admission_sequence`, then fact ID.
Producer-local sequence values are validated per producer but are not compared
across producers. A profile may instead reject multiple covering inputs.
Runtime profiles derive finite target lead/lag bounds from their declared
retention window; standalone contract fixtures may explicitly use an unbounded
development spec. The selected target time and receiver clock domain are
explicit fields of both the input snapshot and evaluation. Legacy
`RobotAction.timestamp` is compatibility lineage only and never participates
in temporal selection or proposal promotion.

A slower source may therefore be reused only while its explicit TTL remains
valid; a faster source may replace only future evaluations. Late completion
never rewrites an existing evaluation. `control_composition` does not sleep,
interpolate, resample robot commands, or own a worker queue. High-rate target,
reference or command generation and physical dispatch remain downstream, bound
to the candidate identity and requested target time.

A ready `CompositionEvaluation` also carries
`composition_supported_until`, the earliest known half-open support boundary
across its participating inputs, enabled source leases, required contexts, and
declared context maximum age. Proposal promotion may shorten this boundary but
cannot extend it, and an already expired result cannot be published. Shadow
evaluation may repeat the same target on a later cut, but successful production
publication is strictly target-monotonic within one composer producer
activation. The current contract does not publish or revise a long future
target range; therefore it has no implicit same-target replacement semantics.
If future scheduling needs that capability, it requires a separate explicit
replacement/version contract.

### Optional guaranteed-next source sampling

Action-chunk owners may opt into source-intent interpolation *before* input
admission.  This does not change the runtime selection rule above.  The
development interface has exactly two modes:

```text
current + no guaranteed next
    -> current action unchanged

current + GuaranteedNextComponentAction
    -> ComponentAction sampled between the two target times
```

The guarantee is a source-owner assertion that the exact next action will
execute immediately after the current step without replacement. A queue peek,
uncommitted prediction, unknown next chunk, or speculative continuation is not
a guarantee and must be represented by `None`. A malformed claimed guarantee
(clock, order, schema or coverage mismatch) is rejected rather than silently
downgraded.

`sample_component_action_source()` currently supports Cartesian pose
`xyz_linear+quaternion_shortest_path_slerp`, joint-position/continuous-native
linear interpolation, and discrete-mode hold-until-next. The returned
`ComponentActionTemporalSample` binds both endpoint facts, the guarantee,
target time, interpolation alpha and algorithm contract. Its sample identity
is included in the core `CompositionInputDraft.source_fact_ids/hashes`; it is
not debug-only metadata.

`build_temporally_sampled_composition_input()` provides the generic draft
adapter. `PolicyCompositionInputAdapter.adapt_interpolated_at()` is the thin
policy-facing port and requires the guarantee argument. A policy caller without
a guarantee must use the established `PolicyCompositionInputAdapter.adapt()`
path, preserving its one-current-step draft, sequence, TTL and identity. No
profile automatically supplies a guarantee. This layer never generates an
execution Reference, runs safety/IK, reads a chunk queue, or grants authority.

### Optional nominal-trajectory editing

The development-only `control_composition.trajectory` layer can latch one
backend-neutral `ComponentActionTrajectory` while a residual session is
active. The canonical trajectory carrier lives in `robot_motion.action`; a
deploy, replay, or other source owner may project its immutable plan into that
carrier without transferring queue or cursor ownership.

The reducer exposes pass-through selection, active-base latching, one bounded
latest shadow candidate, exact lattice-target preparation, and receipt-bound
emission commitment. Session activation now references one
`CompositionActivationScope`, `ControlCut`, `ResidualAnchorReceipt`, immutable
revision-0 `NominalBaseBinding`, and initial publication barrier. The runtime
state keeps interaction mode and nominal-base lifecycle as separate axes and
rejects combinations outside the development legality matrix.
Pass-through trajectories may omit source activation lineage; entering a
residual session may not. The base binding requires source instance plus
activation ID/hash, preventing a late result from an earlier re-arm from being
silently latched.

R3.0 also defines advisory base-transition assessment/revalidation and three
explicit observation capability levels: HOLD-boundary only, Reference-only,
and Reference plus Measurement. An assessment binds the active/candidate base,
the exact residual fact, observation facts, cut, publication basis, future
switch boundary, and corrected seam. It must be revalidated before the switch;
a changed residual/candidate/capability, a stale observation, or a publication
barrier that reaches the boundary rejects that attempt. These are eligibility
contracts, not motion safety or permission. Ordinary automatic base refresh
and handback remain disabled.

The separate stateless temporal-warp seam can currently ramp state-valued
Cartesian translation and arbitrary joint-position residuals; non-identity
pose rotation is rejected until the full-pose gate. A preparation is not an
emission, and its commit receipt is not authority, target admission, reference
activation, or dispatch evidence. Support exhaustion returns typed
`no_action`; it does not request HOLD or synthesize a lower-owned fallback.

Residual recentering is a separate session-local transaction. The immutable
session activation keeps its initial anchor, while
`ActiveResidualAnchorBinding` advances by exactly one revision after an exact
`ResidualRebaseRequest`/`ResidualAnchorReceipt` pair. Accepting a request
immediately fences the old anchor at one uncommitted trajectory keyframe.
Until the fresh receipt arrives, preparation at or after that boundary returns
typed `no_action`; it does not create HOLD or continue the old correction.
The receipt must use the same source/schema/component protocol, follow the
previous request generation, be captured after its barrier and before its
deadline, and, when declared, match the exact trigger-associated source sample.
Completion cannot precede capture or rewrite an emitted target.

The current transaction reanchors the exact component set of one residual
session atomically. It does not partially replace one component's anchor inside
that session. A future partial-rebase profile would require a component-keyed
active-anchor binding and publication-basis contract; it must not be simulated
by mixing receipt generations. Target continuity after the new anchor becomes
effective remains downstream planner/executor work. No automatic detector,
device sampling, Reference generation, safety, authority, or dispatch is owned
by this reducer.

The R3.1a shadow seam connects this preparation to the existing generic
`CompositionInputStore` and formal strategy without introducing a second
composer. Contributor role names and producer stamps are caller-bound; neither
the module nor its tests assume policy/human, arm/hand, MuJoCo, or a particular
robot topology. In residual-active mode it admits the exact nominal keyframe
and the temporally sampled residual on one activation/cut, then calls
`evaluate_composition()` only. It never calls proposal promotion or emission
commit. Pass-through returns the nominal preview without touching the
composition store.

The returned preview distinguishes nominal, corrected, latest-shadow, and
executable views. Executable is always explicitly unavailable in this gate.
Its compact ControlTrace summary records identities/hashes only; the ordinary
composition input/evaluation facts retain the actual participating actions, so
high-rate vectors are not duplicated. The Control Station wrapper is
default-off, rejects `send_actions=true`, and invokes its evaluation factory
lazily so an off profile does not fetch model, replay, or device payloads.
Separately, the source owner may publish one complete
`component_action_trajectory` actual-data fact when it offers a new immutable
plan. That fact preserves all keyframe values once; composition and shadow
summaries continue to reference hashes rather than copying the whole chunk on
every cut.

R3.1b reuses the same derivation through
`prepare_trajectory_composition_inputs(admission_mode="publication_candidate")`.
That function still owns only exact target preparation and ordinary
`CompositionInputStore` admission. It returns the exact admitted input
ID/hash for every role so an application adapter can require the later formal
evaluation to select those same facts. It does not publish a proposal or
advance the emission barrier. Only an external accepted proposal receipt may
be passed to `commit_trajectory_target_emission()`; the optional emitted action
hash is the formally composed candidate, not the nominal keyframe. Target
resolution, safety, IK, Reference/Measurement generation, and dispatch remain
downstream-owned.

This layer prepares one due target at a time and does not publish a corrected
future range. An already emitted target is therefore never revised in place.
`source_immutable_through`, when present, is producer lineage only and must not
be displayed as an executable/committed reference prefix. The existing
absolute weighted and per-cut relative paths remain unchanged, and no
application profile enables trajectory editing by default.

For source-intent visualization,
`build_trajectory_intent_preview()` projects every keyframe of one exact
trajectory revision without touching a queue, cursor, composition store,
closed-target watermark, proposal, or emission barrier. Each point is labeled
either `producer_locked_intent` (inside the source's non-rewritable prefix) or
`speculative_prediction` (replaceable future trend). Neither label claims
target admission, Reference activation, or execution. Actual downstream
Reference/Measurement facts remain a separate overlay.

A whole-horizon *corrected* preview uses a separate non-closing projection
path. `capture_non_closing_projection_cut()` captures producer epochs and the
currently selected context/eligibility facts plus any strategy-required
residual-anchor lifecycle facts under one `ControlCut`, without advancing any
store's closed-target watermark. Source trajectory keyframes, the current
residual, and optional held roles are then represented as
`ProjectedCompositionInput`, not as admitted `CompositionInputRecord`. The
projected carrier deliberately has no receiver admission sequence or
`composition_input_id`.

`build_trajectory_corrected_preview()` evaluates every source-intent keyframe
from that one immutable basis. Production and preview share the pure temporal
selector and formal component arithmetic, but the projected evaluation exposes
no proposal/submission surface and is always non-authoritative,
non-promotable, and `execution_guaranteed=False`. Context, input age, source
lease, and residual support are never extrapolated: once the frozen support
intersection ends, that target is `unavailable`. A runtime with hot strategy
control, pre-mix policy, or a strategy without explicit projected evaluation
support is rejected instead of silently bypassing those semantics.

Preview freshness is an independent display assessment. Producer-locked versus
speculative describes whether the source may revise a keyframe; fresh, aging,
stale-displayable, expired, or unknown describes the age of its explicit
observation basis. `assess_trajectory_intent_preview_freshness()` therefore
requires typed observation/availability evidence and a compatible clock. It
never infers observation age from trajectory `created_at`, and missing or
unmapped evidence remains `unknown` rather than changing source-intent trust.

A current-action-only policy needs no trajectory or model-specific adapter. It
continues through the existing per-cut composition input path. A one-keyframe
`ComponentActionTrajectory` is structurally legal, but it proves only that one
exact lattice target: the reducer does not invent a successor, interpolate a
future target, or promote its hold interval into a chunk guarantee. Timed
trajectory editing is selected from the source fact actually supplied, not
from checkpoint, policy architecture, or model name.

The first MuJoCo publication adapter is activation-local. An enabled instance
must bind an already-open typed Station composition-session barrier and keeps
that barrier generation plus operation-result identity. Pause/release closes
the Station barrier before lower HOLD/cancellation; a later typed re-arm may
reuse the Station barrier object, but it cannot revive the old trajectory
state or controller. The application must construct a fresh trajectory
activation binding. The lazy request supplier is only a bounded read of an
already-produced fact, never model inference, device polling, or cursor
advancement.

### Relative overlay inputs

The formal component strategy supports an explicit stateless
`FOLLOW_BASELINE` contract. Relative inputs are not disguised as absolute
targets: the baseline role uses an `absolute_target` action schema, the overlay
role uses a `relative_delta` schema, and `CompositionSpec` binds the exact input
schema hash for every role plus the absolute output schema hash.

For dataset self-description, the runtime emits each referenced
`ActionPayloadSchema` and `ComponentActionSchema` exactly once per runtime.
Per-frame input/evaluation facts continue to carry only schema hashes and
values. The schema facts contain `value_semantics`, frames, coordinates, units,
semantic parameters, the complete component catalog, and payload bindings.
ControlTrace completeness requires role-specific formal composition to resolve
all input/output action schemas, and each action schema to resolve every payload
schema. This keeps relative/absolute interpretation queryable without
duplicating static schema definitions at control frequency.

The first numerical rules are:

```text
pose:
  p_out = p_baseline + delta_p_reference_frame
  R_out = R_baseline * R_delta_local

qpos:
  q_out = q_baseline + delta_q
```

The pose rule implements world/base-frame translation and baseline-local
right-multiplied rotation. The delta role must explicitly depend on the
baseline role. A required delta can depend only on a required baseline; if an
optional baseline is absent, its dependent delta is removed by the same
snapshot dependency closure. An optional absent delta may use the sampled
baseline unchanged only when the typed rule declares `use_baseline`.

### Extensible coordinate-frame binding

`absolute_target`/`relative_delta` describes value semantics, not a globally
fixed coordinate system. A runtime profile declares the canonical
`reference_frame` and `controlled_frame` for each component; concrete source
adapters convert their native VIVE, replay, policy, simulator, or robot values
before composition admission. The composition evaluator validates exact action
schemas and frame identities, but never loads calibration or searches a
transform graph while evaluating a cut.

The reusable rigid relationship contract lives in
`robot_motion.coordinate_frames`. `CoordinateFrameGraph` is immutable,
hardware-independent, and uses explicit `T_target_source` edges. A new frame
needs only one edge to any registered frame; deterministic path composition
then resolves it to every frame in that connected component. Redundant edges
are allowed only when their cycles are numerically consistent. Resolution
returns the graph hash, exact edge hashes/directions, and resolved matrix so a
source adaptation receipt can preserve the conversion lineage.

The graph does not make all frames global constants. Frame definitions and
calibration edges are profile-owned immutable inputs, so changing a
calibration/recenter binding creates a new graph or source-binding identity
instead of reinterpreting already admitted actions. Dataset
`PoseFrameContract` values may be projected into this runtime relationship by
replay/deployment adapters, but remain dataset contracts rather than
composition dependencies.

`robot_motion.action.bind_action_payload_coordinates()` supplies the additive
schema binding: it links the existing payload-schema hash and frame names to
the exact frame-definition hashes and graph hash. Thus equal strings such as
`desk` do not establish compatibility by themselves. This development binding
is intended for activation/source-adapter receipts; it does not alter existing
action values, run a transform, or grant admission.

Only static rigid coordinate relationships belong in the graph. Human motion
gain, residual anchor/recenter state, time sampling, and representation-specific
pose/twist/wrench application remain source-adapter semantics. Joint-position
components use their ordered coordinate names, units, and schema hashes rather
than spatial-frame edges.

The first device-neutral source implementation is
`teleop_runtime.action.HumanResidualPoseAdapter`. It accepts immutable
source-native pose samples with explicit reference frame, controlled frame,
sample time, and source-coordinate-binding identity. Anchor capture is an
explicit generation-bearing operation. Both anchor and current sample are
converted through the same immutable graph into the delta schema's canonical
frames before computing:

```text
delta_p = configured_gain * (current_p - anchor_p)
delta_R = anchor_R^-1 * current_R
```

Changing the graph/calibration identity, source binding, source/controlled
frame, or measured clock requires a new anchor; an older sample cannot be
adapted through a newer anchor. The returned state-valued `ComponentAction`
contains only the residual pose components and is accompanied by an adaptation
receipt binding the anchor, current samples, exact transform paths, action
coordinate bindings, gain, graph, and output action hash. It remains a
non-authoritative source result. The existing absolute teleop and Sim-HIL
profiles are unchanged until a concrete profile explicitly selects this
adapter and supplies raw pose samples plus recenter lifecycle receipts.

`HumanPoseChannelAdapter` is the narrow observation-side seam for those raw
samples. A profile declares exactly which channel feeds each component and the
source controlled frame; unrelated vision/tactile/device channels are neither
copied nor retained by this adapter. Every selected value is checked against
its receiver-issued `ObservationSampleRef` payload identity before it becomes
a `HumanPoseSample`. Device-specific construction of the coordinate graph and
source binding remains outside this adapter.

When a teleop, replay, or model source already emits a canonical absolute
`ComponentAction`, `HumanPoseSample.from_component_action()` is the preferred
lighter seam. It projects only the requested pose component and binds the
source retarget/calibration identity plus concrete physical frames. It does
not repeat a raw-device coordinate transform. The raw-channel adapter and the
canonical-action projection are alternatives for different source
boundaries, not two mandatory stages.

For VIVE, `build_vive_human_residual_coordinate_binding()` projects the
existing `TeleopCalibration` transform direction and tracker-offset semantics
into the generic graph. It requires an explicit
`T_canonical_vive_retarget` bridge: the historical VIVE retarget frame is never
silently relabeled as dataset/action canonical desk. World, operator, retarget,
canonical, tracker, and component-scoped tool frame IDs are caller-supplied or
profile-bound, not closed enums. The binding exposes distinct concrete
controlled frames such as `left_arm/tool_tcp` and `right_arm/tool_tcp` while
the action payload schema may retain the reusable semantic name `tool_tcp`.

Joint-position human input uses the parallel
`HumanJointPositionResidualAdapter`. Raw Q6/angle-trigger values are not
assumed to be deltas. A device/source adapter first maps them into an exact
absolute target layout; the residual adapter then binds that mapping and
calibration identity, captures an explicit activation/recenter anchor, and
computes:

```text
delta_q(t) = configured_gain * (q_target(t) - q_target(anchor))
```

The absolute source and relative output schemas must have identical component
coverage, coordinate ordering, shape, and units. A changed retarget/calibration
binding, payload schema, or measured clock invalidates the old anchor. This
keeps q6, q16, a 1DoF gripper, and other joint-position layouts generic while
preventing an absolute device state from being added directly to a policy
baseline. The adapter may select one component from a larger retargeted
`ComponentAction`; it does not retain unrelated components or raw sensor
modalities.

This is a state-valued residual, not a per-cycle increment. Re-evaluating the
same input records at later target times cannot accumulate the delta or extend
their support interval. A held delta follows each newly sampled baseline.
Source-side guaranteed-next interpolation remains available for either schema,
but only with the explicit guarantee described above. Without that guarantee,
the current delta or baseline step is held under its existing TTL.

`LATCH_BASELINE_ON_RESIDUAL_ACTIVE` is intentionally not implemented by this
rule: it requires an explicit activation anchor fact and lifecycle reducer.
`CANCEL_BASELINE_ON_TAKEOVER` belongs to authority/execution lifecycle, not to
numerical composition. Neither behavior is inferred from weights, deadband, or
missing samples.

Formal start/re-arm additionally uses `CompositionActivationScope`. The scope
binds one operation spec and session epoch to an explicit receiver-clock input
barrier and required roles. Scoped inputs carry its exact ID/hash; a snapshot
selects only records from that activation whose receiver availability and
effective start meet the barrier, and the result preserves the same identity.
Input admission and snapshot selection use the same canonical activation
freshness evaluator. The receiver-local policy compares `available_at` and
`effective_from` on the activation clock. The strict-source policy additionally
requires `produced_clock_domain_id` to equal that clock and requires
`produced_at > input_barrier_at`; cross-clock source timestamps are never
implicitly compared or converted. Failed admission returns a typed activation
identity, clock, or barrier rejection without advancing the admission
high-watermark.
This identity is independent of producer, authority, and execution epochs.
Unscoped callers retain the v1 input/snapshot/result ID and hash domains.

An explicitly enabled `CompositionSourceEligibilitySpec` adds a second,
orthogonal gate before numerical composition. It binds every composition role
to one stable source slot and producer. `CompositionSourceEligibilityState`
then records receiver-admitted `available`, `paused`, `fault`, `disconnected`,
or `rearming` status, a source lease, status epoch, and optional re-arm
requirement. The action and source-state stores share the same `ControlCut`.
Consequently, a still-unexpired action is missing immediately when its source
is unavailable or its lease has expired.

Source-state heartbeats may extend a lease only without changing status inside
one status epoch. A status transition requires a newer epoch; an old epoch or
late heartbeat is rejected. Once any state declares `rearm_required=True`, the
requirement remains latched across later unavailable states and producer-epoch
restart; returning to `available` requires a typed re-arm receipt ID/hash.
Profiles may permit an explicit new-status-epoch automatic recovery only if no
re-arm requirement was latched. The generic package validates
the reference and ordering; the adapter/profile owns device-specific health
translation and the meaning of the referenced readiness evidence.

Eligibility is default-off. Without a configured store and an evaluation-time
eligibility spec, existing input selection and v1/v2 hashes are unchanged.
Eligibility only converts contributors to missing roles; it does not grant
authority or execute HOLD. A result-bound disposition can submit the exact
ready candidate, emit `no_action` (submit no candidate), or request lower
HOLD. Only the downstream execution owner defines persistence after
`no_action`, or accepts HOLD, advances the execution generation, cancels work,
and returns the authoritative receipt.

## Input, optional action-space checks, and fallback

Source-specific values are normalized by adapters into the runtime-bound
`ComponentAction` before admission. `RobotAction` v1 sources must cross the
explicitly named compatibility adapter; the generic store does not infer or
silently convert that carrier. Invalid identity, clock, producer run,
interval, representation, shape, or numerical data is rejected at that
boundary without clamping or applying motion limits. A finite value with a
large magnitude is still preserved for composition. Composition then handles
declared missing-role behavior, source conflicts, weight normalization,
pose-frame compatibility, quaternion validity, qpos active masks, and bounded
residual/disagreement limits declared by its strategy contract.

An optional, default-off `CandidateActionSpaceCheck` may reject values outside
a statically declared action space as an early integration guard. It does not
rewrite the candidate, produce a `SafeAction`, run IK, inspect backend state,
or grant authority. A failed check remains a candidate rejection reason, not a
canonical safety verdict.

The evaluation outcome is only `ready`, `no_action`, or `invalid`. A
result-bound disposition may submit an exact ready candidate, submit no
candidate for this cycle, or request lower-runtime HOLD.
The request itself is non-authoritative and is not a HOLD/cancellation receipt.
Workspace, joint, velocity, force, IK, smoothing, limit, cancellation and
backend safety remain private responsibilities of the selected downstream
execution capability. Composition fallback can reduce availability; it can
never grant motion authority. `no_action` alone does not globally mean HOLD:
each execution profile explicitly selects wait, request-HOLD/cancel, or fault
behavior. If a downstream profile already owns a currently active target or
reference, only that owner may define whether it continues.

Human takeover follows the same boundary. Mainline human source readiness
binds a fresh emission, measured-state freshness, deadman re-arm, and
retargeter rebase, then obtains a separate lower execution-ready receipt. It
does not run canonical safety before formal proposal admission. Any remaining
`require_canonical_safety` input is a legacy migration target, not a new
composition contract.

### Optional same-cut composition policy

An integration may inject one `CompositionPreMixPolicy` after input/context
snapshot capture and before numerical composition. It receives the exact
immutable cut and optional information-plan context; it does not receive raw
device handles, reread live state, rewrite source values, run IK, or issue a
safety/authority receipt. A configured synchronous policy must declare a
positive `maximum_decision_latency_sec`; runtime records the elapsed time and
maps an overrun to `no_action`. This is a post-call contract check, not thread
preemption: a policy that may block must be moved to an asynchronous advisory
producer instead. With no policy configured, the path is a no-op and existing
v1 behavior and hashes are unchanged.

The initial policy contract only returns `pass`, `no_action`, or `invalid`.
The latter two become composition evaluations and are handled by the configured
downstream-disposition policy. Source-level dropping and bounded
previous-result reuse are
reserved for later additive contracts rather than silently adding persistence
or snapshot semantics now. The policy spec, exact input cut, and decision are
returned in `CompositionEvaluationRecord`; they are protocol facts, not a
canonical safety verdict.

## Public data flow

- `CompositionInputDraft/Record` is the admitted action contribution.
- `CompositionSourceEligibilityDraft/State/Store/Snapshot` is the opt-in
  source-control-plane path. Its spec binds role→source slot→producer without
  adding device fields to action records.
- `CompositionContextDraft/Record/Store/Snapshot` is the generic,
  receiver-admitted context path. `CompositionContext` is its
  representation-neutral frozen strategy view; `PoseMixingContext` remains an
  identity alias for compatibility. A context requirement defaults to
  `same_cut_global`; only `selected_input_ancestor` requests per-role lineage
  binding.
- `CompositionSpec` or a pose/qpos strategy spec freezes the numerical policy,
  source roles, weights, limits, context kinds, TTL behavior, and hash.
- `CompositionEvaluationRecord` is the complete per-evaluation read interface.
  `selected_inputs_by_role` exposes exact source lineage and
  `differences_for_role("human_pose")` retrieves all available pairwise and
  source-to-output residuals involving that role without assuming a particular
  device.
- `CompositeProposalSubmission` carries that evaluation plus the optional,
  later proposal-admission receipt.
- `SourceDifferenceObservation` reports pose translation/SO(3) residuals and
  finite-slot qpos residuals. Visualization and feedback adapters consume these
  non-authoritative facts; they do not feed motion implicitly.
- `CompositionDomainEvent` exposes immutable domain occurrences without
  choosing a dataset schema. An external adapter may project them to
  ControlTrace, visualization, metrics, or no-op observation. The evaluation
  record returns the exact domain-event batch emitted by the current call.

## Dependency rule

Allowed dependency direction:

```text
control_station / sim adapters / teleop adapters / deployment adapters
                  / dataset_interface projection adapters
                              |
                              v
                    control_composition
                              |
                              v
                         robot_motion
```

`control_composition` must not import `control_station`, `sim`, `hardware`,
`teleop_runtime`, `deployments`, `dataset_interface`, or dataset protocol
modules. `robot_motion` and `dataset_contracts` must not import
`control_composition`; pure `dataset_interface` adapters may depend on both the
domain contracts and dataset contracts to project between them.

Concrete MuJoCo, VIVE, robot-vendor, OpenPI, checkpoint, task, and device
bindings stay in their existing owner packages and implement narrow ports.

## Representation boundary

Arm contributors use typed Cartesian pose at the mainline mixed-source
boundary. Target-side resolved arm qpos remains downstream of canonical safety
and verified target resolution. Grippers and dexterous hands use their typed
qpos/native continuous components and mask-aware strategies.

The package must dispatch strategies from declared component and
representation contracts, not from a simulator, model, device, checkpoint, or
robot name.

The generic topology is a set of named components, not a fixed pair of robot
sides. An arm component always carries a Cartesian pose, so a 6DoF, 7DoF, or
other joint topology is **not** an arm-composition schema difference. Joint
count belongs to the downstream target resolver and its capability manifest.
An end-effector component declares its own qpos/native representation,
dimension, ordered names, units, active mask, and contract hash. A component
catalog may therefore bind zero, one, or several end-effectors to one arm.

The current `RobotAction` v1 carrier is a compatibility profile, not this
framework's target topology: it keys values by `left/right`, accepts at most
one hand/native end-effector per side, fixes direct arm qpos to RM65 q6, and
fixes legacy hand slots to q6/q16. Current pose strategies can already select
one arm from the left/right subset, but current formal MuJoCo target manifests
still require the coupled pair.

`robot_motion.action` now provides the additive `ActionPayloadSchema`,
`ControlComponentSpecV2`/`ActionComponentCatalogV2`,
`ComponentActionSchema`, and `ComponentAction` contracts plus the sole
explicit `RobotAction v1 <-> ComponentAction` compatibility bridge. The v2
carrier uses arbitrary component IDs and exact active coordinates; source,
timing, authority, safety, and dispatch lineage remain outside the carrier.
The sole production `ActionProposal` and `ActionArbiter` path now carries
`ComponentAction`. A normal absolute-only activation binds one complete
`ComponentActionSchema`; formal relative composition may instead bind one
explicit input schema per role and one absolute output schema. Each source
proposal may provide only a subset of its component values.
Coverage and selection groups are derived from those values rather than
redeclared by callers. Atomic/dependency checks occur after component
selection. `RobotAction` v1 is materialized only at explicitly identified
legacy observation or target/safety compatibility edges; it is not a second
proposal carrier.

Overlapping component resource claims are not resolved inside composition or
the Arbiter. A lower-owned `ComponentTargetCapabilitySpec` declares exact
active component sets accepted for each resource and returns a structural
assessment. That assessment is not a HOLD, safety verdict, target-admission
receipt, or fallback decision.

The lower execution boundary also owns representation adaptation after the
Arbiter decision has been authoritatively published. `control_composition`
neither imports that adapter nor distinguishes a policy chunk, replay sample,
human input, or any model/backend-specific representation at this stage. It
only supplies the component-keyed candidate and its causal lineage.

## Generic context admission

The runtime now has one receiver-admitted `CompositionContextRecord` with:

- session/domain, channel, context kind, producer epoch and monotonic sequence;
- produced/sample time, receiver availability time, clock domains, effective
  interval/TTL, and admission sequence;
- payload contract hash plus immutable inline semantic payload or stable blob
  references;
- source fact IDs/hashes, parent context IDs, availability/fault state, and
  optional producer/model/artifact/observation-cut lineage.

Policy-specific fields are optional lineage, never generic admission
requirements. `ControlCompositionRuntime`, the synthetic policy-node protocol,
and the MuJoCo adapter all use this generic path. Dataset compatibility and
event-schema migration remain external projection concerns.

The canonical store performs validation, epoch/sequence admission, cut
selection and retention once. Adapters may normalize sensor, replay,
human, policy, model-feature, and task-state inputs, but may not maintain
parallel hidden “latest context” state.

## Predeclared information demand

`CompositionInformationPlan` is independent of activation identity and dataset
capture. Before optional sources acquire, copy, freeze, or admit data, it
declares the latest-value context channels accepted by the current composition
runtime, their payload-contract hashes, maximum age, and whether access is
metadata-only, inline, or through stable blob references. It may also request
the currently implemented composition-local immediate-difference observation
and its diagnostic level. A future same-cut pre-mix view belongs to that same
immediate-difference module; it must not be confused with downstream execution
difference or silently widen the current request/hash field.

This plan is entirely upstream of target execution. Human, replay, policy,
history, task-state, visual, tactile, and model-feature adapters satisfy it
directly through the context admission path. Target capability, IK, planner,
backend, and execution-observation negotiation never participate in obtaining
pre-composition information.

When no plan is supplied, existing callers retain their established behavior.
When a plan is supplied, undeclared channels and incompatible payload contracts
are rejected before entering the context store. Source adapters should query
the plan before acquisition; runtime rejection is a final guard, not a reason
to acquire unused camera or tactile payloads first.

If control-loop resources are tight, a source frequency cannot be sustained,
or copying/serialization competes with control work, first use
`CompositionInformationPlan` to narrow channels, payload access, maximum age,
and optional derived computations. This optimization must not change action
authority, safety, activation-v2 identity, or dataset capture requirements.
Data needed only by a dataset should bypass composition; data needed by both
paths should share one immutable sample while composition receives only the
declared lightweight view/reference.

## Strategy input and latency contract

### Composition semantics versus strategy implementation

`CompositionSpec` and `FormalCompositionStrategySpec` are separate protocol
layers and must never become interchangeable configuration containers:

| Contract | Owns | Must not own |
| --- | --- | --- |
| `CompositionSpec` | input/required roles, algebra, component representations, weights or allowed weighting policy, missing/fallback behavior, disagreement limits, and the typed component-rule-set hash | implementation/module/model identity, context acquisition, latency budget, authority, safety, or dispatch |
| `FormalCompositionStrategySpec` | strategy ID/version, the exact `CompositionSpec` hash it implements, required context channels, implementation dependency hashes, implementation-only parameters, and inline latency budget | a second copy of component weights, fallback behavior, disagreement limits, authority, safety, or dispatch |

The binding is one-way:

```text
ComponentCompositionRuleSet
        |
        | semantic_content_hash
        v
CompositionSpec v2
        |
        | semantic_content_hash
        v
FormalCompositionStrategySpec
```

The component rule set is the single semantic source of truth. A formal
strategy may declare the same rule-set hash in
`implementation_contract_hashes` to prove that its implementation binds that
contract, but it may not redefine the rules in `parameter_payload`. The
framework-built result binds the exact composition spec, strategy spec,
strategy input cut, and output action.

The protocol-neutral event stream emits the complete
`component_composition_rule_set` followed by one
`formal_component_composition_binding` that binds the composition-spec,
rule-set, and formal-strategy hashes. The external v2.4 ControlTrace projection
uses additive record kinds with the same names; established v1 spec and
strategy records are not widened or rehashed.

`formal_component_strategy_v2` is additive. Existing v1 algebras retain their
exact payload and `dexteleop.composition_spec.v1` hash. The v2 mode uses
`dexteleop.composition_spec.v2`, requires one typed component-rule-set hash,
forbids deterministic-v1 numerical fields, and fails closed when no matching
formal strategy is installed. This does not grant the strategy authority:
proposal admission, canonical safety, target resolution, reference generation,
and dispatch remain downstream.

The first formal compound authority profile is consequently defined in
`robot_motion`, not in this package. It binds the hashes emitted here, but
`control_composition` remains unaware of the backend, Arbiter ownership,
safety result, group generation, receiver admission, or dispatch receipts.
This preserves the same composition runtime for sim, real, replay, synchronous
policy chunks, and RTC-produced action streams.

An advanced strategy may use every admitted information class, including
current source actions, bounded action/decision history, future policy chunks,
measured state, lifecycle/authority/source-health state, task labels,
camera/tactile/force features, confidence/uncertainty, and earlier
difference/execution observations. “May use” means the information is first
frozen behind a typed context channel; it never means live access to a model,
sensor, backend, GUI, filesystem, or network.

Every strategy declares requirements per channel:

```text
channel + accepted payload contract(s)
required/optional + missing/expired behavior
latest/history window/count + maximum age
inline/blob-reference access mode
latency class + configured decision deadline
```

One immutable `StrategyInputSnapshot` binds the target time, causal cut,
producer epochs, selected action roles, selected context windows, and previous
decision lineage. A strategy is a pure deterministic mapping from that
snapshot to a result and diagnostics.

Two execution classes are allowed:

1. **Inline deterministic strategy.** No I/O, wait, locks, device/model calls,
   or mutable cache reads. It must finish before the profile's decision
   deadline. The recommended initial budget is p99 no greater than
   `min(2 ms, 10% of the composition period)`; each execution profile must
   publish its actual budget and measurements. A miss follows the declared
   `no_action`/HOLD/fallback path and is traced.
2. **Asynchronous advisory producer.** Any expensive vision, history model, or
   future policy-driven mixer runs outside the control loop and publishes a
   typed, time-bound context result before the cut. The composer never waits
   for it. Late/expired results affect only a later eligible cut or are
   rejected.

A future policy-generated mixer should output a typed advisory containing
component/role weights or route, applicable target interval, confidence,
model/artifact/input-cut identity, and payload contract hash. A small local
deterministic strategy consumes that advisory. The model output never owns
authority, bypasses safety, or directly dispatches; request/result epoch,
deadline, late completion, and default-off behavior follow the same causal
rules as other asynchronous admitted producers.

Integration sequence:

1. define and version the context channel/payload contract;
2. admit it through the generic receiver adapter with clocks, TTL, lineage,
   availability, and faults;
3. declare the strategy's required/optional windows and fallback;
4. build exactly one immutable same-cut snapshot;
5. evaluate the pure strategy within its budget;
6. return result, core lineage, optional differences, and diagnostics;
7. test missing, stale, late, reordered, clock-reset, burst, deadline, and
   producer-restart cases without enabling motion.

## Time-window retention

`CompositionInputStore`, `CompositionContextStore`, and the opt-in
`CompositionSourceEligibilityStore` use an explicit
`TimeWindowRetentionSpec`; the former fixed `4096` lifetime capacity and
`capacity_exhausted` behavior are removed. Every profile declares:

- history retained behind the closed target watermark;
- maximum future lead at receiver admission;
- maximum effective interval duration;
- optional late-arrival allowance for an explicitly logical/replay clock.

Snapshot capture is atomic with window advancement: it first materializes the
exact same-cut selected records, then advances a monotonic
`closed_target_through` watermark and removes full store payloads whose
`effective_until` is outside the retained history window. Existing snapshots
and evaluation records own their immutable selected record objects, so store
pruning cannot alter an already returned result. A compact last-sequence
tombstone per producer epoch survives pruning and continues to reject duplicate
or rolled-back identities. The source-state store additionally retains the
latest state as a fail-closed tombstone, so pruning cannot expose an older
`available` heartbeat after a later pause or fault. `retention_snapshot()`
reports the retention-spec hash, watermark, retained/pruned counts, and
tombstone count.

This is a semantic time window, not a lossy ring overwrite. It does not by
itself bound a malicious producer's record rate or blob size inside the valid
window, and it does not replace durable recorder backpressure. Those
resource quotas remain a separate production hardening gate.

## Difference modules

Difference is split by owner and causal stage, not by version. Both modules are
required, but each can be independently disabled and can evolve behind an
explicit capability/profile.

1. **CompositionImmediateDifference** belongs to `control_composition`. It uses
   one immutable cut and returns a source-to-source view before numerical
   composition. This is the low-resource result that a pre-mix policy, GUI, or
   lightweight feedback path may consume immediately. It never includes the
   composed result, canonical safety, IK, dispatch, or measurement claims.
2. **PostCompositionExecutionDifference** is a separate module. It starts from
   the composed candidate and consumes whichever downstream observations the
   active capability actually exposes: target, optional reference, command,
   dispatch and measurement. The execution portion is normally asynchronous.
   It cannot call a value an actual execution deviation without the
   corresponding downstream identity and time contract, and it never requires
   every backend to expose all intermediate stages.

`off`, `summary`, and `full` are resource modes within either module, not
compatibility versions. The current `SourceDifferenceObservation` is a
provisional transitional carrier: its pairwise fields represent the first
module and its output-residual fields represent the second module. New code
must select an explicit capability/profile instead of treating that combined
carrier as a permanently frozen public shape.

The provisional ports in `observability/modules.py` expose only the common
lineage envelope and opaque capability-owned payload. The included
`SourceDifferenceImmediateObserver` and `SourceToComposedDifferenceObserver`
are replaceable pose/qpos fixtures for the current development path; they are
not wired into the runtime control loop by default. A profile must first
declare its demand and synchronous budget before an immediate observer is
installed; a post-mix observer is installed only by the downstream owner that
can provide the corresponding result/dispatch/measurement receipts.

`observability/execution_difference.py` consumes passive downstream projections;
it does not own reference generation or reconstruct an interpolation timeline.
For an actual reference comparison, the lower owner must emit the exact
reference object used by execution (or a lossless immutable projection of it),
with component identity, target lineage, effective interval, availability and
clock domain. The current metric set supports same-clock `joint_position` and
framed `cartesian_pose` residuals. Pose translation is reported in the declared
reference frame and rotation as a reference-pose-local rotation vector. It
never uses nearest logical target time, compares cross-clock values, treats an
unsupported representation as a generic vector, or infers authority/
cancellation from timestamps. No production caller is enabled merely by
adding the seam.

Reference observation is optional and demand-driven. A downstream
`ExecutionObservationPlan` may request decimated or full-rate reference
frames, independently request measurement frames, and optionally require the
exact representation for every selected component.
`None` or an all-off plan creates no acquisition or publication work. A non-off
plan must bind in full to an `ExecutionObservationCapabilitySpec` during target
construction and requires a nonblocking observation sink; unsupported
components/modes fail construction rather than degrading silently. Every
published value carries the binding/plan identity, and lower-owned
`ExecutionObservationWatermark` facts make sink rejection or missing delivery
detectable without adding one redundant receipt per frame.
Here, `full_frames` means every frame produced by the exact stage named in the
bound capability. It does not silently promote a station-rate observer into a
backend servo-rate or physics-rate tap.

Target lifecycle/identity is not another observation mode. It is already
reported exactly once through request, submission, completion, admission and
activation receipts plus `TargetExecutionStatusSnapshot`; the observation
sink does not duplicate those facts.

The owner computes the reference once and fans out the same object; composition
never receives or executes an interpolation algorithm. MuJoCo uses a
measurement-coupled point sample of the qref selected by the target owner. When
a pose representation is explicitly bound, the MuJoCo owner derives
`FK(active qref)` and `FK(measured qpos)` in private model data and publishes
their raw qpos facts plus the kinematics contract as lineage. This is a
downstream-owned lossless projection, not composition-side FK or reconstructed
interpolation. It avoids retroactively mutating an earlier open interval after
supersession while still allowing an exact same-clock comparison. Unbound
pre-resolver point samples remain incomparable. Provenance may carry an
algorithm ID/hash, but an offline recomputation must be labeled
`ReconstructedReference`, never actual reference.

The two modules return different facts and must not be merged into one record
or treated as successive schema versions. Visualization and future feedback
adapters may consume either or both, but neither observation path grants or
changes motion authority.

## Development-stage contract policy

While these modules are still being developed, only the following are treated
as stable boundaries: ownership, one-cut lineage, stage identity, explicit
availability, optional information demand, and the rule that observations are
non-authoritative. The following remain provisional until real consumers and
replayable fixtures exist: the complete metric catalog, default role/component
pairs, reference/measurement payload shape, cross-clock mapping,
frame-conversion policy, threshold values, payload size limits, debug fields,
and feedback mappings.

Existing hash domains identify already-produced facts and must remain readable,
but a new difference experiment is not a downstream compatibility promise just
because it has a hash. Consumers must opt into a named capability/profile and
handle unavailable or unsupported fields. New fields should be additive or
profile-gated; do not widen an established record merely to anticipate future
modalities.

## Observation versus recording

Composition always returns the immutable domain objects needed to explain its
own numerical decision: selected action/context lineage, cut, epochs, spec,
result, requested differences, and diagnostics. Optional debug computations
must be disabled before construction when not requested. A runtime-bound
`CompositionInformationPlan(diagnostic_level="core")` now retains the compact
decision status/reason but does not construct detailed optional diagnostics or
retain pre-mix elapsed-time diagnostics. `diagnostic_level="debug"` enables
them. A missing information plan preserves the existing compatibility behavior
and therefore still returns detailed diagnostics.

Durable core/debug tiers, media streams, queues, backpressure, episode
membership, seal/commit, and final dataset formation are external recorder
policy. A recorder may subscribe to `CompositionDomainEvent`, but recording
configuration must not cause composition to acquire an otherwise unused
modality. Heavy data used by a strategy should normally enter context through a
stable blob reference; the actual payload is independently captured only when
an external capture profile requests it.

## Package layout

The reusable implementation is now physically owned here:

- `composition/core.py` owns admitted action inputs, target-time snapshots,
  generic algebras, results, and proposal-promotion lineage;
- `composition/context.py` owns immutable multimodal strategy context;
- `composition/pose.py` and `composition/qpos.py` are the stable
  representation-specific public surfaces; their current shared internal
  implementation remains in `composition/strategies.py` to avoid duplicating
  component projection, hashing, and compound-join rules;
- `observability/` owns source/output differences and bounded, non-authoritative
  feedback contracts;
- `evaluation.py` owns the complete per-cut return record and typed diagnostic;
- `events.py` owns protocol-neutral emitted domain occurrences;
- `information.py` owns predeclared composition information demand;
- `eligibility.py` owns opt-in source slot/producer bindings, liveness state,
  leases, status epochs, re-arm lineage, and same-cut eligibility snapshots;
- `runtime.py` owns receiver-side composition evaluation and promotion;
- `policy_node.py` owns the composition-aware synthetic node protocol;
- `lifecycle/` owns handoff, activation, rearm, and the backend-neutral session
  registry.

The former `robot_motion.action.{composition,mixing_strategy,source_difference,
guidance_feedback}`, `robot_motion.deploy.topology_runtime`, and
`robot_motion.deploy.policy_node` implementation paths were removed after all
local callers and the sole external repository, remote OpenPI, were audited.
No compatibility shim is kept because it would reverse the required dependency
direction. `control_station` retains only explicit application adapters for
Station operation results, publication barriers, and HOLD/cancel ordering.
No concrete simulator, device, robot, or model adapter was copied here.

## Legacy/shadow compatibility path

`composition/strategies.py` still contains the original pose and hand-qpos
strategy kernels plus `evaluate_component_mixing_shadow()`. This path projects
an already-built arbitration/proposal snapshot and is retained only as a
non-authoritative numerical compatibility oracle while the formal input path
is completed. It may compute an immutable shadow decision, but it must not
publish a proposal, advance authority or epochs, run canonical safety/IK,
enqueue a reference, dispatch, or own episode state.

The formal path is:

`CompositionInputDraft -> receiver admission -> causal cut/activation/eligibility/context -> CompositionEvaluation -> optional proposal promotion`.

New device, model, simulator, context, liveness, or lifecycle features belong
on that formal path. Do not extend the compatibility path with new production
orchestration. Its reusable SE(3) and active-mask qpos numerical kernels may
be invoked behind the formal strategy protocol described below. The shadow
orchestration can be removed after formal pose/qpos result parity, lineage
parity, and `control_station` caller migration are contract-tested.

## Formal strategy migration

The opt-in v2 seam in `composition/formal_strategy.py` binds four immutable
facts: a strategy spec, the exact action/context/information-plan input, a
canonical v4 `CompositionEvaluation`, and a result binding. The framework builds
the evaluation identity; strategy implementations return only status, reason,
candidate, and the context IDs they actually used. This prevents two strategies
that happen to produce the same action from sharing a result/promotion
identity.

The runtime default remains the deterministic compatibility composer. A new
profile must explicitly inject a formal strategy, whose context channels must
match the runtime `CompositionInformationPlan`. Formal strategies are bounded
inline computations. A model, RPC, or other slow producer must publish a
receiver-admitted advisory/context fact rather than run inside the control
loop. A declared latency overrun is detected and fails closed, but v2 does not
preempt a blocking implementation.

Retirement sequence:

1. run deterministic-v1 and formal-v2 numerical/parity fixtures (complete);
2. expose an opt-in MuJoCo strategy injection and verify the CS6b1h v2
   binding, v4 proposal/trace lineage, and downstream dispatch parity
   (contract fixture complete; no default profile switch);
3. make formal-v2 the default only after all active callers and stored trace
   readers accept it;
4. remove deterministic-v1 orchestration and exact-hash fixtures after the
   documented compatibility window; retain explicit readers for historical
   facts when required.

## Opt-in strategy hot control

Hot control is intentionally smaller than a general dynamic configuration
system. It has two rates and no separate topology/context envelope object:

```text
runtime-fixed compatibility boundary
    action schema + component representations/frames
    input roles/source bindings
    CompositionInformationPlan
    strategy implementation identity

low-frequency immutable revision
    exact CompositionSpec + component rules + formal strategy spec
    fallback and dynamic-parameter contract/defaults

high-frequency parameter record
    complete component-keyed base/overlay alpha values
```

Action, eligibility, deadman, task state, history, and visual/tactile facts
remain ordinary admitted input/context. They are not configuration revisions.
Adding a component, representation, frame, role, context contract, strategy
implementation, or downstream capability crosses the runtime boundary and
requires a new activation/runtime binding.

`CompositionStrategyControl` is optional. With no controller, the existing
`evaluate_composition(spec=...)` path, hashes, and strategy calls are
unchanged. An enabled controller provides:

- one stable strategy instance;
- one immutable initial/current revision plus receiver-linearized successor
  revisions using compare-and-swap predecessor identity;
- receiver-admitted `latched` or `ttl` alpha records bound to one exact
  revision and producer generation;
- `evaluate_hot_composition()`, which selects revision, parameters, action
  inputs, eligibility, and context from one `ControlCut`;
- a v3 formal-strategy input and v6 evaluation identity only on the opt-in hot
  path;
- protocol-neutral revision/parameter/snapshot/binding events and additive
  ControlTrace projections.

The initial high-rate parameter semantics are deliberately narrow:

```text
weight(base_role)    = 1 - alpha[component]
weight(overlay_role) = alpha[component]
alpha in [0, 1]
```

One parameter record carries the complete component vector, so an evaluation
cannot observe a new arm alpha and an old hand alpha. Multiple records admitted
inside one control period do not create catch-up work: the next cut selects the
latest covering record. A rejected record leaves the previously selected
record unchanged. A newly applied revision uses its complete initial alpha
vector; records bound to the predecessor cannot cross the revision boundary.

The first revision implementation is equally narrow: it permits atomic rule
replacement and switching between the installed formal-component strategy's
predeclared `weighted_component_fusion_v1` and
`discrete_component_owner_v1` kernels, with identical
strategy implementation/version, context channels, algebra, input/required
roles, component IDs, representations, and contributing-role topology. It
does not dynamically load unrelated numerical kernels or acquire new context
resources. Arbitrary implementation loading is not a hot configuration
feature. The separate allocation contract below handles predeclared
context-driven mode selection; it does not overload alpha records or strategy
revision.

Changing alpha from zero to one changes the realized contributors while the
composition proposal producer remains the sole downstream authority owner.
Hot control never performs HOLD, authority release/acquire, canonical safety,
IK, reference generation, or dispatch.

## Activity-driven source allocation

`control_composition.allocation` is an opt-in, backend-neutral path for the
common interaction profile:

```text
quiet near active anchor       -> base_only candidate
motion or nonzero residual     -> relative_overlay candidate
large motion/residual evidence -> absolute_overlay_owner candidate
```

Role names are profile data, not hard-coded human/policy semantics. A
`CompositionAllocationSpec` predeclares `base_absolute_role`,
`relative_overlay_role`, `absolute_overlay_role`, and component-keyed threshold
policies. The allocation reducer consumes bounded-window
`ComponentSourceActivityObservation` facts and separate anchor-bound
`ComponentResidualMagnitudeObservation` facts. Motion describes change over a
bounded window; magnitude describes the latest relative action against the
identity of its exact `ResidualAnchorReceipt`. Standard providers cover
Cartesian pose translation/rotation and arbitrary-dimensional
joint-position/continuous vectors. Every observation carries a provider ID and
contract hash; a custom provider must declare both explicitly. Normalization
scales remain profile data.

One admitted source sample may be projected into several fixed roles through
`CompositionInputBatchDraft`. Such a batch is all-or-none at a receiver cut and
shares source lineage, timing, activation, and any semantic binding. The
allocation input path uses this for absolute and residual views of one source;
allocation never changes that source's role dynamically. `AllocationEvidenceWindow`
retains accepted pairs by mapped time, not by an arbitrary item capacity, and a
cut sees a pair only after its final batch admission sequence.

Residual-anchor identity is a formal input semantic binding, not debug
metadata. It participates in the input hash and is recorded separately as
`composition_input_semantic_binding`, keeping the established v2.4 base input
record schema unchanged. Relative and absolute-overlay modes require the
selected input binding to equal the anchor carried by the admitted allocation
context. Projected trajectory inputs apply the same rule.

The reducer distinguishes quiet-at-anchor from quiet-at-nonzero-residual, and
distinguishes both from missing, not-yet-available, or expired evidence.
Base-to-relative entry may use motion **or** magnitude; returning to base
requires motion **and** magnitude below their independent exit thresholds plus
minimum dwell. Multi-component evaluation is atomic and rejects mixed anchor
receipt generations. Missing/stale evidence returns `no_decision`, preserves
state, and does not silently become `base_only`.

A ready allocation decision is published through the existing
`CompositionContextStore`. The reusable `AllocationPublicationRuntime` keeps
the pure reducer result as a candidate until the exact context receipt is
accepted. Only then does it commit the saved next reducer state. A terminal
rejection keeps the old state but consumes the monotonic producer sequence; an
exception or unknown outcome retains one immutable pending draft for exact
retry. A generic `duplicate` receipt counts as success only after the retained
record's producer-side submission hash is proven identical. Ordinary context
admission continues to reject duplicates.

Each allocation scope permits one pending attempt and one latest coalesced
evidence basis. Pending never becomes active, never blocks composition from
using the last admitted context while it remains valid, and never creates a
backlog of missed evaluations. The attempt binds the source-fact basis cut,
component-atomic decision, candidate state, producer sequence, and
`CompositionActivationScope`. Closing that activation permanently invalidates
the local runtime; a late receipt cannot advance a later activation. The first
implementation is in-process receipt-driven consistency, not durable
crash-consistent exactly-once.

The core owns no thread or worker. `offer_basis()` and `resolve_admission()`
are short state transitions suitable for a control-facing seam. The
`attempt_admission()` convenience releases the runtime lock around its callback
but the callback may still block its calling thread; an application therefore
runs it on a bounded non-control-loop worker, or transports the pending draft
asynchronously and returns the typed receipt to `resolve_admission()`.

`AllocatedPoseComponentCompositionRule` and
`AllocatedQposComponentCompositionRule` select the decision on the same formal
action/context cut after its admission, then either copy the base, apply the
declared relative delta, or copy the absolute overlay. All three input roles and schemas are
predeclared at runtime creation; source required/optional behavior continues
to belong to `CompositionSpec` and eligibility, not the allocator.

Profiles that make `absolute_overlay_owner` promotable may attach an
`AllocationTransitionContinuityPolicy`. Its activation-scoped reducer is
component-keyed and target-time driven. On absolute entry it aligns the current
raw absolute pose/qpos to the previous composed target, keeps that alignment
while later human deltas arrive, and on a later mode change decays the target
seam correction over the declared handoff interval. Exact state-before and
prepared-state-after hashes, invocation identity, selected modes, and
correction kinds are returned as `strategy_result_data` actual/replay
information. Preparation does not advance state; only acceptance of that exact
proposal commits it. Exact duplicate invocations are idempotent and a new
non-increasing committed target time is rejected.
Projected trajectory preview intentionally does not mutate this stateful
reducer and therefore exposes the raw advisory composition view.

`absolute_overlay_owner` is deliberately a **soft composition mode**: the
composer remains the sole proposal/authority owner. A true exclusive takeover
still requires the Control Station HOLD/release/rebase/READY lifecycle and is
not driven frame-by-frame by this reducer. Allocation also owns no canonical
safety, IK, Reference generation, command smoothing, or dispatch. A future
learned allocator must publish a bounded-TTL context/advisory asynchronously;
the inline control loop must not wait for it.

Additional stillness, contact, intent, vision, tactile, or learned classifiers
remain advisory providers. Their facts need explicit identity, mapped timing,
TTL, component scope, and lineage. Publishing such a fact cannot itself rebase,
request HOLD, or change authority. An application may later opt into an
`automatic_rebase` profile that converts one declared provider fact into the
same typed rebase transaction.

The generic `control_composition.triggers` layer now supplies the boundary for
that conversion without enabling a profile. A replaceable provider receives an
immutable, caller-selected `CompositionTriggerEvaluationBasis`; an input/state
classifier (for example, large anchor-relative hand motion) returns condition
evidence, while a button or another already-admitted event returns direct-event
evidence. A rule maps either source to the same requested operation contract.
Condition rules emit only on a rising edge and rearm on an explicit inactive
sample. A device adapter must convert a held button level into uniquely
identified press/release edges before using the direct-event origin; distinct
events emit once, and exact duplicates are idempotent.
Every proposal binds the receiver `ControlCut`, activation, source generation,
sequence, TTL, component scope, payload contract, and causal facts.

The result is deliberately a non-authoritative
`CompositionOperationTriggerProposal`. It is not a Station intent, allocation
context, rebase request, HOLD, authority transition, or action proposal. The
owning application profile must explicitly bind it to the existing operation
owner, whose admission receipt and reducer remain canonical. Consequently a
classifier, button callback, GUI, or device adapter can share one trigger seam
without gaining a path to mutate composition or execution directly. No GUI,
device, MuJoCo, replay, deployment, or real-robot caller is wired in this gate.

Clearing a producer-local evidence window is not an immediate rebase fence:
an older admitted input and allocation context can remain mutually consistent
until their support expires. The allocation strategy therefore declares a
`ResidualAnchorLifecycleSpec`, and the runtime selects its receiver-admitted
`pending`/`ready` record at the same `ControlCut` as actions and allocation
context. This low-rate stream has no TTL: a record remains valid within its
activation until a monotonically newer generation supersedes it.

The three facts are parallel inputs to one evaluation, not three serial
processing stages:

```text
one receiver cut
  -> active residual-anchor lifecycle
  -> selected allocation context
  -> selected component inputs
  -> one formal allocation decision
```

At targets before a rebase boundary, the prior ready receipt remains active.
At or after the boundary, pending returns typed `no_action`; it is not HOLD.
Ready may only complete the exact request at the exact pending boundary, and
selected relative/absolute inputs plus allocation context must bind that same
receipt. The allocation publication runtime simultaneously fences pending or
late work from the older generation, so a late accepted context receipt cannot
commit old reducer state. Initial anchor capture publishes the absolute sample
and identity residual as one batch before publishing ready.

Frozen corrected-horizon projection carries this same lifecycle projection and
rebuilds a lifecycle snapshot independently for each future keyframe. It never
rereads a store, predicts a future receipt, or makes the preview promotable.

The reusable reducer, lifecycle receiver/store, receipt-driven publisher,
context adapter, and formal rule are implemented and pure-tested. The separate Station rebase port is
default-off and only binds an already-accepted intent to the pure transaction;
it does not capture a sample or send an action. A default-off MuJoCo shadow
binding exercises the ordinary context store and formal same-cut evaluation
with virtual source facts but cannot admit a proposal. A separate default-off
Station gate can recapture a formal cut and admit that exact ready result to
the existing MuJoCo composer proposal port; it does not step the runtime or
own authority, target/reference generation, safety, IK, or dispatch. No GUI,
deploy frontend, real-device path, or production profile enables either gate
by default.

## Current implementation boundary

Implemented and locally contract-tested:

- deterministic target-time selection across different source rates;
- action-TTL/source-lease orthogonality, source pause/fault/disconnect,
  lease expiry, late-heartbeat rejection, and typed re-arm recovery;
- single-frame pose owner selection and weighted SE(3) pose fusion;
- explicit single-arm subset or current coupled left/right arm pose strategy
  inputs (formal current MuJoCo target remains coupled);
- active-mask-aware 1DoF/q6/q16 hand/gripper qpos interpolation;
- generic frozen context usable by both pose and qpos strategy protocols;
- strict pose-plus-hand structural join in non-authoritative shadow;
- same-cut pose/qpos source differences, human-role lookup, and no-I/O bounded
  guidance contracts;
- complete evaluation return records, domain events, lifecycle,
  handoff, re-arm, proposal promotion, and package dependency firewalls.
- whole-horizon source-intent preview with locked/speculative trust labels and
  no execution claim;
- opt-in component-keyed activity allocation with separate motion and
  anchor-relative magnitude evidence, explicit metric-provider identity,
  pose/qpos metrics, hysteresis/dwell/TTL lineage, existing-context admission,
  and formal base/relative/absolute soft selection;
- session-local active-anchor revisions with pending-rebase fencing, exact
  trigger/fresh-sample receipts, deterministic replay, and a default-off thin
  Station intent port; no application or execution profile is enabled;
- opt-in formal strategy v2 identity, exact context/information-plan lineage,
  canonical v4 result promotion, strict ControlTrace projection, and a
  deterministic migration adapter without changing the default v1 path.
- a backend-neutral MuJoCo strategy injection port and CS6b1h formal-pose
  parity fixture; strategy identity upgrades the application binding to v2
  while the default binding remains byte/hash-compatible v1.
- an optional same-cut `CompositionPreMixPolicy` port with pass/no-action/
  invalid decisions, exact policy input lineage, and default-off behavior;
  it does not own motion safety or alter source values.
- opt-in receiver-linearized component alpha hot updates, immutable
  weighted/discrete-owner revisions, exact same-cut bindings, concurrent
  update/evaluation coverage, and additive ControlTrace projections; no
  Control Station, sim, or real profile enables them yet.

Still gated for later work:

- migration of the remaining legacy `RobotAction` numerical composition
  algorithms to native component values; `component_union` is already native,
  while residual and weighted-pose compatibility algorithms still materialize
  an explicit v1 view. Their promoted candidate enters the sole component-keyed
  proposal/Arbiter path, and target processing remains lower-owned (arm joint
  DOF remains a downstream resolver concern);
- declared context windows, preemptive strategy latency enforcement, and
  asynchronous policy-driven mixing advisories;
- a formal compound pose-plus-hand authoritative execution profile and real
  end-effector safety/acceptance receipts;
- production adapters that admit raw visual/history/state streams as
  `CompositionContext` without embedding mutable backend objects;
- per-profile rate/blob quotas and durable trace backpressure beyond the
  implemented semantic time-window pruning;
- independently selectable immediate and post-execution difference profiles,
  plus strict core/debug recording separation;
- real visualization and haptic/force-feedback adapters (only no-I/O guidance
  validation exists);
- jitter/burst/clock-reset endurance, clean wheel installation, additional sim
  backends, single-arm target capability, and authorized real-device
  verification.
- context-resource prewarming, generic meta-strategy mode selection,
  dynamically loaded strategy implementations, and generalized hard authority
  transition; these are not part of the current alpha hot-control slice.

These gaps do not change existing teleop, replay, policy, sim,
or real deployment defaults. New integrations should implement adapters and
ports around this package rather than add simulator-, model-, robot-, or
device-specific branches inside it.

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
  -> pose/qpos component result + differences + complete evaluation record
  -> proposal admission
  -> authority + canonical safety + target resolver/reference + dispatch
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
- composition results and proposal-promotion eligibility;
- logical composition sessions, activation epochs and source handoff reducers;
- source/output difference observations and non-authoritative feedback facts;
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

The output boundary is a proposal candidate plus lineage, never a motion
permit:

```text
contributors
  -> control_composition
  -> RobotMotion proposal admission
  -> authority/arbitration
  -> canonical safety
  -> target resolution/reference
  -> dispatch permit
```

## Time and rate semantics

Sources do not need to run at the same frequency. Every admitted contribution
declares receiver-clock `[effective_from, effective_until)` validity. For one
`target_effective_at`, the receiver captures a single admission cut and:

1. excludes facts that arrived after the cut or belong to an inactive producer
   epoch;
2. excludes facts whose TTL does not cover the target;
3. selects the deterministic newest valid fact per declared role;
4. evaluates all roles and contexts on that same cut;
5. emits one result for downstream proposal admission.

A slower source may therefore be reused only while its explicit TTL remains
valid; a faster source may replace only future evaluations. Late completion
never rewrites an existing result. `control_composition` does not sleep,
resample robot commands, or own a worker queue. High-rate reference generation
and physical dispatch remain downstream in `robot_motion`, bound to the
result's target time and lineage.

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
authority or execute HOLD. The existing formal authority gate remains the
owner of the downstream HOLD disposition.

## Input, safety, and fallback layers

Source-specific values are normalized by adapters into canonical
`RobotAction` components before admission. Invalid identity, clock, epoch,
interval, representation, shape, or numerical data is rejected at that
boundary. Composition then handles declared missing-role behavior, source
conflicts, weight normalization, pose-frame compatibility, quaternion validity,
qpos active masks, and bounded residual/disagreement limits.

The result is only `ready`, `no_action`, or `invalid`. A formal authority gate
may promote an exact ready result to one proposal or HOLD. Canonical workspace,
joint, velocity, force, IK, reference-generation, generation/epoch, and target
backend safety remain mandatory downstream. Composition fallback can reduce
availability; it can never bypass canonical safety or grant motion authority.

## Public data flow

- `CompositionInputDraft/Record` is the admitted action contribution.
- `CompositionSourceEligibilityDraft/State/Store/Snapshot` is the opt-in
  source-control-plane path. Its spec binds role→source slot→producer without
  adding device fields to action records.
- `CompositionContextDraft/Record/Store/Snapshot` is the generic,
  receiver-admitted context path. `CompositionContext` is its
  representation-neutral frozen strategy view; `PoseMixingContext` remains an
  identity alias for compatibility.
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
still require the coupled pair. Removing those carrier/target limits requires
an additive, component-keyed action carrier and explicit v1 adapters; it must
not mutate v1 hashes, layouts, or default execution behavior.

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
metadata-only, inline, or through stable blob references. It may also bind one
optional source-difference spec and the requested diagnostic level.

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

## Difference return levels

Difference observation is split into two independently configurable stages:

1. **Immediate composition difference** is computed from the same cut's
   selected sources and composed result before IK, smoothing, safety/reference,
   or backend I/O. It is cheap and can be returned synchronously in
   `CompositionEvaluationRecord`. Modes are `off`, `summary`, or `full`;
   `off` must avoid building/copying the observation.
2. **Post-processing/execution difference** is a later, asynchronous fact. It
   may compare the composed desired value with canonical-safe output, resolved
   IK target, smoothed/limited reference, dispatched command, and time-aligned
   measured state. Every comparison names its two stages and carries result,
   safety, resolver, reference, dispatch, and measurement identities as
   available. It cannot claim an “actual” deviation without the corresponding
   dispatch/measurement receipt.

The second stage is produced by downstream `robot_motion`/backend observers,
not by composition. Its modes are also independently `off`, `summary`, or
`full`; callers may request neither, either, or both. Feedback adapters may
later consume either stage, but feedback behavior is outside the current
implementation scope.

## Observation versus recording

Composition always returns the immutable domain objects needed to explain its
own numerical decision: selected action/context lineage, cut, epochs, spec,
result, requested differences, and diagnostics. Optional debug computations
must be disabled before construction when not requested.

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

`CompositionInputDraft -> receiver admission -> causal cut/activation/eligibility/context -> CompositionResult -> optional proposal promotion`.

New device, model, simulator, context, liveness, or lifecycle features belong
on that formal path. Do not extend the compatibility path with new production
orchestration. Its reusable SE(3) and active-mask qpos numerical kernels may
be invoked behind the formal strategy protocol described below. The shadow
orchestration can be removed after formal pose/qpos result parity, lineage
parity, and `control_station` caller migration are contract-tested.

## Formal strategy migration

The opt-in v2 seam in `composition/formal_strategy.py` binds four immutable
facts: a strategy spec, the exact action/context/information-plan input, a
canonical v4 `CompositionResult`, and a result binding. The framework builds
the result identity; strategy implementations return only status, reason,
action, and the context IDs they actually used. This prevents two strategies
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
- opt-in formal strategy v2 identity, exact context/information-plan lineage,
  canonical v4 result promotion, strict ControlTrace projection, and a
  deterministic migration adapter without changing the default v1 path.
- a backend-neutral MuJoCo strategy injection port and CS6b1h formal-pose
  parity fixture; strategy identity upgrades the application binding to v2
  while the default binding remains byte/hash-compatible v1.

Still gated for later work:

- additive component-keyed action carrier/adapters for arbitrary named arm and
  end-effector topology (arm joint DOF remains a downstream resolver concern);
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

These gaps do not change existing teleop, replay, policy, sim,
or real deployment defaults. New integrations should implement adapters and
ports around this package rather than add simulator-, model-, robot-, or
device-specific branches inside it.

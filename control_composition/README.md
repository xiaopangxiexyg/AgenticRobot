# Control Composition

`control_composition` is the reusable, backend-neutral framework for combining
multiple control sources. It turns immutable contributor facts from human,
replay, policy, RTC, or synthetic producers into one time-bound composition
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
- immutable, payload-contract-bound strategy context for history, state,
  future action, vision and arbitrary new modalities;
- replaceable arm-pose and hand/gripper-qpos composition strategies;
- composition results and proposal-promotion eligibility;
- logical composition sessions, activation epochs and source handoff reducers;
- source/output difference observations and non-authoritative feedback facts;
- a complete evaluation return record containing the exact spec, cut, active
  producer epochs, selected inputs, contexts, result, differences, diagnostics,
  and trace facts emitted for that evaluation;
- narrow ports for trace emission and canonical proposal admission.

This package does **not** own:

- canonical `RobotAction` component schemas, authority, arbitration, producer
  or execution epochs, cancellation, safety, IK, reference generation,
  dispatch, or a hardware/sim target;
- a teleoperation device, replay cursor, model worker, RTC queue, dataset
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
- ControlTrace records the full `composition_spec`, admissions, selected
  snapshot, result, source-difference spec/observation, disposition, and
  optional guidance facts. The evaluation record also returns the exact trace
  batch emitted by the current call.

## Dependency rule

Allowed dependency direction:

```text
control_station / sim adapters / teleop adapters / deployment adapters
                              |
                              v
                    control_composition
                              |
                              v
              robot_motion + dataset_contracts
```

`control_composition` must not import `control_station`, `sim`, `hardware`,
`teleop_runtime`, or `deployments`. `robot_motion`, `dataset_contracts`, and
`dataset_interface` must not import `control_composition`.

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
and the MuJoCo adapter all use this generic path. The old
`robot_motion.deploy.PolicyContextRecord/Store` runtime has been removed after
local and the known external OpenPI source tree were audited. Existing
v2.4 `policy_context_*` event schemas remain offline-readable legacy data;
new runtime facts are emitted as `composition_context_*`. They are not
silently rehashed or presented as the same identity.

The canonical store performs validation, epoch/sequence admission, cut
selection, retention, and trace once. Adapters may normalize sensor, replay,
human, policy, model-feature, and task-state inputs, but may not maintain
parallel hidden “latest context” state.

## Strategy input and latency contract

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
rules as RTC producers.

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

`CompositionInputStore` and `CompositionContextStore` use an explicit
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
or rolled-back identities. `retention_snapshot()` reports the retention-spec
hash, watermark, retained/pruned counts, and tombstone count.

This is a semantic time window, not a lossy ring overwrite. It does not by
itself bound a malicious producer's record rate or blob size inside the valid
window, and it does not replace durable ControlTrace/WAL backpressure. Those
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

## Recording levels

Recording has two resource-isolated tiers:

- **Core/actual facts are always retained durably** for an active formal
  profile: admitted inputs/contexts and their clocks/validity/contracts,
  selected lineage, spec/result/disposition, and every authority, safety,
  resolver, reference, dispatch, measurement, fault, fallback, or requested
  difference fact that affected or describes actual behavior. Large payloads
  may be retained by stable blob reference rather than copied inline.
- **Debug facts exist only when debug recording is enabled:** unselected
  alternative payloads beyond required lineage, full pairwise matrices when a
  summary suffices, intermediate weights/solver iterations, detailed timing,
  stack traces, model activations, previews, and duplicate heavy media.

Core and debug use separate queues, quotas, and sinks. `debug=off` means heavy
debug values are not constructed, copied, serialized, or queued. Debug writer
failure never blocks control or core persistence. Core persistence failure
follows the formal profile's fail-closed/seal-ineligible policy. Durable core
acknowledgement is also one precondition for pruning the in-memory window.

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
- `trace.py` owns their ControlTrace projections;
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

## Current implementation boundary

Implemented and locally contract-tested:

- deterministic target-time selection across different source rates;
- single-frame pose owner selection and weighted SE(3) pose fusion;
- explicit single-arm subset or current coupled left/right arm pose strategy
  inputs (formal current MuJoCo target remains coupled);
- active-mask-aware 1DoF/q6/q16 hand/gripper qpos interpolation;
- generic frozen context usable by both pose and qpos strategy protocols;
- strict pose-plus-hand structural join in non-authoritative shadow;
- same-cut pose/qpos source differences, human-role lookup, and no-I/O bounded
  guidance contracts;
- complete evaluation return records, full composition-spec trace, lifecycle,
  handoff, re-arm, proposal promotion, and package dependency firewalls.

Still gated for later work:

- additive component-keyed action carrier/adapters for arbitrary named arm and
  end-effector topology (arm joint DOF remains a downstream resolver concern);
- declared context windows, strategy latency enforcement, and asynchronous
  policy-driven mixing advisories;
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

These gaps do not change existing teleop, replay, synchronous policy, RTC, sim,
or real deployment defaults. New integrations should implement adapters and
ports around this package rather than add simulator-, model-, robot-, or
device-specific branches inside it.

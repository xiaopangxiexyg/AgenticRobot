# robot_motion.deploy.rtc

This package is the target-neutral RTC domain layer.  It is an internal
subpackage of `robot_motion.deploy`, not a new application or production
entrypoint.

Ownership:

- producer cadence and scheduling identity;
- generation/execution time-lattice semantics;
- canonical old-plan overlap selection and plan/epoch lineage;
- future extraction of stable protocol, producer, coordinator, and source
  binding contracts from the existing flat compatibility modules.

Non-ownership:

- model-space prefix encoding, model invocation, capability negotiation,
  framework fields, transport, artifact, or checkpoint selection;
- continuation training/artifact schema, dataset projection, learner, or loss;
- canonical safety, position/reference generation, target admission, dispatch,
  or backend acknowledgement;
- teleoperation, replay, HITL, Control Station, simulation scene, or hardware
  lifecycle.

The dependency direction remains:

```text
dataset_contracts
        ^
robot_motion.deploy.rtc
        ^
deployments.client
        ^
deployments.<model>.client
        ^
teleop_runtime / sim / control_station application bindings
```

RTC is therefore a cross-layer feature, not a cross-layer top-level package.
Do not move model clients or application publication into this package.

## Compatibility migration

Existing modules remain supported:

- `robot_motion.deploy.rtc_protocol`
- `robot_motion.deploy.rtc_runtime`
- `robot_motion.deploy.async_producer`
- `robot_motion.deploy.chunk_plan`
- `robot_motion.deploy.policy_worker`

Only runtime scheduling and canonical plan-continuation semantics land here.
Shared continuation data/artifact schemas belong to `dataset_contracts`,
offline projection belongs to `dataset_interface`, and model-space
codec orchestration/protocol belongs to `deployments.client`; concrete
artifact-bound numerical codecs belong to model providers.  Existing flat modules
contain historical composed contracts and may later become compatibility
surfaces after focused migration and characterization tests; they must not be
moved in a single bulk refactor.

`PolicyAsyncExecutionProfile` is a historical composed profile containing both
producer cadence and application execution settings.  Its fields and semantic
hash remain unchanged.  New producer code consumes its deterministic
`producer_profile` projection, whose identity excludes:

- position sample/reference/IK/command rates;
- position lookahead;
- action-send authorization.

The embedding application continues to own those excluded settings.

## Initial model-RTC boundary

`RtcContinuationTimeline` defines one time relation:

```text
generation_start_at
    + forecast_delay_steps * action_period_sec
    = first_target_effective_at

model generates          full_chunk[0:H]
client returns/admission full_chunk[delay:H]
```

This prevents runtime and model adapters from applying delay compensation
twice.

The canonical training/artifact semantics live in the pure
`dataset_contracts.model_continuation` contracts. Frozen v1 and explicit v2
share one non-persisted normalized runtime view; source schema and source hash
remain authoritative.
The pre-C1 `RtcContinuationEncodingReceipt` is now a frozen deprecated import
surface only. C1 encoding and actual-consumption receipts live in
`deployments.client`; no new caller may use the old receipt or its duplicated
sampler/source constants. It remains temporarily exported solely to preserve
characterized imports before a separate compatibility-removal gate.

MRTC-R1a adds an optional `conditioning_binding_hash` to
`PolicyInferenceRequest`. A model-RTC request may use that causal hash instead
of carrying a model-space ndarray through `robot_motion`; the two forms are
mutually exclusive. The old `PolicyInferencePrefix` and RTC v2 envelope remain
frozen compatibility surfaces, and v2 does not interpret the new binding.
Requests that use it receive a new request-hash domain; raw-RTC and legacy
prefix request hashes remain unchanged. The immutable canonical conditioning
bundle belongs here because it binds the owned timeline, old-plan slice, and
observation context. Model-space encoding and the v3 envelope/receipt belong
to `deployments.client`. The bundle travels directly in
`PolicyInferenceWorkItem`; the causal request carries only its binding hash.
Raw and legacy-prefix work items keep their existing executor call and hash
domain.

`select_rtc_plan_continuation()` now performs the pure timed-plan overlap
selection.  It never mutates the source plan, interpolates, resamples, or calls
a model.  The first contract accepts only:

- typed canonical keyframes on the exact generation lattice and cadence;
- a contiguous selected interval whose exclusive horizon does not exceed
  `forecast_delay_steps`;
- one source plan/epoch/policy-node identity with no request already in flight.

Already-popped leading generation indices may be absent, but a later encoder
must represent those positions as unselected in the versioned prefix mask.
Holes inside the selected interval fail closed.  Empty, unavailable, or shorter
than configured overlap produces an explicit bootstrap outcome; identity,
clock, lattice, and representation failures are not bootstrap.

The current C1 implementation accepts only `flow_inpaint`; other continuation
modes require a new bundle/runtime protocol rather than reinterpretation of
this v1 package. Model tensors, target receipts, position preview, safety, and
dispatch remain outside this package.

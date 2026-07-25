# robot_motion.deploy.rtc

This package is the target-neutral RTC domain layer.  It is an internal
subpackage of `robot_motion.deploy`, not a new application or production
entrypoint.

Ownership:

- producer cadence and scheduling identity;
- generation/execution time-lattice semantics;
- model-neutral continuation-prefix lineage;
- future extraction of stable protocol, producer, coordinator, and source
  binding contracts from the existing flat compatibility modules.

Non-ownership:

- model invocation, framework fields, transport, or checkpoint selection;
- dataset reading, training-window construction, learner, or loss;
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

New model-RTC semantics land here first.  Existing modules may later become
compatibility re-exports after focused migration and characterization tests;
they must not be moved in a single bulk refactor.

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

`RtcContinuationEncodingReceipt` binds producer-local canonical overlap to its
model-space encoding.  Its `prefix_source_role` is deliberately limited to
`producer_local_overlap_v1`.  A sampler may use a clamped or weighted numerical constraint,
but that strength does not prove downstream target acceptance or motion
authority.

The next implementation stage may add pure timed-plan overlap selection and an
injected model-space encoder.  It must not add target receipts, position
preview, safety, or dispatch to this package.

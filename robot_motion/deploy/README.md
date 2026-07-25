# robot_motion.deploy

`robot_motion.deploy` holds sim/real-neutral runtime glue for motion-control
deployment. Model-server client code lives under `deployments.client`.
Real-only hardware adapters stay under `teleop_runtime`; sim-only adapters stay
under `sim`.

Current boundary:

```text
ObservationSource -> Policy -> CanonicalActionPipeline -> Dispatcher
```

- `source.py`: common observation-source protocol.
- `dispatcher.py`: common command-dispatch protocol and dry-run recorder.
- `runtime.py`: small common single-step runtime skeleton.
- `policy_worker.py`: bounded model-neutral RTC inference carrier; no plan,
  safety, target or dispatch ownership. Stop dispatches an optional executor
  cancellation hook in isolation and reports cancellation separately from
  worker-thread drain.
- `rtc_runtime.py`: target-neutral RTC request/result/timed-plan coordinator;
  independent of teleop, replay, Control Station, sim and hardware.
- `rtc/`: canonical home for new target-neutral RTC domain contracts.  It
  starts with the producer-only cadence projection and model-RTC continuation
  timeline/encoding lineage; existing flat RTC modules remain compatibility
  surfaces during incremental migration.  See
  [`rtc/README.md`](rtc/README.md).

RTC request identity is deliberately split. `observation_cut_id` is a small,
versioned causal identity covering session/domain/producer epoch/request
sequence and observation metadata; it never hashes camera or tactile bytes on
the control thread. The immutable actor view is queued with that request, and
the model adapter computes the exact `model_input_hash` inside the worker.
Submission/work-item hashes bind the causal cut and contract; they do not
replace model-input content attestation.

## Four representation boundaries

`PolicyAdapterConfig` is the model-neutral identity/configuration for a policy
stream. It binds the runtime action vector to the model action dimension and
codec, the normalization contract, chunk horizon/executable prefix, and the
absolute/relative pose frame contract. Its `to_runtime_config()` method only
projects these values into the existing `PolicyDeployRuntime`; it does not
create a queue or dispatch authority. Relative pose conversion remains
anchored to the measured TCP snapshot in `pose_adapter.py`.

Raw-RTC freezes that snapshot in a versioned
`PolicyActionRepresentationContext` attached to the inference request.
Relative chunks are expanded once with the original request anchor before
timed-plan admission; absolute chunks use explicit no-anchor semantics. The
context is client-local causal evidence, not a model-server input.

Model-family packages should accept this config and return decoded
`NormalizedAction` chunks. They must not infer a transform from a field name
such as `normalized`. A non-identity numeric transform is executed only by the
owner named in the explicit normalization contract; the generic runtime
validates the contract/hash and never applies a second guessed inverse. Model
adapters also must not reimplement chunk selection, pose-frame conversion, IK,
safety, or target submission. The next layer (`DeployRobotActionAdapter` plus
the target runtime) is shared by real and sim compositions.

### Normalization contract

`robot_motion.deploy.normalization.NormalizationContract` has independent
`state`, `action`, and `image` domains. Each domain declares `owner` (`none`,
`client`, or `server`), a method (`identity`, `zscore`, `quantile_range`,
`affine`, or `range`), an explicit operation (`normalize` or `unnormalize`),
and input/output spaces. Statistics-backed methods must bind both a statistics
reference and its exact artifact hash. The operation is required because the
action output may be the inverse of the state input transform. `none/identity`
is the only implicit default. Legacy payloads are retained as opaque
compatibility metadata and cannot be used to infer a transform. Server
metadata can be required to echo the contract hash through
`DeployPolicyContract.normalization_hash`.

For `quantile_range`, `normalize` is
`(x-q01)/(q99-q01+1e-6)*2-1`; `unnormalize` is
`(x+1)/2*(q99-q01+1e-6)+q01`. The referenced statistics artifact owns q01/q99;
they are never copied into an unrelated client config.

Migration notes:

- Large action/deploy types still live in `teleop_runtime.deploy` during this
  transition. Do not move them in one large edit.
- RTC is a cross-layer feature, not a new top-level package.  Shared
  request/plan/producer/continuation semantics belong under
  `robot_motion.deploy.rtc`; generic model clients remain in
  `deployments.client`, and application execution/publication remains in the
  embedding real/sim/control application.
- `PolicyAsyncExecutionProfile` retains its historical position/send/lookahead
  fields and hash for compatibility.  New RTC producer logic consumes its
  target-neutral `producer_profile` projection instead of treating those
  application settings as producer-owned facts.
- New motion-control common code should be added here first.
- Generic model-server client code belongs in `deployments.client`.
- Model-family-specific client code belongs in `deployments/<model>/client`.
- Real launchers may import from this package; this package should not import
  real hardware backends.
- Sim launchers may import from this package; this package should not import
  MuJoCo-specific backends.
- `teleop_runtime.deploy.policy_worker` and `rtc_frontend` are compatibility
  imports only; new RTC core code must not move back into an application
  frontend.
- Cross-domain launchers such as real-observation-to-sim-dispatch should live
  under `deployments/dexteleop` or another composition layer, not in
  `robot_motion`.

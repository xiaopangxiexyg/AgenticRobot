# deployments.client

This package is the model-agnostic deployment client layer.

It owns:

- remote server metadata validation
- versioned policy-artifact JSON loading and strict input/output tensor binding
- remote policy response normalization into the unified dense action chunk
- generic source -> policy -> action-preparer -> dispatcher runtime glue
- model-family-neutral RTC capability negotiation, response validation and model-space decoder seam

It does not own:

- OpenPI websocket/msgpack details
- ACT/DiffusionPolicy/etc. model-family field names
- real hardware backends
- MuJoCo or other sim backends

Model-specific packages, for example `deployments.openpi.client`, provide the
transport and observation adapter. Real and sim launchers provide their own
observation sources, canonical action preparers, and dispatchers.

## RTC v2 generic seam

`rtc.py` consumes the pure contracts in `robot_motion.deploy.rtc_protocol`. It
accepts any local or remote `RtcModelBackend` that advertises an exact
capability, validates the response/conditioning receipt, then hands model-space
rows to an injected decoder. The resulting `PolicyChunkResult` is still only a
fact for later control-thread plan admission; this client does not own a worker,
timed plan, queue, merge/pop, safety, reference, or dispatch.

`rtc_executor.py` is the narrow adapter from a frozen `PolicyObservationView`
and one admitted request to that client. It is a blocking callable intended to
run inside the caller-owned bounded worker; it creates no thread and still has
no plan-admission or target authority. Model-specific packages may provide a
prepared-input function or executor factory without changing the RTC core.
That worker-side prepared-input function owns the exact `model_input_hash`;
the control-thread `observation_cut_id` and work-item hash are causal bindings
and must not scan bulk sensor payloads.
Its optional `request_cancel()` only forwards a quick backend session signal;
acceptance is not worker, transport, or execution drain evidence.

`rtc_frontend.py` removes the remaining model-family composition boilerplate.
A framework adapter structurally provides only:

1. an instance-derived `rtc_capability`;
2. the exact `request_spec` it intends to run;
3. model input preparation plus model-space decoding inside a blocking worker
   executor;
4. `create_worker_executor()` for each fresh worker session.

`build_rtc_model_shadow_frontend()` validates capability/spec negotiation,
projects artifact/codec/frame/layout/normalization/horizon identity into the
shared coordinator, checks action cadence, and returns an unstarted frontend.
The caller still owns `start/step/close`; the helper cannot initialize an
observation source, target, safety, position reference, dispatcher, or motion
authority. OpenPI satisfies this structural protocol directly, without a
registry entry or a framework-name branch. A local Torch policy, HTTP service,
IPC process or another model server can expose the same four pieces.

New integrations should use the canonical aliases
`build_rtc_model_producer_runtime()` and
`rtc_producer_config_from_model_adapter()`.  The existing
Shadow/Frontend names remain identical callable compatibility surfaces; this
is a naming migration, not a second runtime or producer.

For a sessionless local/HTTP/IPC runtime, `SimpleRtcModelRuntimeAdapter`
packages the four pieces directly, so no adapter subclass is necessary.
Frameworks that need reconnect, metadata refresh or worker-session setup
implement the same structural protocol directly; OpenPI is an example.
For a relative-pose request spec the adapter additionally supplies
`representation_context_builder(observation_view, cut_id)`. It freezes the
original request's measured TCP anchors and declared frames; the shared
frontend rejects a missing or mismatched context before calling the model.
This optional client-side hook does not add fields to the server payload.

Shortest sessionless local/HTTP/IPC integration:

```python
adapter = SimpleRtcModelRuntimeAdapter(
    backend=my_backend,
    decoder=decode_model_actions,
    request_spec=request_spec_from_loaded_artifact,
    prepare_input=prepare_model_input,
    # Required only when request_spec.action_pose_mode == "relative":
    representation_context_builder=build_request_representation_context,
)
frontend = build_rtc_model_shadow_frontend(
    adapter,
    deploy_runtime=deploy_runtime,
    preparer=canonical_action_pipeline,
    # runtime/control/producer/node identities omitted here
)
```

The capability and request spec must come from the actually loaded model/server
instance; neither helper guesses them from a framework, config or checkpoint
name. A framework with custom worker-session lifecycle uses the same protocol
directly:

```python
class MyRtcAdapter:
    @property
    def rtc_capability(self): ...

    @property
    def request_spec(self): ...

    def create_worker_executor(self, *, observe_bundle=None):
        return RtcModelClientInferenceExecutor(
            client=ModelAgnosticRtcClient(
                backend=self.backend,
                decoder=self.decode_actions,
                request_spec=self.request_spec,
            ),
            prepare_input=self.prepare_input,
            observe_bundle=observe_bundle,
        )
```

The application then passes this adapter to
`build_rtc_model_shadow_frontend(...)`; it does not copy protocol identity
fields or implement worker/plan scheduling.

`rtc_final_chunk.py` removes the protocol boilerplate for existing servers
that already return a complete chunk. For a sessionless local/HTTP/IPC client,
the public one-step composition is:

```python
adapter = create_final_chunk_rtc_runtime_adapter(
    binding=instance_binding,
    prepare_input=prepare_actor_observation,
    call_model=existing_client.infer,
    hash_model_input=hash_exact_prepared_input,
    decoder=decode_to_canonical_absolute_actions,
    adapt_output=MappingFinalChunkOutputAdapter(actions_key="actions"),
)
```

`FinalChunkRtcBackend` derives capability and request spec from the single
`FinalChunkRtcBinding`; an integration must not copy those identity or horizon
fields. `MappingFinalChunkOutputAdapter` uses exact configured keys, ignores
unselected diagnostic metadata, and supports a separately typed discrete
matrix. The backend checks the prepared-input hash, validates all rows,
measures local monotonic timing and constructs the RTC response envelope.
It always advertises `continuation=none`; it does not make a legacy final-chunk
model a conditioned RTC model. A framework with a custom session/reconnect
lifecycle implements `RtcModelRuntimeAdapter` directly.

This extraction was driven by the second-framework `vla-scratch` integration:
its framework-specific layer now only builds the processed sample, calls the
unchanged `infer(dict)` interface and extracts `action_chunk.actions`.

`rtc_transport.py` defines optional model-neutral connect/metadata/inference/
close budgets and typed phase receipts. These relative monotonic deadlines are
client transport operations, not model capabilities, request payload fields,
or substitutes for request TTL and plan admission. All budgets default to
`None`, so a concrete adapter retains its legacy behavior unless an RTC caller
opts in. A future HTTP, IPC, local-runtime, or other model adapter can map the
same contract without importing OpenPI.

RTC v2 binds scheduler and continuation independently and validates artifact,
action-space/dimension, pose/component scope, codec/frame/layout/normalization,
horizon, model-input, prefix-mask, and continuation implementation hashes. It
does not infer support from a model name. Continuous float actions and optional
discrete int64 token actions are carried as independent matrices and are both
bound into the response hash. Since the current typed inference prefix only
represents continuous model-space rows, any continuation capability with a
non-zero discrete output dimension fails closed until a typed discrete
prefix/mask/receipt exists.

The deterministic fake backends live only in
`tests/test_rtc_model_agnostic_protocol.py`. The first concrete consumer is the
offline OpenPI final-chunk adapter in `deployments.openpi.client.rtc`; it uses
the same generic seam and advertises `continuation=none` only. This does not
make OpenPI part of the generic protocol and does not wire a production caller.

Current production note: `deployments.openpi.client.real` is a thin supported
launcher over the supported `teleop_runtime.deploy.run_policy` runtime; the generic
`DeploymentClientRuntime` in this package is a composition contract, not the
current RM65 hardware loop. Position-control work must therefore keep common
planning in `robot_motion` and connect it at that existing real-runtime
boundary, rather than creating a second loop here.

## v2.2 policy artifact

`dexteleop_policy_artifact_v1` embeds the canonical `PolicyIOLayout` and its
hash. Input/output float32 dimensions and independent int64 token dimensions
are derived from that layout; a sidecar cannot redefine them independently.
`deploy_contract_from_policy_artifact()` turns it into a strict
`dexteleop_deploy_v2` server handshake. Dataset, checkpoint/server metadata and
runtime must agree on `policy_io_layout_hash`.

For a state-dropout-trained model, the artifact additionally declares
`state_input_mode="masked_normalized_concat_v1"`. Its float input dimension is
twice the layout base-state dimension: normalized state followed by a binary
availability mask. The client/deployment adapter appends an all-one mask at
runtime and requires the server metadata to declare the same mode and doubled
input dimension. Existing artifacts default to `raw_state_v1` and retain the
original state dimension unchanged.

Unmarked models retain `dexteleop_deploy_v1` behavior (26D action, 38D state,
legacy q6 hands). A v2.2 artifact is opt-in and does not reinterpret an old
checkpoint.

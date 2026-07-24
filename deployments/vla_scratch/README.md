# vla-scratch additive integration

This package integrates the external baseline at
`/home/zc/HandMotion/vla-scratch` without making either repository import the
other. The baseline was audited at revision
`fa3cb37a6f3ba620bd2c167a97035b5431f31d19` and was not modified.

## Existing baseline seam

The baseline already exposes the narrow interface needed for final-chunk RTC.
There is one model instance in this path; the other names below are process
roles, not additional models:

```text
ZmqPolicyClient.infer(dict)
  -> ServePolicy.infer(dict)       # serving wrapper/process role
  -> one BasePolicy model.sample_actions(...)
  -> {"action_chunk.actions": float chunk}
```

`BasePolicy` is the model contract and `ServePolicy` is only the unchanged
server-side wrapper. `ZmqPolicyClient` and the DexTeleop RTC adapter do not
load or run a second policy; they serialize requests and validate/decode the
single returned final chunk.

Its processed observation keys are:

```text
observation.images       uint8 [camera, 3, height, width]
observation.image_masks  bool  [camera, 1]
observation.state        float32 [history, state_dim]
task                     non-empty string
```

`dataset_contracts.vla_scratch` freezes those keys, camera ordering,
state/action dimensions and representation IDs in
`VlaScratchProcessedSampleSpec`. The same builder can be used by a future
DexTeleop dataset adapter and by deployment, preventing training/serving input
drift.

`deployments.vla_scratch.client.rtc` wraps only the existing
`infer(dict) -> dict` shape. It imports neither `vla_scratch` nor ZMQ. An
application may inject the baseline's unchanged `ZmqPolicyClient`, a local
callable, or another compatible transport:

```python
adapter = create_vla_scratch_runtime_adapter(
    transport=existing_zmq_policy_client,
    config=explicit_binding,
    decoder=decode_model_actions_to_canonical_absolute_actions,
    task="task instruction bound by the run config",
)
frontend = build_rtc_model_shadow_frontend(adapter, ...)
```

The returned object is the shared `SimpleRtcModelRuntimeAdapter`; no framework
subclass, registry entry, worker, scheduler or plan implementation is added.
The default integration uses
`deployments.client.create_final_chunk_rtc_runtime_adapter`, which derives
capability/request identity from one binding and creates the shared
`FinalChunkRtcBackend`. The vla-scratch layer only validates its processed
input, calls `infer()` and selects the configured action key. The named
`VlaScratchFinalChunkRtcBackend` remains only as an import-compatible thin
wrapper for existing callers.

## Required binding before a real model run

The current baseline server does not publish checkpoint metadata or a
capability handshake. Its serving script also initializes `state_dim`,
`action_dim` and warmup input from the selected dataset. Therefore a real
DexTeleop VLA run must first provide an external, artifact-derived binding for:

- exact checkpoint/artifact digest and baseline source revision;
- ordered logical camera sources and RGB preprocessing contract;
- state history, dimension, representation and normalization-stat digest;
- action horizon, dimension, representation, frame and decoder digest;
- task/instruction source and the dataset view used for warmup.

These values cannot be inferred from `pi-qwen`, a checkpoint filename or the
current LIBERO defaults. In particular, the existing LIBERO 7D action output
must not be interpreted as a DexTeleop absolute pose without an explicit
representation/decoder contract.

## Canonical DexTeleop sample preflight

`dataset_interface.vla_scratch` provides the additive dataset-side seam. It
derives `VlaScratchProcessedSampleSpec` from a v2.3
`PolicyArtifactContract`, preserving artifact camera order and the exact
`PolicyIOLayout` state/action dimensions. A full local fixture covers:

```text
front + left_wrist + right_wrist RGB
  + canonical-desk left/right pose7
  + continuous left/right end-effector state/target
  -> observation.images          uint8   [3,3,H,W]
  -> observation.image_masks     bool    [3,1]
  -> observation.state           float32 [1,26]
  -> action_chunk.actions        float32 [4,26]
```

The `26` and `4` above are fixture values derived from its artifact; the
adapter itself does not fix camera count, state/action dimension, history or
horizon. The full profile requires the logical views
`front/left_wrist/right_wrist`, but their tensor order remains the order in the
artifact rather than mapping insertion order.

The current baseline has no processed discrete channel and its loss has no
action-padding mask. The adapter therefore rejects discrete DexTeleop layouts
and any sample whose `action_is_pad` contains `true`; it never flattens tokens
into floats or trains repeated tail padding as real actions. Non-raw/masked
state also requires a separately contracted shared transform.

The unchanged baseline's own `ToDataSample` accepts the resulting
3-camera/1x26-state/4x26-action key and dtype shape. This is wire/type
compatibility evidence, not checkpoint compatibility. The two repositories'
current virtual environments do not contain one another's complete dependency
set; that is immaterial to the server-client inference boundary. A deployment
launcher can keep the DexTeleop client and vla-scratch server in their native
processes and exchange only the serialized inference payload. If DexTeleop
itself owns the native ZMQ client, install the optional `vla_scratch` extra;
that extra contains only `pyzmq`, not the vla-scratch model/server stack. A
combined environment is optional only for a future in-process training dataset
plugin, not a RTC gate; neither baseline environment is modified here.

For a first real no-action probe, an out-of-tree launcher may construct the
existing `ZmqPolicyClient` and pass it into this package. A later optional
sidecar/proxy may attest metadata without changing `ServePolicy` or
`BasePolicy`; direct baseline edits are not required for final-chunk RTC.

## Current evidence boundary

The original ZMQ client/server codec has completed a localhost no-model
loopback through the generic RTC client: two ordered cameras, one state row and
a 4x7 finite chunk crossed the real wire. No checkpoint, GPU, observation
device, safety, target or action dispatch was used.

The canonical sample preflight additionally uses only virtual arrays and a
synthetic v2.3 artifact/pose-frame contract. It consumes no collected episode,
loads no model and does not authorize a real observation or action path.

The loopback also exposed a shutdown race in the baseline
`ZmqPolicyServer.close()` ordering (socket close before its polling thread has
exited). The probe used an orderly test-only stop and did not modify the
baseline. This lifecycle issue must be isolated or fixed upstream before a
long-running live-server gate; it does not change the inference wire contract.

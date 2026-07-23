# OpenPI Client

This directory contains only OpenPI-specific client pieces:

- OpenPI websocket/msgpack transport
- OpenPI observation field adapter
- OpenPI policy factory for the generic `deployments.client` remote policy
- OpenPI-to-sim adapter bound to the shared `robot_motion.deploy` contract
- artifact-bound OpenPI raw-RTC final-chunk adapter (`continuation=none` only)
- OpenPI startup command wrappers

The generic remote-policy contract, action chunk, and runtime glue live in
`deployments.client`.

Example real dry-run:

```bash
python -m deployments.openpi.client.real \
  --host 127.0.0.1 \
  --port 8000 \
  --model dual_rm65_dmgripper_latest \
  --active-sides both \
  --hand-backend dmtac-gripper \
  --dmtac-config latest
```

The command above expands to the generic real deploy loop with:

```bash
--policy deployments.openpi.client.policy:create_policy
```

`teleop_runtime` still owns real hardware IO and dispatch, but it does not
import or know about OpenPI.

## MuJoCo policy adapter

`OpenPISimAdapter` is the OpenPI-specific server boundary for simulation. It
constructs one artifact-bound `OpenPIClientPolicy`, calls the existing OpenPI
server, and passes the response through the shared normalization-contract,
chunk-prefix, and pose-frame decoder. It returns `OpenPISimPrediction` (a tuple of
`NormalizedAction` values) and owns no MuJoCo stepping, IK, safety, target
authority, or dispatch. `sim.mujoco.deploy.MujocoDeploymentObservationAdapter`
supplies the corresponding read-only `DeploymentObservation`.

This is a library adapter rather than a new production entrypoint. A caller
must explicitly compose its returned actions with the existing generic
`DeployRobotActionAdapter` and `MujocoDeployRuntime` profile. For a synchronous
sim-only composition, `sim.mujoco.deploy.MujocoPolicyRollout` owns that seam;
it does not change the PPO training path or the `sim.mujoco_temp_pipeline.play`
Torch-policy entrypoint.

### Numeric normalization ownership

An artifact may place an explicit `metadata.normalization_contract` beside its
layout. The contract has separate state/action/image domains and names the
executor (`none`, `client`, or `server`) and the operation direction. OpenPI
currently accepts only `none` or server-owned non-identity transforms at this
boundary: state/action values are passed through locally, and the contract hash
is carried in debug metadata and optional server preflight. A client-owned
transform fails closed until a model-specific adapter implements it. Legacy
`normalized` payloads remain opaque compatibility data; they are never used to
guess an inverse transform.

The current remote DexTeleop OpenPI v23 config is an explicit example of the
server-owned case. Its checkpoint asset contains `norm_stats.json` with
`state` and `actions` q01/q99 statistics, and the server config sets
`use_quantile_norm=True`. State is mapped from raw compact 16D to q01/q99
`[-1, 1]` using `(x-q01)/(q99-q01+1e-6)*2-1`; model actions are mapped back to
raw compact 16D using the inverse formula. Images remain raw RGB `uint8` HWC
at this boundary (the server later resizes them to 224x224).
Use `make_openpi_quantile_normalization_contract()` when the artifact has an
exact stats reference and SHA-256. Do not infer this contract from a field such
as `state_input_mode`. The remote registry now publishes additive normalization
profiles for the verified latest, v22, and v23 checkpoints; the server echoes
the contract/hash only for entries that explicitly reference a profile. Legacy
registry entries and clients that do not request a normalization hash remain
unchanged, while new clients can opt into strict preflight.

## RTC-4b-A raw-RTC offline adapter

该 adapter 保持 OpenPI 的既有 final-chunk model/server 接口；它不消费未来动作。需要模型消费
prefix/mask/conditioning receipt 的 model-RTC 属于独立后续实现。

`rtc.py` adapts the existing artifact-validated OpenPI final-chunk response to
the model-neutral RTC v2 envelope. It binds the full artifact and server
metadata, exact prepared input, layout/codec/frame/normalization hashes,
declared action horizon, continuous float outputs, and independent int64 token
outputs. It intentionally advertises only `async_timed_plan +
continuation=none`.

The adapter owns no worker, timed plan, action admission, safety, reference, or
dispatch state. The existing OpenPI launcher wires it only for the explicit
default-off raw profile; that profile forces `send_actions=false` and does not
change `sync_queue_v1` or the RTC-3c builtin-hold frontend. Offline fake
transport plus shared-frontend coverage lives in
`tests/test_openpi_rtc_final_chunk_adapter.py`.
When `typed_canonical_v1` is selected, the client frontend records a versioned
representation context. Absolute actions use an explicit no-anchor context;
relative actions require the original request cut's fixed measured-TCP
anchors and are decoded to canonical absolute actions before plan admission.
The launcher preserves the legacy result path for absolute artifacts and
selects typed results only for an artifact-declared relative raw-RTC request.
Neither branch changes the OpenPI server payload.
Model-RTC prefix-conditioned continuation is a separate RTC-4b-B gate; models with
discrete outputs additionally require a typed discrete prefix/mask/receipt
before that capability can be enabled.

RTC-4b-T1 adds only local transport lifecycle mechanics: an RTC worker executor
may latch cancellation, detach and asynchronously close the current websocket,
then explicitly re-arm a new session for a fresh connection. Typed transport
begin/cancel and close-completion receipts plus connection generations are
separate audit facts; re-arm fails closed while close is pending or failed,
and only worker thread exit proves inference-worker drain. Reconnected server metadata is
re-hashed before a chunk can return. No live server caller, action send, model
sampling change, or legacy websocket payload is introduced by this gate.

RTC-4b-T2a maps the generic `RtcTransportDeadlineSpec` to websocket
`open_timeout`, metadata `recv(timeout=...)`, inference
`recv(timeout=...)`, and connection `close_timeout`. All fields are optional;
the ordinary `OpenPIClientPolicy` still constructs the transport without this
spec and therefore preserves the prior connect/recv call shape and retry
behavior. Each attempted phase records a bounded typed receipt. An explicitly
timed-out metadata or inference exchange retires that socket so a late frame
cannot be paired with the next request; no fallback action chunk is produced.

Fake-socket tests and an actual localhost websocket with a synthetic final
chunk cover timeout, cancel competition, close/re-arm, fresh reconnect, and
metadata continuity without a model, GPU, real observation, or action send.

RTC-4b-T2b additionally exercised this unchanged transport against one existing
remote OpenPI model server using two synthetic observations. Both connections
validated the loaded-instance artifact/metadata and returned finite `50 x 16`
  final chunks; cancel/close/re-arm produced a fresh connection generation with
  an identical metadata hash. This is scoped compatibility evidence, not a
  checkpoint/config allowlist, real-observation test, control-loop timing claim,
  or motion authorization. The probe lived only in remote `/tmp`; the existing
  server/model source was not changed. The explicit OpenPI raw-RTC caller is now
  available as a default-off `send_actions=false` path; real observation scope
  and any motion authority remain separate gates.

Environments with HTTP proxy variables must configure `NO_PROXY` for the
selected direct OpenPI host; production proxy behavior is not silently
overridden by the RTC adapter.

## Explicit live observation shadow

For a real observation/transport probe, add both the raw profile and the
explicit read-only guard:

```bash
python -m deployments.openpi.client.real \
  --deployment-profile rm65_bimanual_dmtac \
  --rtc-profile raw_rtc_final_chunk_15_30_60_120_v1 \
  --rtc-real-shadow \
  --rtc-normalization-hash <artifact-normalization-hash> \
  --checkpoint-dir <checkpoint-dir>
```

This consumes live arm/hand/camera observations and the existing OpenPI
transport, but never sends actuator commands. The guard also removes the lab
profile's qpos-init authority and configures RM65/DMTAC/hand backends for
observation-only startup. The explicit shadow profile is headless: it
auto-starts policy observation, disables manual ready confirmation, and keeps
DMTAC tactile disabled unless a separate gate changes that scope. The ordinary
production profile retains its ready confirmation, qpos-init and configured
tactile behavior. It is not dataset replay: replay/synthetic tests
remain the deterministic complement for codec and plan semantics. The result
is only a real-shadow readiness record, not permission for motion deployment.

With `--debug`, each accepted model exchange also records a
`rtc_transport_exchange` event containing the request/input/response hashes,
chunk horizon and inference/result latency. This event is observational only;
it is not plan-admission, safety, dispatch, or worker-drain evidence.

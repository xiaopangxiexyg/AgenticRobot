# OpenPI Client

This directory contains only OpenPI-specific client pieces:

- OpenPI websocket/msgpack transport
- OpenPI observation field adapter
- OpenPI policy factory for the generic `deployments.client` remote policy
- OpenPI-to-sim adapter bound to the shared `robot_motion.deploy` contract
- artifact-bound OpenPI raw-RTC final-chunk adapter (`continuation=none` only)
- default-off OpenPI model-RTC v3 provider binding
- OpenPI protocol and server/client adapters used by application launchers

The generic remote-policy contract, action chunk, and runtime glue live in
`deployments.client`.

## Optional arm-hold output

For a checkpoint whose `policy_artifact.json` declares
`arm_hold_output_contract`, the OpenPI server handshake must advertise:

```json
{
  "action_semantics": {
    "arm_hold_output_key": "arm_hold_probability",
    "arm_hold_output_contract_hash": "<exact artifact contract hash>"
  }
}
```

Every non-RTC response must then include the declared key as a float
`[action_horizon, 2]` probability matrix in `["left", "right"]` order, plus
the same top-level `arm_hold_output_contract_hash`. The OpenPI adapter does not
apply a second sigmoid: provider logits must be converted to probabilities
before this boundary. The generic client rejects missing, non-finite,
out-of-range, misaligned, or hash-mismatched output before queue admission.
The existing continuous action tensor and its dimension do not change.
This first landing is limited to the synchronous `PolicyDeployRuntime` path;
RTC and `OpenPISimAdapter` reject an arm-hold artifact until their own
emission-time semantic resolution seam is implemented.

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

`teleop_runtime.deploy.openpi_application` owns the real application binding and
imports these OpenPI adapters. `deployments.openpi.client.real` remains only as
the supported historical command path; it must not regain a hardware loop or
real-runtime ownership.

Model id and checkpoint step are instance provenance and are not accepted as
client admission flags. The downstream gate checks the declared wire protocol
and tensor/camera contract. Audit an exact instance from the policy artifact,
server metadata, and runtime receipts without changing online admission.

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
Torch-policy entrypoint. Arms-only remains the default MuJoCo target; OpenPI
hand keyframes require the explicit DMTAC compound target profile and shared
safety callback, otherwise the runtime rejects the chunk rather than dropping
the gripper commands.

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

The resolution boundary is intentional and must not be mistaken for visual
sim-real equivalence. The current MuJoCo OpenPI smoke path may render all
three camera slots at `48x48` so that the transport, policy adapter, action
codec, and sim execution path can be exercised cheaply. The remote OpenPI
policy then applies its model transform `resize_with_pad(224, 224)` before the
vision encoder. This is a valid protocol/control-chain test, but it is not the
same image distribution as the real RM65 cameras (`640x480`, 4:3): a square
`48x48` source is upsampled without padding, while a `640x480` source becomes
`224x168` and receives top/bottom padding. Do not report the smoke result as
visual sim-real alignment; use a camera-contract-matched source profile for
that claim.
Use `make_openpi_quantile_normalization_contract()` when the artifact has an
exact stats reference and SHA-256. Do not infer this contract from a field such
as `state_input_mode`. The remote registry now publishes additive normalization
profiles for the verified latest, v22, and v23 checkpoints; the server echoes
the contract/hash only for entries that explicitly reference a profile. Legacy
registry entries and clients that do not request a normalization hash remain
unchanged, while new clients can opt into strict preflight.

### Backend-neutral emission source

`OpenPIEmissionSourceConfig` and `build_openpi_policy_emission_source()` bind a
prepared OpenPI artifact to the shared `ThreadedPolicyEmissionSource`.  The
caller supplies only a `DeploymentObservation` capture callback; the source has
no MuJoCo, real-robot, GUI, composition, safety, IK, or dispatch dependency.
Model inference runs on one private worker, while prediction admission, chunk
ordering and one-keyframe `PolicyActionEmission` creation remain on the caller
thread.  A failed prefetch cannot revoke the tail of an already admitted chunk,
and an activation generation change discards late completion before emission.

`model_observation_sides` and `output_active_sides` are separate.  This allows a
bimanual model observation to produce an explicitly scoped contributor (for
example right arm and right gripper) without pretending the target supports a
full bimanual command.  The selected output sides must be a subset of the model
observation sides.  A prepared artifact with an explicit executable
`normalization_contract` is mandatory; a raw checkpoint artifact with missing
normalization ownership is rejected.

## Server metadata compatibility

普通 OpenPI 和 model-RTC client 从 server 发布的 `policy_artifact` 读取 state key、
state input mode、连续/离散维度及实际 policy-I/O layout；server 不必在旧顶层字段重复
发布同一信息。已显式发布的顶层字段仍优先检查，不能由 artifact 掩盖实际冲突。
默认兼容性判断比较 layout 的执行结构，而不是 layout 名称、采集注释或哈希。
Client contract 不自动绑定 layout hash、model id 或 checkpoint step；通用接口中显式
请求的 hash 审计保持独立，不是普通部署的必需参数。

## RTC-4b-A raw-RTC offline adapter

该 adapter 保持 OpenPI 的既有 final-chunk model/server 接口；它不消费未来动作。模型消费
prefix/mask/conditioning receipt 时必须使用下面独立的model-RTC v3路径。

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
An artifact-declared arm-hold sidecar is supported only on the absolute
`legacy_normalized_v1` branch. The adapter binds its exact output-contract hash
through capability/request/response, preserves the `[H, 2]` rows beside decoded
actions, and leaves emission-time stage/commit/rollback to `robot_motion`.
Missing, malformed, or mismatched sidecars reject the whole inference. The
frontend also rejects any adapter/runtime contract mismatch before starting a
worker. Artifacts without the sidecar preserve the previous protocol hashes.
When `typed_canonical_v1` is selected, the client frontend records a versioned
representation context. Absolute actions use an explicit no-anchor context;
relative actions require the original request cut's fixed measured-TCP
anchors and are decoded to canonical absolute actions before plan admission.
The launcher preserves the legacy result path for absolute artifacts and
selects typed results only for an artifact-declared relative raw-RTC request.
Neither branch changes the OpenPI server payload.
Models with discrete outputs additionally require a typed discrete
prefix/mask/receipt before model-RTC can be enabled.

## RTC-4b-B model-RTC v3 provider

`rtc_conditioned.py`在既有typed OpenPI raw backend之上增加provider binding，不创建第二个transport、worker、plan或
decoder。canonical future先按artifact layout编码为external action；relative artifact使用请求cut冻结的同一TCP anchor
做absolute→relative逆投影。OpenPI server继续拥有normalize/pad，并在真实`sample_actions_rtc`调用后返回external、
internal和mask receipt。DexTeleop验证receipt后只裁剪一次delay prefix。

OpenPI远端改动是default-off且additive：

```text
ordinary websocket payload -> Policy.infer()       # unchanged
versioned RTC operation     -> Policy.infer_rtc()
```

只有显式`--enable-rtc-continuation`才发布
`rtc_model_continuation_v1` capability。全零mask必须标记
`unconditioned_bootstrap_v1`；未知operation、capability/dimension/hash drift或receipt context不一致均fail closed。
当前实现只支持continuous action prefix；它不修改teleop/replay/HITL、target、安全或dispatch，也不构成真实observation、
checkpoint效果或真机部署证据。
provider同时接受冻结v1 source contract和显式v2 source contract；普通full-chunk checkpoint的推理期hard inpainting
必须绑定`inference_only_projection_v1`且`training=null`，不得伪装成prefix-conditioned训练。
conditioned backend只接受base artifact内正式绑定的source contract；runtime config不再携带第二份contract，只声明
selected prefix schedule与sampler constraint。
server metadata还必须发布由loaded instance生成的provider capability hash、continuation implementation identity和
sampler constraint，call-site receipt逐项回传。客户端context回显不能替代这些provider assertions。
implementation identity由模型声明的versioned sampler描述与`Policy`实际sampling参数生成；server再把loaded
internal horizon/dimension和真实external action dimension纳入provider capability，不绑定checkpoint/config白名单。

artifact binding必须在请求循环外完成。`rtc_artifact_binding.py`接收原artifact、loaded provider capability及显式
delay/prefix范围，生成新的inference-only artifact和receipt；它不会改写checkpoint sidecar。一个可重放的实例位于
`deployments/openpi/configs/model_rtc/`，仅作为`0712-wrist-cube@92599`的loaded-instance证据，不是兼容白名单。

`OpenPIConditionedRtcBackendV3`直接满足通用
`RtcConditionedModelRuntimeAdapterV3`；application可将它交给
`build_rtc_conditioned_model_shadow_frontend_v3()`。该helper只把artifact/runtime contract与loaded capability投影到
既有bounded worker/timed-plan coordinator，并返回尚未启动的default-off frontend。它不复用或降格为raw/v2
request spec，不创建新的worker/plan owner，也不接observation source、target或dispatch。当前production
`teleop_runtime.deploy`尚未选择此profile；本地fake provider已验证bootstrap result进入plan后，下一次请求能消费同一
plan的canonical overlap prefix。默认re-request coverage由artifact contract的maximum forecast delay派生，避免慢模型
沿用raw-RTC的2-step阈值而持续退回bootstrap。

S1b1又在不启动server listener或DexTeleop frontend的前提下，只读加载
`0712-wrist-cube@92599`并直接调用真实GPU `infer_rtc()`。全零mask bootstrap与三帧hard-prefix两次输出均为finite
`50x16`，第二次实际消费indices `[0,1,2]`且三帧external输出与输入prefix完全一致。该结果只关闭loaded sampler
数值gate；OpenPI wire的provider assertion、v3 envelope、真实timed-plan/control-loop仍属于下一独立gate。

S1b2已用同一checkpoint完成localhost wire/frontend no-action shadow。动态delay从2升至16后，第二轮请求从尚未pop的
旧计划选择generation indices `[0..14]`，server consumption receipt、v3 response和plan admission全部通过。该闭环
保持`send_actions=false`、lookahead disabled、legacy queue pop为0；30 Hz control step最大13.88 ms。为此只修正
通用契约/时间语义：v2 continuation wrapper在worker校验前规范化、默认re-request coverage取artifact maximum delay、
model-RTC generation lattice锚定下一枚未pop keyframe。普通OpenPI infer、raw-RTC和production profile均未改变。

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
  --checkpoint-dir <checkpoint-dir> \
  --debug --rtc-debug-level summary \
  --debug-dir logs/openpi_raw_rtc
```

This consumes live arm/hand/camera observations and the existing OpenPI
transport. Without `--send-actions` it never sends actuator commands, while
retaining and validating the lab profile's qpos-init source, target, branch
search, interpolation and collision checks. After planning, no-action uses an
explicit plan-only reference cut and starts inference from the next fresh live
observation; it neither waits through a virtual qpos trajectory nor claims the
hardware reached the dataset start state. With an explicit `--send-actions`,
the same live RTC path retains and actually executes dataset-frame qpos-init
under the existing collision, observation and safety gates; that Active mode
needs separate hardware authorization and cannot inherit plan-only completion.
The explicit shadow profile is headless: it auto-starts policy observation and
defaults to no manual ready confirmation.
For Active use, append `--wait-for-ready-confirm` after the profile arguments
so the final CLI override restores the operator gate. The profile keeps
DMTAC tactile disabled unless a separate gate changes that scope. For every
launch it resets each artifact-selected RealSense by its configured serial,
waits for that device to reconnect, then starts cameras in camera-config order
with a 0.5 s stagger. Reset is limited to the selected serial; it does not
probe or rewrite stream profiles. The ordinary production profile retains its
ready confirmation, qpos-init, configured tactile behavior, and existing
camera startup defaults. It is not dataset replay: replay/synthetic tests
remain the deterministic complement for codec and plan semantics. The result
is only a real-shadow readiness record, not permission for motion deployment.

With `--debug`, each accepted model exchange also records a
`rtc_transport_exchange` event containing the request/input/response hashes,
chunk horizon and inference/result latency. This event is observational only;
it is not plan-admission, safety, dispatch, or worker-drain evidence.
`summary` is the default RTC level; use `trace` only for bounded reproductions
that require every control tick. RTC events use a bounded non-blocking queue;
queue pressure drops debug events and is reported at shutdown rather than
blocking policy or control work.

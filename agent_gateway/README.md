# Agent Gateway

`agent_gateway` is DexTeleop's top-level, upper-framework-neutral attachment
and Skill execution control plane. It is separate from the Control Station GUI.
An upper framework attaches through the published protocol and transport; it
does not import a simulator, device, controller, or Control Station runtime.

RPent is the first reviewed consumer, not a runtime dependency. Wire requests
must not contain an RPent Task, Program, Planner, Memory, Experience, or upper
SkillAttempt object. Route/deployment identity remains a trusted lower binding,
not caller-authored Agent input.

## Package ownership

- `protocol/`: versioned requests, operations, observations, evaluation
  receipts, Skill/schema/I/O catalogs, Artifact and replay-event contracts,
  Route manifests, a construction template, and pure validators.
- `runtime/`: client access policy, exact executor registry, bounded operation
  store, idempotency/reconciliation, passive reads, lifecycle workers, and
  injected lower operation/HOLD ports. `AgentGatewayProtocolRuntime` binds one
  session's discovery surface, replay journal, and optional Artifact cache and
  transfer broker.
- `transport/`: authenticated bounded transports and the transport-neutral wire
  adapter. The current local transport uses Unix peer credentials.
- `skills/`: target-independent Skill definitions and Skill-specific payload
  schemas. `skills/motion` owns Agent-visible motion identities and schemas;
  pure pose primitives remain lower-owned by `robot_motion.primitives`.
  Skills do not select a concrete route, frame, device, safety envelope, or
  deployment.
- `profiles/`: exact application/deployment bindings and executors. The current
  `profiles/mujoco` package owns the no-model TCP-pose Gateway application and
  its motion, observation, perception Artifact, policy-Skill, and CubeLift
  bindings.
  `profiles/real` owns the default-off real TCP-pose observation profile and
  the 6+3 primitive application. The preferred deployment shares one mailbox
  with an independent lower Skill owner under `teleop_runtime.skill_deploy`;
  the historic RealTeleop attachment is compatibility-only. Gateway workers
  perform no device I/O, IK, safety filtering, dispatch, or direct HOLD.

Shared activation-time application of reviewed TCP primitives lives in
`profiles/motion_primitive_resolution.py`; it derives an absolute lower target
from measured state and an exact frame graph but owns no safety, IK, authority,
or dispatch. MuJoCo-specific 6+3 Route IDs and immutable parameter profiles live
in `profiles/mujoco/motion_primitive_profiles.py`. The historic exports from
`profiles/mujoco/motion_deployment.py` and `motion_skill.py` remain available so
this organization-only split does not break consumers.

The neutral `protocol/runtime/transport` core must not import
`control_station`, concrete simulation, hardware, teleop, model deployments,
or an upper Agent framework. A concrete profile may assemble a Control Station
application and lower target through typed ports, but the lower target remains
the only canonical authority/safety/IK/reference/dispatch owner.

## One v2 Skill protocol

The current and only public Skill protocol is `dexteleop.agent_skill.v2`; the
local framed transport is `dexteleop.agent_gateway.unix_json.v2`. This is an
in-place repository-wide upgrade, not a second protocol family. Existing
inline Skills, including the nine TCP primitives, publish explicit empty
Artifact slot lists and run in an Artifact-disabled v2 session. A deployment
that exposes Artifact-bearing Skills enables the same v2 data plane through
its session capability document.

An attaching client discovers five mutually bound documents before invoking a
Skill: Definition Catalog, Schema Catalog, Skill I/O Catalog, Route Manifest,
and Session Capabilities. Hash closure prevents mixing documents from another
runtime session or deployment. The Unix facade additionally exposes bounded
event replay and generic operations to begin/upload/commit an Artifact,
begin/read/acknowledge a download, and acknowledge an upper-owned durable
materialization.

There is no RGB- or depth-image method in the protocol. A capture Skill declares
its selection parameters, output Artifact slots with media/schema/size
constraints, and a normal executor binding. The same machinery supports point
clouds, logs, calibration artifacts, or other finite Skill input/output without
changing transport semantics. Multi-file results use multiple bindings or one
explicitly encoded archive; v2 does not add a second bundle abstraction.

Large bodies never enter an operation or event payload. `AgentArtifactRef` is
the portable content identity; filesystem paths and other locators exist only
inside short-lived, authenticated access grants. The Gateway's managed store
is a transient, quota-bounded handoff cache:

1. reserve quota and create/recover a server-owned staging inode;
2. upload bounded chunks through `chunked_rpc`; an optional reviewed
   same-host `local_path` grant serves only committed read/download bytes;
3. verify byte length and SHA-256, then atomically commit into the local CAS;
4. pin inputs to the operation and retain outputs through the handoff window;
5. issue session/client-bound read grants and receive transfer receipts;
6. record the upper layer's materialization receipt once it has a separate
   durable copy;
7. remove unpinned data after minimum retention/handoff deadlines, retaining a
   tombstone rather than treating a stale path as identity.

Events use monotonic per-session sequence numbers, bounded retention, and an
explicit `gap_detected` result. They are notification hints only. After a gap,
clients reconcile through authoritative operation and discovery queries.

进程内完整载荷消费者使用 `artifact_store.read_bytes(ref, maximum=...)`：在消费者
字节上限内，从同一文件描述符读取并验证实际返回的长度和 SHA-256；损坏内容按原有
策略隔离。该接口不缓存载荷、不创建副本文件，不再让消费者校验路径后重新打开文件。
分块传输和路径授权继续采用既有流式校验，不把完整 Artifact 读入常驻下载缓存。

## Building a Skill

`protocol.skill_registration.SkillRegistrationTemplate` is the recommended
construction seam. A registration supplies one target-independent
`SkillDefinition`, one strict parameter/result/evaluation schema entry, one
`SkillIOContract`, and one or more deployment-independent Routes. The template
checks all identities, schema hashes, parameter profiles, I/O hashes, and
Route closure before `build_skill_registration_catalogs` emits the three
discovery catalogs. Concrete deployment bindings, device acquisition, camera
ownership, safety, and executors remain in `profiles/`; they are deliberately
not hidden inside the template.

## Upper-framework-neutral Skill semantics

This section specifies how an upper planner or orchestration framework must
interpret the existing v2 discovery documents. It documents the current wire
contract; it does not add fields to `SkillDefinition`, change any schema or
hash domain, or move Task/Program behavior into the Gateway.

A Skill is one stable semantic capability. It is not a prompt fragment, model
checkpoint, controller command, Route, deployment, or upper-framework attempt
record. An upper framework may organize Skills into direct calls, finite
Programs, behavior trees, graphs, or another planning representation, but each
lower invocation must resolve to one exact discovered Skill definition, schema
entry, I/O contract, Route, and deployment binding. A same-name Skill from a
different provider is interchangeable only when the versioned semantic
definition and required schema/I/O references match; name equality alone is
insufficient.

### Meaning of the definition fields

| Field | Protocol meaning | Must not be interpreted as |
| --- | --- | --- |
| `skill_id` + `skill_version` | Stable semantic identity selected by an upper planner. | A Route, checkpoint, device, or deployment identity. |
| `skill_kind` | Semantic class used with the Route control-effect closure: `inspection`, `motion`, `policy_task`, or `lifecycle`. | Proof that the caller is authorized or that the lower system is ready. |
| `summary` | Hashed human-facing explanation of the intended capability. Consumers may display it but must not parse it as executable policy. | A substitute for strict schemas, predicates, admission, or evaluation. |
| `intended_target_kind` + `cross_target_deployment_policy` | Whether the semantics are target-agnostic or require an explicit reviewed sim/real use. | Permission to cross-deploy a concrete Route or reuse a safety profile. |
| `controlled_component_ids` | Semantic set of components the capability may intentionally affect. | A resource lease, actuator allowlist, dispatch permit, or claim that every listed component moves. |
| `parameter_semantics_id` | Versioned meaning of the parameter payload. The Schema Catalog separately defines its exact JSON shape. | Caller permission to add parameters or author lower profiles. |
| `execution_result_semantics_id` | Versioned meaning of bounded execution-result facts returned by the lower implementation. | The upper Task result or a semantic success claim by itself. |
| `task_outcome_semantics_id` | Versioned meaning of the Skill-level outcome evaluated from reviewed evidence. | Gateway operation status, upper Task completion, or lower HOLD proof. |
| `entry_predicate_ids` | Named semantic preconditions that must be established by the owning lower Route/profile during admission or activation. | Facts made true by catalog discovery, planner assertion, or prompt text. |
| `task_outcome_predicate_ids` | Named predicates used by the reviewed evaluator to support a Skill-level outcome. | Conditions the upper framework may silently replace or infer from operation termination. |

The three `*_semantics_id` values name meanings; the three strict schemas
constrain bytes. Both are required. A payload can be schema-valid while using
the wrong semantic frame, unit, role, or outcome interpretation, so an adapter
must pin and check the definition hash and schema hashes together. Profiles
bind concrete frames, sources, tolerances, safety envelopes, and evaluators;
those bindings remain trusted assembly rather than Agent-authored arguments.

Entry predicates describe initiation semantics, but discovery does not
evaluate them. The lower owner checks live readiness, fresh measurements,
grants, resource ownership, safety-profile selection, and other Route-specific
conditions at admission or activation. An upper planner may decide not to call
a Skill, request a fresh observation, or replan when a predicate is not known;
it cannot make a lower predicate true by asserting it.

Skill termination has three distinct layers:

1. **Operation termination** reports what happened to the admitted lower
   operation (`succeeded`, `failed`, `cancelled`, `timed_out`, or an unresolved
   state).
2. **Skill outcome evaluation** applies the advertised outcome semantics and
   predicates to the required evidence and produces the typed evaluation
   payload/receipts.
3. **Upper Task or Program progression** is owned by the attaching framework,
   which may combine the Skill outcome with its own guards, retry budget,
   history, Memory, or task policy.

These layers must not be collapsed. In particular, a terminal or
`succeeded` operation does not by itself prove the advertised Skill outcome,
upper Task success, or lower-confirmed HOLD. Conversely, an upper planner's
decision to stop does not cancel, terminate, or hold a lower operation.

### Invocation, recovery, and evidence mapping

An upper-framework adapter should follow this mapping regardless of whether
the planner is an LLM, a symbolic planner, a behavior-tree engine, or another
agent framework:

1. Discover and hash-pin the Definition, Schema, I/O, Route, and Session
   Capability documents for one runtime session.
   不可变 Route Manifest 的语义哈希仅在该实例首次使用时计算，后续请求复用；
   新会话或 binding/Route 等变化仍创建新实例并重新计算。该复用不缓存实时观测、
   参数校验或准入决定，也不改变 wire/hash 协议。
2. Keep upper Task, Program/graph, guard, retry/fallback, planning-generation,
   Memory, and reasoning records outside the Gateway wire request. In RPent,
   `ExecutionIntent`, `SemanticProgramSpec`, and `SkillAttempt` are examples of
   such upper-owned records; they are not Gateway request extensions.
3. Materialize one invocation against the strict parameter schema and one
   reviewed parameter profile, then submit one exact `AgentSkillRequest` bound
   to the discovered definition, Route, deployment, Station revision, client
   sequence, request identity, and idempotency key.
4. Treat the resulting `AgentSkillOperation` as lower execution truth. Events
   are bounded hints; authoritative operation queries and discovery are used
   after a replay gap or reconnect.
5. Validate result metadata, Artifact bindings/content hashes, evaluation
   receipts, and any authoritative lower receipt before projecting a semantic
   result into the upper framework. Artifact materialization and long-term
   retention remain upper/data-plane concerns, not Skill success.
6. On a definite pre-admission rejection, the upper framework may revise its
   plan and create a fresh invocation. If submission may have crossed the
   transport boundary, reconcile only the exact original request; never create
   a new request or retry budget entry until the original acceptance state is
   known.
7. A semantic retry or fallback is a new upper attempt and receives fresh
   Gateway admission. Replacement of an active operation additionally waits
   for the required authoritative lower HOLD/handoff evidence, takes a fresh
   post-HOLD observation, and revalidates the replacement. Cancellation ACK,
   disconnect, elapsed time, event delivery, or ordinary operation termination
   is not lower-confirmed HOLD evidence.

The lower protocol deliberately does not define an upper Program language,
planner prompt, model API, Memory format, retry policy, or Task state machine.
It does define the facts any such framework must preserve: exact identity and
hash closure, strict payloads, fresh admission, idempotent reconciliation,
lower-owned operation status, typed outcome evidence, bounded Artifact
handoff, and fail-closed HOLD semantics. This is the replaceability seam for
changing the upper planning model or framework without changing downstream
execution ownership.

### Compatibility and semantic changes

Treat a change to parameters, units, frames, intended effects, initiation,
termination, outcome predicates, or failure meaning as a Skill semantic
change. An incompatible semantic change requires a new `skill_version` and
corresponding versioned semantics/schema references; it must not be hidden
behind a reused identity. A deployment-only change belongs in the Route or
deployment binding and changes the corresponding hashes rather than the
Agent-visible Skill meaning.

Because `summary` and the semantic reference fields are part of the hashed
definition, even a documentation correction inside a published definition
produces a new definition/catalog hash and requires rediscovery. Adding a new
public definition field or request field is a protocol/schema evolution, not a
documentation-only edit. Do not add upper-specific fields such as RPent Task,
Program, planning-model, Memory, `SkillAttempt`, or private deployment-binding
annotations to the exact Gateway request schema.

Control effect is a closed contract, not a naming convention. An `inspection`
Definition can only use a `read_only` Route; that Route must be
`t0_read_only`, use `cancel_only`, advertise no robot motion or publication
opening, and bind an executor that declares the same effect class. Motion and
policy-task Definitions use motion Routes, while lifecycle Definitions use
lifecycle Routes. Neutral no-I/O test/dry-run ports cannot prove either read or
control execution. The control-domain HOLD port remains separate from passive
read execution.

The exclusive real RM65 profile additionally advertises
`own_domain_emergency_stop_v1`. This is a domain-level safety control, not a
motion Skill: it requires no observation, Station revision, Grant, or active
operation and directly requests the lower owner to stop both arms. Acceptance
requires authoritative acknowledgement from both RM65 backends. The request
locks a one-way Gateway latch before lower I/O, supersedes an in-flight worker
or ordinary cancellation, and rejects every later motion admission. HOLD is
not emergency-stop proof, partial acknowledgement is a failure, and re-arm is
deliberately outside the Agent/RPent surface.

Safety-envelope selection is a motion-admission concern. A Route may publish a
finite set of compatible reviewed envelopes, but every admitted operation pins
one exact envelope ID, content hash, and constraint-set hash. If exactly one
compatible envelope exists, omission retains the single-envelope compatibility
default. If more than one exists and the session advertises
`safety_profile_selection_v1`, omission fails closed with
`safety_profile_selection_required`; the Gateway never guesses a tier from
ordering or names. Passive read and lifecycle Routes do not acquire a motion
envelope through this mechanism.

## Entrypoints

Operator-facing startup, shutdown, risk levels, and remote-developer setup are
maintained in the canonical [`docs/agent_gateway/`](../docs/agent_gateway/README.md)
domain. This package README remains the technical protocol and profile
reference.

The canonical MuJoCo entrypoint is:

```bash
DEXTELEOP_GATEWAY_SOCKET="$(
  python -m agent_gateway allocate-unix-endpoint
)"

python -m agent_gateway mujoco \
  --socket-path "${DEXTELEOP_GATEWAY_SOCKET}" \
  --task-config /absolute/task_config.json \
  --artifact-root /absolute/private/artifact_cache
```

`allocate-unix-endpoint` creates a unique mode-0700 directory under
`<DexTeleop>/temp/ag/` and prints its absent `g.sock` pathname. This separates
the bounded transport namespace from longer history, Artifact, Experience and
log paths. Use `--format json` when the allocation metadata should be retained
alongside a run record.

The allocated endpoint remains explicit: pass the exact returned path to both
the lower Gateway and RPent; neither side discovers or silently relocates it.
Linux limits pathname-based Unix sockets to 107 encoded bytes. Both MuJoCo and
real launchers retain a final validation before constructing a simulator,
device, or lower owner, and the server repeats it immediately before bind. An
advanced caller may pass `--root`, but is then responsible for choosing a root
that leaves enough pathname budget.

每个已认证客户端身份最多保留 **四条**本地持久连接，供会话级操作者界面和当前任务
各使用一条普通请求通道、一条独立急停通道。启动预检在 Dashboard 打开前结束；
临时接线也占用同一个有界额度，不额外放行。第五条连接会被拒绝，服务端日志记录
客户端身份、当前连接数及上限。若额度暂满，握手最多等待 250 ms，让已关闭连接完成
服务端登记清理；仍无空位则拒绝，不挤掉现有连接、不重放请求。这段等待发生在绑定
facade 和执行 RPC 之前，不阻塞控制循环。任务结束只释放自身连接，操作者界面的通道继续保留。
任何已认证通道断开或超时仍走原有的保守断连保护；这里不改变 HOLD 回执、权限或
急停语义。调用方不能把正常任务收尾等同于已经获得下层 HOLD 证明。

`--artifact-root` is optional and default-off. Supplying a real,
non-world-writable directory enables the read-only
`perception.capture_depth`, `perception.capture_rgb`, and
`perception.back_project` Skills. A motion Gateway that also has the optional
segmentation model exposes `perception.segment` and
`perception.plan_placement`. The placement planner consumes two retained
text-prompt segment operation IDs as identity anchors, reruns the registered
localizer on current correlated RGB-D cuts, and returns compact Station-frame
placement geometry. It never dispatches motion, and callers must
use a separate motion Skill after checking
`geometry_eligible_for_motion_planning`. Depth returns role-selected native-resolution
float32 metric data; RGB returns one role-selected JPEG per output Artifact
ordinal. Back-projection takes either one pixel or a half-open pixel region,
captures aligned live depth internally, and returns bounded geometry in the
registered Station frame; raw depth does not enter its control result.
Production RGB v2 also reports the admitted Station revision, an explicit
capture window, and each frame's acquisition timestamp while preserving the
requested semantic-role order. The trusted MuJoCo profile maps semantic roles
to lower camera sources. Inline capture results contain only bounded metadata
and content hashes. Pixel arrays never enter the control frame.

既有原子夹爪 Skill 完成开合时必须得到 `moving=False` 的实测状态；目标位置接近但仍在
运动或状态未知时不能完成。真机停止 ACK 之后还须在有限等待内读取新鲜、无故障的停稳反馈，
早于停止请求、过期、未来时间或不可用反馈均不能确认 HOLD。只读反馈沿用已启动 backend，
不在 Skill 层新建线程、重连或关闭设备。

原生命令下发仅对明确表示未写入设备的 `io_busy` 做有限重试，每次尝试前检查撤销与操作到期；
硬件故障、异常或不确定 ACK 不重试。同步调用返回后、最终状态读取后和 HOLD 确认后也检查
操作截止时间，迟到结果不能标为成功。重试预算不抢占正在进行的原生 I/O；原生调用依赖 backend
自身的 I/O 超时，返回后仍执行失败与 HOLD 处理。撤销与 HOLD 始终引用激活时的下层 owner。

本次接入保留既有 `manipulation.set_gripper_state` 夹爪合同，不注册新的开合或夹爪
状态 Skill。MuJoCo 放置几何读取同一个 scene cut 的 TCP 与实测夹爪状态；当前真机
放置几何适配器尚未绑定夹爪测量，因此返回 `held_gripper_state_unavailable`，不会将结果标为
可用于运动规划。`geometry_eligible_for_motion_planning` 也不代表通过了轨迹碰撞准入、
已抓住物体或可直接执行放置；执行仍需独立运动授权和下层校验。

`python -m control_station agent-gateway-mujoco ...` remains a compatibility
forwarder. It creates no Gateway implementation or runtime state inside
`control_station`.

### Read-only SSH attachment

The canonical host-administration and remote-developer procedure is
[`docs/agent_gateway/SSH_READ_ONLY.md`](../docs/agent_gateway/SSH_READ_ONLY.md).
The details below define the implementation contract rather than a second run
recipe.

`ssh-readonly-proxy` is a small forced-command bridge to an already-running
host-local Unix Gateway. It does not start hardware, create another Gateway,
or own a device. Its fixed method allowlist permits discovery, bounded status,
strictly read-only Skill submission, own-operation query, read-only cancel, and
Artifact downloads. Generic Skill submission, lifecycle, HOLD, Artifact upload,
and all motion routes are absent from the allowlist. The Gateway independently
checks that `submit_read_only_skill_request` resolves to a `t0_read_only`,
`cancel_only`, non-motion Route before admission.

The server-side command is:

```bash
python -m agent_gateway ssh-readonly-proxy \
  --socket-path /absolute/private/gateway.sock
```

Deploy it through a dedicated no-shell SSH account and a key-specific forced
command. Each concurrently attaching developer needs one distinct Unix UID and
one distinct Gateway `client_id`; do not multiplex several developers through
one account or reuse the primary trusted client's identity. Add every remote
identity explicitly when starting the existing Gateway:

```bash
python -m agent_gateway mujoco \
  --socket-path "${DEXTELEOP_GATEWAY_SOCKET}" \
  --task-config /absolute/task_config.json \
  --client-id rpent.dexteleop \
  --client-uid "$(id -u)" \
  --ssh-read-only-client rpent.remote.alice 1101 \
  --ssh-read-only-client rpent.remote.bob 1102
```

`real-rm65` accepts the same repeatable option. With no additional identity,
the endpoint stays mode `0600`. Adding one or more read-only identities changes
only the endpoint mode to `0660`; the Gateway process and the no-shell accounts
must therefore share one dedicated local group. Allocate a traversable session
directory below a pre-created setgid root owned by that group:

```bash
install -d -o gateway-user -g dexteleop-agent-ro -m 2750 \
  /run/dexteleop-agent-gateway

DEXTELEOP_GATEWAY_SOCKET="$(
  python -m agent_gateway allocate-unix-endpoint \
    --root /run/dexteleop-agent-gateway \
    --group-access
)"
```

The allocated session directory is mode `2750`, so the socket inherits the
dedicated group; group users may traverse it but cannot unlink the socket. The
launcher does not change account/group
membership or ownership and does not run `chgrp`. Account and group setup stays
an explicit host-administration step.

The primary `--client-id` remains the only identity eligible for the real
profile's reviewed persistent motion session. Additional identities are marked
read-only in the Gateway itself: generic submission, lifecycle, HOLD, event
replay, Artifact upload, and materialization acknowledgement are rejected even
if a caller bypasses the proxy. Strict read-only Skills, bounded status,
own-operation reads/cancel, and own-operation Artifact downloads remain
available. This keeps remote access read-only without introducing a second
Gateway, broker, lease service, or hardware owner.

The SSH identity maps to the Unix UID accepted by the Gateway, while payload
`client_id` remains non-authoritative. The proxy uses the existing
length-prefixed `dexteleop.agent_gateway.unix_json.v2` frames, so a client does
not need a second hardware protocol. Remote Artifact access is limited to
`chunked_rpc` bytes exposed by an own-operation response on the same SSH
connection; arbitrary cache IDs and host-local paths are denied. If another
operation or lower resource claim occupies the device, the normal Gateway
admission/status result is returned; remote access does not queue, preempt, or
acquire a separate lease.

Two additive discovery calls expose a profile-owned, redacted view:

- `get_hardware_resource_views`: `readable`/`drivable` capability plus exact
  semantic Skill references;
- `get_hardware_status_views`: binary `free`/`occupied` occupancy, separate
  health, and an optional active operation projection.

These views are not a registry or lease system. Concrete profiles bind stable
resource IDs to existing Route hashes and lower-owned control-claim and health
reads; the runtime adds its current operation projection. `occupied` means that
an exclusive lower control/resource claim or an active exclusive operation is
present. It does not mean that a Skill is currently moving the robot. READY,
idle, and lower-confirmed HOLD do not release a claim and therefore never imply
`free`; the optional `active_operation_id` reports execution activity
separately. A status read failure is reported fail-closed as unavailable and
occupied. If a profile provides no binding, discovery returns `enabled=false`
and an empty list rather than guessing hardware identity. Gateway admission,
not this advisory view, remains authoritative for whether a Skill may start.

The exclusive real primitive profile publishes its existing arm `component_id`
as one readable/drivable `robot_arm` ownership unit. Its occupancy comes from
the lower `RealAgentMotionOwnerCut`, including READY and HOLD, while health uses
only cached lower state and freshness. When real RGB/depth capture is enabled,
the profile additionally publishes one shared, read-only
`perception.cameras` bundle. A running lower camera worker projects the bundle
as occupied even when no capture Skill is active; camera lifecycle health is
read without capturing a frame. This grouping matches current lower admission
granularity and does not claim independent per-camera leases that the lower
runtime does not provide.

The MuJoCo TCP-pose profile publishes one readable/drivable
`simulation.runtime` resource. Motion, TCP state/history, and optional RGB/depth
capture all share the same scene and runtime owner, so the profile does not
invent independent simulated arm or camera leases. Before the lower runtime is
started the resource is `free` and `unavailable`; after observation-ready
startup it remains `occupied` through READY and lower-confirmed HOLD until the
application closes. Health is derived from the existing execution-gate state
without stepping physics or rendering a frame, and active Skill execution is
still reported separately through the operation projection.

The dedicated independent RM65 Skill entrypoint is:

```bash
python -m agent_gateway real-rm65 \
  --socket-path /absolute/private/gateway.sock \
  --arm-ip-assignment /absolute/rm65_arm_ips.json \
  --left-tool-profile /absolute/left_tool.json \
  --right-tool-profile /absolute/right_tool.json \
  --envelope-profile /absolute/right_skill_envelope.json \
  --authorized-side right \
  --authorized-action forward \
  --confirm AUTHORIZE_PERSISTENT_REAL_TCP_PRIMITIVES
```

Real RGB/Depth support is opt-in on that same v2 session:

```bash
python -m agent_gateway real-rm65 \
  --socket-path /absolute/private/gateway.sock \
  --arm-ip-assignment /absolute/rm65_arm_ips.json \
  --left-tool-profile /absolute/left_tool.json \
  --right-tool-profile /absolute/right_tool.json \
  --envelope-profile /absolute/right_skill_envelope.json \
  --camera-config /absolute/DexTeleop/teleop_runtime/configs/devices/agent_gateway_realsense.json \
  --camera-registry /absolute/DexTeleop/robot_motion/configs/rm65_camera_registry.json \
  --artifact-root /absolute/private/artifact_cache \
  --authorized-side right \
  --authorized-action forward \
  --confirm AUTHORIZE_PERSISTENT_REAL_TCP_PRIMITIVES
```

For passive camera input without constructing or connecting either RM65 backend,
use the separate camera-only profile:

```bash
python -m agent_gateway real-perception \
  --socket-path /absolute/private/gateway.sock \
  --camera-config /absolute/DexTeleop/teleop_runtime/configs/devices/agent_gateway_realsense.json \
  --camera-registry /absolute/DexTeleop/robot_motion/configs/rm65_camera_registry.json \
  --artifact-root /absolute/private/artifact_cache
```

By default, `real-perception` exposes `perception.capture_rgb`,
`perception.capture_depth`, and `perception.back_project` in
`real_observe` mode. It owns the configured
RealSense inventory and transient Artifact store, starts and admits cameras
before publishing the Unix socket, and has no RM65 backend, motion Route,
operator Grant, command owner, dispatch path, or domain HOLD path. It therefore
does not accept `--arm-ip-assignment`, tool/envelope profiles, an authorized
motion action, or `--confirm`. The selected collision-scene profile is read only
to derive fixed arm-base geometry for static camera poses; it does not construct
or connect a robot backend.

For a session that must never reopen or reset a camera, pass
`--camera-single-start`. This explicit camera-only deployment option disables
startup software retries, hardware resets, runtime reopen policies and the
automatic-exposure-priority write after applying formal startup defaults.
Serial bindings, stream settings, startup priorities, complete RGB-D and
consecutive-fresh-frame admission are unchanged. The effective policy is part
of the source binding identity; missing cameras fail admission, not RGB-only
fallback. The default policy remains unchanged for existing consumers.

可选 `--camera-sdk-debug-log NEW_ABSOLUTE_PATH` 保存原生 DEBUG 日志和 SDK/启动上下文，
不改变相机流、reset 或新鲜度要求。默认不启用；现有文件和符号链接均拒绝覆盖。
启用日志会提前导入 SDK，单次成功不能据此证明初始化时序问题已根治。

相机应用关闭时，先确认 Gateway 已停止准入且所有只读任务已经结束，再释放模型、
相机和 Artifact store。任务等待超时或关闭被拒绝时，应用返回 `FAULT` 并保留这些依赖；
任务结束后可再次调用 `close()` 完成清理。启动失败也使用同一关闭流程。

`real-perception --enable-back-projection` 独立发布
`perception.capture_rgbd@1` 和 `perception.back_project@2`，不加载 SAM2/GraspGenX，
不连接机械臂。既有实时像素／区域 `perception.back_project@1` 及其参数保持兼容；
留存观测调用必须同时按 Skill id 和版本选择 `@2`，不能只取同名 Route 的第一项。
RPent 等按工具名投影接口的客户端应在 `tool_name_by_skill` 中显式区分别名：
`"perception.back_project@1": "gateway_perception_back_project_live"` 与
`"perception.back_project@2": "gateway_perception_back_project_retained"`。
请求的 `skill_version`、Route/hash 和参数 Schema 必须来自同一发现条目。
前者复用已有冻结 RGB-D 编码与相机 owner，在同一 VideoFrame 中
保留 RGB、对齐米制深度、内参及采集时相机位姿，并通过 `rgb_frames` 返回可见预览。
后者传入该采集的 `observation_operation_id`、`role`、`pixels_uv`（1–128 个 `[u,v]`），
只读取同一客户端、同一会话仍保留的观测，批量返回相机系与公开参考系 XYZ。
不重新取帧、补深度洞、调用模型或提供运动权限；结果是被观测表面，不是物体中心或 6D 位姿。
源观测已淘汰、所选视角被排除、像素越界或深度无效时明确失败。历史观测可查询，但结果
保留原始采集时间与 Station revision，不冒充当前场景；运动决策前仍需重新观测。
启用该两步工作流时 `max_retained_operations` 至少为 2；容量不足在装配时拒绝，
其他操作仍可能按既有保留策略淘汰旧观测，不能把 Artifact 尚存等同于 operation 仍可用。
只发布实际目录中对齐到 RGB 且有可信位姿的深度角色；非对齐深度仍可用于既有
`capture_depth`，不能用于逐 RGB 像素反投影。当前 camera-only 默认只有主相机，不能把固定标定
或主相机深度套到腕部图像。动态腕部需要应用层提供对应采集时刻的几何，不能由 Agent 猜测。
既有抓姿消费者继续使用 `capture_grasp_observation` 的静止/qpos 合同，两种入口复用同一编码和采集实现。

普通多视角 `capture_rgb` 先取得并复制本次请求所需的 RGB 像素，检查视角时间偏差，
再按请求顺序编码 JPEG。私有副本不保留深度或完整源帧，编码时逐张释放、调用结束后
不保留像素缓存；临时容量由本次视角数和原始分辨率决定。采集时的新鲜度、设备身份、
generation 与编码后的 Station revision 检查保持有效，不改变相机启动或流配置。

已有 `real-rm65-skill` 双臂会话也可选择 `--enable-back-projection`，复用同一个相机 owner、
lower 测量发布与 `RM65RgbCameraGeometry`。这条入口按已对齐 RGB-D 目录发布主相机与两腕角色，
不需要 SAM2/GraspGenX；普通 `capture_depth` 仍只发布固定外参深度，不改写其既有语义。
腕部没有有效位姿、位姿过期或帧/位姿时间偏差过大时排除该视角；全部缺失时失败，不套用
主相机外参。动态腕部复用已有静止历史选择与帧匹配的测量，不用最新 qpos 替代采集时位姿；
静止历史不完整或帧前后发生运动时排除腕部。此阶段不承诺运动期间的曝光时间配对，
固定主相机不需要静止机械臂。冻结数据保留逐视角 `stationary_only` 和 `camera_pose`
测量/可用时间及证据；后续机器人移动不改变原观测的
投影。只读 Skill 不发送运动/HOLD/STOP，但该既有双臂应用的启动与收尾仍保持原来的
设备/初始 HOLD 行为，不能当成相机-only 启动命令。上层是否暴露动作由独立策略决定。

`--enable-object-localization` adds the supplied-image localization Route after
loading its reviewed model bundle. `--enable-point-grasp` is a separate,
default-off module and requires that same localizer plus a reviewed point-grasp
bundle. GraspGenX in-process inference is the default and does not require a
loopback service; VGN, GraspNet, and service-mode GraspGenX are explicit
compatibility choices:

```bash
python -m agent_gateway real-perception \
  --socket-path /absolute/private/gateway.sock \
  --camera-config /absolute/DexTeleop/teleop_runtime/configs/devices/agent_gateway_realsense.json \
  --camera-registry /absolute/DexTeleop/robot_motion/configs/rm65_camera_registry.json \
  --artifact-root /absolute/private/artifact_cache \
  --enable-object-localization \
  --object-localization-asset-bundle /absolute/assets/perception/sam2_grounding_dino \
  --enable-point-grasp \
  --point-grasp-asset-bundle /absolute/assets/perception/graspgenx-point-grasp \
  --graspgenx-mode in-process \
  --graspgenx-source-root /absolute/AgenticRobot/model/perception/graspgenx
```

SAM2 掩码边界仅扫描行列占用，保持原来的半开区间和空掩码结果，避免展开所有前景像素坐标。
同一模型 owner 在既有锁内最多保留一份精确 RGB 身份及其 DINO 预处理输入；检测框、完整文本掩码
和点选共用这条路径。像素或尺寸变化、编码失败及关闭会使旧输入失效，每个新提示仍重新执行预测，
掩码、检测结果、深度与机器人状态不作缓存；已有分数、二值掩码及点选包含关系校验保留。
DINO 输入额外常驻量受原有 800/1333 resize 限制，float32 至多约 12.2 MiB；新帧和首次推理不承诺加速。
DINO 模型在首次文本推理时绑定所解析的物理设备，后续复用官方 predict，避免重复遍历 model.to；
关闭时由同一 owner 释放引用。权重、精度、阈值和输入 tensor 的官方迁移路径保持不变。
localize 的 last_call_timing_ms 仅报告当前调用的锁等待和推理阶段耗时，不构成控制或测量时间依据。

冻结的 Skill schema entry/catalog 仅复用其静态身份字符串；每次发现仍返回新的可修改 wire 副本，
替换 schema 会重新计算身份，逐请求 payload、Route 闭包和授权校验保持原样。
一次性 client 只在读取相机抓姿提案或创建服务端授权 owner 时加载相应应用代码，现有动作和右夹爪 v1 合同不变。

保留 RGB-D 的原有 NPZ/ZIP schema、NPY 成员、无 pickle 校验和 16 MiB 上限。组合抓取观测通过同一入口
编码并完整验证，每个嵌入视图只解码一次；规划读取直接复用这些已验证数组和采集时 qpos。
新原始数组采用无损 deflate level 3，接近原字节上限时改用原 level 6 再验证；接收已有视图字节的入口
始终逐字节保留输入，不重新压缩。压缩包可能略大，最终字节、解压体积、形状和元数据门禁均保留。

### 可选 SAM3 服务：文本目标与图片点选

`real-rm65` 和 `real-perception` 的 `--object-localization-backend` 默认仍为
`sam2-grounding-dino`。未显式选择 SAM3 时，原模型、阈值、自动分割和启动路径不变，
不会导入 SAM3 client 或连接额外服务。GraspGenX 的 `in-process` 默认也不变。

SAM3 复用 RPent 的通用服务，在它自己的模型环境中独立运行，不向 Gateway 环境
安装 SAM3，也不把相机、抓姿、IK 或运动功能放入服务：

```bash
SAM3_CHECKPOINT_PATH=/absolute/checkpoints/sam3.pt \
  python -m rpent.robots.components.sam3_server \
  --transport http --host 127.0.0.1 --port 8114
```

在现有 Gateway 命令中显式加入下面的选项，并移除
`--object-localization-asset-bundle`；RPent 手动脚本对应移除
`--sam2-grounding-dino-bundle`。相机、工具标定、抓取 profile、深度处理和执行确认保持原样：

```text
--object-localization-backend sam3-service
--sam3-endpoint http://127.0.0.1:8114
--sam3-timeout-sec 120
--sam3-min-score 0.2
```

此版本支持文本提示的**单个最佳实例**和原图像素点选；`max_detections` 仍是上界，
不是承诺返回多个物体。无文本自动分割暂不支持，会明确拒绝，不能以空结果假装成功。
同一实例支持现有 `perception.segment` 的文本和点选请求，复用标准 mask NPY 与
overlay JPEG Artifact 编码、分数阈值及发布时的观测版本检查。
文本没有检出时可按现有交互流程由操作者改为图片点选；不会自动换回 SAM2 或偷偷
降低阈值。Gateway 的 `(u,v)` 只在 client 中转换一次为服务的 `[row,col]`，返回
掩码必须与输入图同尺寸，且点选掩码必须包含点击点。多视角仍区分正常提示未命中与
传输／模型／畸形掩码故障；后者不会被当作可重试的背景点。

适配器只使用 loopback HTTP，显式超时、不重试、不跟随重定向，也不关闭共享服务。
现有 SAM2 命名的下游历史类／计时字段不代表实际选择的模型；部署身份绑定所选
localizer。SAM3 的 profile hash 表示服务协议与 endpoint 配置，**不证明外部服务的
checkpoint 字节身份**，因此应在服务自身的部署记录中保留源码、权重与环境信息。
离线掩码或规划通过不代表深度、标定、真实接触或抓取成功已经验收。

The live `perception.plan_point_grasp@2` request contains only a registered
aligned-depth role, a clicked pixel, and a bounded candidate count. The
application reads RGB and aligned depth from one published `VideoFrame`, binds
source generation/frame index, intrinsics, static camera calibration, Station
revision and frame graph, and returns ranked canonical-Station tool-TCP
proposals. It does not select a candidate, expose an RM65 backend, mint a Grant,
or execute motion. Its visible-scene collision flag is not a complete
scene/robot/path safety proof. The caller-calibrated v1 Route remains available
only to the separate offline tooling.

The DMTAC M adapter distinguishes each model's grasp frame from
the calibrated actual `tool_tcp`. 对完整 4B2 CAD 装配，独立 real-rm65 入口的
GraspGenX 显式绑定模型几何框架：相对真实控制 TCP 横向偏移 24.5 mm、轴向
偏移 8.2 mm，绕 +Z 旋转 90°，使几何 +X 对应实际夹爪开合方向；真实 TCP
标定仍为 flange `[0, 0, 0.140, 0, 0, 0]`。97.001 mm 内部 marker 在完整
装配中的 flange 深度是 105.201 mm，不能把零平移 proxy marker 直接当作
完整安装后的 TCP 几何。模型评分、目标覆盖及多样性检查保持在模型几何
框架内；候选输出明确转换为控制 TCP，渲染、捕获时夹爪自身点云过滤和
可见接近/路径检查使用同一份 `solid_geometry` 转换。快照保留该转换；旧
快照未声明转换时仍按原定义解码，不会静默重解释旧候选。独立离线工具
与其他 backend 的既有默认框架不变。
这是名义双指代理的坐标修正，不代表完整 CAD 手指碰撞、实际开度、闭合
接触或携物退出已通过验证；当前自身过滤与路径检查都近似采用名义最大张开状态，
不代表已测量实际开度。
The combined RM65 profile
rejects point-grasp startup when the selected arm tool profile does not match
that reviewed TCP; RPent never supplies or repairs this tool geometry.

Real RGB and Depth use the shared v2 result contracts. RGB preserves each
camera worker's source timestamp and bounds the selected frames by the minimum
and maximum source timestamps; it does not claim that separate RealSense reads
are one atomic exposure. Each Depth result pins the admitted
`station_revision`, `canonical_station` frame specification, and complete
source-world-to-Station graph hash, and reports
`camera_pose_reference_xyz_quat_xyzw` as
`T_canonical_station_camera_optical_frame`. The camera-only profile constructs
the same typed `dual_arm_solver_world -> canonical_station` graph from the
selected collision-scene URDF fixed-joint chain as the combined RM65 profile;
it never substitutes a geometry digest for a frame graph. The provider retains
its internal `T_dual_arm_solver_world_camera` calibration cut, while the
Gateway rejects a revision change before or during Artifact commit. The legacy
v1 registration builder, RGB `captured_at`, and Depth
`camera_pose_world_xyz_quat_xyzw` projections remain available only for
existing compatibility callers; production camera-only and RM65 perception
composition use RGB/Depth v2.

For a finite input-Skill acceptance run, use the purpose-built probe from a
clean checkout. It fixes the camera config, registry, and collision-scene
profile to the reviewed repository values and exposes no robot, motion,
hardware-reset, or configuration-substitution argument:

```bash
"$(bash scripts/resolve_dev_python.sh)" \
  -m agent_gateway.profiles.real.perception_probe \
  --run-root /absolute/private/real-perception-probe-run
```

The probe starts the real camera-only application, requires the Unix server
socket to be published, then uses one authenticated bound wire adapter to call
`submit_read_only_skill_request` for `perception.capture_rgb` and
`perception.capture_depth`. It downloads every output through `chunked_rpc`,
checks immutable length/SHA-256 descriptors, decodes all three RGB JPEGs, and
loads the front depth Artifact as a native-resolution 2-D float32 metric NPY.
It always closes the application, cameras, executors, socket, and transient
Artifact store. Success prints one `agentic_robot.real_perception_skill_probe.v1`
JSON object and saves the same bounded evidence as
`real_perception_skill_probe.json` under `--run-root`; raw image/depth bytes are
not included in that JSON.

探针以执行器退出之后的新操作快照判断最终状态，避免把退出前读到的 `running` 误报为异常。
执行器已退出而新快照仍非终态、真实失败或超时仍按原合同拒绝；本地替身测试不代表真实相机验收。

This probe covers real Gateway startup, camera admission, both passive Skill
executors, operation polling, and Artifact transfer/decoding. The Skill calls
use the in-process bound wire adapter after a real server socket is established;
an independent RPent AF_UNIX client remains a separate end-to-end acceptance
boundary.

The three perception flags are all-or-none. The checked-in deployment profile
publishes RGB for `main`, `wrist_left`, and `wrist_right`, but publishes depth
only for `main`. The front camera has a reviewed static camera-to-arm-base
transform; wrist depth is deliberately unavailable because those cameras move
with link 6 and the current registry does not provide the activation-time
joint/FK pose needed by the depth result contract. Hardware reset is disabled.

The reviewed front D435 deployment explicitly accepts near-zero RGB payloads
because that installed stream is expected to be dark; those frames remain
counted in camera diagnostics and are published instead of causing restart.
Only that front entry enables two finite software-only startup retries. A retry
stops and rebuilds the SDK pipeline with cancellable delay and never calls
`hardware_reset()`. Wrist cameras and all older profiles retain the default
near-zero-frame rejection/restart behavior and zero software startup retries.

Both capture Routes are `inspection` / `read_only` / `t0_read_only`: invoking
them needs no motion Grant and cannot call command admission, dispatch, or
HOLD. RGB is JPEG with per-frame acquisition time and a bounded multi-camera
capture window; depth is native-resolution float32 metric NPY with aligned
color intrinsics and the trusted canonical-Station camera pose. Payload bytes
are committed to the same generic transient Artifact CAS and retrieved through
`chunked_rpc` or a reviewed same-host `local_path` grant.

For both real profiles, the selected `--collision-scene-profile` is the source
of Station geometry. Its scene URDF fixed-joint chain defines both arm bases in
`dual_arm_solver_world`; the camera registry then defines each fixed camera
relative to its parent link. A Vive/teleoperation `TeleopCalibration` is neither
loaded nor accepted by either entrypoint.

`real-rm65` remains the combined real application. Starting it acquires and
admits all configured cameras, then starts the exclusive dual-RM65 lower owner
before publishing the socket; consequently the persistent motion-session
confirmation and real-hardware safety preflight remain mandatory even when an
upper client intends to invoke only passive Skills. Its runtime TCP-envelope
anchors come from the fresh lower-measured qpos/FK cut after startup HOLD, not
from a teleoperation Home record. A capture invocation itself has no downstream
control effect. Repository offline tests establish neither live camera nor
robot execution for either profile.

### 独立真机 TCP 只读采样

没有运行中的控制 owner 时，可使用同一 dispatcher 的 `real-tcp-observe` profile：

```bash
python -m agent_gateway real-tcp-observe \
  --socket-path /absolute/private/gateway.sock \
  --arm-ip-assignment /absolute/rm65_arm_ips.json \
  --left-tool-profile /absolute/left_tool.json \
  --right-tool-profile /absolute/right_tool.json \
  --envelope-profile /absolute/absolute_standard.json \
  --client-id operator.readonly
```

该会话由一个采样线程持有双 RM65 原始 SDK 连接，读取
`Get_Current_Arm_State(retry=0)`，复用既有 URDF/tool FK 与 Station frame graph
发布 `state.read_tcp_pose@1`。左右臂须绑定不同端点，重复配置在 FK 构造与连接前拒绝。
采样 owner 启动前检查本机已有的 RM65 TCP 连接，
发现占用即拒绝。Gateway 只消费缓存，发布 `readonly_sampling` /
`unavailable_no_command_owner`，控制 epoch/generation 为零；没有运动、夹爪、
HOLD/STOP 或直接刷新设备的调用端口。

工具 profile 决定 TCP；envelope 文件仅作为既有部署 profile 的配置输入，
不裁剪实际测量，也不授予运动准入。SDK 没有传感器时间戳，测量时间采用双臂请求
的最早开始时刻，保守计入等待、双臂时差与 FK；时钟异常、超过 200 ms 的 cut
或 SDK 错误均停止采样，不重连或回退到旧值。启动超时锁定失败并请求取消。

关闭时先取消后续读取与发布，采样线程在在途调用返回后关闭自己持有的 handle；
调用阻塞时显式报告清理未完成。transport 仅在初始观测就绪后发布，启动失败也会
回收已持有的资源。本仓库离线和假 SDK 测试不构成真实设备验收。

The combined three-view point-grasp profile is an explicit opt-in. It uses the
reviewed v4 head-camera placement and derives both wrist-camera poses from the
fresh measured dual-arm qpos consumed by the sole lower control loop:

```bash
python -m agent_gateway real-rm65 \
  --socket-path /absolute/private/gateway.sock \
  --arm-ip-assignment /absolute/rm65_arm_ips.json \
  --left-tool-profile /absolute/left_tool.json \
  --right-tool-profile /absolute/right_tool.json \
  --envelope-profile /absolute/right_skill_envelope.json \
  --camera-config /absolute/AgenticRobot/teleop_runtime/configs/devices/agent_gateway_realsense.json \
  --camera-registry /absolute/AgenticRobot/robot_motion/configs/rm65_camera_registry_v4.json \
  --camera-mount-urdf /absolute/AgenticRobot/assets/rm65_inspire_gripper_v4/rm65_dual_arm_wrist_cameras_v4_head_camera_4b2_cad_with_legacy_housing.urdf \
  --artifact-root /absolute/private/artifact_cache \
  --enable-point-grasp \
  --object-localization-asset-bundle /absolute/assets/perception/sam2_grounding_dino \
  --point-grasp-asset-bundle /absolute/assets/perception/graspgenx-point-grasp \
  --graspgenx-host 127.0.0.1 \
  --graspgenx-port 5556 \
  --authorized-side right \
  --authorized-actions move_tcp_to_pose execute_grasp_approach_plan \
  --confirm AUTHORIZE_PERSISTENT_REAL_TCP_PRIMITIVES
```

For the point-grasp opt-in, the detailed v4 URDF shown above is the default
`--camera-mount-urdf` and the v4 registry shown above is used when
`--camera-registry` is omitted; the explicit arguments keep deployment evidence
self-describing. The generic camera-only profile still requires an explicit
registry. URDF, registry,
Station, arm-base, and live pose-cut disagreement is rejected before a rendered
candidate is returned. This profile connects both arms and requires startup
HOLD even when a client invokes only read-only Skills.

Adding `--enable-point-grasp-arm-reachability` upgrades only this combined
profile to `perception.plan_point_grasp@3`. Its request-gated annotation runs
the existing lower RM65 endpoint IK from one fresh measured-qpos snapshot and
classifies each unchanged candidate as standard (3 mm/1 degree), basic
(5 cm/20 degrees), or difficult within a 1-second/2,880-function-evaluation
per-candidate budget (16 seeds, at most 180 evaluations per seed). No new seed
starts after the wall-clock deadline is observed; one in-flight lower solve is
not preempted and its full duration is reported.
Omitting the flag preserves the exact v2 contracts and hashes; setting
`include_arm_reachability=false` on v3 performs no IK. The annotation is not
path planning, collision clearance, motion admission, or grasp-success proof.

The combined point-grasp profile also publishes a paired contract for the
preferred downstream flow. `perception.plan_grasp_approach@1` accepts only a
retained point-grasp operation ID, candidate ordinal, and speed scale. It binds
the candidate to a fresh measured-qpos cut, applies the deployed TCP envelope,
endpoint IK, production URDF/FCL scene, and staged reach planner, then returns
either `path_found` or a bounded rejection reason. This inspection Route cannot
dispatch motion or request HOLD. A successful result contains an immutable,
session-local plan ID/hash with a maximum 30-second lifetime.

The reach planner first validates a fixed-orientation Cartesian approach along
tool +Z for each pregrasp IK branch, then connects globally to the set of exact,
validated pregrasp joint states using a shared start tree and goal forest. The
selected goal retains its original local approach; IK is not re-solved at the
connection. A failed branch does not discard the remaining branches. It searches
pregrasp distances from 60 mm down to a 20 mm minimum in 10 mm steps, preferring
the longest fully feasible route rather than projecting a blocked pose. Local
targets are at most 5 mm apart. Dense FK revalidation rejects lateral deviation
above 2 mm, orientation deviation above 1 degree, reversed progress, or a joint
branch step above 12 degrees. The final quintic samples also obey the 0.1-degree
geometric resolution used by the collision/envelope checks. An already-reached
exact target remains a HOLD trajectory; it does not back away and approach again.
These values belong to `RM65GraspTrajectoryPlanningConfig`, not caller-controlled
motion permits. Rejected near-field stages return typed reasons and preserve
the complete lower diagnostics in the Gateway log; the published diagnostics use the optional fields and contract migration described below.

Start rejection diagnostics (2026-10-22 integration): `path_start_collision_blocked`
continues to mean that the composed path admission rejected the start; it does
not by itself prove physical FCL contact. Optional `diagnostics.start_qpos_source`
identifies `input_qpos` or the planned `envelope_ingress_endpoint`. Private callers
that do not know the source omit it. The lower copies the actual composed check
already used by the sampler; no additional collision, FK or envelope query runs.
The observed-scene admission also copies its existing check once.

Optional `diagnostics.start_check_json` contains a finite JSON object, limited to
2048 decoded characters (not total wire bytes). It includes safe/available,
distance and reason, plus bounded geometry, observed and composed
`collision_check` / `target_envelope` metadata when representable. A projection
may mark `details_omitted`; full lower details remain in Gateway logs. Its
`detail_scope=published_projection` and `detail_hash` cover exactly the final
published object excluding `detail_hash`, including `detail_scope`, using
`dexteleop.grasp_start_check_projection.v1`. This is not a full-detail hash.
Trimming occurs before hashing. If projection, hashing or encoding fails, the
optional JSON field is omitted and the original refusal remains authoritative.

This intentionally changes the result-schema, IO-contract and planning Route /
deployment-binding hashes for `perception.plan_grasp_approach@1`. Consumers must
rediscover the current catalog and rebind to its hashes before submitting new
requests. An old strict result validator rejects these new optional fields; the
new validator continues to accept legacy results without them. Parameters, Skill
semantics, top-level refusal codes and the separate execution contract retain
their existing meaning. Planning remains read-only, with no motion or HOLD,
no fallback enabled by this diagnostic addition, and the same lock and deadline
owners. Offline tests do not constitute physical grasp acceptance.

Grasp planning uses `dual_arm_hand_proxy_grasp_full`: the existing strict URDF
with non-adjacent real-arm mesh self-collision pairs, complete cross-arm pairs,
and all moving links checked against the fixed table and camera poles. Existing
proxy pairs remain enabled. Coarse link3/hand boxes are not reused for own-arm
self-collision because they intentionally overlap normal arm structure; real
arm meshes cover that check. The live teleop profile and its pair exclusions
remain unchanged. The planner derives conservative convex hulls from the same
CAD arm meshes, matching MuJoCo's mesh collision representation without changing
the URDF, assets, or minimum-clearance requirement. This avoids comparing
triangle-soup and convex-body geometry in replay.
This strengthens arm coverage but does not supply a real,
opening-dependent finger model or target-object contact validation.

MuJoCo admission queries are clipped at the required preventive clearance,
not a broad 50 mm diagnostic radius. On the installed 3.10 runtime, wide
`mj_geomDistance` queries can incorrectly report a negative distance between
separated convex meshes. Threshold-clipped queries preserve the same forbidden
pairs and clearance requirement, including hard-penetration recovery checks.
Reported far-away clearance is consequently a clipped lower bound, not an exact
physical-distance measurement; actual-contact checks remain unchanged.

The final grasp target remains strictly inside the configured envelope. If the
fresh measured TCP of the moving arm starts outside only an absolute translation
bound, the planner may construct a strictly-inside ingress waypoint without
widening that envelope. This route is accepted only when the initial violation
belongs exclusively to the moving arm, is at most 16 cm, monotonically approaches
the envelope, passes dense URDF/FCL collision checks, and never leaves again after
entry. Rotation violations, an out-of-envelope held arm, or any target outside
the envelope still fail closed.

起始测量 TCP 的包络准入检查现在先于多距离/多 IK 搜索进行。
`start_state_safety_envelope_blocked` 表示起始状态问题，不是候选目标被拒绝；
公开 diagnostics 保留投影对象、160 mm 上限及超出量。该修复不会扩大包络。

GraspGenX 保留原模型位置，不再对预测进行未经重新评分的 X/Y 居中修正。
GraspMoE 保留固定版本的插入深度，在 -4～0 cm 区间使用 5 mm 间隔，避免
跳过支撑面附近较窄的合法插入区间；新增候选由 discriminator 重新评分，
不复制旧分数。默认 `server_top_k=-1` 将完整已评分候选交给几何预筛，
不先按模型高分截取 1000 个；最终展示预算仍为最多 5 个。显式正数上限
仍可用于有意限制原始候选的调用，但可能丢失低分的合法方案。
候选预算先分配给接近方向和闭合轴/位置家族，余量保留同家族的已评分
深度/姿态变体；15 mm 家族聚类不再等同于删除，只去掉数值重复的姿态。
`candidate_family_count` 与 `candidate_variant_count` 分别记录家族及独立变体数。
端点 IK 仍只标注，
不改变候选顺序。可见碰撞预筛使用分离的掌部/两指实体，指间空间不填充；
要求零可见点侵入，并检查最大张开代理沿 +Z 接近 60 mm 的解析扫掠体，
终点也使用同一名义张开状态；候选的百分位宽度只是夹持宽度估计，不能拿它
作为非接触接近时的已测开度，也没有因此开放闭合动作。
目标点与背景点分别记录且均不默认放行。成对的保留候选规划还在路径搜索
和最终下发关节样本中检查夹爪代理，原有整臂 FCL、包络、时效和单次票据
保护不变。捕获时自体过滤先以同一 4B2 CAD URDF、双臂测量 qpos 和主相机
外参渲染机器人深度；因快照暂不包含手指关节状态，过滤器只在 URDF 合法范围
内取五个有界手指状态。背景像素必须同时满足距渲染轮廓不超过 12 px、深度差
不超过 15 mm 才归因于机器人，目标标签始终保留。随后仍以名义张开夹爪的
5 mm 解析表面带做第二层补充。两层过滤都只作用于捕获时点云，不会随规划位姿
动态删除点，不填充指间空间，也不删除接近扫掠区域。它们不用于路径碰撞查询，
不改变 FCL 几何或 20 mm 最小间距，并分别记录 URDF、容差、删除计数和语义哈希。
旧的绝对抓姿入口没有保留候选的 RGB-D 快照绑定，不提供这项
观测夹爪检查；本轮验证建议使用成对的只读规划入口。

实时 GraspGenX Route 增加可发现的 `point_grasp_rgbd_snapshot` 输出 Artifact，
保存该次原始 RGB、float32 米制深度、SAM2 掩码、内参/外参和相机附近
200 ms 内的双臂测量 qpos（相机独立模式显式记录无 qpos）。RPent 手动
入口将其下载到 run-root/input/point_grasp_rgbd_snapshot.npz。
推理之前另保留一次输入 Artifact，其路径与哈希写入 Gateway stderr，
即使没有有效候选也能重现输入。缺少或损坏完成态快照时，不准入可执行计划。
快照 capture 与 Gateway 冻结的保留结果先还原为相同 JSON 表示再比较哈希；
不改变全局类型哈希规则，也不放行内容、字节长度或 SHA256 不一致。
快照读取失败的具体类型和有界消息保留在 Gateway stderr；公开 v1 诊断字段不变。
路径规划使用快照同次采集的双臂 qpos；CAD 深度过滤在保留原始目标掩码后构造
固定可见场景，解析夹爪表面带再经下层 FK 补充过滤。最终轨迹诊断分别记录
`captured_robot_observation_filter`、
`excluded_captured_robot_background_points`、
`excluded_captured_gripper_background_points` 和名义开度依据。原有整臂、桌面/相机杆、可见夹爪碰撞
和最终下发轨迹检查保持开启。起点阻挡直接保留具体碰撞证据，不重复换随机种子。
仅相机快照不补造 qpos，不做机器人自身过滤。
GraspGenX 保留模型实际评分的抓姿朝向，不再以“腕部向上”旋转预测，也不将
交换两指的 180° 朝向去重。平行夹爪的局部实体对称不代表完整机器人对称；
真实 TCP 的偏心安装、手腕附件与 IK 分支可能使两种朝向的可行性不同。
相近位置、接近轴和有向闭合轴的重复候选仍去重，输出仍按原评分和有限显示预算
选择；每个保留朝向都须独立通过严格 IK、整臂碰撞与完整路径验证。不会为了
IK 临时生成未评分的旋转候选或沿用另一预测的分数。
抓取链先验证严格预抓取 IK 和连续分支接近段，再搜索包络入场和远场连接；连接
复用已验证局部方案的准确关节分支，不在入场后重新求另一套局部 IK。局部无解时
报告 `local_pregrasp_ik` 或 `local_approach_validation`，不先耗尽远场搜索预算。
包络入场搜索以该次实际起点固定外边界，提前排除超出初始越界距离与已有进度容差、
或扩大越界约束的样本；与密集复核复用投影及边界判定。BiRRT 查询没有路径顺序，
不以随机查询更新“已入场”或最佳进度；完整路径仍须通过单调入场与入场后不再越界
的原有密集复核。搜索到路径但复核拒绝时，日志保留具体复核详情，不只保留原因码。
整臂碰撞搜索先以自适应加密边界（当前 30 mm）查询保守距离下界，远离边界时
不逐对计算精确最小距离；边界附近仍计算所有碰撞对的精确距离，按原规则判定
20 mm 安全余量与是否加密。查询不可用直接拒绝，不用精确查询掩盖故障。
搜索缓存与普通精确缓存隔离，端点及最终密集轨迹复核仍使用原精确查询；不缩小
碰撞几何、不减少碰撞对、不降低采样分辨率，也不放宽包络或 IK 容差。
共享几何层可用 `for_observed_cloud()` 复用静态模型；每个冻结观测仍有独立的 FK、
碰撞查询数据、路径缓存和场景身份，共享 target builder 的规划调用串行执行。
观测体素覆盖全部占用格，顺序与内容哈希保持一致；未知区域不作为已验证自由空间。
捕获自体过滤只缓存不可变 CAD 网格，逐帧重算相机/FK/射线；既有 15 mm 深度门槛、
目标掩码保护、碰撞余量与最终轨迹复核保持不变。
同一抓姿的已验证近场 IK 分支组成全局连接目标集合；每轮共享起点树、目标森林、
碰撞查询缓存和节点/迭代/时间预算，而不是为每个终点重新搜索。搜索范围覆盖全部
目标后逐轮扩展，最后一轮仍为原关节硬限位；确定性种子为基础种子加轮次。返回
准确的目标序号和关节状态，再连接该分支原先验证的接近段。目标碰撞可剔除，
任一查询不可用或输入不合法都停止；密集轨迹复核仍保留原检查。
某条连接被密集复核拒绝时，下轮优先尝试其他分支，但不删除该目标或扩大总预算。
这是同一用户选定抓姿内的多 IK 目标，不会在 v1 Skill 中静默改选其他抓姿。
包络入场的多个目标按剩余总预算分配路由时间片，
该时间片同时覆盖入场、连接到已验证局部分支和组合路径的精确复核。一个路由
超时且总预算尚未耗尽时可尝试下一个路由；碰撞查询不可用仍停止。诊断分别保留
路由时间片、总预算耗尽与各路由失败详情，不把局部超时解释为物理不可达。
感知执行异常将有界类型/消息放入失败操作的 `progress_summary.failure_diagnostic`，
RPent 在重试提示之前显示该诊断；无候选时 stderr 另保留完整筛选漏斗。

这些检查的范围仍是可见点云与名义平行夹爪代理，不是未知空间自由证明，
也没有验证实际夹爪开度、完整闭合运动、携物退出或真实抓取成功。
`path_found` 不能当作端到端真机抓取验收；抓取/移动仍需独立 Skill 和后续验证。

Motion remains a separate decision. `motion.execute_grasp_approach_plan@1`
accepts only the advertised component ID and that exact plan ID/hash. It cannot
replace the target pose or trajectory, replans nothing, rechecks freshness and
start drift at activation, and consumes the plan at most once. Thus a generated
candidate, endpoint reachability annotation, and full-path result are three
different evidence levels; only an explicitly authorized execution request can
move the arm. A target outside the configured envelope remains a rejected path
assessment and is not a reason to widen the envelope automatically. Its operator
action is the distinct `execute_grasp_approach_plan`; authorizing the legacy
`approach_grasp_pose` action does not authorize this Route, and vice versa.

### 分阶段 Skill 边界

显式路径规划对调用者是可选的，但安全准入不是可选的：调用者可以先使用
`perception.plan_grasp_approach@1` 检查并选择计划，再调用
`motion.execute_grasp_approach_plan@1`；已有、经过审阅的绝对抓姿也可以交给
`motion.approach_grasp_pose@1`，由同一个下层规划器内部生成并验证 reach 轨迹。
两条执行路径均只到达抓姿，不闭合夹爪。

| 阶段 | 当前公开 Skill | 边界 |
| --- | --- | --- |
| 获取候选 | `perception.plan_point_grasp` | 仅感知提案；端点 IK 标注不是路径或抓取成功证据 |
| 渲染候选 | `perception.render_point_grasp_candidate` | 只读；不改变候选、姿态或运动状态 |
| 可选的显式 reach 规划 | `perception.plan_grasp_approach@1` | 只读；计划绑定会话、场景版本、起始 qpos 和时效 |
| 到达抓姿 | `motion.execute_grasp_approach_plan@1` / `motion.approach_grasp_pose@1` | 独立运动授权；不自动闭合夹爪 |
| 夹爪闭合 / 松开 | `manipulation.set_gripper_state@1`（显式启用、仅右夹爪） | 独立授权和请求；只证明开闭/停稳，不证明抓牢 |
| 空手移动 | `motion.move_tcp_to_pose@1` | 不能作为携物碰撞验证的替代品 |

当前 reach 准入验证真实机械臂连杆网格与部署的代理几何、固定桌面/相机杆和 TCP
包络。成对规划另验证可见目标/背景与名义张开夹爪的非接触接近，但没有将
观测障碍纳入整臂检查，也尚未验证随开度变化的真实夹爪实体、闭合过程、
带物退出、放置或物理夹持成功。因此 `path_found` 的含义是“reach
阶段在当前模型下通过”，不是“完整抓取链通过”；不能据此自动闭合、试抬
或搬运。后续这些阶段必须有独立的场景/夹爪状态和接触规则证据后才可开放。

#### 可选被动 UDP 反馈

现有真机 Gateway 可用 `--rm65-feedback-source udp --rm65-udp-port 8090` 显式接入
控制器已有的双臂关节与工具坐标系力反馈；默认仍为 SDK。该模式只接收数据，
不修改控制器推送配置，也不在 UDP 失败时切回 SDK 或与 SDK 回调争抢同一端口。
接收端在两臂 SDK 连接及启动 HOLD 前独占绑定端口，确认双臂反馈新鲜、无错误、
关节已使能且力坐标为 tool；端口占用或就绪失败直接拒绝启动。
接收端要求 Linux 64 位内核时间戳支持。测量序号只随新包推进，保留内核接收时间
（不是控制器采集时间）；重复读取不会刷新测量年龄，
已有状态/力反馈过期门禁仍负责拒绝派发与停机。
关闭时先确认测量采样与力监控停止，再释放两臂 backend，最后释放 UDP 接收端；
关闭失败保留尚被引用的资源并允许重试。SDK 尚未启动时的失败清理不会产生 HOLD。
应用及可选右夹爪包装层在尝试启动前登记资源所有权，部分启动失败仍由同一 owner 重试清理。
仅当控制运行时从未启动、入口未发布且 Gateway 从未接纳请求时，才允许无 HOLD 的构造资源清理；
已取得控制资源的正常关闭仍须确认原有下层 HOLD，线程关闭以实际完成凭据为准。

#### 显式夹爪 v2 合同基础

`manipulation.set_gripper_state@1` 及现有右夹爪接入保持默认行为。
可选 v2 必须显式构造双夹爪 deployment、绑定两侧独立 port，并在请求中同时给出
`component_id`（`left_gripper` / `right_gripper`）和 `target_state`。
定义、schema、Route、结果验证与执行器按同一版本闭合；v1 不接受该新增字段。
真机 CLI 默认仍为 `--gripper-skill-version 1`，沿用 `--right-gripper-config`。
显式选择 `--gripper-skill-version 2` 时必须同时提供两侧配置；
`--authorized-gripper-sides left|right [left|right]` 只限定夹爪授权，缺省为已授权臂的一侧，
不扩展手臂动作权限。程序化组合显式传入两侧 `gripper_backends` 时选择 v2，
不得与原 v1 的 `right_gripper_backend` 混用。两侧共享原应用资源 owner 和操作槽位。
MuJoCo 组合显式使用 `--enable-grippers --no-realtime-worker`，或 factory 的
`enable_grippers=True`；默认保持关闭。夹爪与手臂共用同一物理 owner、操作槽位和
域 HOLD；夹爪请求仍逐侧选择，手臂未 HOLD 时拒绝夹爪派发，原碰撞门禁保持。
两版复用原有收敛、取消与下层 HOLD 确认，任何一侧停机失败都不能宣称组合安全。

#### 可选右夹爪接入

`real-rm65 --right-gripper-config /absolute/right-gripper.json` 使用既有
`DMTacGripperBackendConfig` 字段配置独立的右夹爪 backend，配置必须显式提供
`side: "right"` 和 `gripper_port`，不是 USB assignment 文件。默认不启用触觉采集；
部署禁止启动张开、自动清故障、自动重连和触觉 reset。初始化可以设置已有的速度/力矩，
但不发位置命令；反馈表明夹爪正在运动、状态无效或资源启动失败时，不发布 endpoint。
机器人、相机、模型与夹爪均由 application 一次启动和关闭，不按 Skill 请求重建。

仅配置 backend 不授权开闭；操作者还须独立把 `set_gripper_state` 加入
`--authorized-actions`。同一 Gateway 的 `gateway.operation` 独占资源及 operation store
使机械臂和夹爪不能同时执行；不创建第二个 service、控制器或 admission 锁。
已有本机 client 支持 `--action set_gripper_state --side right --target-state open|closed`，
仍需逐请求确认，并消费精确绑定的 one-use Grant 和夹爪约束 profile。
调用方不能设置夹爪速度、力矩、设备地址或扩张运动权限。

下层确认 HOLD 后可以继续发布带真实 `hold/hold` 标签的新鲜观测，不能把它写成
`running/ready`。夹爪位置命令要求同一机械臂 owner 处于新鲜、无故障且已确认的 HOLD；
首次 startup HOLD 也由既有 lower owner 的时钟与控制周期推进。串口忙后的每次重试
也必须重新满足同一检查，不沿用首次下发前的结果。该路径不恢复机械臂。
每次夹爪位置写入尝试均进入 Station control fingerprint，旧冻结观测和路径计划会失效，
不能凭双臂 qpos 未变复用夹爪变化前的计划。读取夹爪状态不改变该 revision。

串口 STOP ACK 不是 HOLD 证明：必须用不超过 200 ms 的新鲜读取确认 `moving=false`，
才可报告夹爪已停止；读取超预算、反馈过期/未来、故障或不明确都不伪装成功。
全域 HOLD、emergency stop 和关闭同时覆盖机械臂及夹爪；停止不完整时保留失败，
不把机械臂的单独 HOLD 作为全域安全证明或继续拆除设备资源。
当前验证包含 fake backend/真实 Gateway wire、启动/互斥/独立授权/关闭失败测试，
以及使用既有独立 MuJoCo 夹爪 port 的双指物理接触测试。它不是硬件串口验收，
也不是同一真实硬件或同一仿真 endpoint 上的完整携物/放置验收。左夹爪不属于 v1，
不能用 `--side left` 偷换绑定。闭爪完成不证明手物关系，普通 TCP 路径仍不验证携物碰撞。
正式真机启用前仍须确认“闭爪后普通空手移动/靠近”的准入策略，并完成硬件验收；
当前接入不能作为已验证携物执行的入口。

The CLI is a concrete lower deployment leaf, not an Agent API. All files and
the selected side/action are trusted operator inputs. The process exposes no
transport until optional cameras have passed consecutive-frame freshness
admission, both RM65 backends have started, lower HOLD is confirmed, a fresh
measured dual-arm cut has seeded the coupled solver, and the Active pre-IK
envelope has been anchored. The exact confirmation authorizes only the selected
side and finite action allowlist for this process session. Each matching active
request still receives a newly minted short-lived, complete-request-bound,
one-use Grant and passes normal Gateway admission; passive capture neither needs
nor consumes a motion Grant. The retired `AUTHORIZE_ONE_REAL_TCP_PRIMITIVE`
token is rejected so an old one-shot confirmation cannot silently become
persistent authority.

The MuJoCo launcher starts no RPent planner and connects no real device. Its
socket parent must already exist, resolve to a real non-world-writable
directory, and contain no existing endpoint at the selected path.

The real CLI resolves an `ExclusiveRM65SkillDeploymentProfile`, constructs the
single lower graph through `assemble_exclusive_rm65_skill_lower`, and binds it
to the existing Skill/Gateway contracts through
`assemble_exclusive_rm65_agent_primitive_composition`. It reuses the shared
RM65 backend, coupled solver, force guard, target-envelope, per-request one-use
Grant, and Unix transport implementations without importing or constructing RealTeleop,
VIVE, collection, GUI, or `control_composition`. The application owns the sole
lower control thread and orders shutdown as Gateway fence -> future-cycle
fence -> lower-confirmed HOLD -> control-thread join -> resource close.

The current evidence is offline construction and lifecycle validation, not
real-hardware proof. `attach_real_agent_primitive_gateway` and the RealTeleop
`--agent-primitive-skill-only` option remain compatibility paths; the
independent launcher is the preferred path. `one_shot_client` is only an
authenticated one-request
Unix client and never initializes or dispatches hardware itself.

同一 Gateway 会话可反复启动 `one_shot_client`；客户端在 socket 旁的
`<socket>.client-sequences.json` 保存按 runtime session / client ID 分组的递增序号，
并用同目录的 `.lock` 文件串行化序号预留和请求提交。目录须允许当前操作者写入；
状态文件不属于源码或机器人配置，不要在当前会话中删除、改写或跨主机复制。
提交前原子保存序号；请求被拒绝或响应丢失也不回退，不自动重发可能已执行的动作。
锁等待受 `--timeout-sec` 限制，状态损坏或写入失败会阻止提交；终态轮询和运动执行
不持有此客户端锁。新 Gateway session 使用独立计数，旧会话的计数仍保留，避免并行旧
客户端覆盖新会话状态。Gateway 的防重放、单操作槽、授权和 HOLD 检查保持不变。

客户端可从 `running/ready` 或 `hold/hold` 提交已授权的新请求。HOLD 再入仍须由下层
取得新 Grant/activation，并验证前一次成功的 HOLD 回执；客户端不调用 resume，
故障或生命周期与执行门禁不一致时拒绝提交。该行为不依赖夹爪 Route 是否存在。

The absolute grasp-pose Route is authorized explicitly with
`--authorized-action move_tcp_to_pose`; publishing the Route without this
operator action does not mint a motion Grant. The matching local client accepts
only an actual tool-TCP pose in the Gateway-advertised `canonical_station`
frame. For example, after an independently authorized and admitted real Gateway
session is already RUNNING/READY:

```bash
python -m agent_gateway.profiles.real.one_shot_client \
  --socket-path /absolute/private/gateway.sock \
  --side right \
  --action move_tcp_to_pose \
  --target-xyz-m 0.31 -0.12 0.28 \
  --target-quat-xyzw 0 0 0 1 \
  --speed-scale 0.20 \
  --confirm EXECUTE_AUTHORIZED_REAL_TCP_PRIMITIVE
```

This is an Active real-motion request, not a probe or dry run. Do not run it
without the real-hardware authorization and site safety checks. The client
normalizes the quaternion, derives all frame/tool/calibration identities from
the same fresh observation and advertised schema, and rejects incomplete poses,
wrong components, stale identity disagreement, and speed scales outside the
reviewed profile. The request still passes the lower TCP envelope, coupled IK,
collision/jump/singularity gates, dispatch admission, convergence evaluation,
and terminal HOLD proof.

For compatibility, a caller that already owns and reviews a raw canonical-Station
tool pose may still authorize and call the combined `approach_grasp_pose` action:

```bash
python -m agent_gateway.profiles.real.one_shot_client \
  --socket-path /absolute/private/gateway.sock \
  --side right \
  --action approach_grasp_pose \
  --target-xyz-m 0.31 -0.12 0.28 \
  --target-quat-xyzw 0 0 0 1 \
  --speed-scale 0.20 \
  --confirm EXECUTE_AUTHORIZED_REAL_TCP_PRIMITIVE
```

`motion.approach_grasp_pose@1` resolves endpoint IK from the latest measured
dual-arm qpos, keeps the inactive arm fixed, plans a collision-aware global path
to pregrasp, and appends the same constrained Cartesian approach used by the
paired read-only planner. Every path is
densely rechecked with the production URDF/FCL collision model and the bound TCP
workspace envelope before the existing lower owner can dispatch it. The lower
owner atomically rechecks the trajectory start and inactive-arm hold at the
PENDING-to-ACTIVE edge. Active control interpolates adjacent certified qpos
samples by its own clock, never re-enters dynamic Cartesian IK, and cannot report
terminal success before the planned duration has elapsed. Missing/stale qpos,
start drift, unauthorized inactive-arm motion, unreachable targets, blocked
endpoints, unavailable collision checking, or exhausted path search return
distinct zero-motion failures. Activation and completion evidence retain the
immutable trajectory identity plus the latest measured/lower state and terminal
HOLD without accumulating an unbounded per-poll or per-frame reference list.
MuJoCo replay and terminal review are acceptance-test tools only; this
production Skill imports and starts neither. New point-grasp clients should use
the paired planning/execution Routes above so path feasibility can be inspected
without first entering a motion Skill.

A legacy grasp generator that consumes the real Station-aware Depth v2 output
may use the same compatibility action without manually converting camera
coordinates. It writes one
bounded UTF-8 JSON proposal and passes its absolute path instead of the direct
`--target-*` arguments:

```bash
python -m agent_gateway.profiles.real.one_shot_client \
  --socket-path /absolute/private/gateway.sock \
  --side right \
  --action approach_grasp_pose \
  --camera-grasp-proposal /absolute/path/grasp-proposal.json \
  --speed-scale 0.20 \
  --confirm EXECUTE_AUTHORIZED_REAL_TCP_PRIMITIVE
```

The proposal schema is `dexteleop.real.camera_relative_grasp_proposal.v1`. It
binds the grasp to the exact Gateway runtime session and Route manifest, source
operation, Station-aware Depth v2 Route/deployment, selected arm, Station
revision, Station frame specification, frame graph, camera pose, and typed
metric-depth Artifact plus its real-capture receipt. The grasp itself is the
actual DMTAC tool-TCP pose in that camera optical frame. An optional
`capture_group_id` may preserve an already-produced cross-modal capture
correlation; it is provenance only and does not weaken any identity check.
Unknown fields, stale identities, a different arm, or an Artifact that is not
the reviewed metric Depth output close the request before submission.
In-process grasp generators should construct it with
`CameraRelativeGraspProposal.from_depth_operation_wire(...)` and serialize
`to_mapping()` so Route, operation, result, and Artifact identities are copied
from the completed Depth operation instead of retyped.

The adapter computes exactly:

```text
T_canonical_station_actual_tool_tcp
  = T_canonical_station_camera_optical_frame
  @ T_camera_optical_frame_actual_tool_tcp
```

It then supplies the resulting canonical-Station pose to the
`motion.approach_grasp_pose@1` request. The proposal is not a Grant, does not make
the perception operation authoritative for motion, and is not forwarded as a
motion input Artifact. The same explicit session authorization, one-use Grant,
fresh RUNNING/READY snapshot, safety envelope, IK/collision admission,
dispatch, convergence, and lower-confirmed HOLD still apply. Direct
`--target-xyz-m`/`--target-quat-xyzw` input remains unchanged and is mutually
exclusive with `--camera-grasp-proposal`.

显式 `dual_arm_hand_proxy_dmtac_grasp` 将抓取专用桌面碰撞位置对齐已登记 CAD，
普通 strict 场景及坐标图保持不变；离线 grasp replay 默认选择该抓取场景，旧记录
可显式指定 strict 对照。绑定腕相机安装轴的 GraspGenX reachability 会筛除侧向／
倾斜接近中相机朝下的候选，普通候选也受此约束；竖直接近保留原自由度。
详细场景身份和使用方式见 `docs/agent_gateway/POINT_GRASP.md`。

The MuJoCo arbitrary-grasp viewer and the real Skill deliberately do not accept
the same numeric input frame. The viewer's explicit input is
`selected_arm_base_from_actual_dmtac_tool_tcp`, while the real Skill public
input is `canonical_station_from_actual_dmtac_tool_tcp`. The physical pose
conversion is:

```text
T_canonical_station_tool_tcp
  = inverse(T_solver_world_canonical_station)
  @ T_solver_world_selected_arm_base
  @ T_selected_arm_base_tool_tcp
```

Both transforms must come from the exact selected collision-scene URDF/profile;
never substitute the sim world origin, copy a numeric arm-base pose into the
real request, or hard-code the DMTAC flange/TCP offset. A camera-frame grasp
proposal must first use the admitted perception result's
`T_canonical_station_camera_optical_frame` and the same pinned Station revision
to become a canonical-Station tool-TCP target. The lower target owner then does
the one and only tool-TCP-to-solver-endpoint conversion described below.

## Skill and execution boundary

The default-off MuJoCo profile publishes one lower motion domain with separate
versioned Skills for absolute TCP targets, semantic cardinal steps, and fixed
level orientations.
`motion.move_tcp_to_pose@1` accepts an exact lower target;
`motion.step_tcp_cardinal@1` accepts `forward`, `backward`, `left`, `right`,
`up`, or `down`. The cardinal target is resolved from measured TCP state only
when the lower Skill is activated. Distance, frame/tool/calibration identities,
primitive hash, safety envelope, IK, dispatch, evaluation, and HOLD remain
trusted deployment facts.

MuJoCo 的现有入口可显式增加 `--enable-safe-tcp-pose`，为会话选定的一臂发布
`motion.move_tcp_to_pose_safe@1`；默认不发布。该 Route 使用当前场景、另一臂的
实测起点和已绑定工具配置，复用下层 RM65 IK、BiRRT-Connect、C2 时间参数化
及原有唯一执行链。逐参考复核继续使用同一已接纳 TCP 包络，绑定改变时拒绝执行。
拐点显式零速度、零加速度，并对齐参考时钟；实测到达拐点后才进入下一段，避免
轨迹平滑或跨采样插值切角。普通 move-to 与既有轨迹默认行为保持不变。
`--safe-tcp-planning-clearance-padding-m` 默认额外预留 0.002 m，执行准入继续
使用原场景阈值。free-space move 禁止夹爪与物体接触，未找到路径时保留拒绝，
不替换为普通 move-to。相机未建模障碍、未建模负载、抓取接近及携物不属于该
Route 的能力；这项接入也不发布双夹爪 v2。完成、取消和关闭仍需下层 HOLD 证据。
模拟验证与真机验收分别记录。

MuJoCo 入口可用 `--motion-side left|right` 选择这一会话唯一可驱动的机械臂，
默认仍为 `right`。factory 对应参数为 `motion_side`；自行构造 target config 的调用方
还须让 `exposed_pose_sides` 与 `mujoco_agent_tcp_pose_execution_observation_plan_v1`
的 `motion_side` 一致。双臂／`both` 请求不会自动拆分，也不支持在同一会话切换控制侧。
左右侧使用各自工具标定、arm-base 与公开 Station 转换、状态 schema 和包络身份；
另一臂仅作为同一 rig 中的实测 support。安全规划与执行复用原 owner、碰撞 oracle、
硬限制、完整路径复验和 HOLD；原有 support 关节漂移检查保持 0.001 rad，不代表另一臂
在物理仿真中绝对零位移。默认右臂的公开合同不变，不改变真机入口。

显式左臂组合目前支持基础 TCP/state、可选 safe TCP、可选 TCP comparison v2，
以及独立启用的只读 RGB-D Artifact capture；尚未接通左臂的 gripper、grasp、spatial goal、
history 或 segmentation 组合会在创建场景／资源前拒绝，不发布只改标签的伪左臂能力。
这些组合仍可按默认右臂入口使用。该配置增加的是已有 Skill 的部署覆盖，不新增日常工具名。

MuJoCo 可显式组合 `--enable-grasp-approach --enable-grippers --no-realtime-worker`，
发布 `motion.approach_grasp_pose@1`。它复用同一 planned-motion、operation slot
及机械臂／双手联合 HOLD，使用独立且只绑定自身 Route 的绝对 TCP 包络；
原 safe move 的增量限值和禁止接触语义不变。公开 assembler 同样要求端口属于
当前 rig、具有构造时声明的 arm-HOLD 约束，且实际 rig 使用确定性物理时钟。
该 Route 只执行接近，不自动闭合夹爪、抬升或证明抓取成功；默认不发布。

同一显式接近组合可另设 `--grasp-contact-target-body`，发布独立的
`motion.approach_grasp_contact@1`。末端接触只绑定任务配置中右手传感器与指定
目标物体的几何对，并保留原硬穿透上限；其他几何对继续使用原预防间隙。
该权限由部署 profile hash 固定，调用者不能另传接触例外。结束或取消时撤销
临时 guard，成功仍需在确认 HOLD 后复核实测接触及工具轴接近走廊。

组合内的候选选择器只在冻结场景上预览接近、近端和末端路径，不激活计划或
推动物理；结果仍需重新提交精确 motion 请求。共享下层 planner 同时增加紧凑
分支选择和路径质量检查，因此普通 safe TCP 的排序／拒绝结果也可能变化，
其默认碰撞政策不变。以上能力均不自动闭合、携物或证明物理抓取成功。

该显式末端接触组合还可用 `--enable-object-state` 发布只读
`state.read_object_state@1`，默认不发布，且必须已绑定 `--grasp-contact-target-body`。
查询只接受目录中绑定的物体、公开 Station 坐标系及 1–50 ms 测量帧龄上限。
当前 MuJoCo 实现从同一 private `MjData` cut 计算 body-origin pose、线/角速度
及目标与支撑、触觉接触面、其他物体的接触；不推进物理、不发送动作或 HOLD，
不读取命令目标冒充测量。结果携带 frame/spec、source binding 和会话物理时钟身份。
速度通过同一 body-origin Jacobian 与 qvel 计算，再用公共 FrameGraph 转换，
不把质心速度或世界坐标直接标成公开 Station。

`measured_at`、`cutoff_at` 属于返回的 native physics clock；暂停仿真时重复读取
同一时刻不会产生新物理证据。这是当前状态反馈，不是连续释放判定：当前适配器固定
返回 `release_observation=null`，不可由一次静止、张爪或动作完成推断放置稳定。
通用结果允许未来显式部署的连续观测 producer 提供窗口，但稳定声明仍须具有完整
窗口起点、实测最大采样缺口和固定稳定策略；缺失测量必须返回 null。
本入口不提供真机物体位姿估计源，也不表示完整抓取、搬运或放置验收已通过。

MuJoCo 的 ``state.read_tcp_pose`` 与可选 TCP history 发布同一公开 Station 坐标系。
其参数及结果校验从公开 state binding 获取坐标身份，不使用内部 IK execution frame；
Agent 必须按当前 schema 查询和复制身份，不能以 arm-base 查询替代 Station 查询。

Target-independent motion semantics and payload schemas live in
`skills/motion`. They also define fixed level-orientation Skills for forward,
forward-left 30 degrees, and forward-right 30 degrees. Together, the cardinal
Skill's six enum choices and the three fixed-orientation Skill identities expose
nine bounded lower primitives. The fixed-single-arm no-model launcher
publishes all nine through exact MuJoCo Routes. Cardinal directions and absolute
headings are resolved in the canonical Station frame, then transformed into the
selected-arm execution frame; the arm-base frame is not relabelled as canonical
desk axes. Fixed orientations use their own route-scoped safety envelope because
they preserve translation but may require a larger angular correction than a
cardinal step or generic small pose move. Publication still grants no motion by
itself: activation preflight, lower safety, IK, dispatch, evaluation, and HOLD
remain mandatory.

The real profile exposes the same nine primitives through four exact
`t2_real_motion` Routes: one cardinal Route with six bounded direction values
and three fixed-orientation Routes. One Gateway process may serve sequential
requests until shutdown, but every request requires its own one-use operator
Grant bound to the complete pre-grant request. Activation resolves the target
from a fresh lower-measured TCP pose, and completion requires distinct fresh
measurement cuts, at least one lower dispatch, the configured stable-arrival
count, and terminal lower-confirmed HOLD. Cancellation, disconnect, timeout,
fault, and Gateway close all use that same lower HOLD boundary.

After a successful terminal HOLD, the exclusive lower runtime remains in HOLD
while idle. It returns to RUNNING only when a separately admitted operation has
published a new pending activation; that transition clears the previous HOLD
receipt so the new operation must produce its own terminal evidence. A failed
HOLD leaves the runtime faulted and cannot be used as a session re-entry point.

The real Gateway does not treat the existing foot switches as an Agent
deadman, and it cannot manufacture owner epochs, frame/calibration identities,
or safety-envelope facts. Those facts come from the selected lower owner and
its Active final pre-IK TCP envelope. Ordinary RealTeleop invocations do not
advertise or start the Gateway, and the repository's offline tests are not
real-hardware validation.

### Real Skill target-frame invariant

The real Skill API is TCP-based, but the RM65 coupled solver is endpoint-frame
based. `RealAgentMotionActivation.target_pose_world`, measured cuts, and the
source value recorded in `BimanualPoseTargetSource` are tool-TCP poses in the
selected execution/world frame. In contrast,
`BimanualPoseTarget.target_pose_by_side` is consumed as a solver-world endpoint
target. Current DMTAC tool profiles commonly introduce roughly `0.14m` of
translation between those frames; the selected profile, not this number, is
the authority.

The only valid path is:

```text
Skill TCP target
  -> lower target owner's solver_world_target_from_tool_tcp_world_target()
  -> BimanualPoseTarget solver endpoint
  -> TCP envelope / IK / dispatch
```

Never pass a TCP pose directly to `BimanualPoseTarget` or duplicate the tool
transform in Gateway/Agent code. A target owner without the conversion method
must be rejected, and tests for this path must use a non-zero offset so an
identity fake cannot hide a frame regression. The canonical action/frame
background is maintained in [`teleop_runtime/README.md`](../teleop_runtime/README.md#action-contract-说明).

Target-independent CubeLift semantics live in `skills/cube_lift`. The MuJoCo
Route, deployment, lifecycle executor, observation binding, evidence, and
terminal evaluator live in `profiles/mujoco/skills/cube_lift`. This keeps task
meaning independent from a simulator while keeping execution policy in the
concrete Gateway profile.

## Observation and visual cuts

Low-rate state observations and pull-based visual captures are separate. RGB is
not a dedicated transport method: `perception.capture_rgb` is an ordinary
`inspection` Skill with a `read_only`, `t0_read_only` Route and generic output
Artifacts. Its deployment profile publishes the trusted semantic roles; the
MuJoCo binding maps `main`, `wrist_left`, and `wrist_right` to current cameras,
renders one scene cut under the existing scene/physics lock, assigns the same
acquisition time to every selected frame, and rejects a cut if its Station
revision changes. Other profiles may bind their own producer to the same
Definition/schema/I/O contract without importing the MuJoCo renderer. The real
binding maps the same roles to an exclusive RealSense inventory, retains each
source timestamp, rejects stale, cross-generation, serial-mismatched,
over-skewed, or revision-racing frames, and commits encoded output through the
generic Artifact lifecycle. RGB intentionally carries no camera pose; geometry
consumers use the Station-aware Depth v2 result. Device lifecycle remains
outside the passive executor.

## Compatibility

Existing imports under `control_station.agent_gateway`,
`control_station.sim_hil_profile.agent`, `control_station.tasks.cube_lift`, and
`control_station.sim_hil_profile.tasks.cube_lift`, plus the historic real
observation Gateway module, are thin re-export shims.
`agent_gateway.protocol.motion_schemas` is also a temporary compatibility
re-export; new motion-Skill code imports `agent_gateway.skills.motion`.
New code must import the canonical `agent_gateway.*` modules. Compatibility
modules may be removed only after production, test, documentation, and external
consumer audits establish that their callers have migrated.

Some immutable Skill/Route identifiers and hash-domain strings still contain
the historical `control-station` spelling. They are protocol identities, not
current Python ownership claims, and remain unchanged so the structural move
does not invalidate manifests, schema snapshots, idempotency records, or
external reconciliation.

### 可选关系目标计算（MuJoCo / Realman 共用）

两端 Gateway 均可加 `--enable-spatial-goals` 发布只读的
`perception.compute_spatial_goal@1`，默认关闭，不改变原有运动 Skill 的参数或执行逻辑。
计算由 `robot_motion.spatial_goals` 统一实现；不启动相机、模型、机械臂或第二个控制循环。
可用的 `component_id` 与 `reference_frame_id` 来自当前部署绑定；现有单臂配置继续保持单臂合同，
此计算只使用该部署已公开的组件和坐标身份。
上层可以在发现此 Route 后选择暴露；缺少它时保留原有接线。外部客户端是否暴露此可选能力
需要由该客户端另行验证，本仓库的测试不证明 RPent 接线已经更新。

参数示例（所有坐标、姿态必须先转换到同一个公开参考系；不要混用相机光学坐标）：

```json
{
  "component_id": "right_arm",
  "reference_frame_id": "canonical_station",
  "current_tcp_pose_xyz_quat_xyzw": [0.2, -0.1, 0.3, 0, 0, 0, 1],
  "anchor_xyz_m": [0.2, -0.1, 0.3],
  "offset_xyz_m": [0, 0, 0.03],
  "offset_frame": "reference",
  "orientation_mode": "preserve",
  "tool_axis": null,
  "axis_reference_xyz": null
}
```

`anchor_xyz_m` 是调用方传入的几何锚点，例如来自同帧分割/深度测量的物体中心，
或来自 `state.read_tcp_pose` 的 TCP 位置。输出明确标为 `caller_supplied_geometry`，
不是新的物体观测；不得用模型臆测坐标替代实测证据。`offset_frame=reference` 表示公开参考系轴，
`tool` 表示输入 TCP 的自身轴，均为米，且偏移使用调整姿态之前的自身轴。
`orientation_mode=preserve` 保持输入姿态；`align_axis` 需指定 `tool_axis=x/y/z`
与公开参考系中的非零 `axis_reference_xyz`，只约束这一个轴并选择相对输入姿态转角最小的解。
这避免为了满足一个轴的目标而人为固定多余的滚转，但不保证该姿态的 IK 或关节路径最短。

输出的 `pose_targets_by_component` 可供现有避障 Skill 构造目标；公开坐标、工具和标定身份
仍须从部署绑定读取，不由本计算生成。它不放行运动、不评估可达性或碰撞，
`path_checked=false`、`motion_executed=false`、`constraint_scope=endpoint_only`。
完整路径仍由现有 `motion.move_tcp_to_pose_safe` 重新取得当前状态、检查场景并规划，
失败不回退普通 move-to。沿整个路径保持水平、绕开指定区域等持续约束尚不由该 Skill 实现，
也不能把终点轴对齐解释为全过程满足该约束。

### 可选实测 TCP 与历史动作目标比较（显式 v2）

两端 Gateway 的 `--enable-tcp-pose-comparison` 在保留 `state.read_tcp_pose@1` 的同时，
额外发布同名 `state.read_tcp_pose@2`，默认关闭。上层通过现有精确 Route 暴露策略选择一个版本，
不应把两个版本当成两个日常工具，也不能缺少 v2 时悄悄回退到 v1。

v2 保留 v1 的四个参数，并要求显式提供 `target_operation_id`：

```json
{
  "component_ids": ["right_arm"],
  "reference_frame_id": "canonical_station",
  "controlled_frame_id": "tool_tcp",
  "max_measurement_age_ms": 200,
  "target_operation_id": null
}
```

`null` 仅读取当前状态；非空值必须引用同一客户端、会话、manifest 中尚被保留的终态
`motion.move_tcp_to_pose_safe@1/@2` **绝对目标请求**。当前部署没有发布对应运动版本时不能引用它。
无法找到记录、记录未终止、参数/身份不匹配时明确失败，不猜测“最近一次动作”，也不接受调用方
重新提供的目标位姿。具体组件、参考系与最大帧龄仍来自当前部署，示例不能跨部署直接套用。

返回原有新鲜单 cut 状态，加 `target_comparison`；其为 `null` 或包含原请求/Route hash、
原 terminal 状态与 execution/task outcome、原目标、各目标组件的平移误差（米）与完整 SO(3)
旋转误差（弧度）。比较覆盖原目标的全部组件，允许一次读取双臂而只比较原来有目标的那一臂。
它复用 `robot_motion` 的位姿转换和距离计算，不引入第二个 FK、采样器、运动 owner 或容差。

本比较不新增历史时间关系声明，不重写样本时间，也不替代原有新鲜度门禁。
误差描述的是**现在的实测值与历史指令目标**，不是历史
执行的终态测量，更不证明物体效果、稳定性、HOLD 或任务成功。即使当前误差为零，原失败、取消或
超时 outcome 仍原样返回。Agent 根据任务给定的容差解释误差，不由本 Skill 新增成功裁决。

### 真机 TCP 预规划的统一接线

`motion.move_tcp_to_pose_safe@1` 需要单独授权 `move_tcp_to_pose_safe`；普通
`move_tcp_to_pose` 的授权不能调用它。该路由通过现有 lower planner 先完成固定场景
碰撞、路径质量、目标包络和最终轨迹检查，再由原控制 owner 执行，不回退到在线 IK。
safe planner 的到位容差绑定当前 convergence profile，不改抓取规划的原有配置。

该 Skill 在 real / MuJoCo 上共用可选请求字段 `pose_goal_policy`：

- 省略或 `within_arrival_tolerance`：保留 v1 原有到达区域规划语义；真机可在部署的
  10 mm / 5° 到达范围内尝试 clearance 平移／旋转救援，小步仍优先局部延续。
- `preserve_target`：固定调用方请求的 TCP 终点，禁止局部及多分支终点搜索主动改写
  平移或姿态目标。沿用原 IK、局部延续、全局关节路径搜索和安全／HOLD owner；
  规划终点按既有 lower 的 3 mm / 1° 检查，部署已有更严容差时保留更严值。
  MuJoCo 用同样上限审核其既有目标 IK。无满足条件的安全路径就拒绝。

此字段只在 `motion.move_tcp_to_pose_safe@1` 接受，非法枚举拒绝；普通 move-to、抓取靠近
和各自原参数含义不变。它选择终点语义，不增加全过程姿态约束或携物碰撞能力。
例如“仅 Z +3 cm、终点姿态保持”可用原有绝对目标加 `"pose_goal_policy": "preserve_target"`。

规划精度与执行到达是两层事实：执行仍按各端既有 arrival 容差、连续实测、完整路径条件和
下层 HOLD 判定；`preserve_target` 不承诺零误差，也不把到达容差称为精确收敛。
成功结果增加可选 `pose_goal_report`，包含 `pose_goal_policy`、`reference_frame_id`、
`requested_pose_xyz_quat_xyzw`、`achieved_pose_xyz_quat_xyzw`、`translation_error_m`、
`rotation_error_rad` 及 `arrival_translation_tolerance_m` / `arrival_rotation_tolerance_rad`。
位姿使用请求的公开参考系；achieved 是到达判定使用的实测位姿，不能当作规划目标。
真机轨迹记录另在 `parameterization.pose_goal` 保存 requested / planned 矩阵、残差及实际规划
容差。新增可选字段会自动更新 definition / schema / Route 等哈希，消费者须重新发现目录。

在已装配抓取 planner 的部署中，普通 `motion.move_tcp_to_pose@1` 也可复用它的
`plan_tcp_pose` 做预规划。固定朝向的平移保留笛卡尔直线，调用共用的求解、连续 C2
时序、硬限制、密集复验和记录路径；它没有携带物碰撞认证或目标接触豁免。
未绑定此可选 planner 的既有普通入口保留原行为。

规划时使用 lower 的 confirmed-HOLD 读取窗口；窗口退出后最多等待 0.5 秒、
且不超过原 operation deadline，取得更新且仍新鲜的 measured qpos 后才可激活。
未来时间、过期、不可用、起始关节漂移、取消及剩余执行时间不足仍按原门禁拒绝并 HOLD。
所有已装配的预规划轨迹都必须执行到原时长结束；即使测量提前进入到位容差，也不能提前判定完成并 HOLD。
lower 只有取得停止命令的 worker／SDK 确认后，才可返回成功 HOLD receipt，或暂停读取窗口内的采样及陈旧测量看门狗；失败和不可读状态仍会拒绝。该确认表示冻结命令已被 SDK 接纳，不是物理静止测量。
本批只接入固定场景规划；RGB-D 障碍场景与会话几何准备另按依赖接入。


观测场景安全移动可显式配置 safe_motion_observation_config，复用已有 RGB-D
provider、相机几何与唯一 lower owner；不需要定位模型。它先冻结同一批静止视图，
再进入已确认 HOLD 的只读窗口，按原深度自体过滤与碰撞规则生成每请求独立的障碍云。
静态规划器按会话复用，请求的 FK／碰撞数据和路径缓存保持隔离；没有目标接触豁免。

planned_trajectory_dir/observations 留存每次尝试的 snapshot.json、原始 RGB-D、
相机/FK 绑定、障碍与 result.json；可选调试记录使用同一批视图，不再采集。
path_found 只证明找到规划路径，execution_verified 仍为 false。采集、场景准备、
规划、记录和请求锁等待都计入剩余操作预算；最后一项落盘后仍检查视图有效期与预算。


#### 目标策略的目录迁移与测量口径

本次保留 `motion.move_tcp_to_pose_safe@1`：新增策略是可选项，省略时维持原规划和执行语义。
参数、结果、definition、I/O 合同、Route 及部署绑定的身份会按内容更新；这是显式的目录迁移。
客户端应重新发现目录并使用同一批目录中的参数 profile、Route、绑定及授权信息，不能复用旧哈希。
旧的严格参数验证器会拒绝新增策略，旧的严格结果验证器会拒绝新增报告；新验证器仍接受旧格式请求与结果。
普通 move-to 与抓取靠近的目录合同不接受这两个新增字段。

real 报告取自触发连续到达判定的不可变测量 cut，与 `final_measured_state_hash` 相同；
随后仍要求下层确认 HOLD，报告并不声称它是 HOLD 后的再次测量。
MuJoCo 报告取自原有 HOLD 后复核的最终样本，与其最终状态哈希使用相同公开位姿。
两端位姿都使用公开参考系，残差从报告发布的 requested / achieved 位姿经下层规范化位姿数学计算，
旋转单位为弧度；不以规划目标或预测位姿代替实测，也不更改既有到达判定。
MuJoCo 原有 already-satisfied 路径仍可在执行 arrival 容差内直接确认 HOLD；
严格策略不会为了达到规划数值上限而强制制造运动，也不保证实测达到 3 mm / 1°。

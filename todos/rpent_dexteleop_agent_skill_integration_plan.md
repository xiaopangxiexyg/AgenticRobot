# RPent ↔ DexTeleop Agent Skill 接入与真机分阶段落地规划

状态：`方向已确认；B1 纯 contract/schema snapshot 已完成；B2 未开始；无 sim/real 启动或运动授权`

更新日期：2026-08-08

关联事实源：

- 目录与 ownership：[`../AGENTS.md`](../AGENTS.md)
- 控制主线与 M5.8/M6：[`dexteleop_control_core_landing_plan.md`](dexteleop_control_core_landing_plan.md)
- Control Station：[`dexteleop_control_station_plan.md`](dexteleop_control_station_plan.md)
- RTC：[`dexteleop_rtc_landing_plan.md`](dexteleop_rtc_landing_plan.md)
- RPent 当前上层框架：`/home/zc/HandMotion/Developing/HierachicalVLA/RPent`

本文是仍需实施的跨仓计划，不是当前代码事实。本文中的接口名在 contract 批次落地前仍是
拟议名称；每批完成后必须把稳定公共契约迁入对应 package README/schema，并从本计划删除
已经失去规划意义的流水内容。

---

## 1. 已确认的方向

以下方向已经确认，后续实现不得悄悄改变：

1. RPent 是 skill-level orchestrator：观察环境、选择离散技能、提交参数、等待结果并继续规划；
   它不生产连续关节/笛卡尔控制流，不拥有 action chunk、reference、IK、安全或 dispatch。
2. 首先闭合 fake/null 与 authoritative sim；真机依次经过 observation-only、dry-run/passive、
   shadow 和独立授权的 Active gate。sim/shadow 证据不自动授权真机 Active。
3. Home/Rearm 可以成为 Agent 可调用能力，但只能作为受控、分型的 privileged lifecycle skill；
   Agent 可以提出和跟踪操作，不能取得 operator lease、motion owner token 或底层 permit。
4. 对外禁止含糊的通用 `home`/`rearm`。第一版拆成四种独立能力：
   `capture_runtime_home_v1`、`move_to_registered_home_v1`、
   `rearm_safety_latch_v1`、`resume_after_hold_v1`。
5. 第一批真机能力限制为 read-only/observation 与无运动的
   `capture_runtime_home_v1`；物理 Home 必须等待 M5.8 lifecycle permit 在 production call seam
   闭合，普通 policy skill 的真机 authority 必须继续等待 M6。

---

## 2. 当前代码事实与阻塞边界

### 2.1 RPent

当前主控制流是：

```text
Runner / EnvSpec
        │
        ▼
PromptBundle → Planner → Toolkit → primitive/client → external service
```

当前 `robots/libero` 将环境当作可 `reset/step` 的 authoritative service，并允许同解释器自动启动
服务或连接外部 endpoint。DexTeleop 不应照搬以下语义：

- RPent 不得自动启动真机设备、Control Station 或 RM65 backend；
- 不得把 DexTeleop 包装成允许任意 action 的 Gym `step()`；
- 不得让 planner 直接持有实时状态、队列、控制线程或硬件句柄；
- 不得因 endpoint 连通就推断 target READY 或 motion authorized。

当前 `rpent.tools.Toolkit` 总是注册文件/IO common tools；Claude Code planner 默认允许
`Bash Read Write Glob Grep`，Codex planner 当前使用 `full_access` sandbox。真机 profile 在这些
默认值未被隔离前，不能把 Gateway 凭据、operator grant 或控制主机文件暴露给 planner。

### 2.2 DexTeleop

当前真实运动 owner 仍是 legacy GUI/deploy/replay 路径；shared runtime 的 authoritative 路径只在
scoped MuJoCo 闭合。Control Station 的真实 frontend 仍是 passive/default-off seam，不能因新增 Agent
RPC 就改变 owner、queue pop 或 dispatch。

`control_station.contracts` 虽公开列出 `home`/`rearm` intent kind，当前
`control_station.state_machine._IMPLEMENTED_INTENTS` 并未实现它们；普通 mutable intent 还要求
`OperatorControlLease`。因此不能通过简单 widening Web allowlist 把 RPent 当作操作员。

当前真实 Home/Rearm owner 是 legacy policy qpos-init 与 position executor manual-rearm：

- Home 的 logical target admission 与每个真实 SDK target 需要分开的 lifecycle admission/permit；
- 每个实际 Home SDK call 必须使用一次性 `LIFECYCLE_CONTROL` permit；
- rearm 只允许在 fresh measured state 和明确 fault class 下推进状态；
- worker/runtime fault 可能要求 restart + re-home，不能被通用 rearm 清除；
- M5.8 production lifecycle state machine 尚未拥有这些 exact permit；
- M6 前 shared runtime 仍没有真机 motion owner token。

---

## 3. 目标架构与 ownership

```text
┌──────────────────────────── RPent host / planner trust zone ────────────────────────────┐
│                                                                                         │
│  PromptBundle → Planner → DexTeleopToolkit → unprivileged client                        │
│                         │                                                               │
│                         │ only exact manifest-derived tools                             │
│                         │ no raw action / no hardware credential                        │
└─────────────────────────┼───────────────────────────────────────────────────────────────┘
                          │ versioned SkillRequest / Status / Cancel
                          ▼
┌──────────────────── DexTeleop application / control host ───────────────────────────────┐
│                                                                                         │
│  credential sidecar or narrow client proxy                                              │
│             │                                                                           │
│             ▼                                                                           │
│  Control Station Agent Gateway                                                          │
│    - client identity and access policy                                                  │
│    - capability manifest                                                                │
│    - request/result CAS, idempotency and bounded operation slots                         │
│    - operator confirmation and SkillExecutionGrant                                      │
│    - publication barrier, status and receipt projection                                 │
│             │ typed lower-owned operation port                                          │
│             ├─────────────────────────────┬──────────────────────────────────────┐       │
│             ▼                             ▼                                      │       │
│  sim/mujoco/deploy capability      teleop_runtime/deploy capability              │       │
│             │                             │                                      │       │
│             └───────────────┬─────────────┘                                      │       │
│                             ▼                                                    │       │
│                   robot_motion                                                   │       │
│       lifecycle / authority / safety / planner / reference / permit              │       │
│                             │                                                    │       │
│                             ▼                                                    │       │
│                    sim backend or hardware backend                               │       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 各层负责内容

| 层 | 负责 | 明确不负责 |
|---|---|---|
| RPent Planner | 任务分解、技能选择、参数形成、根据结果重规划 | 连续动作、设备生命周期、安全判断、permit、dispatch |
| DexTeleopToolkit | manifest 驱动的工具暴露、请求构造、结果呈现 | 自动启动设备、保存高权限凭据、隐式重试运动 |
| Agent Gateway | 认证、准入、幂等、操作状态、operator grant、应用层编排与状态投影 | 技能算法、action queue、IK、安全、motion owner、SDK 调用 |
| lower capability port | 把一个已准入 skill 映射到现有 sim/real runtime | 绕过 Station barrier 或创建第二控制循环 |
| robot_motion | 通用生命周期、动作、安全、规划、reference、owner/permit 合同 | vendor 初始化、Agent、GUI、模型推理 |
| teleop_runtime | 真机设备和 deployment lifecycle、当前 legacy caller | 通用 Agent 协议或跨 backend 技能算法 |
| sim | 仿真实现和可重复故障注入 | 真机包或真机设备 |

### 3.2 强制不变量

1. 每个 control domain 同一时刻只有一个 dispatch owner。
2. Agent grant 不是 motion/lifecycle permit，也不改变 authority epoch。
3. Agent Gateway 不持有 action chunk、reference segment、replay cursor、policy worker 或 episode owner。
4. 接受请求不等于 lower operation 成功，更不等于 command 已 dispatch。
5. 所有结果必须引用实际 lower receipt；没有 receipt 时状态只能是 pending/failed/unknown，不能伪造成功。
6. Agent 连接丢失、deadline 到期、operator deadman 丢失或 Station revision 漂移时，真机 profile 关闭
   publication 并请求 lower HOLD；不能继续后台执行旧操作。
7. HOLD、cancel 和 status 路径不得被长时间技能执行线程阻塞。
8. 第一版每个 control domain 最多一个 active skill operation；busy 时明确拒绝，不做隐式 Agent 队列。

---

## 4. 跨仓协议 ownership

### 4.1 Canonical source

协议 canonical source 位于 DexTeleop。第一批以纯 Python immutable contract 加可生成 JSON Schema 的方式
落地；RPent 保存生成的 schema snapshot 与 semantic hash，不直接 import DexTeleop Python 包。

原因：

- 两仓可能位于不同 Python/conda 环境或不同主机；
- 真机侧必须是 capability 与准入语义的事实源；
- consumer snapshot 可以在启动时做 protocol/hash compatibility preflight；
- 避免两个仓各自修改同名 dataclass 后静默漂移。

建议 hash domain：

```text
dexteleop.agent_skill.protocol.v1
dexteleop.agent_skill.capability_manifest.v1
dexteleop.agent_skill.request.v1
dexteleop.agent_skill.execution_grant.v1
dexteleop.agent_skill.operation.v1
dexteleop.agent_skill.observation_snapshot.v1
```

### 4.2 CapabilityManifest

每次 Gateway runtime 启动生成 immutable manifest，至少包含：

```text
protocol_version
runtime_session_id
control_domain_id
station_manifest_hash
execution_manifest_hash
target_binding_id / target_kind
current_mode
supported_transport_features
skill_capabilities[]
explicit_exclusions[]
manifest_hash
```

每个 `SkillCapability` 至少包含：

```text
skill_id / skill_version
parameter_schema_hash / result_schema_hash
privilege_tier
supported_modes
may_move_robot
requires_operator_confirmation
requires_deadman
required_lower_permit_kind
max_operation_duration_ms
cancellation_policy
max_concurrent_operations (= 1 in v1)
```

RPent 只能把 manifest 中实际允许的 skill 注册为 LLM tool；不得把服务器未声明的 skill 留在 prompt
或工具列表中。manifest/session/hash 变化时，旧工具调用和旧 grant 全部失效。

### 4.3 SkillRequest

第一版请求至少绑定：

```text
client_id
request_id
idempotency_key
client_sequence
runtime_session_id
capability_manifest_hash
expected_station_revision
skill_id / skill_version
parameter_profile_id / parameter_profile_hash
parameters                 # exact schema, additionalProperties=false
observation_snapshot_id
requested_mode
requested_timeout_ms       # 必须不超过 capability/access-policy maximum，超限拒绝
skill_execution_grant_id   # privileged skill only; opaque ID
```

时间约束使用 server monotonic clock。客户端传相对 `requested_timeout_ms`，Gateway 在 admission 时计算
本地 deadline；跨进程绝对 wall-clock 只可用于诊断，不能用于安全判断。

### 4.4 ObservationSnapshot

Agent 获取的是低频、不可变、因果自洽的 snapshot，而不是自行拼接的高频共享状态：

```text
snapshot_id / captured_at / available_at
runtime_session_id / station_revision
authority_epoch / execution_epoch / dispatch_owner_generation
lifecycle_state / execution_gate / active_operation_id
source_binding / producer_epoch
target_binding / execution_manifest_hash
frame_ids / frame_graph_hash / tool_config_hash / calibration_hash
measured_state_summary / freshness / validity
fault_class / rearmable
capability_manifest_hash
```

高频 measured state、controller state 和 SDK telemetry 仍由 lower runtime 本地消费。具体技能启动时必须重新
读取 fresh high-rate state；Agent 使用的 snapshot 只用于规划与 optimistic concurrency，不能代替 lower
pre-dispatch safety。

### 4.5 SkillOperation

统一状态机：

```text
received
  → validated
  → preflighted
  → awaiting_operator_confirmation     # privileged only
  → admitted
  → activated
  → running
  → succeeded | failed | cancelled | timed_out
```

每次投影至少包含：

```text
operation_id / request_hash
status / reason_code / terminal
station_revision_at_admission / current_station_revision
started_at / updated_at / deadline_at
lower_operation_id
receipt_refs[]
progress_summary             # advisory only
result                       # exact result schema when terminal
```

`admitted`、`activated`、`running` 与 `succeeded` 必须严格分开；HTTP 2xx 只能表示请求被解析或接纳，
不能表达 target READY、permit consumed、SDK accepted 或 measured success。

---

## 5. Agent-facing 工具模型

底层 RPC 可以使用通用 operation API，但 RPent 不向 LLM 暴露带自由 `skill_id` 的任意调用器。推荐：

1. 固定暴露少量系统工具：
   - `dexteleop_get_capabilities`
   - `dexteleop_observe`
   - `dexteleop_get_operation`
   - `dexteleop_cancel_operation`
   - `dexteleop_request_hold`
2. `DexTeleopToolkit` 在 run 启动时读取 manifest，为每个允许的技能生成一个精确、独立的 tool schema；
3. 每个 tool handler 内部把固定 `skill_id/version` 翻译成通用 `SkillRequest`；
4. 不暴露 raw qpos、raw pose、SDK method、任意 Python、任意 shell、任意路径写入或通用 `env.step`；
5. Agent 不自动重试有副作用操作。重复网络请求只使用同一 idempotency key 查询原 operation。

### 5.1 技能分级

| Tier | 范围 | 第一版策略 |
|---|---|---|
| T0 | capability、observe、status | read-only，可默认开放给已认证客户端 |
| T1 | 无真机运动或 sim-only 普通技能 | fake/sim 闭合后开放 |
| T2 | exact manifest 下的普通真机运动技能 | 必须显式 session grant；M5.8/M6 gate 后逐项开放 |
| T3 | Home、rearm、enable、recovery 等 lifecycle 技能 | Agent proposal + operator confirmation；单独验收 |

### 5.2 普通技能类别

首版 protocol 支持下列类别，但 capability manifest 只启用实际已有 lower implementation 的项目：

- `policy_task_v1`：提交结构化任务目标，由现有 lower policy runtime 生成 action；Agent 不发送 chunk；
- `registered_motion_v1`：选择服务器登记的命名目标/轨迹 profile，不允许任意 qpos/pose；
- `bounded_cartesian_delta_v1`：严格 envelope 内的微动，只能 sim 首发，真机对齐 CS-4 后另行授权；
- `end_effector_named_state_v1`：命名夹爪/手状态，按 component capability 单独准入，不继承 arms 授权；
- lifecycle 四技能：见下一节。

具体机器人任务技能不应实现在 Gateway 中。策略技能留在 policy/deploy owner；通用运动语义留在
`robot_motion`；sim/real binding 分别留在 `sim` 与 `teleop_runtime`。

---

## 6. Home/Rearm 四种精确语义

### 6.1 `capture_runtime_home_v1`

用途：在机器人保持静止时，将当前 fresh measured robot/source 状态建立为 activation-local runtime anchor。

前置条件：

- target connected 且 station/source publication 为 HOLD；
- 没有 active operation，旧 worker/queue 已 drain；
- measured qpos/TCP、source wrist/tracker 样本、frame/tool/calibration 均 fresh、finite、valid；
- 满足稳定窗口与 tracker-side/role 检查；
- operator 对 exact capture spec 确认。虽然不运动，但它会改变后续映射语义，因此仍是 privileged。

执行效果：

- 不调用任何运动 SDK；
- 创建 `SessionAnchorReceipt`，绑定 measured-state hash、source samples、frame/tool/calibration hash、
  stability interval、station revision 与 execution/producer epoch；
- 使旧 activation、旧 proposal 与旧 runtime-home receipt 失效；
- 完成后仍为 HOLD，不自动开放 publication。

### 6.2 `move_to_registered_home_v1`

用途：执行实际物理 Home/qpos-init 轨迹。

前置条件：

- 使用服务器登记、版本化、可 hash 的 `RegisteredHomeProfile`；
- profile 绑定 exact embodiment、joint order、tool、frame、target 与 execution manifest；
- fresh measured state、路径/碰撞/关节边界 preflight 通过；
- operator 确认 exact profile hash；真机执行期间保持 deadman；
- Station 对普通 source publication 保持 HOLD；lifecycle operation 通过独立 permit 路径执行；
- M5.8 production lifecycle mediation 已覆盖 logical admission、worker/current epoch 和每个 pre-SDK seam。

执行规则：

- 每个实际 SDK target 使用新的一次性 `LIFECYCLE_CONTROL` permit；
- permit 不能跨 tick、retry 或后台 worker 复用；
- deadman 丢失、deadline、epoch/revision 变化、cancel 或 backend fault 立即进入 stop/HOLD 路径；
- 终态必须区分 planned、permit accepted、SDK request、backend ack 与 measured terminal evidence；
- 完成后仍为 HOLD，不自动开始 policy/human control。

### 6.3 `rearm_safety_latch_v1`

用途：在明确可恢复 fault class 下清除指定 safety latch，不恢复运动。

允许条件：

- fault 被 capability/profile 明确标记为 `rearmable`；
- old motion/reference workers 已停止并 join/drain；
- fresh measured state finite、在允许域内，必要时引用 fresh `SessionAnchorReceipt`；
- operator 确认 exact fault、target、session 与 expected post-state；
- lower manual-rearm state machine 接纳并返回 typed receipt。

明确拒绝：

- background worker/runtime integrity fault；
- 需要进程 restart、设备 reconnect、重新校准或物理检查的 fault；
- stale/unknown measurement、未知 fault class、旧 session/epoch；
- 试图通过 rearm 自动恢复 publication 或复用旧 queue/reference。

成功后推进 generation/epoch、确认队列为空并保持 HOLD。

### 6.4 `resume_after_hold_v1`

用途：在 Home/Rearm/普通 HOLD 后显式恢复可发布状态。

前置条件：

- 没有未解决 fault 或 active privileged operation；
- exact target/source binding 仍匹配 manifest；
- fresh measured state、source readiness、deadman/rebase 等 profile-required receipt 完整；
- producer 使用新 epoch，旧 proposal/reference/command 不可复活；
- lower target 明确返回 READY。

只有这些条件完成后 Station 才开放 publication。该技能不做 Home、不清 fault、不改变 motion owner。

---

## 7. 授权模型

### 7.1 三层权限严格分离

| 权限 | owner | 作用 | 不能替代 |
|---|---|---|---|
| `OperatorControlLease` | Control Station/operator | 人类操作会话与 mutable operator intent | Agent grant、motion/lifecycle permit |
| `SkillExecutionGrant` | Control Station access policy | 允许某客户端在窄 scope 内提出/执行某 skill | authority、owner token、pre-SDK permit |
| dispatch permit | robot_motion/lower lifecycle state machine | 授权一次实际 lifecycle/safety/motion call | operator confirmation、skill admission |

RPent 永远不获取 `OperatorControlLease`。Operator 可基于当前 lease 创建 immutable
`SkillExecutionGrant`，至少绑定：

```text
grant_id / grant_hash
operator_authorization_receipt_id
client_id
runtime_session_id / capability_manifest_hash
control_domain_id / target_binding_id
skill_id / skill_version
parameter_profile_hash
authorized_request_hash       # T3 必填；对不含 grant ID 的 operator-visible request 求 hash
allowed_mode
may_move_robot
max_invocations
issued_at / expires_at
```

grant 存在 server side；planner 只看到 opaque reference。runtime restart、manifest/target/profile 变化、
HOLD/ESTOP policy 指定的 revoke、expiry 或 invocation budget 耗尽都会使其失效。Access policy 同时限制
grant 最大 lifetime 与 invocation 数；T3 grant 必须 exact 匹配 `authorized_request_hash`，不能只凭同一
parameter profile 替换具体参数。

### 7.2 Privileged skill 两阶段准入

```text
Agent proposal
  → schema/manifest/revision/grant preflight
  → read-only lower preflight
  → awaiting_operator_confirmation
  → operator confirms exact request hash within short TTL
  → Station HOLD + publication close + cancel/drain
  → lower lifecycle operation admission
  → one-shot permits at actual call seams
  → typed receipts/terminal evidence
  → remain HOLD
  → separate resume_after_hold
```

第一版物理 Home 与 safety rearm 每次调用都要求 operator confirmation。未来即使允许 session-level
预授权，也必须限定 exact profile、短 TTL、次数、deadman 和可取消边界，不能降级为普通 T2 skill。

---

## 8. 进程、传输与安全边界

### 8.1 第一版拓扑

- Agent Gateway 运行在 DexTeleop control host，默认只绑定 Unix socket 或 `127.0.0.1`；
- RPent 与 Gateway 同机时优先 Unix socket；跨机只在后续加入 mTLS/private-network 身份后启用；
- HTTP/MCP handler 只做 parse、auth、admission、nonblocking enqueue 和 status；
- 长任务运行在独立 bounded operation worker，cancel/HOLD 使用独立高优先级路径；
- Gateway 不随 RPent 退出而自动关闭硬件；RPent 也不启动或重启 Gateway/设备。

### 8.2 Planner 隔离

第一版 real profile 必须满足其中一种：

1. 使用 API planner，只向模型暴露 `DexTeleopToolkit` 的结构化工具；或
2. Codex/Claude Code planner 在无控制凭据、无 control-host 文件访问的独立容器/主机运行，凭据由窄 sidecar 持有。

同时需要：

- `Toolkit` 支持 `include_common_tools=False`，LIBERO/default 行为保持 `True`；
- real profile 禁止 Claude built-in Bash/Read/Write/Glob/Grep；
- real profile 禁止 Codex `full_access` 接触 Gateway token 或 Unix socket；
- credential 不进入 prompt、tool result、日志、recipe 或 output directory；
- 请求体、结果、图像和事件均有大小、频率和保留上限；
- auth failure、schema mismatch 和 rate limit 不得触发 lower operation。

---

## 9. 模块落点

### 9.1 RPent：consumer 与 orchestration

建议新增：

```text
robots/dexteleop/
  __init__.py                 # EnvSpec/toolkit factories；默认只连接，不启动真机
  env_spec.py                 # endpoint/profile/planner safety config
  prompt_bundle.py            # skill-level prompt，不描述 raw action
  toolkit.py                  # manifest-derived tool registration
  client.py                   # capability/request/status/cancel/hold client
  contracts_snapshot/
    agent_skill_v1.json       # 由 DexTeleop canonical schema 生成
```

建议测试：

```text
tests/test_dexteleop_env_spec.py
tests/test_dexteleop_toolkit.py
tests/test_dexteleop_client.py
tests/test_dexteleop_contract_snapshot.py
tests/test_dexteleop_real_profile_security.py
```

必要的兼容性修改：

- `rpent/tools/toolkit.py`：增加默认保持现状的 common-tool registration flag；
- planner 配置：real profile 对 built-in tools 与 sandbox 做 fail-closed 检查；
- 不需要中央 env registry 改动，继续使用 `robots.<name>` 动态导入；
- 不修改 LIBERO 的默认 service auto-start 行为。

### 9.2 DexTeleop：canonical Gateway application leaf

建议新增：

```text
control_station/agent_gateway/
  __init__.py
  contracts.py                # 纯外部协议与 hash
  access_policy.py            # client policy 与 SkillExecutionGrant
  operation_port.py           # lower-owned operation protocol/receipt join
  status.py                   # read-only projection
  service.py                  # bounded transport facade；不内联执行
  profiles.py                 # exact fake/sim/passive-real profiles
```

建议测试：

```text
tests/test_control_station_agent_gateway_contracts.py
tests/test_control_station_agent_gateway_access_policy.py
tests/test_control_station_agent_gateway_service.py
tests/test_control_station_agent_gateway_fake_operation.py
tests/test_control_station_agent_gateway_sim_lifecycle.py
tests/test_control_station_agent_gateway_dependency_boundary.py
```

第一批保持以下文件不变：

- `control_station/contracts.py`
- `control_station/state_machine.py`
- `control_station/web/api.py`
- `control_station/operation_coordinator.py`
- `control_station/__main__.py`
- `teleop_runtime/deploy/run_policy.py`
- RM65 backend 与 position executor

这些路径当前已有并行工作且承担现有公共语义。Agent Gateway 首批通过新文件、fake port 和测试构建，
不把 side-effect operation 塞进现有 no-I/O coordinator，不向 generic Web API 添加 Home/Rearm，不新增
production entrypoint。

### 9.3 Lower capability implementation

- sim skill implementation 放在 `sim/mujoco/deploy` 或现有 sim composition 的明确 application binding；
- real policy/task skill port 放在 `teleop_runtime/deploy`，复用现有 policy emission/runtime；
- real Home/Rearm port 暂时包裹现有 qpos-init/manual-rearm owner，但只有 M5.8 production permit closure 后
  才能进入 Active profile；
- 真正 backend-neutral 的 lifecycle contract 才可下沉 `robot_motion/runtime`；
- 不提前创建通用 `skill_runtime`，不在 Gateway 中实现算法或复制 sim/real 控制逻辑。

---

## 10. 分批实施路线

### B0：规格冻结（当前批）

产物：本文。

退出条件：

- ownership、四种 lifecycle skill、三层授权、协议字段、文件落点和 M5.8/M6 依赖明确；
- 不改 runtime，不启动 sim/device/hardware；
- 后续 batch 不再使用通用 `home/rearm` 或 direct Agent→backend 设计。

### B1：纯 contract 与 schema snapshot

执行状态（2026-08-08）：已完成纯 capability/request/observation/operation contract、
`SkillExecutionGrant` 与无副作用 access assessment；DexTeleop canonical schema 与 RPent bundled snapshot
通过字节级 drift 检查。当前没有 service、worker、Station caller、target binding 或 runtime client，最远边界是
纯 Python contract/access decision。

范围：

- DexTeleop 新增纯 `agent_gateway.contracts/access_policy`；
- 生成 RPent schema snapshot；
- 严格 schema、canonical hash、版本兼容、grant 验证测试；
- 不接 Station state machine、operation worker 或任何 target。

退出条件：

- unknown/additional field、旧 session/hash、过期 grant、超限 timeout 全部拒绝；sequence/idempotency 字段已
  进入 hash，依赖历史状态的 duplicate/out-of-order 与同 key 不同 body 检查留在 B2 bounded service；
- schema snapshot 与 canonical hash 可重复生成；
- import boundary 测试证明纯 contract 不导入 sim/teleop/hardware；
- 只能声称 contract complete，不能声称 Gateway 或 skill 已接通。

### B2：fake/null Gateway 与 RPent Toolkit

执行状态（2026-08-08）：**B2a/DexTeleop 已完成，B2b/RPent 在公共 schema 分发决策处暂停**。

B2a 当前落地边界：

- `control_station/agent_gateway/operation_store.py` 只保存 one-active-operation 与 bounded terminal history，
  不含 work queue、线程、transport 或 lower dispatch；同 key 同 body 返回原 operation，同 key 不同 body、
  sequence 回退、跨 operation Station revision 回退均拒绝；
- `operation_port.py` 只允许声明 `no_io=True` 的 deterministic fake/null port；其 receipt 明确
  `authoritative_lower_receipt=false`、`proves_motion_or_hold=false`，不能当作 Station/lower/SDK/measurement
  证据；
- `service.py` 的 `submit_request` 只完成 observation binding、access assessment 与 application admission，
  不调用 operation port；只有显式 fake worker seam 才能把 `admitted` 推进到 `activated/running`；
- observation receiver 只接纳外部已经生成的 immutable snapshot，并核对 runtime session、capability manifest、
  target/execution binding、clock domain、active operation 与 revision；它不读取 Station、sim 或设备；
- cancel/HOLD 在调用 port 前先保留 control 并改变 operation hash，port 调用期间不持有 store/admission lock；
  因此 status、cancel、HOLD 可与 stalled fake activation 并发，late running/success CAS 不能覆盖控制结果；
- cancel/HOLD 失败时 operation 保持 active/control-reserved，不能开放下一 operation；timeout、disconnect 与
  idempotent service close 已有显式 fake/null 路径。

B2a 无设备证据：新增 store/service 测试 23 项通过；B1+B2a 与既有 Control Station contract 合并为
86 passed、1 deselected。deselected 项仍是开始本批前已确认的并行脏树基线
`lower_level_packages_do_not_reverse_depend_on_control_station`，不由 Agent Gateway 改动造成。RPent bundled
schema drift 测试 4 项通过。以上只能证明 pure/fake application semantics，不能证明 HTTP、Station caller、
MuJoCo、真机、HOLD 或 measured success。

B2b 复核 RPent live code 后发现一个公共接口缺口：`AgentSkillCapability` 当前只有
`parameter_schema_hash`、`result_schema_hash` 与 `allowed_parameter_profile_hashes`，没有 schema 正文，也没有
`parameter_profile_id ↔ parameter_profile_hash` 的可信映射。`DexTeleopToolkit` 因而无法仅凭 manifest 生成
精确 LLM tool schema。若在 RPent 内硬编码参数，会破坏“DexTeleop canonical、RPent consumer”的既定 owner。
此项必须先冻结，不能以 fake 实现绕过。

范围：

- bounded fake operation port；
- capability/observe/start/status/cancel/HOLD 全链；
- one-active-operation、duplicate/out-of-order sequence、同 key 不同 body 拒绝、同 key 同 body 原 operation
  lookup、late result CAS、disconnect/timeout failure injection；
- RPent real-safe toolkit profile，不提供 common/builtin shell tools；
- 无 DexTeleop device、无 MuJoCo、无 motion。

退出条件：

- HTTP worker 不执行 operation body；
- cancel/HOLD 不受 stalled operation 阻塞；
- accepted 不被误报为 executed；
- planner 无法读取 credential 或调用未在 manifest 中的工具；
- RPent/LIBERO 默认行为无变化。

### B3：MuJoCo ordinary skill

范围：

- 复用现有 authoritative MuJoCo core 与同一 owner/epoch/trace；
- 先接一个 exact server-side skill，不开放 raw qpos/pose；
- operation receipt 关联 source proposal、authority、safe、resolved target、dispatch 与 measurement lineage；
- abort/timeout/HOLD 进入同一 cancellation path。

退出条件：

- 无第二 queue/Arbiter/dispatch owner；
- success 由 measured terminal evidence 定义；
- deterministic replay、stale observation、duplicate request、late completion、cancel at every transition 全覆盖；
- 只能声称该 exact sim manifest authoritative，不外推真机。

### B4：MuJoCo lifecycle skills

范围：

- 四种技能分别实现并故障注入；
- capture Home 与 physical Home 数值/状态严格分离；
- rearmable 与 restart-required fault 分型；
- Home/Rearm 后保持 HOLD，resume 使用 fresh producer/target READY receipt。

退出条件：

- 任一旧 operation/result 在 HOLD/rearm 后不能复活；
- worker drain、epoch/generation 旋转、队列为空可验证；
- no auto-resume；
- simulated lifecycle permit 每次调用一次，不能复用。

### B5：real observation-only

依赖：Control Station CS-3a cross-rearm binding 与 CS-3b live transport/real-observation readiness。

范围：

- 绑定 existing real frontend 的 passive fanout；
- 只发布 capability、freshness、owner/epoch、gate、fault 与低频 state summary；
- Gateway manifest 明确 `real_motion=false`；
- 即使 Agent 请求 Active skill，也在 operation admission 前拒绝。

该批如需启动真实传感器/设备，必须另行取得具体设备与 no-motion 授权；本文不构成授权。

退出条件：

- target connection 不触发 enable/home/motion；
- 无 SDK write/queue pop/owner change；
- process close/drain 与 status freshness 有真实 evidence；
- 只能声称 real observation readiness。

### B6：复用 M5.8 legacy-owner mediation

本批属于 DexTeleop 控制主线，不由 Agent 接入自行完成或绕过。要求：

- legacy path 是唯一 motion owner；
- shared/Agent path observe-only；
- Home/Rearm lifecycle state machine 在 logical admission、worker/current epoch 与每个 pre-SDK call 闭合；
- 每个 actual SDK tick 使用一次性 permit；
- cancel/stop 有 profile-required acknowledgement；
- fake/dry-run parity 先于任何硬件验证。

Agent Gateway 在该批完成前不得宣称 physical Home/rearm Active-ready。

### B7：首个真机受控 skill

顺序固定：

1. `capture_runtime_home_v1`：无运动、operator confirmed、结束保持 HOLD；
2. Control Station CS-4 exact manifest 下的既有 bounded operation；
3. `move_to_registered_home_v1`：独立硬件授权、deadman、stop/drain/trace/measurement evidence；
4. `rearm_safety_latch_v1`：按 fault class 单独验收；
5. `resume_after_hold_v1`：最后开放，仍不得切换 shared motion owner。

每个 skill 使用独立 capability/profile/grant/evidence，不能由前一个 skill 的通过推导后一个 skill 已通过。

### B8：M6 后 ordinary real policy skill

只有 DexTeleop M6 完成唯一 shared owner 切换后：

- Agent 才能在 exact policy execution manifest 下启动 ordinary real policy skill；
- Gateway 仍只是 application admission owner，不成为 motion owner；
- 切换/rollback 继续复用同一 `DispatchOwnerRegistry`、`ExecutionEpochRegistry` 与 token generation；
- 每次 run 只授权一个 policy instance/selection group；
- hand、replay、human takeover、force/admittance 等能力继续独立 gate，不继承 arms-only policy 授权。

---

## 11. 验收矩阵

| 风险 | 最小证据 |
|---|---|
| 协议漂移 | canonical schema ↔ RPent snapshot hash/round-trip/unknown-field rejection |
| 身份与授权 | wrong client/session/manifest/profile/grant/expiry/count 全拒绝 |
| 幂等 | 相同 key 返回同一 operation；同 key 不同 body 拒绝；sequence 回退拒绝 |
| Station CAS | stale revision、HOLD/ESTOP 抢先、late success 均不能推进运行态 |
| 并发 | 每 domain 单 active operation；busy 不排隐式队；status/cancel/HOLD 不阻塞 |
| planner 隔离 | real profile 无 common file tools、无 shell builtins、无可读 credential |
| 观察一致性 | snapshot 绑定同一 revision/epoch/frame/calibration；过期时 fail closed |
| Home 分型 | capture 不产生 motion；physical Home 只使用 registered profile |
| lifecycle permit | 每个实际调用 one-shot；retry/tick 不复用；旧 epoch 拒绝 |
| rearm 分型 | rearmable 成功；restart-required/unknown fault 拒绝；无 auto-resume |
| cancel/断网 | publication close、旧 work drain、HOLD receipt；late command 被 final seam 拒绝 |
| sim authority | 同一 existing core/owner/trace；无第二 queue/dispatcher |
| real passive | 零 SDK write、零 owner change、零 action admission |
| real Active | exact manifest、明确硬件授权、stop/ack/measurement/rollback 全闭合 |

测试始终使用最小相关集合；初始批次只运行 pure/fake/sim 测试。真实设备、Home、servo、rearm、
motion 或 remote write 必须按批次重新取得范围明确的授权，不能由本文或单元测试推导。

---

## 12. 兼容与回滚

1. 所有新 profile 默认关闭；现有 teleop、collection、policy deploy、LIBERO 和 Control Station Web 默认值不变。
2. B1/B2 只新增文件；若失败可移除新 profile，不需要改变 legacy owner。
3. Gateway/agent 断开不关闭硬件进程，但 real operation 进入 HOLD/cancel；设备生命周期仍由现有 frontend 管理。
4. 协议升级使用显式 version/hash，不做字段含义静默复用。
5. M5.8/M6 rollback 使用既有 owner/epoch/cancellation 事务；Agent 不实现独立 rollback。
6. 不删除或改名现有 Control Station intent，直到 caller inventory、兼容窗口和迁移测试完成。
7. DexTeleop 当前工作树有大量并行修改；每批开始前重新检查 exact target paths 的 status/diff，禁止 reset、stash、
   批量覆盖或顺手整理无关文件。

---

## 13. 当前关闭项与后续待决项

### 已关闭的方向性决策

- RPent 采用 skill-level orchestration，不做连续低层控制；
- Gateway 位于 Control Station application leaf，不位于 `robot_motion`、`teleop_runtime` 或 RPent server；
- canonical protocol 位于 DexTeleop，RPent 使用 schema snapshot；
- Home/Rearm 拆成四个 skill；
- Agent 不获得 operator lease；
- first real = observation-only + runtime Home capture；
- physical Home 等待 M5.8，ordinary real policy authority 等待 M6；
- 第一版 one active operation、无 Agent-side queue、无 auto-resume。

### 不阻塞 B1/B2 的待决项

- 生产远程传输最终采用 mTLS HTTP 还是受控私网 transport；第一版保持 loopback/Unix socket；
- operator confirmation 最终由现有 GUI capability panel、脚踏 deadman 还是独立 operator console 呈现；
- 首个 ordinary sim skill 的具体 task catalog；由现有 lower implementation inventory 在 B3 前选择；
- `RegisteredHomeProfile` 的首个真机目标值、路径、速度和容差；必须在 B7 硬件范围授权前单独冻结，
  不在 contract 阶段猜测；
- hand/gripper、force/admittance、replay 与 human takeover 的 Agent skill；均不进入首个 arms-only profile。

如果上述待决项在对应批次前仍会影响公共语义、安全或 owner，必须暂停并由用户确认；不得通过默认值
悄悄决定。

---

## 14. 推荐立即执行的下一批

当前先冻结 **skill schema/profile 的 canonical 分发方式**，再进入 B2b。推荐方案是：

1. DexTeleop 新增 versioned `AgentSkillSchemaCatalog` canonical contract；catalog entry 同时携带 exact
   parameter/result JSON Schema，以及 `parameter_profile_id/profile_hash/schema_hash` 的明确映射；
2. capability manifest 新增 `skill_schema_catalog_hash`，RPent 获取 catalog 后先校验整体 canonical hash，再逐项
   校验 capability 的 parameter/result/profile 引用；只有全部一致的 skill 才注册为 LLM tool；
3. catalog 仍只描述 schema/profile，不携带凭据、permit、registered Home 数值或 runtime authorization；
   manifest/session/grant 仍按现有边界独立失效；
4. `Toolkit(include_common_tools=False)` 保持 LIBERO/default 为 `True`；DexTeleop toolkit 仅重新注册无文件 I/O
   的 finish tool，不恢复 read/write/list；
5. API planner 增加可关闭隐式本地 `read_image(path)` 的选项，DexTeleop real-safe profile 强制关闭；
   Claude Code 的 Bash/Read/Write/Glob/Grep 与 Codex `full_access` 当前实现均直接拒绝 real-safe profile，直到
   未来有无凭据隔离容器/sidecar 的独立验收。

备选方案是把完整 schema/profile 内联进 capability manifest；实现更短，但 manifest 体积、hash churn 与
重复内容更大。仅在 RPent 本地硬编码 catalog 不建议采用，因为会形成第二 canonical owner 并允许 server 与
prompt/tool schema 漂移。

用户确认该公共 contract 方向后，下一 change set 顺序为：先做 B1.1 schema catalog/snapshot drift，再做
B2b fake client/toolkit/security profile。仍不修改 Station `_IMPLEMENTED_INTENTS`、既有 Web
allowlist/operation coordinator、production entrypoint、teleop/deploy caller、RM65 backend 或任何设备配置。

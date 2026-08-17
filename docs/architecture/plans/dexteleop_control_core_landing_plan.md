# DexTeleop Control Core 当前落地边界

状态：活动结构与安全门禁索引；最后复核日期为 2026-08-17。

本文只维护共享控制核心当前有效的 owner、执行模式、兼容边界和后续 gate。原 M0～M8 逐批设计、
阶段收据、测试数字和 2026-07 现场状态已冻结在
[Control Core 历史落地台账](../reports/control_core_landing_ledger_20260722.md)。需要追溯字段来源或旧
批次推理时可以查阅历史台账，但开始新任务前必须重新核对代码、测试、manifest 和 linked worktree。

长期语义见 [控制架构规划](dexteleop_architecture_planning.md)，当前 direct/composed 运行域事实见
[运行域与可拆分边界](../runtime_domain_boundaries.md)，RTC producer 见
[RTC 落地计划](dexteleop_rtc_landing_plan.md)，应用层装配见
[Control Station 当前计划](dexteleop_control_station_plan.md)。本文不重复这些 owner 的详细实现。

## 1. 当前结论

DexTeleop 已有可复用的控制合同和明确的应用装配边界，但“共享合同存在”不等于所有正式入口已经迁成
同一个 authoritative runtime。

当前必须区分两种执行模式：

| 模式 | 当前入口/装配 | authoritative owner | `control_composition` |
|---|---|---|---|
| 单源 policy deploy | `teleop_runtime.deploy` | 既有 deploy safety/executor/backend | 不加载、不创建 |
| 单源 human teleop | canonical RealTeleop frontend | 既有 session、Rig 和 backend | 不加载、不创建 |
| 单源 replay | `teleop_runtime.replay` | replay executor、hardware Rig/backend | 不加载、不创建 |
| 多源组合/HITL | 显式 Control Station/Sim-HIL profile | 同一个 target runtime 的 safety/dispatch | profile 显式创建 |

GUI、ControlTrace、difference 或 shadow observer 不会把单源入口变成多源控制。只有两个或更多 source
同时参与 selection、fusion 或 handoff 时，application profile 才应显式创建 composition runtime。

本计划不授予真机运动。任何 real authority 迁移都必须是独立、版本化、可回滚的安全批次，不能从 sim、
shadow、fixture、历史测试数字或共享类型存在推导出来。

## 2. 当前控制拓扑与 ownership

```text
human / replay / policy source owner
        │ immutable emission / source fact
        ▼
optional control_composition              # only explicit multi-source profile
        │ candidate + lineage
        ▼
robot_motion proposal / authority / arbitration
        │ selected action
        ▼
canonical safety / target resolution / reference
        │ execution directive / permit
        ▼
one target runtime / one dispatch owner
        │
        └── passive trace/capture projection → one Episode owner
```

| 责任 | 唯一 owner |
|---|---|
| action/proposal/authority/safety/execution contract | `robot_motion.action` |
| identity、epoch、gate、worker drain、dispatch owner | `robot_motion.runtime` |
| backend-neutral multi-source composition、session、handoff、lineage、difference | `control_composition` |
| model client、wire protocol、artifact integration | `deployments` |
| real observation、source/model worker、frontend/backend assembly | `teleop_runtime` 对应 application owner |
| sim state、physics、task、measurement、sim execution | `sim` |
| schema/hash/validation | `dataset_contracts` |
| projection/reader/training view | `dataset_interface` |
| live collection lifecycle/persistence mechanics | `collection_runtime` |
| 非正式 learner/loss/offline evaluation | `experiments/policy_learning` |
| GUI/profile coordination | `control_station` application leaf |

共享代码依赖向 `robot_motion`、`control_composition` 和纯数据 contract 收敛；下层 owner 不得反向导入
Control Station、concrete sim/teleop/deployment application。`robot_motion` 也不得反向依赖
`control_composition`。

## 3. 已建立的共享边界

当前代码已经具有以下可独立复用的机制：

- `robot_motion/action/`：proposal、snapshot、Arbiter、authority、safety anchor、execution target、reference
  和 authoritative trace contract；
- `robot_motion/runtime/`：identity、producer/execution epoch、execution gate、dispatch-owner registry、
  cancellation、worker drain 和 one-shot dispatch seam；
- `control_composition/`：activation、eligibility、composition strategy、allocation、handoff/session、trajectory、
  trigger 和非权威 observability；
- `dataset_contracts/control_trace_validation/`：按 rule family 验证 ControlTrace lineage；
- [运行域边界测试](../../../tests/test_runtime_domain_boundaries.py)：保持默认单源入口不 eager-load
  composition，并以显式 MuJoCo composition binding 作为正向对照。

这些机制的存在只证明 contract 或当前 scoped caller。是否进入正式入口、是否 durable capture、是否
authoritative、是否完成真机验收，必须分别陈述。

### 3.1 Production decision-rate contract

`robot_motion.action.arbitration_runtime.MAX_PRODUCTION_ARBITER_HZ` 当前为 60 Hz。它只限制 production
`ArbitrationDecision` 的发布，不限制 sensor、source、policy inference、reference、IK、command、backend、
physics 或 recording cadence。

HOLD、FAULT 和 safety veto 可以立即生效，但不能伪装成超频的普通 decision。不同 execution profile 的
reference/IK/command rate必须由各自 capability 和验收事实定义，不能从 Arbiter 上限推导。

### 3.2 Runtime 与 dataset 单向依赖

```text
runtime event
  → application encoder/projection
  → dataset_contracts record
  → collection/capture owner
```

- runtime core 不导入 writer、Parquet、Torch、sim 或硬件 SDK；
- dataset contract 不导入 runtime object、frontend 或设备；
- schema/hash domain 由纯 contract owner 定义；
- capture projection 或 WAL 不拥有 authority、safety 或 episode-independent execution state。

### 3.3 V1 compatibility surface

当前 canonical action/execution v1 仍包含 RM65 双臂、left/right、6DoF arm 与 q6/q16 hand 等固定形态。
准确表述是“对 contract-compatible embodiment 可复用”，不是“任意机器人只换 backend”。

现有 v1 import、field 和 artifact layout 在迁移期保持兼容。第二本体或不同 DOF/frame/component 必须发布
新的 typed bridge、layout、capability、sim/real binding 和验证，不能向 generic core 继续增加 shape/name 分支，
也不能把旧 artifact 静默解释为兼容。

## 4. 必须保持的运行不变量

1. 一个 control domain 只有一个 authority source、execution epoch source 和 dispatch owner。
2. 一个 source emission 只由自己的 producer/queue/cursor 推进一次；observer 和 composition 不重复读取。
3. composition 只产生 candidate/lineage，不成为 canonical safety、reference 或 dispatcher。
4. safety、IK、reference、backend 和 actual execution receipt 的顺序由 target runtime 保持。
5. worker stop、queue empty、browser token、operator lease、candidate eligibility 和 motion authority 是不同事实。
6. 一个 formal capture 只有一个 Episode owner；ControlTrace 完整不等于 motion accepted，seal 不等于 commit。
7. source/target/profile/manifest 变化使旧 preflight、capability assessment 和 authorization 失效。
8. default-off、shadow、sim 和 no-action 证据只能支持其实际执行边界，不能升级真实 motion readiness。
9. stateful modifier 的最终 composed output 在最后权威边界受总 envelope 约束；persistent feedback 使用明确的
   incremental baseline。
10. Active-only feedback/state transition 必须先有离线 closed-loop 证据，不能由 shadow parity 授权。

## 5. 后续落地批次

旧 M0～M8 编号只作为历史追溯，不再直接代表当前任务状态。新工作按 ownership 和可验证边界建立窄批次。

### CC-1：保持单源 direct path 独立

- policy、human、replay 默认入口继续不创建 `control_composition`；
- passive observer/fanout 不改变 queue pop、cursor、source cadence、dispatch 或 shutdown；
- application wrapper 保留原 CLI/import 行为，迁移实现时使用薄兼容转发；
- general deploy、teleop、replay 的修复分别由各自 owner 完成，不为了统一而先合并运行环。

### CC-2：收敛 explicit composed profile

- reusable cut、eligibility、strategy、session、handoff 和 observability 只在 `control_composition`；
- human/replay/policy producer 继续 lower-owned，composition 消费同一 immutable emission；
- sim/real target 保持唯一 safety、reference、epoch 和 dispatch owner；
- Control Station profile 只装配 typed ports/manifest，不持有 numerical/runtime implementation；
- 一次只迁移一个职责，并用 hash、数值、调用顺序和 default-off parity 验证。

### CC-3：闭合 capture 与 execution 的单向观察

- runtime/capture 通过 injection port 或 passive observer 连接；
- trace/WAL failure 不得同步授权或改变 motion，但 capture-required profile 可以通过独立策略发 typed HOLD；
- episode membership、seal、commit、native stream 和 artifact receipt 继续由唯一 capture owner 闭合；
- reader/training view 只消费 committed/explicitly-auditable facts，不重建第二 control state。

### CC-4：隔离 embodiment/action-v1 debt

- 用精确 allowlist 阻止 RM65、left/right、q6/q16 和固定 shape 假设向 generic core 扩散；
- 新本体先定义 component/action/frame/capability contract，再增加 profile-specific adapter；
- 保留旧 v1 facade 和 artifact consumer，只有经调用审计、迁移说明和 parity 后才能删除；
- 非正式 learner 源码归 `experiments/policy_learning`，model artifact 归 `model/`；生产模块不得导入实验源码，二者都不拥有 motion semantics。

### CC-5：production authority adoption（独立安全批次）

每个 source/target profile分别证明：

1. exact manifest、inventory、fresh measured state 和 capability；
2. producer/authority/execution epoch 与 dispatch-owner binding；
3. canonical safety、reference admission、command/backend receipt；
4. HOLD/cancellation、worker drain、stop acknowledgement 和 rollback；
5. required ControlTrace/capture completeness；
6. 对应环境的 latency、stability 和 operator authorization。

human、replay、sync policy、RTC policy 和 mixed profile互不继承 motion authorization。第一条真实 profile 通过也
不自动授权其他 source、hand topology、frame、controller 或 target。

## 6. 成熟度口径

文档、代码评审和发布说明统一使用以下范围，不再用一个“完成”混合多种证据：

| 口径 | 含义 |
|---|---|
| contract complete | pure type/algorithm 与 deterministic test 可用；没有 production caller 含义 |
| compatibility seam active | 已接既有入口，但旧 owner/默认行为保持 |
| scoped sim authoritative | 只在精确 sim profile 中拥有 authority |
| passive real/no-action | 消费真实输入或 transport，但不发送动作、不拥有 motion authority |
| durable capture | required facts 已进入对应 seal/commit 边界 |
| production authoritative | exact real profile 已成为执行 owner并通过其安全验收 |

每项结论同时说明：实际 caller、环境、最远执行边界、默认开关以及下一个未验证边界。测试数量只进入带日期
report/evidence，不作为活动计划的长期状态。

## 7. 每批验证

在已激活的 `${DEXTELEOP_ENV_NAME:-dexteleop310}` 环境中，从目标 worktree 根目录选择相关测试：

```bash
python scripts/check_docs_navigation.py
python -m pytest -q tests/test_runtime_domain_boundaries.py
python -m pytest -q tests/test_control_station_boundaries.py
python -m pytest -q tests/test_control_composition_boundaries.py
python -m pytest -q tests/test_embodiment_generalization_firewall.py
git diff --check
```

若实际测试文件名或 profile 已变化，应先搜索当前 suite，不能通过创建空兼容测试来满足旧命令。入口 `--help`、
characterization、formal capture、sim 和 hardware evidence按变更范围递增；没有授权时不启动设备、真实 backend、
remote execution 或 motion。

## 8. 当前优先顺序

1. 继续用 runtime-domain/boundary tests锁住 direct 与 composed 模式，禁止默认入口重新耦合；
2. 先完成 Control Station GUI/profile 减薄，再删除 `control_station/composition` 中的过渡 ownership；
3. 对 capture/projection、embodiment-v1 debt 分别建立窄 owner 批次，不与 authority 迁移混改；
4. RTC producer 只按 RTC 独立计划推进，不把 worker/plan ownership放回 control core application；
5. 任何 production authority 工作从当前 main、exact profile 和现场前提重新立项；历史 M5.8/M6 文本不视为
   当前授权或 readiness。

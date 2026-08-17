# Chunk-Aware Relative HITL 历史实施台账（更新至 2026-08-07）

> 文档类型：历史设计与实施台账。本文中的“当前”、R0-R4 阶段状态、测试数字、
> 工作树约束和下一步记录的是原计划维护期，不再作为当前公开 Sim-HIL profile、
> GUI 布局、真机接入或 motion readiness 的事实。现行 ownership、四类 Sim-HIL
> profile、trajectory seam 与剩余 gate 见
> [HITL 当前边界](../plans/hitl_chunk_relative_migration_plan.md)。本文保留完整的
> ResidualSession、NominalBaseBinding、preview、allocation、rebase 与 handback
> 讨论用于追溯，不授权设备启动、真机动作或旧命令重跑。

更新日期：2026-08-07
状态：R0–R3.0 与 R3.1a 已完成；R3.1b 已完成 default-off、single-target
proposal publication、virtual typed session lifecycle 和既有 MuJoCo 下游/FK
observation 的最小闭环。R3.1c-A/B/C 的 reusable core 已完成：source-intent 与
conditional corrected whole-horizon preview、独立 freshness、窗口运动量与 exact-anchor
residual magnitude 驱动的 component-keyed allocation 均已有开发态契约和纯测试；
R3.1c-D1/D2/D3/D4 已完成 receipt-driven reusable runtime、Control Station
default-off typed worker port、virtual-source MuJoCo
shadow/context/corrected-preview/trace 接线，以及不调用 lower step 的 exact proposal
publication gate。它们尚未成为 production profile；真实 application
coordinator/device binding、durable trace seal/readback 与 release/handback 仍未完成。

本文吸收了历史
[`Chunk-Aware Residual HITL 研究备忘`](../../legacy/reports/hitl_research_memo_20260802.md)
中的有效目标，也是当前 chunk-aware relative HITL 工作的事实源。它不把需求建议直接当作
源码事实；所有新增语义均按当前
`control_composition`、`robot_motion`、Control Station 与下游 observation contract
重新核对。

## 1. 当前结论

R0–R2 的总体方向成立，不需要推翻：

```text
source-owned canonical nominal timed plan
    -> ComponentActionTrajectory
    -> trajectory lifecycle / residual-session view
    -> one due nominal action + one state-valued residual
    -> existing FOLLOW_BASELINE composition
    -> one composed candidate
    -> existing proposal / authority / target / execution
```

当前已经完成的是模型无关、后端无关的纯逻辑骨架，以及一个仅用于证明因果边界的
MuJoCo Sim publication slice；它仍不是可交互运行的完整 HITL 系统。其边界仍然正确：

- policy、RTC、replay 或本地 planner 不需要理解 human；
- trajectory editor 不拥有 source queue/cursor、authority、canonical safety、IK、
  reference、command 或 dispatch；
- human residual 是 state-valued correction，不按 control tick 积分；
- 一次只准备并正式发布一个 due target；
- 现有 absolute weighted 和 per-cut relative 路径保持不变；
- 新路径必须 default-off，从纯测试、shadow、sim 逐步接入；
- 当前阶段不接真机 motion。

新建议中最有价值的修正是：

> 一次连续的 residual session 应保持 human anchor 与 residual 语义稳定，但不应在
> 设计上永久绑定同一条 nominal trajectory；必要时可以在未来明确边界切换到新的
> nominal base。

但是不采用 `BaseEpoch` 命名。项目中已经存在 producer epoch、authority epoch、
execution epoch、clock reset epoch 等对象，继续增加 BaseEpoch 会重新制造歧义。
本文统一使用：

```text
ResidualSession                 一次连续 human residual 交互
NominalBaseBinding              session 当前绑定的 nominal base
base_binding_revision           session 内 base 绑定的单调修订号
```

R3.0 先为多次 base binding 留出协议，但 `base_binding_revision` 固定为 `0`；自动
base refresh 只有在 Sim-HIL 数据证明有必要后才启用。

## 2. 可回退基线与工作树约束

改造前基线：

```text
tag:    hitl-relative-baseline-20260803
commit: 3f94c77b6ea4780533132cb647840f5f1ff5004e
tree:   807c68122348680211ca2652a98d030c30a9a83d
```

该 tag 只是在既有提交上建立稳定别名，不包含后来新增的 R0–R2 未提交文件，也不是
全工作树 checkpoint。禁止据此整仓 reset/checkout。后续比较和恢复必须先检查并行
修改，再按 HITL 路径 manifest 操作。

核心 manifest：

```text
robot_motion/action/trajectory.py
robot_motion/deploy/trajectory_adapter.py
control_composition/trajectory/
control_composition/residual_anchor.py
control_composition/activation.py
tests/test_component_action_trajectory.py
tests/test_trajectory_editing_reducer.py
tests/test_trajectory_residual_warp.py
docs/architecture/plans/hitl_chunk_relative_migration_plan.md
```

后续接线文件必须按实际 gate 逐项加入，不能把整个 `control_station`、RTC、dataset、
real-teleop 或 sim 包视为本任务独占路径。

## 3. 当前源码事实

### 3.1 已完成的 R0–R2

- `ComponentActionTrajectory` 是 component-keyed、model/backend-neutral 的不可变
  timed trajectory carrier。
- 一个 trajectory 内 action schema、component coverage 保持稳定；target time 和
  source index 严格递增。
- `source_immutable_through` 只表达 producer 不再改写的边界，不表达 emission、
  target admission、reference activation 或执行承诺。
- deploy adapter 只读投影一个 `TimedPolicyActionPlanSnapshot`；不推进 queue/cursor，
  不 merge chunk，不 pop，不 emit，也不授予 authority。
- reducer 已支持 `PASS_THROUGH`、`RESIDUAL_ACTIVE`、active-base latch、一个 latest
  shadow、exact-lattice target preparation 和 receipt-bound emission barrier。
- `publication_basis_hash` 不包含非权威 shadow；late shadow 不会使当前 active-base
  preparation 失效。
- temporal warp 已支持 Cartesian translation 与任意 joint-position/qpos 的
  `ramp_and_hold`；非 identity pose rotation 明确拒绝。
- 现有 `FOLLOW_BASELINE` 数值语义不变：reference/world translation、baseline-local
  rotation right-multiply、qpos 加 delta。

### 3.2 当前主线已经存在、无需重复发明的契约

以下能力在新建议中被描述为待新增，但当前源码已经有可复用基础：

1. `CompositionActivationScope`：绑定一次 composition activation 的 identity、clock、
   input barrier、required roles 与 freshness policy。
2. `ResidualAnchorRequest/Receipt`：原子绑定 source role、component set、source binding、
   absolute/relative schema、composition activation 和完整 component anchor identity。
3. `CompositionInputStore`：对 input fact ID、producer sequence、activation 和 time-window
   做接纳、去重与 pruning。
4. `ExecutionObservationPlan/CapabilitySpec/Binding`：上层按需声明是否需要 Reference、
   Measurement 以及 component/频率；下游 capability 必须整单接纳或明确拒绝。
5. `ExecutionReferenceFrameObservation` 与 `ExecutionMeasurementObservation`：由真实
   execution object 导出的 component-keyed observation，带 representation/schema、
   target lineage、scheduled/measured/available time。
6. proposal、target submission/admission/activation、reference admission、dispatch 等
   已有各自 receipt；不得再压缩为一个 `committed_until`。
7. Control Station 已有 typed operation intent，以及 exclusive human takeover 的
   HOLD/invalidation/authority/readiness/READY transaction。

因此 R3 的工作是建立薄的 trajectory-session binding 和 transition assessment，
而不是复制 activation、anchor、observation、takeover 或 publication 状态机。

### 3.3 仍未完成

- 当前 reducer 的一个 residual session 仍有意 latch 单一 active trajectory；自动 base
  refresh 留到 R3.3；
- 没有 nominal base transition、release handback 或 continuity correction 的运行接线；
- virtual fixture 已绑定 typed session barrier、pause/HOLD 与 episode stop，但尚未接入
  真实 application coordinator 的 clutch/deadman/device readiness 和 recorder episode；
- 没有 request-human-component-control 的 profile-neutral request；
- durable ControlTrace seal/readback 尚未覆盖新的 interactive trajectory application；
- full-pose rotation warp 尚未实现。

## 4. 冻结的 ownership 和数据流

```text
raw model/replay/local output
        |
        v
source-owner canonicalization
  queue/cursor/overlap/late-result/representation/clock handling
        |
        v
canonical nominal timed plan
        |
        v
ComponentActionTrajectory projection
        |
        v
ResidualSession + NominalBaseBinding reducer
  active / latest shadow / editable boundary / transition request
        |
        +-- preview-only future projection
        |
        v
select exactly one due nominal keyframe
        |
        + current residual temporal sample
        + optional continuity correction sample
        v
existing FOLLOW_BASELINE composition
        |
        v
one composed candidate
        |
        v
existing proposal / Arbiter / conservative compound authority
        |
        v
target-owned processing / safety / IK / reference / command / dispatch
        |
        v
optional reference / measurement observations and receipts
```

约束：

- trajectory editor 接收的是 source owner 真正准备采用的 canonical plan，不是尚可能
  被 source owner 丢弃、覆盖或重新锚定的 raw model chunk；
- RTC 只是可能的 source owner 实现之一，`control_composition` 不导入 RTC contract；
- replay、同步 policy、本地 planner 可通过同一个 generic trajectory seam 接入；
- trajectory reducer 不选择“最佳模型结果”，首版只保留 receiver-valid latest shadow；
- trajectory reducer 只产生准备结果、transition assessment 和 request，不授予 motion
  authority，也不直接驱动 HOLD；
- 下游是否存在连续 Reference 是 capability 事实，不是通用必经步骤。

## 5. ResidualSession 与 NominalBaseBinding

### 5.1 ResidualSession

一次连续 human intervention 内保持稳定：

```text
residual_session_id
composition_activation_id/hash
activation control-cut identity
residual_anchor_receipt_id/hash
residual source binding id/hash
relative action schema hash
component_ids
residual semantics / frame binding
activated_at
first_editable_target_at
residual validity / TTL lineage
release status
```

Reducer 不复制 human anchor 大 payload；anchor 的数值与 component binding 仍由现有
`ResidualAnchorReceipt` 拥有。session 只引用该 receipt，并验证它与 composition
activation、source binding、schema 和 component set 完全一致。

Human recenter 会产生新 anchor generation，不能在原 session 中静默替换 anchor。
早期计划要求每次 recenter 都建立新的 residual session；按钮/触发器在一次持续交互中
rebase 的新需求表明该规则过粗。推荐保持 `ResidualSessionActivationBinding` 不可变，
另建 session 内单调修订的 active-anchor binding；是否正式采用该方案见 21.4 的公共
语义冻结点。

### 5.2 NominalBaseBinding

一次 session 内某一段时间采用的 nominal base：

```text
base_binding_id
residual_session_id
base_binding_revision            # 0, 1, 2...，不是 epoch
trajectory_id/hash/revision
active_offer_sequence
bound_for_target_at              # 第一次使用该 binding 的 target
effective_support
switch_assessment_id/hash?       # revision > 0 时必需
switch_revalidation_id/hash?     # revision > 0 时必需
predecessor_base_binding_id/hash? # revision > 0 时必需
```

每个 revision 都是新建的 immutable record，禁止原地递增字段：

```text
binding revision 0
  -> assessment + revalidation
  -> new immutable binding revision 1
```

Base binding 只记录 nominal base 的 transition lineage。seam observation 与
continuity correction 由独立 transition/correction facts 记录，避免 binding 与
correction identity 形成循环引用。

R3.0/R3.1 只允许 revision `0`。该字段先存在是为了避免后续自动 refresh 时把
ResidualSession identity 一并推翻，不代表首轮已经支持自动切换。

### 5.3 Active 与 shadow

- `active`：当前 due-target preparation 使用的 `NominalBaseBinding`；
- `shadow`：后到的 nominal trajectory candidate；
- shadow arrival 只改变完整 state hash，不改变当前 `publication_basis_hash`；
- 首版只保留一个 receiver-valid latest shadow；
- rejected shadow 只保留轻量 identity/reason metadata，完整 payload 仅在 debug plan
  请求时保留。

## 6. Session activation 的原子因果切面

建议新增 source-neutral 的 `ResidualSessionActivationBinding`，而不是新增
human-only anchor receipt。

它应一次绑定：

```text
CompositionActivationScope
ControlCut(runtime/session/domain/clock/reset/cutoff/admission sequence)
ResidualAnchorReceipt
selected active ComponentActionTrajectory
active offer sequence
current finalized emission barrier
first editable target
initial NominalBaseBinding revision 0
optional downstream seam observation identity
```

“原子”指这些已经接纳的 immutable facts 由同一个 `ControlCut` 选择并形成一个新
binding，不要求设备采样、policy 结果和 reference 在同一物理纳秒产生。必须保证：

1. anchor receipt、active trajectory 和 publication barrier 都在 cut 内可见；
2. 它们使用相同 runtime/control/clock domain，或有显式 clock mapping；
3. anchor receipt 绑定当前 composition activation；
4. active trajectory 覆盖 `first_editable_target_at`；
5. `first_editable_target_at` 严格晚于已经 finalize 的 emission target；
6. cut 后到达的新 trajectory 只能成为 shadow；
7. session activation admission 自身有唯一 identity，重放同一 binding 幂等，旧
   activation/session 的迟到完成不得修改当前 state。

这样消除“anchor 取自 t、base 取自 t+1、edit boundary 取自 t+2”且没有统一 lineage
的问题，同时复用现有 receiver-side causal cut。

## 7. 两个正交状态维度

不将所有组合展开成大量顶层枚举。R3 采用两个维度：

```text
interaction_mode:
  PASS_THROUGH
  RESIDUAL_ACTIVE
  HAND_BACK
  COMPONENT_CONTROL_REQUESTED

base_lifecycle_status:
  BASE_STABLE
  BASE_REFRESH_PENDING
  BASE_TRANSITION
  BASE_EXHAUSTED
```

首轮实际可达状态：

```text
PASS_THROUGH       + BASE_STABLE
RESIDUAL_ACTIVE    + BASE_STABLE
RESIDUAL_ACTIVE    + BASE_EXHAUSTED
HAND_BACK          + BASE_STABLE/BASE_EXHAUSTED
COMPONENT_CONTROL_REQUESTED + BASE_EXHAUSTED
```

完整合法组合矩阵冻结为：

| interaction mode | 合法 base lifecycle |
|---|---|
| `PASS_THROUGH` | `BASE_STABLE` |
| `RESIDUAL_ACTIVE` | `BASE_STABLE`、`BASE_REFRESH_PENDING`、`BASE_TRANSITION`、`BASE_EXHAUSTED` |
| `HAND_BACK` | `BASE_STABLE`、`BASE_TRANSITION`、`BASE_EXHAUSTED` |
| `COMPONENT_CONTROL_REQUESTED` | `BASE_EXHAUSTED`，或 profile 显式允许的 `BASE_STABLE` |

两个字段必须由同一个 pure reducer 在一个 immutable event 上同时推进，不能由两个
模块分别修改。通常拒绝 `PASS_THROUGH + BASE_REFRESH_PENDING`、
`HAND_BACK + BASE_REFRESH_PENDING` 等无定义组合。

`BASE_REFRESH_PENDING/TRANSITION` 在 R3.3 之前只作为 development contract/test
占位，不从 production profile 进入。

`HOLD`、`FAULT`、authority transition 和 target activation 不属于这两个状态轴；
它们仍由下游 authoritative owner 通过 receipt 表达。

## 8. Base refresh 与 handback 是不同操作

### 8.1 Base refresh

Human 仍 active，anchor 和 residual semantics 不变，只在未来边界把 nominal base 从
binding `n` 切到 `n+1`：

```text
RESIDUAL_ACTIVE(base binding n)
  -> assess latest shadow
  -> schedule switch_at
  -> keep old binding before switch_at
  -> activate binding n+1 at switch_at
  -> preserve same human residual
  -> decay only system continuity correction
```

禁止 `newest shadow wins`，也禁止在当前已不可改写 target 上原地替换。`switch_at`
必须晚于 trajectory editor 的 publication-finalized/pending barrier，并位于 old/new
support 的有效交集内。

### 8.2 Release handback

Human release 后把 residual 从最后状态平滑衰减到 identity，并最终返回 policy-only：

```text
RESIDUAL_ACTIVE
  -> typed release intent
  -> freeze last admitted residual value
  -> select compatible current/shadow nominal base
  -> bind seam observation when profile requires
  -> HAND_BACK
  -> residual/continuity correction decay
  -> PASS_THROUGH
```

Handback 不能隐式清零 residual，也不能因 latest policy arrival 立即替换 active base。
如果在可用 support 内没有满足条件的 base，human 已 release 时只能产生 `no_action`/
`request_hold`，不能自动转为 direct human control。

## 9. Transition compatibility 与 eligibility

Compatibility 不是 canonical safety，也不修改输入数值。它只回答候选 trajectory
是否能作为同一 residual session 的下一 nominal base，以及是否允许自动 transition。

### 9.1 Hard compatibility

失败即不能在当前 session 内直接 transition：

- runtime/control domain 与 clock 可比较；跨 clock 必须有显式 mapping；
- candidate 可投影到当前 residual session 所需的 component action schema；
- 所需 component coverage 完整；
- pose frame/controlled frame 可由已声明 coordinate graph 转换；
- qpos shape/order/unit 可由显式 schema/adapter 无歧义对齐；
- candidate support 覆盖计划的 `switch_at`；
- candidate 属于当前 active producer activation，或存在显式 source transition；
- candidate offer/generation lineage 不回退、不复用旧 identity；
- 对声明 atomic coordination 的 component group，coordination identity 完整一致。

首版 R3.1 可以采用 exact schema hash、exact coverage、exact frame、same clock 的严格
profile，以降低接线风险；但通用协议不得把这些 exact 条件冻结为所有本体和 adapter
的永久限制。

特别修正新建议中的两处过强约束：

- 新 policy replan 的 producer generation 本来就可能变化，因此不要求 generation
  相等；要求的是 activation 合法、receiver order 单调且无 stale/rollback。
- coordination group 只在来源声明 coupled/atomic component plan 时必需；独立单臂、
  单末端或独立 component 不被强制绑定同一 coordination identity。

现有 `ProducerEpochActivation` 已经提供 worker restart/rearm 的 incarnation-equivalent
identity。generic trajectory 不新增另一套 `producer_incarnation` 数字，而应携带：

```text
source_instance_id
source_activation_id/hash
trajectory revision
receiver offer sequence
```

deploy adapter 可把 producer activation 显式投影到这些 source-neutral 字段；只带
generation 而没有 activation identity 的 trajectory 不能启动 residual session。

### 9.2 Soft transition eligibility

只影响 `AUTO_ELIGIBLE / WAIT / OPERATOR_REVIEW / REJECT_CANDIDATE`，不声称动作
安全：

- corrected translation seam；
- corrected rotation seam；
- qpos seam；
- 可用时的 velocity/direction difference；
- shadow freshness 与 remaining support；
- Reference/Measurement tracking validity；
- candidate 是否立即明显撤销 human correction；
- 短窗口 generation 跳变诊断。

首版只实现几何、时间和 freshness，不实现 policy intent classifier、learned gate 或
通用 cosine safety。

### 9.3 Assessment contract

建议使用 `NominalBaseTransitionAssessment`：

```text
assessment_id/hash
residual_session_id
current_base_binding_id/hash
candidate_trajectory_id/hash
candidate_offer_sequence
switch_at
control_cut identity
residual_fact_id/hash/effective_until
publication_basis_hash
hard_compatibility status/reasons
soft_eligibility status/reasons
old/new corrected action hash
corrected seam metrics
reference observation id/hash?
measurement observation id/hash?
tracking assessment id/hash?
coordination assessment?
available_at / clock domain
```

Assessment 是 immutable fact。迟到 assessment 必须同时匹配当前 session、当前 base
binding、candidate hash、offer sequence 和 publication barrier，不能仅按 trajectory
ID 接纳。它是可失效的 advisory result，不是 future switch 的永久许可。

实际 switch 前必须产生 `NominalBaseTransitionRevalidation`，重新检查：

- active binding revision/hash 未变化；
- current shadow 仍是 assessment 中的 candidate/offer；
- residual identity 未变化且仍 fresh；如已变化，必须以当前 residual 重做 assessment；
- Reference/Measurement/tracking assessment 仍是当前允许的 facts；
- capability/profile 未变化；
- `switch_at` 仍严格晚于当前 publication-finalized/pending barrier；
- old/new support 仍覆盖 `switch_at`；
- revalidation cut 尚未越过 `switch_at`。

如果 publication barrier 在等待期间追到或越过 `switch_at`，旧 assessment 作废；必须
选择更晚 boundary 并重新 assessment，不能强行切换已经进入不可改写区间的 base。

## 10. Corrected seam 与 continuity correction

Human residual active 时，seam 比较必须对 old/new base 应用同一个 residual sample：

```text
old_corrected(t) = compose(old_nominal(t), residual(t))
new_corrected(t) = compose(new_nominal(t), residual(t))
seam             = difference(old_corrected(t), new_corrected(t))
```

不能只比较 old/new nominal，也不能给新 base 重抓 human anchor。

系统为了切换连续性生成的补偿必须独立为 `ContinuityCorrection`：

```text
nominal base
+ human residual
+ system continuity correction
-> corrected due candidate
```

它必须有独立 identity、frame/schema、effective support、decay spec 和 source lineage；
不能写回 human residual。否则多次 transition 后 residual 会漂移，release/replay 也
无法解释系统补偿与人类输入的区别。

R3.0 只冻结 contract；R3.2 handback 首次使用，R3.3 才用于 active-session base
refresh。

## 11. Reference 与 Measurement seam

采用以下原则，但增加 capability 和 representation 限制：

```text
Reference 作为命令连续性的优先 seam anchor
Measurement 用于判断该 transition 条件是否仍有效
```

不新增一套 HITL Reference 协议。优先复用：

- `ExecutionObservationPlan`：上层按需声明 observation；
- `ExecutionObservationCapabilitySpec/Binding`：下游声明 component、representation、
  schema、stage 和发布能力；
- `ExecutionReferenceFrameObservation`：执行对象在 `switch_at` 对应的 reference；
- `ExecutionMeasurementObservation`：带 measured/available time 的真实反馈。

严格要求：

1. 用作 seam 的 observation 必须来自下游实际执行对象，不能由 composition 根据
   nominal/corrected preview 伪造；
2. reference 必须覆盖或精确对应 `switch_at`，并带 target/reference lineage；
3. reference/measurement representation 必须与 continuity kernel 可比较，或存在显式
   adapter；例如下游只导出 qpos，而当前 transition kernel 只会 pose，则不允许自动
   pose handback；
4. measurement 通过 association/parent lineage 关联到相应 reference，不能只取“最新
   测量”；
5. tracking error 超过 profile 阈值时，拒绝自动 transition；不能静默用 measurement
   替换 reference；
6. 若 capability 没有 Reference，profile 可以声明 handback 只允许 operator-reviewed
   或 HOLD-boundary transition；不能假装所有 backend 都存在连续 reference；
7. observation 频率和 debug lineage 由 `CompositionInformationPlan`/observation plan
   选择，不隐式复制下游高频 servo 数据。

这里需要的只是旧执行 reference 在 future switch boundary 的可观测锚，不要求上游
预先获得“新 target 经过下游处理后的新 reference”，避免形成先有新 reference 才能
发布新 target 的循环依赖。下游处理后产生的新 reference 继续作为执行事实和事后
一致性观测。

自动 transition capability 分三级：

| Level | 能力 | 允许行为 |
|---|---|---|
| `HOLD_BOUNDARY_ONLY` | 无可比较 seam observation | locked residual、expiry request、operator-reviewed/HOLD-boundary transition；禁止普通自动 handback/refresh |
| `REFERENCE_ONLY` | 可比较 scheduled Reference，无可靠 Measurement | 只在 deterministic sim 或 profile 显式绑定 reference-tracking assumption 时允许自动 transition |
| `REFERENCE_AND_MEASUREMENT` | Reference、fresh Measurement、representation mapping 与 tracking assessment | 完整自动 handback/refresh capability，仍需逐次 assessment/revalidation |

Generic contract 不把 Measurement 永久设为所有 backend 的 required input；但默认
real profile 必须达到 `REFERENCE_AND_MEASUREMENT` 才允许普通自动 handback。MuJoCo
R3.1b 应尽量同时导出 Measurement，即使首轮只用于 trace 和 threshold 标定。

## 12. Publication、non-rewritable boundary 与幂等

继续区分：

```text
source_immutable_through          producer 不再改写 source plan prefix
publication_finalized_target_at  editor 已收到正式 emission 接纳证据
pending_emission_target_at?       已发起但终态未知的最小可选 barrier
target admission/activation      下游逐事实 receipt
reference admission/activation   下游逐事实 receipt
dispatch/measurement             下游逐事实 observation
```

不采用含糊的 `authoritative_committed_through`、`reference_generated_through` 或
`executed_through` 标量，除非相应 owner 将来能证明它确实是无间隙连续 watermark。
现阶段使用具体 identity/receipt，避免从“最后观察到的时间”推导不存在的完整前缀。

当前已有多层去重：

- source-owner 的 pop/emission commit receipt；
- composition input fact ID 与 producer sequence 去重；
- application/backend 对 exact emission 的一次性 submission；
- proposal、target 和 reference 的独立 receipt identity。

因此 R3.0 先做端到端审计：

```text
trajectory preparation
  -> residual sample
  -> composition input admission
  -> evaluation/proposal admission
  -> authoritative publication
```

建议的关联 key 为：

```text
residual_session_id
base_binding_id
target_effective_at
corrected_action_hash
publication_basis_hash
composition_activation_id
```

它是跨层关联 identity，不直接替代各层原有 fact ID。只有审计证明确实存在“已发出、
receipt 未回、同一 target 会再次形成外部副作用”的窗口，才增加最小
`PendingEmissionAttempt`。不得在 trajectory reducer 中先复制完整 publication 状态机。

### 12.1 2026-08-05 源码审计结论

本轮对当前实际链路的结论是：**R3.0 不新增 pending-emission reducer**。

- RTC source owner 只有在 keyframe 已 pop 并形成唯一 emission 后才产生
  `PolicyActionEmissionCommitReceipt`；该 receipt 是 producer-local 事实，不等价于
  proposal/target/reference receipt。
- `CompositionInputStore` 按 input fact identity 与 producer sequence 拒绝 duplicate/
  rollback；`ProposalBuffer` 按 proposal ID 与 `(producer_id, producer_epoch)` sequence
  再次拒绝 duplicate/rollback。
- MuJoCo scheduled target owner 通过 execution epoch、下一 group generation、pending
  reservation 与 predecessor/supersession relation 拒绝重复或乱序 segment；这仍是
  下游自己的接纳，不由 trajectory reducer 重建。
- trajectory reducer 的 `prepare_trajectory_target()` 是无副作用 pure operation；只有
  外部正式 receipt 返回后，`commit_trajectory_target_emission()` 才推进单调 barrier，
  同一或更早 target 随后得到 `already_emitted`。
- shadow arrival 不进入 publication basis，因此不会使已经准备好的 current active-base
  target 因无关 future candidate 失效；session/lifecycle/base/barrier 变化则会改变 basis。

当前仍有一个有意保留到 R3.1a wiring 的窄接口事项：pure reducer 只保存 opaque
`emission_receipt_id/hash`。实际接线必须用一个 typed adapter 明确允许哪一种正式 receipt
推进 editor barrier，并保证只有一个 application call site 做这次 commit。不能把 RTC
producer receipt、proposal receipt、target receipt 和 reference receipt 混为同一阶段。
在出现真实“请求已经造成外部副作用但终态 receipt 丢失”的证据前，不增加第二套
pending 状态。

## 13. Pose、qpos 与频率语义

### 13.1 Pose

正式命名使用 `decoupled_cartesian_residual`：

```text
p_out = p_base + delta_p_reference
R_out = R_base * delta_R_local
```

它不是一个不加说明的通用 `SE3Residual`。translation frame 与 rotation composition
order 必须进入 schema/frame binding。

Rotation temporal warp 后续使用：

```text
delta_R(w) = Exp(w * Log(delta_R))
```

但实现优先级低于 session activation、publication correctness、shadow/sim 接线、
support expiry 和 handback。R3.1 允许 translation-only。

### 13.2 qpos

```text
q_out = q_base + w(t) * delta_q
```

coordinate order、unit、shape、mask、periodicity 必须由 schema/adapter 明确。首版不在
trajectory layer 做 joint limit 或 canonical safety。

### 13.3 多频率

- nominal trajectory 带 explicit target lattice 和 support；
- human residual 带 sample/effective time 与 TTL；
- residual 是状态量，不按 control tick 重复积分；
- current R0–R2 只选择 exact lattice keyframe，不做 nominal off-lattice interpolation；
- 每次只发布一个严格递增 due target，不提前发布 corrected future range；
- source inference/refresh 永不阻塞 control loop；所谓 `WAIT_FOR_NEWER_SHADOW` 只表示
  旧 active 在仍有效 support 内继续运行，不是同步等待 worker；
- missed cut 不 burst catch-up；
- 下游继续负责高频 interpolation、IK、限幅、reference/command resampling。

Temporal support 分层：

1. human adapter：device noise/deadband/timestamp/anchor/recenter/stale；
2. trajectory warp：activation ramp、future-target support、release decay、continuity
   correction；
3. downstream reference owner：控制器需要的高频平滑和动力学约束。

三层不能重复滤波同一个语义。

## 14. Support expiry 与 human-control fallback

优先级：

```text
1. 有 fresh compatible shadow，且自动 refresh 已通过 gate
   -> schedule future base transition

2. 暂无 candidate，但 old active 仍在 support/refresh guard 内
   -> 继续 old active；异步等待下一 source result

3. support 已到期，human active/fresh
   -> request_component_control_fallback

4. support 已到期，human inactive/stale
   -> no_action / request_hold
```

`request_component_control_fallback` 只是 source-neutral 请求，不是 authority grant。
应用层存在两种不同实现，不应在 generic reducer 中写死：

### A. Composer authority 保持不变

当前 conservative compound authority 仍由 composer producer 持有。应用通过新的
composition activation/strategy/source-role binding，把指定 component 从
`nominal + residual` 切换为 direct human absolute control。是否需要先 HOLD/re-arm
由 profile 和下游 lifecycle 决定。

### B. 退出 composer，转为 exclusive human authority

使用现有 Control Station typed `request_human_takeover` transaction：HOLD、source
invalidation、authority release、human takeover、fresh human emission、deadman rearm、
retargeter rebase 和 READY receipts。

不能用现有 full-authority takeover receipt 冒充 A，也不能为 B 再造第二套 takeover
状态机。R3.0 只定义 generic fallback request；R3.4 前必须由具体 profile 冻结采用 A
还是 B。

Direct human control 不能直接复用 residual anchor。它必须基于 authoritative seam 和
当前 device pose 重新建立 absolute-control anchor；否则 residual 会被错误解释为完整
absolute target。

## 15. Preview、记录与资源预算

四类视图严格分开：

```text
ActiveNominalTrajectoryPreview
  当前 human 正在编辑的 base

CorrectedTrajectoryPreview
  nominal + current residual + continuity correction 的非权威只读投影

LatestShadowTrajectoryPreview
  后到 candidate，仅供判断 refresh/handback

ExecutableTrajectoryObservation
  只能来自 authoritative publication、target/reference owner 的真实事实
```

UI 可进一步标注 `Executing / Published / Editable / Unreliable tail`，但每段必须绑定
实际 owner 的 identity，不能从时间位置自行猜测 authority 状态。

记录分层：

### Core replay/因果信息

- session activation binding；
- anchor receipt identity；
- active/shadow trajectory identity 和 source lineage；
- base binding revision 与 transition assessment；
- edit/publication barrier；
- residual fact、warp spec、target time、weight；
- continuity correction identity；
- exact corrected due action 与 composition/proposal lineage；
- support expiry、request/no_action/transition reason；
- 存在时的下游 receipt/reference/measurement identity。

### Decision metrics（低频/事件型）

- residual session duration / activation 时剩余 support；
- boundary-hit 与 expiry-fallback；
- candidate seam summary；
- handback/refresh outcome；
- stale/late candidate/receipt counts。

### Debug/preview-only

- 高密度完整 ghost trajectories；
- 每帧 seam/cosine/fight/jerk 诊断；
- rejected shadow 大 payload；
- UI 绘制中间值。

所有可选高频信息必须经过 `CompositionInformationPlan` 和
`ExecutionObservationPlan` 显式请求。`control_composition` 只发 protocol-neutral domain
events，不导入 dataset v2.4/v2.5、WAL 或 writer。

## 16. 修订后的实施 gates

### R0：通用 trajectory carrier——已完成

- generic immutable trajectory/keyframe；
- source lineage、schema/coverage、time lattice、support；
- source-owned timed plan 的纯 projection；
- 不接运行时。

### R1：active/shadow pure reducer——已完成

- pass-through；
- residual-active active latch；
- latest shadow；
- exact target preparation；
- receipt-bound last emission；
- `publication_basis_hash` 排除 shadow。

### R2：translation/qpos temporal sample——已完成

- Cartesian translation；
- 任意 qpos；
- state-valued `ramp_and_hold`；
- rotation fail-closed；
- 尚未接 Control Station/target。

### R3.0：协议与因果边界冻结——已完成

1. 定义 `ResidualSessionActivationBinding`；
2. 复用并绑定现有 `ResidualAnchorReceipt`；
3. 将 reducer state 拆成 interaction mode 与 base lifecycle status；
4. 冻结 interaction/base legal-combination matrix，并由同一 reducer 维护；
5. 用 immutable `NominalBaseBinding/base_binding_revision` 取代 `BaseEpoch`；
6. revision 首版 runtime 固定为 0，但 pure contract 验证 successor lineage；
7. 定义绑定 residual basis 的 `NominalBaseTransitionAssessment`；
8. 定义 switch 前 `NominalBaseTransitionRevalidation`，覆盖 residual/assessment stale
   和 publication barrier 追过；
9. 定义独立 `ContinuityCorrection` activation contract；
10. 将自动 transition capability 冻结为 HOLD-boundary/reference-only/reference+
    measurement 三级；
11. 使用 source activation identity 防止 restart/rearm 后 generation 复用；
12. 审计 publication/idempotency，不预先实现 pending state；
13. 增加 stale session/base/shadow/assessment/receipt 的纯 reducer 测试；
14. 不接 control loop，不改变现有 profile。

实现收据（2026-08-05）：

- `ComponentActionTrajectory` 增加 source instance/activation lineage；pass-through 可不带
  activation，但 residual base binding 必须带完整 ID/hash；
- `ResidualSessionActivationBinding` 原子引用 activation scope、control cut、anchor
  receipt、initial base binding 与 initial publication barrier；
- `TrajectoryEditingState` 已拆成 interaction/base 两轴，shadow 仍不进入 current
  publication basis；session/base/lifecycle/barrier 变化会使旧 preparation 失效；
- `NominalBaseTransitionAssessment` 绑定 residual/candidate/capability/observation/cut/
  future switch/available time；hard-incompatible candidate 可在不伪造 corrected seam 的
  情况下形成结构化事实；
- `NominalBaseTransitionRevalidation` 检查 base、candidate、residual、capability、Reference、
  Measurement、tracking、assessment visibility 与 publication barrier；Measurement
  freshness 使用 `measured_at` 而不是仅看晚到的 `available_at`；
- pure successor binding 记录 predecessor/assessment/revalidation lineage，但 runtime
  reducer 没有启用 revision 1；非自动 assessment 不能绕过 operator/HOLD-boundary receipt
  构造 successor；
- continuity correction 有独立 action/residual/base/reference/decay lineage，不写回 human
  residual；
- support exhausted 只返回 `no_action`，不在 composition 内请求 HOLD 或构造下游动作。

固定回归组：

```text
# trajectory carrier / reducer / lifecycle / warp
python -m pytest -q \
  tests/test_component_action_trajectory.py \
  tests/test_trajectory_editing_reducer.py \
  tests/test_trajectory_residual_warp.py \
  tests/test_trajectory_session_lifecycle.py

# activation / anchor / observation / relative / publication / RTC-shadow /
# Control Station-MuJoCo boundary（纯测试，不启动设备）
python -m pytest -q \
  tests/test_residual_anchor_contract.py \
  tests/test_execution_observation_contract.py \
  tests/test_composition_temporal_sampling.py \
  tests/test_relative_component_composition.py \
  tests/test_relative_action_schema.py \
  tests/test_action_proposal_contracts.py \
  tests/test_reference_activation_contracts.py \
  tests/test_deploy_rtc_frontend_shadow.py \
  tests/test_control_station_mujoco_sim_hil.py \
  tests/test_control_composition_boundaries.py
```

当前工作树实测：第一组 `32 passed`；第二组 `117 passed`（MuJoCo 依赖产生 1045 条
既有 deprecation/pending-deprecation warnings）。该计数只对应本次路径集合，后续必须
重新实测，不能当作永久历史总数。两组均不构成真机、真实模型服务或正式 runtime
证据。

### R3.1a：default-off virtual/shadow wiring——已完成

- virtual nominal trajectory + virtual human residual；
- 通过 source-owned canonical plan seam，而不是 raw chunk；
- `send_actions=false`；
- PASS_THROUGH/RESIDUAL_ACTIVE；
- revision 0 locked base；
- support expiry request/no_action；
- nominal/corrected/shadow preview；
- executable preview 必须为空或明确 unavailable；
- domain-event/ControlTrace 投影；
- 对现有 policy-only、absolute weighted、per-cut relative 做 hash/parity 检查。

实现收据（2026-08-05）：

- 先将原 1559 行 lifecycle 聚合文件按 session lifecycle、transition assessment、
  final revalidation、continuity correction 拆分；公共 package export 保持不变，最大新增
  轨迹协议文件控制在 648 行；
- `evaluate_trajectory_composition_shadow()` 接收 generic role、producer stamp、
  `ComponentAction` 与既有 `CompositionSpec`，不识别 policy/human、具体本体或 backend；
- RESIDUAL_ACTIVE 走真实 `CompositionInputStore` admission、activation scope、same-cut
  formal strategy 和可选 source-difference；PASS_THROUGH 不触碰 composition store；
- residual 缺失/过期、trajectory support 结束返回 typed `no_action`，不会合成 HOLD、
  authority 或 fallback action；
- latest shadow 只形成 preview，既不参与 current corrected action，也不改变
  `publication_basis_hash`；
- `TrajectoryCompositionShadowResult` 明确暴露 nominal/corrected/shadow，executable
  永远为 unavailable，且拒绝任何带 proposal admission 的结果；
- Control Station 仅增加 default-off 惰性 gate；off 时不调用 evaluation factory，
  `send_actions=true` 直接拒绝；
- ControlTrace 增加 compact shadow summary，只保存动作 hash/lineage；完整动作仍由既有
  composition input/evaluation fact 承担，避免重复高频 payload；
- 新增 virtual arbitrary-component pose+qpos 测试，不启动 sim、模型服务、设备或 motion。

本轮实测收据：trajectory/R3.1a 组 `42 passed`；现有 relative、proposal、Reference、
RTC shadow、MuJoCo Sim-HIL 纯测试及 package boundary 组 `125 passed`，仅有 1045 条
既有第三方 deprecation warning；v2.4 ControlTrace/信息计划/组合投影组 `42 passed`。
这些仍只证明本地 virtual/pure/shadow 兼容，不是真实模型、interactive sim 或真机证据。

### R3.1b：最小 locked-chunk MuJoCo Sim-HIL——publication slice 已完成

本阶段拆成两个边界，不再把 proposal publication 与 interactive application 混为一个
完成条件。

已完成的最小 publication slice：

- 显式 default-off、sim-only profile；off 时不读取 request，且拒绝含糊的
  `enabled/send_actions` 组合；
- 单臂 Cartesian translation residual，hand/qpos component 仍走通用 component-keyed
  carrier；
- 每次只准备并发布一个 exact target；
- trajectory-derived nominal/residual 先进入共享 `CompositionInputStore`，正式
  evaluation 必须选中同一组 input ID/hash，不能被并发更新静默替换；
- 复用现有 composition、compound authority 与 `ProposalAdmissionReceipt`；只有 proposal
  接纳后才推进 trajectory emission barrier，并记录实际 corrected action hash；
- publication controller 不调用下游 `runtime.step()`。future proposal 可以提前接纳，
  但 authority selection、target resolution、reference generation 与 dispatch 仍由既有
  MuJoCo control loop 在目标时刻独占执行；
- support expiry 返回 typed `no_action`；重复 target 不产生第二个 proposal；
- 不构造真机 backend，不接模型 server，不轮询真实遥操设备。

来源能力分流不需要新增模型适配器：当前单步 PPO 继续输出普通
`PolicyActionEmission`，走现有 per-cut relative composition；只有已经提供真实
`ComponentActionTrajectory` 的 source 才进入 locked-base 路径。单 keyframe carrier
只能发布该 exact target，不产生 future successor 或插值保证。R3.1b 使用 virtual timed
policy 验证 chunk 语义，不修改或伪装当前单步 PPO。

application lifecycle seam 已完成的最小部分：

- enabled trajectory publisher 必须绑定已由 typed
  `start_composition_session` result 打开的既有 `StationPublicationBarrier`；
- publisher 绑定 barrier generation 与 operation-result identity，后续 typed re-arm
  不能复活旧 trajectory state/controller；
- virtual integration 已验证 pause 的顺序为“先关闭 publication，再下层 HOLD/cancel”，
  barrier 关闭后不会调用 request supplier；held session 可以再由 typed stop result
  进入 ended；
- request supplier 只允许读取已经生产完成的 mailbox/source fact，不允许在 barrier 内执行
  模型推理、设备轮询或 source cursor 推进。
- v2.4 增加 generic `component_action_trajectory` source fact，由 source owner 对每个
  immutable plan 记录一次完整 keyframe action；它属于复现所需实际数据，不是每 cut
  重复的 debug payload，也不声明 proposal/target/reference/dispatch admission；既有非
  trajectory profile 维持 optional，未来 timed-trajectory capture profile 再显式 require。

仍待 interactive application gate 完成：

- 把既有 Station typed operator intent/readiness 绑定为 residual session 的
  clutch/deadman/release，而不是在 publication controller 内复制设备状态机；当前只验证
  typed session controller 与 virtual intent/result 语义；
- 把 application-owned ControlTrace capture/seal/readback 接到该 profile；generic source
  trajectory fact 已补齐，但 episode profile、durable seal/readback 尚未闭环；
- 验证真实 application coordinator 的 late worker completion、typed re-arm 后新
  trajectory activation 与 session teardown；
- 不在本阶段实现 R3.2 release handback/base transition。

R3.1b 开始时的 live-code readiness 核对与修正（2026-08-05）：现有 MuJoCo runtime
原本已经能按显式 `ExecutionObservationPlan` 输出 active qref Reference、measured qpos
Measurement 与各自 watermark，原始 representation 均为 `joint_position`。这不是
pose-space 可观测性的根本阻塞，因为持有 MuJoCo model、joint layout、末端 frame 与
tool offset 的下游可以做确定性 FK。正确的数据流是：

```text
composed arm pose
  -> downstream IK / limit / planner
  -> active qref
  -> FK(active qref) = pose Reference

measured qpos
  -> FK(measured qpos) = pose Measurement
```

FK 的输入是 qref/qpos，不是已经为 pose 的上游 action。这样可以分别观察“上游 pose
到下游实际接纳 qref 的实现差异”和“active qref 到 measured state 的跟踪差异”。

当前已补齐 additive、default-off 的下游 pose observation seam：

- `ExecutionObservationPlan.required_representation_by_component` 可显式要求
  `cartesian_pose`；未声明时现有 qpos capability 和数值路径不变；
- MuJoCo 使用 private `MjData` 做 FK，不写 live simulator，也不在
  `control_composition` 中引入 FK/IK；
- execution Rig 在同一把 owner lock 下捕获 active qref、measured qpos、model qpos
  background 与 `sampled_at`，避免拼接两个不同执行时刻；
- pose Reference/Measurement 使用上游 arm component 的 exact pose payload schema
  hash，因此 reference frame、controlled frame、shape 和单位可比较；
- 每个派生 pose fact 保留原始 qref/qpos observation 与 kinematics/model contract
  parent lineage；
- generic execution-difference observer 已支持 pose translation residual 与 local-frame
  rotation-vector residual；结果仍是 non-authoritative observation；
- projector 按 component、qpos source、joint names、body frame 和 controlled-frame
  offset 绑定，不把双臂、六关节或固定 component 名写进通用 FK kernel。

publication/lifecycle slice 实现收据（2026-08-06）：

- reusable input preparation 位于 `control_composition/trajectory`，只负责 exact target
  derivation/admission，不拥有 proposal、authority、target、IK 或 dispatch；
- Control Station 的 `MujocoTrajectoryPublicationController` 只做 application profile、
  exact input selection、proposal receipt 与 reducer emission commit；
- controller 复用已有 typed composition-session publication barrier，并绑定单次
  activation generation；pause 先禁发后 lower HOLD/cancel，旧 controller 不能跨 re-arm；
- integration fixture 在 proposal publication 后先证明无 Reference/Measurement，再到
  target time 调用既有 MuJoCo loop，得到正式 target/dispatch；
- pose Reference 精确绑定该下游 `ComponentExecutionTarget`，其数值由
  `FK(active qref)` 得到；pose Measurement 由同一 execution cut 的
  `FK(measured qpos)` 得到。两者都保留 raw qref/qpos 与 FK contract lineage，未从上游
  pose 倒填执行结果。

本轮本地验证收据：trajectory/reducer/shadow/session/publication `53 passed`；Sim-HIL、
pose observation、package boundary `46 passed`；ControlTrace registry/routing/runtime/
episode `40 passed`。这些只证明本地
virtual/sim/contract 兼容，不是真实模型、interactive device、durable capture 或真机
证据。

该实现解决了“当前执行 cut 的 pose Reference/Measurement”缺口，但不声称提前得到未来
`switch_at` 的 Reference。未来 handback/base-refresh 仍必须使用下游在对应时间点正式
发布的 observation，不能由 composition 外推或重算未来 reference。

因此 R3.1b 的下一步是 application lifecycle 与 durable trace 接线；R3.2 的 future
switch/handback publication 仍是后续独立 gate。

### R3.2：release handback

- fresh compatible shadow/current base gate；
- old scheduled reference seam；
- measurement freshness/tracking validity；
- freeze last residual；
- residual decay 与 continuity correction 分离；
- handback cancellation、expiry 和 terminal receipt；
- 无可用 base 时 no_action/request HOLD；
- 仍不启用 active-session 自动 base refresh。

### R3.3：条件启用 active-session base refresh

只有 R3.1/R3.2 数据证明以下问题明显时实现：

- residual session 经常超过 initial active support；
- boundary-hit/expiry-fallback 比例高；
- locked suffix 随时间明显 stale；
- latest shadow 在 future boundary 经常具有可接受 corrected seam。

实现：

```text
RESIDUAL_ACTIVE(binding revision n)
  -> BASE_REFRESH_PENDING
  -> future switch_at
  -> BASE_TRANSITION
  -> RESIDUAL_ACTIVE(binding revision n+1)
```

不实现任意时刻 newest-shadow replacement，不默认增加多-candidate buffer。

### R3.4：component-control fallback

- profile 冻结 composer-stays-authoritative 或 exclusive-human-authority；
- direct-control reanchor；
- component set 与 source-role transition；
- 必要的 HOLD/re-arm/READY receipts；
- arm、gripper、hand 的不同 fallback semantics；
- trajectory reducer 仍只发 request。

### R3.5：数值和 topology 扩展

- rotation temporal warp；
- periodic qpos；
- hand qpos mask；
- gripper discrete override；
- conditional coordination group；
- local warp/rate steering；
- contact-sensitive profile；
- learned/advisory transition policy，且不得阻塞 control loop。

### R4：Interactive sim 稳定后再评估 real shadow

真机只允许先 observation/shadow，不授予 motion permission；必须单独完成 real target
capability、reference/measurement、deadman/rebase、canonical safety 和 dispatch owner
验收。本文不授权真机 motion。

## 17. 验收矩阵

### 17.1 纯 reducer/虚拟策略

1. human inactive 时每个 due action hash 与 nominal 相同；
2. session activation 原子绑定 anchor/base/cut/edit boundary；
3. cut 后 shadow 不改变 active binding 或当前 publication basis；
4. same residual 重算不积累；
5. translation 使用 reference/world frame；
6. rotation 在 R3.5 前继续 fail-closed；
7. qpos 任意维度且 schema/order/unit 不静默变化；
8. optional/required source 与 TTL 行为保持现有 contract；
9. target 只 preparation 一次正式 finalize，重复/回退 target 被拒绝；
10. stale session、旧 base binding、旧 candidate assessment 和 late receipt 被拒绝；
11. residual 与 continuity correction identity 永不合并；
12. shadow buffer 始终有界为 latest-one；
13. no next/unguaranteed next 保持当前 per-step 行为；
14. 不同非整数频率比不要求整数对齐；
15. missed cut 不 burst catch-up；
16. support 到期不 last-value forever；
17. reducer 重放同一 event sequence 得到相同 state/hash。

### 17.2 Default-off/shadow

1. profile 关闭时现有数值、顺序、pop count、emission hash 不变；
2. `send_actions=false` 不构造 target/reference/dispatch；
3. corrected preview 不含虚构 execution claim；
4. executable observation 在无下游 receipt 时明确 unavailable；
5. recorder/debug 关闭时不生成高密度 trajectory payload；
6. late shadow/assessment 不阻塞 control loop。

### 17.3 Sim-HIL

1. typed clutch activation/release 与 composition activation lineage 完整；
2. locked base 避免 policy replan 造成 moving-base；
3. Reference seam 精确覆盖 switch/handback boundary；
4. Measurement 使用 association 和 freshness，而不是 nearest timestamp 猜测；
5. tracking invalid 时拒绝自动 handback并返回明确 request/no_action；
6. support expiry 不自动授权 human；
7. proposal/authority/target/reference/dispatch 仍各有唯一 owner；
8. ControlTrace core facts 可 seal、commit、readback；
9. sim backend 停止、episode end、pause、late completion 不重新开放旧 publication。

## 18. 对新建议的采纳判断

### 18.1 直接采纳

- ResidualSession 与 nominal base lifecycle 分离；
- 两个正交状态维度；
- activation 使用统一 causal cut；
- active/shadow 分离和 latest-one shadow；
- future switch boundary；
- same-residual corrected seam；
- human residual 与 continuity correction 分离；
- Reference continuity / Measurement validity 的职责区分；
- source-owner canonical plan 之后再投影 trajectory；
- nominal/corrected/shadow/executable 四类视图；
- 自动 refresh 由 Sim-HIL 数据驱动，首轮不默认启用；
- rotation warp 低于 causal/runtime/handback correctness 的优先级。

### 18.2 修正后采纳

- `BaseEpoch` 改为 `NominalBaseBinding + base_binding_revision`；
- typed human anchor 不新增，复用现有 generic `ResidualAnchorReceipt`，只增加 session
  activation binding；
- hard compatibility 从“永久 exact hash equality”改为“必须能由声明的 schema/frame/
  adapter 无歧义对齐”；首版 profile 可保持 exact；
- producer generation 不要求相等，只要求 activation、order 和 stale lineage 正确；
- component coverage 要求覆盖当前所需 component，只有 atomic profile 才要求完整
  exact group；
- coordination identity 仅对声明 coupled 的来源强制；
- Reference/Measurement 不是所有 backend 的必选项，而是自动 transition capability；
- `WAIT` 不允许阻塞 control loop；
- human-control fallback 根据 composer authority 与 exclusive authority 分成两种应用
  realization。

### 18.3 暂不采纳或否决

- 不新增顶层 `shared_control/` 包；当前 owner 分布仍以 `robot_motion/action` carrier、
  `control_composition/trajectory` reducer、Control Station wiring 最清晰；
- 不新增第二套 human takeover、anchor、reference 或 publication 状态机；
- 不使用一个 `committed_until` 或未经 owner 证明的 through-watermark 统摄全部阶段；
- 不在审计前增加 `PendingEmissionAttempt`；
- 不保留多个 shadow 或实现“最佳 shadow”选择器；
- 不让 trajectory layer 做 device low-pass、canonical safety、IK 或下游 smoothing；
- 不让 residual 自动升级成 absolute human control；
- 不将 corrected preview 冒充 executable reference；
- 不把 automatic base refresh 作为 R3.1 的前置阻塞项；
- 不默认实现 learned intent/conflict classifier。

## 19. 下一步与阻塞条件

R3.1b 的 single-target publication 与实际 Sim target/FK observation 已闭环，下一步不再
修改 reusable composer 数值语义，而是进入 application-owned integration：

1. 复用既有 Station operator intent、human source readiness、HOLD/re-arm 与 episode
   lifecycle，建立 residual session 的 typed clutch/deadman/release binding；
2. application 按 source fact 分流：current-action-only 继续既有 per-cut 路径；真实 timed
   trajectory 才由 control loop 按 target lattice 调用 publication controller；controller
   不轮询设备、不推进 policy/chunk cursor，也不调用 lower `runtime.step()`；
3. application trace sink 复用既有 ControlTrace capture/seal/readback。trajectory summary、
   composition/proposal/target facts 与 execution observation 必须通过各自既有 owner fanout，
   不在 publication controller 中新增 dataset writer；
4. 增加 pause、episode end、support expiry、duplicate/late request、HOLD/re-arm 与 teardown
   的 virtual-device robustness fixtures；
5. 保持 profile default-off，并继续证明现有 policy-only、absolute weighted 与 per-cut
   relative 路径 hash/顺序不变。

仍冻结的边界：support expiry 只返回 request/no_action，不执行 direct-control fallback；
R3.1b 不做自动 handback；R3.2 自动 handback 才要求在对应 switch cut 上取得可关联的
Reference/Measurement 与 tracking validity。

在进入 R3.4 前必须由 profile 决定 fallback 使用 composer-stays-authoritative 还是
exclusive-human-authority。这个选择影响 Control Station operation binding 和 HOLD/
re-arm 流程，但不阻塞 R3.0、R3.1a 或 R3.2 的纯协议与 shadow 工作。

## 20. R3.1c：未来预览与动态 allocation（当前实施计划）

这两项是通用 HITL 的核心功能，不属于 MuJoCo、RTC、某个 policy checkpoint 或某种
遥操设备。实现按以下两个正交平面组织：

```text
source intent plane
  ComponentActionTrajectory
    -> whole-horizon intent preview
       - producer_locked_intent
       - speculative_prediction

composition decision plane
  source activity facts + one ControlCut
    -> component-keyed allocation decision
       - base_only
       - relative_overlay
       - absolute_overlay_owner       # soft takeover; composer 仍是唯一 authority
```

### 20.1 未来预览的冻结语义

首轮预览只陈述 source owner 已经提供的 intent：

- `producer_locked_intent`：该 keyframe 位于 source 的不可改写前缀内；它仍不承诺
  proposal、target、Reference 或真实执行；
- `speculative_prediction`：该 keyframe 仍可被 source owner 的新结果替换，适合以虚线、
  半透明或 trend 形式显示；
- `source_immutable_through` 只用于上述分类，不升级成“保证执行”；
- snapshot 绑定 exact trajectory revision/hash、完整 keyframe hash 与 preview 生成时刻，
  不读取或推进 source queue/cursor；
- 下游真实 `Reference`/`Measurement` 只能作为独立 execution-observation overlay 显示，
  不能由 composition 伪造为第三种未来 intent。

首轮不把当前 single-target corrected shadow 循环到整个 horizon。当前
`CompositionInputStore`、context store 与 eligibility store 的正式 snapshot 都会推进
closed-target watermark；若直接循环 future target，会让纯预览改变 production 时序。
corrected horizon 必须在后续 gate 使用：

```text
one explicit ControlCut
  -> immutable non-closing input/context/eligibility projections
  -> all target evaluations over the same frozen projections
  -> one coherent corrected-preview snapshot
```

该接口必须保证不可被 `admit_evaluated_composition()` promotion，且不得修改 retention、
closed watermark、proposal、authority 或 emission barrier。

### 20.2 动态 allocation 的冻结语义

动态 allocation 不写死 policy/human，而由 profile 预声明三个 source role：

```text
base_absolute_role
relative_overlay_role
absolute_overlay_role
```

每个 component 独立产生以下 soft composition mode：

```text
base_only
relative_overlay
absolute_overlay_owner
```

其中 `absolute_overlay_owner` 只表示 composition 输出选择该 absolute source，不发生
authority owner 切换；高频阈值变化不得触发 Station HOLD/re-arm transaction。真正的
exclusive/hard takeover 仍由 Control Station 的 typed handoff lifecycle 独占管理。

activity 输入使用 bounded time window，而不是单帧位置阈值。通用 reducer 只依赖
归一化 activity level、validity/TTL 与 source-fact lineage；标准 adapter 可为
`cartesian_pose` 和 `joint_position` 计算 translation、rotation、qpos displacement/path/
rate，其他 representation 通过显式 metric provider 扩展。状态切换必须有：

- residual/takeover 两级 enter/exit 阈值；
- hysteresis；
- minimum dwell；
- stale/missing 与“fresh but static”严格区分；
- component-keyed decision 与输入 fact lineage；
- reducer 重放确定性。

source required/optional、TTL 过期后的 `no_action`/HOLD 仍由既有 eligibility、
CompositionSpec 和下游处理；allocation 不复制另一套安全或 source-liveness 状态机。

### 20.3 分步落地 gate

1. **R3.1c-A（已完成）**：新增 pure whole-horizon intent preview contract/builder；新增
   generic activity observation 与 deterministic allocation reducer；补 pose、任意维 qpos、
   stale、hysteresis、dwell、重放和 source-locked/speculative 边界测试。
2. **R3.1c-B（已完成 reusable core）**：新增 predeclared three-role formal allocation
   rule；ready decision 通过既有 `CompositionContextStore` admission，并与 action inputs 在
   formal evaluation 的同一 `ControlCut` 被选择。不用低频 strategy revision 模拟高频
   切换，不改变既有 alpha-only hot-control 契约。尚未接 application profile。
3. **R3.1c-C（已完成 reusable core）**：新增 non-closing frozen-cut
   projection/evaluator，形成 coherent conditional corrected horizon；preview 不推进任何
   closed watermark，不伪造 admission/receipt，且没有 promotion/submission 接口。独立的
   source-intent freshness assessment 只使用显式 observation/availability evidence，不从
   trajectory `created_at` 推断观测年龄。
4. **R3.1c-D（D1/D2/D3/D4 已完成）**：`control_composition/allocation/` 已完成
   receipt-driven publication runtime；Control Station 已增加 default-off typed worker
   port，只读取已产生的 basis，并在 application-owned worker 上提交 exact context draft。
   Station 既有 default-off、只读且 lazy 的 intent/corrected-preview port 保持不变；
   freshness evidence 只有在 profile 明确请求时才获取。MuJoCo D4 只在同一 Station
   barrier 内重做正式 cut evaluation 并提交 exact proposal，不执行 lower step。后续
   replay、deploy 和真机 adapter 只负责 source/context binding，不实现 allocation 状态机。
5. **R3.1c-E**：接 UI/trace。实际数据记录 trajectory revision、allocation decision 与
   必需 lineage；高密度中间 metrics/candidate matrix 仅在 debug information plan 下保留。

R3.1c-C 采用开发态、可继续演进的公共契约。正式 input store 中的
`CompositionInputRecord` 表示已经过 receiver admission 的单 target 输入；source 已发布的
`ComponentActionTrajectory` 是一个整体 fact，其未来 keyframe 不是逐帧 admitted record。
实现明确禁止以下两种捷径：

- 为预览伪造 `CompositionInputRecord`、receiver admission sequence 或 receipt；
- 为了显示而提前把整段未来 keyframe 正式 admission，进而改变 retention、closed target、
  producer sequence 或 production source selection。

已落地的第三种方案新增只读 `ProjectedCompositionInput`，由 exact trajectory
revision/keyframe、已冻结 residual/context/eligibility fact 和同一 `ControlCut` 派生；“按
target 选择输入”的纯逻辑从 admitted carrier 中抽出，使 production
`CompositionInputRecord` 与 preview projection 共享 selector，但保持不同 identity/hash
domain。每个 projected snapshot 还显式绑定 producer-epoch snapshot 与 expected
activation，避免空选择或失效点丢失 frozen-cut 因果身份。preview evaluator 只返回不可
promotion 的 result，不暴露 `CompositeProposalSubmission`。allocation mode 只沿当前
context 的实际 `effective_until` 投影，绝不预测尚未发生的人类 activity；超过 input、
context、maximum-age、source lease 与 residual 的 support intersection 后明确返回
`unavailable`。

当前仍有三个明确的 application/downstream gate，而不是 composition 算法缺口：

- hot strategy-control 与 pre-mix policy 尚无 projected protocol，因此相关 runtime
  fail closed，不偷跑普通 strategy；
- default-off MuJoCo shadow、最小 UI/trace fact 与 D4 exact proposal gate 已接入；实际
  GUI/application coordinator、真实 source/device binding 与 durable seal/readback 尚未接入；
- Reference/Measurement、真实执行概率和 authoritative future 仍由下游事实提供，不能由
  corrected preview 推断。

### 20.3.1 R3.1c-D allocation publication 冻结语义

allocation 数学 reducer 保持纯函数；新的 publication runtime 只拥有 allocation scope 内的
reducer commit、context producer sequence 和接纳重试一致性：

```text
accepted activity/residual facts + one ControlCut + committed reducer state
  -> pure evaluation
  -> immutable decision + candidate next state
  -> one immutable publication attempt
  -> CompositionContext admission receipt
       accepted new / reconciled exact duplicate
         -> commit exact saved candidate next state
       terminal rejection
         -> keep old state; consume the reserved sequence
       exception / unknown outcome
         -> retain and retry the exact same draft; never re-evaluate it
```

首轮明确采用以下约束：

- 一个 allocation scope 只有一个 in-flight attempt，并只保留一个 latest coalesced basis；
  pending 不阻塞 control loop，也不逐帧补算或 burst catch-up；
- activity basis identity、allocation publication sequence 和 immutable attempt identity 是
  三个不同的 identity；attempt 保存 exact decision、candidate state、draft、ControlCut、
  activation 和相应 hash；
- 当前 `CompositionAllocationDecision` 已经原子覆盖 component set，因此不新增同义的
  `DecisionSet` carrier；真正独立的 component 组使用不同 allocation scope/runtime；
- 普通 `CompositionContextStore.admit()` 的 duplicate 仍是 rejection，不改变既有通用
  admission 契约。只有 publisher 取回既有 record、验证 producer-side submission hash 与
  pending draft 完全一致后，才可将其解释成 `already_accepted_same_fact`；同 identity 不同
  payload/hash 是 terminal contract violation；
- sequence 只要求严格递增而非连续。terminal rejection 消费已保留 sequence，下一次使用
  更高 sequence；rejected candidate 不推进 hysteresis/dwell state；
- attempt 绑定通用 `CompositionActivationScope` 和 allocation producer epoch。Station 的
  publication barrier 只在 application adapter 中约束该 activation，不写进通用 allocation
  协议；pause/re-arm/episode end 使旧 runtime/pending invalid，late receipt 不能推进新
  activation；
- `effective_from >= evaluated_at`；`effective_until` 是 evidence validity 与调用方显式提供的
  source/session/activation/decision support 上界的交集。runtime 不把 stale/missing 默认为
  `base_only`，不制造 HOLD、authority、safety、IK、Reference 或 dispatch；
- 正式 corrected preview 只读取 store 已接纳的 allocation context。pending candidate 仅可
  作为 debug 状态暴露，不能进入 active/corrected/executable 结果；
- 首轮成熟度只声明 in-process receipt-driven commit 与 exact retry reconciliation，不声明
  crash-consistent exactly-once。attempt、receipt、previous/candidate state hash 保留为后续
  durable ControlTrace readback 的恢复依据。

实施顺序固定为：

1. **D1 pure/runtime contract**：immutable attempt、single pending、exact retry、receipt
   classification、commit、sequence gap、activation invalidation、latest-only coalescing和纯测试；
2. **D2 Station default-off port**：只读取已经产生的 typed basis，在 application-owned
   bounded worker 提交 exact draft并回传 outcome；关闭时不读取 supplier、不轮询设备/模型、
   不推进 cursor、不改变 authority；
3. **D3 MuJoCo shadow（已完成）**：virtual source、`send_actions=false`，验证同 cut
   context selection、corrected preview 与 UI/trace；
4. **D4 default-off Sim-HIL publication（已完成最小 gate）**：通过现有 composer
   proposal 路径接纳 exact evaluation，仍不产生 exclusive takeover，也不调用
   `runtime.step()`；最后才接真实遥操设备驱动 sim。

D1 验收必须覆盖：evaluation 未接纳前不修改 state；accepted 后只提交一次 exact candidate；
异常后 draft/fact/sequence/hash 不变；exact duplicate 可经 record reconciliation 提交；冲突
duplicate 和 terminal rejection 不提交但不复用 sequence；pending 只合并最新 basis；旧
activation receipt 不推进新 state；support 到期不继续选择；preview 不使用 pending，也不推进
closed watermark。

D1 首轮实现收据（2026-08-06）：

- `publication_contracts.py` 保存 immutable basis/attempt/outcome/snapshot，
  `publication.py` 只保存 allocation scope runtime 和 admission/commit 状态机，避免把协议、
  reducer 和 application wiring 聚成大文件；
- basis 绑定一个 `ControlCut`、通用 `CompositionActivationScope`、component-keyed activity/
  residual facts、requested effective start 与外部 support 上界；cut 之后才产生的 observation
  不能混入该 basis；
- runtime 只在 accepted new 或 exact-record-reconciled duplicate 后提交 pending 中保存的
  candidate state；exception/unknown 保留同一对象与 draft，terminal rejection 清除 pending、
  保留旧 state 并消费 sequence；
- generic context 新增 producer-side submission semantic hash 与 retained-record lookup；hash
  排除 receiver-owned availability/admission stamps 和 debug-only audit metadata，普通 duplicate
  admission 行为不变；
- 一个 pending admission 在外部 callback 中等待时不持有 runtime lock；新 basis 只保留最新
  一项。activation invalidate 后的 late accepted receipt 只形成 orphan/late outcome；
- allocation context 增加可选 scope/basis/cut/activation binding；formal allocated strategy
  将其与当前 input snapshot activation 对齐，旧 activation context 在新 activation 下 fail
  closed；
- focused allocation/context/formal/preview/boundary 回归为 `100 passed`，其中 publication
  专项 `8 passed`。验证为本地 pure/in-memory contract，不是 Station、MuJoCo、设备、模型、
  authority、motion 或 durable crash recovery 证据。

D2 首轮实现收据（2026-08-07）：

- `control_station/composition/allocation_publication.py` 只拥有 profile switch、一个唤醒
  worker 和有界 outcome cache；exact attempt、latest basis、sequence 与 reducer commit
  仍全部由 `AllocationPublicationRuntime` 拥有；
- profile 关闭时不创建 worker、不读取 basis supplier；启用时只接收已经产生的 typed
  `AllocationPublicationBasis`，控制调用不会等待 context admission；
- 一个 admission 阻塞期间，新 basis 仍由 reusable runtime latest-only coalesce，不建立
  Station 第二份队列；unknown outcome 不自动自旋，只有显式 `request_retry()` 才重试同一
  draft/sequence/hash；
- port 绑定 active typed composition-session barrier generation。关闭时先 invalidate reusable
  runtime，再 join worker；in-flight callback 的迟到 receipt 只能形成 late outcome，不能提交
  reducer state；
- Station publication-barrier observer 的去重、回调失败策略已收敛到单一被动 port；旧
  human/replay/policy profile 的 operation/authority 语义未合并或改变；composition projection
  与 allocation package 的无调用者中间 re-export 已压平；
- focused Station/allocation/composition/trajectory/boundary 回归为 `110 passed`。验证仍为本地
  pure/in-memory/worker contract，不是 MuJoCo、设备、模型、authority、motion 或 durable
  ControlTrace 证据。

D3 首轮实现收据（2026-08-07）：

- `control_station/composition/mujoco_allocation_shadow.py` 是单一 default-off concrete
  profile binding；它复用 Station publication port 与 MuJoCo 已有 formal composition
  topology，不复制 allocation reducer、context store、pose residual 运算或 trajectory
  preview builder；
- allocation basis cut 只负责产生候选。receipt 正式接纳后，后续 evaluation cut 才能选择该
  context；evaluation 的 inputs/context/producer state 由同一个 `ControlCut` 冻结，不能把
  admission 前的 basis cut 与 admission 后的 observation cut伪装为同一 cut；
- 人为阻塞 context admission 时，shadow evaluation 明确返回 `no_action`，pending candidate
  不可见且 reducer state 不推进；解除阻塞后，新 cut 原子选择两个 pose component 的
  `relative_overlay`；
- corrected whole-horizon preview 继续使用 reusable non-closing projection，并逐点核对
  allocation context 是 MuJoCo receiver store 中已接纳的 exact record；preview 保持
  non-authoritative、non-promotable；
- 新增通用 `allocation_publication_outcome` ControlTrace actual-data projection，记录 outcome、
  attempt/decision/context identity 和 committed-state hash；高密度 activity metrics 仍未默认
  进入 actual-data trace；UI latest status 与 trace cache drain互不覆盖；
- focused allocation/trajectory/ControlTrace/Station boundary 回归为 `75 passed`，其中 D3
  专项与 publication 为 `16 passed`。测试使用 virtual MuJoCo runtime/fake scene；proposal
  count 与 physics command count 均为 0，不是实际 MuJoCo task、设备、模型、authority、
  motion 或 durable episode seal/readback 证据。

D4 首轮实现收据（2026-08-07）：

- `control_station/composition/mujoco_allocation_publication.py` 是独立于 D3 shadow 的
  default-off proposal gate；D3 cycle 仍明确 non-promotable，D4 在同一个 typed Station
  barrier 内捕获新 cut、正式求值并只提交该 exact evaluation；
- missing/pending/expired allocation context 返回 `no_action`，不构造 HOLD、安全判决或
  direct-control fallback；ready result 复用既有 `submit_evaluated_composition()` 的
  evaluation identity/idempotent admission，不调用 lower `runtime.step()`；
- ordinary composer promotion scope 从 exact `ComponentAction` 与 runtime catalog 推导，
  不再默认写死 `left_arm/right_arm + arms`；compound profile 仍要求其预声明的全覆盖 scope；
- formal strategy 直接使用的 receiver context 现在进入 proposal `context_ids` compact
  lineage，避免 allocation decision 在 evaluation 到 proposal 的边界丢失；完整 context
  payload 仍由原 context/evaluation fact 持有；
- focused composition/D3/D4/boundary 回归为 `51 passed`；allocation/D3/D4/MuJoCo runtime/
  compound authority/既有 trajectory publication 回归为 `81 passed`。virtual fixture 中
  exact proposal count 为 1、physics command count 为 0；这不是 GUI、真实 task、设备、
  模型、authority selection、target/reference/IK/dispatch、motion 或 durable seal/readback
  证据。

D5 application 接入前的结构核对（2026-08-07）：

- 现有 `control_station/sim_hil.py` 同时承载 lazy factory、source warmup、viewer、GUI
  status、control cadence、pause/re-arm/episode lifecycle 与 fixed-alpha compatibility
  profile，当前约 4970 行；`composition/mujoco_sim_hil.py` 也同时承载旧 authority gate、
  input admission、formal evaluation、proposal 与 execution cycle，当前约 2950 行；
- 现有 application `step()` 直接调用旧 composition cycle，后者会在一次调用中完成
  evaluation、proposal，并继续进入 lower execution。D4 则刻意只到 proposal admission，
  不能作为同名 shadow/result 被原地替换或与旧 proposal producer 并行；
- 现有 residual profile 仍保留旧 `CompositionAuthorityGateSpec` 的 missing/invalid/error
  直接请求 HOLD 与 `canonical_safety_required` 声明。它属于 compatibility profile，不应被
  复制到 allocation 新路径；新路径的 missing/expired 仍只返回 `no_action`，由 lower
  persistence/execution gate 决定是否 HOLD；
- 因此 D5 不应继续把 allocation worker、activity window、anchor/rebase 与 D4 proposal
  gate 塞进上述大文件。推荐先保留现有 fixed-alpha application 行为不变，抽出一个小型
  typed control-cycle port，再以独立 default-off allocation profile 组合现有 source、
  Station session barrier、D2 context worker、D4 proposal gate 和既有 lower step；公共 CLI/
  GUI entrypoint 仍保持原路径；
- 该整理涉及生产 application 文件迁移与 compatibility facade，实施前需要单独确认移动
  范围；在确认前不改现有 `sim_hil.py`、真实设备构造或 execution cycle。

D5a typed control-cycle seam 实施收据（2026-08-07）：

- `control_station/sim_hil.py` 已先按应用 concern 完成无行为拆分；稳定 CLI/facade 保留，
  core lifecycle 位于 `sim_hil_profile/application.py`，没有把通用 composition 算法或 lower
  execution 搬入 Station；
- 新增小型 `SimHILControlCyclePort`。application 只根据 port spec 决定是否需要 policy
  source，以及旧 application-local fallback 是否属于当前 profile；一个 cycle 的输入接纳、
  composition/proposal 与 lower execution 仍由被选择的 port 及 lower owner 实现；
- 现有 factory 显式绑定 `FixedAlphaCompatibilityControlCyclePort`，其 `run_cycle()` 精确透传
  原 `composition.step()` 参数和返回对象。RTC、replay、teleop、真机、dataset 与公开 CLI
  均未改动；
- lifecycle 顺序已固定为：composition activation + fresh input prime 后 activate port；HOLD
  和 re-arm lower transition 前先 deactivate port；lower composition/runtime close 前先 close
  port。该顺序为 D2 worker/barrier 的后续接入留出唯一 fence/join 位置；
- allocation profile 禁用旧 fixed-alpha 的 blend-failure 与 human-timeout alpha fallback。
  missing/pending/expired allocation 仍必须沿 D4 返回 `no_action`，不能在 application 层改写
  为 HOLD、source latch 或安全判决；
- 专项与兼容回归：control-cycle `3 passed`、GUI/acceptance `78 passed`、MuJoCo Sim-HIL
  `34 passed`、allocation/publication `20 passed`、Station boundary `8 passed`。均为本地
  virtual/contract 证据，没有设备、模型、viewer、physics、authority selection 或 motion。

D5b source binding 首轮实施收据（2026-08-07）：

- profile 固定三个 role，但 role 名仍是 profile data：policy 产生 `base_absolute_role`；
  同一个 human source sample 由 source adapter 一次性派生
  `absolute_overlay_role` 与 `relative_overlay_role`。allocation 只选择 mode，不让一个
  source 在运行中改变 role；
- 两个 human role 使用 `CompositionInputBatchDraft` 原子接纳，共享 source lineage、
  receiver `available_at`、effective interval、activation 及 exact residual-anchor binding。
  任一 role 失败则两个都不可见；
- `AllocationEvidenceWindow` 按时间窗口保留接纳后的 sample pair；cut 只有同时覆盖 sample
  time、receiver availability 和 batch 最后 admission sequence 才能看见该 pair。标准
  provider 支持 pose 与任意维 continuous/qpos，具体 profile 也允许显式注入带 identity 的
  custom provider；
- motion activity 基于 absolute human window，residual magnitude 基于同一 sample 派生的
  anchor-relative action。两者都绑定同一 `ResidualAnchorReceipt`，不会把“缺失/过期”解释成
  “静止”；
- 输入的非 debug semantic binding 已进入 input hash。v2.4 基础
  `composition_input`/`composition_contributor_input` record 保持原 schema，binding 另存为
  `composition_input_semantic_binding` actual-data lineage fact；
- corrected trajectory preview 使用相同 binding 校验。future residual/absolute projection 与
  allocation context 不是同一 anchor 时只返回 `no_action`，不会放宽 preview 的因果约束；
- 新 MuJoCo policy/human allocation profile、human input controller 与 worker/publication
  binding 均 default-off、`send_actions=false`，没有接 GUI、公开 Sim-HIL factory、真实设备、
  model worker、proposal authority 或 execution。

D5b lifecycle gate 已落地为独立的低频 receiver stream，而不是复用普通 TTL context：
`ResidualAnchorLifecycleStore` 接纳 target-fenced pending/ready，formal allocation strategy
在同一 cut 校验 lifecycle、allocation context 与 selected inputs 的 exact receipt identity。
Station 的请求顺序固定为 pending admission -> allocation generation fence -> adapter commit；
首个 fresh complete sample 的 absolute/identity-residual batch 接纳后，才允许 exact ready。
旧 generation 的 in-flight context receipt 只能作为 late evidence，不能提交 reducer state。
该闭环仍未接入公开 Sim-HIL application/GUI/真实设备，因此只能称 generic core 与
default-off MuJoCo fixture 完成，不能称实际操作链路已验证。

D5c application control-cycle 首轮实施收据（2026-08-07）：

- 新增独立 `AllocationProposalExecutionControlCyclePort`，只在 application 已建立 exact
  `CompositionActivationScope` 后惰性创建一个 activation-local binding；pause、episode end、
  re-arm 都先 fence 旧 binding 再进入既有 lower HOLD，terminal close 在 HOLD 后 join；
- 新增薄 `MujocoAllocationControlCycleBinding`，由具体 profile 注入 source publisher 与
  allocation-basis factory。它不推断 policy/human role、anchor、阈值、设备、模型或拓扑；
- 一个 cycle 的顺序固定为 source admission -> optional non-blocking D2 basis offer -> D4 exact
  proposal publication -> one lower `runtime.step()`。`no_action` 仍会调用 lower step，不在
  Station 生成 HOLD、fallback、安全判断或 direct-control action；
- source publication 显式携带接纳完成后的 receiver `observed_at`。basis cut、proposal cut 和
  lower step 使用该闭合 cut，避免同步 human poll 后仍按 pre-poll timestamp 选择输入；
- read-only status 可显示 allocation publication status、selected context 与 component mode。
  现有 public factory 仍只选择 fixed-alpha compatibility port，因此真实 VIVE/Q6、GUI rebase
  event 与 durable seal/readback 仍未接入；
- activation priming ownership 已成为显式 port capability。fixed-alpha port 保留原
  application-owned、prime-before-activate 顺序；allocation port 则先建立 exact
  activation-local binding，再通过其注入 source publisher 做 source-only prime。prime 不发布
  proposal、不执行 lower cycle；prime 成功前禁止进入 `run_cycle()`，re-arm 必须新建并重新
  prime binding；
- pause/episode-end 的旧 binding 清理拆为两阶段：先 non-blocking fence 并使 late receipt
  不可提交，随后立即建立 lower HOLD；terminal close 只在 HOLD 之后 join retired worker。
  即使 fence 抛错也仍尝试 lower HOLD，再向 application 报告双边结果；
- 首帧 human anchor sample 以 identity residual 正式接纳，重复 emission 返回同一 immutable
  submission；pose identity 为零平移加单位四元数，qpos identity 为零向量；
- focused lifecycle/allocation/input/publication/Sim-HIL/boundary 回归为 `135 passed`。验证使用
  virtual source/fake MuJoCo scene，physics command count 为 0，不是设备、模型、viewer 或运动
  证据。

下一落地点已收敛为实际 profile wiring，而不是继续扩充 composition core：具体 Sim-HIL
profile 需要提供 policy/human source publisher、anchor/rebase binding 与 allocation-basis
factory，并由 public factory 在显式 default-off 选项下选择 allocation port。随后补 GUI typed
rebase event 和 actual-data durable seal/readback。完成这些接线与虚拟闭环验收前，public
factory 继续只选择 fixed-alpha compatibility port。

public wiring 前仍需单独冻结第一个实际 allocation profile 的 takeover 边界。generic reducer
已经能选择 `absolute_overlay_owner`，但若 application 仅在阈值越界时把 composed target 从
`policy + residual` 改成设备 absolute pose，目标值可能发生不连续跳变。不得用未经标定的
阈值或额外 pre-mix safety 把该问题隐藏起来。首轮有两个可实施 gate：

1. 推荐先接通 `base_only <-> relative_overlay` 的 default-off Sim-HIL production proposal；
   absolute evidence/decision 仍可观测，但不允许 promotion。待连续 takeover receipt 完成后
   再开放第三种 mode；
2. 直接接三种 mode，但必须先定义显式 transition transaction：进入 absolute 时绑定 exact
   pre-transition composed target、human sample/anchor 与首个可用 target boundary，并决定是
   downstream-only interpolation，还是建立可追溯的 takeover offset/anchor。该 transaction
   未冻结前不得由 public factory 选择三态 profile。

两种方案都不改变 fixed-alpha 默认路径；阈值、normalization、window、TTL 和 dwell 应作为
development profile data，而不是写死进 generic port 或 CLI 算法。

方案 A 首轮落地收据（2026-08-07）：

- generic reducer 继续完整产生三种 mode；限制只放在具体 proposal publication profile；
- `MujocoPolicyHumanAllocationProfileConfig` 默认只允许 `base_only` 与
  `relative_overlay` promotion，resource-free Station profile builder 将该 allowlist 原样绑定到
  唯一 D4 proposal gate，避免 application factory 漏配；
- `absolute_overlay_owner` 的 ready evaluation 返回 `promotion_blocked`，保存 selected/blocked
  component mode，不提交 proposal；application 仍调用一次 lower step，由下游已有
  persistence/HOLD 语义处理；
- allocation profile 显式拒绝旧 fixed-alpha hot control；fixed-alpha port 的 capability 与公开
  factory 选择不变；
- activation prime 显式携带 barrier 后的新 human emission。异步 source 首次 poll 无数据时在
  既有 timeout 内等待 fresh emission，不复用 cached pose；prime 不发布 proposal、不调用 lower
  execution；
- passive human emission 到非权威 ComponentAction/trace facts 的适配已抽为共享 source helper，
  旧 fixed-alpha 路径已改用该 helper，为 allocation source publisher 消除第二套 identity/时间
  转换实现；
- 本收据仅由 pure/virtual 与非硬件 MuJoCo contract tests 验证，不是 viewer、设备、模型或
  motion 证据。公开 Sim-HIL factory 仍选择 fixed-alpha compatibility。

### 20.4 兼容性验收

- 新功能关闭时，既有 policy-only、absolute weighted、per-cut relative、teleop、replay、
  deploy 的数值、顺序、pop count、proposal/emission hash 不变；
- intent preview 对任意 component topology 工作，不要求双臂、固定 DOF 或特定模型；
- speculative keyframe 永不显示为 locked/executable；locked keyframe 也永不显示为
  Reference；
- allocation 输入 missing/stale 不得伪装为“人静止”；
- 阈值附近抖动不造成 mode chatter；
- learned allocator 以后只能异步发布带 TTL 的 advisory fact，control loop 不等待模型；
- 当前 D3/D4 只启动 virtual MuJoCo runtime/fake scene，不加载实际 task；D3 proposal
  count 为 0，D4 只验证 proposal admission，二者 physics command count 均为 0；其余首轮
  验证为 pure/virtual，均不启动设备、模型 server 或 motion。

R3.1c-A/B 首轮实现收据（2026-08-06）：

- `control_composition/trajectory/preview.py` 只投影 exact trajectory revision，locked 与
  speculative 分类由 `source_immutable_through` 唯一决定，所有 point 明确
  `execution_guaranteed=False`；
- `control_composition/allocation/` 按 contracts/decision/metrics/reducer/context/strategy
  拆分，标准 activity metric 支持 pose 和任意维 continuous/qpos，其他 representation
  必须显式提供 provider；
- ready allocation decision 作为 ordinary typed context 接纳，formal allocated rules 在
  base/relative/absolute 三个预声明 role 间选择；`absolute_overlay_owner` 不产生 authority
  变更；
- shared pose/qpos 数值运算从 formal strategy 聚合文件抽到
  `composition/component_operations.py`，避免为 allocation 重复实现 relative math；
- `control_station/composition/trajectory_intent_preview.py` 只做 default-off application
  gate；关闭时不读取 trajectory supplier，且任何配置都不能发送动作；
- focused preview/allocation/relative/formal-component/Station-boundary 组 `206 passed`；
  allocation context 同时显式禁止将新决策回填到早于其 `evaluated_at` 的 target。验证仅为本地
  pure/virtual contract，不是 application、sim、模型服务、真实设备或 motion 证据。

R3.1c-C 首轮实现收据（2026-08-06）：

- `composition/input_selection.py` 是 admitted 与 projected carrier 共用的 pure selector；
  production store 的 admission、retention 和 closed-target ownership 保持不变；
- `composition/projected_*.py` 定义无 admission identity、无 authority、无 promotion 的
  projection/evaluation 契约；formal component strategy 复用同一数值规则；
- `trajectory/frozen_projection.py`、`projected_inputs.py` 与 `corrected_preview.py` 从单个
  non-closing cut 生成整段 conditional corrected horizon，future context/activity 不预测；
- `trajectory/preview_freshness.py` 将 intent trust 与 observation freshness 分离，缺少证据或
  clock mapping 时为 `unknown`，显式过期才不可显示；
- focused trajectory/composition/allocation/formal regression 为 `121 passed`。验证仅为本地
  pure/virtual contract，不是 application、sim、模型服务、真实设备或 motion 证据。

## 21. Anchor-relative neutrality 与 typed interaction event（纯协议首轮已落地）

### 21.1 对“静止”的修正

human 是否仍在影响 composition，必须以当前 residual 相对其 active anchor 的幅值为
主要事实之一，不能只看最近窗口内是否在移动：

```text
quiet near active anchor
  residual magnitude ~= identity + motion quiet
  -> base_only candidate

quiet away from active anchor
  residual magnitude != identity + motion quiet
  -> keep relative_overlay

explicit HOLD
  -> Station operator intent / lifecycle
  -> 不是 allocation 对“静止”的推断
```

当前标准 activity metric 只计算窗口首尾位移、path 和 rate。human 移到非零 residual
后保持不动时，这些量会逐步归零，因此当前 reducer 可能错误退回 `base_only`。这不是
调阈值可以解决的问题，必须把以下证据维度分开：

1. `anchor_relative_magnitude`：最新 residual 相对 identity 的幅值；
2. `motion_activity`：窗口内 path/rate/变化幅度；
3. `freshness/eligibility`：证据是否可在本 target 使用；
4. 可选 detector facts：静止、接触、意图、模态标签等外部 observation。

pose residual 的 magnitude 使用世界平移相对零向量与局部旋转相对 identity quaternion；
joint-position/qpos residual 使用相对零向量的显式 norm。其他 representation 必须由
provider 声明 metric 与 normalization，不能被强制解释成 pose/qpos。magnitude fact 必须
绑定 exact `ResidualAnchorReceipt` generation，禁止跨 rebase window 混用样本。

首轮推荐 transition 判定：

- `base_only -> relative_overlay`：magnitude 或 motion 越过 enter 条件；
- `relative_overlay -> base_only`：magnitude 与 motion 同时低于各自 exit 条件，并满足
  dwell；
- `relative_overlay <-> absolute_overlay_owner`：使用独立的 profile 条件与 continuity/
  anchor transition，不仅改变 mode enum；
- missing/stale detector 或 source 不是“静止”，不得产生 base-only 证据。

### 21.2 事件边界

不新增一个能同时改变 lifecycle、anchor、allocation 和 authority 的全局 event bus。
事件按作用边界分成三类：

1. **operator/device command intent**：按钮、clutch、trigger、GUI 操作等显式请求，由
   Control Station 接纳；
2. **observation/advisory fact**：静止判别器、活动度、接触或 learned classifier 等只读
   证据，通过 typed context/provider 进入同一 causal cut；默认没有 rebase、HOLD、
   authority 或 mode-change 权力；
3. **causal transition request/receipt**：真正改变 residual anchor 的
   `ResidualAnchorRequest/Receipt` 与 session 内 active-anchor binding revision。

可扩展事件/observation 的公共 envelope 只负责 identity 和时间因果，不携带本体特例：

```text
event kind / provider contract
subject source role
component set
produced_at + clock domain
available_at / valid_until
source fact lineage
profile-selected parameters or payload schema hash
```

事件种类应 namespaced、由 profile 显式声明消费规则。新增设备、判别器或模态只需增加
provider/adapter；generic reducer 不 import 设备、sim、模型或 GUI。

### 21.3 `rebase-and-clear` 的推荐因果流程

按钮或显式 trigger 的含义已冻结为“在当前 source 状态建立新 anchor，并从后续
composition target 清除 residual 影响”，而不是 HOLD 或 authority transfer：

```text
typed rebase-and-clear intent
  -> bind residual session + expected active-anchor revision + component set
  -> ResidualAnchorRequest generation n+1
  -> input barrier fences pre-trigger source samples
  -> exact trigger-associated sample，或 barrier 后第一条 fresh complete sample
  -> ResidualAnchorReceipt generation n+1
  -> ActiveResidualAnchorBinding revision n+1
  -> next not-yet-prepared target uses identity residual
  -> allocation reevaluates to base_only when its other conditions permit
```

如果设备按钮与 pose/qpos 来自同一原子 packet，receipt 可以绑定该 packet；如果事件与
动作流分离，“当前状态”严格定义为 barrier 后第一条 fresh complete source sample，并受
capture timeout 约束。旧样本不能回填新 anchor。

`control_composition` 只改变 CompositionResult/target 的内容，不生成 Reference、Command、
IK、平滑或限幅。rebase 后 target 回到 base 所需的连续插值仍由下游 executor/planner
负责；已正式 publication 的 target 不撤销、不重写。

### 21.4 Session 内 anchor revision（已冻结）

现有 `ResidualSessionActivationBinding` 证明 session 的初始 activation/cut/anchor/base，
应继续不可变。为了支持按钮反复 rebase，而不把一次连续的 UX 交互切成多个 session，
已新增独立的：

```text
ActiveResidualAnchorBinding
  residual_session_id/hash
  anchor_binding_revision
  residual_anchor_receipt_id/hash
  effective_for_target_at
  predecessor_binding_id/hash
  trigger intent/advisory lineage
```

Reducer 只允许 revision 单调加一、predecessor 完整、effective target 尚未 prepared。
这样 session identity 表示一次连续 HITL 交互，anchor binding revision 表示该交互内的
recenter 历史；既不会静默替换初始 receipt，也不强迫 Control Station stop/start session。

### 21.5 已冻结的公共语义

1. **pending rebase**：trigger 一经接纳，旧 anchor 从声明的未提交 target 边界起立即被
   fence；新 receipt 尚未形成时返回 typed `no_action`，不由 composition 生成 HOLD 或继续
   沿用旧 correction。
2. **生效边界**：新 anchor 只从 next not-yet-prepared keyframe 生效；已 emission 的 target
   不 replacement。物理连续性由下游处理，不在 composition 中复制插值器。
3. **detector 权限**：额外静止/模态判别器默认只发布 advisory fact。只有未来 profile
   显式开启 `automatic_rebase`，并声明 provider、TTL、debounce 和 component scope 时，
   才能把该 fact 转换成同一个 typed transaction；当前未实现或启用自动 profile。
4. **absolute overlay**：本节 rebase 只重置 relative residual。`absolute_overlay_owner`
   的 engagement/rebase/handback 仍需独立 transition contract。
5. **首轮 component scope**：一次 rebase 原子更新当前 residual session 的完整 component
   set，禁止在一次 allocation cut 混用不同 receipt generation。单 session 内的部分
   component rebase 需要后续 component-keyed active-anchor 协议，当前不伪装支持。

### 21.6 对未来轨迹建议的采纳

采纳“artifact stage 与 freshness 正交”，但不把不同 owner 的事实压成一个单枚举：

- source preview 继续只区分 producer-locked 与 speculative intent，并增加独立 freshness；
- composition corrected preview 是基于一个 frozen cut 的 conditional projection；
- authoritative target 与 scheduled Reference 只能由各自下游 receipt/observation 提供；
- UI 可形成 nominal、latest shadow、conditional corrected、executable observation 四个
  view，但 lineage/owner 保持分离；
- R3.1c-C 使用独立 projected carrier/evaluation，不能伪造正式
  `CompositionInputRecord`，也不能 promotion。

### 21.7 实施收据与后续 gate

已完成：

1. allocation 将 `motion_activity` 与 exact-anchor-bound `residual_magnitude` 分离；标准
   pose/qpos provider 和所有自定义 provider 都有显式 contract identity；只有两类证据都
   低于 exit 条件才允许回到 base-only；
2. `ActiveResidualAnchorBinding`、`ResidualRebaseRequest`、pending fence、fresh/trigger-
   associated receipt completion 与 target/emission barrier 已进入纯 reducer；
3. Station 增加 default-off `request_residual_rebase_and_clear` intent/port，receipt 绑定 exact
   intent hash，关闭时不会求值 command supplier，也不会获取设备数据或发送动作；
4. 纯测试覆盖 near-anchor quiet、nonzero-residual quiet、pose/qpos、mixed-anchor rejection、
   stale evidence、trigger-associated sample、pre-barrier/late/duplicate event、capture deadline、
   replay hash deterministic 和 old target 不重写。
5. D5b 已增加独立 `ResidualAnchorLifecycleSpec/Store/Snapshot`：同一 receiver cut 下同时
   选择 active anchor、allocation context 与 component inputs；pending 从显式 target boundary
   起返回 `no_action`，ready 只能完成 exact request/boundary。Station 先接纳 pending 再改变
   adapter，并以 generation fence 阻止旧 reducer attempt 的 late receipt 回写；首个 fresh
   sample 原子发布 absolute + identity residual 后才发布 ready。
6. lifecycle 已进入 non-closing corrected-horizon projection和 v2.4 ControlTrace adapter；每个
   future keyframe 从同一 frozen cut 重建 lifecycle snapshot，记录 request/receipt/boundary/
   receiver identity 但不复制 action vectors，也不产生 promotion 或 execution claim。

尚未完成，不能宣称 application/sim/production 可用：

- 具体按钮/trigger packet 到 anchor request/receipt 的设备 adapter；
- detector advisory 的异步 producer/context adapter，以及显式 automatic-rebase profile；
- 单 residual session 内部分 component rebase；
- 公共 MuJoCo Sim-HIL application profile/GUI 的 lifecycle 状态展示与端到端虚拟设备验证；
  当前只有 default-off shadow/proposal fixture 的薄接线，不代表公开 profile 已启用；
- 任意 real-device 接入或 motion 证据。

后续仍保持 application binding default-off，先做 pure/virtual，再做 sim；不得因本节协议
完成而直接启动真机 motion。

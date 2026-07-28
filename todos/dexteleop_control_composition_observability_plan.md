# Control Composition：差异观测与多源组合实践及后续规划

更新日期：2026-07-26

本文是 `control_composition` 当前实践、差异观测边界和后续落地顺序的独立索引。
它不是 v2.4 数据协议，也不是 teleop/replay/deploy 的实施流水账。

相关事实源：

- 目录 ownership：[`AGENTS.md`](../AGENTS.md)
- 组合框架契约：[`control_composition/README.md`](../control_composition/README.md)
- 主架构索引：[`dexteleop_architecture_planning.md`](dexteleop_architecture_planning.md)
- ControlTrace/v2.4：[`dataset_contracts/README.md`](../dataset_contracts/README.md)
- Control Station 装配：[`dexteleop_control_station_plan.md`](dexteleop_control_station_plan.md)
- RTC 部署计划：[`dexteleop_rtc_landing_plan.md`](dexteleop_rtc_landing_plan.md)

---

## 1. 目标与范围

`control_composition` 是与本体、sim、真机设备、模型架构和数据集格式无关的多源动作组合库。
它接收已经由上游 adapter 接纳的动作/context 事实，在一个确定的 receiver-side causal cut
上完成选择或数值融合，并返回带完整 lineage 的 composition evaluation/proposal candidate。

目标链路：

```text
human / replay / policy / synthetic / future source
                    │
                    ▼
          source adapter + admission
                    │
                    ▼
       immutable input/context snapshot
                    │
                    ▼
       optional difference observations
                    │
                    ▼
             replaceable strategy
                    │
                    ▼
       composition evaluation + lineage
                    │
                    ▼
             proposal admission
                    │
                    ▼
    authority -> canonical safety -> IK/reference -> dispatch
                    │
                    ├── sim backend
                    └── real backend
```

本包不拥有：

- authority、arbitration、execution epoch、dispatch owner；
- canonical workspace/joint/velocity/force/collision safety；
- IK、reference generation、target/backend capability；
- teleop device、replay cursor、model worker、RTC queue；
- episode、WAL、ControlTrace writer、最终 v2.4 dataset；
- 真实力反馈或任何可以改变 motion authority 的反馈路径。

---

## 2. 当前长期实践

### 2.1 唯一 owner 和依赖方向

对同一个逻辑资源，source production、cursor/plan advancement、authority、canonical safety、
dispatch 和 episode commitment 各自只能有一个 authoritative owner。observer、fanout queue
和 UI cache 只能只读、有限、不可重新构造 owned state。

依赖方向：

```text
control_station / teleop / deployment / sim adapters
                         │
                         ▼
              control_composition
                         │
                         ▼
                   robot_motion
```

`control_composition` 可以依赖 `robot_motion` 的纯动作/运行时契约，但不得导入具体 sim、硬件、
teleop、deployment 或 `control_station` 模块。新增设备、本体、模型或 sim 时优先增加边缘 adapter
和 capability，而不是复制组合算法或执行 owner。

### 2.2 动作表示和拓扑

- 机械臂的 canonical composition representation 是 Pose；6DoF、7DoF 或其他关节数属于
  downstream resolver，不应成为通用 arm composition schema。
- 夹爪、灵巧手和其他末端使用各自 typed qpos/native component。
- 通用 carrier 以任意 `component_id` 为键，不要求左右双臂，也不限制一个 arm 对应一个末端。
- 当前 `RobotAction` v1 仍是兼容 profile，保留 legacy left/right、固定布局和旧 hash；component-keyed
  carrier、proposal 和 Arbiter shadow 是 additive、non-authoritative seam。
- arm-qpos mixing 只保留在明确 debug/未来同构臂 profile，不是当前主线。

动作阶段必须分开：

```text
Action value
    -> SourceActionSample
    -> ComposedActionCandidate
    -> downstream ExecutionTarget / optional ReferenceFrameObservation
    -> Command / Measurement
```

不能用一个模糊的 action vector 代表所有阶段。`control_composition` 在
`ComposedActionCandidate` 结束；后续阶段由 target/execution capability 按需拥有，
不是所有 backend 的强制流水线。

### 2.3 接纳、时间和生命周期

每条输入经过 receiver admission 后才成为可见事实：

1. 校验 identity、clock、epoch/sequence、representation 和数值可计算性；
2. deep-freeze，并赋予 `available_at` 与 admission sequence；
3. 按 target-effective time 和 explicit TTL 选择；
4. 在同一 `ControlCut` 上构造 input/context snapshot；
5. cut 之后到达的事实只影响下一次 evaluation。

不同频率的来源不需要同步频率。慢来源仅在 TTL 覆盖 target 时复用，快来源只能替换未来 evaluation。
late completion 不会重写已经形成的 result。

当前 store 使用语义 time-window retention，而非固定 4096 ring overwrite：已返回 snapshot/result
持有不可变对象，store pruning 不会改写历史结果；producer epoch 的 compact tombstone 仍保留以拒绝
重复或回退 identity。

source action TTL 与 source liveness/lease 是两条独立语义。source eligibility 是 default-off；
paused/fault/disconnected/rearming 只把角色变为不可用，不授予 authority，也不直接执行 HOLD。

### 2.4 Composition 和安全边界

composition 负责：

- role/component selection；
- pose selection 或 weighted SE(3) fusion；
- end-effector qpos interpolation；
- explicit missing/fallback 和 bounded strategy residual；
- result 的 deterministic identity 和 proposal lineage。

source admission 只做最小的协议完整性检查，不提前执行 motion limit、clamp、workspace、IK 或
target-specific safety。有限但数值很大的原始输入仍应保真进入 composition。真实安全由 composition
之后的 canonical safety、resolver、reference 和 backend execution 层负责。

composition evaluation 只有：

```text
ready / no_action / invalid
```

它不是 motion permit。绑定该result的disposition可以向下游提交candidate或
`request_hold`，但请求不等于HOLD；HOLD、FAULT、cancel和execution epoch只有在
下游owner接纳并返回receipt后才成为权威事实。`no_action`如何映射由具体execution
profile显式声明。

### 2.5 Context 和信息需求

`CompositionContext` 是 receiver-admitted、不可变、representation-neutral 的策略上下文，可以
承载 history、state、future action、label、vision、tactile、force 或其他 modality 的 typed
payload/ref。

策略只能读取 cut 后冻结的 context，不得读取 live device、model、sim、GUI、filesystem 或 network。

`CompositionInformationPlan` 负责在 acquisition 前声明：

- 需要哪些 context channel；
- payload contract hash；
- maximum age；
- metadata-only、inline 或 blob-ref；
- 当前可选的 difference demand 和 diagnostic level。

只为 dataset 使用的视觉/触觉 native stream 直接进入 capture owner；只有策略确实需要时才通过
`CompositionContext` 接入轻量 immutable view/ref。

### 2.6 Pre-mix policy

当前已经实现可选的 `CompositionPreMixPolicy`，位于输入/context snapshot 之后、正式 composition
之前，且 default-off。

```text
same-cut input/context
        -> CompositionPreMixPolicy
        -> formal strategy / deterministic composer
```

policy 输入包含：

- runtime session/domain/target；
- exact `CompositionInputSnapshot`；
- optional `CompositionContextSnapshot`；
- `CompositionInformationPlan`；
- exact policy input hash。

policy 只能返回：

```text
pass / no_action / invalid
```

它不能：

- 改写 source 数值；
- 丢弃单个 source；
- 隐式复用 previous result；
- 签发 authority/safety receipt；
- 执行 IK、reference 或 dispatch。

同步 policy 必须声明正的 `maximum_decision_latency_sec`。runtime 在调用返回后检查耗时，超时转为
`no_action` 并记录 diagnostic。这不是线程抢占；可能阻塞的高级 policy 应改为异步 advisory producer。

policy spec、input、decision 和实际耗时进入 `CompositionEvaluationRecord`。未配置 policy 时，原有
行为和 v1 hash 不变。

---

## 3. 当前差异观测实践

差异按 owner 和 causal stage 分为两个独立模块，不用 v1/v2 表示先后关系。

### 3.1 `CompositionImmediateDifference`

归 `control_composition`，使用同一 input snapshot，在正式混合前比较 source-to-source：

```text
human pose  <-> policy pose
human qpos  <-> replay qpos
```

目标用途（当前尚未接入 runtime policy input）：

- pre-mix policy 输入；
- GUI/可视化立即显示；
- 低资源的同步反馈准备。

它不包含 composed result、canonical safety、IK、dispatch 或 measurement 事实。

当前实现：

- [差异计算 helper](../control_composition/observability/difference.py) 已能在同一 cut 上计算
  pose translation/SO(3) 和 finite-slot qpos 差异；
- [SourceDifferenceImmediateObserver](../control_composition/observability/modules.py) 将当前
  pose/qpos 结果包装为 provisional opaque payload；
- 缺失 pair 或无法计算时返回 `unavailable`，不自动阻塞 composition；
- frame mismatch、qpos mask 和 representation 不做隐式转换或零填充。

### 3.2 `PostCompositionExecutionDifference`

从 composed result 开始，可覆盖：

```text
source -> composed
composed -> safe
safe -> resolved IK
resolved -> limited/reference
reference -> dispatched
dispatched -> measured
```

混合后的 source-to-composed residual 可以是同周期 passive fact；真实 sim/真机执行部分必须由
下游 receipt owner 异步提供。没有 dispatch/measurement identity 时，不能称为 actual execution
deviation。

当前实现：

- [SourceToComposedDifferenceObserver](../control_composition/observability/modules.py) 已实现
  provisional source-to-composed passive observation；
- [DifferenceReceiptReference](../control_composition/observability/modules.py) 只记录下游 receipt
  identity/hash/stage，不复制下游 receipt 语义；
- 真实 safe/resolved/reference/dispatch/measured observer 尚未接入生产 sim/real owner。

### 3.3 Transitional carrier 和兼容边界

现有 `SourceDifferenceObservation` 同时包含：

- `pairwise_differences`：第一模块的 source-to-source 部分；
- `output_residuals`：第二模块的 source-to-composed 部分。

它是过渡性 carrier，不是长期冻结的公共 schema。已有调用方继续兼容；新调用方应选择明确的
capability/profile，不应假设所有 payload 字段永远存在。

当前 observation envelope 只固定：

- module/capability identity；
- status/reason；
- control domain、clock、cut、target；
- input snapshot lineage；
- optional composition evaluation lineage；
- optional receipt references；
- opaque capability-owned payload。

暂不固定 metric catalog、默认 pair、阈值、frame conversion、debug payload 或 feedback mapping。

### 3.4 Feedback

现有 guidance feedback 仅是 non-authoritative、bounded、TTL 约束的 projection：

- 当前仅对 pose 差异生成 dimensionless guidance cue；
- qpos 差异可以记录，但没有通用反馈映射；
- `NoIoGuidanceFeedbackAdapter` 只做验证，不访问设备；
- feedback 不能获得、续期或改变 authority/HOLD/dispatch。

真实力反馈仍属于后续 target/device adapter，不属于通用 composition 算法。

---

## 4. 记录、ControlTrace 和数据集边界

`CompositionEvaluationRecord` 是 composition 自己的完整只读解释接口，包含：

- exact composition spec；
- ControlCut 和 producer epoch snapshot；
- selected input/context lineage；
- result；
- requested difference/diagnostic；
- policy input/decision；
- protocol-neutral domain events。

记录分两层：

- core：实际参与 composition 和因果解释的事实，正式 profile 保留；
- debug：候选矩阵、中间权重、preview、model internals 和重诊断，仅 debug profile 构造。

`control_composition` 不负责：

- v2.4 ControlTrace schema；
- dataset writer、WAL、episode membership、seal/commit；
- media retention、backpressure 和最终训练 view。

外部 recorder 可以订阅 `CompositionDomainEvent` 或读取 evaluation record，但“需要保存数据”不能
反向迫使 composition acquisition 视觉/触觉等未声明模态。

---

## 5. 当前实现成熟度

差异观测的优先级不能高于执行事务闭环。当前 scheduled MuJoCo 路径已经具有分阶段事实：

```text
ProposalAdmissionReceipt
-> arbitration decision(authority epoch)
-> canonical safety result
-> ExecutionTargetAdmissionReceipt(execution epoch + group generation)
-> ReferenceSegmentAdmissionReceipt(supersedes + admission guarantee)
-> CommandDispatchReceipt
-> ExecutionCancellationReceipt
```

因此不新增一个笼统的 `execution_lease` 或“总 final receipt”。不同 target profile 的实际
线性化点不同：scheduled reference 以 target/reference receiver admission 为准，direct
backend 以 backend accepted receipt 为准。proposal admission 只表示候选事实进入 receiver，
不表示获得 authority 或运动许可。

当前还必须保持的执行不变量：

- authority 变更前必须已有 lower HOLD/FAULT/TRANSITIONING fence 和完成的 backend ack；
- authority API 不隐式执行 cancellation；
- 已处于 lower HOLD 时不重复 force-cancel/推进 execution epoch；
- producer restart/re-arm 通过 producer epoch/activation barrier 使旧输入和旧 proposal 失效；
- dispatch context 在 receiver 侧再次校验 owner generation、authority epoch、execution epoch
  和 group generation；
- lower HOLD/cancel 失败必须向上抛出，不能包装为缺少 receipt 的成功 cycle。

| 能力 | 当前状态 | 边界/不能外推为 |
|---|---|---|
| input/context admission | 已有 receiver clock、cut、TTL、epoch、time-window store | 不是 authority 或 execution owner |
| source eligibility | default-off，已有 lease/status epoch/re-arm lineage | 不是 HOLD/FAULT 执行器 |
| pose composition | 单 owner selection、weighted SE(3) 和 formal component seam | 不代表真实 target safety/IK |
| qpos composition | end-effector active-mask-aware interpolation | arm-qpos 不是主线 |
| component topology | component-keyed additive carrier/proposal shadow | 旧 RobotAction v1 production 仍是兼容路径 |
| pre-mix policy | default-off same-cut pass/no_action/invalid、耗时诊断 | 不是异步 worker，也不负责 source rewrite |
| immediate difference | provisional source-to-source observer 和 pose/qpos helper | 尚未 runtime demand/budget 接入 |
| post-mix difference | provisional source-to-composed observer | 尚无真实 dispatch/measurement observer |
| execution transaction | scheduled MuJoCo 已有 target/reference/dispatch/cancel receipt 与 trace | real backend 的统一 production binding 尚未完成 |
| execution difference | A0 receipt lineage + A1 same-clock qpos association 已有 provisional seam 和 MuJoCo passive adapter | 未接 production caller；cross-clock、pose/FK、real binding 待完成 |
| guidance feedback | pose bounded no-I/O projection | 不是物理力反馈 |
| evaluation/lineage | exact record、hash、domain events 已有 | 不是 dataset finalization |
| default compatibility | 未启用新 profile 时保留旧行为 | 新 observer 不应隐式启用 |

---

## 6. 接下来落地规划

### System P0：保持并扩展 no-motion/sim 执行事务闭环

当前 sim 已验证 proposal、authority、canonical safety、target/reference admission、dispatch、
HOLD/cancel、re-arm 和 stale epoch rejection。下一步只做已有契约的边缘接线与负面用例：

1. virtual policy / `send_actions=false` 只产生 typed emission/input，不获得 motion authority；
2. late result、旧 producer epoch、旧 authority/execution epoch 和旧 group generation 均在
   影响 reference 前拒绝；
3. pause、takeover、fault 和 re-arm 的 cancellation/READY 因果链可重放；
4. real target 先做 passive/default-off binding，不从 sim 证据外推真机 readiness；
5. 不修改 proposal v1/v2，不新增与现有 epoch/generation 重复的 execution lease。

### Observation P1：为 ImmediateDifference 增加按需声明和预算

前提是先选一个真实 consumer（优先 sim GUI/可视化或 replayable test consumer），明确：

- 是否需要 pre-mix source-to-source observation；
- 需要哪些 role/component pair；
- summary/full 的资源边界；
- 同步允许的最大耗时；
- unavailable 时是仅记录还是供 policy 决策。

落地要求：

1. 以 additive/profile-gated 方式扩展 `CompositionInformationPlan`，不覆盖现有
   `source_difference_spec_hash` 语义；
2. runtime 仅在显式 demand 下调用 immediate observer；
3. default-off 且不 acquisition 未声明的 modality；
4. 超过同步预算只产生诊断或 `no_action`，不启动隐式阻塞 worker；
5. 将 observation 写入 evaluation record 的独立字段，不与 post/execution 字段混合。

验收：同一 cut、不同来源频率、缺失/late/frame mismatch/qpos mask、重复调用和 default-off parity。

### Observation P2：接入 PostComposition 的 passive residual observer

`SourceToComposedDifferenceObserver` 的 provisional 计算实现已经存在；下一步是在
sim/no-motion shadow 中完成它的按需接入和独立事件/返回链路：

- composed result 与 selected source 的 source-to-composed residual；
- result identity 和 input snapshot identity 校验；
- observation 的 valid-until 和 target time；
- `CompositionEvaluationRecord` 之外的独立 observer event/fanout；
- 不影响 proposal、authority、canonical safety 或 execution owner。

该阶段只做 composition-local passive fact，不声称 safe、IK、reference 或 actual execution。

### Observation P3：实现 downstream ExecutionDifferenceObserver

由 `robot_motion`/target backend 提供 passive observation adapter。它只覆盖
该 capability 实际拥有并由 `CompositionInformationPlan` 请求的阶段，例如：

```text
composed candidate
execution target
optional actual reference frame
dispatch receipt
measurement sample
```

每个阶段只提交自己的 identity/hash/status/time。观察器只读，不重建 authority，
不复制 safety、IK、平滑、限幅或插值逻辑。sim 先验证 deterministic receipt
lineage，再由真机 profile 接入 stop/latency/operator gate。

当前已完成的最小切片：

1. **A0 receipt/lineage**：通用层以 opaque
   `DifferenceReceiptReference` 绑定下游 fact ID/hash/stage/time，不复制 receipt
   的 authority/safety 语义，也不新增 execution lease；
2. **A1 same-clock qpos**：component-keyed actual-reference/measurement projection
   只在 control domain、clock domain、representation、schema 和维度一致时计算
   `joint_position` residual；
3. 关联规则固定为 lower owner 导出的唯一 actual reference frame，其
   `[effective_from, effective_until)` 覆盖 `measurement.measured_at`，并通过
   target/reference identity 关联；逻辑 target time 只保留 lineage，禁止
   nearest-target-time 匹配，也禁止上层从 segment 自行重算 actual reference；
4. MuJoCo 薄适配应把内部
   `ReferenceSegmentAdmissionReceipt -> actual reference activation ->
   optional ReferenceSegmentTerminal -> MujocoQposMeasurement` 投影为新的
   downstream-owned identity，不把旧 execution epoch/group generation 名称
   泄漏进通用协议；
5. 该能力没有 production caller，默认不构造、不计算、不影响 control loop。

后续仍分开推进：

- **actual reference contract**：下游只计算一次并把执行所用的同一不可变对象
  nonblocking fanout；支持 off、lifecycle、decimated-frame、full-frame demand；
  provenance 重算只能标为 `ReconstructedReference`；
- **A2 cross-clock**：只有存在显式 clock-mapping receipt、误差界和有效区间后才允许；
- **pose actual difference**：由下游 FK/frame adapter 从 measured qpos 产生
  frame-qualified measured pose，再使用 pose metric；不得在通用 observer 中把
  qpos 与 pose 直接相减；
- **target/dispatch stages**：只绑定 capability 实际暴露的 receipt，不强制
  safe/reference 等内部阶段，也不做一个笼统 final receipt；
- **real binding**：先 passive/default-off，单独验证 stop、latency、operator gate
  和 backend failure。

### Observation P4：迁移 transitional carrier

只有当至少一个真实可视化 consumer、一个 replayable fixture 和一个 sim passive observer 使用新端口后，
才考虑拆分 `SourceDifferenceObservation` 的 pairwise/output-residual carrier。

迁移方式：

- 保留旧读取路径；
- 新字段 additive/profile-gated；
- 旧 carrier 只作为 compatibility projection；
- 不原地重写既有 hash、ControlTrace 或历史 dataset；
- 记录 migration note、consumer 清单和删除条件。

### Observation P5：可视化与 feedback

差异链稳定后再接：

- sim GUI overlay；
- human/policy/replay 对比视图；
- bounded haptic guidance；
- 后续 force feedback。

feedback 只消费 difference，不能成为第二 authority 或第二 dispatch path。

### Observation P6：ControlTrace/dataset 外部投影

最后由 `dataset_interface`/capture runtime 选择：

- core 事实是否持久化；
- debug observation 是否保留；
- 如何映射到 v2.4 ControlTrace；
- episode/WAL/seal/reader 如何处理异步 execution difference。

不把这些协议回写为 `control_composition` 的通用职责。

---

## 7. 当前必须保持的兼容性规则

- teleop、replay、policy deploy 默认链路不因 observer 存在而改变；
- 新 observer 不自动启用，不新增第二 cursor、queue、dispatcher 或 recorder owner；
- source 只通过 adapter 进入 composition；设备/model/backend 对象不得进入 snapshot；
- pre-mix 只保留原始合法数值，不增加没有实际效果的安全冗余；
- post/execution difference 不得冒充 canonical safety verdict；
- 没有 dispatch/measurement receipt，不得称为实际执行偏差；
- sim 通过不等价于真机 motion readiness；
- `send_actions=false`、shadow、debug 和 no-I/O profile 不得产生运动授权；
- 新 carrier、field、metric 和 hash 必须 additive 或 profile-gated；
- 旧代码迁移到 `legacy/` 前必须确认无 production/test/doc 调用，并记录替换与删除条件。

---

## 8. 验证矩阵

### Pure/contract

- snapshot/result/receipt lineage mismatch；
- same-cut and target-time selection；
- TTL、window pruning、late/reordered input；
- missing、unavailable、invalid、frame mismatch；
- pose SO(3) metric、qpos active mask、arbitrary component key；
- opaque payload immutability and hash determinism；
- observer exception isolation and default-off parity。

### Replayable/sim

- human/replay/policy source pair 的 source-to-source immediate observation；
- compose 后 source-to-composed passive residual；
- safe/resolved/reference/dispatch/measurement receipt chain；
- no-action、HOLD、FAULT、pause、episode-end、re-arm；
- sim/real adapter 只替换边缘 receipt，不复制 composition algorithm。

### Real shadow/real execution

只有在 sim/passive evidence 完成后，才检查：

- actual dispatch acceptance；
- stop/latency/epoch cancellation；
- measurement clock alignment；
- operator gate 和 backend fail-closed；
- feedback observer failure 不影响 authority。

---

## 9. 完成判据

这条差异与组合主线只有同时满足以下条件，才可称为可复用闭环：

1. 新 source、本体、模型和 target 只需增加 adapter/capability，不复制 composition 算法；
2. immediate 和 post/execution 是两个独立、可分别关闭的模块；
3. 两个模块均能返回 exact cut/target/lineage，且不拥有 authority；
4. 真实 execution difference 能绑定 safe、resolved、dispatch 和 measurement receipt；
5. runtime 只 acquisition `CompositionInformationPlan` 声明的信息；
6. debug 关闭时不构造大 payload、不占用不必要的 control-loop 资源；
7. default legacy 行为、动作顺序、pop count、hash 和 teleop/replay/deploy 入口保持不变；
8. ControlTrace/dataset projection 可以独立演进，不反向污染通用 composition；
9. sim、shadow 和真机 evidence 能区分 composition difference、execution difference 与实际测量偏差。

在真实 consumer 和可重放 fixture 出现前，不继续扩大字段集合或冻结新的 public schema。

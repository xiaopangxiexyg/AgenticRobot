# DexTeleop 控制架构规划

更新日期：2026-07-26

本文是控制主线的架构索引，不是实施流水账。它只保留：

- 长期有效的模块边界和控制语义；
- 当前代码已经具备的能力及其准确成熟度；
- 接下来需要跨越的安全与兼容性 gate。

具体字段、批次、测试记录和历史决策见配套文档与 Git 历史：

| 主题 | 事实源 |
|---|---|
| 目录与 ownership | [`AGENTS.md`](../AGENTS.md) |
| 多源组合接口 | [`control_composition/README.md`](../control_composition/README.md) |
| 组合实践、差异观测与后续落地 | [`dexteleop_control_composition_observability_plan.md`](dexteleop_control_composition_observability_plan.md) |
| 模型、动作与部署 schema | [`DEPLOY_SCHEMA.md`](../DEPLOY_SCHEMA.md) |
| 控制主线分批落地 | [`dexteleop_control_core_landing_plan.md`](dexteleop_control_core_landing_plan.md) |
| Control Station | [`dexteleop_control_station_plan.md`](dexteleop_control_station_plan.md) |
| RTC | [`dexteleop_rtc_landing_plan.md`](dexteleop_rtc_landing_plan.md) |
| 图示 | [`dexteleop_architecture_visual_guide.md`](dexteleop_architecture_visual_guide.md) |
| v2.4 数据契约 | [`dataset_contracts/README.md`](../dataset_contracts/README.md) |

---

## 1. 总体目标

DexTeleop 要支持 human、replay、policy 和 synthetic source 以不同频率产生动作，
经同一套组合、authority、安全、执行和记录边界运行于 sim 或真机：

```text
human / replay / policy / synthetic
              │
              ▼
     immutable source facts
              │
              ▼
 control_composition（可选数值融合）
              │ proposal candidate
              ▼
 RobotMotion proposal admission
              │
              ▼
 authority / arbitration / persistence
              │
              ▼
 canonical safety
              │
              ▼
 target resolution / IK / reference
              │
              ▼
 one dispatch owner
              │
              ├── sim backend
              └── real backend

每一层的事实 ──→ ControlTrace / episode capture
```

增加新机械臂、遥操设备、sim 环境或模型架构时，应替换边缘 adapter 和 capability，
而不是复制 composition、authority、safety、dispatch、cursor 或 recorder。

---

## 2. Ownership 边界

`AGENTS.md` 是最终约束；本节只给出主线摘要。

| 包 | 负责 | 明确不负责 |
|---|---|---|
| `control_composition` | 多来源接纳、同一 causal cut、TTL、source eligibility、replaceable strategy、difference 与 lineage | authority、canonical safety、IK/reference、dispatch、设备/模型 worker、replay cursor、episode |
| `robot_motion` | canonical action、proposal、authority、epoch/cancellation、safety、resolver、reference、execution contract | vendor 初始化、GUI、模型推理、训练 |
| `control_station` | GUI/API、typed operator intent、profile 选择、应用 lifecycle、薄 source/target 装配 | 通用融合算法、worker/cursor、canonical safety、dispatch、dataset writer |
| `deployments` | 模型 artifact、协议、transport 和输出解码 adapter | motion authority 与机器人执行 |
| `sim` | 仿真 backend、task 和 sim adapter | 真机设备与 `teleop_runtime` |
| `teleop_runtime` | 真机遥操、collection 和 policy deployment frontend | 通用组合或训练语义 |
| `dataset_contracts` | 纯 schema、identity、hash 和完整性契约 | runtime 与设备 |
| `dataset_interface` | projection、reader、window/batch 和 validation | 控制 owner 与 episode owner |

依赖方向必须保持：

```text
control_station / source adapters / target adapters
                         │
                         ▼
              control_composition
                         │
                         ▼
                  robot_motion
```

下层包不得导入 `control_station`；`robot_motion` 不得反向依赖
`control_composition`；`sim` 不得依赖真机 runtime。

---

## 3. 不可破坏的架构不变量

1. **每个资源只有一个 authoritative owner。** source production、chunk/cursor
   advancement、authority、canonical safety、dispatch 和 episode commitment
   分别只能有一个 owner。observer、fanout 和 UI cache 必须是只读且有界的。
2. **接收端决定事实何时可见。** producer 提供 `produced_at`；receiver 完成
   validation、deep-freeze、epoch/sequence 检查后才赋予 `available_at` 和
   admission sequence。
3. **一次计算只读取一个不可变 cut。** cut 后到达的 source、context、authority、
   gate 或 epoch 变化只能影响下一次 decision。
4. **Composition 不拥有 authority。** 它可以选择或融合数值并产生 proposal
   candidate，但不能签发 motion permit，也不能绕过下游 safety。
5. **Arbiter 不做数值融合。** base/residual、SE(3) blend、qpos interpolation
   属于 strategy；Arbiter 只按 authority、selection group 和有效性选择 proposal。
6. **机械臂混合的 canonical 表示是 Pose。** 6DoF、7DoF 或其他关节数属于
   downstream resolver。夹爪、灵巧手和其他末端可使用各自 typed qpos/native
   表示。arm-qpos mixing 仅保留为明确 debug 或未来同构臂 profile。
7. **值、候选和下游事实使用独立 schema。** `Action` 只是 component-keyed
   数值 carrier；`SourceActionSample` 表达来源事实，`ComposedActionCandidate`
   表达 composition 结果。下游可按自身能力产生 `ExecutionTarget`、可选
   `ReferenceFrameObservation`、`Command` 和 `Measurement`；这些阶段不得共用一个
   模糊的“action vector”，也不得由 `control_composition` 伪造。
8. **Composition 前不宣称 canonical safety。** source admission 只做 schema、
   identity、时序、有限值和数值可计算性校验，默认保真接纳原始数值。可选且
   default-off 的 `CandidateActionSpaceCheck` 只能提供静态动作空间早期防护；
   其结果不能改名为 safe action、不能替代下游 motion limit、IK、平滑、限幅、
   backend guard 或 motion permit。下游执行 capability 独立拥有这些语义。
9. **有效输入先保真进入 composition。** 可选的 `CompositionPreMixPolicy` 只能基于
   同一 immutable cut 做 composition-level 的 pass/no_action/invalid 决策；它不改写
   source 数值、不签发 safety/authority receipt。同步 policy 必须声明最大决策时延，
   超时转为 `no_action`；runtime 不做线程抢占，可能阻塞的实现必须改为异步 advisory。
   source drop 和 previous-result 复用作为后续 additive contract，不能隐式加入首版。
10. **`no_action`、`hold` 和 `fault` 不等价。** `no_action` 不创建或续期动作；
   composition 可以基于 exact result 向下游发出 `request_hold`，但请求本身不是
   HOLD。只有 lower execution owner 接纳请求、推进 execution epoch/generation、
   取消未执行 work 并返回 receipt 后，HOLD/FAULT 才成为权威事实。每个 profile
   必须显式声明 `no_action` 是等待、保持已接纳 reference、request-HOLD 还是 fault。
   已处于 lower HOLD 的周期不得重复 force-cancel 或推进 execution epoch。
11. **最终 dispatch 必须硬互斥。** human、replay、policy 是 proposal source，
    不分别持有 dispatcher。配置开关不能代替 owner token、epoch 和 permit。
    authority 只能在 lower HOLD/FAULT/TRANSITIONING 且 backend acknowledgement
    完成后变更；authority API 不得隐式插入 cancellation 或 backend 操作。
12. **合理的上层契约优先于旧下游兼容。** 开发阶段允许显式破坏性升级，但必须
    记录字段迁移、受影响 caller 和删除条件；不得为了迁就旧 adapter 在新公共协议
    中保留错误阶段或模糊身份。任何接线仍按 default-off、passive、sim、real shadow
    的顺序验证，协议可破坏不等于默认控制行为可以改变。
13. **来源与本体分别适配。** VIVE、手柄或 dataset 字段不得进入通用 Arbiter；
    RM65 SDK、joint order 或 actuator 类型不得进入通用 source/composition contract。
14. **模型 artifact 固定布局。** 通用 component topology 不意味着 checkpoint
    tensor 自动动态化；模型输出必须由 artifact-bound codec 解码。
15. **正式数据不可原地改写。** sealed/committed capture 只能通过带 parent
    identity 的 copy-on-write derived dataset 演进。

---

## 4. 时间、多频率与生命周期

### 4.1 必须区分的时间

| 时间 | 含义 |
|---|---|
| `sample_at` | 传感器或状态对应的物理采样时间 |
| `produced_at` | producer 形成事实的时间 |
| `available_at` | receiver 接纳完成、事实可参与 cut 的时间 |
| `[effective_from, effective_until)` | 动作或 context 的有效区间 |
| `target_effective_at` | 本次 decision 希望影响的目标时刻 |
| `dispatched_at` / `measured_at` | 实际下发与测量时间 |

跨 clock domain 的时间不能直接比较。需要比较时必须有显式 clock mapping 或统一的
receiver monotonic clock。

### 4.2 多频率规则

- 不要求频率为整数倍；选择、重放和 deadline 全部基于 timestamp。
- 当前 composition 的对齐语义是 **same-target temporal snapshot selection**，
  不是插值或 resampling：在一个 `ControlCut` 上，为同一
  `target_effective_at` 选择有效区间覆盖该时刻的每个来源最新事实。
- chunk 来源可在 admission 前显式使用可选
  `GuaranteedNextComponentAction` 做 source-intent 采样。只有来源 owner 能保证
  exact next step 将紧接 current 执行时才提供该值；没有 next 或无法保证时传
  `None`，保持当前 step/TTL 行为。采样结果以独立
  `ComponentActionTemporalSample` 绑定端点、目标时间、alpha 和算法契约，再作为
  普通 `CompositionInputDraft` 接纳；该接口不改变 composition snapshot 规则，
  也不是下游 Reference/command resampling。
- `CompositionTemporalSelectionSpec` 冻结 overlap、target lead/lag 和 action
  timestamp 语义。默认排序为 `effective_from -> receiver admission sequence ->
  fact ID`；不同 producer 的本地 sequence 不可相互比较。`RobotAction.timestamp`
  只保留 source lineage，不参与对齐。
- 慢 source 只可在 TTL 内复用；快 source 只能替换未来尚未形成的 evaluation。
- 来源参与要求属于 `CompositionSpec`：默认全部 role 必需，optional 必须显式
  声明。相对/从属分支使用无环的 role dependency；若 `B depends_on A`，A 缺失
  时 A/B 同时不参与，B required 则 A 必须 required。依赖闭包在同一 cut 的
  TTL/epoch/activation/source-eligibility 选择之后、strategy 之前执行。
- `FOLLOW_BASELINE` 相对动作已使用显式 role schema 落地：baseline 输入和输出为
  `absolute_target`，delta 输入为 `relative_delta`。Pose 采用 reference/world
  平移相加与 baseline-local 旋转右乘，qpos 采用逐元素增量相加。同一 delta 是
  状态型 offset，不按 composition tick 积分；baseline 更新时只对新 baseline
  重新应用一次。可选 delta 缺失仅在 typed rule 声明 `use_baseline` 时退化为
  baseline。LATCH 仍需独立 anchor/lifecycle 协议，CANCEL 属于 authority/execution，
  均不得由该数值规则隐式推断。
- ready evaluation 的 `composition_supported_until` 是实际参与 input TTL、
  已启用 source lease、必需 context TTL 与 maximum age 的最早截止。proposal
  只能缩短，不能把它续长；正式 publication 在同一 composer activation 内要求
  target 严格递增。shadow 可对同一 target 重新评估，但不能据此再次正式发布。
- 当前不提前发布较长未来 target 范围，也不支持对已发布 target 原地修订；因此
  不增加 replacement/version。未来若确有该需求，必须以显式 replacement identity
  和版本规则单独升级，不能复用新的 cut 暗中覆盖。
- control loop 不等待模型、视觉、网络或高级 mixing policy。
- 高频 IK、平滑、限幅、reference/command resampling 属于下游
  `robot_motion`/target，不属于 `control_composition`。Reference 不是通用必经阶段；
  需要观测时由实际生成它的 owner 从同一执行对象导出。
- store 按语义时间窗口保留数据；prune 不允许 sequence identity 被复用。
- debug 队列拥塞可以丢诊断，不能阻塞控制或 core trace。

### 4.3 暂停与 episode 结束

policy pause、worker failure、replay EOF、human deadman release 和 episode end
必须分别表达：

```text
source unavailable / TTL expired
        -> 不再产生 eligible proposal
        -> required group 无替代来源时按profile request-HOLD
        -> lower execution owner 接纳后 cancel/HOLD

episode end
        -> 关闭 publication
        -> drain/cancel 未完成 execution generation
        -> seal/commit capture
        -> 不隐式关闭仍连接的 human device
```

human 设备仍在线不代表自动取得 authority。恢复执行必须经过新的 source readiness、
必要的 re-arm receipt、authority transition 和 execution generation。source
readiness只绑定fresh emission、deadman、rebase和所需状态新鲜度；mainline不在正式
proposal前运行canonical-safety preview，安全仍由下游authoritative path执行。

---

## 5. 多源组合与反馈

### 5.1 Observation 与 policy node

不同模态保持各自 native stream，不先拼成一个全局“最新同步帧”。每次模型推理通过
`ObservationCut`选择可见事实，再由 artifact-bound `PolicyObservationView`形成该模型
需要的固定输入布局。

单策略只是只有一个 action policy node；快慢策略可以由一个 context producer 和一个
action producer 组成。二者仍通过 immutable context/action fact 连接，不共享 mutable
model state，也不新增第二套 action queue 或 authority。

### 5.2 Composition 输入

所有 contributor 先成为 immutable、receiver-admitted fact：

- action input：当前 human/replay/policy/synthetic 动作；
- context：bounded history、future action、task/lifecycle state、source health、
  vision/tactile/force feature、label、confidence 或外部 advisory；
- information plan：声明 strategy 真正需要的 channel、最大 age 和
  `metadata_only/inline/blob_ref` 访问方式。

strategy 不得在 evaluation 中读取 live device、model、sim、GUI 或 mutable backend。
昂贵或模型驱动的 mixing policy 应作为异步 advisory producer；本地 deterministic
strategy 负责 deadline、合法性和 fallback。

### 5.3 当前最小策略

- arm pose：单 owner selection 或 weighted SE(3) fusion；
- end-effector：active-mask-aware qpos interpolation；
- pose 与 end-effector component 可组合为一个带完整 lineage 的结果；
- source TTL 与 source liveness/lease 分离；
- missing、invalid、paused、fault、disconnected、rearming 均 fail closed。

高级历史/视觉/标签策略仍是开放扩展点，不应固化在主线架构中。

### 5.3.1 Strategy 热控制的最小边界

热控制不引入独立 topology/context envelope 配置树，只有两层：

- 高频 `alpha_by_component`：一份完整、receiver-admitted、same-`ControlCut`
  参数记录，`base=1-alpha`、`overlay=alpha`；
- 低频 immutable revision：完整绑定 `CompositionSpec`、component rules、
  formal strategy spec、参数 contract 与初始 alpha，并通过 predecessor hash
  做 CAS 原子切换。

action schema、component ID/representation/frame、input roles/source binding、
`CompositionInformationPlan` 和 strategy implementation 是 runtime 固定边界；
越界变化要求新 activation/runtime。action、deadman、eligibility、task/vision/history
是输入事实，不是配置。当前只实现同一 formal-component kernel 内的规则 revision，
并预注册 weighted fusion/discrete owner 两种本地 mode；不实现任意 kernel 热加载、
context 资源动态扩张或 hard authority handoff。alpha
变化只改变 realized contributor；下游 authority owner 仍是唯一 composer producer。

### 5.4 Difference 与 feedback

差异分成两个独立模块，而不是两个版本；两者都需要实现，也可以分别关闭：

1. **CompositionImmediateDifference**：由 `control_composition` 在同一 immutable
   cut 上提供 source-to-source 的 pre-mix 低资源同步观测，供 pre-mix policy、GUI
   或轻量反馈立即消费。它不包含 composed result、canonical safety、IK、dispatch 或
   measurement 事实。
2. **PostCompositionExecutionDifference**：从 composed result 开始，可比较 selected
   source、safe、IK/resolved、limited/reference、dispatched 与 measured stage。混合后
   的 passive residual 可以在同周期返回；真实执行部分通常由 `robot_motion`/target
   backend 异步提供，所有实际偏差必须绑定对应 receipt。没有 dispatch/measurement
   receipt 时不得称为实际执行偏差。

当前 A0/A1 provisional slice 已能让 backend 以 opaque receipt lineage 提交
component-keyed reference/measurement，并在同一 clock domain 内按“实际生效的
reference 半开区间覆盖 `measured_at`”计算 qpos residual。它不使用最近目标时间，不推断
authority/cancellation，也尚未接入默认 runtime。跨时钟必须等待显式 clock-mapping
receipt；arm pose 的实际偏差必须由下游 FK/frame adapter 另行产生 measured pose。

两模块使用不同的 owner、输入和生命周期，不应通过“v1/v2”表达先后关系，也不应把
第二模块的执行观测字段塞回 composition 的即时记录。当前
`SourceDifferenceObservation` 只是过渡性 carrier：pairwise 字段对应第一模块，
output-residual 字段对应第二模块，后续可在 capability/profile 下拆分，不把当前字段
集合当作长期冻结的公共接口。可视化或力反馈 adapter 可以消费任一模块，但不能借反馈
链路改变 authority。

开发阶段只冻结 owner、causal cut、stage identity、availability、信息按需声明和
non-authoritative 边界；metric catalog、默认 pair、阈值、frame 转换、debug payload
和反馈映射均保持 profile/实验级可替换。已有 hash 只用于已产生事实的 lineage，不自动
构成下游兼容承诺；新字段必须 additive 或 profile-gated，避免下游误用后再被迫维持。

---

## 6. Action、topology 与 authority

### 6.1 Component-keyed action

新增通用 carrier 使用任意 `component_id`，每个 component 绑定 exact payload
schema、frame、unit、coordinate、resource claim、dependency 和 atomic group。
标准 payload family 包括：

- `cartesian_pose`
- `joint_position` / `joint_velocity` / `joint_effort`
- `spatial_twist` / `spatial_wrench`
- `continuous_native` / `discrete_mode`

`ComponentAction` 是唯一 production proposal/Arbiter carrier。`RobotAction v1`
仅在明确的 legacy target/safety/observation edge 形成无损 materialized view；
不得根据数组长度猜 action family，也不得静默修改 v1 数值顺序。

当前 carrier 主干已经完成：

- generic catalog、payload schema 与 `ComponentAction`；
- v1 双向兼容 bridge；
- receiver-admitted component proposal；
- session/embodiment/schema/epoch/TTL same-cut reduction；
- 唯一 payload-opaque production Arbiter；
- human/replay/policy/composer 复用同一 buffer、cut、authority 和 dispatcher；
- execution manifest/ControlTrace 绑定完整 activation action schema；
- 已无调用的平行 shadow proposal/Arbiter 已归档到 `legacy/`。

现有 pose/qpos profile 在 lower compatibility edge 物化 v1 后沿用原 target、
canonical safety、IK/reference 和 dispatch。新增 velocity/wrench/effort 等表示
仍需 target capability 与 handler 才能执行。这些能力由 `robot_motion`/target
owner 实现，不属于 `control_composition`，也不要求每个 backend 暴露完全相同的
内部阶段。

### 6.2 Authority

authority 以 selection group 为单位；多个来源可以同时贡献不同 component，但最终
motion 仍通过一个 control-domain owner。

正式 compound authority 的当前下游 profile 采用保守语义：required
component/group 必须完整；任一缺失、过期、安全失败或 downstream rejection
都会请求并由 execution owner 接纳为整个 control domain HOLD。
未来若允许 pose+wrench 等共享资源组合，必须由 target capability 明确声明兼容性，
不能由 Arbiter 或 payload shape 推断。

### 6.3 为什么 target capability 是晋级前置条件

`resource_claim` 只表示“这个 component 可能影响哪些下游资源”，不表示两个
component 可以直接合并。例如 arm pose 与 spatial wrench 都 claim 同一个 TCP：
它们可能是互斥控制模式，也可能由支持 hybrid control 的 target 协同处理；
Arbiter 不能从 payload family、数组维度或来源优先级推断答案，也不会为此自动
HOLD。它只输出 candidate；target capability assessment 对未声明的组合返回
rejected，执行 owner 再决定 no-action、保持既有目标、HOLD 或其他已声明 fallback。

一个下游可执行的 target capability 至少要冻结：

1. accepted component/action schema、frame、单位和 representation handler；
2. 每个 resource 的参与 component，以及被接受的 exact active component sets；
   不允许 target 用隐式 priority 丢弃已经选中的 component；
3. candidate 如何被接纳或拒绝，以及 target/command/dispatch receipt 的身份、
   调度时间和 clock domain；
4. capability hash，以及哪些可选观测阶段可被导出，例如 target、reference、
   command 或 measurement。

`control_composition` 不统一 position、velocity、wrench、effort 的 canonical
safety state 或 handler。不同 backend 可以具有不同安全和执行管线；通用边界只要求
这些下游事实拥有准确身份、时间和状态，不能由 composition 根据 payload shape
推断或自行重算。

所以统一 `ActionArbiter` 的结果只能证明“在同一 cut 下选中了哪些
component”。该 candidate 可以向下游提交，但实际执行必须由唯一 target/execution
owner 接纳并签发自己的 receipt。下游内部是否经历 safety、IK、reference 或直接
command 是 capability 私有实现，不应反向成为 composition 的必选阶段。

当前已实现前两项的结构层：

- `ComponentTargetCapabilitySpec` 精确绑定 activation action schema；
- 每个 component 绑定 input payload、handler contract 和 output target schema；
- `TargetResourceCombinationSpec` 使用 exact active component sets 表达互斥或
  pose+wrench 等联合处理；
- `TargetCapabilityAssessment` 只返回 accepted/rejected 与原因，不产生 HOLD、
  safety 或 dispatch 语义；
- 现有 MuJoCo pose/qpos profile 通过显式 RobotAction-v1 compatibility capability
  绑定，production 数值与下发行为未改变；
- execution manifest/ControlTrace 绑定 capability hash。

表示适配采用方案 B，已经接入唯一 production execution seam：

```text
AuthoritativeArbitrationDecision
  -> ActionAdaptationRequest
  -> TargetCapabilityAssessment
  -> AdaptedActionCandidate
  -> ActionAdaptationReceipt
  -> ExecutionDirective
  -> canonical safety / resolver / target
```

这里的 `ActionAdaptationReceipt` 只证明指定 receiver 完成了结构接纳和表示转换，
不是 safety verdict、HOLD、ExecutionTarget admission 或 motion receipt。拒绝时不
进入 safety/resolver/dispatch；执行 owner 再按既有 persistence/compound 规则决定
no-action、保持既有目标或 domain HOLD。当前兼容 adapter 只在这一 lower edge
精确物化 non-safe `RobotAction v1`，不把 v1 重新带回 proposal/Arbiter。

通用薄下游协议已经建立在 `robot_motion.action.execution_handoff`：

- `TargetExecutionRequest` 绑定 exact authority、adaptation、capability、component
  coverage、target time、`supported_until` 与 authority/gate/fence sequence；
- `TargetExecutionSubmissionReceipt` 只证明 handler 接收或拒绝处理，不是 target
  admission、HOLD 或 motion permission；
- `TargetExecutionCompletionReceipt` 必须关闭每个 accepted submission，明确
  `resolved`、`rejected`、`failed` 或 `cancelled`，避免出现无 target 的悬空 processing；
- `ComponentExecutionTarget` 不要求统一 `SafeAction`，具体 safety、IK、planner、
  limiter 事实仅作为 opaque `TargetProcessingFactReference`；
- `ComponentExecutionTargetAdmissionReceipt` 与后续 activation、reference、
  command、dispatch、measurement 继续分离；target handler 还必须声明 activation
  为 `required` 或 `omitted`，required 时返回独立 activation receipt。
- 时间支持上界从 selected proposal 的最早 `effective_until` 依次进入
  `ArbitrationResult`、authoritative decision、adaptation/request 与
  `ComponentExecutionTarget`；publication、processing acceptance、target
  admission、activation、scheduled interval 和 persistence segment 都只能缩短，
  不能续命。该规则只约束时序有效性，不引入通用 safety/IK/reference。

MuJoCo Sim-HIL 已通过显式 default-off profile gate 迁入该链：adaptation 后由
MuJoCo handler 接收唯一 request，target-private canonical safety 与 coupled IK
继续复用已有数值实现，随后产生无 `SafeAction` 声明的 point target。已有
scheduled-reference receiver 是唯一 target admission/activation owner，并对
`arms + left_hand + right_hand` 执行一次原子 activation。request、submission、
completion、target、admission、activation 六类实际事实均进入 ControlTrace。

旧 `execution_target.ResolvedExecutionTarget` 仍服务于尚未迁移的 MuJoCo profile；
Sim-HIL 不再使用它。待其他 target owner 分别迁移并确认无 caller 后，再按 legacy
policy 归档旧 carrier。submission/adaptation receipt 仍不得冒充 completion、
admission、activation 或 dispatch。

### 6.4 Reference 的最终边界

Reference 仅表示某些连续控制 profile 在给定时刻实际使用的下游连续目标。它不是
`ComposedActionCandidate` 的别名，也不是所有 backend 的必经中间产物。

- 需要精确在线观测时，Reference owner 必须从**与执行相同的已计算对象**导出
  nonblocking immutable observation；禁止上层复制插值算法并重算后声称为 actual。
- 默认只导出轻量 lifecycle/identity；按 `ExecutionObservationPlan` 可选择关闭、
  降采样或导出高频 frame。观测拥塞不得阻塞执行。
- 下游必须同时导出 target lineage、scheduled time、actual effective interval、
  receiver availability 和 clock domain，关联依靠 identity/hash 与区间，不依靠
  arrival order 或 nearest timestamp。
- 插值算法 ID/hash 只用于 provenance 和离线复现。离线结果必须命名为
  `ReconstructedReference`，不能冒充 actual reference。
- 直接 command、backend trajectory 或不具备连续 reference 的 capability 可以明确
  声明 unsupported/omitted。

---

## 7. RTC 与模型部署边界

RTC 是 policy source 的生产机制，不是另一套 control system：

```text
observation cut
    -> async inference request
    -> model server returns action chunk
    -> local artifact-bound decode
    -> timed plan / one due keyframe
    -> PolicyActionEmission
    -> ordinary proposal/composition boundary
```

raw-RTC 与 model-RTC 的 owner 不完全相同：

```text
raw-RTC
  robot_motion timed plan + deployments final-chunk client

model-RTC
  dataset_contracts    版本化 continuation/artifact/training schema
  dataset_interface    离线 artifact-action prefix/target/mask projection
  robot_motion         在线 timeline、old-plan overlap、canonical conditioning bundle、epoch/plan identity
  deployments.client   injected codec编排、model-space input、protocol envelope/receipt
  deployments.<model>  artifact-bound codec、模型字段、sampler 与 transport
  future model/        learner、loss 与训练实现
```

model-RTC 是跨层 feature，不由单一 package 完整拥有，也不新增顶级 `rtc/`
package。原始 dataset parquet/video/episode schema 首阶段保持不变；训练所需
continuation 作为 additive derived view 和 artifact-bound contract 提供。
`robot_motion` 可以消费纯数据契约，并拥有在线计划因果及其 immutable canonical
conditioning bundle；它不拥有 model-space tensor、训练样本、sampler、loss 或模型消费 receipt。

长期规则：

- raw-RTC 模型 server 只需接收既有 observation 并返回 final chunk；model-RTC
  server 才显式消费 prefix/mask。模型特定适配留在 deployment/model 层，不进入
  `robot_motion` 或 `control_composition`。
- dataset 中的普通 future target、trajectory-overlap teacher prefix 和真实历史
  policy-output prefix 是三种不同事实，必须用显式 provenance 区分，不能互相冒充。
- sync queue 仍是默认路径；RTC 必须显式启用。
- worker 不得阻塞 control loop；request/result/epoch/plan identity 必须唯一。
- 每个 keyframe 只经过一次正式 emission；late completion 不能改写已 pop/commit
  的 prefix。
- RTC underrun、TTL、HOLD、FAULT 和 close/drain 必须 fail closed。
- `send_actions=false` 永远不授权 motion。
- RTC profile 当前禁止 position lookahead；在 typed preview identity、safety
  verdict、downstream receipt 和 identity promotion 完成前，不得让 future
  successor 提前影响 position reference。
- 对 composition 而言，RTC 输出只是带时间和 lineage 的 policy action source；
  composition 不读取 request、worker、chunk merge 或 transport 状态。

RTC 的具体实现状态和后续 gate 只在
[`dexteleop_rtc_landing_plan.md`](dexteleop_rtc_landing_plan.md)维护。

---

## 8. Control Station、sim HIL 与真机

Control Station 是 application leaf：

- 选择 source/target/profile；
- 提交 typed lifecycle/authority/capture intent；
- 展示可由 lower-owned facts 重建的状态；
- 通过薄 adapter 组合 human、replay、policy 与 null/sim/real target。

它不得：

- 再读一次遥操设备、dataset cursor 或 policy chunk；
- 重建一套 authority、epoch、safety、dispatch 或 episode lifecycle；
- 在 HTTP/UI 线程内执行动作或 inline side effect；
- 因 GUI 断线、重连或轮询改变控制事实。

主线验证顺序：

1. null/no-I/O contract；
2. deterministic sim；
3. sim 中接真实遥操设备，但 target 仍为 sim；
4. passive real shadow；
5. 单 source real authority；
6. 明确验收后才扩展 mixed-source real authority。

真实 VIVE/Q6 等设备配置应复用正式真机遥操数采链路的 public builder，不复制端口、
标定、retarget 或 hand mapping。新增设备只实现 source adapter；新增机械臂或 sim
只实现 target capability/binding。

---

## 9. ControlTrace 与 v2.4 数据

控制层只产生事实，不直接形成训练数据集：

```text
source/context facts
proposal + arbitration
composition + difference
safety + target/reference
dispatch + measurement
lifecycle + task outcome
        │
        ▼
ControlTrace / native streams
        │
        ▼
episode seal + commit
        │
        ▼
dataset_interface projection
        │
        ▼
TrainingView / replay / RL transition
```

记录分两层：

- **core**：实际参与控制、重放和因果解释的事实及 lineage，正式 profile 必须保留；
- **debug**：候选矩阵、中间权重、preview、model internals 和重诊断，仅在 debug
  profile 构造与保存。

视觉、触觉等 native stream 若只供数据集使用，应直接进入 capture owner；只有
strategy 也需要时才通过 `CompositionInformationPlan` 接轻量 immutable view/ref。

ControlTrace schema、writer、episode membership、WAL、seal/commit 和 reader 属于
`dataset_contracts`/`dataset_interface`/capture runtime，不属于
`control_composition`。

---

## 10. 当前成熟度

| 能力 | 当前状态 | 不能外推为 |
|---|---|---|
| `control_composition` 通用包 | context/input admission、显式 temporal selection、time window、source eligibility、pose/qpos 最小策略、lineage、default-off pre-mix、immediate/source-to-composed observer，以及 downstream-owned reference/measurement qpos association 已有 pure/scoped 验证 | observer 的 production demand/budget wiring、cross-clock/pose/real active-reference difference、production mixed-source authority 或 motion safety |
| strategy hot control | opt-in component alpha latest-covering stream、same-cut binding、weighted/discrete-owner 原子 revision、CAS 与 ControlTrace 投影已完成 scoped contract 验证 | 已接 Control Station、动态 kernel 加载、context 驱动 meta-strategy、hard authority handoff 或 sim/real production |
| component-keyed carrier | EG1/EG2 完成；EG3 proposal/Arbiter shadow 完成；正式 evaluation 名称已改为 candidate 语义 | 已完成统一 production proposal/target 迁移 |
| human/replay/policy source | 已有各自 emission/adapter 与 scoped sim 证据 | 统一 production GUI 已迁移 |
| Control Station | 已有 typed intent、publication barrier 和 scoped sim/HIL 装配工作 | 真机 motion authorization |
| RTC | async producer、timed plan、single emission、TTL/epoch/close 等 scoped 验证；默认关闭 | production worker/hardware readiness |
| sim execution | 已有 authoritative/scheduled reference 的 scoped fixture | 真机 stop/latency/receipt 证据 |
| real execution | legacy owner 仍是正式路径；shared path 主要为 passive/default-off seam | owner 已迁移 |
| v2.4 trace | schema、WAL、episode seal/reader 和多个 formal fixture 已有 | 所有 production entrypoint 已接 writer |

测试通过只能证明其声明范围。pure contract、deterministic sim、no-action shadow 和
真机 motion authorization 必须分别报告，不能互相替代。

---

## 11. 下一步 gates

### Gate A：完成 sim HIL 主线

目标是已有 sim policy、sim target 和真实遥操设备可在 Control Station 中组合，并
形成完整 capture。

退出条件：

- source adapter 不复制 device/model/replay owner；
- arm 保持 pose，末端保持 typed qpos/native；
- pause、deadman、EOF、worker fault、episode end、re-arm corner case 全部有
  deterministic HOLD/authority 行为；
- immediate source difference 可返回；
- core ControlTrace 可 seal/commit/read；
- legacy teleop/replay/deploy 默认行为 parity。

### Gate B：component-keyed carrier 收口（已完成主干）

唯一 production `ActionProposal` 已直接承载 `ComponentAction`；human、replay、
policy、composer 复用同一 ProposalBuffer、cut、Arbiter、authority 与 dispatcher。
一次 activation 绑定完整 schema，source 只提交其实际覆盖的 component；coverage/
selection group 不再由调用方重复声明。旧 `RobotAction` 只在明确的 target/safety
或 legacy observation bridge 物化。

剩余工作：

- 把仍使用 `RobotAction` 的 legacy composition-input 数值算法迁到 native
  component values；
- 将 Sim-HIL 之外仍使用 MuJoCo `ResolvedExecutionTarget` compatibility carrier
  的 profile 分别迁到 component-keyed handoff；不得用全局开关一次改变所有部署；
- Sim-HIL command、dispatch receipt 已关联 exact request/target；继续把相同薄协议
  接到其他 sim/real target owner，并由各自 owner 报告 activation/dispatch 时间；
- production observation demand wiring 仍需按 target capability 接入。

最小晋级链路是：

```text
ComponentAction candidate
  -> one proposal admission/Arbiter/authority path
  -> target capability schema/resource decision
  -> downstream-private processing
  -> target/command/dispatch receipt
```

`control_composition` 不冻结 position、velocity、wrench、effort 的 safety handler；
这些是具体 target owner 的内部职责。退出条件是所有 active source 共用一个 proposal
owner，旧/新 carrier 不形成并行 authority，且 sim 中 identity/hash 和数值 parity
通过；在此之前不得把 shadow carrier 接 real authority。

### Gate C：RTC 接入普通 policy source

RTC 独立完善后，只通过正式 `PolicyActionEmission` adapter 接入。Control Station 和
composition 不得接收完整 RTC frontend step，也不得管理 worker、plan 或 inference epoch。

退出条件：

- default-off 与 sync parity；
- cold/warm/late/underrun/close/drain 鲁棒测试；
- no-action real observation shadow；
- source binding、artifact/capability 和 ControlTrace lineage 完整。

### Gate D：real shadow 与 authority 迁移

顺序固定为：

```text
passive observation
  -> counterfactual safety/target
  -> no-command real shadow
  -> owner/worker drain evidence
  -> explicit single-source motion authorization
  -> mixed-source authorization
```

必须具备真实 stop/latency、epoch/generation guard、backend acceptance、measured
deviation 和 operator approval。sim 通过不能替代这些证据。

---

## 12. 近期不做

- 不一次重写现有 GUI、teleop、replay、policy deploy 或 position executor。
- 不让 `control_composition` 感知 RTC、checkpoint、MuJoCo、VIVE 或 RM65。
- 不为尚未出现的本体预建大量空 adapter；由明确第二本体驱动 capability 验证。
- 不在缺少 typed preview receipt 时恢复 RTC position lookahead。
- 不把高级 model-based mixing 放进 control loop。
- 不让 v2.4 writer、视觉/触觉采集或 feedback 成为 composition 的隐式职责。
- 不因 shadow/sim 测试通过而授权真机 motion。

---

## 13. 完成判据

架构主线只有同时满足以下条件才可称为通用闭环：

1. 新 source 仅通过 adapter 即可参与同一 proposal/composition contract；
2. 新本体仅通过 component/capability/resolver/backend binding 接入；
3. 新 model 仅替换 artifact-bound observation/action codec 和 transport；
4. sim 与 real 复用相同 proposal、authority、safety 和 trace 语义，但各自独立
   capability 与 motion authorization；
5. 不同频率、late completion、pause、fault、episode end 和 re-arm 可确定性重放；
6. source、composed、safe、resolved、dispatched、measured 的差异和 lineage 可查询；
7. 默认 legacy 行为、动作顺序、pop count 和 hash 在未启用新 profile 时保持不变；
8. 所有正式动作都能追溯到 exact input cut、authority、safety、target receipt 和
   episode commit；
9. 任何 observer、GUI、debug recorder 或 dataset projection 失效都不能获得或改变
   motion authority；
10. 真机执行只在具体 profile 的安全、停止、时延和 operator gate 全部通过后启用。

这份文档只在上述长期边界、当前成熟度或 gate 顺序发生变化时更新；单次实现、测试数
和临时问题不再追加到主文。

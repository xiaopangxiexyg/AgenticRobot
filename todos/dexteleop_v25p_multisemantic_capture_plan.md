# DexTeleop v2.5-p 高层多语义通用采集落地计划

状态：已由 `todos/dexteleop_v25p2_collection_semantics_plan.md` 取代；保留作为
v2.5-p1 的宽语义探索记录，不应直接实施  
参考：`todos/long_horizon_dataset_plan.md`、`todos/dexteleop_v25p_native_landing_plan.md`

## 1. 目标与边界

本计划把长程任务建议收敛为 v2.5-p 的下一条可实施路径。目标不是立即把
所有高层语义写进真机 GUI，而是先建立一个通用、可回退、支持在线采集与后标注
共存的语义事实层。

第一条纵向链路只需可靠表达：

1. 当前执行哪个 Task Node；
2. 当前处于该 Task Node 的哪个 Phase；
3. 每次 Attempt 的成功、失败、中止或未知结果；
4. 同一 Task Node 可以失败后再次成功，但默认不允许两个相同的成功 Attempt
   连续出现；
5. Episode 结束时分别记录任务结果、场景结果和终止原因；
6. 后标注可以补充或修正语义，但不能重写已封存的原始采集事实。

本阶段明确不做：

- 不推进长程任务 GUI；
- 不实现 v2.4 的多来源、authority、reference、dispatch 或 ControlTrace 时序；
- 不把 Task Node、Phase、Attempt 塞进 v2.3 的 action/state 物理列；
- 不改变当前 v2.3 正式 writer、LeRobot v2.1 兼容输出或默认真机/UMI 命令；
- 不建立第二个 Episode owner；
- 不用一个任意嵌套的“大语义 JSON”代替明确契约。

## 2. 现状判断

当前 v2.5-p 已具备以下基础：

- Episode owner 生成并向 legacy/native 路径传递同一个
  `EpisodeCaptureIdentity`；
- component-keyed native stream 能表达不同本体、单双臂、末端和任意合法张量；
- RealTeleop 与 UMI 已有 default-off producer adapter 和 dual-write 等价性验证；
- native store 已具备 CJSON 与 Parquet chunk 两种物理布局；
- v2.3 exact projection、omission receipt 与兼容 reader 已存在。

现有 `dataset_contracts.task_semantics` 只覆盖简化的 V1 Subtask Attempt：

- 它适合保持现有数据可读；
- 它不足以表达 Task Node、Phase、Scenario、Invocation、Episode Outcome 和后标注
  revision；
- 当前事件逐文件保存以及语义随 native capture 一起封存，不适合长程任务和
  capture seal 后的标注；
- 当前在线入口以 wall clock 和 frame count 为主，不能作为最终统一的因果时间
  合同。

因此不应继续给 V1 `LongHorizonSubtask` 和 `SubtaskAttemptEvent` 追加可选字段，
也不应静默把 `subtask` 重新解释为新的 Task Node。

## 3. 必须冻结的兼容边界

### 3.1 V1 保持原义

- 现有 V1 schema ID、hash domain、序列化形式和 reader 保持不变；
- 旧 artifact 继续由 V1 reader/projector 读取；
- 新 writer 不生成带旧 hash、却按新语义解释的 artifact；
- 如需迁移，使用显式 V1 -> V2 adapter，并生成 migration receipt；
- V1 Subtask 可以映射为 V2 Task Node，V1 Attempt 可以映射为 V2 Attempt；
  缺失的 Phase、Scenario 和 Episode Outcome 必须保持缺失，不能猜测。

### 3.2 默认正式采集保持不变

高层语义 writer 必须和 native v2.5-p 一样是 subordinate writer：

- Episode 身份由现有 dataset episode owner 生成；
- 它不能自行开始、停止、编号、重排或提交 legacy Episode；
- 任一语义写入失败不得伪造 legacy 成功；
- 在显式开关落地前，RealTeleop/UMI 默认命令不加载新 writer。

## 4. 所有权与包边界

```text
dataset_contracts
  纯 profile、fact、identity、hash 和 validation contract
          ↑
dataset_interface
  迁移、投影、reader、resolved view、window/batch、导出
          ↑
teleop_runtime.collection
  接收 episode owner 的身份和采样时钟，持久化 capture facts
          ↑
control_station / offline annotation frontend
  只产生 typed semantic intents，不拥有 writer 或 episode lifecycle
```

具体约束：

- `dataset_contracts` 不 import runtime、GUI、device 或控制模块；
- `dataset_interface` 不成为第二个状态机、Episode owner 或 control authority；
- `teleop_runtime.collection` 拥有语义 capture port 到持久化的绑定；
- `control_station` 未来只发 `BeginAttemptIntent`、`BeginPhaseIntent` 等 typed
  intents，不重建 canonical timeline；
- post-annotation tool 产生独立 annotation set，不修改 capture bundle；
- dense mechanism stream 仍由 v2.5-p component stream 表达，不新建一份重复的
  robot/action 大记录。

## 5. 概念模型

### 5.1 静态定义

#### `TaskProfileV2`

只描述任务拓扑，不保存某次执行结果：

- `task_profile_id`、schema/hash；
- Task Node 定义；
- 每个 Task Node 可用的 Phase 定义；
- Task Node 和 Phase 的允许迁移；
- 默认成功重复策略；
- 可选的人类说明、训练说明和 guideline 引用。

Task Node 表达可重试的高层工作单元。Phase 表达一次 Attempt 内部的执行阶段。
不能用 `failed_align`、`recovery_grasp` 这类结果或机制状态污染 Phase vocabulary；
失败、恢复和异常分别由 Attempt outcome、Invocation、Scenario/Mechanism fact 表达。

#### `CollectionScenarioProfileV1`

描述“本次计划采什么”，而不是“实际发生了什么”：

- 所绑定的 Task Profile；
- nominal/off-nominal purpose；
- 计划的 entry/setup/perturbation；
- target result；
- stop condition 和安全边界；
- 可选的参数 schema。

计划目标不得被 reader 自动当作实际 `scenario_result`。

### 5.2 Episode 绑定与最终结果

#### `EpisodeSemanticBindingV1`

必须在第一条语义事实前封存：

- `episode_capture_identity_hash`；
- Task Profile hash；
- 可选 Collection Scenario Profile hash；
- task instruction；
- recording scope；
- context source；
- entry context；
- open-boundary seal policy；
- producer/build 信息。

#### `EpisodeOutcomeV1`

在停止或封口时产生为一条 final typed fact，三类结果必须分开：

- `task_outcome`：任务是否成功；
- `scenario_result`：计划场景是否实际发生及其结果；
- `termination`：为何结束，如 completed、operator_stop、collision、fault、timeout。

碰撞停止不自动等于任务失败，除非 profile 或标注策略明确这样解释。

### 5.3 执行事实

所有执行事实使用统一 envelope，但 payload 是严格的 tagged union，而不是任意 dict。

#### `SemanticFactEnvelopeV1`

公共字段只保存一次：

- `fact_id`；
- Episode identity hash；
- capture/annotation scope 内单调 `sequence_index`；
- `effective_at` 与其 clock；
- `available_at` 与其 clock；
- origin、actor/annotator；
- 可选 confidence；
- `source_fact_ids`；
- `supersedes_fact_ids`；
- typed payload schema ID/hash。

首批 payload：

- `TaskAttemptBoundaryV2`：begin/finish、node、attempt、track、outcome；
- `PhaseBoundaryV1`：begin/end、phase、attempt、phase track；
- `EpisodeOutcomeV1`。

后续 payload：

- `InvocationBoundaryV1`：人工/策略/工具调用及独立的 outcome/termination；
- `ScenarioEventV1`：计划参数与实际参数、实际结果；
- `MechanismEventV1`：稀疏事件及其源 stream/record/detector 引用。

### 5.4 为什么 Attempt 和 Phase 都要有边界

Attempt 边界是 canonical，不完全由 Phase 反推。原因是：

- 在线采集可以知道正在执行某个 Task Node，却暂时没有 Phase 标注；
- Episode 可在两个 Phase 之间被中止；
- post annotation 可能只补 Task Node，不补细 Phase；
- 用 Phase 合并推断 Attempt 会把“缺标注”误判成“不在 Attempt”。

Phase interval 可以由边界事件在 seal/read 时物化；物化结果是派生索引，不是第二份
可独立修改的真源。后标注若直接画区间，也必须先转换成有 provenance 的 typed
annotation fact。

## 6. 时间、定位与因果语义

### 6.1 两个时间不可合并

- `effective_at`：事实在 Episode 内实际生效的时间；
- `available_at`：该事实何时可被在线系统或训练样本观察到。

在线操作通常两者接近。后标注可以对早期 capture time 生效，但其 availability
属于标注时刻。因果训练 view 必须能排除当时尚不可见的后标注。

### 6.2 Episode owner 提供采样时钟

首批 runtime adapter 不再自己用任意 wall clock 作为主时钟：

- collection owner 提供 Episode 内 monotonic capture time；
- frame/sample index 只作为定位提示，不作为唯一时间真源；
- 若事实落在采样间隔中，保存有效时间和相邻 sample locator；
- 若引用具体样本，优先引用 causal sample hash/record identity；
- 不同 clock 之间没有明确 binding 时禁止数值比较。

## 7. 状态约束与验证规则

最小 fail-closed 规则：

1. 每个 Task track 同时最多一个 active Task Node Attempt；
2. Phase 必须属于当前 active Attempt；
3. 同一 phase track 不允许重叠；
4. Task Node/Phase 必须存在于绑定的 profile；
5. transition 必须满足 profile；
6. 同一个 Task Node 允许 failure 后再次 success；
7. 默认禁止两个相同 Task Node 的 successful Attempt 连续出现；
8. finish 必须引用已 begin 且尚未 finish 的 ID；
9. Phase/Attempt 越过 Episode 边界时必须显式 truncated/auto-closed；
10. auto-close 不得伪装成人工事实，必须携带 synthesized origin 和原因；
11. planned scenario target 不得作为 actual result；
12. 缺失语义使用 null/unknown/not_observed，不得编码成 false、zero 或 success；
13. capture facts 与 annotation facts 不能共享可变 sequence owner；
14. supersession 必须保持被替代事实可读，禁止原地覆写。

Profile 可选择 seal 时遇到 open Attempt 的策略：

- `fail_seal`；或
- 由 Episode owner 合成 `unknown/aborted` finish，并记录明确 provenance。

正式配置选哪一种应由 `EpisodeSemanticBindingV1` 明示，不能由 writer 猜测；它是
本次采集的封口策略，不污染可复用的 Task Profile 拓扑。

## 8. 在线事实与后标注的不可变组织

当前 native capture manifest 不应因后标注而失效。建议分成两个不可变层：

```text
meta/v25p/native/
  profiles/
    task_v2/<task_profile_hash>.cjson
    scenario_v1/<scenario_profile_hash>.cjson
  episodes/<episode_uuid>/
    semantics/
      binding.cjson
      capture/
        chunks/<chunk_index>.<physical_format>
        manifest.cjson
  annotations/<annotation_set_id>/
    binding.cjson
    chunks/<chunk_index>.<physical_format>
    manifest.cjson
  semantic_views/<resolved_view_receipt_hash>.cjson
```

- capture bundle 与 native sample capture 一起封存；
- `EpisodeOutcomeV1` 位于 capture fact log 中，由 manifest 建立唯一 final-outcome
  索引，不再另存一份可分叉的 canonical outcome；
- annotation set 绑定 Episode identity、native capture manifest、Task Profile 和
  annotation policy；
- 每个 annotation set 独立不可变，可以在 Episode seal 后新增；
- resolved view 明示选择了哪些 capture/annotation set、冲突策略、因果模式和
  resolver 版本，并产生 receipt；
- reader 默认返回 raw layers，不静默选“最新标注”。

## 9. 物理存储与逻辑 hash 解耦

逻辑事实 hash 不依赖“每条事实一个文件”或“一个 chunk 一个文件”。首批采用
`semantic_event_chunks_v1`：

- 小规模可用 compact CJSON chunk；
- 长程规模优先 Parquet chunk；
- 每个 chunk 保存有序 facts，并声明 sequence/time range、row count 和 bundle hash；
- manifest 保存 chunk 顺序和总事实 hash；
- 默认 128 或 256 facts/chunk，具体值通过基准测试决定；
- 单条/区间随机读取由 chunk index 支持；
- crash 时未封存 chunk 不得形成可验证 manifest；
- 不为每个 Phase/Attempt/Mechanism event 发布一个小文件。

静态 Task/Scenario Profile 按内容 hash 只保存一次。

## 10. Mechanism semantics 与 native component stream

`long_horizon_dataset_plan.md` 中的连续机制信号，不应再汇总成第二份
`mechanism_sample`：

- action、qpos、pose、velocity、wrench、contact 等连续量仍是 v2.5-p component
  streams；
- 只有 producer 真正提供并有稳定 schema 的信号才能成为 stream；
- 碰撞、急停、grasp-loss、slip 等稀疏事件使用 `MechanismEventV1`，并引用源
  stream records、detector profile/version 和阈值；
- TCP speed、jerk、workspace cell、contact summary 等可重建量默认由
  `dataset_interface` 派生，输出 derivation receipt；
- 不复制大数组进事件 payload，除非该值无法从被引用 artifact 重建；
- source/accepted/dispatched 等 value stage 只有在现有 provenance 能准确提供时
  才写入，不为未来 v2.4 合并提前伪造。

## 11. Reader、训练 view 与 LeRobot/v2.3 兼容

### 11.1 原始数据不改旧行结构

- legacy v2.3 writer 不变；
- v2.5-p 高层语义是绑定同一 Episode identity 的独立事实层；
- 不把 Task Node/Phase/Attempt ID 拼进 action vector；
- 不用默认值填充缺失标注。

### 11.2 显式投影

`dataset_interface` 提供单独的 V2 projector：

- frame view：按请求生成 active node、attempt、phase 和 outcome columns；
- interval/window view：适合 invocation、VLM 或长程策略训练；
- causal view：只使用 `available_at` 当时可见的事实；
- retrospective view：可显式选择后标注 annotation set；
- export receipt 记录 profile、annotation sets、resolver、missingness 和列映射。

旧 `project_task_semantics_to_frames` 保持 V1 语义，新 V2 不复用其名称或静默改变
返回结果。

导出到 LeRobot v2.1/v2.3 时有两种合法方式：

- 保留 sidecar/episode metadata；或
- 由显式 converter 加声明过的 frame/episode columns。

两种方式都必须产生 conversion receipt；不得把高层语义隐式混入 observation 或
action。

## 12. 为 v2.4 合并预留但暂不实现的接口

本阶段只预留稳定引用位置：

- final v2.5 commit 将来可以引用 native sample capture manifest；
- 可以引用 capture semantic manifest；
- 可以引用选定 annotation sets/resolved-view receipt；
- 可以引用现有 immutable v2.4 capture commit；
- sparse semantic fact 可以引用外部 ControlTrace fact/hash。

本阶段不把 authority、execution route、reference、source precedence 或
multi-source timing 放进 Task/Phase schema。未来 v2.4 是 provenance/mechanism
证据层，Task semantics 是高层执行事实层，两者通过不可变 ID/hash 关联，而不是
彼此嵌套或复制。

## 13. 分阶段落地

### M0：术语和 V1 冻结

- 在 contracts README 明确 Subtask V1 与 Task Node V2 不同；
- 固定新 schema ID/hash domain；
- 写出 V1 -> V2 migration 的可表达性/缺失 receipt；
- 不改 runtime。

退出条件：旧 fixture hash 不变，旧 reader 测试通过。

### M1：纯 V2 contracts

先落地：

- `TaskProfileV2`；
- `CollectionScenarioProfileV1`；
- `EpisodeSemanticBindingV1`；
- `SemanticFactEnvelopeV1`；
- Attempt/Phase/EpisodeOutcome payload；
- state validation 和 CJSON/hash round-trip。

退出条件：failure -> same-node success 合法；same-node success -> success 默认失败；
Phase 嵌套、transition、unknown/truncated 和 clock validation 都有负例。

### M2：chunk store、reader 与 annotation set

- capture/annotation 两种 subordinate store；
- CJSON/Parquet chunk 逻辑 hash 等价；
- crash-incomplete、seal、random read 和小文件数量测试；
- immutable annotation set 与 resolved-view receipt。

退出条件：seal 后新增 annotation 不改变 native/capture manifest；原始与 superseded
facts 都可验证读取。

### M3：default-off collection semantic port

- Episode owner 把同一 identity、sample clock 和 locator 传给 semantic port；
- RealTeleop/UMI 使用各自 adapter，不互相 import；
- 只接 Attempt、Phase、EpisodeOutcome，不接 GUI；
- thread/process/interrupt/restart 使用类真机 fixture 验证；
- legacy v2.3 输出逐字段保持不变。

退出条件：未开显式 flag 时零行为变化；开启后语义失败不会伪造 legacy/native
commit；相同 Episode identity 在所有 manifest 中一致。

### M4：后标注与训练投影

- offline typed annotation API；
- capture-only、causal、retrospective 三种明确 view；
- frame/interval/window projector；
- LeRobot/v2.3 显式 converter 与 receipt。

退出条件：同一 raw Episode 可生成不同可复现 view，而 raw capture 不变。

### M5：Scenario、Invocation 与 Mechanism event

- 按真实 producer 能力逐类接入；
- 连续信号仍进 component stream；
- event 只保存不可重建事实和引用；
- detector/version/threshold 有完整 provenance。

退出条件：planned 与 actual 结果不会混淆；derived facts 可由 receipt 重建。

### M6：GUI（明确延期）

在 contracts、store、offline intents 和 resolved views 稳定后，才由
`control_station` 增加 typed operator intents。GUI 不直接写文件、不推进 Episode
sequence、不拥有 Task state machine。

### M7：v2.4 合并 gate（明确延期）

最终 v2.5 commit 共同验证 native samples、task semantics、compatibility receipts
以及选定的 v2.4 capabilities。只有该 final contract 可以宣称 native-v2.5
eligible。

## 14. 第一批实现范围建议

下一次代码落地只做 M0 + M1，不直接铺开全部实体，也不接真机：

```text
TaskProfileV2
EpisodeSemanticBindingV1
TaskAttemptBoundaryV2
PhaseBoundaryV1
EpisodeOutcomeV1
SemanticFactEnvelopeV1
V1 -> V2 migration receipt
```

`CollectionScenarioProfileV1` 可以在 M1 一并定义纯契约，但不在首批 runtime 中
选择。Invocation、Scenario event、Mechanism event 等到 M2/M5 的存储与来源边界
稳定后再接。

这个切片已经能够无歧义地满足“当前子任务阶段”和“某次子任务成功/失败”，同时
避免为未来每种语义各建一套 writer、时钟、Episode lifecycle 和物理文件布局。

## 15. 回退策略

- 所有新 schema 使用新 ID/hash domain；
- 默认采集不开启新 semantic port；
- 回退运行行为只需不附加 semantic writer；
- V1/V2 readers 可并存，已封存 artifact 不重写；
- 删除 provisional writer 不能删除其 pure reader；
- 如 M2 物理布局需要回退，可换回同一逻辑事实 hash 的 CJSON chunk，不改事实
  契约；
- v2.4 merge 未完成前继续保持 `native_v25_eligible=false`。

## 16. 验收总表

- **内容正确**：Task/Phase/Attempt/Outcome 不混义，缺失不伪造；
- **通用性**：契约不包含 RM65、VIVE、UMI 或单双臂特例；
- **单一所有权**：一个 Episode owner、一个 capture sequence owner、每个 annotation
  set 一个 append owner；
- **因果性**：effective/available time 可区分，跨 clock 必须有 binding；
- **可追加标注**：seal 后 annotation 不改变 raw capture；
- **存储效率**：事件 chunk 化，无每事件小文件爆炸；
- **随机读取**：manifest 可定位 sequence/time range；
- **兼容性**：v2.3/LeRobot 导出显式且有 receipt；
- **可回退性**：默认关闭、旧 hash 不变、旧 reader 不删；
- **v2.4 准备**：只保留 immutable reference/hash 插槽，不提前混入其 authority 和
  multi-source 语义。

# DexTeleop v2.5-p2 通用采集进度与后标注规划

状态：capture-truth 合同与 application-neutral progress core 已离线落地；默认采集行为未修改
前置：v2.5-p1 native component capture、episode identity、兼容投影
旧 long-horizon/multisemantic 草案的有效目标已吸收到本文；本文是 p2 的当前规划真源。

命名说明：`v2.5-p2` 是数据协议和迁移阶段名称，不是 production Python package
名称。运行时正式领域名统一为 `collection_runtime.progress`；不得新增
`v25p`、`v25p2` 或 `subtask_interface` 之类并行实现目录。

当前实现边界：

- 已落地 `dataset_contracts.progress_profile/progress_facts`；
- 已落地 `collection_runtime.progress` requests、reducer、durable controller、
  append-only store、restart/outbox recovery 和 termination seal；
- Profile、Episode identity、request、fact envelope 和 store 配置均已改为严格类型
  校验，不接受 `"false"`、字符串时间或非整数 ordinal 等有损隐式转换；
- 已落地 `dataset_interface.progress_timeline` 第一版只读 resolver：独立校验
  capture facts，重建 canonical Attempt/transition/intentional-pause timeline，并校验
  sealed manifest；
- 已落地线性 `REASSIGN_FAILURE_TO_ATTEMPT` 的只读 correction 原型：只绑定 sealed
  progress manifest，不改 capture facts；它只验证 immutable retrospective overlay
  的可行性，不作为最终后期矫正合同；
- 已落地版本无关的 `dataset_interface.sample_axis/progress_samples` 只读投影：
  将同一 Episode 时钟上的 causal sample cut 与 Attempt/intentional-pause 时间区间
  映射为明确的半开 sample interval，不读取或改写 stream payload；
- 已落地 `dataset_interface.progress_exclusions` coarse-only 只读训练视图：
  只对 sealed capture-time timeline 中的在线失败 Attempt 生成
  `[Attempt start, capture-causal Failure cut]` 排除区间（普通 operator-report 或
  已验证的 promoted-Hold cut），不推断 refinement 或 retrospective correction；
- 已落地最小 `dataset_contracts.progress_refinement` 与
  `dataset_interface.progress_refinement` salvage-anchor 原型：profile 只冻结边界身份、
  顺序、前缀证明与必要 predecessor coverage；
- 已落地最小 `dataset_contracts.progress_annotations`、
  `collection_runtime.progress.RefinementAnnotationStore` 与
  `dataset_interface.progress_annotations`：annotation set 只绑定 sealed progress
  manifest 与 refinement profile，boundary facts 按小型 immutable chunk 追加，seal 后
  reader 可独立校验并重建 selected read view；annotator provenance、supersession 与
  correction 仍未冻结；
- 已落地 `dataset_interface.progress_coverage` 结构性 Attempt coverage 派生视图，
  以及 `progress_coverage_catalog` 可删除、可完整重建的 SQLite WAL 查询缓存；
  selected/training/coverage/catalog 均保留精确 annotation manifest hash lineage，
  coverage 不使用 confidence，training/salvage policy 仍显式独立；catalog 当前格式为
  v2，旧派生数据库必须删除重建，canonical 数据不受影响；
- 增加必要的 `EpisodeProgressStartedV1`，因为 binding 不能单独恢复首个 Attempt
  identity；
- store 接收 Episode owner 指定的目录，尚未冻结全局 `meta/v25p2` 物理落点；
- Progress fact append/seal 遇到 write/fsync 不确定失败后会 poison 当前 store，
  只能关闭重开并由完整 record replay 或 torn-tail fail-closed；
- 同一 Episode 的 capture-time fact `effective_at` 必须单调不减；Hold/Resume
  operation 的 request/effect 时间不得晚于其 receipt 的 `available_at`；这些约束由
  live reducer 和 read-time resolver 对称执行，避免“在线已封存、离线不可读”；
- seal 先 fsync 并关闭 append fd，再发布 manifest；manifest 是最后的 commit marker，
  且其 `sealed_at` 不得早于最后一条 termination fact；
- ROOTED_TREE 合同和离线 reducer 已支持，但 controller 默认 gate 为 LINEAR；
- `FULL_TASK` 和 `PROGRESS_STEP_WINDOW` 已有明确且经过测试的结束语义；
  `TRANSITION_WINDOW` 和通用 `EVENT_WINDOW` 尚无专用 target/boundary binding；
  RealTeleop 仅对同时具有精确 `ProgressStageEntryBindingV1` 且
  `entry_condition=RECOVERY_FROM_FAILURE` 的恢复补采使用保守
  `EVENT_WINDOW`，它只表达从指定失败 stage 恢复的非完整 capture window，
  不推断 event 已完整覆盖或 full-task success；
- general correction、annotation-set selection 和派生视图发布 manifest 仍未落地，
  明确延后到真实采集数据能够驱动其最小设计之后；
- RealTeleop 已落地默认关闭的一个或多个 step 的 `LINEAR/FULL_TASK` adapter；
  typed Advance/Undo/Failure、Failure→Hold/Pause→Resume→首个 writer-accepted sample
  Retry 已接线，Control Station 只通过 structural ports 协调，真实 Session/recorder
  effect 仍由 RealTeleop owner 执行；Control Station 已能从当前正式 task 对应的
  managed LINEAR profile 中进行显式选择，Tree 和 UMI 尚未接入。未提供 profile
  或 GUI 配置时现有默认数据输出不变；
- RealTeleop 数据集级 progress profile 使用既有 `TaskProgressProfileV1` 直接作为
  不可变 canonical 绑定，落点为 `meta/progress/profile.cjson`，不新增冗余 binding
  schema。首个 progress Episode 后按完整 semantic hash 锁定；重启自动恢复，同一
  数据集拒绝不同 profile、相同 ID 的不同内容，以及 profile-less/profiled Episode
  混装。
- 已落地 `TaskStageProfileV1`、capture-time `ProgressStageEntryBindingV1` 与数据集级
  `meta/progress/stages.cjson` 唯一绑定；非首 fine stage 入口必须显式声明
  `RECOVERY_FROM_FAILURE`，同一数据集拒绝 stage profile 混装；
- 已落地独立的 Control Station 细粒度阶段标注应用。它只发现 `episodes.jsonl`
  已发布且 progress manifest 已 sealed 的旧 Episode，一个数据集同时只允许一个
  annotation writer，但可以与新 Episode 采集并行；启动命令无需 dataset/annotator
  参数，二者在 loopback GUI 中确认后才延迟打开 workspace 和取得 writer lock，
  CLI 参数仅作为可选预填；
- 已落地 immutable stage occurrence/abnormal annotation set、原子草稿和
  `dataset_interface.progress_stages` 派生索引：Failure 排除从当前失败 stage occurrence
  起点到原始 Failure cut，Retry 继续同一 stage；whole-Episode abnormal 只改变默认
  training eligibility，不删除原始数据或已有标注；

## 0. 数据采集目标优先于架构

p2 首先服务数据采集，而不是追求语义实体数量或软件层次完整。后续任何合同、
目录、状态机和索引设计都必须先回答：它是否让数据更正确、更容易采、更容易补标，
以及更容易用于训练。

按优先级排列，目标是：

1. **原始内容严格正确**：传感、动作、时间和 Episode 身份不能因语义扩展而被
   改写、错位或丢失；
2. **数采员负担最小**：在线只管理粗子任务和异常 failure，不要求实时完成研究级
   精细标注；
3. **成功数据尽可能保留**：failure 不能粗暴废弃整条 Episode；后标注越精细，
   越能恢复 failure 之前仍然有效的片段；
4. **失败数据含义明确**：失败事实、无效训练区间和人工恢复不能混进默认成功轨迹；
5. **采集热路径稳定**：语义记录和后标注不能造成控制、相机、传感或主 writer
   周期性阻塞；
6. **采集与标注可并行**：补标不重写高带宽原始数据，不要求停止新的 Episode
   采集；
7. **训练选择可复现**：include/exclude 不是散落在帧里的临时布尔值，而是由
   failure、标注层和显式 resolver policy 生成的可验证索引；
8. **本体无关、逐步升级**：不同机器人、单双臂、末端和传感器都可以使用同一
   Episode/时间/语义绑定机制，旧数据继续可读。

如果某个架构抽象不能直接改善这些目标，也不是未来兼容所必需，就不应在 p2 首批
实现中加入。

## 1. p2 的准确目标

v2.5-p2 不再尝试让前期数采员实时维护完整的长程任务语义。在线接口只暴露
两个能力：

1. 管理当前粗粒度子任务进度；
2. 在异常时记录当前 Attempt 失败。

正常轨迹默认成功。进入下一个子任务时，当前 Attempt 隐式成功；Episode 正常
结束时依据本次 recording scope 结束当前 Attempt，但只有完整任务从合法 terminal
step 结束时才能推断 full-task success。数采员不需要额外按“成功”。

子任务内部的 A1/A2/A3 等精细划分全部属于后处理。数据协议必须允许：

- 只有粗粒度在线语义；
- 只有一部分 Attempt 完成了精细标注；
- 同一数据集同时存在未标注、部分标注和完整标注 Episode；
- 后标注不重写原始帧、视频或 native stream；
- 新标注可以立刻改变训练有效区间和查询结果；
- 未标注、部分标注以及特定 profile 版本下未完成的数据可以快速检索。

## 2. 版本与回退边界

- v2.5-p1 的 Subtask V1、schema ID、hash domain、reader 和 artifact 保持原义；
- p2 使用新的 schema ID、hash domain、capability 名称和 reader；
- p1 artifact 不按 p2 规则静默解释；
- p1 -> p2 只能通过显式 migration adapter 和 receipt；
- p2 默认关闭，不改变 v2.3、LeRobot v2.1、RealTeleop 或 UMI 默认输出；
- v2.4 合并完成前继续保持 `native_v25_eligible=false`。

## 3. 在线粗进度的精确定义

### 3.1 Profile

`TaskProgressProfileV1` 只定义在线需要的粗粒度任务进度：

- 有序 `ProgressStepDefinitionV1` 列表；
- 完整任务的默认 initial step；
- progress topology；
- 是否允许同一 step 失败后重试；
- 合法 full-task terminal steps；
- profile identity/hash。

每个 progress step 必须分开身份、语义类型与展示文案：

```text
progress_step_id   = pick_cup_02       # 时间线中的唯一实例
semantic_type      = PICK_OBJECT       # 可以在多个 step 重复
display_name       = 拿起第二个杯子
ordinal            = 3
```

在线状态、Attempt、Advance 和 annotation 都绑定 `progress_step_id`，不能使用
`display_name` 或 `semantic_type` 充当身份。Retry 在同一个 `progress_step_id` 上
创建新 Attempt；两个连续的 PICK_OBJECT 可以是两个不同且合法的 step。

Profile 支持两种 topology，默认仍是单一线性：

```text
topology_kind = LINEAR | ROOTED_TREE
```

`LINEAR`：

- step 按唯一 ordinal 严格递增；
- advance 只能进入下一个 ordinal；
- 默认只有一个后继和一个 terminal leaf；
- 不需要 UI 选择 target。

`ROOTED_TREE`：

- 每个 step 显式声明 `allowed_next_step_ids`；
- profile initial step 是唯一 root；
- 除 root 外每个 step 恰好有一个父节点，禁止分支汇合；
- 所有 step 必须从 root 可达，禁止环和孤立节点；
- child ordinal 必须大于 parent ordinal，保证每条 root-to-leaf 分支内单调；
- terminal steps 必须与 reachable leaves 完全一致，不能留下无后继且不可完成的
  dead-end；
- 一个后继时可以直接推进；多个后继时 request 必须显式指定
  `target_progress_step_id`。
- `ordinal` 在整个 Profile 内全局唯一，只用于稳定排序和
  parent-to-child 单调性验证；它不表示树中存在唯一执行顺序，sibling 也不能复用
  ordinal。

p2 不实现通用 DAG planner、分支合并或运行时回跳。树模式只是把 profile 中合法的
下一步从一个扩展为有限多个；进入某一分支后仍沿该分支单调前进。

它不包含 A1/A2/A3，不包含机器人、本体、GUI、动作空间或存储布局。

### 3.2 本次 Episode 的实际入口

Profile 的 initial step 只是完整任务默认入口。本次采集从哪里开始由
`ProgressEpisodeBindingV1` 决定：

```text
bound_clock_domain_id:
recording_scope:
  FULL_TASK | PROGRESS_STEP_WINDOW | TRANSITION_WINDOW | EVENT_WINDOW
entry_step_id:
entry_mode:
  PROFILE_INITIAL | ACTUAL_TRUNCATED_HISTORY | SCENARIO_SETUP | NO_PREHISTORY
history_complete:
assumed_completed_path_step_ids:
truncated_at_start:
```

当前支持级别不是四种 scope 等价：

- `FULL_TASK`：合同、reducer、resolver 和 termination fixture 已验证；
- `PROGRESS_STEP_WINDOW`：通用局部窗口结束与
  `TRUNCATED_AT_CAPTURE_END`/显式 step-complete 语义已验证；
- `TRANSITION_WINDOW`、`EVENT_WINDOW`：仅为前向兼容保留 enum。当前 binding 没有
  transition target、通用 event identity 或窗口边界合同，底层只能按普通
  non-full-task window 退化处理。唯一已接入的窄例外是 RealTeleop 恢复补采：
  它必须同时绑定精确 coarse step、fine stage 和
  `RECOVERY_FROM_FAILURE`，并且不得宣称完整 event/full-task 结果；其余 P4
  adapter 必须拒绝，直到专用语义经 fixture 验证。

约束：

- `FULL_TASK` 必须使用 `PROFILE_INITIAL`、从 profile initial step 开始、
  `history_complete=true` 且 `truncated_at_start=false`；否则 binding fail；
- 非 `FULL_TASK` scope 可以从任意合法 `entry_step_id` 开始，但这不等价于上述
  reserved scope 已具备完整任务语义；
- `assumed_completed_path_step_ids` 必须是从 profile root 到 `entry_step_id` 父节点
  的完整、有序、无空洞祖先路径；在线性模式下就是严格前缀闭包；树模式下是唯一
  root-to-parent path；
- 该 path 只表示本次绑定的明确前置假设，不伪造未采集的 Attempt；
- `history_complete=true` 且 entry 不是 root 时，必须提供完整 ancestor path；
- `history_complete=false` 可以不声明任何已完成 path；一旦声明，仍必须是完整祖先
  path，禁止 `[A, C]` 这类带空洞上下文；
- `SCENARIO_SETUP` 或 `NO_PREHISTORY` 可以从非 root step 进入且 path 为空，但不得
  推断任何 ancestor 已真实完成；
- `history_complete=false` 时，不得推断当前是该 step 在真实任务历史中的第一次
  Attempt；
- Attempt 只保存 `attempt_ordinal_in_capture`，不声明任务全历史绝对 ordinal；
- `truncated_at_start=true` 时，训练/评估 view 必须能区分 Episode 边界和真实任务
  起点；
- 局部窗口正常结束不等于 full-task success。

这使单独采一个节点、中段开始、人工 setup、failure/off-nominal 专项窗口和截断历史
都能被准确表达，而不修改可复用 Profile。

### 3.3 应用无关的进度请求

Control Station、脚踏板或其他前端可以继续向数采员显示“下一子任务”，但 GUI
operator intent、控件可用性和应用工作流由 `control_station` 拥有。进入
`collection_runtime` 后统一转换成 application-neutral request：

- `AdvanceProgressStepRequest`：进入合法的下一 progress step；分岔点必须给出
  `target_progress_step_id`；
- `UndoLastProgressAdvanceRequest`：撤销最近一次误触的 advance；
- `ReportAttemptFailureRequest`：当前 Attempt 失败。

以下是生命周期输入，不是额外的数采员语义按钮：

- `StartEpisodeProgress`：Episode owner 启动进度；
- `ResumeFailedAttempt`：现有 Hold/Resume 流程恢复后，在同一 coarse progress step
  创建新 Attempt；
- `CompleteEpisodeProgress`：Episode owner 正常结束 Episode；
- `AbortEpisodeProgress`：故障、碰撞或人工放弃等非正常终止。

因此 UI 或脚踏板仍然只需提供“子任务管理”和“失败”两个能力；canonical contract、
代码类型和日志统一使用 `progress_step`，不再引入新的 Subtask p2 类型。

在线的 `UndoLastProgressAdvanceRequest` 不是通用历史矫正。它只撤销当前
`progress_sequence_index` 上最后一个、仍然最新鲜的 Advance，并且必须发生在任何
后续 progress fact 之前。一次 Undo 原子恢复原 Attempt、撤销其隐式 success，并移除
刚创建的新 Attempt；因此不存在需要解释 B2、B->C、C1 的 suffix。数采员如果已经
提交 Failure、Retry、另一次 Advance 或 termination，就不能再通过在线 Undo 改写
历史，只能在 Episode seal 后进入后期矫正层。

### 3.4 统一事实记录，但不扩张领域实体

在线 progress log 中的 accepted event、effect receipt 和 termination 使用一个共同
的 `ProgressFactEnvelopeV1`。它只是统一 identity、因果时间和 append 顺序的存储
envelope，不是新的任务语义实体：

```text
fact_id
episode_capture_identity_hash
progress_sequence_index
effective_at + effective_clock_domain_id
available_at + available_clock_domain_id
origin + actor_id
payload_schema_id + payload_schema_hash
payload
source_fact_ids
```

约束：

- `progress_sequence_index` 由唯一 progress append owner 从 1 连续递增；
- `ProgressEpisodeBindingV1.bound_clock_domain_id` 绑定本 Episode 的采样时钟；
  capture progress 的所有 `effective_at`、正式 capture Pause/Resume 边界和 seal
  必须使用该时钟；`available_at` 可使用独立接收时钟；
- effective/availability time 沿用 native 数据已有的因果时间原则，不另建简化时间
  模型，也不增加另一套 `ticks_ns` 字段；
- payload 是 P1-A 明确列举的严格 tagged union，不接受任意字典；
- payload-specific identity，例如 Attempt、Failure、annotation set，仍放在对应 payload；
- correction 对被修正事实的引用也放在 typed correction payload，不把
  `supersedes_fact_ids` 冻结成所有 capture facts 都必须携带的通用关系图；
- schema definition、hash 和 timestamp value type 尽量复用现有
  `dataset_contracts`/v2.5-p 基础合同，不复制第二套实现。

P1-A 冻结的是这组最小公共语义和 tagged-union 边界，不提前冻结物理 Parquet 列、
派生索引列或未来所有 provenance 字段。Refinement/correction annotation 由每个
annotation set 的独立 append owner 排序，不能竞争 `progress_sequence_index`；
P1-A 只冻结 typed annotation payload 与 sealed-manifest binding，annotation record
的物理 envelope 等 P3 store fixture 后再决定是否复用同一 header 实现。

### 3.5 Advance 是一个复合 canonical fact

一次 `A -> B` advance 在语义上同时完成，并显式绑定 from/to
`progress_step_id`：

1. 将 A 的当前 Attempt 结束为 success；
2. 记录切换时间；
3. 将当前 coarse progress step 更新为 B；
4. 创建 B 的新 Attempt。

持久化层只写一条 `ProgressAdvanceEventV1`，而不是四条需要跨记录事务组合的
Attempt begin/finish/transition facts：

```text
from_progress_step_id
from_attempt_id
from_attempt_outcome = SUCCESS
to_progress_step_id
to_attempt_id
```

外层 `ProgressFactEnvelopeV1.fact_id` 就是该 transition 的唯一身份，不再重复增加
第二个 transition identity。Attempt timeline 是 reducer/resolver 对这条复合事实的
物化结果。Failure 同样由一条 `AttemptFailureEventV1` 完整关闭当前 Attempt，不另写
一个可与它分叉的 Attempt-finish 真源。

## 4. 误触撤销为什么使用追加事件

物理删除最后一个时间戳虽然直观，但会破坏已 flush/chunk 的不可变数据，也会在
采集和标注并行时制造重写和竞态。p2 使用：

```text
Advance A -> B @ t1, fact_id=17
Undo advance_fact_id=17 @ t2
Advance A -> B @ t3, fact_id=19
```

规范 resolver 的结果是：

- transition 17 不再属于有效进度时间线；
- A 的原 Attempt 没有在 t1 成功结束，而是连续保持到 t3；
- transition 17 创建的 B Attempt 被整体撤销；
- transition 19 才结束 A 并创建新的 B Attempt；
- 原始日志仍保留误触和撤销，便于审计；
- retrospective/training view 中 t1 到 t2 仍属于 A；
- 如未来需要严格在线因果分析，causal view 可以重建系统在 t1 到 t2 当时认为
  自己处于 B 的事实。

撤销的 fail-closed 条件：

1. 只能指向当前规范时间线中最后一个有效 advance；
2. 必须引用精确 `advance_fact_id`，不能用“最近时间戳”猜测；
3. 该 advance 之后不能已有另一个 advance、failure、Episode 终止或已封存
   annotation dependency；
4. undo、被撤销 advance 以及事务内 Attempt 变化由同一个 reducer 验证；
5. 后续事件必须从 resolver 返回的有效状态生成，不能从 raw log 最后一行生成。

这保证追加式撤销不会产生悬空 Attempt、重复 success 或歧义边界。

## 5. Failure、Hold、Retry 与训练无效区间

### 5.1 在线行为

Failure 到 Retry 使用无空档状态机：

```text
ACTIVE_ATTEMPT
  -> AWAITING_HOLD_RECEIPT
  -> PAUSED_NO_ACTIVE_ATTEMPT
  -> RESUME_RECEIVED
  -> ACTIVE_RETRY_ATTEMPT
```

Failure 的 accepted fact 与 Attempt 关闭是同一个 reducer transition，因此实现中不保留
一个可单独停留的 `FAILURE_REPORTED` phase。`AWAITING_HOLD_RECEIPT` 已经完整表达
“Failure 已持久化、Attempt 已关闭、外部 Hold/Pause receipt 尚未确认”。

`ReportAttemptFailureRequest` 生效时立即：

1. 在当前 `progress_step_id` 的 active Attempt 上记录 failure；
2. 将该 Attempt 结束为 failure；
3. 清空 `active_attempt_id`；
4. 返回 typed `ProgressEffectRequest`，其中
   `requested_operations=(APPLICATION_HOLD, FORMAL_CAPTURE_PAUSE)`；
5. 进入 `AWAITING_HOLD_RECEIPT`。

该状态机还必须保证：A Attempt failure 后，Resume 只能在 A 上创建新的 Retry
Attempt。`ResumeFailedAttemptRequest` 不接受目标 step 参数；在 Retry Attempt
成功并产生显式 `A -> B` Advance 前，任何进入 B 的请求都 fail closed。在线采集
不允许出现“A failure 后没有任何进度更新却直接开始 B”的时间线。

正常的“直接报告 Failure”路径中，Failure fact 必须先持久化，并携带需要执行的
Hold/Pause operations：

```text
requested_operations = APPLICATION_HOLD, FORMAL_CAPTURE_PAUSE
# 幂等键直接使用 containing ProgressFactEnvelopeV1.fact_id，不增加字段
```

这里不增加第二套独立 outbox 服务，也不增加与 Failure 一一对应的第二个 request
identity：durable Failure fact 本身就是 outbox，`fact_id` 同时作为应用执行和 receipt
去重的幂等键。只有 Failure fact 持久化成功后才能把 operations 发布给
`control_station`/应用 effect port。重启后可以重复发布同一 Failure fact；应用必须
幂等执行，重复 receipt 只能被 reducer 确认一次。这样既覆盖至少一次传递，也不会
出现“Attempt 已失败但 Hold 请求从未落盘”的空档。

collection semantics 本身不控制机器人、不执行 Hold，也不拥有安全状态机。
`control_station` 或非 GUI application coordinator 只负责把 effect 委托给现有
机器人/采集 owner，并汇总其 typed result，返回 `HoldPauseEffectReceiptV1`：

```text
source_failure_fact_id
application_hold:
  requested_at
  effective_at
  status = APPLIED | REJECTED | TIMEOUT
formal_capture_pause:
  requested_at
  effective_at
  status = APPLIED | REJECTED | TIMEOUT
```

只有 `application_hold.status == APPLIED` 且
`formal_capture_pause.status == APPLIED` 才能进入
`PAUSED_NO_ACTIVE_ATTEMPT`。聚合状态可以由 UI/view 派生，不持久化一份可能和两个
分项漂移的 `overall_status`。其他状态不得静默 retry；应用必须继续等待、异常终止
Episode，或将 progress capability 标为 incomplete。

`APPLICATION_HOLD` 是应用生命周期效果，不等同于某种机器人命令。RealTeleop 只有
在下层机器人 Hold 已得到确认后才能报告 `APPLIED`；UMI、仿真或无机器人应用则由
各自 application owner 按同一效果语义确认。progress core 不知道具体本体。

现阶段结合项目实际，Hold 会完整暂停正式 raw/formal capture。p2 不设计 Hold 期间
继续采 raw、不区分 training/raw 两种 pause，也不在首批合同增加
`CaptureGapIntervalV1`。暂停边界由 failure、已应用的 hold receipt 和 retry 明确
限定；如果未来 Hold 行为改变，再以新 capability/schema 增加更细的 capture-gap
合同。

数据质量 resolver 可以从
`formal_capture_pause.effective_at -> formal_capture_resume_effective_at` 派生
`INTENTIONAL_HOLD_PAUSE` 区间，用于和 `SENSOR_GAP`、`WRITER_FAILURE`、
`FRAME_DROP` 区分。该区间不是新的 canonical capture fact，也不表示 Hold 期间继续
采集 raw 数据。

另有一个受约束的在线提升路径：数采员可能因紧急情况先请求普通 Hold，在同一个未恢复
Hold 内才判断当前 Attempt 应记为 Failure。此时不改写或删除原 Hold，也不重新执行
Hold/Pause；应用 coordinator 只允许复用同一 Episode、同一 active Attempt、且确由
该人工 Hold 新建的精确正式暂停边界。自动安全 Hold、Hold 前已经存在的模态暂停、
恢复后的旧 Hold 和跨 Episode Hold 均 fail closed。

提升后的 capture fact 使用：

```text
effective_at = original formal_capture_pause_effective_at
available_at = later operator classification time
reported_at = effective_at
timing_semantics = PROMOTED_HOLD_TIME
```

其 `HoldPauseEffectReceiptV1` 绑定原 `hold_requested_at`、原
`application_hold_effective_at` 和原 `formal_capture_pause_effective_at`。因此它是
“已有效果的追加式语义分类”，不是违反普通 outbox 顺序的第二条 effect 路径。Hold
期间只开放这一个 Failure 提升 intent；Advance/Undo 均禁用。若精确原始 receipt
上下文已经丢失，保持 Hold/Pause 且不猜测性补写 Failure。

人工调整并通过现有 Resume 后，`AttemptRetryEventV1` 必须显式引用：

```text
source_failure_fact_id
hold_pause_receipt_fact_id
resume_receipt_fact_id
new_attempt_id
```

`resume_receipt_fact_id` 指向由应用的权威 Hold/Resume 状态转换生成的
`ResumeAfterFailureReceiptV1`：

```text
source_failure_fact_id
hold_pause_receipt_fact_id
resume_operation_id
application_resume_effective_at
formal_capture_resume_effective_at
application_resume_status = APPLIED | REJECTED | TIMEOUT
formal_capture_resume_status = APPLIED | REJECTED | TIMEOUT
```

collection runtime 只验证其 failure/episode/hold-receipt 绑定与两个分项的 `APPLIED`
状态，不自行判断应用是否已经恢复。新 Attempt 的 `effective_at` 使用
`formal_capture_resume_effective_at` 或其后的第一个可用采样时刻，不能使用操作者
按下 Resume 的时间。

按架构所有权，Resume operator intent 和 application operation identity 应由
`control_station`/应用 lifecycle owner 持有，具体应用恢复仍由 lower runtime owner
执行；当前 application-neutral p2 core 只消费 typed receipt，P4 的 Control Station
adapter 尚未实现。因此 p2 不复制第二个 Resume outbox。receipt 中的稳定
`resume_operation_id` 是跨端口幂等身份。如果未来某个非 Station frontend 把 Resume
effect 委托给 collection controller，也必须通过同一 port 语义提供稳定 operation
identity，而不能另建不兼容协议。

不能依据“当前 step 最近失败过”猜测来源。Retry 在同一 `progress_step_id` 创建新
Attempt，Episode 保持打开。

因此可以得到：

```text
A / attempt_1 = failure
A / attempt_2 = success
```

Hold 期间不存在 active Attempt，也不写正式采集帧。调整动作若未来需要作为
recovery 数据，应通过独立显式采集模式记录，不能被当前默认行为悄悄混入成功轨迹。

### 5.2 Failure 时间语义

在线数采员通常只能可靠提供报告时间，不能声称知道物理 failure onset。因此普通
capture-truth `AttemptFailureEventV1` 使用：

```text
reported_at
effective_at
timing_semantics = OPERATOR_REPORT_TIME
reported_at == effective_at
```

当且仅当同一未恢复的普通 Hold 被在线提升为 Failure 时，还允许
`timing_semantics=PROMOTED_HOLD_TIME`：`effective_at/reported_at` 是已验证的原
formal Pause 边界，fact `available_at` 是数采员稍后按下 Failure 的时间。它表达
“在该暂停边界结束失败 Attempt”，不声称这是不可观测的物理故障 onset。

`ESTIMATED_ONSET` 和 `POSTHOC_CORRECTED_ONSET` 不进入在线 Failure 合同；它们只属于
后标注/correction overlay。Hold 真正生效的时间只来自 receipt。后标注可以增加
onset/outcome correction，但不能改写原始 failure event。

failure 生效后到 Retry begin 之间没有 active Attempt；Pause 生效前可能产生的控制
状态也不能归入 retry Attempt。由于当前 Hold 完整暂停采集，p2 首批不为这段创建
额外 raw stream。

### 5.3 训练排除区间不是采集时写死的布尔列

Failure 是 canonical fact；训练排除区间是可重建 view。`Tf` 是所选 resolver policy
下的有效 failure 时间：未选择 correction 时使用在线 `effective_at`，选择合法的
onset correction 时使用修正时间。对于 `Tf`：

- 如果没有精细标注，起点是当前 coarse Attempt 的开始时间 `Tc`；
- 如果存在所选 refinement profile 下、通过验证的 salvage anchor，起点是 `Tf`
  之前最近的合格 anchor `Tr`；
- 排除区间是从该起点到 failure cut；
- 后期增加 A3 边界后，排除区间可以从 `[Tc, Tf]` 收缩为 `[TA3, Tf]`；
- A1/A2 不需要重写，即可重新进入训练选择。

映射到样本时使用半开区间：

```text
[first_sample_at_or_after(segment_start),
 first_sample_strictly_after(Tf))
```

这样不会依赖浮点时间恰好相等，也能明确包含 failure 前最后一个采样帧。

当前只读 sample locator 已固定映射算法、但没有增加持久化 schema：

- causal sample 必须从 `sample_index=0` 连续编号，cut time 非递减，并与
  `ProgressEpisodeBindingV1.bound_clock_domain_id` 使用同一 Episode 时钟；
- Advance 关闭的 Attempt 使用 `[start, advance)`，避免 transition cut 同时属于
  前后两个 Attempt；
- Failure 或 termination 关闭的 Attempt 使用 `[start, end]`，因此 failure cut
  对应的同时间样本被包含；
- intentional formal-capture Pause 使用 `[pause, resume)`；
- 冻结的 p1 artifact 即使仍可被 p1 reader 读取，只要 causal cut 使用外部时钟、
  index 不连续或时间倒退，就不能进入 p2 progress-aware 投影。

这只是 `dataset_interface` 的 fail-closed derived view，不改变 p1 native writer、
manifest、hash 或兼容 reader。

当前 coarse-only resolver 已先验证最保守基线：每个在线失败 Attempt 从 Attempt
起点排除到 operator-report Failure cut；Retry、成功 Attempt 和 intentional Pause
不会被隐式合并。没有 causal sample 落入该时间段时仍保留空 exclusion 供审计，
而不是删除 Failure 事实。它只接受 sealed、terminated 的 capture-time timeline；
retrospective correction 必须由后续带独立 resolution identity 的 resolver 处理，
不能把修改后的时间线冒充原 capture view。当前 provisional Python view 使用非持久化
`resolution_kind` 做 fail-closed gate；它不是新的 dataset 字段，最终 correction
identity 仍等后期 resolver 原型稳定后再冻结。

未来若发布固定训练视图，其小型 resolved-view manifest 应绑定派生排除 interval
artifact、来源 failure fact、resolved failure cut、实际应用的 correction facts、
所选 annotation set/profile 和 resolver policy identity。它不是 raw truth，可以
低成本重建。当前不冻结 `TrainingExclusionIndexV1` 或独立 exclusion receipt；具体
artifact、列和 hash domain 等真实采集、训练与导出消费者共同验证后再确定。

一个 refinement boundary 只有同时满足以下条件才能成为 salvage anchor：

1. 属于同一个 coarse Attempt 且位于 failure 之前；
2. Profile 声明 `eligible_as_exclusion_anchor=true`；
3. Profile 声明它能够 `certifies_valid_prefix=true`；
4. 从 Attempt 开始到该 anchor 的 required predecessor coverage 连续完整；
5. 没有 unresolved conflict 或 invalid correction；
6. confidence 达到所选 training resolver policy 的阈值；
7. boundary semantics 能证明此前前缀有效，而不只是视觉分段。

如果没有合格 anchor，排除起点保持 coarse Attempt start。系统宁可暂时多排除，也
不能因为一个孤立 A3 标签错误释放失败相关动作。

当前原型已经覆盖上述收缩 gate：只有 candidate 及其全部
`required_predecessor_boundary_ids` 在同一 coarse Attempt 中各有一个无冲突观察、
时间顺序合法、使用 Episode clock、位于 Failure cut 之前且均达到显式 confidence
阈值时才选择最新合格 anchor。任何 `UNANNOTATED`、`INCOMPLETE`、`CONFLICTED`、
`INVALID`、`BELOW_CONFIDENCE` 或 `NO_VALID_ANCHOR` 结果都继续使用 coarse Attempt
起点，不释放数据。这些状态目前只是 derived Python view，不是冻结的 receipt enum。

未来 reward/penalty 不应复用或改写这个布尔排除合同；它们可以使用同一套 interval
reference 另建训练信号 overlay。

### 5.4 正常结束的 fail-closed 条件

`CompleteEpisodeProgress` 至少要求：

- 当前存在 active Attempt；
- 没有未撤销 failure；
- 没有 pending Hold/Pause effect；
- 没有未确认 Resume；
- 没有未完成 advance transaction；
- 对 `FULL_TASK`，当前 step 必须是 profile 允许的 terminal step。

`EpisodeProgressTerminationV1` 分开记录：

```text
termination_cause =
  FULL_TASK_COMPLETED | WINDOW_COMPLETED | MANUAL_STOP |
  SAFETY_STOP | ABORTED | UNKNOWN
active_attempt_disposition =
  SUCCESS | FAILURE | ABORTED | TRUNCATED_AT_CAPTURE_END |
  UNKNOWN | NO_ACTIVE_ATTEMPT
full_task_outcome = SUCCESS | FAILURE | UNKNOWN | NOT_APPLICABLE
```

`FULL_TASK` 从合法 terminal step 正常完成，才能隐式产生 current Attempt success 和
full-task success。当前已验证的 `PROGRESS_STEP_WINDOW` 正常结束时，
`termination_cause` 可以是 `WINDOW_COMPLETED`，但默认
`active_attempt_disposition=TRUNCATED_AT_CAPTURE_END` 且
`full_task_outcome=NOT_APPLICABLE`。只有窗口显式覆盖并验证了完整 progress step，
才能把 active Attempt 写成 `SUCCESS`。`WINDOW_COMPLETED` 永远不是 Attempt outcome。
`TRANSITION_WINDOW`、通用 `EVENT_WINDOW` 在专用 binding 冻结前只能沿用该保守
退化结果，不能据此推断 transition/event 已被完整采集。恢复补采的精确 stage entry
binding 只证明入口语义，不证明窗口已完整。

### 5.5 Abort、Crash 与 Seal 的闭环

Episode 可以在任何 reducer 非终态终止。首批不为每个 pending effect 再增加一组
`CANCELLED_*` receipt enum，而由一条 `EpisodeProgressTerminationV1` 显式引用并关闭
本 Episode 内尚未确认的 effect operation/outbox：

| reducer state | termination 后 Attempt disposition | progress seal |
| --- | --- | --- |
| `ACTIVE_ATTEMPT` | `ABORTED` 或 `TRUNCATED_AT_CAPTURE_END` | 可 COMPLETE |
| `AWAITING_HOLD_RECEIPT` | 无 active Attempt；termination 引用 unresolved request | 可 COMPLETE |
| `PAUSED_NO_ACTIVE_ATTEMPT` | `NO_ACTIVE_ATTEMPT` | 可 COMPLETE |
| `RESUME_RECEIVED` 但尚无首帧 | retry 尚未开始，`NO_ACTIVE_ATTEMPT` | 可 COMPLETE |
| 存在半写入 append/batch | 不生成完整 progress manifest | 不可 COMPLETE |

`EpisodeProgressTerminationV1` 对 pending operation 只记录 typed unresolved refs，
例如 `source_failure_fact_id` 或 `resume_operation_id`；它说明 receipt 缺失已被终止
事实完整解释，并不伪造外部 effect 已成功、被取消或从未发生。于是：

- 没有 termination 解释的 pending state 仍是 `INCOMPLETE`；
- 被有效 termination 完整引用、且所有已写 facts 可验证时可以 `COMPLETE`；
- `COMPLETE` 只表示 progress 事实闭合可解析，不表示任务成功或外部副作用成功；
- crash 留下半条/半批物理记录时仍禁止 manifest seal，不能靠 termination 猜测修复。

## 6. 后期精细标注

### 6.1 为什么需要独立 refinement profile

在线 coarse progress 与后期 A1/A2/A3 有不同生命周期：

- coarse profile 在采集前必须冻结；
- refinement profile 可以后续增加或升级；
- 新的 refinement 不能改变当时在线 coarse 状态；
- 同一 raw Episode 可以按不同 refinement profile 生成不同训练 view。

因此使用两个必要而非冗余的 profile：

- `TaskProgressProfileV1`：在线粗进度；
- `TaskRefinementProfileV1`：后期细分层级、允许边界和覆盖要求。

Refinement profile 必须绑定兼容的 coarse profile hash。

每个 refinement boundary definition 至少声明：

```text
progress_step_id
refinement_boundary_id
parent_refinement_boundary_id
ordinal
semantics = ENTER_STAGE | COMPLETE_STAGE
eligible_as_exclusion_anchor
certifies_valid_prefix
required_predecessor_boundary_ids
```

`refinement_boundary_id` 在 refinement profile 内唯一；`parent_refinement_boundary_id`
表达细粒度层级，显示路径由 profile 派生。身份、hash 和 resolver 不解析 `A/A3` 这类
展示字符串。

普通分段边界默认不能作为训练排除收缩 anchor。只有 Profile 明确声明、且实际
annotation coverage 满足前置条件时，resolver 才能使用它释放前缀数据。

### 6.2 标注事实

已落地的最小 `RefinementBoundaryAnnotationV1` 只表达：

- annotation identity；
- coarse Attempt identity；
- `refinement_boundary_id`；
- Episode clock 上的 effective timestamp；
- confidence。

Episode/progress-manifest、refinement-profile、annotation-set identity 和 bound clock
统一保存在 `RefinementAnnotationSetBindingV1`，不在每条 boundary 中重复。sample
locator 由 `EpisodeSampleAxis` 按时间确定性派生，不作为第二个时间真源写入。
annotator/origin、supersedes/source references 和 correction graph 暂不冻结；在其
resolver 与选择 policy 稳定前，需要修订时创建新的 immutable annotation set，由上层
显式选择，不在原 set 中猜测性覆写。

标注只追加到独立 annotation set，不修改 raw frames、videos、native streams 或
capture progress facts。一个正在编辑的 annotation set 是 append-only draft；完成后
封存 manifest。修改通过新的事实/supersession 完成，而不是原地覆写旧 chunk。

当前 `RefinementAnnotationSetBindingV1` 必须引用已经 sealed 的
`source_progress_manifest_hash`。不允许 annotation 绑定仍在采集的 active progress
log。这里“采集与标注并行”指采集新 Episode 的同时标注已封存旧 Episode，而不是
两个 writer 同时修改同一个 Episode。未来若确实需要 active-Episode 近实时标注，
再以独立 capability 增加 watermark/sequence 合同。

当前物理集合由 `binding.cjson + chunks/chunk-*.cjson + manifest.cjson` 构成。每次
append 批次被限制为小型 immutable chunk；重启只恢复已经原子发布的连续 chunk，
manifest 精确绑定 ordered annotation/chunk hashes。离线 reader 严格要求 seal、
progress manifest、refinement profile 和 clock 全部匹配，随后重建
`SelectedRefinementAnnotationSet`。兼容导出名 `RefinementBoundaryObservation`
直接指向同一持久化 boundary 合同，不维护第二套字段真源。

### 6.3 混合覆盖率

覆盖率以 `(episode, coarse_attempt, refinement_profile_hash)` 为基本单位：

- `not_applicable`：该 coarse step 在所选 refinement profile 中没有边界定义；
- `unannotated`：没有可用精细边界；
- `partial`：有部分边界，但未满足 profile 完整性；
- `complete`：满足 profile 要求且时间线通过验证；
- `conflicted`：多个事实无法按选定 policy 唯一解析；
- `invalid`：越界、非法顺序或引用错误。

同一数据集可以同时包含所有状态。缺失标注保持缺失，不转换成默认 A1 或 success。
当前派生实现要求同一 Attempt 的每个 profile boundary 恰好有一条合法 annotation；
重复边界为 `conflicted`，错时钟、越界、错 step 或非法时间顺序为 `invalid`。低
confidence annotation 仍计入结构 coverage，因为 confidence 是训练选择而不是
“是否完成标注”的维度。

### 6.4 后期矫正是 sealed 时间轴 overlay，不是在线 Undo 的延伸

在线 Undo 只处理数采员刚刚误按“下一阶段”的即时回退。后期矫正面对的是已经 seal
的完整 Episode，标注员可以结合视频、动作、状态和任务上下文，对时间轴进行充分
调整，包括：

- 重定 progress step 的进入/退出边界；
- 修正 Attempt 所属 `progress_step_id`；
- 合并、拆分或作废错误 Attempt 区间；
- 修正 Advance 的时间与目标；
- 设置或修正 success/failure/unknown outcome；
- 修正 Failure cut/onset；
- 在 profile 允许范围内重建完整 retrospective progress timeline。

后期矫正遵循以下边界：

- correction/annotation set 必须绑定 sealed progress manifest；
- capture-time facts、raw/native stream 和原 manifest 永远不改写；
- causal view 永远保留数采时的在线认知；
- retrospective/training view 只有显式选择 annotation set 与 resolver policy 后才
  使用修正时间轴；
- 修正后的 Attempt 区间必须无非法重叠，边界必须落在 Episode 时钟上，stage
  sequence 必须通过所选 progress profile；
- partial correction 可以保持 partial/unresolved，不得被静默当成完整训练时间轴；
- 多个矫正产生不唯一解释时标为 `conflicted`，不能按“最新一条”覆盖；
- correction 会使 coverage/exclusion view 失效并重建，但不重写高带宽数据。

因此后期矫正不需要把 suffix 简化为“全部作废”或“自动逐段猜测”二选一。标注员
可以直接给出所需范围内的修正时间轴；未被充分解释的范围保持未解析/不可用于默认
progress-aware training。具体持久化形式应先通过 resolver 原型比较“边界事实 overlay”
和“replacement segment overlay”，再冻结最小 schema，避免提前堆积
`INVALIDATE_*`、`RETIME_*` 等大量过程型操作实体。

当前 `dataset_interface.progress_corrections` 中的线性 terminal-Failure reassignment
只是一项窄版 immutable-overlay 验证：证明原 capture bundle 不变时可以重建不同的
retrospective timeline。它不是最终后期矫正合同，其“存在 Retry/suffix 时 fail
closed”也不是最终能力限制。后续 prototype 应以完整时间轴 overlay 为目标，而不是
继续扩张这一 convenience operation。

### 6.5 已确认的协议与实现边界

以下边界先记录，但未实现部分延后到真实采集数据可以驱动设计时再继续：

- correction 的 immutable fact/schema 未来属于 `dataset_contracts`；correction 的
  应用、冲突处理和 retrospective timeline 重建属于 `dataset_interface`；
- annotation set 的 binding/chunk/manifest 属于数据协议；多套 annotation set 的选择、
  合并和冲突 policy 属于 `dataset_interface`，不进入 capture 协议；
- 某次数据集发布实际选中的 annotation/correction/profile/policy hashes 可以记录在
  薄的 resolved-view manifest 中，但选择算法本身不持久化为协议；
- coverage 是可重建结构视图，继续使用 Python view 与可删除 SQLite cache；当前不单独
  冻结 canonical coverage receipt；
- exclusion/eligibility 是训练 policy 派生结果，不成为 Episode 基础事实，也不写成
  每帧永久布尔列；
- 当真实训练和导出至少形成两个稳定消费者后，优先评估一个统一、很薄的
  resolved-view manifest，而不是分别新增大型 coverage/exclusion receipt；
- `collection_runtime.progress` 只负责在线 request/reducer/controller/store 与通用
  annotation persistence，不拥有 correction policy、annotation selection、coverage
  resolver 或 training exclusion。

## 7. “原地标注”与低 IO 的实现

这里的“原地”定义为：针对同一 Episode 增加 overlay，并立即更新可查询视图，而
不是改写 Episode 主数据。

当前已经实现的物理结构由调用方提供根目录，不冻结仓库级或数据集级落点：

```text
<caller-supplied progress_dir>/
  profile.cjson
  binding.cjson
  facts.cjsonl
  manifest.cjson

<caller-supplied annotation_dir>/
  binding.cjson
  chunks/chunk-*.cjson
  manifest.cjson

<caller-supplied derived-cache path>/
  semantic_catalog.sqlite
```

Progress 事件量很低，当前使用一条 append-only `facts.cjsonl`，不是 progress
chunk。Annotation 才使用 bounded immutable chunk。未来接入既有 Episode owner 时，
可以在不改变 logical fact/manifest hash 的前提下定义 `meta/...` 落点或物理 batching；
在 P4 前不得把示例路径误当成 canonical dataset topology。

未来 Episode semantic bundle 需要独立声明 progress capability seal 状态；该
capability contract 当前尚未落地：

```text
progress_v2:
  status = ABSENT | COMPLETE | INCOMPLETE | FAILED
  manifest_hash
  failure_reason
```

- 规划规则是：未启用 p2 为 `ABSENT`，不是失败；
- 完整校验并 seal 是 `COMPLETE`；
- open transaction、未被 termination 显式解释的 pending Hold/Resume 或缺 receipt 是
  `INCOMPLETE`；合法 termination 已引用的 unresolved effect 不阻止语义闭合；
- writer/hash/contract 验证失败是 `FAILED`；
- raw/native Episode 可以在 progress 不完整时仍保持自身有效；
- 只有 `COMPLETE` 默认进入 progress-aware training view。

原则：

- capture 与 annotation 写不同目录和文件，不争用 raw writer；
- annotation 只绑定 sealed progress manifest；
- 每个 annotation set 只有一个 append owner；不同标注任务使用不同 set；
- capture progress 追加到低事件量 fact log；annotation 按 bounded chunk 写入，
  两者都避免一个事件一个文件；
- 标注只产生小型边界事实，不重写高带宽数据；
- coverage 和 exclusion 是小型派生表，可增量或整体重建；
- `semantic_catalog.sqlite` 是可选、可删除、可重建的查询缓存，不属于 canonical
  dataset identity；
- SQLite 使用 WAL 支持采集期间的读写查询；catalog 更新失败不影响 canonical
  annotation commit，可稍后重建；
- 可移植的 canonical 来源始终是 profile、progress fact log、annotation chunks 和
  manifests，不要求训练端信任 SQLite。

典型查询：

```text
未开始标注的 Attempt
某 refinement profile 下 partial 的 Attempt
包含 failure 但尚未细分到 failure 所在阶段的 Attempt
已 complete 且无 unresolved conflict 的 Episode
某 resolver/profile 下可训练的 sample intervals
```

## 8. 顶层包边界

### 8.1 不建议新建 `long_horizon/`

`long_horizon` 是使用场景，不是稳定所有权。它容易逐渐混入：

- 数据合同；
- GUI；
- writer；
- 任务执行器；
- 后标注工具；
- 模型训练逻辑。

p2 的核心能力同样适用于短任务、多阶段质检、仿真和非遥操数据，因此不应以
long-horizon 命名顶层生产包。

### 8.2 使用已经拆出的 `collection_runtime/`

当前仓库已经把 bounded async writer、native/v2.5-p persistence、generic producer、
episode maintenance 和 layout mechanics 迁入 `collection_runtime`，并已有 package
discovery、README、兼容 shim 和 boundary tests。p2 不再新建顶层包，只在现有边界
内增加一个小型 progress 子域：

```text
collection_runtime/
  progress/
    annotation_store.py       # frozen sparse refinement-boundary chunks/seal
    stage_annotation_store.py # ordered stage/abnormal drafts and immutable seal
    requests.py               # application-neutral progress requests
    reducer.py                # canonical progress state reducer
    controller.py             # validate request -> event/effect
    store.py                  # capture progress append/seal; subordinate to episode owner
```

accepted fact/effect/receipt 的持久化 payload 继续由 `dataset_contracts` 定义；
`collection_runtime.progress` 不另建一份 `events.py` schema。若实现阶段发现 runtime
内部确实需要非持久化 wrapper，优先放在现有 controller/request 边界；只有形成独立
稳定职责并被两个以上应用证明复用时才增加新模块。目前不存在 `ports.py`，也不应
为了目录对称提前创建。

严格边界：

- 继续遵守现有 `AGENTS.md`：可以依赖 `dataset_contracts`、`dataset_interface` 和纯
  domain contracts；
- 不依赖 `teleop_runtime`、`control_station`、`hardware`、`sim` 或具体本体；
- 不拥有 operator intent；应用 adapter 把按钮/脚踏板 intent 转成 progress request；
- 不产生机器人命令，不拥有 safety/authority/dispatch；
- 初期不夺取现有 Episode owner，只接收其 identity 和 clock；
- 后续只有在 RealTeleop、UMI 和至少一个非遥操 fixture 都通过同一 port 后，才考虑
  将通用 Episode ownership 迁入；
- 不增加新的生产 entrypoint。

顶层 package discovery 和架构边界已经落地。实现 p2 时只补充 progress ownership
说明、子包导出和针对新依赖方向的 boundary tests；没有依赖变化时不修改
`uv.lock`。

### 8.3 其他包的职责

`dataset_contracts/`：

- progress/refinement profile；
- per-Episode recording scope、actual entry 和 history completeness binding；
- event/receipt/annotation payload 的持久化合同；
- Episode/Attempt/transition/annotation identity；
- Hold/Pause receipt 与已落地 manifest schema；
- 未来 immutable correction fact、progress capability 和薄的 resolved-view manifest
  schema 的所有权仍在这里，但真实数据/消费者验证前不视为已实现或已冻结；
- 纯验证和 hash；
- 不包含 reducer、writer、IO 或 GUI。

`dataset_interface/`：

- raw event + undo 的规范 timeline resolver；
- 已落地窄版 terminal-Failure reassignment correction 原型和 validated
  salvage-anchor resolver；
- general coarse/outcome correction、annotation-set selection 和 p1 -> p2 migration
  adapter 的所有权仍在这里，但当前尚未完整实现；
- refinement annotation sealed reader/verification boundary；
- coverage builder；
- training exclusion/index builder；
- frame/interval/window projection与查询；
- rebuildable SQLite catalog adapter；
- 不拥有在线 current-progress-step 状态或 Episode lifecycle。

`control_station/`：

- 已是 RealTeleop/UMI 的 GUI-facing application control plane；拥有按钮、脚踏板等
  operator intents、控件可用性、输入优先级和应用 workflow；
- 未来 P4 adapter 将把 operator intent 翻译成 collection progress request，并协调
  controller 返回的 hold/pause effect 与 lower-owned operation receipt；该 adapter
  当前尚未实现；
- 不直接操作设备、机器人或 dataset writer，也不复制 progress reducer。
- `control_station.web`、`control_station.real_teleop` 和 `control_station.umi` 是
  canonical GUI owner；`teleop_runtime.gui` 的现存路径仅作兼容 re-export/alias，
  不得重新加入实现。

`teleop_runtime/`、UMI frontend、sim application：

- 提供设备/runtime status、机器人 Hold/Resume、Episode lifecycle 等 typed ports；
- 把现有 Episode identity、sample clock、resume/stop receipt 传入 ports；
- 不拥有 GUI policy，不复制 progress reducer，不直接构造持久化 event。

## 9. 数据流

```text
operator input / application lifecycle
              ↓ control_station or non-GUI application adapter
collection_runtime progress request
              ↓ controller + reducer
accepted event + requested effect
        ↙                     ↘
progress store          application hold/pause
        ↓                     ↓ receipt
v2.5-p2 facts       collection_runtime next state
        ↓
dataset_interface resolver
        ↓
coarse timeline + refinement overlay
        ↓
coverage index + training exclusion intervals
```

每个逻辑资源仍只有一个 owner：

- Episode identity/commit：现有 Episode owner，迁移前保持不变；
- 在线 progress state：`collection_runtime` reducer；
- capture fact append sequence：progress store；
- 单个 annotation set append sequence：该 annotation writer；
- resolved view：`dataset_interface` 按显式 policy 生成；
- Hold/safety：lower runtime 的现有控制/安全 owner；`control_station` 只协调 typed
  operation/receipt。

## 10. p1 到 p2 的迁移

迁移不搬动 raw sample：

- p1 successful attempt finish + next begin 可映射为一个 p2 advance；
- p1 failed finish + same-node begin 可映射为 failure + retry；
- p1 无法证明从完整任务入口开始时，p2 binding 必须使用
  `history_complete=false`，不能补造全局 Attempt ordinal；
- 无法确认是 operator undo 的历史事件不得猜测为 undo；
- p1 没有 refinement coverage，迁移后为 `unannotated`；
- migration receipt 列出已映射、缺失、歧义和被保留的 p1 fact；
- p1 reader 和 artifact 永久保留。

## 11. 系统迁移顺序

以下状态描述的是本文件对应的 p2 工作，不代表整个 DexTeleop 数据框架的完成度。

### P0：文档与所有权冻结（已完成）

- 冻结 p1；
- 以已经落地的 `collection_runtime` README、`AGENTS.md`、package discovery 和
  boundary tests 为基线；
- 补充 progress request/reducer/store 的所有权，不扩大 operator intent 或 Episode
  owner 权限；
- 明确 teleop entrypoint 不变；
- 不移动现有文件。

### P1-A：优先冻结真源合同（核心已完成，扩展项待定）

- 已冻结 reducer 闭环所必需的最小公共 envelope、payload tagged union 和引用约束；
- origin/actor 扩展词表、物理列、通用 supersession 图、额外 provenance、派生 receipt
  payload 均不在首批过度固化；
- 已落地 linear/rooted-tree progress profile、refinement profile 和唯一 progress-step
  identity；
- 已落地 recording scope、actual entry、history completeness binding；
- 已落地 identities、advance/undo/failure/retry/complete events 和 Hold/Pause receipt；
- 已落地 sealed annotation binding、chunks 和 manifest；
- general coarse/outcome correction 与 progress capability/seal status 尚未冻结；
- 核心 serialization/hash/严格类型负例测试已通过。

### P2-L：linear resolver 与 reducer（核心已完成）

- 已落地 `collection_runtime` controller/reducer；
- 已落地 `dataset_interface` capture-causal resolver，以及窄版 terminal-Failure
  reassignment retrospective 原型；
- advance/undo、entry、截断历史、`FULL_TASK`/`PROGRESS_STEP_WINDOW` termination、
  failure/retry、normal/abnormal termination 和 validated salvage-anchor fixture 已通过；
- general retrospective correction 和 annotation-set selection 尚未落地；
- 覆盖任意合法事件序列的属性测试尚未落地，当前是确定性正负例与 fault fixture。

### P2-T：rooted-tree gate（部分完成，仅离线显式开启）

- 已在同一 P1-A contract 上落地 tree profile validation、ancestor path、离线 reducer
  和显式 `allow_rooted_tree=True` controller gate；
- 显式 branch target、缺失 target fail-closed，以及同 request 的 branch-target
  幂等重放已验证；
- branch correction、Tree Failure reassign、完整 Undo/terminal-leaf 组合矩阵尚未
  验证；
- 不增加 DAG、merge 或运行时回跳实体；
- P2-T 通过前，tree profile 只能做纯合同/离线 fixture，不能进入真机 adapter。

### P1-B：真实数据驱动的派生视图发布合同（延期）

- 当前 derived Python views 和 SQLite cache 不是 canonical dataset truth；
- 当前不分别冻结 `AnnotationCoverageReceiptV1` 与
  `TrainingExclusionReceiptV1`，避免重复 lineage/policy 字段和过早固化训练策略；
- 先由 P4 产生真实 progress/Failure/annotation 数据，再让训练和导出消费者验证
  selection、correction、coverage 与 exclusion 的实际最小交集；
- 两个以上稳定消费者确实需要固定发布结果时，再评估一个薄的 resolved-view
  manifest，绑定输入 manifest/policy/output artifact hashes；
- P1-B 结果不得反向改变 P1-A capture/annotation truth，也不得把 SQLite cache 提升为
  canonical source。

### P3：store 与并发 annotation fixture（核心 fixture 已完成）

- refinement annotation 的 bounded immutable chunk store、crash-incomplete、
  seal、restart 已落地；capture progress 继续使用低事件量 `facts.cjsonl`，暂不为形式
  一致性改写；
- 新 Episode progress 写入与已 sealed 旧 Episode annotation 写入并行 fixture 已通过；
- coverage catalog replace、完整 rebuild、delete/rebuild 和 active-reader WAL snapshot
  fixture 已通过；
- 并发 fixture 验证旧 Episode raw payload 未被改写；
- progress append/fsync 不确定失败会 poison live store，重开后完整 record replay，
  torn tail 继续 fail-closed。

### P4：类真机 default-off 接入（RealTeleop P4-B 代码接线已完成）

接入顺序固定为 RealTeleop 先、UMI 后。RealTeleop 已有完整的 Episode
start/pause/resume/异步 finalize 和 Hold 流程，适合先验证 Failure/Retry 语义；UMI 的
sample clock 和暂停模型不同，应复用同一 progress core，但不能假装与 RealTeleop
使用同一种应用回执。

完整 P4 接入继续遵循以下 thin-adapter 边界：

- RealTeleop 与 UMI 使用同一个 `collection_runtime.progress` reducer/store；
- 首次类真机和真机 gate 只允许 `topology_kind=LINEAR`；Tree 在 P2-T 及分支选择
  adapter 单独通过后再开放；
- 已落地的 RealTeleop adapter 允许一个或多个 step 的 `LINEAR/FULL_TASK`，并允许
  精确 `entry_step + entry_stage + RECOVERY_FROM_FAILURE` 的恢复补采
  `EVENT_WINDOW`；`PROGRESS_STEP_WINDOW` 仍待专用 Episode binding，其他
  `TRANSITION_WINDOW`/通用 `EVENT_WINDOW` 在专用 binding 落地前必须拒绝；
- 一个可选 progress-profile 配置存在时才写 progress fact；配置缺失时不创建
  Episode-local progress 目录、不改 Episode metadata、不改变 Hold/暂停/恢复/收尾
  路径。profile 可由 `--collection-progress-profile` 预选，也可在显式启用
  `--collection-progress-gui` 后、首个 Episode 前，从当前 task 的 managed LINEAR
  catalog 中选择；GUI 模式未选择 profile 时拒绝 Episode start；
- 不增加新的生产 entrypoint，也不把 p1 `begin_subtask/finish_subtask` 当作 p2 接口；
- 首轮不增加长程任务 profile 编辑器或 Tree profile 准入。独立后标注 GUI 已按
  sealed-old-Episode-only gate 落地，不接触 active Episode。

#### P4 Episode 生命周期

现有 dataset Episode owner 仍是唯一身份与 commit owner。接入必须按以下顺序：

1. 在 countdown/warmup 前加载并纯验证 progress profile；失败只拒绝本次 Episode
   启动，不影响遥操；
2. Episode owner 生成现有 `EpisodeCaptureIdentity` 后，用同一 identity 创建
   `ProgressEpisodeBindingV1` 和 subordinate progress participant；
3. progress binding 使用正式样本时间轴，不要求等于 identity 的 preparation clock：
   RealTeleop 使用 `SessionStepResult.timestamp` 的 `local_wall_time`，UMI 使用
   sample timestamp 的 `local_wall_clock`；
4. participant 的准备、legacy writer 启动和绑定必须有显式 rollback；未绑定的临时
   progress 目录不能被发布为有效 capability；
5. owner metadata 只保存 progress profile/binding/manifest 的引用和 identity hash，
   不内嵌 reducer 状态或派生 coverage/exclusion；
6. 零帧 legacy Episode 被丢弃时 subordinate progress 也丢弃；legacy finalize
   失败时 progress 关闭但不 seal；只有 legacy Episode 已成功发布且 progress 已有合法
   termination 时，后台 finalizer 才 seal progress；
7. Ctrl+C、异常退出和 safety stop 必须先写相应 Abort/termination，再走同一
   close/finalize 路径；不能把进程退出误写为正常完整任务成功。人工停止也只有在
   `ACTIVE_ATTEMPT + terminal progress step` 时才隐式成功；非 terminal、Failure
   暂停、等待 Resume 等状态统一以 `MANUAL_STOP` Abort 闭合。

初始 RealTeleop adapter 已采用 capability 名称而不是 provisional 版本名：

```text
meta/progress/episodes/<episode_uuid>/
```

目录中继续使用 store 已有的 `profile.cjson`、`binding.cjson`、`facts.cjsonl` 和
`manifest.cjson`。数据集根另外保存唯一的
`meta/progress/profile.cjson`；它就是完整 TaskProgressProfile 合同本身，而不是
第二套 identity。显式 CLI profile 或 GUI 选择均可启用该落点；当前可直接使用
`catalog:dmtac_rm65_bimanual_position_v1_progress`。统一的只读 profile catalog 位于
`collection_runtime/progress/profiles/`，按 `linear/` 与 `tree/` 分区；当前 RealTeleop
仍只准入 LINEAR。任务自己的 progress/refinement 文件构成 self-contained bundle，
跨任务仅复用可选 `semantic_type` 检索词汇，不引入运行时继承模板。配置缺失时
不创建 progress 目录，也不改变正式数采 metadata 和生命周期。

#### P4 Failure、Hold 与 Retry

Failure 的应用接入必须保持三段事实，不允许用一个按钮时间替代真实效果：

```text
durable Failure fact
  → application Hold + formal capture pause
  → typed Hold/Pause receipt
```

- `ProgressController.report_failure()` 成功并完成 fsync 后，adapter 才把 effect request
  交给 Control Station workflow；
- Control Station 只协调 typed operation；机械臂 Hold 仍由 RealTeleop/session owner
  执行，formal capture pause 仍由 Episode recorder 执行；
- RealTeleop 当前控制循环先请求 `pause_recording()` 与 Hold，随后在既有 Hold
  分支中调用 `session.hold_hardware()` 并释放位控流。只有 Session Hold 返回成功、
  位控流释放未报错且 recorder 暴露真实 Pause 生效时间，adapter 才能写
  `HoldPauseEffectReceiptV1(APPLIED, APPLIED)`；调用 `request_hold()` 本身不算
  receipt；
- 任一效果失败或超时都保持无 active Attempt，并以分项状态 fail closed；不得自动
  创建 Retry。

恢复同样分成：

```text
application resume/Home 生效
  → formal recorder resume 生效
  → 第一条真正被正式 writer 接纳的样本
  → AttemptRetryEvent
```

现有 `record_step_async()` 的返回值只说明请求已提交或入队，不证明 process/thread
writer 已经接纳样本。P4-B 已补齐小型、单 owner 的 accepted-sample receipt port：
thread 路径使用 `handle_step()` 的正式返回，process 路径使用 child terminal ack；
禁止使用 enqueue 时间创建 Retry。通用 receipt 值对象和 bounded/fail-closed retention
机制位于 `collection_runtime.sample_acceptance`，具体 writer 仍是唯一 acceptance
判定者；该 port 不复制 writer cursor，也不是持久化 dataset schema。

这里的 `accepted sample` 具体定义为：

- authoritative Episode writer 已完成该 causal sample 的合同校验；
- writer-owned sample/frame cursor 已经前进，并且所有 formal required jobs 已被
  writer 接纳；它不表示 Episode 已 seal，也不表示每个高带宽 artifact 已 fsync；
- thread 路径在 `handle_step` 正式返回后产生，process 路径必须由 worker terminal ack
  返回实际 `episode_index + frame_index + sample effective time`，父进程 enqueue/
  submitted counter 不能充当 receipt；
- optional media 沿用既有显式 missing/drop contract，不能被 receipt 偷换成“所有可选
  视频均已落盘”；
- Retry 的 `effective_at` 使用 application Resume 与 formal capture Resume 两项均
  生效之后第一条 accepted sample 的 Episode 时间；没有该 receipt 就保持
  `RESUME_RECEIVED`，不得创建 Retry Attempt。

#### P4 最小操作面与分阶段调试

为先获得真实 artifact、避免同时引入大 GUI，分两步接入：

- P4-A（已实现并通过离线 recorder fixture）：单 step `FULL_TASK` profile。Episode
  开始自动创建首个 Attempt，正常 Episode 停止完成该 Attempt；不新增 progress
  操作控件。已验证同一 identity、`local_wall_time` clock、capability 目录、正常/
  safety/interrupt termination、legacy finalize 后 seal、legacy finalize 失败不 seal、
  零帧 discard 和 disabled-path 等价性；并已用真实 v2.3 Parquet writer 验证 legacy
  Episode metadata 与 sealed progress artifact 的 identity/profile/binding hash 一致，
  再通过 `dataset_interface.progress_artifacts` 纯只读加载并重放 timeline；
- P4-B（代码接线与离线 fixture 已完成）：`control_station` 已增加 profile 选择、
  “下一阶段”、
  “报告失败”和“撤销上一次阶段推进”的 typed intent/availability，以及只依赖
  structural ports 的 application workflow。RealTeleop 控制循环负责把该 workflow
  接到现有 Session Hold、formal recorder Pause/Resume 与 writer receipt；Station
  不直接写 fact 或操作机器人。独立的 default-off Control Station 插件在提供
  `--collection-progress-gui` 时显示；它只提交 typed profile/进度 intent，profile
  加载、task 匹配、数据集 hash 锁定仍由下层 adapter/recorder 完成。删除/禁用插件
  不改变原 RealTeleop 页面、A/B/C 或 progress 数据合同。Tree、后标注和 profile
  编辑继续延期。

P4-A 已通过 thread/process、termination、零帧 discard、legacy finalize 失败、未开启
配置以及真实 v2.3 Parquet writer 的类真机 fixture。受控远端非运动验证还确认了：
向已写入一帧的实际 writer fixture 进程投递 `SIGINT` 后，legacy Episode 先完成
finalize，progress Attempt 以 `MANUAL_STOP/ABORTED` 封存；随后在同一 dataset root
重建 recorder/writer 时，legacy episode index 和新的 Episode identity 正确推进，前后
两个 sealed progress artifact 均可纯只读重放。
P4-B 已通过 Hold/Pause 未确认时的 durable-outbox 重放、真实 receipt 前不确认、
Resume 后忽略过期 sample、首个 accepted sample、wrong-Episode/overflow fail-close、
独立 GUI 插件和 disabled-path fixture；相关代码已通过远端 Python 3.10/PyArrow
非运动回归。下一 gate 是明确安全范围内的受控真机 artifact 验证。
真实数据稳定前继续延期：

- general retrospective correction；
- annotation-set selection；
- canonical coverage/exclusion 发布 receipt 或 resolved-view manifest；
- `TRANSITION_WINDOW`/`EVENT_WINDOW`；
- Tree operator workflow。

### P5：训练视图（部分完成）

- 已落地 coarse-only failure exclusion；
- 已落地 partial refinement progressive narrowing 和 validated complete refinement；
- 已落地 unannotated/partial/complete/conflicted/invalid coverage 查询及 SQLite catalog；
- 已落地确定性 Episode sample-axis 与 interval 映射；
- LeRobot/v2.3 显式 sidecar/column conversion receipt 尚未落地；
- 基于 resolved eligibility interval 的随机训练窗口采样尚未落地。

### P6：通用 collection core 迁移评估（已有基础迁移，p2 评估待 P4）

当前通用 writer/persistence 已有一部分迁入 `collection_runtime`。只有 P4 稳定后再
审计仍留在 `teleop_runtime.collection` 的通用部分：

- 只迁移被 RealTeleop、UMI 和非遥操 fixture 共同证明通用的
  identity/sample-clock/commit ports；
- source-specific profile、producer 和 application assembly 继续作为 adapters；
- 硬件、camera、control loop 继续留在对应 `teleop_runtime`/sim owner，GUI policy
  继续留在 `control_station`；
- 保留 `python -m teleop_runtime.collection` 兼容入口；
- 每批移动都保留 shim 和 boundary test，禁止一次性搬迁大目录。

### P7：v2.4 合并（延期）

最终 v2.5 commit 可以引用 p2 progress/annotation/resolved-view manifest 以及 immutable
v2.4 commit。p2 不提前复制 authority、source lineage、reference 或 dispatch 字段。

## 12. 首批代码范围

已落地的 capture-truth/reducer 最小闭环：

```text
ProgressFactEnvelopeV1
ProgressStepDefinitionV1
TaskProgressProfileV1
ProgressEpisodeBindingV1
EpisodeProgressStartedV1
ProgressAdvanceEventV1
ProgressUndoEventV1
AttemptFailureEventV1
HoldPauseEffectReceiptV1
ResumeAfterFailureReceiptV1
AttemptRetryEventV1
EpisodeProgressTerminationV1
ProgressEpisodeManifestV1

collection_runtime.progress requests/reducer/controller/store

RefinementBoundaryDefinitionV1
TaskRefinementProfileV1
RefinementBoundaryAnnotationV1
RefinementAnnotationSetBindingV1
RefinementAnnotationChunkV1
RefinementAnnotationManifestV1
collection_runtime.progress refinement annotation store
dataset_interface sealed refinement annotation reader
dataset_interface selected refinement read view + validated salvage anchor
```

真实数据前不继续冻结的后期语义范围：

```text
immutable progress correction fact/schema（具体最小形态待真实数据验证）
ProgressCapabilityStatusV1
```

P1-B 延后评估：

```text
thin resolved-view manifest（名称、payload、hash domain 尚未冻结）
P1ToP2MigrationReceiptV1
```

当前 canonical timeline resolver 已用 linear/undo/failure/retry、部分 effect、
termination、全局 effective-time 单调性、operation receipt availability 和 manifest
commit-order fixture 验证；sample locator、coarse-only failure
exclusion、validated salvage anchor 及 annotation persistence/restart/seal 也已闭环。
结构性 Attempt coverage 与 SQLite WAL query cache 已用 mixed-status、冲突、非法
引用、精确 annotation-manifest lineage 和 delete/rebuild equivalence fixture 验证；
ROOTED_TREE 分岔 target 的首次请求与幂等重放也已 fail-closed 验证。

下一步按以下顺序推进：

1. RealTeleop P4-B 的远端 Python 3.10/PyArrow 非运动回归已完成；GUI、Session
   Hold 和机器人控制均未下沉到 `collection_runtime`，UMI 路线保持延期；
2. 下一步在明确的安全范围内做受控真机
   Failure→Hold/Pause→Resume→accepted-sample
   artifact 验证；继续保持 progress profile 和独立 GUI 插件分别显式启用，并确认
   禁用 GUI 或全部 progress 开关时的两级回退；
3. 由受控真机采集产生首批真实 progress/Failure/Retry 数据，并用独立 stage
   annotation GUI 验证首批 failure/recovery occurrence；
4. 再依据真实数据原型化 annotation-set selection 与通用 retrospective correction；
5. `PROGRESS_STEP_WINDOW`、`TRANSITION_WINDOW`/`EVENT_WINDOW`、Tree operator
   workflow、resolved-view
   manifest 和派生持久化列都由
   真实消费者需求决定；在专用 binding 落地前 adapter 显式拒绝。

## 13. 验收重点

以下是 p2 整体验收目标，包含已经通过的离线能力和仍属于 P1-B/P2/P4 的未来 gate；
阶段完成状态以第 11 节为准，不能仅凭本节条目推断已经接入应用或真机。

- progress envelope 的 sequence 连续、effective/availability clock 合法，payload 只能是
  已注册 tagged union；
- 误触 advance + undo 后，规范时间线与从未误触完全等价；
- undo 同时撤销隐式 success 和错误新 Attempt；
- 完整任务、任意 step 入口、截断历史和局部窗口不会被互相误判；
- 重复 semantic type 通过唯一 `progress_step_id` 保持可区分；
- 默认 linear profile 只能进入下一个 ordinal；rooted-tree profile 禁止环、汇合、
  孤立节点和非单调 child，多后继时必须显式选 target；
- assumed completed context 必须是 entry 的完整祖先 path，跳过祖先时 fail；
- failure 后立即没有 active Attempt；Hold 与 formal capture pause 任一未 APPLIED 都
  禁止 retry；
- durable Failure fact 落盘后才发布 Hold/Pause 副作用；同一 Failure `fact_id`
  重放不得重复执行语义或重复确认 receipt；
- formal capture resume 未 APPLIED 时不得创建新 Attempt；新 Attempt 不得早于正式
  capture 恢复后的第一个样本；
- retry 保持 coarse progress step 不变并创建新 Attempt；
- coarse-only failure 排除范围正确；
- 只有连续覆盖且证明有效前缀的 salvage anchor 才能收缩排除范围；
- coarse/failure correction 只改变显式 retrospective view，不改 capture fact；
- failure 可以原子重归属到正确 Attempt，相关错误 advance/Attempt 不得留下悬空
  引用；
- window 正常结束不会产生错误的 full-task success；
- `WINDOW_COMPLETED` 只作为 termination cause；中途结束的 Attempt 必须 truncated；
- reducer 每个非终态都能被显式 termination 闭合；被 termination 引用的 unresolved
  effect 可 COMPLETE，但半写入 fact/batch 绝不能 seal；
- refinement annotation 通过 `coarse_attempt_id + refinement_boundary_id` 定位，不能
  依赖显示 path；
- annotation 绑定 active progress log 时 fail，绑定 sealed manifest 时 pass；
- selected annotation/coverage/training view 必须保存其精确 annotation manifest hash
  lineage，不能只凭 Episode 或 profile 猜测 annotation set；
- annotation chunk/index 必须连续，重复 annotation ID 仅在内容完全一致时幂等；
  缺块、多块、foreign clock、hash 冲突或 manifest 不完整时 reader/store 都 fail
  closed；
- 新 Episode 采集与旧 sealed Episode 标注并行时不重写或锁住 raw 数据；
- intentional Hold pause 可由 receipt 时间派生，不误报为 sensor/writer gap；
- progress capability 的 ABSENT/COMPLETE/INCOMPLETE/FAILED 不影响 raw/native 自身
  有效性，只有 COMPLETE 默认进入 progress training view；
- coverage 可按 Episode、Attempt、profile、状态快速查询；
- SQLite catalog 格式版本不匹配时必须拒绝旧派生缓存；删除 cache 后可从 canonical
  manifests 完整重建，重建结果与增量写入结果等价；
- progress store 在 write/fsync 结果不确定后必须 poison，禁止继续 append/seal；
  重开时只能完整 replay 或对 torn tail fail closed；
- p1/v2.3/LeRobot 默认输出保持不变；
- 所有 runtime 接入 default-off 且可通过禁用 adapter 回退。
- P4 初次接入只接受 LINEAR；Tree 必须通过 P2-T 和分支选择 adapter 的独立 gate。
- P4 在专用 binding 落地前必须拒绝 `TRANSITION_WINDOW` 和通用
  `EVENT_WINDOW`，不能以通用 fallback 冒充完整支持；精确 stage recovery entry
  是唯一窄例外，并始终按非完整窗口解释。

## 14. 当前已确认的假设

- 在线不支持有意退回更早 coarse progress step；唯一回退是撤销最后一次错误
  advance；
- topology 默认 LINEAR，也允许无环、无汇合且分支内单调的 ROOTED_TREE；
- tree ordinal 全局唯一，只承担确定性排序和单调验证，不代表唯一执行顺序；
- 可以从任意合法 coarse step 开始；Profile initial 只用于完整任务默认入口；
- Attempt ordinal 只在本次 capture 内有效，不声称是全任务历史 ordinal；
- 正常 advance 隐式 success；Episode 结束按 recording scope/terminal 条件解析；
- failure 立即请求 Hold/Pause，同一 Episode 内 resume 后重试当前 progress step；
- 当前项目 Hold 完整暂停正式采集；p2 首批不继续 raw capture，也不增加
  training-only/raw-capture 两套 pause 模式或 canonical CaptureGap 合同；receipt 仍
  分开验证 application hold 与 formal capture pause 是否真正生效；
- 后标注是同一 Episode 的 overlay，不物理改写原始数据；
- coverage 按 coarse Attempt + refinement profile version 计算；
- partial annotation 只有通过 Profile、连续 coverage、conflict 和 confidence 检查的
  validated salvage anchor 才能渐进缩小排除范围；
- causal view 保留在线认知，retrospective/training view 才应用显式 correction。

## 15. 面向后续模块和机制的升级原则

p2 仍只是基础框架。未来可能增加新的传感器、机制判断、人类偏好、reward 或其他
高层语义。兼容这些扩展不意味着现在预先加入大量空字段，而是固定少数稳定的组合
边界。

### 15.1 五层数据必须分开

```text
1. raw measurement/action streams
2. capture-time operator facts
3. mechanism signals/events
4. post-capture annotations
5. derived training views/indexes
```

- 原始触觉、图像、qpos、pose、action 属于第 1 层；
- advance、undo、failure 属于第 2 层；
- 接触概率、slip、碰撞或抓取状态属于第 3 层；
- A1/A2/A3 和连续人工偏好属于第 4 层；
- failure exclusion、采样权重和训练窗口属于第 5 层。

下层事实可以被上层引用，但上层不得反向改写下层。训练 view 不是原始事实。

### 15.2 新传感器与机制判断

例如增加触觉传感器并产生“是否接触”：

- 原始触觉值使用新的 native component stream/profile；
- contact detector 使用独立、带版本和参数 hash 的机制合同；
- 输出可以是连续 contact probability stream，也可以是稀疏 contact begin/end
  event；
- 输出引用源触觉 records、detector identity 和 effective/available time；
- 在线计算与后处理计算可以共存，通过 origin/availability 区分；
- 旧 Episode 缺少该 capability 时保持 missing，不填充为 `false` 或 `0`；
- 是否把 contact 用于训练由显式 training view 决定，不修改原始触觉。

触觉算法不进入 `collection_runtime`；设备 producer、机制分析模块和 dataset-boundary
adapter 各自保持原有 owner。

### 15.3 连续偏好不能扩展成 success/failure 的附加枚举

未来的连续偏好可能是标量、向量、成对比较或一段时间内变化的信号。它应拥有独立
typed annotation/stream contract，至少声明：

- value semantics、dtype/shape、范围/单位；
- actor/annotator；
- effective/available time；
- 所绑定的 Episode、Attempt 或 interval；
- profile/schema hash 和 provenance。

success/failure 继续只表示 Attempt outcome。reward、preference、quality score 和
training weight 不复用该字段，也不由 reader 猜测互相等价。

### 15.4 混合 capability 和覆盖率是正常状态

“数据集已完整标注”不能是一个全局布尔值。覆盖率必须至少按以下维度查询：

```text
episode / attempt
semantic or mechanism capability
profile/schema version
annotation set or producer
coverage status
```

因此同一数据集中可以合法存在：

- 无触觉和有触觉 Episode；
- 只有 raw tactile、但没有 contact 推断的 Episode；
- coarse-only、partial refinement 和 complete refinement Attempt；
- 无偏好、稀疏偏好和连续偏好数据。

训练入口必须显式声明 required/optional capabilities 和 missingness policy。

### 15.5 扩展方式

- 新语义用新 schema ID/hash domain，不修改旧 schema 的含义；
- 每种新增 layer 有独立 manifest，可由 Episode semantic bundle 通过 typed
  layer reference 组合；
- 公共 reference 只固定 identity、time、interval、schema、provenance 和 coverage；
- 具体 payload 使用独立 typed contract，不建立任意嵌套的万能 dict；
- 物理格式与逻辑 hash 分离，可以在不改语义的情况下从 CJSON 迁移到 chunked
  Parquet；
- 新 capability 默认可选、default-off，并通过 adapter/receipt 接入；
- 只有实际 producer 或标注流程存在时才落地具体字段和 profile，避免为假想需求
  增加冗余。

# DexTeleop v2.5-p2 通用采集进度与后标注规划

状态：规划确认，尚未修改运行代码或默认采集行为  
前置：v2.5-p1 native component capture、episode identity、兼容投影  
参考：`todos/long_horizon_dataset_plan.md`

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
- 默认单调迁移；
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

它不包含 A1/A2/A3，不包含机器人、本体、GUI、动作空间或存储布局。

### 3.2 本次 Episode 的实际入口

Profile 的 initial step 只是完整任务默认入口。本次采集从哪里开始由
`ProgressEpisodeBindingV1` 决定：

```text
recording_scope:
  FULL_TASK | SUBTASK_WINDOW | TRANSITION_WINDOW | EVENT_WINDOW
entry_step_id:
entry_mode:
  PROFILE_INITIAL | ACTUAL_TRUNCATED_HISTORY | SCENARIO_SETUP | NO_PREHISTORY
history_complete:
assumed_completed_step_ids:
truncated_at_start:
```

约束：

- `FULL_TASK + PROFILE_INITIAL` 默认从 profile initial step 开始；
- 其他 scope 可以从任意合法 `entry_step_id` 开始；
- `assumed_completed_step_ids` 只能引用同一 profile 中 entry 之前的合法 step，且只
  表示本次绑定的明确前置假设，不伪造未采集的 Attempt；
- `history_complete=false` 时，不得推断当前是该 step 在真实任务历史中的第一次
  Attempt；
- Attempt 只保存 `attempt_ordinal_in_capture`，不声明任务全历史绝对 ordinal；
- `truncated_at_start=true` 时，训练/评估 view 必须能区分 Episode 边界和真实任务
  起点；
- 局部窗口正常结束不等于 full-task success。

这使单独采一个节点、中段开始、人工 setup、failure/off-nominal 专项窗口和截断历史
都能被准确表达，而不修改可复用 Profile。

### 3.3 操作语义

在线领域意图包括：

- `AdvanceSubtaskIntent`：进入下一个粗子任务；
- `UndoLastAdvanceIntent`：撤销最近一次误触的 advance；
- `ReportFailureIntent`：当前 Attempt 失败。

以下是生命周期输入，不是额外的数采员语义按钮：

- `StartEpisodeProgress`：Episode owner 启动进度；
- `ResumeFailedAttempt`：现有 Hold/Resume 流程恢复后，在同一 coarse subtask
  创建新 Attempt；
- `CompleteEpisodeProgress`：Episode owner 正常结束 Episode；
- `AbortEpisodeProgress`：故障、碰撞或人工放弃等非正常终止。

因此 UI 或脚踏板仍然只需提供“子任务管理”和“失败”两个能力。

### 3.4 Advance 是一个原子事务

一次 `A -> B` advance 在语义上同时完成，并显式绑定 from/to
`progress_step_id`：

1. 将 A 的当前 Attempt 结束为 success；
2. 记录切换时间；
3. 将当前 coarse subtask 更新为 B；
4. 创建 B 的新 Attempt。

四项共享一个 `progress_transition_id`，不能分别提交。持久化层只接受完整事务。

## 4. 误触撤销为什么使用追加事件

物理删除最后一个时间戳虽然直观，但会破坏已 flush/chunk 的不可变数据，也会在
采集和标注并行时制造重写和竞态。p2 使用：

```text
Advance A -> B @ t1, transition_id=17
Undo transition_id=17 @ t2
Advance A -> B @ t3, transition_id=19
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
2. 必须引用精确 `progress_transition_id`，不能用“最近时间戳”猜测；
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
  -> FAILURE_REPORTED
  -> AWAITING_HOLD_RECEIPT
  -> PAUSED_NO_ACTIVE_ATTEMPT
  -> RESUME_RECEIVED
  -> ACTIVE_RETRY_ATTEMPT
```

`ReportFailureIntent` 生效时立即：

1. 在当前 `progress_step_id` 的 active Attempt 上记录 failure；
2. 将该 Attempt 结束为 failure；
3. 清空 `active_attempt_id`；
4. 返回 `request_hold_and_capture_pause=true` effect；
5. 进入 `AWAITING_HOLD_RECEIPT`。

collection semantics 本身不控制机器人、不执行 Hold，也不拥有安全状态机。真机、
仿真或其他应用执行 effect，并返回 `HoldPauseEffectReceiptV1`：

```text
failure_event_id
request_issued_at
hold_effective_at
capture_pause_effective_at
status = APPLIED | PARTIALLY_APPLIED | REJECTED | TIMEOUT
```

只有 `APPLIED` 才能进入 `PAUSED_NO_ACTIVE_ATTEMPT`。其他状态不得静默 retry；应用
必须继续等待、异常终止 Episode，或将 progress capability 标为 incomplete。

现阶段结合项目实际，Hold 会完整暂停正式 raw/formal capture。p2 不设计 Hold 期间
继续采 raw、不区分 training/raw 两种 pause，也不在首批合同增加
`CaptureGapIntervalV1`。暂停边界由 failure、已应用的 hold receipt 和 retry 明确
限定；如果未来 Hold 行为改变，再以新 capability/schema 增加更细的 capture-gap
合同。

人工调整并通过现有 Resume 后，`AttemptRetryEventV1` 必须显式引用：

```text
source_failure_event_id
hold_pause_receipt_id
resume_receipt_id
new_attempt_id
effective_at
```

`resume_receipt_id` 指向由应用的权威 Hold/Resume 状态转换生成的
`ResumeAfterFailureReceiptV1`；collection runtime 只验证其 failure/episode 绑定与
`APPLIED` 状态，不自行判断机器人是否已经恢复。

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

在线数采员通常只能可靠提供报告时间，不能声称知道物理 failure onset。因此
`AttemptFailureEventV1` 区分：

```text
reported_at
effective_at
timing_semantics = OPERATOR_REPORT_TIME | ESTIMATED_ONSET | POSTHOC_CORRECTED_ONSET
```

在线默认 `reported_at == effective_at` 且使用 `OPERATOR_REPORT_TIME`。Hold 真正
生效的时间只来自 receipt。后标注可以增加 onset/outcome correction，但不能改写
原始 failure event。

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
 first_sample_strictly_after(failure_effective_at))
```

这样不会依赖浮点时间恰好相等，也能明确包含 failure 前最后一个采样帧。

`TrainingExclusionIndexV1` 只保存派生的排除 interval、原因、来源 failure fact、
所选 annotation set/profile 和 resolver 版本。它不是 raw truth，可以低成本重建。

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
termination_cause
current_attempt_outcome = SUCCESS | FAILURE | WINDOW_COMPLETED | ABORTED | UNKNOWN
full_task_outcome = SUCCESS | FAILURE | UNKNOWN | NOT_APPLICABLE
```

`FULL_TASK` 从合法 terminal step 正常完成，才能隐式产生 current Attempt success 和
full-task success。`SUBTASK_WINDOW`、`TRANSITION_WINDOW`、`EVENT_WINDOW` 正常结束
时可以产生 `WINDOW_COMPLETED`，但 `full_task_outcome` 必须是 `NOT_APPLICABLE`；如
调用方要声明窗口中的 coarse Attempt 完整成功，必须使用显式、可验证的 completion
语义，不能从“Episode 正常关掉”自动推断。

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
boundary_id
path
ordinal
semantics = ENTER_STAGE | COMPLETE_STAGE
eligible_as_exclusion_anchor
certifies_valid_prefix
required_predecessor_boundary_ids
```

普通分段边界默认不能作为训练排除收缩 anchor。只有 Profile 明确声明、且实际
annotation coverage 满足前置条件时，resolver 才能使用它释放前缀数据。

### 6.2 标注事实

`RefinementBoundaryAnnotationV1` 至少包含：

- Episode identity；
- coarse Attempt identity；
- refinement profile hash；
- hierarchical path，例如 `A/A3`；
- effective timestamp 和 sample locator；
- annotator/origin/confidence；
- annotation set identity；
- supersedes/source references。

标注只追加到独立 annotation set，不修改 raw frames、videos、native streams 或
capture progress facts。一个正在编辑的 annotation set 是 append-only draft；完成后
封存 manifest。修改通过新的事实/supersession 完成，而不是原地覆写旧 chunk。

### 6.3 混合覆盖率

覆盖率以 `(episode, coarse_attempt, refinement_profile_hash)` 为基本单位：

- `unannotated`：没有可用精细边界；
- `partial`：有部分边界，但未满足 profile 完整性；
- `complete`：满足 profile 要求且时间线通过验证；
- `conflicted`：多个事实无法按选定 policy 唯一解析；
- `invalid`：越界、非法顺序或引用错误。

同一数据集可以同时包含所有状态。缺失标注保持缺失，不转换成默认 A1 或 success。

### 6.4 后标注必须能修正 coarse 与 outcome

在线 undo 只解决刚发生且仍是最后一个有效 advance 的误触。对于事后发现的漏报、
晚报或错误结果，annotation 层预留两个 typed correction：

`CoarseProgressCorrectionAnnotationV1`：

```text
INVALIDATE_ADVANCE
RETIME_ADVANCE
CHANGE_TARGET_STEP
```

`AttemptOutcomeCorrectionAnnotationV1`：

```text
SET_SUCCESS
SET_FAILURE
SET_UNKNOWN
RETIME_FAILURE
```

约束：

- correction 必须引用被修正的 capture event、transition 或 Attempt；
- correction 不能修改或删除 capture-time fact；
- causal view 保留当时在线系统的认知；
- retrospective/training view 只有在显式选择 annotation set 和 resolver policy 后
  才应用 correction；
- retime/change-target 必须重新验证完整进度顺序、Attempt 嵌套和 profile 迁移；
- 冲突修正产生 `conflicted` coverage，不能按“最新一条”静默覆盖；
- correction 会使相关 coverage/exclusion view 失效并重建，但不重写 raw/native
  数据。

首批 contract 冻结这两个 tagged union 和引用边界；完整 correction resolver 在
P2 实现。

## 7. “原地标注”与低 IO 的实现

这里的“原地”定义为：针对同一 Episode 增加 overlay，并立即更新可查询视图，而
不是改写 Episode 主数据。

推荐组织：

```text
meta/v25p2/
  profiles/
    progress/<hash>.cjson
    refinement/<hash>.cjson
  episodes/<episode_uuid>/
    progress/
      binding.cjson
      chunks/<chunk>.cjson|parquet
      manifest.cjson
  annotations/<annotation_set_id>/
    binding.cjson
    chunks/<chunk>.cjson|parquet
    manifest.cjson
  derived/
    coverage/<view_hash>.parquet
    training_exclusions/<view_hash>.parquet
  indexes/
    semantic_catalog.sqlite
```

原则：

- capture 与 annotation 写不同目录和文件，不争用 raw writer；
- 每个 annotation set 只有一个 append owner；不同标注任务使用不同 set；
- 事件按 chunk 写入，避免一个事件一个文件；
- 标注只产生小型边界事实，不重写高带宽数据；
- coverage 和 exclusion 是小型派生表，可增量或整体重建；
- `semantic_catalog.sqlite` 是可选、可删除、可重建的查询缓存，不属于 canonical
  dataset identity；
- SQLite 使用 WAL 支持采集期间的读写查询；catalog 更新失败不影响 canonical
  annotation commit，可稍后重建；
- 可移植的 canonical 来源始终是 profile、fact chunk 和 manifest，不要求训练端
  信任 SQLite。

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

### 8.2 建议新建 `collection_runtime/`

该包拥有与设备、控制和 UI 无关的采集期语义状态与 subordinate lifecycle：

```text
collection_runtime/
  __init__.py
  semantics/
    intents.py          # typed domain intents
    events.py           # accepted domain events/effects/receipts
    reducer.py          # canonical progress state reducer
    controller.py       # validate intent -> event/effect
    ports.py            # clock, identity, persistence, hold/pause receipts
  persistence/
    progress_store.py   # append/chunk/seal; subordinate to episode owner
  episode/
    ports.py            # generic episode identity/sample-clock boundary
```

严格边界：

- 可以依赖 `dataset_contracts`；
- 不依赖 `teleop_runtime`、`control_station`、`hardware`、`sim` 或具体本体；
- 不产生机器人命令，不拥有 safety/authority/dispatch；
- 初期不夺取现有 Episode owner，只接收其 identity 和 clock；
- 后续只有在 RealTeleop、UMI 和至少一个非遥操 fixture 都通过同一 port 后，才考虑
  将通用 Episode ownership 迁入；
- 不增加新的生产 entrypoint。

实现该包时需要同步更新 `AGENTS.md`、架构文档和 `pyproject.toml` package discovery；
没有依赖变化时不需要修改 `uv.lock`。

### 8.3 其他包的职责

`dataset_contracts/`：

- progress/refinement profile；
- per-Episode recording scope、actual entry 和 history completeness binding；
- intent/event payload 的持久化合同；
- Episode/Attempt/transition/annotation identity；
- Hold/Pause receipt、correction、manifest、coverage/exclusion receipt schema；
- 纯验证和 hash；
- 不包含 reducer、writer、IO 或 GUI。

`dataset_interface/`：

- raw event + undo 的规范 timeline resolver；
- coarse/outcome correction 和 validated salvage-anchor resolver；
- p1 -> p2 migration adapter；
- refinement annotation reader/writer boundary；
- coverage builder；
- training exclusion/index builder；
- frame/interval/window projection与查询；
- rebuildable SQLite catalog adapter；
- 不拥有在线 current-subtask 状态或 Episode lifecycle。

`teleop_runtime/`、`control_station/`、UMI、sim：

- 把按钮、脚踏板或其他来源翻译成 collection intents；
- 执行 controller 返回的 hold/pause effect；
- 把现有 Episode identity、sample clock、resume/stop receipt 传入 ports；
- 不复制 progress reducer，不直接构造持久化 event。

## 9. 数据流

```text
operator input / application lifecycle
              ↓ thin adapter
collection_runtime typed intent
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
- Hold/safety：应用现有控制/安全 owner。

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

### P0：文档与所有权冻结

- 冻结 p1；
- 在 `AGENTS.md` 规划 `collection_runtime` 边界；
- 明确 teleop entrypoint 不变；
- 不移动现有文件。

### P1：纯 p2 contracts

- progress/refinement profiles 和唯一 progress-step identity；
- recording scope、actual entry、history completeness binding；
- identities、advance/undo/failure/retry/complete events；
- Hold/Pause receipt、coarse/outcome corrections；
- manifests、annotation、coverage 与 exclusion receipts；
- 序列化/hash/负例测试。

### P2：resolver 与 reducer

- `collection_runtime` controller/reducer；
- `dataset_interface` retrospective 与 causal resolver；
- advance/undo 原子事务测试；
- 任意合法 entry、截断历史和局部窗口 termination 测试；
- failure/retry、normal completion 和 abnormal termination 测试；
- correction 和 validated salvage-anchor 测试；
- 属性测试覆盖任意合法事件序列。

### P3：store 与并发 annotation fixture

- chunked progress/annotation store；
- crash-incomplete、seal、restart；
- collection 写入与 annotation 写入并行；
- coverage/catalog 增量更新和完整 rebuild 等价；
- raw/native artifact hash 不变。

### P4：类真机 default-off 接入

- RealTeleop 与 UMI 只增加 thin adapter；
- 两者使用同一个 reducer 和 store；
- thread/process/interrupt/resume 测试；
- 未开 flag 时行为和 legacy 输出完全不变；
- 不增加长程任务 GUI。

### P5：训练视图

- coarse-only failure exclusion；
- partial refinement progressive narrowing；
- complete refinement；
- unannotated/partial/complete 查询；
- LeRobot/v2.3 显式 sidecar/column conversion receipt；
- 随机窗口和 interval sampling。

### P6：通用 collection core 迁移评估

只有 P4 稳定后再审计 `teleop_runtime.collection`：

- 可复用 Episode identity/sample clock/commit ports 迁入 `collection_runtime`；
- RealTeleop/UMI writer 变成 adapters；
- 硬件、GUI、camera、control loop 继续留在应用包；
- 保留 `python -m teleop_runtime.collection` 兼容入口；
- 每批移动都保留 shim 和 boundary test，禁止一次性搬迁大目录。

### P7：v2.4 合并（延期）

最终 v2.5 commit 可以引用 p2 progress/annotation/resolved-view manifest 以及 immutable
v2.4 commit。p2 不提前复制 authority、source lineage、reference 或 dispatch 字段。

## 12. 首批代码范围

下一次实现只建议进行 P0 + P1，不创建 runtime writer，不接真机：

```text
ProgressStepDefinitionV1
TaskProgressProfileV1
TaskRefinementProfileV1
ProgressEpisodeBindingV1
ProgressAdvanceEventV1
ProgressUndoEventV1
AttemptFailureEventV1
HoldPauseEffectReceiptV1
ResumeAfterFailureReceiptV1
AttemptRetryEventV1
EpisodeProgressTerminationV1
RefinementBoundaryAnnotationV1
CoarseProgressCorrectionAnnotationV1
AttemptOutcomeCorrectionAnnotationV1
AnnotationCoverageReceiptV1
TrainingExclusionReceiptV1
P1ToP2MigrationReceiptV1
```

在这些合同通过评审后，再创建 `collection_runtime` reducer。这样先冻结公共数据
语义，再迁移 runtime ownership，避免新包建立在仍会变化的事件定义上。

## 13. 验收重点

- 误触 advance + undo 后，规范时间线与从未误触完全等价；
- undo 同时撤销隐式 success 和错误新 Attempt；
- 完整任务、任意 step 入口、截断历史和局部窗口不会被互相误判；
- 重复 semantic type 通过唯一 `progress_step_id` 保持可区分；
- failure 后立即没有 active Attempt，Hold/Pause receipt 未 APPLIED 前禁止 retry；
- retry 保持 coarse subtask 不变并创建新 Attempt；
- coarse-only failure 排除范围正确；
- 只有连续覆盖且证明有效前缀的 salvage anchor 才能收缩排除范围；
- coarse/failure correction 只改变显式 retrospective view，不改 capture fact；
- window 正常结束不会产生错误的 full-task success；
- annotation 与 capture 并行时不重写或锁住 raw 数据；
- coverage 可按 Episode、Attempt、profile、状态快速查询；
- 删除 SQLite cache 后可从 canonical manifests 完整重建；
- p1/v2.3/LeRobot 默认输出保持不变；
- 所有 runtime 接入 default-off 且可通过禁用 adapter 回退。

## 14. 当前已确认的假设

- 在线不支持有意退回更早 coarse subtask；唯一回退是撤销最后一次错误 advance；
- 可以从任意合法 coarse step 开始；Profile initial 只用于完整任务默认入口；
- Attempt ordinal 只在本次 capture 内有效，不声称是全任务历史 ordinal；
- 正常 advance 隐式 success；Episode 结束按 recording scope/terminal 条件解析；
- failure 立即请求 Hold/Pause，同一 Episode 内 resume 后重试当前 subtask；
- 当前项目 Hold 完整暂停正式采集；p2 首批不继续 raw capture、不拆分两种 pause、
  不增加 CaptureGap 合同；
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

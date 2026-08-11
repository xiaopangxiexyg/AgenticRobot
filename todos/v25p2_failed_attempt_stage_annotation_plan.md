# v2.5-p2 失败 Attempt 子阶段标注校准方案

状态：GUI/application reflow 已落地；数据合同与 capture path 未修改。

## 1. 结论

失败是 coarse Attempt 的结束结果，也是该 Attempt 内最后一个 fine stage
occurrence 的结果；它不应把整个 Attempt 压缩成不可再细分的单一“失败子任务”块。

现有合同已经能够表达：

```text
Attempt a1:
  A1 SUCCESS
  A2 FAILURE  -> source_failure_fact_id=failure:1

Attempt a2:
  A2 SUCCESS
  A3 FAILURE  -> source_failure_fact_id=failure:2

Attempt a3:
  A3 SUCCESS
```

因此本问题不需要新增并行的失败协议，也不应改写 capture-time Failure。应校准
Control Station 的草稿构造和编辑交互，让标注员能显式选择“本次 Failure 发生在哪个
子阶段”，由编辑器按 profile 顺序物化该失败 Attempt 内的 stage occurrences。

## 2. 当前实现为什么看起来无法添加

`StageOccurrenceAnnotationV1`、`dataset_interface.progress_stages` 以及已有测试已经
允许一个失败 Attempt 包含多个 occurrence。当前 GUI 的初始草稿却采用最保守的折叠：

```text
失败 Attempt -> 当前 expected stage 的一个 FAILURE occurrence
```

现有“在当前画面前完成上一阶段”操作可以递进拆分：

```text
A1 FAILURE
  -> 在边界 t1 拆分
A1 SUCCESS + A2 FAILURE
  -> 在边界 t2 再拆分
A1 SUCCESS + A2 SUCCESS + A3 FAILURE
```

但这项能力存在明显的交互缺陷：

1. 按钮名称没有说明它是在给失败 Attempt 增加子阶段；
2. 必须先把播放游标放进当前失败块内部，条件不明显；
3. 多阶段 Failure 需要重复执行相同操作；
4. 时间轴只显示当前已物化的失败块，没有展示尚未到达的 profile stage，用户会误以为
   数据模型只允许一个阶段；
5. 失败 stage 的选择、边界划分和 Retry 起点三个概念被一个间接的 split 操作耦合，
   冲突时缺少可读解释。
6. 当前 split 只改变被拆分的失败 Attempt；随后调用的 occurrence normalize 只过滤
   foreign stage 并重排 occurrence index，不会把后续 bootstrap Retry 从旧 stage
   重新对齐到新的 failed stage。因此即使界面上拆分成功，封存时仍可能因 Retry 倒序
   而 fail closed。现有使用文档声称“后续 Retry 会重新对齐”，与实现并不完全一致。

所以这不是底层数据能力缺失，而是 GUI 将已有能力暴露得过于隐蔽。

## 3. 推荐交互：先选失败 stage，再调整边界

不提供任意的“+ 子阶段”按钮。任意添加会产生重复、倒序、跳级和 Failure 归属不明。
对每个 capture-time 失败 Attempt，提供一个受 profile 约束的编辑卡片：

```text
本次 Attempt 从：A1 开始（由 entry/前序 Attempt 推导，只读）
原始 Failure：00:12.430（只读）
失败发生于：[A1] [A2] [A3]
```

若标注员选择 `A3`，编辑器一次性生成：

```text
A1 SUCCESS | A2 SUCCESS | A3 FAILURE
```

并在 Attempt 的有效采样时间上均分放置两个初始边界。随后标注员通过视频和游标精调
`A1/A2`、`A2/A3` 边界。最后的 `A3 FAILURE` 必须结束于原始 Failure cut，并引用原
`source_failure_fact_id`。

原逐次拆分入口已从 GUI 移除，避免它只改当前 Attempt、却留下后续 Retry 错序。正式
交互统一为上述“失败发生于哪个 stage”的直接选择。

## 4. 必须保持的状态约束

### 4.1 单个失败 Attempt

假设本次 Attempt 的入口是 stage `Si`，标注员选择失败 stage `Sj`：

- 必须满足 `j >= i`；
- `Si ... S(j-1)` 各出现一次且 outcome 为 `SUCCESS`；
- `Sj` 恰好出现一次且 outcome 为 `FAILURE`；
- `Sj` 必须是本 Attempt 最后一个 occurrence；
- `Sj.end_at` 必须等于原始 Failure cut；
- `Sj.source_failure_fact_id` 必须等于关闭该 Attempt 的 Failure fact；
- Failure 后到 Resume 之间没有 active Attempt，不能生成 stage occurrence；
- 不允许在同一 Attempt 的 Failure 后继续添加阶段。

### 4.2 跨 Attempt

- 一个失败 stage 可以在后续 Retry 中重复；
- Retry 的入口 stage 必须等于前一失败 Attempt 的 terminal failed stage；
- 已成功 stage 不得在正常顺序中重复；
- 后一个 Attempt 不能倒退到更早 stage；
- coarse step 最终成功时，从 Episode entry stage 起的成功 stage 序列必须完整、无重复、
  严格按 profile 顺序；
- recovery window 从非首 stage 开始时，以 capture-time
  `entry_condition=RECOVERY_FROM_FAILURE` 和 `entry_stage_id` 为准，不补造缺失前缀。

### 4.3 改变失败 stage 后的传播

改变某个失败 Attempt 的 terminal stage 会改变后续 Retry 的 expected entry。编辑器
不得静默重写已经人工调整过的后续 occurrences：

- 后续仍是未编辑的 bootstrap 草稿时，可以确定性重新物化；
- 后续已有人工边界或 stage 编辑时，先显示冲突，要求标注员确认整体重排或撤销本次
  选择；
- 已封存 annotation set 永不原地修改，只能基于它创建新草稿和新版本。

## 5. 时间与显示

- canonical annotation 始终保存 Episode clock 上的真实 `start_at/end_at`；
- 原始 Failure 时间、视频时间和 Parquet sample axis 不变；
- Hold/Pause 只在 GUI display axis 上压缩，不得进入 Attempt stage 区间；
- 初始均分应在 Attempt 的有效采样跨度上完成，而不是跨越 Hold 的 wall time；
- 所有拖动边界吸附或最终校验到正式 sample axis，避免空区间和一帧重叠。
- 若正式 sample 数不足以让所选的每个 occurrence 至少占一个非空样本区间，应禁止该
  选择并提示“有效帧不足”，不能仅在连续时间上造出看似有效、实际为空的阶段。

## 6. 未知或证据不足时

第一版不新增 `UNKNOWN_STAGE` 或伪造一个 stage。若标注员无法判断 Failure 所属阶段：

- 允许保存未完成草稿；
- 不允许把它封存成 `COMPLETE` stage view；
- 可以标记“需要二次复核”；
- 若整条数据本身不可解释，再使用整条异常，而不是猜测 stage。

这样避免为了 GUI 便利扩张正式协议。

## 7. 训练和索引语义

完成细标后，默认成功训练视图应：

- 保留失败 Attempt 中 Failure 之前已经完整成功的 stage occurrences；
- 排除 terminal failed stage occurrence 的 `[stage_start, failure_cut]`；
- 排除 Failure 到正式 Resume/首个 accepted retry sample 之间的暂停区间；
- 接纳后续 Retry 中成功完成的同名 stage occurrence；
- 通过 `coarse_attempt_id + occurrence_index_in_attempt + stage_id` 高效检索；
- 原始 Failure fact 与后标注 stage occurrence 保持明确 lineage。

这与现有 `dataset_interface.progress_stages` 的派生方式一致，不需要重写 Parquet、视频或
capture facts。

## 8. 可回退性

建议的第一阶段只修改可变草稿/UI 行为：

- 保持 `TaskStageProfileV1`、`StageOccurrenceAnnotationV1` 和 immutable annotation set
  不变；
- 旧数据、旧封存版本和旧草稿继续可读；
- 新编辑器仍输出现有 occurrence payload；
- 若新交互发现问题，可回退 application/frontend，不需要迁移数据或改 hash domain。

只有真实数据证明现有合同无法表达“未知 failure stage”时，才单独讨论新 schema；当前
不应提前加入。

## 9. 实现前验收用例

1. `A1 FAILURE`：失败发生在首 stage，无新增成功块；
2. `A1 SUCCESS -> A2 FAILURE`：一次选择和一个边界；
3. `A1 SUCCESS -> A2 SUCCESS -> A3 FAILURE`：两个边界；
4. `A2 FAILURE -> retry A2 SUCCESS -> A3 SUCCESS`：Retry 从同 stage 继续；
5. 同一 stage 连续失败两次：两个 Attempt 各有独立 `A2 FAILURE`；
6. recovery Episode 从 A2 开始：不得补造 A1；
7. 改变前一 Failure 为 A3、但后续人工标为 A2 Retry：必须报冲突，不静默修正；
8. Failure 与 abnormal interval 相交：Failure fact 保留，stage/异常规则分别校验；
9. 不可判断失败 stage：草稿可保存、COMPLETE seal fail closed；
10. 旧 annotation set 重新编辑：生成新版本，旧版本 hash 和内容不变。

## 10. 推荐落地顺序

1. 已由 Control Station application 接收失败 stage 编辑并确定性重建 occurrence suffix；
2. 已补跨 Attempt Retry 重排和封存验证测试；
3. 已将 GUI 改为失败 stage 选择器，并移除不安全的递进 split 主入口；
4. 待用一条真实失败 Episode 做交互验收；
5. capture path 和正式 schema 保持不变。

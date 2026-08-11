# v2.5-p2 数据格式简明说明

本文面向数据采集、标注、训练和数据检查人员，说明 DexTeleop 当前
**v2.5-p2 长程任务数据集**的基本结构和语义。真机运行参数、标注 GUI 操作和完整合同
分别以对应专题文档为准；本文只回答三个常见问题：

1. v2.5-p2 和已有 v2.3 数据是否兼容；
2. 子任务、尝试、子阶段、失败和恢复分别表示什么；
3. 读取基础轨迹或训练语义时应使用哪一层数据。

需要把一个实际数据集交给其他人时，另见
[数据集内容与交付说明](./DATASET_CONTENT_AND_HANDOFF.md)。该页以
`2026-08-04_dmtac_rm65_bimanual_position_v2_writing_4_stage` 为实例，说明模态、字段、
标注状态、磁盘体量、训练准入和对外交付清单。

## 1. 一句话理解

当前真机 v2.5-p2 不是用新格式替换 v2.3 主表，而是：

```text
v2.3 Parquet / 视频 / Episode 元数据       基础轨迹与传感器数据
                    +
meta/progress/                              v2.5-p2 长程任务语义
                    +
derived/training_views/                     可删除、可重建的训练索引
```

因此，**v2.3 仍是现行真机数据的物理基础层和权威 Episode 发布路径**；v2.5-p2 使用
同一个 Episode identity 和同一条正式 sample 时间轴，追加 Progress Step、Attempt、
Stage、Failure、Recovery 和后标注信息。Progress 功能默认关闭；关闭时不会创建
`meta/progress/`，原 v2.3 路径保持不变。

## 2. 与 v2.3 的兼容性结论

当前兼容性适合渐进迁移，但需要区分“基础数据兼容”和“新语义兼容”。

本项目要求的兼容性实际包含两个独立目标：

1. **混合训练**：v2.5-p2 Episode 忽略 Progress/Stage 语义后，仍能和普通 v2.3
   Episode 一起按 v2.3 基础字段训练；
2. **旧数据升级**：普通 v2.3 Episode 不重写 Parquet/视频，通过显式后期标注和迁移
   receipt 补齐 p2 语义。

当前第 1 项已经具备；第 2 项的数据合同边界已经具备，但端到端 migration adapter 和
旧数据粗进度标注入口**尚未落地**，不能把普通 v2.3 目录直接交给现有 Stage 标注 GUI
并称为完成升级。

| 使用方式 | 兼容性 | 说明 |
|---|---|---|
| 旧 v2.3 reader 读取 Parquet、视频、action、state | 直接兼容 | p2 不重写 v2.3 行，也不改变视频内容 |
| `validate-v23` 检查 p2-enabled 数据集 | 兼容 | 基础层仍须完整通过 v2.3 验证 |
| 旧训练代码忽略 `meta/progress/` | 兼容 | 得到原始完整轨迹，但不会自动避开 Failure/异常区间 |
| 按子任务或子阶段选取数据 | 需要 p2-aware reader | 使用 `dataset_interface` 解析语义区间 |
| 自动避开 Failure、异常区间或整条异常 Episode | 需要 p2-aware reader | 排除结果是可重建派生视图，不写回 Parquet |
| 把全部 p2 语义交给只认识 v2.3 的第三方 loader | 不直接兼容 | 需要显式导出/adapter，不能假设新语义已成为 v2.3 列 |

### 2.1 目标一：忽略阶段，与 v2.3 混合训练

这条路径当前可用。p2-enabled Episode 的 action、state、timestamp、视频等基础内容仍在
v2.3 物理层中；训练代码忽略 Episode metadata 中的 Progress 引用和
`meta/progress/`，即可把它当作普通 v2.3 Episode。

混合前仍须满足一般训练数据兼容条件：

- 使用兼容的 action/state 字段、shape、dtype 和 pose-frame contract；
- task、embodiment 和 policy-I/O 语义确实允许共同训练；
- 分别通过 v2.3 校验，再由训练数据清单显式组合，不能仅复制文件并合并
  `episodes.jsonl`；
- 这条兼容路径不会应用 p2 Failure 或异常排除。若要使用这些语义，应将普通 v2.3
  数据先升级，或者使用支持两类输入的显式训练选择层。

当前 `build_prepared_training_view(...)` 是 p2 严格训练视图，只接收带封存 Progress 和
Stage profile 的 Episode。因此“忽略 p2 与 v2.3 混合”目前应使用原 v2.3 loader/训练
清单，而不是让 prepared view 猜测 profile-less Episode 的语义。

### 2.2 目标二：通过后标注把 v2.3 升级为 v2.5-p2

这条路径设计上可行，并且不需要改动低维主表或视频；但当前还缺少显式的
`v2.3 -> p2` 后期迁移工具。现有保护行为是有意的：

- 采集侧拒绝向已经有 Episode 的 profile-less 数据集直接补绑 Progress profile；
- Stage 标注 GUI 只列出 `episodes.jsonl` 已发布、具有 Progress 引用、Progress manifest
  已封存且带 Stage profile 的 Episode；
- reader 不会根据文件名、task name 或时间猜测缺失的 Step、Attempt、Failure。

正确的后期升级链路应是：

1. 只读验证原 v2.3 Episode、Parquet、视频和正式 sample 时间轴；
2. 为整个数据集显式选择唯一 Progress profile 和 Stage profile；
3. 为旧 Episode 建立稳定 identity，并用源 Episode record/hash 绑定迁移来源；
4. 后标注 coarse Progress Step 边界、Attempt outcome、Failure 归属以及可选 Stage；
5. 将这些信息记录为 retrospective/migrated provenance，不能伪装成采集时在线按钮、
   Hold 或 Pause receipt；
6. 发布不可变 p2 artifact 和 migration receipt，逐项列出已映射、未标注、歧义及排除
   内容；
7. 由 p2 reader 验证 receipt 后，才把该 Episode 纳入 p2 prepared training view。

Stage 可以保持未标注或部分标注，之后继续追加不可变 annotation set；但最低限度的
Progress profile、Episode identity、coarse timeline 和迁移 provenance 必须明确。若一个
旧 v2.3 目录实际混有不同任务/profile，应先按显式清单拆成不同逻辑数据集，不能在同一个
p2 数据集绑定中混装。

已经有离线合同测试覆盖以下关键边界：

- 真实 v2.3 writer 输出和 p2 Progress artifact 共用同一个 Episode identity；
- 启用 p2 后基础数据仍通过 v2.3 validator；
- 关闭 p2 时不产生 Progress metadata 或目录；
- provisional native component stream 到 v2.3 的投影要求唯一、显式的字段绑定，歧义或
  缺失会 fail closed，未投影流必须出现在 omission receipt 中。

### 2.3 仍需注意的边界

- 纯 v2.3 reader 看不到 Progress Step、Stage、Failure 和异常区间；这不是数据丢失，
  而是它只读取基础层。
- Stage 标注和训练排除区间不会被复制成逐帧布尔列。训练端应读取派生的 sample interval
  或 prepared training view，避免重写大体量 Parquet。
- 同一 Episode 可以有多个不可变 annotation set。存在多版时必须显式选择一版，系统
  不会用“目录中最新的一版”静默覆盖训练语义。
- 一个正式数据集目录只能绑定一种 Progress profile 和一种 Stage profile，也不能混装
  profile-less 与 profiled Episode。日期目录可以在整理阶段组成逻辑训练集合，但所选
  数据必须具有相同的 profile hash 和兼容的物理 schema。
- `meta/v25p/native/` 下的 component-native shadow 是另一条仍属 provisional、默认关闭的
  路径。它不等于本文的 p2 Progress 语义层，也不是当前 v2.3 真机主表的替代品。

## 3. 核心语义

### 3.1 Task Progress Profile：整条任务的进度合同

一个数据集只绑定一个 `TaskProgressProfile`。它定义有序的粗粒度
**Progress Step（面向数采员可显示为“子任务”）**、合法后继和终止节点。

Progress Step 是 VLM 能够区分的合理语义粒度，也是后续“一个具体 VLA 负责一个子任务”
的候选边界。例如：

```text
右手拿取毛笔 → 左手取下笔盖 → 右手书写 → 右手放回毛笔
```

每个 Step 有唯一 `progress_step_id`。显示名称或 `semantic_type` 可以重复，身份不能靠
显示字符串推断。合同支持 LINEAR 和有限 ROOTED_TREE；当前 RealTeleop 真机入口只准入
LINEAR profile。

### 3.2 Attempt：同一子任务的一次尝试

Attempt 是一个 Progress Step 的一次实际执行：

- 正常 Advance：当前 Attempt 隐式成功，进入下一个 Step 并创建新 Attempt；
- Failure：当前 Attempt 立即结束为失败；
- 人工调整期间没有 active Attempt，也不写正式训练帧；
- Resume 生效并恢复正式采样后，在同一个 Step 创建新的 Retry Attempt。

因此，同一个 Step 可以出现多个 Attempt，例如先失败、后恢复成功；不能仅凭 Step ID
把它们合并。

### 3.3 Stage：子任务内部的有序执行阶段

Stage 是 Progress Step 内更细的执行粒度，用于非 0/1 progress、失败归属和 recovery
入口。它主要由采集后的细粒度标注补齐，而不是要求数采员在线逐阶段点击。

例如“右手拿取毛笔”可以包含：

```text
A1 靠近毛笔 → A2 正确抓住毛笔 → A3 提起并送至左手
```

Stage 遵循以下规则：

- 在同一 Step 内严格按 profile 顺序推进；
- 已成功的 Stage 不得倒退或在同一条顺序中重复成功；
- 某个 Stage 可以失败多次，Retry 必须继续这个失败 Stage；
- Episode 可以在开头从中间 Stage 进入，但必须在采集时记录
  `entry_condition=RECOVERY_FROM_FAILURE`；
- Episode 也可以在中间结束，此时尾部是截断窗口，不强行补成完整成功任务。

一个典型时间线是：

```text
Attempt a1: A1 SUCCESS, A2 FAILURE
Attempt a2:             A2 SUCCESS, A3 FAILURE
Attempt a3:                         A3 SUCCESS
```

这三个 Attempt 合起来完成了 A1、A2、A3 的完整、有序且不重复的成功序列，同时保留
两次失败发生在哪个 Stage 的事实。

### 3.4 Failure、failure trigger 与训练排除

采集时的 Failure 是不可变原始事实，默认记录数采员报告时间，并触发 Hold 和正式采样
暂停。它不会因为后标注而被删除。

后标注可以进一步给出 `failure_trigger`，表示失败行为开始的正式 sample：

```text
failed Stage occurrence 的起点
        → failure trigger：可视为失败前仍正常的边界
        → Failure fact：数采员报告并结束 Attempt
```

默认训练视图会排除失败相关区间并在此处截断 chunk。没有可用的 trigger 时，使用已解析
的失败 Stage 区间进行保守排除；不能定位到 Stage 时再按更粗粒度 fail closed。原始
Failure、Stage occurrence 和排除策略是三层不同信息，loader 不应把它们折叠成一个
布尔字段。

### 3.5 Recovery

Recovery 不是覆盖原失败 Attempt，而是新的采集窗口或 Retry Attempt。若一条 Episode
从失败 Stage 开始补采，必须保存准确的 `entry_step_id`、`entry_stage_id` 和
`RECOVERY_FROM_FAILURE`。这样训练端可以：

- 只训练正常成功片段；
- 从 Failure/recovery anchor 开始采样；
- 单独构建 recovery 数据集；
- 保持 failure 前后的因果边界和 Attempt 边界。

### 3.6 异常与后标注

后标注不修改 Parquet、视频或第一次采集事实，而是生成不可变 annotation set。当前可
表达：

- 有序 Stage occurrence 和边界；
- Failure 所属 Stage；
- 局部异常区间；
- 整条 Episode 异常或需要二次复核；
- 特权 coarse Progress 边界修正；
- capture Failure 的可选 failure trigger。

异常区间等价于训练时间线中的断点：区间本身不进入默认训练，前后可分别形成合法样本
窗口。多版 annotation set 永久并存，训练发布时必须显式选择。

## 4. 目录结构

典型 p2-enabled 数据集保留 v2.3 基础目录，并增加 `meta/progress/`：

```text
dataset_root/
  data/chunk-000/episode_000000.parquet        # v2.3 低维主表
  videos/observation.images.front/...mp4       # v2.3 视频
  meta/
    info.json
    episodes.jsonl                             # v2.3 Episode 发布记录
    episodes_stats.jsonl
    progress/
      profile.cjson                            # 数据集唯一 Progress profile
      stages.cjson                             # 数据集唯一 Stage profile（可选）
      episodes/<episode_uuid>/
        profile.cjson
        binding.cjson
        stages.cjson                           # Stage profile（若启用）
        stage_entry.cjson                      # 本 Episode 的 Stage 入口
        facts.cjsonl                           # Advance/Undo/Failure/Receipt/Retry/结束
        failure_trigger_capture/               # capture-time 默认 trigger receipt（可选）
        manifest.cjson                         # 最后发布，表示 Progress 已封存
      stage_annotation_drafts/<episode_uuid>/  # 可修改草稿，不是合同真源
      stage_annotations/<episode_uuid>/<set_id>/
        binding.cjson
        annotations.cjson
        failure_triggers.cjson                 # 可选
        manifest.cjson                         # 不可变标注集发布点
  derived/training_views/<view_hash>/          # 可删除、可重建的逻辑训练视图
```

`manifest.cjson` 已存在只表示对应 artifact 已闭合：Progress manifest 和 Stage
annotation manifest 是两次不同的封存。一个 Episode 显示“Progress 已封存”并不表示它
已经完成二次 Stage 标注。

## 5. 应该怎样读取

### 5.1 只需要原始轨迹

继续使用现有 v2.3 reader 读取 `meta/episodes.jsonl`、Episode Parquet 和视频即可。正式
验收命令是只读的：

```bash
python -m utils.dataset_postprocess validate-v23 /absolute/path/to/dataset
```

这种方式不会应用 Stage、Failure 或异常排除，适合兼容检查、回放或尚未迁移的训练代码。

### 5.2 需要长程任务语义

使用 `dataset_interface` 的只读接口：

- `load_dataset_progress_profile(...)`：读取数据集唯一 Progress profile；
- `load_dataset_task_stage_profile(...)`：读取数据集唯一 Stage profile；
- `load_sealed_progress_episode(...)`：验证单个 Episode 的 profile、binding、facts 和 seal；
- `list_progress_annotation_work_items(...)`：列出已发布且可标注的 Episode；
- `resolve_stage_training_index(...)`：将 Stage occurrence 映射到正式 sample axis；
- `build_episode_training_semantic_index(...)`：构建 Step/Stage/Failure/异常的区间和 anchor。

这些接口只解析和验证，不修改源数据。

### 5.3 整理多个日期数据集供训练

`build_prepared_training_view(...)` 可以把多个不可变日期数据集组织成一个逻辑训练视图。
它不会复制或重写原始 Parquet/视频，只生成：

```text
episodes.parquet            Episode 与源文件索引
semantic_intervals.parquet  Step/Stage/排除区间
semantic_anchors.parquet    Failure、Recovery 等采样起点
source_inventory.cjsonl     源文件及内容校验
manifest.cjson              profile、策略和输入 hash
```

默认构建仍要求所有选中数据具有同一个 Progress profile、同一个 Stage profile 和兼容的
Parquet projection schema；一个 Episode 有多版 annotation set 时必须通过
`annotation_selection` 显式指定。这使 loader 可以按子任务、Stage、正常/失败/Recovery
anchor 高效筛选，同时保持 v2.3 源数据可独立搬运和验证。

如果要从不同任务 Profile 中抽取经过审核的同一个可训练子任务，可以显式提供
`SkillProjectionCatalog + target_skill_id`。该模式只生成额外的
`skill_intervals.parquet` 和 `skill_projection_receipts.cjsonl`，不会改变源 Progress、
Stage、Parquet、视频或标注：

```python
from dataset_interface import (
    PreparedSourceDataset,
    build_prepared_training_view,
    load_builtin_writing_skill_projection_catalog,
)

result = build_prepared_training_view(
    organized_dataset_root,
    (
        PreparedSourceDataset("writing-3", writing_3_root),
        PreparedSourceDataset("writing-4", writing_4_root),
    ),
    skill_projection_catalog=(
        load_builtin_writing_skill_projection_catalog()
    ),
    target_skill_id="writing.write_with_brush",
)
```

这不是按名称模糊匹配：每条 projection 精确绑定源 Progress/Stage profile hash、源 Step、
有序 Stage、受控组件、入口/成功谓词和必要上下文。任一条件缺失都会 fail closed；失败、
异常、Pause 和 Attempt 边界也不会被跨越。当前只构建一个 `target_skill_id` 的训练视图，
不把多个 Skill 拼成一条伪造的连续 Episode。

Writing 内置映射明确区分：

- 3-step 的 `lift_ready` 对应“进入直接书写准备”；
- 4/5-step 的 `lift_and_present` 对应“送至左手取笔盖”。

二者不是同一个完整拿笔 Skill；只有共同的 `approach + grasp` 前缀可以投影到公共 Skill。
`right_write` 和 `right_return_brush` 可以跨 3/4/5-step 复用，其中不同前序任务状态作为
显式条件上下文保留。

## 6. 数据维护规则

- 不要手工编辑 `.cjson`、`.cjsonl` 或 seal manifest；canonical hash 会因此失效。
- 不要为了加标签重写 Parquet 或视频；标注写独立 annotation set，训练选择写派生视图。
- 不要把两个不同 profile 的日期目录直接视为同一个训练集合；默认先按 profile hash
  分组。只有存在经过审核且精确 hash 绑定的 Skill projection 时，才能构建单子任务的
  跨 Profile 派生视图。
- 不要用目录修改时间自动选择 annotation set；把选择写入可复现的训练配置或 prepared
  view manifest。
- 搬运数据时应整体保留 v2.3 基础文件和 `meta/progress/`。只复制 Parquet 会丢失视频与
  p2 语义，只复制 `meta/progress/` 则没有对应 sample axis。

## 7. 相关文档

- 数据集内容、实例数据卡与交付清单：
  [`DATASET_CONTENT_AND_HANDOFF.md`](./DATASET_CONTENT_AND_HANDOFF.md)
- 真机采集与 Progress Profile：
  [`TELEOP_COLLECTION.md`](../../real/TELEOP_COLLECTION.md)
- 二次 Stage 标注 GUI：
  [`LONG_HORIZON_STAGE_ANNOTATION.md`](../../real/LONG_HORIZON_STAGE_ANNOTATION.md)
- 数据后处理与测试：
  [`POSTPROCESS_AND_TESTING.md`](../POSTPROCESS_AND_TESTING.md)
- v2.3 验收：
  [`ACCEPTANCE_V23.md`](../ACCEPTANCE_V23.md)
- 纯合同与 reader 边界：
  [`dataset_contracts/README.md`](../../../dataset_contracts/README.md)、
  [`dataset_interface/README.md`](../../../dataset_interface/README.md)

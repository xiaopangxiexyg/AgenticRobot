# DexTeleop 可迁移训练预处理方案

## 1. 目标与结论

在不改写 canonical capture、视频、Parquet、Progress facts 和后标注事实的前提下，
为一个已封存的数据集生成小型、可删除、可完整重建的 `prepared training view`。

该 view 解决四个问题：

1. 数据集移动到另一台机器后，所有引用仍然有效；
2. loader 不必在每个 worker/epoch 重扫大量 Progress/annotation 小文件；
3. 子任务、阶段、Failure、异常和 Recovery 选择可以直接做表式查询；
4. view 精确绑定 source manifests、annotation set 和 resolver policy，旧缓存不会被
   新标注静默复用。

0802 writing 数据只读实测表明，当前核心数值列、视频和语义 resolver 已经足够快，
不需要提前重写物理数据。最有价值的是元数据归一化、派生区间物化和路径可迁移性检查。

## 2. 所有权与边界

- `dataset_contracts`：只在至少两个稳定消费者证明需要后，才冻结 prepared-view 的
  最小 manifest 合同；第一版可以保持 provisional。
- `dataset_interface`：拥有只读预处理、验证、派生表构建和 prepared-view reader。
- `collection_runtime`：不生成训练 view，不解释训练 eligibility。
- `teleop_runtime`、`control_station`：不参与训练预处理。
- canonical Episode owner、capture facts 和 annotation artifacts 永不被该流程改写。

prepared view 是训练侧派生物，不是新的数据真源；删除它以后必须能从 source dataset
得到完全相同的逻辑结果。

## 3. 推荐目录

```text
<dataset_root>/
  data/
  videos/
  meta/
  ... canonical capture and annotation artifacts ...

  derived/
    training_views/
      <training_view_hash>/
        manifest.cjson
        episodes.parquet
        semantic_intervals.parquet
        semantic_anchors.parquet
        source_inventory.cjsonl
```

约束：

- `derived/` 不参与 canonical dataset identity；
- 所有 locator 必须是相对 dataset root 的 POSIX 路径；
- 不允许保存绝对路径、主机名、Conda 路径或临时 mount point；
- 不原地更新同一个 view；输入或 policy 改变时创建新的 `<training_view_hash>`；
- view 完整写入并校验后，最后原子发布 `manifest.cjson`；
- loader 只接受 manifest 已发布且所有 hash/row count 一致的 view。

如果不希望把派生物放进数据集，也可以使用 caller-supplied cache root；其逻辑内容与
dataset-local view 完全一致。dataset-local 形式更方便整体搬运，外部 cache 更适合共享
只读数据集。

### 3.1 采集期间的增量发布

prepared view 不必等待整个数据集停止采集。推荐采用“sealed-Episode watermark +
immutable snapshot”模型：

```text
active Episode
  -> 不进入 prepared view

Episode seal + Episode catalog commit
  -> 生成/验证该 Episode 的 immutable semantic object
  -> 发布一个引用截至当前 watermark 的新 snapshot manifest

新的 Episode 或 annotation set
  -> 生成新 object / 新 snapshot
  -> 不修改已经发布的旧 snapshot
```

必须区分两层：

1. **live build cache**：可以随采集增量更新，允许使用可删除 SQLite/WAL 或 staging
   文件；它不是可迁移真源，也不能被训练任务当作已提交 view；
2. **published prepared snapshot**：只引用 sealed Episode 和 sealed annotation，
   manifest 最后发布，发布后不可变。

数据集仍在增长时，可以周期性、按 Episode batch 或训练启动请求发布 snapshot。
训练任务绑定具体 snapshot hash 和 Episode watermark，因此同一次训练运行看到稳定数据，
不会因采集端新增 Episode 而在 epoch 中途变化。

后标注同理：未标注 Episode 可以在某个 snapshot 中明确记录为 `UNANNOTATED`；标注员
提交新的 sealed annotation set 后，生成新的 Episode semantic object 和新 snapshot。
旧 snapshot 继续保持原解释，不会被“最新标注”静默改变。

整个数据集停止采集后可以生成 final snapshot，并将 episode-level 小 object compact 为
较少的 Parquet 文件以方便搬运和顺序读取。该 final compact 是物理优化，不是语义封存
的唯一时机；逻辑结果必须与相同 watermark/source hashes 的增量 snapshot 完全等价。

**当前项目决策**：首版不接入采集热路径，也不在每个 Episode seal 后自动运行。
prepared view 只在数据整理、筛选、合并或训练发布阶段显式生成。上述 watermark 和
immutable snapshot 规则仍保留，用于保证以后若开放增量整理，不需要修改 artifact
语义。首版也不需要 live build daemon；SQLite/WAL 仅作为整理工具运行期间的临时缓存。

### 3.2 按日期采集数据的逻辑合并

当前数据按日期形成独立 dataset root，prepared view 应以 Episode identity 为主键进行
逻辑合并，不应通过拼接 `episode_index`、重命名源文件或复制媒体来实现合并。

推荐整理后的逻辑结构：

```text
<organized_dataset_root>/
  meta/
    source_datasets.cjson
    selected_episodes.cjsonl
  derived/
    training_views/<training_view_hash>/...
```

`source_datasets.cjson` 为每个日期数据集分配稳定 `source_dataset_id`，绑定：

```text
source dataset identity/fingerprint
source root locator
progress/stage profile hashes
available modality/capability summary
```

`selected_episodes.cjsonl` 每行只引用：

```text
source_dataset_id
episode_capture_identity_hash
source episode index（仅 provenance）
selected annotation manifest hash
include/exclude decision and reason
```

合并规则：

- `episode_capture_identity_hash` 是合并主键；各日期目录中的 `episode_000000` 允许重复，
  不作为全局身份；
- 同一 Episode identity 且 source/content hashes 完全一致时可以逻辑去重；
- 同一 Episode identity 指向不同内容时 fail closed；
- 合并后的连续 ordinal、global sample offset 和 train/validation split 都是派生字段，
  不回写日期数据集；
- stats、eligible intervals 和样本数量从选中的 Episode 重算，不能直接相加旧 stats；
- annotation set 逐 Episode 显式绑定，不使用各 source 目录中的“最新标注”；
- 一个 organized dataset 继续只允许一个精确 Task Progress Profile 和一个精确 Stage
  Profile；Profile hash 不同的日期数据默认不能混装；
- 动作/观测物理 schema 可以不同，但必须被同一个显式 training projection 无损或按
  合同允许地投影；不能投影的 source 必须拆分为另一个 organized dataset；
- resolver policy、chunk contract 和 modality selection 属于 prepared view，不属于
  source Episode。

工作区逻辑合并允许 source locator 指向多个只读日期目录，因此不复制数百 GiB 媒体。
若要把合并结果交付到另一台机器，则另做 portable materialization：按 inventory 复制
被选择的 source files 到一个自包含 bundle，校验相对路径、文件大小、content hash 和
尾部解码后再发布。逻辑合并与物理搬运必须是两个明确步骤。

不建议用外部符号链接伪装成自包含 release。工作区 locator 可以是受验证的 source-root
映射；正式 portable bundle 内的所有路径必须解析在 bundle root 内。

## 4. 最小派生物

### 4.1 `manifest.cjson`

只保存重建和验证所需的最小信息：

```text
training_view_schema
training_view_hash
source_dataset_identity/fingerprint
source_episode_catalog_hash
selected_progress_profile_hash
selected_stage_profile_hash
selected_annotation_manifest_hash_by_episode
selected_failure_trigger_manifest_hash_by_episode
resolver_policy_hash
modality_selection
column_projection
episode_count
interval_count
anchor_count
derived_file hashes
build implementation/version
```

`column_projection` 应使用逻辑字段名，并验证 dtype/shape；它不是第二份 embodiment 或
task profile。annotation set 必须由调用者显式选择，或满足“唯一可用 set”的明确策略；
禁止根据目录名猜测 `latest`。

### 4.2 `episodes.parquet`

一行一个 Episode，只保存 loader 调度所需字段：

```text
episode identity/index
frame count
sample clock/domain
start/end timestamp
Parquet relative path
camera video relative paths
progress manifest hash
annotation manifest hash
profile hashes
capability/completeness status
```

不内嵌逐帧 metadata，不复制动作/状态数组，不复制视频信息字典。

### 4.3 `semantic_intervals.parquet`

一行一个半开 sample interval：

```text
episode identity/index
sample_begin
sample_end_exclusive
semantic_kind
training_domain
progress_step_id
attempt_id
stage_id
outcome
entry_condition
source fact/annotation identity
resolution kind
```

它物化 `dataset_interface.training_semantics` 的结果。正常区间允许跨成功阶段和
Progress Step；Failure、异常、Pause 和 continuity boundary 已经切断。

只物化 interval，不预先展开成每帧 label，也不为每个 action horizon 生成所有窗口。
loader 根据 chunk contract 从 interval 计算合法 start 和 padding/drop，因此不把训练
超参数固化进数据。

### 4.4 `semantic_anchors.parquet`

一行一个稀疏事件锚点，例如：

```text
PROGRESS_STEP_BEGIN
STAGE_BEGIN
FAILURE_REGION_BEGIN
FAILURE_TRIGGER
FAILURE_END
RECOVERY_BEGIN
```

用于从 Failure、Recovery 或未来新标签出发采样。未来增加 contact/preferences 时，
优先投影为新的 typed interval/anchor，而不是修改旧帧表。

### 4.5 `source_inventory.cjsonl`

用于迁移前检查：

```text
relative path
logical role
file size
required/optional
content hash or bound canonical manifest hash
```

第一版可以只对 metadata、Parquet 和本次启用的视频计算强 hash。未进入训练的 tactile
可以记录 `OMITTED_BY_MODALITY_SELECTION`，不必为 66 GiB 内容重复做昂贵 hash。

必须检测并拒绝：

- 指向 dataset root 外部的符号链接；
- 绝对路径 locator；
- 丢失的视频/Parquet；
- profile 混装；
- annotation 与 Progress manifest 不一致；
- source path 大小或 hash 在 view 发布后发生变化。

## 5. 不建议提前做的处理

以下处理暂时收益不足或会破坏回退性：

- 不把子任务、失败或异常布尔值写回原始 Parquet 每一帧；
- 不重写现有 89 个 Episode Parquet；
- 不转码 MP4；
- 不复制一份 core action/state 数据；
- 不提前生成每种 horizon 的全部 window-start 表；
- 不把 padding 后的帧保存下来；
- 不使用 Pickle 或 Python 对象作为可迁移格式；
- 不让 SQLite 成为唯一可读 artifact；
- 不把“最新 annotation set”作为隐式语义；
- 不将 derived view 提升为 capture truth。

SQLite 可以作为本机可删除查询缓存，但 portable artifact 应使用 CJSON + Parquet，
以便 Python、C++、Rust、LeRobot 和其他训练框架直接读取。

## 6. Loader 预期路径

```text
打开 prepared manifest
  -> 校验 source 与 annotation hashes
  -> 一次读取 episodes/intervals/anchors
  -> 按查询筛选 Episode 与合法 interval
  -> 按 Episode 加载严格裁剪的数值列
  -> 复用每 worker 的视频 decoder
  -> 按 chunk contract 生成 window 与 padding mask
```

核心 Parquet projection 默认只包含模型实际使用的 action/state/qpos/pose/timestamp。
`sample_metadata_json`、设备诊断和完整 causal debug 字段只有显式请求时才加载。

## 7. 0802 数据对应收益

当前 89-Episode 数据实测：

- 核心 11 列全量读取：约 0.53 秒、18.9 MiB；
- work-item discovery：约 0.14 秒；
- sample-axis 构建：约 0.50 秒；
- 77 个后标注 Episode 的语义索引：约 0.065 秒；
- Progress/annotation 小文件约 945 个；
- 全部语义准备约 0.7 秒。

prepared view 不会显著改变核心列或视频解码速度；它的主要收益是：

- worker/epoch 不再重复打开数百个语义小文件；
- 数据集搬迁后不需要修复路径；
- 训练选择具有精确 annotation lineage；
- 新标注只产生新 view，不重写 raw 数据；
- 子任务/Failure/Recovery 查询可直接做列式过滤。

## 8. 分阶段落地

### P0：只读预检与 provisional view

1. 在 `dataset_interface` 增加 dataset-root preflight；
2. 显式选择 annotation set；
3. 支持一个或多个日期 source root 的逻辑 Episode selection；
4. 生成 episodes、semantic intervals、anchors；
5. 生成 source inventory 和 provisional manifest；
6. snapshot 只纳入 Episode catalog 中已经提交且 Progress/annotation 已 sealed 的输入；
7. 用 0802 数据验证原 resolver 与 prepared view 查询结果逐项等价；
8. 模拟合并两个日期 root，验证重复 `episode_index` 不冲突、Episode identity 去重和
   Profile/hash 不兼容时 fail closed；
9. 模拟追加新 Episode/新 annotation，确认旧 snapshot 不变且新 snapshot 只增加或替换
   显式绑定的 Episode semantic object；
10. 删除 view 后重建，要求内容 hash 完全一致。

#### 当前落地状态（2026-08-02）

已在 `dataset_interface.prepared_training` 子域完成首个 provisional 实现，并通过
`dataset_interface.prepared_training_view` 保留稳定门面：

- builder 只接收 caller 显式给出的一个或多个 source root，不接采集热路径；
- 只选择 `episodes.jsonl` 已提交且 Progress 已 sealed 的 Episode；
- 按 Episode capture identity 合并，`episode_index` 只保留为 provenance；
- 一致内容允许去重，同 identity 的 raw/annotation/trigger 内容不一致则 fail closed；
- 强制一个 Progress Profile、一个 Stage Profile 和一个一致的 projected Parquet schema；
- annotation set 唯一时可采用，存在多个 set 时必须显式按 Episode 选择；
- 生成 `episodes.parquet`、`semantic_intervals.parquet`、
  `semantic_anchors.parquet`、source selection 和强 hash inventory；
- source root 与 source file locator 均为相对路径；workspace view 可以引用 root 外的日期
  source，portable release 可用 `require_self_contained=True` 额外拒绝这种布局；
- private staging 完整写入、校验和 fsync 后以 view hash 目录原子发布；重复构建幂等；
- strict reader 校验 manifest、派生表 schema/hash/row count、selection 交叉绑定，并可选
  重算全部 source file hash；
- Stage annotation 的纯 reader 已从 `collection_runtime` writer 边界拆到
  `dataset_interface`，原 import 路径保留兼容转发。

离线 fixture 已覆盖两日期 root、重复 source `episode_index`、已标注/未标注混合、多个
annotation set 的显式选择、profile 冲突、同 identity 精确去重/内容冲突、源文件零修改、
非自包含识别、派生表损坏拒绝和重复构建。尚未完成的是在真实 0802 全量数据上实际发布
prepared view 并逐项比较 direct resolver；该步骤仍保持只读 source、写入独立 organized
root，待明确运行位置后执行。

### P1：训练 loader 接入

1. dataset-root reader 优先读取合法 prepared view；
2. 缺失 view 时允许直接 resolver fallback；
3. 实现 Episode 数值列 cache 和 per-worker video decoder cache；
4. 实现子任务、Failure、Recovery、异常排除和 chunk contract 查询；
5. 对比 fallback/prepared 两条路径的样本 identity、mask 和 payload 完全一致。

### P2：稳定后再冻结合同

只有当至少训练 loader 和导出/审计工具两个消费者稳定使用后，再决定：

- 是否冻结统一的 thin resolved-view manifest；
- 是否将 source inventory 合并进 dataset release manifest；
- 是否提供显式 LeRobot v2.1 materialization/export receipt；
- 是否将大量小型 annotation source artifacts 额外打包为传输归档。

## 9. 验收条件

- prepared view 的生成不修改任何 source inode/content；
- active/unsealed Episode 永远不进入 published snapshot；
- 新 Episode 或新 annotation 不得原地修改已经发布的 snapshot；
- 同一次训练运行绑定固定 snapshot hash/watermark；
- 数据集整体移动或换 mount root 后无需改写 view；
- source/annotation/policy 任一 hash 改变时旧 view fail closed；
- direct resolver 与 prepared view 的 interval/anchor 完全等价；
- 相同输入重复生成得到相同逻辑 hash；
- 删除 derived 目录后 canonical 数据仍完整可读；
- 默认 loader 不读取 `sample_metadata_json`；
- 未启用 tactile 时，不打开 tactile manifest 或 binary stream；
- padding/drop 仍由 chunk contract 决定，不固化在预处理结果中。

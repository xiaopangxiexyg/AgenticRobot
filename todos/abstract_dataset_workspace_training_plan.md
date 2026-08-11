# 抽象数据集 Workspace、视角约束与混训架构方案

> 状态：架构规划，待按阶段实施
>
> 日期：2026-08-04
>
> 适用范围：v2.3 retrospective 数据、v2.5p2 native 数据、单子任务数据、4/5-stage
> 全流程数据、front-only/局部视角/三视角训练，以及后续频繁增加或停用日期数据集。

## 1. 结论

建议采用用户提出的总体方向，并做两点边界修正：

1. Stage annotation、retrospective Progress、Episode capability 等事实应逻辑归属于对应源
   数据集，并以追加式 sidecar 放在数据集内；不能改写 v2.3 Parquet、视频、既有
   `episodes.jsonl` 或把 retrospective 伪装成 native capture facts。
2. 源数据集只声明“有什么”和“已经验证了什么”；“三视角训练要求三路 RGB”“只训练
   writing 第一子任务”“哪些来源参与本次训练”等选择性约束由上层
   `AbstractDatasetProfile` 声明，workspace 编译器负责 fail-closed 检查。

训练程序不直接管理日期目录，不扫描“最新标注”，也不在运行时推断相机兼容性。训练端只
读取一个或多个已冻结的 `AbstractDatasetSnapshot`；多个 snapshot 的混合比例由独立
`TrainingMixtureProfile` 管理。

## 2. 总体分层

```text
┌───────────────────────────────────────────────────────────────┐
│ Source dataset：0803、0804、v2.3、v2.5p2 ...                  │
│ raw Parquet/video + dataset-local semantic sidecars           │
│ 事实：identity、schema、camera capability、Progress、Stage    │
└──────────────────────────────┬────────────────────────────────┘
                               │ dataset_id + immutable hashes
┌──────────────────────────────▼────────────────────────────────┐
│ Dataset Workspace：管理面                                     │
│ source registry + locator bindings + profiles + build reports │
│ 可注册/停用数据集，不能拥有或改写 raw/annotation truth         │
└──────────────────────────────┬────────────────────────────────┘
                               │ compile + validate
┌──────────────────────────────▼────────────────────────────────┐
│ Abstract Dataset Snapshot：不可变训练输入快照                  │
│ exact Episodes/intervals/modalities/annotations/splits/hashes │
└──────────────────────────────┬────────────────────────────────┘
                               │ one or more compatible snapshots
┌──────────────────────────────▼────────────────────────────────┐
│ Training Mixture：训练配置                                    │
│ snapshot hash + sampling weight + batching policy             │
└───────────────────────────────────────────────────────────────┘
```

一个 workspace 可以管理多个源数据集和多个抽象数据集 profile。一个抽象数据集由一个
versioned profile 管理；每次解析 profile 都产生一个新的不可变 snapshot。profile 表示意图，
snapshot 表示当时实际解析出的精确成员。

### 2.1 当前 writing 标注功能的严格边界

本轮新增的 retrospective writing 标注能力只服务于 `dexteleop_dataset_v2.3`：

- 输入是一个明确的 v2.3 source dataset root 和该数据集自己的严格 Episode selection；
- 只为这些 v2.3 Episodes 生成 `RETROSPECTIVE_MIGRATED` one-step window 与 writing 第一子任务
  三 Stage；
- migration、draft、sealed annotation set 和 capability index 都写入该数据集自己的 named
  semantic layer；
- 不负责注册其他日期数据集、跨数据集浏览、抽象数据集组合、相机训练约束、split、sampling
  weight 或 prepared-view 发布；
- 不修改 v2.3 raw Parquet、视频、既有 `info.json`/`episodes.jsonl`，也不声称生成 native
  v2.5p2 capture facts。

这个限制只约束 writing v2.3 retrospective adapter。既有通用 refinement-annotation GUI 仍可
服务带 native Progress/Stage 的 v2.5p2 数据；共享 GUI 只提供时间轴编辑和封存交互，不拥有
v2.3 migration 解释。上层 workspace/compiler 只读取已经完成的数据集本地语义，不参与人工
标注生命周期。

## 3. 源数据集内的 semantic sidecar

### 3.1 推荐布局

避免假设一个物理数据集只能拥有一个 Progress/Stage profile。对 v2.3 retrospective 或未来
其他补标语义，使用数据集内的 namespaced semantic layer：

```text
<source-dataset>/
  meta/info.json
  meta/episodes.jsonl
  data/...
  videos/...
  semantic_layers/
    index.cjson
    writing_stage1_retrospective_v1/
      profile.cjson
      stages.cjson
      episodes/<episode_identity_hash>/
        source_binding.cjson
        progress_binding.cjson
        retrospective_window.cjson
        migration_manifest.cjson
      meta/progress/stage_annotation_drafts/...
      meta/progress/stage_annotations/...
      capability_index.cjsonl
```

对于 native v2.5p2 capture Progress，既有 `meta/progress` 是隐式 native semantic layer，不
复制 facts。统一 reader 将 native layer 和 namespaced retrospective layer 都投影为相同的
`SemanticEpisodeSource` 接口。

### 3.2 完整性边界

- raw fingerprint 只覆盖原始 schema、catalog、Parquet、视频及 capture receipts；添加 semantic
  sidecar 不改变 raw identity。
- 每个 semantic layer 有独立 layer hash、profile/stage hash、Episode migration/annotation hash。
- sidecar 只能追加或发布新 immutable set；不能覆盖旧 annotation set。
- “移除数据”不删除 sidecar，只在上层 profile 中 exclude/disable。
- 写入前验证目标不是 hardlink raw 副本；任何视频变换仍遵守新 inode、帧数和 tail decode 门。
- 若 source dataset 必须保持操作系统只读，可使用 colocated overlay 路径，但 registry 必须把
  overlay 明确绑定到同一 `dataset_id + raw_fingerprint`；语义上仍归属于该数据集，不能成为
  一个无归属的训练 workspace。

## 4. 数据集事实与训练约束的区别

### 4.1 数据集记录事实 capability

每个 Episode 发布 `EpisodeCapabilityV1`，至少包括：

```text
episode_identity_hash
raw_dataset_id / raw_fingerprint
available_camera_slots
camera_role_binding                 # source key -> canonical role
declared/decoded/aligned frame count
numeric field schema hashes
action/state layout hashes
Progress profile/stage profile hashes
available Skill projection ids
sealed annotation-set ids/hashes
RGB/mask/tactile readiness
quality/review/abnormal status
```

例如 front-only Episode 的事实是：

```text
available_camera_slots = [base_0_rgb]
```

三视角 Episode 的事实是：

```text
available_camera_slots = [base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb]
```

不要在 Episode 内硬编码 `three_view_training_allowed=false`。它没有缺陷，只是不满足某个训练
profile 的输入要求。

### 4.2 抽象数据集 profile 声明约束

`AbstractDatasetProfileV1` 建议包含：

```text
abstract_dataset_id / version
source_dataset_selectors             # 引用稳定 dataset_id，不写绝对路径
episode include/exclude/query
target_skill_id + version/hash
accepted Progress-to-Skill projections
semantic interval selection          # 整个 Skill、指定 Step 或 Stage subset
required/optional/forbidden modalities
canonical camera input signature
missing-modality policy
numeric/action/state schema contract
annotation selection policy
quality/review/abnormal gates
dedup policy
split policy
optional per-source sampling metadata
```

profile 必须把相机约束写成可机械检查的 input signature。例如：

```text
writing_stage1_front_v1:
  required_camera_slots = [base_0_rgb]
  model_camera_order = [base_0_rgb]
  missing_modality_policy = REJECT

writing_stage1_three_view_v1:
  required_camera_slots =
    [base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb]
  model_camera_order =
    [base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb]
  missing_modality_policy = REJECT
```

因此 three-view profile 自动拒绝 front-only Episode。不能用零图、重复 front 或静默缺失 wrist
来凑三视角。若未来模型明确支持缺失模态，必须定义新的 input contract，包含 availability mask、
mask 进入模型的语义以及相应训练验证；它不是 three-view profile 的隐式宽松模式。

## 5. 单子任务与全流程数据如何合并

### 5.1 单独采集的第一子任务

- v2.3 writing 数据在自己的 source dataset 内拥有
  `writing_stage1_retrospective_v1` semantic layer；
- 一个 migrated Attempt 覆盖完整 Episode；
- 三个 Stage 和 annotation set 也留在该数据集 layer 内；
- 通过 dedicated one-step profile 投影到共享 writing Stage-1 Skill v2。

### 5.2 4-stage/5-stage 全流程数据

- 保留原生完整 Progress profile、Attempts、Stage annotation；
- 不复制或截断 raw Episode；
- abstract profile 使用显式 Progress-to-Skill projection，仅选择
  `right_acquire_brush` 对应的 accepted sample intervals；
- action horizon 不得越过该 Skill 的 interval 尾进入取笔盖步骤；Stage 内部正常边界是否允许
  跨越继续由训练 window contract 决定。

### 5.3 合并门槛

来自 one-step、4-stage、5-stage 的 Episode 只有同时满足以下条件才进入同一 snapshot：

- 投影到相同 Skill contract hash；
- action/state/component schema 可相等或存在显式受验证 projection；
- 相机 input signature 满足当前 abstract profile；
- annotation selection 唯一且 sealed；
- sample interval 完整、无 review-pending/不允许异常；
- global Episode identity 无冲突，重复内容按 dedup policy 只保留一个逻辑成员；
- split assignment 由 global identity/采集 block 决定，不因 source/profile 改变。

## 6. Workspace 与稳定 locator

### 6.1 Source registry

workspace 保存环境相关的 locator，但 profile 只引用稳定 dataset ID：

```text
workspace/
  source_registry.cjson
  locator_bindings.local.json
  profiles/<abstract_dataset_id>/<version>.cjson
  snapshots/<abstract_dataset_id>/<snapshot_hash>/...
  build_reports/...
  mixture_profiles/...
```

`source_registry.cjson` 中的每项包括：

```text
dataset_id
expected_raw_fingerprint
semantic layer ids/hashes
capability summary hash
enabled/disabled
registration provenance
```

绝对路径放在机器本地的 `locator_bindings.local.json`，不进入抽象数据集 semantic hash。数据从
一块盘迁到另一块盘时，只需重新绑定 locator 并验证 fingerprint；不重做标注、不改变 profile。

### 6.2 Profile 与 snapshot

- profile 是 versioned、可编辑的组合规则；修改 source selector、视角或质量门即产生新版本；
- compiler 解析 registry、source capabilities、annotations 和 Skill projections；
- snapshot 固定 exact source fingerprints、Episode identities、sample intervals、selected
  annotation hashes、camera order、split 和所有 receipts；
- 训练只能引用 snapshot hash，不直接引用 mutable profile 名称。

## 7. 频繁添加、停用与恢复

### 添加数据集

1. 注册新的 `dataset_id` 和 locator；
2. 验证 raw fingerprint、schema、media/camera capability；
3. 发现或生成 dataset-local semantic layer；
4. 完成 annotation selection；
5. 对受影响的 abstract profiles 做增量 plan；
6. 生成新 snapshot，并报告新增、拒绝、重复 Episode；
7. 旧 snapshot 不变。

标注 pilot 不属于这里的“添加数据集”。pilot 只是在已注册源数据集自己的 semantic layer 中
增加一个 annotation campaign receipt；它不会产生新 `dataset_id`，也不会影响未来 split。

### 停用数据集

1. registry 中将 dataset 标为 disabled，或在 profile 中 exclude；
2. 编译新 snapshot；
3. receipt 明确 removed Episode identities 和原因；
4. 不删除 raw、semantic layer、annotation set 或旧 snapshot。

### 恢复数据集

重新 enable 后复验 fingerprint。内容未漂移则可以复用原 annotations/splits；内容变化必须作为
新 dataset identity 或新 semantic revision 处理，不能静默复用旧 receipt。

## 8. 多个抽象数据集混训

训练端输入一个 `TrainingMixtureProfileV1`：

```text
mixture_id/version
members:
  - abstract_dataset_snapshot_hash
    sampling_weight
input compatibility policy
batching policy
global seed
```

默认要求所有成员具有相同：

- target Skill/action contract；
- model observation signature；
- camera slot 顺序及像素预处理 contract；
- state/action schema 或显式 projection 后 schema；
- timebase/chunk/padding contract。

front-only snapshot 和 three-view snapshot 不能直接进入同一个固定三视角模型 batch。允许的方式
只有：

1. 分别训练/评估；
2. 模型明确支持两种 observation signature，并由 mixture profile 指定 adapter/head；
3. 从三视角数据显式投影 front-only，形成 common-front 对照；
4. 使用经过定义和测试的 modality-dropout contract，且提供缺失 mask。

训练端可以实现 sampling 和 batching，但不得自行改变成员资格或相机兼容判断；这些都必须在
mixture compile/validation 阶段形成 receipt。

## 9. 与现有实现的关系和缺口

当前 prepared-training 已有可复用能力：

- 多个 `PreparedSourceDataset`；
- `include_episode_identities`；
- per-Episode `annotation_selection`；
- `modality_selection` 和 column projection；
- 跨 Progress profile 的显式 Skill projection；
- immutable prepared-view hash 与 source inventory。

尚缺：

- dataset-local namespaced retrospective semantic layer；
- stable source registry 与 relocatable locator binding；
- workspace-aware/native-aware 统一 `SemanticEpisodeSource`；
- typed camera capability/input signature constraint；
- versioned `AbstractDatasetProfile` 与 resolved snapshot；
- enable/disable、增量 diff/build receipt；
- `TrainingMixtureProfile` 与跨 snapshot compatibility validator；
- prepared builder 对“semantic root 与 raw root 分离”的支持。

## 10. 分阶段实施

### Phase A：合同与只读解析

- 在 `dataset_contracts` 增加 source registry、Episode capability、abstract profile、snapshot、
  mixture profile 纯合同；
- canonical camera role 使用 `base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb`；
- 定义 raw fingerprint 与 semantic revision 的独立 hash domain；
- 增加 capability/input-signature validator 和负例。

### Phase B：标注归位

- 增加 writing-v2.3 专用 admission/initializer，将 retrospective writer 的写入目标严格限制为
  当前 v2.3 dataset root 下的 dataset-local named semantic layer；
- 现有未正式发布的 pilot 可直接使用新布局，无需迁移训练资产；
- GUI 由薄 adapter 打开一个 v2.3 source dataset + semantic layer，草稿与 immutable
  annotation set 写回该 layer；不在标注会话中加载 workspace registry 或其他数据集；
- source raw 文件保持只读，sidecar 写入范围严格限于 layer 目录；
- 对必须只读的 source 支持显式 bound overlay fallback。

### Phase C：Workspace registry/compiler

- 实现 register/verify/enable/disable/relocate plan API；
- profile 只引用 dataset ID 与 selectors；
- 编译 exact membership、dedup、annotation selection、Skill intervals、camera compatibility、split；
- 输出 diff：added/removed/rejected/unchanged/conflicted。

### Phase D：Prepared snapshot

- 把现有 prepared-training builder 输入泛化为 `SemanticEpisodeSource`；
- native 4/5-stage 和 retrospective one-step 都生成统一 `PreparedEpisode`；
- 发布 front-full、front-common、three-view 等不同 input signature snapshots；
- reader 验证 snapshot、source registry binding、semantic layer 和 raw inventory。

### Phase E：Training mixture

- 实现 mixture compatibility validator；
- 只允许相同 input/action contract 默认混训；
- 异构视角必须指定 typed adapter 或 modality-dropout contract；
- 训练日志记录所有 snapshot hash、weight、seed 和 resolved sample count。

## 11. 包所有权

| 包 | 所有权 |
|---|---|
| `dataset_contracts` | registry/profile/snapshot/capability/mixture schemas 与 hashes |
| `dataset_interface` | source discovery、constraint resolver、Skill/interval projection、snapshot compiler/reader |
| `collection_runtime` | 通用 dataset-local semantic layer 原子/不可变持久化；不知道 v2.3 writing、cohort 或训练约束 |
| `control_station` | 通用标注交互；writing-v2.3 adapter 只把单个 source dataset/layer 转成 work items |
| `model` 或训练包 | 消费 snapshot、按已验证 mixture 做 sampling/batching；不重新解释 dataset membership |
| `utils.dataset_postprocess` | writing-v2.3 layer 初始化及 workspace CLI 门面；调用下层 API，不复制合同 |

### 11.1 建议代码组织

```text
dataset_contracts/
  semantic_layers.py                 # 通用 layer/index/binding 纯合同
  abstract_dataset.py                # profile/snapshot/capability 合同
  training_mixture.py                # mixture 与 input signature 合同

dataset_interface/
  semantic_sources.py                # native/retrospective 统一只读接口
  writing_stage1_v23.py              # 仅 v2.3 writing admission/selection/初始化计划
  abstract_dataset_compiler.py       # registry/profile -> snapshot

collection_runtime/
  semantic_layer_store.py            # 通用原子写入、锁、immutable publish

control_station/refinement_annotation/
  application.py                     # 通用 Stage 编辑协调
  writing_v23_adapter.py             # 单 dataset + layer 的薄 work-item adapter

utils/dataset_postprocess/
  cli.py                              # init/validate/register/compile 命令门面
```

禁止新增一个 writing 专用 GUI、第二套 Stage annotation store，或让
`control_station`/`collection_runtime` 读取 7.23/7.24 cohort 规则。cohort/provenance 过滤属于
`dataset_interface.writing_stage1_v23`；上层 workspace 组合属于 abstract-dataset compiler。

## 12. 验收标准

- three-view profile 对任意缺少 wrist 的 Episode fail closed；
- front profile 可同时使用 front-only 与 three-view 数据的 front 投影，且 identity/annotation/split
  不变；
- full 4/5-stage Episode 只输出第一 Skill interval，不修改 raw Episode；
- 添加/停用 source 只产生新 snapshot，旧 snapshot hash 和内容不变；
- locator 迁移后 fingerprint 相同则 snapshot 可重新解析，profile hash 不变；
- duplicate Episode identity/content conflict fail closed；
- 多 annotation set 无显式 selection 时 fail closed；
- 不兼容 observation signature 的 snapshots 默认不能混训；
- source raw fingerprint 与 semantic revision 分离；添加 Stage annotation 不改变 raw identity；
- 所有 source/profile/snapshot/mixture diff 都有机器可读 receipt；
- 原始 Parquet、视频、既有 metadata 的 size/hash/inode 在标注和编译前后不变。

## 13. 推荐决策

采纳“标注归属于源数据集、workspace 管理抽象数据集、训练消费抽象数据集”的结构。实现时将
“一个抽象数据集由一个 profile 管理”具体化为：

```text
one versioned AbstractDatasetProfile
    -> many immutable resolved snapshots over time
```

这样既能频繁增删日期数据集，又不会牺牲训练复现；视角约束也在进入训练之前被统一证明，
而不是由训练 loader 在运行中临时猜测。

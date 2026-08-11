# v2.3 Writing 第一子任务后标注与双训练视图方案

> 状态：dataset-local 合同、发布器、严格 reader 与 GUI adapter 已实现并通过回归；待执行
> 12 条 pilot 人工标注与远端正式发布
>
> 基线日期：2026-08-04
>
> 数据范围：7.23 / 7.24 writing 第一子任务，严格媒体可用 719 Episodes
>
> 目标：不改写 v2.3 Parquet、视频和既有 metadata，只在对应 v2.3 数据集内追加可审计的
> retrospective Progress/Stage semantic layer；跨数据集组织和训练视图由上层 workspace 管理。

## 0. 实施进度（2026-08-04）

已完成：

- 增加 `RETROSPECTIVE_MIGRATED` source binding、window、migration manifest 纯合同及严格 reader；
- 增加专用 one-step Progress profile、三 Stage profile、双臂 Skill v2，以及专用窗口和现有
  4/5-step profile 到该 Skill 的显式 projection；
- 增加严格 cohort builder，实测得到 7.23 front-valid 479、7.24 three-view-valid 240，合计
  719；按 provenance 排除 three-view logical manifest 中混入的 83 条其他批次数据；
- 增加 external semantic-root 的不可变 writer/reader seam 作为合同原型，并在加载时复验
  info、Episode record、Parquet 和已准入视频 hash；该原型不作为正式 writing 标注目录结构；
- retrospective timeline 已接入既有 sample/Stage resolver；测试证明三段可以在同一 Attempt 上
  正时长、无重叠地覆盖完整 accepted sample axis，并可提交 immutable Stage annotation set；
- 已在远端数据上完成 12 条分层 pilot（7.23/7.24 各 6 条）的只读预检：36 个数据文件全部
  通过 size/hash、声明/解码帧数和 source binding 检查，Episode 帧数范围为 61--321；原始
  dataset 未被改写。
- 既有 `control_station refinement-annotation` 的通用 GUI/Stage 闭环已接入正式
  dataset-local campaign reader；只有同时具备 definition 与 publication receipt、且通过源绑定
  复验的 Episode 才进入标注队列，未完成 campaign 不可见。
- `utils.dataset_postprocess writing-stage1-pilot --publish-dataset-local` 已实现：分别向 7.23、
  7.24 的 `semantic_layers/<layer-id>` 发布 campaign definition、6 条 Episode migration 和
  publication receipt；`--workspace-root` 只保留为兼容原型。
- 正式发布前会同时预检两批数据，强制 7.23/7.24 各来自唯一且不同的源数据集、各 6 条；
  dataset-local source locator 固定为 `.`，数据集整体移动后仍可解析。semantic layer 和目标
  layer 的符号链接逃逸会被拒绝，campaign receipt 会复验定义、成员和全部 source binding。

当前 cohort semantic hash 为
`2f7b0c4ee3bef645bd32879477e133cfc5e741ad9c35c9742c89d4a3c8a583b0`。本轮仅证明
source admission、合同和持久化边界闭合，**不代表 12 条 pilot 已完成人工 B1/B2 标注**。

12 条 pilot 不是新的数据集，也不是 train/val/test 中的 test split：7.23 的 6 条直接在
7.23 数据集自己的 layer 内标注，7.24 的 6 条直接在 7.24 数据集自己的 layer 内标注。后续
扩展时继续向这两个 layer 追加，其 sealed annotation set 不复制、不迁移、不重标。

待完成：12 条 pilot 的 B1/B2 人工标注/复核、远端 dataset-local semantic layer 的正式发布，
以及上层 abstract-dataset workspace/prepared views。尚未向远端源数据集写入 semantic layer，
也尚未发布任何正式训练 view。

## 1. 已确认事实

### 1.1 数据与任务语义

- 每条候选 Episode 只记录 writing 4/5-step 任务的第一个子任务，不是完整长程任务。
- 每条 Episode 的成功终点是：毛笔仍带笔盖，右手已经稳定抓住毛笔，并将毛笔送到左手
  可以取盖的位置。
- 下一子任务应为 `left_remove_brush_cap`，但历史 Episode 在它开始前正常结束。
- 第一子任务是双臂协同运动：左臂和右臂都移动，右夹爪完成抓笔；左夹爪大部分保持原状态。
- 4-step 与 5-step writing 的该子任务入口、三段执行语义和成功终点相同。

远端对 719 条严格候选的全量低维复核结果：

| 指标 | 左臂 | 右臂 |
|---|---:|---:|
| action 平移 excursion 中位数 | 0.162 m | 0.228 m |
| action 旋转 excursion 中位数 | 45.2° | 39.3° |
| measured state 平移 excursion 中位数 | 0.155 m | 0.193 m |

719/719 条的左右 action 平移 excursion 都超过 5 mm，711/719 条的左右侧都超过 1 cm。
因此不能把左臂标成 `fixed_hold`，也不能用只监督 `right_arm + right_hand` 的旧 Skill
合同代表完整第一子任务。

### 1.2 严格 cohort

```text
7.23：503 个已整理 v2.3 Episodes
      - 24 个 front 视频声明/解码帧数不一致
      = 479 个严格 front-valid Episodes

7.24：241 个已整理 v2.3 Episodes
      - 1 个 front/right-wrist 视频声明/解码帧数不一致
      = 240 个严格 three-view-valid Episodes

严格后标注 cohort = 479 + 240 = 719 Episodes
```

源数据都是 `dexteleop_dataset_v2.3`，action/state 为相同的 16 维合同，pose frame 为
canonical desk absolute；它们来自 v2.2 migration compatibility，
`native_v23_causal_snapshot=false`。这不会阻止基础训练，但必须保留迁移 provenance，不能
伪称为 native v2.3 capture-causal 数据。

25 个媒体不完整 Episode 保留在历史源和 exclusion receipt 中，不删除、不进入 719 标注
队列，也不因 Parquet 仍可读取就进入多模态默认训练。

## 2. 核心决策

### 2.1 标注一次，按模态派生视图

```text
                    719-Episode semantic cohort
                               |
              retrospective Progress window + 3 Stage labels
                               |
                    one sealed annotation set / Episode
                               |
                 +-------------+-------------+
                 |                           |
          front-only full view         three-view view
             719 candidates             240 candidates
        front + state + action     front + L/R wrist + state + action
```

Stage、任务成功、异常和复核结论与相机数量无关。三视角 Episode 可以在 GUI 中切换视角帮助
判断边界，但不能生成另一套 Stage 真值。两个训练视图必须引用同一个 Episode identity、
migration manifest 和被显式选择的 annotation-set hash。

### 2.2 不把历史 Episode 任意绑定成 4-step 或 5-step 完整任务

一个 p2 数据集/语义工作区只能绑定一个 Progress profile。若把历史数据直接绑定为 4-step，
会让 5-step 复用依赖任意选择；反过来也一样，而且历史采集本身从未记录后续拓扑。

推荐新增一个只描述真实 capture window 的 source profile：

```text
profile_id: dmtac_bimanual_writing_acquire_present_window_v1
recording_scope: PROGRESS_STEP_WINDOW
entry_step: right_acquire_brush
entry_mode: PROFILE_INITIAL
history_complete: true
truncated_at_start: false
termination: WINDOW_COMPLETED
active_attempt_disposition: SUCCESS
full_task_outcome: NOT_APPLICABLE
```

该 profile 只表示“第一子任务窗口采完”，不声称完整 4/5-step task 成功。它通过显式
Progress-to-Skill projection 投影到 4-step 和 5-step 共用的第一子任务 Skill。

### 2.3 第一子任务使用新的双臂 Skill 合同

推荐发布新版本，而不是原地改变可能已经被 receipt 引用的旧 Skill hash：

```text
skill_id: writing.bimanual_acquire_brush_and_present_for_cap_removal
skill_version: 2

active controlled components:
  - left_arm
  - right_arm
  - right_hand

full-16D action 中继续发布的 hold component:
  - left_hand（有效的保持/开放目标，不丢列、不伪装成 active grasp）

entry predicates:
  - capped brush is available
  - right hand is ready to approach/grasp
  - both arms are in the declared initial workspace

success predicates:
  - brush is stably held by the right hand
  - capped brush is presented within the left-hand cap-removal region
  - both arms end in a valid coordinated pose
```

主训练 view 保留完整 16 维 action。若模型只输出 active components，必须生成显式 component
projection 和 omission receipt；不得靠切片位置默默删除 left-hand target。

## 3. 第一子任务的标注协议

### 3.1 Coarse Progress

每条 Episode 只有一个 retrospective Attempt：

```text
Attempt start = Parquet 第一个正式 accepted sample timestamp
Attempt end   = Parquet 最后一个正式 accepted sample timestamp
Step          = right_acquire_brush
Outcome       = SUCCESS（用户确认的正常第一子任务窗口）
Termination   = WINDOW_COMPLETED
```

禁止生成以下不存在的 capture facts：

- 不生成进入 `left_remove_brush_cap` 的 Advance；
- 不生成第二步的零长度 Attempt；
- 不把正常停录标成 Failure、Abort 或 full-task truncation；
- 不生成在线 operator button、Hold、Pause、Resume receipt；
- 不把 annotation 时间冒充 capture effective/available time。

### 3.2 三个 fine Stage

三个 Stage 是双臂协同任务阶段，不是只描述右臂的标签。左臂在三个区间中的运动保留在相同
sample timeline 上。

| Stage | 起止判据 | 标注员主要观察 |
|---|---|---|
| `right_acquire_brush.approach` | Episode 首样本至右手开始形成抓取 | 双臂离开起始位；右手靠近带盖毛笔；左臂进入后续取盖协同区域 |
| `right_acquire_brush.grasp` | 抓取开始至毛笔已稳定随右手运动 | 右夹爪闭合/接触；毛笔不滑脱；左臂允许继续协同移动 |
| `right_acquire_brush.lift_and_present` | 稳定抓取后的首次提起至 Episode 尾 | 毛笔离开支撑并被送到左手可取盖位置；最终姿态满足已确认成功条件 |

两个可编辑边界都吸附到正式 sample axis：

1. `B1 approach -> grasp`：右夹爪出现有意闭合/接触并开始建立抓取的第一帧；
2. `B2 grasp -> lift_and_present`：抓取已经稳定，毛笔开始随右手离开支撑并进入提送运动的
   第一帧。

区间语义：

```text
A1 = [sample 0, B1)
A2 = [B1, B2)
A3 = [B2, final accepted sample]  # 尾部闭区间
```

三段必须正时长、顺序固定、不重叠，合起来覆盖整个有效 Attempt。正常 Stage 边界只是语义
边界，主第一子任务 VLA 的 action chunk 可以跨越 A1->A2 和 A2->A3；只有异常区间、Episode
边界、Failure/review exclusion 或真实 continuity break 才切断训练连续性。

### 3.3 Leading/trailing 内容

- 短暂的正常静止准备可以归入 A1；成功终点处的短暂稳定保持可以归入 A3。
- 若 Episode 开头包含与任务无关的人工摆放/复位，标为 `BAD_CAPTURE` abnormal interval，
  不新增虚构 Stage。
- 若成功到位后又发生回撤、复位或无关动作，尾部标为 abnormal interval；不得把它并入
  `lift_and_present` 的 nominal supervision。
- 视频遮挡导致 B1/B2 无法可靠判定时，标记 `WHOLE_EPISODE_REVIEW_PENDING`，不按三等分
  猜边界。

### 3.4 自动建议只用于提速

GUI 可以生成 mutable draft 建议：

- B1 候选：右 hand action 开始持续闭合的 sample；
- B2 候选：右 hand action 已进入稳定区，同时右侧 pose 出现持续提起/提送的 sample；
- 对三视角 Episode，候选边界在三个视频上共享同一 sample index。

候选检测不产生正式标签，不能直接 seal。最终边界必须由人观看视频确认；低维信号与视觉
冲突时进入二次复核。

## 4. Retrospective artifact 与 dataset-local semantic layer

### 4.1 只追加 namespaced layer，不补绑原 capture catalog

现有保护逻辑会拒绝向已有 Episode 的 profile-less 数据集补写在线 Progress profile。应创建
位于对应 v2.3 数据集内、与 raw/capture metadata 隔离的 named semantic layer：

```text
<v2.3-dataset>/semantic_layers/writing_stage1_retrospective_v1/
  profile.cjson
  stages.cjson
  campaigns/<campaign_id>/definition.cjson
  campaigns/<campaign_id>/publication_receipt.cjson
  episodes/<episode_identity>/...
  meta/progress/stage_annotation_drafts/...
  meta/progress/stage_annotations/...
```

该 layer 是追加式语义 sidecar，不得改写 `meta/info.json`、`meta/episodes.jsonl`、Parquet、
视频或 capture Progress。禁止复制/hardlink raw 数据。上层 workspace 以后只注册 7.23、7.24
等 dataset ID 和各自 semantic layer，不拥有 Stage annotation。

### 4.2 Layer 内的 eligibility 与 annotation campaign

每个数据集的 layer 分开保存“全部严格可标 Episode”和“本次实际标注批次”：

```text
semantic_layers/writing_stage1_retrospective_v1/
  campaigns/
    pilot_v1/
      definition.cjson               # eligible=479/240；selected=6
      publication_receipt.cjson       # 6 个 migration-manifest hash
    expansion_0001/
      definition.cjson
      publication_receipt.cjson
  episodes/<episode_identity>/...
  meta/progress/stage_annotations/...
```

- campaign receipt 只引用同一 source dataset 内的 Episode identities；禁止跨 dataset locator；
- `pilot_v1` 是标注流程 QA 批次，不参与训练 split 语义；
- 后续 batch 只追加未迁移/未标注 Episode，已封存 pilot annotation 原地复用；
- 每个 campaign 有自己的 actor、created-at、selection hash 和 completion/status receipt；不能要求
  未来追加批次伪用 pilot 的时间戳；
- 重跑同一 campaign 时从该 campaign receipt 恢复 actor/timestamp 并验证成员完全一致；不同
  campaign 可以有不同创建时间；
- 从 layer 中“移除”只改变上层 selection/profile，不删除任何 Episode migration 或 annotation。

### 4.3 身份与 source binding

719 cohort manifest 对每条 Episode 固定：

- 现有 logical `episode_identity`；
- source dataset root、source episode index 和 provenance source index；
- source `info.json` / `episodes.jsonl` receipt hash；
- Episode record hash、Parquet hash、已准入视频 hash；
- `front_valid`、`three_view_valid`、mask readiness 等模态 capability；
- 7.23 / 7.24 batch 与历史 migration receipts；
- 25 个媒体异常 Episode 的独立 exclusion receipt hash。

同一 identity 内容冲突、相同 source locator identity 不同、source hash 漂移或同一 Episode
重复出现时 fail closed。front-only view 对 7.24 Episode 必须复用 three-view manifest 已有的
identity，不能因少加载两路相机而重新生成身份。

### 4.4 retrospective 不能伪装成 capture-causal

当前 `resolve_progress_timeline()` 固定输出 `CAPTURE_CAUSAL`，不能直接用两条伪造的 start/
termination capture facts 来迁移旧数据。需要新增显式合同和 reader：

```text
schema: dexteleop.retrospective_progress_window.v1
resolution_kind: RETROSPECTIVE_MIGRATED
source_manifest_hash: ...
source_episode_record_hash: ...
sample_axis_hash: ...
assertion_provenance: dataset-owner-confirmed writing-stage1 intent
window_attempt: first sample -> last sample
migration_created_at / actor_id: annotation clock provenance
```

`dataset_interface` 将它解析为统一的 resolved Attempt view；Stage annotation 和 prepared
view 必须同时绑定 migration manifest hash 与 sample-axis hash。任何只接受 capture-causal
Failure/receipt 的逻辑继续拒绝 `RETROSPECTIVE_MIGRATED`，不能放宽全局安全门。

## 5. 一次标注、两类训练产品

### 5.1 Front-only full view

```text
view_id: writing_stage1_front_full_v1
source candidates: 719
modalities:
  - front RGB
  - observation.state
  - action
  - selected optional tactile fields（只有训练合同显式要求时）
annotation selection: 每个 Episode 显式绑定一个 sealed set hash
target skill: writing.bimanual_acquire_brush_and_present_for_cap_removal.v2
```

7.24 的 240 条在这个 view 中只选择 front，但身份和标注仍与三视角 view 相同。

### 5.2 Three-view view

```text
view_id: writing_stage1_three_view_v1
source candidates: 240（当前仅 7.24 strict three-view rows）
modalities:
  - front RGB
  - left-wrist RGB
  - right-wrist RGB
  - observation.state
  - action
annotation selection: 与 front-only view 相同的 sealed set hash
target skill: 与 front-only view 相同
```

当前 240 条的 RGB 媒体完整性可用于 RGB-only three-view；若训练依赖 final wrist keep-mask，
则在 mask receipt 完成前继续标为 `augmentation_not_ready`，不能把 RGB-ready 等同于
mask-ready。

### 5.3 公平相机对照 view

仅比较 `front_full_719` 与 `three_view_240` 会同时改变相机数量和训练数据量，无法归因。
因此额外生成只用于实验对照的：

```text
writing_stage1_front_common240_v1
```

它与 three-view view 使用完全相同的 240 个 Episode、split 和 annotation，仅选择 front。
正式训练产品仍是两类；common-240 是可删除、可重建的评估视图。

### 5.4 Split 一致性

train/val/test assignment 只计算一次，并由 stable Episode identity 或更保守的连续采集 block
生成。要求：

- 同一 Episode 在 full-front、common-front 和 three-view 中永远属于同一 split；
- 不因 annotation 新版本或相机投影重新抽签；
- 优先按连续采集 block 分组，避免高度相似的相邻演示跨 train/test 泄漏；
- split receipt 绑定 cohort manifest hash。

### 5.5 频繁增删数据集与完整流程数据复用

完整的 source-local semantic layer、workspace registry、视角 input signature、抽象数据集
snapshot 与训练 mixture 规划见
[`abstract_dataset_workspace_training_plan.md`](abstract_dataset_workspace_training_plan.md)。

多数据集管理分为三层，不能全部放进训练进程，也不能把一个 annotation workspace 当成
长期可变的数据湖目录：

```text
Source registry（可变管理面）
  dataset_id -> locator/fingerprint/capabilities/enabled
                         |
                         v
Semantic workspace（追加式标注真值）
  Episode identity -> Progress/Stage/selected annotation candidates
                         |
                         v
Training recipe -> immutable prepared view（训练输入快照）
  Skill/modality/include/exclude/weights/split/annotation selection
```

- 同样只采第一子任务、并绑定相同 one-step retrospective profile 的新数据，可以追加进入同一
  semantic campaign；删除只在 registry/recipe 中 disable 或 exclude，不删除历史 migration
  seal 和 annotation set。
- 原生 4-stage/5-stage 全流程数据保留原 profile 和原 workspace；训练时通过显式
  Progress-to-Skill projection 只选择 `right_acquire_brush` 的 Attempt/Stage intervals，不能为
  方便合并而把全流程 Episode 改写成 one-step retrospective Episode。
- front-only、three-view、未来其他相机组合是 recipe 的 modality projection，不产生另一套
  Stage 真值。
- 频繁添加数据集只新增稳定 `dataset_id`、验证 fingerprint/capability 并重编译 view；频繁
  移除只生成新的 recipe/view hash。旧 prepared view 仍然可重现，源数据和标注不删除。
- 训练端只接受一个已冻结的 prepared-view hash，可以在训练配置中设置 source/skill sampling
  weight，但不得在运行时扫描目录、选择“最新标注”或自行解释 4/5-stage Step。

当前 prepared-training builder 已支持多个 native source dataset、显式 Episode include、显式
annotation selection，以及通过 Skill projection 合并不同 Progress profile；但尚不能直接把
external retrospective workspace 与其只读 v2.3 sources 作为一个 source 输入，因为当前
`PreparedSourceDataset` 仍假设 Episode catalog、Progress、annotation、Parquet 和视频都位于同一
dataset root。需要增加 workspace-aware semantic source adapter，而不是把这项路径解析逻辑下放
到训练代码。

## 6. 标注作业流程

### Gate A：冻结 source cohort

1. 从现有逻辑清单选择 7.23 front-valid 479 条；
2. 从 three-view 清单只选择 provenance 为 7.24 的 240 条，排除混入的 83 条 7.12 数据；
3. 对 identity 去重并验证总数恰为 719；
4. 重验 v2.3 schema、pose-frame、policy-I/O、Parquet/视频 membership；
5. 发布只读 source manifest 和 modality coverage receipt。

### Gate B：在每个 v2.3 数据集内生成 retrospective semantic layer

1. 验证目标为 `dexteleop_dataset_v2.3`，并将严格 selection 绑定到当前 dataset fingerprint；
2. 为当前数据集内每个入选 Episode 建立 source binding 和 sample-axis hash；
3. 在 named layer 写入 one-step window profile 与三 Stage profile；
4. 生成一个 `RETROSPECTIVE_MIGRATED` window Attempt；
5. 验证没有 Advance、第二 Attempt、Failure、Hold/Pause/Resume；
6. manifest 最后原子发布；重复运行内容一致，hash 不变。

### Gate C：一审 Stage 标注

1. 默认打开 front；三视角 Episode 允许随时切 wrist，不改变边界；
2. 检查终点成功；
3. 调整 B1、B2 到正式 sample；
4. 检查 leading/trailing 无关内容；
5. 保存 mutable draft；
6. 校验三段 coverage 后 seal immutable annotation set。

### Gate D：二次复核

二次复核覆盖：

- 所有一审标记 review/低置信度/视觉与低维冲突的 Episode；
- 双臂活动异常低的尾部样本（至少包含未达到双侧 1 cm 的 8 条）；
- 所有 abnormal interval 或 endpoint 有疑问的 Episode；
- 7.23 和 7.24 各按采集 block 分层抽取至少 10% 的普通成功 Episode。

B1/B2 两位标注员相差不超过 3 个 accepted samples 时可采用复核版本；超过 3 samples 必须
人工仲裁并发布新的 `SECOND_REVIEW` set。旧 set 永久保留。

### Gate E：发布训练视图

1. 显式生成 per-Episode annotation selection manifest；
2. 先发布 `front_common240` 与 `three_view240` 并核对共同 cohort；
3. 再发布 `front_full719`；
4. 输出 nominal、review-pending、abnormal、media-invalid 数量；
5. 训练只消费已发布 view hash，不扫描“最新 annotation”。

## 7. 实现所有权

| 包 | 应承担的工作 | 禁止承担的工作 |
|---|---|---|
| `dataset_contracts` | retrospective window、migration manifest、modality capability、双臂 Skill v2 纯合同 | 文件扫描、GUI、训练选择 |
| `dataset_interface` | writing-v2.3 admission、严格 selection、source hash、semantic-layer reader、Stage/Skill projection | 原始 Episode 生命周期、GUI、workspace 组合管理 |
| `collection_runtime` | 通用 named-layer 与 immutable annotation 持久化原语 | v2.3/writing 解释、cohort、训练投影、GUI |
| `control_station` | 通用 refinement GUI；writing-v2.3 薄 adapter 一次打开一个 dataset/layer | 跨数据集 workspace、迁移算法、训练约束、控制设备 |
| `utils.dataset_postprocess` | `writing-stage1-pilot` 等现有 CLI 门面 | 新合同、GUI、workspace 编译或绕过 validator |

不增加新的真机或生产控制入口。迁移/标注能力接入现有 dataset-postprocess 和
`control_station refinement-annotation` 入口。

## 8. 计划中的代码工作包

### WP1：合同与 golden fixture

- 新增 retrospective source binding/window/manifest 合同；
- 新增 `RETROSPECTIVE_MIGRATED` resolution kind；
- 新增 one-step source profile、三 Stage profile和双臂 Skill v2；
- 增加 4-step/5-step source Step -> shared Skill v2 的显式 projections；
- 为 1 个 front-only、1 个 three-view、1 个 media-invalid Episode 建 golden fixture。

### WP2：719 cohort builder

- 读取现有 logical manifests，不重扫或重写 raw；
- 按 provenance 精确剔除 7.12 的 83 条；
- 验证 479/240/719 count、identity 唯一性和 source receipt；
- `--plan` 只输出报告，publish 使用全新目录并最后写 manifest。

### WP3：v2.3 dataset-local semantic layer reader/writer

- 增加 named-layer index/selection/capability receipt，明确 layer 归属的 v2.3 dataset fingerprint；
- raw Parquet/视频只能从同一 dataset root 读取并逐 Episode 验证 record/hash；
- writer 只能创建或追加 `semantic_layers/<layer_id>`，禁止修改任何既有 raw/capture 文件；
- 对操作系统只读数据集，未来可用显式 bound overlay fallback，但不是本轮默认流程。

### WP4：GUI 三阶段模板与复核

- writing-v2.3 adapter 输入为一个 dataset root 和一个 semantic layer ID，不接受 abstract
  workspace、多数据集 source list 或训练 profile；
- 使用 gripper/pose 生成 B1/B2 draft suggestions；
- front-only 不展示不可用 wrist 按钮；three-view 使用同一 sample cursor；
- 明确显示 `RETROSPECTIVE_MIGRATED / PROGRESS_STEP_WINDOW / FULL TASK N/A`；
- 支持 review-pending、异常区间、annotation-set 版本选择和 second review。

### WP5：prepared training views

- 新增稳定 source registry：`dataset_id` 与可重定位 locator/fingerprint/capability 分离，支持
  enable/disable 和增量验证，不通过删除文件表达训练成员变化；
- 新增 semantic-layer-aware prepared source，使 native capture layer 与 v2.3 retrospective
  layer 都投影成统一 `PreparedEpisode`；
- 新增不可变 training recipe/snapshot：绑定 Skill、include/exclude、annotation selection、
  modality specs、source/skill weights 和 split policy；
- 支持一个 annotation selection manifest 配多个 modality specs；
- 发布 full-front、common-front、three-view 三个可重建索引；
- 绑定 Skill v2、component projection、split receipt 和所有 source/annotation hashes；
- loader 遇到 source 漂移、模态缺失、多个未选择 annotation set 时 fail closed。

## 9. 验收标准

### 9.1 Source 与零修改

- 719 cohort 恰为 479 + 240，identity 零重复；
- 25 media-invalid 与 83 个非本范围 7.12 Episode 都不进入 cohort；
- migration 前后所有 source Parquet、视频、metadata 的 inode/size/hash 零变化；
- 不创建 raw hardlink 副本，不修改现有 logical catalog。

### 9.2 Progress/Stage

- 每个 Episode 恰有一个 window Attempt；
- termination 为 `WINDOW_COMPLETED/SUCCESS/NOT_APPLICABLE`；
- resolution kind 为 `RETROSPECTIVE_MIGRATED`；
- 无 Advance、Failure、Hold、Resume 或第二 Attempt；
- sealed nominal Episode 恰有 A1/A2/A3 三个 SUCCESS occurrence；
- occurrence 顺序、正时长、不重叠、sample coverage 和 final closed interval 全部通过。

### 9.3 训练视图

- pre-semantic-exclusion candidate coverage：front=719，three-view=240，common-front=240；
- post-annotation 实际 admitted count 由发布 receipt 明确报告，不能假定仍为 719/240；
- three-view identity/split/annotation selection 是 full-front 的精确子集；
- common-front 与 three-view 的 identity、split、action/state/sample intervals 完全一致；
- full 16D action 或显式 component projection 均通过 layout/hash 验证；
- nominal Stage 边界不错误切断第一子任务连续 chunk；异常/review/媒体问题不得进入默认 nominal view。

### 9.4 最小测试

- retrospective contracts/profile/Skill hash round-trip；
- profile-less source 保持不变；
- cohort count、去重、provenance 过滤和 source hash 漂移负例；
- semantic-layer path escape/symlink/hash-change 与 raw-write-attempt 负例；
- Stage coverage、边界吸附、review/abnormal 与 second-review 版本测试；
- prepared full-front/common-front/three-view subset 与 split 一致性测试；
- `validate-v23` 对所有 source roots 仍通过；
- `git diff --check` 和 refinement annotation `--dry-run`。

## 10. 发布顺序与停止条件

推荐顺序：

1. 在两个原 v2.3 数据集的同名 layer 内分别发布 `pilot_v1` receipt，各选择 6 条覆盖长短轨迹；
2. 两名标注员对 pilot 达到边界一致性门槛；
3. 在同一 layer 追加每批 50 条的 campaign，验证标注速度和 source hash 稳定性；
4. 完成 479/240 全部严格 eligible Episode，不建立新的“719 数据集”；
5. 标注完成后冻结 selection/split/view receipts；
6. 先跑 common-240 相机对照，再训练 full-front-719。

遇到以下任一情况立即停止 publish，不继续猜测：

- source receipt/hash 漂移或 identity 冲突；
- Episode 不符合“带盖毛笔被送到左手取盖位置”的终点语义；
- B1/B2 无法在 sample axis 上可靠落点；
- 视频出现任何新的 decoder warning、帧数不一致或 tail decode 失败；
- 4/5 profile 或 Skill component 语义仍在并行修改、hash 尚未冻结；
- prepared view 不能证明两个相机视图引用同一 annotation truth。

## 11. 最终交付物

- 7.23、7.24 对应 v2.3 数据集内的 writing-stage1 semantic layers；
- 每个数据集自己的 annotation 状态与 sealed-set selection manifest；
- 上层 abstract-dataset profile/snapshot，将两个数据集及未来来源组合为逻辑数据集；
- `writing_stage1_front_full_v1`；
- `writing_stage1_three_view_v1`；
- `writing_stage1_front_common240_v1` 对照视图；
- cohort、migration、annotation、split、modality、Skill projection 和 prepared-view receipts；
- nominal/review/abnormal/media-invalid 汇总与二审一致性报告。

最终原则：**719 条只产生一套第一子任务语义真值；front-only 和 three-view 只是对同一
语义时间轴的不同 observation 投影。**

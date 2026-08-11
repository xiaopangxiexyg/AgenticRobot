# DexTeleop 最小数据增广两阶段正式规划：V2.3 Plus / V2.4 Plus → V2.5

> 状态：正式规划
>
> 基线日期：2026-07-23
>
> 当前实现状态：Plus 原型已归档到 `legacy/dataset_semantic_plus_v0/`；新的最小 Plus v1 协议、兼容 reader、外部 append-only sidecar，以及 RM65/UMI 共用 GUI 的 default-off 接线已落地。
>
> 归档说明：[`legacy/dataset_semantic_plus_v0/README.md`](../legacy/dataset_semantic_plus_v0/README.md)。
>
> 适用范围：DexTeleop 正式 RM65 GUI 遥操数采、UMI 数采、离线标注、数据集验收与训练视图。
>
> 本文是该主题的唯一有效规划。文中的 `规划`、`研究` schema 名称均为目标设计，不代表当前代码已经实现。

## 0. 总体决策

本规划分为两个严格隔离的阶段：

| 阶段 | 当前优先级 | 目标 | 是否修改既有采集/控制链 |
|---|---|---|---|
| 第一阶段：`V2.3 Plus` / `V2.4 Plus` | 当前优先 | 对已 finalize、committed 或由 reader 明确证明 compatibility membership 的数据附加可选的 episode 级增广，支持基本筛选、分组和训练视图 | 否 |
| 第二阶段：`V2.5` | 长远规划 | 利用 proposal→admission→dispatch→measurement→world effect 的充分链路事实，构建更细语义和选择性重生成监督 | 可能；必须另行评审 |

`Plus` 不是新的 raw dataset schema，也不是把 `dexteleop_dataset_schema` 从 v2.3/v2.4 原地改名。
它固定表示：

```text
原始 V2.3 或可验证的 V2.4 source（native committed / compatibility view）
  + source identity profile
  + immutable semantic annotation overlay
  + overlay manifest / validator / training view
```

规划上第一阶段只有一套 Plus overlay。V2.3/V2.4 的差异只由 source adapter 负责，两个版本共享
annotation schema、publisher、reader、split 和 training-view 接口；V2.4 原生提交优先，兼容视图是
显式 fallback，不是第二套 Plus 产品：

```text
common Plus overlay = dexteleop_plus_episode_curation_v1

V2.3 source binding = v23_compat_content_locator_v1
V2.4 native binding = v24_committed_capture_membership_v1
V2.4 compatibility binding = v24_compatibility_content_locator_v1
```

V2.4 compatibility binding 只表示 v2.4 reader 明确提供的 v2.3 compatibility view（`v2.3_single_action_row`）。
它可以在 native capture commit 尚不可用时让 V2.4 数据进入同一 Plus 离线流程，但不声称存在
`episode_uuid`、capture commit、Arbiter/ControlTrace 或其他 v2.4 原生事实。调用方必须显式选择该视图，
不能由目录、文件名或 legacy index 自动猜测。

| source 输入 | Plus adapter/profile | 可引用的 native 事实 | 输出 |
|---|---|---|---|
| v2.3 finalized RM65/UMI | `v23_plus_source` / `v23_compat_content_locator_v1` | v2.3 finalized manifest/content membership | common Plus overlay |
| v2.4 native committed | `v24_plus_source` / `v24_committed_capture_membership_v1` | 已验证的 v2.4 UUID、capture commit、可选 trace | common Plus overlay |
| v2.4 explicit compatibility view | `v24_plus_source` / `v24_compatibility_content_locator_v1` | 仅 compatibility membership；native UUID/commit/trace 为 unavailable | common Plus overlay |

第一阶段允许增加显式的 default-off Plus 参数和 GUI 旁路适配器，但不得修改 raw writer、row schema、episode
保存/丢弃规则、控制权或 ControlTrace。第二阶段 V2.5 的任何细粒度在线能力继续视为 default-off future
feature；未提供 `--plus-episode-curation`（RM65）或对应 UMI 参数时，Plus 对既有链路完全不可见。

本规划将长期数据建设主线确定为：

```text
构建具有因果证据、反事实对照和明确本体边界的多粒度机器人数据
```

V2.5 的目标不再是单纯增加 segment、primitive、keypose 或 caption，而是让数据能够回答：

1. 一段行为为什么被判断为 grasp、search、correction 或 recovery；
2. 多次执行之间，哪些变量有证据地保持不变，哪些变量被有意改变；
3. 偏差出现后，实际改变了控制残差、交互计划、几何实现、局部子目标还是整个策略；
4. 哪些语义可以在 UMI 与 RM65 间共享，哪些事实必须保持本体专用；
5. 一条样本为什么允许或禁止进入某个训练视图。

工程底线保持不变：

```text
不可变事实层
    + 版本化语义与证据层
    + 可重建训练视图
```

任何语义扩展都不得修改既有动作事实、压缩真实时间、使用未来快照、把缺失值补成零、把推断伪装成传感器测量，或干扰真机控制线程。

第二阶段的在线部分统一视为 **future feature**：即使未来开始实现也默认关闭，只能通过显式实验参数启用。第一阶段不注册这些参数；现有 RM65/UMI 正式命令、`docs/real/teleop_args.sh` 配方和历史脚本均保持原行为。

### 0.1 最新目标：最小可选 episode 级增广

Plus 现在不再承担粗阶段、事件、反事实或证据本体。第一阶段只增加一层可跳过的 episode 级
`curation overlay`，解决“这条已保存数据应如何使用、属于哪个预先定义的采集变体”两个最直接的
数据集问题。它不声称给出任务成功真值，也不替代当前已经存在的系统质量、暂停、拒绝和碰撞统计。

| 信息 | 取值/约束 | 采集时机 | 直接用途 |
|---|---|---|---|
| `disposition` | `keep_nominal` / `keep_recovery` / `exclude` / `unknown` | episode finalize/commit 后一次确认 | 生成 nominal、recovery、excluded 训练视图 |
| `reason_code` | 仅在 `keep_recovery` 或 `exclude` 时可填：`task_failure` / `operator_recovery` / `system_issue` / `safety_abort` / `other` / `unknown` | 同一确认窗口，可不填 | 过滤、失败统计和复核抽样；不覆盖系统诊断 |
| `variant_id`（可选） | 仅使用 task profile 预先注册的有限枚举；无 profile 时字段不存在，禁止自由文本 | episode 开始前一次选择 | 变体分组、评估切分或条件训练；按 profile 决定是否可作监督 |

以上字段均为人工整理事实或采集意图，不由模型、VLM、裸传感器阈值或文件状态自动推断。
`disposition` 未确认时写 `unknown`；`reason_code` 不得把 `trajectory_quality`、`force_collision_*`、
`pose_quality` 等既有事实改写成新的语义。`variant_id` 表示预先声明的变体，不等于执行后证明的
`observed_changed`。

Plus 不重复现有的 `task_id`、任务描述/类型、`round_trip_phase`、`teleop.arm_mode`、pose/trajectory
quality、pause/rejection、媒体完整性和 writer/capture integrity；这些字段继续由当前 RM65/UMI 链路
负责。只有 task profile 明确声明且现有任务 metadata 无法表达的变体，才增加 `variant_id`。

Plus 的附加交互只允许两次轻量操作：有 task profile 时在独立的采集表/小工具中于开始前选一次变体，
结束后在 GUI/离线复核窗口选择一次 disposition（必要时选择 reason）。它们在 episode finalize 后再与
现有 episode locator 绑定；绑定不明确则为 `unknown`。没有响应、GUI 不可用或 Plus 未启用时，原
episode 仍按既有链路保存/丢弃，overlay 缺失或为 `unknown`。不增加逐帧快捷键、时间轴拖拽、事件
点选、传感器采样、在线队列或额外控制线程。

Plus 的启用边界是独立的 post-process/review 命令或显式 overlay root；现有 RM65/UMI 采集命令不增加
Plus 参数不会因为 profile、sidecar 目录或环境变量存在而自动启用。启用时 GUI 只增加开始前变体选择
和 finalize 后一次整理提交；写入在独立后台线程完成，失败只报告为 annotation error，不改变原始 episode
结果或控制循环。

训练视图默认只把 `disposition` 用于筛选，`reason_code` 用于统计，`variant_id` 只有在 task profile
明确 allowlist 后才用于分组/条件输入；三个字段都不默认进入 observation。粗 phase、interaction
state/event、EvidenceBundle、campaign/invariant、ControlTrace cause chain 和
`minimal_replan_level` 全部移到 V2.5 或研究抽样。

## 1. 实现现状与状态标记

本文使用四种状态：

| 标记 | 含义 |
|---|---|
| `已实现` | 当前生产/正式路径已有明确实现和测试 |
| `部分实现` | 合同或构件存在，但正式入口、持久化或全链路尚未闭合 |
| `规划` | 本文批准建设的工程能力 |
| `研究` | 需要实验验证，不承诺进入当前 runtime 合同 |

### 1.1 当前最有价值的已实现资产

当前项目的优势不是字段数量，而是已经具备以下可审计底座：

- `已实现` v2.3 `T0` 因果切点：`source_timestamp <= T0` 且 `available_timestamp <= T0`；
- `已实现` operator/command/reference/transport/actual 分层；
- `已实现` v2.3 pose-frame contract、canonical desk 和 contract hash；
- `已实现` RM65 measured qpos 与按项目运动学/TCP 计算的 `actual_tcp_pose`；
- `已实现` 相机帧索引、source/available 时间及 episode 视频完整性统计；
- `已实现` DMTAC 外置触觉流、snapshot sequence、时间与 manifest；
- `已实现` pause/resume 与 `collection.continuity_segment_index`；
- `已实现` `teleop.arm_mode`、pose quality、fault 和 episode 级采集完整性；
- `已实现` UMI snapshot rejection 及按原因汇总；
- `已实现` `validate-v23` 只读验收；
- `已实现` copy-on-write 后处理与 dataset episode 维护基础。

这些事实是后续证据化语义的来源，不能被新的标签层替换。

### 1.2 当前尚未闭合的能力

以下内容当前不能写成“已有正式能力”：

- `已实现` 外部 annotation identity、兼容 reader 和最小 curation sidecar writer；
- `已实现` Plus v1 的 episode `disposition`、bounded `reason_code` 和可选 `variant_id` 合同，以及 RM65/UMI 共用的 default-off GUI 轻量复核入口；
- `规划` 仅用于专门研究样本的稀疏 event、campaign/group 与 invariant 记录；
- `研究预留` `semantic_annotations.py` 中完整的 state/event/evidence 合同，以及
  `counterfactual.py` 中完整的 campaign/group/initial-state 合同；它们不属于首版必填字段；
- `研究预留` 通用多模态时间轴、candidate detector、关联图 split、ControlTrace cause chain 和
  minimal replan level；
- `研究` UMI/RM65 跨本体分层模型与选择性重生成。

此前的丰富 `Plus-0`、`Plus23-1`、`Plus24-1` 原型仍仅保存在 `legacy/dataset_semantic_plus_v0/`；当前实现
只注册最小 `episode_curation` contract/reader/store，并不恢复完整 source adapter、training view、timeline、
candidate、phase、关联图 split 或 evidence profile。V2.3/V2.4 通过显式 `source_binding` 进入同一 sidecar
协议，native v2.4 identity 仍由主架构另行提供，不能从目录或文件名猜测。

身份与发布现状也必须如实表述：

- 当前 RM65/UMI 核心 v2.3 writer 没有统一的 `dataset_instance_uuid` / `episode_uuid` 合同；
- ControlTrace 的 `episode_uuid` 只存在于可选 trace episode 路径，不能充当所有数据的通用身份；
- `已实现` Plus 外部 overlay 使用排除 annotation 目录的 source base fingerprint 合同；完整 snapshot hash、
  connected-component split 和发布 supersession 仍为后续增强；
- ControlTrace episode manifest 已有临时文件、`os.replace`、文件与目录 `fsync` 的原子发布范式；Plus v1
  当前对 manifest 使用原子写入，对业务 JSONL 使用 append+fsync，完整目录级 staging/commit marker 仍是规划；
- 当前没有按反事实关联图连通分量生成 split 的实现。

v2.4 ControlTrace 需要特别说明：

- `部分实现` 已有 ControlTrace 合同、WAL、health ledger、episode seal/commit、reader 和测试构件；
- `部分实现` `formal_real_human_shadow_v1` profile 被标记为 runtime-integrated；
- canonical GUI 的 human shadow 默认仍为 `off`，正常构建使用 dry-run profile；
- formal episode sink 当前只在测试或显式构建路径中出现，没有在 canonical GUI 正式数采中自动挂接；
- UMI 没有接入该 formal ControlTrace episode 链路；
- 因此当前语义规划不能依赖 ControlTrace 一定存在。

## 2. 两条正式数采链路的事实边界

### 2.1 RM65 正式 GUI 遥操数采

canonical 入口保持：

```bash
python -m teleop_runtime.debug.run_vive_rm65_real_teleop --gui
```

正式 position 配方当前关键频率是：

```text
dataset sample    15 Hz
IK / qref         60 Hz
qcmd             120 Hz
```

position 主 action 是 accepted qref 的 FK pose。必须保留以下链路差异：

```text
VIVE/operator target
    -> retarget / IK candidate
    -> accepted qref
    -> planner qcmd
    -> SDK send/receipt
    -> measured qpos
    -> measured TCP
```

训练和标注不得混淆：

- `teleop.action_source.<side>_*`：操作端/retarget 来源；
- 根 `action`：模型 action 合同；
- `action.reference.*`：position reference/qref 审计；
- `position.transport.*`：attempted/accepted/measured 传输事实；
- `actual_hardware_qpos`：RM65 硬件关节反馈；
- `actual_tcp_pose`：项目按 measured qpos、运动学和工具 TCP 计算的标准 pose；
- `actual_hardware_pose_6d`：SDK/controller-base 审计值，不是 canonical relative anchor。

impedance/力控是独立控制链。position 的 accepted-qref、IK iteration、qcmd planner 语义不得套用到 impedance 数据；跨 control modality 对照必须把 profile 和 modality 作为 `intended_preserved` / `intended_varied` 显式声明，并在执行后验证实际变化。

位控正式 writer 可运行在 spawn 子进程中。共享槽、queue 或 writer 过载时，episode 显式暂停；pause 前后属于不同 continuity segment。训练窗口和 semantic segment 都不得跨越该边界。

### 2.2 UMI 正式数采

UMI canonical 子命令保持：

```bash
python -m teleop_runtime.collection umi \
  --config teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json
```

UMI 的事实语义是：

- 不构造 arm backend，`embodiment_has_arm=false`；
- 双侧 VIVE tracker 经各自 `T_tcp_to_tracker` 得到 UMI 工具 TCP；
- 双腕 RGB、AngleTrigger、DMTAC measured state 和双侧 tactile 组成样本；
- `observation.arm.qpos` 是 12 维无效审计槽，不是真机关节状态；
- UMI `teleop.arm_mode` 是 annotation-only，`control_effect=false`；
- pose、camera、gripper 和 required tactile 任一不合格时拒绝该 source sample。

UMI arm action 使用 next-accepted-observation：

```text
observation_arm[t] = umi_tcp_pose[t]
action_arm[t]      = umi_tcp_pose[t+1]
action_gripper[t]  = angle_trigger_command[t]
```

因此：

- `N` 个 accepted source sample 只写 `N-1` 个 row；
- terminal source sample 被丢弃是合同，不是漏帧；
- arm action 有独立 source frame index/timestamp；
- gripper action 仍是同帧命令，不能一起 shift；
- UMI qpos 不得进入 loss、normalization、replay init 或 feasibility 判断。

当前默认 freshness 上限：

| 信号 | 上限 |
|---|---:|
| VIVE pose | `0.1 s` |
| wrist camera | `0.1 s` |
| AngleTrigger | `0.2 s` |
| gripper measured state | `0.5 s` |
| tactile snapshot | `0.5 s` |

这些是当前 config 值，不是所有任务的永久通用阈值。

### 2.3 UMI 与 RM65 的共享边界

批准采用“语义层可共享、低层事实按本体解释”的原则：

| 层级 | 是否共享 | 约束 |
|---|---|---|
| task goal | 可共享 | 使用共同任务模板 |
| strategy | 通常可共享 | 必须保留 embodiment 条件 |
| left/right role | 可共享 | 角色不等于具体设备 side binding |
| interaction state/event | 条件共享 | 证据 profile 可不同 |
| contact region/keypoint | 条件共享 | 需要坐标和对象身份对齐 |
| EEF geometry | 部分共享 | 需 tool/TCP 和 feasibility 条件 |
| qref/qcmd/qpos | 不共享 | UMI 无真实 arm state |
| admission/dispatch latency | 不共享 | runtime 拓扑不同 |
| collision/joint feasibility | 不共享 | 必须按目标机器人验证 |

`研究` 模型可以共享 strategy/interaction 表示，但 UMI 与 RM65 必须使用本体专用 geometry/control adapter。单纯因为两者 action 都是 16 维而拼接训练是禁止的。

## 3. 正式数据分层

### 3.1 Raw fact layer

原始事实层继续使用当前 v2.3 layout：

```text
dataset_root/
  meta/info.json
  meta/tasks.jsonl
  meta/episodes.jsonl
  meta/episodes_stats.jsonl
  data/chunk-xxx/episode_xxxxxx.parquet
  videos/...
  tactile/...
  wrist_depth/...             # 可选
  meta/v24/control_trace/...  # 可选且当前非普遍存在
```

规则：

- finalize 后核心 Parquet、媒体、合同和 capture commit 不原地重写；
- committed v2.4 capture 存在时，删除/重编号应使用 copy-on-write 派生数据集；
- semantic annotation 不回写根 action/state；
- 后处理产物保留 `postprocess_history` 和源合同 hash。

### 3.2 Versioned annotation layer（规划）

Plus v1 使用**数据集外置**的 annotation set。`plus_annotation_root` 必须由命令行/配置显式传入，不能从
source dataset root、源码目录或当前工作目录隐式推导。当前最小实现使用一个 root 对应一个 annotation set：

```text
plus_annotation_root/
  manifest.json                    # source fingerprint/profile/status
  episode_curation.jsonl           # Plus v1 唯一业务记录
```

V2.3 Plus 与 V2.4 Plus 都只在该外置根发布，source dataset/capture 全程只读。V2.3 Plus 通过
content locator 绑定 finalize 后的 v2.3 snapshot；V2.4 Plus 优先通过主架构已正式 committed 的 capture
membership/identity 绑定，也允许通过明确的 v2.3 compatibility view 绑定。两条 V2.4 路径都不新建
capture UUID 或第二套 capture commit；兼容路径不复制或推断 native identity。把 annotation materialize
回 `meta/annotations/`、追加 sealed capture root、或让原 writer 同步生成 overlay，均不属于第一阶段。
V2.5 若需要 `cause_chains.jsonl`，必须由独立 schema/version 和发布评审加入，不能预留空文件冒充
已有充分链路事实。

两条 Plus 链路在 overlay 侧必须使用同一个 `PlusSourceEpisodeBinding`、`AnnotationSetManifest`、
publisher、reader 和训练视图接口；绑定对象只把 `source_profile`、`source_dataset_schema` 和
parent locator 的来源差异显式保留下来。V2.3 adapter 提供 finalized manifest content locator，
V2.4 adapter 优先提供 native committed membership，也可在 reader 明确声明时提供不伪造 native
身份的 compatibility view。下游不得按版本复制一套 annotation record schema 或发布协议。

Plus v1 必须 append-only：同一 episode 的冲突修订不能静默覆盖旧记录；后续发布系统再通过新
`annotation_set_id` 表达 supersession。首版只要求
`episode_curation.jsonl`；不生成逐帧 `segments/events/evidence`，也不要求 campaign/relation 文件。
`variant_id` 的枚举由 task profile 在 annotation set manifest 中声明，不能在 episode 记录中写自由文本。
manifest 至少绑定：

- annotation schema/version；
- source profile、source dataset schema 和 source dataset base fingerprint；
- task profile（若启用）与 created time；
- annotation set status（`draft|complete|partial|invalid|superseded`）。

当前 v1 记录固定保存 `source_episode_manifest_hash`、legacy `episode_index`、review status 和
annotation identity；author/tool/completeness report、source snapshot content hash 与 parent set
留到后续事务 publisher，不作为 GUI 接线的前置条件。

#### 3.2.1 Base fingerprint 边界

首版区分两个 hash，不能混成一个：

- `source_dataset_base_fingerprint`：新写入使用 append-stable v2；由
  `meta/info.json` 的稳定字段（明确排除 seal 时更新的 episode/frame 等计数）、
  `meta/tasks.jsonl` 和 `meta/modality_validity.json` 的相对路径/内容 hash 组成，并排除会随
  append 变化的 `meta/episodes.jsonl`；不含目录位置或外部 annotation root。旧 provisional
  v1 manifest 只作兼容读取/续写，不用新算法静默重解释。单个 episode 另存
  `source_episode_manifest_hash`，完整 snapshot projection 仍属于后续版本；
- `source_snapshot_content_hash`：一个显式、不可变 `SourceDatasetSnapshot` 的内容绑定，固定成员
  episode、逻辑 manifest record 边界，以及成员的 Parquet、已声明媒体/触觉/depth 和存在且已
  commit 的 ControlTrace；每项记录规范相对路径、大小和内容 hash。

当前实现只在 finalize 后由 sidecar store 调用；调用方应把 dataset 视为冻结 source。后续若需要跨
append snapshot 复用 lineage，必须升级到独立 snapshot contract，不得把当前 v1 hash 误称为完整
append-stable identity。

它必须明确排除：

```text
meta/annotations/**
annotation scratch/staging/**
derived training views/**
```

不得使用目录 mtime、inode 或绝对路径参与 hash。dataset 后续新增 episode 复用同一 base
fingerprint 并形成新的 snapshot content hash，不能使旧 snapshot 或已有 episode identity 失效；
reader 按旧 snapshot 成员清单验证逻辑 record 与 artifact 仍未改变。新增、修订或 supersede
annotation set 不能改变两者；lineage anchor 改变必须得到新 base fingerprint，snapshot 成员
raw fact/manifest record 改变必须得到新 snapshot hash，并使旧 binding fail-closed。

#### 3.2.2 发布事务与状态

完整发布器（后续增强）应遵循与现有 ControlTrace 原子发布范式一致的原则，但使用独立事务。当前
Plus v1 只对 manifest 使用原子替换，对 JSONL 追加使用 flush+fsync；它是可审计的 draft/partial
authoring store，不宣称已经提供目录级 publish transaction：

1. 写入 `semantic_overlay_root` 同一文件系统下的
   `.staging/<source_snapshot_content_hash>/<publish_attempt_id>/`；
2. 完成所有文件、canonical hash、引用、base fingerprint 和 completeness 校验；
3. `fsync` 文件与 staging 目录；
4. 原子 rename 到上述外置 derived overlay 目标，随后 `fsync` 父目录；
5. 若目标文件系统不能保证目录原子 rename，则先完整落盘，最后原子发布 commit marker；
6. 崩溃残留只能作为 staging/draft 恢复或清理，reader 不得将其视为已发布集合。

状态至少包括：

```text
draft | complete | partial | invalid | superseded
```

当前 GUI 产生的 set 初始为 `draft`，训练/导出工具不得把 draft 当作 complete。后续发布器仍应保证
`draft` 只存在于 staging；`superseded` 通过后继 annotation set 的 append-only supersession
record 表达，不回写旧 manifest。训练视图默认只接受 `complete`；`partial` 必须由训练 profile
显式白名单，并记录接受原因和缺失范围；`draft/invalid/superseded` 禁止进入新训练 artifact。

### 3.3 Rebuildable training view

训练视图通过 `dataset_interface` 和现有 `utils.dataset_postprocess` 扩展生成，不新增生产入口：

```text
raw facts
  + chosen annotation set
  + split manifest
  + task/embodiment profile
  -> nominal / recovery / excluded / variant-filtered views
```

首版训练视图只允许从 episode 的 `disposition`、既有质量/完整性事实和 task profile 的 `variant_id`
重建过滤与采样 mask。`reason_code` 默认只用于统计；relative action、action chunk、sample weight
仍可由已有训练管线派生，但不把 provenance、review 状态或 variant 选择作为模型输入。任何扩展字段都
必须在训练 profile 中显式 allowlist，并能由固定输入和版本确定性重建。

### 3.4 落地模块与依赖方向

第一阶段批准 pure contract、兼容 reader、外部 sidecar store，以及一个 default-off 的 RM65/UMI 共用
GUI 适配器；该适配器只消费 episode finalize 结果和人工一次性选择，不进入控制权/原始 writer。
目标布局与当前实现对应为：

```text
dataset_contracts/
  semantic_annotations.py       # 研究预留：完整 state/event/evidence 合同
  counterfactual.py              # 研究预留：campaign/group/invariant 合同
  plus_episode_curation.py       # 首版 disposition/reason/variant 最小合同
  annotation_manifest.py         # annotation set、identity/ref 合同
  annotation_identity.py         # 外部 annotation identity 纯合同与 domain hash

dataset_interface/
  episode_curation.py          # 缺失 sidecar 仍返回 None 的兼容 reader
  annotations.py                 # annotation set 验证、join、查询
  v23_plus_source.py             # finalized v2.3 read-only source adapter
  v24_plus_source.py             # native/compatibility v2.4 read-only source adapters
  semantic_training_view.py      # 最小 curation/variant 训练视图

utils/dataset_postprocess/semantic_annotations/
  publish.py                     # 外置 overlay 事务 publisher
  validate.py                    # identity/episode/curation 验收
  episode_curation.py            # 一次 post-episode 复核或离线补录
  split.py                       # 默认 episode split；关系子集再启用 component split
  profiles/                      # 可选 task/variant profile，不定义全局 ontology

tests/
  test_semantic_annotation_contracts.py
  test_semantic_annotation_source_profiles.py
  test_semantic_annotation_publication.py
  test_plus_episode_curation_contract.py
  test_plus_training_view.py

teleop_runtime/collection/
  episode_curation.py          # default-off sidecar store/session
  umi/controller.py            # Plus GUI 旁路接线
teleop_runtime/gui/
  calibration_web_ui.py        # 共用 variant/review panel
  real_teleop_control.py       # RM65 event/session adapter
```

第一阶段依赖方向固定为：

```text
offline postprocess + dataset_interface
        -> dataset_contracts

finalized v2.3 / v2.4 native commit or explicit v2.3 compatibility view
        -> read-only source adapter
        -> external immutable overlay
```

额外边界：

- `dataset_contracts` 不导入 runtime、hardware、GUI、postprocess；
- `dataset_interface` 不启动设备、不拥有在线 episode lifecycle；
- annotation contract/postprocess 不导入 RTC worker、timed plan、producer admission 或
  `robot_motion.deploy`；
- 离线 detector 不放入 `teleop_runtime/debug`，也不在控制进程中运行；
- 不创建新的 production entrypoint，离线能力扩展现有 `utils.dataset_postprocess` CLI；
- `control_station` 暂不作为该 feature 的依赖，因为 `teleop_runtime` 不得反向导入应用叶；
  后续 composition root 迁移必须另立架构任务。

V2.5 才可以讨论 `teleop_runtime/collection/semantic_annotations/`、细粒度 RM65/UMI semantic bridge、
V2.5 在线 GUI panel 或 ControlTrace bridge；当前 Plus curation panel 不属于该 semantic panel。即便进入 V2.5，这些能力也不得堆进 `debug` 入口，且仍需 architecture/RTC
owner 单独批准。实际创建顺序遵循第 18 节；首批不得为了完整目录树预建空文件或修改 package export。

### 3.5 V2.5 预留：在线 feature 开关与零干扰合同

本节全部属于第二阶段，**不是 V2.3 Plus / V2.4 Plus 的实现内容或验收依赖**。第一阶段不注册
以下 V2.5 参数，因而也不存在第一阶段的 semantic online `off/shadow/record` 运行模式；其零干扰通过“不修改、
不导入、不接线现有采集链”直接成立。

统一 master 参数规划为：

```text
--semantic-annotation-mode off|shadow|record
```

该参数只控制未来 RM65/UMI **在线采集集成**。V2.5 开工前必须重新核对主架构、RTC 和当时的
canonical entrypoint，并通过单独合同评审；本文仅冻结最低安全语义。若未来实施，默认值必须保持
`off`，不得因为存在 campaign config、overlay 目录或 ControlTrace 而自动启用。

模式语义：

| mode | 允许行为 | 禁止行为 |
|---|---|---|
| `off` | 仅解析默认值 | 不构造组件、不读 config、不创建 UUID/线程/文件/GUI、不改 metadata |
| `shadow` | 显式 scratch root 中生成诊断/candidate，允许内存统计 | 不写 dataset、不改 episode lifecycle、不弹 post-review 阻塞正式流程 |
| `record` | 写版本化 annotation sidecar，显示 pre/post episode UI | 不改 row/action/state，不拥有 capture admission，不影响已有 episode 成败 |

配套参数：

```text
--semantic-annotation-campaign-config PATH
--semantic-annotation-profile PATH
--semantic-annotation-output-root PATH
--semantic-annotation-control-trace-bridge off|reference
```

校验规则：

- `mode=off` 时传入任何配套参数都报 CLI error，避免用户误以为已经记录；
- `shadow|record` 必须显式提供 campaign config；
- `record` 必须显式提供 annotation profile；
- `shadow` 必须显式提供 output root；该 scratch root 必须与 dataset root、源码树分离，
  不能默认落到 `teleop_runtime/debug`，也不能从 dataset root 推导；
- `record` 对 v2.3/v2.4 都固定写显式 derived overlay root，禁止写 source dataset 或 sealed
  capture。两者都不得写 Parquet/media；该模式传任意未由 identity profile 验证的
  `--semantic-annotation-output-root` 必须报错，避免旁路发布；
- ControlTrace bridge 默认 `off`；选择 `reference` 时必须发现完整、正式绑定的 trace，
  否则 feature 自身 fail-closed，但不得使既有 dataset episode finalize 失败；
- 配置/合同错误应在设备初始化和 episode start 之前暴露；
- feature 运行期失败只把 annotation set 标为 incomplete/invalid，不能请求 Hold、停止运动、
  pause 原数采、丢弃原 episode 或改变其 `collection_integrity_valid`。

#### `off` 模式的严格等价要求

不带参数或显式 `--semantic-annotation-mode off` 时：

- RM65 与 UMI 不能因本 feature 额外生成 `dataset_instance_uuid`、`episode_uuid` 或
  annotation metadata；已有独立能力若已生成 identity，保持原行为；
- 不导入重型 VLM/视觉/标注依赖；
- 不新增 thread/process/queue/shared-memory slot；
- 不修改 GUI DOM、快捷键、脚踏映射或启动时长；
- 不改变 writer config、row schema、Parquet columns、episode summary 和目录结构；
- 不改变 UMI snapshot acceptance、`N -> N-1` alignment 或 rejection count；
- 不改变 RM65 control/sample/qref/qcmd 频率、pause、finalize 和 overload 行为；
- `validate-v23` 对现有数据输出保持一致；
- CLI help 可以显示 future feature 参数，但 canonical 正式命令不包含它们。

#### `shadow` 模式的隔离要求

- 使用独立 bounded queue 和独立输出根；
- queue overflow 只增加 `annotation_drop_count`；
- 不复用正式 writer queue、process shared slots 或 finalize barrier；
- 不写 dataset root；
- 不阻塞 control/sample thread；
- 关闭失败不改变原数据集 finalize 结果。

#### `record` 模式的隔离要求

- annotation sink 与 LeRobot writer 生命周期从属但相互独立；
- 原 episode 成功 finalize 后，即使 annotation publish 失败，原 episode 仍保持成功；
- annotation manifest 单独记录第 3.2.2 节状态；staging `draft` 和已发布
  `complete/partial/invalid/superseded` 不影响原 episode 状态；
- post-episode review 可以影响 annotation curation/disposition，不能反写 `collection_integrity_valid`；
- 任一额外 CPU/IO 预算必须通过正式长程 A/B 验收后才能进入推荐命令。

在该 feature 被正式验收前，`docs/real/teleop_args.sh`、正式 position/impedance 配方和 UMI
默认 config 不加入上述参数。测试和小规模 campaign 必须在命令行显式传入 mode。

## 4. 稳定 annotation identity 是 P0 前置条件

当前 `episode_index` 会在 GUI 删除 episode 后连续重编号，ControlTrace UUID 又不是 RM65/UMI
通用事实。为满足 Gate 0，首版明确选择“外部稳定映射”，不修改正式 writer 身份合同。

### 4.1 Plus v1：外部稳定映射

每个已 finalize 的 source episode 使用：

```text
source_dataset_base_fingerprint
source_episode_manifest_record_hash
legacy_episode_index
```

组成 `AnnotationEpisodeLocator`，再用 domain-separated canonical hash 生成
`annotation_episode_id`。其中：

- base fingerprint 遵循第 3.2.1 节，表示 append-stable lineage；annotation set 另绑定显式
  source snapshot content hash，两者都排除 `meta/annotations/**`；dataset 后续 append 不改变
  已有 episode identity；
- episode manifest record hash 来自该 episode 在正式 manifest 中的规范化记录；
- legacy index 只是 locator 的显式组成部分，不能单独作为身份；
- 若 source episode 有已 commit 的 ControlTrace `episode_uuid`，仅作为可验证 auxiliary ref，
  不替代上述通用 locator；
- V2.5 若实现 online record，可先使用随机 `annotation_attempt_id` 写 staging；只有原 episode 成功 finalize、
  manifest record 可读后，才能绑定并发布 `annotation_episode_id`；
- discarded/未 finalize episode 不发布正式 annotation identity。

因此，无论数据来自当前还是历史 RM65/UMI，都可以在事后以相同规则建立
稳定 annotation identity，而无需改变原数采。

identity profile 必须显式区分：

```text
v23_compat_content_locator_v1
v24_committed_capture_membership_v1
v24_compatibility_content_locator_v1
```

- 当前 RM65/UMI v2.3 使用前者；
- `v24_committed_capture_membership_v1` 只能引用主架构已验证的 native v2.4
  `episode_uuid + capture_hash/commit`；
- `v24_compatibility_content_locator_v1` 只能引用 v2.4 reader 明确声明的
  `v2.3_single_action_row` compatibility view，不携带 native UUID/commit 字段；
- 本规划不得为 native v2.4 再生成竞争性的 episode UUID、capture hash 或 membership lifecycle；
- 三种 profile 都可生成 overlay-local `annotation_episode_id`，但 manifest 必须保留 source identity
  profile、兼容来源和完整 parent binding，不能只存一个无类型字符串。

### 4.2 未来核心 UUID 的边界

所有正式数采的 `dataset_instance_uuid` / `episode_uuid` 属于主架构 native v2.4 capture 主线，
必须由其核心数据合同、writer/maintenance/resume 评审和 production gate 落地；它不属于本 future
feature 首版，也不得借 `semantic-annotation-mode` 隐式上线。未来该能力正式可用时，本规划只
消费 core UUID/capture commit，并发布 compat locator -> native membership 的显式 mapping，不能
静默换主键。

### 4.3 维护约束

- 未发布 staging 可以随失败 attempt 清理，但不能修改 source dataset；
- 已发布 annotation set 或 committed ControlTrace 存在时禁止原地删除/压紧；
- 维护必须生成派生 dataset、新 base fingerprint 和显式 old-locator -> new-locator mapping；
- annotation 引用统一使用 `annotation_episode_id`；不得使用文件名或裸 index 猜 episode；
- reader 必须重新计算 locator 并拒绝 stale/missing/mismatched binding。

## 5. Plus v1 最小增广合同（当前实现）

以下是第一阶段唯一需要实现、复核和进入训练视图的业务信息。它不定义任务本体，不要求操作员
判断 strategy、interaction、geometry、phase 或事件边界。所有记录绑定已 finalize/commit 的 source
episode，写入外置 overlay；缺失就写 `unknown` 或省略可选字段，不回写原始 Parquet、媒体或
`episodes.jsonl`。

### 5.1 EpisodeCuration：一次确认完成样本分流

```json
{
  "schema": "dexteleop_plus_episode_curation_v1",
  "annotation_episode_id": "ann_ep_7b2f...",
  "task_profile_id": "cable_clip_v1",
  "disposition": "keep_recovery",
  "reason_code": "operator_recovery",
  "variant_id": "approach_left",
  "review_status": "confirmed"
}
```

`disposition` 固定为：

```text
keep_nominal | keep_recovery | exclude | unknown
```

它是训练数据的人工整理意见，不是自动计算的 task outcome。`keep_nominal` 表示可作为常规示范，
`keep_recovery` 表示有价值但应与常规示范分视图，`exclude` 表示不进入训练视图，`unknown` 表示未
复核。已有 `summary.trajectory_quality`、`force_collision_label`、`pose_quality`、pause/rejection
和 writer integrity 事实继续由原链路提供；Plus 不复制、覆盖或把它们重新命名为 `disposition`。
这里的 `exclude` 只表示训练视图排除，不等于既有采集链的 discard；只有已按原链路 finalize/commit 的
source episode 才能产生这条 curation 记录。

当 `disposition` 为 `keep_recovery` 或 `exclude` 时，`reason_code` 可填写以下有限枚举：

```text
task_failure | operator_recovery | system_issue | safety_abort | other | unknown
```

`reason_code` 只用于筛选、统计和抽样，不能替代已有系统 reason，也不允许自由文本。若操作员没有
足够依据，写 `unknown`。

`task_profile_id` 和 `variant_id` 仅在 task profile 明确声明时出现。`variant_id` 必须来自有限枚举，
例如预先登记的对象实例、接近方向或采集策略；它表示采集前意图，不等于执行后已验证的变化事实。
没有 variant profile 时不生成字段，也不要求操作员手填“无”。

### 5.2 ReviewStatus：发布审计而非模型标签

`review_status` 只描述 overlay 记录本身：

```text
confirmed | provisional | conflicted | rejected | unknown
```

首版训练视图只接受 `confirmed`；如确需使用 `provisional`，必须由训练 profile 显式允许并记录
缺失范围。它不表示 task success probability，不提供未经校准的 confidence 分数。

### 5.3 研究预留，不进入 Plus v1

以下内容全部移到 V2.5 或独立 Research-Preview：粗阶段和连续 segment、interaction state/event、
稀疏 timestamp event、`SemanticEvidenceBundle`、campaign/group/invariant、association-graph
split、ControlTrace cause chain、`observed_recovery_scope` 和 `minimal_replan_level`。它们即使已有
纯 Python 草案，也不能要求每个 episode 生成 JSONL，不能成为 Plus v1 发布或训练的前置。

## 5R. 研究预留语义合同（V2.5）

以下名称是规划中的 draft schema；实现前放入 `dataset_contracts`，通过 canonical encoding、round-trip 和 invalid-case 测试后才成为正式合同。

### 5R.1 CampaignSpec：采集前保存不可事后恢复的信息

反事实元数据提前到 P0.5。每次 campaign 预先定义：

```json
{
  "schema": "dexteleop_campaign_spec_v1",
  "campaign_id": "cable_clip_cf_20260722",
  "campaign_type": "controlled_counterfactual",
  "task_template_id": "cable_clip_v1",
  "allowed_strategies": [
    "right_stabilize_left_insert",
    "left_stabilize_right_insert"
  ],
  "safety_review_ref": "review_cable_clip_cf_v1",
  "annotation_profile_id": "cable_clip_interaction_v1",
  "source": "campaign_author",
  "version": "1"
}
```

不得把 `safety_review_ref` 当成运行时运动授权。它只证明 campaign 设计经过审查；每次真机运行仍遵循现有 safety/preflight/motion authority。

### 5R.2 CounterfactualGroup：episode 等价类

数据基本单位扩展为具有“计划保持/计划变化”的 episode group。采集设计与执行事实必须分开：

```json
{
  "schema": "dexteleop_counterfactual_group_v1",
  "counterfactual_group_id": "clip2_group_004",
  "campaign_id": "cable_clip_cf_20260722",
  "shared_initial_state_id": "scene_snapshot_031",
  "initial_state_evidence_level": "visual_snapshot_match",
  "intended_preserved": [
    "task_goal",
    "strategy",
    "role_assignment",
    "target_clip"
  ],
  "intended_varied": [
    "approach_direction",
    "wrist_orientation"
  ],
  "expected_invariants": [
    "contact_sequence",
    "final_object_relation"
  ]
}
```

禁止使用含义模糊的 `preserved_variables` 字段：`intended_preserved` / `intended_varied`
只描述 campaign 设计，不证明执行中实际保持或改变。

group 收集后另行发布 assessment：

```json
{
  "schema": "dexteleop_counterfactual_group_assessment_v1",
  "counterfactual_group_id": "clip2_group_004",
  "member_annotation_episode_ids": ["ann_ep_a", "ann_ep_b"],
  "observed_changed": ["wrist_orientation", "contact_point"],
  "invariant_checks": [
    {
      "variable": "strategy",
      "result": "verified",
      "evidence_bundle_ids": ["evidence_group_004_strategy"]
    },
    {
      "variable": "contact_sequence",
      "result": "violated",
      "evidence_bundle_ids": ["evidence_group_004_contact"]
    },
    {
      "variable": "final_object_relation",
      "result": "not_observable",
      "evidence_bundle_ids": []
    }
  ],
  "review_status": "confirmed"
}
```

`invariant_check.result` 只允许：

```text
verified | violated | not_observable | unknown
```

只有 `verified` 能进入 counterfactual consistency loss；`violated` 是实际偏离证据，
`not_observable/unknown` 必须被 mask。`observed_changed` 表示观测到的变化，不声明变化由预定
perturbation 唯一造成，也不证明未列出的变量保持不变。

`initial_state_evidence_level` 必须显式：

```text
not_verified
operator_reset_claim
visual_snapshot_match
tracked_object_state_match
```

只有证据等级相同且验证通过的 episode 才能宣称共享初始状态。相同任务名、相同桌面或操作者口头复位均不足以自动升级为 `tracked_object_state_match`。

批准三类首批 group：

1. 同策略、不同低层实现；
2. 同任务、不同策略；
3. 成功—安全偏差—恢复。

偏差必须经过单独真机安全审查；不允许为了制造 recovery 数据绕过 collision、joint limit、freshness 或 motion authority。

### 5R.3 EpisodeAnnotation：结果不是文件状态

研究用完整 episode annotation（不属于 Plus v1 最小字段）：

```json
{
  "schema": "dexteleop_episode_annotation_v1",
  "annotation_episode_id": "ann_ep_7b2f...",
  "campaign_id": "cable_clip_cf_20260722",
  "counterfactual_group_id": "clip2_group_004",
  "task_template_id": "cable_clip_v1",
  "scene_instance_id": "board_a_layout_03",
  "planned_strategy": "right_stabilize_left_insert",
  "executed_strategy": "right_stabilize_left_insert",
  "role_assignment": {"left": "manipulate", "right": "stabilize"},
  "outcome": "success",
  "completion_fraction": 1.0,
  "quality": "clean",
  "failure_reasons": [],
  "contains_search": true,
  "contains_correction": false,
  "operator_reviewed": true,
  "source": "operator_post_episode_review",
  "review_status": "confirmed"
}
```

枚举：

```text
outcome:
  success | partial_success | failure | aborted | unknown

quality:
  clean | usable_with_warning | recovery_example
  negative_example | system_invalid | unknown
```

`saved != success`。若操作员跳过确认，必须写 `unknown`，不能默认成功。

`review_status` 使用 `confirmed | provisional | conflicted | rejected | unknown`，不使用未经
校准的统一浮点 confidence。`completion_fraction` 是明确来源的任务进度判断，也不能解释为
概率。

`system_invalid` 与 task failure 严格分离：相机断流、标定错误、未来快照、writer 丢数据不是机器人负样本。

### 5R.4 SemanticSegment：任务阶段

segment 使用半开区间，且不能跨 continuity segment：

```json
{
  "schema": "dexteleop_semantic_segment_v1",
  "segment_id": "ann_ep_7b2f_seg04",
  "annotation_episode_id": "ann_ep_7b2f...",
  "continuity_segment_index": 0,
  "start_frame_index": 83,
  "end_frame_index_exclusive": 121,
  "start_timestamp": 1780000005.12,
  "end_timestamp": 1780000007.65,
  "subtask": "seat_cable_in_clip_2",
  "primitive": "insert",
  "target_entity": "cable_tip",
  "reference_entity": "clip_2",
  "behavior_role": "intentional_search",
  "segment_outcome": "success",
  "semantic_group_id": "ann_ep_7b2f_subtask_clip2",
  "evidence_bundle_ids": []
}
```

任务 phase 与 interaction state 不强制共享边界。interaction state 使用独立半开区间：

```json
{
  "schema": "dexteleop_interaction_state_interval_v1",
  "state_interval_id": "ann_ep_7b2f_state09",
  "annotation_episode_id": "ann_ep_7b2f...",
  "continuity_segment_index": 0,
  "start_timestamp": 1780000005.91,
  "end_timestamp": 1780000006.44,
  "interaction_state": "compliant_search",
  "evidence_bundle_ids": ["evidence_ann_ep_7b2f_state09"]
}
```

同一 state track 内区间不得重叠或跨 continuity；已复核但无法判断的时段写 `unknown`。未标注
区域与 `unknown` 必须区分，不能为追求稠密覆盖自动补齐状态。

primitive 是 V2.5 研究中的辅助描述，不是 Plus v1 字段。V2.5 词表保持小规模：

```text
move | reorient | approach | grasp | release | lift | transport
push | pull | press | slide | pivot | insert | extract | search | hold
```

行为角色：

```text
nominal | intentional_search | local_correction | recovery | retry
strategy_switch | hesitation | idle | accidental | unknown
```

### 5R.5 SemanticEvent：瞬时事件与稀疏时间锚点

`SemanticEvent` 是 derived annotation overlay 类型，不重定义主架构 native v2.4 `TaskEvent` 或
ControlTrace event。source capture 已有对应正式 event 时只保存 typed parent ref；没有时才保存明确
标为 operator/offline/detector source 的 annotation claim。

`interaction_state` 只能写在 state interval record；瞬时变化写为 event。interaction event
V2.5 研究固定为：

```text
contact_onset | contact_loss | slip_onset | jam_onset
grasp_established | release_begin | release_complete
```

任务/安全/审计事件单独使用 `event_family=task|safety|operator`，例如：

```text
subgoal_reached | retry_begin | recovery_begin | strategy_changed
collision | excessive_force | scene_changed | operator_mark_invalid
```

interaction event 必须允许记录 `pre_state` / `post_state` 和对应证据；未知状态显式写
`unknown`，不能把 `stable_grasp` 等状态冒充瞬时事件。在线只写轻量 anchor，详细类型事后
确认；在线按钮不得改变 motion authority、arm mode、设备命令或 writer admission。

### 5R.6 SemanticEvidenceBundle：有证据的语义假设

项目已有 `RM65ExecutionEvidenceBundle`，因此本规划使用独立名称 `SemanticEvidenceBundle`，避免与执行安全证据混淆。

每个关键 event/segment claim 保存：

```json
{
  "schema": "dexteleop_semantic_evidence_bundle_v1",
  "evidence_bundle_id": "evidence_ann_ep_7b2f_evt04",
  "claim": {
    "claim_type": "interaction_state",
    "value": "stable_grasp"
  },
  "anchor_timestamp": 1780000006.41,
  "window": {
    "start_timestamp": 1780000005.91,
    "end_timestamp": 1780000007.21,
    "continuity_segment_index": 0
  },
  "supporting_evidence": [
    {
      "evidence_type": "tactile_onset",
      "modality_ref": "tactile:left:sequence:314",
      "detector_id": "dmtac_contact_candidate_v1",
      "calibration_ref": "session_031_baseline"
    },
    {
      "evidence_type": "gripper_measured_closed",
      "row_ref": "annotation_episode_id:frame:102"
    },
    {
      "evidence_type": "operator_confirmation",
      "annotation_event_ref": "review_event_007"
    }
  ],
  "contradicting_evidence": [],
  "missing_evidence": ["tracked_object_pose"],
  "semantic_source": "auto_candidate_operator_confirmed",
  "detector_version": "contact_fusion_v1",
  "detector_score": {
    "name": "contact_fusion_raw_score",
    "value": 2.73,
    "scale_id": "contact_fusion_v1_raw",
    "calibration_status": "uncalibrated"
  },
  "review_status": "confirmed",
  "evidence_completeness": "partial"
}
```

约束：

- evidence 使用引用，不复制 RGB/触觉大数组；
- 支持证据、反证和缺失证据分开；
- 自动 detector 负责物理变化和时间候选，操作员确认意图；
- `detector_score` 是算法原始输出，不宣称为概率，不同 detector/version 的 score 默认不可比；
- `review_status` 使用 `confirmed | provisional | conflicted | rejected | unknown`；
- `evidence_completeness` 使用 `complete | partial | insufficient`；
- 只有通过 Gate F 校准后，后续 schema 才可增加 `calibrated_probability`，并必须绑定
  calibration artifact、适用 operator/session/scene 域和 detector version；
- evidence window 不能跨 continuity segment；
- detector 升级生成新 annotation set，不覆盖旧结果；
- `intentional_search`、`local_correction`、`hesitation` 不能仅由轨迹曲率决定；
- V2.5 只要求关键 interaction event、group invariant check 和研究抽样具有
  `SemanticEvidenceBundle`，不强制每个普通 free-space segment 建复杂 bundle；Plus v1 不发布此文件。

## 6. V2.5 研究预留：Interaction state

以下模型层次是 V2.5 的研究主线，不是 Plus v1 的采集字段。Plus v1 只保存第 5 节的 episode curation
overlay；只有在任务 profile、标注一致性和训练收益都通过单独试验后，才扩展到 phase/state/geometry/residual。

批准使用以下模型层次作为长期研究组织主线：

```text
strategy
  -> interaction state / contact relation / constraint structure
  -> geometry / keypose / contact region
  -> control residual
```

不以无限扩张的 primitive 词表作为核心。

V2.5 `interaction_state` vocabulary：

```text
free_space
approach
contact_maintained
stable_grasp
sliding_contact
constrained_translation
constrained_rotation
compliant_search
unknown
```

V2.5 `interaction_event` vocabulary：

```text
contact_onset
contact_loss
slip_onset
jam_onset
grasp_established
release_begin
release_complete
```

state 与 event 是不同枚举、不同 record，不允许用同名字符串混写。每个任务 profile 必须声明
有限状态转移：

```text
pre_state --interaction_event--> post_state
```

并继续声明：

- 允许的 interaction transitions；
- 每个 state 的 required/supporting/contradicting evidence；
- 每个 event 的时间边界、允许 pre/post state 和证据要求；
- DMTAC baseline/calibration 需求；
- onset/offset hysteresis；
- gripper、运动、视觉和触觉的组合规则；
- operator review 项；
- unavailable modality 的退化策略。

DMTAC 禁止使用跨 sensor、跨 session 的固定裸阈值。触觉变化只能支持 contact candidate，不能单独证明稳定抓取、对象身份或任务成功。

V2.5 只选择 1 个任务建立完整 profile；后续扩展优先：

- cable clip/cable routing；
- 插入或装配；
- 双手稳定—操作任务；
- 具有多种接近策略和安全可复核恢复的任务。

## 7. V2.5 研究预留：反事实采集规则

反事实不是 Plus v1 的普遍采集要求。第一阶段只允许在已有 campaign 或研究抽样中附加
`group_id/variant_id/observed_changed/invariant_result`；不要求每次采集前填写完整的
`intended_preserved/intended_varied`，也不以反事实 group、初始状态证据或 connected-component
split 作为普通训练数据的发布门槛。以下规则仅约束真正启动反事实研究时的记录质量。

### 7.1 A 组：同策略、不同低层实现

计划保持 task、初始状态证据、strategy、role 和 contact sequence；计划改变路径、速度、腕姿、
接触点或残差。执行后必须用 group assessment 分别记录实际变化和每项 invariant 结果。

用途：只有在对应 invariant check 为 `verified` 时，才能监督 fine/geometry 变化下 strategy 和
interaction state sequence 保持。

### 7.2 B 组：同任务、不同策略

例如左右手角色互换、一次对准与接触搜索、左侧绕行与右侧绕行。

用途：避免把高层多模态误当低层噪声。

### 7.3 C 组：成功—偏差—恢复

记录而不是先验猜测：

- perturbation type/severity；
- 原计划保持的变量；
- 操作者实际改变的变量；
- correction 起止时间；
- 最终 outcome；
- system 是否正常接收和执行。

数据侧只记录带证据的实际修正范围：

```text
observed_recovery_scope:
  residual_changed
  interaction_changed
  geometry_changed
  subgoal_changed
  strategy_changed
```

该字段允许多选，表示观察到操作者实际改了什么；它不证明这些修改都是必要的，也不证明更低
层修改不足以恢复。

`minimal_replan_level` 属于研究派生标签，不属于 Plus v1 数据事实。只有模型或 oracle 逐层执行
反事实重生成、按照预注册 success criteria 验证更低层均无法恢复后，才能生成，并绑定：

```text
derivation_method
tested_lower_levels
counterfactual_rollout_refs
success_criteria_ref
review_status
```

仅凭人类示范中出现了 geometry/strategy 修改，不得推断最小重规划层级。

### 7.4 UMI feasibility

UMI group 增加：

```text
robot_feasibility:
  unchecked
  offline_ik_pass
  offline_ik_fail
  sim_collision_checked
  real_replay_verified
```

这些状态是逐级证据，不互相等价。`offline_ik_pass` 仍不能证明动态跟踪、自碰撞、环境碰撞、控制时序或真实接触稳定性。

任何 `real_replay_verified` 都属于独立真机 replay 安全流程，不能由 UMI 数采规划自动触发。

## 8. V2.5 预留：ControlTrace 与语义层的桥接边界

### 8.1 当前决策

本节只定义第二阶段的远期边界。V2.3 Plus 不依赖 ControlTrace；V2.4 Plus native path 可以把
source capture 中**已经 committed 且可验证**的 trace range 当普通 evidence reference 读取，compatibility
path 则明确标记 trace unavailable。两条路径都不得新建 bridge、cause chain、trace writer 或在线挂接，
也不得把 trace 缺失视为 Plus 失败。

ControlTrace 和 semantic annotation 保持两套合同，通过 ID/time span 引用，不合并：

```text
semantic intent/claim
    -> source/candidate/reference
    -> admission/decision
    -> dispatch/receipt
    -> measured response
    -> physical interaction evidence
    -> semantic outcome
```

ControlTrace 可以证明控制事实，不能证明世界语义。semantic evidence 可以描述 world effect，不能反推未持久化的 admission/dispatch 事实。

### 8.2 启用前置条件

V2.5 cause-chain bridge 只有在以下条件满足后才可进入正式 annotation：

1. canonical RM65 GUI 正式启用并自动挂接 formal episode sink；
2. profile、WAL、seal、capture commit 和 episode UUID 在 production path 验收；
3. out-of-session DMTAC dispatch 覆盖问题闭合；
4. UMI 若需要桥接，先定义其适用/不适用 ControlTrace profile；
5. annotation reader 能区分 trace unavailable 与 trace incomplete；
6. committed trace maintenance 继续 fail-closed。

### 8.3 CauseChain 草案

条件满足后，V2.5 可通过独立 schema/version 增加：

```json
{
  "schema": "dexteleop_semantic_cause_chain_v1",
  "cause_chain_id": "cause_ann_ep_7b2f_004",
  "semantic_event_id": "ann_ep_7b2f_evt07",
  "control_trace_range_ref": "control_trace_episode_uuid:journal:314-328",
  "operator_intent": "local_correction",
  "admission_result": "accepted",
  "dispatch_result": "received",
  "measurement_effect": "tcp_shifted_4mm",
  "world_effect": "slip_stopped",
  "semantic_result": "recovery_success"
}
```

字段只能由对应证据产生；缺失必须明确，不允许从最终动作反推整个 cause chain。

## 9. 第一阶段最小附加工作流

### 9.1 Task profile 准备（可选）

若一个任务确实需要区分有限的对象实例、接近方向或采集策略，单独准备只读 task profile，声明
`task_profile_id` 和 `variant_id` 枚举。profile 不传入 RM65/UMI 控制链，不要求所有任务都有 variant，
也不允许现场自由输入新字符串。IP、TCP、相机 serial、layout 或 calibration hash 不重复手填；source
finalize/commit 后仍由 read-only adapter 从正式 manifest/config binding 读取。

### 9.2 Episode 开始前（仅在 Plus 明确启用时）

有 task profile 时，操作员在独立采集表或不持有控制权的小工具中选择一次 `variant_id`；没有 profile
或未选择时不写该字段。该记录在 finalize 后按显式 episode locator 绑定，绑定不清晰则保持 unknown。
RM65/UMI 当前共用 GUI 在显式 Plus 开关下提供同一选择；选择作为 event payload 传到 runtime session，
不创建新的 UUID，不改 task、round-trip、raw metadata 或 writer metadata，也不改变 episode 是否开始。

### 9.3 Episode 期间

操作员完全按既有 V2.3/V2.4 数采流程工作。第一阶段不增加逐帧快捷键、脚踏、语音、event anchor 或
candidate detector；Plus 不读取/修改控制状态，不参与 pause、hold、admission、finalize 或 discard
决策。启用时仅保留一个 GUI event queue 项和独立 sidecar 写线程，线程只在 finalize 后人工提交时运行，
不向控制/采样 writer 提交任务。

### 9.4 Episode finalize/commit 后

已冻结 source episode 后，GUI 的轻量 post-review 窗口或离线批处理只要求一次确认：

1. `disposition`：`keep_nominal`、`keep_recovery`、`exclude` 或 `unknown`；
2. 对 recovery/exclude 可选一个 `reason_code`；
3. 若采集前已有 profile，显示并确认对应 `variant_id`。

RM65/UMI GUI 直接提供一次提交；离线 reader 也可补录。结果写入外置 `episode_curation.jsonl`，不回写 `episodes.jsonl`、Parquet 或媒体。没有响应、
窗口关闭或 review 工具失败时，原 episode 状态不变，annotation 记录为缺失/`unknown`，不阻塞验收。
同一 episode 的修订发布新 annotation set，不能原地覆盖旧记录。

### 9.5 失败数据策略

- 现有 clean/failure episode 的保存或 discard 规则完全不变；Plus GUI 只做 finalize 后 curation，不新增在线 outcome 判定；
- 只有 source 已按原链路保留，离线 overlay 才能把它划入 nominal/recovery/exclude 视图；
- `reason_code=system_issue` 只作为操作员补充筛选码，系统完整性仍以原有事实为准；
- 原始事实保留，训练集通过 copy-on-write 生成。

## 10. V2.5 研究预留：离线多模态证据时间轴

Plus v1 不建设通用 timeline、phase 或 candidate detector。本节只属于 V2.5：只有人工语义边界成本
已经成为瓶颈，且 episode curation 的筛选收益得到验证后，才增加自动候选。

### 10.1 可用事实源

| 候选边界/证据 | 当前可用信号 |
|---|---|
| gripper open/close | target、measured scalar、joint timestamp |
| contact candidate | calibrated tactile statistics、snapshot time |
| keypose/stationary | canonical measured TCP 速度低谷 |
| correction candidate | 曲率、方向反转、jerk、目标/actual 误差 |
| safety/IK change | pose quality、fault、reference modification reason |
| pause/recovery | continuity、pause summary、arm mode |
| UMI gap | rejection reason、相邻 row timestamp gap |
| modality desync | source/available age、sequence、frame index |
| world effect | 视频/VLM/对象 tracker，若有 |
| control fact | ControlTrace，若正式存在且完整 |

### 10.2 对齐规则

- 先在原始 timestamp 上建立 window，再映射 row；
- 使用半开区间；
- 不跨 episode/continuity；
- UMI 边界同时检查 observation time 与 next-action source time；
- 大 gap 两侧默认拆段，不插值伪造；
- 视频离线语义可使用完整 evidence window，但必须标为 offline evidence，不能冒充在线可观测事实；
- 同一个 tactile snapshot 被多 row 复用是合法的，不按 row 数重复计作独立接触证据。

### 10.3 标注原则

```text
自动系统负责发现时间和物理变化
操作员负责确认意图、对象和任务结果
```

自动 candidate 未确认时可以用于检索和弱监督，但不能升级成高置信人工真值。

## 11. 数据验收门

### 11.0 Gate 0：现有数采零干扰

第一阶段以结构隔离证明零干扰，优先级高于 annotation 功能正确性：

- RM65/UMI canonical 命令、parser、config 和帮助文本均不因 Plus 改变；
- 不修改或导入现有 raw writer、episode lifecycle、control loop、RTC 和 ControlTrace producer；
- Plus GUI adapter、事件队列和 sidecar 后台线程只在显式开关下构造，不创建 UUID、不写 dataset metadata，
  不参与 raw writer/control dispatch；
- source dataset/capture 只读，before/after member bytes/hash 与 `validate-v23` 结论不变；
- overlay 发布失败只影响外置 annotation set，不进入现有 `collection_integrity`；
- V2.3 Plus 与 V2.4 Plus 的所有 fixture 在 `/tmp` 或测试临时目录运行，不启动设备。

Gate 0 未通过时，不得发布 Plus overlay。第一阶段不进行 feature-on 真机 A/B；未来 V2.5 如需
在线接线，必须在当时重新定义并通过 off/on 长程 A/B，不能继承本 Gate 的授权。

### 11.1 Gate A：合同与因果事实

- layout/hash 自洽；
- pose-frame/hash 一致；
- causal snapshot/hash 一致；
- episode/parquet 计数一致；
- action/state 副本一致；
- 外置视频/触觉引用可解析；
- calibration/pose contract 未混写；
- `validate-v23` 通过。

### 11.2 Gate B：采集完整性

RM65：

- collection integrity valid；
- submitted/completed 一致；
- actuator trace drop/error 为 0；
- reference 未 stale；
- writer overload/pause 可解释；
- actual measured TCP 因果有效。

UMI：

- `source_sample_count = written_rows + 1`；
- terminal source drop 合同成立；
- attempt/accepted/rejection 算术一致；
- 双腕 video 和双侧 tactile 完整；
- calibration/TCP extrinsic 有效；
- qpos invalid mask 符合 arm-free 语义。

### 11.3 Gate C：Annotation integrity

- annotation set manifest/hash 有效；
- base fingerprint 与 snapshot content hash 重算一致，且两者输入排除
  `meta/annotations/**`；
- 所有 annotation identity、row/modality refs 可解析；
- staging/draft 不会被 reader 当成已发布集合；
- 发布满足原子 rename 或 commit-marker-last 合同；
- `complete/partial/invalid/superseded` 训练可见性符合第 3.2.2 节；
- `episode_curation.jsonl` 中的 `disposition/reason_code/variant_id` 满足合同枚举和条件约束；
- `review_status`、task profile 和 source binding 完整；
- source binding 与 review provenance 完整；
- Plus v1 不要求 event/relation/phase；若研究记录存在，按独立 schema 验证，不影响普通 overlay；
- 修订通过新 annotation set 发布。

### 11.4 Gate D：任务质量

```text
nominal view:
  disposition=keep_nominal

recovery view:
  disposition=keep_recovery
  reason_code 可选，仅作筛选/统计

excluded view:
  disposition=exclude

unknown:
  未复核或证据不足；不能自动当作 nominal、recovery 或 exclude
```

既有 `trajectory_quality`、媒体完整性、暂停/拒绝和 writer integrity 仍是独立的验收事实；Plus
不得用一次人工 disposition 覆盖它们。

### 11.5 Gate E：训练视图

- 只使用 layout 声明的有效 action/state；
- DMTAC 按每侧 gripper1，不消费 padding；
- normalization 排除 invalid qpos、padding 和 chunk pad；
- window 不跨 episode/continuity；
- relative pose anchor 使用 measured actual TCP；
- UMI/RM65 显式携带 embodiment/source/alignment；
- audit/debug 字段默认不进入模型；
- annotation set 和 split manifest hash 固化进 artifact metadata。

### 11.6 Gate F：标签稳定性

首版只对 episode curation 记录报告：

- inter-annotator agreement；
- intra-annotator repeatability；
- `disposition` 和条件性 `reason_code` 的复核一致性；
- `variant_id` 的枚举命中率与遗漏率（仅配置了 variant profile 的任务）。

detector precision/recall、phase boundary error、calibrated probability、跨 sensor/session 稳定性和
modality degradation 留到 V2.5 timeline/candidate profile，不阻塞 Plus v1。

规划不预先固定全任务统一阈值。每个 task profile 先建立 baseline，再批准阈值；示例数字不能直接升级成通用验收标准。

### 11.7 Gate G：防数据泄漏

首版默认按完整 `annotation_episode_id` 切分，禁止按 row 或短 segment 随机切分。同一 episode
下的所有 row/window/derived segment 必须在同一 split。只有研究数据实际提供了 group/scene/session
关系时，才启用关联图：

```text
episode
  <-> counterfactual_group
  <-> shared_initial_state
  <-> real_contact_sequence
  <-> scene/operator/collection_session/calibration_session（按评估策略启用）
```

研究关系形成无向 edge/hyperedge：

- `counterfactual_group_id`；
- `shared_initial_state_id`；
- 同一真实 contact sequence；
- 同一 scene instance，若任务要求 scene generalization；
- operator/session/calibration group，按评估目标选择隔离。

启用关联图时，先对所启用关系求 connected components，再以整个 component 作为最小切分单位。session、operator
和 calibration 不能无条件连边，否则可能把整个 campaign 连接成单一分量；启用哪些 edge family
必须由预注册 evaluation isolation policy 决定。

同一 `annotation_episode_id` 下的 row/window/segment 在建图前先归并为一个 episode node，任何
情况下都不得拆到不同 split。

split manifest 必须记录：

- graph rule/version 和启用的 edge families；
- relation source annotation set/hash；
- 每个 component ID、成员 episode 和 relation evidence；
- component -> split assignment；
- canonical assignment hash；
- component 规模导致目标比例不可满足时的显式偏差报告。

首版没有 relation 记录时不强行创建空 graph；一旦启用 graph，任一 connected component 跨 split
都是 Gate G 失败。

## 12. 两阶段实施里程碑

### 12.1 第一阶段：V2.3 Plus / V2.4 Plus（重新评审后的规划重点）

第一阶段当前交付四项：

1. 不修改正式 writer 的外部 `annotation_episode_id`、source binding 和 append-only overlay；
2. 一次轻量 post-review 得到 episode `disposition`，必要时补充 bounded `reason_code`；
3. 对确有需要的 task profile，采集前一次选择有限枚举 `variant_id`，并生成可重建的筛选/分组视图。
4. RM65/UMI 共用 GUI 的 default-off 旁路接线：开始前选择 variant，保存后提交 disposition/reason；
   sidecar 写入失败不改变原始 episode 结果。

第一阶段不以以下内容为完成条件：

- 逐帧在线快捷键、脚踏、语音、candidate detector 或多模态 timeline；（Plus 的两次 episode 级 GUI 操作已在上一项交付。）
- 所有任务统一 ontology，或 primitive/role/subtask 三套并行标签；
- 粗阶段、interaction state/event、稀疏时间锚点和每段 evidence bundle；
- ControlTrace cause chain、完整 state/event/evidence bundle；
- 自动或仅凭示范派生的 `minimal_replan_level`；
- 完整 UMI/RM65 跨本体模型和反事实 consistency loss；
- 关联图 split（没有实际关系记录时只做 episode-level split）。

这些内容保留到 V2.5 或单独 Research-Preview，不得成为 Plus v1 的发布/训练前置。

#### Plus-0：共享 episode curation 合同与现状审计（已实施）

目标：把本文 draft 落成纯合同、兼容 reader 和 fixture，同时建立外部 annotation identity、base
fingerprint 与现有数采 baseline；Plus 接线仅为 default-off GUI/event adapter，不改 writer/action schema。

此前原型曾实现 `plus_minimal.py`、identity/manifest、丰富 semantic/counterfactual 合同及对应 tests；
现已全部移动到 `legacy/dataset_semantic_plus_v0/`，不再属于当前实现。若恢复本切片，必须重新确认
合同字段和与 architecture/RTC 并行主线的 owner，再从独立 fixture 开始。

规划位置：

- `dataset_contracts/episode_curation.py`（当前合同；旧原型文件名不恢复）；
- `dataset_interface/episode_curation.py`（当前缺失 sidecar 兼容 reader）；
- `teleop_runtime/collection/episode_curation.py`（当前外部 sidecar store/session）；
- `dataset_contracts/semantic_annotations.py` / `counterfactual.py`（研究预留候选）；
- `dataset_contracts/annotation_manifest.py` / `annotation_identity.py`（候选）；
- `dataset_interface/semantic_training_view.py`（候选，只输出 curation/variant 视图）；
- 对应 `tests/test_*`。

验收：

- canonical hash 稳定；
- round-trip；
- 非法 timestamp/range/ref/enum fail-closed；
- episode curation schema 可 round-trip，条件枚举 fail-closed；
- base fingerprint/snapshot hash 排除 annotations/staging/derived views；append 新 episode 只改变
  新 snapshot hash，不改变 base lineage 或旧 episode identity；成员 raw 改动能被检测；
- locator/hash 对 RM65/UMI fixtures 稳定，stale binding fail-closed；
- contract 不导入 runtime/hardware/UI；
- 不修改 `dataset_contracts/__init__.py`、`dataset_interface/__init__.py` 或任何当前入口；
- canonical 正式命令、config 和输出 byte/layout 均不变化。

#### Plus-1：V2.3 Plus read-only source profile（规划）

目标：只对已 finalize 的 v2.3 RM65/UMI fixture/副本验证外部 annotation identity、episode
curation 记录绑定和可选 variant profile；不接任何在线模式。

规划改动：

- `dataset_interface/v23_plus_source.py` 复用 `annotation_identity.py`，对 finalize 后数据计算 locator 和
  `annotation_episode_id`；
- `dataset_interface/annotations.py` 提供 reader/validation，不修改两者 writer；
- skipped review 写 `unknown`；
- 不要求普通 episode 有 campaign/group/phase/event 字段；
- episode maintenance 强制 copy-on-write 并生成 locator mapping。

验收：

- 同一未修改 source dataset 重算 base fingerprint/locator 一致；
- annotation set 新增不改变 source base fingerprint；
- copy-on-write/reindex 产生新 locator 并保留可验证 mapping；
- RM65/UMI curation 行为一致；
- UMI `N -> N-1` 不变；
- fixture 验证完全在 `/tmp` 或测试临时目录，不写 source dataset；
- 不导入 `teleop_runtime`、`control_station`、RTC 或 hardware。

#### Plus-2：V2.4 Plus native/compatibility read-only source adapters（规划）

目标：让 V2.4 Plus 自然兼容两种真实来源，同时只输出一套 Plus overlay schema：

1. **native path（优先）**：消费主架构已经正式 committed 的 v2.4 capture identity、membership 和可选
   trace reference；
2. **compatibility path（可用时）**：消费 v2.4 reader 明确提供的 `v2.3_single_action_row` compatibility
   view，用 v2.3 manifest/content locator 规则绑定 episode，但不把它包装成 native v2.4 commit。

两条路径都不补造 UUID、不写 capture、不改变 seal/commit，也不要求 ControlTrace 存在。两条路径必须
通过同一个 `PlusSourceEpisodeBinding` 进入 publisher/reader/split/training view；差异只保留在
`source_profile`、`compatibility_origin` 和 trace completeness 中。

规划改动：

- `dataset_interface/v24_plus_source.py` 提供 `bind_v24_plus_episode` 统一入口，native commit 存在时
  选择 `v24_committed_capture_membership_v1`，否则只有 caller 明确声明 compatibility-view membership
  才能选择 `v24_compatibility_content_locator_v1`；
- compatibility path 的 locator 固定写 `compatibility_origin=v2.3_single_action_row`，不携带
  `native_episode_uuid`/`capture_commit_hash`，也不通过目录扫描或 legacy index 猜测 membership；
- native path 缺少正式 commit 或 membership 不完整时仍 fail-closed；这表示 native source unavailable，
  不阻止已由 reader 证明的 compatibility path；
- trace unavailable、trace incomplete 与 trace complete 分开记录，均不反推 cause chain；
- curation/variant 继续使用 Plus-0 的同一最小合同；native/compatibility 只影响来源 binding。

验收：

- native source capture 和 sealed artifacts before/after hash 等价；compatibility view 的 source members
  同样 before/after 等价，且保留 `v2.3_single_action_row` 来源标记；
- 同一 native committed capture 或同一 compatibility view 重算 binding 稳定，source record/membership
  改变时 stale binding 被拒绝；
- native 与 compatibility 两种输入均产出同 schema 的 overlay record、reader result 和 training view；
- 不产生新的 dataset/capture/episode UUID，不建立第二套 commit；compatibility path 不声称 native UUID；
- 不导入 capture writer、ControlTrace writer、RTC 或在线 runtime；
- 没有任何可验证 membership 时明确报告 `source_profile_unavailable`，不得退回目录猜测。

#### Plus-3：外置版本化 annotation storage、reader、post-review 与最小 split

目标：只以离线 authoring 路径事务化发布、验证、读取最小 curation overlay，并按完整 episode
生成可重建训练视图。当前 Plus-3 已接入 RM65/UMI 共用 GUI；训练 view、完整事务 publisher 和 split
仍可复用该 sidecar reader，通过离线 CLI/回放 GUI 扩展，不进入 raw writer。
若研究子集已有关系记录，再单独生成 connected-component split；没有关系时不创建复杂 graph。

规划位置：

- `dataset_interface/annotations.py`；
- `utils/dataset_postprocess/semantic_annotations/publish.py`；
- `utils/dataset_postprocess/semantic_annotations/validate.py`；
- `utils/dataset_postprocess/semantic_annotations/episode_curation.py`；
- `utils/dataset_postprocess/semantic_annotations/split.py`（只做 episode-level，关系子集再扩展）；
- 首批直接调用新 submodule，不修改现有 `utils.dataset_postprocess` CLI；CLI 注册留到共享文件协调窗口。

验收：

- append-only publish；
- staging -> validate/fsync -> atomic rename 或 commit-marker-last；
- crash/retry/idempotency 和目标冲突 fail-closed；
- `draft/complete/partial/invalid/superseded` 状态与训练可见性正确；
- parent/version/hash 正确；
- stale/missing refs 拒绝；
- legacy identity mapping 可读；
- episode-level split 不拆 episode；
- 有关系记录时先求 connected components，再按 component split；
- 使用 graph 时 rule/component/assignment hash 可复现；
- copy-on-write 派生视图保留 provenance；
- finalized v2.3 与 committed v2.4 都不被原地维护；
- 默认 off 采集的数据可在 finalize 后由离线路径补录 curation；
- 不创建 V2.5 控制线程/queue/shared slot，不导入 V2.5 online runtime；Plus v1 的 sidecar save thread
  仅处理外部 JSONL，不拥有 motion/data admission。
- 没有任何 Plus 参数加入正式 RM65/UMI 配方。

### 12.2 第二阶段：V2.5（长远规划，不进入当前代码批次）

V2.5 的目标是利用在线链路的充分事实构造更细粒度因果语义，而不是继续给第一阶段 overlay
堆字段。它必须基于届时正式的 architecture/RTC/capture 合同重新设计；以下工作包只表达依赖与
安全边界，不授权现在修改采集链。

#### V25-0：显式 shadow 接线候选

前置：第一阶段完成，且 architecture/control/RTC 并行负责人确认 V2.5 合同、共享文件窗口和
exact base revision。

实施顺序：

1. 先新增 `teleop_runtime/collection/semantic_annotations/{cli,config,runtime}.py` 和 mock fixture，
   不注册 caller；
2. 再新增 RM65/UMI bridge 与 GUI presentation adapter 的独立文件；
3. 最后以单独小 patch 修改现有 RM65/UMI parser/composition，每个入口只注册共享参数并注入 bridge；
4. `--semantic-annotation-mode shadow` 只写显式 scratch root；默认/显式 `off` 不 import 重型模块、
   不构造对象、不改变 GUI、writer、episode metadata 或目录。

该批禁止导入 `control_station`、`robot_motion.deploy`、`deployments.client`、RTC frontend/plan/worker
或修改 ControlTrace schema。RTC/Control Station 如需观察 annotation health，只能以后消费 immutable
observer event，不得成为 shadow runtime owner。

验收：

- 无参数、显式 off 与合入前 baseline 的 help 以外行为、输出和 timing 等价；
- shadow queue/sink 故障不影响 RM65/UMI episode lifecycle；
- mock/headless/GUI contract 测试通过后，才申请独立无动作现场观察；
- 在线参数仍不进入正式数采配方。

#### V25-1：显式 experimental record 候选

前置：V25-0、Plus-3 和 V2.5 专属 online Gate 通过；publisher 必须直接复用 Plus-3 的事务实现，不能在线再造第二套
manifest、fingerprint、commit 或 split 逻辑。

- v2.3/v2.4 都只在原 episode 成功 finalize/commit 后绑定 identity，并发布外部 derived overlay；
- annotation queue、finalize status 与 `collection_integrity` 独立；
- record 失败只产生 annotation `partial/invalid`，不能丢弃、暂停或改写原 episode；
- RM65 与 UMI 分两个小批接线和验收，不在同一个 patch 同时改两个 active recorder；
- 真机长程 A/B 与正式配方评审单独进行，不从 shadow/mock 结果继承授权。

#### V25-2：反事实 campaign 与 profile 扩展

目标：在 Plus 单任务试验后扩展 A/B/C campaign、更多 task profile 和可选跨本体验证。
V25-2 不是 Plus 完成条件。通过 V25-2 不自动改变默认值，也不自动把实验参数加入正式
数采配方。

验收：

- shared initial state 有证据等级；
- intended preserved/varied 在采集前固化；
- observed changed/invariant checks 在执行后复核；
- perturbation 有安全审查；
- split 按启用关联图的 connected components 隔离；
- system invalid 不混入 recovery；
- UMI feasibility 状态不夸大；
- campaign 配置在设备初始化前完成校验；
- 未提供 master feature 参数的原有 campaign 与历史命令保持逐项等价。

#### V25-3：充分链路事实与 ControlTrace semantic bridge

目标：在 ControlTrace 正式生产挂接完成后，使用
`--semantic-annotation-control-trace-bridge reference` 建立可选 cause-chain refs；该参数仅在
master mode 为 `record` 时有效，默认 `off`。

验收前置见第 8.2 节。没有 trace 时 annotation 仍可完整运行；trace incomplete 必须显式降级。
bridge 故障不得回写 ControlTrace，不得阻塞既有 episode seal/commit/finalize。
V25-3 不阻塞任何 Plus 里程碑；它必须分别声明 proposal、admission、dispatch/receipt、measurement
与 world-effect 的 source/completeness，不能只因 ControlTrace 存在就宣称链路充分。

#### V25-4：多粒度研究与选择性重生成

Plus-3 完成后即可用最小 curation 数据做研究预览，不等待通用 detector 或 ControlTrace；
正式 V2.5 结论仍必须使用充分链路 profile。该阶段属于未来 `model/`，不是当前 runtime entrypoint。
V2.5 首轮固定：

- 1 个 contact-rich 任务；
- A/B/C 三类小规模反事实组；
- 在 Plus episode curation 之上，单独人工确认 phase、strategy、interaction state/event、geometry 和 observed recovery scope；
- 标准 Flow/Diffusion baseline 与分层 latent baseline；
- 仅用 `verified` invariant 做 consistency supervision。

后续研究顺序：

1. 标准 Flow/Diffusion baseline；
2. keypose/multiscale action baseline；
3. strategy conditioned；
4. strategy + interaction + geometry/residual；
5. counterfactual consistency/sensitivity；
6. 在逐层反事实 rollout/oracle 证据具备后研究 minimal replan routing；
7. 后续 human correction latent supervision。

每个模型 artifact 必须绑定 raw dataset、annotation set、split manifest、layout 和 pose contract hash。

### 12.3 阶段依赖图

```text
第一阶段共享：Plus-0
               ├─> Plus-1 V2.3 source profile ─┐
               └─> Plus-2 V2.4 native/compatibility source adapters ─┴─> Plus-3 external publisher/reader
                                                    -> post-review curation + split/training view

第二阶段：第一阶段稳定输出 + 届时正式链路合同
          -> V25-0/V25-1 可选在线接线
          -> V25-2 campaign/profile 扩展
          -> V25-3 充分链路 cause-chain
          -> V25-4 多粒度模型与选择性重生成
```

V2.3 Plus 可以独立交付；V2.4 Plus 的 native 分支等待主架构正式 committed capture 合同，但在该
合同尚未覆盖的历史/兼容数据上，可先使用 reader 明确提供的 v2.3 compatibility view。本文不替主架构
补齐 native identity，也不允许无证据 fallback。任何 V2.5 工作包都不阻塞第一阶段交付，也不能借
第一阶段的只读/离线验收取得在线修改授权。

## 13. 研究假设与非目标

### 13.1 研究假设

待验证的核心假设：

```text
不同粒度随机变量能够对应可证伪的行为不变量
```

建议层次：

```text
z_strategy     -> strategy
z_interaction  -> contact/constraint plan
z_geometry     -> keypose/contact region/path realization
z_residual     -> dense control residual
```

反事实 group 提供：

- `verified` invariant 的一致性监督；
- `intended_varied` 与 `observed_changed` 对照后的敏感性监督；
- recovery outcome 的有效性约束；
- observed recovery scope 的弱路由监督；
- 经逐层反事实 rollout/oracle 验证后，才可能获得 minimal replan level 的强路由监督。

这些是研究假设，不应提前固化成 runtime action schema。

研究主张收敛为“反事实不变量 + 选择性重生成”：数据系统、sidecar 和 GUI 是支撑，
ControlTrace 是可选因果证据增强，UMI/RM65 是后续泛化验证；它们都不取代以下核心问题：模型
能否只修改必要层级，同时保持已验证的上层行为不变量。

### 13.2 明确非目标

本规划不做：

- 为每帧生成长 caption；
- 一开始覆盖所有任务的巨大 ontology；
- 把 primitive 数量当创新指标；
- 把 VLM 输出直接当真值；
- 用固定 tactile 裸阈值跨设备判接触；
- 把 UMI qpos 或 feasibility 冒充真机事实；
- 把 ControlTrace 当任务 success；
- 把 system failure 当机器人 negative；
- 在采集控制线程运行重型标注/VLM；
- 把 schema、sidecar writer、detector、split 或 GUI 状态机堆进
  `teleop_runtime/debug/run_vive_rm65_real_teleop.py`；
- 在 shared annotation runtime 中导入 RM65、VIVE、DMTAC 等 vendor backend；
- 新增独立正式数采入口；
- 在验收前修改 future feature 的默认值，或通过配置存在、目录存在、环境变量、
  ControlTrace 状态进行隐式启用；
- 在验收前把实验参数写入 `docs/real/teleop_args.sh`、正式 RM65/UMI 配置或历史启动脚本；
- 在未安全审查时自动执行 perturbation 或真实 replay；
- 当前阶段实现 learner、DAgger 或 RL runtime。

## 14. 风险与缓解

| 风险 | 缓解 |
|---|---|
| annotation schema 过大 | 冻结 episode disposition/reason/variant；phase/state/event/evidence 只作研究预留 |
| 身份能力干扰现有 writer | 首版使用外部 locator/hash；核心 UUID 只由主架构 native v2.4 合同落地 |
| 与 native v2.4 创建重复身份/commit | compat/native identity profile 分型；native 只消费主架构 capture membership |
| annotation 反向改变 dataset fingerprint | base fingerprint 明确排除 annotations/staging/derived views |
| 操作员负担破坏示范 | Plus 只在开始前选一次有限 variant、结束后做一次 curation review，不占用在线控制操作 |
| 自动标签被当真值 | 首版不把 detector 或文件状态当训练真值；人工 disposition 只用于数据分流 |
| 未校准 score 被解释为概率 | detector/概率字段不进入 Plus v1；V2.5 单独做 calibration artifact |
| 过度依赖单一传感器 | 首版不新增 sensor-derived semantic label；tactile 只在 V2.5 candidate 中使用 |
| UMI/RM65 shape 误拼接 | embodiment/source/alignment 强制条件 |
| group 初始状态名不副实 | Plus v1 不要求 group；研究子集才保留 evidence level + snapshot/ref |
| 计划不变量被误当执行事实 | Plus v1 不把计划当标签；研究关系才分 intended/observed |
| state/event 混写破坏时序语义 | Plus v1 不发布 state/event；V2.5 扩展时再使用独立 record |
| recovery 数据引入安全风险 | Plus v1 只做 post-review 分流；真实 perturbation 仍需独立 campaign 和安全审查 |
| ControlTrace 被过度承诺 | 仅列入 V25-3，两个 Plus profile 都不依赖 |
| 多键传递关系导致隐性泄漏 | 按启用 edge families 求关联图 connected components，再整分量切分 |
| observed scope 被误称最小重规划 | 数据侧只记实际修改；minimal level 需要逐层反事实验证 |
| annotation 修改 raw 或半发布 | staging + 全量校验/fsync + 原子 rename/commit marker + append-only supersession |
| Plus 逻辑影响控制频率 | GUI/event 仅在显式开关下出现；sidecar 保存为独立后台线程，不进入 raw writer/control dispatch |
| 功能继续堆入真机 `debug` 入口 | 第一阶段禁止触达；V2.5 也只能在独立评审后做薄接线 |
| V2.5 future feature 被旧配置或环境意外启用 | 第二阶段单一 master 参数、默认 `off`、禁止推断启用 |
| 标注故障污染现有 episode 结果 | annotation health 与 collection integrity 分域；失败只标 annotation incomplete/invalid |
| 实验输出混入源码或正式数据目录 | 两个 Plus profile 都要求显式外置 overlay root；source 全程只读 |
| 并行主线互相覆盖或同名合同分叉 | 新文件优先、shared-file lease、exact base revision、每切片 touched-file allowlist |

## 15. 完成定义

### 15.1 V2.3 Plus 完成

- 只读取已 finalize 的 v2.3 snapshot，通过 external content locator 建立稳定 identity；
- source dataset before/after member byte/hash 等价，`validate-v23` 结论不变；
- overlay 只发布到显式外置根，事务、append-only、stale binding 与 crash recovery 验收通过；
- RM65/UMI raw writer、row schema、episode lifecycle 和正式默认配方零修改；Plus CLI 参数和共用 GUI
  旁路仅在显式开关下出现，旧命令行为不变。

### 15.2 V2.4 Plus 完成

- native path 只读取主架构正式 committed 的 v2.4 capture identity/membership；
- compatibility path 只读取 reader 明确提供的 `v2.3_single_action_row` view，并保留
  `v24_compatibility_content_locator_v1` 与 `compatibility_origin`，不以此冒充 native v2.4；
- 没有任何可验证 native/compatibility membership 时明确 fail-closed，不做目录猜测；
- 不生成 core UUID、不建立第二套 capture commit、不写 sealed capture；
- 可选 trace 只作已提交 evidence ref，unavailable/incomplete/complete 状态明确，未生成 cause chain；
- 产出与 V2.3 Plus 同 schema 的外置 overlay 和训练视图。

### 15.3 第一阶段共同完成

- `dataset_contracts`、`dataset_interface` 和 offline postprocess 依赖方向符合第 3.4 节；
- 外部 annotation identity、source binding、append-only 发布和 stale 检查通过；
- `disposition/reason_code/variant_id` 不由文件状态猜测，条件枚举和 review 状态验收通过；
- 一个可选 task profile 能稳定生成 variant 分组；未配置 profile 时无额外字段和操作负担；
- 默认 episode-level split 不拆 episode；有真实关系记录时才启用 connected-component split；
- UMI/RM65 共享 episode curation 高层语义，但保留 embodiment/source/alignment 事实边界；
- phase/event/relation 缺失不影响普通训练视图；
- Plus 参数、GUI panel 和 sidecar 后台线程均 default-off；未启用时不构造 profile/store、不写文件、
  不改变 canonical runtime 行为；V2.5 的在线语义 bridge 仍未实现；
- 最小 pure/fixture/dependency tests、source immutability checks 和 `git diff --check` 完成。

### 15.4 V2.5 进入条件

V2.5 不因第一阶段完成而自动开始。它需要 architecture、RTC、capture/ControlTrace owner 重新冻结
充分链路事实与 shared-file 窗口，并单独完成 default-off、off/on A/B、motion authority、failure-domain
和真机安全评审。其 V2.5 在线接线、cause chain、自动 minimal replan、跨本体模型与选择性重生成均不属于
V2.3 Plus / V2.4 Plus 的完成条件。

V2.5 研究成功仍由可证伪实验判断：固定细粒度 latent 时经验证的上层 invariant 是否保持；只改变
指定粒度时预期变量是否敏感；选择性重生成是否优于全量重生成。

## 16. 项目内权威参考

- [`dexteleop_architecture_planning.md`](./dexteleop_architecture_planning.md)：action/time/authority、
  native v2.4 capture、ControlTrace、derived overlay 与长期目录 ownership 的上位规划；冲突时优先。
- [`dexteleop_rtc_landing_plan.md`](./dexteleop_rtc_landing_plan.md)：RTC request/worker/timed-plan/
  emission 的独立落地主线；本规划不得修改其 producer truth 或把 RTC internal state 当语义事实。
- [`DATASET_SCHEMA.md`](../DATASET_SCHEMA.md)：当前 v2.2/v2.3 数据、pose、causal、媒体和质量合同。
- [`docs/real/README.md`](../docs/real/README.md)：真实系统 canonical 入口。
- [`docs/real/TELEOP_COLLECTION.md`](../docs/real/TELEOP_COLLECTION.md)：RM65 正式 position/impedance 配方与验收。
- [`docs/real/UMI.md`](../docs/real/UMI.md)：UMI 标定、验证和正式启动。
- [`teleop_runtime/collection/README.md`](../teleop_runtime/collection/README.md)：writer、episode、pause、维护和输出结构。
- [`teleop_runtime/collection/umi/README.md`](../teleop_runtime/collection/umi/README.md)：UMI action alignment、rejection 和设备语义。
- [`dataset_contracts/README.md`](../dataset_contracts/README.md)：v2.3/v2.4 合同边界与当前生产状态说明。
- [`ACTION_LABEL_PLAN.md`](./ACTION_LABEL_PLAN.md)：稳定 arm mode 与 audit phase 的分离规则。

## 17. 实施顺序摘要

```text
当前第一阶段已完成最小 Plus v1 的合同、兼容 reader、外置 sidecar 和 RM65/UMI default-off GUI 接线；
后续只推进离线发布/训练视图和显式 v2.4 native source adapter，不回写 raw 链路：

Plus-0 共享 curation 合同/identity/hash
  ├─> Plus-1 V2.3 finalized source adapter
  └─> Plus-2 V2.4 native/compatibility source adapters（native 依赖上位 v2.4，compatibility 依赖显式 reader view）
       -> Plus-3 外置事务 publisher/reader/post-review/split/training view
```

第一阶段的实现边界固定在外置 overlay 和可重建训练视图，不进入采集 runtime：

```text
finalized/committed/explicit-compatibility source（只读）
 -> 离线 source profile
       -> immutable semantic overlay
       -> validate / curation review / episode split / training view
```

V2.5 另行立项后，才讨论 online shadow/record、充分链路 cause chain、更多 campaign/profile、
多粒度模型和选择性重生成。一次 Plus 发布成功、研究预览有效或 ControlTrace 可用，都不能跳过
V2.5 的合同与安全评审。

近期最优先交付不是新的自动标注模型，而是：

```text
外部稳定 annotation identity
+ episode disposition（nominal/recovery/exclude/unknown）
+ bounded reason_code（仅必要时）
+ 可选 task-profile variant_id
+ source 全程只读且不接现有数采链
```

这些信息可以在 episode finalize/commit 后由人工快速确认（variant 在开始前一次选择），并直接支持
过滤、分组和基础训练；phase、intended/observed、evidence、关联图和多粒度变量留到有明确研究问题时再增加。

## 18. 与 architecture / RTC 并行主线的落地协议

### 18.1 文档权威和责任边界

三条主线不互相复制合同：

| 主线 | 唯一拥有的语义 | 本规划如何依赖 |
|---|---|---|
| `dexteleop_architecture_planning.md` | action/time/authority/safety、native v2.4 capture、core episode UUID、ControlTrace、sealed/derived 边界 | 上位规范；直接复用 canonical encoding、capture commit 和 native identity，不另造 |
| `dexteleop_rtc_landing_plan.md` | inference request/result、worker/session、timed plan、due-pop、`PolicyActionEmission` | annotation 只能引用已正式持久化的事实；不读取 RTC mutable plan，不参与 admission/pop |
| 本规划 | v2.3 compatibility identity、最小 episode curation overlay、可选研究 relation/evidence | 不拥有 motion、episode capture、RTC producer 或 ControlTrace writer |

发生冲突时，主架构对公共语义优先，RTC plan 对 RTC 局部实现优先。本规划只能调整自己的
overlay/adapter；不能通过修改本文绕过其 owner、epoch、safety、capture commit 或真机 gate。

### 18.2 当前并行热区与文件规则

截至本文基线，`dataset_contracts`、`dataset_interface`、`teleop_runtime/collection`、canonical
RM65 debug 入口和 `control_station` 都存在并行改动；architecture plan 本身也在修改，RTC landing
plan 尚是独立新文件。状态只说明工作树活跃，不说明这些改动属于本规划，也不得覆盖或格式化它们。

首批优先新增独立文件：

```text
dataset_contracts/{semantic_annotations,counterfactual,annotation_manifest,annotation_identity}.py
dataset_contracts/plus_episode_curation.py
dataset_interface/{annotation_identity,annotations}.py
utils/dataset_postprocess/semantic_annotations/*.py
tests/test_semantic_annotation_*.py
tests/test_plus_*.py
```

以下属于 shared hot file，修改前必须确认负责人、exact base revision、允许文件列表和合入窗口：

```text
dataset_contracts/__init__.py
dataset_interface/__init__.py
utils/dataset_postprocess/{__init__.py,cli.py}
DATASET_SCHEMA.md / package README
teleop_runtime/collection 现有 writer、run_collect 和 umi 文件
teleop_runtime/debug/run_vive_rm65_real_teleop.py
teleop_runtime/gui/real_teleop_control.py
control_station/**
robot_motion/deploy/**
deployments/client/**
```

RTC 当前批次明确不修改 dataset schema，因此 Plus v1 不进入 RTC 或 ControlTrace producer；即使新增
文件没有文本冲突，也必须避免与并行主线同时定义同名 contract 或不同的 native v2.4 identity。

### 18.3 推荐合入切片

以下把已落地切片和后续候选切片分开。每个切片仍需独立 review、独立测试、独立回滚，不把多个阶段
塞入一个 patch：

| 切片 | 内容 | 允许触达 | 明确不触达 | 退出条件 |
|---|---|---|---|---|
| `Coord-0` | 三线边界、文件 ownership、base revision 确认 | planning docs/read-only audit | runtime/schema/code | 三方无同名合同或热文件冲突 |
| `Plus-0` | pure schema、external identity、canonical hash | `dataset_contracts/episode_curation.py` + tests | raw writer/row schema | 已通过 pure/invalid/hash fixtures |
| `Plus23-1` | finalized v2.3 RM65/UMI compatibility reader、minimal sidecar store | `dataset_interface/episode_curation.py` + `teleop_runtime/collection/episode_curation.py` | source dataset write、raw writer | 已通过 missing-sidecar、idempotency/conflict tests |
| `Plus24-1` | v2.4 native membership/identity + explicit v2.3 compatibility-view adapter | 新 v2.4 source adapter + fixtures | capture/trace writer、UUID/commit owner、目录猜测 fallback | native/compatibility binding、同 schema 输出与 unavailable fail-closed |
| `Plus-2` | external transactional publisher、validator、episode-level split | 后续 postprocess 子目录 + `/tmp` fixtures | raw writer、V2.5 online sink | crash/status/split tests |
| `Plus-3` | RM65/UMI 共用 GUI post-review、variant profile 与 training view | `teleop_runtime/gui` + `teleop_runtime/collection`（GUI/sidecar 已接线）；训练 view 后续补齐 | 通用 VLM、phase/state/event/evidence、所有任务 ontology、真机控制 | 已通过 default-off GUI、sidecar append、v2.3/v2.4 compatibility fixture |
| `Research-Preview` | phase/event/relation 与最小 baseline（可选） | immutable input + derived training view | online feature、RTC、ControlTrace 前置 | 只验证研究假设，不扩大 Plus 合同 |
| `V25-*` | 在线接线、充分链路、cause chain、选择性重生成 | 未来独立评审批准的文件 | 当前第一阶段所有代码批次 | 新合同、owner、A/B 与安全 Gate 全部重审 |

`Research-Preview` 可以在 `Plus-3` 后迭代并行，但不是第一阶段完成条件。
`V25-*` 不能因为离线研究有效而自动提前。

### 18.4 每个切片的并行修改协议

开始前：

1. 记录 `git status --short`、目标文件 diff、base revision 和本切片 touched-file allowlist；
2. 重新读取两份上位规划的当前状态/不得修改清单；
3. 对 shared hot file 确认没有未合入 owner patch；无法确认则改走新文件或等待；
4. 明确本批 Plus source profile、外置 overlay root 和不包含范围。

实现中：

- 首批通过直接 submodule import 测试，不顺手改 `__init__.py`/README/总 CLI；
- 一次只落 contract、单一 source adapter、publisher 或 offline task 中的一层；
- 不批量 rename/format 并行文件，不移动现有入口，不吸收无关 dirty diff；
- annotation 不导入 RTC/Control Station；RTC/Control Station 也不成为 annotation lifecycle owner；
- 若主架构已经出现等价 canonical type/hash/manifest，改为 adapter/reuse，停止创建平行类型。

合入前：

1. 再次检查 touched files 是否被并行修改；有重叠先停下重新审计；
2. 运行本切片 targeted tests、dependency-boundary tests 和 `git diff --check`；
3. 对 Plus23-1、Plus24-1 和 Plus-2 保留 source byte/hash before/after 等价证据；
4. evidence 只声明本切片 scope，不把 architecture/RTC 的历史通过数吸收到本批结论。

### 18.5 与 RTC 的数据接口防火墙

- RTC worker/timed plan 是 producer-local mutable preparation，不是 semantic evidence；
- 只有正式 `PolicyActionEmission`、ControlTrace explicit event、capture membership 和 durable receipt
  才能被 annotation 引用；debug metric 只能标为 non-formal supporting evidence；
- annotation detector、review queue、publisher 不能进入 RTC control tick、worker callback、plan merge 或
  due-pop 路径；
- semantic label、disposition、future frame 和 post-episode review 默认不能进入部署 actor observation；若研究
  需要 conditioning，必须经主架构 `PolicyObservationView`/artifact allowlist 另立合同；
- 未来 Research-Preview artifact 若使用 RTC 部署，仍需 RTC capability/codec/horizon 与 real authority 独立验收，
  数据集成功不提供 motion authorization。

### 18.6 暂停条件和最近一步

出现以下任一情况立即暂停当前切片，不自行合并语义：

- shared hot file 在实施期间出现新的并行 diff；
- architecture 发布 native v2.4 identity/overlay type，与本批 compat type 重叠；
- RTC 或 Control Station 请求 annotation runtime 提供 queue、episode owner、plan cursor 或 actor input；
- source dataset 的目录、metadata、线程、timing 或 validate 结果因 Plus 发生变化；
- publisher 只能通过修改 sealed/committed capture 才能工作；
- 任一 Plus 切片要求修改 RM65 或 UMI active recorder 才能继续。

此前的丰富原型已移入 `legacy/dataset_semantic_plus_v0/`；当前生产批次只保留最小 episode curation
contract、独立 fixture、外置 sidecar 和共用 GUI panel。完整 state/event/evidence、timeline、candidate、
关联图和 V2.5 online 能力不得顺带恢复。Plus v1 全程不改 export、raw writer、ControlTrace、RTC、
`control_station` 或任何已有 RM65/UMI 采集默认行为。

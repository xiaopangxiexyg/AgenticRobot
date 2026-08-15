# Arm Hold Label 数据 Profile 与派生视图规划

## 1. 结论

为轻量 `hold-label prediction` 建立一个**模型框架无关、源数据只读**的数据
profile 是合理且必要的，但不应复用或扩写成 OpenPI 专用训练配置。

首个 profile 管理以下三个独立 source domain：

1. `2026-08-03_dmtac_rm65_bimanual_position_v2_writing_4_stage`
2. `2026-08-04_dmtac_rm65_bimanual_position_v2_writing_4_stage`
3. `2026-08-05_dmtac_rm65_bimanual_position_v2_writing_4_stage`

0805 提供当前集合中唯一有一定数量的左侧 `fixed_hold->teleop` 样本；0803、0804
提供跨日期覆盖、稳定的其他三类切换和更多自然稳定区间。三者不得先物理合并，也不得按
symlink 路径重复计入。

profile、manifest 和 cache 分工冻结为：

```text
checked-in ArmHoldDatasetProfile
    声明 source identity、admission、split 和 horizon 规则
                 |
                 v
generated ArmHoldDatasetManifest
    冻结 exact Episode membership、split、coverage 和 source receipts
                 |
                 v
rebuildable window/cache
    读取 front RGB/state，生成 [H, 2] target/mask；可删除重建
```

本轮 profile 不拥有模型结构、optimizer、loss 权重、augmentation、checkpoint、RTC、
resolver 或执行语义。

## 2. 已确认的数据事实

### 2.1 物理层级

迁移后的 canonical 数据根为：

```text
/home/user/zc/datasets/RealmanDMTAC/DexTeleop_0803/datasets/
```

旧路径仍是逐数据集 symlink compatibility view：

```text
/home/user/zc/DexTeleop_0803/datasets/<dataset>
    -> /home/user/zc/datasets/RealmanDMTAC/DexTeleop_0803/datasets/<dataset>
```

因此 source identity 不能绑定 resolved absolute path。profile 使用一个运行时传入的
workspace root 加 source-relative locator；resolved path 只进入主机本地 receipt，不进入
profile semantic hash。compatibility path 与 canonical path 必须生成同一个 source identity。

每个数据集当前具有同构的只读层级：

```text
<dataset>/
  data/                     # Episode Parquet
  videos/                   # front / left_wrist / right_wrist
  tactile/                  # external tactile stream
  wrist_depth/              # wrist depth mask stream
  meta/
    info.json               # dataset/layout/label declarations
    episodes.jsonl          # authoritative Episode catalog
    episodes_stats.jsonl
    modality_validity.json
    tasks.jsonl
    devices/
    progress/
      profile.cjson
      stages.cjson
      episodes/
      stage_annotations/
      stage_annotation_drafts/
```

本实验只读取 Parquet、front video、`info.json`、`episodes.jsonl` 和必要的 sealed
progress/annotation artifact。tactile、wrist depth 和 draft annotation 不进入输入，也不进入
cache identity。

### 2.2 三个 source 的原始覆盖

| source | Episodes | frames | hold Episodes | front/state | 备注 |
|---|---:|---:|---:|---|---|
| 0803 writing-4-stage | 202 | 86,667 | 191 | 202/202 | 跨日域 1 |
| 0804 writing-4-stage | 355 | 142,576 | 348 | 355/355 | 最大单日集合 |
| 0805 writing-4-stage | 261 | 102,887 | 259 | 261/261 | 含稀有 left exit |
| 合计 | 818 | 332,130 | 798 | 818/818 | raw inventory，不等于 eligible cohort |

逐侧原始 label 行数：

| side/label | rows |
|---|---:|
| left/teleop | 144,260 |
| left/fixed_hold | 187,870 |
| right/teleop | 268,787 |
| right/fixed_hold | 63,343 |

实际双臂组合为：

| pair `[left,right]` | rows |
|---|---:|
| `00` teleop/teleop | 80,917 |
| `01` teleop/fixed_hold | 63,343 |
| `10` fixed_hold/teleop | 187,870 |
| `11` fixed_hold/fixed_hold | 0 |

`11` 缺失是当前四阶段任务事实，不应伪造、过采样或作为 profile admission 失败；manifest
必须显式记录该组合未覆盖。

Episode 内原始切换事件：

| direction | 0803 | 0804 | 0805 | total |
|---|---:|---:|---:|---:|
| left `teleop->fixed_hold` | 189 | 346 | 248 | 783 |
| left `fixed_hold->teleop` | 0 | 0 | 12 | 12 |
| right `teleop->fixed_hold` | 191 | 347 | 260 | 798 |
| right `fixed_hold->teleop` | 189 | 345 | 250 | 784 |

因此 split 不能只按总类别比例分层；必须保护 0805 中原始的 12 个 left-exit 事件。严格
Episode admission 后实际保留 9 个，分别来自 9 个不同 Episode；coverage gate 以 eligible
cohort 的 9 个事件为准，不复制窗口或改写 label。

### 2.3 已确认的共同合同

三批数据均声明：

- dataset schema：`dexteleop_dataset_v2.3`；
- nominal rate：15 Hz；
- state/action：16D 双臂 desk-absolute pose + 双侧 DM-Tac gripper scalar；
- arm-mode schema：`dexteleop_arm_mode_v2`；
- hold projection：`arm_hold_intent_v1`；
- hold projection hash：
  `8ff801c71736d28c89b496c3406496bc348cac5aeea14527312d5a21f90c6600`；
- positive scope：`arm_and_end_effector`；
- layout hash：
  `sha256:2567572f0f9170281b3e775ed13ecba903fbdeb199fd88a5bb1b7b1f4e96db16`；
- Progress Profile hash：
  `0874f6b97b3b5255240dd34e6cfac01c76a95c1077df8a20af6b9b3119f3a438`；
- Stage Profile hash：
  `b3e3d19daefa9dbf9fe870c769c062c72b24e1442cfb400d52fae9664dc3934b`。

这允许一个 profile 管理三批数据，同时保留日期/source domain。

## 3. 与已有 profile 的关系

仓库已有四个：

```text
deployments/openpi/configs/training/
  writing_stage1_0804_0805_{front,full}_h16_{absolute,relative}_v1.json
```

它们属于 `dexteleop.openpi_prepared_skill_training_profile.v1`，负责：

- 0804+0805 的严格 602-Episode Skill cohort；
- action pose mode、camera slots、OpenPI base/config/asset/experiment name；
- target Skill 与 stage annotation selection；
- OpenPI source mixing 和 action training。

新 profile 与它们的关系为：

```text
ArmHoldDatasetProfile                 OpenPI Prepared Skill Profile
  source/cohort/split/window            model/action/Skill integration
  framework-neutral                     OpenPI-specific
  full-task label visibility             selected Skill intervals
             \                           /
              reuse the same source facts
```

首轮不得修改现有四个 OpenPI profile，也不得让它们引用尚未稳定的新 profile。新实现可以
复用 prepared-training 中已有的 source fingerprint、Episode identity、inventory、annotation
selection 和 conflict detection 逻辑，但不能复制一套同义实现。

## 4. Profile V1 的最小职责

建议 provisional schema：

```text
dexteleop.arm_hold_dataset_profile.provisional.v1
```

先由 `dataset_interface` 所有；完成一次真实 build/train/eval 后，再决定是否晋升为
`dataset_contracts` 的稳定公共合同。这样避免在实验前冻结过多字段。

建议 future asset：

```text
dataset_interface/arm_hold_profiles/
  writing_0803_0804_0805_front_h16_v1.json
```

建议 loader/builder：

```text
dataset_interface/arm_hold_profile.py
```

不把现有 `dataset_interface/arm_hold.py` 从模块改成 package；后者继续只拥有 label
projection、mask 和 coverage。

### 4.1 Profile 中应有的字段

```text
schema
profile_id
purpose = lightweight_hold_label_prediction

workspace_key
sources[]:
  source_dataset_id
  relative_dataset_root
  expected_dataset_name
  expected_episode_count
  expected_frame_count

required_contracts:
  dataset_schema
  layout_hash
  arm_hold_projection_hash
  progress_profile_hash
  stage_profile_hash

episode_admission:
  sealed_only
  success/non-abnormal policy
  review-pending exclusion policy
  annotation selection policy
  required modalities/columns

sample_view:
  observation = front RGB + current observation.state
  horizon = 16
  tail_policy = drop

required_contracts 中的 `arm_hold_projection_id/hash` 已唯一绑定 same-index
`arm_hold_intent_v1`；`sample_view` 不重复声明 target。`tail_policy=drop` 已完整表达首轮不
padding，因此也不再保留同义布尔字段。

split:
  unit = episode_capture_identity
  strategy = source_and_transition_stratified_v1
  train/validation ratio
  deterministic seed/domain
  minimum coverage gates
```

### 4.2 不应进入 Profile 的字段

- absolute machine path；
- image resize/crop implementation；
- backbone/head hidden size；
- optimizer、learning rate、batch size、epoch；
- BCE/class/transition/lambda 权重；
- sampling/oversampling 实验开关；
- checkpoint、OpenPI config 或 server endpoint；
- threshold/debounce/resolver/RTC 参数；
- 逐帧 debug payload。

首个训练实验关闭 augmentation，但该开关属于 experiment/cache config，不进入 dataset
profile 或其 semantic hash。

这些属于后续 experiment config、model adapter 或 deploy artifact。profile 只回答“哪些只读
source facts 构成哪一个可重建数据视图”。

## 5. Cohort 与 split 语义

### 5.1 Episode admission

首轮推荐使用**成功、sealed、无 abnormal interval 的完整 Episode**，而不是只使用现有
`writing.bimanual_acquire_brush_and_present_for_cap_removal` Skill interval。原因是 hold mode
切换跨越整个四阶段任务；裁成单一 Skill 会丢失 right exit、left enter 以及稀有 left exit。

Progress/stage/annotation 只用于 Episode admission 和异常排除；正式 hold target 始终来自
同 keyframe 的 recorded effective arm-mode，不从 progress step 重新生成。

`episode_review_request` 使用独立的 `exclude_episode_review_pending_v1` 原因排除，不混称为
abnormal。多个 sealed set 的三个 `SUCCESS` Episode 已显式绑定到与现有 OpenPI training
profile 一致的审阅版本；两个 `UNKNOWN` Episode 在 terminal-success gate 前置排除，不需要
annotation selector。

首次 no-write preflight 已确认 exact eligible count 为 774。不能把 raw 818 Episodes 直接写成
eligible count。已知现有 OpenPI 0804/0805 Skill profile 的 344+258=602 只证明其 Skill cohort，
不等于本 profile 的完整 Episode cohort。

2026-08-06 的重复 no-write dry-run receipt 为：44 个 Episode 被排除，其中 41 个 terminal
outcome 非 `SUCCESS`、3 个存在 whole-Episode review pending；619 个 Episode 进入 train、155 个
进入 validation。按 source 的 train/validation 数量分别为 0803=`148/37`、0804=`275/69`、
0805=`196/49`。两次独立重建得到相同 dry-run hash
`8c219c70ed4b3b6beb5ee68c100404126d6ffca93d2970955ce0751dadef7eb6`。这只是可重建的
只读 evidence，尚未生成或写入 frozen split manifest。

### 5.2 Split

split 必须先按 Episode 完成，再生成窗口：

```text
source facts
  -> Episode eligibility
  -> Episode feature signature
       source date
       observed pair states
       per-side transition directions
  -> deterministic stratified assignment
  -> frozen Episode split manifest
  -> H=16 windows
```

V1 使用 80/20 train/validation，但不是简单 hash-mod：

- 每个 source date 同时出现在 train 和 validation；
- 左右侧的 teleop/fixed_hold 都必须在两侧 split 出现；
- `00/01/10` 都必须在两侧 split 出现；
- 四个 transition direction 都必须在 train 和 validation 出现；
- 先解析严格 admission 后承载 0805 的 9 个 left-exit event 的 Episode；至少保留 2 个 event
  给 validation、7 个给 train；
- 其余 Episode 在固定 seed 下按 source 和 signature 分层；
- 同一 Episode identity、内容相同的 symlink alias 或重复 source 永远只能出现一次；
- identity 相同但内容冲突必须 fail closed。

分层算法与 seed 进入 profile hash；最后的 Episode membership 进入 generated manifest hash。
不得在每个训练 run 中重新随机划分。

### 5.3 Window

每个 observation index `t` 对应：

```text
input:
  front_rgb[t]
  observation.state[t]

target:
  arm_hold_target[t:t+16, left/right]
  arm_hold_valid_mask[t:t+16, left/right]
  arm_hold_transition_mask[t:t+16, left/right]
```

窗口不得跨 Episode、continuity segment、frame gap、非 RUNNING 区间或 padded row。label
采用 `drop_window_if_any_side_invalid_v1`：horizon 内任一时刻任一侧 `arm_hold_valid_mask=false`
时丢弃整个窗口，不进行单侧 masked training。
transition 必须先在完整 Episode 上投影，确保窗口首行能够读取同 continuity segment 的
`t-1`，再切窗口；不得在每个窗口内部重新计算 transition。

2026-08-06 对三批 finalized 数据的只读统计为：818 Episode / 332,130 RUNNING rows 中
left-only、right-only、both-invalid 均为 0；严格 admission 后 774 Episode / 322,839 rows
仍均为 0。H=16、同 continuity 的 raw/eligible 候选窗口分别为 255,237/248,196，因 label
invalid 丢弃均为 0。因此该策略不改变 V1 样本数量，但冻结了未来数据的 fail-closed 行为。

Episode-level transition coverage 与 H-window coverage 必须分开报告。current manifest 中 train
的 7 个 left-exit event 只有 5 个可被至少一个合法 H16 window 覆盖，validation 的 2 个均可
覆盖；对应含 left-exit 的 overlapping windows 为 train=10、validation=4。其他三方向的
`source event / covered event / windows` 为：train left-enter=`522/522/6602`、right-enter=
`594/594/7297`、right-exit=`519/519/6567`；validation left-enter=`131/131/1657`、
right-enter=`151/151/1849`、right-exit=`130/130/1641`。因此 V1 两侧 split 仍具备四方向的
实际训练窗口，但训练启动前不能只复用 manifest 的 Episode-level coverage gate，必须运行
window inventory coverage preflight。

V1 tail 直接 drop，不使用 padding。previous-mode/dwell 可以由 loader 作为后续输入消融派生，
但不改变 Episode split 或正式 same-index target。

## 6. Generated Manifest 与 cache 层级

P1 只冻结 Episode admission 和 split，不提前伪装成完整训练 view。为避免六个小文件之间重复
声明 identity/hash，当前采用单一、内容寻址的 canonical artifact，并且不写入任一 source
dataset：

```text
<organized_root>/derived/arm_hold_splits/<manifest_hash>/
  manifest.cjson
```

`manifest.cjson` 自包含：规范化 profile snapshot、portable source receipt、included/excluded
Episode、exact train/validation membership 和当前 label/pair/transition coverage。目录名是完整
payload 的 semantic hash；loader 必须重新计算并验证目录名。重复发布相同内容是幂等的，已有
目标损坏或冲突时不得覆盖。

这里的 `source receipt` 不包含 resolved host absolute path，后者只属于运行时本地诊断；split
manifest 也暂不声称绑定 Parquet/RGB bulk bytes。P2 构建可训练 window/view 时，再通过
source-relative locator、逐 Episode Parquet/front receipt、合法 H-window 统计绑定实际训练输入。
这一区分保证 split membership 可以先冻结，同时不会把尚未实现的数据物化承诺写进 P1
协议。

2026-08-06 已完成首次正式原子发布与严格回读：

```text
organized root:
  /home/user/zc/DexTeleop/datasets/training_workspaces/
    arm_hold_writing_0803_0804_0805_front_h16_v1

current manifest hash:
  c007d85aec111cb71dca3042f1a5aba63269f2ec85f290ad20a17ad15acdc838

current cohort dry-run hash:
  8a01f53c846524d76a1cd69bbcf315bba07724aeda855819707e20f9f4d81fc1

current profile hash:
  d3a104498ff8ff72b223e01f7273166f56dcd28989c419acb837998acf5f8379
```

首次 materialize 返回 `idempotent=false`，相同输入的第二次 materialize 返回
`idempotent=true`；严格 loader 回读为 774 eligible、44 excluded、619 train、155 validation。
发布目录/文件权限分别为 `0775/0664`，且无遗留 `.building-*`。该 receipt 不改变
`DexTeleop_0803` 或任一 source dataset。

较早的 manifest
`228a9fd730682c5ae204fd219949ee703f118af6dd44dedebcd703b68df7c35c` 对应尚未显式声明
invalid-label window policy 的 profile revision。它作为不可变历史 artifact 保留，但不再是
P2 的 current input；不得删除或原地覆盖。

RGB/state cache 另放：

```text
<organized_root>/cache/<manifest_hash>/<cache_config_hash>/
```

cache identity 包含 resize/codec/dtype 等派生实现，但不反向改变 dataset manifest。cache 可删除
重建，不能被 checkpoint 当作 source truth。

## 7. 实施阶段

### P0：Profile contract 与纯验证

1. 在 `dataset_interface` 定义 provisional profile dataclass、规范化和 semantic hash。
2. 加入一个 checked-in 0803/0804/0805 front-H16 profile asset。
3. 复用现有 source fingerprint/Episode identity/arm-hold projection，不复制 canonical hashing。
4. 单测覆盖未知字段、重复 source、绝对路径进入 semantic identity、contract mismatch 和
   compatibility/canonical path identity 等价。

### P1：只读 inventory、cohort 和 split manifest

1. 读取三个 source 的 sealed facts，不写 source。
2. 执行 admission，并给出 rejected Episode 原因。
3. 生成 deterministic stratified split 和 coverage gates。
4. 先 dry-run 输出摘要；显式确认后才在远端 derived workspace 原子写单文件、内容寻址的
   split manifest。
5. debug-off 只打印总计/gate；debug-on 才输出逐 Episode signature、拒绝原因和分层决策。

### P2：轻量 window/cache reader

1. 按 frozen split manifest 枚举窗口。
2. first-pass 使用 lazy video decode；确认 IO 成本后再决定是否生成 RGB cache。
3. 支持 state-only、front-only、front+state 三种输入，但它们共享同一 Episode split。
4. previous-mode/dwell 作为后续可选 derived context，不进入 canonical state。
5. 增加 single-batch、one-Episode、shuffled-label 和 worker determinism 测试。

2026-08-06 current manifest 的 no-cache inventory receipt：

| split | Episodes | pre-continuity candidates | continuity rejects | legal H=16 windows |
|---|---:|---:|---:|---:|
| train | 619 | 249,077 | 50,364 | 198,713 |
| validation | 155 | 62,152 | 12,669 | 49,483 |
| total | 774 | 311,229 | 63,033 | 248,196 |

`LABEL_INVALID/STATE_INVALID/FRONT_FRAME_INVALID/NON_RUNNING/FRAME_GAP` 在当前 eligible
cohort 中均为 0。三个 source 各读取一个真实 `front_state` sample 后确认 front 为
`480x640x3 uint8 RGB`、state 为 `16D float32`、target 为 `16x2`，且 valid/loss 全真；
未写 cache 或图像文件。

### P3：独立轻量训练

1. 训练代码只消费 manifest，不直接扫描日期目录。
2. 第一轮关闭 augmentation，固定 `[H=16,2]` 输出。
3. 在同一 split 上比较 majority、persistence、state/front/front+state。
4. loss/sampling 消融写入独立 experiment config，不改 dataset profile。

2026-08-06 已完成 P3 的最小实现与真实数据 GPU smoke；该实现现归属 `policy_learning.arm_hold`，
仅提供小型 RGB encoder、state MLP、并行 `[H,2]` head、单一 masked weighted BCE 和
input-free constant baseline，不是新的生产 entrypoint，也不依赖 OpenPI adapter。三源各取一个
真实 `front_state` window 组成 batch，在已有 OpenPI venv 的空闲 GPU 上完成前向、反向和一步
optimizer update：42,528 parameters，初始/一步后 loss=`0.69414/0.69059`，梯度有限且非零。
该 smoke 未写 checkpoint、日志或 cache，也未修改 OpenPI、source dataset 或部署链。

环境边界同时得到验证：DexTeleop 的常规 `dexteleop310` 环境未安装 torch，因此数据
profile/window reader 仍可独立工作；训练验证显式选择已有
`/home/user/zc/openpi/.venv`（torch 2.7.1+cu126），不通过给 DexTeleop runtime 环境补装模型
依赖来制造隐式耦合。当前证据只证明 batch/model/loss/GPU plumbing 正确，不等价于正式
训练结果或 hold-label 可学习性结论。

随后完成一个显式 `--no-motion`、不写 artifact 的 state-only 单 Episode overfit probe。probe
从 train split 中选择真实包含 left-exit 合法窗口的 0805 Episode，使用 128 个 H16 windows
（含 2 个 left-exit target element）训练 17,696-parameter head 600 steps；loss 从 `0.69072`
降至 `0.02585`，同批 overall balanced accuracy=`0.99268`，2/2 left-exit keyframe prediction
正确。input-free constant baseline 的 balanced accuracy=`0.5`。这证明真实 manifest -> window
-> tensor -> loss -> optimizer -> aggregate metric 链路可过拟合且不会必然退化为 constant label；
它仍不是独立 Episode validation 泛化结果。

首个全量 train/validation 基线同样通过：使用全部 198,713 个 train windows、自然采样、
unweighted BCE、state-only、1 epoch，在 49,483 个 frozen validation windows 上得到
pooled balanced accuracy=`0.96591`、左右 macro balanced accuracy=`0.96307`、PR-AUC=
`0.99182`，pair exact match=`0.94837`；总耗时 67.5 秒。按 train prior 拟合后应用到
validation 的 per-side constant predictor，左右 balanced accuracy 均为 `0.5`。pooled constant
balanced accuracy 会因左右侧先验不同达到 `0.69644`，因此正式比较必须同时报告 per-side 与
macro-side，不能单独使用 pooled overall。

该自然基线的 independent-window exact-keyframe recall 仍显示 transition 是主要缺口：
left-enter=`0.3796`、left-exit=`1.0`（仅 4 个 overlapping window elements，对应 2 个 unique
events）、right-enter=`0.3889`、right-exit=`0.5521`。因此下一步应按既定顺序比较 class-balanced
loss 与 transition-centered sampler；当前结果不支持跳过消融或直接加入 persistence loss。

class-balanced BCE 的单因素对照也已完成，权重严格由 train window prevalence 得到：left/right
positive weight=`0.74228/4.14818`。macro balanced accuracy 从 unweighted 的 `0.96307` 小幅变为
`0.96618`，但 pair exact 从 `0.94837` 降至 `0.92911`；transition recall 从
left-enter/right-enter/right-exit=`0.3796/0.3889/0.5521` 变为
`0.2142/0.9627/0.0280`，left-exit 仍为 4/4。即 class weighting 主要把右侧决策强烈推向
fixed-hold，并没有同时改善进入与退出切换。它暂不应成为默认配置；下一步 sampler 若要对四个
方向做重采样，必须先冻结重复倍率/上限和稀有 left-exit 的防记忆策略，不能直接使用无上限的
inverse-frequency oversampling。

对 sampler 进行二次反思后，补充了 compute-matched random bonus control，避免把“额外
optimizer exposure”误判为 transition 定向收益。train 中实际有 13,909 个 unique
transition windows；三组均使用相同 seed、模型、unweighted BCE 和 1 epoch：natural 为
198,713 exposures，random/transition bonus 均为 212,622 exposures。每个 bonus window
至多额外一次，validation 始终 natural，constant/class prior 始终来自 natural inventory。

| sampling | macro balanced | pair exact | left enter | left exit | right enter | right exit |
|---|---:|---:|---:|---:|---:|---:|
| natural | 0.9631 | 0.9484 | 0.3796 | 1.0000 | 0.3889 | 0.5521 |
| natural + random-once | 0.9698 | 0.9530 | 0.3271 | 1.0000 | 0.5971 | 0.3224 |
| natural + transition-once | 0.9675 | 0.9464 | 0.3802 | 1.0000 | 0.6409 | 0.2633 |

random control 的 13,909 个 bonus 中自然命中 947 个 transition windows；transition 组的
13,909 个 bonus 全部为 transition windows。transition 定向相对 random 只进一步提高 enter，
同时降低 right-exit、macro 和 pair exact；相对 natural 也呈现明显 enter/exit tradeoff。
因此 `natural_plus_transition_once` 仅保留为可重放实验模式，不晋升为默认 sampler。
`natural_plus_random_once` 也只是 compute control，单 seed 的 aggregate 改善不能作为正式方案。

exact-transition element weight 随后在 natural sampling、class weight=1 下独立扫描。seed=23
中 weight=`1/2/4/8` 的 macro balanced 为 `0.9631/0.9650/0.9673/0.9683`，pair exact 为
`0.9484/0.9503/0.9542/0.9569`，且三个有充分 support 的 transition direction 均随权重改善。
但 paired seed=31 给出不同 Pareto 关系：

| exact weight | macro balanced | pair exact | left enter | right enter | right exit |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.9650 | 0.9516 | 0.7302 | 0.9470 | 0.8349 |
| 4 | 0.9622 | 0.9496 | 0.7990 | 0.9730 | 0.8787 |
| 8 | 0.9600 | 0.9488 | 0.8503 | 0.9811 | 0.9080 |

即更大 exact weight 能稳定强化 transition elements，但不保证自然 validation 的稳定/pair
指标；不同 seed 下未加权 transition recall 本身也有很大方差。left-exit 在所有实验中均为
4/4 overlapping elements，只对应 2 个 unique validation events，不能参与权重优选。因此当前
实现保留正数 `exact_transition_weight` 参数，但默认仍为 `1.0`，不冻结 2/4/8 中任何值。
正式选择至少还需要更多 seed、Episode-level aggregation 和尚未实现的 timing/bouncing/dwell
指标；不能依据单 seed 或 independent-window recall 将较大权重接入 OpenPI。

2026-08-07 已补齐两个必须分型、不可相互替代的 sequence evaluation scope，仍只作用于
轻量 baseline 的验证输出，不修改 loss、数据协议、RTC 或部署：

1. `fixed_forecast_lead_episode_timelines_v1`：固定一个 `lead_offset=k`，沿 Episode 时间轴
   比较连续 observation 所产生的 `prediction[t,k] -> target[t+k]`。16 个 lead 分开报告，
   禁止跨 lead 平均。这衡量滚动预测在固定 forecast lead 上的切换一致性，不等价于顺序执行
   一个 chunk。
2. `within_output_chunk_timelines_v1`：对每个独立产生的 H16 输出，在 offset 轴内比较标签
   切换、±1 timing、stable false switch、bounce 和 dwell。由于 H16 windows 重叠，transition
   support 明确是 `per_overlapping_output_window_v1`，不是 unique Episode event，也不是 RTC
   admission 后真正执行的 plan。

两者共享确定性 one-to-one transition matcher：先最大化 tolerance 内匹配数，再最小化总绝对
timing error。mask/gap 会断开序列；chunk offset 0 没有 chunk 内 predicted predecessor，若该点
是 canonical transition，只计 `boundary_unsupported`，不伪造前态。debug 默认关闭，只有显式
开启才保留逐 segment identity/dwell 细节。

同一 seed=23 natural/unweighted/weight=1 真实 validation 重跑后，aggregate 结果逐值复现
（macro balanced=`0.96307`、pair exact=`0.94837`），证明 sequence observer 没有改变训练行为。
但新指标说明 independent-window 高分不能推出标签 chunk 时序正确：

| within-output-chunk | enter exact / ±1 | exit exact / ±1 | stable false switch | single-frame interior segment |
|---|---:|---:|---:|---:|
| left | 0.0458 / 0.1348 | 0 / 0（support=2） | 0.00326 | 0.7200 |
| right | 0.0858 / 0.2592 | 0.0443 / 0.1276 | 0.00488 | 0.7968 |

fixed-lead 同样较弱：例如 lead 0 的 left-enter exact/±1=`0.0094/0.0377`，right-enter=
`0.0619/0.1239`，right-exit=`0.0190/0.0476`；后续 lead 有所变化但仍不能与
independent-window transition element recall 混用。较低的 stable false-switch rate 与较高的
single-frame interior segment rate 不矛盾：稳定边界总数巨大，少量额外切换却常形成短暂的
chunk 内 bounce。

因此当前证据确认的是数据/训练/预测 plumbing 正确，并同时否定“只看 aggregate 或
independent-window recall 就足以冻结 head/loss”的假设。它不说明 raw-RTC admission/execution
有 bug，也不回答真机任务效果。下一阶段应先在此轻量框架内比较能显式约束 H-step 连贯性的
最小结构/loss，并使用同一 frozen split 与上述两类 scope 验证；在此之前不接入 OpenPI
production head，也不调 raw-RTC。

#### P3.1：最小 persistence-loss gate（冻结后执行）

下一轮只验证一个不会改变输入/输出合同的候选：保持 current state-only 输入、并行
`[H,2]` occupancy logits、natural sampling、普通 state BCE 不变，额外加入：

```text
stable boundary =
    target[j] == target[j-1]
    and current/previous valid+loss mask are both true

L = L_state_bce
    + stable_persistence_weight
      * mean(abs(sigmoid(logit[j]) - sigmoid(logit[j-1])))
        over stable boundaries only
```

正式 transition 边界不施加 persistence penalty；mask/gap 两侧不建立邻接。配置默认
`stable_persistence_weight=0.0`，因此现有 baseline 数值和行为保持不变。首轮只预声明一个
非零候选 `0.1`，不根据结果继续扫描 0.05/0.2，不同时改变 transition weight、sampling、模型
结构或训练预算。对照使用 seed 23 和 31、1 epoch，并同时报告 aggregate、fixed-lead 与
within-output-chunk 指标。

`previous mode + dwell` 暂不与本 gate 混做。方向本身合理，但当前 source 能派生的是
`previous_canonical_collected_mode`，不是部署 resolver 的 `previous_resolved_mode` receipt；还需
先冻结以下合同：continuity/bootstrap 时的 unavailable 表达、dwell bucket、teacher-forcing
provenance，以及部署时只由已执行 due-pop receipt 更新 history。该上下文属于后续 P3.2，不能
用默认值伪装成已经接通的 resolved history，也不能让 future chunk label 提前进入历史。

2026-08-07 已完成 P3.1 本地实现：`ArmHoldLossConfig` 新增默认关闭的
`stable_persistence_weight=0.0`，完整组合函数命名为 `arm_hold_loss`，避免继续用
`weighted_bce` 误称 additive persistence loss。实现会验证 window 内所有 effective adjacent
target 与 canonical transition topology 一致；只在两侧 mask 连续且 transition=false 时计算
概率 L1。zero-weight 直接返回原 state loss。包含默认逐位兼容、stable-only、transition exclusion、
mask gap、topology conflict、负权重和 gradient 的范围内测试共 84 项通过。

远端实验尚未启动：只读资源检查发现同一 GPU 正在运行另一条正式
`deployments.openpi.training.prepared_skill_launch train`，因此按资源隔离原则不并发、不停止
该进程。待 GPU 空闲后再执行已冻结的 seed 23/31、weight 0/0.1 四组无写入对照；等待期间不
改变实验矩阵。

GPU 空闲后四组已按冻结矩阵完成，均 `exit_code=0`、`writes_artifacts=false`，未产生
checkpoint/cache/log。zero-weight seed=23 与旧 baseline 的 train/validation/sequence 数值逐值
一致。主要结果如下：

| seed | persistence | macro balanced | pair exact | chunk stable false L/R | chunk enter ±1 L/R | chunk right-exit ±1 |
|---:|---:|---:|---:|---:|---:|---:|
| 23 | 0 | 0.96307 | 0.94837 | 0.003258 / 0.004881 | 0.13475 / 0.25922 | 0.12760 |
| 23 | 0.1 | 0.96575 | 0.95159 | 0.002955 / 0.004552 | 0.13411 / 0.27880 | 0.12826 |
| 31 | 0 | 0.96501 | 0.95164 | 0.001855 / 0.003314 | 0.11476 / 0.09332 | 0.09049 |
| 31 | 0.1 | 0.96527 | 0.95179 | 0.001635 / 0.002705 | 0.10832 / 0.08295 | 0.06315 |

`0.1` 在两个 seed、左右侧都降低 stable false-switch，并小幅提高 aggregate/pair；说明
persistence term 确实约束了其声明的稳定区间。但它没有稳定改善 chunk transition timing：
seed 23 只有 right-enter 有明显改善，seed 31 的 left-enter/right-enter/right-exit ±1 均下降。
single-frame interior segment 也混合变化：seed 23 left `0.72 -> 0`、right
`0.7968 -> 0.8571`；seed 31 left `0.8889 -> 0.7308`，right 两组均无 interior segment support。
fixed-lead 0/7/15 同样是混合变化。

因此 P3.1 的决定是：实现保留、默认继续为 `0.0`，`0.1` 只作为可重放实验值，不进入正式
配置；不追加权重扫描。单独的 stable persistence regularization 不是解决 chunk transition
timing 的充分机制。下一步进入 P3.2 history-context 合同确认，首先解决 canonical-collected
history 与 deployment-resolved history 的 provenance/bootstrap 对齐，再决定是否实现模型输入。

#### P3.2：最小 previous-mode / dwell context gate（已冻结并进入验证）

本阶段只回答“轻量 label head 能否正确消费显式历史上下文”，不接入 OpenPI、不改变 RTC，
也不把训练 teacher-forcing 伪装成部署闭环。三组保持相同 state-only 输入、`[H,2]` 输出、
natural sampling、unweighted BCE、`stable_persistence_weight=0`、1 epoch 和 seed 23/31：

1. `none`：已有 P3.1 zero-weight baseline；
2. `previous_mode`：增加左右侧前一 canonical collected mode；
3. `previous_mode_dwell`：在相同 mode context 上增加离散 dwell bucket。

训练上下文使用 typed contract `dexteleop.arm_hold_history_context.v1`，provenance 固定为
`previous_canonical_collected_mode_v1`。每侧独立携带 `previous_mode`、`available_mask` 与
`dwell_steps`；不可用时 mode/dwell 的存储值都为 0。它只能从当前 window start 的前一 source
row 向后派生，并且不得跨越 Episode start、frame gap、continuity break、非 RUNNING row、
invalid/loss-masked label。当前候选 bucket 固定为：

```text
0 = unavailable
1 = dwell 1
2 = dwell 2
3 = dwell 3
4 = dwell 4..7
5 = dwell 8..15
6 = dwell 16+
```

模型以三类 mode embedding（unavailable/teleop/fixed-hold）编码每侧；dwell 版本将七类
dwell embedding 与 mode embedding相加。mode-only 与 mode+dwell 在同一 seed 下共享相同的
state encoder、mode embedding 和 head 初始化，因此比较只增加 dwell 信息，不混入公共参数
初始化差异。`label_context_mode=none` 保持默认，reader 也默认不派生 context，现有 profile、
manifest/hash、loss 和无上下文数值路径均不改变。

部署侧尚未接线。未来只能由 authoritative `ArmHoldSemanticResolver` commit 返回的 resolved
receipt 推进 `previous_resolved_mode` history；它不声称硬件已经执行，也不能从 future chunk
label、未提交的 staged resolution、未 admission 的预测或 canonical dataset label 冒充
deployment history。bootstrap/epoch/continuity reset 必须显式产生 unavailable。该
receipt-to-context adapter 是模型接入前的独立 gate，不属于本轮训练。

本地训练前验证已覆盖 context 派生、per-side invalid、全部结构边界、不可变存储、dwell
bucket、typed collation、context 模型双向 fail-closed、相同 seed 公共初始化与真实 Parquet
reader 路径；arm-hold 聚焦测试共 102 项通过。远端实验必须继续满足
`writes_artifacts=false`、`--no-motion`、只读 source dataset，不生成 checkpoint/cache/log。

远端冻结矩阵已完成。四个新增 context run 均 `exit_code=0`、
`writes_artifacts=false`，使用相同 manifest/profile、198,713 train windows、49,483 validation
windows 与 13,909 transition-window union；没有 checkpoint/cache/log。validation 每侧均有
48,528/49,483（98.0700%）context available，955 个 unavailable 与 955 个 continuity groups
的 bootstrap 数完全一致。两侧 dwell bucket 均覆盖 0..6，证明 unavailable 与 bucket 接线没有
退化成常量或隐式填充。

| seed | context | params | macro balanced | pair exact | chunk stable false L/R | chunk enter ±1 L/R | chunk right-exit ±1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 23 | none | 17,696 | 0.96307 | 0.94837 | 0.003258 / 0.004881 | 0.13475 / 0.25922 | 0.12760 |
| 23 | previous mode | 19,768 | 0.97948 | 0.97670 | 0.002089 / 0.004443 | 0.15925 / 0.19412 | 0.18750 |
| 23 | mode + dwell | 19,824 | 0.97911 | 0.97642 | 0.002136 / 0.004478 | 0.16699 / 0.19297 | 0.16797 |
| 31 | none | 17,696 | 0.96501 | 0.95164 | 0.001855 / 0.003314 | 0.11476 / 0.09332 | 0.09049 |
| 31 | previous mode | 19,768 | 0.98302 | 0.97714 | 0.002238 / 0.004907 | 0.21986 / 0.32431 | 0.22982 |
| 31 | mode + dwell | 19,824 | 0.98317 | 0.97698 | 0.002202 / 0.004961 | 0.21792 / 0.30818 | 0.22005 |

`previous_mode` 在两个 seed 都显著提高 aggregate/pair，并在 seed 31 同时提高三个有充分
support 的 chunk transition ±1 recall；说明历史 mode 是值得保留的最小候选。但 seed 23 的
right-enter 下降，stable false-switch 在 seed 31 也高于 no-context，不能只根据 aggregate
宣称 sequence 问题已解决。dwell 相对 mode-only 的 aggregate、pair、stable 与 transition
指标均为混合或轻微下降，没有跨 seed 的稳定增益。因此 P3.2 决定为：

1. 保留 `previous_mode` 与 `previous_mode_dwell` 实验实现，默认仍为 `none`；
2. `previous_mode` 晋升为下一 gate 的最小候选，但尚不进入 OpenPI/production；
3. dwell 不晋升、不调 bucket、不追加 embedding/loss sweep；
4. 下一步先实现模型框架无关的 deployment-resolved history state/adapter，并用 synthetic
   formal resolution receipt 做 reset、duplicate、late/out-of-order、epoch rotation 与 free-running
   rollout 正确性测试；该 adapter 不拥有 RTC plan、authority、safety 或 target execution；
5. 只有 resolved-history 输入与本训练 context 的 shape/availability 语义一致后，才考虑
   OpenPI hold head adapter。训练仍不得消费 future chunk label 作为 history。

执行期间主线依次运行 writing stage2/3/4 的 sample-check/norm-stats。首个 context run 后段与
stage4 norm-stats 有一段 CPU/I/O 重叠，因此 elapsed time 不用于性能比较；其余三组在主线任务
退出后串行执行。该重叠不改变 frozen seed、数据顺序、模型/loss 或报告数值语义。

#### P3.3：deployment-resolved history 独立 gate（已实现）

2026-08-08 完成了模型框架无关、target 无关的 resolved-history 基础层：

1. `ArmHoldHistoryContext` 从 `dataset_interface` 迁移到纯 `dataset_contracts`；旧
   `dataset_interface.arm_hold_windows` import 继续兼容。V1 shape/availability/dwell 约束不变，
   provenance 明确分为 `previous_canonical_collected_mode_v1` 与
   `previous_resolved_mode_v1`。
2. `dataset_interface` 只产生 canonical-collected teacher-forcing context；baseline batch
   可验证并消费两种 typed provenance，但现有训练 reader 仍只接受 canonical provenance。
3. `robot_motion.deploy.arm_hold_history` 提供显式 binding/epoch、content-addressed update、
   bounded state 与 snapshot。update 只保留 active binding hash、caller-owned sequence/id 和
   正式 resolution receipt，不重复镜像可推导的 contract/content hash。首次 committed
   resolution 后 context 才 available；每侧 mode 独立累计 dwell。
4. duplicate 完全幂等；同 sequence 不同 identity、stale、gap 与旧 binding update 全部
   fail-closed 且不修改状态。gap 可在缺失 sequence 补齐后按序恢复；binding/epoch rotation
   立即回到 unavailable。
5. 状态不保存无界 trace，不读取 future chunk，不拥有 model worker、RTC plan、authority、
   canonical safety、target preparation、dispatch 或 hardware acknowledgement。

synthetic free-running 测试按 `00 -> 10 -> 10 -> 11` 的 committed resolution 序列验证：每次
model cut 只能看到上一次 commit，得到的 per-side dwell 依次为 `[1,1] -> [1,2] -> [2,3]
-> [3,1]`，不存在 future-label leakage。连同现有 sync/raw-RTC arm-hold、OpenPI final-chunk、
package boundary 和训练测试，共 283 项通过。

本阶段刻意没有接入 `PolicyDeployRuntime`。审计发现两条兼容链的 commit 时点并不相同：

```text
sync_queue_v1:
    resolver.resolve()  # 内部 stage + commit
    -> pre-safety / PolicyActionEmission / target preparation

raw-RTC:
    resolver.stage()
    -> PolicyActionEmission + producer emission commit
    -> resolver.commit()
    -> application-owned target path
```

因此统一历史源必须定义为 authoritative resolver commit，而不是硬件执行，也不能简单监听
`PolicyActionEmission`：否则 sync 在后续 pre-safety/preparation failure 时可能与 resolver 的
真实内部状态错位。下一接线 gate 建议在 `PolicyDeployRuntime` 内为 resolver commit 分配独立、
单调的 `resolution_sequence`，并在 sync 的 `resolve()` 返回后和 RTC 的
`commit_staged_arm_hold()` 返回后各推进一次同一 adapter。该开关必须默认关闭，并与
`reset_arm_hold_semantics()` 同步 rotation。该 resolver-commit seam 已确认接受并在 P3.4
完成接线。

#### P3.4：默认关闭的 runtime commit-seam 接线（已实现）

`PolicyDeployRuntime` 构造函数新增可选 typed `arm_hold_history_binding`，没有 binding 时不创建
history state，现有 config、action、queue、emission 和 target 调用路径不变。启用时在构造阶段
验证 binding contract 与 `arm_hold_output` 完全一致；模型或本体 adapter 不需要复制 history
状态机。

推进点严格限定为：

```text
sync_queue_v1:
    resolver.resolve() committed
    -> resolved history advance

raw-RTC:
    formal producer emission admitted
    -> commit_staged_arm_hold()
    -> resolved history advance
```

staged-but-discarded、reference failure、formal emission failure 均不推进 history。每个 runtime
epoch 内 resolution sequence 从 1 单调递增；`reset_arm_hold_semantics()`、queue clear、prediction
schedule reset 和 RTC close 沿用既有 lifecycle。P3.5a 进一步收紧为：只有当前 epoch 已经存在
committed resolution 时，首次 reset 才将同一 binding id 旋转到下一 epoch并恢复 explicit
unavailable bootstrap；已经 unavailable 的重复 reset 完全幂等。snapshot 只声明 resolver
commit，不声明 target acceptance、dispatch 或硬件执行。

验证覆盖 sync 两次 commit/dwell、构造期 contract mismatch、default-off、reference failure 不推进、
history off/on action、emission hash 与 pop count 等价、raw-RTC emission commit、raw-RTC fault 不推进
及 close epoch rotation。arm-hold/OpenPI/package 组合回归共 287 项通过；额外 63 项无 arm-hold 的
queue/emission/next-step/RTC runtime 回归通过。
当前尚未把 snapshot 编码进任何 OpenPI 或其他模型输入；该步骤属于下一模型 adapter gate。

#### P3.5a：真实 lifecycle 下的幂等 history reset（已实现）

真实部署审计确认，故障锁存期间 application 可能在多个 control tick 重复调用 queue clear，部分
恢复路径还会连续调用 queue clear 与 prediction schedule reset；RTC HOLD/FAULT/close 也可能与
application reset 叠加。原先“每次 reset 都 epoch+1”会在正式启用 history binding 后产生没有
新 resolution 的虚假 epoch rotation。

当前规则冻结为：

1. history default-off 时不执行任何额外状态逻辑；
2. 当前 epoch 有 committed resolution 时，首次 reset 轮转一次、清空 context，并把下一
   resolution sequence 恢复为 1；
3. context 已经 unavailable 时，后续 semantic reset、queue clear、schedule reset、RTC fault 或
   close 不再轮转；
4. 新 epoch 再次 commit 后，下一真实 lifecycle boundary 可以正常再轮转一次；
5. checkpoint、artifact 或 policy source 变化不得借普通 reset 表达，必须构造新的 binding/runtime；
6. late worker completion 继续由 RTC producer/session epoch 拒绝，history epoch 不替代该职责。

验证覆盖首次 commit 后 reset、bootstrap 前重复 reset、queue-clear + schedule-reset 组合、重新
commit 后第二次 lifecycle boundary、raw-RTC emission 后 close + application 重复 reset，以及
raw-RTC 在 formal emission 前 fault/close 不制造 history epoch。arm-hold/RTC/OpenPI 聚焦回归
145 项通过；不启用 history 的 deploy/RTC/OpenPI/observation/position 兼容回归 138 项通过。

### P4：后续模型框架接入

1. 冻结最小输入/head/loss 后，OpenPI adapter 消费同一个 manifest。
2. 旧 OpenPI prepared Skill profiles、action loss、训练部署逻辑保持不变。
3. 0806 或未来 cube+hold 数据以新 profile revision 加入，不修改 V1 source membership。

## 8. 验收门

实现进入训练前必须满足：

1. 三个 source 的 schema/layout/hold/progress/stage identity 全部匹配。
2. profile 从 compatibility path 与 canonical relocated path解析得到相同 semantic identity。
3. source dataset 全程只读，所有派生物位于 organized root。
4. exact Episode identity 无重复、无 train/validation 泄漏。
5. train/validation 均覆盖左右类别、`00/01/10` 和四个 transition direction。
6. 稀有 left exit 满足最小 train/validation 事件门；否则 split build 失败。
7. window inventory 另行确认四个 transition direction 在 train/validation 均有 covered event
   和合法 window；Episode-level event gate 不得替代该检查。
8. 所有窗口不跨 Episode/continuity/invalid row，transition 首行语义正确。
9. profile、manifest、cache config 三种 hash 不混用。
10. debug-off 不生成逐 Episode 日志；debug-on 信息不进入 semantic hash。
11. 不修改现有 OpenPI profiles、source Parquet/video、RTC、Control Station、
    control composition、teleop 或 replay。

## 9. 实现前的决策门

建议按本规划进入 P0，但在 P1 真正冻结 cohort 前再确认一次：

- Episode admission 使用“成功、sealed、无 abnormal 的完整 Episode”；
- 0803/0804/0805 三个 source 全部进入 V1；
- 0806 暂不加入，未来作为新 profile revision 或 cross-day evaluation；
- `11` 不存在只记录 coverage gap，不作为失败；
- left-exit 稀缺由 episode-stratified split 保护，不通过复制窗口或改 label 解决。

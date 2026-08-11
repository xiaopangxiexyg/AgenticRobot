# Action Label Plan

## 目标

特殊控制必须保留可审计的事实记录，但不应为了记录控制过程而增加模型的分类难度。

- 模型侧对 init 只看到一个稳定标签：`init_pose`。
- 数采侧仍区分 `init_move` 与 `init_hold`，用来审计“是否真正到位且静止”。
- `init_move`/`init_hold` 是审计字段，不进入 dataset interface、训练 batch 或部署
  observation，也不是 mode target、embedding、token 或辅助 head 的类别。
- `task_hold` 是遥操员调整 retarget Home、暂停数采或 tracker recovery 的内部控制状态；
  它不作为数据集样本标签，不进入模型或部署接口。暂停时本来就不写样本；若需要诊断，
  只保留 episode/runtime 级审计记录。

## 对外模型语义

`teleop.arm_mode` 保持逐帧、逐侧的单一 mode 标签：

| ID | mode | 模型含义 |
| --- | --- | --- |
| 0 | `unknown` | 无法可靠判断 |
| 1 | `teleop` | 正常遥操 |
| 2 | `fixed_hold` | 特殊控制器固定该侧 |
| 3 | `init_pose` | 特殊控制器执行 init；不区分移动/静止 |
| 4 | `stopped` | 停止或非运行状态 |

`init_pose=3` 不会被拆成新的 mode ID。训练与部署只允许消费这一 mode；不得把
`init_move`、`init_hold`、`task_hold` 加入 action、observation、token、condition、
mode loss 或 phase loss。

## 仅审计的 init 记录

在 `teleop.arm_mode.<side>` 的完整 JSON 和可选 Parquet 审计列中记录：

- `init_phase`: `not_active`、`init_move`、`init_hold` 或 `unknown`。
- `init_phase_valid`: 是否有可信的真实反馈、init target 与连续时间基准。
- `init_target_position_error_m`、`init_target_rotation_error_deg`。
- `init_translation_speed_m_s`、`init_rotation_speed_deg_s`、`init_dwell_sec`。

判定只基于真实的、因果有效的 TCP 反馈和 init target，绝不根据下发 command 猜测。
当前默认基准与现有手眼标定自动静止检测一致：目标误差不超过 1.5 mm / 0.25°，并连续
满足 3 s。速度由相邻真实 TCP 反馈导出，并使用同一静止窗口量级的阈值；任一误差或速度
超限、时间倒退、目标切换或反馈无效都会重置 dwell，并标为 `init_move` 或 `unknown`。
这些阈值是审计判定，绝不改变 init 控制器的运动、限速或安全行为。

旧数据只有在同时具备真实 TCP、init target 和时间连续性时才可离线派生；否则写
`init_phase=unknown`、`init_phase_valid=0`。

## fix_arm 语义

`fix_arm` 的固定范围是整侧已配置的执行器：该侧机械臂以及已配置的 hand/gripper。
锁存和重放必须覆盖所有这些 effector，而不只覆盖 arm pose。

若某一侧没有末端执行器配置，允许退化为 `arm_only`，但必须在状态、command metadata
和 Parquet 审计中给出明确 warning；不得静默声称已经固定末端。审计 scope 为
`arm_and_end_effector` 或 `arm_only`。

## task_hold 与停止

- UI hold、retarget Home 重采、tracker loss/recovery 和 recording pause 不生成
  `task_hold` 数据帧。
- 非 RUNNING session 或已经写入的停止路径继续使用既有 `stopped`；它不是正常任务行为，
  不作为 BC action 目标。
- 运行时可保留 pause/re-home 的 episode summary，供问题定位；其内容不导入训练或部署。

## 数据边界

Parquet 是事实与审计的存储层，允许展开上述 init/fix 审计列；`sample_metadata_json`
保留完整上下文。`dataset_interface`、训练 adapter、canonical batch、部署 observation
和 policy 均只读取稳定的 `teleop.arm_mode.*` mode 合同，忽略 audit 列。

模型 batch 保持：

```text
executed_mode_history, requested_mode, mode_target, mode_valid,
affected_action_mask, policy_action_mask, special_controller_mask,
reference_action, reference_valid
```

不包含 `phase_target`、`phase_valid` 或任何 task-hold 字段。特殊控制器仍负责 init 的
安全归位和 fix 的锁存；模型只可理解或建议稳定的 mode。

## 验证与实施

1. 将 plan、dataset schema 与 writer manifest 固化为 mode/audit 分离的合同。
2. 在采集端从真实 TCP 反馈生成 init 审计 phase 与 Parquet 列；无真实反馈时明确无效。
3. 保持 `init_pose` 的既有 mode ID、26 维 action 和 observation 形状不变。
4. 将 fix-arm 收紧为整侧所有已配置 effector；无末端时 arm-only + warning。
5. 单测覆盖：init move、到位但未静止、dwell 后 hold、反馈缺失、目标切换、整侧 fix、
   arm-only warning，以及 Parquet 列稳定性。
6. 回归确认 dataset interface、训练 batch 和部署 observation 均未读取 audit 字段。

## 当前阶段：真机数据上的轻量 Hold Label 预测框架

### 目标冻结

当前目标不是修改 OpenPI 主模型，也不是启动 raw-RTC 真机执行，而是使用现有真机
采集数据建立一个独立、轻量、可重复的 hold-label prediction 框架，回答以下问题：

1. 现有 `teleop/fixed_hold` 标签是否能由部署时可获得的轻量输入学习；
2. 模型是否会退化为持续预测同一 label；
3. 哪种最小输入、输出结构和 label loss 能兼顾稳定区间与 mode transition；
4. 在固定数据划分上，训练结果能否稳定复现并通过 corner-case 检查。

这一阶段只评价 label prediction 的实现正确性和可学习性，不评价抓取成功率、碰撞率或
真实控制效果。实验不写回原始数据集，不启动设备，不下发动作，也不修改 teleop、replay、
Control Station、control composition 或 RTC producer 行为。

### 总体顺序

```text
真实数据只读审计
    -> episode-level train/validation split
    -> 独立轻量 label predictor
    -> 输入/输出/结构/loss 消融
    -> 冻结最小 label 方案
    -> 以可选 adapter 接入已有模型
    -> 保持 action 训练逻辑不变，只扫描全局 hold-label loss 权重
    -> no-action shadow 验证输出与部署合同
```

在轻量框架给出明确证据前，不提前修改既有模型 backbone、action head、RTC 请求协议或
部署 observation 合同。

### 阶段 L0：真实数据与划分合同

0803/0804/0805 writing-4-stage 数据的 source/cohort/split/window profile 采用独立、
模型框架无关的详细规划：
[`arm_hold_dataset_profile_plan.md`](arm_hold_dataset_profile_plan.md)。该 profile 不替代
OpenPI prepared Skill profile，也不拥有 loss、模型或部署参数。

数据源使用包含双臂 `teleop.arm_mode` 的真实机器人数据。首次实验固定：

- 只使用合法 `teleop/fixed_hold` target；`unknown/init_pose/stopped` 继续 mask；
- `fixed_hold` 必须满足 `arm_and_end_effector` scope；
- 原始 Parquet、视频和 metadata 全部只读；派生 cache、split manifest 和结果写入独立目录；
- 先按 Episode/轨迹划分 train/validation，再生成窗口，禁止同一轨迹窗口跨 split 泄漏；
- validation 保持自然分布，不做 transition oversampling；
- train 和 validation 都必须覆盖左右侧、进入/退出 hold、稳定 teleop、稳定 hold，以及数据中
  实际存在的双臂 mode 组合；
- split manifest 记录 dataset identity、episode membership、随机种子、horizon、label contract
  hash 和内容 hash，后续实验不得静默重划分；
- 第一轮关闭 RGB/state 数据增强，避免把增强收益或缺陷混入 label 可学习性判断。

窗口起点 `t` 的 transition 判定必须能够读取同 continuity segment 的 `t-1`；不得因为
窗口从 `t` 开始而漏掉首个 target keyframe 的切换。episode/continuity 边界、缺帧、padding
和被 mask 的 mode 都不得提供虚假的 previous-mode 或 dwell context。

### 阶段 L1：独立轻量预测基线

框架与现有 OpenPI 主模型解耦，先做单独分类实验。首轮候选输入按增量消融：

```text
A. state only
B. front RGB only
C. front RGB + state
D. C + previous applied mode + valid
E. D + previous-mode dwell steps
```

其中 previous applied mode 在训练数据中来自上一连续 keyframe 的 recorded effective mode；
它不是 action delta 推导出的 motion label。`valid=false` 覆盖 episode 起点、continuity reset、
ignored mode 和历史缺失。精确 dwell 由数据接口派生，分桶或 embedding 属于实验模型，不写死
进数据协议。

第一版模型保持小型、可单卡快速训练：

```text
current image encoder + state MLP
    -> fused current feature
    -> small hold head
    -> hold_logits[H, 2]
```

`H` 与计划接入的 action chunk horizon 绑定并记录在 manifest，不为某个 checkpoint 在代码中
硬编码。label head 并行预测整个 horizon，不做逐 keyframe 自回归。若使用 mode/dwell context，
它只进入 hold head，不进入 image/state backbone 或未来 action head。

必须同时保留以下退化基线：

- majority/constant predictor；
- previous-label persistence predictor；
- single-batch overfit；
- one-trajectory train/eval overfit；
- shuffled-label negative control。

这些基线用于区分“模型真正利用输入”与“仅依靠类别占比或轨迹泄漏”。

### 阶段 L2：Label loss 与 sampling 消融

基础 loss 为逐侧 `BCEWithLogits`，只在 `valid AND not-padding` 位置计算。第一轮使用一个
明确的 weighted BCE，而不是叠加多个语义不清的重复 BCE：

```text
L_label_base = weighted_BCE(
    logits,
    target,
    valid_mask,
    class_weight,
    exact_transition_weight,
)
```

消融顺序固定为：

1. natural sampling + unweighted BCE；
2. class-balanced loss 或 sampler，二者先分别测试；
3. natural windows + transition-centered train windows；
4. exact-transition weighted BCE；
5. 只有 raw probability 明显碎片化时，才测试非 transition 区间的低权重 persistence loss；
6. 只有 teacher-forcing gap 明显时，才测试受控的 early/late transition history corruption。

第一版 `persistence loss` 权重为零。未来启用时，pairwise mask 必须要求两端 label 有效、
非 padding、source index 连续、continuity 相同且 ground-truth 未切换。不得通过 label shift、
soft-label 扩散或修改正式 same-keyframe target 来改善 transition 指标。

### 阶段 L3：验证指标与最小方案选择

不能使用 aggregate accuracy 单独选方案。每次实验至少报告：

- 左右侧分别及总体的 balanced accuracy、precision、recall、F1、PR-AUC；
- `teleop->fixed_hold`、`fixed_hold->teleop` 的 exact-keyframe recall；
- transition 的 `+-1` keyframe recall 和 signed timing offset；
- 稳定区间误切换率、单帧 segment 数和预测 mode dwell 分布；
- `00/01/10/11` 中数据实际覆盖组合的混淆情况；
- probability calibration、阈值扫描以及 constant/persistence baseline 对照；
- 按 Episode、task phase、side 和 transition direction 的误差切片。

最小方案按以下优先级选择：

1. 不发生 constant-label collapse；
2. validation transition 指标显著优于 persistence baseline；
3. 稳定区间不出现不可接受的 mode bouncing；
4. train/validation gap 和不同 Episode 间方差可解释；
5. 在性能近似时选择输入更少、head 更小、部署状态更少的方案。

如果 `front RGB + state` 已足够，不增加 mode history；如果 previous mode 有明确增益但 dwell
没有，则不保留 dwell。完整 raw label history、transition head、动态规划 decoder 和 observation
长历史都不是第一阶段默认项。

### 阶段 L4：与已有模型的可选集成

只有 L3 冻结输入、`[H,2]` 输出、head 结构和 `L_label_base` 后，才进入已有模型项目。
集成必须满足：

- hold head 是 additive、artifact-bound、default-off；
- 没有 hold 合同的旧 checkpoint、训练配置、server/client 和部署路径完全不变；
- 连续 action 输出维度、action loss、action sampling 和 action deploy decode 保持原样；
- 如果最终方案需要 previous mode/dwell，再新增独立、typed、可选的 hold-context 输入合同；
  不追加到 canonical robot state，不藏入 metadata，也不复用无关 state augmentation；
- 训练端用 recorded previous applied mode，部署端用 previous committed semantic resolution；
  不读取硬件 execution acknowledgement，也不把未 due-pop 的 future label 当成历史；
- raw-RTC 将 context 作为 observation-time immutable input；普通 dwell 变化不使 active request
  失效，HOLD/FAULT/re-arm/profile epoch 变化仍沿用现有失效规则。

该阶段先做 one-trajectory overfit、固定 batch 和 no-action inference；通过后才扩大训练。

### 阶段 L5：联合训练只调整 Hold Label Loss 权重

模型集成后，冻结 L3 已选择的：

```text
label input
label output
hold-head structure
label target/mask
class/transition weighting
sampling
optimizer and action-training configuration
```

联合目标只增加一个全局系数：

```text
L_total = L_action_existing + lambda_hold * L_label_base
```

正式对照只扫描 `lambda_hold`，至少包含 `0` 作为完全兼容基线。不得在同一组实验中同时修改
action loss、数据增强、backbone、action horizon、RTC 参数或其他训练超参数，否则无法判断
差异是否来自 hold-label supervision。

通过条件包括：

- `lambda_hold=0` 数值行为与旧训练链路一致；
- 非零权重下 action loss、梯度和 action 输出保持有限且无非预期退化；
- hold head 不退化为 constant prediction；
- train/validation label 指标可复现；
- 输出严格满足 `[H,2]`、side order、horizon、概率和 artifact hash 合同。

### 阶段 L6：只读/No-action 部署验证

完成联合训练后，先进行真实 observation、`send_actions=false` 的部署验证：

- server/client 正确协商可选 hold input/output 合同；
- action 与 label horizon/keyframe identity 一致；
- raw-RTC crop/merge 后 label 始终随对应 action keyframe；
- 只有正式 due-pop label 进入 resolver；
- future label 不提前更新 history；
- threshold/debounce、HOLD/FAULT/reset 和 late result 行为可重放；
- debug-off 不增加额外逐帧日志，debug-on 才记录 context、logit、resolution 和 timing evidence。

该阶段仍不构成真机 motion 授权。是否进入受控动作测试另行确认。

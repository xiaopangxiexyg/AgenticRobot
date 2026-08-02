# Action Label：双臂 Hold 语义、训练接口与部署框架

> 状态：已实现数据契约、训练数据视图、同步部署和 default-off raw-RTC
> final-chunk 部署语义。OpenPI 内部的模型 head、训练循环和 checkpoint 导出仍由
> OpenPI 项目实现。
>
> 最后核对日期：2026-07-30。

## 1. 一句话结论

当前 v2.3 的 `fixed_hold` 标签表示某一侧的**机械臂与该侧末端执行器一起保持**，
不是“只让手臂静止、末端仍可运动”。

模型不改变原有连续 action 的维度，而是额外输出每个 action keyframe 对齐的两个
概率：

```text
arm_hold_probability[h] = [P(left_hold), P(right_hold)]
```

左右两侧是两个独立的 Bernoulli 语义，因此自然支持：

| 组合 | 左臂 | 右臂 |
|---|---|---|
| `00` | 运动 | 运动 |
| `01` | 运动 | Hold |
| `10` | Hold | 运动 |
| `11` | Hold | Hold |

这不是由 TCP 速度或关节速度计算出来的“物理静止分类”，而是数据采集时记录的
**操作/控制意图**。

## 2. 四层对象不要混用

| 层级 | 名称 | 形状/取值 | 含义 |
|---|---|---|---|
| 原始数据 | `teleop.arm_mode` | 每行、每侧一个枚举 | 采集时的控制模式事实 |
| 训练视图 | `arm_hold_target` | `[N, 2]`，0/1 | 从原始 mode 投影出的辅助监督 |
| 模型输出 | `arm_hold_probability` | `[B, H, 2]` 或部署时 `[H, 2]` | 每侧 Hold 概率 |
| 部署状态 | `moving` / `hold` | 每侧独立状态机 | 经过阈值、滞回和 debounce 后的执行语义 |

核心原则：

- 原始标签是事实，不能被训练代码重写。
- 训练 target 是只读派生视图，不增加或修改 Parquet action 列。
- 模型输出是概率，不是 logits；sigmoid 必须在模型/服务边界之前完成。
- 部署状态不是对单帧概率直接取整，而是 checkpoint 契约指定的有状态解析结果。

## 3. v2.3 原始标签协议

### 3.1 稳定 schema 和 mode ID

原始协议位于
[`dataset_contracts/arm_mode.py`](../../dataset_contracts/arm_mode.py)，schema 为：

```text
dexteleop_arm_mode_v2
```

稳定 mode ID：

| Label | ID | 是否进入 hold loss | 说明 |
|---|---:|---|---|
| `unknown` | 0 | 否 | 标签不可用或该侧不活跃 |
| `teleop` | 1 | 是，target=0 | 正常运动控制 |
| `fixed_hold` | 2 | 是，target=1 | 显式 Hold 意图 |
| `init_pose` | 3 | 否 | 初始化运动，不应学成正常移动或 Hold |
| `stopped` | 4 | 否 | 暂停、session 非 RUNNING 或显式停止 |

`stopped` 与 `fixed_hold` 必须区分：

- `fixed_hold` 是模型需要学习的动作语义；
- `stopped` 是生命周期/采集状态，当前训练契约将其忽略；
- 不能把暂停、故障或停止帧当成 Hold 正样本。

### 3.2 每行的关键列

writer 会保留 JSON 审计信息，并展开以下训练相关列：

```text
teleop.arm_mode.schema
teleop.arm_mode.available
teleop.arm_mode.left.id
teleop.arm_mode.left.label
teleop.arm_mode.left.scope
teleop.arm_mode.right.id
teleop.arm_mode.right.label
teleop.arm_mode.right.scope
teleop.arm_mode_json
```

另外还有 source、tags、init audit 等审计列。完整展开逻辑见
[`teleop_runtime/collection/arm_mode_labels.py`](../../teleop_runtime/collection/arm_mode_labels.py)
和
[`teleop_runtime/collection/schema_v21.py`](../../teleop_runtime/collection/schema_v21.py)。

### 3.3 `fixed_hold` 为什么包含末端

采集端 `fix_arm` 默认写入：

```json
{
  "label": "fixed_hold",
  "scope": "arm_and_end_effector"
}
```

训练投影只有在 `fixed_hold` 同时满足
`scope == "arm_and_end_effector"` 时才接受为正样本。否则：

- 普通模式下该侧 mask 为 invalid；
- `strict=True` 时直接抛出 `positive_scope_mismatch`。

因此当前协议不允许把“只 Hold 手臂”混入同一个正类。如果未来需要区分
`arm_only`、`end_effector_only`，应增加新的版本化语义，而不是改变 v1 的解释。

UMI 的纯操作员 annotation 使用不同的 scope（`tool_tcp_and_gripper`）；它不会被当前
RM65 `arm_hold_intent_v1` 误收为正样本，而会得到 `positive_scope_mismatch`。若未来
需要训练 UMI 同类语义，应定义独立的 embodiment/projection 契约。

### 3.4 新数据 metadata

当前 writer 在 `meta/info.json` 中声明：

```json
{
  "label_specs": {
    "teleop.arm_mode": {
      "schema": "dexteleop_arm_mode_v2",
      "columns_optional": true,
      "sides": ["left", "right"],
      "mode_ids": {
        "unknown": 0,
        "teleop": 1,
        "fixed_hold": 2,
        "init_pose": 3,
        "stopped": 4
      },
      "training_projections": {
        "arm_hold_intent_v1": {
          "schema": "dexteleop.arm_hold_training_contract.v1",
          "source_schema": "dexteleop_arm_mode_v2",
          "side_order": ["left", "right"],
          "positive_modes": ["fixed_hold"],
          "negative_modes": ["teleop"],
          "ignored_modes": ["unknown", "init_pose", "stopped"],
          "required_positive_scope": "arm_and_end_effector",
          "missing_label_policy": "mask_invalid_v1",
          "target_alignment": "same_indices_as_action_target_chunk_v1",
          "target_dtype": "float32",
          "contract_hash": "8ff801c71736d28c89b496c3406496bc348cac5aeea14527312d5a21f90c6600"
        }
      }
    }
  }
}
```

该稳定训练契约的 hash 由 canonical encoding 计算，不应由训练项目手写另一份算法。

## 4. 对已有数据的兼容

兼容策略是 additive、read-only：

1. **已带 `training_projections` 的新 v2.3 数据**
   使用 metadata 中的显式契约，并校验 contract hash。
2. **已有 `dexteleop_arm_mode_v2`，但没有 `training_projections` 的数据**
   自动绑定冻结的 `arm_hold_intent_v1`，标记
   `compatibility_mode=True`。
3. **完全没有 arm-mode label spec 的旧数据**
   仍可正常训练连续 action；arm-hold view 全部 invalid，辅助 loss 为 0。
4. **行级列缺失或标签不可用**
   不猜成 `teleop`，也不猜成负样本；只将对应位置 mask 掉。
5. **schema、ID/label 或 positive scope 自相矛盾**
   默认 mask 并记入 receipt；严格预检模式直接失败。

所以接入该 label 不需要迁移或重写旧数据，也不会改变旧 action tensor。

已知样例数据 `2026-07-27_dmtac_rm65_bimanual_position_v2_responsive`
应在正式训练前通过下述 coverage preflight 再确认具体 episode 的标签覆盖，而不应仅凭
数据集名称假设 `00/01/10/11` 都已覆盖。

## 5. `dataset_interface` 训练视图

公共入口：

```python
from dataset_interface import (
    project_arm_hold_training_view,
    summarize_arm_hold_coverage,
)
```

实现位于
[`dataset_interface/arm_hold.py`](../../dataset_interface/arm_hold.py)。

### 5.1 最小使用示例

```python
view = project_arm_hold_training_view(
    rows_or_columns,
    metadata=dataset_info,
    source_indices=action_target_indices,
    target_is_pad=action_target_is_pad,
    strict=True,
)

training = view.to_training_dict()
target = training["arm_hold_target"]                  # float32 [H, 2]
valid_mask = training["arm_hold_valid_mask"]          # bool [H, 2]
loss_mask = training["arm_hold_loss_mask"]            # bool [H, 2]
transition_mask = training["arm_hold_transition_mask"] # bool [H, 2]
```

对 batch 进行 stack 后，推荐的框架侧形状是：

```text
arm_hold_target          [B, H, 2] float32
arm_hold_valid_mask      [B, H, 2] bool
arm_hold_loss_mask       [B, H, 2] bool
arm_hold_transition_mask [B, H, 2] bool
```

side order 永远是：

```text
index 0 = left
index 1 = right
```

### 5.2 四种 mask/数组的区别

- `targets`：有效位置为 0/1；无效位置的数值没有训练含义。
- `valid_mask`：标签协议和行级值是否有效。
- `loss_mask`：`valid_mask AND NOT target_is_pad`，训练 loss 只看它。
- `transition_mask`：在同一 episode continuity segment 内，相邻且有效的 target
  是否发生切换；padding 和跨 continuity 边界不会被标记。
- `source_mode_ids`：保留原始枚举 ID，便于审计，不直接作为二分类 target。

### 5.3 与 action target 的对齐

训练契约规定：

```text
target_alignment = same_indices_as_action_target_chunk_v1
```

也就是说，若 action target 使用 episode 中的索引：

```python
action_target_indices = [t, t + 1, ..., t + H - 1]
```

arm-hold target 必须使用完全相同的 `source_indices`。不能按 observation 输入索引、
chunk 起点或预测完成时刻重新偏移一次。

如果 action chunk 做了 horizon prefix selection，label 必须用同一批 index 一起裁剪。

### 5.4 数据覆盖率预检

```python
report = summarize_arm_hold_coverage(view)
print(report.to_dict())
```

必须至少审查：

- 左右两侧 valid/positive/negative 数量；
- `00/01/10/11` 四种 pair 数量；
- `missing_pair_states`；
- 每侧 transition 数量；
- `invalid_reason_counts`；
- 是否 `perfectly_complementary`。

若数据只有 `01` 和 `10`，模型可能只学会“总有一只臂 Hold”，不能据此宣称它已学会
独立的双 Bernoulli 语义。`00` 和 `11` 必须有真实覆盖或在评估中明确列为缺失。

建议 coverage 至少按以下维度分别统计，而不是只看全数据汇总：

- train / validation / test split；
- task 或 task variant；
- dataset / collection date；
- 左右侧；
- episode；
- 四种组合；
- transition 与稳定区间。

## 6. OpenPI 训练 adapter 应怎样接入

DexTeleop 已提供纯 NumPy、框架无关的 target/mask/contract。OpenPI 项目需要实现的只是
tensor adapter、模型 head、loss 和 artifact 导出。

### 6.1 模型 head

推荐在不改变原 action head 输出维度的前提下增加：

```text
arm_hold_logits: [B, H, 2]
```

训练时 logits 直接交给 BCE-with-logits。部署响应前执行：

```python
arm_hold_probability = sigmoid(arm_hold_logits)
```

注意：

- 左右侧独立，不能在最后一维使用 softmax；
- 不建议改成一个 4-class head，因为协议的基本语义是两个独立侧；
- 原 action loss、action shape 和 action codec 均保持不变；
- action 与 label 必须共享同一个 horizon `H`。

### 6.2 基线 loss

令：

```text
z[b,h,s] = arm_hold_logits
y[b,h,s] = arm_hold_target
m[b,h,s] = arm_hold_loss_mask
```

基线辅助 loss：

```text
element_loss = BCEWithLogits(z, y)
L_hold = sum(element_loss * m) / max(sum(m), 1)
L_total = L_action + lambda_hold * L_hold
```

PyTorch 风格伪代码：

```python
per_item = torch.nn.functional.binary_cross_entropy_with_logits(
    arm_hold_logits,
    arm_hold_target,
    reduction="none",
)
mask = arm_hold_loss_mask.to(per_item.dtype)
hold_loss = (per_item * mask).sum() / mask.sum().clamp_min(1.0)
loss = action_loss + config.arm_hold_loss_weight * hold_loss
```

必须满足：

- 分母是有效、非 padding 的 side-keyframe 数；
- 全 batch 都没有有效 label 时，`L_hold=0`，不能产生 NaN；
- 无 label 数据仍贡献正常 action loss；
- 不用 target 数值代替 mask；
- 不让 padding、`unknown`、`init_pose`、`stopped` 进入 BCE。

### 6.3 class/transition weighting

当前 `dataset_interface` 故意不隐藏 class weight 或 transition weight。应先看 coverage，
再在训练配置中显式声明。

可选形式：

```text
weight = side_weight[s]
       * class_weight[s, y]
       * (transition_weight if transition_mask else 1)
```

但第一版建议从普通 masked BCE 开始，原因是：

- transition 过度加权可能产生概率尖峰和部署抖动；
- 只按全局正负比例加权可能掩盖 `00/11` 缺失；
- 左右侧分布可能不同；
- 部署已经有 hysteresis 和 debounce，训练 loss 不应偷偷复制第二套状态机。

任何 weighting 都应写入 checkpoint 训练配置和评估报告，而不是修改数据契约。

### 6.4 推荐训练指标

除 action 指标外，至少输出：

- masked BCE；
- 每侧 AUROC / AUPRC（样本足够时）；
- 每侧 precision、recall、F1；
- 四种组合的 confusion matrix；
- transition 附近与稳定区间分别统计的指标；
- 概率 calibration，例如 reliability diagram / ECE；
- 使用候选 enter/exit threshold 和 debounce 回放后的状态级指标；
- 每个 split 的 valid coverage。

最终部署阈值不能只根据单帧 accuracy 决定，而要根据状态回放评估：

- 误进入 Hold；
- 漏进入 Hold；
- Hold 释放延迟；
- transition 抖动；
- 每分钟错误切换次数；
- 四种组合的状态级 precision/recall。

### 6.5 OpenPI adapter 的最小职责

训练 adapter 应：

1. 用 action target 的同一 `source_indices` 调用
   `project_arm_hold_training_view()`；
2. 将返回数组转换为框架 tensor；
3. stack 成 `[B,H,2]`；
4. 计算显式 masked BCE；
5. 保留原 action loss；
6. 保存训练契约 hash、loss 配置和阈值校准结果；
7. 导出 `PolicyArmHoldOutputContract`；
8. 推理服务返回概率和 exact output contract hash。

训练 adapter 不应：

- 修改 v2.3 原始行；
- 从速度重新生成标签；
- 将缺失 label 填成 0；
- 改变连续 action dimension；
- 在 dataset_interface 内加入 PyTorch/JAX/OpenPI 依赖；
- 把部署 debounce 状态放入 dataset reader。

## 7. Checkpoint / policy artifact 契约

训练完成后，checkpoint 的 `policy_artifact.json` 在原有字段旁添加：

```json
{
  "arm_hold_output_contract": {
    "schema": "dexteleop_policy_arm_hold_output_v1",
    "output_key": "arm_hold_probability",
    "training_contract_hash": "8ff801c71736d28c89b496c3406496bc348cac5aeea14527312d5a21f90c6600",
    "side_order": ["left", "right"],
    "score_semantics": "hold_probability_v1",
    "target_alignment": "same_action_keyframe_v1",
    "positive_scope": "arm_and_end_effector",
    "supported_combinations": ["00", "01", "10", "11"],
    "enter_threshold": 0.8,
    "exit_threshold": 0.2,
    "enter_consecutive_steps": 2,
    "exit_consecutive_steps": 2
  },
  "arm_hold_output_contract_hash": "<由上述完整契约 canonical 计算>"
}
```

上面的 `0.8/0.2/2/2` 只是格式示例，不是框架默认值。每个 checkpoint 必须通过
validation replay 校准并显式写入：

```text
enter_threshold > exit_threshold
enter_consecutive_steps >= 1
exit_consecutive_steps >= 1
```

契约实现位于
[`dataset_contracts/policy_action_semantics.py`](../../dataset_contracts/policy_action_semantics.py)。

该字段位于原有 `outputs` 旁边，而不是增加一个 action tensor field，因此：

- 原 action dimension 不变；
- 原 layout hash 不因 action 维度变化；
- 没有该字段的旧 artifact 序列化保持不变；
- 有该字段时，部署端必须严格消费 label，不能静默忽略。

## 8. 推理服务响应

### 8.1 同步/OpenPI 响应

声明该 artifact 后，每个响应必须同时提供：

```json
{
  "actions": "<float32 [H, action_dim]>",
  "arm_hold_probability": "<float32 [H, 2]>",
  "arm_hold_output_contract_hash": "<exact artifact contract hash>"
}
```

校验要求：

- shape 必须精确为 `[H,2]`；
- `H` 与 action horizon 完全一致；
- 值必须 finite 且在 `[0,1]`；
- side order 固定为 `[left,right]`；
- hash 必须与 artifact 完全一致；
- 不接受 logits；
- 任一项失败时拒绝整个 chunk。

OpenPI handshake 还必须声明：

```json
{
  "action_semantics": {
    "arm_hold_output_key": "arm_hold_probability",
    "arm_hold_output_contract_hash": "<exact artifact contract hash>"
  }
}
```

### 8.2 为什么 label 要附着到每个 action

响应解析后，每一行被转换为：

```text
NormalizedAction
  ├── continuous/discrete action
  └── ArmHoldKeyframePrediction(probabilities, contract_hash)
```

这样 chunk prefix selection、delay cropping、queue/timed-plan freeze 都会同时处理 action
和 label，不需要维护第二个 index queue，也不会发生 action/label 错位。

## 9. 部署执行语义

### 9.1 总体顺序

```text
model/server
  -> action chunk + [H,2] probability sidecar
  -> contract/hash/shape/value validation
  -> 每个 probability row 附着到对应 action keyframe
  -> pose decode / absolute conversion
  -> keyframe 实际 pop 时解析 Hold 状态
  -> 用 Hold reference 替换该侧 arm + end-effector target
  -> canonical safety
  -> formal emission / dispatch
```

关键点是：**状态机在 keyframe 实际执行时推进，而不是在模型返回 chunk 时推进。**

否则被裁剪、被覆盖或永远没有执行的预测也会错误改变 Hold latch。

### 9.2 每侧状态机

每侧独立维护：

```text
moving --(p >= enter_threshold, 连续 N_enter 次)--> hold
hold   --(p <= exit_threshold,  连续 N_exit 次) --> moving
```

中间概率区间形成 hysteresis：

```text
exit_threshold < p < enter_threshold
```

此时保持当前状态，不立即切换。

### 9.3 进入 Hold 时保存什么

进入某侧 Hold 时，resolver 一次性锁存：

- 该侧机械臂的实际保持 reference；
- 若该侧存在末端执行器，则锁存该末端的保持 target；
- 若 action 使用 joint target，则锁存有限的 measured qpos。

持续 Hold 时复用同一 snapshot，不跟随测量噪声逐帧漂移。

若任一必须的 reference 不存在或非 finite：

- 不产生新的动作；
- 不允许只提交左侧、再让右侧失败；
- 双侧状态更新保持原子性。

实现位于
[`robot_motion/deploy/arm_hold.py`](../../robot_motion/deploy/arm_hold.py)。

### 9.4 与 safety 的关系

Hold 解析位于 pose decode 之后、canonical safety 之前：

```text
model action -> absolute decoded action -> hold resolution -> canonical safety
```

Hold 语义不能绕过 safety。被替换后的 target 仍需通过现有 workspace、step、IK、
collision 等执行安全边界。

启用 arm-hold resolver 时，不暴露未解析的 next-action lookahead，避免 position
lookahead 绕过下一 keyframe 的语义状态解析。

### 9.5 同步路径

同步 `PolicyDeployRuntime`：

1. 模型 chunk 解析后逐行绑定 label；
2. action 和 label 一起进行 execution-prefix selection；
3. queue pop 时用当前 observation 解析状态；
4. 生成包含 model action、resolved action 和 resolution receipt 的 emission；
5. reset/re-arm/clear/fault 时释放 latch。

## 10. raw-RTC final-chunk 路径

### 10.1 当前支持范围

| 路径 | arm-hold label |
|---|---|
| 同步 absolute `NormalizedAction` | 支持 |
| raw-RTC final chunk、`continuation=none`、absolute `legacy_normalized_v1` | 支持 |
| raw-RTC typed canonical | 暂不支持，fail closed |
| raw-RTC relative action | 暂不支持，fail closed |
| virtual RTC | 暂不支持，fail closed |
| model-RTC / flow-inpaint | 暂不支持，fail closed |
| OpenPI sim adapter | 暂不支持，fail closed |

当前 raw-RTC profile 仍是 default-off、shadow/no-action。已经验证的是完整的 label
协议、timed-plan 对齐和 emission 语义事务，不代表已经打开真实硬件 dispatch。

### 10.2 RTC 协议绑定

raw-RTC 的以下对象都携带 exact `arm_hold_output_contract_hash`：

- continuation capability；
- request spec；
- model response envelope；
- RTC emission source binding。

response 还携带 `[H,2]` probability matrix。adapter 和
`PolicyDeployRuntime` 的 contract hash 在 frontend 创建时必须一致，否则 worker
启动前就失败，避免“模型输出了 label，但 runtime 静默按普通动作执行”。

### 10.3 两阶段提交

RTC keyframe 的状态推进使用事务式流程：

```text
due-pop
  -> stage semantic resolution
  -> build formal emission
  -> producer commit emission
  -> commit arm-hold state/latch
```

`stage` 只计算候选状态、resolved action 和 receipt，不修改 resolver 的正式状态。

以下情况会 discard/reset staged state：

- reference 缺失；
- action adapter 失败；
- formal emission 构建失败；
- producer commit 被拒绝；
- lifecycle 进入 HOLD/FAULT；
- frontend close/reset；
- staged token 过期或重复提交。

因此 producer 没有接受的动作，不会推进 Hold debounce 或 latch。

## 11. 可回退与 fail-closed 的准确含义

“可回退”表示功能是 additive、default-off、artifact-bound：

- 不带 `arm_hold_output_contract` 的旧 checkpoint 走原路径；
- 没有 arm-mode 标签的旧数据仍可训练 action；
- 移除/切换到不带该契约的 artifact 即回到旧行为；
- 不改变 action dimension、旧 row layout 或旧无-label hash；
- raw-RTC 仍需显式选择对应 profile。

但对于**已经声明 arm-hold contract 的 artifact**，运行中不能把异常 label
“回退成 moving”：

- 缺少 sidecar：拒绝；
- hash 不一致：拒绝；
- shape/value 非法：拒绝；
- reference 不可用：no action / fault；
- runtime 没有 resolver：启动失败。

这是为了避免把协议错误解释为机械臂继续运动。

## 12. 训练与部署交付清单

### 数据侧

- [ ] `meta/info.json` 声明 `dexteleop_arm_mode_v2`；
- [ ] 新数据带 `arm_hold_intent_v1` training projection；
- [ ] `fixed_hold` scope 为 `arm_and_end_effector`；
- [ ] 按 split/task/side/episode 生成 coverage report；
- [ ] 确认 `00/01/10/11` 覆盖；
- [ ] 确认 transition 数量和 continuity 边界；
- [ ] 记录所有 invalid reason。

### OpenPI 训练侧

- [ ] action 和 hold target 使用完全相同的 source indices；
- [ ] 新增 `[B,H,2]` logits head，不改 action head；
- [ ] 使用 masked BCE-with-logits；
- [ ] label 全 invalid 时 loss 为 0；
- [ ] weighting 显式配置并记录；
- [ ] 评估概率 calibration 和状态回放；
- [ ] 校准 enter/exit threshold 与 debounce；
- [ ] 导出完整 artifact contract 和 hash。

### 推理服务侧

- [ ] handshake 回显 output key 和 exact hash；
- [ ] 返回 `[H,2]` probability，而不是 logits；
- [ ] action 和 probability horizon 一致；
- [ ] 不对概率执行第二次 sigmoid；
- [ ] 不在缺失 label 时填 0。

### 部署侧

- [ ] artifact、server、adapter、runtime 的 contract hash 一致；
- [ ] 仅在当前支持路径启用；
- [ ] Hold 解析位于 decode 后、safety 前；
- [ ] lifecycle reset 释放 latch；
- [ ] missing reference 为 no-action/fault；
- [ ] raw-RTC 保持 shadow，直到独立完成真实 dispatch gate。

## 13. 建议的验证命令

数据/训练视图：

```bash
python -m pytest -q tests/test_arm_hold_training_view.py
```

同步部署状态机与四种组合：

```bash
python -m pytest -q tests/test_deploy_arm_hold_semantics.py
```

raw-RTC/OpenPI/协议：

```bash
python -m pytest -q \
  tests/test_rtc_model_agnostic_protocol.py \
  tests/test_rtc_generic_deploy_runtime.py \
  tests/test_openpi_rtc_final_chunk_adapter.py \
  tests/test_deploy_rtc_frontend_shadow.py \
  tests/test_deploy_rtc_producer_state.py
```

这些测试均为 contract、fake transport 或 shadow/no-action 测试，不应启动真实设备。

## 14. 代码所有权与入口索引

| 责任 | 代码 |
|---|---|
| 原始 mode ID、训练投影契约、兼容解析 | [`dataset_contracts/arm_mode.py`](../../dataset_contracts/arm_mode.py) |
| policy artifact 的 hold output 契约 | [`dataset_contracts/policy_action_semantics.py`](../../dataset_contracts/policy_action_semantics.py) |
| 训练 target/mask/coverage | [`dataset_interface/arm_hold.py`](../../dataset_interface/arm_hold.py) |
| 采集 mode 事实生成 | [`teleop_runtime/collection/arm_mode_labels.py`](../../teleop_runtime/collection/arm_mode_labels.py) |
| writer metadata | [`teleop_runtime/collection/writer_v21.py`](../../teleop_runtime/collection/writer_v21.py) |
| 通用 remote response sidecar | [`deployments/client/action.py`](../../deployments/client/action.py) |
| OpenPI client 与 raw-RTC adapter | [`deployments/openpi/client/rtc.py`](../../deployments/openpi/client/rtc.py) |
| 每 keyframe 绑定 | [`robot_motion/deploy/action_semantics.py`](../../robot_motion/deploy/action_semantics.py) |
| Hold 状态机和事务 | [`robot_motion/deploy/arm_hold.py`](../../robot_motion/deploy/arm_hold.py) |
| 同步/timed emission | [`robot_motion/deploy/runtime.py`](../../robot_motion/deploy/runtime.py) |
| RTC producer commit 协调 | [`robot_motion/deploy/rtc_runtime.py`](../../robot_motion/deploy/rtc_runtime.py) |
| RTC capability/request/response | [`robot_motion/deploy/rtc_protocol.py`](../../robot_motion/deploy/rtc_protocol.py) |
| 部署 frontend/profile gate | [`teleop_runtime/deploy/run_policy.py`](../../teleop_runtime/deploy/run_policy.py) |

## 15. 后续 RTC 规划边界

下一阶段若接入 model-RTC / flow-inpaint，不能直接解除当前拒绝门。需要先明确：

1. conditioned prefix 是否携带已经解析的 Hold 状态，还是原始概率；
2. prefix mask、action mask 和 semantic mask 如何同一索引对齐；
3. flow/inpaint 重写 action 时是否必须同时重写 label；
4. model continuation receipt 如何绑定 semantic contract hash；
5. typed canonical/relative action 在哪个阶段附着并解析 sidecar；
6. 计划替换、delay crop 和 overlap merge 对 debounce 的因果语义；
7. producer commit 与 semantic commit 的唯一权威仍由谁持有；
8. RTC 真实 dispatch gate、fault recovery 和重新推理策略。

在这些协议冻结前，保持 model-RTC 路径 fail closed 是当前合理边界。

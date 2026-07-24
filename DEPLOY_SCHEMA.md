# DexTeleop Deploy Schema

本文档是 policy 与 DexTeleop deployment runtime 之间的规范契约。它定义模型输入、模型输出、
artifact/session identity 和兼容边界；不定义 dataset 磁盘格式，也不授予硬件 dispatch 权限。
离线数据以 [`DATASET_SCHEMA.md`](DATASET_SCHEMA.md) 为准，RTC 实施过程以
[`todos/dexteleop_rtc_landing_plan.md`](todos/dexteleop_rtc_landing_plan.md) 为准。

规范文字使用以下含义：`MUST`/“必须”是启动或推理时的硬约束，`MUST NOT`/“不得”表示
fail-closed 条件，`SHOULD`/“应”是兼容性建议，`MAY`/“可”表示可选能力。JSON/Python 示例
只用于说明结构；实际对象必须通过 typed contract 校验。

## 1. 版本、权威来源与边界

### 1.1 版本关系

这些版本位于不同层次，不能互相替代：

| 层次 | 标识/对象 | 作用 |
| --- | --- | --- |
| Deploy schema | `dexteleop_deploy_v1` / `dexteleop_deploy_v2` | runtime 与 policy 的总接口 |
| Policy artifact | `dexteleop_policy_artifact_v1` | checkpoint/server sidecar 序列化格式 |
| Dataset schema | `dexteleop_dataset_v2.2` / `dexteleop_dataset_v2.3` | 训练数据与 pose provenance |
| Tensor layout | `PolicyIOLayout` | state/action channel、维度、token vocabulary 的唯一语义来源 |
| RTC protocol | `dexteleop_rtc_protocol_v2` | 可选异步推理扩展，不改变基础 state/action 语义 |

`dexteleop_deploy_v2` 当前覆盖 v2.2 与 v2.3 dataset semantics；v2.3 是 pose/action-chunk
扩展，不是新的 deploy schema version。未携带 v2 artifact/layout 信号的旧模型继续按显式
`dexteleop_deploy_v1` 处理；不得仅根据向量宽度把旧模型重解释为 v2。

| Deploy mode | 输入/输出核心 | 当前边界 |
| --- | --- | --- |
| `legacy_v1` | 固定 38D state、26D action、q6 hand slots | 保持同步兼容 |
| `v22_q6_bridge` | 双臂 pose14 + typed end-effector，经显式 q6 slot projection | 继续走 legacy safety/dispatch |
| `v22_native` | 双臂 pose14 + typed `EndEffectorTarget` | contract 已定义；未配置 native safety/backend 时必须拒绝 |

### 1.2 权威来源顺序

```text
PolicyArtifactContract
  -> PolicyIOLayout / keys / dimensions / pose / camera identity
Server metadata
  -> 对 artifact 与协议的运行时 attestation；不得覆盖 artifact
ResolvedDeploySessionContract
  -> artifact 与本次 runtime end-effector/frame capability 的不可变合并
DeployStepInput
  -> T0 之前冻结的 causal observation cut
Policy output
  -> 必须匹配 artifact/layout，先解码再进入 runtime
RobotAction / safety / dispatcher
  -> 最终执行边界，模型不得绕过
```

部署输入、输出和 runtime execution 分为三层：

1. Runtime input：real、sim 或 replay 提供当前 observation；三者共享本契约，差异只在 source/adapter。
2. Policy output：模型返回 typed action chunk，由 artifact/layout 解释。
3. Runtime execution：canonical action 经过 safety、IK/reference 和 dispatcher；这些不属于模型 output schema。

`metadata`、`state_dict`、自由 JSON 扩展只能提供诊断信息，不能覆盖 typed artifact、session、layout、
pose-frame 或 modality contract。

## 2. 共同不变量

- v2 模型张量的唯一语义来源是 `PolicyIOLayout`；`arm_action_specs_by_side` 与
  `hand_action_specs_by_side` 仅用于 legacy v1/q6 execution compatibility。
- v2 固定保留 `left_pose7 + right_pose7`。当前 v2 session 要求 `active_sides=("left", "right")`；
  末端可以按侧为 `none` 或显式配置。
- v2 连续值使用 `float32`，离散 target/state 使用独立的 `int64` 矩阵；不得拼接、cast、静默丢弃 token。
- 缺失模态使用 `available`、`valid`、`mask` 和 `faults` 表示，不通过改变 schema 或改变 tensor 顺序表示。
- 模型输出 arm pose 和配置化 end-effector target；低层 arm qpos 只用于审计、初始化、debug 或特殊 controller，
  不进入 v2 默认 model state/action。
- `source_timestamp <= T0` 且 `available_timestamp <= T0`。模型 view 必须由不可变历史构造，不能在推理时读取
  可能已更新的 mutable backend state；`T0` 是 observation/action cut，不是组装完成时间。
- 任何 artifact、server metadata、frame、layout、normalization、horizon 或 dtype 不一致都必须在 safety/dispatch
  之前 fail-closed。
- safety、IK、position reference、hardware SDK 和 emergency hold 不得由 policy output 绕过。

## 3. Policy artifact contract

实现来源：[`dataset_contracts/policy_artifact.py`](dataset_contracts/policy_artifact.py)。
`PolicyArtifactContract` 是 checkpoint/server sidecar；artifact 必须能被 typed factory 或
`PolicyArtifactContract.from_dict()` 解析。

### 3.1 字段与派生规则

| 字段 | 必须性 | 语义 |
| --- | --- | --- |
| `schema` | 必须 | `dexteleop_policy_artifact_v1` |
| `dataset_schema` | 必须 | `dexteleop_dataset_v2.2` 或 `dexteleop_dataset_v2.3` |
| `deploy_schema_version` | 必须 | `dexteleop_deploy_v2` |
| `model_id` / `checkpoint_step` | 通用 v2 可选；OpenPI raw-RTC 及声明 endpoint identity 的 RTC profile 必须 | 已加载实例身份 |
| `source_datasets` | 可选 | 训练数据 provenance |
| `policy_io_layout` | 必须 | 嵌入的 `PolicyIOLayout` |
| `policy_io_layout_hash` | 必须 | 必须等于 layout canonical hash |
| `state_key` / `state_discrete_key` | 必须 | 输入字段名；当前默认 `state` / `state_discrete` |
| `action_key` / `action_discrete_key` | 必须 | 输出字段名；当前默认 `actions` / `actions_discrete` |
| `state_input_mode` | 必须 | `raw_state_v1` 或 `masked_normalized_concat_v1` |
| `state_augmentation` | 必须为 mapping | 训练期增强审计，不改变 dataset schema |
| `camera_input` | 视觉 artifact 必须；state-only 可为 `null` | 逻辑相机到 model/server 字段的有序映射 |
| `pose_action_chunk_contract` | v2.3 pose artifact 必须 | absolute/relative、anchor、tail、horizon |
| `pose_action_chunk_contract_hash` | 有 pose contract 时必须 | action-chunk contract 的 canonical hash |
| `pose_training_data_contract` | 有 pose contract 时必须 | native v2.3 或 v2.2 raw-identity provenance |
| `pose_frame_contract_hashes` | 有 pose contract 时至少一个 | 训练/运行 pose-frame identity |
| `metadata` | 可选 | 非权威扩展，不能覆盖上述字段 |

`inputs` 和 `outputs` 在 artifact JSON 中可以出现，但它们是由 layout、keys 和
`state_input_mode` 派生的 generated fields，不能单独修改维度或 dtype。float/token horizon、
dimension、layout hash 任一不一致都必须拒绝；即使 token dimension 为 0，也保留稳定的 int64 字段。

### 3.2 最小 artifact 形状

```json
{
  "schema": "dexteleop_policy_artifact_v1",
  "dataset_schema": "dexteleop_dataset_v2.2",
  "deploy_schema_version": "dexteleop_deploy_v2",
  "model_id": "my_policy",
  "checkpoint_step": 100000,
  "source_datasets": ["org/task-v22"],
  "policy_io_layout": "<canonical PolicyIOLayout object>",
  "policy_io_layout_hash": "sha256:<canonical-layout-hash>",
  "state_input_mode": "raw_state_v1",
  "state_augmentation": {},
  "camera_input": "<PolicyCameraInputContract or null>",
  "inputs": "<generated from layout and keys>",
  "outputs": "<generated from layout and keys>",
  "metadata": {}
}
```

当 `state_input_mode="masked_normalized_concat_v1"` 时，模型 float state 维度为
`2 * layout.state_dim`：前半为归一化连续 state，后半为同长度的 availability mask。模型/server
只能归一化前半，不能对 mask 做均值方差归一化。server metadata 必须声明相同
`input_state_mode` 与 `input_state_dim`。

### 3.3 Camera input contract

每个 `camera_input.inputs` 项必须绑定：

| 字段 | 含义 |
| --- | --- |
| `source_key` | DexTeleop 逻辑键，如 `front`、`left_wrist`、`right_wrist` |
| `model_input_key` | 送入模型的精确图像字段名 |
| `server_camera_slot` | server metadata 声明的能力槽 |

`inputs` 顺序就是模型输入顺序。`required=true` 时，启动必须提供完整同名逻辑视角；
`required=false` 允许无图像运行，但仍必须保留已提供图像的键、格式和 availability。旧 artifact 没有
`camera_input` 时可走 legacy compatibility；OpenPI raw-RTC adapter 则要求显式 camera contract，
state-only 模型使用 `required=false` 和空输入列表表示。artifact 还必须固定
`color_space="rgb"`、`layout="HWC"`、`dtype="uint8"`；NCHW/float 转换属于 model adapter。

### 3.4 v2.3 pose/action-chunk contract

`pose_action_chunk_contract` 只允许出现在 `dataset_schema=dexteleop_dataset_v2.3` 的 artifact 中，
并且必须同时提供 `pose_training_data_contract` 与至少一个 `pose_frame_contract_hashes`。
serialized artifact 使用字段名 `pose_action_chunk_contract`、`pose_training_data_contract`、
`pose_action_chunk_contract_hash` 和 `pose_frame_contract_hashes`；hash 必须分别匹配 typed contract 的 canonical hash。

```json
{
  "schema": "dexteleop_pose_action_chunk_v1",
  "action_horizon": 32,
  "action_pose_mode": "relative",
  "last_action_pose_mode": "relative",
  "absolute_pose_frame": "canonical_desk",
  "observation_state_pose_frame": "canonical_desk",
  "tail_policy": "drop",
  "padding_mask_key": "action_is_pad",
  "padding_value_policy": "repeat_last_converted_action",
  "relative_action_anchor": "chunk_start_actual_tcp",
  "relative_last_action_anchor": "current_actual_tcp",
  "normalization_excludes_padding": true
}
```

规则：

- `action_pose_mode` 与 `last_action_pose_mode` 当前必须相同；CLI、structured contract 与旧
  `metadata.pose_representation.action` 冲突时启动失败。
- state pose 是 canonical desk absolute pose；runtime observation 的 `actual_tcp_pose` 必须是 T0 之前最新测量值。
- absolute canonical-desk action 在 runtime 中先转换为当前机器 raw；relative action 因共同 world-frame 左变换可直接使用
  同一 T0 measured raw TCP 解码为 raw absolute；两条路径在 canonical safety/IK 前汇合。
- relative chunk 使用 chunk-start measured TCP 作为固定 anchor；relative action 在 safety/IK/dispatch 前解为
  runtime raw absolute pose，不得回退到 current/target/last command。
- v2.2 replay bridge 可以在每次 chunk prediction 时将存储的 absolute action 转为 relative，再走同一 runtime 解算器；
  该过程不得改写旧数据。
- 原生 v2.3 training 使用显式 runtime raw→canonical-desk frame contract。
- v2.2 raw identity compatibility 必须声明 `expected_runtime_source_id`，且 runtime raw→desk 对每侧均为 identity；
  不得把非 identity frame 当作该模型的兼容绑定。
- 原生 v2.3 的 `runtime_pose_binding` 必须是显式 runtime-raw→canonical-desk；v2.2 兼容路径必须是
  same-source identity raw→desk，并核对 `expected_runtime_source_id`。两者都必须保留
  `pose_action_chunk_contract_hash`，不能只放在自由 metadata 中。
- `tail_policy` 和 `action_horizon` 用于 training/checkpoint 审计；deploy 以 server 实际 chunk 为准，不在真机侧复现训练 padding。
- Vive 中间 frame 统一称为 `vive_retarget_frame`，不能直接当作 canonical dataset `desk`；RM65 `hardware_pose_6d` 是
  SDK base-frame audit，不是 canonical pose 或 relative anchor。

构造 provenance 应使用 typed factory，例如：

```python
from dataset_contracts import PoseTrainingDataContract

v22_compat = PoseTrainingDataContract.v22_raw_identity(active_sides=("left", "right"))
native_v23 = PoseTrainingDataContract.native_v23()
```

## 4. Resolved deploy session

实现来源：[`robot_motion/deploy/session_contract.py`](robot_motion/deploy/session_contract.py)。
session 是启动前解析出的不可变合并结果，不是第二份 artifact。

```json
{
  "deploy_schema_version": "dexteleop_deploy_v2",
  "policy_contract_mode": "v22_q6_bridge",
  "active_sides": ["left", "right"],
  "action_dim": 15,
  "action_discrete_dim": 1,
  "state_dim": 15,
  "state_discrete_dim": 1,
  "state_input_mode": "raw_state_v1",
  "policy_absolute_frame": "canonical_desk",
  "runtime_pose_frame_contract_hash": "sha256:<frame-contract-hash>",
  "policy_io_layout_hash": "sha256:<matching-layout-hash>",
  "end_effector_sides": ["left", "right"],
  "runtime_end_effectors": {},
  "q6_bridge_compatible": true,
  "hardware_output": false,
  "warnings": [],
  "metadata": {}
}
```

`policy_contract_mode` 只能是：

- `legacy_v1`：旧 26D/38D 路径，不要求 v2 layout hash。
- `v22_q6_bridge`：所有已配置 action channel 必须声明 `legacy_q6_slot_indices`，然后进入现有
  26D safety/dispatch 链。
- `v22_native`：不投影到 q6，使用 typed `EndEffectorTarget`。如果 native safety/backend adapter 未配置，
  必须 fail-closed，不得直接构造 backend command。

`v22_native + hardware_output=true` 还要求每个启用侧的
`RuntimeEndEffectorCapability.safety` 与 layout action channel 精确绑定：

```text
adapter + value_type + ordered names + units
continuous -> lower/upper + optional max_step/max_rate_per_sec/bounded rate interval
discrete   -> per-component token allowlist + optional transition allowlist
```

safety token不得超出layout vocabulary；continuous layout未声明units时不得进入真实native输出。模型artifact只声明
action channel，不能提供或覆盖runtime safety数值。

`policy_absolute_frame` 只能是 `runtime_raw` 或 `canonical_desk`；它描述模型 absolute pose 的解释，不能替代
`runtime_pose_frame_contract`。legacy v1 session 继续保留 `active_sides`、`enabled_effectors`、`control_hz`/
`policy_hz`、分侧 `arm_action_specs_by_side`/`hand_action_specs_by_side`、`camera_contract`、`tactile_contract`
和 `history_contract`。这些字段用于旧执行兼容与诊断；v2 visual policy 的 camera mapping 仍以 artifact 为准。

v2 runtime 必须在创建真实 backend 前完成 artifact、server metadata、layout、frame、end-effector capability
和 requested mode 的一致性检查。启用 layout 中不存在的末端是错误；真实输出缺少 layout 中需要的末端也是错误，
除非显式使用 dry-run 或 `allow_missing_v22_end_effectors` 兼容选项并记录 warning。

## 5. Server metadata handshake

server metadata 是 loaded endpoint 对 artifact 的运行时 attestation。客户端应核对至少以下字段：

| metadata 事实 | 必须与谁一致 |
| --- | --- |
| protocol/deploy schema | client contract / artifact |
| model id、checkpoint step | artifact（若 artifact 声明） |
| state key、state dim、state input mode、state token key/dim | artifact/layout |
| action key、action dim、action token key/dim | artifact/layout |
| `policy_io_layout_hash` | artifact layout hash |
| `action_horizon` | 当前 server 实际返回 chunk |
| camera slots | artifact `camera_input.server_camera_slots` |

任何 missing、drift、dtype/shape mismatch 或 horizon 不一致，都必须在 action 进入 safety 前拒绝。
server metadata 是证明，不是新的张量配置；自由 metadata 不能覆盖 artifact。

## 6. Deploy step input

每次 policy 推理使用一个逻辑 `DeployStepInput`。实际模型字段可以由 model-family adapter 映射，但
低维 key、dtype、dimension、layout hash 和 causal semantics 必须保持一致。

```json
{
  "schema_version": "dexteleop_deploy_v2",
  "timestamp": 1783420000.123,
  "monotonic_time": 12345.678,
  "step_index": 42,
  "active_sides": ["left", "right"],
  "enabled_effectors": ["left_arm", "right_arm"],
  "task": {"task_id": 0, "instruction": "pick up the object"},
  "observation": {},
  "history": {},
  "metadata": {}
}
```

组装完成的 `observation` 至少暴露 `state`、`state_discrete`、`policy_io_layout_hash`、`images`、
`tactile_by_side`、`last_action`、`last_action_discrete`、`last_action_valid` 和 `faults`；字段名可由
artifact 的 key contract 映射，但不得改变其语义。

### 6.1 Causal observation cut

每个模态应保留 `snapshot_sequence`、`source_timestamp`、`available_timestamp`、`source_clock`、
`available_clock` 和 `causal_snapshot_contract`。手部 telemetry、tactile、camera 各自记录自己的 sequence/time。
T0 前没有合格快照时，必须给出 `available=false`/fault，不得使用推理循环中刚读到的未来值。

### 6.2 Typed low-dimensional state

```text
state(float32)  = [left_pose7, right_pose7, left_eef_state_cont?, right_eef_state_cont?]
action(float32) = [left_pose7, right_pose7, left_eef_target_cont?, right_eef_target_cont?]
state_discrete(int64)  = [left_eef_state_tokens?, right_eef_state_tokens?]
action_discrete(int64) = [left_eef_target_tokens?, right_eef_target_tokens?]
```

- `layout.state_dim = 14 + 左右连续末端 state 维度之和`。
- `layout.action_dim = 14 + 左右连续末端 target 维度之和`。
- state/action end-effector dimensions 可以不同，但必须由 layout 显式声明。
- `observation.arm.qpos` 可作为 `[left_qpos6, right_qpos6]` 的 12D audit/init 字段；不属于默认 v2 model state。
- `state_dict` 是诊断视图，不是第二份模型 tensor 权威。
- 当前 OpenPI 默认低维键为 `state` / `state_discrete`；其他框架必须由 artifact 显式声明键名。
- legacy v1 的 `state_valid` 与 38D `state` 等长；未启用或不可用的 segment 可用 `NaN` 表示，且必须将
  对应 mask 置为 `false`。adapter 可以在模型侧将 NaN 替换为 0，但不得丢掉 validity 信息。

当使用 `masked_normalized_concat_v1` 时，model `state` 长度为 `2 * layout.state_dim`，mask 规则见 3.2。
部署 observation 不读取 init/fix audit columns，也不读取 `task_hold`。

### 6.3 Images

`images` 按 logical camera key 命名。每项至少包含：

```json
{
  "available": true,
  "timestamp": 1783420000.100,
  "age_sec": 0.023,
  "data": "<HWC uint8 RGB tensor>",
  "color_space": "rgb"
}
```

`required=false` 时保留 key 并置 `available=false`/fault；`required=true` 时缺失应阻止推理或进入 hold。
模型需要 NCHW/float 等转换时，由 model adapter 完成，不能改变 artifact 中的 logical-key 顺序。
artifact 选择的视角数与 server camera slots 必须在打开相机前核对；required visual artifact 不得以 `camera-mode=none`
或不足视角启动。旧 artifact 没有 `camera_input` 时才保留 legacy permissive fallback。

### 6.4 Tactile

目前 tactile 是 runtime modality/availability contract，不是一个全局固定的 artifact tensor schema。不同末端可使用
backend-native vector 或 dict；缺失时保留 side 并置 `available=false`，不得把 stale data 伪装成有效数据。
如果未来模型直接消费 typed tactile channels，应先增加 artifact-level contract，再宣称跨框架一致。

### 6.5 History and last action

history 是可选的 model-adapter contract，不替代当前 observation。每个 stream 必须带 timestamps 与
available/valid mask；不足 horizon 时可 repeat-last 或 zero-pad，但必须标记 padding。`last_action` 必须来自
已经 safety-filter 接受或实际 dispatch 的 action，而不是未过滤 raw output。relative policy 的 last-action frame/mode
必须与 `pose_action_chunk_contract` 一致。

## 7. Policy output

### 7.1 v2 typed output

```json
{
  "actions": "<float32[horizon, action_dim]>",
  "actions_discrete": "<int64[horizon, action_discrete_dim]>",
  "policy_io_layout_hash": "sha256:<matching-layout-hash>",
  "metadata": {}
}
```

`actions` 与 `actions_discrete` horizon 必须一致；continuous 值必须 finite；token 必须是精确 int64 且属于 layout
声明的 vocabulary。兼容键 `action` / `action.discrete` 只有在 server/artifact 显式声明时才可使用。

v2 output 的两条执行路径：

```text
v22_q6_bridge:
typed output -> explicit legacy_q6_slot_indices -> NormalizedAction(26D)
             -> existing safety -> RobotAction -> TeleopCommand -> dispatcher

v22_native:
typed output -> RobotAction(typed EndEffectorTarget) -> TeleopCommand(typed target)
             -> native safety/backend adapter
```

离散 target 在 typed path 中保持 int64；只有 layout category、backend value 和 q6 slot 都显式声明时，q6 bridge 才可执行。

### 7.2 Legacy v1 output

v1 固定布局：

```text
state = [left_arm_qpos6, left_arm_pose7, left_hand6,
         right_arm_qpos6, right_arm_pose7, right_hand6]  # 38D
action = [left_arm_pose7, right_arm_pose7, left_hand6, right_hand6]  # 26D
```

完整 v1 canonical action chunk 的语义是：

```json
{
  "schema_version": "dexteleop_deploy_v1",
  "horizon": 1,
  "actions": [
    {
      "active_effectors": ["right_arm"],
      "arm_pose_by_side": {
        "right": [0.30, -0.10, 0.25, 0.0, 0.0, 0.0, 1.0]
      },
      "hand_qpos_by_side": {},
      "metadata": {}
    }
  ],
  "metadata": {"policy_name": "my_policy"}
}
```

当前 legacy runtime 还接受以下显式兼容 short forms：

- `np.ndarray(shape=(26,))`：full action vector；
- `np.ndarray(shape=(T, 26))`：legacy action chunk；
- active-side compact arm pose `(7,)` 或双臂 `(14,)`；
- `{right_arm_pose, right_hand}` 等带 side 的 mapping；
- 单臂时 `{arm, hand}` aliases。

这些 short forms 只属于 legacy v1 或显式 q6 bridge；新 checkpoint 应保留 canonical specs/metadata。完整 canonical JSON
在进入 runtime 前必须转换为 `NormalizedAction` 或 typed `RobotAction`。

### 7.3 Chunk horizon and execution prefix

artifact/training horizon、server-return horizon 与 runtime execution horizon 不是同一字段。runtime 可配置
`action_chunk_execution_steps=X`：`None` 保持完整 chunk，`X>0` 只 admission 前 X 个 action。该限制只改变调度 admission，
不得改变 pose representation、relative anchor、safety 或 history semantics；real 与 MuJoCo deploy 应保持同一语义。

## 8. Runtime normalization and fault handling

### 8.1 Normalization boundary

```text
policy output
  -> typed/model-space validation
  -> pose/codec decode to canonical absolute action
  -> NormalizedAction (v1/q6 bridge) or typed RobotAction (v2 native)
  -> canonical safety / IK / reference
  -> TeleopCommand
  -> backend dispatcher
```

RTC逐项执行的顺序固定为：

```text
timed-plan due-pop
  -> PolicyActionEmission
  -> proposal/source admission
  -> canonical safety (safe | modified | fault)
  -> resolver/reference/backend
  -> downstream receipt
```

Emission记录pre-safety intent，不以safety或backend receipt为前置条件；只有RTC-3d future preview/lookahead需要在正式
emission前取得独立preview acceptance。

`RobotAction` 是 action semantic boundary，`TeleopCommand` 是 dispatch boundary。model adapter 不得 merge/pop plan、调用
safety/reference/SDK 或取得 dispatch authority。

额外执行规则：v2 缺少任一 pose7、出现非有限值或 layout 中声明的末端缺失时，整条 candidate/chunk 拒绝；v1
只有 inactive side 可以省略，active effector 缺失只能显式 hold/fallback。q6 bridge 仍使用既有 slot projection 与
范围裁剪，裁剪/拒绝结果必须进入 action metadata；不得由 adapter 直接发送 backend command。

### 8.2 Fail-closed matrix

| fault | 处理 |
| --- | --- |
| artifact/schema/layout/hash mismatch | 启动前拒绝 |
| server metadata mismatch/drift | 不接收 result；重新 handshake 或退出 |
| required camera/telemetry/tactile snapshot 缺失 | 阻止本次推理或进入 hold |
| optional modality 缺失 | 保留 key，置 availability/mask/fault |
| output shape/dtype/token/finite 校验失败 | 丢弃整个 chunk，不部分下发 |
| pose/frame/anchor 无效 | 在 safety/dispatch 前拒绝 |
| safety/IK/reference fault | hold/取消后续动作，记录 fault |
| worker/transport late result | 按 request/epoch/session identity 拒绝，不能静默 fallback 同步推理 |

每次 fallback 必须记录 `fallback_reason`、是否进入 safety、是否 dispatch；fault 与 hold 的最终 owner 属于 runtime，
不是模型。

## 9. Optional RTC extension

RTC 是 `dexteleop_deploy_v2` 的可选推理扩展，默认关闭，不新增 state/action tensor 语义。

### 9.1 raw-RTC

```text
existing observation -> final action chunk
model/server unchanged
client worker -> timed plan -> admission/TTL/epoch -> PolicyActionEmission
continuation=none
```

raw-RTC 的模型不消费 future action prefix；request 不得携带 prefix/mask/conditioning receipt，且必须声明
`continuation=none`。OpenPI 当前 final-chunk 接入属于 raw-RTC。raw-RTC adapter 只负责 input preparation、
blocking model call、response validation 和 model-space decode，不能拥有 worker、plan、safety、reference 或 dispatch authority。

RTC 的 `observation_cut_id` 是 versioned causal identity，不是大体积 sensor payload 的内容哈希。v2 identity 必须绑定
runtime session、control domain、producer/epoch、policy node、单调 request sequence、observation contract、timestamp 和
step index；control thread 不得为此扫描 camera/tactile bytes。冻结的 actor view 随 work item 进入 worker，worker 再对实际
model adapter 输出计算 `model_input_hash`；response 必须回显该 hash。两者职责不可混用：

```text
observation_cut_id = request/epoch/anchor/plan 的因果身份
model_input_hash   = 本次实际送入模型内容的精确证明
```

absolute raw-RTC 的 decoder 必须在 `PolicyChunkResult` 前产生 typed canonical action，并用同一 receipt 绑定
`observation_cut_id`、codec/frame/layout/normalization、conversion contract、source chunk hash、canonical chunk hash
和连续 source index。identity conversion 也必须显式留据；typed 与 legacy result 不得混装。OpenPI typed output
必须由显式 `typed_canonical_v1` 配置启用，默认 legacy codec/result 保持不变。

relative raw-RTC 必须在 client request 构造时冻结
`PolicyActionRepresentationContext`。该 versioned context 绑定 source pose mode、observation
cut/contract/timestamp/step、active arm group、每侧 reference/controlled frame、SE(3) 约定，以及每侧同一
observation cut 的 measured actual-TCP anchor；整个 chunk 只能使用这组 anchor：

```text
T_absolute[side, i] = T_request_anchor[side] @ T_relative[side, i]
```

不得在 completion、merge 或 emission 时重新采样/重锚定。SE(3) 变换只作用于 arm pose，typed gripper/hand
continuous target 与 discrete token 保持原 component identity。context hash 进入 request hash，relative request 缺失
context、cut/time/frame/arm coverage 不一致或 receipt anchor identity 不匹配时，必须在模型调用或 plan admission 前
fail closed。该 context 是 client/runtime 内部因果事实，不加入 raw-RTC model/server payload；模型仍只接收既有
observation 并返回 final chunk。absolute frontend 可携带显式 `anchor_policy=none_v1` context；没有 context 的既有
absolute request/hash继续按 v1 compatibility 解释。

### 9.2 model-RTC

model-RTC 才允许模型实际消费 versioned future-action prefix/mask，并返回可验证的
`RtcConditioningReceiptV2`，证明实际使用的 prefix、mask、delay、implementation parameters 和 execution horizon。
`continuation != none` 必须走 `async_timed_plan`；当前 typed prefix 不支持离散 token continuation 时必须拒绝。
model-RTC 的 receipt/identity promotion 不得回写 raw-RTC 的 legacy model/server 调用。

### 9.3 通用 adapter boundary

adapter 必须从已加载模型实例导出 capability、request spec、artifact/layout/frame/normalization/horizon identity，
不得按 model name、checkpoint name、config name 或固定维度猜测。通用入口是结构化
`RtcModelRuntimeAdapter`/`SimpleRtcModelRuntimeAdapter`；具体 worker/session/transport lifecycle 见
[`deployments/client/README.md`](deployments/client/README.md) 和
[`todos/dexteleop_rtc_landing_plan.md`](todos/dexteleop_rtc_landing_plan.md)。

对于无自定义 session/reconnect 生命周期且直接返回完整 chunk 的接口，优先使用
`create_final_chunk_rtc_runtime_adapter()`：框架只提供 instance-derived `FinalChunkRtcBinding`、input preparation、
blocking model call、精确 input hash、output mapping 和 canonical decoder；capability/request spec 必须由同一 binding
派生一次。mapping response 的动作 key 必须显式且精确，未选择的诊断字段可忽略；连续与离散输出保持独立 typed matrix。
需要自定义 session/metadata/reconnect 的 transport 继续直接实现 `RtcModelRuntimeAdapter`，不得为复用 helper 削弱生命周期校验。

raw-RTC source 与 execution target 的兼容性使用
`assess_policy_emission_target_compatibility()`做纯评估：必须复用既有 `TargetRuntimeActionContract`、
`ActionComponentCatalog`和`ReferenceActivationSpec`，绑定 target clock、显式授权 component scope、coupled group、
action space、pose frame/transform及 activation lead。legacy representation receipt 与 typed canonicalization receipt
是两种不同 source proof；legacy pose frame fact 还必须携带独立 binding hash，不能由调用方只抄 target frame。
两类 proof 均不等于 target safety、IK、reference acceptance 或 motion authority。assessment accepted 只允许进入
后续 target preflight，不得直接 dispatch。

通过 assessment 后可调用 `prepare_policy_emission_target_proposal()` 形成 target-neutral
`ActionProposalDraft`。该 preparation 必须绑定 exact emission ID/hash、assessment、target contract、component catalog、
activation spec、representation proof 和 producer identity；target 的 control domain/embodiment identity只能来自
`TargetRuntimeActionContract`。它固定 `authority_request=none`，不写 ControlTrace，也不等于 proposal admission、
target admission、safety receipt 或 dispatch。assessment rejected 时不得产生 proposal draft。

RTC capability/negotiation 至少绑定：`capability_id`/`provider_id`/`artifact_hash`/`protocol_version`、scheduler modes、
continuation mode、implementation id/`implementation_parameters_hash`、model action space 与 float/token dims、支持的 pose modes
和 component scope、`action_codec_hash`/`frame_contract_hash`/`layout_hash`/`normalization_hash`、prediction/execution
horizon 范围、`prefix_requirement`，以及 prefix-mask/conditioning-receipt 能力。客户端与 server 必须逐项核对并保留 typed rejection；
capability drift、hash drift 或 horizon 不满足时不得降级成另一种 model/RTC 语义。

RTC profile MUST remain default-off；`--rtc-profile off` 时 server metadata、legacy `Policy.infer()`、sampler、websocket
payload、sync queue 顺序和 existing action-chunk behavior 不得改变。OpenPI raw-RTC 的 launcher/profile 细节见
[`deployments/openpi/client/README.md`](deployments/openpi/client/README.md)。

### 9.3.1 Live observation/transport no-action shadow

`--rtc-real-shadow` 是 raw-RTC 的显式部署开关，语义是：

```text
real arm/hand/camera observation
  -> existing model/server transport
  -> RTC worker/timed-plan/admission/safety shadow
  -> no actuator dispatch
```

它不是数据集 replay；数据集/合成 observation 只能验证离线 codec、adapter 和 plan，不覆盖真实设备采样、相机时序或
网络延迟。该开关必须与 `raw_rtc_final_chunk_15_30_60_120_v1`、`send_actions=false` 一起使用；没有显式开关时，raw-RTC
不接受 live arm/hand endpoints。

live shadow 下 backend MUST expose measured observation while `observation_only=true`：RM65 不得配置奇异点阈值、同步工具
帧、启动 force/CANFD 输出或发送/停止 motion；DMTAC/hand 不得清 fault、写 speed/torque、open/reset 或发送 command。该
模式只产生 `rtc_real_shadow_ready` 类验证证据，不授予 motion authorization；teleop、replay、sync dry-run 和 model-RTC
均不继承此开关。

若开启 `--debug`，raw-RTC 还会记录 model-neutral 的 `rtc_transport_exchange` 事件，包含 observation cut、输入/输出
identity hash、chunk horizon 以及 inference/result latency；该事件只用于观测，不能替代 plan admission、safety、dispatch
或 worker-drain evidence。

`--rtc-request-validity-steps`和`--rtc-worker-stop-timeout-sec`只在显式RTC profile下有效，分别绑定request/plan TTL和
frontend close的bounded worker wait；默认值`10`/`0`保持既有fail-closed与non-waiting语义。模型包、checkpoint名称或
server latency不得隐式改写这两个值，非RTC caller设置override必须拒绝。

### 9.4 vla-scratch migration boundary

外部 `vla-scratch` 仓库保持不修改。DexTeleop 侧只提供 artifact-bound processed-sample binding 和
`infer(dict) -> final chunk` adapter；server/client 通过 transport 进程边界互通，不要求共享 virtual environment。
输入顺序、state/action dimension、horizon、frame/layout/normalization identity 必须来自外部 artifact；不得把 LIBERO 默认
7D action 或 dataset 推导值猜成 DexTeleop canonical action。无 server attestation 时只允许 scoped no-action verification。
详细接入见 [`deployments/vla_scratch/README.md`](deployments/vla_scratch/README.md)。

## 10. Compatibility matrix

This matrix describes current model/deployment adapter profiles; it is not the
topology contract of `control_composition`. At the mixed-source boundary an
arm is always a Cartesian pose component. A robot's 6DoF, 7DoF, or other joint
count is a downstream resolver/capability property, not a mixer dimension.
Current left/right pose14 and per-side end-effector rules remain unchanged for
compatibility. A future component-keyed carrier may support single arm and
multiple end-effectors per arm only through a new versioned layout/adapter; it
must not reinterpret an existing checkpoint or silently change v1/v2 tensor
order.

| 维度 | legacy v1 | v2 |
| --- | --- | --- |
| active sides | 单臂或双臂 | layout 固定双臂 pose core；session 当前要求两侧 |
| arm state | 38D 中含 qpos + pose | pose14；qpos 仅 audit/init |
| arm action | 26D pose + q6 slots | typed continuous/token channels |
| hand | side-specific q6 slot spec | per-side `EndEffectorSpec`，可 none/continuous/discrete |
| camera | legacy camera contract / permissive fallback | artifact `camera_input` 为 authority |
| tactile | runtime-native availability | 仍是 runtime modality，未来 typed channel 需新 contract |
| output | short forms permitted | layout-bound matrices only，除非显式 bridge |
| single arm | compact aliases或 full vector + invalid side | 新增单臂 v2 必须有新 layout/schema，不能删除 pose14 slot |

### End-effector rules

v2 每侧 `EndEffectorSpec` 至少声明 `side`、`adapter`、state/target `value_type`、dimension、names/units/normalization，
以及离散 vocabulary；legacy q6 projection 使用可选 `legacy_q6_slot_indices`。layout 中存在的末端必须有同侧 runtime capability
且 adapter 一致；runtime 额外启用 layout 中不存在的末端也是错误。dry-run 可以缺 backend，但必须 warning。

v1 hand spec 使用 `slot_count`、`active_dof`、`active_mask`、`joint_names`、`command_space`、`slot_map` 和可选
`invalid_slot_value`。DMTAC 当前常见为 6-slot vector、active slot 0；inactive slot 使用声明的有限 padding，runtime
只向 backend 发送 active value。tactile availability（当前实现可同时记录 `tactile_valid_by_side`）与 hand command
availability 独立。

### Single-arm rules

legacy `active_sides=["right"]` 可只提供 right state/image/tactile，left side 在 full vector 中以 invalid/NaN 表示，或返回
compact `(7,)` / `{arm, hand}`。v2 当前固定双臂 pose core；真实 runtime 缺少一侧必须失败，不能用 NaN、默认姿态或上一条命令伪造。

## 11. Minimal conformance checklist

一个新模型/框架接入至少必须证明：

1. artifact、dataset/deploy schema、layout hash 可解析且一致；
2. input/output keys、float/token dims、dtype、horizon 与 server metadata 一致；
3. camera logical key、model input key、server slot 顺序一致；
4. state_input_mode、normalization、mask 规则明确；
5. v2.3 pose mode、anchor、frame contract、training provenance 可重建；
6. observation 在 T0 前冻结，所有 modality 有 timestamp/availability/mask；
7. output 经过 finite、shape、token vocabulary、layout/hash validation；
8. adapter 只返回 decoded candidate/result，不拥有 safety、plan、pop、dispatch；
9. runtime fault、hold、late result 和 fallback 可观测且 fail-closed；
10. legacy sync path、sim/replay compatibility 和 default-off behavior 未改变；
11. 如启用 raw-RTC，明确 `continuation=none`；如启用 model-RTC，提供可验证 conditioning receipt；
12. 至少有 fake/local deterministic test，真实 observation/hardware/action send 另行授权。

## 12. Relation to dataset schema

`DATASET_SCHEMA.md` 是 offline sample columns、media storage 和 dataset provenance 的权威；本文件是 online deployment
observation、artifact、session、policy output 和 runtime compatibility 的权威。

- Dataset rows 是持久化样本，deploy step 是带 availability/causal metadata 的在线 observation。
- Dataset `action` 是 training target，deploy output 是 prediction，必须在 runtime 中归一化为 `RobotAction` 或 typed target。
- v2 dataset 的 `PolicyIOLayout` 必须嵌入 artifact 并核对 hash；v1 26D/38D 只有经过显式、可审计 adapter 才能进入 v2。
- `observation.arm.qpos` 可以由 replay/qpos-init 读取，但不进入默认 v2 model observation。
- 改变 state/action/token/pose/frame 语义时，必须同步更新 dataset layout、artifact、server metadata、adapter 和本文档；
  runtime availability、fault、session preflight 与 output normalization 只在部署契约层维护。

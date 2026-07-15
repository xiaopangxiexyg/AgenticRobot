# DexTeleop Deploy Schema

本文档定义模型部署时 policy 与 DexTeleop runtime 之间的接口契约。它不是
dataset 存储格式；离线数据格式见 `DATASET_SCHEMA.md`。部署 schema 关注每次
在线推理 step 的输入、输出，以及如何兼容单臂/双臂、不同末端、可选相机、可选触觉和历史窗口。

当前建议版本为 `dexteleop_deploy_v2`。未携带 v2.2 layout/artifact 信号的旧模型继续按
`dexteleop_deploy_v1` 解释；不得仅根据向量宽度把旧模型自动重解释为 v2。

版本关系：

| Deploy schema | 数据/模型合同 | 低维 state/action | 状态 |
| --- | --- | --- | --- |
| `dexteleop_deploy_v1` | legacy/unversioned | 固定 38D state、26D action、q6 hand slots | 完整兼容保留 |
| `dexteleop_deploy_v2` | `dexteleop_dataset_v2.2` + `PolicyIOLayout` | 双臂 pose14 固定前缀 + 分侧配置化末端；离散值为独立 int64 token | 新模型推荐 |

`dexteleop_deploy_v2` 不改变 LeRobot-v2.1-style 的数据目录组织；它改变的是模型和部署
低维张量的语义绑定。

## 设计边界

部署链路应分成三层：

- Runtime input: 从真机、仿真或 replay 中采集当前观测，构造 policy 输入。
- Policy output: 模型输出动作，动作语义由 action specs 解释。
- Runtime execution: 将动作解析为 `RobotAction`，再解析为 `TeleopCommand`，最后由具体 backend 执行。

部署 schema 只定义前两层的输入/输出契约。IK、限步、防碰撞、硬件下发和 emergency hold 属于 runtime execution，不应由模型输出绕过。

核心原则：

- v1 中 `active_sides` 是单臂/双臂兼容的主开关。v2 的模型合同固定保留左右 arm pose7；
  当前 v2 runtime session 因此要求 `active_sides=["left", "right"]`，末端则可以按侧缺省。
- v2 的唯一模型张量语义来源是 `PolicyIOLayout`；`arm_action_specs_by_side`、
  `hand_action_specs_by_side` 继续用于 v1/q6 执行兼容，不得覆盖 v2 layout。
- 缺失模态通过 `available`, `valid`, `mask`, `faults` 表达，不通过换 schema 表达。
- 模型输出 arm 7D pose 和配置化末端 target；低层 arm qpos 只作为独立审计、初始化、
  调试或特殊控制输入，不进入 v2 默认模型 state/action。
- 所有在线视觉/触觉/arm state 输入必须来自不可变历史，并满足
  `source_timestamp <= T0 AND available_timestamp <= T0`；不允许在推理时直接读取
  可能已经更新的 mutable backend state。这里的 `T0` 是本次动作/控制信息建立的切点，
  不是 observation 组装结束时间。

## v2.3 Absolute/Relative Arm Pose

v2.3 默认兼容 v2.2 的 tensor layout，但显式增加 pose representation 选择：

- state pose 始终是 desk absolute pose；部署 observation 中的
  `arm_states.<side>.actual_tcp_pose` 必须是当前 `T0` 之前最新的实际测量 TCP。
- policy action 的 `absolute`（默认）或 `relative` 由 session/CLI 明确声明，不能按
  向量宽度猜测。relative chunk 的 anchor 是 chunk 开始时的 measured TCP，chunk 内固定；
  `last_action` 如送入 relative policy，则使用当前 observation 的 measured TCP。
- relative policy 输出在 canonical safety、IK 和任何硬件 dispatch 之前由 runtime 解算成
  absolute pose。解算只接受 `actual_tcp_pose`，缺失、无效或不可转换时 fail-closed，
  不得回退到 `current/target/last_command`。
- v2.2 replay bridge 可以把存储的 absolute action 在每次 chunk prediction 时即时转换为
  relative，再走同一部署解算器；这用于不改写旧数据的兼容和端到端一致性测试。
- 若 observation 带 `pose_frame_contract`，runtime 先按 source raw/desk 声明将实际 TCP
  转到 desk；没有 metadata 的旧 RM65 runtime 暂按 identity compatibility profile，
  但新 session 应显式发布 source/topology/变换配置。

因此模型输出的 pose 表示是可替换的接口层选项，硬件执行层永远只接收 canonical desk
absolute target。

## Session Contract

每个部署 session 在启动时必须固定一个不可变能力声明。v2 使用两层来源：

1. `PolicyArtifactContract`：checkpoint/server sidecar，内嵌唯一 `PolicyIOLayout`，声明模型
   输入/输出键、float/token 维度、模型标识、layout hash，以及（视觉模型必须声明的）相机输入契约。
2. `ResolvedDeploySessionContract`：启动时临时生成，将 artifact/layout 与本次 runtime
   左右末端 backend 能力合并；它不是另一份持久化张量配置。

真实 backend 必须在 session 解析成功后创建。dataset、checkpoint/server、显式 layout 与
runtime 的 `policy_io_layout_hash` 必须完全一致。

### v2 Policy Artifact

默认字段结构：

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
  "camera_input": {
    "inputs": [
      {
        "source_key": "front",
        "model_input_key": "observation.images.base_0_rgb",
        "server_camera_slot": "base_0_rgb"
      }
    ],
    "required": true,
    "color_space": "rgb",
    "layout": "HWC",
    "dtype": "uint8"
  },
  "state_input_mode": "raw_state_v1",
  "state_augmentation": {},
  "inputs": {
    "state": {"key": "state", "dim": 15, "dtype": "float32"},
    "state_discrete": {"key": "state_discrete", "dim": 1, "dtype": "int64"}
  },
  "outputs": {
    "actions": {"key": "actions", "dim": 15, "dtype": "float32"},
    "actions_discrete": {"key": "actions_discrete", "dim": 1, "dtype": "int64"}
  }
}
```

默认 `state_input_mode="raw_state_v1"` 时，float state 维度只能由内嵌 layout 的 `state_dim`
派生，sidecar 不能独立修改。启用训练期 state dropout 的模型必须显式声明
`"masked_normalized_concat_v1"`：float 输入维度固定为 `2 * layout.state_dim`，前半是
归一化后的连续 state，后半是相同长度的 0/1 availability mask。`state_augmentation` 记录训练
配置以供审计；它不修改磁盘 dataset schema。即使某一 token 维度为 0，对应字段仍可保留为稳定的
零宽 int64 合同。

#### 相机输入契约

`camera_input` 是新视觉模型的唯一相机语义来源，`inputs` 的顺序也是模型输入顺序。每一项必须同时
绑定：

- `source_key`：DexTeleop 逻辑相机键，例如 `front`、`left_wrist` 或 `right_wrist`；`front`
  对应头部/主视角，不是要求物理设备的名称也叫 front。
- `model_input_key`：送到该模型服务器的精确图像字段名。
- `server_camera_slot`：服务端 metadata 必须声明的能力槽，用于启动期握手。

因此，左腕单视角模型可以把 `left_wrist` 映射到模型的 `observation.images.base_0_rgb`；部署不会再
把同一帧错误地标为 `front`，也不会把配置文件中未选中的头部或另一腕相机送入模型。`required=true`
时，启动必须配置完整的同名逻辑视角；`required=false` 允许无图像运行，但仍固定已提供图像的键和格式。
旧 artifact 没有 `camera_input` 时保持 v1/早期 v2 的默认 OpenPI front 路径，以便旧 checkpoint 直接部署；
迁移后的视觉 checkpoint 应补写此字段，而不要依赖自由 metadata。

### Resolved v2 Session

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
  "policy_io_layout_hash": "sha256:<canonical-layout-hash>",
  "end_effector_sides": ["left", "right"],
  "runtime_end_effectors": {
    "left": {"adapter": "dmtac_gripper_v1", "backend": "dmtac-gripper", "enabled": true},
    "right": {"adapter": "two_state_gripper_v1", "backend": "dmtac-gripper", "enabled": true}
  },
  "q6_bridge_compatible": true,
  "hardware_output": false,
  "warnings": []
}
```

`policy_contract_mode` 只能是：

- `legacy_v1`：旧 26D/38D 路径，不强制 layout hash。
- `v22_q6_bridge`：每个已配置 action channel 都显式声明
  `legacy_q6_slot_indices`，转换后继续走既有 26D safety/dispatch 链。
- `v22_native`：不投影到 q6，末端以 typed `EndEffectorTarget` 经
  `RobotAction -> TeleopCommand` 传递。类型载体已经实现；当前真机 `run_policy` 尚无完整
  native safety/backend adapter，因此该模式 fail-closed，不会直接下发硬件。

### Legacy v1 Session

以下原有字段继续用于 v1：

```json
{
  "schema_version": "dexteleop_deploy_v1",
  "robot": "dual_rm65",
  "active_sides": ["left", "right"],
  "enabled_effectors": ["left_arm", "left_hand", "right_arm", "right_hand"],
  "control_hz": 15.0,
  "policy_hz": 15.0,
  "arm_action_specs_by_side": {
    "left": {
      "side": "left",
      "controlled_frame": "tool_tcp",
      "reference_frame": "left_arm_base",
      "representation": "xyz_quat",
      "dof": 7
    }
  },
  "hand_action_specs_by_side": {
    "left": {
      "side": "left",
      "hand_model": "dmtac_gripper",
      "slot_count": 6,
      "active_dof": 1,
      "active_mask": [true, false, false, false, false, false],
      "joint_names": ["gripper_close"],
      "command_space": "normalized_0_0.9",
      "slot_map": {"gripper_close": 0}
    }
  },
  "camera_contract": {
    "selected_keys": ["front", "left_wrist", "right_wrist"],
    "required": false,
    "color_space": "rgb",
    "layout": "HWC",
    "dtype": "uint8"
  },
  "tactile_contract": {
    "enabled": true,
    "required_by_side": {"left": false, "right": false},
    "representation": "backend_native_vector_or_dict"
  },
  "history_contract": {
    "enabled": true,
    "observation_horizon": 1,
    "action_horizon": 1,
    "pad_policy": "repeat_latest_with_valid_mask"
  }
}
```

说明：

- `enabled_effectors` 使用 `<side>_arm` / `<side>_hand` 命名。只启用右臂时可为 `["right_arm"]` 或 `["right_arm", "right_hand"]`。
- session metadata 中的 `camera_contract.selected_keys` 是运行时诊断记录；对 v2.2 visual policy，实际
  选择、模型字段映射和 server slot 必须来自 artifact 的 `camera_input`，不能由该 metadata 覆盖。
- `required=false` 表示缺相机时仍可推理，但 input 中必须给出 availability/fault。
- `hand_action_specs_by_side` 对 DMTAC gripper、Inspire hand 和未来末端采用同一套 active slot 语义。

## Step Input

每次 policy 推理接收一个 `DeployStepInput`。容器中的图像、触觉、任务和 fault 字段可由
模型家族 adapter 改名；低维字段的键、dtype、维度和 layout hash 必须与 policy artifact
一致。当前 OpenPI adapter 默认使用 `state` / `state_discrete`。

推荐顶层结构：

```json
{
  "schema_version": "dexteleop_deploy_v2",
  "timestamp": 1783420000.123,
  "monotonic_time": 12345.678,
  "step_index": 42,
  "active_sides": ["left", "right"],
  "enabled_effectors": ["left_arm", "left_hand", "right_arm", "right_hand"],
  "task": {
    "task_id": 0,
    "instruction": "pick up the object"
  },
  "observation": {},
  "history": {},
  "metadata": {}
}
```

v2 实际送入模型的低维字段等价于：

```json
{
  "state": "<float32[state_dim]>",
  "state_discrete": "<int64[state_discrete_dim]>",
  "policy_io_layout_hash": "sha256:<canonical-layout-hash>"
}
```

当 artifact 的 `state_input_mode` 为 `masked_normalized_concat_v1` 时，`state` 长度是
`2 * layout.state_dim`；部署 adapter 对原始 base state 自动追加全 1 mask。模型/server 必须只
归一化前半 state，不能对 mask 再做均值方差归一化。server metadata 必须声明相同
`input_state_mode` 和 `input_state_dim`，否则启动握手失败。

离散 token 不得拼接或转换进 float state。部署 observation 不读取 init/fix 审计列，也不
读取 `task_hold`；这些字段不属于模型合同。

### Observation

`observation` 是当前帧的在线观测包。

在线观测的 `metadata`/`state_dict` 应保留每个模态的 causal selection 审计字段：
`snapshot_sequence`、`source_timestamp`、`available_timestamp`、`source_clock`、
`available_clock` 和 `causal_snapshot_contract`。手部 telemetry 与 tactile 分别记录
自己的 snapshot sequence/时间；如果任一 stream 在 T0 前没有合格快照，应报告
`available=false`/fault，而不是使用推理循环中刚读取的最新值。VIVE 校时后的
`source_timestamp` 在本机 wall-clock 域，`available_timestamp` 仍表示本机接收发布完成时刻。

```json
{
  "state": "<float32[state_dim]>",
  "state_discrete": "<int64[state_discrete_dim]>",
  "policy_io_layout_hash": "sha256:<canonical-layout-hash>",
  "state_dict": {
    "left_arm.pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    "right_arm.pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    "right_end_effector.state_token": [0]
  },
  "images": {
    "front": {
      "data": "<HWC uint8 RGB tensor>",
      "available": true,
      "timestamp": 1783420000.100,
      "age_sec": 0.023,
      "shape": [480, 640, 3],
      "color_space": "rgb"
    }
  },
  "tactile_by_side": {
    "left": {
      "available": true,
      "timestamp": 1783420000.110,
      "age_sec": 0.013,
      "data": "<backend-native tactile vector or dict>",
      "shape": [64]
    }
  },
  "last_action": "<float32[action_dim]>",
  "last_action_discrete": "<int64[action_discrete_dim]>",
  "last_action_valid": true,
  "faults": []
}
```

`state_dict` 是可选诊断视图，不是第二份模型张量权威。arm qpos 如需在线审计应放在独立
runtime metadata/diagnostic 字段；不得重新拼入 v2 `state`。

### v2 Typed Low-Dimensional Layout

连续向量始终先放固定双臂 pose：

```text
state(float32)  = [left_pose7, right_pose7, left_eef_state_cont?, right_eef_state_cont?]
action(float32) = [left_pose7, right_pose7, left_eef_target_cont?, right_eef_target_cont?]

state_discrete(int64)  = [left_eef_state_tokens?, right_eef_state_tokens?]
action_discrete(int64) = [left_eef_target_tokens?, right_eef_target_tokens?]
```

- `layout.state_dim = 14 + 左右连续末端 state 维度之和`。它始终表示 dataset 的 base state；
  masked artifact 的模型输入维度为其两倍，而不是改变该 layout。
- `action_dim = 14 + 左右连续末端 action 维度之和`。
- state/action 末端维度默认一致，但 layout 可以显式声明不同维度。
- 左右末端可为 `none`、`continuous` 或 `discrete`，也可以使用不同 adapter。
- token 顺序、类别 id、名称和可选 backend value 均由 layout 声明；未声明类别必须拒绝。
- `observation.arm.qpos` 是独立 12D float32 审计/回放初始化字段，顺序为
  `[left_qpos6, right_qpos6]`；它不属于上述默认 model state。

例如左侧无末端、右侧 Inspire 6D 时，float state/action 默认为 20D；左侧连续 1D、
右侧离散 1 token 时，float state/action 为 15D，token state/action 为 1D。

### Legacy v1 Vector Layout

```text
state dim = 38
state = [
  left_arm_qpos6, left_arm_pose7, left_hand6,
  right_arm_qpos6, right_arm_pose7, right_hand6
]

action dim = 26
action = [
  left_arm_pose7,
  right_arm_pose7,
  left_hand6,
  right_hand6
]
```

v1 的 `state_valid` 与 `state` 等长。未启用或不可用的 segment 填 `NaN`，对应
`state_valid=false`。模型输入适配器可把 NaN 替换为 0，但必须保留 mask。

### Images

`images` 是按 camera key 命名的字典。部署时可选择任意子集：

- `front`
- `left_wrist`
- `right_wrist`
- 其他在 camera registry 中声明的 key

每个 image entry 必须带：

- `available`: 当前帧是否可用。
- `timestamp`: 原始相机帧时间戳，可为 `null`。
- `age_sec`: `step.timestamp - image.timestamp`，不可用时可为 `null`。
- `data`: 图像 tensor。推荐 RGB, HWC, uint8；若模型需要 NCHW/float，应在 model adapter 中转换。

当相机缺失：

- 若 session `camera_contract.required=false`，保留 key，设 `available=false`，`data=null` 或填充占位 tensor，并在 `faults` 中记录。
- 若 `required=true`，runtime 应阻止推理或进入 hold，而不是让 policy 猜。

### Tactile

`tactile_by_side` 按 side 提供。不同末端可以有不同 representation：

- Inspire hand 可使用扁平 tactile vector。
- DMTAC gripper 可使用 per-sensor dict 或后处理后的 force/dist vector。
- 没有触觉的末端应保留 `available=false`，不要伪造有效触觉。

推荐字段：

```json
{
  "right": {
    "available": true,
    "backend": "dmtac_gripper",
    "representation": "distributed_force",
    "data": {"inner": [0.0], "outer": [0.0]},
    "timestamp": 1783420000.110,
    "age_sec": 0.013,
    "sensor_keys": ["inner", "outer"]
  }
}
```

训练/部署 reader 不应要求所有任务都有触觉。触觉缺失时应依赖 `available=false` 和 mask，而不是改变 observation schema。

### History

`history` 是可选字段，用于需要历史窗口的 policy。它不替代当前帧 observation。

推荐结构：

```json
{
  "observations": {
    "state": [[0.0]],
    "state_valid": [[true]],
    "timestamps": [1783420000.023]
  },
  "actions": {
    "action": [[0.0]],
    "timestamps": [1783420000.010],
    "valid": [true]
  },
  "images": {
    "front": {
      "data": ["<image t-1>", "<image t>"],
      "timestamps": [1783419999.900, 1783420000.100],
      "available": [true, true]
    }
  },
  "tactile_by_side": {
    "left": {
      "data": ["<tactile t-1>", "<tactile t>"],
      "timestamps": [1783419999.910, 1783420000.110],
      "available": [false, true]
    }
  }
}
```

History 规则：

- 每个 history stream 必须有 `timestamps` 和 `available`/`valid`。
- 不足 horizon 时可重复最新帧或 zero-pad，但必须用 mask 表示 padding。
- action history 使用已经实际下发或被 safety filter 接受后的 action，而不是未过滤 raw policy output。

### Metadata

`metadata` 必须包含本次 step 解释动作所需的 specs：

```json
{
  "arm_action_specs_by_side": {},
  "arm_controlled_frame_by_side": {},
  "arm_reference_frame_by_side": {},
  "hand_action_specs_by_side": {},
  "hand_active_mask_by_side": {},
  "modality_available": {
    "images": {"front": true},
    "tactile": {"left": false, "right": true}
  },
  "runtime": {
    "source": "real|sim|replay",
    "dry_run": false
  }
}
```

## Policy Output

v2 policy 输出两个独立矩阵，horizon 必须一致：

```json
{
  "actions": "<float32[horizon, action_dim]>",
  "actions_discrete": "<int64[horizon, action_discrete_dim]>",
  "policy_io_layout_hash": "sha256:<canonical-layout-hash>",
  "metadata": {}
}
```

也接受兼容键 `action` / `action.discrete`，但 policy artifact 中声明的正式 server 键是握手
权威。float/token horizon、维度、token 类别和 hash 任一不匹配都必须在进入 safety gate 前
拒绝。

v2 action 解析有两条显式路径：

```text
v22_q6_bridge:
typed output -> explicit legacy_q6_slot_indices projection -> NormalizedAction(26D)
             -> existing safety -> RobotAction -> TeleopCommand -> dispatcher

v22_native:
typed output -> RobotAction(typed EndEffectorTarget) -> TeleopCommand(typed target)
             -> native safety/backend adapter（当前 run_policy 未完成，fail-closed）
```

离散 target 在整个 typed 路径中保持 int64；不得为了调用旧 backend 而静默转换成连续值。
只有 layout 中离散 category 显式提供可用 backend value 且声明 q6 slot 时，bridge 才可执行。

### Legacy v1 Policy Output

v1 policy 输出 `DeployActionOutput`，runtime 会将它归一化为 `RobotAction`，再由
`RobotActionResolver` 转成 `TeleopCommand`。当前代码支持后文 short forms；完整
canonical JSON 是建议接口，policy adapter 仍需显式转换到现有
`NormalizedAction`/`RobotAction`。

当前 `normalize_policy_output()` 不直接接收下面完整 canonical JSON dict。
policy wrapper 若内部使用该结构，必须在返回给 runtime 前转换成后文
Accepted Short Forms、`NormalizedAction`，或直接走 `RobotAction` 兼容路径。

Legacy canonical output target contract:

```json
{
  "schema_version": "dexteleop_deploy_v1",
  "action_space": "bimanual_ee_pose_v1",
  "timestamp": 1783420000.123,
  "horizon": 1,
  "actions": [
    {
      "active_effectors": ["left_arm", "left_hand", "right_arm", "right_hand"],
      "arm_pose_by_side": {
        "left": [0.30, 0.10, 0.25, 0.0, 0.0, 0.0, 1.0],
        "right": [0.30, -0.10, 0.25, 0.0, 0.0, 0.0, 1.0]
      },
      "hand_qpos_by_side": {
        "left": [0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
        "right": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
      },
      "metadata": {
        "confidence": 0.92
      }
    }
  ],
  "metadata": {
    "policy_name": "my_policy",
    "raw_output_shape": [26]
  }
}
```

Field semantics:

- `actions` may contain a chunk. Runtime may execute only the first item or buffer the chunk.
- `arm_pose_by_side.<side>` is 7D `[x, y, z, qx, qy, qz, qw]` in `arm_action_specs_by_side.<side>.reference_frame`, controlling the frame declared by `controlled_frame`.
- `hand_qpos_by_side.<side>` uses the fixed slot width declared by `HandActionSpec.slot_count`. For current DMTAC gripper datasets this is 6 slots with only active slot 0.
- Inactive hand slots are padding. For q6 DMTAC data they should be finite padding such as `0.0`; for larger generic slots `NaN` padding is allowed only when specs explicitly support it.
- `arm_qpos_by_side` is allowed only for low-level debug or special controllers. Preferred model output is arm pose.

### Accepted v1 Short Forms

Legacy runtime also accepts these policy outputs and normalizes them：

1. Full action vector:

```python
np.ndarray(shape=(26,), dtype=np.float32)
```

2. Action chunk:

```python
np.ndarray(shape=(T, 26), dtype=np.float32)
```

3. Active-side compact arm pose:

```python
# active_sides=("right",)
np.ndarray(shape=(7,), dtype=np.float32)

# active_sides=("left", "right")
np.ndarray(shape=(14,), dtype=np.float32)
```

4. Mapping with explicit components:

```json
{
  "right_arm_pose": [0.30, -0.10, 0.25, 0.0, 0.0, 0.0, 1.0],
  "right_hand": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
  "metadata": {
    "policy": "example"
  }
}
```

5. Single-arm aliases when only one side is active:

```json
{
  "arm": [0.30, -0.10, 0.25, 0.0, 0.0, 0.0, 1.0],
  "hand": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
}
```

这些 short forms 仅属于 v1 或显式 q6 bridge。saved checkpoints 和 production
policy wrappers 应保留 canonical specs/metadata；在 canonical JSON parser 落地前，
wrapper 返回 runtime 前仍需转换成当前 accepted form。

## Runtime Normalization

v1 和 q6 bridge 使用以下序列：

```text
policy output
  -> NormalizedAction
  -> RobotAction
  -> TeleopCommand
  -> backend-specific send_command
```

`NormalizedAction.vector` 在此路径使用 legacy 26D action layout。
`RobotAction` is the semantic boundary. `TeleopCommand` is the dispatch boundary.

v2 native 不以 26D `NormalizedAction.vector` 为语义来源；它使用 layout-bound typed
`RobotAction.end_effector_targets_by_side`。在 native safety/dispatch adapter 完成前，不能
通过让 policy adapter 直接构造 backend command 来绕过这一边界。

Rules:

- v2 必须给出完整双臂 pose14；任一侧缺失或非有限都是整条 action fault，不允许只下发另一侧。
- v2 layout 中已配置的连续末端值必须有限；离散 token 必须是精确整数且属于声明 vocabulary。
- v1 中 missing inactive side output 可以忽略或保持；missing active effector 只有在 policy
  显式 hold 或 safety fallback 时才可退化为 hold。
- v1 policy output 中 active arm pose/qpos 或 active hand slot 非有限时，不得向其余有效
  effector 部分下发。
- Runtime safety filters 可以裁剪 workspace、限制 pose/qpos 步长、拒绝动作或 hold。q6
  bridge 继续使用现有 hand q6 `[0, 1]` clip；这些修改必须进入 action metadata。

## Single-Arm Compatibility

### Legacy v1

For `active_sides=["right"]`:

- Input may contain only right-side valid state, images and tactile.
- `state` and `last_action` may still use full 38D/26D layouts with left side as invalid/NaN.
- Target canonical output may contain only `right_arm` / `right_hand`; current direct runtime returns should use compact `(7,)` or mapping aliases `arm` and `hand`, unless a policy adapter converts canonical JSON first.
- Runtime must not require left-side qpos, left images or left tactile.

For `active_sides=["left", "right"]`:

- Outputs should specify side names or use full 26D layout.
- Compact `(14,)` arm pose means ordered sides `left`, then `right`.
- Hand compact output follows ordered sides `left`, then `right`.

### v2 Fixed Arm Core

当前 v2 `PolicyIOLayout` 固定包含 `left_pose7 + right_pose7`，因此 model-facing session
要求左右 arm 都存在。它不使用“删除未激活 arm slot”的方式表达单臂任务。

- 某侧可以没有末端：例如左侧 `none`、右侧 6D hand。
- 某侧 arm 若在真实 runtime 中不可用，v2 session 应在创建 backend 前失败，而不是用 NaN、
  默认姿态或上一条命令伪造模型输入。
- 若未来需要真正的单臂 v2 模型，应定义新的显式 layout/schema 版本；不得复用当前 v2.2
  hash 后改变双臂 pose14 核心。

## End-Effector Compatibility

### v2 Per-Side End-Effector Spec

每侧 `EndEffectorSpec` 独立声明：

- `side`、`adapter`、`state_source_field`；
- `target` 与 `state` 各自的 `value_type`、`dim`、名称、单位和 normalization metadata；
- 离散 channel 的 token vocabulary 与稳定 int64 category id；
- 仅用于旧执行路径的可选 `legacy_q6_slot_indices`。

左右配置可以不同，例如：

```json
{
  "left": {
    "side": "left",
    "adapter": "none_v1",
    "target": {"value_type": "none", "dim": 0},
    "state": {"value_type": "none", "dim": 0}
  },
  "right": {
    "side": "right",
    "adapter": "inspire_tac_v1",
    "target": {"value_type": "continuous", "dim": 6},
    "state": {"value_type": "continuous", "dim": 6}
  }
}
```

真实输出时，layout 中存在的末端必须有同侧 runtime capability 且 adapter 一致；runtime
额外启用 layout 中不存在的末端同样是错误。dry-run 可以缺 backend，但必须给出 warning。

### Legacy v1 Hand Slot Spec

v1 end-effector 共用 hand action spec schema：

```json
{
  "side": "right",
  "hand_model": "dmtac_gripper",
  "slot_count": 6,
  "active_dof": 1,
  "active_mask": [true, false, false, false, false, false],
  "joint_names": ["gripper_close"],
  "command_space": "normalized_0_0.9",
  "slot_map": {"gripper_close": 0},
  "invalid_slot_value": "zero_padding"
}
```

Compatibility rules:

- Policy should output full slot vector by side unless the wrapper explicitly supports active-dof output.
- Runtime converts full slot vector to backend command by selecting active slots.
- DMTAC gripper backend receives 1D active value after resolver mapping, even if policy emitted a 6-slot vector.
- v1 future hands can use different `slot_count`, `active_mask`, `joint_names` and
  `command_space`；v2 新末端必须通过新的 canonical layout/hash 绑定。
- Tactile availability is independent from hand command availability.

## Modality Compatibility

### No Camera

If camera input is disabled:

- `images={}` or all configured entries have `available=false`.
- `metadata.modality_available.images` records false values.
- Policy must be trained/adapted for no-image operation, or runtime should use hold/fallback.

### Camera Selection

If only a subset of cameras is selected:

- Keep original DexTeleop logical keys (`front`/`left_wrist`/`right_wrist`).
- Use artifact `camera_input.inputs` to record the ordered logical key → model field → server slot mapping.
- `--policy-artifact-json` makes `run_policy` open only those logical cameras. A single-camera serial source
  is automatically assigned to the artifact's sole logical key; an explicitly conflicting `--camera-key` fails
  before opening the device.
- An artifact that requires two or more views rejects `--camera-mode single`; a required visual artifact rejects
  `--camera-mode none`. Legacy artifacts without `camera_input` keep the old permissive behavior.

### No Tactile

If tactile is disabled or unavailable:

- `tactile_by_side.<side>.available=false`.
- `tactile_valid_by_side.<side>=false` in current implementation.
- Do not fill random or stale tactile as valid data.

### Historical Context

If the model needs history:

- Runtime should provide `history` with masks and timestamps.
- The current frame must still be present outside history.
- If history is shorter than expected after startup/reset, masks must mark padded entries invalid.

## Faults And Fallback

`faults` is part of the input because policy and runtime may make different decisions based on degraded sensors.

Common observation faults:

- `<side>_arm_qpos_unavailable`
- `<side>_arm_pose_unavailable`
- `<side>_hand_state_unavailable`
- `camera_frame_unavailable:<camera_keys>`
- `<side>_hand_telemetry:<reason>`

Recommended behavior:

- Hard hardware faults should stop dispatch or enter hold before policy output is sent.
- Soft modality faults may still run policy if the session contract marks that modality optional.
- Every fallback action should be logged with `fallback_reason` and whether it was dispatched.

## Minimal Examples

### v2: Left No End Effector, Right Discrete Gripper

Layout-derived model fields：

```json
{
  "state": "<float32[14] = left_pose7 + right_pose7>",
  "state_discrete": "<int64[1] = right_gripper_state_token>",
  "actions": "<float32[T,14] = left_pose7 + right_pose7>",
  "actions_discrete": "<int64[T,1] = right_gripper_target_token>",
  "policy_io_layout_hash": "sha256:<matching-layout-hash>"
}
```

左侧不会生成 hand command。右侧 token 只有在 category 和 adapter 校验通过后，才进入
显式 q6 bridge 或 native typed target。

### v2: Left 1D Gripper, Right 6D Inspire

默认 state/action 末端维度一致时：

```text
state/action float dim = 14 + 1 + 6 = 21
state/action token dim = 0
```

采集、训练、checkpoint/server、replay 与部署必须使用同一 layout hash。仅 backend 配置
一致而 hash 不一致仍必须拒绝。

### Legacy v1: Right Arm Only, No Camera, No Tactile

Input:

```json
{
  "schema_version": "dexteleop_deploy_v1",
  "step_index": 1,
  "active_sides": ["right"],
  "enabled_effectors": ["right_arm"],
  "observation": {
    "state": "<38D float32, right valid, left NaN>",
    "state_valid": "<38D bool>",
    "images": {},
    "tactile_by_side": {},
    "last_action": "<26D float32>"
  },
  "metadata": {
    "arm_action_specs_by_side": {
      "right": {"controlled_frame": "tool_tcp", "representation": "xyz_quat", "dof": 7}
    }
  }
}
```

Output:

```json
{
  "arm": [0.30, -0.10, 0.25, 0.0, 0.0, 0.0, 1.0]
}
```

Runtime normalizes this to `right_arm` cartesian pose.

### Legacy v1: Dual Arm With DMTAC Grippers

Output:

```json
{
  "left_arm_pose": [0.30, 0.10, 0.25, 0.0, 0.0, 0.0, 1.0],
  "right_arm_pose": [0.30, -0.10, 0.25, 0.0, 0.0, 0.0, 1.0],
  "left_hand": [0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
  "right_hand": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
}
```

Runtime uses `hand_action_specs_by_side` to send only active gripper values to each backend.

## Relation To Dataset Schema

`DATASET_SCHEMA.md` remains the source of truth for offline sample columns and media storage.
部署与数据集共享同一模型语义，但容器不同：

- Dataset rows are persisted samples.
- Deploy step inputs are online observations with availability masks.
- Dataset `action` is the training target.
- Deploy output is the model prediction that must be normalized to `RobotAction`.
- v2 dataset 的 `PolicyIOLayout` 是张量语义权威；deploy artifact 内嵌同一 layout 并验证 hash。
- `observation.arm.qpos` 可以被 replay/qpos-init 读取，但 dataset interface、训练 batch 和默认
  deployment model observation 不读取它。
- v1 的 26D/38D 仍保持原定义；旧数据/模型只有经过显式、可审计的 adapter 或复制式转换
  才能进入 v2，不允许原地修改或按维度猜测。

任何改变 action/state/token 语义的修改必须同时更新 dataset layout、policy artifact、server
metadata、replay 和本文档；在线 fault、模态 availability、session preflight 与输出归一化只在
`DEPLOY_SCHEMA.md` 描述。

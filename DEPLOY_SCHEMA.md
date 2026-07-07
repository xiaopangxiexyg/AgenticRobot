# DexTeleop Deploy Schema

本文档定义模型部署时 policy 与 DexTeleop runtime 之间的接口契约。它不是
dataset 存储格式；离线数据格式见 `DATASET_SCHEMA.md`。部署 schema 关注每次
在线推理 step 的输入、输出，以及如何兼容单臂/双臂、不同末端、可选相机、可选触觉和历史窗口。

当前建议版本为 `dexteleop_deploy_v1`。

## 设计边界

部署链路应分成三层：

- Runtime input: 从真机、仿真或 replay 中采集当前观测，构造 policy 输入。
- Policy output: 模型输出动作，动作语义由 action specs 解释。
- Runtime execution: 将动作解析为 `RobotAction`，再解析为 `TeleopCommand`，最后由具体 backend 执行。

部署 schema 只定义前两层的输入/输出契约。IK、限步、防碰撞、硬件下发和 emergency hold 属于 runtime execution，不应由模型输出绕过。

核心原则：

- `active_sides` 是单臂/双臂兼容的主开关。未激活 side 的输入可以缺失，输出会被忽略或保持。
- `arm_action_specs_by_side` 和 `hand_action_specs_by_side` 是动作维度和语义的权威来源，不要按 DMTAC/Inspire 等设备名硬编码输出维度。
- 缺失模态通过 `available`, `valid`, `mask`, `faults` 表达，不通过换 schema 表达。
- 模型推荐输出 arm 7D pose 和 hand slot target；低层 arm qpos 只作为调试或特殊控制输出。
- 所有在线视觉/触觉输入必须是 `latest <= action_timestamp` 语义，不允许读取未来帧。

## Session Contract

每个部署 session 在启动时应固定一个能力声明。该声明可以来自 CLI、robot config、dataset metadata、checkpoint metadata 或它们的合并结果。

推荐字段：

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
- `camera_contract.selected_keys` 是本次部署实际送入模型的相机集合。相机可少于训练集相机，但 policy 必须能识别这个选择。
- `required=false` 表示缺相机时仍可推理，但 input 中必须给出 availability/fault。
- `hand_action_specs_by_side` 对 DMTAC gripper、Inspire hand 和未来末端采用同一套 active slot 语义。

## Step Input

每次 policy 推理接收一个 `DeployStepInput`。当前代码中的
`DeploymentObservation.model_input()` 已接近这个结构；后续可以逐步收敛到本节字段。

推荐顶层结构：

```json
{
  "schema_version": "dexteleop_deploy_v1",
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

### Observation

`observation` 是当前帧的在线观测包。

```json
{
  "state": [0.0],
  "state_valid": [true],
  "state_layout": "dexteleop_state_v1",
  "state_dict": {
    "left_arm.qpos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "left_arm.pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    "left_hand": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
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
  "last_action": [0.0],
  "last_action_valid": true,
  "faults": []
}
```

Current vector layouts:

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

`state_valid` 与 `state` 等长。未启用或不可用的 segment 填 `NaN`，对应
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

推荐 policy 输出 `DeployActionOutput`，runtime 会将它归一化为 `RobotAction`，
再由 `RobotActionResolver` 转成 `TeleopCommand`。当前代码已经支持后文的
short forms；完整 canonical JSON 是建议收敛的目标接口，落地时应在
policy adapter 中显式转换到现有 `NormalizedAction`/`RobotAction`。

当前 `normalize_policy_output()` 不直接接收下面完整 canonical JSON dict。
policy wrapper 若内部使用该结构，必须在返回给 runtime 前转换成后文
Accepted Short Forms、`NormalizedAction`，或直接走 `RobotAction` 兼容路径。

Canonical output target contract:

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

### Accepted Short Forms

Current runtime also accepts these policy outputs and normalizes them:

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

Short forms 是当前 runtime 可直接接收的返回格式。saved checkpoints 和 production
policy wrappers 应保留 canonical specs/metadata；在 canonical JSON parser 落地前，
wrapper 返回 runtime 前仍需转换成当前 accepted form。

## Runtime Normalization

Runtime must normalize every accepted output to this internal sequence:

```text
policy output
  -> NormalizedAction
  -> RobotAction
  -> TeleopCommand
  -> backend-specific send_command
```

`NormalizedAction.vector` uses the 26D action layout from `DATASET_SCHEMA.md`.
`RobotAction` is the semantic boundary. `TeleopCommand` is the dispatch boundary.

Rules:

- Missing inactive side output is acceptable; runtime keeps or ignores inactive side values.
- Missing active arm output should default to hold only when policy explicitly requests hold or safety fallback is active.
- Missing active hand output may hold previous hand target if `hands_enabled=true` and the hand backend supports hold; otherwise it is a policy fault.
- Policy output with non-finite active arm pose, active qpos, or active hand slots is invalid; runtime must not dispatch a partial command for the remaining valid effectors.
- Runtime safety filters may clip workspace, rate-limit pose/qpos values, reject unsafe actions, or hold last safe command. Current deploy clips hand q6 to `[0, 1]`; hand rate-limit is reserved but not enabled yet. These modifications must be recorded in action metadata.

## Single-Arm Compatibility

For `active_sides=["right"]`:

- Input may contain only right-side valid state, images and tactile.
- `state` and `last_action` may still use full 38D/26D layouts with left side as invalid/NaN.
- Target canonical output may contain only `right_arm` / `right_hand`; current direct runtime returns should use compact `(7,)` or mapping aliases `arm` and `hand`, unless a policy adapter converts canonical JSON first.
- Runtime must not require left-side qpos, left images or left tactile.

For `active_sides=["left", "right"]`:

- Outputs should specify side names or use full 26D layout.
- Compact `(14,)` arm pose means ordered sides `left`, then `right`.
- Hand compact output follows ordered sides `left`, then `right`.

## End-Effector Compatibility

All end-effectors share the hand action spec schema:

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
- Future hands can use different `slot_count`, `active_mask`, `joint_names` and `command_space` without changing deploy top-level schema.
- Tactile availability is independent from hand command availability.

## Modality Compatibility

### No Camera

If camera input is disabled:

- `images={}` or all configured entries have `available=false`.
- `metadata.modality_available.images` records false values.
- Policy must be trained/adapted for no-image operation, or runtime should use hold/fallback.

### Camera Selection

If only a subset of cameras is selected:

- Keep original camera keys.
- Do not rename `left_wrist` to positional indices unless the model adapter records the mapping.
- Checkpoint metadata should store expected camera keys and ordering.

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

### Right Arm Only, No Camera, No Tactile

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

### Dual Arm With DMTAC Grippers

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
This deploy schema reuses the same action/state vector layouts and action specs, but changes the container:

- Dataset rows are persisted samples.
- Deploy step inputs are online observations with availability masks.
- Dataset `action` is the training target.
- Deploy output is the model prediction that must be normalized to `RobotAction`.

Any future dataset or model change should update both schemas when it changes action semantics, but only `DEPLOY_SCHEMA.md` needs to describe online fault handling, modality availability and policy output normalization.

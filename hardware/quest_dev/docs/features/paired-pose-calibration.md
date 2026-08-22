# Quest/Robot Paired-Pose Calibration R0

## 目标与坐标契约

把 Quest 手柄刚性固定在机械臂 TCP 后，每个静止采样窗口形成一对：

- first observation：`A_i = T_robotReference_from_tcp(i)`；
- second observation：`B_i = T_questReference_from_controllerGrip(i)`。

两个未知且在数据集内固定的变换为：

- `X = T_questReference_from_robotReference`；
- `Y = T_tcp_from_controllerGrip`。

观测满足：

```text
B_i = X A_i Y
```

R0 通过相对运动构造 `A_ij Y = Y B_ij`，以 NumPy SVD 解旋转、最小二乘解平移，
再由所有样本平均求 `X`。不估计 scale，也不假设手柄原点与 TCP 原点重合。

## 数据集契约

顶层 schema 为 `quest_dev.visualization.paired_pose_dataset.v1`：

```json
{
  "schema": "quest_dev.visualization.paired_pose_dataset.v1",
  "frames": {
    "first_reference": "robot_world",
    "first_moving": "tcp",
    "second_reference": "quest_local_floor",
    "second_moving": "controller_grip"
  },
  "pairing_method": "operator_static_window",
  "source_identity": {
    "quest_session_id": "xr-...",
    "reference_space_type": "local-floor",
    "reference_space_generation": 0,
    "controller_source_session_id": "input-1",
    "robot_source_instance_id": "robot-feedback-epoch-owner",
    "quest_reference_epoch_id": "xr-...:local-floor:0",
    "robot_source_epoch_id": "robot-epoch-...",
    "tool_profile_hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  },
  "samples": [
    {
      "sample_id": "pose-000",
      "first_observation": {
        "observation_id": "robot-000",
        "source_instance_id": "robot-feedback-epoch-owner",
        "source_epoch_id": "robot-epoch-...",
        "sequence": 10,
        "observed_at": 123.4,
        "clock_domain_id": "robot-monotonic",
        "stationary_evidence_id": "stable-window-000",
        "matrix_reference_from_moving": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
      },
      "second_observation": {
        "observation_id": "quest-000",
        "source_instance_id": "input-1",
        "source_epoch_id": "xr-...:local-floor:0",
        "sequence": 20,
        "observed_at": 567.8,
        "clock_domain_id": "quest-xr-frame",
        "stationary_evidence_id": "stable-window-000",
        "matrix_reference_from_moving": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
      }
    }
  ]
}
```

矩阵在 JSON 中使用常规 nested 4×4 row/column 表示。controller wire 的 flat
column-major 数组只能通过 `controller_wire_matrix_to_transform()` 转换；该边界会按
`reshape(order="F")` 还原并重新验证 SE(3)，避免隐式转置或求逆。

所有 source/epoch/tool identities 必须在数据集内一致；tool profile 使用
`sha256:<64 lowercase hex>`；sample/observation/sequence 必须唯一。解析器拒绝未知字段，
语义 SHA-256 在构造时生成，并在 fit 入口对当前规范化内容重新解析、重算和核验。

真实采集必须使用 `operator_static_window`：同一 pair 的两侧必须引用同一个、且全数据集
唯一的 `stationary_evidence_id`。时间戳明确保留各自 clock domain；R0 不会假装两个时钟
可直接相减。`synthetic_replay` 只用于离线测试，不要求静止证据。

## 拟合与输出

```bash
python -m hardware.quest_dev.tools.fit_paired_pose_calibration paired_poses.json
```

输出 schema 为 `quest_dev.visualization.paired_pose_fit.v1`，包含：

- 输入数据集与拟合输出各自的 canonical SHA-256；
- frame endpoints 与 tool-profile hash；
- `T_questReference_from_robotReference` 和 `T_tcp_from_controllerGrip`，各自同时提供
  nested 4×4 与 WebGL column-major；
- 每样本与 RMS/max 的平移、旋转残差；
- 有效相对旋转数、最大转角、最大轴分离角、rotation constraint/nullspace ratios、
  nullspace gap，以及 translation singular-value ratio。

少于 3 个姿态会拒绝。数值可解门槛还要求：至少两个 `>=5°` 的有效相对旋转、有效轴
最大分离 `>=15°`、清晰的 rotation 唯一零空间，以及不病态的 translation system。
这些是保守的数值可解性与数据一致性 admission，用来阻止求解器输出任意大而残差看似
很小的结果；它们仍不是最终标定质量验收。特别是 nullspace-ratio 门槛需要在真实 Quest
噪声数据上补充预期可通过的 fixture 后再复核，当前选择保持 fail-closed。实际采集建议
至少 15 个覆盖工作空间、绕两个以上非平行轴充分旋转的静止姿态。

## Fit 不是 Acceptance

输出固定包含：

```json
{"accepted": null, "acceptance_note": "fit only; a reviewed acceptance profile is required"}
```

R0 不定义残差阈值，不写 calibration 文件，不安装 frame graph，不启动机器人，也不让
visualization 使用该结果。后续验收至少还需绑定数据集 hash、fit hash、tool profile、
robot epoch、Quest session/reference generation，审查残差/可观测性和独立验证姿态；
reference-space reset、重新安装手柄或 tool profile 改变都必须使旧结果失效。

## 尚未实现

- 机器人权威只读 pose 与 Quest latest sample 的同步采集器；
- 静止窗口检测/证据生成；
- mount 刚性检查与独立验证姿态；
- reviewed acceptance profile 和持久化 receipt；
- 接入 future visualization 的只读 frame/trajectory 投影。

因此当前工具只能处理人工构造或后续采集器导出的数据集，不能作为真实标定完成证明。

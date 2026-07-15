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

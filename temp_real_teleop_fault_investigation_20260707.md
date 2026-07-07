# RealTeleop FAULT 卡住排查记录

日期：2026-07-07

## 问题描述

最近一次真机遥操过程中，主循环仍在打印状态和姿态信息，但机械臂不再响应，表现为“卡住”。用户提供的关键日志片段时间为 `14:55:46`：

```text
[RealTeleop] step=2252 hz=14.9 state=FAULT
left_hw_init=0:0 right_hw_init=0:0
left_cmd=imp=0,loop=not_started,send=hold
right_cmd=imp=0,loop=not_started,send=hold
faults=[]
pose_quality=unknown score=- available=0 reason=no_command
hand_q6=['left=ok', 'right=ok']
gripper_control=on
gripper=['left=connected/...', 'right=connected/...']
```

## 结论

这次不是主循环死锁，也不是 VIVE/Q6/夹爪输入链路断开导致的卡死。系统已经进入 `FAULT` 状态；进入 `FAULT` 后，`TeleopSession.run_step()` 会直接返回，不再生成或下发 arm command，所以后续持续显示：

- `pose_quality=unknown ... reason=no_command`
- 双臂 `send=hold`
- 双臂 `imp=0, loop=not_started`

更可能的触发路径是姿态阻抗控制 guard 检测到 RM65 impedance 后端异常或未启用，然后主动 `hold_motion()` 并将 session 置为 `FAULT`。

## 关键证据

### 1. FAULT 后不会再生成 command

代码位置：[teleop_runtime/session.py](/home/zc/HandMotion/Developing/DexTeleop/teleop_runtime/session.py:351)

```python
if self.state_machine.state in {
    TeleopRunState.FAULT,
}:
    return SessionStepResult(...)
```

因此日志里的 `reason=no_command` 是 `FAULT` 的结果，不是最初原因。

### 2. 姿态阻抗 guard 会触发 FAULT

代码位置：[teleop_runtime/debug/run_vive_rm65_real_teleop.py](/home/zc/HandMotion/Developing/DexTeleop/teleop_runtime/debug/run_vive_rm65_real_teleop.py:6760)

`_impedance_force_control_issue()` 会检查：

- `fault`
- `impedance_enabled`
- `impedance_loop_state == "stopped"`

若发现问题，`_guard_impedance_force_control()` 会：

1. 记录 `ForceControl` 日志；
2. 调用 `rig.hold_motion(...)`；
3. 调用 `components.session.fault(reason)`。

对应代码位置：[teleop_runtime/debug/run_vive_rm65_real_teleop.py](/home/zc/HandMotion/Developing/DexTeleop/teleop_runtime/debug/run_vive_rm65_real_teleop.py:6821)

### 3. hold_motion 正好会产生日志中的后端状态

代码位置：[teleop_runtime/backends/arms/rm65_impedance.py](/home/zc/HandMotion/Developing/DexTeleop/teleop_runtime/backends/arms/rm65_impedance.py:461)

`RM65ImpedanceArmBackend.hold_motion()` 会：

- `_enabled = False`
- `_last_send_result_text = "hold"`
- 停止 impedance loop
- loop thread 结束后，`get_state()` 显示 `loop_state = "not_started"`

这与日志中的 `imp=0,loop=not_started,send=hold` 一致。

## 已排除项

- **不是 Q6 输入失效**：日志显示 `hand_q6=['left=ok', 'right=ok']`。
- **不是夹爪/触觉主链路直接导致 arm 卡住**：日志中 gripper 仍 `connected/q6`，并且仍有命令/测量值。
- **不是常规 safety fault**：`faults=[]`，没有 `missing_channel`、`stale_channel`、`invalid_pose`、`translation_jump`、`rotation_jump`、`low_control_rate` 等 `FaultEvent`。
- **不是已拉回的 14:44 debug timing 那次运行**：`debug_timing_20260707_144430_012.jsonl` 共 1213 帧，全部 `RUNNING`，双臂均为 `imp=True, loop=alive, send=ok`。

## 仍缺失的直接证据

缺少首次进入 `FAULT` 前的原始控制台日志。理想情况下应能看到类似：

```text
[ForceControl] main_loop: <side>_arm pose impedance is disabled before command dispatch: ...
[Debug Info] FAULT triggered due to: main_loop: ...
```

当前可见日志只保留了 `FAULT` 后的状态，因此只能从状态组合和代码路径反推触发来源。

## 相关远端日志检查

- `/home/user/zc/DexTeleop/teleop_records/debug_timing/debug_timing_20260707_144430_012.jsonl`
  - 已拉到本地 `/tmp/dexteleop_debug_timing_144430/`
  - 结论：该文件对应运行没有 FAULT。
- `/tmp/xxxtmp.log`
  - 已拉到本地 `/tmp/dexteleop_xxxtmp_20260707/`
  - 内容是输入法/DBus 噪声，不包含 `RealTeleop` 或 `ForceControl`。
- `/home/user/zc/DexTeleop/logs/DmTac_M*.log`
  - 14:55:56 左右有 DMTac 设备显式断开记录；
  - 时间晚于 14:55:46 的 FAULT 状态，更像进程退出或收尾结果，不像 arm FAULT 根因。

## 建议修复方向

1. 在 `TeleopSession.fault(reason)` 或 RealTeleop 主循环状态输出中持久化 `last_fault_reason`。
2. 在 Web UI / `RealTeleop` 状态行里显示最近一次 `ForceControl` guard reason。
3. 在 debug timing frame 中加入 session 状态机的 fault reason，而不仅仅是 `fault_codes`。
4. 下次复现时优先保存包含首次 `FAULT triggered due to` 的控制台 stdout/stderr。

## 当前判断

最可能原因：

```text
RM65 pose impedance backend 在运行中变为 disabled/stopped/fault
=> _guard_impedance_force_control() 触发安全 hold
=> session 进入 FAULT
=> 后续 arm command 不再生成
=> RealTeleop 持续打印 state=FAULT, reason=no_command, send=hold
```

## 2026-07-07 本轮补充排查

### 新增日志核对

- 远端 `/home/user/zc/DexTeleop/teleop_records/debug_timing/debug_timing_20260707_144430_012.jsonl` 实际大小约 166MB，本地 `/tmp/dexteleop_debug_timing_144430/` 解析为 `1213` 个 frame，全部 `RUNNING`，双臂 `impedance_loop_state=alive`、`send_result_text=ok`，没有记录到 FAULT。
- 远端更新的 `/home/user/zc/DexTeleop/teleop_records/debug_timing/debug_timing_20260707_151356_569.jsonl` 已拉到 `/tmp/dexteleop_debug_timing_151356/`，解析为 `118` 个采样 frame，全部 `RUNNING`，双臂阻抗线程均 alive/send ok，也没有记录到 14:55 的 FAULT。
- 这说明 14:55 控制台片段里的 FAULT 原因仍缺少首帧直接证据；现有 debug timing 只能证明 14:44/15:13 两次采样没有 session FAULT。

### 阻抗相关确认的问题

15:13 最新结果里有明显的位置轴异常证据：

- 多数大偏差时 `active_wrench` 线性分量已经为 0 或很小，`applied_offset` 也接近 0。
- 但 `command_to_measured_linear_m` 仍存在 2-7cm 级别偏差，例如左臂 y 轴末尾约 `-36.6mm`，右臂 y 轴最高约 `54.7mm`。
- 这不是简单的“力控 offset 还在推”，而是命令/参考 TCP 与实测 TCP 没有对齐；体感上会像卡在受阻位置、小幅抖动，或在释放外力后仍被命令轨迹拉回。

进一步检查发现 antiwindup 逻辑有一个真实问题：

```text
旧逻辑用 6D active_wrench 的总范数判断是否允许 antiwindup。
只要某个角向力矩分量非零，即使线性力为 0，也可能把 IK/关节限速后的线性 realized offset 反写进 admittance offset。
```

这会把并非由线性外力产生的位置偏差写回阻抗状态，造成位置轴的虚假回正/卡住/抵抗感。

### 已修改

- `hardware/rm65/impedance.py`
  - 将 admittance antiwindup 改为逐轴同步。
  - 只有对应轴同时有 active wrench 和 active target/requested offset 时，才允许该轴被 IK realized offset 反写。
  - 角向力矩不再允许反写 x/y/z 线性 offset；线性力也不再无条件反写角向 offset。
  - debug record 新增 `admittance_offset_antiwindup_active_axes`。
- `teleop_runtime/debug/run_vive_rm65_real_teleop.py`
  - debug timing frame 新增 `session_fault_reason`。
  - 状态行新增 `fault_reason=...`，下次如果 guard 触发 FAULT，不会只剩 `send=hold/no_command` 这种后果状态。
  - compact debug 中保留 `admittance_offset_antiwindup_active_axes`。
- `teleop_runtime/session.py`
  - `TeleopSession.fault(reason)` 持久保存 `last_fault_reason`。

### 验证

- 本地：`PYTHONPATH=. uv run pytest tests/test_rm65_impedance.py -q`，`47 passed`。
- 本地：`PYTHONPATH=. uv run pytest tests/test_rm65_impedance.py tests/test_real_teleop_gui_collection_visualization.py -q`，`112 passed`。
- 本地：`PYTHONPATH=. uv run pytest tests/test_arm_endpoint_action.py tests/test_rm65_impedance.py tests/test_real_teleop_gui_collection_visualization.py -q`，`136 passed`。
- 远端 sync check：`hardware/rm65/impedance.py`、`teleop_runtime/debug/run_vive_rm65_real_teleop.py`、`teleop_runtime/session.py` 等已同步一致。
- 远端 pytest 被 wrapper 安全规则拦截为 control-risk，未绕过执行。

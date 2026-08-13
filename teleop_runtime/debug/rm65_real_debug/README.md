# RM65 Real Debug

这个目录提供一个**只读真机状态**的系统检查入口，用于在不上 teleop、不发送任何运动指令的前提下，把真实 RM65 的当前关节状态，以及可选 InspireTac 手部关节状态，映射回 SAPIEN。

当前也支持 `--arm-mode virtual`，直接渲染默认双臂姿态，不连接真机。

它的目标是：

1. 连接左/右 RM65
2. 可选连接左/右 InspireTac
3. 只调用 `Get_Current_Arm_State`、`Get_Current_Tool_Frame` 和手部只读遥测
4. 用当前 `desk -> base` 标定和 URDF 正向运动学计算真实 TCP
5. 在 SAPIEN 中显示双臂、双手、desk 坐标系、arm base 坐标系和当前 TCP 坐标系

这一步适合在接入真机前确认：

- 左右臂 base 相对关系是否正确
- 当前 `hand_base` 的 TCP 是否落在合理位置
- 当前 `T_D_B_left/right` 和真实安装关系是否一致

## 安全说明

这个工具是只读的。

- 它不会调用 `Movej_CANFD`
- 不会发送任何动态轨迹
- 不会改动真机姿态
- 显式禁用共享 backend 启动期的 `Algo_Kin_Set_Singularity_Thresholds` 写入

机械臂侧允许的真机读操作是：

- `Get_Current_Arm_State`
- `Get_Current_Tool_Frame`
- `InspireTac.read_joint_pos()`

状态行会同时显示同一次 `Get_Current_Arm_State` 已返回的 `arm_err` 和
`sys_err`；这不会增加 SDK 调用，也不等价于使能、急停回路或运动就绪验收。

## M5.8-b0 stop-only preflight

`run_rm65_stop_only_preflight` 不是只读工具。它只用于已经单独授权的
M5.8-b0 窄验收，并向每侧各发送一次 `Arm_Slow_Stop`。入口具有以下代码级限制：

- 必须同时提供 execute、arms-stationary 和 ESTOP-ready 三个确认参数；
- 启动前读取两次 fresh qpos，检查 error code、URDF 限位和静止漂移；
- motion adapter 为 deny-all，不提供 `Movej_CANFD`/force-position permit；
- lifecycle 和其他 safety seam 均不能从 stop-only provider 获得 permit；
- 每侧在本次 authorization 下最多尝试一次，失败后不会自动重试；
- 不允许从 `Arm_Slow_Stop` 回退到 `Arm_Stop`。

该入口只产生 stop-only 现场证据，不表示完整 M5.8-b、worker drain、measured-stop
acknowledgement 或 M6 authority 已通过。

## M5.8-b1-real-A idle-worker drain/stop-only preflight

`run_rm65_idle_worker_drain_preflight` 是另一个独立、需要现场授权的状态改变入口。它会启动真实 RM65
background worker，但不会给 worker 提供任何 target；确认 worker 已运行且 `send_count=0` 后，推进一次
execution epoch，依次 drain position-reference、left sender、right sender，最后向每侧最多发送一次
`Arm_Slow_Stop`。

这个入口采用专用 profile `m5_8b1_real_a_idle_worker_drain_stop_only_v1`，其保守配置与正式操作严格隔离：

- 不修改 `m6_candidate_policy_arm_qpos_v1`、canonical deploy 参数或 backend 默认值；
- 不提供 motion、home、enable、rearm、force-position 或 lifecycle authority；
- authorization、owner token、epoch registry 和 worker 只在本次进程内有效；
- 证据默认仍写入 `teleop_records/preflight_evidence/m5_8b1_real_a/<runtime_session_id>/`；显式传入
  `--artifact-layout runtime-artifacts-v1` 时改写到
  `runtime_artifacts/evidence/preflight/rm65_idle_worker_drain_stop_only/<runtime_session_id>/`，而
  `--evidence-root` 始终优先；两种布局都会在 `evidence.json` 记录实际存储选择；
- 诊断 WAL 与 `evidence.json` 明确不能 seal/commit 为公开 v2.4 capture；
- 后续真实运动必须重新构造对应 execution profile 和 authorization，不能继承本入口的 deny-motion、
  无target或一次性stop设置。

该批次只验证真实 idle worker 的 epoch/drain barrier 和 drain 后的 final Slow-Stop，不证明带target streaming、
motion permit、physical measured-stop、owner切换或M6 authority。

2026-07-21 的 scoped 现场批次已完成：两侧worker在无target条件下分别运行103/93 tick且`send_count=0`，
三个required participant完成execution epoch 0→1 drain；两次Slow-Stop均accepted且SDK返回0，诊断trace为
10/10 durable并保持`capture_commit_eligible=false`。因为没有target或in-flight旧调用，这份证据只关闭
idle-worker profile；带target streaming与production mediation仍需独立授权和验收。

## 入口

主脚本：

- `teleop_runtime/debug/rm65_real_debug/run_rm65_real_state_viz.py`

推荐启动方式：

```bash
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"

python -m teleop_runtime.debug.rm65_real_debug.run_rm65_real_state_viz \
  --left-arm-ip 192.168.101.21 \
  --right-arm-ip 192.168.1.20 \
  --left-hand-ip 192.168.12.210 \
  --right-hand-ip 192.168.11.210 \
  --tcp-mode hand_base
```

如果已经有 calibration profile，希望直接复用其中的 `T_D_B`：

```bash
python -m teleop_runtime.debug.rm65_real_debug.run_rm65_real_state_viz \
  --left-arm-ip 192.168.101.21 \
  --right-arm-ip 192.168.1.20 \
  --tcp-mode hand_base \
  --load-profile teleop_records/calibration_profiles/your_profile.json
```

如果只想看终端状态，不开 SAPIEN：

```bash
python -m teleop_runtime.debug.rm65_real_debug.run_rm65_real_state_viz \
  --left-arm-ip 192.168.101.21 \
  --right-arm-ip 192.168.1.20 \
  --skip-renderer
```

如果你只是想用同一套 viewer / desk / tcp 可视化，但不连真机：

```bash
python -m teleop_runtime.debug.rm65_real_debug.run_rm65_real_state_viz \
  --arm-mode virtual \
  --tcp-mode hand_base
```

左臂示例：

```bash
python -m teleop_runtime.debug.rm65_real_debug.run_rm65_real_state_viz \
  --left-arm-ip 192.168.101.21 \
  --left-hand-ip 192.168.12.210 \
  --tcp-mode hand_base
```

右臂示例：

```bash
python -m teleop_runtime.debug.rm65_real_debug.run_rm65_real_state_viz \
  --right-arm-ip 192.168.1.20 \
  --right-hand-ip 192.168.11.210 \
  --tcp-mode hand_base
```

如果你发现“真实左臂的运动本身对，但它落在了逻辑右臂位置”，先确认你用的是当前版本的共享 viewer 视角：

- 屏幕左边 = `desk -Y` = 逻辑 `left_arm`
- 屏幕右边 = `desk +Y` = 逻辑 `right_arm`

只有在这个视角下仍然确认**物理 IP 到逻辑 side 的分配反了**时，才使用：

```bash
--swap-arm-sides
```

例如：

```bash
python -m teleop_runtime.debug.rm65_real_debug.run_rm65_real_state_viz \
  --left-arm-ip 192.168.101.21 \
  --right-arm-ip 192.168.1.20 \
  --tcp-mode hand_base \
  --swap-arm-sides
```

启动后终端会明确打印：

- `left_arm_ip=... -> logical_right_arm`
- `right_arm_ip=... -> logical_left_arm`

## 参数

- `--arm-mode {real,virtual}`: `real` 读真机，`virtual` 只渲染默认姿态
- `--left-arm-ip` / `--right-arm-ip`: 左右 RM65 IP；在 `arm-mode=real` 下提供哪一侧，就读取哪一侧真机
- `--left-hand-ip` / `--right-hand-ip`: 左右 InspireTac IP；可独立于 arm 连接，未提供时该手保持固定张开姿态
- `--swap-arm-sides`: 交换提供的两个 IP 对应的逻辑左/右臂
- `--tcp-mode hand_base`: 历史兼容参数；当前 action 主 frame 是 `tool_tcp`
- `--load-profile PATH`: 载入已有 calibration profile
- `--save-initial-alignment PATH_OR_DIR`: 指定初始对齐状态保存位置；运行时按 `s` 保存
- `--hz`: 状态轮询和渲染频率，默认 `30`
- `--status-sec`: 终端状态心跳周期，默认 `0`，表示只打印启动信息、状态变化、错误和保存事件；需要周期性姿态摘要时传入正数
- `--headless`: 禁用 SAPIEN viewer
- `--skip-renderer`: 不创建 SAPIEN，只做终端输出
- `--max-steps`: 限制最大循环步数

## 可视化内容

SAPIEN 中当前会显示：

- `desk`
- `left_arm_base`
- `right_arm_base`
- `left_tcp_actual`
- `right_tcp_actual`

当前 viewer 默认视角已经固定成：

- 屏幕左边 = 逻辑 `left_arm` = `desk -Y`
- 屏幕右边 = 逻辑 `right_arm` = `desk +Y`

这个视角与 `teleop_runtime` 里的 `vive` / calibration 可视化使用同一套共享场景布局与相机朝向。
单臂模式下只显示已连接那一侧的 `arm_base` 和 `tcp_actual`，避免未连接一侧的 debug frame 继续干扰判断。

终端默认每 `0.5s` 左右输出一次当前摘要，例如：

```text
[20:12:05] [RM65Real] left result=0 q=[...] tcp_xyz=[...] tcp_rpy_deg=[...]
[20:12:05] [RM65Real] right result=0 q=[...] tcp_xyz=[...] tcp_rpy_deg=[...]
```

其中：

- `q` 是当前真实关节角，单位 `rad`
- `tcp_xyz` / `tcp_rpy_deg` 是当前选中 `tcp_mode` 在 desk 坐标系下的位姿
- 如果连接了 InspireTac，还会额外输出 `left_hand q6=[...]` / `right_hand q6=[...]`

## 保存初始对齐状态

需要从环境准备、只读检查、按 `s` 保存到遥操作复用的完整步骤时，参见
[`INITIAL_ALIGNMENT_WORKFLOW.md`](./INITIAL_ALIGNMENT_WORKFLOW.md)。

运行时按：

- `s`: 把当前左右臂 `arm_qpos + tcp_pose`，以及可用的手部 `q6`，保存成一份初始对齐状态 JSON

如果没有指定 `--save-initial-alignment`，默认会写到：

- `teleop_records/initial_alignment_states/<timestamp>_rm65_initial_alignment.json`

如果指定了目录，例如：

```bash
python -m teleop_runtime.debug.rm65_real_debug.run_rm65_real_state_viz \
  --left-arm-ip 192.168.101.21 \
  --right-arm-ip 192.168.1.20 \
  --save-initial-alignment teleop_records/initial_alignment_states
```

则按 `s` 后会在该目录下生成一个带时间戳的 JSON。

保存下来的初始对齐状态可以直接给虚拟运行入口加载，这样就不需要在启动时再读真机起始状态：

```bash
python -m teleop_runtime.debug.run_vive_sapien_virtual \
  --host 0.0.0.0 \
  --port 9999 \
  --arm-mode real \
  --control-mode absolute \
  --load-profile teleop_records/calibration_profiles/your_profile.json \
  --load-initial-alignment teleop_records/initial_alignment_states/your_alignment.json
```

这里 `arm-mode=real` 仍然保留真机风格的恢复/限步语义，但 preview 起始姿态来自保存文件，不再依赖 `Get_Current_Arm_State`。

## 建议检查顺序

1. 先加载 profile，确认左右 TCP 是否位于桌面上方，左右分布是否与真实安装一致
2. 如果接了 InspireTac，确认手指开合方向和手腕安装方向一致
3. 需要复用当前起始姿态时，按 `s` 保存一份初始对齐状态
4. 如果看到明显偏差，再回头检查 `T_D_B_left/right` 或 URDF/TCP 定义

## 退出

- 在交互终端中按 `q`

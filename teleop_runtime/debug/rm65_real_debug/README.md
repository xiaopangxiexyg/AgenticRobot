# RM65 Real Debug

这个目录提供一个**只读真机状态**的系统检查入口，用于在不上 teleop、不发送任何运动指令的前提下，把真实 RM65 的当前关节状态，以及可选 InspireTac 手部关节状态，映射回 SAPIEN。

当前也支持 `--arm-mode virtual`，直接渲染默认双臂姿态，不连接真机。

它的目标是：

1. 连接左/右 RM65
2. 可选连接左/右 InspireTac
3. 只调用 `Get_Current_Arm_State` 和手部只读遥测
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

唯一的真机读操作是：

- `Get_Current_Arm_State`
- `InspireTac.read_joint_pos()`

## 入口

主脚本：

- [run_rm65_real_state_viz.py](/home/user/zc/dex_inspire_teleop/teleop_runtime/debug/rm65_real_debug/run_rm65_real_state_viz.py)

推荐启动方式：

```bash
conda activate grasp_py310_copy

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
- `--tcp-mode hand_base`: 当前 runtime 固定只使用 `hand_base`
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

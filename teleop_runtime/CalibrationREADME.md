# Calibration README

这份文档只说明**当前已经落地的** VIVE -> RM65 标定调试流程。  
它对应的入口是：

- [teleop_runtime/debug/run_vive_rm65_calibration_debug.py](./debug/run_vive_rm65_calibration_debug.py)

当前这套流程的核心目标是：

1. 用 `auxiliary tracker` 标定一个**虚拟 desk** 坐标系 `V`
2. 用左右 wrist tracker **同步**标定 `tracker -> hand control frame` 外参
3. 可选地在 SAPIEN 里查看 `desk/base/raw tracker/calibrated hand` 坐标系并做 wrist 细调
4. 保存一个可复用的 tracker/station calibration profile

这套流程**不会**重标 `desk -> base`。  
机器人侧继续沿用当前 preset 里的 `T_D_B_left/right`。
它现在也**不会**采集 session `home anchor`。`home` 已经移到
[teleop_runtime/debug/run_vive_sapien_virtual.py](./debug/run_vive_sapien_virtual.py)
和后续真机部署入口中，每次启动单独采。

当前终端日志默认带 `[HH:MM:SS]` 时间戳，并且只在状态变化、关键采样进度节点或异常时输出，不再持续刷固定频率状态行。
当前默认只在 `TRACKER_FRAME_VIZ` 阶段执行 SAPIEN `render_data(...)`；`LINK_CHECK`、`VirtualDesk`、`WristCoarse` 都不会做逐帧渲染，以避免拖慢采样。
当前 timed capture 的 `LIVE ...` 心跳默认约每 `0.5s` 一次。
当前 action 主 frame 已统一为 `tool_tcp`；`hand_base` / `arm_link6` 只作为 RM65
后端内部或兼容 frame 使用。

## 1. 坐标系定义

当前工具里涉及的主要坐标系：

- `W`: `vive_world`
- `V`: `virtual desk`
- `D`: 机器人侧当前 preset 使用的 desk frame
- `B_left / B_right`: 左右机械臂 base
- `T_left / T_right`: 左右 wrist tracker
- `H_left / H_right`: 左右人手控制参考系
- `E_left / E_right`: 左右机器人 TCP

当前流程里，真正会求的是：

- `T_W_V`
- `T_T_H_left`
- `T_T_H_right`

不会求的是：

- `T_D_B_left/right`

当前 `teleop_runtime` 的 `D` 约定固定为：

- `+X_D`: forward
- `+Y_D`: left

注意：RM65 的共享 SAPIEN 可视化布局沿用当前项目已有的双臂摆放约定，逻辑 `left_arm` 位于 `desk -Y`，逻辑 `right_arm` 位于 `desk +Y`。因此“desk 轴语义”和“左右臂在 viewer 中的槽位”不要混为一谈，真机检查与 VIVE 虚拟跑法都使用同一套布局。
- `+Z_D`: up

在默认双臂 preset 下：

- `B_left` 位于 `D.y > 0`
- `B_right` 位于 `D.y < 0`
- 所有共享 SAPIEN viewer 统一从 `D.x < 0` 一侧看向原点，所以屏幕左边就是逻辑左臂，屏幕右边就是逻辑右臂

## 2. 这套流程为什么叫“虚拟 desk”

因为操作者不是固定在真实桌面边缘工作的。  
如果强依赖“把 tracker 插到桌面固定槽位”，会与实际 teleop 使用方式冲突。

因此这里的思路是：

- 让操作者通过一组**动作**定义“前/左/上”三个方向
- 用这些动作拟合出一个虚拟工作平面 `V`
- 再把人手运动先映射到 `V`
- 最后再映射到机器人侧当前使用的 `D`

所以：

- `V` 决定人侧运动方向解释
- `D` 决定机器人侧固定 desk/base 关系

## 3. 工具入口

推荐命令：

```bash
conda activate dexteleop310

python hardware/VIVE/mocap/vive_data_sender.py \
  --host 127.0.0.1 \
  --port 9999 \
  --hz 60 \
  --tracker tracker_1:left_wrist \
  --tracker tracker_2:right_wrist
```

第三只 `tracker_3:auxiliary` 可选。没有 auxiliary 时，calibration debug 会用左或右
wrist tracker 完成 virtual desk 扫动。

```bash
python -m teleop_runtime.debug.run_vive_rm65_calibration_debug \
  --host 0.0.0.0 \
  --port 9999 \
  --gui \
  --arm-mode real \
  --left-arm-ip 192.168.101.21 \
  --right-arm-ip 192.168.1.20
```

`--gui` 会打开浏览器控制面板，等价于 `--ui-mode web`。如果只想使用终端按键，
可以改用 `--no-gui` 或 `--ui-mode terminal`。

如果本机没有可用渲染设备，可以先加：

```bash
--skip-renderer
```

如果这一步只想做 tracker/station 标定，不连接真机机械臂，也可以显式切到纯虚拟渲染：

```bash
python -m teleop_runtime.debug.run_vive_rm65_calibration_debug \
  --host 0.0.0.0 \
  --port 9999 \
  --arm-mode virtual
```

## 4. 当前状态机

这个 app 当前实现的顺序是：

1. `LINK_CHECK`
2. `VIRTUAL_DESK_X`
3. `VIRTUAL_DESK_Y`
4. `VIRTUAL_DESK_Z`
5. `WRIST_COARSE`
6. `TRACKER_FRAME_VIZ` 可选
7. `SAVE`

## 5. 按键说明

### 5.1 通用按键

- `q`: 退出
- `c`: 开始当前 timed capture
- `r`: 中止并重置当前 capture
- `f`: 需要时切换 `virtual desk` 的 `XY` 符号翻转，也就是绕 `+Z` 追加 `180°`

### 5.2 链路检查阶段

- `n`: 在 `left/right/aux` 三个 tracker 都在线后继续

### 5.3 Tracker Frame Visualization / Fine Tune 阶段

- `1`: 选择左手做细调
- `2`: 选择右手做细调
- `j/l`: 当前选中侧 `yaw -/+`
- `i/k`: 当前选中侧 `pitch +/-`
- `u/o`: 当前选中侧 `roll -/+`
- `x`: 当前选中侧细调角度清零
- `v`: 切换 `frame_viz_mode = calibrated / raw / both`
- `s`: 保存 calibration profile 并退出

注意：

- `timed capture` 默认不需要第二次按 `c` 结束，工具会在时间窗结束时自动校验并结束
- `timed capture` 的样本数按“新收到的 VIVE 包”计数，不按主循环次数计数

说明：

- 这个阶段需要显式加 `--show-tracker-frame-viz`
- 默认 `frame_viz_mode=calibrated`，避免 `raw tracker` 和 `calibrated hand` 在平移偏置为 `0` 时叠成 6 根轴
- 可用 `v` 或 `--frame-viz-mode` 在 `calibrated / raw / both` 之间切换
- 如果发现前后和左右符号同时反了，可按 `f` 切换一次 `XY` 符号翻转
- 它会打开 SAPIEN viewer，并持续显示：
  - `desk` 坐标系
  - 左右 arm base 坐标系
  - 左右 raw wrist tracker 坐标系
  - 左右 calibrated hand-control 坐标系
  - 左右当前 TCP 坐标系
- viewer 相机可以直接用 SAPIEN 自带交互控制移动视角
- 如果关闭 SAPIEN viewer 窗口，校准脚本会继续在无 3D 窗口模式下运行；仍可在终端或 Web UI 里按 `s` 保存退出

### 5.5 Timed Capture 默认时长

- `VirtualDesk X/Y/Z`: 默认 `3.0s`
- `WristCoarse`: 默认 `2.0s`

对应 CLI 参数：

- `--sweep-capture-sec`
- `--hold-capture-sec`
- `--show-tracker-frame-viz`
- `--frame-viz-mode`
- `--debug-frame-axis-length`
- `--load-profile`
- `--virtual-desk-post-yaw-deg`

## 6. 详细动作要求

### 6.1 LINK_CHECK

目标：

- 确认 `left_wrist_pose`
- 确认 `right_wrist_pose`
- 确认 `auxiliary_tracker_pose`

通过条件：

- 三个 channel 都在线且 `valid=True`

终端会定期输出 `LIVE ... samples=... valid=...`，默认约每 `0.5s` 一次，即使样本数没有跳到关键阈值，也能看出当前 capture 没有卡住。

### 6.2 Virtual Desk X

动作：

- 按一次 `c`
- 手持 auxiliary tracker
- 做 `后 -> 前` 扫动
- 建议从更靠近身体后方的位置开始，在自动结束前停在身体前方
- 高度尽量保持不变
- 振幅建议 `20-40 cm`

为什么要这样做：

- 当前工具会用轨迹的首尾趋势给主轴定正负号，所以 `X` 必须按 `后 -> 前`

### 6.3 Virtual Desk Y

动作：

- 按一次 `c`
- 做 `右 -> 左` 扫动
- 建议从身体右侧开始，在自动结束前停在身体左侧
- 保持高度尽量不变

### 6.4 Virtual Desk Z

动作：

- 按一次 `c`
- 做一次明确的低到高抬升
- 建议从低处开始，在自动结束前停在高处
- 不需要特别大幅，但要清楚
- 不要在同一个 capture 窗口里上去又回来，否则 `start_end_distance` 容易不足

三段 sweep 完成后，工具会自动把所有 sweep 样本的位置均值作为 `origin_V`。
这个原点只用于形成一个完整的坐标系；后续 runtime home 会重新绑定当前人手
和机器人 TCP，因此不需要额外做虚拟 desk 原点 hold。

### 6.5 Wrist Coarse

建议 neutral 姿态：

- 按一次 `c`
- 左右手同时进入 neutral canonical pose
- 前臂自然前伸
- 手背朝上
- 手指大致朝 `+X_V`
- 姿态在整个时间窗里尽量稳定

工具会在**同一个 capture 窗口里**分别计算左右手的粗外参：

- `T_T_H_side`

默认只靠这一步做旋转粗对齐。  
平移偏置先用参数输入，默认是 `0,0,0`。

### 6.7 Session Home 不在本工具中

`run_vive_rm65_calibration_debug.py` 现在只负责保存可复用的 tracker/station calibration。

每次启动都变化的内容，例如：

- 真机当前起始姿态
- 本次操作者双手的起始姿态
- 本次 teleop 的 `home anchor`

都不在这一步保存。  
这些内容现在放到 `run_vive_sapien_virtual.py` 和后续真机部署入口里再采。

## 7. 虚拟 desk 拟合方法

当前实现里：

- `X/Y/Z` 三段 sweep 都只用**位置轨迹**
- 对每段轨迹做 PCA / SVD
- 取第一主方向作为该动作的主轴

然后：

```text
x_raw = principal_axis(X-sweep)
y_raw = principal_axis(Y-sweep)
z_raw = principal_axis(Z-sweep)
```

当前实际做法是：

```text
x_V = x_raw
y_V = normalize(y_raw - dot(y_raw, x_raw) * x_raw)
z_V = normalize(cross(x_V, y_V))
if dot(z_V, z_raw) < 0:
    y_V = -y_V
    z_V = -z_V
```

也就是：

- `X sweep` 作为主参考方向
- `Y sweep` 只做最小必要正交化
- `Z sweep` 主要用来判定 `X/Y` 定义出的手性方向是否一致

这样不会因为某一段 sweep 有少量串轴就直接失败，但仍会记录：

- raw sweep 之间的互相不正交程度
- 最终正交基相对 raw 轴各自修正了多少角度
- `Z` 与 `X/Y` 推导结果在翻转前的夹角

## 8. 默认质量门槛

当前默认要求：

- 每段 sweep 至少 `8` 帧
- 每段 sweep 的 `extent >= 0.15 m`
- 每段 sweep 的 PCA 主方向 `variance_ratio >= 0.50`
- 每段 sweep 的 `start_end_distance >= 0.08 m`
- 拟合后的 `x/y/z` 任一轴相对 raw sweep 方向的修正角度 `<= 20 deg`
- hold / wrist 默认至少 `20` 帧

这些值都可以通过 CLI 改。

## 9. 保存内容

默认保存到：

- `teleop_records/calibration_profiles/<timestamp>_vive_rm65_virtual_desk.json`

其中会包含：

- `T_W_V`
- `virtual_desk_post_yaw_deg`
- 左右 wrist 的 coarse/final tracker offsets
- 虚拟 desk 拟合质量信息
- 本次保存时间
- 不包含 session home / `arms[*].home_pose_*` / `home_qpos`

如果你已经有 profile，可以直接加载并跳过前面的虚拟 desk / wrist coarse：

```bash
python -m teleop_runtime.debug.run_vive_rm65_calibration_debug \
  --host 0.0.0.0 \
  --port 9999 \
  --load-profile teleop_records/calibration_profiles/your_profile.json
```

对于旧版 `vive_rm65_virtual_desk` profile，如果当时是按旧说明做的 `X: 前 -> 后`、`Y: 左 -> 右` 扫动，
当前 runtime 会在加载时自动补一次等效 `post_yaw=180deg` 的兼容修正，
把虚拟 desk 统一到 `+x=forward, +y=left, +z=up`。

如果你还希望连 `run_vive_sapien_virtual.py` 的起始机械臂姿态也一起复用，可以先在
`teleop_runtime.debug.rm65_real_debug.run_rm65_real_state_viz` 里按 `s` 保存一份
initial alignment，然后启动虚拟入口时额外传：

```bash
python -m teleop_runtime.debug.run_vive_sapien_virtual \
  --host 0.0.0.0 \
  --port 9999 \
  --load-profile teleop_records/calibration_profiles/20260416_231341_vive_rm65_virtual_desk.json \
  --load-initial-alignment teleop_records/initial_alignment_states/20260418_154512_rm65_initial_alignment.json
```

这样 preview 起始姿态会直接用保存下来的 `arm_qpos`，不再依赖启动时读取真机状态。
它也会在 `run_vive_sapien_virtual.py` 里作为 robot home anchor 的来源使用。
但它不包含当前操作者双手的 home anchor，所以
`run_vive_sapien_virtual.py` 仍然会在你按下 `p` 后做一次 runtime home calibration：

- tracker home：始终重新采
- robot home：优先用 `--load-initial-alignment`
- 如果没有 `--load-initial-alignment` 且真机已连接，则读取当前真机 arm home
- 否则使用当前 preview arm pose

## 10. 典型验证动作

保存前建议至少做这些动作：

1. 左手单独向前推一点，`left_hand_calibrated` 应主要沿机器人前方解释
2. 左手单独向左移一点，`left_hand_calibrated` 应主要沿机器人左方解释
3. 左手向上抬一点，`left_hand_calibrated` 应向上
4. 右手重复一次
5. 左右手同时对称前推，两侧 `hand_calibrated` 坐标系应整体对称前伸

如果方向不对，优先怀疑顺序：

1. `VIRTUAL_DESK_X/Y/Z`
2. 当前侧 wrist coarse/fine tune

## 11. 常见失败现象

### 11.1 X/Y/Z 标定一直被拒绝

常见原因：

- sweep 幅度太小
- 高度变化太大
- 开始结束方向不清楚
- 中途停顿太多

### 11.2 左右手方向镜像错了

优先排查：

- 左右 wrist tracker role 是否写反
- 左右手 neutral pose 是否一致
- fine tune 后是否已经重新保存 profile 并在 session runner 中重新加载
- 当前 wrist coarse 默认 canonical hand frame 是 XYZ Euler `[0, 0, 0]`；neutral hold 直接定义 tracker-to-hand 粗标外参

### 11.3 一调 wrist offset，后续 session 方向又不对

优先检查：

- 你是否已经重新保存 calibration profile
- `run_vive_sapien_virtual.py` 是否重新加载了新 profile
- session home 是否在新 profile 生效后重新采集

### 11.4 看起来“能动”，但姿态总是别扭

通常说明：

- 虚拟 desk 方向大致对了
- 但 wrist tracker 的旋转外参还不够准

这时应在 `TRACKER_FRAME_VIZ` 阶段用 `j/l/i/k/u/o` 做小步微调。

## 12. 当前限制

- 现在是“虚拟 desk”标定，不恢复物理桌面的绝对世界位置
- 现在 wrist 外参主要靠 neutral pose + 细调，不是完全自动辨识
- SAPIEN 里现在主要显示 `desk / arm_base / tracker_raw / hand_calibrated / tcp` 坐标系，`W` 不直接渲染
- `desk -> base` 外参仍沿用 preset，不在这套流程里重标

## 13. 建议工作流

1. 跑 sender
2. 跑 calibration debug app
3. 做 `LINK_CHECK`
4. 做 `VIRTUAL_DESK_X/Y/Z/NEUTRAL`
5. 做同步 `WRIST_COARSE`
6. 如果需要，进入 `TRACKER_FRAME_VIZ` 检查 raw tracker / calibrated hand 坐标系
7. 如果需要，进 `TRACKER_FRAME_VIZ` 做小步微调 wrist 外参
8. 满意后 `s` 保存
9. 后续每次启动 `run_vive_sapien_virtual.py` 或真机部署时，再做 session home capture

保存好的 profile 后续可直接用：

```python
from teleop_runtime import TeleopCalibration

calibration = TeleopCalibration.load_json("path/to/profile.json")
```

# Teleop Runtime

> 本页说明模块接口和局部示例。正式真机遥操/数采的组合命令统一从
> [`docs/real/README.md`](../docs/real/README.md) 复制。

`teleop_runtime` 是新的通用遥操作运行时骨架。当前版本已经能把下面两条链路接到统一接口里：

- `VIVE tracker -> pose_to_arm -> RM65`
- `hand_q6 reader -> joints_to_dexhand -> Inspire Tac`
- `hand_q6 reader -> scalar_to_gripper -> DMTAC gripper`

它的目标不是替代所有旧代码，而是把“输入源 / retargeter / backend / rig / session”拆开，后续可以继续接别的输入设备、别的机械臂、别的末端。

更完整的设计背景见 [teleop_runtime/blueprint.md](./blueprint.md)。

## 当前已实现

- 通用数据模型：`ObservationChannel`、`EffectorCommand`、`TeleopSession`
- 通用标定对象：`TeleopCalibration`
- `VivePoseSource`
- `HandQ6Source`
- `PoseToArmRetargeter`
- `JointsToDexHandRetargeter`
- `ScalarToGripperRetargeter`
- `RM65ArmBackend`
- `InspireTacHandBackend`
- `DMTacGripperBackend`
- `BimanualRig`
- headless/兼容数据采集 CLI runner（正式 GUI 数采主线不再从这里扩展第二套控制实现）：
  `python -m teleop_runtime.collection`
  （兼容旧入口 `python -m teleop_runtime.collection.run_collect`）
- 数据采集使用说明：
  [teleop_runtime/collection/README.md](./collection/README.md)
- 独立策略部署 CLI runner：
  `python -m teleop_runtime.deploy`
  （同一受支持运行时也提供直接入口
  `python -m teleop_runtime.deploy.run_policy`；OpenPI launcher 当前依赖其 `main`）
- 策略部署使用说明：
  [teleop_runtime/deploy/README.md](./deploy/README.md)
- 部署动作链路现在以 `RobotAction` 为 sim/real 通用语义边界，以
  `TeleopCommand` 为 launcher/backend 下发边界；真机分发逻辑只保留在
  `teleop_runtime.deploy.dispatch`。后续硬件无关 action/deploy preparation 目标是
  迁到 `robot_motion/action` 和 `robot_motion/deploy`。
- 交互式 calibration debug app：
  [teleop_runtime/debug/run_vive_rm65_calibration_debug.py](./debug/run_vive_rm65_calibration_debug.py)
- 默认录制：`teleop_records/<timestamp>/`

## 当前未实现

- 基于固定物理工装的 `desk/world` 绝对外参向导
- 键盘事件采集线程

所以当前推荐用法是：

1. 先用 `connect_hardware=False` 做 dry-run
2. 先完成启动标定和虚拟验收，再接真硬件
3. 真机遥操作和正式 GUI 数采走
   `teleop_runtime.debug.run_vive_rm65_real_teleop --gui`；
   `teleop_runtime.collection` 仅作为 headless/兼容入口保留

## 目录

- `teleop_runtime/sources`: 输入源
- `teleop_runtime/retargeters`: 输入到命令的映射
- `teleop_runtime/backends`: 具体硬件执行
- `teleop_runtime/rigs`: 多 backend 组合
- `teleop_runtime/session.py`: 主循环
- `teleop_runtime/calibration.py`: 标定和坐标变换

## Action Contract 说明

当前 RM65 + hand/gripper 主线仍使用 26 维 action：

- `left_arm_pose7 + right_arm_pose7 + left_hand_q6 + right_hand_q6`

arm pose7 的正式语义是 `T_arm_base_tool_tcp`：

- `tool_tcp` 是 action 主 frame；tracker retarget、数据集 arm pose、policy pose 输出和真机执行目标都按 TCP 解释。
- TCP 与末端工具 profile 绑定，左右臂分开加载，例如
  `hardware/rm65/utils/configs/calibrations/rm65_right_empty.json`。
- 如果只控制机械臂、不加载末端工具 profile，`tool_tcp` 会退回为机械臂
  `arm_endpoint` 本身。
- 加载 profile 时会检查 TCP 相对 `arm_endpoint` 的偏移；距离超过 `0.20m` 或总旋转
  偏差超过 `90deg` 时会打印显式 warning。
- RM65 后端内部会把 `tool_tcp` 转到 SDK/IK 需要的 hand_base 或 arm_link6 frame；这些内部 frame 不应作为模型 action 语义。
- `arm_endpoint` / `hand_base` 仅保留为兼容和调试 frame。
- 数据集和部署 metadata 会写 `arm_action_specs_by_side`、
  `arm_controlled_frame_by_side` 和 `arm_reference_frame_by_side`；下游代码应按这些
  spec 解释 arm pose，不要凭维度猜测 pose 控制的是 TCP 还是内部 frame。
- 旧参数 `--tcp-mode rm_eef` 现在按 `tool_tcp` 归一；旧 profile 里显式写出的
  `hand_base` 仍保留为兼容 frame。
- 旧采集 row 如果缺少 canonical action metadata，`teleop_runtime.action.row_replay`
  会显式按历史 `arm_endpoint` pose contract 回放；带 metadata 的新数据不走该兼容分支。

为兼容夹爪和高自由度灵巧手，action 层已经预留 hand16 contract：

- `left_arm_pose7 + right_arm_pose7 + left_hand_q16 + right_hand_q16`

hand16 中的空槽必须按无效值处理，而不是按 0 处理：

- 有效槽由 `hand_active_dof_by_side` / `hand_active_mask_by_side` / 本体 hand spec 定义
- 无效槽值约定为 `NaN`
- 下游模型、数据统计、loss、replay 和部署下发都只能使用有效槽
- 旧数据集和旧模型应自行通过 adapter 映射到当前 contract，runtime 主线不为旧维度保留额外语义兼容

## 依赖

按链路分别准备：

- VIVE 发送端：`triad_openvr`
- RM65 arm backend：RM65 SDK，可导入 `hardware.rm65.robotic_arm_package.robotic_arm`
- Inspire hand backend：`hardware/inspire_tac/inspire_tac_modbus.py` 依赖的 `pymodbus`
- arm IK：`rm65_inspire_tac.ik.SingleArmIKSolver` 依赖 `pinocchio`
- 推荐环境：`conda activate dexteleop310`

如果只做 dry-run，可以先把 arm / hand backend 的 `connect_hardware=False`。

## VIVE 发送端

推荐发送端在 [hardware/VIVE/mocap/vive_data_sender.py](../hardware/VIVE/mocap/vive_data_sender.py)。它每帧只取一次 OpenVR pose snapshot，再复用到所有 tracker，并会打印 loop/get_pose/json/send 耗时。

sender 只接受两个 GenericTracker，并按设备名字典序自动映射：较小者为
`left_wrist`，较大者为 `right_wrist`。发现多于两个 tracker 会直接报错。

示例：

```bash
python hardware/VIVE/mocap/vive_data_sender.py \
  --host 192.168.20.90 \
  --port 9999 \
  --hz 300
```

发送协议是结构化 JSON UDP，`teleop_runtime` 也保留了 legacy float 包兼容。仓库里只保留这一个 sender 入口，避免误跑旧的 float/满速 UDP sender。

## Calibration Tool

当前已经有一个可运行的交互式标定调试入口：

- [teleop_runtime/debug/run_vive_rm65_calibration_debug.py](./debug/run_vive_rm65_calibration_debug.py)

它当前实现的是：

- `auxiliary/left_wrist/right_wrist tracker -> virtual desk` 标定
- 左右 wrist tracker 同步 coarse 标定
- 可选的 tracker/hand frame 可视化
- `TRACKER_FRAME_VIZ` 中的 wrist 姿态细调

这套流程的详细动作、按键和质量门槛见：

- [teleop_runtime/CalibrationREADME.md](./CalibrationREADME.md)

当前终端日志默认带 `[HH:MM:SS]` 时间戳，并且优先按“状态变化 / 关键进度 / 异常”输出，不再固定频率重复刷屏。
如果当前终端支持 ANSI 颜色，关键提示还会用颜色和大写步骤名强调。
当前 timed capture 的 `LIVE ...` 心跳默认约每 `0.5s` 一次。

## 标定流程建议

当前已经落地的主线不是“固定物理 desk 绝对外参恢复”，而是：

1. 用 auxiliary 或任意一只 wrist tracker 标定一个 `virtual desk`
2. 用左右 wrist tracker 同步标定 `tracker -> hand control frame` 外参
3. 每次 session 再做 `home anchor`

也就是先解决“人侧前/左/上如何解释”，再解决“tracker 安装姿态如何解释”，最后解决“本次从哪里开始跟”。
当前 `home anchor` 会对齐**完整末端位姿**：

- 位置：按 tracker 相对 home 的位移增量叠加到 robot home
- 方向：按 tracker 相对 home 的姿态变化叠加到 robot home

这样保持 tracker 不动并按下 `p` 后，机器人会留在当前 robot home，而不会只对齐 xyz 后再因为姿态不一致被拉走。

如果你需要完整动作说明，优先看：

- [teleop_runtime/CalibrationREADME.md](./CalibrationREADME.md)

### 1. 结构层

这部分仍然沿用当前 preset，不在这套标定工具里重标：

- `arms[*].T_D_B`
- `workspace_min/workspace_max`
- 双臂相对基距
- `tcp_mode` 对应的 TCP frame

当前入口默认使用：

- `build_annotated_dual_rm65_calibration(...)`
- `left_right_base_distance = 0.525`
- `desk` 轴定义固定为：`+X` 是操作者面向桌面的方向，`+Y` 是从 workspace near-boundary 原点朝向操作者左侧的方向，`+Z` 是垂直桌面向上的方向
- 默认双臂 base 摆放为 `left_arm @ -y`, `right_arm @ +y`
- 所有共享 SAPIEN viewer 统一使用“从 `desk +x` 看向原点”的视角，因此屏幕左边就是逻辑 `left_arm(-y)`，屏幕右边就是逻辑 `right_arm(+y)`

如果机器人底座、桌面定义或 workspace 变了，这部分要单独重新确认。

workspace 现在由独立配置文件管理，不需要手改 calibration profile：

- [teleop_runtime/configs/arm_workspace.json](./configs/arm_workspace.json)

其中 desk `+x` 是 forward；如果只是前伸被卡住，通常先把
`default.workspace_max` 的第一个值调大，例如从 `0.60` 调到 `0.75` 或 `0.80`。
左右臂需要不同范围时，在 `sides.left` 或 `sides.right` 里覆盖单侧。

真机遥操作入口也支持临时命令行覆盖，方便现场试值：

```bash
python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  --load-profile teleop_records/calibration_profiles/your_profile.json \
  --arm-workspace-min -0.60 -0.80 0.00 \
  --arm-workspace-max 0.80 0.80 0.90
```

启动优先级是：命令行 `--arm-workspace-*` > `--arm-workspace-config` 指定的 JSON >
calibration profile 里的历史 workspace 字段。新保存的 calibration profile 不再写入
workspace 数值，只保留 tracker/desk 标定。

### 2. 输入映射层

这部分解决：

- `tracker_name -> tracker_role`
- `tracker_offsets[*]`
- 左右 wrist 的安装方向是否一致

当前代码里，receiver 会优先按 `tracker_role` 绑定，不再优先依赖设备名顺序。

### 3. 会话层

这部分是每次启动都要重做的：

- `home anchor`

前两项固定标定由 [run_vive_rm65_calibration_debug.py](./debug/run_vive_rm65_calibration_debug.py) 负责；`home anchor` 已经移到 [run_vive_sapien_virtual.py](./debug/run_vive_sapien_virtual.py) 和后续真机部署入口中按 session 重做。

## 当前推荐的标定入口

完整标定入口：

```bash
python -m teleop_runtime.debug.run_vive_rm65_calibration_debug \
  --host 0.0.0.0 \
  --port 9999
```

如果你已经保存过一个可用 profile，可以直接载入并跳过 `virtual desk` 和 `wrist coarse`：

```bash
python -m teleop_runtime.debug.run_vive_rm65_calibration_debug \
  --host 0.0.0.0 \
  --port 9999 \
  --load-profile teleop_records/calibration_profiles/your_profile.json
```

这一步会按顺序完成：

1. `LINK_CHECK`
2. `VIRTUAL_DESK_X`
3. `VIRTUAL_DESK_Y`
4. `VIRTUAL_DESK_Z`
5. `WRIST_COARSE`
6. `TRACKER_FRAME_VIZ` 可选
7. `SAVE`

当前这套流程的核心是：

- virtual desk tracker 不需要固定在物理桌面上；没有第三只 tracker 时直接使用左/右 wrist tracker
- virtual desk 原点由三段 X/Y/Z sweep 样本均值自动给出，不再额外采 `VIRTUAL_DESK_NEUTRAL`
- 通过一组扫动动作拟合出 `virtual desk`
- 机器人侧仍沿用现有 `desk -> base` 预设

## 标定动作的职责划分

### 1. Virtual Desk

这一步只解决人侧坐标系方向：

- `X sweep`: 从身体后侧扫到身体前侧，使拟合出的 `+X` 指向前方
- `Y sweep`: 从身体右侧扫到身体左侧，使拟合出的 `+Y` 指向左方
- `Z sweep`: 操作者上方
- `neutral hold`: 当前 session 的虚拟工作中心

这里不追求恢复物理桌面的绝对世界原点，只追求：

- 前/左/上方向正确
- 每次基站移动后都能重新稳定定义同一类操作语义

### 2. Wrist Tracker Offset

这一步只解决左右 wrist tracker 的安装外参：

- coarse：通过双手同步 neutral canonical pose 求初值
- fine：在验证阶段用 `j/l/i/k/u/o` 做小步旋转细调

平移偏置目前仍然走 CLI 参数：

- `--left-tracker-translation`
- `--right-tracker-translation`
- `--show-tracker-frame-viz`
- `--frame-viz-mode`

运行时 VIVE tracker 到手腕/TCP 的默认局部高度修正集中在
`teleop_runtime/hand_tcp_alignment.py` 的 `ViveTrackerAlignmentConfig`。
现有默认值保持 `height_correction_m = 0.04`，各入口统一暴露
`--vive-tracker-height-correction-m` 和 `--no-vive-tracker-height-correction`。

### 3. Home Anchor

这一步现在不在 calibration tool 中做。  
它已经移到 `run_vive_sapien_virtual.py` 和后续真机部署入口里，每次启动根据：

1. 当前真机起始姿态
2. 当前操作者双手起始姿态

重新建立本次 session 的映射锚点。

## 标定结果保存与加载

当前 calibration tool 会在最后 `SAVE` 阶段写出 tracker/station calibration profile。默认输出到：

- `teleop_records/calibration_profiles/<timestamp>_vive_rm65_virtual_desk.json`

保存内容包括：

- 当前 `vive_retarget_frame` 写回后的 `T_W_O / T_O_VR`（旧键 `T_O_D` 兼容保留）
- `virtual_desk_post_yaw_deg`
- 左右 wrist tracker 的 `coarse_offset / fine_tune_rpy_deg / final_offset`
- 不包含 session `home capture`
- 不包含当前 `arms[*].home_pose_*` / `home_qpos`
- 质量和来源 metadata

再次使用时有两种方式：

1. 继续进入 calibration tool，只跳过前面的标定阶段：

```bash
python -m teleop_runtime.debug.run_vive_rm65_calibration_debug \
  --host 0.0.0.0 \
  --port 9999 \
  --load-profile teleop_records/calibration_profiles/your_profile.json
```

2. 直接进入虚拟跑法，复用 profile，再做一次 runtime home calibration：

```bash
python -m teleop_runtime.debug.run_vive_sapien_virtual \
  --host 0.0.0.0 \
  --port 9999 \
  --control-mode absolute \
  --load-profile teleop_records/calibration_profiles/your_profile.json
```

注意：

- `--load-profile` 会复用 `virtual desk` 和 wrist 外参，但不会跳过本次 session 的 `home` 对齐
- `--load-initial-alignment` 会恢复保存下来的机器人起始姿态和 robot home anchor，但不会跳过本次 session 的 tracker home 采样
- 如果你在 `TRACKER_FRAME_VIZ` 里又改了 wrist fine tune 或按了 `f`，最好重新保存一份 profile
- 如果你只想一开始就把前后和左右同时翻转，可直接传 `--virtual-desk-post-yaw-deg 180`

## 什么时候用哪个入口

### `run_vive_rm65_calibration_debug.py`

用于：

- 第一次搭系统
- 基站位置变化后重新校准
- wrist tracker 重新安装后重做外参
- 需要 SAPIEN 内做小动作验收和 fine tune

### `run_vive_sapien_virtual.py`

用于：

- 已有可用 calibration profile 后的快速虚拟跑法
- 每次启动都重新做 session `home anchor`
- 简单验证 `absolute`
- 非完整标定场景下的 home-only 调试

它仍然有价值，但它不替代完整标定流程。

## 基站、tracker、机器人变化时该怎么处理

### 只移动了 VIVE 基站

推荐直接重跑完整 calibration app：

1. 重新做 `virtual desk`
2. 重新确认左右 wrist offset

之后再在 session runner 里重新采本次 `home capture`。

### 只重新安装了 wrist tracker

至少重做：

1. `LINK_CHECK`
2. 对应侧 wrist coarse
3. `TRACKER_FRAME_VIZ` 里的 fine tune
4. 在 session runner 里重新采 `HOME_CAPTURE`

### 机器人或桌面位置变化

这超出了当前 calibration app 的范围。需要重新确认：

- `arms[*].T_VR_B`（旧键 `T_D_B`）
- workspace box
- 双臂基距和 `vive_retarget_frame` 定义

## 当前代码里标定字段的实际含义

`TeleopCalibration` 当前最关键的字段：

- `T_W_O`
- `T_O_VR`（旧键 `T_O_D`）
- `tracker_offsets[*]`
- `arms[*].T_VR_B`（旧键 `T_D_B`）

对于当前 virtual-desk 标定，保存时采用的是：

- `T_W_O = I`
- `T_O_VR = T_W_V`

也就是把 virtual-desk 标定结果写入 `world -> vive_retarget_frame` 链路。这个历史
retarget frame 与 v2.3 dataset contract 的 canonical `desk` 无关。

如果在验证阶段额外做了 `XY` 符号翻转：

- `virtual_desk_post_yaw_deg = 180`

这个后处理角也会一起保存，后续 `--load-profile` 会自动恢复。

如果你已经保存过 profile，可以直接载入：

```python
from teleop_runtime import TeleopCalibration

calibration = TeleopCalibration.load_json("path/to/calibration.json")
```

## 最小 Dry-Run 示例

下面的例子会启动 source、retargeter 和 session，但不连接真机械臂/灵巧手。

```python
import time
import numpy as np

from teleop_runtime import (
    ArmCalibration,
    KeyboardButtonSource,
    RM65ArmBackendConfig,
    TeleopCalibration,
    TeleopSessionConfig,
    build_bimanual_session,
)

calibration = TeleopCalibration(
    T_W_O=np.eye(4),
    T_O_D=np.eye(4),
    arms={
        "left_arm": ArmCalibration(
            name="left_arm",
            side="left",
            T_D_B=np.eye(4),
            workspace_min=np.array([-0.6, -0.6, -0.2]),
            workspace_max=np.array([0.6, 0.6, 0.8]),
        ),
        "right_arm": ArmCalibration(
            name="right_arm",
            side="right",
            T_D_B=np.eye(4),
            workspace_min=np.array([-0.6, -0.6, -0.2]),
            workspace_max=np.array([0.6, 0.6, 0.8]),
        ),
    },
)

keyboard = KeyboardButtonSource()
keyboard.set_button("left_clutch_button", False, side="left")
keyboard.set_button("right_clutch_button", False, side="right")

session = build_bimanual_session(
    calibration=calibration,
    session_config=TeleopSessionConfig(
        mode="bimanual",
        record_enabled=True,
    ),
    keyboard_source=keyboard,
    left_arm_backend_config=RM65ArmBackendConfig(
        arm_ip="0.0.0.0",
        side="left",
        connect_hardware=False,
    ),
    right_arm_backend_config=RM65ArmBackendConfig(
        arm_ip="0.0.0.0",
        side="right",
        connect_hardware=False,
    ),
)

session.start()
session.set_calibration_ready()
session.mark_debug_passed()
session.arm_hardware()

try:
    while True:
        result = session.run_step()
        if result.faults:
            print("faults:", [fault.code for fault in result.faults])
        time.sleep(0.02)
finally:
    session.close()
```

## 真机用法

把 builder 里的 backend 配置切到真实 IP 即可：

```python
from teleop_runtime import RM65ArmBackendConfig, InspireTacHandBackendConfig
```

- 机械臂：`RM65ArmBackendConfig(arm_ip=..., connect_hardware=True)`
- 灵巧手：`InspireTacHandBackendConfig(hand_ip=..., connect_hardware=True)`
- 手部输入：给 `build_bimanual_session(...)` 传 `left_hand_serial_port` / `right_hand_serial_port`

如果左右 RM65 backend 都存在，`build_bimanual_session(...)` 现在默认会启用 joint-space 双臂碰撞规避 retargeter；只有单臂场景才退回到独立 `pose_to_arm` 路径。

## VIVE + SAPIEN 虚拟跑法

当前已经补了一个真实 VIVE 输入、SAPIEN 双臂可视化的虚拟跑法入口。  
它现在支持两种机械臂渲染模式：

- `--arm-mode real`：读取真 RM65 当前起始姿态，只把它复制到本地 preview，不发送任何运动指令
- `--arm-mode virtual`：完全不连真机，直接用本地 unlimited-speed preview

同时它现在也支持**可选的手部串口输入预览**：

- 传 `--left-hand-port` / `--right-hand-port` 后，会把 `hardware/teleop_hand/hand_q6_reader.py` 的 `q6` 输入接到 runtime
- 手部输入只影响 SAPIEN 中的手部渲染，不会改变 arm teleop 主链路
- 手部输入线程和 VIVE 线程解耦，某一侧手部设备掉线不会阻断双臂控制环

- [teleop_runtime/debug/run_vive_sapien_virtual.py](./debug/run_vive_sapien_virtual.py)

建议启动顺序：

1. 在 `dexteleop310` 环境里启动 VIVE 发送端
2. 再启动虚拟跑法

VIVE 发送端：

```bash
python hardware/VIVE/mocap/vive_data_sender.py \
  --host 127.0.0.1 \
  --port 9999 \
  --hz 300
```

SAPIEN 虚拟跑法：

```bash
python -m teleop_runtime.debug.run_vive_sapien_virtual \
  --host 0.0.0.0 \
  --port 9999 \
  --arm-mode real \
  --left-arm-ip 192.168.101.21 \
  --right-arm-ip 192.168.1.20 \
  --left-hand-port /dev/serial/by-id/your_left_hand \
  --right-hand-port /dev/serial/by-id/your_right_hand \
  --control-mode absolute
```

如果已经有 profile，推荐直接带上：

```bash
python -m teleop_runtime.debug.run_vive_sapien_virtual \
  --host 0.0.0.0 \
  --port 9999 \
  --arm-mode virtual \
  --control-mode absolute \
  --load-profile teleop_records/calibration_profiles/your_profile.json
```

说明：

- 这个入口使用“之前标注结果”的双臂相对关系预设，当前默认 `left_right_base_distance = 0.525`
- 默认标定预设来自 `build_annotated_dual_rm65_calibration(...)`
- 默认先等待你按 `p` 或回车，再开始 runtime home calibration
- `arm-mode=real` 下会在开始 home capture 前读取真机当前关节状态，并把它作为本地 preview 的起始姿态；整个入口不会发送任何真机动作指令
- `--load-initial-alignment` 会在你按下 `p` 时提供机器人侧的 home pose/qpos；如果没有它，而又连了真机，则 `p` 会同时采当前真机 arm home；如果既没有它也没连真机，则 `p` 用当前 preview arm pose 作为 robot home
- 默认会启用 joint-space 双臂碰撞规避，scene profile 默认 `dual_arm_hand_proxy_open`；它使用明确指定的 12 对碰撞检测集合，覆盖固定张开手/arm-link3 proxy 与中央/左右 camera pole、table 的关键组合
- 默认碰撞模式是 `boundary_slide`；如果要在接触阈值附近直接保持最近安全关节位姿，可改用 `--collision-avoidance-mode freeze_on_contact`
- 如果想直接退回旧的独立双臂链路，可以加 `--no-collision`；旧参数 `--disable-dual-arm-collision-avoidance` 仍兼容
- 当前预留的 scene profile 顺序是：`dual_arm` -> `dual_arm_hand_proxy_open` -> `dual_arm_hand` -> `dual_arm_cylinder_table`
- runtime home calibration 只负责 home 锚点，不负责完整 desk/world 对齐
- 启动时会打印 `expecting ...` 和 `observed ...`，这是当前最直接的 tracker 绑定验收信息
- 当前 action 主 frame 是 `tool_tcp`；`--tcp-mode hand_base` 只作为历史兼容参数保留
- 如果加了 `--left-hand-port` / `--right-hand-port`，对应手部 `q6` 会实时映射成 SAPIEN 手部关节；不传这些参数时，行为和之前完全一致
- 手部默认 `--hand-joint-order` 是 `5,4,3,2,0,1`，也就是把 reader 的 `[thumb_yaw, thumb_flex, index, middle, ring, pinky]` 重排成 InspireTac 顺序 `[pinky, ring, middle, index, thumb_yaw, thumb_flex]`
- 手部输入默认是**附加功能**：手部输入异常不会触发 arm teleop 的 `FROZEN`
- 如果当前机器没有可用渲染设备，可以先用 `--skip-renderer` 验证 teleop 主链是否正常跑通
- joint-space 双臂碰撞规避依赖 `pinocchio + pink + qpsolvers`
- RM65 默认 impedance 遥操不再调用 SDK 奇异解检测；其独立的 IK、关节限位、力控与故障保护不变。普通位控遥操仍会在目标从正常区进入 SDK 判定奇异区时保持当前姿态。默认双臂 position replay 则先以原有效关节步幅的 0.2 生成慢速目标，并在同一 12 对碰撞复核通过后才下发；复核失败会 fault/hold，且该 replay 专用规则不影响 deploy。
- 更详细的控制模式说明见 [teleop_runtime/README_control_modes.md](./README_control_modes.md)
- 如果发现 `+X/+Y` 同时反了，先回 calibration tool 按 `f`，或者在标定入口直接加 `--virtual-desk-post-yaw-deg 180`

## RM65 真机只读检查

当前也补了一个只读真机状态的检查入口：

- [teleop_runtime/debug/rm65_real_debug/run_rm65_real_state_viz.py](./debug/rm65_real_debug/run_rm65_real_state_viz.py)

它不会发送任何动态指令，只会：

- 连接左/右 RM65
- 轮询 `Get_Current_Arm_State`
- 把真实关节状态映射回 SAPIEN
- 显示 `desk / arm_base / tcp_actual`

更详细的使用方式见：

- [teleop_runtime/debug/rm65_real_debug/README.md](./debug/rm65_real_debug/README.md)

## RM65 真机遥操作

现在也补了双臂真机遥操作入口，并支持**可选的双手附加控制**：

- [teleop_runtime/debug/run_vive_rm65_real_teleop.py](./debug/run_vive_rm65_real_teleop.py)

推荐启动方式：

```bash
python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  --left-arm-ip 192.168.101.21 \
  --right-arm-ip 192.168.1.20 \
  --left-hand-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130705-if00 \
  --right-hand-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130707-if00 \
  --left-hand-ip 192.168.12.210 \
  --right-hand-ip 192.168.11.210 \
  --load-profile teleop_records/calibration_profiles/your_profile.json
```

如果需要保留旧版位控链路，并让 `tool_tcp` 直接等于机械臂末端、跳过 payload/impedance
profile 依赖，可以显式加载位控专属 profile：

```bash
python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  --arm-control-mode position \
  --left-tool-profile position_endpoint \
  --right-tool-profile position_endpoint
```

如果你当前已经在用双臂 `50Hz`、较高 arm 角速度上限，并且想在此基础上打开 InspireTac 触觉读取检查频率是否合理，可以直接从下面这组参数开始：

```bash
python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  --left-arm-ip 192.168.101.21 \
  --right-arm-ip 192.168.1.20 \
  --load-profile teleop_records/calibration_profiles/your_profile.json \
  --hz 50 \
  --arm-max-angvel 1.2,1.2,1.2,1.8,1.8,1.8 \
  --left-hand-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130705-if00 \
  --right-hand-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130707-if00 \
  --left-hand-ip 192.168.12.210 \
  --right-hand-ip 192.168.11.210 \
  --hand-telemetry-hz 10 \
  --hand-tactile-regions all
```

`--hand-tactile-regions` 默认就是 `all`，所以上面这一行也可以省略；保留它可以让采集命令更显式。如果只是排查控制链路、需要临时关闭触觉读取：

```bash
--hand-tactile-regions ""
```

真实遥操作默认仅在状态变化、故障和生命周期事件时打印终端日志，避免常规状态刷屏掩盖真实问题。若需要周期性状态摘要，显式追加 `--status-sec 1`；若需要连续姿态调试，追加 `--debug-orientation-console`。

如果希望在开始前先验证当前真机起始关节和某份对齐状态是否足够接近：

```bash
python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  --left-arm-ip 192.168.101.21 \
  --right-arm-ip 192.168.1.20 \
  --load-profile teleop_records/calibration_profiles/your_profile.json \
  --load-initial-alignment teleop_records/initial_alignment_states/your_alignment.json
```

说明：

- `--load-initial-alignment` 在这个入口里只用于启动前的关节误差检查，不会替代 `p` 时刻的 live robot home capture
- 如果同时给了 `--left/right-hand-port` 和 `--left/right-hand-ip`，runtime 会把手部串口 `q6` 映射后下发到 InspireTac
- `--gui` 会显示 SAPIEN 真机预览窗口；`--no-gui`/`--headless` 会关闭窗口但保留遥操逻辑；`--skip-renderer` 则完全不创建 renderer
- 如果操作者左右 VIVE tracker 在开始前需要互换，使用 `--swap-operator-sides` 或在 `--gui` 控制面板点“交换 Tracker”。它只交换 VIVE tracker 到 logical left/right 的绑定，手部串口、真手、profile 和机械臂 IP 都不变；机械臂 IP 左右互换仍然使用 `--swap-arm-sides`
- 默认手部映射顺序仍是 `5,4,3,2,0,1`，对应 reader `q6=[thumb_yaw, thumb_flex, index, middle, ring, pinky] -> InspireTac q6=[pinky, ring, middle, index, thumb_yaw, thumb_flex]`
- 如果只给 `--left/right-hand-port`，则只做 hand preview，不会连接真手
- 如果只给 `--left/right-hand-ip`，则只读手部遥测并渲染当前真手状态，不会有 hand teleop 输入
- `--hand-telemetry-hz` 控制每只手后台 telemetry 线程的轮询频率；这个线程会读取 `joint_pos`，默认还会读取全 17 区域 tactile
- `--hand-tactile-regions` 默认是 `all`，等价于读取全部 17 个区域；传空字符串 `--hand-tactile-regions ""` 才会只做手部关节遥测、不读 tactile
- `--inspire-hand-max-force N` 控制启动时写入 InspireTac 的每指 force，默认 `800`；旧参数 `--max-fingertip-force` 仍可用，`--max-force` 是简写
- 目前每只手只保留一个 Modbus 连接，所以同一只手的 `set_joint_pos`、`read_joint_pos`、`read_tac_data` 会在该手内部串行，但左右手之间仍然并行
- 建议的频率排查顺序是：
  1. 默认采集使用 `--hand-tactile-regions all`，先把 `--hand-telemetry-hz` 设为 `10` 确认全区 tactile 链路稳定
  2. 如果频率和跟手性仍稳定，再逐步升到 `15` 或 `20`
  3. 只有在隔离排查控制链路延迟时，才临时传 `--hand-tactile-regions ""` 关闭 tactile
  4. 不建议把 `--hand-telemetry-hz` 直接拉到 `50`；那更适合作为压力测试，而不是默认工作配置
- 如果只是想单独看手部 teleop + tactile 对频率的影响，优先用 `teleop_runtime.debug.run_hand_q6_inspire_real_teleop` 做隔离测试；这样更容易把 arm 链路和 hand/tactile 链路分开看
- 手部控制链是附加子链：手部输入掉线或无效不会阻断 arm teleop；arm 的 `FROZEN -> RECOVERING` 逻辑保持原样
- `--swap-arm-sides` 仍然只影响 arm 逻辑侧映射；`--teleop-hardware-ip` / `--teleop-hardware-port` 继续保留但当前仍未启用
- `FROZEN -> RECOVERING` 逻辑和 episode 录制格式沿用虚拟入口，方便后续 replay / live teleop 统一
- impedance 模式默认会把 `--left/right-tool-profile` 中的 TCP 和 payload/CoM 写入或创建到 RealMan 控制器工具坐标，并自动 `Change_Tool_Frame`；可用 `--no-auto-sync-tool-frame` 临时关闭
- `--left/right-tool-profile position_endpoint` 是位控兼容 profile：TCP 固定在 RM65 机械臂末端，不要求 payload、重力补偿或 impedance 信息；双臂防碰撞 IK 仍会按这个 endpoint TCP 求解

当前支持的 tactile region 名称是：

```text
little_1, little_2, little_3,
ring_1, ring_2, ring_3,
middle_1, middle_2, middle_3,
index_1, index_2, index_3,
thumb_1, thumb_2, thumb_3, thumb_4,
palm
```

`delta_clutch` 已硬封存，active 入口不再接受该参数；历史行为仅保留在版本历史中。
- `t`: 开始一个 episode 轨迹录制
- `s`: 结束当前 episode 轨迹录制
- `q`: 退出

## Session 状态机

主要状态：

- `CALIBRATION`
- `DEBUG_ONLY`
- `ARMED`
- `RUNNING`
- `FROZEN`
- `FAULT`

典型顺序：

1. `session.start()`
2. `session.set_calibration_ready()`
3. `session.mark_debug_passed()`
4. `session.arm_hardware()`
5. 循环调用 `session.run_step()`

## 离合和控制模式

`PoseToArmRetargeter` 当前支持：

- `absolute`
- `delta_clutch`（已硬封存，仅作历史说明）

默认是 `absolute`。当前 active runtime 不再接受 `delta_clutch`。

## 录制与安全

`run_vive_rm65_real_teleop.py` 默认不再写 legacy session JSONL；只有显式传入
`--record` 时才会生成：

- `teleop_records/<timestamp>/raw_inputs.jsonl`
- `teleop_records/<timestamp>/commands.jsonl`
- `teleop_records/<timestamp>/faults.jsonl`
- `teleop_records/<timestamp>/state_transitions.jsonl`
- `teleop_records/<timestamp>/calibration.json`

GUI 里的 IK debug 记录单独保存到：

- `teleop_records/ik_debug/ik_debug_<timestamp>.jsonl`

正式 dataset collection 会写入 `datasets/<date>_<task>/` 下的 LeRobot 数据集。轻量
debug trajectory 不会在正式 dataset 采集中默认重复保存；仅在 `--no-collection`、
`--episode-root` 或 `--debug-trajectory-recording` 时生成：

- `teleop_records/<timestamp>/trajectories/episode_000001/manifest.json`
- `teleop_records/<timestamp>/trajectories/episode_000001/frames.jsonl`
- `teleop_records/<timestamp>/trajectories/episode_000001/events.jsonl`

推荐的 replay 验收顺序：

1. 录一段 episode
2. `python -m teleop_runtime.debug.validate_recorded_episode /path/to/episode_000001`
3. `python -m teleop_runtime.debug.run_recorded_episode_replay /path/to/episode_000001 --arm-mode virtual`
4. 再切到 `--arm-mode real --left-arm-ip ... --right-arm-ip ...`

`--arm-mode real` replay 会在首帧前读取当前双臂真实关节，并和 episode 首帧 `qpos` 做误差检查；默认任一关节误差超过 `5 deg` 或加权误差超过 `12 deg` 会直接拒绝开播。

当前 safety monitor 会检查：

- 缺失 channel
- stale channel
- 无效 pose
- 平移跳变
- 旋转跳变
- 控制频率低于下限

默认策略是发现 fault 后整套 rig 进入 `FROZEN`。

## 已知限制

- 还没有真实键盘监听，`KeyboardButtonSource` 目前需要你主动 `set_button(...)`
- `run_vive_rm65_calibration_debug.py` 已可用，但当前还没有把 `W / V / D / target axes` 显式叠加画到 SAPIEN 里
- 双臂 joint-space 碰撞规避当前默认推荐 `dual_arm_hand_proxy_open`；它用固定张开手/arm-link3 proxy、中央与左右 camera pole、固定桌面 box 的 12 对碰撞检测集合维持 12DoF arm-only 求解。`dual_arm_hand` 仍保留给完整手模型实验路径，`dual_arm_cylinder_table` 仍是特定场景 profile
- 当前 `virtual desk` 标定不恢复物理桌面的绝对世界原点
- wrist 外参目前是“neutral pose 粗标 + 人工细调”，不是完全自动辨识；当前 coarse 默认 canonical hand frame 为 `[0, 0, 0]`，neutral hold 直接定义 tracker-to-hand 粗标外参

## 建议的当前工作流

1. 先跑 `hardware/VIVE/mocap/vive_data_sender.py --hz 300`
2. 跑 `run_vive_rm65_calibration_debug.py` 完成 `LINK_CHECK -> virtual desk -> wrist offset -> save profile`
3. 启动 `run_vive_sapien_virtual.py`，读取 profile，并在当前真机/虚拟起始姿态上做 session `home capture`
4. 用 `control_mode="absolute"` 做虚拟验收，确认前/左/上和双臂相对关系
5. 启动 `run_vive_rm65_real_teleop.py` 接入真机，统一按 `tool_tcp` action 执行
6. 正式采集时再打开 `--gui`、相机和需要的 hand/gripper backend

如果下一步继续开发，建议优先补：

- SAPIEN workspace / TCP 调试可视化

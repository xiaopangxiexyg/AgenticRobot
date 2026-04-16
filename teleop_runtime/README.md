# Teleop Runtime

`teleop_runtime` 是新的通用遥操作运行时骨架。当前版本已经能把下面两条链路接到统一接口里：

- `VIVE tracker -> pose_to_arm -> RM65`
- `hand_q6 reader -> joints_to_dexhand -> Inspire Tac`

它的目标不是替代所有旧代码，而是把“输入源 / retargeter / backend / rig / session”拆开，后续可以继续接别的输入设备、别的机械臂、别的末端。

更完整的设计背景见 [teleop_runtime/blueprint.md](/home/zc/HandMotion/RealRobotExperiments/dex_inspire_teleop/teleop_runtime/blueprint.md)。

## 当前已实现

- 通用数据模型：`ObservationChannel`、`EffectorCommand`、`TeleopSession`
- 通用标定对象：`TeleopCalibration`
- `VivePoseSource`
- `HandQ6Source`
- `PoseToArmRetargeter`
- `JointsToDexHandRetargeter`
- `RM65ArmBackend`
- `InspireTacHandBackend`
- `BimanualRig`
- 交互式 calibration debug app：
  [teleop_runtime/debug/run_vive_rm65_calibration_debug.py](/home/zc/HandMotion/RealRobotExperiments/dex_inspire_teleop/teleop_runtime/debug/run_vive_rm65_calibration_debug.py)
- 默认录制：`teleop_records/<timestamp>/`

## 当前未实现

- 通用真机 CLI runner
- 基于固定物理工装的 `desk/world` 绝对外参向导
- runtime 内部的双臂耦合 IK backend
- 键盘事件采集线程

所以当前推荐用法是：

1. 先用 `connect_hardware=False` 做 dry-run
2. 先完成启动标定和虚拟验收，再接真硬件
3. 在你自己的入口脚本里驱动 `TeleopSession`

## 目录

- `teleop_runtime/sources`: 输入源
- `teleop_runtime/retargeters`: 输入到命令的映射
- `teleop_runtime/backends`: 具体硬件执行
- `teleop_runtime/rigs`: 多 backend 组合
- `teleop_runtime/session.py`: 主循环
- `teleop_runtime/calibration.py`: 标定和坐标变换

## 依赖

按链路分别准备：

- VIVE 发送端：`triad_openvr`
- RM65 arm backend：RM65 SDK，可导入 `hardware.rm65.robotic_arm_package.robotic_arm`
- Inspire hand backend：`hardware/inspire_tac/inspire_tac_modbus.py` 依赖的 `pymodbus`
- arm IK：`rm65_inspire_tac.ik.SingleArmIKSolver` 依赖 `pinocchio`
- 推荐环境：`conda activate grasp_py310_copy`

如果只做 dry-run，可以先把 arm / hand backend 的 `connect_hardware=False`。

## VIVE 发送端

新的发送端在 [hardware/VIVE/mocap/vive_data_sender.py](/home/zc/HandMotion/RealRobotExperiments/dex_inspire_teleop/hardware/VIVE/mocap/vive_data_sender.py)。

默认 tracker 映射：

- `tracker_1 -> left_wrist`
- `tracker_2 -> right_wrist`
- `tracker_3 -> auxiliary`

示例：

```bash
python hardware/VIVE/mocap/vive_data_sender.py \
  --host 192.168.20.90 \
  --port 9999 \
  --tracker tracker_1:left_wrist \
  --tracker tracker_2:right_wrist \
  --tracker tracker_3:auxiliary
```

发送协议是结构化 JSON UDP，`teleop_runtime` 也保留了 legacy float 包兼容。

## Calibration Tool

当前已经有一个可运行的交互式标定调试入口：

- [teleop_runtime/debug/run_vive_rm65_calibration_debug.py](/home/zc/HandMotion/RealRobotExperiments/dex_inspire_teleop/teleop_runtime/debug/run_vive_rm65_calibration_debug.py)

它当前实现的是：

- `auxiliary tracker -> virtual desk` 标定
- 左右 wrist tracker 外参粗标定
- session home capture
- SAPIEN dry-run 验证与 wrist 姿态细调

这套流程的详细动作、按键和质量门槛见：

- [teleop_runtime/CalibrationREADME.md](/home/zc/HandMotion/RealRobotExperiments/dex_inspire_teleop/teleop_runtime/CalibrationREADME.md)

## 标定流程建议

当前已经落地的主线不是“固定物理 desk 绝对外参恢复”，而是：

1. 用 `auxiliary tracker` 标定一个 `virtual desk`
2. 用左右 wrist tracker 标定 `tracker -> hand control frame` 外参
3. 每次 session 再做 `home anchor`

也就是先解决“人侧前/左/上如何解释”，再解决“tracker 安装姿态如何解释”，最后解决“本次从哪里开始跟”。

如果你需要完整动作说明，优先看：

- [teleop_runtime/CalibrationREADME.md](/home/zc/HandMotion/RealRobotExperiments/dex_inspire_teleop/teleop_runtime/CalibrationREADME.md)

### 1. 结构层

这部分仍然沿用当前 preset，不在这套标定工具里重标：

- `arms[*].T_D_B`
- `workspace_min/workspace_max`
- 双臂相对基距
- `tcp_mode` 对应的 TCP frame

当前入口默认使用：

- `build_annotated_dual_rm65_calibration(...)`
- `left_right_base_distance = 0.525`

如果机器人底座、桌面定义或 workspace 变了，这部分要单独重新确认。

### 2. 输入映射层

这部分解决：

- `tracker_name -> tracker_role`
- `tracker_offsets[*]`
- 左右 wrist 的安装方向是否一致

当前代码里，receiver 会优先按 `tracker_role` 绑定，不再优先依赖设备名顺序。

### 3. 会话层

这部分是每次启动都要重做的：

- `virtual desk` 方向拟合
- 左右 wrist coarse/fine 标定
- `home anchor`

这也是当前 [run_vive_rm65_calibration_debug.py](/home/zc/HandMotion/RealRobotExperiments/dex_inspire_teleop/teleop_runtime/debug/run_vive_rm65_calibration_debug.py) 真正负责的内容。

## 当前推荐的标定入口

完整标定入口：

```bash
python -m teleop_runtime.debug.run_vive_rm65_calibration_debug \
  --host 0.0.0.0 \
  --port 9999 \
  --tcp-mode rm_eef
```

这一步会按顺序完成：

1. `LINK_CHECK`
2. `VIRTUAL_DESK_X`
3. `VIRTUAL_DESK_Y`
4. `VIRTUAL_DESK_Z`
5. `VIRTUAL_DESK_NEUTRAL`
6. `LEFT_WRIST_COARSE`
7. `RIGHT_WRIST_COARSE`
8. `HOME_CAPTURE`
9. `VALIDATE`
10. `SAVE`

当前这套流程的核心是：

- `auxiliary tracker` 不需要固定在物理桌面上
- 通过一组扫动动作拟合出 `virtual desk`
- 机器人侧仍沿用现有 `desk -> base` 预设

## 标定动作的职责划分

### 1. Virtual Desk

这一步只解决人侧坐标系方向：

- `X sweep`: 操作者前方
- `Y sweep`: 操作者左方
- `Z sweep`: 操作者上方
- `neutral hold`: 当前 session 的虚拟工作中心

这里不追求恢复物理桌面的绝对世界原点，只追求：

- 前/左/上方向正确
- 每次基站移动后都能重新稳定定义同一类操作语义

### 2. Wrist Tracker Offset

这一步只解决左右 wrist tracker 的安装外参：

- coarse：通过 neutral canonical pose 求初值
- fine：在验证阶段用 `j/l/i/k/u/o` 做小步旋转细调

平移偏置目前仍然走 CLI 参数：

- `--left-tracker-translation`
- `--right-tracker-translation`

### 3. Home Anchor

这一步只解决本次 session 的起点：

1. 人工把双臂 move 到本次 teleop home
2. 双手摆到自然起始姿态
3. 采样人手 home 和机器人 TCP home
4. 建立本次 session 的映射锚点

这一步不是 desk/world 外参。

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
- 简单验证 `absolute / delta_clutch`
- 非完整标定场景下的 home-only 调试

它仍然有价值，但它不替代完整标定流程。

## 基站、tracker、机器人变化时该怎么处理

### 只移动了 VIVE 基站

推荐直接重跑完整 calibration app：

1. 重新做 `virtual desk`
2. 重新确认左右 wrist offset
3. 重新采 `home`

只做 `home capture` 往往能“看起来能动”，但方向解释未必还对。

### 只重新安装了 wrist tracker

至少重做：

1. `LINK_CHECK`
2. 对应侧 wrist coarse
3. `VALIDATE` 里的 fine tune
4. `HOME_CAPTURE`

### 机器人或桌面位置变化

这超出了当前 calibration app 的范围。需要重新确认：

- `arms[*].T_D_B`
- workspace box
- 双臂基距和 desk 定义

## 当前代码里标定字段的实际含义

`TeleopCalibration` 当前最关键的字段：

- `T_W_O`
- `T_O_D`
- `tracker_offsets[*]`
- `arms[*].T_D_B`

对于当前 virtual-desk 标定，保存时采用的是：

- `T_W_O = I`
- `T_O_D = T_W_V`

也就是把 `virtual desk` 直接写入 `world -> desk` 链路里。  
这让现有 retargeter 不需要改接口，就能直接把 `vive_world` 映射到当前 session 的 `virtual desk`。

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
        tcp_mode="hand_base",
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

注意当前 `build_bimanual_session(...)` 默认总会创建左右两侧的 `pose_to_arm` retargeter，所以如果只启一侧硬件，另一侧 VIVE 数据也最好同时存在，或者你自己写一个单侧 builder。

## VIVE + SAPIEN 虚拟跑法

当前已经补了一个真实 VIVE 输入、SAPIEN 双臂可视化、但不连真 RM65 的虚拟跑法入口：

- [teleop_runtime/debug/run_vive_sapien_virtual.py](/home/zc/HandMotion/RealRobotExperiments/dex_inspire_teleop/teleop_runtime/debug/run_vive_sapien_virtual.py)

建议启动顺序：

1. 在 `grasp_py310_copy` 环境里启动 VIVE 发送端
2. 再启动虚拟跑法

VIVE 发送端：

```bash
python hardware/VIVE/mocap/vive_data_sender.py \
  --host 127.0.0.1 \
  --port 9999 \
  --tracker tracker_1:left_wrist \
  --tracker tracker_2:right_wrist \
  --tracker tracker_3:auxiliary
```

SAPIEN 虚拟跑法：

```bash
python -m teleop_runtime.debug.run_vive_sapien_virtual \
  --host 0.0.0.0 \
  --port 9999 \
  --tcp-mode rm_eef \
  --control-mode absolute
```

说明：

- 这个入口使用“之前标注结果”的双臂相对关系预设，当前默认 `left_right_base_distance = 0.525`
- 默认标定预设来自 `build_annotated_dual_rm65_calibration(...)`
- 默认先等待你按 `p` 或回车，再开始 runtime home calibration
- runtime home calibration 只负责 home 锚点，不负责完整 desk/world 对齐
- 启动时会打印 `expecting ...` 和 `observed ...`，这是当前最直接的 tracker 绑定验收信息
- 当前 `rm_eef` 如果在 arm-only URDF 中没有同名 frame，会自动回退到可用 frame，不会直接崩
- 如果当前机器没有可用渲染设备，可以先用 `--skip-renderer` 验证 teleop 主链是否正常跑通
- 更详细的控制模式说明见 [teleop_runtime/README_control_modes.md](/home/zc/HandMotion/RealRobotExperiments/dex_inspire_teleop/teleop_runtime/README_control_modes.md)

如果你想用终端键盘做 delta clutch：

```bash
python -m teleop_runtime.debug.run_vive_sapien_virtual \
  --control-mode delta_clutch \
  --keyboard-clutch
```

键位：

- `z`: 左臂 clutch 开关
- `x`: 右臂 clutch 开关
- `c`: 左右同时切换
- `r`: 全部释放
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
- `delta_clutch`

默认是 `delta_clutch`。注意当前实现里，第一次进入 `delta_clutch` 模式时必须先按下 clutch 建立 anchor，否则 retargeter 会返回空命令，这是刻意的安全行为，用来避免启动瞬间跳变。

## 录制与安全

默认会生成：

- `teleop_records/<timestamp>/raw_inputs.jsonl`
- `teleop_records/<timestamp>/commands.jsonl`
- `teleop_records/<timestamp>/faults.jsonl`
- `teleop_records/<timestamp>/state_transitions.jsonl`
- `teleop_records/<timestamp>/calibration.json`

当前 safety monitor 会检查：

- 缺失 channel
- stale channel
- 无效 pose
- 平移跳变
- 旋转跳变
- 控制频率低于下限

默认策略是发现 fault 后整套 rig 进入 `FROZEN`。

## 已知限制

- 还没有现成的 `run_bimanual_session.py`
- 还没有真实键盘监听，`KeyboardButtonSource` 目前需要你主动 `set_button(...)`
- `run_vive_rm65_calibration_debug.py` 已可用，但当前还没有把 `W / V / D / target axes` 显式叠加画到 SAPIEN 里
- 当前双臂路径是“两个单臂 backend 并行”，不是统一多任务 IK
- 当前 `virtual desk` 标定不恢复物理桌面的绝对世界原点
- wrist 外参目前是“neutral pose 粗标 + 人工细调”，不是完全自动辨识

## 建议的当前工作流

1. 先跑 `hardware/VIVE/mocap/vive_data_sender.py`
2. 跑 `run_vive_rm65_calibration_debug.py` 完成 `LINK_CHECK -> virtual desk -> wrist offset -> home -> validate`
3. 保存 calibration profile
4. 先用 `control_mode="absolute"` 做虚拟验收，确认前/左/上和双臂相对关系
5. 再切到 `control_mode="delta_clutch"` 验证真实遥操作手感
6. 接入单臂 RM65，先用 `tcp_mode="rm_eef"`
7. 再切到 `tcp_mode="hand_base"`
8. 最后再接 `hand_q6 -> InspireTac`

如果下一步继续开发，建议优先补：

- `run_bimanual_session.py`
- SAPIEN workspace / TCP 调试可视化
- calibration profile 加载到 `run_vive_sapien_virtual.py` 的统一 CLI

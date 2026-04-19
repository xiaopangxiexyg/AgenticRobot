# rm65_inspire_tac_portable

`teleop/rm65_inspire_tac` 里的主线功能很多，但长期混在一起了：重定向、单臂 IK、双臂多任务 IK、可视化、硬件控制、数据采集、实验脚本都堆在同一层。这个 portable 版的目标不是逐文件兼容，而是把真正有复用价值的能力重新整理成一套更清晰、可组合、易复用的包。

## 这版保留了什么

- `VisionPro / OpenTeleVision / 自定义输入 -> RM65 + Inspire Tac` 的手部重定向主链路
- 基于 Pinocchio 的单臂增量 IK
- 基于 Pink 的双臂多任务 IK 与碰撞场景 profile
- SAPIEN 双臂+灵巧手可视化
- RM65 + Inspire Tac 的便携硬件控制封装

## 这版刻意没保留什么

- 原目录里按任务散落的 `run_hardware_teleop*.py`
- 相机采集、episode 录制、视频落盘这类数据采集逻辑
- `train_mlp_classification.py`、`visualize_data_tacile.py` 这类下游分析脚本
- 各种历史 debug / collision / multitask 变体脚本的逐份副本

这些内容并不是没价值，而是不适合作为 portable 主入口。这里把它们背后的“核心能力”抽出来重新组织了。

## 新目录结构

```text
teleop/rm65_inspire_tac_portable/
├── __init__.py
├── paths.py
├── schemas.py
├── retargeting.py
├── ik.py
├── multitask_ik.py
├── pipeline.py
├── rendering.py
├── hardware.py
├── examples/
│   └── offline_retarget_demo.py
└── README.md
```

## 原始代码到 portable 模块的映射

| 原始来源 | portable 模块 | 归纳后的职责 |
| --- | --- | --- |
| `realtime_retarget.py` | `retargeting.py` | 单/双手重定向、手部关节映射、腕部位姿归一化 |
| `teleop/pin_robot.py` | `ik.py` | 单臂增量 IK，去掉了对 `torch/cuda` 的硬依赖 |
| `multi_task_IK/multi_task_ik.py` | `multitask_ik.py` | 多任务 IK 求解器，去掉硬编码路径 |
| `realtime_retarget_multitask_dual_arm*.py` | `pipeline.py` | 双臂场景 profile、场景化碰撞配置、结果拆分 |
| `render_sapien.py` | `rendering.py` | SAPIEN 机械臂+灵巧手渲染器 |
| `hardware/rm65/rm65_inspire_tac_controller.py` | `hardware.py` | 单臂/双臂硬件控制封装，补了更干净的 `close()` |

## 核心对象

### 1. `HandMotionInput` / `TeleopFrame`

这是 portable 版最重要的输入抽象。它把“动作源”从原始 AVP/VisionPro 脚本里剥离出来了。

- `HandMotionInput`: 一只手的输入
- `TeleopFrame`: 一帧双手输入

最常见的两种用法：

1. 直接喂原始 AVP 数据字典，让 `RM65InspireTacRetargeter` 自动解析
2. 自己构造 `TeleopFrame`，把别的 mocap 源统一接进来

### 2. `RM65InspireTacRetargeter`

负责把上游动作输入变成：

- 机械臂腕部目标位姿 `wrist_pose_robot`
- 18 维 Inspire Tac 手部关节 `retargeted_hand_qpos`
- 6 维硬件手指控制量 `hardware_hand_qpos`

支持三种动作源：

- `VisionProSource`
- `OfflineVisionProSource`
- `OpenTeleVisionSource`

也支持外部直接传 `TeleopFrame`。

### 3. `IndependentArmSimulator`

这是原始 `HardwareSimulator` 的整理版：

- 左右臂分开做单臂 IK
- 输入是 `BimanualRetargetingResult`
- 输出是 `BimanualExecutionResult`

适合：

- 单臂调试
- 快速验证重定向结果
- 不想引入双臂耦合碰撞约束时

### 4. `DualArmMultiTaskSimulator`

这是原始多任务 IK 变体的统一入口。内置三个 profile：

- `dual_arm`
- `dual_arm_hand_proxy_open`
- `dual_arm_hand`
- `dual_arm_cylinder_table`

适合双臂协同和带碰撞约束的场景。

### 5. `RM65InspireTacController` / `BimanualRM65InspireTacSystem`

这是硬件层封装：

- `RM65InspireTacController`: 单臂 RM65 + Inspire Tac
- `BimanualRM65InspireTacSystem`: 左右手组合控制

它接收 `BimanualExecutionResult`，直接下发到硬件。

## 依赖说明

### 基础依赖

- `numpy`
- `scipy`

### 重定向依赖

- `dex_retargeting`
- `avp_stream` 或 `teleop.TeleVision` 对应的运行环境

### 单臂 IK

- `pinocchio`

### 双臂多任务 IK

- `pink`
- `qpsolvers`
- 可选：`loop-rate-limiters`

### 渲染

- `sapien`

### 硬件控制

- RM65 官方 SDK
- `pymodbus`

portable 包本身做了尽量多的“延迟导入”。不使用某部分功能时，不会强制拉起对应依赖。

## 快速开始

### 1. 只做离线重定向

```bash
python -m teleop.rm65_inspire_tac_portable.examples.offline_retarget_demo \
    /path/to/offline_avp_stream.pkl \
    --mode both \
    --steps 10
```

### 2. 从离线 AVP 走到单臂/双臂仿真

```python
from teleop.rm65_inspire_tac_portable import (
    OfflineVisionProSource,
    RM65InspireTacRetargeter,
    IndependentArmSimulator,
)

source = OfflineVisionProSource("/path/to/offline_avp_stream.pkl")
retargeter = RM65InspireTacRetargeter(mode="both", source=source)
simulator = IndependentArmSimulator()

teleop = retargeter.retarget(include_keypoints=False)
execution = simulator.step(teleop)

print(execution.left.arm_qpos.shape)   # (6,)
print(execution.right.arm_qpos.shape)  # (6,)
```

### 3. 使用双臂多任务 IK

```python
from teleop.rm65_inspire_tac_portable import (
    OfflineVisionProSource,
    RM65InspireTacRetargeter,
    DualArmMultiTaskSimulator,
)

retargeter = RM65InspireTacRetargeter(
    mode="both",
    source=OfflineVisionProSource("/path/to/offline_avp_stream.pkl"),
)
simulator = DualArmMultiTaskSimulator(scene_profile="dual_arm_hand")

teleop = retargeter.retarget()
execution = simulator.step(teleop)
```

### 4. 接自定义 mocap / tracker 输入

如果你不是用 AVP，而是自己已经有：

- `4x4` 腕部位姿
- `21x3` 或 `25x3` 手部关键点

可以直接构造 `TeleopFrame`：

```python
import numpy as np

from teleop.rm65_inspire_tac_portable import (
    HandMotionInput,
    RM65InspireTacRetargeter,
    TeleopFrame,
)

retargeter = RM65InspireTacRetargeter(mode="right")

right_input = HandMotionInput(
    wrist_pose=np.eye(4),
    hand_keypoints=np.zeros((21, 3)),
    wrist_in_operator_frame=True,
    hand_keypoints_in_operator_frame=False,
)

frame = TeleopFrame(right=right_input)
result = retargeter.retarget_frame(frame)
```

说明：

- `wrist_in_operator_frame=True` 时，会套用原始 AVP 版本的手腕坐标系旋转
- `hand_keypoints_in_operator_frame=False` 是默认值，表示关键点已经在适合 retarget 的手部坐标系里
- 如果你传的是原始手部坐标系关键点，可以改成 `True`

### 5. 下发到真实硬件

```python
from teleop.rm65_inspire_tac_portable import (
    ArmHardwareConfig,
    BimanualRM65InspireTacSystem,
    DualArmMultiTaskSimulator,
    OfflineVisionProSource,
    RM65InspireTacRetargeter,
)

retargeter = RM65InspireTacRetargeter(
    mode="both",
    source=OfflineVisionProSource("/path/to/offline_avp_stream.pkl"),
)
simulator = DualArmMultiTaskSimulator(scene_profile="dual_arm")

hardware = BimanualRM65InspireTacSystem.from_configs(
    left=ArmHardwareConfig(arm_ip="192.168.101.21", hand_ip="192.168.11.218"),
    right=ArmHardwareConfig(arm_ip="192.168.1.20", hand_ip="192.168.11.219"),
)

teleop = retargeter.retarget()
execution = simulator.step(teleop)
hardware.send_execution(execution)
hardware.close()
```

## 推荐工作流

1. 先用 `OfflineVisionProSource` 跑通 `retargeting.py`
2. 再接 `IndependentArmSimulator`，确认腕部位姿和关节映射没问题
3. 需要双臂碰撞约束时，再换 `DualArmMultiTaskSimulator`
4. 最后再接 `hardware.py`

这样比原始目录里直接跑大一统脚本更容易定位问题。

## 与原版相比的主要改动

- 去掉了 `sys.path.append(...)` 这类路径注入
- 去掉了 `PathConfig` 依赖，统一改成相对仓库根路径解析
- 单臂 IK 改成纯 `pinocchio + numpy` 路线，不再默认依赖 `torch/cuda`
- 双臂 profile 做成显式配置，不再靠多个高度重复的脚本维护
- 硬件控制补了更干净的资源释放接口，避免直接 `sys.exit(0)`
- 输入层抽象成 `TeleopFrame`，后续接 VIVE / VD / 自定义 mocap 会轻很多

## 当前边界

- 这个 portable 包不负责相机采集与 episode 数据落盘
- 不负责策略训练和触觉数据后处理
- 不自动做不同 mocap 系统之间的坐标系标定，`TeleopFrame` 只是提供统一入口

如果后面你要把 `VD_with_VIVE_tracker` 也并进来，最自然的方式是给它补一个 adapter，把输出整理成 `TeleopFrame`，而不是再起一套新的 `realtime_retarget_*.py`

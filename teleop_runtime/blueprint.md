# 通用遥操作模块 Blueprint

本文档把当前仓库重构成一套**通用、易用、可组合**遥操作模块的实施蓝图。

目标是构建一条统一主线：

- 用 `VIVE` 提供机械臂腕部/末端位姿输入
- 用 `minimal_hand_teleop` / `teleop_hand` 提供灵巧手 `q6` 输入
- 用 `rm65_inspire_tac` 提供 RM65 + Inspire Tac 的 IK 与硬件执行
- 把三者组织成一套清晰的、显式标定、可扩展到单臂/双臂的运行时系统

不要求前向兼容。允许重构现有目录、类名和入口，只保留真正值得复用的底层能力。
再次强调：**当前已有代码仅供参考，不是必须迁就的约束。** 如果某个现有脚本、目录结构、接口命名会妨碍最终系统的清晰性、调试性或安全性，应直接重构。

注意：本文档是历史实施蓝图，不代表当前 frame 语义。当前已落地实现以
`tool_tcp` 作为 action 主 frame；`rm_eef` 会归一为 `tool_tcp`，`hand_base` /
`arm_endpoint` 仅作为兼容、调试或后端内部 frame。
本文把 SAPIEN 写为优先 3D 调试后端的段落同样属于历史方案，已被当前决策取代：SAPIEN
只冻结保留旧 preview，不再新增功能或调用方；新的仿真/可视化工作使用 MuJoCo、Web UI
或 backend-neutral observer。

## 0. 本轮已确认的设计决策

这一节记录当前已经明确、不再作为开放问题的约束。后续实现以这里为准。

- 第一可用版本必须支持：`左右臂 + 左右手`
- VIVE 默认使用左右手腕各一个 tracker，最高支持 `3` 个 tracker
- 第 3 个 tracker 是可选辅助 tracker，可临时拆装，用于 `T_W^O` 或其他标定用途
- VIVE 发送端协议可以重写，但 `triad_openvr` 包本身不改
- arm 控制 TCP 支持两种：
  - `rm_eef`：用于单独控制机械臂和坐标调试
  - `hand_base`：用于更精细的末端控制
- arm 默认要求完整 `6DoF`
- 控制模式主路径：`absolute`；`delta_clutch` 已硬封存
- 默认控制模式：`absolute`
- 每次启动都必须重新做 runtime calibration
- 机械臂 home 由人工移动到位后采集，程序不自动规划 home
- 异常后的默认行为：优先冻结到最后一个已确认安全的位置
- hand 第一版只要求：`q6 -> Inspire`
- hand 输入层保留 `keypoints3d` 能力接口，但不是第一版必做项
- 灵巧手后续优先支持 dexterous hand 类末端；两指夹爪是下一优先级
- 调试可接受多窗口
- 3D 调试后端优先 `SAPIEN`
- 数据录制做成可选项，但默认开启
- 控制频率约束：
  - 最低可接受：`15 Hz`
  - 目标：`30 Hz`
  - 开启 Inspire 触觉读取后，触觉链路不能阻塞主控制环
- 当前没有现成 workspace，需要通过 `SAPIEN + 安全范围立方体` 调试可视化来交互确定
- 网络拓扑固定：
  - 一台机器最多控制一套双臂双手系统
  - RM65 的 IP 固定
  - Inspire 手 IP 允许调整
- clutch 第一版输入源：键盘
- 任何关键部件部分掉线或部分失效，整个 rig 进入 `FROZEN`

## 1. 核心设计原则

### 1.1 共享参考系优先

要保证双臂遥操作中双手的相对位置能够正确映射到机械臂上，核心原则仍然是：

**绝对不能让左右臂独立进行“手-眼”映射，必须通过共享参考系作为桥梁。**

这个原则也适用于单臂。单臂时虽然没有“左右相对位姿保持”的要求，但仍然应该经过统一的 `Operator Frame` 和 `Desk Frame`，否则系统会很快退化成不可维护的坐标系补丁集合。

### 1.2 输入层和执行层彻底解耦

新的主模块必须把以下概念拆开：

- 原始设备输入：VIVE UDP、手套串口、键盘/脚踏板
- 标定状态：`T_W^O`、`T_D^B`、`T_tracker^H`
- 任务映射：绝对位姿映射、离合增量映射、姿态对齐策略
- 执行后端：单臂 IK、双臂 IK、真实硬件、仿真

任何一个输入源都不应该直接知道机器人 IP、URDF、Modbus 细节。
任何一个机器人执行器也不应该直接去读 VIVE UDP 或串口。

### 1.3 机械臂和灵巧手是两条独立子链

这里的目标不是“把手部重定向也强塞进 arm pipeline”。
更合理的做法是：

- arm 子链：`VIVE -> pose calibration -> arm target -> IK / hardware`
- hand 子链：`hand_q6 reader -> q6 mapping -> Inspire hand hardware`

然后在运行时把两条子链组合到同一个 session 中。

这样有几个直接好处：

- 单独调 arm 不需要依赖手套
- 单独调 hand 不需要启动 VIVE
- arm 和 hand 的频率、滤波、离合、安全策略可以分开设计
- 后续把 hand 输入从 `minimal_hand_teleop` 换成视觉手部估计时，arm 子链几乎不受影响

但要补一个当前项目的具体约束：

- 第一版最终运行态虽然是 `bimanual arm + bimanual hand`
- 但 arm 子链和 hand 子链仍然要能独立启动、独立调试
- 这不是为了长期只跑单臂，而是为了把调试、验收和故障定位拆开

### 1.4 标定状态必须显式化

当前仓库里很多坐标变换都藏在某个 adapter 或脚本内部。新模块必须把标定变成显式状态对象，而不是“某个工具类里偷偷乘了个旋转矩阵”。

至少要显式保存：

- `T_W^O`: VIVE 世界系到操作者参考系的运行时标定
- `T_D^B_left`, `T_D^B_right`: 机械臂基座在桌面参考系下的固定外参
- `T_tracker^H_left`, `T_tracker^H_right`: Tracker 到手腕/末端控制点的固定偏置
- 机器人工作空间、姿态偏置、缩放比例、离合模式

### 1.5 单臂是一等公民，双臂是自然扩展

当前 `blueprint` 的核心原理来自双臂，而当前**第一可用版本目标**也是：

- `VIVE -> RM65` 的左右双臂 arm teleop
- `q6 -> Inspire` 的左右双手 hand teleop

实现过程允许分阶段逐步打通，但对外的第一可用里程碑必须是双臂双手完整支持。

因此新架构必须默认支持以下三种模式：

- `right_only`
- `left_only`
- `bimanual`

而不是只为双臂写一套过度耦合的大脚本。

### 1.6 调试优先，安全优先

对于这类真机遥操作系统，第一优先级不是“尽快动起来”，而是：

- 坐标系能被清楚观测和校正
- 错误输入能被及时识别和阻断
- 设备掉线或异常时系统行为可预测
- 每一步从仿真到真机都有明确准入门槛

因此新系统必须把下面两个能力作为一等公民，而不是开发时临时加的辅助脚本：

- `VIVE tracker -> operator frame -> desk frame -> robot eef` 的可视化调试链路
- 面向掉线、畸形数据、IK 失败、硬件异常的鲁棒状态机

### 1.7 面向能力抽象，而不是面向具体设备写死

当前目标虽然是：

- `VIVE -> RM65 arm`
- `hand_q6 -> Inspire hand`

但架构不能围绕这组组合硬编码，否则后续接入：

- 非 VIVE 的 6DoF tracker / 视觉 pose 估计 / 外骨骼 FK
- 非 Inspire 的灵巧手 / 两指夹爪 / 其他末端执行器

时，系统又会退化成“每来一类设备就新开一套 teleop 脚本”。

因此整个 runtime 必须围绕**能力接口**组织，而不是围绕设备名组织：

- `Pose6DSource`: 提供一个或多个 6DoF pose 流
- `JointVectorSource`: 提供关节向量、q6、qN 或标量控制流
- `Retargeter`: 把输入能力转换成末端能力
- `EffectorBackend`: 接收通用 command，下发到具体机器人/灵巧手/夹爪
- `Rig`: 把多个 effector 组合成一个完整系统

结论：

- `VIVE` 只是第一种 `Pose6DSource`
- `teleop_hand/hand_q6_reader.py` 只是第一种 `JointVectorSource`
- `rm65` 只是第一种 arm backend
- `InspireTac` 只是第一种 dexterous-hand backend

不过当前已知的未来优先扩展方向要明确收敛为：

- hand 输入侧保留 `keypoints3d`
- 末端侧下一优先级是两指夹爪

其他 pose source 扩展能力要保留，但不应反过来干扰第一版的 VIVE 主线落地。

## 2. 现有仓库现状与问题

### 2.1 现有可复用能力

当前仓库已经有三块非常有价值的底层能力：

1. `teleop_runtime/sources/pose/vive.py`
   提供当前 VIVE JSON UDP 主链路的最新帧接收逻辑。

2. `teleop_hand/hand_q6_reader.py`
   已经提供了单手/双手 `q6` 读取、自动零位标定和归一化输出。

3. `rm65_inspire_tac`
   已经提供了：
   - `ik.py`: 单臂 IK
   - `multitask_ik.py`: 双臂多任务 IK
   - `hardware.py`: RM65 + Inspire Tac 硬件下发
   - `pipeline.py`: 双臂执行编排
   - `rendering.py`: SAPIEN 可视化渲染

这些模块都应该尽量复用。

### 2.2 当前不适合作为最终架构的部分

#### VIVE 链路的问题

- `VIVE/utils.py` 里的旧 float receiver wrapper 已不再作为 VIVE 主链路使用
- 当前唯一 sender 是 `hardware/VIVE/mocap/vive_data_sender.py`，默认发送到 `9999`
- 没有 side、tracker 名称、丢包超时、有效性检查等概念
- 没有显式区分 `W`、`O`、`D`、`B` 坐标系

因此它现在更像 demo，而不是可以长期维护的输入适配器。

#### hand teleop 链路的问题

- `minimal_hand_teleop/inspire_hand_teleop.py` 是一个直接读输入并直接发 Modbus 的脚本
- 它的 import 还写成了 `teleop_hardware.hand_q6_reader`，与当前仓库真实目录 `teleop_hand/hand_q6_reader.py` 不一致
- 映射参数是 CLI 逻辑，不是可复用的 runtime component

因此它适合作为“最小可跑样例”，但不适合继续承担系统主入口。

#### rm65_inspire_tac 链路的问题

- `retargeting.py` 的核心是 AVP / VisionPro 手部关键点重定向，不是 VIVE arm teleop 的主线
- `pipeline.py` 当前以 `BimanualRetargetingResult` 为中心，仍然偏向“已有重定向结果 -> 执行”
- 缺少统一 runtime orchestration，无法把 arm 子链和 hand 子链优雅组合

结论：`rm65_inspire_tac` 应该保留为机器人执行后端，而不是继续承担整个 teleop 系统的顶层架构。

## 3. 目标架构

建议新增一个顶层包，例如：

```text
teleop_runtime/
├── __init__.py
├── registry.py
├── schemas.py
├── transforms.py
├── calibration.py
├── clutch.py
├── safety.py
├── diagnostics.py
├── recording.py
├── session.py
├── interfaces/
│   ├── __init__.py
│   ├── source.py
│   ├── retargeter.py
│   ├── backend.py
│   └── rig.py
├── sources/
│   ├── pose/
│   │   ├── __init__.py
│   │   ├── vive.py
│   │   ├── vision_pose.py
│   │   └── exo_fk.py
│   ├── hand/
│   │   ├── __init__.py
│   │   ├── q6_serial.py
│   │   ├── keypoints.py
│   │   └── glove.py
│   └── misc/
│       ├── __init__.py
│       └── keyboard.py
├── retargeters/
│   ├── __init__.py
│   ├── pose_to_arm.py
│   ├── joints_to_dexhand.py
│   ├── keypoints_to_dexhand.py
│   └── scalar_to_gripper.py
├── backends/
│   ├── arms/
│   │   ├── __init__.py
│   │   └── rm65.py
│   ├── hands/
│   │   ├── __init__.py
│   │   ├── inspire_tac.py
│   │   └── allegro.py
│   └── grippers/
│       ├── __init__.py
│       └── parallel_gripper.py
├── rigs/
│   ├── __init__.py
│   ├── arm_only.py
│   ├── hand_only.py
│   ├── arm_hand.py
│   └── bimanual.py
├── configs/
│   ├── __init__.py
│   ├── source_profiles.py
│   ├── embodiment_profiles.py
│   └── teleop_profiles.py
├── debug/
│   ├── __init__.py
│   ├── vive_frames.py
│   ├── tracker_to_eef_debug.py
│   ├── sim_overlay.py
│   └── workspace_tuner.py
└── apps/
    ├── run_vive_rm65_calibration_debug.py
    ├── run_vive_rm65_arm.py
    ├── run_hand_inspire.py
    └── run_rm65_full_teleop.py
```

设计意图：

- `teleop_runtime` 负责系统编排和通用抽象
- `interfaces/` 定义稳定契约，防止后续每加一种设备就污染主循环
- `registry.py` 负责按 profile 组装 source / retargeter / backend / rig
- `sources/` 只负责采数据，不知道机器人细节
- `retargeters/` 只负责能力映射，不直接碰硬件
- `backends/` 只负责下发，不直接读传感器
- `rigs/` 负责把多个 backend 组合成系统
- `recording.py` 负责会话录制与离线重放输入，默认开启但可关闭
- `rm65_inspire_tac` 保留为第一批可复用实现来源，但不再作为最终架构中心
- `VIVE` 和 `teleop_hand` 保留低层驱动能力，但不再负责系统级编排
- `minimal_hand_teleop` 退化成 `apps/` 的 thin wrapper 或历史兼容脚本
- `debug/` 目录承载正式调试模式，而不是临时 notebook / demo 脚本

## 4. 统一数据模型

新的模块不要继续让每条链路发自己随意拼装的 `dict`。内部核心数据模型应优先采用**通用 channel / effector schema**，而不是把 `VIVE wrist` 和 `Inspire q6` 写死在顶层结构里。

### 4.1 原始观测层

```python
@dataclass
class ObservationChannel:
    channel_name: str            # "right_wrist_pose", "left_hand_q6", "clutch_button"
    capability: str              # "pose6d" / "joint_vector" / "keypoints3d" / "scalar" / "bool"
    side: str | None
    value: Any
    frame_name: str | None
    timestamp: float
    source_name: str
    valid: bool = True
    critical: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 4.2 标定后的统一观测层

```python
@dataclass
class TeleopObservation:
    channels: dict[str, ObservationChannel] = field(default_factory=dict)
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

说明：

- 对当前第一版来说，最常见的 channel 会是：
  - `right_wrist_pose_operator`
  - `left_wrist_pose_operator`
  - `right_hand_q6`
  - `left_hand_q6`
- 但后续也可以自然扩展为：
  - `right_hand_keypoints`
  - `right_gripper_scalar`
  - `clutch_button`
  - `mode_switch`

### 4.3 任务目标层

```python
@dataclass
class EffectorCommand:
    effector_name: str            # "right_arm", "right_hand", "right_gripper"
    side: str | None
    command_type: str             # "cartesian_pose" / "joint_vector" / "scalar"
    value: Any
    frame_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class TeleopCommand:
    commands: dict[str, EffectorCommand] = field(default_factory=dict)
    timestamp: float | None = None
```

### 4.4 执行结果层

这里可以尽量复用 `rm65_inspire_tac/schemas.py` 里的结果结构，但 runtime 核心不应该只围绕 AVP retargeting 或 RM65/Inspire 命名。

为降低实现难度，可以允许：

- runtime 内部使用通用 `ObservationChannel` / `EffectorCommand`
- `rm65_inspire_tac` 这类 backend 自己维护一层 typed adapter，把通用 command 转成当前 backend 所需结构

此外需要显式区分：

- `critical control channels`：掉线会触发整个 rig `FROZEN`
- `aux channels`：如触觉遥测、调试统计、可选监控，不应阻塞主控制环

## 5. 标定模型

### 5.1 静态物理空间标定

这是 `Setup Once`，写入配置文件：

- 定义桌面坐标系 `D`
- 写入 `T_D^B_left` / `T_D^B_right`
- 写入工作空间包围盒
- 写入 arm eef frame、姿态修正矩阵、默认 home pose
- 写入 hand/gripper backend 的基础参数

建议落到配置文件：

```yaml
embodiment_profile: rm65_plus_inspire_bimanual
desk_frame:
  name: desk
  axes: {x: forward, y: left, z: up}
effectors:
  left_arm:
    type: cartesian_arm
    T_D_B: [[...], [...], [...], [...]]
    workspace_min: [x, y, z]
    workspace_max: [x, y, z]
    home_qpos: runtime_captured
    tcp_debug_frame: rm_eef
    tcp_fine_frame: hand_base
  right_arm:
    type: cartesian_arm
    T_D_B: [[...], [...], [...], [...]]
    workspace_min: [x, y, z]
    workspace_max: [x, y, z]
    home_qpos: runtime_captured
    tcp_debug_frame: rm_eef
    tcp_fine_frame: hand_base
  left_hand:
    type: dexterous_hand
    backend: inspire_tac
    open_command: [...]
  right_hand:
    type: dexterous_hand
    backend: inspire_tac
    open_command: [...]
```

### 5.2 运行时操作者标定

每次启动都执行，且这是强制流程。

推荐启动顺序：

1. 人工把左右机械臂移动到本次 session 的期望 home
2. 程序读取当前 arm qpos，保存为本次 session 的 `home_qpos`
3. 记录操作者中心参考系 `T_W^O`
4. 记录 tracker 到控制点偏置 `T_tracker^H`
5. 初始化 clutch anchor 状态

VIVE 标定流程必须同时支持：

- `2 tracker` 模式：左右手腕 tracker 常驻；其中一个 tracker 可临时拆下用于 `T_W^O` 标定后再装回
- `3 tracker` 模式：左右手腕 tracker 常驻，第 3 个 tracker 作为专用辅助标定 tracker

对于 VIVE 这类 tracker-based pose source，运行时核心公式仍然保留：

$$T_O^H = (T_W^O)^{-1} \cdot T_W^{tracker} \cdot T_{tracker}^H$$

对于单臂：

$$T_D^{E_{target}} = \Phi(T_O^H, \text{clutch state}, \text{scale}, \text{orientation policy})$$

$$T_B^{E_{target}} = (T_D^B)^{-1} \cdot T_D^{E_{target}}$$

其中 `Phi(...)` 不是简单固定一种策略，而是可配置：

- 绝对位姿映射
- 位置增量 + 姿态绝对
- 位置增量 + 姿态锁定
- 只跟踪平移，不跟踪姿态

当前项目的明确默认值：

- 默认控制模式：`absolute`
- `delta_clutch` 已硬封存
- 默认姿态策略：完整 `6DoF`
- 默认精细控制 TCP：`hand_base`
- 默认调试 / 纯 arm 控制 TCP：`rm_eef`

但要明确：**不是所有 source 都必须经过这套 tracker 标定公式。**

例如：

- 视觉系统如果直接输出 `T_O^H`，则不需要 `T_W^O`
- 外骨骼 FK 如果直接输出末端 pose，只需要零位和尺度标定
- glove / qN / scalar 输入只需要数值零点、归一化或线性映射

因此 `calibration` 模块必须支持：

- 全局空间标定
- source-specific 标定
- effector-specific 标定

而不是把所有输入都硬塞进 `tracker -> operator -> desk` 一条路径。

### 5.3 双臂一致性

双臂时仍然使用同一个 `O` 和同一个 `D`。
这样自动满足：

$$\Delta E = (T_D^{E_L})^{-1} \cdot T_D^{E_R} = (T_O^{H_L})^{-1} \cdot T_O^{H_R} = \Delta H$$

这仍然是双臂方案的理论核心。

### 5.4 Tracker 到机械臂末端的调试与矫正

这是本次修正里必须提升优先级的一块。系统不能只提供“最终标定结果”，还必须提供**可观察、可交互、可反复验证**的调试模式。

需要显式支持三层检查：

1. `raw tracker debug`
   直接看 `T_W^tracker` 是否稳定、旋转矩阵是否有效、不同 tracker 的朝向定义是否一致。

2. `operator-frame debug`
   在完成 `T_W^O` 后，可视化 `T_O^H`，确认手的前后左右上下和系统定义一致。

3. `robot-eef debug`
   在完成 `T_tracker^H`、`T_D^B` 后，同时显示：
   - tracker 原始坐标系
   - operator frame 下的控制 pose
   - desk frame 下的目标末端 pose
   - 机器人当前实际 eef pose
   - target / actual 误差
   - 当前使用的 TCP 模式：`rm_eef` / `hand_base`

需要明确区分两类“矫正”：

- 坐标系矫正：修正轴方向、手性、旋转补偿矩阵
- 几何偏置矫正：修正 tracker 固定点到控制点的平移/姿态偏置

建议加入一个正式的 `tracker_to_eef_debug` 模式，其职责不是发真机，而是：

- 读取 VIVE 数据
- 加载当前 calibration
- 通过 `SingleArmIKSolver` 或仿真后端计算期望姿态
- 在可视化里并排展示 `tracker / target eef / actual eef`
- 输出平移误差、姿态误差、跳变检测结果
- 允许在 `rm_eef` 和 `hand_base` 两种 TCP 间切换比较
- 允许交互调整安全范围立方体，并导出 workspace 配置

这个模式必须优先复用现有的：

- VIVE tracker frame 的调试绘制思路
- `rm65_inspire_tac/rendering.py` 的 SAPIEN 机器人可视化

但不要求兼容现有 demo 结构，可以直接重写成新的统一 debug app。

## 6. 运行时子模块职责

### 6.1 `interfaces.*` 与 `registry`

这是模块化的关键层。

需要至少定义：

- `ObservationSource`
- `Retargeter`
- `EffectorBackend`
- `Rig`

它们的最小职责：

- source: 产出 `ObservationChannel`
- retargeter: `TeleopObservation -> TeleopCommand`
- backend: 消费 `EffectorCommand`
- rig: 聚合多个 backend，并暴露统一 lifecycle

`registry` 负责通过 profile 把这些对象组装起来，避免主程序里写大量 `if source == "vive"` / `if hand == "inspire"`。

### 6.2 `sources.pose.*`

职责：

- 提供 pose6d 能力的输入源
- 输出 `ObservationChannel(capability="pose6d")`
- 处理最新帧缓存、超时、无效帧、零矩阵过滤

当前第一批实现可以包括：

- `VIVE`: UDP tracker
- `vision_pose`: 视觉估计得到的腕部/末端 pose
- `exo_fk`: 外骨骼关节经 FK 得到的 pose

其中 `VIVE` 模块应替代当前 `VIVE/utils.py` 的 demo 风格接口。

建议功能：

- `ViveReceiver`: 封装 `VIVEDataReceiver`
- `ViveTrackerConfig`: `name -> side -> role`
- `VivePoseSource.get_latest() -> dict[str, ObservationChannel]`

VIVE 在本项目中的明确约束：

- 最大 tracker 数：`3`
- 默认角色：
  - `left_wrist`
  - `right_wrist`
  - `aux_calibration`
- 如果只接 2 个 tracker，也必须支持完整运行和标定

由于 `triad_openvr` 包不能改，建议直接重写发送端协议，发送以下字段：

- `seq`
- `sender_timestamp`
- `tracker_name`
- `tracker_role`
- `valid`
- `pose_matrix_4x4`

不要继续使用当前“按固定切片解释裸 float 数组”的协议作为长期方案。

### 6.3 `sources.hand.*`

职责：

- 提供手部或末端控制相关的数值输入
- 输出 `joint_vector`、`keypoints3d`、`scalar` 等 channel
- 不再让 CLI 参数成为唯一接口

当前第一批实现可以包括：

- `q6_serial`: 包装 `teleop_hand.hand_q6_reader`
- `keypoints`: 未来视觉手关键点
- `glove`: 未来不同手套输入

建议：

- 保留 `SingleHandQ6Reader` / `BimanualHandQ6Reader`
- 在新模块里提供 `HandQ6Source`
- 支持 side 选择、启动校准、时间戳、异常传播

当前明确边界：

- 第一版只实现 `q6_serial -> Inspire`
- `keypoints3d` 接口保留，但实现可以后置

### 6.4 `calibration`

职责：

- 管理静态配置和运行时校准状态
- 提供 `world -> operator`、`operator -> desk`、`desk -> base` 变换
- 提供保存/加载配置能力

核心对象建议：

```python
@dataclass
class TeleopCalibration:
    T_W_O: np.ndarray | None
    T_D_B_left: np.ndarray | None
    T_D_B_right: np.ndarray | None
    T_tracker_H_left: np.ndarray | None
    T_tracker_H_right: np.ndarray | None
```

必须把“是否已完成 setup calibration / runtime calibration”做成显式状态检查。

此外，`calibration` 模块还必须支持：

- 校验矩阵是否是合法 SE(3)
- 保存多套 calibration profile
- 记录 profile 版本和最近一次调试时间
- 输出供 debug app 直接消费的可视化 frame 集合
- 标记本次 session 的 `home_qpos`
- 区分 `rm_eef` 与 `hand_base` 两种 TCP 的标定和调试结果

### 6.5 `clutch`

职责：

- 维护离合状态机
- 在按下离合瞬间记录 human anchor 和 robot anchor
- 生成 delta pose target

最关键的是：离合状态不能藏在主循环局部变量里，必须做成独立对象，便于测试和复用。

### 6.6 `retargeters.pose_to_arm`

职责：

- 输入 `pose6d channel` + `TeleopCalibration` + `ClutchState`
- 输出 `EffectorCommand(command_type="cartesian_pose")`

它是第一版最核心的 retargeter。

建议支持的姿态策略：

- `absolute`
- `yaw_only`
- `position_only`
- `fixed_orientation`

建议支持的平移策略：

- `absolute_1x`
- `delta_clutch`
- `delta_clutch_scaled`（历史实现）

当前明确要求：

- 第一版必须支持完整 `6DoF`
- 默认 `absolute`
- `delta_clutch` / `delta_clutch_scaled` 已硬封存，不属于 active runtime
- 第一版不需要以 `yaw_only` / `position_only` 作为主路径

还必须内置输入合法性检查：

- 位置是否有限值
- 旋转矩阵是否近似正交、行列式是否接近 `+1`
- 当前帧与上一帧的平移/旋转跳变是否超过阈值
- 当前目标是否越出 workspace / velocity / orientation 限制

### 6.7 `retargeters.hand_to_effector`

职责：

- 输入 `joint_vector` / `keypoints3d` / `scalar` channel
- 输出某个具体 effector 所需 command
- 允许一个输入能力映射到不同末端

第一批可实现的 retargeter：

- `joints_to_dexhand`: `q6/qN -> dexterous hand`
- `keypoints_to_dexhand`: `3D keypoints -> dexterous hand`
- `scalar_to_gripper`: `0..1 -> gripper open/close`

也就是说，把当前 `minimal_hand_teleop/InspireHandTeleopBridge` 从 CLI 脚本里抽成 `joints_to_dexhand + inspire backend` 的组合，而不是继续做成单体脚本。

当前明确优先级：

- 第一版只实现 `joints_to_dexhand`
- `keypoints_to_dexhand` 仅保留接口
- `scalar_to_gripper` 为两指夹爪预留

### 6.8 `backends.arms.*`

职责：

- 接收 `EffectorCommand(command_type="cartesian_pose")`
- 选择单臂 IK 或双臂 IK
- 输出实际下发的 arm qpos

具体复用策略：

- 单臂优先复用 `rm65_inspire_tac.ik.SingleArmIKSolver`
- 双臂优先复用 `rm65_inspire_tac.pipeline.DualArmMultiTaskSimulator`

这里不负责读输入，不负责标定，只负责执行。

### 6.9 `backends.hands.*` / `backends.grippers.*`

职责：

- 接收 hand / gripper 相关的 `EffectorCommand`
- 下发到具体末端

第一批可以有：

- `hands.inspire_tac`
- `hands.allegro`
- `grippers.parallel_gripper`

也就是说，hand backend 必须从第一天开始允许“同一套上游输入，切换不同末端实现”。

同时需要明确区分：

- `control path`
- `tactile telemetry path`

其中触觉读取是可选辅助通道，不能阻塞主控制环。
如果开启 Inspire 触觉信息读取，应使用独立线程/进程或独立 listener，以避免把控制频率从目标 `30 Hz` 直接拖垮。

### 6.10 `rigs.*`

职责：

- 把多个 backend 组合成一个系统级 rig
- 管理 side、资源生命周期、联合启动和关闭顺序

第一批可以有：

- `arm_only`
- `hand_only`
- `arm_hand`
- `bimanual`

当前第一可用版本必须交付 `bimanual`。

### 6.11 `debug.*`

至少要有三个 debug 入口：

1. `vive_frames`
   只看原始 tracker 坐标系和数据质量。

2. `tracker_to_eef_debug`
   重点看从 `T_W^tracker` 到 `T_D^E_target` 的全过程，专门用于标定和矫正。

3. `sim_overlay`
   结合 `SAPIEN`，把机器人模型、workspace、目标末端、实际末端、误差向量一起画出来。

4. `workspace_tuner`
   在 SAPIEN 里交互调节安全范围立方体，并导出配置。

可视化里建议固定显示：

- `W / O / D / B / E_target / E_actual` 坐标轴
- 当前 side、控制模式、clutch 状态
- stale / invalid / dropped / frozen 状态
- 平移误差与姿态误差
- 当前是否允许进入真机模式
- 本次 session 的 `home_qpos`
- 当前控制频率与最近窗口平均频率

调试显示明确采用多窗口是可接受的：

- tracker/raw 数据窗口
- SAPIEN 机器人与 workspace 窗口
- 可选日志 / 状态窗口

## 7. 目录重构建议

### 7.1 保留并复用

- `rm65_inspire_tac/ik.py`
- `rm65_inspire_tac/multitask_ik.py`
- `rm65_inspire_tac/hardware.py`
- `teleop_hand/hand_q6_reader.py`
- `teleop_runtime/sources/pose/vive.py`

这些都更适合作为新架构里的 backend/source 实现来源，而不是最终顶层组织方式。

### 7.2 应该降级为示例或 wrapper

- `VIVE/utils.py`
- `minimal_hand_teleop/inspire_hand_teleop.py`

它们不再应该承载核心架构，只保留为：

- 示例
- 兼容入口
- 开发调试脚本

### 7.3 `rm65_inspire_tac/retargeting.py` 的定位

它保留，但从“主 teleop 架构中心”降为“AVP / VisionPro 手部重定向插件”。

也就是：

- 它继续解决 `hand keypoints -> wrist_pose_robot + hand qpos`
- 但不参与新的 VIVE arm teleop 主流程

### 7.4 面向未来设备扩展的组织要求

后续每新增一类设备，原则上只能新增以下其中一层：

- 一个新的 `source`
- 一个新的 `retargeter`
- 一个新的 `backend`
- 一个新的 `rig`

如果新增设备时需要直接修改 `session.run()` 主循环或复制一份新的 app 主脚本，说明模块边界设计仍然不够好。

## 8. 第一版落地范围

第一版不要试图一次做完所有输入源和双臂协同。建议按下面顺序落地。

### Phase 1: 通用骨架搭建

交付：

- `teleop_runtime/interfaces/*`
- `teleop_runtime/registry.py`
- `teleop_runtime/schemas.py`
- `teleop_runtime/calibration.py`
- `teleop_runtime/session.py`
- `teleop_runtime/transforms.py`
- 历史版本中的 `teleop_runtime/clutch.py`（已从 active runtime 移除）
- `teleop_runtime/safety.py`
- `teleop_runtime/diagnostics.py`
- `teleop_runtime/recording.py`

目标：

- 把通用数据结构、能力接口、registry 和状态机先立住
- 不急着接真机

### Phase 2: VIVE 控 RM65 arm

交付：

- `debug/vive_frames.py`
- `debug/tracker_to_eef_debug.py`
- `debug/sim_overlay.py`
- `debug/workspace_tuner.py`
- `sources/pose/vive.py`
- `retargeters/pose_to_arm.py`
- `backends/arms/rm65.py`
- `rigs/bimanual.py`
- `apps/run_vive_rm65_calibration_debug.py`
- `apps/run_vive_rm65_arm.py`

目标：

- 跑通 `最多 3 tracker` 的 VIVE 输入协议
- 跑通双臂 `VIVE -> calibration -> arm target -> IK -> sim overlay`
- 再跑通双臂 `VIVE -> calibration -> arm target -> IK -> RM65`
- 先支持键盘离合，脚踏板后接
- 真机前必须经过 tracker-to-eef debug 模式验收
- 支持 `rm_eef` 与 `hand_base` 两种 TCP 调试/控制模式

### Phase 3: hand_q6 控 Inspire 手

交付：

- `sources/hand/q6_serial.py`
- `retargeters/joints_to_dexhand.py`
- `backends/hands/inspire_tac.py`
- `rigs/bimanual.py`
- `apps/run_hand_inspire.py`

目标：

- 跑通左右双手 `q6 -> Inspire`
- 把当前 `minimal_hand_teleop` 真正组件化
- 修正 import 路径和 CLI-only 架构问题
- 让同一层 retargeter 未来能切到其他 hand backend
- 触觉读取作为可选辅助通道接入，不阻塞控制环

### Phase 4: arm + hand 合流

交付：

- `rigs/bimanual.py`
- `apps/run_rm65_full_teleop.py`

目标：

- 同时控制 `左右双臂 + 左右双手`
- arm 与 hand 支持不同更新频率
- 支持单独使能/禁用 arm 与 hand
- 这是第一可用版本的最终验收阶段

### Phase 5: 双臂扩展

交付：

- 双臂 `Desk Frame` 一致性强化校验
- `2 tracker` / `3 tracker` 标定流程稳健化
- 可选接入 `DualArmMultiTaskSimulator`

目标：

- 让 `blueprint` 中的双臂相对位姿约束在真机上稳定落地

### Phase 6: 扩展验证

交付：

- `keypoints3d` hand input 预演验证
- 至少验证一种两指夹爪或 gripper backend

目标：

- 验证 runtime 的模块化不是纸面设计，而是真能支持 hand 输入与末端执行器扩展

## 9. 关键接口设计

### 9.1 Session 配置

建议用一个明确的配置对象统一驱动：

```python
@dataclass
class TeleopSessionConfig:
    mode: str                        # right_only / left_only / bimanual
    source_profile: str              # "vive_right_tracker", "vision_pose_right", ...
    embodiment_profile: str          # "rm65_inspire_right", "rm65_allegro_right", ...
    teleop_profile: str              # mapping + safety + clutch policy
    tcp_mode: str                    # "rm_eef" / "hand_base"
    scale: float
    record_enabled: bool
    allow_hardware_after_debug_pass: bool
```

### 9.2 Session 主循环

主循环逻辑应该固定化：

1. 拉取最新输入
2. 检查时间戳和有效性
3. 记录原始输入与状态事件
4. 做 runtime calibration 变换
5. 生成 `TeleopObservation`
6. 通过 retargeter 生成 `TeleopCommand`
7. 通过 rig/backend 下发
8. 记录状态、错误、频率

也就是说，新模块真正的主入口应该是：

```python
session = TeleopSession(...)
session.run()
```

而不是再出现多个 300 行 CLI 脚本各自拼一遍循环。

建议再显式引入一层安全配置：

```python
@dataclass
class TeleopSafetyConfig:
    max_translation_jump_m: float
    max_rotation_jump_deg: float
    stale_timeout_sec: float
    invalid_frame_limit: int
    freeze_on_fault: bool
    minimum_control_hz: float        # 15
```

并建议为可扩展性补一层 profile：

```python
@dataclass
class SourceProfile:
    source_type: str                 # vive / vision_pose / q6_serial / ...
    capability: str                  # pose6d / joint_vector / ...
    params: dict[str, Any]

@dataclass
class EmbodimentProfile:
    rig_type: str                    # arm_only / arm_hand / bimanual
    arm_backend: str | None
    hand_backend: str | None
    gripper_backend: str | None
    params: dict[str, Any]
```

并建议加入录制配置：

```python
@dataclass
class RecordingConfig:
    enabled: bool = True
    save_raw_inputs: bool = True
    save_calibration: bool = True
    save_commands: bool = True
    save_fault_events: bool = True
```

## 10. 安全策略

必须从第一版开始内置：

- 输入超时保护：VIVE / 串口数据过期即保持或停车
- 工作空间裁剪：所有 `T_D^E_target` 在 desk frame 下裁剪
- 关节步长限制：继续复用 `max_joint_step` / `max_angvel`
- 启动握手：arm/hand 未 ready 不进入运行态
- 紧急停止：键盘中断、未来脚踏板释放、异常退出统一 `close()`
- 初始位姿：进入 teleop 前由人工移动到本次 session 的 home，并由程序采集记录

除此之外，必须把 fault handling 做成**正式状态机**，而不是 scattered `if`：

- `INIT`
- `CALIBRATION`
- `DEBUG_ONLY`
- `ARMED`
- `RUNNING`
- `FROZEN`
- `FAULT`
- `ESTOP`

状态转移原则：

- 未通过 debug 验收，不能从 `DEBUG_ONLY` 进入 `ARMED`
- 输入 stale / invalid / jump 过阈值时，先进入 `FROZEN`
- 连续故障超过阈值时，进入 `FAULT`
- 人工确认前，`FAULT` 不能自动恢复到 `RUNNING`
- 任一关键部件部分掉线或部分失效时，整个 rig 进入 `FROZEN`

必须覆盖的异常类型：

- VIVE UDP 断流
- tracker 返回全零或畸形旋转矩阵
- 手套串口断开或反复超时
- IK 求解失败或解跳变过大
- RM65 控制进程未 ready、退出、卡死
- Inspire 手 Modbus 连接失败

建议默认行为：

- arm：冻结到最近一个已确认安全的 target
- hand：停止发送新命令，必要时张开到安全姿态
- UI / 日志：明确打印 fault source、首次发生时间、最近恢复尝试时间

需要明确区分：

- `critical path failure`：触发整个 rig `FROZEN`
- `aux telemetry failure`：不阻塞控制环，但必须报警并录制

## 11. 测试与验收

### 11.1 单元测试

至少覆盖：

- `T_W^O`、`T_D^B`、`T_tracker^H` 组合变换
- clutch 增量逻辑
- `q6` invert / gain / bias / order 映射
- workspace clipping

### 11.2 离线集成测试

需要支持：

- 用录制的 VIVE 包重放 arm target
- 用录制的 `q6` 包重放 hand target
- 不接真机做 dry-run
- 对无效帧、跳变帧、全零帧、stale 帧做 fault injection
- 对 `2 tracker` / `3 tracker` 两种 VIVE 配置都要覆盖

### 11.3 调试模式验收

在进入真机前，必须先通过 debug 模式验收：

1. `raw tracker`
   不移动 tracker 时抖动在阈值内；旋转矩阵合法；断流能被识别。

2. `operator frame`
   操作者向前/向左/向上移动，`O` 系下的显示方向与预期一致。

3. `tracker -> eef target`
   可视化中目标末端与手部操作方向一致；姿态补偿矩阵调整后能立即观察变化。

4. `sim overlay`
   在不接真机时，目标轨迹、IK 结果、workspace 裁剪和错误状态都能被可视化确认。
5. `workspace tuner`
   能通过 SAPIEN 中的立方体安全范围交互调参，并回写配置。

### 11.4 真机验收标准

双臂第一阶段验收：

- 完成一次 `T_W^O` 标定后，左右手 tracker 都能稳定驱动对应 RM65 末端
- 离合按下/释放时，机械臂不会跳变
- 断流后机械臂不会继续漂移
- 非法旋转矩阵、突变数据、连续 stale 输入不会直接下发到硬件
- 从 `DEBUG_ONLY` 切换到 `RUNNING` 需要显式 arm/confirm 动作
- `rm_eef` 与 `hand_base` 两种 TCP 模式都能通过调试模式验收
- 人工设定的 home 被程序正确采集并用于本次 session

灵巧手验收：

- 左右手 `q6` 输入与 Inspire 手动作方向一致
- 调整 `invert / gain / bias / joint_order` 后行为可预测
- 串口断开、Modbus 失败时能安全退出
- 开启触觉读取后，主控制环仍不低于最低可接受频率

双臂验收：

- 左右手在 `O` 中的相对位姿变化，能在 `D` 中复现
- 任一侧 tracker / IK / hardware 异常时，整个 rig 进入 `FROZEN`
- 故障录制数据足够离线重放和复现

## 12. 当前实现时的明确取舍

第一版明确不做：

- 不把视频落盘、episode 采集混进主模块
- 不把 AVP/VisionPro retargeting 与 VIVE arm teleop 强耦合
- 不为所有旧脚本保留逐文件兼容
- 不在主循环里隐式写死坐标补偿

第一版明确要做到：

- 一个清晰的 runtime 主入口
- 一套 capability-first 的 source / retargeter / backend / rig 分层
- 一个显式的 calibration 状态对象
- 一套 side-based 通用 command schema
- 一个正式的 tracker-to-eef 调试与矫正模式
- 一个默认开启的 fault detection / freeze 机制
- 一个默认开启的数据录制机制
- `VIVE(<=3 trackers) -> 双臂 RM65 arm`
- `双手 q6 -> 双手 Inspire`
- 可组合成同一 bimanual session

## 13. 最终落地结论

这个仓库后续应该形成如下分层：

```text
raw drivers:
  VIVE/mocap
  teleop_hand
  inspire_tac
  rm65 sdk

generic teleop runtime:
  teleop_runtime

robot-specific backend:
  rm65_inspire_tac

apps / examples:
  minimal_hand_teleop
  teleop_arm/apps
```

这会把当前“设备脚本、映射逻辑、IK、硬件控制、运行时状态混在一起”的结构，重构成：

- 输入源清晰
- 标定链路清晰
- arm/hand 解耦
- 单臂/双臂模式统一
- 真机和 dry-run 都有明确入口

后续实现时，优先级严格按以下顺序推进：

1. 统一 schema 与 calibration
2. 跑通 `VIVE(<=3 trackers) -> 双臂 RM65 arm`
3. 跑通 `双手 q6 -> 双手 Inspire`
4. 合并成 `双臂 + 双手` 单 session
5. 再做 `keypoints3d / gripper` 扩展验证

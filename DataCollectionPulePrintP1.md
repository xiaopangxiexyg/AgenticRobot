# Data Collection Blueprint P1

## 0. Status

本文档是基于 **2026-04-19** 当前仓库代码状态修订后的 `Phase 1` 规划。

`Phase 1` 的目标调整为先打通 **双臂优先、兼容单臂退化** 的数据采集闭环，但实现基线已经变了：

- 采集框架应当是 **独立于当前 teleop 入口脚本** 的 collector runtime
- 但要尽量复用当前 `teleop_runtime` 已有的 `interfaces / schemas / source / retargeter / backend / calibration` 能力
- 在此基础上新增 **LeRobot v2.1 风格数据集输出层**

并且根据当前代码现状，需要明确一件事：

- **双臂真机遥操作主链路已经打通**
- 当前 `Phase 1` 的剩余工作重点已经不再是 teleop bring-up
- 而是实现一套独立 collector runtime，并把数据写入做成高并行异步流水线

当前 `Phase 1` 固定范围：

- 统一按 **双臂 schema** 设计与实现
- 采集侧配置支持：`left / right / both`
- 单臂采集时，未采集 side 的字段置空
- `single RGB camera`
- 输出为 **LeRobot v2.1 风格**数据集
- **不使用** `lerobot` 官方 Python 包

本文档只做规划，不做实现。

---

## 1. 当前代码基线

这一节只描述仓库里 **已经存在** 的能力，后续任务拆分必须以这里为基线。

### 1.1 已经存在的 runtime 主干

当前 `teleop_runtime` 已经具备下面这些模块：

- `teleop_runtime/session.py`
  - `TeleopSession`
  - `SessionStepResult`
  - 主控制循环、状态机、安全冻结、`SessionRecorder`
- `teleop_runtime/schemas.py`
  - `ObservationChannel`
  - `TeleopObservation`
  - `EffectorCommand`
  - `TeleopCommand`
- `teleop_runtime/calibration.py`
  - `TeleopCalibration`
  - calibration profile 的保存/加载
- `teleop_runtime/sources/pose/vive.py`
  - `VivePoseSource`
  - 已支持 `sender_timestamp / receiver_timestamp / tracker_role`
- `teleop_runtime/sources/hand/q6_serial.py`
  - `HandQ6Source`
- `teleop_runtime/sources/misc/keyboard.py`
  - `KeyboardButtonSource`
- `teleop_runtime/retargeters/pose_to_arm.py`
  - `PoseToArmRetargeter`
- `teleop_runtime/retargeters/joints_to_dexhand.py`
  - `JointsToDexHandRetargeter`
- `teleop_runtime/backends/arms/rm65.py`
  - `RM65ArmBackend`
  - 已支持 `refresh_state_from_hardware()` 和 `get_state()`
- `teleop_runtime/backends/hands/inspire_tac.py`
  - `InspireTacHandBackend`
  - 已支持 telemetry 线程和 `get_state()`
- `teleop_runtime/rigs/bimanual.py`
  - `BimanualRig`

### 1.2 已经存在的录制/回放能力

当前代码里已经有两套录制语义：

#### A. `SessionRecorder`

文件：

- `teleop_runtime/recording.py`

用途：

- debug
- 原始 channel / command / fault JSONL 记录

不适合做：

- 视频写盘
- LeRobot parquet writer

#### B. `EpisodeRecorder`

文件：

- `teleop_runtime/episodes.py`
- `teleop_runtime/debug/run_recorded_episode_replay.py`
- `teleop_runtime/debug/validate_recorded_episode.py`

用途：

- 当前 teleop episode 的 JSONL 记录
- 当前 replay / validate 工具链

这一套已经被：

- `teleop_runtime/debug/run_vive_sapien_virtual.py`
- `teleop_runtime/debug/run_vive_rm65_real_teleop.py`

实际使用。

结论：

- `SessionRecorder` 保留为 debug recorder
- `EpisodeRecorder` 保留为当前 teleop episode/replay 格式
- `Phase 1` 的数据集输出必须是 **第三层**，不能把前两者硬改成 LeRobot writer

### 1.3 当前还没有的能力

截至当前代码：

- runtime 内 **没有** camera source
- runtime 内 **没有** camera config 层
- runtime 内 **没有** LeRobot 风格 writer
- runtime 内 **没有**通用 data collection runner
- runtime 内 **没有** dataset sample assembler / dataset stats accumulator

### 1.4 当前真实入口

当前真实双臂真机 teleop 入口是：

- `teleop_runtime/debug/run_vive_rm65_real_teleop.py`

它已经验证了下面这条主线是能工作的：

- `VivePoseSource -> PoseToArmRetargeter -> TeleopSession -> BimanualRig -> RM65ArmBackend`
- `HandQ6Source -> JointsToDexHandRetargeter -> InspireTacHandBackend`

并且当前入口已经具备：

- 双臂真机 arm teleop
- 双手 q6 -> InspireTac hand teleop
- `--hand-telemetry-hz`
- `--hand-tactile-regions`
- `EpisodeRecorder` sidecar 录制
- command / preview / state backend 分离
- 启动阶段双臂 state 并行读取

因此 `Phase 1` 不应该另起一套平行 runtime，而应该在这个主线上做 **数据采集扩展层**。

### 1.5 重新划线后的剩余缺口

在当前代码基线上，`Phase 1` 真正还缺的能力主要只剩三类：

1. 相机与视频链路
   - `RealSenseRGBSource`
   - camera config
   - mp4 writer
2. dataset 抽样与落盘链路
   - sample assembler
   - LeRobot parquet/meta writer
3. 观测语义与时间戳补齐
   - hand telemetry timestamp
   - runtime arm state source 的明确口径
   - dataset 级 metadata/schema 固化
4. 并行写盘流水线
   - 低维样本与视频帧解耦
   - 异步 mp4 编码
   - 异步 parquet/meta 落盘
   - bounded queue / backpressure 策略

---

## 2. P1 冻结约束

### 2.1 模态与范围

- `enable_depth = False`
- 只采单路 `RGB`
- arm / hand / tactile 统一按双臂字段设计
- `active_sides` 允许为：`left`、`right`、`both`
- 单臂采集时，未采集 side 的字段置空，并在 dataset meta 中显式声明 `active_sides`
- tactile 默认启用，但允许 `--with-tactile=false` 先调通主链路
- 多相机不是 `Phase 1` 范围

### 2.2 频率与时间

- 控制主循环目标：`30Hz`
- 数据集按 **实际执行频率**采样，不因 `dataset nominal fps` 强制降采样
- `dataset nominal fps` 只作为 metadata 参照值，不代表真实帧间隔，也不决定实际采样触发
- 每条样本必须保存真实 `timestamp`
- 每个模态尽量保存：
  - `source timestamp`
  - `receive timestamp`
  - `sample timestamp`
  - `age_sec`

说明：

- `VivePoseSource` 当前已经有 `sender_timestamp / receiver_timestamp`
- `RM65ArmBackend` 当前已经有 `hardware_refresh_time`
- `InspireTacHandBackend` 当前 **还没有** telemetry timestamp，需要补
- 相机 source 需要自己提供 camera timestamp / receive timestamp

### 2.3 action 原则

`Phase 1` 的主 `action` 采用：

- **机器人侧 cartesian target pose in robot base frame**

并且统一按双臂字段实现：

- `left_arm` 和 `right_arm` 都保留固定位置
- 未采集 side 在固定低维槽位里置空

实际来源是当前各 side `EffectorCommand` 真正发给 arm backend 的：

- `step_result.command.commands["left_arm"].value`
- `step_result.command.commands["right_arm"].value`

如果对应 command 的类型为：

- `cartesian_pose`

则这是当前系统真正执行的主动作语义。

同时额外保存：

- `teleop.action_source.left_wrist_pose_operator`
- `teleop.action_source.left_wrist_pose_desk`
- `teleop.action_source.left_hand_q6`
- `teleop.action_source.right_wrist_pose_operator`
- `teleop.action_source.right_wrist_pose_desk`
- `teleop.action_source.right_hand_q6`
- `action.replay.left_arm_joint_target`
- `action.replay.left_hand_joint_target`
- `action.replay.right_arm_joint_target`
- `action.replay.right_hand_joint_target`

### 2.4 与旧版规划相比的调整

以下几点要明确改口径：

#### A. `arm force` 不再作为 P1 基线必做

原因：

- 当前 `RM65ArmBackend.get_state()` **没有**输出 6D force
- 当前 `refresh_state_from_hardware()` 也没有 force 字段

因此：

- `arm force` 只能作为后续增强项
- 不能再把它写成 `Phase 1` 已冻结依赖

#### B. `Phase 1` 不再先做一套独立 state source 再接 session

当前真机 runner 的已落地路径是：

- 启动 / 标定阶段使用独立 state backend 读取 RM65 硬件状态
- 运行阶段沿用当前 `runtime_arm_state_source = last_commanded_action`
- 手部 telemetry 由 `InspireTacHandBackend` 后台线程维护

因此 `Phase 1` 应优先复用这条已验证路径，而不是先造新的平行 listener 体系。

#### C. `Phase 1` 的数据集层必须与现有 episode 层并存

当前 replay 工具绑定的是：

- `teleop_runtime/episodes.py`

不是 LeRobot 数据集。

所以：

- `Phase 1` 写出 LeRobot 数据集时
- 当前 `EpisodeRecorder` 最好继续保留为 sidecar debug / replay 产物

---

## 3. 架构决策

## 3.1 三层职责划分

`Phase 1` 需要明确分成三层：

### 层 1：runtime 控制层

直接复用当前代码：

- `TeleopSession`
- `TeleopCalibration`
- `ObservationSource`
- `Retargeter`
- `Rig`
- `RM65ArmBackend`
- `InspireTacHandBackend`

职责：

- 采输入
- 算命令
- 下发硬件
- 管状态机和安全

### 层 2：debug / replay 记录层

直接复用当前代码：

- `SessionRecorder`
- `EpisodeRecorder`

职责：

- 调试
- 故障回放
- 当前 JSONL replay

### 层 3：dataset collection 层

这是 `Phase 1` 新增内容。

建议新增：

```text
teleop_runtime/collection/
├── __init__.py
├── schema_v21.py
├── sample_assembler.py
├── writer_v21.py
├── stats.py
├── run_collect.py
└── replay_bridge.py
```

职责：

- 维护独立于当前 debug 入口的 collector main loop
- 从 `TeleopSession` 和 backend state 抽取可训练样本
- 把写盘工作交给异步并行 writer pipeline
- 写出 LeRobot v2.1 风格目录
- 不干扰现有 debug / replay 逻辑

## 3.2 P1 应复用哪些现有模块

`Phase 1` 建议直接复用：

- `teleop_runtime/interfaces/*.py`
- `teleop_runtime/session.py`
- `teleop_runtime/calibration.py`
- `teleop_runtime/schemas.py`
- `teleop_runtime/sources/pose/vive.py`
- `teleop_runtime/sources/hand/q6_serial.py`
- `teleop_runtime/sources/misc/keyboard.py`
- `teleop_runtime/retargeters/pose_to_arm.py`
- `teleop_runtime/retargeters/joints_to_dexhand.py`
- `teleop_runtime/backends/arms/rm65.py`
- `teleop_runtime/backends/hands/inspire_tac.py`

只把下面这些作为**参考实现**，不要让 collector runtime 直接依赖其主循环：

- `teleop_runtime/debug/run_vive_rm65_real_teleop.py`
- `teleop_runtime/debug/run_vive_sapien_virtual.py`

不要复用为 dataset writer 的模块：

- `teleop_runtime/recording.py`
- `teleop_runtime/episodes.py`

它们可以并行开着，但不要把它们改造成 parquet/mp4 writer。

## 3.3 双臂优先采集链路的推荐组装方式

### 输入源

- `VivePoseSource`
  - 默认同时配置 `left_wrist_pose` 和 `right_wrist_pose`
  - `auxiliary_tracker_pose` 可选
- `KeyboardButtonSource`
  - 用于 clutch / record control
- `HandQ6Source.from_bimanual(...)`
  - 单臂采集时允许只提供单侧 serial port

### retargeter

- `PoseToArmRetargeter(side="left", input_channel_name="left_wrist_pose")`
- `PoseToArmRetargeter(side="right", input_channel_name="right_wrist_pose")`
- `JointsToDexHandRetargeter(side="left", input_channel_name="left_hand_q6")`
- `JointsToDexHandRetargeter(side="right", input_channel_name="right_hand_q6")`

### backend

- `RM65ArmBackend(side="left")`
- `RM65ArmBackend(side="right")`
- `InspireTacHandBackend(side="left", enable_telemetry=True)`
- `InspireTacHandBackend(side="right", enable_telemetry=True)`

### rig

当前优先做法：

- 继续用 `BimanualRig`
- 按 `active_sides` 决定实际挂哪些 backend
- dataset schema 仍统一按双臂字段输出

理由：

- 当前 `BimanualRig` 本质上只是 `dict[str, EffectorBackend]`
- 并不强依赖必须左右各一套
- 但 collector / schema 应统一以双臂字段为主，避免后续再做 schema 扩张

### collector runtime

这里的关键原则是：

- collector runtime 是 **独立入口**
- 它可以参考 `run_vive_rm65_real_teleop.py` 的组件组装方式
- 但不应直接 import 那个脚本里的主循环、状态推进和临时 helper 作为生产采集框架的核心依赖

更合理的边界是：

- collector 复用 `source / retargeter / backend / rig / session`
- collector 自己维护：
  - 采样节拍
  - sample 组装
  - writer pipeline
  - backpressure 策略

### calibration

当前优先做法：

- 继续用 `TeleopCalibration.load_json(...)`

当前直接接受双臂 profile；单臂采集时只驱动 `active_sides` 对应的一侧，但 dataset schema 仍保留双侧字段。

## 3.4 状态采样策略

### RM65 arm state

最终口径：

- `Phase 1 collector` **不保留**运行期 `state_backend.refresh_state_from_hardware()`
- 启动 / 标定阶段的 state backend 仍可继续服务 bring-up / 对位 / 校准
- 但 dataset 运行期 arm observation 固定按 `runtime_arm_state_source = last_commanded_action`

collector 里 arm state 的具体来源统一为：

- `current_qpos` 取 command/preview backend 的本地状态
- `tcp pose` 优先由当前本地 `qpos` 做 FK 计算
- 如果没有可用 FK，再退回最后一次 command 附带的 cartesian target metadata

处理结论：

- `Phase 1` 不再保留“持续真实硬件测量态”作为默认方案
- 如果后续确认训练必须依赖 measured state，再作为单独增强项加回
- collector 必须把 `arm_state_source=last_commanded_action` 写进 sample 或 meta

### Inspire hand + tactile state

当前建议：

- 直接复用 `InspireTacHandBackend(enable_telemetry=True)`
- collector loop 只调用 `get_state()`

当前已经具备：

- `telemetry_hz`
- `tactile_regions`

仍然必须补的能力是：

- `joint_timestamp`
- `tactile_timestamp`
- `telemetry_timestamp`

否则无法正确算 `age_sec`。

### RGB state

当前必须新增：

- `RealSenseRGBSource`

文件建议：

- `teleop_runtime/sources/camera/realsense_rgb.py`

它应该是：

- 一个带后台缓存的 frame provider

而不是：

- 把大图像塞进 `ObservationChannel.value`

## 3.5 并行写盘流水线

这是本次规划新增的一等约束。

### 原则

- 控制线程只做：
  - `session.run_step()`
  - 抽取 backend state
  - 组装 sample
  - 提交写盘任务
- 不在控制线程内执行：
  - mp4 编码
  - parquet flush
  - 大块 metadata 统计/聚合

### 推荐流水线

```text
collector main loop
    -> sample queue
    -> low-dim writer worker(s)
    -> video writer worker(s)
    -> metadata/stats worker
```

### 并行约束

- low-dim data 必须 lossless
- metadata 必须 lossless
- 视频优先走独立 worker，避免阻塞控制线程
- 所有队列必须有 `maxsize`
- backlog 过大时必须有明确策略：
  - `block`
  - `raise`
  - 或仅允许对非关键视频任务做受控丢弃

### 当前建议

- `Phase 1` 默认：
  - 低维样本 `block or fail-fast`
  - 视频写盘异步并行
  - metadata/stats 异步并行
- 不允许静默丢掉低维样本

## 3.6 action / observation 语义冻结

### 主 `action`

固定为：

```text
[left_target_pose_base_xyz_quat(7),
 right_target_pose_base_xyz_quat(7),
 left_hand_joint_target(6),
 right_hand_joint_target(6)]
```

总维度：

- `26`

来源：

- `left_arm`: `step_result.command.commands["left_arm"].value`
- `right_arm`: `step_result.command.commands["right_arm"].value`
- `left_hand`: `step_result.command.commands["left_hand"].value`
- `right_hand`: `step_result.command.commands["right_hand"].value`

单臂采集时：

- 未采集 side 的低维固定槽位填 `NaN`
- 独立结构化 side 字段写 `null`
- `meta/info.json` 与 episode metadata 记录 `active_sides = left/right/both`

### `observation.state`

当前代码基线下，建议先冻结为：

```text
[left_arm_joint_pos(6),
 left_eef_pose_base_xyz_quat(7),
 left_hand_joint_pos(6),
 right_arm_joint_pos(6),
 right_eef_pose_base_xyz_quat(7),
 right_hand_joint_pos(6)]
```

总维度：

- `38`

说明：

- 旧版文档中的 `right_arm_6d_force` 先移出 P1 基线
- tactile 单独存，不进 `observation.state`
- 单臂采集时未采集 side 的固定槽位填 `NaN`

### 单独保存的大字段

- `observation.tactile.left`
- `observation.tactile.right`
- `teleop.action_source.left_wrist_pose_operator`
- `teleop.action_source.left_wrist_pose_desk`
- `teleop.action_source.left_hand_q6`
- `teleop.action_source.right_wrist_pose_operator`
- `teleop.action_source.right_wrist_pose_desk`
- `teleop.action_source.right_hand_q6`
- `action.replay.left_arm_joint_target`
- `action.replay.left_hand_joint_target`
- `action.replay.right_arm_joint_target`
- `action.replay.right_hand_joint_target`
- `observation.last_action`

### 关键提取口径

#### 单侧 target pose base

直接取：

- `EffectorCommand.value`

不要再从 `target_pose_desk` 反推一次。

#### 单侧 hand target

优先取：

- 对应 side 的 hand command 最终 `value`

原始手套输入另存：

- `source_q6`

#### 单侧当前 TCP pose

优先取：

- `RM65ArmBackend.compute_tcp_pose()`

不要把：

- `hardware_pose_6d`

当成数据集主 pose 字段，因为它现在只是 SDK 原始输出，语义和格式还没在 runtime 里完全冻结。

## 3.7 数据集目录

建议仍然对齐 LeRobot v2.1 风格：

```text
dataset_root/
├── data/chunk-000/episode_000000.parquet
├── videos/observation.images.front/chunk-000/episode_000000.mp4
└── meta/
    ├── info.json
    ├── episodes.jsonl
    ├── tasks.jsonl
    ├── episodes_stats.jsonl
    ├── calibration/
    │   └── teleop_profile.json
    └── devices/
        └── realsense.json
```

当前建议：

- dataset episode 编号由 dataset writer 自己管理
- 不复用 `EpisodeRecorder` 的目录/编号逻辑

---

## 4. 更新后的 Phase 1 任务清单

## 4.1 任务总览

| ID | 任务 | 目标产物 | 完成判据 |
| --- | --- | --- | --- |
| T0 | 固化 P1 语义 | 本文档 + schema 冻结 | action / observation / 目录结构不再反复改 |
| T1 | 补齐当前 runtime 的采集 metadata 缺口 | retargeter/backend metadata 补丁 | writer 不需要靠猜字段写盘 |
| T2 | 设备配置层 + RGB source | `realsense.json` + `RealSenseRGBSource` | 单相机 RGB 稳定缓存，不阻塞控制 |
| T3 | 独立 collector runtime 组装层 | independent collect runtime builder | collector 独立于当前 debug 入口，但复用底层 teleop 接口 |
| T4 | sample assembler | `sample_assembler.py` | 能把 `TeleopSession` + backend state 组装成 row |
| T5 | 并行 writer pipeline | `async_writer.py` + `writer_v21.py` + `stats.py` | parquet/mp4/meta 异步并行落盘，不阻塞控制线程 |
| T6 | collector runner | `run_collect.py` | 真机录出一个 `left/right/both` episode |
| T7 | replay bridge / 校验 | `replay_bridge.py` 或等价脚本 | 数据集 action 可回放或可转回当前 replay 格式 |
| T8 | 验收与回归 | checklist + smoke test | 结构、时序、回放、性能通过 |

## 4.2 推荐实现顺序

推荐顺序：

1. `T0` 固化语义
2. `T1` 补 metadata 缺口
3. `T2` 做 camera config + RGB source
4. `T3` 做独立 collector runtime builder
5. `T4` 做 sample assembler
6. `T5` 做并行 writer pipeline
7. `T6` 做 runner
8. `T7` 做 replay bridge
9. `T8` 验收

---

## 5. 任务细化

## 5.1 T1 补齐当前 runtime 的采集 metadata 缺口

这一项是新版规划里最重要的变化。

旧版文档默认很多字段还不存在；现在其实大部分链路都有了，但还差几个关键元数据。

### 需要补齐的点

#### A. `PoseToArmRetargeter` metadata 口径补齐

当前已有：

- `EffectorCommand.metadata["target_pose_desk"]`
- `EffectorCommand.metadata["target_pose_operator"]`
- `tracker_role`
- `control_mode`
- `orientation_mode`

但当前命名存在问题：

- `target_pose_desk` 确实是机器人 target in desk
- `target_pose_operator` 实际上更接近 **source wrist pose in operator**

因此 `Phase 1` 里建议补成更明确的字段：

- `source_pose_operator`
- `source_pose_desk`
- `target_pose_desk`
- `target_pose_base`
- `home_human_pose_desk`
- `home_robot_pose_desk`

其中：

- `target_pose_base` 也可以由 collector 从 `EffectorCommand.value` 直接拿
- 但字段名仍建议在 metadata 层显式化，降低 assembler 歧义

#### B. `InspireTacHandBackend.get_state()` 需要时间戳

当前已有：

- `joint_pos`
- `tactile`
- `tactile_dict`
- `telemetry_error`
- `last_sent_command`

还需要补：

- `last_sent_command_timestamp`
- `joint_timestamp`
- `tactile_timestamp`
- `telemetry_timestamp`
- `tactile_regions`

当前口径约定：

- `joint_timestamp`：成功发起并完成本次 `read_joint_pos()` 读取的请求起始时刻
- `tactile_timestamp`：成功发起并完成本次 `read_tac_data()` 读取的请求起始时刻
- `telemetry_timestamp`：本轮 telemetry cycle 完成并写入缓存的时刻

#### C. `RM65ArmBackend.get_state()` 可以直接复用，但要明确采样口径

当前已有：

- `current_qpos`
- `last_target_qpos`
- `hardware_refresh_time`
- `hardware_result_code`
- `hardware_result_text`
- `ik_metadata`

`Phase 1` 里约定：

- 如果数据集 arm state 走 `state_backend`，则 `hardware_refresh_time` 作为 source timestamp
- 如果数据集 arm state 走 `command/preview backend`，则当前还缺少显式 `last_command_timestamp`
- `current_qpos` 是当前关节状态
- `last_target_qpos` 是 replay backup 字段

因此这里建议新增：

- `arm_state_source`
- `last_command_timestamp`

### Done 条件

- assembler 不需要通过猜 key 名来判断 action/source/state 含义

## 5.2 T2 设备配置层 + RGB source

### 目标

去掉旧代码里散落的 RealSense serial 硬编码。

### 建议新增

- `teleop_runtime/configs/devices/realsense.json`
- `teleop_runtime/sources/camera/realsense_rgb.py`

### 建议配置格式

```json
{
  "front": {
    "serial": "017322074878",
    "width": 1280,
    "height": 720,
    "fps": 30
  }
}
```

### 实现要求

- source 内部维护一个很小的最新 RGB frame 缓存
- 提供 camera timestamp / receive timestamp / frame counter
- 每条 sample 只匹配并保留与该 sample 时间最接近的那一帧
- 不额外保留未被 sample 选中的其它视频帧
- 不把大图像塞进 `ObservationChannel`

### Done 条件

- 单相机长时间输出稳定
- 不阻塞 teleop 主循环

## 5.3 T3 独立 collector runtime 组装层

### 目标

实现一套独立 collector runtime。

它不直接依赖当前 debug 入口脚本的主循环，但尽量复用 teleop runtime 的底层接口和组件。

### 建议文件

- `teleop_runtime/collection/build_collect_components.py`
- `teleop_runtime/collection/runtime.py`

或直接内聚到：

- `teleop_runtime/collection/run_collect.py`

### 责任

- 加载 `TeleopCalibration`
- 复用当前已经稳定的接口级组件：
  - `VivePoseSource`
  - `KeyboardButtonSource`
  - `HandQ6Source.from_bimanual(...)`
  - `PoseToArmRetargeter(left/right)`
  - `JointsToDexHandRetargeter(left/right)`
  - `RM65ArmBackend(command/preview/state)`
  - `InspireTacHandBackend(left/right)`
  - `TeleopSession`

### Done 条件

- `left`、`right`、`both` 三种模式都能正常启动
- 单臂时未采集 side 不参与硬件链路，但 dataset schema 仍完整保留
- collector runtime 不直接依赖 debug 入口脚本主循环

## 5.4 T4 `SampleAssembler`

### 目标

把 dataset row 构建逻辑集中到一个模块。

### 文件

- `teleop_runtime/collection/sample_assembler.py`

### 输入

- `SessionStepResult`
- `TeleopCalibration`
- `left/right RM65ArmBackend.get_state()`
- `left/right InspireTacHandBackend.get_state()`
- `RealSenseRGBSource.get_latest_frame()`
- 当前 episode 上下文

### 输出

- 一条内存中的 sample
- 包含：
  - `observation.state`
  - `action`
  - 各模态时间戳
  - `age_sec`
  - 视频引用
  - replay backup 字段

### 关键责任

- 从 `step_result.command.commands` 中抽左右 arm / hand action
- 从 backend state 中抽左右 joint / tactile / replay joint target
- 显式写出本条样本的 `arm_state_source`
- 从 VIVE channel metadata 中抽 `sender_timestamp / receiver_timestamp`
- 对未采集 side 统一填充空值策略：
  - 固定低维向量填 `NaN`
  - 独立 side 字段写 `null`
- 统一生成：
  - `index`
  - `frame_index`
  - `episode_index`

### Done 条件

- 在不写盘的情况下可单测

## 5.5 T5 `LeRobotV21Writer`

### 目标

实现不依赖官方包的 LeRobot 风格 writer，并通过并行 writer pipeline 避免阻塞控制线程。

### 文件

- `teleop_runtime/collection/schema_v21.py`
- `teleop_runtime/collection/async_writer.py`
- `teleop_runtime/collection/writer_v21.py`
- `teleop_runtime/collection/stats.py`

### 产物

- parquet
- mp4
- `meta/info.json`
- `meta/episodes.jsonl`
- `meta/tasks.jsonl`
- `meta/episodes_stats.jsonl`

### 重要约束

- 视频像素不进 parquet
- parquet episode-per-file
- 数据集 `fps` 表示 nominal fps，不表示真实帧间隔
- 真实时间以样本 `timestamp` 为准
- 写盘线程和控制线程解耦
- 队列必须有 backpressure 策略

### Done 条件

- 手工造一个假 episode 可以完整落盘

## 5.6 T6 `run_collect.py`

### 目标

新增一个支持 `left/right/both` 的真机采集入口。

### 文件

- `teleop_runtime/collection/run_collect.py`

### 负责的事情

- 载入 calibration profile
- 做 session home capture
- 初始化 `active_sides` 对应的 arm / hand / tactile / RGB
- 运行 `TeleopSession.run_step()`
- 拉取 arm state / hand state / RGB
- 调 `SampleAssembler`
- 把样本交给异步 writer pipeline
- 默认并行开启 `EpisodeRecorder`

### 推荐 CLI 参数

- `--task`
- `--task-description`
- `--active-sides`
- `--load-profile`
- `--left-arm-ip`
- `--right-arm-ip`
- `--left-hand-ip`
- `--right-hand-ip`
- `--left-hand-port`
- `--right-hand-port`
- `--camera-config`
- `--with-tactile`
- `--hand-tactile-regions`
- `--dataset-root`
- `--nominal-fps`
- `--max-steps`
- `--record-debug-episode`

### Done 条件

- 真机 `left/right/both` 任一种链路上都能录出一个最小 episode

## 5.7 T7 replay bridge / 校验

### 目标

尽快验证当前 action 设计是可 replay 的。

### 现实约束

当前已有 replay 工具：

- `teleop_runtime/debug/run_recorded_episode_replay.py`

它读的是：

- `EpisodeRecorder` 格式

不是 LeRobot 数据集。

### 因此 P1 推荐方案

二选一都可以：

#### 方案 A

采集时继续并行写：

- `EpisodeRecorder`

这样：

- 当前 replay 工具不需要改

#### 方案 B

新增：

- `replay_bridge.py`

把 LeRobot episode 转成当前 replay 所需最小 JSON episode。

### Done 条件

- 至少有一条 P1 数据可 replay 验证

## 5.8 T8 验收与回归

### 结构验收

- LeRobot 目录结构正确
- metadata 文件齐全
- parquet 可读

### 时序验收

- 实际样本时间戳单调、完整
- 实际平均采样频率与真实执行频率一致
- `dataset nominal fps` 仅作为 metadata 参照，不与实际采样频率冲突
- 各模态时间戳完整

### 性能验收

- 采集不明显拖慢 teleop
- tactile 不阻塞主控制环

### replay 验收

- 至少一条 episode 可被 replay 或桥接 replay

---

## 6. 当前明确的处理策略

这一节是本次修订最重要的结论。

## 6.1 `PoseToArmRetargeter` metadata 命名和语义不完全一致

当前代码里：

- `target_pose_desk` 是机器人侧 target pose in desk
- `target_pose_operator` 更像操作者 wrist source pose in operator

这会直接误导 dataset writer。

处理结论：

- 在 runtime 里补新的明确字段
- assembler 不再直接依赖 `target_pose_operator` 这种歧义命名

本次文档冻结为新增：

- `source_pose_operator`
- `source_pose_desk`
- `target_pose_desk`
- `target_pose_base`
- `home_human_pose_desk`
- `home_robot_pose_desk`

## 6.2 `InspireTacHandBackend` 缺 telemetry 时间戳

处理结论：

- tactile 时间戳按“成功读到本次触觉数据的请求发送时刻”定义
- telemetry cycle 完成时刻单独记为 `telemetry_timestamp`

具体口径：

- `tactile_timestamp = t_request_sent(read_tac_data)`，前提是本次请求最终成功返回并写入缓存
- `telemetry_timestamp = t_cycle_finish`
- `age_sec = sample_timestamp - tactile_timestamp`

## 6.3 `RM65ArmBackend` 当前没有 arm force

旧版文档把：

- `right_arm_6d_force`

写进了 P1 的 `observation.state`。

但当前代码没有这条数据。

处理结论：

- `arm force` 从 P1 基线移除
- 后续如果 RM65 SDK 补齐稳定力读取，再作为增强项重新评估

## 6.4 当前 replay 工具和 LeRobot 数据集不是同一格式

当前：

- replay 工具绑定 `EpisodeRecorder`

未来：

- collector 输出 LeRobot 数据集

处理结论：

- 两者并存
- 两者独立实现
- 不能假设“写完 LeRobot 数据集就自动能被当前 replay 脚本读取”

## 6.5 当前 calibration preset 是双臂默认，不是单臂专用

当前默认构造入口：

- `build_annotated_dual_rm65_calibration(...)`

处理结论：

- P1 先整体按双臂实现
- 单臂采集时未采集 side 的数据字段置空
- dataset meta 明确记录 `active_sides = left/right/both`

因此当前直接接受双臂 preset：

- profile 里保留左右臂字段
- collector 根据 `active_sides` 决定实际驱动哪一侧
- schema 不再为了单臂额外分叉

## 6.6 runtime 里还没有 camera config / camera source

这意味着：

- 旧版文档里写的 `realsense.json`
- `RealSenseRGBSource`

仍然需要新增。

而且当前仓库里相关 serial 硬编码还散落在：

- `hardware/handeye/handeye_utils.py`

处理结论：

- 继续按新增任务处理
- 这部分不是“接线工作”，而是明确的新任务

## 6.7 当前真机 runner 的 hand telemetry 频率口径不适合直接照搬到数据采集

当前真机 runner 里：

- 已经提供：
  - `--hand-telemetry-hz`
  - `--hand-tactile-regions`

因此这部分已经不再是“能力缺失”，而是“dataset 如何定义时间语义”的问题。

处理结论：

- `dataset nominal fps` 只作为参照 metadata
- 数据采集按实际执行频率进行
- `telemetry_hz` 与 `dataset nominal fps` 解耦，并允许显式配置

## 6.8 理想 collector 不应直接依赖当前 debug 入口主循环

当前：

- `run_vive_rm65_real_teleop.py` 已经是稳定的真机 teleop 参考实现

但理想 collector：

- 应独立于这个脚本
- 不能把 debug 入口里的临时 glue 逻辑变成长期数据采集核心依赖

处理结论：

- 复用接口和组件
- 不复用 debug 入口主循环
- collector 自己维护采样/写盘/backpressure 生命周期

## 6.9 当前运行期 arm state 默认不是持续硬件测量态

当前 `run_vive_rm65_real_teleop.py` 已经把 episode metadata 写成：

- `runtime_arm_state_source = last_commanded_action`

并且启动阶段的硬件 state readers 在 calibration 后就会进入 idle。

这意味着当前代码基线下，dataset 里 arm observation 最自然的来源是：

- command/preview backend 的状态

而不是：

- 持续真实硬件测量态

处理结论：

- `Phase 1` 文档先承认这一现状
- collector 必须把 `arm_state_source` 写入样本或 meta
- 如果后续要持续采真实硬件态，作为单独增强项实现，不再默认假设“当前代码已经有”

## 6.10 高并行写盘需要明确 backlog 策略

如果 collector 只追求“尽量并行”，但不定义 backlog 语义，最终会得到不可控阻塞或隐式丢样本。

处理结论：

- 低维样本和 metadata 不允许静默丢失
- 视频任务允许单独配置更激进的并发和背压策略
- 所有 async writer 都必须暴露：
  - queue size
  - worker 异常
  - drain/flush 状态

## 6.11 `EpisodeRecorder` 的 episode 编号逻辑不要复用到 LeRobot writer

当前 `EpisodeRecorder` 默认是：

- `episode_000001`

而 LeRobot 风格通常更适合：

- `episode_000000`

处理结论：

- dataset writer 自己管理 episode index
- 不复用 `EpisodeRecorder._next_episode_dir()`

---

## 7. 推荐下一步

当前语义已经冻结：

1. arm observation 固定来自 `last_commanded_action`
2. 视频只保留每条 sample 命中的最近帧

因此更合理的下一步是：

1. 做 `T2`
   - `realsense.json`
   - `RealSenseRGBSource`
2. 继续补 `T3/T6`
   - collector runtime builder
   - `run_collect.py`
3. 用真机最小 episode 把 `T5` writer 链路接上
4. 最后做 `T7`
   - replay bridge 或 sidecar replay 验证

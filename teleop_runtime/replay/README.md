# DexTeleop Replay

`teleop_runtime.replay` 用于查看已有数据集，并可在通过检查后把数据集轨迹复现到 RM65 真机。它只做 replay，不创建新 episode，也不写采集数据。

## 启动 GUI

```bash
conda activate dexteleop310
cd /home/zc/HandMotion/Developing/DexTeleop

python -m teleop_runtime.replay \
  --host 127.0.0.1 \
  --port 8765 \
  --no-open-browser
```

浏览器打开：

```text
http://127.0.0.1:8765/
```

也可以启动时直接加载数据集：

```bash
python -m teleop_runtime.replay \
  --dataset-root /path/to/lerobot_dataset_root \
  --host 127.0.0.1 \
  --port 8765 \
  --no-open-browser
```

省略 `--episode` 会自动选择第一条可用 episode；若要指定，请使用数据集真实
`episode_index` 或 `episode_id`，例如 `--episode 10` 或 `--episode episode_000010`。

数据集根目录通常需要包含：

```text
meta/episodes.jsonl
data/chunk-*/episode_*.parquet
```

页面中执行：`Load` 数据集 -> 选择 episode -> `Play/Pause` 或拖动时间轴。若 episode 元数据或 parquet 行字段中记录了视频/触觉路径或 inline 触觉字段，摄像头与 DMTAC 触觉会自动显示。

## 真机轨迹 Replay

当前真机 replay 面向 `teleop_runtime.debug.run_vive_rm65_real_teleop`
采集的数据集。默认语义是：

- `Move Start`: 只使用首帧 qpos 做安全初始化。优先用
  `action.replay.<side>_arm_joint_target`；没有该字段时退回
  `observation.arm.<side>.current_qpos`，再退回
  `observation.arm.<side>.actual_hardware_qpos`。
  首帧 QPOS 初始化速度可通过启动参数或 GUI 的 `QPOS Init Vel`
  调整；传 1 个 rad/s 值应用到 6 个关节，也可传 6 个逗号分隔的逐关节 rad/s 值。
- `Start`: 按帧读取数据集 26 维 `action` 中 active side 的 7D
  `[x, y, z, qx, qy, qz, qw]` pose，构造成
  `TeleopCommand/EffectorCommand(cartesian_pose)`，再走 RM65 backend 的
  IK/限步/下发路径。

这一步用于验证后续模型预测的 action 是否能对齐真机下游接口。显式
`qpos_target` 模式仅作为低层调试路径，需要数据集中存在全帧 finite 的
`action.replay.<side>_arm_joint_target`。

## 末端 Replay

默认只 replay 机械臂。需要同时验证手/夹爪接口时，启动时打开
`--end-effectors-enabled`，并为每个 active side 指定末端 backend：

```bash
python -m teleop_runtime.replay \
  --dataset-root /path/to/lerobot_dataset_root \
  --active-sides both \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --left-tool-profile hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json \
  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json \
  --end-effectors-enabled \
  --hand-backend dmtac_gripper \
  --left-gripper-port /dev/ttyUSB_LEFT \
  --right-gripper-port /dev/ttyUSB_RIGHT \
  --host 127.0.0.1 \
  --port 8765 \
  --no-open-browser
```

`Move Start` 仍然只做机械臂首帧 qpos 初始化，不移动末端。`Start` 阶段
会从数据集 `action` 中读取 active side 的 hand slot，与 arm pose 一起构造
同一个 `TeleopCommand`，再由 replay hardware rig 按 `left_arm`,
`left_hand`, `right_arm`, `right_hand` 分发到对应 backend。

末端维度以 dataset metadata 中的 `hand_action_specs_by_side` 为准。DMTAC
gripper 当前使用 `dmtac_gripper` spec，将 6-slot 兼容 hand target 映射到
1 维 gripper target；如果缺少 spec，resolver 会退回 q6 手部默认语义，
这通常不适合夹爪部署，需要先补齐数据集 metadata 或做后处理转换。

当前通用 replay 动作构造集中在
`teleop_runtime/replay/action_frame.py`。该文件刻意不依赖真机硬件，只负责
把 dataset frame 转成 `RobotAction` / `TeleopCommand`；后续大规模整理时，
它应迁移到 sim/real 共享的 replay/action package，`teleop_runtime` 只保留
真机 backend、GUI 和安全执行壳。

右臂示例：

```bash
python -m teleop_runtime.replay \
  --dataset-root /path/to/lerobot_dataset_root \
  --active-sides right \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json \
  --host 127.0.0.1 \
  --port 8765 \
  --no-open-browser
```

双臂示例：

```bash
python -m teleop_runtime.replay \
  --dataset-root /path/to/lerobot_dataset_root \
  --active-sides both \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --left-tool-profile hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json \
  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json \
  --host 127.0.0.1 \
  --port 8765 \
  --no-open-browser
```

GUI 中按顺序操作：

1. 确认 `Hardware Replay` 面板里的 active side、IP、TCP profile。
2. 默认 `Arm Output` 关闭；确认安全后再勾选。
3. 点击 `Preflight`。GUI 会先应用当前表单配置，并在需要时自动 `Connect`。
4. 确认所有阻断项通过后，点击 `Move Start` 执行 qpos init plan 到轨迹首帧。
5. 点击 `Start`，按 `action` pose 复现轨迹并验证下游 action 接口。
6. 需要停止时点 `Abort`；replay 会请求 arm hold 并恢复 replay speed override。

不满足当前状态条件的按钮会自动置灰。例如未选 episode 时不能
`Preflight`，preflight 未通过时不能 `Move Start`，未到 `start_ready` 时不能
`Start`。`Connect` 按钮只保留为手动诊断入口，正常流程不需要单独点击。

## Preflight 检查

真机 replay 会在开始运动前检查：

- 数据集是否有有效轨迹。
- 默认 `action_pose` 模式下，active side 的 `action` pose 是否全帧 finite 且可转换为 pose matrix。
- 是否能为 active side 选择到 finite 的首帧 qpos init target。
- 当前真机 qpos 与选中的首帧 qpos init target 是否在允许范围内。
- 数据集记录的 tool profile TCP 与当前 GUI/CLI 配置的 TCP 数值是否一致。
- 当前 TCP 位姿到轨迹首帧位姿的距离会显示为提示信息。

`Preflight` 本身不执行轨迹运动；它只会连接/读取当前硬件状态并生成阻断项。
真正的机械臂运动从 `Move Start` 开始，真正的 action replay 从 `Start` 开始。

默认不会向真机发送动作。只有勾选 GUI 的 `Arm Output`，或启动时显式添加 `--send-actions`，才允许 `Move Start` 和 `Start` 下发命令。

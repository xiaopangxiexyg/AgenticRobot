# DexTeleop Replay

`teleop_runtime.replay` 用于查看已有数据集，并可在通过检查后把数据集轨迹复现到 RM65 真机。它只做 replay，不创建新 episode，也不写采集数据。

## 启动 GUI

```bash
conda activate dexteleop310
cd /home/password-1/DexTeleop

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
  --episode 0 \
  --host 127.0.0.1 \
  --port 8765 \
  --no-open-browser
```

数据集根目录通常需要包含：

```text
meta/episodes.jsonl
data/chunk-*/episode_*.parquet
```

页面中执行：`Load` 数据集 -> 选择 episode -> `Play/Pause` 或拖动时间轴。若 episode 元数据里有 `video_paths` 和 `tactile_paths`，摄像头与 DMTAC 触觉会自动显示。

## 真机轨迹 Replay

当前真机 replay 面向 `teleop_runtime.debug.run_vive_rm65_real_teleop`
采集的数据集。机械臂控制不使用 26 维 `action` pose 直接下发，而是使用：

```text
action.replay.left_arm_joint_target
action.replay.right_arm_joint_target
```

`Move Start` 会先用 `robot_motion` 的 qpos init planner 从当前真机 qpos 规划到轨迹首帧 qpos；`Start`
之后按帧把 recorded qpos target 包成 `TeleopCommand/EffectorCommand(joint_vector)`，再走
RM65 backend 下发路径。

右臂示例：

```bash
python -m teleop_runtime.replay \
  --dataset-root /path/to/lerobot_dataset_root \
  --episode 0 \
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
  --episode 0 \
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
3. 点击 `Connect`。
4. 点击 `Preflight`，确认所有阻断项通过。
5. 点击 `Move Start`，执行 qpos init plan 到轨迹首帧。
6. 点击 `Start`，按 `action.replay.<side>_arm_joint_target` 复现轨迹。
7. 需要停止时点 `Abort`；replay 会请求 arm hold 并恢复 replay speed override。

## Preflight 检查

真机 replay 会在开始运动前检查：

- 数据集是否有有效轨迹。
- 数据集中是否存在 `action.replay.<side>_arm_joint_target`，且帧数与 replay 轨迹一致、全帧 finite。
- 当前真机 qpos 与轨迹首帧 qpos 是否在允许范围内。
- 数据集记录的 tool profile TCP 与当前 GUI/CLI 配置的 TCP 数值是否一致。
- 当前 TCP 位姿到轨迹首帧位姿的距离会显示为提示信息。

默认不会向真机发送动作。只有勾选 GUI 的 `Arm Output`，或启动时显式添加 `--send-actions`，才允许 `Move Start` 和 `Start` 下发命令。

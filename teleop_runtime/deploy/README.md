# teleop_runtime.deploy

`teleop_runtime.deploy` 是独立于采集主循环和 `LeftArmFixed` 的策略部署入口，面向 RM65 单臂/双臂，可选 InspireTac 手和 RealSense。默认 dry-run；真实下发必须显式加 `--send-actions`。

部署频率兼容旧接口：

- 默认仍可用 `--hz` 指定控制/发送频率。
- 如果提供 `--debug-replay-dataset` / `--debug-replay-parquet`，且没有显式传 `--action-hz`，部署器会优先从数据集实际记录的 action 行间 `timestamp` / episode stats / `control_hz` 推断真实 action 执行频率，并按该频率发送 action。
- 也可以用 `--action-hz-dataset` 单独指定一个参考数据集来推断发送频率，而不进入 replay 模式。
- `--policy-hz` 可以省略；省略时，普通部署保持历史默认 `10Hz`，而 replay / 参考数据集频率模式会默认跟随推断出的 action 频率。

## Action / Observation Layout

部署器兼容 `teleop_runtime.collection` 的数据布局：

- `action`: 26 维 `[left_arm_pose7, right_arm_pose7, left_hand_q6, right_hand_q6]`
- `observation.state`: 38 维 `[left_qpos6, left_pose7, left_hand6, right_qpos6, right_pose7, right_hand6]`
- arm pose 是 base frame 下的 `[x, y, z, qx, qy, qz, qw]`

策略输出支持：

- `np.ndarray` shape `(26,)` 或 `(chunk, 26)`
- 单臂 active 时 shape `(7,)` 或 `(chunk, 7)`，表示该侧 arm pose
- 双臂 active 时 shape `(14,)` 或 `(chunk, 14)`，顺序 `left,right`
- hand-only shape `(12,)`，顺序 `left,right`
- dict: `{"action": ...}` 或 `{"action_chunk": ...}`
- dict: `{"left_arm_pose": pose7, "right_arm_pose": pose7, "left_hand": q6, "right_hand": q6}`
- dict: `{"left_arm_qpos": q6, "right_arm_qpos": q6}` 可直接下发 joint target

## 单臂 Dry-Run

```bash
python -m teleop_runtime.deploy.run_policy \
  --policy hold \
  --active-sides right \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --max-steps 200
```

未传 `--auto-start` 时启动后按 `g` 才执行策略；按 `h` 暂停并清空 action chunk，按 `q` 退出。

## 单臂真实下发

```bash
python -m teleop_runtime.deploy.run_policy \
  --policy /abs/path/my_policy.py:Policy \
  --policy-kwargs-json '{"checkpoint": "/abs/path/model.pt"}' \
  --active-sides right \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --send-actions
```

## 双臂 + 双手

```bash
python -m teleop_runtime.deploy.run_policy \
  --policy /abs/path/my_policy.py:create_policy \
  --active-sides both \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --enable-hands \
  --left-hand-ip "$LEFT_HAND_IP" \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --hand-tactile-regions all \
  --send-actions
```

## Dataset Replay Debug

可以直接 replay `teleop_runtime.collection` 采集出的 action。默认仍是 dry-run；真实下发仍需 `--send-actions`。
debug replay 在 `--action-queue-policy auto` 下会自动 append 预取的 action
chunk，避免按 `--min-action-queue` 补队列时跳过轨迹尾部；普通模型 policy 仍默认
replace pending queue。

```bash
python -m teleop_runtime.deploy.run_policy \
  --debug-replay-dataset /abs/path/to/dataset \
  --debug-replay-episode 0 \
  --active-sides both \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --auto-start
```

上面命令在未显式传 `--action-hz` 时，会自动读取该 episode 的真实频率并按接近真机的节奏发送/消费 action。

## 用真机数据集校准部署频率

```bash
python -m teleop_runtime.deploy.run_policy \
  --policy /abs/path/my_policy.py:Policy \
  --policy-kwargs-json '{"checkpoint": "/abs/path/model.pt"}' \
  --active-sides right \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --action-hz-dataset /abs/path/to/real_dataset \
  --action-hz-episode 12 \
  --send-actions
```

## 安全边界

- arm cartesian pose 会按 `--arm-workspace-min/--arm-workspace-max` 裁剪位置。
- arm pose 每 tick 默认限速：`--max-arm-xyz-step-m 0.03`、`--max-arm-rot-step-deg 8`。
- arm joint target 每 tick 默认限速：`--max-arm-joint-step-rad 0.08`。
- hand q6 会 clip 到 `[0, 1]`，并由 `--max-hand-step` 限速。
- `--require-ready-observation` 会在 active state 或图像缺失时跳过策略和下发。
- `--inspire-hand-max-force` 默认继承 `teleop_runtime` 的 `800`。

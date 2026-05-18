# OpenPI05 真机部署说明

这个目录提供了 OpenPI 到 `teleop_runtime` 的部署适配层，当前面向两类 Realman 本体：

- `realman_dual_arm_hands`
- `realman_right_arm_right_hand`

同时兼容它们的别名输入：

- `realman_dual_arm_hand`
- `realman_right_arm_hand`

当前实现以单相机部署为前提，默认读取 `front`。

## 0. realman setting:

```bash
export REPO=$(pwd)
export PROFILE=/home/ai/Desktop/DexTeleop/teleop_records/calibration_profiles/20260506_122501_vive_rm65_virtual_desk.json
export CAMERA_CONFIG=$REPO/teleop_runtime/configs/devices/realsense.json
export DATASET_ROOT=$REPO/datasets
export VIVE_TARGET_HOST=192.168.20.90

export LEFT_ARM_IP=192.168.101.21
export RIGHT_ARM_IP=192.168.1.20
export LEFT_HAND_IP=192.168.12.210
export RIGHT_HAND_IP=192.168.11.210
export LEFT_HAND_PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130705-if00
export RIGHT_HAND_PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130707-if00
```


## 1. 目录职责与环境分工

本目录的职责是保持 `teleop_runtime.deploy.run_policy` 主循环不大改，只在策略侧补上这些适配：

- OpenPI websocket metadata 校验
- 真机观测到 OpenPI 请求格式的转换
- action chunk 顺序推理与顺序消费
- 带触觉 / 不带触觉的严格兼容检查
- 基于数据集轨迹的 startup initial-state
- 基于数据集的调试输入模式

当前推荐的环境分工如下：

- OpenPI server：`/home/ai/zc/openpi`，使用 `uv`
- 真机 client：`/home/ai/Desktop/DexTeleop`，使用 `conda activate grasp310_new`

虽然 server 和 client 在同一台机器上运行，但仍然按 client-server 方式通信，地址默认使用
`127.0.0.1:8000`。

## 2. 核心控制语义

部署时最关键的行为约束如下：

- 每次推理向 server 请求一个完整 action chunk。
- chunk 内的动作按真机控制频率连续下发。
- 当前 chunk 执行完后，才开始下一次推理。
- chunk 之间允许存在推理间隙，不强求无缝拼接。
- 这样可以保证部署时的动作消费方式和训练时的 chunk 监督方式尽量一致。

如果 server 对应模型要求触觉，而真机侧任一必需 tactile token 缺失：

- dry-run 可按策略配置放宽
- 正式下发动作时会进入 `policy error -> pause`
- 随后关闭 client 连接并退出，避免继续执行

## 3. `use_dataset_initial_state` 默认行为

`use_dataset_initial_state` 现在默认就是 `true`。

这意味着：

- 只要同时提供了 `dataset_root`
- 并且部署命令带了 `--init-source auto`

client 就会在启动阶段读取选中轨迹的首帧 `observation/state`，通过已有的 startup init 逻辑，
让机械臂平滑插值到这条轨迹的起始状态，再进入策略循环。

注意两点：

- 这里使用的是数据集的首帧 `observation/state`，不是首个 action target。
- 如果没有提供 `dataset_root`，即便 `use_dataset_initial_state=true`，也不会强行报错，只是不会生成
  dataset startup init。

## 4. Server / Client 交互流程

推荐按两个终端来运行。

### 终端 A：启动 OpenPI server

```bash
cd /home/ai/zc/openpi

# 首次部署或依赖有变化时执行
uv sync

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/serve_policy.py \
  --host=127.0.0.1 \
  --port=8000 \
  --warmup \
  policy:checkpoint \
  --policy.dir=/home/ai/zc/openpi/checkpoints/<manifest_config>/<exp_name>/<step>
```

paper fold:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/serve_policy.py \
  --host=127.0.0.1 \
  --port=8000 \
  --warmup \
  policy:checkpoint \
  --policy.dir=/home/ai/zc/openpi/checkpoints/pi05_realman_dual_hands_paper_fold_0428_three_2026_04_28_large_brown_paper_fold_2026_04_28_small_brown_paper_fold_tactile_manifest/paper_fold_0428_three_tactile_lora/15000
```

说明：

- 同机部署时固定使用 `127.0.0.1` 即可。
- `--default-prompt` 可省略；省略后 server 会继续使用 checkpoint 中保存的默认 prompt。
- 老 checkpoint 如果没有 manifest，可能还需要显式传 `--policy.config=...`。
- 如果 checkpoint 内没有随附 norm stats，可能还需要补 `--norm-stats-dir=...`。

server 启动后，可以单独检查 metadata：

```bash
curl http://127.0.0.1:8000/metadata
```

建议重点核对：

- `deployment.checkpoint_dir`
- `deployment.checkpoint_step`
- `local_robot.embodiment`
- `local_robot.output_action_dim`
- `model.action_horizon`
- `model.tactile_kind`
- `model.tactile_tokens`

### 终端 B：启动 DexTeleop client

```bash
source /home/ai/miniconda3/etc/profile.d/conda.sh
conda activate grasp310_new
cd /home/ai/Desktop/DexTeleop
```

先做 dry-run 更安全。下面是双臂双手单相机的推荐基础命令：

```bash
python -m teleop_runtime.deploy.run_policy \
  --policy /home/ai/Desktop/DexTeleop/openpi05_deployment/policy.py:create_policy \
  --policy-kwargs-json '{
    "host":"127.0.0.1",
    "port":8000,
    "embodiment":"realman_dual_arm_hands",
    "camera_key":"front",
    "dataset_root":"/path/to/dataset"
  }' \
  --active-sides left right \
  --enable-hands \
  --camera-mode single \
  --camera-config /home/ai/Desktop/DexTeleop/teleop_runtime/configs/devices/realsense.json \
  --require-camera-frame \
  --require-ready-observation \
  --action-queue-policy replace \
  --min-action-queue 0 \
  --init-source auto \
  --init-time-s 6.0 \
  --arm-workspace-min -0.60 -0.80 0.00 \
  --arm-workspace-max 0.60 0.80 0.90
```

paper fold:

```bash
python -m teleop_runtime.deploy.run_policy \
  --policy /home/ai/Desktop/DexTeleop/openpi05_deployment/policy.py:create_policy \
  --policy-kwargs-json '{
    "host":"127.0.0.1",
    "port":8000,
    "embodiment":"realman_dual_arm_hands",
    "camera_key":"front",
    "dataset_root":"/home/ai/zc/datasets/paper_fold_datasets/2026-04-28_black_paper_fold"
  }' \
  --active-sides both \
  --enable-hands \
  --camera-mode single \
  --camera-config /home/ai/Desktop/DexTeleop/teleop_runtime/configs/devices/realsense.json \
  --require-camera-frame \
  --require-ready-observation \
  --action-queue-policy replace \
  --min-action-queue 0 \
  --init-source auto \
  --init-time-s 6.0 \
  --arm-workspace-min -0.60 -0.80 0.00 \
  --arm-workspace-max 0.60 0.80 0.90 \
  --send-actions \
  --auto-start \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --left-hand-ip "$LEFT_HAND_IP" \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --inspire-tac-max-fingertip-force 800
```

如果确认状态正常、准备真实执行，则在命令末尾追加：

```bash
--send-actions \
--left-arm-ip "$LEFT_ARM_IP" \
--right-arm-ip "$RIGHT_ARM_IP" \
--left-hand-ip "$LEFT_HAND_IP" \
--right-hand-ip "$RIGHT_HAND_IP" \
--inspire-tac-max-fingertip-force 800 \
```

右臂右手版本只需要改两处：

- `embodiment` 改成 `realman_right_arm_right_hand`
- `--active-sides` 改成 `right`

也就是说：

```bash
--policy-kwargs-json '{"host":"127.0.0.1","port":8000,"embodiment":"realman_right_arm_right_hand","camera_key":"front","dataset_root":"/path/to/dataset"}'
--active-sides right
```

## 5. 为什么这样能保证 client-server 语义正确

当前交互顺序是：

1. client 连接 server，先读取 metadata
2. client 校验 embodiment、action horizon、触觉需求、输出维度
3. client 组织当前观测并发送给 server
4. server 返回一个完整 chunk
5. client 将 chunk 缓存起来
6. client 按真机频率逐条下发该 chunk
7. chunk 清空后，再发起下一次推理

这保证了：

- 推理结果不会被半截打断
- chunk 内动作的时间语义是稳定的
- 部署时的动作消费方式与训练监督方式保持一致

## 6. 数据集驱动的两种调试模式

除了 `input_source="live"`，还支持两种调试模式。

### `dataset_compare`

作用：

- 不用真机观测作为 policy 输入
- 改为读取数据集轨迹观测送给 server
- 将 server 输出和数据集 GT action 做误差比较
- 不允许 `--send-actions`

示例：

```bash
python -m teleop_runtime.deploy.run_policy \
  --policy /home/ai/Desktop/DexTeleop/openpi05_deployment/policy.py:create_policy \
  --policy-kwargs-json '{
    "host":"127.0.0.1",
    "port":8000,
    "embodiment":"realman_dual_arm_hands",
    "input_source":"dataset_compare",
    "dataset_root":"/home/ai/zc/datasets/paper_fold_datasets/2026-04-28_black_paper_fold",
    "dataset_episode":12
  }' \
  --init-source none \
  --auto-start \
  --enable-hands
```

`dataset_compare` 是纯离线对比模式，不需要把机器人移动到数据集初始位姿。
因此建议使用 `--init-source none --auto-start`，避免在没有真机状态时等待 startup init。

### `dataset_execute`

作用：

- 同样使用数据集轨迹观测作为 policy 输入
- 但会把 server 返回的 chunk 真实执行到机器人
- 支持连续执行多个 chunk

示例：

```bash
python -m teleop_runtime.deploy.run_policy \
  --policy /home/ai/Desktop/DexTeleop/openpi05_deployment/policy.py:create_policy \
  --policy-kwargs-json '{
    "host":"127.0.0.1",
    "port":8000,
    "embodiment":"realman_dual_arm_hands",
    "input_source":"dataset_execute",
    "dataset_root":"/home/ai/zc/datasets/paper_fold_datasets/2026-04-28_black_paper_fold",
    "dataset_episode":12
  }' \
  --active-sides left right \
  --init-source auto \
  --auto-start \
  --enable-hands \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --left-hand-ip "$LEFT_HAND_IP" \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --inspire-tac-max-fingertip-force 800 \
  --send-actions
```

`dataset_execute` 会在 `--init-source auto` 下优先使用 policy 提供的 dataset 初始状态：
先把机器人平滑移动到 `dataset_start_frame` 对应的 `observation/state`，初始化完成后再从同一帧数据集观测开始向 server 推理，并连续执行返回的 action chunk。
不加 `--auto-start` 时需要手动按 `g` 开始；加 `--send-actions` 时必须提供对应 active side 的 arm/hand IP。
`--inspire-tac-max-fingertip-force` 会传到 `InspireTacModbus(max_fingertip_force=...)`，启动时写入六个手指的 force limit；默认是 `800`，也兼容简写 `--hand-max-force`、`--max-force`。
真机 `--send-actions` 默认启用二段确认：到达初始状态后会停在 `ready=wait:A`，按 `A` 后才开始 policy 推理和动作执行。
如需恢复旧行为，可添加 `--no-wait-for-ready-confirm`。
默认初始化完成阈值是 `30mm / 20deg`；如果需要更严格或更宽松，可以用 `--init-arm-position-tolerance-m` 和 `--init-arm-rotation-tolerance-deg` 显式覆盖。

共享的常用策略参数包括：

- `dataset_root`
- `dataset_episode`
- `dataset_episode_selection`：`specified` 或 `random`
- `dataset_start_frame`
- `dataset_camera`
- `dataset_prompt`
- `dataset_loop`
- `dataset_seed`
- `dataset_stride`

运行时状态栏会显示：

- 当前 episode / frame
- server 推理耗时
- 与数据集 GT 的误差摘要

注意：这两个模式当前替换的是 policy 输入，但最外层 deploy loop 仍可能继续轮询已连接的硬件后端。
如果后续要做到“完全不读真机输入”，还需要继续重构 `run_policy` 的观测来源。

## 7. 数据采集时的工作空间默认值

如果采集时没有显式提供 calibration JSON，当前采集侧默认工作空间是：

- `workspace_min = (-0.60, -0.80, 0.00)`
- `workspace_max = (0.60, 0.80, 0.90)`

对应代码位置：

- [teleop_runtime/configs/presets.py](/home/ai/Desktop/DexTeleop/teleop_runtime/configs/presets.py:30)
- [teleop_runtime/collection/run_collect.py](/home/ai/Desktop/DexTeleop/teleop_runtime/collection/run_collect.py:1690)

有一个容易混淆的例外：

- 采集侧固定轨迹回放时，内部会临时使用 `(-10, -10, -10)` 到 `(10, 10, 10)` 的超大范围
- 这是 replay-only 的放宽设置，不是正常采集默认值

对应位置：

- [teleop_runtime/collection/run_collect.py](/home/ai/Desktop/DexTeleop/teleop_runtime/collection/run_collect.py:4632)

因此，真机部署时建议把：

```bash
--arm-workspace-min -0.60 -0.80 0.00
--arm-workspace-max 0.60 0.80 0.90
```

至少先作为默认起点，再根据你真实采集所用 calibration 做一致化调整。

## 8. 其他注意事项

- `openpi_client` 现在最多等待 server 10 分钟，避免无限期卡死。
- tactile 是否必需，由 server metadata 决定；client 会在连接后严格检查。
- 当前 README 默认假设单相机部署，`camera_key` 默认 `front`。
- 正式真机执行前，建议遵循：`metadata 检查 -> dry-run -> dataset_compare -> dataset_execute -> live + --send-actions`。

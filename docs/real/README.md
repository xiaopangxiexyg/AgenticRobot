# 真实系统运行入口

真实输出前必须完成设备检查、标定、工作空间和急停确认。遥操与数采参数见
[TELEOP_COLLECTION](./TELEOP_COLLECTION.md)。

| 专题 | 用途 |
|---|---|
| [TELEOP_COLLECTION](./TELEOP_COLLECTION.md) | 启动顺序、正式 profile 参数、数采验收和回退 |
| [REPLAY](./REPLAY.md) | 离线/真机 replay 参数和安全门 |
| [DEPLOY](./DEPLOY.md) | 通用 deploy 合约和验收 |
| [OPENPI](./OPENPI.md) | OpenPI 两进程部署 |
| [UMI](./UMI.md) | UMI 配置、标定和数采 |

## 0. 运行前确认

- 从仓库根目录使用 Bash，并已激活 `dexteleop310`。
- 已完成 [设备只读检查](../hardware/README.md)；左右 side、RM65 IP、tool/TCP、Q6、DMTAC
  和相机配置与现场一致。
- [`teleop_args.sh`](./teleop_args.sh) 是当前站点参数，不是通用默认值。正式数采前检查其中的 CPU 编号、数据目录
  和设备默认值；当前远端 Ryzen 9 9950X 是 16 个物理核心/32 个逻辑 CPU，
  配方使用 `0-15` 中每个物理核心的一个线程，`16-31` 留给系统和其他进程。
- H2/M 命令必须有现场操作员、可用急停和已清空的工作空间。

## 1. Human teleop 与正式数采

### 1.1 VIVE sender（H1）

```bash
python hardware/VIVE/mocap/vive_data_sender.py \
  --host "${VIVE_RECEIVER_IP:?set VIVE_RECEIVER_IP}" \
  --port 9999 \
  --hz 300
```

`VIVE_RECEIVER_IP` 是运行接收端的主机地址；sender 与接收端不在同一机器时尤其不能沿用
别人的 IP。

### 1.2 GUI composition root（M）

```bash
python -m teleop_runtime.debug.run_vive_rm65_real_teleop --gui
```

### 1.3 正式 position 数采（M）

```bash
cd /home/user/zc/DexTeleop
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate /home/user/miniconda3/envs/dexteleop310
source docs/real/teleop_args.sh

taskset -c 0-7 \
python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  "${COMMON_REAL_TELEOP_ARGS[@]}" \
  "${DMTAC_CAPTURE_COMMON_ARGS[@]}" \
  "${DMTAC_A_ARGS[@]}" \
  "${DMTAC_TACTILE_WORKER_2CPU_ARGS[@]}" \
  "${DATASET_COMMON_ARGS[@]}" \
  "${POSITION_FORMAL_V1_ARGS[@]}" \
  "${POSITION_COLLECTION_PROCESS_ARGS[@]}" \
  "${POSITION_COLLECTION_PRODUCTION_ARGS[@]}" \
  --dataset-root "$DATASET_BASE" \
  --collection-task-name dmtac_rm65_bimanual_position_v1 \
  --collection-task-description "formal rm65 bimanual position v1: accepted-qref FK action, IK 1.0/0.35 min2/max4, 15/60/60/120" \
  --q6-config latest --q6-config-required \
  --dmtac-config latest --require-dmtac-config \
  --camera-config cameras/configs/rm65/devices/realsense.json \
  --write-parquet --write-video
```

### 1.4 Impedance/力控 15→40 Hz 数采（M）

```bash
cd /home/user/zc/DexTeleop
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate /home/user/miniconda3/envs/dexteleop310
source docs/real/teleop_args.sh

taskset -c 0-11 \
python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  "${COMMON_REAL_TELEOP_ARGS[@]}" \
  "${DMTAC_CAPTURE_COMMON_ARGS[@]}" \
  "${DMTAC_A_ARGS[@]}" \
  "${DATASET_COMMON_ARGS[@]}" \
  "${IMPEDANCE_40HZ_ARGS[@]}" \
  "${CONTROL_DEBUG_SUMMARY_ARGS[@]}" \
  --dataset-root "$DATASET_BASE" \
  --collection-task-name dmtac_impedance_15_40hz \
  --collection-task-description "full-device bimanual impedance 15Hz capture / 40Hz backend" \
  --q6-config latest --q6-config-required \
  --dmtac-config latest --require-dmtac-config \
  --camera-config cameras/configs/rm65/devices/realsense.json \
  --write-parquet --write-video
```

### 1.5 无 collection、无 debug 写盘的卡顿基线（M）

这个命令保留双臂、Q6、DMTAC 夹爪/触觉和 RGB 相机，但不创建 dataset
writer、不启动腕部 depth stream，也不写 standalone debug 文件。它只在终端输出
超过 66 ms 的慢帧和主循环 stall，用于判断卡顿是否仍存在于控制/传感器链路。
慢帧文本通过有界后台队列输出，终端或 SSH 反压不会在遥操控制线程内执行
同步 `flush`。

```bash
cd /home/user/zc/DexTeleop
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate /home/user/miniconda3/envs/dexteleop310
source docs/real/teleop_args.sh

taskset -c 0-7 \
python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  "${COMMON_REAL_TELEOP_ARGS[@]}" \
  "${DMTAC_CAPTURE_COMMON_ARGS[@]}" \
  "${DMTAC_A_ARGS[@]}" \
  "${DMTAC_TACTILE_WORKER_2CPU_ARGS[@]}" \
  "${POSITION_FORMAL_V1_ARGS[@]}" \
  "${NO_COLLECTION_SENSOR_BASELINE_ARGS[@]}" \
  --q6-config latest --q6-config-required \
  --dmtac-config latest --require-dmtac-config \
  --camera-config cameras/configs/rm65/devices/realsense.json \
  "${CONSOLE_TIMING_ARGS[@]}" \
  "${NO_DEBUG_DISK_WRITE_ARGS[@]}"
```

不要在这个基线命令中加入 `DATASET_COMMON_ARGS`、
`POSITION_COLLECTION_PROCESS_ARGS`、`--write-parquet` 或 `--write-video`。
`CONSOLE_TIMING_ARGS` 已由 `teleop_args.sh` 定义；不要用未定义的 Bash 数组占位。

### 1.6 Headless/兼容 collection（先 R0）

先看帮助：

```bash
python -m teleop_runtime.collection --help
```

### 1.7 UMI 配置验证（R0）

要求左右 hand-eye 标定已存在：

```bash
python -m teleop_runtime.collection umi \
  --config teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json \
  --validate-only
```

详细 UMI 流程见 [UMI](./UMI.md)。

## 2. Replay

离线 Replay GUI（R1）：

```bash
python -m teleop_runtime.replay \
  --dataset-root "${DATASET_ROOT:?set DATASET_ROOT}" \
  --host 127.0.0.1 \
  --port 8765 \
  --no-open-browser
```

真机 replay 仍使用同一入口，但必须显式配置 arms/tools/backend，并在 GUI 中人工执行
`Move Start` 和 `Start`。参数和安全边界见 [REPLAY](./REPLAY.md)。MuJoCo replay 放在
[仿真入口](../sim/README.md)，不混在真实系统命令中。

## 3. Deploy

Canonical CLI 帮助（R0）：

```bash
python -m teleop_runtime.deploy --help
```

OpenPI artifact contract 检查（R0）：

```bash
python -m deployments.openpi.client.real \
  --checkpoint-dir "${CHECKPOINT_DIR:?set CHECKPOINT_DIR}" \
  --contract-check-only
```

实机观测和推理、不发送动作（H1）：

```bash
python -m deployments.openpi.client.real \
  --checkpoint-dir "${CHECKPOINT_DIR:?set CHECKPOINT_DIR}" \
  --deployment-profile rm65_bimanual_dmtac \
  --host "${POLICY_SERVER_HOST:?set POLICY_SERVER_HOST}" \
  --port 8078 \
  --max-steps 100
```

只有完成 qpos-init、ready gate 和人工确认后才允许增加 `--send-actions`。通用说明见
[DEPLOY](./DEPLOY.md)，OpenPI 两进程流程见 [OPENPI](./OPENPI.md)。

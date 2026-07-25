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

进入 runtime Home 前，系统会用已标定的 VIVE desk 横向轴检查左右 wrist tracker：
三轴标定中的“身体右侧 → 身体左侧”定义为操作者坐标 `+Y`。检查会先去除配置的
`virtual_desk_post_yaw_deg`，因此默认 `180°` 场景旋转不会把左右判断颠倒。
正常情况要求 logical left tracker 的操作者坐标 Y 大于 logical right；若顺序相反，
或双手横向间距小于 5 cm 无法可靠判断，GUI 会在 Home 真正开始前弹窗并要求二次
确认。建议取消后点击“交换左右 Tracker”并重试；选择继续不会自动交换机械臂、
Q6、真手或任何持久配置。
双侧样本还必须有效且接收时间不超过 0.5 秒；按下开始后会等待首个可判定的新鲜
结果，并在准备倒计时结束、真正采集 Home 前再复核一次。运行中双臂映射已锁定时，
弹窗不会提供不可用的“交换”建议，而是要求恢复左右手位置，或安全退出后重新启动。
左右关系检查只在当前 GUI 进程第一次开始遥操时执行一次：首次通过或经人工二次
确认后，本进程后续的普通 Home、Q6 重采和双踏板切换不再重复计算左右关系，避免
把它变成持续运行监控。后续 Home 仍会独立检查 Tracker 新鲜度和稳定性。首次操作
若来自双踏板 Hold 恢复，则左右检查通过后仍走原有快速 Home。双踏板切换的稳定判定为
平移速度 ≤90 mm/s、旋转速度 ≤15°/s、连续稳定 120 ms，窗口漂移 ≤6 mm/1.5°；普通
runtime Home 仍使用更严格的 15 mm/s、2.5°/s、250 ms、3 mm/0.75°。Init 单臂
到位判定另采用平移速度 ≤30 mm/s、旋转速度 ≤5°/s、连续稳定 2 s；这些阈值不改变
Tracker/相机新鲜度要求。
进入指定单臂时只采集即将 Fix 的目标侧 Home，另一侧沿用已有 runtime anchor，不会因其
Tracker 仍在运动而阻塞切换；从单臂恢复双臂时只采集需要释放的固定侧，Hold 直接恢复
双臂时仍采集双侧。
Tracker 等待与稳定 Home 采样共享 1 秒总时限；超时会取消切换并显示 3 秒红色提醒。人工处理首次左右
警告弹窗的时间不计入，二次确认后重新开始这 1 秒窗口。普通 Home 与 Q6 Home
不会被双踏板模式误识别为快速 Home。

双踏板模式下 B 是“双臂遥操 / Hold”状态键，并不始终等同于普通“开始遥操”：
双臂遥操时按 B 进入 Hold；Hold 时按 B 经过快速 Home 恢复双臂遥操；任一单臂
Fix 状态按 B 时，当前运动臂和数采保持连续，固定臂继续冻结到后台 Home 就绪后
再原子释放为双臂遥操，不触发全局 Hold，也不暂停数采。Init 移动、精确 qpos
或 Init Home 重采期间不会接受 B/踏板 1/2 的模式切换；此时 C 仍始终可用，用于
先进入常规 Hold，再重新选择单臂/双臂遥操。

### 踏板:

```bash
sudo -E /home/user/miniconda3/envs/dexteleop310/bin/python -m teleop_runtime.configure_devices --fix-permissions

python -m hardware.foot_pad --count 2
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
  --camera-config teleop_runtime/configs/devices/realsense.json \
  --camera-color-format bgr8 \
  --write-parquet --write-video
```

真机多相机默认使用 `bgr8` 颜色传输；它不改变 640×480 等请求分辨率，只避免多台
RealSense 同时协商 `yuyv` 时出现首帧迟到和底层 SDK 重启。启动首帧等待（5 秒）与
运行期新鲜度等待（1 秒）分开，启动期间不会因首帧迟到反复重建 pipeline。只有在
已验证的单站环境中才应显式改用 `--camera-color-format yuyv`。

`POSITION_FORMAL_V1_ARGS` 会强制启用独立的双臂位控受力急停：通过左右 RM65
状态连接以 20 Hz 读取 tool-zero 六维力，默认任一侧 `|F| > 15 N` 或
`|T| > 2 Nm` 即锁存并并行请求左右臂 `Arm_Stop`。启动时必须看到
`force_limits_armed=true stale_watchdog=deferred_until_camera_ready`，随后在相机启动
完成且任何归位/遥操动作前看到
`stale_watchdog=armed_after_fresh_post_camera_samples motion_admission=ready`。
相机初始化期间真实超力和连续读取错误仍会 fail-closed；只有数据陈旧 watchdog
延后启用，且启用前强制要求左右各产生一帧新的有效力样本。运行期力读取连续失败、
数据陈旧或启动后无法取得双侧新鲜样本都会 fail-closed。触发后不能在当前进程
重新使能，必须检查设备并重启。这里的保护与 impedance 使用的
`--force-tare-sec` 无关。

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
  --camera-config teleop_runtime/configs/devices/realsense.json \
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
  --camera-config teleop_runtime/configs/devices/realsense.json \
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

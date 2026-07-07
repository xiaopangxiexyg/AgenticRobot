(dexteleop310) password-1@1:~/DexTeleop$ python -m teleop_runtime.debug.run_vive_rm65_real_teleop   --active-sides right   --right-arm-ip "$RIGHT_ARM_IP"   --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json   --load-profile /home/password-1/DexTeleop/teleop_records/calibration_profiles/20260702_152040_vive_rm65_real_teleop.json   --gui --hand-backend dmtac-gripper --gripper-task-profile teleop_runtime/configs/gripper_task_profile.example.json



export LEFT_ARM_IP=192.168.101.21
export RIGHT_ARM_IP=192.168.1.20


python -m teleop_runtime.debug.run_vive_rm65_real_teleop   --active-sides both   --right-arm-ip "$RIGHT_ARM_IP" --left-arm-ip "$LEFT_ARM_IP"  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json  --left-tool-profile hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json   --load-profile /home/password-1/DexTeleop/teleop_records/calibration_profiles/20260702_152040_vive_rm65_real_teleop.json   --gui --hand-backend dmtac-gripper --gripper-task-profile teleop_runtime/configs/gripper_task_profile.example.json --q6-hold-recalibration-gesture --no-episode-recording 



(dexteleop310) password-1@1:~/DexTeleop$ python -m teleop_runtime.debug.run_vive_rm65_real_teleop   --active-sides both   --right-arm-ip "$RIGHT_ARM_IP" --left-arm-ip "$LEFT_ARM_IP"  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json  --left-tool-profile hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json   --load-profile /home/password-1/DexTeleop/teleop_records/calibration_profiles/20260702_152040_vive_rm65_real_teleop.json   --gui --hand-backend dmtac-gripper --gripper-task-profile teleop_runtime/configs/gripper_task_profile.example.json --q6-hold-recalibration-gesture --no-episode-recording --dmtac-with-tactile --hz 15 \
--arm-io-threads 2 \
--dmtac-with-tactile \
--dmtac-tactile-read-hz 15 \
--dmtac-tactile-process-read \
--dmtac-tactile-worker-cpus 4,5,6,7 \
--dmtac-tactile-worker-nice 15 \
--dmtac-tactile-sequential-read \
--dmtac-tactile-global-read-lock \
--dmtac-tactile-no-denoise \
--dmtac-tactile-no-flat-vector \
--no-session-record-commands \
--session-record-flush-every 30 \
--session-record-compact-json \
--writer-workers 4 \
--storage-queue-size 64 \
--dmtac-tactile-worker-queue-size 4 \
--dmtac-tactile-global-read-lock


taskset -c 0-11 python -m teleop_runtime.debug.run_vive_rm65_real_teleop   --active-sides both   --right-arm-ip "$RIGHT_ARM_IP" --left-arm-ip "$LEFT_ARM_IP"  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json  --left-tool-profile hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json   --load-profile /home/password-1/DexTeleop/teleop_records/calibration_profiles/20260702_152040_vive_rm65_real_teleop.json   --gui --hand-backend dmtac-gripper --gripper-task-profile teleop_runtime/configs/gripper_task_profile.example.json --q6-hold-recalibration-gesture --no-episode-recording --dmtac-with-tactile \
--hz 15 \
--arm-control-hz 40 \
--arm-io-threads 4 \
--dual-arm-ik-hz 15 \
--dmtac-with-tactile \
--dmtac-tactile-read-hz 15 \
--dmtac-tactile-process-read \
--dmtac-tactile-worker-cpus 12,13,14,15 \
--dmtac-tactile-worker-nice 15 \
--dmtac-tactile-sequential-read \
--dmtac-tactile-global-read-lock \
--dmtac-tactile-no-denoise \
--dmtac-tactile-no-flat-vector \
--no-session-record-commands \
--session-record-flush-every 60 \
--session-record-compact-json \
--writer-workers 6 \
--writer-queue-size 1024 \
--storage-queue-size 128


## 16 核机器：触觉并行读取两档配置

前提：16 核机器上尽量把 DMTAC 触觉相机分散到更宽裕的 USB root/controller；如果四个触觉仍共享同一个 USB2 480M root，总线仍可能限制触觉实际写入频率。建议先跑方案 A，确认稳定后再跑方案 B。

共同资源划分：

- 主遥操作进程固定在 CPU `0-11`
- 两个 DMTAC 触觉 worker 固定在 CPU `12-15`
- 触觉读取目标 `15 Hz`
- GUI 会显示 `采集频率` 和 `触觉写入`，优先看这两个数

### 方案 A：左右串行，手内并行（推荐先试）

含义：每只手内部两个 DMTAC 传感器并行读；左右手之间仍用全局锁串行，降低 USB/SDK 同时读四路的风险。

关键开关：

```text
--dmtac-tactile-parallel-read
--dmtac-tactile-global-read-lock
```

完整命令：

```bash
conda activate dexteleop310
cd /home/password-1/DexTeleop

export LEFT_ARM_IP=192.168.101.21
export RIGHT_ARM_IP=192.168.1.20

taskset -c 0-11 python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  --active-sides both \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json \
  --left-tool-profile hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json \
  --gui \
  --load-profile teleop_records/calibration_profiles/20260702_152040_vive_rm65_real_teleop.json \
  --hand-backend dmtac-gripper \
  --gripper-task-profile teleop_runtime/configs/gripper_task_profile.example.json \
  --q6-hold-recalibration-gesture \
  --no-episode-recording \
  --hz 15 \
  --arm-control-hz 40 \
  --arm-io-threads 4 \
  --dual-arm-ik-hz 15 \
  --dmtac-with-tactile \
  --dmtac-tactile-read-hz 15 \
  --dmtac-tactile-process-read \
  --dmtac-tactile-worker-cpus 12,13,14,15 \
  --dmtac-tactile-worker-nice 15 \
  --dmtac-tactile-worker-queue-size 4 \
  --dmtac-tactile-parallel-read \
  --dmtac-tactile-global-read-lock \
  --dmtac-tactile-no-denoise \
  --dmtac-tactile-no-flat-vector \
  --no-session-record-commands \
  --session-record-flush-every 60 \
  --session-record-compact-json \
  --writer-workers 6 \
  --writer-queue-size 1024 \
  --storage-queue-size 128 \
  --wrist-depth-mask
```

采集固定一侧的: 添加--fix-arm


taskset -c 0-11 python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  --active-sides right \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json \
  --gui \
  --load-profile teleop_records/calibration_profiles/20260702_152040_vive_rm65_real_teleop.json \
  --hand-backend dmtac-gripper \
  --q6-hold-recalibration-gesture \
  --no-episode-recording \
  --hz 15 \
  --arm-control-hz 40 \
  --arm-io-threads 4 \
  --dual-arm-ik-hz 15 \
  --no-session-record-commands \
  --session-record-flush-every 60 \
  --session-record-compact-json \
  --writer-workers 6 \
  --writer-queue-size 1024 \
  --storage-queue-size 128



### 方案 B：全部并行（USB 确认宽裕后再试）

含义：左右手同时读，每只手内部两个 DMTAC 传感器也并行读；最多四个触觉 SDK getter 同时工作。16 核且 USB root/controller 分布宽裕时，这个方案最可能提高触觉写入频率，但也最容易暴露 USB/SDK 并发不稳定。

关键开关：

```text
--dmtac-tactile-parallel-read
--dmtac-tactile-no-global-read-lock
```

完整命令：

```bash
conda activate dexteleop310
cd /home/password-1/DexTeleop

export LEFT_ARM_IP=192.168.101.21
export RIGHT_ARM_IP=192.168.1.20

taskset -c 0-11 python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  --active-sides both \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json \
  --left-tool-profile hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json \
  --load-profile /home/password-1/DexTeleop/teleop_records/calibration_profiles/20260702_152040_vive_rm65_real_teleop.json \
  --gui \
  --hand-backend dmtac-gripper \
  --gripper-task-profile teleop_runtime/configs/gripper_task_profile.example.json \
  --q6-hold-recalibration-gesture \
  --no-episode-recording \
  --hz 15 \
  --arm-control-hz 40 \
  --arm-io-threads 4 \
  --dual-arm-ik-hz 15 \
  --dmtac-with-tactile \
  --dmtac-tactile-read-hz 15 \
  --dmtac-tactile-process-read \
  --dmtac-tactile-worker-cpus 12,13,14,15 \
  --dmtac-tactile-worker-nice 15 \
  --dmtac-tactile-worker-queue-size 4 \
  --dmtac-tactile-parallel-read \
  --dmtac-tactile-no-global-read-lock \
  --dmtac-tactile-no-denoise \
  --dmtac-tactile-no-flat-vector \
  --no-session-record-commands \
  --session-record-flush-every 60 \
  --session-record-compact-json \
  --writer-workers 6 \
  --writer-queue-size 1024 \
  --storage-queue-size 128
```

判断标准：

- 若方案 A 已经能稳定 `采集频率` 接近 15 Hz，但 `触觉写入` 仍低，再试方案 B。
- 若方案 B 出现触觉窗口/SDK 错误、触觉写入抖动变大、总采集频率下降，退回方案 A。
- 方案 B 稳定后，再考虑把 `--dmtac-tactile-worker-nice 15` 调小到 `10` 或 `5`，给触觉 worker 更多调度权重。




## 极简 replay 用法

```bash
conda activate dexteleop310
cd /home/password-1/DexTeleop

# 空启动 GUI，在页面顶部 Dataset 输入数据集路径后点 Load。
python -m teleop_runtime.replay --host 127.0.0.1 --port 8765 --no-open-browser
```

浏览器打开：

```text
http://127.0.0.1:8765/
```

如果启动时已经知道数据集路径和 episode：

```bash
python -m teleop_runtime.replay \
  --dataset-root /path/to/lerobot_dataset_root \
  --episode 0 \
  --host 127.0.0.1 \
  --port 8765 \
  --no-open-browser
```

数据集根目录需要包含：

```text
meta/episodes.jsonl
data/chunk-*/episode_*.parquet
```

页面操作：Load 数据集 -> 选择 episode -> Play/Pause/拖动时间轴。若数据集中有 `video_paths` 和 `tactile_paths`，摄像头与 DMTAC 触觉会自动显示。


## DMTAC raw-only 实现与调试记录

目标：数采和遥操实时链路只采集 DMTAC 原始图像，避免在主链路里实时解算触觉力；后续可以再用 SDK 根据 raw 图像离线解算 force / distributed_force。

实现要点：

- 默认采集 raw 图：`tactile_fields=("raw",)`，`tactile_compute_profile="dist-only"`。
- 新增可配置触觉字段：`raw`、`infer`、`deformation2d`、`shear`、`depth`、`force`、`distributed_force` 等。
- raw-only 默认配置；显式写法：

```bash
--dmtac-with-tactile \
--dmtac-tactile-fields raw \
--dmtac-tactile-compute-profile requested \
--dmtac-tactile-no-flat-vector \
--external-tactile
```

- 回到原始 force 实时路径：

```bash
--dmtac-with-tactile \
--dmtac-tactile-fields distributed_force \
--dmtac-tactile-compute-profile dist-only
```

- 如果旧脚本依赖扁平触觉向量，再加：

```bash
--dmtac-tactile-flat-vector
```

离线可行性依据：SDK 的 `dmrobotics.utils.init_h5/read_h5` 存的是 `(N,H,W) uint8` raw 图像和 serial，`read_h5()` 会恢复成 `DMTacImage`，再交给 `Sensor.process(...)` 解算。现在 dataset external tactile 也保留 raw 的 `uint8` dtype 和 shape，后续可以按 manifest 拼回同样的输入。

数采 timing/debug 已保存内容：

- dataset parquet 每帧会保存 DMTAC timing 列：
  `observation.hand.<side>.gripper_read_elapsed_s`、
  `tactile_read_elapsed_s`、`tactile_gate_wait_s`、`telemetry_elapsed_s`、
  `tactile_timings_ms_json`。
- `sample_metadata_json` 每帧会保存 `sample_builder_debug`，包含 arm/hand/camera snapshot 耗时和 camera/hand debug snapshot。
- 正在录制 dataset 时，`sample_metadata_json` 还会保存主循环 `loop_phase_timings_ms` 和 `loop_elapsed_before_record_s`。
- episode finalize summary 会保存 `writer_pool_stats`、`storage_queue_stats`、`collection_frequency`；用于判断 writer 队列、相机队列、触觉写入频率是否拖慢。
- GUI/日志在数采频率低于 `--collection-warning-hz` 时会打印 `DatasetFreq` 和 `DatasetFreqDebug`，包含 writer、storage queue、camera、hand/DMTAC 状态。

建议开启的调试保存方式：

```bash
--debug-timing \
--debug-timing-threshold-sec 0.066 \
--debug-timing-report-interval-sec 1.0 \
--session-record-root teleop_records/debug_sessions \
--session-record-compact-json \
--session-record-flush-every 30
```

说明：

- `--debug-timing`：慢帧时打印主循环分段耗时；15Hz 目标下 `0.066s` 对应一帧预算。
- `--session-record-root`：开启额外 JSONL debug session，目录形如 `teleop_records/debug_sessions/<timestamp>_vive_rm65_real_teleop/`。
- 不要加 `--no-record`，否则 session JSONL 不会保存。
- 如果想减少同步写盘影响，可以保留 `--session-record-compact-json --session-record-flush-every 30`。
- 如果不想保存同步 `commands.jsonl`，可以加 `--no-session-record-commands`；正式 dataset 不受影响。
- 若使用独立 `teleop_runtime.collection.run_collect`，对应卡顿诊断开关是：

```bash
--debug-frozen \
--debug-frozen-threshold-sec 0.066 \
--debug-frozen-report-interval-sec 1.0
```

raw-only 双臂 15Hz 示例：

```bash
taskset -c 0-11 python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  --active-sides both \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json \
  --left-tool-profile hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json \
  --gui \
  --load-profile teleop_records/calibration_profiles/20260702_152040_vive_rm65_real_teleop.json \
  --hand-backend dmtac-gripper \
  --gripper-task-profile teleop_runtime/configs/gripper_task_profile.example.json \
  --q6-hold-recalibration-gesture \
  --hz 15 \
  --arm-control-hz 40 \
  --arm-io-threads 4 \
  --dual-arm-ik-hz 15 \
  --dmtac-with-tactile \
  --dmtac-tactile-fields raw \
  --dmtac-tactile-compute-profile requested \
  --dmtac-tactile-read-hz 15 \
  --dmtac-tactile-process-read \
  --dmtac-tactile-worker-cpus 12,13,14,15 \
  --dmtac-tactile-worker-nice 15 \
  --dmtac-tactile-worker-queue-size 4 \
  --dmtac-tactile-parallel-read \
  --dmtac-tactile-global-read-lock \
  --dmtac-tactile-no-denoise \
  --dmtac-tactile-no-flat-vector \
  --external-tactile \
  --writer-workers 6 \
  --writer-queue-size 1024 \
  --storage-queue-size 128 \
  --debug-timing \
  --debug-timing-threshold-sec 0.066 \
  --session-record-root teleop_records/debug_sessions \
  --session-record-compact-json \
  --session-record-flush-every 30
```

排查顺序：

1. 先看 GUI 的 `采集频率` 和 `触觉写入` 是否接近目标。
2. 频率低时看终端 `DatasetFreqDebug`：writer 队列满、storage queue drop/high_water、camera age、hand tactile timing 哪个先异常。
3. 保存后看 parquet 的 `sample_metadata_json` 和 `observation.hand.*.tactile_timings_ms_json`。
4. 若 raw-only 仍慢，优先降低相机/触觉读频率或切回 `--dmtac-tactile-global-read-lock` 串行读；若只有 force 路径慢，继续用 raw-only。

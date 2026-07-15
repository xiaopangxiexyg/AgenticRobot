# DMTAC 真机遥操作命令分层

本文只重新整理真机遥操作命令。相机、DMTAC、Q6、gripper、数据集写盘等与机械臂控制链无关的配置保持原值；机械臂执行模式、各阶段频率和 debug 作为独立 overlay 选择。

不要把 `safe-debug`、`disable-hold`、关闭奇异保护或 CANFD controller smoothing 当成普通 debug 开关；这些选项会改变控制行为，统一放在文末“高级控制实验”中。

## 1. 环境与 CPU 分配

```bash
conda activate dexteleop310

export LEFT_ARM_IP=192.168.101.21
export RIGHT_ARM_IP=192.168.1.20
```

16 核机器沿用以下资源划分：

- 主遥操作进程：CPU `0-11`
- 两个 DMTAC tactile worker：CPU `12-15`
- DMTAC 读取目标：`15 Hz`
- GUI 重点观察：`采集频率`、`触觉写入`

如果四个触觉仍共享同一个 USB2 480M root，总线仍可能限制实际频率。先运行方案 A，稳定后再试方案 B。

## 2. Common：设备、遥操作输入与数采

以下数组是双臂 DMTAC raw-only 数采的公共参数。控制模式和 debug 不放在这里。

```bash
COMMON_REAL_TELEOP_ARGS=(
  --active-sides both
  --right-arm-ip "$RIGHT_ARM_IP"
  --left-arm-ip "$LEFT_ARM_IP"
  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json
  --left-tool-profile hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json
  --gui
  --load-profile teleop_records/calibration_profiles/20260702_152040_vive_rm65_real_teleop.json
  --hand-backend dmtac-gripper
  --gripper-task-profile teleop_runtime/configs/gripper_task_profile.example.json
  --q6-hold-recalibration-gesture
  --no-episode-recording
  --arm-io-threads 4
)

DMTAC_CAPTURE_COMMON_ARGS=(
  --dmtac-with-tactile
  --dmtac-tactile-fields raw
  --dmtac-tactile-compute-profile requested
  --dmtac-tactile-read-hz 15
  --dmtac-tactile-process-read
  --dmtac-tactile-worker-cpus 12,13,14,15
  --dmtac-tactile-worker-nice 15
  --dmtac-tactile-worker-queue-size 4
  --dmtac-tactile-parallel-read
  --dmtac-tactile-no-denoise
  --dmtac-tactile-no-flat-vector
  --external-tactile
)

DATASET_COMMON_ARGS=(
  --no-session-record-commands
  --session-record-flush-every 60
  --session-record-compact-json
  --writer-workers 6
  --writer-queue-size 1024
  --storage-queue-size 128
  --wrist-depth-mask
)
```

说明：

- `--no-episode-recording` 只关闭轻量 debug trajectory，不关闭正式 LeRobot 数据集。
- session JSONL 默认关闭；仅设置 `--session-record-*` 不会自动启用，确实需要时必须额外传 `--record`。
- `--external-tactile` 保持原 external tactile 存储方式。
- `raw + requested + no-flat-vector` 避免在实时链路中解算完整触觉力。

## 3. DMTAC 方案 A / B

### 方案 A：左右串行，手内并行（推荐）

每只手内部两个 DMTAC 并行读取；左右手通过全局锁串行，降低 USB/SDK 四路并发风险。

```bash
DMTAC_A_ARGS=(
  --dmtac-tactile-global-read-lock
)
```

### 方案 B：左右与手内全部并行

最多四个 tactile SDK getter 同时运行。仅在 USB root/controller 分布宽裕、方案 A 已稳定后使用。

```bash
DMTAC_B_ARGS=(
  --dmtac-tactile-no-global-read-lock
)
```

旧的全串行读取仍保留为诊断备选：从 common 中移除
`--dmtac-tactile-parallel-read`，再追加 `--dmtac-tactile-sequential-read`；不要把
parallel 与 sequential 同时作为 canonical 配置。

判断标准保持不变：

1. 方案 A 的采集频率接近 15 Hz、但触觉写入仍低时，再试 B。
2. B 出现 SDK/窗口错误、写入抖动或总采集频率下降，退回 A。
3. B 稳定后，可再把 worker nice 从 `15` 调到 `10` 或 `5`。

## 4. 控制分支

四个频率配置明确分开，但当前只有三个实际调度时钟：sample、position
reference/IK outer loop、backend qcmd。`--dual-arm-ik-hz` 是 solver 的 nominal
frequency/步长预算，不会另起一个 IK worker。

| 阶段 | impedance 分支 | position 分支 | CLI |
|---|---:|---:|---|
| 数据采样 | 15 Hz | 15 Hz | `--hz` |
| 外层 reference | 15 Hz pose | 75 Hz qpos | `--hz` / `--position-reference-hz` |
| dual-arm IK solver nominal frequency | 15 Hz | 75 Hz | `--dual-arm-ik-hz` |
| backend / qcmd | 40 Hz | 150 Hz | `--arm-control-hz` |

表中的75 Hz只是`--position-reference-hz`当前默认值，IK、collision、replay/deploy和debug
均读取实际参数，不存在45/75专属算法分支；`--dual-arm-ik-hz`通常保持与reference一致。

三个调度时钟按时间推进；默认15:75:150恰好为整数频率比，但实现仍按时间戳推进，
不依赖固定帧计数。IK 实际调用节拍跟随
outer/reference，因此 canonical 与自定义配置中应让 `--dual-arm-ik-hz` 等于对应的
`--hz`（impedance）或 `--position-reference-hz`（position）；写成不同值会产生 warning。

### 分支 1：力控 impedance 40 Hz

```bash
IMPEDANCE_40HZ_ARGS=(
  --arm-execution-profile impedance_40hz
  --arm-control-mode impedance
  --hz 15
  --dual-arm-ik-hz 15
  --dual-arm-temp-max-xyz-step-m 0.003
  --dual-arm-temp-max-rot-step-deg 0.5
  --arm-control-hz 40
  --no-arm-canfd-follow
)
```

该分支保持现有正常的 40 Hz impedance backend，不启用 position CANFD background stream。

### 分支 2：位控 75 Hz reference → 150 Hz qcmd

```bash
POSITION_CONTROL_ARGS=(
  --arm-execution-profile position
  --arm-control-mode position
  --hz 15
  --position-reference-hz 75
  --dual-arm-ik-hz 75
  --position-reference-max-age-sec 0.2
  --arm-control-hz 150
  --arm-canfd-follow
  --position-post-ik-qpos-smoothing-alpha 0.9
  --arm-follow-qpos-alpha 0.1
  --arm-canfd-trajectory-mode 0
  --arm-canfd-radio 0
)
```

### 分支 2B：当前60 Hz reference/IK → 120 Hz qcmd，8×跟踪 + 0.675° step guard

这是当前真机平滑性复测使用的 position-only overlay。独立 step guard 直接接到
120 Hz acceleration-limited planner，不通过降低 `--arm-max-angvel` 获得，因此不会同时
压低持续跟踪速度。它位于 final alpha=0.9 之前，正常跟踪时实际 qcmd 单步最大约为
`0.675° × 0.9 = 0.6075°`。

运行前仍需确认 `robot_motion/configs/joint_velocity_limits.json` 中
`profiles.rm65.max_velocity_deg_s` 已在受控测试窗口临时设置为
`[321,321,413,688,688,917]`，否则 qref final step 仍不是完整8×。

```bash
POSITION_60_120_8X_STEP_GUARD_ARGS=(
  --arm-execution-profile position
  --arm-control-mode position
  --hz 15
  --position-reference-hz 60
  --dual-arm-ik-hz 60
  --position-reference-max-age-sec 0.2
  --arm-control-hz 120
  --arm-canfd-follow
  --position-command-planner acceleration_limited
  --ik-tuning-json teleop_runtime/configs/rm65_ik_tuning.json
  --position-ik-solve-max-iterations 1
  --position-ik-local-step-clip-scale 8
  --arm-max-angvel 5.6,5.6,7.2,12,12,16.0
  --position-max-acceleration-rad-s2 56,56,72,120,120,160
  --position-max-command-step-deg 0.675
  --position-velocity-ramp-sec 0.10
  --position-post-ik-qpos-smoothing-alpha 0.17
  --position-post-ik-qpos-smoothing-fast-alpha 0.50
  --position-post-ik-qpos-smoothing-adaptive
  --arm-follow-qpos-alpha 0.9
  --position-degraded-speed-scale 0.50
  --arm-canfd-trajectory-mode 0
  --arm-canfd-radio 0
)
```

标准 position 分支不再同时设置 `--arm-canfd-follow-stream-hz`。该 legacy 参数仍兼容，但显式大于 0 时会覆盖实际 background stream 频率；它与 `--arm-control-hz` 不同会打印 warning。

若需要自定义频率，直接修改上面对应的频率参数。position 分支应满足 reference 不低于 sample、qcmd 不低于 reference；违反时 resolved metadata 会记录 ordering warning。

例如只做非默认频率实验时，可用下列数组替换 position overlay 中的四个频率值：

```bash
CUSTOM_POSITION_RATE_ARGS=(
  --hz 12
  --position-reference-hz 50
  --dual-arm-ik-hz 50
  --arm-control-hz 160
)
```

| 自定义项 | 是否可独立设置 | 建议约束 |
|---|---|---|
| sample Hz | 是 | 正数 |
| position reference Hz | 是 | `reference_hz >= sample_hz` |
| IK solver nominal Hz | 可配但不是独立时钟；未写时跟随 outer reference | 应与 reference Hz 相同 |
| qcmd Hz | 是 | `command_hz >= reference_hz` |

## 5. Debug overlay

`--control-debug` 是纯观测 contract，不会自动开启或修改 smoothing、hold、奇异保护、碰撞、安全限位或 dispatch 行为。

debug contract 只表达记录请求和输出参数；runtime recorder 负责实际落盘。为兼容已有排查方式，full overlay 同时组合现有 action/timing 日志。

### Debug off

```bash
CONTROL_DEBUG_OFF_ARGS=(
  --control-debug off
)
```

### Debug summary contract

```bash
CONTROL_DEBUG_SUMMARY_ARGS=(
  --control-debug summary
  --control-debug-root teleop_records/control_debug
  --control-debug-queue-size 8192
  --control-debug-flush-every 30
)
```

summary 应记录：各阶段 p50/p95/p99/max、reference/command 实际频率、deadline miss、最大 gap、发送成功/失败/stale/fault 计数，以及 debug queue high-water/drop count；不保存逐 tick qpos trace。

### Debug full + 当前已有日志

```bash
CONTROL_DEBUG_FULL_ARGS=(
  --control-debug full
  --control-debug-root teleop_records/control_debug
  --control-debug-queue-size 8192
  --control-debug-flush-every 30
  --debug-action
  --debug-action-log-root teleop_records/action_debug
  --debug-action-flush-every 30
  --debug-action-queue-size 8192
  --debug-timing-log
  --debug-timing-console
  --debug-timing-ik
  --debug-timing-threshold-sec 0.0222
  --debug-timing-report-interval-sec 1.0
)
```

full 在 summary 之外实际记录完整的 position reference 与 qcmd trace：epoch/sequence、wall/monotonic 时间、q_ik_raw、q_ref、pose_ref、修改原因、backend target sequence、150 Hz tick/alpha/qcmd 和 SDK 结果。左右两侧事件会绑定到同一 reference sequence，可离线计算 send skew；本次没有实现前述组级原子 coordinator。

当前已有日志可记录：

- outer loop 各阶段耗时、慢帧和状态/故障变化；
- 逐 outer frame action、IK、q_ref 相关 metadata；
- backend background stream 的累计 send/accepted/failed、actual Hz、deadline miss、gap、stale、fault snapshot；
- 正式数采时的 15 Hz sample metadata 和 transport health。

实际输出如下：

- session：`<control-debug-root>/<timestamp>_vive_rm65_real_teleop/`
- 正式数采 episode：`<dataset-root>/meta/position_control/episode_XXXXXX/`
- summary/ full 都写 `position_summary.json`
- full 额外写 `position_reference.jsonl`、`position_command_150.jsonl`、`position_timing.jsonl`
- episode recorder 会从动态 fanout 中无锁挂载/摘除，结束时由后台线程收尾；控制线程只做有界 `put_nowait`，队列满时增加 drop counter，不等待写盘

`position_reference`使用频率无关的schema/event名称；`position_command_150`暂时保留既有
v1名称。实际目标频率看 `position_summary.json.context.arm_execution`，实际
运行频率看 summary percentiles 和逐事件 metadata，不要从文件名反推。

最终应检查 `position_summary.json` 的 `writer_error`、`enqueue_drop_count`、queue high-water、deadline/skip 计数和各阶段 p50/p95/p99/max。debug contract 本身不改变控制行为。

如需 legacy session JSONL，可额外组合：

```bash
LEGACY_SESSION_DEBUG_ARGS=(
  --record
  --session-record-root teleop_records/debug_sessions
  --no-session-record-commands
  --session-record-compact-json
  --session-record-flush-every 30
)
```

原有采集卡顿诊断选项也保留，不与 control-debug 混为控制开关：

```bash
COLLECTION_TIMING_DEBUG_ARGS=(
  --debug-timing
  --debug-timing-threshold-sec 0.066
  --debug-timing-report-interval-sec 1.0
  --collection-warning-hz 15
)
```

`--debug-timing` 在 15 Hz 外循环下以 66 ms 为慢帧阈值；session JSONL 需要组合上面的
`LEGACY_SESSION_DEBUG_ARGS`（必须有 `--record`，不要加 `--no-record`）。若单独使用
`teleop_runtime.collection.run_collect`，原有等价诊断仍是
`--debug-frozen --debug-frozen-threshold-sec 0.066 --debug-frozen-report-interval-sec 1.0`。

## 6. 完整运行示例

### Canonical 命令 1：方案 A + impedance 40 Hz + debug off

```bash
taskset -c 0-11 python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  "${COMMON_REAL_TELEOP_ARGS[@]}" \
  "${DMTAC_CAPTURE_COMMON_ARGS[@]}" \
  "${DMTAC_A_ARGS[@]}" \
  "${DATASET_COMMON_ARGS[@]}" \
  "${IMPEDANCE_40HZ_ARGS[@]}" \
  "${CONTROL_DEBUG_OFF_ARGS[@]}"
```

### Canonical 命令 2：方案 A + position 75→150 Hz + full debug

```bash
taskset -c 0-11 python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  "${COMMON_REAL_TELEOP_ARGS[@]}" \
  "${DMTAC_CAPTURE_COMMON_ARGS[@]}" \
  "${DMTAC_A_ARGS[@]}" \
  "${DATASET_COMMON_ARGS[@]}" \
  "${POSITION_CONTROL_ARGS[@]}" \
  "${CONTROL_DEBUG_FULL_ARGS[@]}"
```

### 当前平滑复测：方案 A + position 60→120 Hz + 8× + 0.675° guard

```bash
taskset -c 0-11 python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  "${COMMON_REAL_TELEOP_ARGS[@]}" \
  "${DMTAC_CAPTURE_COMMON_ARGS[@]}" \
  "${DMTAC_A_ARGS[@]}" \
  "${DATASET_COMMON_ARGS[@]}" \
  "${POSITION_60_120_8X_STEP_GUARD_ARGS[@]}" \
  "${CONTROL_DEBUG_FULL_ARGS[@]}"
```

### 方案 B + position 75→150 Hz + summary contract

```bash
taskset -c 0-11 python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  "${COMMON_REAL_TELEOP_ARGS[@]}" \
  "${DMTAC_CAPTURE_COMMON_ARGS[@]}" \
  "${DMTAC_B_ARGS[@]}" \
  "${DATASET_COMMON_ARGS[@]}" \
  "${POSITION_CONTROL_ARGS[@]}" \
  "${CONTROL_DEBUG_SUMMARY_ARGS[@]}"
```

## 7. 单侧采集

原右臂单侧配置保留。使用时应替换 `COMMON_REAL_TELEOP_ARGS` 中的 active side、arm IP 和 tool profile，而不是简单追加，否则 inactive left 配置会触发校验错误。

```bash
RIGHT_ONLY_COMMON_ARGS=(
  --active-sides right
  --right-arm-ip "$RIGHT_ARM_IP"
  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json
  --gui
  --load-profile teleop_records/calibration_profiles/20260702_152040_vive_rm65_real_teleop.json
  --hand-backend dmtac-gripper
  --q6-hold-recalibration-gesture
  --no-episode-recording
  --arm-io-threads 4
)
```

## 8. 高级控制实验：不会被 debug 自动启用

以下旧选项继续保留，但都可能改变控制语义：

```text
固定一侧采集时按需追加：       --fix-arm
impedance 中关闭力导纳：       --arm-disable-force-admittance
旧 120 Hz follow stream：      --arm-canfd-follow --arm-canfd-follow-stream-hz 120
CANFD 曲线拟合：               --arm-canfd-trajectory-mode 1 --arm-canfd-radio 100
CANFD controller 滤波：        --arm-canfd-trajectory-mode 2 --arm-canfd-radio 100
修改IK后/轨迹前 qpos EMA：     --position-post-ik-qpos-smoothing-alpha 0.8
修改最终高频发送 qpos EMA：    --arm-follow-qpos-alpha 0.8
独立限制position planner单步： --position-max-command-step-deg 0.675
恢复旧 position safety/filter：--safe-debug
在 safe-debug 下关闭 hold：     --safe-debug --disable-hold
严格五分之一关节限速：         --strict-degree-limit
关闭非阻抗奇异保护：           --no-arm-singularity-protection
```

关闭奇异保护仅用于受控诊断；不要放入标准 position profile 或 debug overlay。

## 9. DMTAC raw-only 与离线解算

实时链路默认只保存 raw image：

```bash
--dmtac-with-tactile \
--dmtac-tactile-fields raw \
--dmtac-tactile-compute-profile requested \
--dmtac-tactile-no-flat-vector \
--external-tactile
```

回到原始 distributed force 实时路径：

```bash
--dmtac-with-tactile \
--dmtac-tactile-fields distributed_force \
--dmtac-tactile-compute-profile dist-only
```

旧脚本确实依赖扁平触觉向量时，再加：

```bash
--dmtac-tactile-flat-vector
```

SDK 的 `dmrobotics.utils.init_h5/read_h5` 保存 `(N,H,W) uint8` raw 图像和 serial；external tactile 同样保留 raw dtype/shape，因此后续可以离线恢复 `DMTacImage` 并调用 `Sensor.process(...)` 解算。

正式 dataset 每帧已有 DMTAC timing、sample builder debug、loop phase timing；episode finalize summary 已有 writer/storage queue 和 collection frequency。排查顺序保持：

1. GUI 查看采集频率和触觉写入。
2. 查看 `DatasetFreqDebug` 中 writer、storage、camera age、hand/DMTAC timing。
3. 查看 parquet 的 `sample_metadata_json` 和 `observation.hand.*.tactile_timings_ms_json`。
4. raw-only 仍慢时先降低相机/触觉读频率或退回全局锁；只有 force 路径慢时继续使用 raw-only。

## 10. 极简 replay 查看数据

```bash
conda activate dexteleop310
python -m teleop_runtime.replay --host 127.0.0.1 --port 8765 --no-open-browser
```

浏览器打开：

```text
http://127.0.0.1:8765/
```

启动时指定数据集与 episode：

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

页面操作仍是：Load 数据集 → 选择 episode → Play/Pause/拖动时间轴；存在 `video_paths` 和 `tactile_paths` 时会自动显示相机与 DMTAC。

## 11. real replay / deploy 的位控链路

位控默认执行顺序已经接通：

```text
15 Hz raw action
  -> 15 Hz finite/quaternion/workspace/pose step safety
  -> 75 Hz timestamp pose reference（xyz linear + shortest-path quaternion SLERP）
  -> coupled dual-arm collision-aware IK
  -> 绝对关节限位 + 每 75 Hz reference 步长限位
  -> 限位后重新计算所选 scene profile 的 endpoint collision distance
  -> 75 Hz safe endpoint qpos
  -> sim/real 共用 75→150 Hz qpos planner
  -> 最后 150 Hz sink 才区分 sim controller / real Movej_CANFD
```

这里没有额外 low-pass/EMA。15 Hz keyframe safety 只执行一次，不会在五个75 Hz子点上重复施加。real replay 是 offline，可使用已知的后继 action，并优先保留 parquet 的逐帧 timestamp；时间戳缺失或非法时整段明确退回 nominal grid。online deploy 严格因果，因此显式延迟一个 action period。手/夹爪继续保持 15 Hz 离散值且不插值；混合 arm+hand action 会在同一个延迟后75 Hz reference tick 做逻辑调度，随后由独立一槽 latest-wins hand worker 下发，避免 hand I/O 拖慢机械臂 reference。调度落后时不会补发手部 burst；真实 backend 拒绝会触发 fault/arm hold，debug JSONL 会记录 callback 实际起止时间、receipt、skip/coalesce 和 fault。这里不承诺两个物理设备同时生效，backend 自身的重连也可能产生内部 retry；若 fault 出现时一个 arm callback 已进入 backend，该次调用最多仍可能完成，随后以 `accepted_before_async_abort` 记录并 hold。真实 backend callback 必须配置有限 I/O timeout。

安全边界：real replay/deploy 的双臂75 Hz endpoint checker 已接通，且 distance 不可用、
初始双臂 qpos 不完整、IK/限位后无安全候选都会 fail-closed。仍未完成的是相邻75 Hz
endpoint 之间75→150插值整段的连续碰撞认证；碰撞自由空间非凸，安全端点不能证明中间
线段安全。默认 `dual_arm_hand_proxy_open` 还是固定张开手/link proxy，也不能泛化为任意
DMTAC/自定义工具外形已经完整建模。debug 会同时记录
`endpoint_collision_checked=true` 与 `segment_collision_certified=false`。

position 真机输出当前要求 `active_sides=both`、左右 backend 启动后的实际 TCP，以及完整
left+right 权威当前 qpos；单臂没有被动臂状态源时会拒绝，不能用默认姿态猜测。

replay 的无硬件 GUI/旧格式检查示例（显式关闭 collision required，只允许非运动 dry-run，
不验证75 Hz collision IK）：

```bash
python -m teleop_runtime.replay \
  --dataset-root /path/to/lerobot_dataset_root \
  --episode 0 \
  --arm-control-mode position \
  --position-reference-hz 75 \
  --arm-control-hz 150 \
  --arm-canfd-follow \
  --no-position-collision-required \
  --position-workspace-min -0.60 -0.80 0.00 \
  --position-workspace-max 0.60 0.80 0.90 \
  --position-max-xyz-step-m 0.10 \
  --position-max-rot-step-deg 15 \
  --position-max-joint-step-rad 0.10472 \
  --no-connect-hardware \
  --no-open-browser
```

deploy 的 position collision dry-run 需要双臂有效初始 qpos、两侧实际 TCP 和 IK 依赖；以下
示例只展示配置形状，不加 `--send-actions`，且不会猜测缺失的初始状态：

```bash
python -m teleop_runtime.deploy.run_policy \
  --policy hold \
  --active-sides both \
  --left-tool-profile hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json \
  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json \
  --hz 15 \
  --arm-control-mode position \
  --position-reference-hz 75 \
  --position-collision-check \
  --position-collision-scene-profile dual_arm_hand_proxy_open \
  --position-collision-min-distance-m 0.03 \
  --arm-control-hz 150 \
  --arm-canfd-follow
```

deploy 中 `send_actions=false` 仍会运行可用的15 Hz safety、因果75 Hz插值、coupled
planner、canonical command resolution 与统计，但不会调用 backend dispatch；如果没有完整
双臂 qpos/TCP 或 IK 依赖，会在 planner prepare/reset 明确失败。若只想做旧单臂非运动格式
检查，可显式加 `--no-position-collision-check`；该开关不能与 position `--send-actions`
组合。impedance replay/deploy 沿用原 action-rate → 40 Hz backend，不经过 position executor。

排查 deploy 位控异步链路时追加：

```bash
--debug --debug-dir teleop_records/deployment_debug
```

`action_step.dispatch_mode=position_async_queue` 只表示安全关键帧已入队；实际结果看后续
`position_executor_arm_reference_dispatch`、`position_executor_discrete_dispatch_queued`、
`position_executor_discrete_dispatch` 和 `position_executor_executor_fault`。这些事件包含 source
index、parent/alpha、arm command、hand logical tick 与实际 callback 时间、queue lag、backend
receipt、skip/coalesce 以及 fault/hold，因而能区分“入队、逻辑调度、backend 接受”三个阶段。
collision debug 还包含 raw IK qpos、最终 qpos、target/realized world pose、每次 distance
attempt、限位原因与 solve/budget 时间；关闭 debug 时命令 metadata 只保留摘要。
真机 collision planner reset 还会记录双臂 hardware read/measured sequence、测量年龄与
timestamp skew；读取失败或过期不会回退 cached qpos。coupled qpos 的 requested/final/backend
target、singularity gate 与 150 Hz background health 会进入相应 debug；deploy 事件另有
backend receipt，而 replay ring 当前不声称保存完整 receipt。replay collision debug 为三个
health 阶段保留固定槽，无 hand 的 tick 标记 `not_applicable`，失败 gate 先记录异常再
fail-closed。replay debug 使用进程内 512 条有界 ring，`/api/state` 查看 last/计数，完整窗口可通过
`GET /api/position-collision-debug?limit=100` 读取；状态只有在真实 endpoint 检查成功且当前
未 fault 时才显示已认证。

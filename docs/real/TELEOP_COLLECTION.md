# 真机遥操与数采参数说明

运行命令见 [real README](./README.md)，完整参数语义见
[`teleop_runtime/README.md`](../../teleop_runtime/README.md) 和
[`teleop_runtime/collection/README.md`](../../teleop_runtime/collection/README.md)。

正式 teleop/collection 命令均为 M 级：可能连接并驱动机械臂、手或夹爪。单独的 `--help`、
`--validate-only` 和 VIVE sender 按各自入口标注的较低等级处理。

## 1. 启动顺序

1. 激活 `dexteleop310`，进入仓库根目录。
2. 停止占用相同 RM65、Q6、DMTAC、相机和 VIVE 端口的旧进程。
3. 执行设备 `--check`，确认左右 side、IP、串口、tool/TCP profile 和相机配置。
4. 按 [real README 的第 1.1 节](./README.md) 启动 VIVE sender。
5. 先做 virtual/calibration 验收；真机工作空间清空、急停可用后再启动 GUI。
6. GUI preflight 完成后人工放行，不使用未经审核的 auto-start 参数。

## 2. GUI human teleop / 正式采集入口

可直接复制的入口见 [real README 的第 1.2 节](./README.md)。它只是
canonical composition root；实际站点应显式提供设备、profile、dataset 和 debug 参数。
不要从旧日志中复制一串未命名的调参 flag。

## 3. 正式 position profile

当前正式 profile 是 `rm65_bimanual_position_v1`。它固定 15 Hz sample、60 Hz IK/qref、
120 Hz qcmd、accepted-qref FK main action、1.0/0.35 IK、adaptive min2/max4、position-only
奇异 branch 策略和低扰动数采配置。

正式可执行命令只在 [real README 的第 1.3 节](./README.md) 维护；本节负责
解释该命令选中的 profile 和验收字段。

参数数组只由 [`teleop_args.sh`](./teleop_args.sh) 加载，不会在 `source` 时启动设备。
旧 A/B、奇异问题定位过程和被替代命令保存在
[`docs/legacy/position_control/test-7.11.md`](../legacy/position_control/test-7.11.md)。

不要与 `POSITION_CONTROL_ARGS`、`POSITION_ACCELERATION_LIMITED_*`、`POSITION_IK_*`
叠加；这些是历史对照/调参数组，会覆盖正式 profile。

### 3.1 Formal profile 参数

| 层 | 当前值 | 应用范围 |
|---|---|---|
| Sample / IK / qref / qcmd | `15 / 60 / 60 / 120 Hz` | Live position teleop |
| Main action | 最后 accepted qref 的 FK pose | 写入 dataset；replay/deploy 不重复 pre-action 整形 |
| Inactive enter scale | `0.5` | 仅 live human source |
| Post-IK adaptive alpha | `0.14 → 0.45` | 仅 live human source、左右臂独立 |
| IK position/orientation | `1.0 / 0.35` | Pose action → qref |
| IK damping | `0.03` | Pose action → qref |
| IK iterations | adaptive `min=2 / max=4` | 60 Hz solve budget内 |
| Non-IK total reserve | `5.0 ms` | 其中 candidate certification `3.5 ms`，post-candidate `1.5 ms` |
| Candidate local clip | `8×` | Position-only candidate search |
| Singularity policy | wrist/elbow/shoulder 优先非奇异；无替代时允许认证后的奇异解 | 只影响位控 |
| qref velocity profile | `rm65_bimanual_position_v1_qref` | 不修改共享 RM65/力控 profile |
| Planner | `acceleration_limited` | Accepted qref → 120 Hz qcmd |
| qcmd max velocity | `5.6,5.6,7.2,12,12,16 rad/s` | 逐关节 |
| qcmd max acceleration | `56,56,72,120,120,160 rad/s²` | 逐关节 |
| Target-velocity filter | `tau=0.03 s` | 120 Hz planner |
| Velocity ramp | `0.10 s` | 120 Hz planner |
| Command step envelope | `0.675°` | 每次 qcmd proposal |
| Final follow alpha | `0.9` | 最终 120 Hz SDK qpos |
| Degraded speed | `0.5×` | 已分类异常目标，不改变正常 qdd 上限 |
| Reference max age | `0.2 s` | Stale 时 fail-safe hold |

参数真值位于版本化
[`robot_motion/rm65/position_profile.py`](../../robot_motion/rm65/position_profile.py)。
文档表用于运行检查，不应成为第二份配置实现。

### 3.2 Bash 数组职责与覆盖顺序

| 数组 | 责任 | 不负责 |
|---|---|---|
| `COMMON_REAL_TELEOP_ARGS` | 双臂、tool/TCP、碰撞与 GUI 基础配置 | 不选择 position/force profile |
| `DMTAC_CAPTURE_COMMON_ARGS` | DMTAC 数采公共选项 | 不决定左右并行策略 |
| `DMTAC_A_ARGS` | 当前推荐 DMTAC 调度方案 | 不改变 arm control |
| `DMTAC_TACTILE_WORKER_2CPU_ARGS` | 触觉 worker CPU `12,13` 与 nice | 不改变触觉数据语义 |
| `DATASET_COMMON_ARGS` | Dataset、相机、sample 公共配置 | 不改变位控参数 |
| `POSITION_FORMAL_V1_ARGS` | 选择正式 position profile 和 control mode | 不配置 writer/debug |
| `POSITION_COLLECTION_PROCESS_ARGS` | Spawn writer、共享内存槽、CPU `8-11` | 不改变 main action |
| `POSITION_COLLECTION_PRODUCTION_ARGS` | 关闭高频 JSON/FK debug，保留预分配数值 trace | 不关闭正式 dataset |
| `NO_COLLECTION_SENSOR_BASELINE_ARGS` | 禁用 collection/episode writer，形成控制与传感器基线 | 不禁用 RGB 相机或 DMTAC 触觉 |
| `CONSOLE_TIMING_ARGS` | 通过有界后台队列输出慢帧/stall，不启用 timing JSONL | 不提供正式性能结论 |
| `NO_DEBUG_DISK_WRITE_ARGS` | 关闭所有 standalone debug 写盘 | 不关闭正式 dataset |

Argparse 使用后出现的值覆盖前值。正式命令中不要在这些数组之后随意追加历史 position flag。

### 3.3 启动后确认

Dataset `meta/info.json` 的 runtime metadata 至少应满足：

- `position_control_profile_resolved=rm65_bimanual_position_v1`；
- `main_action_contract=accepted_qref_fk_pose`；
- `position_ik_orientation_weight=0.35`；
- `position_ik_solve_min_iterations=2`、`position_ik_solve_max_iterations=4`；
- `position_ik_iteration_reserve_ms=5`；
- `position_ik_candidate_certification_reserve_ms=3.5`；
- `position_singularity_candidate_certificate=true`；
- `position_reference_hz=60`、`arm_control_hz=120`；
- `position_target_velocity_filter_tau_sec=0.03`；
- `position_force_guard.enabled=true`、`source=tool`；
- `position_force_guard.max_force_n=15`、`max_torque_nm=2`；
- `position_force_guard.fail_closed=true`、`latched=true`、`stop_scope=bimanual`；
- `position_max_command_step_deg=0.675`；
- `collection_worker_mode=process`、`control_debug.level=off`；
- `actuator_trace=true`、`actuator_singularity_trace=true`。

正常退出后还应确认 `collection_integrity_valid=true`、`submitted_samples == completed_samples`
以及 actuator trace 的 drop/error count 均为 `0`。

### 3.4 定向回退

只回退 shoulder/coaxial branch 纠正：

```bash
--no-position-prefer-non-shoulder-singular-branch
```

只回退 candidate 预算拆分和最终 qref 证书：

```bash
--position-ik-candidate-certification-reserve-ms 0 \
--no-position-singularity-candidate-certificate
```

只关闭稀疏 singularity trace：

```bash
--no-actuator-singularity-trace
```

这些是定向诊断回退，不代表旧 profile 重新成为正式配置。

### 3.5 Fix / Init 与数采边界

- 单臂精确归位到位后保持 Init 状态；操作员显式触发该侧 Home 重采后
  才恢复普通遥操。
- “双臂顺序精确归位”只执行左臂→右臂 qpos，不修改 tracker Home；
  双臂到位或任一步失败后进入常规 Hold。这个 Hold 需显式重采
  runtime Home 才能恢复，不会自动采 Home。
- GUI 可分别设置左/右臂精确归位速度倍率；`1.0` 对应
  `rm65_init_slow` 基线，默认均为 `5.0`。CLI 等价参数为
  `--init-arm-left-qpos-speed-scale` 和
  `--init-arm-right-qpos-speed-scale`。启动前归位、运行/数采过程归位、
  踏板归位和双臂顺序归位使用同一组倍率；归位进行中及 episode 忙碌
  期间不能修改。执行器按规划时间戳插值，不再按外层主循环调用次数推进，
  因而启动前、位控和力控链路不会因循环频率不同而改变归位速度。
- GUI、浏览器快捷键和终端 `t` 共用同一数采准入门：常规 Hold、
  Fix-Hold、exact-qpos 规划/执行/保持、Init 移动/Home 重采/失败恢复期间
  都不能新建 episode。已经打开的 episode 在 Hold/Fix-Hold 时暂停，恢复
  后进入新的 continuity segment。
- Fix/Init 的 Home 重采只计数新鲜且连续静止的 tracker 帧；网络中断、
  重复 timestamp、运动或采样窗漂移会继续等待或重置计数，不会将旧帧当成 Home。

详细标签、pause 和 continuity 语义见
[`DATASET_SCHEMA.md`](../../DATASET_SCHEMA.md#arm-mode-特殊状态标签)。

### 3.6 可选 Plus episode 整理

Plus 是外置、default-off 的 episode 级旁路，不改 RM65 raw writer、Parquet、媒体或 episode
成败。需要时显式启用：

```bash
python -m teleop_runtime.debug.run_vive_rm65_real_teleop --gui \
  --plus-episode-curation \
  --plus-annotation-root /path/to/plus_annotations
```

GUI 仅在开始前提供有限 `variant` 选择（配置 task profile 时）以及保存后一次
`disposition/reason` 提交；提交失败不影响已保存 episode。Plus 提交绑定当前
pending episode token，旁路写入期间不会允许删除/重编号。旧 v2.3 数据和显式的 v2.4
compatibility view 由同一 reader 读取，缺少 sidecar 时按“无 Plus 标注”处理。v2.4 native
capture identity 仍由上位 capture adapter 提供，Plus 不从目录名推断。

## 4. Impedance/力控数采

力控没有被 position profile 替代。当前设备全开 15→40 Hz 命令只在
[real README 的第 1.4 节](./README.md) 维护；历史来源保存在
[`docs/legacy/position_control/test-7.11.md`](../legacy/position_control/test-7.11.md)。

Position-only profile、奇异 branch、qref 平滑和 position planner 参数不得加入此命令。

## 5. Headless/兼容 collection

入口命令见 [real README 的第 1.6 节](./README.md)。该入口保留给
headless/兼容场景。已有无相机、q6-only、arm-only 和单臂示例继续由
[`teleop_runtime/collection/README.md`](../../teleop_runtime/collection/README.md#5-最小启动命令)
维护。新正式 GUI 数采不要改走 `teleop_runtime.collection.run_collect`。

UMI 是 collection 的显式子命令，验证命令见
[real README 的第 1.7 节](./README.md)。`--validate-only` 仍会读取配置引用的标定文件；左右
`teleop_records/umi_hand_eye/latest_{side}.json`（`side=left/right`）缺失或过期时会明确
失败，应先完成两侧 `utils.umi_hand_eye`。验证通过后去掉 `--validate-only` 才正式启动；完整流程见
[UMI 数采](./UMI.md)。

## 6. Debug 档位

- 正式长程数采：`POSITION_COLLECTION_PRODUCTION_ARGS`，保留预分配 actuator/singularity
  数值 trace，关闭高频 JSON/FK 写盘。
- Session summary：仅用于检查整体时序。
- Full/session forensic debug：只用于短时复现，会与 60/120 Hz 控制线程竞争 GIL。
- `NO_DEBUG_DISK_WRITE_ARGS`：临时关闭 standalone debug，不关闭正式 dataset。
- 正式热路径和 30 Hz 实测状态 poller 只读取 RM65 轻量状态；完整 FK 与 timing
  percentile 快照留给低频诊断调用，不能重新放回 60/120 Hz 控制链路。

不要把 full debug 当成正式采集默认档。debug 档位数组和检查字段见
[`teleop_args.sh`](./teleop_args.sh)。历史 debug 组合见
[`docs/legacy/position_control/test-7.11.md`](../legacy/position_control/test-7.11.md)。

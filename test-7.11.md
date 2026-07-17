已落地两个修复，没有执行任何真机运动。

主要改动：

- Replay collision debug 现在保存三个固定 transport-health 槽，包括成功快照、失败异常和无 hand 时的 `not_applicable`，并通过现有 ring/API 返回。[executor.py](/home/zc/HandMotion/Developing/DexTeleop/teleop_runtime/replay/executor.py:1301)
- 新增 episode 级模式契约校验。[execution_mode_contract.py](/home/zc/HandMotion/Developing/DexTeleop/teleop_runtime/execution_mode_contract.py:110)
  - 记录模式与 replay/deploy 所选模式不一致时拒绝。
  - 旧数据缺少 `arm_control_mode` 且进入真机链路时，按 `impedance` 处理。
  - dry-run 缺字段只警告。
- Replay 在连接和 preflight 前校验；deploy debug replay 在创建 backend 前校验。[run_policy.py](/home/zc/HandMotion/Developing/DexTeleop/teleop_runtime/deploy/run_policy.py:2689)
- Position-only collector 现在明确写入 `arm_control_mode=position`。[run_collect.py](/home/zc/HandMotion/Developing/DexTeleop/teleop_runtime/collection/run_collect.py:2594)
- 修正了 replay、deploy 和临时状态文档中过度陈述 backend receipt/持久化的内容。

验证结果：

- 模式契约与 replay execution：`53 passed`
- Replay GUI、频率、数采、deploy 相邻回归：`117 passed, 2 skipped`
- `git diff --check` 通过。
- 仅有既存的 `manotorch.egg-info/PKG-INFO` 缺失警告，不影响测试。

以下命令假定已经执行过 `source temp_args.md`，并设置：

- `DATASET_BASE`
- `FORCE_DATASET`、`POSITION_DATASET`：采集完成后的实际日期子目录
- `LEFT_GRIPPER_PORT`、`RIGHT_GRIPPER_PORT`

### 1. 设备全开数采：15→40 Hz 力控

```bash
source temp_args.md
taskset -c 0-11 python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  "${COMMON_REAL_TELEOP_ARGS[@]}" \
  "${DMTAC_CAPTURE_COMMON_ARGS[@]}" \
  "${DMTAC_A_ARGS[@]}" \
  "${DATASET_COMMON_ARGS[@]}" \
  "${IMPEDANCE_40HZ_ARGS[@]}" \
  "${CONTROL_DEBUG_SUMMARY_ARGS[@]}" \
  --dataset-root "$DATASET_BASE" \
  --collection-task-name dmtac_impedance_15_40hz \
  --collection-task-description "full-device bimanual impedance 15Hz capture / 40Hz backend" \
  --q6-config latest \
  --q6-config-required \
  --dmtac-config latest \
  --dmtac-config-required \
  --camera-config cameras/configs/rm65/devices/realsense.json \
  --write-parquet \
  --write-video
```

### 2A. 设备全开数采：15→60→120 Hz 纯位置位控（linear）

该命令是对照组。`POSITION_CONTROL_ARGS` 已明确包含
`--position-command-planner linear`，不使用速度/加速度连续规划。

```bash
source temp_args.md && taskset -c 0-11 python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  "${COMMON_REAL_TELEOP_ARGS[@]}" \
  "${DMTAC_CAPTURE_COMMON_ARGS[@]}" \
  "${DMTAC_A_ARGS[@]}" \
  "${DATASET_COMMON_ARGS[@]}" \
  "${POSITION_CONTROL_ARGS[@]}" \
  "${CONTROL_DEBUG_FULL_ARGS[@]}" \
  --dataset-root "$DATASET_BASE" \
  --collection-task-name dmtac_position_linear_15_60_120hz \
  --collection-task-description "full-device bimanual pure-position linear planner: 15Hz capture / 60Hz reference / 120Hz qcmd" \
  --q6-config latest \
  --q6-config-required \
  --dmtac-config latest \
  --dmtac-config-required \
  --camera-config cameras/configs/rm65/devices/realsense.json \
  --write-parquet \
  --write-video
```

### 2B. 设备全开数采：15→60→120 Hz 位置+速度位控（acceleration_limited）

该命令只在2A之后叠加`POSITION_ACCELERATION_LIMITED_ARGS`；设备、数采、频率、
碰撞IK和debug配置均与2A一致。规划器在新60 Hz安全目标到达时继承当前速度，并限制
速度和加速度；manual hold、碰撞/无有效解及stale target仍走快速保持。速度上限由
`--arm-max-angvel`直接调节，加速度上限由`--position-max-acceleration-rad-s2`
独立调节；异常50%降速不会同步降低加速度上限。

`POSITION_CONTROL_ARGS`中的两层EMA含义不同：

- `--position-post-ik-qpos-smoothing-alpha 0.14` 到
  `--position-post-ik-qpos-smoothing-fast-alpha 0.45`：左右臂分别依据各自有效 tracker
  线/角速度，在
  IK/安全限步之后、轨迹规划之前自适应；
- `--arm-follow-qpos-alpha 0.9`：直接作用于最终120 Hz SDK qpos，实际发送值会回写规划器；
  manual/collision/stale quick-stop绕过该EMA。

```bash
source temp_args.md && taskset -c 0-11 python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  "${COMMON_REAL_TELEOP_ARGS[@]}" \
  "${DMTAC_CAPTURE_COMMON_ARGS[@]}" \
  "${DMTAC_A_ARGS[@]}" \
  "${DATASET_COMMON_ARGS[@]}" \
  "${POSITION_CONTROL_ARGS[@]}" \
  "${POSITION_ACCELERATION_LIMITED_ARGS[@]}" \
  "${CONTROL_DEBUG_FULL_ARGS[@]}" \
  --dataset-root "$DATASET_BASE" \
  --collection-task-name dmtac_position_acceleration_limited_15_60_120hz \
  --collection-task-description "full-device bimanual position+velocity acceleration-limited planner: 15Hz capture / 60Hz reference / 120Hz qcmd" \
  --q6-config latest \
  --q6-config-required \
  --dmtac-config latest \
  --dmtac-config-required \
  --camera-config cameras/configs/rm65/devices/realsense.json \
  --write-parquet \
  --write-video
```

### 2C. URDF 根坐标系 X 轴（前后）专项诊断：15→60→120 Hz

此命令与 2B 使用相同的真实硬件位控链路，但 full debug 会额外记录在同一 URDF 根
（`desk`）帧下的三种 FK：原始 tracker 目标、最终 60 Hz `q_ref`、以及**仅在 SDK
accepted 后**的 120 Hz `q_cmd`。不读取或伪造硬件关节反馈，也不会改变任何控制量。

进入 RUNNING 后按以下顺序操作：静止 3 秒；仅沿 URDF 根 X 正向平移 8 秒；静止
3 秒；仅沿 X 负向平移 8 秒；静止 3 秒。另一只手保持静止，尽量不叠加 Y/Z 或姿态
变化。该操作会驱动真机，仍需按正常 GUI preflight、工作空间和现场安全流程执行。

```bash
source temp_args.md && taskset -c 0-11 python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  "${COMMON_REAL_TELEOP_ARGS[@]}" \
  "${DMTAC_CAPTURE_COMMON_ARGS[@]}" \
  "${DMTAC_A_ARGS[@]}" \
  "${DATASET_COMMON_ARGS[@]}" \
  "${POSITION_CONTROL_ARGS[@]}" \
  "${POSITION_ACCELERATION_LIMITED_ARGS[@]}" \
  "${CONTROL_DEBUG_FULL_ARGS[@]}" \
  --dataset-root "$DATASET_BASE" \
  --collection-task-name dmtac_position_root_x_diagnostic_15_60_120hz \
  --collection-task-description "URDF-root X diagnostic: source target, qref FK, and SDK-accepted qcmd FK" \
  --q6-config latest \
  --q6-config-required \
  --dmtac-config latest \
  --dmtac-config-required \
  --camera-config cameras/configs/rm65/devices/realsense.json \
  --write-parquet \
  --write-video
```

进程正常退出后，将本次 `teleop_records/control_debug/<timestamp>...` 替换进下面命令。
它按 `reference_epoch/reference_sequence` 精确关联 qref 和 SDK accepted qcmd，输出三个
根 X 残差的 p50/p95/max，以及最大的逐条样本（其中 `inactive_release` 是此次
inactive 收敛速度释放系数）：

```bash
python -m teleop_runtime.debug.analyze_position_root_x \
  --control-debug-dir teleop_records/control_debug/<timestamp>_vive_rm65_real_teleop \
  --side both \
  --threshold-cm 1 \
  --top 30
```

判读：`target - qref` 大而 `qref - accepted qcmd` 小，说明问题在 tracker/IK/静止
限速之前；前者小而后者大，说明问题在 120 Hz 规划或 SDK 命令链路；两者都小则该次
命令链路的 URDF 根 X 正常，需另查真实硬件跟随/机械侧。

### 2D. 位控 IK 迭代次数 A/B（仅 position，不影响力控）

最外层参数已经暴露。位控默认 `--position-ik-solve-min-iterations 2`、
`--position-ik-solve-max-iterations 2` 和 `--position-ik-damping 0.03`；力控仍固定为
最少/至多1次和 damping `0.01`。最少次数优先于 pose tolerance 与迭代时间预算，预算只会
截断超过最少次数的可选迭代。最新离线结果建议使用位置/旋转 `1.0/0.5`：

```bash
source temp_args.md && taskset -c 0-11 python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  "${COMMON_REAL_TELEOP_ARGS[@]}" \
  "${DMTAC_CAPTURE_COMMON_ARGS[@]}" \
  "${DMTAC_A_ARGS[@]}" \
  "${DATASET_COMMON_ARGS[@]}" \
  "${POSITION_CONTROL_ARGS[@]}" \
  "${CONTROL_DEBUG_FULL_ARGS[@]}" \
  --position-ik-orientation-weight 0.5 \
  --position-ik-solve-min-iterations 2 \
  --position-ik-solve-max-iterations 2 \
  --position-ik-damping 0.03 \
  --position-ik-adaptive-iterations \
  --position-ik-iteration-reserve-ms 5 \
  --dataset-root "$DATASET_BASE" \
  --collection-task-name dmtac_position_ik_1x05_adapt2_15_60_120hz \
  --collection-task-description "position-only IK: weights 1.0/0.5, min/max 2 solver iterations, damping 0.03" \
  --q6-config latest \
  --q6-config-required \
  --dmtac-config latest \
  --dmtac-config-required \
  --camera-config cameras/configs/rm65/devices/realsense.json \
  --write-parquet \
  --write-video
```

`--position-ik-orientation-weight` 同时覆盖 single/dual IK 的旋转权重，并将相应位置权重
固定为 `1.0`。它只在 position mode 生效，不修改共享 JSON；力控/阻抗模式传入会被拒绝。

如果后续要让 solver 在每帧预算内尝试更多迭代，可以继续提高安全上限，例如：

```bash
--position-ik-solve-max-iterations 4 \
--position-ik-solve-min-iterations 2 \
--position-ik-adaptive-iterations \
--position-ik-iteration-reserve-ms 5
```

自适应模式会保留 5ms 给后续候选筛选和安全检查；`--solve-max-iterations` 是上述参数
的兼容别名。力控/阻抗模式传入这些 position 参数会直接拒绝，避免误改力控链路。

INACTIVE 进入条件也暴露为 position-only 参数：

```bash
--position-inactive-enter-threshold-scale 0.5
```

`1.0` 完全保持原逻辑；`0.5` 将进入 INACTIVE 所需的线速度、角速度、平移窗口跨度、
旋转窗口跨度阈值都乘以 `0.5`，即四项都必须小于原阈值的一半才开始 quiet dwell。
`enter_dwell_sec=0.30s`、退出速度/位移阈值和退出 debounce 均不改变。有效范围为 `(0,1]`，
力控/阻抗入口拒绝非默认值。

候选 local-step 裁剪可单独做 A/B，例如：

```bash
--position-ik-local-step-clip-scale 2
```

默认值为 `1`。该参数仅放大候选分支评分时的 local-step 裁剪；最终 qref 的执行限速不变，
因此不会提高机械臂的最大执行速度。力控/阻抗模式传入该参数同样会直接拒绝。

### 2E. 位控关节速度 profile、qref 限速与 120 Hz planner

位控双臂 IK 使用的默认关节速度 profile 文件是：

```text
robot_motion/configs/joint_velocity_limits.json
```

默认加载路径由 `robot_motion/joint_limits.py` 的 `default_joint_velocity_limits_path()`
给出；双臂 retargeter 默认选择该 JSON 的 `profiles.rm65`。位控入口现在通过
`--position-joint-velocity-limit-profile` 独立选择 final qref cap profile；默认仍为
`rm65`。`rm65_position_2x/4x/8x` 只在显式选择时作用于位控，力控/阻抗模式传入非默认值
会直接拒绝，因此不再需要修改共享 `rm65`。

对 real position teleop，IK 侧真正使用的每关节步长是（单位统一为 rad）：

```text
qref_step_limit(j, dt) = min(
    deg2rad(joint_velocity_limits[selected_position_profile][j]) * dt,
    --arm-max-angvel[j] * dt
)
```

其中 `dt` 是实际外层参考周期（会使用实际 loop-dt 的受限估计）。因此只修改
`joint_velocity_limits.json` 并不会自动提速：当前 `--arm-max-angvel` 为
`[0.7, 0.7, 0.9, 1.5, 1.5, 2.0] rad/s`，已经低于默认 profile 的
`[85, 85, 95, 105, 120, 120] deg/s`，实际由 `--arm-max-angvel` 限制。

此前离线 A/B 的“8×”是对 **candidate-local clip 和 final qref cap 同时**乘以 8。
完整8×不再修改 `profiles.rm65`，而是显式选择 `rm65_position_8x`；最终仍与
`--arm-max-angvel` 逐关节取最小值。

8× `--arm-max-angvel` 换算为约 `[321,321,413,688,688,917] deg/s`，它仍是独立
profile 之外的最终 ceiling。在当前60 Hz IK/reference 下，相当于每个 outer qref 最多约
`[5.35,5.35,6.88,11.46,11.46,15.28] deg`；数值本身已说明8×只能作为上界档，
而不是当前推荐的首轮真机档。
这**不是机械臂额定速度，也不是可直接上真机的推荐配置**；它只是保证 profile 不会阻挡
离线 8× 因果试验。不要修改共享 `rm65`；逐级真机验证分别使用：

```bash
--position-joint-velocity-limit-profile rm65_position_2x
--position-joint-velocity-limit-profile rm65_position_4x
--position-joint-velocity-limit-profile rm65_position_8x
```

同时，若希望 120 Hz accepted qcmd 也能获得同等速度能力，必须调整 planner；只放宽
candidate/final qref cap 会使 120 Hz planner 继续成为瓶颈。planner 的限制来源如下：

| 限制 | 参数/来源 | 8×时的对应值 | 是否需要调整 |
| --- | --- | --- | --- |
| 最大速度与默认 command-step 上限 | `--arm-max-angvel`；backend 以它建立 planner 的 `max_velocity`，未显式设置独立 step guard 时还会建立 `max_command_step` | `5.6,5.6,7.2,12,12,16 rad/s` | 若要测试完整 8×持续速度，必须同步调整；仅改 JSON 不影响 planner。 |
| 独立 planner 单步保护 | `--position-max-command-step-deg`，作用于120 Hz acceleration-limited proposal、位于 final EMA 之前 | `0.675 deg` | 本轮平滑性测试显式启用；它限制峰值步长，但不降低上述持续速度上限。 |
| 最大加速度 | `--position-max-acceleration-rad-s2`，当前 `7,7,9,15,15,20 rad/s^2` | 保持 0.10 s ramp 的等比例值为 `56,56,72,120,120,160 rad/s^2` | 必须同步调整，否则到达 8×速度约需 0.8 s，仍会明显滞后。 |
| 目标速度反馈 horizon | `--position-velocity-ramp-sec`，当前 `0.10 s` | 保持 `0.10 s` | 无需修改；在速度与加速度均按 8×缩放时，归一化 ramp 行为不变。 |
| accepted qref 目标速度低通 | `--position-target-velocity-filter-tau-sec` | `0.03 s` | 对 qref 差分得到的 feed-forward 速度做有界因果低通；`0` 可严格旁路，便于 A/B。只作用于 acceleration-limited 位控 planner。 |
| final 120 Hz EMA | `--arm-follow-qpos-alpha`，当前 `0.9` | `1.0` 为无额外 EMA | 不是硬速度上限，但高速度 A/B 中会增加相位滞后；先保留 0.9 记录结果，再单独 A/B 1.0。 |
| endpoint safety envelope | acceleration planner 内部根据当前速度、加速度和目标端点计算 | 无单独“放宽”参数 | 不应关闭；提高前两项后它仍可能标记为 limited，但这是保证可刹停的必要约束。 |

`--position-command-planner acceleration_limited` 不应改成关闭或改为绕过 safety envelope。
此前 debug 中 `command_step_limited=0`，说明旧的 `arm-max-angvel × command_dt` 上限没有解决
抖动。本轮改为显式 `0.675°` 独立 step guard，用来约束120 Hz峰值步长，同时保留8×持续速度。

若只为离线/无机械臂 dry-run 对齐 8×参数，参数组为：

```bash
--arm-max-angvel 5.6,5.6,7.2,12,12,16.0 \
--position-max-acceleration-rad-s2 56,56,72,120,120,160 \
--position-max-command-step-deg 0.675 \
--position-target-velocity-filter-tau-sec 0.03 \
--position-velocity-ramp-sec 0.10
```

真机不要直接使用该组。推荐先完成 profile 名称/路径的显式 CLI 支持，再按较小的、逐关节
不超过已确认安全速度的值进行 A/B；每次检查 `position_command_*.jsonl` 中的
`trajectory_max_velocity_rad_s`、`trajectory_max_acceleration_rad_s2`、
`trajectory_endpoint_velocity_limited`、`trajectory_acceleration_limited` 及最终
`accepted_qcmd` FK 误差。

### 2F. 显式请求的全流程 8× 上界测试：1.0 / 1.0 IK，15→60→120 Hz

这是一条**受控真机上界测试**，不是日常位控配置。它固定
`robot_motion/configs/rm65_ik_tuning.json` 的 `orientation_priority`，即位置/旋转权重
均为 `1.0`；保持固定2次 IK solve，以便本次只检验限速放宽的端到端效果。

“全流程 8×”在本测试中严格指：

- candidate-local step clip：`--position-ik-local-step-clip-scale 8`；
- final 60 Hz qref step cap：`--arm-max-angvel` 8×，且 joint-velocity profile 同步不再更严格；
- 120 Hz planner 最大速度：`--arm-max-angvel` 8×；
- 120 Hz planner 最大加速度：`--position-max-acceleration-rad-s2` 8×；
- 120 Hz planner 独立单步保护：`--position-max-command-step-deg 0.675`。
- accepted qref 目标速度低通：`--position-target-velocity-filter-tau-sec 0.03`。

最后一项不会降低 planner 的持续速度上限。它在 `--arm-follow-qpos-alpha 0.9` 之前生效，
因此正常跟踪时最终 accepted qcmd 的单步上限约为 `0.675° × 0.9 = 0.6075°`。

命令通过 `--position-joint-velocity-limit-profile rm65_position_8x` 选择独立 profile，
不修改共享 `rm65`，也不改变力控/阻抗链路。保留 collision、joint-limit、inactive 规则及 planner 的 braking
safety envelope；后者不是可关闭的速度参数。

```bash
source temp_args.md && taskset -c 0-11 python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  "${COMMON_REAL_TELEOP_ARGS[@]}" \
  "${DMTAC_CAPTURE_COMMON_ARGS[@]}" \
  "${DMTAC_A_ARGS[@]}" \
  "${DATASET_COMMON_ARGS[@]}" \
  "${POSITION_CONTROL_ARGS[@]}" \
  "${POSITION_ACCELERATION_LIMITED_8X_1X1_ARGS[@]}" \
  "${CONTROL_DEBUG_FULL_ARGS[@]}" \
  --dataset-root "$DATASET_BASE" \
  --collection-task-name dmtac_position_1x1_fullpath_8x_step0675_15_60_120hz \
  --collection-task-description "controlled position-only test: IK 1.0/1.0, qref and planner V/A 8x, planner step guard 0.675deg" \
  --q6-config latest \
  --q6-config-required \
  --dmtac-config latest \
  --dmtac-config-required \
  --camera-config cameras/configs/rm65/devices/realsense.json \
  --write-parquet \
  --write-video
```

该命令会驱动真机；在 GUI preflight 和工作空间检查完成后，先只做短距离、单方向、低幅度
动作，再逐步增加范围。结束后确认：`position_summary.json` 的
`position_ik_local_step_clip_scale=8`、`position_joint_velocity_limit_profile=rm65_position_8x`
和固定2次 iteration；action debug 的首个
`event=config` 中 `arm_max_angvel_rad_s=[5.6,5.6,7.2,12,12,16]` 且
`position_max_command_step_deg=0.675`；以及
`position_command_150.jsonl` 的 `trajectory_max_velocity_rad_s` 与
`trajectory_max_acceleration_rad_s2=[56,56,72,120,120,160]`。`orientation_priority=1.0/1.0`
由命令显式固定的 tuning JSON 确认。同时在 `position_reference.jsonl` 确认
`joint_velocity_limit_max_step_deg` 不再被旧 profile 截断，并以
`position_command_150.jsonl` 的 accepted qcmd FK 误差作为最终评价，不以 qref 误差替代。

### 2G. 最新推荐复测：1.0/0.5 + adaptive max 2 + INACTIVE 0.5×

该命令沿用0.675°保护，只替换本次需要验证的位控参数。共享
`rm65_ik_tuning.json` 和共享 `rm65` joint-velocity profile 都不会被修改，力控链路不受影响。
当前推荐从 `rm65_position_2x` 开始，确认后再仅替换为4x；最新两组离线数据中8x对位置
相较4x没有额外收益。

```bash
source temp_args.md && taskset -c 0-11 python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  "${COMMON_REAL_TELEOP_ARGS[@]}" \
  "${DMTAC_CAPTURE_COMMON_ARGS[@]}" \
  "${DMTAC_A_ARGS[@]}" \
  "${DATASET_COMMON_ARGS[@]}" \
  "${POSITION_CONTROL_ARGS[@]}" \
  "${POSITION_ACCELERATION_LIMITED_8X_1X05_ADAPT2_TIGHT05_ARGS[@]}" \
  "${CONTROL_DEBUG_FULL_ARGS[@]}" \
  --dataset-root "$DATASET_BASE" \
  --collection-task-name dmtac_position_1x05_adapt2_tight05_15_60_120hz \
  --collection-task-description "position-only: IK 1.0/0.5, min/max2, damping0.03, inactive enter thresholds 0.5x" \
  --q6-config latest \
  --q6-config-required \
  --dmtac-config latest \
  --dmtac-config-required \
  --camera-config cameras/configs/rm65/devices/realsense.json \
  --write-parquet \
  --write-video
```

运行后先检查 `position_summary.json`：

- `position_ik_orientation_weight=0.5`；
- `position_ik_solve_min_iterations=2`；
- `position_ik_solve_max_iterations=2`；
- `position_ik_damping=0.03`；
- `position_ik_adaptive_iterations=true`；
- `position_inactive_enter_threshold_scale=0.5`。
- `position_joint_velocity_limit_profile=rm65_position_2x`。

再检查 action debug 中 `ik_solve_iterations` 的1/2次分布和
`target_activity.<side>.state`。有效 INACTIVE 进入阈值应为：线速度 `0.015m/s`、角速度
`0.09rad/s`、0.1秒窗口平移跨度 `0.00125m`、旋转跨度 `0.0125rad`；dwell仍为0.30秒。

### 2H. 保持现有 TCP 的最新最小设备遥操版本

该版本继承2G的最新位控、IK和平滑参数，但只启动双臂 RM65、VIVE 输入和 GUI。
`NO_GRIPPER_COMMON_ARGS` 显式设置 `--hand-backend none`、`--no-camera` 和
`--no-collection`，因此不会发现或连接 Q6、DMTAC 夹爪、DMTAC 视触觉及 RealSense，
也不会创建正式数据集。

这里的“不带夹爪”仅表示**不连接、不读取、不控制夹爪设备**，不表示机械末端或控制坐标系
发生变化。左右臂仍使用与设备全开版本完全相同的 `rm65_left/right_DMTAC_M.json`，因此
DMTAC_M 的 TCP、工具配置和双臂 IK 目标语义保持不变；默认
`dual_arm_hand_proxy_open` 碰撞场景也保持不变。

命令中不要展开 `DATASET_COMMON_ARGS`（其中的 `--wrist-depth-mask` 会要求相机）、
`DMTAC_CAPTURE_COMMON_ARGS`、`DMTAC_A_ARGS`，也不要传入 `--camera-config`、
`--q6/dmtac-config*` 或 `--write-parquet/video`。

默认碰撞场景中的固定开放手形代理只用于保守避碰，并继续提供桌面和相机杆碰撞检查；
它不会创建夹爪 backend 或发送夹爪命令。不要额外覆盖 `--collision-scene-profile`，否则
就不再与最新设备全开配置对齐。

```bash
source temp_args.md && taskset -c 0-11 python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  "${NO_GRIPPER_COMMON_ARGS[@]}" \
  "${POSITION_CONTROL_ARGS[@]}" \
  "${POSITION_ACCELERATION_LIMITED_8X_1X05_ADAPT2_TIGHT05_ARGS[@]}" \
  "${CONTROL_DEBUG_FULL_ARGS[@]}"
```

这仍是2F/2G定义的受控 8× 真机测试配置，不是日常低速配置。启动前必须满足2F的 joint
velocity profile 和现场安全前置条件；GUI preflight 通过后先做短距离、单方向、低幅度动作。
启动日志应显示 `hand_backend=none`、`--no-camera` 和 collection disabled；左右 hand
backend 均为空，不应出现 Q6/DMTAC、RealSense 设备连接或夹爪 warmup；左右 tool profile
仍应为 `DMTAC_M`，TCP 数值应与设备全开版本一致。

### 2I. 位控正式数采进程隔离 + max4 两组严格 A/B

这两条命令用于正式数采，不写 standalone debug。位控主进程由 `taskset -c 0-7`
约束在物理核心0-7；数采子进程固定核心8-11、nice=5。以下两条命令额外展开
`DMTAC_TACTILE_WORKER_2CPU_ARGS`，将触觉读取子进程限制到核心12-13、nice=15，
释放14-15给系统和其他后台任务。远端 Ryzen 9 9950X 的0-15是16个不同物理核心，
因此三条数据面不会共享同一个物理核心。

触觉CPU亲和性不是生产代码常量：可在 `source temp_args.md` 前设置
`DMTAC_TACTILE_WORKER_CPUS`，或者在命令末尾展开2核/4核profile。后展开者优先：

```bash
# 例：在source前选择2核；不设置时默认兼容值为12,13,14,15
export DMTAC_TACTILE_WORKER_CPUS=12,13
source temp_args.md

# 或者source后，在正式命令中选一个：
"${DMTAC_TACTILE_WORKER_2CPU_ARGS[@]}"  # 12,13
"${DMTAC_TACTILE_WORKER_4CPU_ARGS[@]}"  # 12,13,14,15
```

正式数采采用预分配共享内存传递相机帧和原始触觉ndarray；每槽额外预留8 MiB，避免
约1.23 MiB/sample的四传感器原始触觉数据在父进程内经Queue pickle。sample metadata/action组装、schema、视频编码、
触觉写入、统计和Parquet finalize全部在spawn子进程。主进程只在已有低优先级snapshot线程
冻结因果机械臂/夹爪状态并复制相机数组。整个sample队列或共享内存槽耗尽时，当前episode
会显式进入 `collection_overload:*` 暂停，不能反向阻塞60 Hz IK/120 Hz下发，也不会静默
继续产生缺critical row的数据集。

`--collection-worker-mode thread` 是完整回退开关。`auto` 只对position解析成process，
force/impedance仍保持原thread路径；显式给力控传process会被CLI拒绝。

#### A. 位置/旋转 1.0/0.5，min2/max4

```bash
source temp_args.md && taskset -c 0-7 python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  "${COMMON_REAL_TELEOP_ARGS[@]}" \
  "${DMTAC_CAPTURE_COMMON_ARGS[@]}" \
  "${DMTAC_A_ARGS[@]}" \
  "${DMTAC_TACTILE_WORKER_2CPU_ARGS[@]}" \
  "${DATASET_COMMON_ARGS[@]}" \
  "${POSITION_CONTROL_ARGS[@]}" \
  "${POSITION_ACCELERATION_LIMITED_8X_1X05_ADAPT2_TIGHT05_ARGS[@]}" \
  "${POSITION_COLLECTION_PROCESS_ARGS[@]}" \
  "${POSITION_IK_1X05_MAX4_ARGS[@]}" \
  "${NO_DEBUG_DISK_WRITE_ARGS[@]}" \
  --dataset-root "$DATASET_BASE" \
  --collection-task-name dmtac_position_process_1x05_min2_max4_15_60_120hz \
  --collection-task-description "position formal collection: process-isolated writer, IK 1.0/0.5 min2 max4 damping0.03" \
  --q6-config latest \
  --q6-config-required \
  --dmtac-config latest \
  --dmtac-config-required \
  --camera-config cameras/configs/rm65/devices/realsense.json \
  --write-parquet \
  --write-video
```

#### B. 位置/旋转 1.0/0.35，min2/max4

```bash
source temp_args.md && taskset -c 0-7 python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  "${COMMON_REAL_TELEOP_ARGS[@]}" \
  "${DMTAC_CAPTURE_COMMON_ARGS[@]}" \
  "${DMTAC_A_ARGS[@]}" \
  "${DMTAC_TACTILE_WORKER_2CPU_ARGS[@]}" \
  "${DATASET_COMMON_ARGS[@]}" \
  "${POSITION_CONTROL_ARGS[@]}" \
  "${POSITION_ACCELERATION_LIMITED_8X_1X05_ADAPT2_TIGHT05_ARGS[@]}" \
  "${POSITION_COLLECTION_PROCESS_ARGS[@]}" \
  "${POSITION_IK_1X035_MAX4_ARGS[@]}" \
  "${NO_DEBUG_DISK_WRITE_ARGS[@]}" \
  --dataset-root "$DATASET_BASE" \
  --collection-task-name dmtac_position_process_1x035_min2_max4_15_60_120hz \
  --collection-task-description "position formal collection: process-isolated writer, IK 1.0/0.35 min2 max4 damping0.03" \
  --q6-config latest \
  --q6-config-required \
  --dmtac-config latest \
  --dmtac-config-required \
  --camera-config cameras/configs/rm65/devices/realsense.json \
  --write-parquet \
  --write-video
```

两组除旋转权重外必须保持完全一致。启动日志应出现
`worker_mode=process`；最终episode metadata/summary中检查：

- `collection_worker_mode=process`；
- `collection_integrity_valid=true` 且 `collection_integrity_reasons=[]`；
- `collection_process_rejected_busy=0`；
- `sample_assembly_skipped_busy=0`；
- process worker `worker_error=null`；
- `submitted_samples == completed_samples`；
- `writer_pool_stats_after_flush.timing_ms` 中检查 `parent.pack_shared`、
  `parent.queue_put` 及各 `worker.handle.*` 的p95/max；
- `shared_payload_bytes.p50` 应覆盖三路相机和约1.23 MiB原始触觉，且
  `parent.queue_put` 只传共享内存描述符；
- episode未出现 `collection_overload:*` pause segment。

这两条命令最后展开 `NO_DEBUG_DISK_WRITE_ARGS`，所以不会生成
`position_summary.json`/action debug。若需要确认四次iteration分布，只做一轮短时诊断：临时移除
`NO_DEBUG_DISK_WRITE_ARGS`，加入 `CONTROL_DEBUG_FULL_ARGS`，正常退出后查看debug；不要把full debug
作为正式长程数采配置。

### 2J. 下一轮 elbow singularity 卡住验证：数采 + debug（直接运行）

最新一组 `20260717_135323_617815` debug 产生于 elbow 候选优先/显式回退代码更新之前；其中
右臂出现两段 `elbow_singularity -> hold_entry`。下一轮使用当前远端代码同时采集正式数据和
位控/action debug，以便把相机、触觉、15 Hz主action、60 Hz qref和120 Hz qcmd按时间
对齐检查。控制配置保持该次测试的 15 Hz sample / 60 Hz IK与qref / 120 Hz qcmd、
位置/旋转权重1.0/0.35、adaptive min2/max4、0.675° command envelope和现有平滑参数。

资源隔离与2I一致：遥操主进程使用核心0-7，数采spawn子进程使用8-11，四路触觉读取进程
限制到12-13。默认命令使用“session full + 抽样action/timing”：不再创建episode级重复full
sidecar，action限制15 Hz，timing限制5 Hz并将slow阈值提高到40 ms。状态/故障变化不受timing
抽样限制。它保留完整session qref/qcmd，且避免上一轮每帧重复JSON写盘。

命令末尾显式开启 wrist/elbow 两类非奇异候选优先，保证其覆盖前面数组或环境中的同名配置。
主验证命令**不得**添加 `--no-position-prefer-non-elbow-singular-branch`；该参数只用于之后的
明确回退A/B。

```bash
source temp_args.md && taskset -c 0-7 python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  "${COMMON_REAL_TELEOP_ARGS[@]}" \
  "${DMTAC_CAPTURE_COMMON_ARGS[@]}" \
  "${DMTAC_A_ARGS[@]}" \
  "${DMTAC_TACTILE_WORKER_2CPU_ARGS[@]}" \
  "${DATASET_COMMON_ARGS[@]}" \
  "${POSITION_CONTROL_ARGS[@]}" \
  "${POSITION_ACCELERATION_LIMITED_8X_1X05_ADAPT2_TIGHT05_ARGS[@]}" \
  "${POSITION_COLLECTION_PROCESS_ARGS[@]}" \
  "${POSITION_IK_1X035_MAX4_ARGS[@]}" \
  "${POSITION_COLLECTION_AB4_SAMPLED_FULL_ARGS[@]}" \
  --position-prefer-non-wrist-singular-branch \
  --position-prefer-non-elbow-singular-branch \
  --dataset-root "$DATASET_BASE" \
  --collection-task-name dmtac_position_elbow_singularity_validation_1x035_max4_15_60_120hz \
  --collection-task-description "position elbow-singularity candidate/fallback validation: data+sampled-session-debug, IK 1.0/0.35 min2 max4" \
  --q6-config latest \
  --q6-config-required \
  --dmtac-config latest \
  --dmtac-config-required \
  --camera-config cameras/configs/rm65/devices/realsense.json \
  --write-parquet \
  --write-video
```

同一条命令可按下面顺序做严格A/B；每轮只替换上面这一项，不要与历史
`CONTROL_DEBUG_FULL_ARGS`叠加：

1. `"${POSITION_COLLECTION_AB1_NO_DEBUG_ARGS[@]}"`：正式数采、全部独立debug写盘关闭；
2. `"${POSITION_COLLECTION_AB2_SUMMARY_ARGS[@]}"`：只保留session summary；
3. `"${POSITION_COLLECTION_AB3_SESSION_FULL_ARGS[@]}"`：完整session qref/qcmd，不写episode重复sidecar；
4. `"${POSITION_COLLECTION_AB4_SAMPLED_FULL_ARGS[@]}"`：在第3档上增加15 Hz action和最多5 Hz timing。

四档分别使用不同`--collection-task-name`，建议在现有名字末尾追加
`_ab1_no_debug`、`_ab2_summary`、`_ab3_session_full`、`_ab4_sampled_full`，避免写入同一数据集。

该命令会连接并驱动双臂、夹爪、触觉和相机真机。完成 GUI preflight、设备检查和工作空间
检查后，开始一个正式episode，再复现此前自然出现问题的连续低幅往返动作；不要为了触发
日志主动靠近已知奇异位形。若自然出现卡住，保持 tracker 不动约2秒，记录大致时刻，正常
结束episode，然后安全退出并等待 dataset writer 和 debug recorder 全部正常关闭。

正常退出后，先确认新数据集的episode完整且 `collection_integrity_valid=true`、
`submitted_samples == completed_samples`，再检查session级 `position_summary.json` 和 action debug
首个 `event=config`：

- `position_prefer_non_elbow_singular_branch=true`；
- `position_ik_orientation_weight=0.35`；
- `position_ik_solve_min_iterations=2`、`position_ik_solve_max_iterations=4`；
- `position_reference_hz=60`、`arm_control_hz=120`；
- `episode_sidecar_requested=false`、`debug_action_sample_hz=15`、
  `debug_timing_log_sample_hz=5`、`debug_timing_threshold_sec=0.04`；
- 第4档不应出现 `--no-debug-disk-write`，否则本轮没有足够信息验证修复。

在疑似奇异区间，action debug 必须满足下面两条路径之一：

- 找到替代候选：`ik_candidate_selection.candidate_eval_mode=base_then_auxiliary_singularity_search`，
  `position_singularity_preference.selected.any_elbow_singular=false`，backend gate应为 `pass`；
- 没有可行替代候选：`position_singularity_preference.fallback_to_singular=true`，对应侧
  `position_elbow_singularity_fallback_requested=true`、
  `position_elbow_singularity_fallback_allowed=true`，backend gate应为
  `allow_position_elbow_singular`，而不是 `hold_entry`。

若仍出现 `target_singularity_label=elbow_singularity` 且 `singularity_gate_status=hold_entry`，
保留该次完整目录和 action debug；这将说明候选层证书没有传到backend，或SDK分类与J3代理
不一致，不能再归因于120 Hz发送或硬件拒收。

### 2K. 120 Hz real-like actuator辨识：独立5 ms真机反馈sidecar

现有正式数据可以对齐15 Hz主action、60 Hz accepted qref和120 Hz实际发送qcmd，但正式
Parquet中的机械臂反馈只按15 Hz sample保存；action debug中的
`observation_backend_by_side.*.current_qpos`在部分运行里来自`last_commanded_action`，不能当作
动态真机反馈。因此，已有数据足够重放控制器/planner，却不足以可靠辨识8.33 ms级执行器延迟、
跟随带宽、死区和速度连续性。

下面的sidecar与遥操/数采框架独立，不导入`teleop_runtime`，也不发送move、servo、CAN-FD或
任何其他运动命令。它只临时设置RM65 UDP状态上报为5 ms，将原始包、解析后的关节位置/速度、
本机wall/monotonic接收时间写入独立JSONL，结束后恢复启动前逐字段保存的完整配置。
RM65协议中的原始`cycle`是5 ms间隔的计数，因此`cycle=1`表示5 ms/200 Hz；命令行仍使用
易读的`--cycle-ms 5`，脚本会安全换算为控制器字段`cycle=1`。

恢复保护是强制的：修改前先fsync recovery journal、双臂做原配置set/get回读、启动独立guardian，
临时设置与恢复都必须set/get一致；父进程被kill或SSH断开时guardian会恢复；若整机掉电导致当时
无法连接控制器，未完成journal会保留，下一次运行capture前必须先恢复成功，否则拒绝继续修改。
不要删除`.recovery.json`或`.guardian.log`。若不能接受掉电期间只能在主机恢复后执行恢复，放弃
该测试，已有软件无法在无电/无网络时对控制器作即时回写。

先确保没有遥操、replay或其他RM65 UDP监听进程在运行。在终端A做只读检查（不会改配置）：

```bash
python -m hardware.rm65.utils.scripts.rm65_udp_actuator_trace inspect \
  --left-arm-ip 192.168.101.21 \
  --right-arm-ip 192.168.1.20 \
  --left-local-ip 192.168.101.10 \
  --right-local-ip 192.168.1.10 \
  --push-port 8099
```

确认双臂均可读后，仍在终端A启动独立采集器：

```bash
python -m hardware.rm65.utils.scripts.rm65_udp_actuator_trace capture \
  --left-arm-ip 192.168.101.21 \
  --right-arm-ip 192.168.1.20 \
  --left-local-ip 192.168.101.10 \
  --right-local-ip 192.168.1.10 \
  --push-port 8099 \
  --cycle-ms 5 \
  --duration-sec 300 \
  --output-root teleop_records/actuator_feedback \
  --apply \
  --i-understand-no-motion
```

只有看到`READY trace=...`后，才在终端B运行2J的“数采 + debug”完整命令。按2J的小范围动作
复现要求操作，先正常结束episode并退出遥操，等dataset/debug recorder完成关闭，再回终端A
按Ctrl-C。不要先杀终端A；固定300秒到期也会自动进入恢复。

成功必须同时看到终端A最后一行：

```text
RESTORED verified=true rows=<nonzero> dropped=0 trace=<path>
```

并检查同目录文件：

- `rm65_udp_actuator_trace_<timestamp>.recovery.json`的`status`必须为`restored`，左右臂都在
  `restored_config_by_side`中，`restore_failures=[]`；
- trace最后一个`event=stop`必须为`controller_config_restored=true`、`queue_drop_count=0`；
- `receiver_counters_by_side`左右臂的`feedback`、`joint_position`、`joint_speed`都必须大于0，
  `unexpected_source=0`；
- feedback应同时包含`receive_wall_time_ns`、`receive_monotonic_time_ns`、原始
  `payload_base64`、`joint_position_deg`与`joint_speed_deg_s`。脚本缺少其中任一侧的位置或
  速度字段时，会在恢复配置后以失败退出，这份数据不能用于执行器辨识。

如果终端A没有打印`RESTORED verified=true`，不要再次启动遥操。使用错误信息指向的journal执行：

```bash
python -m hardware.rm65.utils.scripts.rm65_udp_actuator_trace restore \
  --journal teleop_records/actuator_feedback/<对应文件>.recovery.json
```

恢复命令也必须打印`RESTORED verified=true`；随后重新运行只读`inspect`，与journal内双臂
`original_config`逐字段核对。只有这两项都通过后才能继续真机测试。

### 3. Replay：力控数据 15→40 Hz

```bash
python -m teleop_runtime.replay \
  --dataset-root "$FORCE_DATASET" \
  --episode 0 \
  --active-sides both \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --left-tool-profile hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json \
  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json \
  --arm-control-mode impedance \
  --arm-control-hz 40 \
  --no-arm-canfd-follow \
  --force-tare-sec 3 \
  --strict-tool-profile \
  --end-effectors-enabled \
  --hand-backend dmtac_gripper \
  --left-gripper-port "$LEFT_GRIPPER_PORT" \
  --right-gripper-port "$RIGHT_GRIPPER_PORT" \
  --arm-web-hardware-output \
  --host 127.0.0.1 \
  --port 8765
```

### 4. Replay：位控数据 15→75→150 Hz

```bash
python -m teleop_runtime.replay \
  --dataset-root "$POSITION_DATASET" \
  --episode 0 \
  --active-sides both \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --left-tool-profile hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json \
  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json \
  --arm-control-mode position \
  --position-reference-hz 75 \
  --arm-control-hz 150 \
  --arm-canfd-follow \
  --position-collision-required \
  --position-collision-debug \
  --end-effectors-enabled \
  --hand-backend dmtac_gripper \
  --left-gripper-port "$LEFT_GRIPPER_PORT" \
  --right-gripper-port "$RIGHT_GRIPPER_PORT" \
  --arm-web-hardware-output \
  --host 127.0.0.1 \
  --port 8765
```

两条 replay 命令不会启动即运动；需要在 GUI 中依次完成 `Preflight → Move Start → Start` 并显式开启输出。

## 位控遥操作速度连续规划：建议复测动作与检查项

上面的2A与2B分别是纯位置和位置+速度全设备数采命令，不要在同一次进程中切换模式。
2B 的 60 Hz 最终安全 `q_ref` 历史用于因果估计 `qd_ref`，120 Hz 记录
`q_cmd/qd_cmd/qdd_cmd`、速度/加速度/单步/envelope限幅、异常50%降速、软件quick-stop、
proposal/SDK结果及时序。VIVE原始速度仍只用于观测，不直接驱动本规划器。

为便于把卡顿和速度变化对齐，建议 runtime home 后只做一轮约 30 秒的受控小范围运动：

1. 静止 5 秒。
2. 单方向慢速平移 8 秒，尽量保持手腕姿态不变。
3. 在原地停止 5 秒。
4. 沿相反方向慢速返回 8 秒，再静止 4 秒。

不要为了触发问题主动靠近已知奇异位形或碰撞边界。如果自然出现 singularity hold，
保持 tracker 不动 2 秒后安全退出，并记录大致发生时刻。退出后需要检查最新文件：

- `teleop_records/control_debug/<latest>/position_reference.jsonl`
- `teleop_records/control_debug/<latest>/position_command_150.jsonl`
- `teleop_records/control_debug/<latest>/position_timing.jsonl`
- `teleop_records/control_debug/<latest>/position_summary.json`
- `teleop_records/action_debug/action_debug_<latest>.jsonl`

重点检查 `position_command_150.jsonl` 的：

- `metadata.position_command_planner=acceleration_limited`
- `trajectory_qd_reference_rad_s / trajectory_qd_command_rad_s / trajectory_qdd_command_rad_s2`
- `trajectory_proposal_token / trajectory_generation / trajectory_target_preempted`
- `trajectory_acceleration_limited / trajectory_command_step_limited / trajectory_safety_envelope_limited`
- `trajectory_braking_guard_qpos / trajectory_braking_guard_direction`
- `trajectory_endpoint_velocity_limited / trajectory_safety_velocity_infeasible`
- `trajectory_qdd_limit_exceeded_after_safety_clamp`（正常运动必须始终为 `false`）
- `trajectory_mode / trajectory_reason / trajectory_speed_scale / trajectory_quick_stop`
- `sdk_send_gap_sec / skipped_tick_count / sdk_accepted / transport_block_reason`

### 临时关闭全部 debug 写盘

需要验证 debug recorder 是否影响实时性时，在原运行命令最后追加：

```bash
"${NO_DEBUG_DISK_WRITE_ARGS[@]}"
```

或者直接追加 `--no-debug-disk-write`。该总开关优先级高于
`CONTROL_DEBUG_FULL_ARGS` 以及单独的 action/timing/IK debug 写盘参数，会关闭：

- control debug 的 reference/command/timing sidecar；
- standalone action 和 timing JSONL；
- GUI IK debug log；
- legacy `--record` session JSONL；
- fix-arm pose snapshot 和 lightweight debug trajectory。

console timing、IK profile、watchdog 和正式 LeRobot dataset collection 不受影响。
因此可以保留原测试命令的全部参数，只在末尾追加该开关做真机 A/B。开启 full control
debug 时，运行期间也不再周期性生成昂贵的 percentile summary checkpoint；
`position_summary.json` 会在进程正常关闭 recorder 时一次性生成，所以必须正常退出后再检查。

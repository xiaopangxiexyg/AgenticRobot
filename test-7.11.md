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

- `--position-post-ik-qpos-smoothing-alpha 0.18` 到
  `--position-post-ik-qpos-smoothing-fast-alpha 0.7`：依据有效 tracker 线/角速度，在
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

最外层参数已经暴露。默认 `--position-ik-solve-max-iterations 1` 保持现有行为；建议
先使用固定 2 次做单变量 A/B：

```bash
source temp_args.md && taskset -c 0-11 python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  "${COMMON_REAL_TELEOP_ARGS[@]}" \
  "${DMTAC_CAPTURE_COMMON_ARGS[@]}" \
  "${DMTAC_A_ARGS[@]}" \
  "${DATASET_COMMON_ARGS[@]}" \
  "${POSITION_CONTROL_ARGS[@]}" \
  "${CONTROL_DEBUG_FULL_ARGS[@]}" \
  --position-ik-solve-max-iterations 2 \
  --dataset-root "$DATASET_BASE" \
  --collection-task-name dmtac_position_ik_iter2_15_60_120hz \
  --collection-task-description "position-only IK A/B: fixed 2 solver iterations" \
  --q6-config latest \
  --q6-config-required \
  --dmtac-config latest \
  --dmtac-config-required \
  --camera-config cameras/configs/rm65/devices/realsense.json \
  --write-parquet \
  --write-video
```

如果要让 solver 在每帧预算内尽可能多迭代，设置一个安全上限并打开自适应模式，例如：

```bash
--position-ik-solve-max-iterations 4 \
--position-ik-adaptive-iterations \
--position-ik-iteration-reserve-ms 5
```

自适应模式会保留 5ms 给后续候选筛选和安全检查；`--solve-max-iterations` 是上述参数
的兼容别名。力控/阻抗模式传入这些 position 参数会直接拒绝，避免误改力控链路。

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
给出；双臂 retargeter 默认选择该 JSON 的 `profiles.rm65`。当前运行入口没有暴露
profile 名称或 JSON 路径的 CLI，因此在**不改代码**的现状下，修改 `profiles.rm65` 才会
影响此位控链路；仅新增一个 profile 名称不会被本命令选中。

对 real position teleop，IK 侧真正使用的每关节步长是（单位统一为 rad）：

```text
qref_step_limit(j, dt) = min(
    deg2rad(joint_velocity_limits["rm65"][j]) * dt,
    --arm-max-angvel[j] * dt
)
```

其中 `dt` 是实际外层参考周期（会使用实际 loop-dt 的受限估计）。因此只修改
`joint_velocity_limits.json` 并不会自动提速：当前 `--arm-max-angvel` 为
`[0.7, 0.7, 0.9, 1.5, 1.5, 2.0] rad/s`，已经低于默认 profile 的
`[85, 85, 95, 105, 120, 120] deg/s`，实际由 `--arm-max-angvel` 限制。

此前离线 A/B 的“8×”是对 **candidate-local clip 和 final qref cap 同时**乘以 8。
若仅从数学上要求 JSON profile 不再截断这一 8×目标，则 `profiles.rm65` 的
`max_velocity_deg_s` 至少需为：

```json
[321, 321, 413, 688, 688, 917]
```

该数组来自 `8 * [0.7, 0.7, 0.9, 1.5, 1.5, 2.0] rad/s` 换算为 deg/s 后向上取整。
在后续 30 Hz IK/reference 下，这相当于每个 outer qref 最多约
`[10.70, 10.70, 13.75, 22.92, 22.92, 30.56] deg`；数值本身已说明它只能作为离线
因果上界，而不是可直接采用的真机步长。
这**不是机械臂额定速度，也不是可直接上真机的推荐配置**；它只是保证 profile 不会阻挡
离线 8× 因果试验。直接把共享 `rm65` profile 改为该值会影响所有引用它的路径，应先改为
显式选择一个独立的 position-experimental profile，再做逐级真机验证。

同时，若希望 120 Hz accepted qcmd 也能获得同等速度能力，必须调整 planner；只放宽
candidate/final qref cap 会使 120 Hz planner 继续成为瓶颈。planner 的限制来源如下：

| 限制 | 参数/来源 | 8×时的对应值 | 是否需要调整 |
| --- | --- | --- | --- |
| 最大速度与每个 outer-step 的 command-step 上限 | `--arm-max-angvel`；backend 以它建立 planner 的 `max_velocity` 和 `max_command_step` | `5.6,5.6,7.2,12,12,16 rad/s` | 若要测试完整 8× qcmd，必须同步调整；仅改 JSON 不影响 planner。 |
| 最大加速度 | `--position-max-acceleration-rad-s2`，当前 `7,7,9,15,15,20 rad/s^2` | 保持 0.10 s ramp 的等比例值为 `56,56,72,120,120,160 rad/s^2` | 必须同步调整，否则到达 8×速度约需 0.8 s，仍会明显滞后。 |
| 目标速度反馈 horizon | `--position-velocity-ramp-sec`，当前 `0.10 s` | 保持 `0.10 s` | 无需修改；在速度与加速度均按 8×缩放时，归一化 ramp 行为不变。 |
| final 120 Hz EMA | `--arm-follow-qpos-alpha`，当前 `0.9` | `1.0` 为无额外 EMA | 不是硬速度上限，但高速度 A/B 中会增加相位滞后；先保留 0.9 记录结果，再单独 A/B 1.0。 |
| endpoint safety envelope | acceleration planner 内部根据当前速度、加速度和目标端点计算 | 无单独“放宽”参数 | 不应关闭；提高前两项后它仍可能标记为 limited，但这是保证可刹停的必要约束。 |

`--position-command-planner acceleration_limited` 不应改成关闭或改为绕过 safety envelope。
最新 debug 中 `command_step_limited=0`；需要重点处理的是 planner 的最大速度和加速度，
而不是另找一个 120 Hz 的固定 step cap。

若只为离线/无机械臂 dry-run 对齐 8×参数，参数组为：

```bash
--arm-max-angvel 5.6,5.6,7.2,12,12,16.0 \
--position-max-acceleration-rad-s2 56,56,72,120,120,160 \
--position-velocity-ramp-sec 0.10
```

真机不要直接使用该组。推荐先完成 profile 名称/路径的显式 CLI 支持，再按较小的、逐关节
不超过已确认安全速度的值进行 A/B；每次检查 `position_command_*.jsonl` 中的
`trajectory_max_velocity_rad_s`、`trajectory_max_acceleration_rad_s2`、
`trajectory_endpoint_velocity_limited`、`trajectory_acceleration_limited` 及最终
`accepted_qcmd` FK 误差。

### 2F. 显式请求的全流程 8× 上界测试：1.0 / 1.0 IK，15→60→120 Hz

这是一条**受控真机上界测试**，不是日常位控配置。它固定
`teleop_runtime/configs/rm65_ik_tuning.json` 的 `orientation_priority`，即位置/旋转权重
均为 `1.0`；保持 1 次 IK solve，以便本次只检验限速放宽的端到端效果。

“全流程 8×”在本测试中严格指：

- candidate-local step clip：`--position-ik-local-step-clip-scale 8`；
- final 60 Hz qref step cap：`--arm-max-angvel` 8×，且 joint-velocity profile 同步不再更严格；
- 120 Hz planner 最大速度：`--arm-max-angvel` 8×；
- 120 Hz planner 最大加速度：`--position-max-acceleration-rad-s2` 8×。

执行命令前，必须人工临时将
`robot_motion/configs/joint_velocity_limits.json` 的 `profiles.rm65.max_velocity_deg_s`
设为下面的值；否则 profile 会继续截断 final qref cap，运行结果**不是**全流程 8×：

```json
[321, 321, 413, 688, 688, 917]
```

这是共享 profile，只能在本次专门试验窗口使用，并在结束后恢复。命令本身不改该 JSON，
也不改变力控/阻抗链路。保留 collision、joint-limit、inactive 规则及 planner 的 braking
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
  --collection-task-name dmtac_position_1x1_fullpath_8x_15_60_120hz \
  --collection-task-description "controlled position-only upper-bound: IK 1.0/1.0, qref and planner V/A 8x" \
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
`position_ik_local_step_clip_scale=8` 和固定 1 次 iteration；action debug 的首个
`event=config` 中 `arm_max_angvel_rad_s=[5.6,5.6,7.2,12,12,16]`；以及
`position_command_150.jsonl` 的 `trajectory_max_velocity_rad_s` 与
`trajectory_max_acceleration_rad_s2=[56,56,72,120,120,160]`。`orientation_priority=1.0/1.0`
由命令显式固定的 tuning JSON 确认。同时在 `position_reference.jsonl` 确认
`joint_velocity_limit_max_step_deg` 不再被旧 profile 截断，并以
`position_command_150.jsonl` 的 accepted qcmd FK 误差作为最终评价，不以 qref 误差替代。

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

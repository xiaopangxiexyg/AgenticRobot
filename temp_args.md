#!/usr/bin/env bash
# 仅加载 DMTAC 真机遥操作参数，不会执行任何命令。
#
# 在当前终端加载：
#   source /home/zc/HandMotion/Developing/DexTeleop/temp_args.md
#
# 加载后再自行选择最终命令，例如：
#   taskset -c 0-11 python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
#     "${COMMON_REAL_TELEOP_ARGS[@]}" \
#     "${DMTAC_CAPTURE_COMMON_ARGS[@]}" \
#     "${DMTAC_A_ARGS[@]}" \
#     "${DATASET_COMMON_ARGS[@]}" \
#     "${IMPEDANCE_40HZ_ARGS[@]}" \
#     "${CONTROL_DEBUG_OFF_ARGS[@]}"

export LEFT_ARM_IP="${LEFT_ARM_IP:-192.168.101.21}"
export RIGHT_ARM_IP="${RIGHT_ARM_IP:-192.168.1.20}"

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

DMTAC_A_ARGS=(
  --dmtac-tactile-global-read-lock
)

DMTAC_B_ARGS=(
  --dmtac-tactile-no-global-read-lock
)

# 使用全串行方案时，以此数组替换 DMTAC_CAPTURE_COMMON_ARGS；避免 parallel/sequential 同传。
DMTAC_CAPTURE_SEQUENTIAL_ARGS=(
  --dmtac-with-tactile
  --dmtac-tactile-fields raw
  --dmtac-tactile-compute-profile requested
  --dmtac-tactile-read-hz 15
  --dmtac-tactile-process-read
  --dmtac-tactile-worker-cpus 12,13,14,15
  --dmtac-tactile-worker-nice 15
  --dmtac-tactile-worker-queue-size 4
  --dmtac-tactile-sequential-read
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

IMPEDANCE_40HZ_ARGS=(
  --arm-execution-profile impedance_40hz
  --arm-control-mode impedance
  --hz 15
  --dual-arm-ik-hz 15
  --arm-control-hz 40
  --no-arm-canfd-follow
)

POSITION_CONTROL_ARGS=(
  --arm-execution-profile position
  --arm-control-mode position
  --hz 15
  --position-reference-hz 60
  --dual-arm-ik-hz 60
  --position-reference-max-age-sec 0.2
  --arm-control-hz 120
  --arm-canfd-follow
  --position-post-ik-qpos-smoothing-alpha 0.18
  --position-post-ik-qpos-smoothing-fast-alpha 0.7
  --position-post-ik-qpos-smoothing-adaptive
  --arm-follow-qpos-alpha 0.9
  --arm-canfd-trajectory-mode 0
  --arm-canfd-radio 0
  --position-command-planner linear
)

# 位控平滑实验叠加项：放在 POSITION_CONTROL_ARGS 之后展开。
# qd_max 与 qdd_max 均显式暴露在这里，便于复测时直接调节；二者互不派生。
# 可恢复异常目标速度降至50%，但qdd_max保持下面的原始上限，不随速度比例缩小。
# manual hold、碰撞/无有效IK解、stale target走固定当前qpos的软件快速保持；
# 已被后端拒绝并替换为当前qpos的奇异区入口按异常目标降至50%，避免额外瞬时速度清零。
# 有限且已通过碰撞/关节安全检查的IK pose残差不触发异常降速；规划器保留正常速度，
# 并通过position error反馈继续收敛到静止qref，避免“目标停止后停在远处”。
POSITION_ACCELERATION_LIMITED_ARGS=(
  --position-command-planner acceleration_limited
  --arm-max-angvel 0.7,0.7,0.9,1.5,1.5,2.0
  --position-max-acceleration-rad-s2 7,7,9,15,15,20
  --position-velocity-ramp-sec 0.10
  --position-degraded-speed-scale 0.50
)

# 全流程 8× 位置跟踪上界测试（仅 position mode）。
# 固定 IK 为 rm65_ik_tuning.json 的 orientation_priority，即位置/旋转=1.0/1.0；
# 维持当前的一次 IK solve，避免与“放宽限速”这个变量混在一起。
#
# 重要前置条件：先人工确认 robot_motion/configs/joint_velocity_limits.json 的
# profiles.rm65.max_velocity_deg_s 已临时设为
# [321, 321, 413, 688, 688, 917]。该 profile 仍是 qref final step cap 的一部分；
# 未满足此前置条件时，下面的 --arm-max-angvel 无法使 qref 真正达到 8×。
# 这是共享 profile，只能在专门的受控真机试验窗口临时修改，试验后恢复。
#
# 四个放宽点：candidate-local clip、final qref step、120Hz planner V、120Hz planner A。
# 另在120Hz planner增加独立0.675°单次proposal envelope；它不降低持续V上限。
# 该envelope位于0.9 final EMA之前，因此正常跟踪时实际qcmd单步最大约0.6075°。
# 保留 0.10s horizon、0.9 final EMA、碰撞/关节/endpoint braking safety envelope；
# 不传 force/impedance 参数，position-only 参数在 force mode 会被入口拒绝。
POSITION_ACCELERATION_LIMITED_8X_1X1_ARGS=(
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
)

# 历史15→45→150数据或显式对照实验使用；不再是默认位控分支。
POSITION_45_150HZ_ARGS=(
  --arm-execution-profile position_45_150hz
  --arm-control-mode position
  --hz 15
  --position-reference-hz 45
  --dual-arm-ik-hz 45
  --position-reference-max-age-sec 0.2
  --arm-control-hz 150
  --arm-canfd-follow
  --position-post-ik-qpos-smoothing-alpha 0.9
  --arm-follow-qpos-alpha 0.1
  --arm-canfd-trajectory-mode 0
  --arm-canfd-radio 0
  --position-command-planner linear
)

CUSTOM_POSITION_RATE_ARGS=(
  --hz 12
  --position-reference-hz 50
  --dual-arm-ik-hz 50
  --arm-control-hz 160
)

CONTROL_DEBUG_OFF_ARGS=(
  --control-debug off
)

CONTROL_DEBUG_SUMMARY_ARGS=(
  --control-debug summary
  --control-debug-root teleop_records/control_debug
  --control-debug-queue-size 8192
  --control-debug-flush-every 30
)

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
  --debug-timing-threshold-sec 0.0133
  --debug-timing-report-interval-sec 1.0
)

LEGACY_SESSION_DEBUG_ARGS=(
  --record
  --session-record-root teleop_records/debug_sessions
  --no-session-record-commands
  --session-record-compact-json
  --session-record-flush-every 30
)

COLLECTION_TIMING_DEBUG_ARGS=(
  --debug-timing
  --debug-timing-threshold-sec 0.066
  --debug-timing-report-interval-sec 1.0
  --collection-warning-hz 15
)

# 高级控制实验 overlays：仅在明确需要时手动追加。
FIX_ARM_ARGS=(--fix-arm)
DISABLE_FORCE_ADMITTANCE_ARGS=(--arm-disable-force-admittance)
LEGACY_CANFD_120HZ_ARGS=(--arm-canfd-follow --arm-canfd-follow-stream-hz 120)
CANFD_CURVE_FITTING_ARGS=(--arm-canfd-trajectory-mode 1 --arm-canfd-radio 100)
CANFD_CONTROLLER_FILTER_ARGS=(--arm-canfd-trajectory-mode 2 --arm-canfd-radio 100)
POST_IK_QPOS_EMA_08_ARGS=(--position-post-ik-qpos-smoothing-alpha 0.8)
FINAL_QPOS_EMA_08_ARGS=(--arm-follow-qpos-alpha 0.8)
SAFE_DEBUG_ARGS=(--safe-debug)
SAFE_DEBUG_DISABLE_HOLD_ARGS=(--safe-debug --disable-hold)
STRICT_DEGREE_LIMIT_ARGS=(--strict-degree-limit)
NO_SINGULARITY_PROTECTION_ARGS=(--no-arm-singularity-protection)

# DMTAC raw-only 与 distributed-force 路径的独立参数组。
DMTAC_RAW_ONLY_ARGS=(
  --dmtac-with-tactile
  --dmtac-tactile-fields raw
  --dmtac-tactile-compute-profile requested
  --dmtac-tactile-no-flat-vector
  --external-tactile
)

DMTAC_DISTRIBUTED_FORCE_ARGS=(
  --dmtac-with-tactile
  --dmtac-tactile-fields distributed_force
  --dmtac-tactile-compute-profile dist-only
)

DMTAC_FLAT_VECTOR_ARGS=(--dmtac-tactile-flat-vector)

echo "DMTAC 参数已加载；尚未执行任何遥操作命令。"

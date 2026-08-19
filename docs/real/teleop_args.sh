#!/usr/bin/env bash
# 从仓库根目录加载 DMTAC 真机遥操作参数，不会连接设备或执行遥操命令。
#
# 在当前终端加载：
#   source docs/real/teleop_args.sh
#
# 加载后再自行选择最终命令，例如：
#   taskset -c 0-11 python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
#     "${COMMON_REAL_TELEOP_ARGS[@]}" \
#     "${DMTAC_CAPTURE_COMMON_ARGS[@]}" \
#     "${DMTAC_A_ARGS[@]}" \
#     "${DATASET_COMMON_ARGS[@]}" \
#     "${IMPEDANCE_40HZ_ARGS[@]}" \
#     "${CONTROL_DEBUG_OFF_ARGS[@]}"
# 本文件会为 LEFT_ARM_IP、RIGHT_ARM_IP、DATASET_BASE 提供默认值；如果要做
# 参数校验，必须放在 source 本文件之后。bash 和 zsh 都支持下面 README 中的
# `: "${NAME:=default}"` 写法，不要在 source 之前使用 `${NAME:?message}`。

export LEFT_ARM_IP="${LEFT_ARM_IP:-192.168.101.21}"
export RIGHT_ARM_IP="${RIGHT_ARM_IP:-192.168.1.20}"
# 新正式采集统一写入 canonical 数据盘；仍可在 source 前显式覆盖，用于旧数据兼容、
# 离线测试或受控回退。已有数据和历史路径不会被迁移或改写。
export DATASET_BASE="${DATASET_BASE:-/home/user/zc/datasets/RealmanDMTAC/FormalCollection}"
# 触觉读取子进程CPU亲和性可在source本文件之前覆盖；默认值保持原4核行为。
# 例如只使用2核：export DMTAC_TACTILE_WORKER_CPUS=12,13
export DMTAC_TACTILE_WORKER_CPUS="${DMTAC_TACTILE_WORKER_CPUS:-12,13,14,15}"
# 已通过双臂验收的 60 Hz profile-compliance 响应参数。可在 source 本文件前覆盖；
# weight 是相对验收基线的顺应倍率（1.0 保持验收行为），不是 IK objective weight。
# 当前正式链路没有独立的 M/B/K 物理阻尼项；两个 tau 才是实际生效的受力/扭矩
# 输入阻尼，值越大越平滑但响应越慢。最终平移/旋转及 slew 安全包络不会随它们变化。
export POSITION_ADMITTANCE_TRANSLATION_RESPONSE_WEIGHT="${POSITION_ADMITTANCE_TRANSLATION_RESPONSE_WEIGHT:-1.0}"
export POSITION_ADMITTANCE_ROTATION_RESPONSE_WEIGHT="${POSITION_ADMITTANCE_ROTATION_RESPONSE_WEIGHT:-1.0}"
export POSITION_ADMITTANCE_TRANSLATION_FILTER_TAU_SEC="${POSITION_ADMITTANCE_TRANSLATION_FILTER_TAU_SEC:-0.010857362}"
export POSITION_ADMITTANCE_ROTATION_FILTER_TAU_SEC="${POSITION_ADMITTANCE_ROTATION_FILTER_TAU_SEC:-0.010857362}"
# Default-off 60 Hz output follower and velocity-continuity limits for the
# low-frequency force-modifier candidate.  The accepted/parity arrays below do
# not consume these values; only the explicit DAMPED successor does.  Setting
# either acceleration value to zero disables that limiter exactly.
export POSITION_ADMITTANCE_TRANSLATION_OUTPUT_FOLLOW_TAU_SEC="${POSITION_ADMITTANCE_TRANSLATION_OUTPUT_FOLLOW_TAU_SEC:-0.025}"
export POSITION_ADMITTANCE_ROTATION_OUTPUT_FOLLOW_TAU_SEC="${POSITION_ADMITTANCE_ROTATION_OUTPUT_FOLLOW_TAU_SEC:-0.025}"
export POSITION_ADMITTANCE_TRANSLATION_MAX_OUTPUT_ACCELERATION_M_S2="${POSITION_ADMITTANCE_TRANSLATION_MAX_OUTPUT_ACCELERATION_M_S2:-1.2}"
export POSITION_ADMITTANCE_ROTATION_MAX_OUTPUT_ANGULAR_ACCELERATION_RAD_S2="${POSITION_ADMITTANCE_ROTATION_MAX_OUTPUT_ANGULAR_ACCELERATION_RAD_S2:-3.0}"

COMMON_REAL_TELEOP_ARGS=(
  --active-sides both
  --right-arm-ip "$RIGHT_ARM_IP"
  --left-arm-ip "$LEFT_ARM_IP"
  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json
  --left-tool-profile hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json
  --gui
  --hand-backend dmtac-gripper
  --gripper-task-profile teleop_runtime/configs/gripper_task_profile.example.json
  --q6-hold-recalibration-gesture
  --no-episode-recording
  --arm-io-threads 4
)

# 双臂最小设备遥操：保留当前 DMTAC_M 工具/TCP、碰撞场景及最新控制参数，
# 只是不连接相机、Q6、DMTAC 夹爪/触觉，也不启动正式数据采集。
# 与最新位控参数组组合使用；不要再展开 DATASET_COMMON_ARGS、DMTAC_CAPTURE_*
# 或 DMTAC_A/B_ARGS。
NO_GRIPPER_COMMON_ARGS=(
  --active-sides both
  --right-arm-ip "$RIGHT_ARM_IP"
  --left-arm-ip "$LEFT_ARM_IP"
  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json
  --left-tool-profile hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json
  --gui
  --hand-backend none
  --no-q6-default-config
  --no-dmtac-default-config
  --no-camera
  --no-collection
  --no-episode-recording
  --arm-io-threads 4
)

RIGHT_ONLY_COMMON_ARGS=(
  --active-sides right
  --right-arm-ip "$RIGHT_ARM_IP"
  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json
  --gui
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
  --dmtac-tactile-worker-cpus "$DMTAC_TACTILE_WORKER_CPUS"
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

# CPU亲和性profile放在DMTAC_CAPTURE_*和DMTAC_A/B_ARGS之后展开，利用argparse
# 的最后值覆盖规则。2核profile释放逻辑CPU14-15给其他进程/系统任务。
DMTAC_TACTILE_WORKER_2CPU_ARGS=(
  --dmtac-tactile-worker-cpus 12,13
)

DMTAC_TACTILE_WORKER_4CPU_ARGS=(
  --dmtac-tactile-worker-cpus 12,13,14,15
)

# 使用全串行方案时，以此数组替换 DMTAC_CAPTURE_COMMON_ARGS；避免 parallel/sequential 同传。
DMTAC_CAPTURE_SEQUENTIAL_ARGS=(
  --dmtac-with-tactile
  --dmtac-tactile-fields raw
  --dmtac-tactile-compute-profile requested
  --dmtac-tactile-read-hz 15
  --dmtac-tactile-process-read
  --dmtac-tactile-worker-cpus "$DMTAC_TACTILE_WORKER_CPUS"
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
  --storage-queue-backpressure-policy block
  --wrist-depth-mask
)

# 无正式 collection 的控制/传感器卡顿诊断不得展开 DATASET_COMMON_ARGS：
# 该数组显式保留“没有 dataset writer、没有腕部 depth stream”的基线。
# 相机 RGB 是否连接仍由最终命令中的 --camera-config/--no-camera 决定。
NO_COLLECTION_SENSOR_BASELINE_ARGS=(
  --no-collection
  --no-episode-recording
)

IMPEDANCE_40HZ_ARGS=(
  --arm-execution-profile impedance_40hz
  --arm-control-mode impedance
  --hz 15
  --dual-arm-ik-hz 15
  --arm-control-hz 40
  --no-arm-canfd-follow
)

# Conservative rollback for the position-mode force guard. Append this array
# after POSITION_FORMAL_V1_ARGS or any V2/V3 position array so argparse's
# last-value-wins rule restores the previous complete 15 N / 2 Nm guard.
POSITION_FORCE_GUARD_CONSERVATIVE_ARGS=(
  --position-force-guard
  --position-force-guard-source tool
  --position-force-guard-max-force-n 15
  --position-force-guard-max-torque-nm 2
  --position-force-guard-hz 20
  --position-force-guard-consecutive-samples 1
  --position-force-guard-max-read-errors 3
  --position-force-guard-stale-sec 0.4
)

# 正式版双臂位控 profile。该数组单独使用，不要再叠加下面历史调试用的
# POSITION_CONTROL_ARGS / POSITION_ACCELERATION_LIMITED_*，否则会显式覆盖正式值。
# live teleop 会在主 action 生成前应用 inactive 与 0.14->0.45 自适应 qref 整形；
# replay/deploy 读取已经整形的主 action，不会重复这些阶段。
# shoulder/coaxial纠正的独立回退开关是
# --no-position-prefer-non-shoulder-singular-branch（默认正式profile开启）。
# 候选预算与证书的独立回退分别是
# --position-ik-candidate-certification-reserve-ms 0 和
# --no-position-singularity-candidate-certificate；默认正式profile使用3.5 ms候选认证
# reserve，并校验最终qref指纹。
# 正式位控在证书有效时将 SDK 的 J3/J5 奇异边界作为 advisory，优先采用
# 本地候选分类；力控、回放和无证书直发路径仍保留 SDK 进入式保护。
POSITION_FORMAL_V1_ARGS=(
  --position-control-profile rm65_bimanual_position_v1
  --arm-control-mode position
  # Stable position baseline: one physical IK step per 60 Hz reference tick,
  # q5 escape hysteresis, and the guarded wrist-branch candidate set.  The
  # three options remain individually reversible on the command line.
  --position-ik-step-mode single_step_singularity_aware
  # Keep the 2026-08-02 q3/q5 anchor and forced limit-escape experiments out
  # of the production baseline.  These explicit rollback flags make the
  # command stable even if parser defaults are changed during later research.
  --no-position-ik-branch-continuity-anchor
  --position-joint-limit-guard-mode legacy
  --position-wrist-branch-switch-guard-deg 12.0
  --position-wrist-q5-hysteresis-release-deg 6.0
  --position-force-guard
  --position-force-guard-source tool
  --position-force-guard-max-force-n 30
  --position-force-guard-max-torque-nm 5
  --position-force-guard-hz 20
  --position-force-guard-consecutive-samples 1
  --position-force-guard-max-read-errors 3
  --position-force-guard-stale-sec 0.4
)

# Opt-in responsive successor.  V1 above remains the exact rollback profile.
# V2 preserves the same IK candidate policy, force guard, qref velocity limits,
# 15/60/60/120 clocks, and high-rate execution planner.  It only bypasses the
# broad post-IK EMA for a per-side healthy normal tracking step and requires a
# real admitted q5 escape before release-band hysteresis may hold that wrist.
POSITION_FORMAL_V2_RESPONSIVE_ARGS=(
  --position-control-profile rm65_bimanual_position_v2_responsive
  --arm-control-mode position
  --position-ik-step-mode single_step_singularity_aware
  --no-position-ik-branch-continuity-anchor
  --position-joint-limit-guard-mode legacy
  --position-wrist-branch-switch-guard-deg 12.0
  --position-wrist-q5-hysteresis-release-deg 6.0
  --position-force-guard
  --position-force-guard-source tool
  --position-force-guard-max-force-n 30
  --position-force-guard-max-torque-nm 5
  --position-force-guard-consecutive-samples 1
  --position-force-guard-max-read-errors 3
  --position-force-guard-stale-sec 0.4
)

# Frozen stable smoothness baseline for responsive V2.  The last profile
# argument is intentional: it preserves the complete V2 command/safety surface
# while only selecting the balanced source-shaping profile.  Switching back to
# POSITION_FORMAL_V2_RESPONSIVE_ARGS is the atomic rollback.
POSITION_FORMAL_V2_BALANCED_ARGS=(
  "${POSITION_FORMAL_V2_RESPONSIVE_ARGS[@]}"
  --position-control-profile rm65_bimanual_position_v2_balanced
)

# Formal pure-position release promoted from the 2026-08-11 dual-arm real run.
# This selects the frozen V2 Balanced source/IK contract plus the strictly-newer
# 60 Hz qref -> 120 Hz jerk horizon owner.  It enables no admittance modifier;
# POSITION_FORMAL_V2_BALANCED_ARGS remains the complete execution rollback.
# Frozen profile SHA256:
# dc97dc0e07a6ffc486ed50042899bcfef448a844f48ff883201ee1542f900380
POSITION_FORMAL_V2_BALANCED_60HZ_JERK_V2_ARGS=(
  "${POSITION_FORMAL_V2_BALANCED_ARGS[@]}"
  --position-control-profile rm65_bimanual_position_v2_balanced_60hz_jerk_v2
)

# 正式双臂 60 Hz 平移+旋转阻抗 overlay。该组保持 V2 Balanced 的 60 Hz 解 IK、
# 碰撞/关节/动力学 admission、30 N / 5 Nm 力急停与数采语义，只在 pre-IK
# copied target 上加入已经验收的 bounded modifier。该基线使用 response weight 与
# input filter tau 四个环境变量；显式 DAMPED successor 再使用两个 output tau；
# Active 会拒绝超出响应权重 [0.25,4] 或 filter tau > 0.1 s 的配置。
POSITION_FORMAL_V2_BALANCED_60HZ_ADMITTANCE_ARGS=(
  "${POSITION_FORMAL_V2_BALANCED_ARGS[@]}"
  # Keep fix-arm available for the five-pedal side-switch contract.  The
  # active admittance installer already gates each fixed side through its
  # side_activity callback, so disabling the controller globally is neither
  # required for safety nor compatible with formal pedal collection.
  --fix-arm
  --no-position-admittance-shadow
  --position-admittance-active
  --position-admittance-active-side both
  --position-admittance-active-envelope normal_parity
  --position-admittance-active-endpoint-policy formal_teleop
  --position-admittance-rotation-active
  --position-admittance-combined-active
  --position-admittance-rotation-active-profile rotation-emphasis-5deg
  --position-admittance-response-model profile_compliance
  --position-admittance-translation-response-weight "$POSITION_ADMITTANCE_TRANSLATION_RESPONSE_WEIGHT"
  --position-admittance-rotation-response-weight "$POSITION_ADMITTANCE_ROTATION_RESPONSE_WEIGHT"
  --position-admittance-shadow-axes xyz
  --position-admittance-shadow-stiffness-n-m 250
  --position-admittance-shadow-deadband-n 0.45
  --position-admittance-translation-filter-tau-sec "$POSITION_ADMITTANCE_TRANSLATION_FILTER_TAU_SEC"
  --position-admittance-rotation-filter-tau-sec "$POSITION_ADMITTANCE_ROTATION_FILTER_TAU_SEC"
  --position-admittance-shadow-max-offset-m 0.08
  --position-admittance-shadow-max-velocity-m-s 0.06
  --position-admittance-shadow-max-return-velocity-m-s 0.42
  --position-admittance-shadow-max-slew-m-s 0.06
  --position-admittance-shadow-return-deadband-n 0.45
  --position-admittance-shadow-return-rate-s 4
  --position-admittance-response-reference-hz 40
  --position-admittance-shadow-stale-sec 0.15
  --position-admittance-shadow-max-force-dt-sec 0.15
)

# Default-off qref-to-qcmd execution successor for the formal 60 Hz combined
# admittance path.  The final repeated profile option intentionally replaces
# only V2 Balanced's execution profile while retaining the complete admitted
# target, IK, collision, force-guard and admittance argument surface above.
# POSITION_FORMAL_V2_BALANCED_60HZ_ADMITTANCE_ARGS is the atomic rollback.
POSITION_FORMAL_V2_BALANCED_60HZ_JERK_ADMITTANCE_ARGS=(
  "${POSITION_FORMAL_V2_BALANCED_60HZ_ADMITTANCE_ARGS[@]}"
  --position-control-profile rm65_bimanual_position_v2_balanced_60hz_jerk_v1
)

# Default-off clock-ownership successor for the 60 Hz combined path.  It keeps
# JERK V1's IK, collision, qref/qcmd dynamics and safety arguments unchanged;
# only a strictly newer source timestamp may replace the cached 120 Hz horizon.
# POSITION_FORMAL_V2_BALANCED_60HZ_JERK_ADMITTANCE_ARGS is the atomic rollback.
POSITION_FORMAL_V2_BALANCED_60HZ_JERK_V2_ADMITTANCE_ARGS=(
  "${POSITION_FORMAL_V2_BALANCED_60HZ_ADMITTANCE_ARGS[@]}"
  --position-control-profile rm65_bimanual_position_v2_balanced_60hz_jerk_v2
)

# Reversible successor for the 20 Hz wrench -> 60 Hz modifier sample-and-hold
# observed in the 2026-08-11 free-motion traces.  It filters only the already
# bounded translation/rotation modifier output and limits its 60 Hz velocity
# change.  Input force/torque response, the released pure-position profile,
# shared 60 Hz IK and every final hard offset/slew/endpoint/force-guard boundary
# remain unchanged.  The JERK V2 admittance array above is the exact rollback.
POSITION_FORMAL_V2_BALANCED_60HZ_JERK_V2_DAMPED_ADMITTANCE_ARGS=(
  "${POSITION_FORMAL_V2_BALANCED_60HZ_JERK_V2_ADMITTANCE_ARGS[@]}"
  --position-admittance-translation-output-follow-tau-sec "$POSITION_ADMITTANCE_TRANSLATION_OUTPUT_FOLLOW_TAU_SEC"
  --position-admittance-rotation-output-follow-tau-sec "$POSITION_ADMITTANCE_ROTATION_OUTPUT_FOLLOW_TAU_SEC"
  --position-admittance-translation-max-output-acceleration-m-s2 "$POSITION_ADMITTANCE_TRANSLATION_MAX_OUTPUT_ACCELERATION_M_S2"
  --position-admittance-rotation-max-output-angular-acceleration-rad-s2 "$POSITION_ADMITTANCE_ROTATION_MAX_OUTPUT_ANGULAR_ACCELERATION_RAD_S2"
)

# Default-off successor for residual ~30 Hz micro-vibration.  It changes only
# the ordinary-region alpha from balanced 0.55 to 0.45.  BALANCED above is the
# complete rollback and must remain a separate task/profile dataset.
POSITION_FORMAL_V2_SMOOTH_ARGS=(
  "${POSITION_FORMAL_V2_BALANCED_ARGS[@]}"
  --position-control-profile rm65_bimanual_position_v2_smooth
)

# Opt-in 15 Hz action-aligned / 120 Hz jerk-limited execution experiment.
# It keeps V1 source shaping, 60 Hz IK/collision evaluation, force guard, and
# all operator semantics.  Only one qref endpoint per 15 Hz source bucket is
# admitted; the endpoint is projected onto the 120 Hz q/qd/qdd/jerk reachable
# set and is also the accepted-qref FK action recorded by the dataset.  The
# three intermediate 60 Hz IK frames are still evaluated upstream for input
# freshness/safety but do not preempt the cached 120 Hz control cut.
# POSITION_FORMAL_V1_ARGS is the
# complete rollback profile.
POSITION_FORMAL_SAMPLE_ALIGNED_JERK_V1_ARGS=(
  "${POSITION_FORMAL_V1_ARGS[@]}"
  --position-control-profile rm65_bimanual_position_sample_aligned_jerk_v1
)

# Reversible successor to the first sample-aligned experiment.  Compared with
# V1 it keeps the full admitted target under degraded execution, restores the
# guarded responsive source path, continues converging after the tracker stops,
# and regularizes the 120 Hz jerk horizon.  The V1 array above is the complete
# rollback and remains unchanged.
POSITION_FORMAL_SAMPLE_ALIGNED_JERK_V2_ARGS=(
  "${POSITION_FORMAL_V1_ARGS[@]}"
  --position-control-profile rm65_bimanual_position_sample_aligned_jerk_v2
)

# Default-off V3 execution experiment.  It preserves V2's 15 Hz action and the
# complete terminal q/qd/qdd state, and only smooths the eight executed 120 Hz
# jerk samples inside that terminal-state null space.  V2 above is the complete
# rollback.
POSITION_FORMAL_SAMPLE_ALIGNED_JERK_V3_ARGS=(
  "${POSITION_FORMAL_V1_ARGS[@]}"
  --position-control-profile rm65_bimanual_position_sample_aligned_jerk_v3
)

# Opt-in Pink QP motion-envelope probe. This keeps the stable V1 source,
# singularity-candidate, collision, force-guard, and downstream 120 Hz command
# semantics, but moves the existing formal qref velocity/acceleration interval
# into the 60 Hz IK QP as well. POSITION_FORMAL_V1_ARGS remains unchanged and
# is the complete rollback; do not combine this array with the V2 collection
# profile until each profile has been validated independently.
POSITION_FORMAL_V3_QP_CONSTRAINED_ARGS=(
  "${POSITION_FORMAL_V1_ARGS[@]}"
  --position-ik-admitted-motion-limit
)

# 奇异感知单周期 IK 参数数组（正式 profile 已包含同值）。为保留旧链路回退，
# 非正式 position profile 可显式展开本数组；新模式每个 60 Hz 外层
# 周期只推进一次 Pink 物理积分，并使用已正式放行 qref 推导的 qdot 作为
# 下一周期加速度约束状态；q5 共轴带内保持 q4+q6 不变并抑制 q4-q6
# 等价角重分配。后置 qpos 平滑也沿上一侧把 q5 推向 3.25° 逃逸边界；
# q4-q6 只有超过 5° 的 null-direction step 才会获得完整抑制，
# 小幅操作者输入保持跟手，避免正式 qref 被重新平滑回 SDK 奇异带。
# 另外在 q5 +/-12° guard band 内不让等价 +/-pi 腕部分支参与候选竞争，
# 并拒绝局部步长裁剪后跨入该 guard 另一侧的辅助候选，防止两个等价解
# 来回翻转。q5 已逃离 3.25° 边界后，默认在 +/-6° release band 内保持
# 上一侧的向内候选，防止 singularity_escape 候选在边界两侧开关；设为 0
# 可单独回退该回滞，guard 设为 0 可单独回退分支保护。
# 原子回退方式：
#   --position-ik-step-mode legacy_iterative
# 或者不要展开本数组。数据集 action 仍为 accepted qref FK，不变。
POSITION_SINGULARITY_AWARE_SINGLE_STEP_ARGS=(
  --position-ik-step-mode single_step_singularity_aware
  --no-position-ik-branch-continuity-anchor
  --position-joint-limit-guard-mode legacy
  --position-wrist-branch-switch-guard-deg 12.0
  --position-wrist-q5-hysteresis-release-deg 6.0
)

# 历史兼容/调试数组。显式 --arm-execution-profile position 会选择 legacy 回退，
# 保留原有逐项调参方式。
POSITION_CONTROL_ARGS=(
  --arm-execution-profile position
  --arm-control-mode position
  --hz 15
  --position-reference-hz 60
  --dual-arm-ik-hz 60
  --position-reference-max-age-sec 0.2
  --arm-control-hz 120
  --arm-canfd-follow
  --position-post-ik-qpos-smoothing-alpha 0.14
  --position-post-ik-qpos-smoothing-fast-alpha 0.45
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
  --position-target-velocity-filter-tau-sec 0.03
  --arm-max-angvel 0.7,0.7,0.9,1.5,1.5,2.0
  --position-max-acceleration-rad-s2 7,7,9,15,15,20
  --position-velocity-ramp-sec 0.10
  --position-degraded-speed-scale 0.50
)

# 全流程 8× 位置跟踪上界测试（仅 position mode）。
# 固定 IK 为 rm65_ik_tuning.json 的 orientation_priority，即位置/旋转=1.0/1.0；
# 位控使用至少/至多2次 IK solve，并采用 Pink damping=0.03。
#
# qref final cap 使用独立 position-only profile；共享 rm65 profile 保持不变。
# 2x/4x/8x profile 都仍与 --arm-max-angvel 逐关节取最小值。
#
# 四个放宽点：candidate-local clip、final qref step、120Hz planner V、120Hz planner A。
# 另在120Hz planner增加独立0.675°单次proposal envelope；它不降低持续V上限。
# 该envelope位于0.9 final EMA之前，因此正常跟踪时实际qcmd单步最大约0.6075°。
# 保留 0.10s horizon、0.9 final EMA、碰撞/关节/endpoint braking safety envelope；
# 不传 force/impedance 参数；目标速度低通仅由 acceleration-limited 位控 planner 消费，
# force/impedance 链路不会读取或应用该滤波。
POSITION_ACCELERATION_LIMITED_8X_1X1_ARGS=(
  --position-command-planner acceleration_limited
  --position-target-velocity-filter-tau-sec 0.03
  --ik-tuning-json robot_motion/configs/rm65_ik_tuning.json
  --position-ik-solve-min-iterations 2
  --position-ik-solve-max-iterations 2
  --position-ik-damping 0.03
  --position-ik-local-step-clip-scale 8
  --position-joint-velocity-limit-profile rm65_position_8x
  --arm-max-angvel 5.6,5.6,7.2,12,12,16.0
  --position-max-acceleration-rad-s2 56,56,72,120,120,160
  --position-max-command-step-deg 0.675
  --position-velocity-ramp-sec 0.10
  --position-post-ik-qpos-smoothing-alpha 0.14
  --position-post-ik-qpos-smoothing-fast-alpha 0.45
  --position-post-ik-qpos-smoothing-adaptive
  --arm-follow-qpos-alpha 0.9
  --position-degraded-speed-scale 0.50
)

# 最新离线 A/B 推荐的 position-only 组合。共享 IK JSON 不变，因此力控仍使用其原始设置：
# - position/orientation = 1.0/0.5；
# - 每帧固定至少/至多2次局部solve；时间预算只约束未来把max提高到2以上后的附加迭代；
# - INACTIVE进入速度/窗口跨度阈值收紧为默认值的0.5倍；dwell与退出阈值不变。
POSITION_ACCELERATION_LIMITED_8X_1X05_ADAPT2_TIGHT05_ARGS=(
  --position-command-planner acceleration_limited
  --position-target-velocity-filter-tau-sec 0.03
  --ik-tuning-json robot_motion/configs/rm65_ik_tuning.json
  --position-ik-orientation-weight 0.5
  --position-ik-solve-min-iterations 2
  --position-ik-solve-max-iterations 2
  --position-ik-damping 0.03
  --position-ik-adaptive-iterations
  --position-ik-iteration-reserve-ms 5
  --position-inactive-enter-threshold-scale 0.5
  --position-ik-local-step-clip-scale 8
  --position-joint-velocity-limit-profile rm65_position_2x
  --arm-max-angvel 5.6,5.6,7.2,12,12,16.0
  --position-max-acceleration-rad-s2 56,56,72,120,120,160
  --position-max-command-step-deg 0.675
  --position-velocity-ramp-sec 0.10
  --position-post-ik-qpos-smoothing-alpha 0.14
  --position-post-ik-qpos-smoothing-fast-alpha 0.45
  --position-post-ik-qpos-smoothing-adaptive
  --arm-follow-qpos-alpha 0.9
  --position-degraded-speed-scale 0.50
)

# 位控正式数采的进程隔离数据面。远端 Ryzen 9 9950X 是16个物理核心/32个
# 逻辑CPU；CPU 0-15分别对应16个不同物理核心，16-31是其SMT sibling。
# 推荐最终命令用 taskset -c 0-7 约束遥操主进程，数采子进程固定8-11，
# DMTAC读取子进程固定12-15，避免三者共享物理核心；16-31留给系统和其他进程。
# auto 在位控也会解析为process；这里显式写出便于运行记录与回退。
# writer-workers/storage-queue属于旧thread路径，process模式不依赖它们。
POSITION_COLLECTION_PROCESS_ARGS=(
  --collection-worker-mode process
  --collection-process-queue-size 4
  --collection-process-shared-slots 4
  --collection-process-extra-array-mb 8
  --collection-worker-cpus 8,9,10,11
  --collection-worker-nice 5
)

# 在完整现有1.0/0.5位控配置之后展开：保持min=2，允许在帧预算内最多4次。
POSITION_IK_1X05_MAX4_ARGS=(
  --position-ik-orientation-weight 0.5
  --position-ik-solve-min-iterations 2
  --position-ik-solve-max-iterations 4
  --position-ik-damping 0.03
  --position-ik-adaptive-iterations
  --position-ik-iteration-reserve-ms 5
)

# 与上组仅旋转权重不同，用于严格A/B；位置权重仍固定1.0。
POSITION_IK_1X035_MAX4_ARGS=(
  --position-ik-orientation-weight 0.35
  --position-ik-solve-min-iterations 2
  --position-ik-solve-max-iterations 4
  --position-ik-damping 0.03
  --position-ik-adaptive-iterations
  --position-ik-iteration-reserve-ms 5
)

# 只替换 60 Hz qref final execution-step profile 的逐级 A/B；其余参数保持不变。
# 下一次真机从2x开始，确认后再用4x；当前离线数据不支持直接使用8x。
POSITION_QREF_FINAL_CAP_2X_ARGS=(
  --position-joint-velocity-limit-profile rm65_position_2x
)
POSITION_QREF_FINAL_CAP_4X_ARGS=(
  --position-joint-velocity-limit-profile rm65_position_4x
)
POSITION_QREF_FINAL_CAP_8X_ARGS=(
  --position-joint-velocity-limit-profile rm65_position_8x
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

# 正式数采默认档：正式 dataset 仍由隔离进程写入；关闭会竞争GIL的JSON/FK sidecar，
# 只保留已验证低扰动的预分配数值trace：120 Hz qcmd、60 Hz IK shadow、原始VIVE
# pose/OpenVR速度和对应时间戳。trace在命令热路径不排队、不做JSON/FK、不写盘，
# 机械臂命令worker停止后才一次性保存。该档不改变位控、力控或主 action 语义。
POSITION_COLLECTION_PRODUCTION_ARGS=(
  --control-debug off
  --no-debug-action
  --no-debug-timing-auto-log
  --actuator-trace
  --actuator-trace-root teleop_records/actuator_identification
  --actuator-trace-max-duration-sec 600
)

# 正式数采性能A/B。四档必须一次只运行一档，且按1→4逐级增加诊断负载。
# AB3/AB4 是 forensic debug，不是正式数采默认档。
POSITION_COLLECTION_AB1_NO_DEBUG_ARGS=(
  --no-debug-disk-write
)

POSITION_COLLECTION_AB2_SUMMARY_ARGS=(
  --control-debug summary
  --control-debug-root teleop_records/control_debug
  --control-debug-queue-size 8192
  --control-debug-flush-every 30
  --no-control-debug-episode-sidecar
  --no-debug-action
  --no-debug-timing-auto-log
)

POSITION_COLLECTION_AB3_SESSION_FULL_ARGS=(
  --control-debug full
  --control-debug-root teleop_records/control_debug
  --control-debug-queue-size 8192
  --control-debug-flush-every 30
  --no-control-debug-episode-sidecar
  --no-debug-action
  --no-debug-timing-auto-log
)

# 重型 forensic 诊断档：保留完整session qref/qcmd审计；action按正式数据频率15 Hz，
# standalone timing最多5 Hz且只把40 ms以上帧视为slow。注意 control-debug 自身仍会
# 写全量120 Hz qcmd和逐阶段position timing，不能作为正式数采常开配置。
POSITION_COLLECTION_AB4_SAMPLED_FULL_ARGS=(
  --control-debug full
  --control-debug-root teleop_records/control_debug
  --control-debug-queue-size 8192
  --control-debug-flush-every 30
  --no-control-debug-episode-sidecar
  --debug-action
  --debug-action-log-root teleop_records/action_debug
  --debug-action-flush-every 30
  --debug-action-queue-size 8192
  --debug-action-sample-hz 15
  --debug-timing-log
  --debug-timing-log-sample-hz 5
  --debug-timing-console
  --debug-timing-ik
  --debug-timing-threshold-sec 0.040
  --debug-timing-report-interval-sec 1.0
)

# 一次性关闭所有 standalone debug 写盘。它覆盖 CONTROL_DEBUG_*、debug action/
# timing/IK sidecar、legacy session JSONL、fix-arm pose snapshot 和 debug trajectory；
# 不影响正式 LeRobot dataset。可将该数组放在命令参数的最后，便于临时关闭写盘。
NO_DEBUG_DISK_WRITE_ARGS=(
  --no-debug-disk-write
)

# 只通过有界后台控制台队列输出慢帧和主循环stall诊断，不启用JSONL、
# RM65 backend或IK profiling；终端flush不在控制线程执行。
# 与 NO_DEBUG_DISK_WRITE_ARGS 组合时不会写 standalone debug 文件。
CONSOLE_TIMING_ARGS=(
  --debug-timing-console
  --debug-timing-watchdog
  --no-debug-timing-auto-log
  --debug-timing-threshold-sec 0.066
  --debug-timing-report-interval-sec 1.0
  --collection-warning-hz 15
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

# 可回退的空闲期触觉无损压缩 overlay。默认不启用；仅适用于 external tactile
# 且热路径保持 --tactile-compression none 的正式 collection。删除该数组或在命令
# 末尾追加 NO_IDLE_TACTILE_COMPRESSION_ARGS 即回到原始 raw-only 行为。
IDLE_TACTILE_COMPRESSION_ARGS=(--idle-tactile-compression)
NO_IDLE_TACTILE_COMPRESSION_ARGS=(--no-idle-tactile-compression)

echo "遥操作参数已加载（含 DMTAC 与无夹爪版本）；尚未执行任何遥操作命令。"

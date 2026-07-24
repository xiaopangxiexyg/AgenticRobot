# IK 安全配置说明

本文解释 `default_ik_safety_profiles.json` 中各字段的实际含义和调参影响。该文件由 `robot_motion.profiles.load_ik_safety_profiles()` 读取成 `IKSafetyProfile`，主要被 `robot_motion.ik_resolver.IKSafetyResolver` 使用；`dual_arm_teleop` 还会被 `teleop_runtime.retargeters.dual_arm_collision_aware` 的双臂局部步进逻辑读取。

## RM65 相机 registry（过渡配置）

`rm65_camera_registry.json` 是 RM65/Rig 共享的相机映射配置，供 Real2Sim 相机挂载发现、离线投影和运行时相机快照使用。当前 v1 为兼容旧数据，仍保留历史 registry 中的序列号、profile 和外参字段；新配置不要继续扩展这种混合结构。后续迁移会把遥操作工作站的设备分配、机器人挂载 profile 与运行时快照拆成独立契约。

## 配置结构

顶层字段：

| 字段 | 含义 | 影响 |
| --- | --- | --- |
| `profiles` | profile 名到参数对象的映射。当前包含 `init_arm`、`teleop`、`deploy`、`dual_arm_teleop`。 | 运行时通过 profile 名选择一组 IK 安全策略。不同 profile 可在精度、实时性、恢复能力之间取不同权衡。 |

当前 profile 的意图：

| Profile | 主要用途 | 当前特征 |
| --- | --- | --- |
| `init_arm` | 初始化或回到初始姿态。 | 候选更多、允许昂贵 seed 扩展、不启用局部步进，偏向一次性找到可靠解。 |
| `teleop` | 单臂实时遥操作。 | 有 35 ms 预算、启用 local-first、粗跟踪、限位/奇异/姿态恢复，偏向高频连续跟随。 |
| `deploy` | 部署运行。 | 与 `teleop` 类似，但候选数更少、FK 容差更紧、目标容差更宽，偏向稳定和算力可控。 |
| `dual_arm_teleop` | 双臂高频控制。 | 候选更多、允许并行、50 ms 预算、启用双臂组件级进度判断，偏向双臂同时保持进展。 |

## 基础目标与候选搜索

| 字段 | 单位/类型 | 实际含义 | 调参影响 |
| --- | --- | --- | --- |
| `target_tolerance.position_m` | 米 | 目标位姿的位置容忍半径。用于 pose fraction 计算、粗跟踪阈值、局部步进进度评分、双臂误差评分。 | 变小会要求更精确，容易出现 hold、失败或抖动；变大会更容易接受近似解，但末端可能离目标更远。 |
| `target_tolerance.orientation_deg` | 度 | 目标姿态角误差容忍值。用途同上。 | 变小会提高姿态精度要求，但奇异附近更容易卡住；变大会降低姿态约束，跟手性更柔和但方向误差更大。 |
| `relaxation_fractions` | 0 到 1 数组 | 全量 IK 依次尝试的目标放松比例。`0.0` 表示原始目标；大于 0 时会把目标向当前参考位姿拉回，最多拉回 `target_tolerance * fraction`。 | 增加或放大可提高远目标、突变目标的成功率，但结果状态会变成 relaxed，实际末端目标可能不是原始目标。减少可更严格追原始目标，但更容易失败。 |
| `enable_expensive_seed_expansion` | 布尔 | 是否允许 IK provider 做更昂贵的 seed 扩展。 | 开启通常提高逃出局部解、跨分支找解的机会；代价是耗时增加，实时 teleop 中可能降低帧率。 |
| `allow_parallel_candidate_generation` | 布尔 | 是否允许 provider 并行生成候选。 | 开启可能降低墙钟时间，尤其候选多时；但会增加 CPU 占用和调度不确定性。 |
| `max_candidate_count` | 个数 | 每轮向 provider 请求或保留的最大候选数量。local step 中会再限制到较小上限。 | 增大可提高找到好解的概率，但增加 IK、FK、奇异、碰撞检查成本；减小更快但可能漏掉安全候选。 |
| `provider_time_budget_ms` | 毫秒或 `null` | 单帧 provider 预算。`resolve_step` 会据此建立 frame deadline，并传给 full/local provider。 | 变小更实时但候选可能不足；变大提高成功率但可能造成 GUI 或控制循环卡顿。`null` 表示不显式限制。 |
| `allow_smoothing_recovery` | 布尔 | 全量 IK 全部失败时，是否返回 `SMOOTHING_REQUIRED` 而不是直接 failed。 | 开启后上层可进入平滑/保持恢复路径；关闭会更快暴露失败，但控制链更可能中断。 |
| `stable_recovery_frames` | 帧数 | 上层恢复逻辑可用的稳定帧计数参数，当前 profile 只保存并暴露该值。 | 值越大，恢复状态通常越保守；值越小，恢复切换更快但可能抖动。 |

## 关节跳变与分支保护

| 字段 | 单位/类型 | 实际含义 | 调参影响 |
| --- | --- | --- | --- |
| `hard_branch_joint_indices` | 关节索引数组 | 对这些关节施加硬分支跳变检查。索引为 0 基。 | 覆盖的关节越多，越能阻止肘部或肩部突然翻支；也越可能拒绝可行但跨分支的解。 |
| `hard_branch_max_delta_deg` | 度数组 | 与 `hard_branch_joint_indices` 一一对应，候选相对参考位姿的最大允许关节变化。 | 变小会减少突跳，但容易卡在坏分支；变大更容易跨分支恢复，也更可能突然大动作。 |
| `per_step_joint_limit_deg` | 度数组或 `null` | 对所有关节的单步绝对变化硬限制。数组长度必须等于 qpos 长度才生效；`null` 表示关闭。 | 开启且较小可强约束每步动作；过严会拒绝大量候选，尤其初始化或远目标。 |

注意：如果配置了 `metadata.joint_posture_preference.hard_branch_escape_*`，某些硬分支跳变可在“参考姿态很差且候选显著改善姿态”时被允许。

## FK 误差、奇异和评分惩罚

| 字段 | 单位/类型 | 实际含义 | 调参影响 |
| --- | --- | --- | --- |
| `fk_error_tolerance_m` | 米 | 候选 FK 到有效目标的位置误差阈值。实际判定使用 `max(fk_error_tolerance_m, target_tolerance.position_m)`。 | 变小不一定比 `target_tolerance` 更严；只有大于目标容差时才放宽 FK 接受。变大可接受更粗的 IK 解。 |
| `fk_error_tolerance_deg` | 度 | 候选 FK 到有效目标的姿态误差阈值。实际判定使用 `max(fk_error_tolerance_deg, target_tolerance.orientation_deg)`。 | 同上。过大可能接受明显偏离的姿态。 |
| `reject_singular` | 布尔 | 候选是否进行奇异拒绝。 | 开启更安全，能避开奇异位姿；关闭可能更容易跟到目标，但速度放大、控制不稳定风险更高。 |
| `soft_singularity_labels` | 字符串数组 | 这些奇异标签不直接拒绝，而是作为软奇异候选加分惩罚。当前常见为 `wrist_singularity`。 | 加入标签会让该类奇异“可用但不优先”；移除则会被硬拒绝。 |
| `soft_singularity_penalty_deg` | 度量惩罚 | 软奇异或允许的姿态奇异逃逸候选，在评分中额外增加的关节角惩罚。 | 变大更不愿选择奇异附近解；变小更愿意穿过或停留在奇异附近。 |
| `pose_slack_penalty_deg` | 度量惩罚 | 当候选 pose fraction 超出理想容差时，给评分加的惩罚；粗跟踪接受时也按超出比例加罚。 | 变大更偏向姿态/位置更准的解；变小更偏向关节动作小或姿态偏好好的解。 |

评分大体是“优先最小化带惩罚的最大关节变化，再看关节变化范数”。所以 penalty 字段不会直接拒绝候选，但会改变候选排序。

## 粗跟踪 `coarse_tracking_*`

粗跟踪用于全量 IK 候选 FK 误差超过普通容差时的“有进展就先接收”策略。它不会无条件放宽：必须足够远、误差没有超过上限、相对参考位姿有足够进展，并且位置/姿态退化在限制内。

| 字段 | 单位/类型 | 实际含义 | 调参影响 |
| --- | --- | --- | --- |
| `coarse_tracking_enabled` | 布尔 | 是否允许粗跟踪接受超出普通 FK 容差但有进展的候选。 | 开启可避免目标很远时完全卡住；关闭会更严格，但远目标或手柄突变时更容易失败。 |
| `coarse_tracking_start_fraction` | 倍数 | 只有参考位姿到目标的 pose fraction 大于等于该值时才启用粗跟踪。 | 变大后只有更远的目标才粗跟踪；变小会更频繁接受近似解。 |
| `coarse_tracking_max_tolerance_fraction` | 倍数 | 粗跟踪候选允许的最大 pose fraction。 | 变大可接受更粗的候选；过大可能明显偏离目标。 |
| `coarse_tracking_max_position_error_m` | 米 | 粗跟踪候选的绝对位置误差上限。若为 0，则使用 `target_tolerance.position_m * max_tolerance_fraction`。 | 变大更容易不断运动；变小更容易在远目标处等待更准的 IK。 |
| `coarse_tracking_max_orientation_error_deg` | 度 | 粗跟踪候选的绝对姿态误差上限。若为 0，则使用 `target_tolerance.orientation_deg * max_tolerance_fraction`。 | 变大更能先追位置或离开坏姿态；过大会产生较大朝向误差。 |
| `coarse_tracking_min_fraction_progress` | fraction | 粗跟踪要求的最小综合进展。比较参考 pose fraction 与候选 pose fraction，也比较 balanced score。 | 变大要求每步改善更多，稳定但容易卡；变小更易移动但可能进展很慢。 |
| `coarse_tracking_orientation_recovery_enabled` | 布尔 | 是否允许“姿态显著改善但位置略变差”的粗跟踪恢复。 | 开启有助于从姿态误差或腕部奇异中恢复；关闭更保护位置误差。 |
| `coarse_tracking_orientation_recovery_min_progress_deg` | 度 | 姿态恢复路径要求的最小姿态改善。 | 变大只接受更明显的姿态恢复；变小更容易牺牲一点位置去修姿态。 |
| `coarse_tracking_orientation_recovery_max_position_worsen_m` | 米 | 姿态恢复路径允许的位置变差上限。 | 变大更容易先转正姿态；过大可能末端位置后退明显。 |

## 局部步进总开关与 full/local 调度

`resolve_step()` 面向高频控制。全量 IK 可能一步跨很远，所以 profile 可以先尝试一个受速度限制的 local step，再按周期或必要时做 full solve。

| 字段 | 单位/类型 | 实际含义 | 调参影响 |
| --- | --- | --- | --- |
| `local_step_enabled` | 布尔 | 是否启用局部步进恢复路径。还要求存在关节速度 profile 和 local provider。 | 开启可降低高频遥操作延迟和大步跳变；关闭则直接用 full IK 结果。 |
| `local_step_first_enabled` | 布尔 | 是否在 full solve 前先尝试 local step。 | 开启更实时，尤其目标连续变化时；但如果 local provider 质量差，可能延迟 full solve。 |
| `local_step_first_full_solve_interval` | 帧数 | local-first 模式下，每隔多少帧强制尝试一次 full solve。0 表示不周期强制。 | 变小会更频繁校正到全量 IK；变大更依赖 local step，CPU 更省但可能漂。 |
| `local_step_first_skip_full_when_far` | 布尔 | 周期 full solve 到期时，如果 local step 已成功且仍离目标较远，是否跳过 full solve。 | 开启可避免远目标时把预算花在 full solve；关闭更规律地校正。 |
| `local_step_first_skip_full_pose_fraction` | 倍数 | 上述“离目标较远”的 pose fraction 阈值。 | 变小更容易跳过 full solve；变大则只有很远时才跳过。 |
| `local_step_first_skip_full_if_budget_low` | 布尔 | 如果 local step 后剩余预算不足，是否跳过周期 full solve。 | 开启可保护实时帧预算；关闭可能提高校正机会但有超时风险。 |
| `local_step_first_full_solve_min_remaining_ms` | 毫秒 | 判定预算不足的剩余时间阈值。 | 变大更保守、更常跳过 full solve；变小更愿意冒险做 full solve。 |
| `local_step_trigger_ratio` | 倍数 | full IK 目标经速度限制后，如果所需步长超过当前最大步长该倍数，才触发 local recovery。 | 变小更频繁启用 local step；变大只有明显大步才启用。 |
| `local_step_hold_on_failure` | 布尔 | local recovery 失败时是否保持参考位姿，而不是回退到 full target。 | 开启更安全，避免无进展的大跳；关闭更激进，可能直接跟 full IK。 |
| `local_step_full_target_step_enabled` | 布尔 | 是否把已接受的 full target 作为 local step 候选之一，再经速度限制评估。 | 开启能平滑过渡到 full 解；关闭更依赖 local provider 自己生成的候选。 |

## 局部步进候选生成与步长

| 字段 | 单位/类型 | 实际含义 | 调参影响 |
| --- | --- | --- | --- |
| `local_step_max_iterations` | 次数 | 当 `local_step_iteration_counts` 为空时，local provider 使用的默认最大迭代数。 | 增大可提高 local IK 质量，但增加耗时。 |
| `local_step_iteration_counts` | 次数数组 | local provider 会按这些迭代次数分别生成候选。会去重并升序。 | 多个 horizon 可提高候选多样性；数组越长耗时越多。 |
| `local_step_scale_factors` | 0 到 1 数组 | 对候选关节增量做缩放。第一个是 primary，其余在失败或逃逸激活时作为 fallback。 | 加入较小 scale 可让系统在大步不安全时尝试小步；太多 scale 会增加评估成本。 |

## 局部步进接受条件

局部步进不是“只要更近就接受”。它会比较当前参考位姿和候选输出位姿到目标的误差，接受以下情况之一：位置进展且姿态退化有限、姿态进展且位置退化有限、综合分数有进展、在容差内、姿态恢复、限位/奇异/姿态/腕部卸载的有界恢复。

| 字段 | 单位/类型 | 实际含义 | 调参影响 |
| --- | --- | --- | --- |
| `local_step_min_progress_fraction` | fraction | 综合 pose score 的最小改善量。 | 变大要求更明确进展，减少无效小步；变小更容易连续微动。 |
| `local_step_accept_position_progress_m` | 米 | 位置改善达到该值，且姿态变差不超过上限时，可接受。 | 变大更看重明显位置改善；变小更容易接受细小位置进展。 |
| `local_step_accept_orientation_progress_deg` | 度 | 姿态改善达到该值，且位置变差不超过上限时，可接受。 | 变大更保守；变小更容易为了姿态做小调整。 |
| `local_step_max_position_worsen_m` | 米 | 普通 local step 中允许的位置变差上限。 | 变大更容易绕开局部困境；过大可能先远离目标。 |
| `local_step_max_orientation_worsen_deg` | 度 | 普通 local step 中允许的姿态变差上限。 | 变大更容易优先追位置；过大可能姿态抖动或漂移。 |
| `local_step_accept_worse_within_tolerance` | 布尔 | 候选已经在 `target_tolerance` 内时，是否即使略退化也接受。若为 false，则至少要求不明显退化。 | 开启可减少容差内抖动拒绝；关闭更严格，可能在目标附近 hold。 |

## 动态退化窗口 `local_step_dynamic_worsen_*`

当当前误差已经很大时，普通退化窗口可能过窄，导致系统不能绕开障碍、奇异或坏分支。动态窗口会按当前误差的一定比例放宽 `local_step_max_*_worsen_*`，但不超过最大值。

| 字段 | 单位/类型 | 实际含义 | 调参影响 |
| --- | --- | --- | --- |
| `local_step_dynamic_worsen_enabled` | 布尔 | 是否启用动态退化窗口。 | 开启更容易在远目标时保持运动；关闭更严格。 |
| `local_step_dynamic_worsen_error_start_m` | 米 | 当前位置误差大于等于该值时激活动态位置窗口。 | 变小会更常放宽；变大只在更远时放宽。 |
| `local_step_dynamic_worsen_error_start_deg` | 度 | 当前姿态误差大于等于该值时激活动态姿态窗口。 | 同上。 |
| `local_step_dynamic_worsen_position_fraction` | 比例 | 动态位置退化上限可取当前误差的该比例。 | 变大更容易接受绕行；过大可能明显后退。 |
| `local_step_dynamic_worsen_orientation_fraction` | 比例 | 动态姿态退化上限可取当前误差的该比例。 | 变大更允许姿态临时变差。 |
| `local_step_dynamic_worsen_max_position_m` | 米 | 动态位置退化窗口的硬上限。 | 是放宽的刹车。过大降低安全感，过小动态窗口效果不明显。 |
| `local_step_dynamic_worsen_max_orientation_deg` | 度 | 动态姿态退化窗口的硬上限。 | 同上。 |

## 局部姿态恢复 `local_step_orientation_recovery_*`

| 字段 | 单位/类型 | 实际含义 | 调参影响 |
| --- | --- | --- | --- |
| `local_step_orientation_recovery_enabled` | 布尔 | 是否允许“姿态大幅改善、位置有界变差”的 local step。 | 开启有助于从腕部姿态错误中恢复；关闭更保护位置。 |
| `local_step_orientation_recovery_min_progress_deg` | 度 | 姿态恢复要求的最小姿态改善。 | 变大更保守；变小更频繁为了姿态牺牲位置。 |
| `local_step_orientation_recovery_max_position_worsen_m` | 米 | 姿态恢复允许的位置变差上限。 | 变大恢复能力强但位置可能后退；变小更严格。 |
| `local_step_orientation_recovery_max_after_position_error_m` | 米 | 姿态恢复后的位置误差上限。0 或更小表示不限制。 | 能避免“姿态好了但位置太远”的候选被接受。 |

## 双臂组件级进度 `local_step_component_progress_*`

这些字段主要用于 `dual_arm_teleop`。双臂总分可能因为一侧轻微退化而看起来没有整体进展，组件级进度允许“至少某一侧/某一组件有明确改善，同时其它组件退化有界”的候选。

| 字段 | 单位/类型 | 实际含义 | 调参影响 |
| --- | --- | --- | --- |
| `local_step_component_progress_enabled` | 布尔 | 是否启用双臂组件级进度接受路径。 | 开启可减少双臂互相牵制导致的卡顿；关闭更依赖总分进展。 |
| `local_step_component_progress_min_position_m` | 米 | 任一侧位置改善达到该值可满足组件进度。 | 变大要求更明显位置进展；变小更容易微动。 |
| `local_step_component_progress_min_orientation_deg` | 度 | 任一侧姿态改善达到该值可满足组件进度。 | 同上。 |
| `local_step_component_progress_max_score_regression` | 分数 | 允许总体双臂 pose score 退化的最大值。 | 变大更宽松；过大可能整体误差变差。 |
| `local_step_component_progress_max_position_worsen_m` | 米 | 组件级接受时，各侧位置允许变差上限。 | 变大更容易通过狭窄区域；过大可能一侧明显偏离。 |
| `local_step_component_progress_max_orientation_worsen_deg` | 度 | 组件级接受时，各侧姿态允许变差上限。 | 变大更宽松。 |
| `local_step_component_progress_max_after_position_error_m` | 米 | 接受后任一侧最大位置误差上限。0 或更小表示不限制。 | 防止某一侧虽然有局部改善但绝对误差已经太大。 |

## 关节限位逃逸 `local_step_limit_escape_*`

当参考关节离位置限位很近时，该路径会生成“远离限位”的候选，并优先考虑最小限位余量有改善的候选。

| 字段 | 单位/类型 | 实际含义 | 调参影响 |
| --- | --- | --- | --- |
| `local_step_limit_escape_enabled` | 布尔 | 是否启用近限位逃逸候选。 | 开启可避免关节贴限位后卡死；关闭更纯粹追目标但可能顶限位。 |
| `local_step_limit_escape_activation_margin_deg` | 度 | 关节离上下限的最小余量小于等于该值时激活。 | 变大更早远离限位；变小只有接近极限才触发。 |
| `local_step_limit_escape_step_fraction` | 0 到 1 | 限位逃逸步长占速度 profile 允许步长的比例。 | 变大逃离更快；过大可能影响末端跟随。 |
| `local_step_limit_escape_max_candidates` | 个数 | 最多生成多少个限位逃逸候选。 | 增大可覆盖更多关节/方向；耗时更多。 |
| `local_step_limit_escape_accept_worse_within_tolerance` | 布尔 | 如果候选在目标容差内且限位余量改善，是否允许小幅退化。 | 开启可在目标附近主动从限位退出来；关闭更严格维持目标误差。 |
| `local_step_limit_escape_accept_bounded_worse` | 布尔 | 不在目标容差内时，是否允许“限位余量改善且 pose 退化有界”的候选。 | 开启更能逃出限位；关闭只接受同时追目标的逃逸。 |
| `local_step_limit_escape_max_position_worsen_m` | 米 | 限位逃逸路径允许的位置变差上限，会与普通 local worsen 上限取较大值。 | 变大逃逸更强；过大可能末端明显离开目标。 |
| `local_step_limit_escape_max_orientation_worsen_deg` | 度 | 限位逃逸路径允许的姿态变差上限。 | 同上。 |

## 奇异逃逸 `local_step_singularity_escape_*`

当参考状态已经奇异时，该路径会尝试每个关节正负方向的小步，优先选择能改善奇异指标且 pose 退化有界的候选。

| 字段 | 单位/类型 | 实际含义 | 调参影响 |
| --- | --- | --- | --- |
| `local_step_singularity_escape_enabled` | 布尔 | 是否启用 local 奇异逃逸候选。 | 开启可从奇异状态中主动挪出；关闭可能依赖 full IK 或保持。 |
| `local_step_singularity_escape_step_fraction` | 0 到 1 | 奇异逃逸步长占速度 profile 允许步长的比例。 | 变大逃离更快但扰动更大；变小更平滑但可能慢。 |
| `local_step_singularity_escape_max_candidates` | 个数 | 最多生成多少个奇异逃逸候选。 | 增大覆盖更多方向；增加 FK/奇异检查成本。 |
| `local_step_singularity_escape_max_position_worsen_m` | 米 | 奇异逃逸允许的位置变差上限。 | 变大更容易脱离奇异；过大会牺牲位置跟随。 |
| `local_step_singularity_escape_max_orientation_worsen_deg` | 度 | 奇异逃逸允许的姿态变差上限。 | 同上。 |

## `metadata.joint_posture_preference`

该块由 `robot_motion.posture_preferences.posture_preference_from_metadata()` 读取，用于给肘部/腕部姿态打分、允许坏分支到好分支的例外、生成 local 姿态逃逸和腕部卸载候选。它不是纯注释，实际影响候选评分和接受。

### 基础姿态评分

| 字段 | 单位/类型 | 实际含义 | 调参影响 |
| --- | --- | --- | --- |
| `enabled` | 布尔 | 是否启用该姿态偏好。为 false 时整块不生效。 | 关闭后不会对肘部好分支加偏好，也不会生成姿态/腕部恢复候选。 |
| `name` | 字符串 | 姿态偏好名称，写入 metadata 便于诊断。 | 不影响算法，只影响可读性。 |
| `j2_index` | 关节索引 | 用作 RM65 肘部姿态判断的第二关节索引。 | 索引错会让所有姿态评分失真。 |
| `j3_index` | 关节索引 | 用作 RM65 肘部姿态判断的第三关节索引。 | 同上。 |
| `j2_high_deg` | 度 | 当 j2 高于该值且 j3 低于坏分支阈值时，认为是高 j2、低 j3 的坏分支。 | 变低会更容易标记坏分支；变高更宽松。 |
| `j2_preferred_min_deg` | 度 | j2 偏好窗口下界。 | 窗口越窄，preferred_bonus 越难触发。 |
| `j2_preferred_max_deg` | 度 | j2 偏好窗口上界。 | 同上。 |
| `j3_zero_margin_deg` | 度 | j3 接近 0 的惩罚区间。绝对值越接近 0，惩罚越大。 | 变大更远离零位穿越；变小更容易经过零位。 |
| `j3_low_deg` | 度 | j3 低值惩罚阈值。j3 低于该值会进入 low zone。 | 变大更强烈推动 j3 抬高；变小更宽松。 |
| `j3_bad_max_deg` | 度 | 与 `j2_high_deg` 组合判断坏分支的 j3 上界。 | 变大更容易把高 j2、较低 j3 判为坏分支。 |
| `j3_preferred_min_deg` | 度 | j3 偏好窗口下界。 | 偏好好分支的核心参数，变高会更强推 j3 到更大角度。 |
| `j3_preferred_max_deg` | 度 | j3 偏好窗口上界。 | 过低会让高 j3 姿态难以获得 bonus；过高可能贴近上限。 |
| `j3_upper_warning_deg` | 度 | j3 高于该值后进入上沿警告区并加惩罚。 | 变低更早防止 j3 过高；变高更宽松。 |
| `bad_branch_penalty_deg` | 度量惩罚 | 被判定为坏分支时加的基础惩罚。 | 变大更强烈避开坏分支；过大可能宁可牺牲 pose。 |
| `j3_negative_penalty_deg` | 度量惩罚 | j3 为负的额外惩罚。 | 变大更不愿进入 j3 负分支。 |
| `high_j2_low_j3_penalty_deg` | 度量惩罚 | 高 j2、低 j3 坏分支的额外惩罚。 | 变大更强烈逃离该组合。 |
| `j3_zero_penalty_deg` | 度量惩罚 | j3 零位附近惩罚。 | 变大更避免穿越 j3 零附近，也可能降低跨分支能力。 |
| `j3_low_penalty_deg` | 度量惩罚 | j3 低值区惩罚。 | 变大更偏向抬高 j3。 |
| `j3_upper_penalty_deg` | 度量惩罚 | j3 上沿警告区惩罚。 | 变大更避免 j3 过高。 |
| `preferred_bonus_deg` | 度量奖励 | 当 j2、j3 都在偏好窗口内时，从姿态惩罚中扣除的奖励。 | 变大更偏向好分支；过大可能压过末端精度或关节平滑。 |

### 硬分支和奇异例外

| 字段 | 单位/类型 | 实际含义 | 调参影响 |
| --- | --- | --- | --- |
| `hard_branch_escape_enabled` | 布尔 | 是否允许候选因显著改善坏分支而越过 `hard_branch_max_delta_deg`。 | 开启可从坏肘形跨回好肘形；关闭会严格禁止大跳。 |
| `hard_branch_escape_allowed_joint_indices` | 关节索引数组 | 只有这些关节超出硬分支阈值时，才允许姿态逃逸例外。 | 范围越大越容易跨分支；范围越小越安全保守。 |
| `hard_branch_escape_min_improvement_deg` | 度量改善 | 允许硬分支逃逸所需的最小姿态分数改善。 | 变大只允许明显变好；变小更容易大步跨分支。 |
| `singular_escape_enabled` | 布尔 | 是否允许候选因从坏分支穿越到好分支而例外通过某些奇异。 | 开启有助于肘部坏分支恢复；关闭则奇异仍被拒绝。 |
| `singular_escape_labels` | 字符串数组 | 允许上述例外的奇异标签，当前为 `elbow_singularity`。 | 加入标签会让对应奇异可在姿态改善时穿越；误加会降低奇异保护。 |
| `singular_escape_min_improvement_deg` | 度量改善 | 允许姿态奇异逃逸所需的最小姿态分数改善。 | 变大更保守；变小更容易穿越奇异。 |
| `singular_escape_j3_abs_max_deg` | 度 | 姿态奇异逃逸时，候选 j3 绝对值必须不超过该值。 | 限制只在接近 j3 零附近穿越；变大穿越范围更宽。 |

### Local 姿态逃逸

| 字段 | 单位/类型 | 实际含义 | 调参影响 |
| --- | --- | --- | --- |
| `local_escape_enabled` | 布尔 | 是否生成朝偏好 j2/j3 窗口移动的 local 候选。 | 开启可主动修肘形；关闭只靠普通 IK 候选。 |
| `local_escape_max_candidates` | 个数 | 姿态逃逸最多生成多少个候选。 | 增大候选更多，耗时更多。 |
| `local_escape_step_fraction` | 0 到 1 | 姿态逃逸步长占速度 profile 允许步长的比例。 | 变大修正快但末端扰动更明显；变小更平滑。 |
| `local_escape_max_position_worsen_m` | 米 | 姿态逃逸接受时允许的位置变差上限。 | 变大更能修肘形；过大可能离目标。 |
| `local_escape_max_orientation_worsen_deg` | 度 | 姿态逃逸接受时允许的姿态变差上限。 | 同上。 |
| `local_escape_min_improvement_deg` | 度量改善 | 生成/接受姿态逃逸候选所需的最小姿态分数改善。 | 变大只接受明显改善；变小更频繁修姿态。 |

### 腕部卸载

腕部卸载用于 j4/j5/j6 接近边界或 wrap 时，通过 j1、j4、j6 的小步组合降低腕部负担。代价函数约为 `j4_excess + 0.5 * j5_excess + j6_excess`。

| 字段 | 单位/类型 | 实际含义 | 调参影响 |
| --- | --- | --- | --- |
| `wrist_unload_enabled` | 布尔 | 是否启用腕部卸载候选。 | 开启可降低腕部贴边和 j6 wrap 风险；关闭更专注追末端。 |
| `wrist_unload_j1_index` | 关节索引 | 用于卸载腕部时可调整的 j1 索引。 | 索引错会移动错误关节。 |
| `wrist_unload_j4_index` | 关节索引 | 腕部 j4 索引。 | 同上。 |
| `wrist_unload_j5_index` | 关节索引 | 腕部 j5 索引。 | 同上。 |
| `wrist_unload_j6_index` | 关节索引 | 腕部 j6 索引，会按 wrap 后角度评估。 | 同上。 |
| `wrist_unload_j4_abs_activation_deg` | 度 | `abs(j4)` 超过该值时认为 j4 需要卸载。 | 变低更早触发；变高更接近限位才触发。 |
| `wrist_unload_j5_abs_activation_deg` | 度 | `abs(j5)` 超过该值时认为 j5 弯折过大。 | 同上。 |
| `wrist_unload_j6_wrap_activation_deg` | 度 | wrap 后 `abs(j6)` 超过该值时认为接近 wrap 边界。 | 变低更早避免绕圈；变高更宽松。 |
| `wrist_unload_step_fraction` | 0 到 1 | 腕部卸载步长占速度 profile 允许步长的比例。 | 变大卸载更快；过大末端扰动更明显。 |
| `wrist_unload_max_candidates` | 个数 | 腕部卸载最多生成多少个候选。 | 增大覆盖更多组合，耗时更多。 |
| `wrist_unload_max_position_worsen_m` | 米 | 腕部卸载接受时允许的位置变差上限。 | 变大更容易卸载腕部；过大可能明显离目标。 |
| `wrist_unload_max_orientation_worsen_deg` | 度 | 腕部卸载接受时允许的姿态变差上限。 | 同上。 |
| `wrist_unload_min_cost_improvement_deg` | 度量改善 | 接受腕部卸载候选所需的最小腕部代价改善。 | 变大只接受更明显卸载；变小更频繁做卸载小动作。 |

## 其它 metadata

| 字段 | 含义 | 影响 |
| --- | --- | --- |
| `metadata.intended_for` | 给 profile 使用场景的文字标记。当前 `dual_arm_teleop` 使用 `high_frequency_dual_arm_control`。 | 当前核心解析器不直接使用，主要用于诊断和人工阅读。 |
| `metadata.note` | 说明性文字。 | 当前核心解析器不直接使用。 |

## 常见调参方向

| 现象 | 优先关注字段 | 调整思路 |
| --- | --- | --- |
| 末端卡住不动，但目标明显可达 | `coarse_tracking_*`、`local_step_min_progress_fraction`、`local_step_accept_*`、`local_step_dynamic_worsen_*` | 适度降低进展阈值或放宽远目标动态退化窗口。 |
| 末端跟随太松、误差偏大 | `target_tolerance.*`、`fk_error_tolerance_*`、`coarse_tracking_max_*`、`pose_slack_penalty_deg` | 收紧容差或提高 slack penalty，必要时降低粗跟踪最大误差。 |
| 出现关节突然翻支 | `hard_branch_*`、`per_step_joint_limit_deg`、`hard_branch_escape_*` | 收紧硬分支阈值，或提高姿态逃逸所需 improvement。 |
| 坏肘形/贴限位后很难恢复 | `joint_posture_preference.*`、`local_step_limit_escape_*`、`local_escape_*` | 开启并适度放宽姿态/限位逃逸，但保持 pose worsen 上限。 |
| 腕部奇异或 j6 wrap 附近抖动 | `soft_singularity_*`、`local_step_singularity_escape_*`、`wrist_unload_*` | 提高软奇异惩罚，开启奇异逃逸/腕部卸载，控制允许退化上限。 |
| 实时循环/GPU 或 CPU 压力大 | `max_candidate_count`、`provider_time_budget_ms`、`enable_expensive_seed_expansion`、`allow_parallel_candidate_generation`、`local_step_iteration_counts` | 减候选、减迭代、关昂贵扩展；并行需要结合机器负载评估。 |

## 修改建议

1. 一次只改一类参数，例如只改粗跟踪或只改限位逃逸，方便从 metadata 判断效果。
2. 优先看运行时 metadata 中的 `reject_reason`、`local_step_progress_accept_reason`、`coarse_tracking.reject_reason`、`local_step_*_bounded_*`，再决定放宽哪一组阈值。
3. 对真实机械臂，先保守调整 `*_max_position_worsen_m` 和 `*_max_orientation_worsen_deg`。这些字段直接决定为了恢复可以暂时离目标多远。
4. 修改 `joint_posture_preference` 的关节索引前，先确认当前机器人 qpos 顺序。索引错误会让姿态偏好产生反效果。

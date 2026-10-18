"use strict";

const token = document.querySelector('meta[name="csrf-token"]').content;
const el = id => document.getElementById(id);
let state = null;
let heartbeatEnabled = false;
let heartbeatQueued = false;
let requestQueue = Promise.resolve();
let cycleAngleInitialized = false;
let refreshInFlight = false;
let stopControlsAvailable = false;
const STAGED_CONSUMED_PHASES = new Set(["breakaway", "releasing", "drop", "track", "settle", "brake", "soft_hold", "authority_exhausted"]);
const ACTION_NAMES = {arm: "添加驱动力", set: "设置驱动力", set_target: "设置位移目标",
  zero: "驱动力归零", withdraw: "撤销驱动力", select_joint: "切换关节", select_mode: "选择测试模式",
  authorize_breakaway: "授权静摩擦起步", stop_and_hold: "停止动作并保持实测停点"};

function number(value, digits = 3) {
  return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function freshJointDispatch(value) {
  const dispatch = value?.last_dispatch;
  return dispatch && dispatch.joint_name === value.joint_name
    && Number.isFinite(value.dispatch_age_sec) && value.dispatch_age_sec >= 0 && value.dispatch_age_sec <= 0.25
    ? dispatch : null;
}

function finalCommandDelta(dispatch) {
  return Number.isFinite(dispatch?.conditioned_total_nm) && Number.isFinite(dispatch?.gravity_nm)
    ? dispatch.conditioned_total_nm - dispatch.gravity_nm : null;
}

function finalZeroConfirmed(dispatch) {
  const delta = finalCommandDelta(dispatch);
  const commands = dispatch?.joint_commands;
  // Display confirmation uses the owner's complete composed-command receipt, not a zero request.
  return delta !== null && Math.abs(delta) <= 1e-9 && dispatch.overlay_after_profile_nm === 0
    && ["gravity_nm", "overlay_nm", "total_nm"].every(key =>
      Array.isArray(commands?.[key]) && commands[key].length === 6 && commands[key].every(Number.isFinite))
    && commands.total_nm.every((total, i) =>
      Math.abs(total - commands.gravity_nm[i]) <= 1e-9 && commands.overlay_nm[i] === 0);
}

function directDriveLabel(value) {
  // Presentation only: authority and command history remain owned by the owner.
  if (!value) return "未连接";
  if (value.phase === "fault") return "保护已触发";
  if (!value.owner_ready) return "等待 owner 就绪";
  if (!Number.isFinite(value.feedback_age_sec) || value.feedback_age_sec < 0 || value.feedback_age_sec > 0.15) return "反馈状态待确认";
  if (["onset_release", "velocity_check"].includes(value.phase)) return "撤力与停稳确认中";
  if (value.phase === "arm_requested") return "准备中：检查六轴静止";
  const dispatch = freshJointDispatch(value);
  if (value.phase === "inactive") {
    return finalZeroConfirmed(dispatch) && value.effective_assist_nm === 0
      ? "仅重力补偿" : "等待最终零助力发送确认";
  }
  if (value.phase !== "armed") return value.phase;
  if (!dispatch || !Number.isFinite(dispatch.overlay_after_profile_nm)) return "已授权：等待新鲜发送回执";
  if (value.desired_assist_nm === 0 && value.effective_assist_nm === 0) {
    return finalZeroConfirmed(dispatch) ? "已授权：零目标，最终附加命令为零" : "零目标：等待最终归零回执";
  }
  const action = lastOperatorEvent(value.events);
  const finalDelta = finalCommandDelta(dispatch);
  if (dispatch.overlay_after_profile_nm === 0 || finalDelta === null || Math.abs(finalDelta) <= 1e-9
      || (action?.action === "set" && dispatch.feedback_timestamp_sec < action.time_sec)) {
    return "目标已接收：等待加力发送确认";
  }
  return value.onset_capture?.probe?.state?.stage === "hold" ? "定力观察：暂停加力" : "正在加力：已有发送回执";
}

function stopCaptureMessage(value) {
  const motion = value?.motion_control;
  if (!value?.motion_control_enabled || !value.owner_ready || (!value.operator_armed && value.phase !== "releasing")
      || !Number.isFinite(value.feedback_age_sec) || value.feedback_age_sec < 0 || value.feedback_age_sec > 0.15
      || !motion || !["brake", "soft_hold", "releasing"].includes(value.phase)
      || (value.phase === "releasing" ? !["brake", "soft_hold"].includes(motion.phase) : motion.phase !== value.phase)) return null;
  const reasons = {
    selected_raw_velocity: "选中关节瞬时位置差分速度未通过静止门槛",
    selected_window_velocity: "选中关节窗口速度未通过静止门槛",
    support_window_velocity: "支撑关节窗口速度未通过静止门槛",
    support_raw_velocity: "支撑关节瞬时位置差分速度未通过静止门槛",
    measured_position_changed: "停稳观察期间位置变化，计时已重置",
    stationary_dwell_incomplete: "速度门已通过，等待连续停稳时间",
    braking_timeout: "制动超时，驱动力撤销",
  };
  return motion.phase === "soft_hold"
    ? "已捕获实际停点；请同时检查原目标误差，停点保持不等于到位。"
    : `等待停点捕获：${reasons[motion.stop_capture_block_reason] || "等待 owner 门控信息"}。连续停稳 ${number(motion.stop_capture_dwell_elapsed_sec, 3)} / ${number(motion.stop_capture_required_dwell_sec, 3)} s；静止门槛 ${number(motion.stop_capture_velocity_limit_rad_s, 4)} rad/s（不是硬保护限速）。`;
}

function isStaged() {
  return state?.controller_mode === "staged_assist"
    || (isAutomatic() && state.automatic_calibration.stage === "motion");
}

function isAutomatic() { return state?.controller_mode === "automatic_calibration"; }

function isSupportedCycle() {
  return isAutomatic() && ["supported_cycle", "workspace"].includes(state?.mode_selection?.selected);
}

function pageVisibilityIntent(hidden, backgroundCycle) {
  if (backgroundCycle) return {action: "visibility", visible: !hidden};
  return hidden ? {action: "withdraw", source: "page_hidden"} : null;
}

function renderAutomatic() {
  const automatic = isAutomatic();
  const selection = automatic ? state.mode_selection : null;
  el("mode-workspace").hidden = !selection?.available_modes?.includes("workspace");
  el("mode-selector").hidden = !selection;
  el("cycle-angle-control").hidden = !selection || selection.selected !== "supported_cycle";
  if (selection) {
    if (!cycleAngleInitialized && Number.isFinite(selection.cycle_angle_deg)) {
      el("cycle-angle").value = String(selection.cycle_angle_deg);
      cycleAngleInitialized = true;
    }
    el("cycle-angle").max = selection.maximum_cycle_angle_deg ?? 20;
    el("cycle-angle").disabled = selection.locked || selection.selected !== "supported_cycle";
    if (selection.locked && Number.isFinite(selection.cycle_angle_deg)) {
      el("cycle-angle").value = String(selection.cycle_angle_deg);
    }
    el("cycle-angle-note").textContent = `可调范围：大于 0°，不超过 ${selection.maximum_cycle_angle_deg ?? 20}°（本次连接前已准入的上限）。随“开始完整流程”提交；启动后锁定，不改变本轮返回起点。`;
    const draftAngle = Number(el("cycle-angle").value);
    if (selection.selected === "supported_cycle" && (!Number.isFinite(draftAngle)
        || draftAngle <= 0 || draftAngle > (selection.maximum_cycle_angle_deg ?? 20))) {
      el("cycle-angle-note").textContent = `角度无效，尚不能启动：请填写大于 0° 且不超过 ${selection.maximum_cycle_angle_deg ?? 20}° 的数值。`;
    }
    el("mode-fixed-position").setAttribute("aria-pressed", String(selection.selected === "fixed_position"));
    el("mode-supported-cycle").setAttribute("aria-pressed", String(selection.selected === "supported_cycle"));
    el("mode-workspace").setAttribute("aria-pressed", String(selection.selected === "workspace"));
    for (const id of ["mode-fixed-position", "mode-supported-cycle", "mode-workspace"]) el(id).disabled = selection.locked;
    const label = selection.selected === "workspace" ? "J2/J3 多位姿自动采集" : selection.selected === "supported_cycle" ? `完整流程（抬升 ${el("cycle-angle").value}° → 测试 → 返回）` : "当前位置小范围测试";
    el("mode-selection-status").textContent = selection.locked
      ? `本轮已锁定：${label}。可随时撤力或停机，但不能中途切换模式。`
      : !selection.ready ? `已选择：${label}；等待 owner 确认零驱动力周期，尚未开始。`
      : `后端已选定：${label}。选择不会驱动；声明无人介入后，点击下方“开始”执行。`;
  }
  el("automatic-panel").hidden = !automatic;
  el("arm").textContent = automatic ? "开始自动标定（授权本批驱动）" : "允许驱动（先准备）";
  el("drive-instructions").textContent = automatic
    ? "先声明无人施加外力，再启动一次完整批次，无需填目标。保持页面可见；只做重力补偿、有人介入或连接超时都会撤力。结束后需重新运行 owner 才能开始新批次。"
    : "允许驱动只授权和检查静止，不会自动加力。通过后填写下方目标并点击提交；以 owner 的发送回执为准。";
  for (const id of ["step-label", "step", "minus", "plus", "target-label", "target", "apply", "zero"]) {
    el(id).hidden = automatic;
  }
  el("cycle-disable-hint").hidden = true;
  el("full-disable").textContent = "软件急停：J1–J6 全失能";
  if (!automatic) return;
  const batch = state.automatic_calibration;
  el("automatic-sequence-note").textContent = "先采集正负起动与撤力回退，再采集双速度小范围运动。编码器进展不等于输出侧运动或真实摩擦力；不会自动安装补偿。";
  const startButton = el("arm");
  startButton.disabled = !state.owner_ready || batch.state !== "idle" || batch.start_requested
    || (selection && !selection.ready)
    || batch.human_intervention !== false || state.transparency_assessment?.available !== true;
  el("joint-select").disabled = true;
  el("authorize-breakaway").hidden = true;
  el("stop-and-hold").hidden = true;
  el("arm").setAttribute("aria-pressed", String(batch.active || batch.start_requested));
  el("withdraw").setAttribute("aria-pressed", String(!(batch.active || batch.start_requested)));
  el("command-state").textContent = "自动批次由唯一 owner 执行；当前附加软件命令和发送回执见下方。";
  el("gate").textContent = `${batch.reason}。硬速度、位移、反馈和最终命令保护保持启用。`;
  if (batch.workspace) {
    renderWorkspace(batch);
    return;
  }
  el("withdraw").textContent = batch.cycle ? "停止本轮并撤销驱动力" : "停止采集并撤销驱动力";
  const cycleDisableReady = batch.cycle?.phase === "complete"
    && batch.cycle.disable_confirmation_available === true
    && batch.cycle.zero_assist_base_damping_confirmed === true
    && state.stop_controls_available === true;
  el("automatic-mode-title").textContent = batch.cycle ? "模式二：支撑原位出发的完整流程" : "模式一：当前位置定点小范围采集";
  el("automatic-mode-note").textContent = batch.cycle
    ? "包含指定角度的搬运、重力观察、局部采集和返回；正常返回支撑起点并确认后继续保持重力补偿与基础阻尼，等待你在 GUI 明确确认 J1–J6 全失能。角度在启动前设置，开始后锁定。"
    : "围绕启动时的当前位置采集，不做抬升搬运，不自动返回旧原位或自动失能；结束后保留重力保护保持。开始前可用上方按钮选择。";
  if (!batch.cycle) el("arm").textContent = "开始当前位置小范围采集";
  if (batch.cycle) {
    el("arm").textContent = "开始完整流程：抬升 → 测试 → 返回 → 等待确认失能";
    el("drive-instructions").textContent = `一次启动执行完整流程；起点须有可靠支撑，并清空 J3 增加 ${el("cycle-angle").value}° 及局部采集的运动空间。切换标签或最小化时流程继续，后台许可最长 ${number(batch.cycle.background_permit_timeout_sec, 2)} s 无心跳后撤力停止；关闭/刷新页面或点击“停止本轮”会撤力，不自动返回旧位置。正常返回并确认停稳后继续保持重力补偿/阻尼，等待 GUI 人工确认全失能。`;
  } else {
    el("withdraw").textContent = "停止采集并撤销驱动力";
    el("drive-instructions").textContent = "自动采集由唯一 owner 执行；有人介入、页面隐藏、关闭页面或连接超时会撤力停止。切换前先撤销附加驱动力，物理急停优先。";
  }
  if (cycleDisableReady) {
    el("cycle-disable-hint").hidden = false;
    el("cycle-disable-hint").textContent = "完整流程已完成：J3 已返回支撑原位，附加助力为零且基础阻尼已确认。owner 仍在运行重力补偿与阻尼；确认机械臂已由可靠支撑承托后，点击下方“确认返回支撑原位并全失能”完成掉能。";
    el("full-disable").textContent = "确认返回支撑原位并全失能 J1–J6";
  }
  const names = {idle: "等待启动", onset_prepare: "准备支撑轴阻尼与六轴停稳", onset_arm: "准备起动采集",
    onset_scan: "渐增加力 / 定力观察", onset_release: "撤力并确认停稳", motion_prepare: "准备低速运动",
    onset_restore: "恢复基础阻尼并确认停稳",
    motion: "低速运动采集", motion_cancel: "撤销运动并停稳", complete: "采集批次完成",
    failed: "采集未完成", cancelled: "已停止采集", positioning_out: "缓慢移动 J3 至测试位置",
    gravity_observe: "撤去驱动力，观察纯重力补偿漂移", calibration: "运行既有完整标定",
    returning: "缓慢返回起始位置", return_observe: "返回后撤力停稳确认", stopping: "整轮已撤销，等待撤力停稳"};
  const deviations = (batch.motion?.completed_legs || []).filter(leg => leg.tracking_target_reached === false).length;
  const label = batch.state === "complete" && deviations
    ? "采集完成（含定位偏差）" : names[batch.state] || batch.state;
  el("phase").textContent = label;
  el("automatic-progress").textContent = `${label}；起动试次 ${batch.completed_onset_trials}/${batch.onset_count}；运动段 ${batch.motion?.step_index ?? 0}/${batch.motion?.step_count ?? batch.planned_motion_steps ?? 14}；原因 ${batch.reason}`;
  if (batch.cycle) {
    const cycle = batch.cycle;
    const origin = cycle.origin_qpos_rad?.[2];
    const actual = cycle.current_qpos_rad?.[2];
    el("automatic-progress").textContent += `；整轮 ${cycle.phase}；J3 起点 ${number(origin, 5)} / 当前 ${number(actual, 5)} / 抬升目标 ${number(cycle.outbound_target_rad, 5)} rad；相对起点 ${number((actual-origin)*180/Math.PI, 2)}°；整轮命令暴露 ${number(cycle.command_exposure_nm_sec, 2)} N·m·s（不是测得的能量）。`;
    if (cycle.page_visible === false) {
      el("automatic-progress").textContent += `；GUI 页面在后台，流程继续运行；后台心跳许可 ${number(cycle.background_permit_timeout_sec, 2)} s。`;
    }
    if (cycle.positioning) el("automatic-progress").textContent += ` 搬运状态 ${cycle.positioning.state}，${cycle.positioning.reason}；J3 位置到位门 ${number(cycle.positioning_limits?.arrival_tolerance_rad, 5)} rad。`;
    const control = cycle.positioning_control;
    if (control) el("automatic-progress").textContent += ` 最近搬运轨迹阶段 ${control.phase} / ${control.reason}；距目标 ${number(control.target_error_rad, 5)} rad；软制动续行次数 ${control.speed_governor_resume_count}；停稳门 ${control.stop_capture_block_reason ?? "—"}；零力矩回执门 ${control.speed_governor_resume_block_reason ?? "—"}。`;
    if (control?.continuous_speed_scale != null) el("automatic-progress").textContent += ` 连续调速：朝目标窗口速度 ${number(control.continuous_speed_observed_rad_s, 4)} rad/s，减力起点 ${number(control.continuous_speed_start_rad_s, 4)}，停稳软门 ${number(control.tracking_speed_brake_limit_rad_s, 4)} rad/s；调速比例 ${number(control.continuous_speed_scale * 100, 0)}%；跟踪加力 ${number(cycle.positioning_limits?.tracking_torque_rise_nm_s, 2)} / 正常减力 ${number(cycle.positioning_limits?.tracking_torque_fall_nm_s, 2)} N·m/s。保护制动和反转仍用快速通道；以上非实测输出力。`;
  }
  if (batch.outcome?.severity === "warning") {
    el("phase").textContent = "采集告警：本批未完成";
    el("automatic-progress").textContent += "；这是采集能力或质量告警，不是已确认的硬件故障；本批已经停止，不会自动重试。已完成数据保留，控制故障和数据能否拟合另行判定。";
  } else if (batch.outcome?.severity === "error") {
    el("automatic-progress").textContent += "；故障：需要排查原因，不能按普通采集告警忽略。";
  }
  if (batch.motion_plan === "motion_check") {
    el("automatic-progress").textContent += "；本轮验证负向低速运动及正向返回，共两段，不用于摩擦模型拟合。";
  } else if (batch.motion_plan === "local_identification") {
    el("automatic-progress").textContent += "；本轮复用已验证的受限驱动做局部双向、多速度采集，训练段与整段留出验证分开。起步、制动和到位过程不冒充稳态摩擦；不自动修改补偿参数。";
  }
  const onsetTrials = batch.onset_trials || [];
  const confirmedOnsetCount = onsetTrials.filter(trial => trial.outcome === "motion_onset_confirmed").length;
  el("automatic-data-status").textContent = onsetTrials.length
    ? `起步数据独立记录：${onsetTrials.length} 个已结束试次，其中 ${confirmedOnsetCount} 次确认编码器进展。后续运动段或撤力阶段失败不会回溯作废这些已结束试次；未确认进展仍按各自结果标记。`
    : "起步试次单独记录；后续运动采集状态不会改变已结束起步试次的有效性。";
  if (batch.stage === "motion" && batch.motion) {
    const motion = batch.motion;
    const consumed = motion.batch_budget_counts_withdrawal ? motion.batch_command_exposure_nm_sec : motion.active_command_exposure_nm_sec;
    el("automatic-progress").textContent += `；采集到位门 ${number(motion.acquisition_arrival_tolerance_rad, 5)} rad，跟踪质量门 ${number(motion.tracking_arrival_tolerance_rad, 5)} rad；本段命令暴露 ${number(motion.leg_command_exposure_nm_sec, 2)} / ${number(motion.max_leg_command_exposure_nm_sec, 2)}，整批 ${number(consumed, 2)} / ${number(motion.max_batch_command_exposure_nm_sec, 2)} N·m·s（不是实测能量）。已记录定位偏差 ${deviations} 段；数据能否拟合仍需查看报告。`;
  }
  if (batch.stage === "onset" && batch.active && batch.onset_baseline && batch.onset_baseline.gate !== "trial_started") {
    const baseline = batch.onset_baseline;
    el("automatic-progress").textContent += `；首次加力前等待零助力与连续停稳：${number(baseline.dwell_sec, 2)} / ${number(baseline.required_dwell_sec, 2)} s，${preparationReasonLabel(baseline.gate)}。尚未开始新的起动试次。`;
  }
  if (batch.stage === "onset" && batch.support_damping) {
    const gains = batch.support_damping.last_acknowledged_kd_nm_s_per_rad
      ? batch.support_damping.last_acknowledged_kd_nm_s_per_rad.map(v => number(v, 3)).join(" / ") : "尚无回执";
    const damping = batch.support_damping;
    const release = damping.release_braking_enabled
      ? `加力扫描时 J3 Kd 为零；撤力后按限速升至 ${number(damping.release_target_kd_nm_s_per_rad?.[2])}，下一扫描前恢复零。起动 J3 速度保护 ${number(damping.selected_joint_velocity_limit_rad_s)} rad/s，支撑轴与运动段保护仍为 ${number(damping.support_joint_velocity_limit_rad_s)} rad/s。`
      : "J3 保持零 Kd。";
    el("automatic-progress").textContent += `；起动阶段最近完整发送 Kd：${gains}。${release}这是阻尼参数，不是实测力矩。`;
  }
  if (batch.feedback_velocity_timebase === "joint_receive") {
    el("automatic-progress").textContent += ` 差分速度按各关节有效接收时间对齐；运动段速度硬门 ${number(state.limits.max_joint_velocity_rad_s, 3)} rad/s。快照周期、反馈年龄和配对时钟分别校验；六轴异步间隔差异仅记时序告警，真正时基无效仍停止助力。整臂快照差分独立记录。`;
  }
  const releaseObservation = state.onset_capture?.release_support_observation;
  if (batch.stage === "onset" && releaseObservation) {
    const message = releaseObservation.state === "observing"
      ? "撤力后支撑轴微动告警：保持零助力与阶段阻尼，正在限时确认，不恢复原加力。"
      : releaseObservation.state === "confirmed"
        ? "支撑轴微动观察已确认；仍须完成原六轴停稳条件，才结束本试次。"
        : `支撑轴撤力观察故障：${releaseObservation.reason}。`;
    el("automatic-progress").textContent += ` ${message} 支撑轴总变化范围上限 ${number(releaseObservation.position_range_limit_rad, 6)} rad；观察时钟不续期。`;
  }
  if (batch.stage === "motion" && state.tracking_recovery) {
    const recovery = state.tracking_recovery;
    el("automatic-progress").textContent += `；TRACK 维持项：PD ${number(recovery.pd_request_nm)} + 恢复 ${number(recovery.bias_request_nm)} N·m；速度反馈削减 ${number(recovery.velocity_feedback_cut_nm ?? 0)} N·m；期望 / 窗口速度 ${number(recovery.target_velocity_rad_s, 4)} / ${number(recovery.window_velocity_rad_s, 4)} rad/s；总上限 ${number(recovery.total_cap_nm)} N·m。以上是软件需求，实际发送以派发回执为准。`;
    if (recovery.dispatch_credit_mode === "maintenance_first_bounded_by_sent_total") {
      el("automatic-progress").textContent += ` 回执对维持项削减 ${number(recovery.dispatch_tracking_cut_nm ?? 0)} N·m；已有维持项优先分配实际总助力，不假定请求的 PD 已完整执行。`;
    }
  }
  if (batch.stage === "motion" && state.motion_control?.arrival_stabilization_active) {
    const motion = state.motion_control;
    el("automatic-progress").textContent += state.motion_control.arrival_unloading
      ? ` 正常到位渐卸助力中；提前制动门 ${number(motion.arrival_brake_error_rad, 5)} rad，位置到位门 ${number(motion.arrival_unload_exit_error_rad, 5)} rad。须确认最终零助力，再重新计满停稳窗口。`
      : ` 正常到位稳定中；位置到位门 ${number(motion.arrival_unload_exit_error_rad, 5)} rad；保留受限维持助力，回退将校正当前目标，原时限及预算不重置。`;
  }
  el("automatic-trials").textContent = (batch.onset_trials || []).map((trial, i) =>
    `${i + 1}. ${trial.direction > 0 ? "正向" : "负向"}：${trial.outcome}；软件派发峰值 ${number(trial.peak_dispatched_overlay_nm)} N·m；净编码器位移 ${number(trial.final_qpos_rad?.[2] - trial.baseline_qpos_rad?.[2], 6)} rad`
  ).join("\n") || "尚无已撤力并停稳的试次。";
}

function renderWorkspace(batch) {
  const workspace = batch.workspace;
  const task = workspace.current_task;
  const names = {idle: "等待启动", position: "分段换位", sweep: "双向双速采集",
    observe: "零助力重力观察", return: "分段返回起点", return_observe: "返回后观察",
    advancing: "准备下一阶段", stopping: "撤力停稳", complete: "多位姿采集完成",
    failed: batch.outcome?.severity === "warning" ? "采集未完成 · 告警，数据保留" : "采集故障 · 已停止",
    cancelled: "已撤销"};
  const label = names[workspace.phase] || workspace.phase;
  const ranges = workspace.acquisition_ranges_deg || {joint2: [0, 10], joint3: [0, 10]};
  el("automatic-mode-title").textContent = "J2/J3 多位姿自动采集 · 重力优先";
  el("automatic-mode-note").textContent = `本批采集范围 J2 ${ranges.joint2.join("–")}°、J3 ${ranges.joint3.join("–")}°；从 J2/J3 零位附近（不超过 ${workspace.maximum_start_angle_deg ?? 2}°）开始，先展开 J3 再移动 J2，返回时顺序相反。完整机械角度范围不等于任意组合可达；本功能不替代自碰撞/环境检查。J2 −0.3° 只作已有反馈边界处理，不生成负目标。`;
  el("arm").textContent = "开始 J2/J3 多位姿自动采集";
  el("withdraw").textContent = "停止采集并撤力（不自动返回）";
  el("drive-instructions").textContent = "起点须有可靠支撑，并清空整个采集区域。无人介入后启动；后台继续，关闭页面或人工介入撤力停止。正常完成后逐轴返回起点，等待 GUI 人工确认失能。";
  if (["failed", "cancelled"].includes(workspace.phase)) {
    const gravityOnly = batch.outcome?.severity === "warning" && state.owner_ready && !state.fault;
    el("drive-instructions").textContent = gravityOnly
      ? "本批采集已停止，当前仅重力补偿；不会自动重试、继续加力或返回起点。数据已保留。需要结束时，先确认可靠支撑，再使用全失能按钮；这不是程序仍在等待下一段。"
      : "本批采集已停止提交新目标，不会自动重试或返回。撤力、保护保持和失能是否完成，以上方 owner 状态及回执为准；停止采集不等于全失能。";
  }
  el("phase").textContent = label;
  const target = Number.isFinite(task?.target) ? `，换位目标 ${number(task.target * 180 / Math.PI, 2)}°` : "";
  const reason = workspace.reason === "leg_timeout" ? "本段在时限内未完成（leg_timeout）" : workspace.reason;
  el("automatic-progress").textContent = `${label}；任务 ${workspace.task_index}/${workspace.task_count}；关节采集 ${workspace.completed_joint_acquisitions}/${workspace.expected_joint_acquisitions}；当前 ${task?.joint || "六轴观察"}${target}；原因 ${reason}；命令暴露 ${number(workspace.command_exposure_nm_sec, 2)} / ${number(workspace.max_command_exposure_nm_sec, 2)} N·m·s（非实测能量）。`;
  const centres = (workspace.pose_centres_rad || []).map(q => `(${number(q[0] * 180 / Math.PI, 0)}°, ${number(q[1] * 180 / Math.PI, 0)}°)`).join(" → ");
  el("automatic-data-status").textContent = `配置组：${workspace.setup_label}。采样中心 (J2,J3)：${centres}。重力参数本轮冻结；保存已完成段与未完成段，数据质量和流程状态分开。不自动安装重力或摩擦补偿；本阶段不执行静摩擦阈值扫描。`;
  el("automatic-trials").textContent = "";
  el("automatic-sequence-note").textContent = "本阶段先采集零驱动力下的重力漂移，再采集 J2/J3 双向双速度运动，不执行额外静摩擦阈值扫描。软件力矩命令不是实测受力；不会自动安装补偿。";
  const ready = batch.cycle?.disable_confirmation_available === true && state.stop_controls_available === true;
  el("cycle-disable-hint").hidden = !ready;
  if (ready) {
    el("cycle-disable-hint").textContent = "J2/J3 已返回起始位置，零助力和基础阻尼已确认；请确认机械臂确实由可靠支撑承托，再点击确认失能。角度到位不是物理支撑传感器。";
    el("full-disable").textContent = "确认已支撑并全失能 J1–J6";
  }
}

function preparationReasonLabel(reason) {
  const reasons = {
    ready: "六轴窗口速度、位移范围和零助力派发已通过",
    feedback_unavailable: "尚无完整反馈",
    feedback_not_fresh: "反馈时间不在当前 40 ms 新鲜窗口内",
    velocity_window_not_ready: "反馈窗口尚未完整",
    raw_velocity_hard_guard: "原始差分速度超过硬门限",
    not_stationary: "窗口速度仍超过静止门限",
    velocity_window_displacement_exceeded: "窗口内位置变化范围过大",
    zero_dispatch_unconfirmed: "尚无新鲜的六轴最终零助力派发确认",
    onset_capture_position_not_settled: "采集准备/停稳窗口内仍有超过一个计数的位置变化",
    continuous_baseline_dwell: "正在连续确认停稳",
    support_damping_or_zero_not_settled: "等待支撑阻尼与六轴零助力回执稳定",
  };
  return reasons[reason] || reason || "尚无准备判定";
}

function onsetCaptureDetails(capture) {
  const trial = capture?.trial;
  if (!capture) return "";
  const crossings = [1, 2, 3].map(count => {
    const observation = trial?.[`first_${count}_count_observation`];
    return `${count} 计数首见 ${number(observation?.dispatch?.overlay_after_profile_nm, 3)} N·m / ${number(observation ? observation.feedback_timestamp_sec - trial.started_at_sec : null, 3)} s`;
  }).join("；");
  const criterion = trial?.first_criterion_observation;
  if (capture.probe) {
    const outcomeNames = {
      motion_onset_confirmed: "已确认新的编码器进展（非输出侧运动验收）",
      probe_cumulative_displacement_withdrawn: "整轮累计位移到门后撤力（未确认持续移动）",
      probe_inconclusive: "定力观察不明确，已撤力",
      inconclusive_velocity_spike: "速度尖峰先撤力，不能作为起动样本",
      operator_zero: "操作者置零结束",
      observing: "正在观察，尚无确认",
    };
    return `已停稳结束 ${capture.trials.length} 轮；当前结果：${outcomeNames[trial?.outcome] || trial?.outcome || "尚未开始"}；本轮软件派发峰值 ${number(trial?.peak_dispatched_overlay_nm, 3)} N·m（不是静摩擦阈值）。整轮净位移 ${number(capture.probe.trial_displacement_rad, 6)} rad / 撤力门 ${number(capture.probe.cumulative_displacement_limit_rad, 6)} rad；局部短暂停住不重置整轮基线。${crossings}（只是位置首见）。只有确认记录才显示相应命令 ${number(trial?.onset_dispatch?.overlay_after_profile_nm, 3)} N·m；即使确认也不是实际电机力矩或输出侧摩擦测量。完整姿态、保持/恢复和撤力原因写入 trace/result，下一轮须手动添加驱动力。`;
  }
  return `已停稳结束 ${capture.trials.length} 轮；当前结果 ${trial?.outcome || "尚未开始"}；本轮软件派发助力峰值 ${number(trial?.peak_dispatched_overlay_nm, 3)} N·m；${crossings}（只是位置首见，可能为量化/预滑，不是起动判据）。首次满足完整位移/速度判据 ${number(criterion?.dispatch?.overlay_after_profile_nm, 3)} N·m / ${number(criterion ? criterion.feedback_timestamp_sec - trial.started_at_sec : null, 3)} s；连续确认起动时最近完整派发助力 ${number(trial?.onset_dispatch?.overlay_after_profile_nm, 3)} N·m，确认位移 ${number(trial?.onset_displacement_rad, 5)} rad。置零目标也会结束本轮，下一轮须重新添加驱动力。完整姿态、方向、时间及命令写入 trace/result；不是实际电机或人体受力测量。`;
}

function gateMessage(value) {
  if (value.phase === "fault") return value.controller_mode === "staged_assist"
    ? "助力已撤销；查看 owner 状态和保护原因。"
    : "已请求撤销助力；最终发送状态请查看 owner 回执和保护原因。";
  if (!value.owner_ready) return "等待 STARTUP ACTIVE / TORQUE CONSOLE READY。";
  if (value.feedback_age_sec === null || value.feedback_age_sec > 0.15) {
    return "反馈尚未足够新鲜，暂不能准备。";
  }
  if (value.onset_capture) {
    const capture = value.onset_capture;
    const probe = capture.probe;
    if (probe && value.phase === "inactive" && probe.human_intervention !== false) {
      return "带力观察候选：先明确声明无人施加外力，再添加驱动力。有人/未知状态不能继续加力；没有自动接触检测。";
    }
    if (value.phase === "onset_release") {
      if (probe && capture.trial?.outcome === "probe_cumulative_displacement_withdrawn") {
        return `整轮累计位移已达到撤力门 ${number(probe.cumulative_displacement_limit_rad, 6)} rad，停止继续加力；不是持续移动/静摩擦阈值确认。${capture.first_zero_completed_at_sec == null ? "等待六轴零助力派发" : "零派发已确认，等待停稳"}；不会恢复旧目标。`;
      }
      return `${capture.trial?.outcome === "motion_onset_confirmed" ? (probe ? "已确认新的编码器位置进展（非输出侧验收）" : "已确认持续起动") : "本轮撤力"}：助力请求已清零，${capture.first_zero_completed_at_sec == null ? "等待六轴最终零助力派发" : "零派发已确认，等待完整停稳窗口"}。GUI 和重力补偿保留；不能继续加力，不会自动开始下一轮，真实超速/位移/反馈故障仍进入保护。`;
    }
    if (value.phase === "inactive" && capture.trial?.completed_at_sec != null) {
      return `${capture.trial.outcome === "motion_onset_confirmed" ? "起动记录已完成" : "本轮已撤力结束（不是有效起动样本）"}；试次结束时已确认六轴零助力与停稳窗口，当前发送状态见回执。重新点击添加驱动力才会开始下一轮，旧驱动力不会恢复。`;
    }
    if (value.phase === "armed") {
      if (value.desired_assist_nm === 0 && value.effective_assist_nm === 0 && capture.trial == null) {
        return "零目标准备已完成，尚未开始采集；最终是否归零请查看发送确认。填写非零目标并点击提交；已准备状态的原低速保护仍启用。";
      }
      if (probe) {
        const hold = probe.state;
        const trialProgress = `整轮净位移 ${number(probe.trial_displacement_rad, 6)} / ${number(probe.cumulative_displacement_limit_rad, 6)} rad；累计到门即撤力，不因局部停住继续升级。`;
        return hold?.stage === "hold"
          ? `定力观察：保持附加助力 ${number(hold.hold_nm, 3)} N·m，不向目标继续爬升；${hold.first_hold_completed_at_sec == null ? "等待最终保持命令派发" : "保持已确认，观察新反馈"}。新的位置进展 ${number(hold.maximum_progress_rad, 6)} rad；不同新位置 ${hold.distinct_progress_samples}/${capture.settings.confirm_samples}。至少观察 ${number(probe.settings.hold_sec, 3)} s，最长 ${number(probe.maximum_hold_sec, 3)} s；微位移后须重新确认最近窗口停稳才恢复爬升，继续进展则撤力。${probe.reason === "waiting_recent_plateau" ? "正在等待新的稳定平台。" : ""}${trialProgress}重力仍实时更新。`
          : `渐增加力：${probe.reason}；出现新的同向候选位置时定力观察，不把旧滑窗重复帧当成运动确认。${trialProgress}整轮上限 ${number(probe.settings.max_trial_sec, 2)} s；累计软件命令暴露 ${number(probe.command_exposure_nm_sec, 4)} / ${number(probe.settings.max_command_exposure_nm_sec, 4)} N·m·s，不是实测力矩冲量。`;
      }
      return `起动界限采集：首次非零目标建立新基线，后续调力不重置；同向位移至少 ${number(capture.settings.displacement_rad, 5)} rad、窗口速度至少 ${number(capture.settings.window_velocity_rad_s, 4)} rad/s，连续 ${capture.confirmation_streak}/${capture.settings.confirm_samples} 帧后自动撤力。无自动人工接触检测，当前记录不等于实测静摩擦力矩。`;
    }
  }
  if (value.quantized_velocity_debug) {
    const debug = value.quantized_velocity_debug;
    const velocity = value.velocity_window?.ready ? value.control_qvel_rad_s?.[2] : undefined;
    if (value.phase === "velocity_check") {
      const spans = debug.confirmation_position_span_rad_six;
      const position = debug.confirmation_first_feedback_at_sec == null
        ? "等待零派发后的首帧实测反馈"
        : spans ? `零派发后六轴最大位置范围 ${number(Math.max(...spans), 6)} rad`
          : `零派发后位置范围 ${number(debug.confirmation_position_span_rad, 6)} rad`;
      return `单计数速度尖峰${debug.withdrawal_axes ? `（${debug.withdrawal_axes.join("、")}）` : ""}：驱动力请求已归零；${debug.zero_dispatch_confirmed ? "六轴零助力派发已确认" : "等待最终六轴零助力派发"}。${position}；${value.velocity_window?.ready ? `撤力后窗口速度 ${number(velocity, 5)} rad/s` : "撤力后观察窗口尚未完整"}；须记录 ${number(debug.confirmation_window_sec, 3)} s 完整窗口，总确认期限 ${number(debug.confirmation_timeout_sec, 3)} s。此时不能调力或重新授权，持续运动/撤力超时仍进入保护。`;
    }
    if (value.phase === "inactive" && value.fault?.startsWith("arm_rejected_")) {
      return `上次准备已拒绝：${preparationReasonLabel(value.fault.slice("arm_rejected_".length))}。当前准备门：${preparationReasonLabel(debug.preparation_gate_reason)}；条件满足后需重新点击添加驱动力，不会自动授权。`;
    }
    if (debug.reason === "zero_confirmed_manual_rearm_required") {
      return "本次尖峰后已确认零助力与小范围位置变化；不是实测速度安全证明。旧驱动力已清空，必须重新点击添加驱动力并重新设置目标；不会自动恢复。";
    }
    if (value.phase === "inactive") {
      return `当前选择仅重力补偿模式，最终发送状态见回执。准备门：${preparationReasonLabel(debug.preparation_gate_reason)}；点击添加驱动力只完成零助力准备，仍需另行设置目标。`;
    }
  }
  if (value.controller_mode === "staged_assist") {
    if (value.phase === "inactive" && value.fault?.startsWith("breakaway_")) {
      return `本轮起步预算已结束：${value.fault}。驱动力已撤销，当前只做重力补偿；需要重新静止准备、设置目标并单独授权，不会自动重复发力。`;
    }
    const messages = {
      inactive: "先让机械臂静止，再准备并捕获当前实测位置。",
      arm_requested: "等待 owner 检查六轴静止并记录锚点。",
      armed: "已准备。设置目标相对位移后，先感受 CUE；起步需另行授权。",
      passive: "目标已更新；owner 正撤销旧意图并重置阶段。",
      cue: value.reason === "awaiting_breakaway_authorization"
        ? "短时方向提示已结束；仍在等待你的起步授权。确认方向安全后可随时授权一次。"
        : value.reason === "breakaway_authorization_rejected_not_stationary"
          ? "关节尚未静止，起步授权未生效；等速度回到门限内后再授权。"
          : value.reason === "breakaway_authorization_rejected_small_target_error"
            ? "目标剩余误差太小，起步授权未生效；设置更明确的相对目标后重试。"
        : "方向提示阶段：确认方向无误后，才可授权一次 BREAKAWAY。",
      breakaway: "有界起步脉冲正在运行；若观察到运动，控制器会立即撤掉脉冲。",
      releasing: "撤力请求已提交；保留驱动保护和必要阻尼，等待最终命令撤力确认与停稳。此时尚非纯重力模式。",
      drop: "起步运动已确认，起步脉冲已归零。",
      track: "低增益目标跟随阶段。",
      brake: "旧目标已作废；仅用受限速度反馈制动，停稳后捕获实际位置。只做重力补偿会撤销制动。",
      soft_hold: "正在小力保持实际停点，不返回启动位置或旧目标。只做重力补偿会撤销保持。",
      settle: "目标附近稳定阶段；积分受限并会泄漏。",
      human_override: "人工覆盖优先；旧目标不会自动恢复。",
      authority_exhausted: value.reason === "breakaway_duration_exhausted_before_impulse_budget"
        ? `起步时长先耗尽、冲量预算未用完；本次助力已归零。修改速率需重启 owner；新目标至少改变 ${number(value.limits.new_intent_delta_rad, 3)} rad 才能重试。`
        : value.reason === "breakaway_impulse_budget_exhausted"
          ? `起步冲量上限已用完；本次助力已归零。需设置至少改变 ${number(value.limits.new_intent_delta_rad, 3)} rad 的新目标再重新授权。`
          : `附加驱动力已撤销：${value.reason || "权限耗尽"}。检查原因后设置新目标，不会自动重试。`,
    };
    return messages[value.phase] || "等待 staged owner 状态。";
  }
  if (value.phase === "inactive") return "当前选择仅重力补偿模式，最终发送状态见回执。添加驱动力前，请让六轴静止。";
  if (value.phase === "armed") return "可逐步调节驱动力；查看发送确认，再观察关节运动。";
  return "等待 owner 确认静止状态。";
}

function post(path, body, hintId = "hint", fetchOptions = {}) {
  const send = async () => {
    try {
      const response = await fetch(path, {
        method: "POST",
        headers: {"Content-Type": "application/json", "X-B601-Console-Token": token},
        body: JSON.stringify(body),
        ...fetchOptions,
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
      if (path === "/api/stop") {
        el(hintId).textContent = "停机请求已接收，尚非电机执行回执；请观察 owner 终端状态并保持物理急停可用。";
      } else if (body.action !== "heartbeat") {
        el(hintId).textContent = `${ACTION_NAMES[body.action] || body.action}请求已接收；请查看发送确认和实测位移。`;
      }
    } catch (error) {
      if (body.action !== "heartbeat") el(hintId).textContent = `未接收：${error.message}`;
      if (body.action === "select_joint" && state) el("joint-select").value = state.joint_name;
    }
  };
  // Shutdown must not wait behind a stalled adjustment or heartbeat request.
  if (path === "/api/stop" || fetchOptions.keepalive) return send();
  if (body.action === "heartbeat" && heartbeatQueued) return requestQueue;
  if (body.action === "heartbeat") heartbeatQueued = true;
  requestQueue = requestQueue.then(send).finally(() => {
    if (body.action === "heartbeat") heartbeatQueued = false;
  });
  return requestQueue;
}

function intent(action, extra = {}, fetchOptions = {}) {
  // A cached armed status must not renew a lease after the operator withdrew it.
  if (action === "withdraw") heartbeatEnabled = false;
  if (action === "arm") heartbeatEnabled = true;
  return post("/api/intent", {action, ...extra}, "hint", fetchOptions);
}

function setTorque(value) {
  const maximum = state?.limits?.max_abs_assist_nm ?? 0;
  if (!Number.isFinite(value) || Math.abs(value) > maximum) {
    el("hint").textContent = `目标必须在 ±${number(maximum)} N·m 内。`;
    return;
  }
  el("target").value = number(value);
  intent("set", {torque_nm: value});
}

function setTargetOffset(value) {
  const maximum = state?.limits?.maximum_target_offset_rad ?? 0;
  if (!Number.isFinite(value) || Math.abs(value) > maximum) {
    el("hint").textContent = `目标偏移必须在 ±${number(maximum, 3)} rad 内。`;
    return;
  }
  el("target").value = number(value);
  intent("set_target", {target_offset_rad: value});
}

function adjustTarget(direction) {
  const step = Number(el("step").value);
  if (!Number.isFinite(step) || step <= 0) {
    el("hint").textContent = "调整步长必须为正数。";
    return;
  }
  if (isStaged()) {
    setTargetOffset((state?.target_offset_rad ?? 0) + direction * step);
  } else {
    setTorque((state?.desired_assist_nm ?? 0) + direction * step);
  }
}

el("joint-select").onchange = () => intent("select_joint", {joint_name: el("joint-select").value});
el("arm").onclick = () => {
  const selection = state?.mode_selection;
  const fields = selection ? {operation_mode: selection.selected} : {};
  if (selection?.selected === "supported_cycle") {
    const angle = Number(el("cycle-angle").value);
    if (!Number.isFinite(angle) || angle <= 0 || angle > selection.maximum_cycle_angle_deg) {
      el("cycle-angle-note").textContent = `抬升角度须大于 0° 且不超过 ${selection.maximum_cycle_angle_deg}°；尚未提交驱动。`;
      return;
    }
    fields.cycle_angle_deg = angle;
  }
  return intent("arm", fields);
};
el("mode-fixed-position").onclick = () => intent("select_mode", {operation_mode: "fixed_position"});
el("mode-supported-cycle").onclick = () => intent("select_mode", {operation_mode: "supported_cycle"});
el("mode-workspace").onclick = () => intent("select_mode", {operation_mode: "workspace"});
el("authorize-breakaway").onclick = () => intent("authorize_breakaway");
el("stop-and-hold").onclick = () => intent("stop_and_hold");
el("zero").onclick = () => {
  el("target").value = "0";
  intent("zero");
};
el("withdraw").onclick = () => {
  el("target").value = "0";
  intent("withdraw", {source: "operator_button"});
};
el("apply").onclick = () => {
  const target = Number(el("target").value);
  if (isStaged()) setTargetOffset(target);
  else setTorque(target);
};
el("minus").onclick = () => adjustTarget(-1);
el("plus").onclick = () => adjustTarget(1);
el("protective-hold").onclick = () => post("/api/stop", {action: "protective_hold"}, "stop-hint");
el("full-disable").onclick = () => {
  const cycleReady = state?.automatic_calibration?.cycle?.phase === "complete"
    && state.automatic_calibration.cycle.disable_confirmation_available === true
    && state.automatic_calibration.cycle.zero_assist_base_damping_confirmed === true;
  const message = cycleReady
    ? "自动流程已回到支撑原位并确认零助力/基础阻尼。请先确认机械臂已由可靠支撑承托；继续将请求 J1–J6 全失能。"
    : "确认请求 J1–J6 全失能？机械臂可能下坠。物理急停仍应优先使用。";
  if (window.confirm(message)) {
    post("/api/stop", {action: cycleReady ? "confirm_cycle_full_disable" : "full_disable"}, "stop-hint");
  }
};

document.addEventListener("visibilitychange", () => {
  if (!heartbeatEnabled) return;
  const visibilityIntent = pageVisibilityIntent(document.hidden, isSupportedCycle());
  if (!visibilityIntent) return;
  const {action, ...fields} = visibilityIntent;
  intent(action, fields);
  if (!document.hidden && isSupportedCycle()) intent("heartbeat");
});
// Closing/reloading remains an explicit withdrawal; keepalive lets the owner
// receive it even while the browser tears down the page.
window.addEventListener("pagehide", () => {
  if (heartbeatEnabled) {
    intent("withdraw", {source: "page_unload"}, {keepalive: true});
  }
});
setInterval(() => {
  const armed = isAutomatic() ? state.automatic_calibration.active || state.automatic_calibration.start_requested
    : isStaged() ? state?.operator_armed : state?.phase === "armed";
  if (heartbeatEnabled && (isSupportedCycle() || !document.hidden) && armed) intent("heartbeat");
}, 100);

function plot(history) {
  const canvas = el("chart");
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#131d27";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "#334353";
  context.lineWidth = 1;
  for (let index = 0; index < 5; index++) {
    const y = 20 + index * (height - 40) / 4;
    context.beginPath();
    context.moveTo(40, y);
    context.lineTo(width - 14, y);
    context.stroke();
  }
  if (!history?.length) return;
  const staged = isStaged();
  const torqueMaximum = Math.max(0.01, state.limits.max_abs_assist_nm);
  const offsetMaximum = Math.max(0.001, state.limits.maximum_target_offset_rad || 0.087266);
  const series = staged
    ? [
        ["target_offset_rad", "#6ab1ff", value => value / offsetMaximum],
        ["overlay_after_profile_nm", "#ffbc72", value => value / torqueMaximum],
        ["displacement_rad", "#7bd4a1", value => value / offsetMaximum],
      ]
    : [
        ["desired_assist_nm", "#6ab1ff", value => value / torqueMaximum],
        ["overlay_after_profile_nm", "#ffbc72", value => value / torqueMaximum],
        ["displacement_rad", "#7bd4a1", value => value === null ? NaN : value * 180 / Math.PI / 5],
      ];
  for (const [key, color, scale] of series) {
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.beginPath();
    let started = false;
    history.forEach((item, index) => {
      const raw = staged && key === "overlay_after_profile_nm"
        ? (item.overlay_after_profile_nm ?? item.effective_assist_nm)
        : item[key];
      const value = scale(raw);
      if (!Number.isFinite(value)) {
        started = false;
        return;
      }
      const y = height / 2 - value * (height * 0.38);
      const x = 40 + index * (width - 54) / Math.max(1, history.length - 1);
      if (started) context.lineTo(x, y);
      else context.moveTo(x, y);
      started = true;
    });
    context.stroke();
  }
}

function lastOperatorEvent(events) {
  return [...(events || [])].reverse().find(event => [
    "arm", "set", "set_target", "zero", "withdraw", "select_joint", "authorize_breakaway", "stop_and_hold",
  ].includes(event.action));
}

function commandState(value) {
  const action = lastOperatorEvent(value.events);
  const dispatch = value.controller_mode === "staged_assist" ? value.last_dispatch : freshJointDispatch(value);
  if (value.controller_mode === "staged_assist") {
    if (!action) return "尚未提交 staged 目标；设置目标后会显示阶段、起步授权和 owner 发送回执。";
    if (action.action === "stop_and_hold") {
      if (!dispatch || dispatch.feedback_timestamp_sec < action.time_sec) return "停止动作请求已接收，等待 owner 周期确认；尚未确认停稳。";
      return value.phase === "soft_hold"
        ? `owner 已捕获实际停点 ${number(value.motion_control?.hold_position_rad, 5)} rad；小力保持已启用，不代表物理锁止。`
        : `owner 已处理停止请求；当前阶段 ${value.phase}，原因 ${value.reason || "—"}。`;
    }
    if (action.action === "set_target") {
      if (!dispatch || dispatch.feedback_timestamp_sec < action.time_sec || dispatch.joint_name !== value.joint_name) {
        return `目标偏移 ${number(value.target_offset_rad, 4)} rad 已接收；等待 owner 周期确认。当前阶段：${value.phase}。`;
      }
      return `目标 ${number(value.target_offset_rad, 4)} rad → 阶段 ${value.phase}；owner 已完整发送，附加力矩 ${number(dispatch.overlay_after_profile_nm)} N·m。`;
    }
    if (action.action === "authorize_breakaway") {
      const phase = value.phase || value.stage_phase;
      if (phase === "authority_exhausted") {
        const ending = value.reason === "breakaway_duration_exhausted_before_impulse_budget"
          ? "达到起步时长上限时，冲量预算尚未用完"
          : value.reason === "breakaway_impulse_budget_exhausted"
            ? "起步冲量预算已用完"
            : (value.reason || "起步预算已耗尽");
        return `owner 已消费本次起步授权；脉冲结束：${ending}。已用冲量 ${number(value.breakaway_impulse_nm_sec, 4)} / ${number(value.limits?.breakaway_max_impulse_nm_sec, 4)} N·m·s，条件化软件命令峰值差 ${number(value.breakaway_peak_conditioned_delta_nm, 3)} N·m；不是电机实测。若需重试，目标至少改变 ${number(value.limits?.new_intent_delta_rad, 3)} rad。`;
      }
      if (STAGED_CONSUMED_PHASES.has(phase)) {
        return `owner 已消费本次起步授权；当前阶段 ${phase}，原因：${value.reason || "—"}。软件请求峰值 ${number(value.breakaway_peak_overlay_nm, 3)} N·m，条件化命令峰值差 ${number(value.breakaway_peak_conditioned_delta_nm, 3)} N·m（非电机实测）。`;
      }
      if (phase === "cue" && value.breakaway_authorized) {
        return `起步授权请求已接收；owner 当前仍在 CUE 阶段，等待 owner 周期推进。序号 ${value.latest_policy_sequence ?? "—"}。`;
      }
      return `起步授权请求尚未由 owner 确认；当前阶段 ${phase || "—"}，门控：${value.breakaway_gate_reason || "—"}。`;
    }
    if (action.action === "arm") return value.operator_armed ? "静止检查通过；捕获了实测位置，可设置相对目标。" : "准备请求已接收；等待 owner 检查六轴静止。";
    return `${ACTION_NAMES[action.action] || action.action}请求已接收；附加驱动力正在归零或撤销。`;
  }
  if (!action) return "尚未提交操作；最终命令是否归零请查看发送回执。允许驱动并提交目标后，这里显示发送确认。";
  if (action.action === "set" || action.action === "zero") {
    if (!dispatch || dispatch.feedback_timestamp_sec < action.time_sec || dispatch.joint_name !== value.joint_name) {
      return `GUI 已接收 ${number(action.requested_nm)} N·m；等待 owner 完整发送回执。`;
    }
    return `目标已接收 → owner 已完整发送；profile 附加命令 ${number(dispatch.overlay_after_profile_nm)} N·m，最终命令相对重力基线差 ${number(finalCommandDelta(dispatch))} N·m（不是实测力矩）。`;
  }
  if (action.action === "arm") {
    if (value.quantized_velocity_debug) {
      if (value.phase === "armed") return finalZeroConfirmed(dispatch)
        ? "静止检查已通过，最终六轴零助力命令已确认；现在可以设置目标。"
        : "静止检查已通过；最终零助力发送尚未确认，请查看最终命令差。";
      if (value.phase === "arm_requested") return "准备请求已提交，等待 owner 下一周期判定。";
      if (value.phase === "inactive" && value.fault?.startsWith("arm_rejected_")) {
        return `准备已拒绝：${preparationReasonLabel(value.fault.slice("arm_rejected_".length))}；不会继续等待或自动授权，请查看当前准备门后重新点击。`;
      }
      return "本次准备已撤销或停止；不会自动授权，请查看 owner 当前状态。";
    }
    return value.phase === "armed" ? "静止检查已通过；现在可以设置驱动力。" : "添加驱动力请求已接收；等待六轴静止检查。";
  }
  if (action.action === "select_joint") return `${value.joint_name} 已选定；添加驱动力前需再次检查静止。`;
  if (action.action === "withdraw" && value.phase === "inactive" && dispatch
      && dispatch.feedback_timestamp_sec >= action.time_sec && finalZeroConfirmed(dispatch)) {
    return "撤力已由 owner 确认：附加驱动力为零，当前只做重力补偿。";
  }
  return `${ACTION_NAMES[action.action] || action.action}请求已接收；附加驱动力正在撤回。`;
}

async function refresh() {
  if (refreshInFlight) return;
  refreshInFlight = true;
  try {
    const response = await fetch("/api/status?view=operator", {cache: "no-store"});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state = await response.json();
    stopControlsAvailable = state.stop_controls_available === true;
    if (isSupportedCycle() && ["complete", "cancelled", "failed"].includes(state.automatic_calibration?.cycle?.phase)) {
      heartbeatEnabled = false;
    }
    const staged = isStaged();
    el("connection").textContent = state.feedback_age_sec === null
      ? "等待反馈"
      : state.feedback_age_sec > 0.25 ? `反馈延迟 ${number(state.feedback_age_sec, 2)} s` : "本地连接正常";
    const phaseNames = {
      inactive: "仅重力补偿", arm_requested: "检查静止状态", armed: "可调节驱动力",
      velocity_check: "驱动力归零与速度确认",
      onset_release: "起动采集：撤力与停稳确认",
      fault: "保护已触发", cue: "方向提示 CUE", breakaway: "静摩擦起步 BREAKAWAY",
      releasing: "撤力与制动中",
      drop: "起步撤力 DROP", track: "低增益跟踪 TRACK", settle: "目标稳定 SETTLE",
      brake: "正在制动 BRAKE", soft_hold: "保持实际停点 SOFT HOLD",
      human_override: "人工覆盖", authority_exhausted: "附加驱动已撤销",
    };
    el("phase").textContent = phaseNames[state.phase] || state.phase;
    if (!staged) el("phase").textContent = directDriveLabel(state);
    el("fault").textContent = state.fault || "";
    const jointSelect = el("joint-select");
    for (const option of jointSelect.options) option.disabled = Boolean(state.motion_control_enabled || state.quantized_velocity_debug) && option.value !== "joint3";
    if (document.activeElement !== jointSelect) jointSelect.value = state.joint_name;
    jointSelect.disabled = state.phase !== "inactive" || Math.abs(state.effective_assist_nm) > 1e-10;
    const jointIndex = Number(state.joint_name.replace("joint", "")) - 1;
    const selectedAngle = state.qpos_rad?.[jointIndex];
    el("qpos").textContent = `${number(Number.isFinite(selectedAngle) ? selectedAngle * 180 / Math.PI : null, 3)}° / ${number(selectedAngle, 5)} rad`;
    const selectedGuard = state.automatic_calibration?.stage === "onset" && jointIndex === 2
      ? state.automatic_calibration.support_damping?.selected_joint_velocity_limit_rad_s ?? state.limits.max_joint_velocity_rad_s
      : state.limits.max_joint_velocity_rad_s;
    el("qvel").textContent = `${number(state.qvel_fd_rad_s?.[jointIndex], 4)} / ${number(selectedGuard, 3)} rad/s`;
    const displacement = state.displacement_rad ?? state.last_observed_displacement_rad;
    el("displacement").textContent = Number.isFinite(displacement)
      ? `${number(displacement, 5)} rad / ${number(displacement * 180 / Math.PI, 3)}°`
      : "未准备";
    const drifts = state.support_drift_rad ?? state.last_observed_support_drift_rad;
    const largest = drifts && Object.entries(drifts).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))[0];
    el("support-drift").textContent = largest
      ? `${largest[0]} ${number(largest[1], 5)} rad / 限 ${number(state.limits.max_support_drift_rad, 5)}`
      : "未准备";
    el("desired-label").textContent = staged ? "目标相对位移" : "目标驱动力";
    el("desired").textContent = staged
      ? `${number(state.target_offset_rad, 4)} rad`
      : number(state.desired_assist_nm);
    el("effective").textContent = `${number(state.effective_assist_nm)} N·m`;
    const displayDispatch = staged ? state.last_dispatch : freshJointDispatch(state);
    el("dispatched").textContent = `${number(displayDispatch?.overlay_after_profile_nm)} N·m`;
    el("receipt").textContent = displayDispatch
      ? `MIT #${displayDispatch.receipt_sequence} / ${number(state.dispatch_age_sec, 2)} s`
      : "尚无当前关节的新鲜回执";
    el("gravity").textContent = `${number(displayDispatch?.gravity_nm)} N·m`;
    el("total").textContent = `${number(displayDispatch?.conditioned_total_nm)} N·m`;
    el("conditioned-delta").textContent = `${number(finalCommandDelta(displayDispatch), 4)} N·m`;
    el("command-state").textContent = commandState(state);
    el("target-label").textContent = staged ? "目标相对位移 · rad" : "目标驱动力 · N·m";
    el("apply").textContent = staged ? "提交位移目标（先提示）" : "提交力矩目标";
    el("step-label").textContent = staged ? "位移调整步长 · rad" : "力矩调整步长 · N·m";
    const targetLimit = staged ? state.limits.maximum_target_offset_rad : state.limits.max_abs_assist_nm;
    el("target").min = -targetLimit;
    el("target").max = targetLimit;
    el("step").max = targetLimit;
    el("authorize-breakaway").hidden = !staged;
    el("authorize-breakaway").disabled = !staged || !state.breakaway_eligible;
    el("stop-and-hold").hidden = !state.motion_control_enabled;
    el("stop-and-hold").disabled = !state.operator_armed || state.target_offset_rad === null || state.target_update_pending || ["fault", "releasing"].includes(state.phase);
    el("motion-state").hidden = !state.motion_control_enabled;
    const stopMessage = stopCaptureMessage(state);
    el("stop-capture-state").hidden = stopMessage === null;
    el("stop-capture-state").textContent = stopMessage ?? "";
    if (state.motion_control_enabled) {
      const motion = state.motion_control;
      el("motion-state").textContent = motion
        ? `运动参考 ${number(motion.q_reference_rad, 5)} rad / ${number(motion.qvel_reference_rad_s, 4)} rad/s；目标误差 ${number(motion.target_error_rad, 5)} rad；控制窗口速度 ${number(motion.control_qvel_rad_s, 4)} rad/s；位置项 ${number(motion.position_torque_nm)} / 速度项 ${number(motion.velocity_torque_nm)} N·m；制动触发前参考误差 ${number(motion.brake_trigger_lag_rad, 5)} rad；实际停点 ${number(motion.hold_position_rad, 5)} rad（停点保持不代表原目标已到位）。只做重力补偿会撤销制动和保持。`
        : "J3 交互模式已启用。先准备、设置目标并授权起步；停止动作会先制动，停稳后捕获实测位置。没有自动人体接触检测。";
      if (motion?.reference_position_reserve_rad != null && motion.reference_speed_scale === 0) {
        el("motion-state").textContent += " 参考正在减速暂停，给机械臂追赶当前目标；TRACK 维持助力与原预算继续生效。";
      }
      if (motion?.motion_evidence_block_reason) {
        el("motion-state").textContent += ` 运动保护具体原因：${motion.motion_evidence_block_reason}。`;
      }
      if (motion?.tracking_recent_position_span_rad != null) {
        el("motion-state").textContent += ` 近期解析速度 ${number(motion.tracking_recent_velocity_rad_s, 4)} rad/s；近期位置跨度 ${number(motion.tracking_recent_position_span_rad, 6)} rad。半窗口仅提前触发软撤力，单计数抖动不触发；起动、停稳和原始硬速度保护仍按原判据。`;
      }
      if (motion?.speed_governor_active) {
        el("motion-state").textContent += ` 调速告警（非整批故障）：先制动，等待六轴停稳与最终归零回执后受限恢复同一目标。已恢复 ${motion.speed_governor_resume_count} 次${motion.speed_governor_resume_block_reason ? `；等待原因：${motion.speed_governor_resume_block_reason}` : ""}。时限与命令预算不重置；硬保护仍优先。`;
      } else if (motion?.speed_governor_resume_count > 0) {
        el("motion-state").textContent += ` 降速后已受限恢复 ${motion.speed_governor_resume_count} 次；没有重新授权起步脉冲。`;
      }
      const dispatchedKd = state.last_dispatch?.driver_kd_nm_s_per_rad;
      el("motion-state").textContent += ` 六轴阶段请求 Kd：${(state.drive_driver_kd_nm_s_per_rad ?? []).map(v => number(v, 3)).join(" / ")}；助力环最后完整下发 Kd（历史回执，保护保持不更新）：${dispatchedKd ? dispatchedKd.map(v => number(v, 3)).join(" / ") : "尚无回执"} N·m·s/rad。支撑轴软判据/原始差分速度最大值：${number(state.support_max_velocity_rad_s, 4)} / ${number(state.support_raw_max_velocity_rad_s, 4)} rad/s；窗口跨度 ${number(state.velocity_window_span_sec, 3)} s。此阻尼仅在添加驱动力后生效；只做重力补偿恢复标定基础 Kd。命令回执不是实测电机力矩。`;
      if (state.velocity_window?.enabled) {
        el("motion-state").textContent += ` 准备/授权使用因果窗口与窗口位移；窗口${state.velocity_window.ready ? "已就绪" : "尚未就绪，请等待完整反馈窗口"}。原始单帧速度硬保护不变。`;
        if (state.phase === "breakaway" && state.owner_ready && state.operator_armed
            && Number.isFinite(state.feedback_age_sec) && state.feedback_age_sec >= 0 && state.feedback_age_sec <= 0.15
            && state.limits.motion_onset_confirm_displacement_rad != null) {
          if (state.breakaway_probe_settings) {
            const probe = state.breakaway_probe;
            el("motion-state").textContent += ` 定力观察：${probe?.stage ?? "缓升等待微动"}；保持命令 ${number(probe?.hold_nm, 4)} N·m；实际定力派发${probe?.first_hold_completed_at_sec == null ? "尚未确认" : "已确认"}；派发后的不同位置进展 ${probe?.distinct_progress_samples ?? 0} / ${state.motion_onset_required_samples ?? "—"}；新增进展 ${number(probe?.maximum_progress_rad, 6)} rad。观察 ${number(state.breakaway_probe_settings.hold_sec, 2)} s，无持续进展则在原预算内恢复缓升；重复位置不算新进展，越速/位移/预算仍撤力。此处不是测力。`;
          } else {
            el("motion-state").textContent += ` 起动确认 ${state.motion_onset_confirm_count ?? 0} / ${state.motion_onset_required_samples ?? "—"} 个连续反馈：同向净位移至少 ${number(state.limits.motion_onset_confirm_displacement_rad, 5)} rad，窗口运动方向不能相反。${state.reason === "motion_onset_candidate_pulse_paused" ? "疑似起动：大起步脉冲已暂停为零，尚未确认运动，不进入跟踪。" : "在峰值、时长和冲量预算内递增；疑似起动立即撤脉冲。"}`;
          }
        }
      }
      if (state.phase === "releasing") {
        const delta = state.last_dispatch?.conditioned_total_nm - state.last_dispatch?.gravity_nm;
        el("motion-state").textContent += ` 最后完整软件派发相对重力基线的命令差 ${number(delta, 4)} N·m；请求归零不等于最终发送归零。正在${state.release_restoring_base_damping ? "确认基础阻尼恢复" : "受限撤力/制动与停稳检查"}。没有电机输出力传感。`;
        el("motion-state").textContent += ` 含最终发送尾量的累计软件命令暴露 ${number(state.conditioned_command_impulse_nm_sec, 4)} / ${number(state.limits.breakaway_max_impulse_nm_sec, 4)} N·m·s；变化的重力基线可能使撤力预留失效，越界会报告故障。这不是实测力矩冲量。`;
      }
    }
    el("staged-details").hidden = !staged && !state.onset_capture;
    el("staged-details-box").hidden = !staged && !state.onset_capture;
    el("staged-details-box").querySelector("summary").textContent = state.onset_capture ? "起动界限采集记录" : "分阶段驱动详情";
    const envelope = state.breakaway_command_envelope;
    el("breakaway-envelope").hidden = !staged || !envelope;
    if (staged && envelope) {
      el("breakaway-envelope").textContent = `起步配置峰值 ${number(envelope.nominal_peak_nm)} N·m；无运动、满权限时的理论命令上界 ${number(envelope.optimistic_command_peak_ceiling_nm)} N·m；${envelope.nominal_peak_within_optimistic_ceiling ? "参数组未发现理论峰值冲突" : "时长或冲量限制使配置峰值不可达"}。这不是实际下发值，不证明能克服静摩擦或安全到位。`;
    }
    if (staged) {
      const authorizationLabel = state.breakaway_authorized
        ? "本次授权已提交"
        : STAGED_CONSUMED_PHASES.has(state.phase) ? "本次已消费" : "未授权";
      el("staged-details").textContent =
        `阶段原因：${state.reason || "—"}；当前 policy 权限 ${number(state.policy_authority, 2)}（爬升率 ${number(state.limits.authority_rise_per_sec, 2)}/s）；目标周期确认：${state.target_update_pending ? "等待 owner" : "已确认"}；授权门控：${state.breakaway_eligible ? "可授权" : (state.breakaway_gate_reason || "—")}；误差至少 ${number(state.limits.minimum_breakaway_error_rad, 3)} rad、速度不高于 ${number(state.limits.stuck_max_velocity_rad_s, 3)} rad/s；相邻重试目标至少改变 ${number(state.limits.new_intent_delta_rad, 3)} rad；起步授权：${authorizationLabel}；起步冲量 ${number(state.breakaway_impulse_nm_sec, 4)} / ${number(state.limits.breakaway_max_impulse_nm_sec, 4)} N·m·s；起步时长上限 ${number(state.limits.breakaway_max_duration_sec, 2)} s；起步独立上升率 ${number(state.limits.breakaway_rise_slew_nm_s, 3)} N·m/s；软件请求 overlay 峰值 ${number(state.breakaway_peak_overlay_nm, 3)} N·m；owner 回执中的 profile overlay 峰值 ${number(state.breakaway_peak_dispatched_nm, 3)} N·m；最终条件化总力矩相对重力基线的峰值差 ${number(state.breakaway_peak_conditioned_delta_nm, 3)} N·m（均为软件命令诊断，不是电机实测）；授权序号 ${state.latest_policy_sequence ?? "—"}。`;
      el("chart-legend").innerHTML = "<span class=\"blue\">● 目标偏移</span><span class=\"orange\">● 已发送助力</span><span class=\"green\">● 实测位移</span>";
    } else {
      const capture = state.onset_capture;
      el("staged-details").textContent = onsetCaptureDetails(capture);
      el("chart-legend").innerHTML = "<span class=\"blue\">● 目标驱动力</span><span class=\"orange\">● 已接受发送的驱动力</span><span class=\"green\">● 关节位移（5°/满量程）</span>";
    }
    el("arm").disabled = !state.owner_ready || state.phase !== "inactive" || (state.velocity_window?.enabled && !state.velocity_window.ready)
      || (state.onset_capture?.probe && state.onset_capture.probe.human_intervention !== false)
      || (state.quantized_velocity_debug && state.quantized_velocity_debug.preparation_gate_reason !== "ready")
      || Math.abs(state.effective_assist_nm) > 1e-10
      || state.feedback_age_sec === null || state.feedback_age_sec > 0.15;
    const driveRequested = state.operator_armed || state.phase === "armed" || state.phase === "arm_requested";
    el("arm").setAttribute("aria-pressed", String(Boolean(driveRequested)));
    el("withdraw").setAttribute("aria-pressed", String(!driveRequested));
    const targetEnabled = staged ? state.operator_armed && state.phase !== "releasing"
      && (state.limits.motion_onset_confirm_displacement_rad == null || state.target_offset_rad == null) : state.phase === "armed";
    for (const id of ["minus", "plus", "target", "apply"]) el(id).disabled = !targetEnabled;
    el("zero").disabled = staged ? !state.operator_armed : state.phase !== "armed";
    el("protective-hold").disabled = !state.stop_controls_available || state.phase === "fault";
    el("full-disable").disabled = !state.stop_controls_available;
    el("gate").textContent = gateMessage(state);
    el("terminal-statuses").textContent = state.terminal_statuses?.length
      ? state.terminal_statuses.join("\n") : "等待 owner 状态…";
    el("log-path").textContent = state.runtime_log_path
      ? `完整运行日志：${state.runtime_log_path}` : "";
    renderAutomatic();
    plot(state.history);
    document.dispatchEvent(new CustomEvent("b601-status", {detail: state}));
  } catch (_error) {
    el("cycle-disable-hint").hidden = true;
    el("full-disable").textContent = "软件急停：J1–J6 全失能";
    el("stop-capture-state").hidden = true;
    el("stop-capture-state").textContent = "";
    document.dispatchEvent(new CustomEvent("b601-status", {detail: null}));
    el("connection").textContent = "连接中断；助力将超时撤销";
    el("phase").textContent = "未连接";
    el("gate").textContent = "无法取得 owner 状态；请检查终端。";
    for (const id of ["minus", "plus", "target", "apply", "arm", "joint-select", "authorize-breakaway", "stop-and-hold", "zero", "mode-fixed-position", "mode-supported-cycle", "mode-workspace", "cycle-angle"]) {
      el(id).disabled = true;
    }
    el("protective-hold").disabled = !stopControlsAvailable;
    el("full-disable").disabled = !stopControlsAvailable;
  } finally {
    refreshInFlight = false;
  }
}

setInterval(refresh, 200);
refresh();

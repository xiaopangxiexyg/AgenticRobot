"""Pure operator-facing projection for the Sim-HIL control GUI.

This module turns existing application and acceptance snapshots into a small
workflow view.  It is deliberately non-authoritative: it neither decides
readiness nor changes which intents the acceptance controller will admit.
"""

from __future__ import annotations

from typing import Any, Mapping


SIM_HIL_OPERATOR_GUIDANCE_SCHEMA_V1 = (
    "dexteleop.control_station.sim_hil_operator_guidance.v1"
)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _intent_action(
    intent_kind: str,
    label: str,
    *,
    destructive: bool = False,
    control_mode: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "interaction": "intent",
        "intent_kind": intent_kind,
        "label": label,
        "destructive": bool(destructive),
    }
    if control_mode is not None:
        result["control_mode"] = control_mode
    return result


def _focus_action(target_id: str, label: str) -> dict[str, object]:
    return {
        "interaction": "focus",
        "target_id": target_id,
        "label": label,
        "destructive": False,
    }


def _classify_error(raw_error: str) -> tuple[str, str, str]:
    """Return a best-effort presentation summary while retaining raw_error."""

    lowered = raw_error.lower()
    if "active_forbidden_contact" in lowered:
        return (
            "mujoco_forbidden_contact",
            "MuJoCo 禁碰接触保护已触发",
            "MuJoCo 已观测到超过硬穿透限值的禁碰接触；当前 episode "
            "只能 Reset。",
        )
    if "collision_margin_low" in lowered:
        return (
            "lower_collision_margin_hold",
            "规划器安全间隙保护已触发",
            "下游规划器的代理模型低于预防间隙；这不表示 MuJoCo 已经"
            "发生物理接触。当前 episode 只能 Reset，不能 rearm 或"
            "切换来源继续。",
        )
    if "target_private_resolution_failure" in lowered:
        return (
            "lower_target_resolution_hold",
            "执行目标未能解析，已保持输出",
            "组合结果已经送达下游，但 IK、限位或目标解析未通过；"
            "在执行诊断中查看具体拒绝原因。",
        )
    if "rotation is not orthonormal" in lowered or "pose rotation" in lowered:
        return (
            "human_pose_invalid",
            "VIVE 位姿未通过输入校验",
            "检查当前来源映射与坐标配置；原始设备数据仍可在诊断区查看。",
        )
    if "human input acceptance timed out" in lowered:
        return (
            "human_input_timeout",
            "等待人类输入超时",
            "确认被选中的 VIVE 来源和夹爪输入正在更新，然后重新启动。",
        )
    if (
        "requires start_allocation_runtime" in lowered
        or "does not use fixed-alpha" in lowered
        or "use set_allocation_mode" in lowered
    ):
        return (
            "profile_intent_mismatch",
            "控制界面与所选分配方案不一致",
            "刷新页面后使用动态分配区域；不要使用 fixed-alpha 模式按钮。",
        )
    if "controltrace" in lowered or "capture" in lowered or "writer" in lowered:
        return (
            "capture_fault",
            "记录链路发生故障",
            "输出会保持；先在诊断区确认写入故障和存储状态。",
        )
    if "cancel" in lowered or "interrupted" in lowered:
        return (
            "operation_cancelled",
            "操作已中断",
            "当前控制状态未被当作成功启动；可在条件就绪后重新操作。",
        )
    return (
        "operation_failed",
        "最近一次操作失败",
        "先查看原始错误和对应子系统状态，再选择恢复或重建。",
    )


def _execution_fault_context(
    status: Mapping[str, Any],
) -> dict[str, object] | None:
    """Preserve lower rejection facts that a generic gate reason may hide."""

    composition = _mapping(status.get("composition"))
    last_cycle = _mapping(composition.get("last_cycle"))
    execution = _mapping(last_cycle.get("execution"))
    failure = _mapping(execution.get("resolution_failure"))
    gate = _mapping(status.get("execution_gate"))
    contact = _mapping(status.get("sim_collision_contact"))
    unsafe_contact = bool(
        contact.get("available") and not contact.get("safe", True)
    )
    reason_code = failure.get("reason_code")
    message = failure.get("message")
    outcome_code = execution.get("target_completion_outcome_code")
    gate_reason = gate.get("reason")
    searchable = " ".join(
        str(value)
        for value in (reason_code, message, outcome_code, gate_reason)
        if value not in {None, ""}
    )
    if not searchable and not unsafe_contact:
        return None
    lowered = searchable.lower()
    category = (
        "collision"
        if unsafe_contact or "collision" in lowered
        else "target_resolution"
        if failure or outcome_code
        else "execution_gate"
    )
    diagnostics = dict(failure.get("diagnostics") or {})
    collision_summary = None
    collision_semantics = None
    if category == "collision":
        parts: list[str] = []
        if unsafe_contact:
            collision_semantics = "mujoco_forbidden_contact_hard_penetration"
            if contact.get("geom1") and contact.get("geom2"):
                parts.append(
                    f"实际接触 {contact['geom1']} ↔ {contact['geom2']}"
                )
            try:
                parts.append(
                    "穿透深度 "
                    f"{1000.0 * float(contact.get('penetration_depth_m') or 0.0):.3f} mm"
                )
            except (TypeError, ValueError):
                pass
        else:
            collision_semantics = (
                "planner_proxy_preventive_clearance_not_physical_contact"
            )
            geometry_1 = diagnostics.get("collision_geometry_1")
            geometry_2 = diagnostics.get("collision_geometry_2")
            distance = diagnostics.get("best_candidate_distance_m")
            threshold = diagnostics.get("collision_threshold_m")
            if geometry_1 and geometry_2:
                parts.append(f"代理对 {geometry_1} ↔ {geometry_2}")
            try:
                if distance is not None:
                    parts.append(
                        f"最近间隙 {1000.0 * float(distance):.3f} mm"
                    )
                if threshold is not None:
                    parts.append(
                        f"有效阈值 {1000.0 * float(threshold):.3f} mm"
                    )
            except (TypeError, ValueError):
                pass
        collision_summary = " · ".join(parts) or None
    return {
        "category": category,
        "collision_semantics": collision_semantics,
        "collision_summary": collision_summary,
        "reason_code": reason_code,
        "message": message,
        "outcome_code": outcome_code,
        "gate_status": gate.get("status"),
        "gate_reason": gate_reason,
        "diagnostics": diagnostics,
        "searchable_summary": (
            searchable
            or "mujoco_forbidden_contact_hard_penetration"
        ),
    }


def _human_demo_workflow(
    status: Mapping[str, Any],
    *,
    acceptance: Mapping[str, Any],
    setup: Mapping[str, Any],
    capture: Mapping[str, Any],
    recommended_action: Mapping[str, object] | None,
    phase: str,
) -> dict[str, object]:
    """Project one concise, non-authoritative demo collection journey."""

    started = bool(status.get("started", False))
    held = bool(status.get("held", False))
    episode_ended = bool(status.get("episode_ended", False))
    preview_ready = bool(acceptance.get("sim_preview_ready", False))
    capture_state = str(capture.get("status") or "ready")
    committed = capture_state == "committed"
    input_ready = bool(setup.get("input_ready", False))
    profile_ready = bool(setup.get("profile_ready", False))
    human_ready = bool(
        setup.get("human_runtime_launch_ready", False)
    )
    active_intent_kind = (
        None
        if acceptance.get("active_intent_kind") is None
        else str(acceptance.get("active_intent_kind"))
    )
    started_at_value = acceptance.get("formal_runtime_started_at")
    finalized_at_value = acceptance.get("formal_runtime_finalized_at")
    sampled_at_value = status.get("sampled_at")
    recording_elapsed_sec = None
    try:
        if started_at_value is not None:
            endpoint = (
                finalized_at_value
                if finalized_at_value is not None
                else sampled_at_value
            )
            if endpoint is not None:
                recording_elapsed_sec = max(
                    0.0,
                    float(endpoint) - float(started_at_value),
                )
    except (TypeError, ValueError):
        recording_elapsed_sec = None

    preview_state = "done" if preview_ready or started or committed else (
        "active" if active_intent_kind == "prepare_sim_preview" else "ready"
    )
    input_state = (
        "done"
        if started or committed
        else "active"
        if active_intent_kind == "start_runtime"
        else "ready"
        if preview_ready and human_ready
        else "blocked"
        if preview_ready
        else "waiting"
    )
    record_state = (
        "done"
        if committed
        else "fault"
        if capture_state == "faulted"
        else "active"
        if capture_state == "recording" or (started and not episode_ended)
        else "ready"
        if preview_ready and human_ready
        else "waiting"
    )
    save_state = (
        "done"
        if committed
        else "active"
        if active_intent_kind == "finalize_capture"
        else "ready"
        if capture_state == "recording"
        else "fault"
        if capture_state == "faulted"
        else "waiting"
    )
    operation_labels = {
        "prepare_sim_preview": "正在构造静态仿真并打开 viewer",
        "start_runtime": "正在接入 Human、采集 Home 并开始记录",
        "finalize_capture": "正在保持输出、封存并写入磁盘",
        "rearm": "正在重新检查并恢复遥操",
        "capture_runtime_home": "正在重新采集 Runtime Home",
        "rebuild_runtime": "正在关闭本次仿真并返回准备阶段",
        "stop_runtime": "正在退出",
    }
    if committed:
        state_label = "已保存"
    elif active_intent_kind is not None:
        state_label = operation_labels.get(active_intent_kind, "操作进行中")
    elif started and held:
        state_label = "记录保持中"
    elif capture_state == "recording":
        state_label = "遥操与记录中"
    elif preview_ready:
        state_label = "静态预览中"
    else:
        state_label = "等待打开仿真"

    return {
        "phase": str(phase),
        "state_label": state_label,
        "active_intent_kind": active_intent_kind,
        "operation_label": (
            None
            if active_intent_kind is None
            else operation_labels.get(active_intent_kind, "操作进行中")
        ),
        "primary_action": (
            None
            if recommended_action is None
            else dict(recommended_action)
        ),
        "steps": [
            {
                "id": "preview",
                "label": "打开静态仿真",
                "state": preview_state,
                "detail": "仅 viewer；不接设备、不控制、不记录",
            },
            {
                "id": "human_ready",
                "label": "确认 Human 输入",
                "state": input_state,
                "detail": (
                    "VIVE 与坐标配置已就绪；扳机在启动时检查"
                    if human_ready
                    else "等待 VIVE 与坐标配置"
                ),
            },
            {
                "id": "recording",
                "label": "遥操并记录",
                "state": record_state,
                "detail": "Human-only 控制与 ControlTrace 同时启动",
            },
            {
                "id": "saved",
                "label": "封存并保存",
                "state": save_state,
                "detail": "terminal HOLD 后执行 seal 与 durable commit",
            },
        ],
        "input": {
            "vive_ready": input_ready,
            "coordinate_profile_ready": profile_ready,
            "angle_trigger": (
                "connected"
                if started
                else "checked_when_recording_starts"
            ),
        },
        "recording": {
            "active": capture_state == "recording",
            "held": held,
            "elapsed_sec": recording_elapsed_sec,
        },
        "capture": {
            "status": capture_state,
            "commit_status": capture.get("commit_status"),
            "session_root": capture.get("session_root"),
            "message": capture.get("message"),
        },
    }


def build_sim_hil_operator_guidance(
    status: Mapping[str, Any],
) -> dict[str, object]:
    """Build one rebuildable, non-authoritative operator workflow snapshot."""

    acceptance = _mapping(status.get("acceptance"))
    setup = _mapping(acceptance.get("setup"))
    calibration = _mapping(setup.get("calibration"))
    startup = _mapping(status.get("startup"))
    capture = _mapping(acceptance.get("capture"))
    composition = _mapping(status.get("composition"))
    allocation_operations = _mapping(composition.get("allocation_operations"))
    last_cycle = _mapping(composition.get("last_cycle"))
    composition_step = _mapping(last_cycle.get("composition_step"))
    allocation_cycle = _mapping(composition_step.get("allocation"))
    effective_allocation_modes = {
        str(component_id): str(mode)
        for component_id, mode in _mapping(
            allocation_cycle.get("selected_mode_by_component")
        ).items()
    }

    started = bool(status.get("started", False))
    held = bool(status.get("held", False))
    episode_ended = bool(status.get("episode_ended", False))
    collision_terminal = bool(
        episode_ended
        and str(status.get("episode_end_reason") or "").startswith(
            "sim_collision_stop:"
        )
    )
    runtime_constructed = bool(startup.get("mujoco_constructed", False))
    restart_required = bool(startup.get("restart_required", False))
    shutdown_requested = bool(acceptance.get("shutdown_requested", False))
    exclusive_profile = bool(
        acceptance.get("exclusive_hitl_enabled", False)
    )
    # The exclusive profile reuses the streaming-allocation implementation,
    # but its operator contract is only Policy versus Human.  Treating that
    # implementation detail as generic allocation advertises the wrong start
    # intent and creates a duplicate control path in the GUI.
    allocation_profile = bool(
        acceptance.get("allocation_runtime_enabled", False)
    ) and not exclusive_profile
    human_demonstration = (
        str(acceptance.get("capture_profile") or "")
        == "human_demonstration"
    )
    sim_preview_ready = bool(
        acceptance.get("sim_preview_ready", False)
    )
    active_operation = bool(
        acceptance.get("active_intent_id")
        or int(acceptance.get("pending_intent_count") or 0) > 0
    )
    input_ready = bool(setup.get("input_ready", False))
    profile_ready = bool(setup.get("profile_ready", False))
    policy_launch_ready = bool(
        setup.get("policy_runtime_launch_ready", runtime_constructed)
    )
    human_launch_ready = bool(
        setup.get("human_runtime_launch_ready", runtime_constructed)
    )
    workflow_active = bool(calibration.get("workflow_active", False))
    calibration_stage = str(calibration.get("stage") or "")
    raw_error_value = acceptance.get("last_error") or status.get("last_error")
    raw_error = None if not raw_error_value else str(raw_error_value)
    execution_fault = _execution_fault_context(status)
    capture_fault = str(capture.get("status") or "") == "faulted"

    blockers: list[dict[str, str]] = []
    notices: list[str] = []
    if allocation_profile:
        if not input_ready:
            input_preflight = _mapping(calibration.get("input_preflight"))
            notices.append(
                str(
                    input_preflight.get("message")
                    or "Human 输入尚未就绪；可先以策略基线启动。"
                )
            )
        if not profile_ready:
            notices.append(
                "尚未选用有效的 VIVE 坐标配置；Human 模式暂不可用。"
            )
        if workflow_active:
            notices.append(
                "VIVE 标定进行中；不影响纯策略启动。"
            )
        if bool(setup.get("unsaved_calibration", False)):
            notices.append(
                "标定结果尚未保存；保存前 Human 模式不可用。"
            )
    if not human_demonstration and not policy_launch_ready:
        blockers.append(
            {
                "code": "policy_runtime_not_ready",
                "message": "策略运行时当前不能启动；请查看策略与启动诊断。",
            }
        )

    supported_signals = {
        str(item) for item in allocation_operations.get("supported_signals", ())
    }
    if (
        allocation_profile
        and "allocation_absolute_overlay_owner" not in supported_signals
    ):
        notices.append(
            "当前分配方案未声明人类绝对接管；Human 仅使用已声明的修正模式。"
        )
    if (
        not human_demonstration
        and not allocation_profile
        and policy_launch_ready
        and not human_launch_ready
    ):
        notices.append(
            "Policy 可以独立启动；Human 仍需 VIVE 输入与坐标配置。"
            if exclusive_profile
            else "纯 PPO 可以独立启动；遥操和融合仍需 VIVE 输入与坐标配置。"
        )
    if not started and not sim_preview_ready:
        notices.append("夹爪设备、策略和 MuJoCo 会在启动过程中接入并校验。")

    phase: str
    severity: str
    headline: str
    detail: str
    action: dict[str, object] | None
    issue_code: str | None = None

    if shutdown_requested:
        phase, severity = "closing", "warning"
        headline = "正在退出控制"
        detail = "控制线程已收到退出请求，请等待进程完成收尾。"
        action = None
    elif active_operation:
        phase, severity = "operation_pending", "warning"
        active_kind = str(acceptance.get("active_intent_kind") or "")
        operation_copy = {
            "prepare_sim_preview": (
                "正在打开静态仿真",
                "正在加载场景并创建 viewer；此时不会接入遥操或开始记录。",
            ),
            "start_runtime": (
                "正在开始遥操与记录",
                "正在接入 VIVE/扳机、检查输入并采集 Runtime Home；全部通过后才会运行。",
            ),
            "finalize_capture": (
                "正在封存并保存",
                "正在保持输出并完成 seal、durable flush 与 commit，请不要直接关闭进程。",
            ),
        }.get(active_kind)
        if human_demonstration and operation_copy is not None:
            headline, detail = operation_copy
        else:
            headline = "操作正在进行"
            detail = "启动、设备接纳或恢复可能需要等待；期间仍可中断当前操作。"
        action = _intent_action("cancel_operation", "中断当前操作")
    elif collision_terminal:
        # Collision termination dominates generic restart/fault projection.
        # It must never regress to rearm, Home capture, or source switching
        # merely because another status flag is present at the same time.
        phase, severity = "episode_ended", "error"
        actual_contact = bool(
            execution_fault
            and execution_fault.get("collision_semantics")
            == "mujoco_forbidden_contact_hard_penetration"
        )
        headline = (
            "MuJoCo 禁碰接触保护已触发"
            if actual_contact
            else "规划器安全间隙保护已触发"
        )
        detail = (
            "MuJoCo 已观测到超过硬穿透限值的禁碰接触。当前 episode "
            "只能 Reset，不能恢复或切换来源继续。"
            if actual_contact
            else (
                "代理模型的预防间隙不足；这不等于 MuJoCo 已发生物理接触。"
                "当前 episode 只能 Reset，不能恢复或切换来源继续。"
            )
        )
        action = _intent_action(
            "rebuild_runtime",
            "Reset 仿真",
            destructive=True,
        )
    elif restart_required:
        phase, severity = "restart_required", "error"
        headline = "当前运行时必须重建"
        detail = "现有运行时不能安全恢复；结束当前仿真并回到准备页。"
        action = (
            _intent_action(
                "rebuild_runtime",
                "重建运行时",
                destructive=True,
            )
            if runtime_constructed and not episode_ended
            else _intent_action("stop_runtime", "退出控制", destructive=True)
        )
    elif raw_error or execution_fault or capture_fault:
        phase, severity = "faulted", "error"
        classification_input = " ".join(
            value
            for value in (
                raw_error,
                None
                if execution_fault is None
                else str(execution_fault["searchable_summary"]),
                str(capture.get("message") or "") if capture_fault else None,
            )
            if value
        )
        issue_code, headline, detail = _classify_error(
            classification_input or "capture fault"
        )
        if capture_fault:
            action = _focus_action("operator-diagnostics", "查看记录故障")
        elif human_demonstration and sim_preview_ready and not started:
            headline = "Human 输入未通过，视角仍已保留"
            detail = (
                "检查 VIVE、坐标配置或角度扳机后可直接重试；静态 viewer 不会重建。"
            )
            action = _intent_action(
                "start_runtime",
                "重试开始遥操并记录",
                control_mode="human_teleop",
            )
        elif human_demonstration and not started:
            headline = "上次操作失败，可重新打开静态仿真"
            detail = (
                "当前仍未接入遥操或开始记录；检查提示后可安全重试静态预览。"
            )
            action = _intent_action(
                "prepare_sim_preview",
                "重新打开仿真并调整视角",
            )
        elif issue_code == "lower_collision_margin_hold" and started:
            headline = "规划器安全间隙保护已触发"
            detail = (
                "代理模型低于预防间隙；这不表示已发生物理接触。"
                "当前 episode 已终止，不能 rearm、重采 Home 或切换 "
                "Policy/Human 继续，只能 Reset 仿真。"
            )
            action = _intent_action(
                "rebuild_runtime",
                "Reset 仿真",
                destructive=True,
            )
        elif started and held and not episode_ended:
            action = _intent_action("rearm", "重新检查并恢复")
        elif allocation_profile and not started and not blockers:
            action = _intent_action(
                "start_allocation_runtime",
                "修正后重新启动",
            )
        else:
            action = _focus_action("operator-diagnostics", "查看诊断信息")
    elif episode_ended:
        phase, severity = "episode_ended", "warning"
        headline = "当前回合已结束"
        detail = (
            "本回合已经封存并自动保存；确认提交状态后退出控制。"
            if human_demonstration
            else "本回合不会再次恢复执行；完成检查后退出控制。"
        )
        action = _intent_action(
            "stop_runtime",
            "退出控制",
            destructive=True,
        )
    elif started and held:
        phase, severity = "held", "warning"
        headline = "输出已保持"
        detail = "策略、混合和执行不会继续推进；可重采 Runtime Home 后恢复。"
        action = _intent_action("rearm", "重新检查并恢复")
    elif started:
        phase, severity = "running", "ready"
        headline = (
            "Human Demo 正在遥操并记录"
            if human_demonstration
            else "Sim-HIL 正在运行"
        )
        if human_demonstration:
            detail = (
                "Human 是唯一控制来源；完成本回合后封存会事务性提交并自动保存。"
            )
        elif allocation_profile:
            mode_id = str(
                allocation_operations.get("allocation_override_mode")
                or "automatic"
            )
            mode = {
                "automatic": "自动分配",
                "base_only": "仅策略基线",
                "relative_overlay": "人类相对修正",
                "absolute_overlay_owner": "人类绝对接管",
            }.get(mode_id, mode_id)
            selected_modes = set(effective_allocation_modes.values())
            effective = (
                {
                    "base_only": "仅策略基线",
                    "relative_overlay": "人类相对修正",
                    "absolute_overlay_owner": "人类绝对接管",
                }.get(next(iter(selected_modes)), next(iter(selected_modes)))
                if len(selected_modes) == 1
                else "按 component 分配"
                if selected_modes
                else "等待首个正式 composition 周期"
            )
            detail = (
                f"分配规则：{mode}；本周期实际选择：{effective}。"
                "所有动作仍由唯一 MuJoCo 执行链路下发。"
            )
        elif exclusive_profile:
            mode = str(acceptance.get("active_control_mode") or "")
            source = {
                "ppo_only": "Policy",
                "human_teleop": "Human",
            }.get(mode, "待确认来源")
            detail = (
                f"当前控制来源：{source}。使用同一组 Policy / Human 来源卡切换；"
                "切换事务会先 HOLD，再完成 Home 或策略重置。"
            )
        else:
            mode = str(acceptance.get("active_control_mode") or "未报告")
            detail = f"当前控制模式：{mode}。可在下方切换模式或保持输出。"
        action = (
            _intent_action(
                "finalize_capture",
                "结束回合、封存并保存",
                destructive=True,
            )
            if human_demonstration
            else _intent_action("hold", "保持输出")
        )
    elif human_demonstration and sim_preview_ready:
        phase, severity = "sim_preview", "ready"
        headline = "静态仿真已打开，可调整视角"
        if human_launch_ready:
            detail = (
                "当前未接入遥操、未推进物理、未开始记录；视角确认后可一次性开始遥操与记录。"
            )
            action = _intent_action(
                "start_runtime",
                "开始遥操并记录",
                control_mode="human_teleop",
            )
        else:
            detail = (
                "可以继续调整视角；开始记录前仍需 VIVE、角度扳机和坐标配置就绪。"
            )
            action = _focus_action("vive-setup-panel", "检查 Human 输入")
    elif human_demonstration:
        phase, severity = "ready_for_preview", "ready"
        headline = "先打开静态仿真"
        detail = (
            "预览只构造场景并刷新 viewer，不接入遥操、不推进物理、也不创建回合记录。"
        )
        action = _intent_action(
            "prepare_sim_preview",
            "打开仿真并调整视角",
        )
    elif workflow_active:
        phase, severity = "calibrating", "warning"
        headline = "VIVE 坐标标定进行中"
        detail = "按当前方向完成采样；标定保存前不会构造正式运行时。"
        action = _focus_action("vive-setup-panel", "继续标定")
    elif allocation_profile and not blockers and policy_launch_ready:
        phase, severity = "ready_to_start", "ready"
        headline = "动态分配已可启动"
        detail = (
            "先以策略基线构造 MuJoCo；进入 Human 模式时再接入设备并采集 Runtime Home。"
            if not human_launch_ready
            else "可先以策略基线启动，随后按需进入 Human 分配模式。"
        )
        action = _intent_action(
            "start_allocation_runtime",
            "以策略基线启动",
        )
    elif not allocation_profile and policy_launch_ready:
        phase, severity = "ready_to_start", "ready"
        headline = "请选择启动模式"
        if human_launch_ready:
            if exclusive_profile:
                detail = (
                    "Policy 与 Human 都已就绪；在来源卡中选择任一来源启动。"
                )
                action = _focus_action(
                    "exclusive-source-modules",
                    "选择控制来源",
                )
            else:
                detail = "纯遥操、纯 PPO 和比例融合均可作为初始模式。"
                action = _focus_action("control-modes", "选择启动模式")
        else:
            detail = (
                "当前可先以 Policy 启动；Human 来源卡会持续显示缺少的准备条件。"
                if exclusive_profile
                else "当前可先启动纯 PPO；Human 相关模式尚未满足输入条件。"
            )
            action = _intent_action(
                "start_runtime",
                "以 Policy 启动" if exclusive_profile else "以纯 PPO 启动",
                control_mode="ppo_only",
            )
    else:
        phase, severity = "setup_required", "warning"
        if not policy_launch_ready and not allocation_profile:
            headline = "策略运行时尚未就绪"
            detail = "这不是 VIVE 前置条件；请查看策略和启动诊断。"
            action = _focus_action("operator-diagnostics", "查看诊断信息")
        elif calibration_stage == "completed":
            headline = "标定结果尚未保存"
            detail = "保存并选用当前坐标配置后即可继续。"
            action = _focus_action("vive-setup-panel", "保存并选用标定")
        elif input_ready:
            headline = "需要 VIVE 坐标配置"
            detail = "实时输入已经就绪；完成并保存坐标标定后即可启动。"
            action = _intent_action("begin_vive_calibration", "开始 VIVE 标定")
        else:
            headline = "等待有效 VIVE 输入"
            detail = "启动所选方案前，需要被选中的 VIVE 来源持续提供新鲜数据。"
            action = _focus_action("vive-setup-panel", "查看输入设置")

    workflow = (
        _human_demo_workflow(
            status,
            acceptance=acceptance,
            setup=setup,
            capture=capture,
            recommended_action=action,
            phase=phase,
        )
        if human_demonstration
        else None
    )
    return {
        "schema": SIM_HIL_OPERATOR_GUIDANCE_SCHEMA_V1,
        "phase": phase,
        "severity": severity,
        "headline": headline,
        "detail": detail,
        "issue_code": issue_code,
        "raw_error": raw_error,
        "execution_fault": execution_fault,
        "blockers": blockers,
        "notices": notices,
        "recommended_action": action,
        "human_demo_workflow": workflow,
    }


__all__ = [
    "SIM_HIL_OPERATOR_GUIDANCE_SCHEMA_V1",
    "build_sim_hil_operator_guidance",
]

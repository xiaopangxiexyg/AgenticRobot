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
    runtime_constructed = bool(startup.get("mujoco_constructed", False))
    restart_required = bool(startup.get("restart_required", False))
    shutdown_requested = bool(acceptance.get("shutdown_requested", False))
    allocation_profile = bool(
        acceptance.get("allocation_runtime_enabled", False)
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
    if not policy_launch_ready:
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
    if not allocation_profile and policy_launch_ready and not human_launch_ready:
        notices.append(
            "纯 PPO 可以独立启动；遥操和融合仍需 VIVE 输入与坐标配置。"
        )
    if not started:
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
        headline = "操作正在进行"
        detail = "启动、设备接纳或恢复可能需要等待；期间仍可中断当前操作。"
        action = _intent_action("cancel_operation", "中断当前操作")
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
    elif raw_error or capture_fault:
        phase, severity = "faulted", "error"
        issue_code, headline, detail = _classify_error(
            raw_error or str(capture.get("message") or "capture fault")
        )
        if started and held and not episode_ended:
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
        detail = "本回合不会再次恢复执行；完成检查后退出控制。"
        action = _intent_action("stop_runtime", "退出控制", destructive=True)
    elif started and held:
        phase, severity = "held", "warning"
        headline = "输出已保持"
        detail = "策略、混合和执行不会继续推进；可重采 Runtime Home 后恢复。"
        action = _intent_action("rearm", "重新检查并恢复")
    elif started:
        phase, severity = "running", "ready"
        headline = "Sim-HIL 正在运行"
        if allocation_profile:
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
        else:
            mode = str(acceptance.get("active_control_mode") or "未报告")
            detail = f"当前控制模式：{mode}。可在下方切换模式或保持输出。"
        action = _intent_action("hold", "保持输出")
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
            detail = "纯遥操、纯 PPO 和比例融合均可作为初始模式。"
            action = _focus_action("control-modes", "选择启动模式")
        else:
            detail = "当前可先启动纯 PPO；Human 相关模式尚未满足输入条件。"
            action = _intent_action(
                "start_runtime",
                "以纯 PPO 启动",
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

    return {
        "schema": SIM_HIL_OPERATOR_GUIDANCE_SCHEMA_V1,
        "phase": phase,
        "severity": severity,
        "headline": headline,
        "detail": detail,
        "issue_code": issue_code,
        "raw_error": raw_error,
        "blockers": blockers,
        "notices": notices,
        "recommended_action": action,
    }


__all__ = [
    "SIM_HIL_OPERATOR_GUIDANCE_SCHEMA_V1",
    "build_sim_hil_operator_guidance",
]

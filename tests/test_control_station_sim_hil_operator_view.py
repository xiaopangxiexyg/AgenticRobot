from __future__ import annotations

from copy import deepcopy

from control_station.sim_hil_profile.operator_view import (
    SIM_HIL_OPERATOR_GUIDANCE_SCHEMA_V1,
    build_sim_hil_operator_guidance,
)


def _status(*, allocation: bool = True) -> dict[str, object]:
    return {
        "sampled_at": 25.0,
        "started": False,
        "held": False,
        "episode_ended": False,
        "startup": {
            "mujoco_constructed": False,
            "restart_required": False,
        },
        "composition": {
            "allocation_operations": {
                "supported_signals": [
                    "allocation_automatic",
                    "allocation_base_only",
                    "allocation_relative_overlay",
                    "residual_rebase_and_clear",
                ],
                "allocation_override_mode": "automatic",
            }
        },
        "acceptance": {
            "allocation_runtime_enabled": allocation,
            "active_intent_id": None,
            "pending_intent_count": 0,
            "last_error": None,
            "shutdown_requested": False,
            "active_intent_kind": None,
            "formal_runtime_started_at": None,
            "formal_runtime_finalized_at": None,
            "capture": {"status": "ready"},
            "setup": {
                "input_ready": True,
                "profile_ready": True,
                "policy_runtime_launch_ready": True,
                "human_runtime_launch_ready": True,
                "calibration": {
                    "workflow_active": False,
                    "stage": "idle",
                    "input_preflight": {
                        "ready": True,
                        "message": "VIVE input ready",
                    },
                },
            },
        },
    }


def test_allocation_ready_guidance_uses_only_allocation_start_intent() -> None:
    guidance = build_sim_hil_operator_guidance(_status())

    assert guidance["schema"] == SIM_HIL_OPERATOR_GUIDANCE_SCHEMA_V1
    assert guidance["phase"] == "ready_to_start"
    assert guidance["recommended_action"] == {
        "interaction": "intent",
        "intent_kind": "start_allocation_runtime",
        "label": "以策略基线启动",
        "destructive": False,
    }
    assert any("人类绝对接管" in item for item in guidance["notices"])


def test_allocation_human_setup_is_not_a_policy_start_blocker() -> None:
    status = _status()
    setup = status["acceptance"]["setup"]
    setup["input_ready"] = False
    setup["profile_ready"] = False
    setup["human_runtime_launch_ready"] = False
    setup["calibration"]["input_preflight"]["message"] = "等待 VIVE 数据"

    guidance = build_sim_hil_operator_guidance(status)

    assert guidance["phase"] == "ready_to_start"
    assert guidance["blockers"] == []
    assert guidance["recommended_action"]["intent_kind"] == (
        "start_allocation_runtime"
    )
    assert any("等待 VIVE 数据" in item for item in guidance["notices"])
    assert any("Human 模式暂不可用" in item for item in guidance["notices"])


def test_fixed_alpha_policy_only_start_does_not_treat_vive_as_global_blocker() -> None:
    status = _status(allocation=False)
    setup = status["acceptance"]["setup"]
    setup["input_ready"] = False
    setup["profile_ready"] = False
    setup["human_runtime_launch_ready"] = False

    guidance = build_sim_hil_operator_guidance(status)

    assert guidance["phase"] == "ready_to_start"
    assert guidance["blockers"] == []
    assert guidance["recommended_action"]["intent_kind"] == "start_runtime"
    assert guidance["recommended_action"]["control_mode"] == "ppo_only"


def test_exclusive_profile_is_presented_as_two_sources_not_dynamic_allocation() -> None:
    status = _status()
    status["acceptance"]["exclusive_hitl_enabled"] = True

    guidance = build_sim_hil_operator_guidance(status)

    assert guidance["phase"] == "ready_to_start"
    assert guidance["headline"] == "请选择启动模式"
    assert guidance["recommended_action"] == {
        "interaction": "focus",
        "target_id": "exclusive-source-modules",
        "label": "选择控制来源",
        "destructive": False,
    }
    assert all(
        item != "当前分配方案未声明人类绝对接管；Human 仅使用已声明的修正模式。"
        for item in guidance["notices"]
    )


def test_exclusive_profile_can_start_policy_while_human_is_not_ready() -> None:
    status = _status()
    acceptance = status["acceptance"]
    acceptance["exclusive_hitl_enabled"] = True
    setup = acceptance["setup"]
    setup["input_ready"] = False
    setup["profile_ready"] = False
    setup["human_runtime_launch_ready"] = False

    guidance = build_sim_hil_operator_guidance(status)

    assert guidance["phase"] == "ready_to_start"
    assert guidance["recommended_action"]["intent_kind"] == "start_runtime"
    assert guidance["recommended_action"]["control_mode"] == "ppo_only"
    assert "Human 来源卡" in guidance["detail"]


def test_running_exclusive_profile_names_the_actual_selected_source() -> None:
    status = _status()
    status.update({"started": True, "held": False})
    status["startup"]["mujoco_constructed"] = True
    acceptance = status["acceptance"]
    acceptance["exclusive_hitl_enabled"] = True
    acceptance["active_control_mode"] = "human_teleop"

    guidance = build_sim_hil_operator_guidance(status)

    assert guidance["phase"] == "running"
    assert "当前控制来源：Human" in guidance["detail"]
    assert "Policy / Human 来源卡" in guidance["detail"]


def test_active_start_takes_precedence_over_unconstructed_setup_badge() -> None:
    status = _status()
    status["acceptance"]["active_intent_id"] = "start:1"

    guidance = build_sim_hil_operator_guidance(status)

    assert guidance["phase"] == "operation_pending"
    assert guidance["recommended_action"]["intent_kind"] == "cancel_operation"


def test_running_allocation_guidance_separates_rule_from_effective_selection() -> None:
    status = _status()
    status.update({"started": True, "held": False})
    status["startup"]["mujoco_constructed"] = True
    status["composition"]["last_cycle"] = {
        "composition_step": {
            "allocation": {
                "selected_mode_by_component": {
                    "right_arm": "base_only",
                    "right_gripper": "base_only",
                }
            }
        }
    }

    guidance = build_sim_hil_operator_guidance(status)

    assert guidance["phase"] == "running"
    assert "分配规则：自动分配" in guidance["detail"]
    assert "本周期实际选择：仅策略基线" in guidance["detail"]


def test_held_and_terminal_states_have_distinct_recovery_actions() -> None:
    held = _status()
    held.update({"started": True, "held": True})
    held["startup"]["mujoco_constructed"] = True
    held_guidance = build_sim_hil_operator_guidance(held)
    assert held_guidance["phase"] == "held"
    assert held_guidance["recommended_action"]["intent_kind"] == "rearm"

    ended = deepcopy(held)
    ended["episode_ended"] = True
    ended_guidance = build_sim_hil_operator_guidance(ended)
    assert ended_guidance["phase"] == "episode_ended"
    assert ended_guidance["recommended_action"]["intent_kind"] == "stop_runtime"
    assert ended_guidance["recommended_action"]["destructive"] is True


def test_runtime_operation_fault_never_recommends_duplicate_start() -> None:
    status = _status()
    status.update({"started": True, "held": False})
    status["startup"]["mujoco_constructed"] = True
    status["acceptance"]["last_error"] = "RuntimeError:allocation operation failed"

    guidance = build_sim_hil_operator_guidance(status)

    assert guidance["phase"] == "faulted"
    assert guidance["recommended_action"]["interaction"] == "focus"
    assert guidance["recommended_action"]["target_id"] == "operator-diagnostics"


def test_known_device_error_is_summarized_without_dropping_raw_evidence() -> None:
    status = _status()
    raw = (
        "TimeoutError:Human input acceptance timed out: "
        "missing_channels=['right_angle_trigger']"
    )
    status["acceptance"]["last_error"] = raw

    guidance = build_sim_hil_operator_guidance(status)

    assert guidance["issue_code"] == "human_input_timeout"
    assert guidance["headline"] == "等待人类输入超时"
    assert guidance["raw_error"] == raw


def test_lower_collision_rejection_is_explained_without_weakening_safety() -> None:
    status = _status()
    raw = (
        "target_private_resolution_failure.collision_margin_low: "
        "compound authority failed"
    )
    status["acceptance"]["last_error"] = raw

    guidance = build_sim_hil_operator_guidance(status)

    assert guidance["issue_code"] == "lower_collision_margin_hold"
    assert guidance["headline"] == "规划器安全间隙保护已触发"
    assert "代理模型低于预防间隙" in guidance["detail"]
    assert "不表示 MuJoCo 已经发生物理接触" in guidance["detail"]
    assert "只能 Reset" in guidance["detail"]
    assert guidance["raw_error"] == raw


def test_structured_lower_collision_is_visible_when_gate_error_is_generic() -> None:
    status = _status()
    status.update(
        {
            "started": True,
            "held": True,
            "episode_ended": True,
            "episode_end_reason": (
                "sim_collision_stop:collision_margin_low"
            ),
            "last_error": (
                "ExecutionGateNotReady:hold:compound_authority_failure"
            ),
            "execution_gate": {
                "status": "hold",
                "reason": "compound_authority_failure",
            },
        }
    )
    status["startup"]["mujoco_constructed"] = True
    # Collision termination must keep Reset as the only continuation even if
    # another lower fault also marks the runtime as requiring reconstruction.
    status["startup"]["restart_required"] = True
    status["composition"]["last_cycle"] = {
        "execution": {
            "target_completion_outcome_code": (
                "target_private_resolution_failure.collision_margin_low"
            ),
            "resolution_failure": {
                "reason_code": "collision_margin_low",
                "resolver_status": "rejected",
                "message": "projected link distance is below margin",
                "diagnostics": {
                    "best_candidate_distance_m": 0.0019,
                    "collision_threshold_m": 0.0019,
                    "collision_geometry_1": "right_hand_proxy_0",
                    "collision_geometry_2": "table_0",
                },
            },
        }
    }

    guidance = build_sim_hil_operator_guidance(status)

    assert guidance["phase"] == "episode_ended"
    assert guidance["issue_code"] is None
    assert guidance["execution_fault"]["category"] == "collision"
    assert guidance["execution_fault"]["reason_code"] == (
        "collision_margin_low"
    )
    assert guidance["execution_fault"]["collision_semantics"] == (
        "planner_proxy_preventive_clearance_not_physical_contact"
    )
    assert guidance["execution_fault"]["collision_summary"] == (
        "代理对 right_hand_proxy_0 ↔ table_0 · 最近间隙 1.900 mm · "
        "有效阈值 1.900 mm"
    )
    assert guidance["recommended_action"] == {
        "interaction": "intent",
        "intent_kind": "rebuild_runtime",
        "label": "Reset 仿真",
        "destructive": True,
    }

    status["acceptance"]["capture_profile"] = "human_demonstration"
    demo_guidance = build_sim_hil_operator_guidance(status)
    assert demo_guidance["recommended_action"]["intent_kind"] == (
        "rebuild_runtime"
    )


def test_actual_mujoco_forbidden_contact_is_not_described_as_proxy_margin() -> None:
    status = _status()
    status.update(
        {
            "started": True,
            "held": True,
            "episode_ended": True,
            "episode_end_reason": (
                "sim_collision_stop:active_forbidden_contact"
            ),
            "sim_collision_contact": {
                "available": True,
                "safe": False,
                "geom1": "right_gripper_collision",
                "geom2": "table_collision",
                "penetration_depth_m": 0.0015,
            },
        }
    )
    status["startup"]["mujoco_constructed"] = True

    guidance = build_sim_hil_operator_guidance(status)

    assert guidance["headline"] == "MuJoCo 禁碰接触保护已触发"
    assert "已观测到" in guidance["detail"]
    assert guidance["execution_fault"]["collision_semantics"] == (
        "mujoco_forbidden_contact_hard_penetration"
    )
    assert guidance["execution_fault"]["collision_summary"] == (
        "实际接触 right_gripper_collision ↔ table_collision · "
        "穿透深度 1.500 mm"
    )


def test_human_demo_guidance_requires_preview_before_teleop_recording() -> None:
    status = _status(allocation=False)
    status["acceptance"]["capture_profile"] = "human_demonstration"
    status["acceptance"]["sim_preview_ready"] = False

    before = build_sim_hil_operator_guidance(status)

    assert before["phase"] == "ready_for_preview"
    assert before["recommended_action"]["intent_kind"] == (
        "prepare_sim_preview"
    )
    before_workflow = before["human_demo_workflow"]
    assert [step["state"] for step in before_workflow["steps"]] == [
        "ready",
        "waiting",
        "waiting",
        "waiting",
    ]
    assert before_workflow["primary_action"]["intent_kind"] == (
        "prepare_sim_preview"
    )

    status["startup"]["mujoco_constructed"] = True
    status["acceptance"]["sim_preview_ready"] = True
    preview = build_sim_hil_operator_guidance(status)

    assert preview["phase"] == "sim_preview"
    assert preview["recommended_action"] == {
        "interaction": "intent",
        "intent_kind": "start_runtime",
        "label": "开始遥操并记录",
        "destructive": False,
        "control_mode": "human_teleop",
    }
    preview_workflow = preview["human_demo_workflow"]
    assert [step["state"] for step in preview_workflow["steps"]] == [
        "done",
        "ready",
        "ready",
        "waiting",
    ]
    assert preview_workflow["input"]["angle_trigger"] == (
        "checked_when_recording_starts"
    )


def test_running_human_demo_guidance_finalizes_and_saves() -> None:
    status = _status(allocation=False)
    status.update({"started": True, "held": False})
    status["startup"]["mujoco_constructed"] = True
    status["acceptance"]["capture_profile"] = "human_demonstration"
    status["acceptance"]["capture"] = {"status": "recording"}
    status["acceptance"]["formal_runtime_started_at"] = 20.0

    guidance = build_sim_hil_operator_guidance(status)

    assert guidance["headline"] == "Human Demo 正在遥操并记录"
    assert guidance["recommended_action"]["intent_kind"] == (
        "finalize_capture"
    )
    assert guidance["recommended_action"]["destructive"] is True
    workflow = guidance["human_demo_workflow"]
    assert workflow["recording"]["active"] is True
    assert workflow["recording"]["elapsed_sec"] == 5.0
    assert workflow["input"]["angle_trigger"] == "connected"
    assert workflow["steps"][2]["state"] == "active"
    assert workflow["steps"][3]["state"] == "ready"


def test_human_demo_preflight_failure_recommends_retry_without_losing_preview() -> None:
    status = _status(allocation=False)
    status["startup"]["mujoco_constructed"] = True
    acceptance = status["acceptance"]
    acceptance["capture_profile"] = "human_demonstration"
    acceptance["sim_preview_ready"] = True
    acceptance["last_error"] = (
        "TimeoutError:Human input acceptance timed out: "
        "missing_channels=['right_angle_trigger']"
    )

    guidance = build_sim_hil_operator_guidance(status)

    assert guidance["headline"] == "Human 输入未通过，视角仍已保留"
    assert guidance["recommended_action"]["intent_kind"] == "start_runtime"
    assert guidance["recommended_action"]["control_mode"] == "human_teleop"
    assert guidance["human_demo_workflow"]["steps"][0]["state"] == "done"


def test_human_demo_prepreview_failure_can_retry_preview() -> None:
    status = _status(allocation=False)
    acceptance = status["acceptance"]
    acceptance["capture_profile"] = "human_demonstration"
    acceptance["sim_preview_ready"] = False
    acceptance["last_error"] = "previous setup operation failed"

    guidance = build_sim_hil_operator_guidance(status)

    assert guidance["headline"] == "上次操作失败，可重新打开静态仿真"
    assert guidance["recommended_action"]["intent_kind"] == (
        "prepare_sim_preview"
    )


def test_capture_fault_never_recommends_rearm() -> None:
    status = _status(allocation=False)
    status.update({"started": True, "held": True})
    status["startup"]["mujoco_constructed"] = True
    acceptance = status["acceptance"]
    acceptance["capture_profile"] = "human_demonstration"
    acceptance["capture"] = {
        "status": "faulted",
        "message": "writer fsync failed",
    }

    guidance = build_sim_hil_operator_guidance(status)

    assert guidance["recommended_action"] == {
        "interaction": "focus",
        "target_id": "operator-diagnostics",
        "label": "查看记录故障",
        "destructive": False,
    }
    assert guidance["human_demo_workflow"]["steps"][2]["state"] == "fault"
    assert guidance["human_demo_workflow"]["steps"][3]["state"] == "fault"


def test_human_demo_active_operation_copy_names_the_actual_boundary() -> None:
    status = _status(allocation=False)
    acceptance = status["acceptance"]
    acceptance["capture_profile"] = "human_demonstration"
    acceptance["active_intent_id"] = "intent:start"
    acceptance["active_intent_kind"] = "start_runtime"

    guidance = build_sim_hil_operator_guidance(status)

    assert guidance["headline"] == "正在开始遥操与记录"
    assert "采集 Runtime Home" in guidance["detail"]
    workflow = guidance["human_demo_workflow"]
    assert workflow["operation_label"] == (
        "正在接入 Human、采集 Home 并开始记录"
    )

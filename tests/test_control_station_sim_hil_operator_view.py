from __future__ import annotations

from copy import deepcopy

from control_station.sim_hil_profile.operator_view import (
    SIM_HIL_OPERATOR_GUIDANCE_SCHEMA_V1,
    build_sim_hil_operator_guidance,
)


def _status(*, allocation: bool = True) -> dict[str, object]:
    return {
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

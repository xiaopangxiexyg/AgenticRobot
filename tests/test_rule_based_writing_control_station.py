from __future__ import annotations

from pathlib import Path

import pytest

from control_station.rule_based_writing import (
    RuleBasedWritingApplication,
    RuleBasedWritingIntent,
    RuleBasedWritingWebUi,
    RuleWritingTaskChainPhase,
)


def _task_chain_configuration() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    options = {
        "source_candidates_by_phase": {
            phase: ["a", "b"] for phase in ("pen_down", "write", "pen_up")
        },
        "trajectory_ids": ["digit_1_v1", "letter_s_v1"],
        "directions": ["up"],
        "numeric_bounds": {
            "writing_size_m": [0.005, 0.2],
            "initial_alignment_speed_scale": [0.05, 1.0],
        },
    }
    first = {
        "pen_down_candidate_id": "a",
        "write_candidate_id": "a",
        "pen_up_candidate_id": "a",
        "trajectory_id": "digit_1_v1",
        "direction": "up",
        "writing_size_m": 0.08,
        "initial_alignment_speed_scale": 1.0,
    }
    second = {
        **first,
        "write_candidate_id": "b",
        "trajectory_id": "letter_s_v1",
        "initial_alignment_speed_scale": 0.5,
    }
    return options, first, second


def _task_chain_ready_status(**updates: object) -> dict[str, object]:
    return {
        "started": True,
        "armed": False,
        "execution_begun": False,
        "completed": False,
        "aborted": False,
        "held": False,
        "session_state": "ARMED",
        "rule_start_available": True,
        "configuration_available": True,
        "gripper_control_available": True,
        "gripper_sides": ["right"],
        "height_calibration_available": True,
        **updates,
    }


def test_rule_application_exposes_one_start_and_translates_to_writer_intent() -> None:
    application = RuleBasedWritingApplication()
    application.observe_runtime_publication(
        {
            "stage": "RM65 真机遥操运行中",
            "active_controls": [
                "episode_start",
                "episode_delete",
                "fix_arm_toggle",
                "progress_advance",
            ],
        }
    )
    application.bind_rule_runtime()
    application.observe_rule_status({"rule_start_available": True})

    ready = application.snapshot()
    assert ready["phase"] == "ready"
    assert ready["can_start"] is True
    assert "active_controls" not in ready
    assert application.submit_intent(RuleBasedWritingIntent.START) is True
    assert application.consume_runtime_events() == ["episode_start"]
    assert application.consume_rule_intents() == ("rule_start",)


def test_rule_ui_opens_only_its_armed_pre_episode_writer_window() -> None:
    application = RuleBasedWritingApplication()
    application.observe_runtime_publication(
        {"active_controls": ["episode_start"]}
    )
    application.bind_rule_runtime()
    application.observe_rule_status({"rule_start_available": True})

    assert application.allows_pre_episode_armed_warmup() is False
    assert application.submit_intent(RuleBasedWritingIntent.START) is True
    assert application.allows_pre_episode_armed_warmup() is True

    # Writer preparation and rule status publication are asynchronous.  The
    # accepted dedicated START latch must remain authoritative even if a
    # transient callback projects ``preparing`` instead of ``armed``.
    application._state.phase = "preparing"
    assert application.allows_pre_episode_armed_warmup() is True

    application.observe_runtime_publication(
        {
            "startup_blockers": {
                "rig.session": (
                    "Session state=ARMED; runtime Home and hardware arm admission "
                    "are required before rule collection."
                )
            }
        }
    )
    assert application.allows_pre_episode_armed_warmup() is True
    assert application.pre_episode_start_admission_reason() is None
    application.observe_rule_status(
        {
            "started": True,
            "armed": True,
            "execution_begun": False,
            "completed": False,
            "aborted": False,
            "held": False,
            "session_state": "ARMED",
            "readiness_blockers": {
                "dataset.lifecycle": (
                    "cancelled rule Episode lifecycle is still settling"
                )
            },
        }
    )
    assert application.allows_pre_episode_armed_warmup() is True
    assert application.pre_episode_start_admission_reason() is None
    application.observe_rule_status(
        {
            "started": True,
            "armed": False,
            "execution_begun": False,
            "completed": False,
            "aborted": False,
            "held": False,
            "session_state": "ARMED",
        }
    )
    assert application.allows_pre_episode_armed_warmup() is True

    application.observe_runtime_publication(
        {"startup_blockers": {"right_hand": "DMTAC fault"}}
    )
    assert application.allows_pre_episode_armed_warmup() is False


def test_rule_ui_does_not_ignore_discard_lifecycle_or_device_faults() -> None:
    application = RuleBasedWritingApplication()
    application.observe_runtime_publication({"active_controls": ["episode_start"]})
    application.bind_rule_runtime()
    application.observe_rule_status({"rule_start_available": True})
    assert application.submit_intent(RuleBasedWritingIntent.START) is True

    application.observe_rule_status(
        {
            "started": True,
            "armed": True,
            "execution_begun": False,
            "completed": False,
            "aborted": False,
            "held": False,
            "session_state": "ARMED",
            "readiness_blockers": {
                "dataset.lifecycle": (
                    "cancelled rule Episode is still active and awaiting discard"
                )
            },
        }
    )
    assert application.allows_pre_episode_armed_warmup() is False

    application = RuleBasedWritingApplication()
    application.observe_runtime_publication({"active_controls": ["episode_start"]})
    application.bind_rule_runtime()
    application.observe_rule_status({"rule_start_available": True})
    assert application.submit_intent(RuleBasedWritingIntent.START) is True
    application.observe_rule_status(
        {
            "started": True,
            "armed": True,
            "execution_begun": False,
            "completed": False,
            "aborted": False,
            "held": False,
            "session_state": "ARMED",
            "readiness_blockers": {
                "hand.right": "right_hand telemetry_failed: Write timeout",
                "dataset.lifecycle": (
                    "cancelled rule Episode lifecycle is still settling"
                ),
            },
        }
    )
    assert application.allows_pre_episode_armed_warmup() is False
    reason = application.pre_episode_start_admission_reason()
    assert reason is not None
    assert "hand.right" in reason


def test_rule_ui_preserves_dynamic_blocker_for_pre_episode_admission_diagnostic() -> None:
    application = RuleBasedWritingApplication()
    application.observe_runtime_publication({"active_controls": ["episode_start"]})
    application.bind_rule_runtime()
    application.observe_rule_status({"rule_start_available": True})
    assert application.submit_intent(RuleBasedWritingIntent.START) is True

    application.observe_runtime_publication(
        {"startup_blockers": {"hand.right": "right_hand telemetry_failed: Write timeout"}}
    )

    reason = application.pre_episode_start_admission_reason()
    assert reason is not None
    assert "hand.right" in reason
    assert "Write timeout" in reason


def test_rule_ui_keeps_accepted_start_window_across_stale_status_heartbeat() -> None:
    application = RuleBasedWritingApplication()
    application.observe_runtime_publication(
        {"active_controls": ["episode_start"]}
    )
    application.bind_rule_runtime()
    application.observe_rule_status(
        {
            "started": True,
            "armed": False,
            "execution_begun": False,
            "completed": False,
            "aborted": False,
            "held": False,
            "session_state": "ARMED",
            "rule_start_available": True,
            "configuration_available": True,
        }
    )

    assert application.submit_intent(RuleBasedWritingIntent.START) is True
    application.observe_rule_status(
        {
            "started": True,
            "armed": False,
            "execution_begun": False,
            "completed": False,
            "aborted": False,
            "held": False,
            "session_state": "ARMED",
            "rule_start_available": True,
            "configuration_available": True,
        }
    )

    state = application.snapshot()
    assert state["phase"] == "armed"
    assert application.allows_pre_episode_armed_warmup() is True


def test_rule_runtime_bind_waits_for_first_readiness_status() -> None:
    """Binding the extension alone must not advertise a usable start gate."""

    application = RuleBasedWritingApplication()
    application.bind_rule_runtime()

    state = application.snapshot()
    assert state["phase"] == "preparing"
    assert state["title"] == "规则采集环境准备中"
    assert state["can_start"] is False
    assert state["rule_start_available"] is False

    application.observe_rule_status({"rule_start_available": True})
    ready = application.snapshot()
    assert ready["phase"] == "ready"
    assert ready["can_start"] is True


def test_task_chain_explains_missing_right_gripper_instead_of_inviting_click() -> None:
    application = RuleBasedWritingApplication()
    application.bind_rule_runtime()
    application.observe_rule_status(
        {
            "collection_mode": "rule_based_writing",
            "rule_start_available": True,
            "configuration_available": True,
            "gripper_control_available": False,
            "gripper_sides": [],
            "height_calibration_available": True,
        }
    )

    state = application.snapshot()
    assert state["task_chain_can_advance"] is False
    assert "等待夹爪控制拓扑就绪" in state["task_chain_detail"]
    assert "按“继续任务链”" not in state["task_chain_detail"]


def test_failed_runtime_home_returns_to_a_retryable_blocked_state() -> None:
    application = RuleBasedWritingApplication()
    application.bind_rule_runtime()
    application.observe_rule_status(
        {
            "started": True,
            "session_state": "DEBUG_ONLY",
            "rule_start_available": False,
            "readiness_blockers": {
                "rig.session": "Session state=DEBUG_ONLY; runtime Home required"
            },
        }
    )

    assert application.snapshot()["can_runtime_home"] is True
    assert application.submit_intent(RuleBasedWritingIntent.RUNTIME_HOME_RETRY) is True
    assert application.snapshot()["runtime_home_pending"] is True
    application.observe_rule_status({"session_state": "ARMED"})
    assert application.snapshot()["runtime_home_pending"] is True
    application.observe_runtime_publication(
        {
            "capture_stage": "runtime_home",
            "capture_phase": "ready",
            "control_hint": "已禁止新的机械臂/夹爪命令，正在等待底层停机确认。",
        }
    )
    assert application.snapshot()["runtime_home_pending"] is True

    application.observe_runtime_publication(
        {
            "capture_stage": "runtime_home",
            "capture_phase": "failed",
            "home_ready": False,
            "control_hint": "实时 qpos 快照失败；机械臂保持 Hold。",
        }
    )
    failed = application.snapshot()
    assert failed["runtime_home_pending"] is False
    assert failed["can_runtime_home"] is True
    assert failed["phase"] == "blocked"
    assert "实时 qpos 快照失败" in failed["detail"]

    # A second click is a genuine retry, not a permanently consumed request.
    assert application.submit_intent(RuleBasedWritingIntent.RUNTIME_HOME_RETRY) is True


def test_rule_start_does_not_depend_on_generic_episode_control_projection() -> None:
    application = RuleBasedWritingApplication()
    application.observe_runtime_publication(
        {"stage": "ARMED", "active_controls": ["h", "q"]}
    )
    application.bind_rule_runtime()
    application.observe_rule_status({"rule_start_available": True})

    assert application.snapshot()["can_start"] is True
    assert application.submit_intent(RuleBasedWritingIntent.START) is True
    assert application.consume_runtime_events() == ["episode_start"]
    assert application.consume_rule_intents() == ("rule_start",)


def test_rule_debug_toggle_reuses_runtime_diagnostic_event_and_projects_path() -> None:
    application = RuleBasedWritingApplication()
    application.bind_rule_runtime()

    assert application.snapshot()["can_toggle_debug"] is True
    assert application.submit_intent(RuleBasedWritingIntent.DEBUG_TOGGLE) is True
    assert application.consume_runtime_events() == ["d"]
    assert application.snapshot()["debug_mode_pending"] is True
    assert application.snapshot()["can_toggle_debug"] is False

    application.observe_runtime_publication(
        {
            "debug_ik_logging": True,
            "ik_debug_log_path": "/tmp/rule-debug.jsonl",
        }
    )
    state = application.snapshot()
    assert state["debug_mode_enabled"] is True
    assert state["debug_mode_pending"] is False
    assert state["can_toggle_debug"] is True
    assert state["debug_log_path"] == "/tmp/rule-debug.jsonl"


def test_height_calibration_controls_convert_mm_and_follow_idle_authority() -> None:
    application = RuleBasedWritingApplication()
    application.bind_rule_runtime()
    application.observe_rule_status(
        {
            "started": True,
            "session_state": "ARMED",
            "rule_start_available": True,
            "collection_mode": "rule_based_writing",
            "height_calibration_available": True,
        }
    )
    state = application.snapshot()
    assert state["can_capture_height"] is True
    assert state["can_clear_height"] is False

    assert application.submit_height_calibration(
        {
            "operation": "capture",
            "lower_tolerance_mm": 1.5,
            "upper_tolerance_mm": 2.0,
        }
    )
    assert application.consume_height_calibration_requests() == (
        {
            "operation": "capture",
            "lower_tolerance_m": pytest.approx(0.0015),
            "upper_tolerance_m": pytest.approx(0.002),
        },
    )
    assert application.snapshot()["can_start"] is False
    application.complete_height_calibration(error=None)
    application.observe_rule_status(
        {
            "started": True,
            "session_state": "ARMED",
            "rule_start_available": True,
            "collection_mode": "rule_based_writing",
            "height_calibration_enabled": True,
            "height_calibrated_z_m": 0.321,
            "height_lower_bound_z_m": 0.3195,
            "height_upper_bound_z_m": 0.323,
            "height_lower_tolerance_m": 0.0015,
            "height_upper_tolerance_m": 0.002,
            "height_calibration_available": True,
            "height_calibration_clear_available": True,
        }
    )
    state = application.snapshot()
    assert state["can_clear_height"] is True
    assert state["height_calibrated_z_m"] == pytest.approx(0.321)
    assert application.submit_height_calibration({"operation": "clear"})
    assert application.consume_height_calibration_requests() == (
        {"operation": "clear"},
    )


def test_height_calibration_is_hidden_from_full_replay_mode() -> None:
    application = RuleBasedWritingApplication()
    application.bind_rule_runtime()
    application.observe_rule_status(
        {
            "started": True,
            "session_state": "ARMED",
            "rule_start_available": True,
            "collection_mode": "full_trajectory_replay",
            "height_calibration_available": True,
            "height_calibration_clear_available": True,
        }
    )
    state = application.snapshot()
    assert state["can_capture_height"] is False
    assert state["can_clear_height"] is False


def test_rule_stop_atomically_blocks_rule_and_requests_writer_stop_and_hold() -> None:
    application = RuleBasedWritingApplication()
    application.observe_runtime_publication(
        {"active_controls": ["episode_start", "episode_stop", "h"]}
    )
    application.bind_rule_runtime()
    application.observe_rule_status({"rule_start_available": True})
    assert application.submit_intent(RuleBasedWritingIntent.START) is True
    application.consume_runtime_events()
    application.consume_rule_intents()
    application.observe_rule_status(
        {
            "armed": True,
            "execution_begun": True,
            "completed": False,
            "aborted": False,
            "held": False,
            "recording_ready": True,
            "step_id": "right_write.write",
            "directive_kind": "custom_action_segment",
            "sequence_index": 4,
        }
    )

    assert application.submit_intent(RuleBasedWritingIntent.STOP) is True
    assert application.consume_rule_intents() == ("rule_stop",)
    assert application.consume_runtime_events() == [
        "episode_stop:rule_based_hold_cancel",
        "h",
    ]
    assert application.snapshot()["can_stop"] is False

    application.observe_runtime_publication(
        {
            "capture_stage": "runtime_home",
            "capture_phase": "ready",
            "startup_blockers": {
                "rig.session": "Session state=DEBUG_ONLY; runtime Home required"
            },
        }
    )
    application.observe_rule_status(
        {
            "armed": False,
            "execution_begun": False,
            "completed": False,
            "aborted": False,
            "held": True,
            "rule_start_available": False,
            "session_state": "DEBUG_ONLY",
            "readiness_blockers": {
                "rig.session": "Session state=DEBUG_ONLY; runtime Home required"
            },
        }
    )
    held = application.snapshot()
    assert held["phase"] == "blocked"
    assert held["can_start"] is False
    assert held["can_runtime_home"] is True


def test_rule_failure_stop_requests_one_non_success_writer_reclaim() -> None:
    application = RuleBasedWritingApplication()

    application.request_failure_stop()
    application.request_failure_stop()

    assert application.consume_runtime_events() == [
        "episode_stop:rule_based_abort"
    ]


def test_failed_rule_attempt_exposes_rehome_after_confirmed_hold_reset() -> None:
    application = RuleBasedWritingApplication()
    application.bind_rule_runtime()
    application.observe_rule_status(
        {
            "started": True,
            "armed": False,
            "execution_begun": True,
            "aborted": True,
            "held": False,
            "session_state": "RUNNING",
            "rule_start_available": False,
        }
    )

    failed = application.snapshot()
    assert failed["phase"] == "aborted"
    assert failed["can_runtime_home"] is False

    application.observe_runtime_publication(
        {
            "capture_stage": "runtime_home",
            "capture_phase": "ready",
            "startup_blockers": {
                "rig.session": "Session state=DEBUG_ONLY; runtime Home required"
            },
        }
    )
    application.observe_rule_status(
        {
            "started": True,
            "armed": False,
            "execution_begun": False,
            "completed": False,
            "aborted": False,
            "held": True,
            "session_state": "DEBUG_ONLY",
            "waiting_reason": "runtime_home_required_after_hold",
            "rule_start_available": False,
            "readiness_blockers": {
                "rig.session": "Session state=DEBUG_ONLY; runtime Home required"
            },
        }
    )

    recoverable = application.snapshot()
    assert recoverable["phase"] == "blocked"
    assert recoverable["can_start"] is False
    assert recoverable["can_runtime_home"] is True
    assert application.submit_intent(RuleBasedWritingIntent.RUNTIME_HOME_RETRY) is True
    assert application.consume_runtime_events() == ["runtime_home_retry"]


def test_transient_writer_pause_keeps_running_controls_stable() -> None:
    application = RuleBasedWritingApplication()
    application.bind_rule_runtime()

    application.observe_rule_status(
        {
            "started": True,
            "armed": True,
            "execution_begun": True,
            "completed": False,
            "aborted": False,
            "held": False,
            "session_state": "RUNNING",
            "waiting_reason": "formal_writer_paused",
            "step_id": "right_write.pen_down",
        }
    )

    paused = application.snapshot()
    assert paused["phase"] == "running"
    assert paused["can_stop"] is True
    assert paused["can_runtime_home"] is False
    assert "保持最后一个安全目标" in paused["detail"]
    assert "连续新鲜后自动恢复" in paused["detail"]

    application.observe_rule_status(
        {
            "started": True,
            "armed": True,
            "execution_begun": True,
            "completed": False,
            "aborted": False,
            "held": False,
            "session_state": "RUNNING",
            "waiting_reason": "",
            "step_id": "right_write.pen_down",
        }
    )
    resumed = application.snapshot()
    assert resumed["phase"] == "running"
    assert resumed["can_stop"] is True
    assert resumed["can_runtime_home"] is False
    assert resumed["detail"] == "当前规则步骤：right_write.pen_down"


def test_rule_ui_asset_contains_no_teleop_episode_progress_or_pedal_controls() -> None:
    package = (
        Path(__file__).resolve().parents[1]
        / "control_station"
        / "rule_based_writing"
    )
    asset = (package / "static" / "index.html").read_text(encoding="utf-8")
    script = (package / "static" / "app.js").read_text(encoding="utf-8")
    transport = (package / "control_ui.py").read_text(encoding="utf-8")

    assert "开始规则采集" in asset
    assert "停止并 Hold" in asset
    assert 'id="topology"' in asset
    assert 'id="position"' in asset
    assert 'id="samples"' in asset
    assert 'id="batch-progress"' in asset
    assert 'id="add-to-batch"' in asset
    assert 'id="batch-items"' in asset
    assert 'id="collection-mode"' in asset
    assert 'id="full-replay-source"' in asset
    assert 'id="pre-pen-down-source"' in asset
    assert 'id="pre-pen-down-normal-episodes"' in asset
    assert 'id="pen-down-source"' in asset
    assert 'id="write-source"' in asset
    assert 'id="pen-up-source"' in asset
    assert 'id="trajectory"' in asset
    assert 'id="direction"' in asset
    assert 'id="writing-speed"' in asset
    assert 'id="alignment-speed-input"' in asset
    assert 'id="tactile-bundle-root"' in asset
    assert 'id="tactile-min-free"' in asset
    assert 'id="writer-workers"' in asset
    assert 'id="camera-frame-max-age"' in asset
    assert 'id="sample-hz"' in asset
    assert 'id="position-reference-hz"' in asset
    assert 'id="arm-control-hz"' in asset
    assert 'id="debug-mode"' in asset
    assert 'data-event="debug_toggle"' in asset
    assert 'data-event="runtime_home_retry"' in asset
    assert 'id="task-chain-advance"' in asset
    assert "Alt+N" in asset
    assert 'postJson("/task-chain"' in script
    assert 'pendingRequests.has("task-chain")' in script
    assert "event.altKey" in script
    assert '"/task-chain"' in transport
    for forbidden in ("开始遥操", "Episode", "长程进度", "踏板", "夹爪控制"):
        assert forbidden not in asset


def test_rule_ui_satisfies_operator_port_without_starting_http_server() -> None:
    ui = RuleBasedWritingWebUi(open_browser=False)
    assert ui.diagnostic_mode_profile == "rule-writing"
    ui.publish(active_controls=["episode_start"])
    ui.bind_rule_runtime()
    ui.publish_rule_status({"rule_start_available": True})
    assert ui.snapshot()["can_start"] is True
    assert ui.consume_events() == []


def test_rule_ui_shows_dual_gripper_topology_stage_position_and_counts() -> None:
    application = RuleBasedWritingApplication()
    application.bind_rule_runtime()
    application.observe_rule_status({"rule_start_available": True})
    application.observe_rule_status(
        {
            "armed": True,
            "execution_begun": True,
            "recording_ready": True,
            "step_id": "right_write.write",
            "directive_kind": "custom_action",
            "sequence_index": 17,
            "stage_label": "书写",
            "stage_index": 2,
            "stage_frame_index": 18,
            "stage_frame_total": 40,
            "collection_position": "阶段 2 / 3 · 帧 18 / 40",
            "recorded_sample_count": 27,
            "planned_sample_count": 83,
            "initial_alignment_speed_scale": 0.5,
            "gripper_sides": ["left", "right"],
            "control_topology": "双臂 + 左夹爪 + 右夹爪",
            "gripper_control_available": False,
        }
    )

    state = application.snapshot()
    assert state["control_topology"] == "双臂 + 左夹爪 + 右夹爪"
    assert state["stage_label"] == "书写"
    assert state["collection_position"] == "阶段 2 / 3 · 帧 18 / 40"
    assert state["recorded_sample_count"] == 27
    assert state["planned_sample_count"] == 83
    assert state["initial_alignment_speed_scale"] == 0.5


def test_rule_ui_explains_startup_home_continuity_without_claiming_motion() -> None:
    application = RuleBasedWritingApplication()
    application.bind_rule_runtime()
    application.observe_rule_status(
        {
            "armed": True,
            "execution_begun": False,
            "completed": False,
            "aborted": False,
            "held": False,
            "waiting_reason": "startup_home_continuity_dwell",
        }
    )

    state = application.snapshot()
    assert state["phase"] == "armed"
    assert "runtime Home 连续性验证" in state["detail"]
    assert "不会下发运动或写入样本" in state["detail"]


def test_rule_ui_exposes_rehome_only_after_devices_recover() -> None:
    application = RuleBasedWritingApplication()
    application.bind_rule_runtime()
    application.observe_runtime_publication({"active_controls": ["device_retry", "q"]})
    application.observe_rule_status(
        {
            "started": True,
            "session_state": "DEBUG_ONLY",
            "readiness_reason": (
                "Session state=DEBUG_ONLY; runtime Home and hardware arm admission "
                "are required before rule collection."
            ),
            "readiness_blockers": {
                "hand.right": "right_hand error_code=16",
                "rig.session": "Rig is not ready: right_hand error_code=16",
            },
        }
    )

    blocked = application.snapshot()
    assert blocked["can_start"] is False
    assert blocked["can_retry_devices"] is True
    assert blocked["can_runtime_home"] is False

    application.observe_rule_status(
        {
            "started": True,
            "session_state": "DEBUG_ONLY",
            "readiness_reason": (
                "Session state=DEBUG_ONLY; runtime Home and hardware arm admission "
                "are required before rule collection."
            ),
            "readiness_blockers": {
                "rig.session": (
                    "Session state=DEBUG_ONLY; runtime Home and hardware arm admission "
                    "are required before rule collection."
                )
            },
        }
    )

    recovered = application.snapshot()
    assert recovered["can_retry_devices"] is False
    assert recovered["can_runtime_home"] is True
    assert "重新安全初始化" in recovered["detail"]
    assert application.submit_intent(RuleBasedWritingIntent.RUNTIME_HOME_RETRY) is True
    assert application.consume_runtime_events() == ["runtime_home_retry"]
    assert application.consume_rule_intents() == ()
    assert application.snapshot()["runtime_home_pending"] is True


def test_rule_device_retry_holds_running_control_before_automatic_retry() -> None:
    application = RuleBasedWritingApplication()
    application.bind_rule_runtime()
    application.observe_runtime_publication(
        {
            "active_controls": ["h", "q"],
            "startup_blockers": {
                "camera.front": "camera did not produce a fresh frame",
            },
        }
    )
    application.observe_rule_status(
        {
            "started": True,
            "session_state": "ARMED",
            "readiness_blockers": {},
        }
    )

    assert application.snapshot()["can_retry_devices"] is True
    assert application.submit_intent(RuleBasedWritingIntent.DEVICE_RETRY) is True
    assert application.consume_runtime_events() == ["h"]
    waiting = application.snapshot()
    assert waiting["device_retry_pending"] is True
    assert waiting["can_retry_devices"] is False
    assert "底层 Hold" in waiting["detail"]

    application.observe_runtime_publication({"active_controls": ["q", "d"]})
    assert application.consume_runtime_events() == []

    application.observe_runtime_publication(
        {"active_controls": ["device_retry", "q", "d"]}
    )
    assert application.consume_runtime_events() == ["device_retry"]
    assert "底层停机已确认" in application.snapshot()["detail"]

    application.complete_device_retry(error=None)
    completed = application.snapshot()
    assert completed["device_retry_pending"] is False
    assert completed["can_retry_devices"] is True


def test_rule_device_retry_is_unavailable_without_lower_retry_or_hold_control() -> None:
    application = RuleBasedWritingApplication()
    application.bind_rule_runtime()
    application.observe_runtime_publication(
        {
            "active_controls": ["q", "d"],
            "startup_blockers": {
                "camera.front": "camera did not produce a fresh frame",
            },
        }
    )
    application.observe_rule_status(
        {"started": True, "session_state": "ARMED", "readiness_blockers": {}}
    )

    assert application.snapshot()["can_retry_devices"] is False
    assert application.submit_intent(RuleBasedWritingIntent.DEVICE_RETRY) is False


def test_rule_device_retry_never_turns_active_episode_into_implicit_hold() -> None:
    application = RuleBasedWritingApplication()
    application.bind_rule_runtime()
    application.observe_runtime_publication(
        {
            "active_controls": ["h", "q"],
            "startup_blockers": {
                "camera.front": "camera stopped producing frames",
            },
        }
    )
    application.observe_rule_status(
        {
            "started": True,
            "session_state": "RUNNING",
            "armed": True,
            "execution_begun": True,
            "readiness_blockers": {},
        }
    )

    assert application.snapshot()["phase"] == "running"
    assert application.snapshot()["can_retry_devices"] is False
    assert application.submit_intent(RuleBasedWritingIntent.DEVICE_RETRY) is False
    assert application.consume_runtime_events() == []


def test_rule_device_retry_stops_waiting_after_negative_lower_hold_receipt() -> None:
    application = RuleBasedWritingApplication()
    application.bind_rule_runtime()
    application.observe_runtime_publication(
        {
            "active_controls": ["h", "q"],
            "startup_blockers": {
                "camera.front": "camera did not produce a fresh frame",
            },
        }
    )
    application.observe_rule_status(
        {"started": True, "session_state": "ARMED", "readiness_blockers": {}}
    )
    assert application.submit_intent(RuleBasedWritingIntent.DEVICE_RETRY) is True
    assert application.consume_runtime_events() == ["h"]

    application.observe_runtime_publication(
        {
            "active_controls": ["h", "q", "d"],
            "hardware_hold_confirmed": False,
            "control_hint": "底层停机确认失败；禁止维护动作。",
        }
    )

    failed = application.snapshot()
    assert failed["device_retry_pending"] is False
    assert failed["title"] == "安全停臂未确认"
    assert failed["detail"] == "底层停机确认失败；禁止维护动作。"
    assert application.consume_runtime_events() == []


def test_rule_ui_keeps_session_debug_only_blocked_after_retry_projection() -> None:
    application = RuleBasedWritingApplication()
    application.bind_rule_runtime()
    application.observe_rule_status(
        {
            "started": True,
            "session_state": "DEBUG_ONLY",
            "readiness_blockers": {
                "rig.session": (
                    "Session state=DEBUG_ONLY; runtime Home and hardware arm admission "
                    "are required before rule collection."
                )
            },
        }
    )

    application.submit_intent(RuleBasedWritingIntent.DEVICE_RETRY)
    application.consume_runtime_events()
    application.observe_runtime_publication({"startup_blockers": {}})

    state = application.snapshot()
    assert state["phase"] == "blocked"
    assert state["can_start"] is False
    assert state["can_runtime_home"] is True


def test_rule_status_does_not_clear_independent_startup_blockers() -> None:
    application = RuleBasedWritingApplication()
    application.bind_rule_runtime()
    application.observe_runtime_publication(
        {
            "startup_blockers": {
                "camera.front": "camera did not produce a fresh frame",
            },
        }
    )
    application.observe_rule_status(
        {
            "started": True,
            "session_state": "ARMED",
            "readiness_blockers": {},
            "rule_start_available": True,
        }
    )

    state = application.snapshot()

    assert state["startup_blockers"] == {
        "camera.front": "camera did not produce a fresh frame",
    }
    assert state["can_start"] is False


def test_rule_configuration_validates_all_independent_combinations_and_numeric_speed() -> None:
    application = RuleBasedWritingApplication()
    application.bind_rule_runtime()
    application.observe_rule_status({"rule_start_available": True})
    options = {
        "configuration_core_fields": [
            "pre_pen_down_replay_candidate_id",
            "pre_pen_down_replay_normal_episodes",
            "pen_down_candidate_id",
            "write_candidate_id",
            "pen_up_candidate_id",
            "trajectory_id",
            "direction",
            "writing_size_m",
            "initial_alignment_speed_scale",
        ],
        "choice_fields": {
            "pre_pen_down_replay_candidate_id": "pre_pen_down_replay_candidates",
        },
        "pre_pen_down_replay_candidates": ["", "a", "b"],
        "source_candidates_by_phase": {
            phase: [
                {"value": "a", "label": "A"},
                {"value": "b", "label": "B"},
            ]
            for phase in ("pen_down", "write", "pen_up")
        },
        "trajectory_ids": ["digit_1_v1", "digit_9_v1"],
        "directions": ["up", "left"],
        "numeric_bounds": {
            "pre_pen_down_replay_normal_episodes": [0, 10000],
            "writing_size_m": [0.005, 0.2],
            "initial_alignment_speed_scale": [0.05, 1.0],
        },
        "integer_fields": ["pre_pen_down_replay_normal_episodes"],
    }
    selected = {
        "pre_pen_down_replay_candidate_id": "",
        "pre_pen_down_replay_normal_episodes": 0,
        "pen_down_candidate_id": "a",
        "write_candidate_id": "a",
        "pen_up_candidate_id": "a",
        "trajectory_id": "digit_1_v1",
        "direction": "up",
        "writing_size_m": 0.08,
        "initial_alignment_speed_scale": 1.0,
    }
    application.configure_rule_options(options, selected)
    changed = {
        **selected,
        "pre_pen_down_replay_candidate_id": "b",
        "pre_pen_down_replay_normal_episodes": 3,
        "pen_down_candidate_id": "b",
        "pen_up_candidate_id": "b",
        "trajectory_id": "digit_9_v1",
        "direction": "left",
        "initial_alignment_speed_scale": 0.35,
    }

    assert application.submit_configuration(changed) is True
    assert application.snapshot()["can_start"] is False
    assert application.consume_configuration_requests() == (changed,)
    application.complete_configuration(selected=changed, error=None)
    application.observe_rule_status({"rule_start_available": True})
    state = application.snapshot()
    assert state["rule_configuration"] == changed
    assert state["can_start"] is True

    assert application.submit_configuration(
        {**changed, "pre_pen_down_replay_normal_episodes": 1.5}
    ) is False
    assert "必须是整数" in application.snapshot()["configuration_error"]


def test_rule_configuration_rejects_out_of_range_speed_without_queueing() -> None:
    application = RuleBasedWritingApplication()
    application.bind_rule_runtime()
    application.observe_rule_status({"rule_start_available": True})
    options = {
        "source_candidates_by_phase": {
            phase: ["only"] for phase in ("pen_down", "write", "pen_up")
        },
        "trajectory_ids": ["digit_1_v1"],
        "directions": ["up"],
        "numeric_bounds": {
            "writing_size_m": [0.005, 0.2],
            "initial_alignment_speed_scale": [0.05, 1.0],
        },
    }
    selected = {
        "pen_down_candidate_id": "only",
        "write_candidate_id": "only",
        "pen_up_candidate_id": "only",
        "trajectory_id": "digit_1_v1",
        "direction": "up",
        "writing_size_m": 0.08,
        "initial_alignment_speed_scale": 1.0,
    }
    application.configure_rule_options(options, selected)

    assert application.submit_configuration(
        {**selected, "initial_alignment_speed_scale": 1.5}
    ) is False
    assert application.consume_configuration_requests() == ()
    assert "[0.05, 1.0]" in application.snapshot()["configuration_error"]


def test_rule_configuration_accepts_an_ordered_bounded_batch() -> None:
    application = RuleBasedWritingApplication()
    application.bind_rule_runtime()
    application.observe_rule_status({"rule_start_available": True})
    options = {
        "source_candidates_by_phase": {
            phase: ["a", "b"] for phase in ("pen_down", "write", "pen_up")
        },
        "trajectory_ids": ["digit_1_v1", "digit_2_v1"],
        "directions": ["up"],
        "numeric_bounds": {
            "writing_size_m": [0.005, 0.2],
            "initial_alignment_speed_scale": [0.05, 1.0],
        },
    }
    first = {
        "pen_down_candidate_id": "a",
        "write_candidate_id": "a",
        "pen_up_candidate_id": "a",
        "trajectory_id": "digit_1_v1",
        "direction": "up",
        "writing_size_m": 0.08,
        "initial_alignment_speed_scale": 1.0,
    }
    second = {
        **first,
        "write_candidate_id": "b",
        "trajectory_id": "digit_2_v1",
        "initial_alignment_speed_scale": 0.5,
    }
    application.configure_rule_options(options, first)

    request = {"batch": [first, second]}
    assert application.submit_configuration(request) is True
    assert application.consume_configuration_requests() == (request,)
    application.complete_configuration(selected=request, error=None)
    application.observe_rule_status(
        {
            "rule_start_available": True,
            "rule_configuration": first,
            "rule_configuration_batch": [first, second],
            "batch_index": 0,
            "batch_count": 2,
            "batch_completed_count": 0,
        }
    )

    state = application.snapshot()
    assert state["rule_configuration"] == first
    assert state["rule_configuration_batch"] == [first, second]
    assert state["batch_count"] == 2
    assert state["can_start"] is True


def test_full_replay_configuration_accepts_only_complete_episode_choices() -> None:
    application = RuleBasedWritingApplication()
    application.bind_rule_runtime()
    application.observe_rule_status({"rule_start_available": True})
    options = {
        "collection_mode": "full_trajectory_replay",
        "configuration_core_fields": [
            "full_replay_candidate_id",
            "initial_alignment_speed_scale",
        ],
        "choice_fields": {
            "full_replay_candidate_id": "full_replay_candidates",
        },
        "full_replay_candidates": ["episode-a", "episode-b"],
        "numeric_bounds": {
            "initial_alignment_speed_scale": [0.05, 1.0],
        },
    }
    selected = {
        "full_replay_candidate_id": "episode-a",
        "initial_alignment_speed_scale": 0.5,
    }
    application.configure_rule_options(options, selected)

    assert application.snapshot()["collection_mode"] == "full_trajectory_replay"
    assert application.submit_configuration(
        {
            "full_replay_candidate_id": "episode-b",
            "initial_alignment_speed_scale": 0.25,
        }
    ) is True
    assert application.consume_configuration_requests() == (
        {
            "full_replay_candidate_id": "episode-b",
            "initial_alignment_speed_scale": 0.25,
        },
    )
    application.complete_configuration(selected=selected, error=None)
    assert application.submit_configuration(
        {
            "full_replay_candidate_id": "outside-pool",
            "initial_alignment_speed_scale": 0.25,
        }
    ) is False
    assert "不是可用选项" in application.snapshot()["configuration_error"]


@pytest.mark.parametrize("terminal_key", ("completed", "aborted"))
def test_stopped_terminal_status_keeps_configuration_enabled(
    terminal_key: str,
) -> None:
    application = RuleBasedWritingApplication()
    application.bind_rule_runtime()
    options = {
        "source_candidates_by_phase": {
            phase: ["only"] for phase in ("pen_down", "write", "pen_up")
        },
        "trajectory_ids": ["digit_1_v1"],
        "directions": ["up"],
        "numeric_bounds": {
            "writing_size_m": [0.005, 0.2],
            "initial_alignment_speed_scale": [0.05, 1.0],
        },
    }
    selected = {
        "pen_down_candidate_id": "only",
        "write_candidate_id": "only",
        "pen_up_candidate_id": "only",
        "trajectory_id": "digit_1_v1",
        "direction": "up",
        "writing_size_m": 0.08,
        "initial_alignment_speed_scale": 1.0,
    }
    application.configure_rule_options(options, selected)
    application.observe_rule_status(
        {
            terminal_key: True,
            "held": True,
            "configuration_available": True,
            "rule_start_available": False,
            "readiness_blockers": {"rig.session": "runtime Home required"},
        }
    )

    state = application.snapshot()
    assert state[terminal_key] is True
    assert state["can_configure"] is True
    assert state["can_start"] is False
    assert state["can_runtime_home"] is False
    changed = {**selected, "initial_alignment_speed_scale": 0.5}
    assert application.submit_configuration(changed) is True
    assert application.consume_configuration_requests() == (changed,)


def test_writer_prepare_failure_blocks_only_collection_and_retries_writer_once() -> None:
    application = RuleBasedWritingApplication()
    application.bind_rule_runtime()
    application.observe_rule_status({"rule_start_available": True})

    assert application.submit_intent(RuleBasedWritingIntent.START) is True
    assert application.consume_runtime_events() == ["episode_start"]
    assert application.consume_rule_intents() == ("rule_start",)

    warning = "正式 dataset 预备失败: TactileBundleStorageError: insufficient free space"
    application.publish_writer_prepare_result(error=warning)
    state = application.snapshot()
    assert state["phase"] == "blocked"
    assert state["writer_blocked"] is True
    assert state["writer_warning"] == warning
    assert state["can_start"] is True
    assert state["can_configure"] is True
    assert state["can_stop"] is True
    assert state["shutdown"] is False

    assert application.submit_intent(RuleBasedWritingIntent.START) is True
    assert application.consume_runtime_events() == ["episode_start"]
    assert application.consume_rule_intents() == ()
    assert application.snapshot()["writer_retry_pending"] is True

    application.publish_writer_prepare_result(error=None)
    state = application.snapshot()
    assert state["phase"] == "armed"
    assert state["writer_blocked"] is False
    assert state["writer_retry_pending"] is False


def test_writer_blocked_configuration_validates_runtime_fields_and_stays_blocked() -> None:
    application = RuleBasedWritingApplication()
    application.bind_rule_runtime()
    application.observe_rule_status({"rule_start_available": True})
    options = {
        "source_candidates_by_phase": {
            phase: ["only"] for phase in ("pen_down", "write", "pen_up")
        },
        "trajectory_ids": ["digit_1_v1"],
        "directions": ["up"],
        "runtime_configuration_fields": [
            "writing_speed_m_s",
            "tactile_bundle_root",
            "tactile_bundle_min_free_gb",
            "tactile_compression",
            "writer_workers",
            "writer_queue_size",
            "collection_process_queue_size",
            "storage_queue_size",
            "collection_warning_hz",
            "write_parquet",
            "write_video",
        ],
        "integer_fields": [
            "writer_workers",
            "writer_queue_size",
            "collection_process_queue_size",
            "storage_queue_size",
        ],
        "select_fields": {"tactile_compression": ["none", "lz4", "zstd"]},
        "boolean_fields": ["write_parquet", "write_video"],
        "text_fields": ["tactile_bundle_root"],
        "numeric_bounds": {
            "writing_size_m": [0.005, 0.2],
            "initial_alignment_speed_scale": [0.05, 1.0],
            "writing_speed_m_s": [0.001, 0.2],
            "tactile_bundle_min_free_gb": [0.0, 4096.0],
            "writer_workers": [1, 32],
            "writer_queue_size": [1, 65536],
            "collection_process_queue_size": [1, 1024],
            "storage_queue_size": [0, 65536],
            "collection_warning_hz": [0.1, 120.0],
        },
    }
    selected = {
        "pen_down_candidate_id": "only",
        "write_candidate_id": "only",
        "pen_up_candidate_id": "only",
        "trajectory_id": "digit_1_v1",
        "direction": "up",
        "writing_size_m": 0.08,
        "writing_speed_m_s": 0.08,
        "initial_alignment_speed_scale": 1.0,
        "tactile_bundle_root": "/data/tactile",
        "tactile_bundle_min_free_gb": 40.0,
        "tactile_compression": "lz4",
        "writer_workers": 4,
        "writer_queue_size": 512,
        "collection_process_queue_size": 4,
        "storage_queue_size": 64,
        "collection_warning_hz": 15.0,
        "write_parquet": True,
        "write_video": True,
    }
    application.configure_rule_options(options, selected)
    application.publish_writer_prepare_result(error="磁盘余量不足")

    assert application.submit_configuration({**selected, "writer_workers": 2.5}) is False
    assert "必须是整数" in application.snapshot()["configuration_error"]
    changed = {
        **selected,
        "tactile_bundle_min_free_gb": 20.0,
        "writer_workers": 2,
        "write_video": False,
    }
    assert application.submit_configuration(changed) is True
    assert application.consume_configuration_requests() == (changed,)
    application.complete_configuration(selected=changed, error=None)
    state = application.snapshot()
    assert state["phase"] == "blocked"
    assert state["can_configure"] is True
    assert state["can_start"] is True
    assert state["rule_configuration"] == changed


def test_manual_gripper_pending_blocks_start_and_reverse_action() -> None:
    application = RuleBasedWritingApplication()
    application.bind_rule_runtime()
    application.observe_rule_status(
        {
            "rule_start_available": True,
            "gripper_control_available": True,
            "gripper_sides": ["right"],
        }
    )

    assert application.submit_intent(RuleBasedWritingIntent.CLOSE_GRIPPERS) is True
    state = application.snapshot()
    assert state["gripper_action_pending"] is True
    assert state["can_start"] is False
    assert state["can_release_grippers"] is False
    assert state["can_close_grippers"] is False
    assert application.submit_intent(RuleBasedWritingIntent.RELEASE_GRIPPERS) is False
    assert application.submit_intent(RuleBasedWritingIntent.START) is False


def test_task_chain_advances_one_gate_at_a_time_and_reuses_existing_queues() -> None:
    application = RuleBasedWritingApplication(task_chain_choice_index=lambda count: count - 1)
    options, first, second = _task_chain_configuration()
    application.bind_rule_runtime()
    application.configure_rule_options(options, first)
    application.observe_rule_status(_task_chain_ready_status())

    batch = {"batch": [first, second]}
    assert application.submit_configuration(batch) is True
    assert application.consume_configuration_requests() == (batch,)
    application.complete_configuration(selected=batch, error=None)
    application.observe_rule_status(_task_chain_ready_status())
    assert application.snapshot()["task_chain_can_advance"] is True

    assert application.submit_task_chain_advance({"event": "advance"}) is True
    assert application.consume_rule_intents() == ("release_grippers",)
    assert application.snapshot()["task_chain_phase"] == "initial_release_pending"
    application.observe_rule_status(
        _task_chain_ready_status(
            gripper_action_pending=True,
            gripper_action_status="已排队：松开夹爪",
        )
    )
    application.observe_rule_status(
        _task_chain_ready_status(
            gripper_action_pending=False,
            gripper_action_status="夹爪实测已松开",
        )
    )
    assert application.snapshot()["task_chain_phase"] == "ready_close"

    assert application.submit_task_chain_advance({"event": "advance"}) is True
    assert application.consume_rule_intents() == ("close_right_gripper",)
    application.observe_rule_status(
        _task_chain_ready_status(
            gripper_action_pending=False,
            gripper_action_status="夹爪实测已松开",
        )
    )
    assert application.snapshot()["task_chain_phase"] == "close_pending"
    application.observe_rule_status(
        _task_chain_ready_status(
            gripper_action_pending=True,
            gripper_action_status="已排队：闭紧夹爪",
        )
    )
    application.observe_rule_status(
        _task_chain_ready_status(
            gripper_action_pending=False,
            gripper_action_status="夹爪实测已闭紧",
        )
    )
    assert application.snapshot()["task_chain_phase"] == "ready_height"

    assert application.submit_task_chain_advance(
        {
            "event": "advance",
            "lower_tolerance_mm": 1.5,
            "upper_tolerance_mm": 2.0,
        }
    ) is True
    assert application.consume_height_calibration_requests() == (
        {
            "operation": "capture",
            "lower_tolerance_m": pytest.approx(0.0015),
            "upper_tolerance_m": pytest.approx(0.002),
        },
    )
    application.complete_height_calibration(error=None)
    assert application.snapshot()["task_chain_phase"] == "ready_rule"

    assert application.submit_task_chain_advance({"event": "advance"}) is True
    assert application.consume_configuration_requests() == (second,)
    assert application.snapshot()["task_chain_phase"] == "configuration_pending"
    application.complete_configuration(selected=second, error=None)
    assert application.consume_rule_intents() == ("rule_start",)
    assert application.consume_runtime_events() == ["episode_start"]
    state = application.snapshot()
    assert state["task_chain_phase"] == "rule_pending"
    assert state["task_chain_selected_configuration"] == second
    assert state["task_chain_selected_configuration_index"] == 1

    application.observe_rule_status(
        _task_chain_ready_status(
            armed=True,
            execution_begun=True,
            session_state="RUNNING",
            rule_start_available=False,
            configuration_available=False,
            height_calibration_available=False,
            batch_count=1,
            batch_completed_count=0,
        )
    )
    assert application.snapshot()["task_chain_phase"] == "rule_pending"
    application.observe_rule_status(
        _task_chain_ready_status(
            batch_count=1,
            batch_completed_count=1,
            batch_cycle_complete=True,
        )
    )
    assert application.snapshot()["task_chain_phase"] == "ready_final_release"

    assert application.submit_task_chain_advance({"event": "advance"}) is True
    assert application.consume_rule_intents() == ("release_grippers",)
    application.observe_rule_status(
        _task_chain_ready_status(
            gripper_action_pending=True,
            gripper_action_status="已排队：松开夹爪",
            batch_count=1,
            batch_completed_count=1,
            batch_cycle_complete=True,
        )
    )
    application.observe_rule_status(
        _task_chain_ready_status(
            gripper_action_pending=False,
            gripper_action_status="夹爪实测已松开",
            batch_count=1,
            batch_completed_count=1,
            batch_cycle_complete=True,
        )
    )
    state = application.snapshot()
    assert state["task_chain_phase"] == "complete"
    assert state["task_chain_cycle_count"] == 1
    assert state["task_chain_can_advance"] is True


def test_task_chain_starts_after_stale_start_gate_recovers() -> None:
    application = RuleBasedWritingApplication()
    options, first, _second = _task_chain_configuration()
    application.bind_rule_runtime()
    application.configure_rule_options(options, first)
    application.observe_rule_status(
        _task_chain_ready_status(
            rule_start_available=False,
            configuration_available=True,
        )
    )

    # Model the asynchronous configuration callback: the previous status
    # receipt still says that START is unavailable, even though validation has
    # succeeded.  This must not become a configuration failure or reconfigure
    # the same choice on the next click.
    application._task_chain.phase = RuleWritingTaskChainPhase.CONFIGURATION_PENDING
    application.complete_configuration(selected=first, error=None)
    state = application.snapshot()
    assert state["task_chain_phase"] == "ready_rule"
    assert state["task_chain_can_advance"] is False
    assert "配置已验证" in state["task_chain_detail"]
    assert application.consume_rule_intents() == ()
    assert application.consume_configuration_requests() == ()

    # The generic runtime can still publish its ARMED/no-command diagnostic
    # while this validated chain waits for the dedicated START receipt.  That
    # projection is expected here and must not strand the chain in READY_RULE.
    application.observe_runtime_publication(
        {
            "startup_blockers": {
                "rig.session": (
                    "Session state=ARMED; runtime Home and hardware arm admission "
                    "are required before rule collection."
                )
            }
        }
    )

    application.observe_rule_status(_task_chain_ready_status())
    assert application.consume_rule_intents() == ("rule_start",)
    assert application.consume_runtime_events() == ["episode_start"]
    assert application.snapshot()["task_chain_phase"] == "rule_pending"

    # Later status heartbeats must not submit a second START for the same
    # validated chain selection.
    application.observe_rule_status(_task_chain_ready_status())
    assert application.consume_rule_intents() == ()
    assert application.consume_runtime_events() == []


def test_task_chain_does_not_offer_rule_retry_while_device_is_blocked() -> None:
    application = RuleBasedWritingApplication()
    options, first, _second = _task_chain_configuration()
    application.bind_rule_runtime()
    application.configure_rule_options(options, first)
    application.observe_rule_status(_task_chain_ready_status())

    assert application.submit_configuration({"batch": [first]}) is True
    assert application.consume_configuration_requests() == ({"batch": [first]},)
    application._task_chain.phase = RuleWritingTaskChainPhase.CONFIGURATION_PENDING
    application.complete_configuration(selected={"batch": [first]}, error=None)
    assert application.snapshot()["task_chain_phase"] == "rule_pending"

    application.observe_rule_status(
        _task_chain_ready_status(
            aborted=True,
            held=True,
            session_state="DEBUG_ONLY",
            rule_start_available=False,
            readiness_blockers={"right_hand": "DMTAC fault"},
        )
    )
    state = application.snapshot()
    assert state["task_chain_phase"] == "ready_rule"
    assert state["task_chain_can_advance"] is False
    assert "DMTAC fault" in state["task_chain_detail"]


def test_task_chain_rejects_manual_step_replacement_and_does_not_skip_failure() -> None:
    application = RuleBasedWritingApplication()
    options, first, _second = _task_chain_configuration()
    application.bind_rule_runtime()
    application.configure_rule_options(options, first)
    application.observe_rule_status(_task_chain_ready_status())

    assert application.submit_task_chain_advance({"event": "advance"}) is True
    assert application.submit_intent(RuleBasedWritingIntent.START) is False
    assert application.submit_intent(RuleBasedWritingIntent.CLOSE_GRIPPERS) is False
    assert application.submit_configuration(first) is False
    assert application.submit_height_calibration({"operation": "capture"}) is False

    application.consume_rule_intents()
    application.observe_rule_status(
        _task_chain_ready_status(
            gripper_action_pending=False,
            gripper_action_status="夹爪控制超时",
        )
    )
    state = application.snapshot()
    assert state["task_chain_phase"] == "idle"
    assert "夹爪控制超时" in state["task_chain_detail"]
    assert state["task_chain_cycle_count"] == 0


def test_task_chain_waits_for_writer_commit_and_stops_on_rule_hold() -> None:
    application = RuleBasedWritingApplication()
    options, first, _second = _task_chain_configuration()
    application.bind_rule_runtime()
    application.configure_rule_options(options, first)
    application.observe_rule_status(_task_chain_ready_status())

    application._task_chain.phase = RuleWritingTaskChainPhase.CONFIGURATION_PENDING
    application.complete_configuration(selected=first, error=None)
    application.consume_rule_intents()
    application.consume_runtime_events()
    application.observe_rule_status(
        _task_chain_ready_status(
            armed=True,
            execution_begun=True,
            rule_start_available=False,
            configuration_available=False,
            height_calibration_available=False,
        )
    )
    application.observe_rule_status(
        _task_chain_ready_status(
            completed=True,
            rule_start_available=False,
            configuration_available=False,
            height_calibration_available=False,
            batch_completed_count=0,
        )
    )
    assert application.snapshot()["task_chain_phase"] == "rule_pending"

    application.observe_rule_status(
        _task_chain_ready_status(
            aborted=True,
            held=True,
            rule_start_available=False,
            configuration_available=False,
            height_calibration_available=False,
        )
    )
    state = application.snapshot()
    assert state["task_chain_phase"] == "ready_rule"
    assert state["task_chain_can_advance"] is False
    assert "中止" in state["task_chain_detail"]

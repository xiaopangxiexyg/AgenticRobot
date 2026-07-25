from __future__ import annotations

from itertools import product

from teleop_runtime.gui.calibration_web_ui import CalibrationWebUi
from teleop_runtime.real_teleop.gui_controls import (
    GuiControlEvents,
    GuiControlState,
    current_active_controls,
    episode_busy,
)


def test_gui_control_policy_matrix_keeps_lifecycle_operations_disjoint() -> None:
    """Exercise the combinations that previously produced grey/dead buttons."""

    for (
        teleop_started,
        hold_active,
        recording_active,
        recording_paused,
        countdown_active,
        finalizing_active,
        qpos_active,
        fix_hold_active,
        foot_pad_available,
        foot_pad_test_active,
    ) in product((False, True), repeat=10):
        state = GuiControlState(
            teleop_started=teleop_started,
            hold_active=hold_active,
            episode_recording_available=True,
            episode_recording_active=recording_active,
            recording_paused=recording_paused,
            episode_countdown_active=countdown_active,
            episode_finalizing_active=finalizing_active,
            fix_arm_available=True,
            init_arm_available=True,
            init_arm_qpos_active=qpos_active,
            fix_hold_active=fix_hold_active,
            foot_pad_available=foot_pad_available,
            foot_pad_count=2 if foot_pad_available else 0,
            foot_pad_test_active=foot_pad_test_active,
        )
        controls = set(current_active_controls(state, GuiControlEvents()))

        if episode_busy(state):
            assert "episode_start" not in controls
            assert "device_retry" not in controls
            assert "foot_pad_test_start" not in controls
            assert "movement_multiple_up" not in controls
            assert "gripper_force_set" not in controls
        if teleop_started and not hold_active:
            assert "device_retry" not in controls
            assert "foot_pad_test_start" not in controls
        if not foot_pad_available:
            assert "foot_pad_test_start" not in controls
            assert "foot_pad_test_stop" not in controls
        elif foot_pad_test_active:
            assert "foot_pad_test_stop" in controls
            assert "foot_pad_test_start" not in controls
        if qpos_active:
            assert "init_qpos_emergency_stop" in controls


def test_web_ui_one_shot_reconnect_and_episode_events_cannot_double_queue() -> None:
    ui = CalibrationWebUi(open_browser=False)
    try:
        ui.publish(active_controls=["q", "device_retry"])
        assert ui.push_event("device_retry") is True
        assert ui.push_event("device_retry") is False
        assert ui.consume_events() == ["device_retry"]

        ui.publish(active_controls=["episode_start", "episode_stop"])
        assert ui.push_event("episode_start") is True
        assert ui.push_event("episode_start") is False
        assert ui.consume_events() == ["episode_start"]
    finally:
        ui.close()


def test_web_ui_rejects_payloads_for_inactive_controls_without_queueing() -> None:
    ui = CalibrationWebUi(open_browser=False)
    try:
        ui.publish(active_controls=["q"])
        assert ui.push_event("fix_arm_select:left") is False
        assert ui.push_event("movement_multiple_set:2.0") is False
        assert ui.consume_events() == []
    finally:
        ui.close()


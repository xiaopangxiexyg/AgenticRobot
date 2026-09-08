from __future__ import annotations

from types import SimpleNamespace

from robot_motion.runtime import run_best_effort_cleanup

from control_station.real_teleop.control_availability import (
    GuiControlEvents,
    GuiControlState,
    current_active_controls,
    episode_countdown_controls,
    hold_controls,
    pre_home_controls,
    teleop_controls,
)
from teleop_runtime.collection.camera_preview import CameraPreviewer
from teleop_runtime.debug import run_vive_rm65_real_teleop as legacy_entrypoint
from teleop_runtime.real_teleop import operator_control_runtime, terminal_input
from teleop_runtime.real_teleop.components import RealTeleopComponents
from teleop_runtime.real_teleop.movement import apply_movement_multiple


def test_operator_control_uses_canonical_lower_runtime_dependencies() -> None:
    assert operator_control_runtime.CameraPreviewer is CameraPreviewer
    assert operator_control_runtime.RealTeleopComponents is RealTeleopComponents
    assert operator_control_runtime._apply_movement_multiple is apply_movement_multiple
    assert (
        operator_control_runtime.run_best_effort_cleanup
        is run_best_effort_cleanup
    )


def test_legacy_control_input_delegates_active_controls_to_extracted_policy() -> None:
    control = legacy_entrypoint.RealTeleopControlInput.__new__(
        legacy_entrypoint.RealTeleopControlInput
    )
    control.args = SimpleNamespace(
        no_collection=False,
        no_episode_recording=False,
        camera_visualization_available=True,
        tactile_visualization_available=False,
    )
    control.components = SimpleNamespace(
        active_sides=("left",),
        fix_arm_controller=object(),
        init_arm_controller=None,
    )
    control._teleop_started = True
    control._hold_active = True
    control._hardware_hold_confirmed = True
    control._episode_recording_ui_active = False
    control._episode_countdown_ui_active = False
    control._episode_finalizing_ui_active = False
    control._pose_snapshot_recorder = object()

    state = GuiControlState(
        teleop_started=True,
        hold_active=True,
        hardware_hold_confirmed=True,
        episode_recording_available=True,
        camera_visualization_available=True,
        fix_arm_available=True,
        pose_snapshot_available=True,
        active_side_count=1,
    )
    assert control._current_active_controls() == current_active_controls(
        state,
        GuiControlEvents(),
    )


def test_extracted_gui_controls_preserve_pre_home_and_running_control_order() -> None:
    events = GuiControlEvents()
    ready = GuiControlState(
        episode_recording_available=True,
        camera_visualization_available=True,
        tactile_visualization_available=True,
        fix_arm_available=True,
        init_arm_available=True,
        pose_snapshot_available=True,
        active_side_count=2,
    )
    assert pre_home_controls(ready, events) == [
        "p",
        "w",
        "d",
        "q",
        "fix_arm_select",
        "init_arm_select",
        "init_arm_qpos_speed_set",
        "toggle_camera_vis",
        "toggle_tactile_vis",
        "calibrate_vive_coordinate_profile",
        "save_vive_calibration_profile",
        "load_vive_calibration_profile",
        "movement_multiple_down",
        "movement_multiple_up",
        "movement_multiple_reset",
        "movement_multiple_set",
        "task_select",
        "round_trip_toggle",
        "gripper_force_set",
        "gripper_force_persist",
    ]

    running = GuiControlState(
        **{
            **ready.__dict__,
            "teleop_started": True,
        }
    )
    assert teleop_controls(running, events) == [
        "q",
        "d",
        "h",
        "c",
        "fix_arm_toggle",
        "fix_arm_select",
        "g",
        "init_arm_toggle",
        "init_arm_select",
        "init_arm_qpos_speed_set",
        "init_arm_move_to_qpos",
        "init_arm_move_both_to_qpos",
        "record_pose_snapshot",
        "episode_start",
        "episode_stop",
        "task_select",
        "round_trip_toggle",
        "gripper_force_set",
        "gripper_force_persist",
        "toggle_camera_vis",
        "toggle_tactile_vis",
        "save_vive_calibration_profile",
    ]

    no_recording = GuiControlState()
    assert "task_select" in pre_home_controls(no_recording, events)
    assert "round_trip_toggle" not in pre_home_controls(no_recording, events)
    assert "gripper_force_set" in pre_home_controls(no_recording, events)


def test_armed_lifecycle_without_live_dispatch_exposes_only_recovery_controls() -> None:
    state = GuiControlState(
        teleop_started=True,
        teleop_dispatch_confirmed=False,
        episode_recording_available=True,
        camera_visualization_available=True,
        tactile_visualization_available=True,
        fix_arm_available=True,
        init_arm_available=True,
        pose_snapshot_available=True,
        foot_pad_available=True,
        foot_pad_count=2,
        active_side_count=2,
    )

    assert current_active_controls(state, GuiControlEvents()) == ["q", "d", "h"]


def test_application_hold_without_lower_receipt_exposes_only_fail_safe_controls() -> None:
    state = GuiControlState(
        teleop_started=True,
        hold_active=True,
        hardware_hold_confirmed=False,
        episode_recording_available=True,
        camera_visualization_available=True,
        tactile_visualization_available=True,
        tactile_solving_available=True,
        fix_arm_available=True,
        init_arm_available=True,
        pose_snapshot_available=True,
        foot_pad_available=True,
        foot_pad_count=2,
        active_side_count=1,
    )

    assert current_active_controls(state, GuiControlEvents()) == ["q", "d", "h"]


def test_recording_without_live_dispatch_keeps_stop_and_hold_but_no_new_motion() -> None:
    state = GuiControlState(
        teleop_started=True,
        teleop_dispatch_confirmed=False,
        episode_recording_available=True,
        episode_recording_active=True,
        fix_arm_available=True,
        init_arm_available=True,
        pose_snapshot_available=True,
        active_side_count=2,
    )

    controls = current_active_controls(state, GuiControlEvents())
    assert controls == ["q", "d", "episode_stop", "h"]
    assert "c" not in controls
    assert "g" not in controls
    assert "record_pose_snapshot" not in controls


def test_pre_home_exact_qpos_controls_require_live_startup_executor() -> None:
    events = GuiControlEvents()
    unavailable = GuiControlState(
        init_arm_available=True,
        init_arm_qpos_move_available=True,
    )
    assert "init_arm_move_to_qpos" not in pre_home_controls(unavailable, events)
    assert "init_arm_move_both_to_qpos" not in pre_home_controls(unavailable, events)

    available = GuiControlState(
        init_arm_available=True,
        init_arm_qpos_move_available=True,
        startup_qpos_maintenance_enabled=True,
    )
    controls = pre_home_controls(available, events)
    assert "init_arm_move_to_qpos" in controls
    assert "init_arm_move_both_to_qpos" in controls


def test_extracted_gui_controls_preserve_countdown_and_single_arm_hold_boundaries() -> None:
    events = GuiControlEvents()
    countdown = GuiControlState(
        teleop_started=True,
        episode_recording_available=True,
        episode_countdown_active=True,
        camera_visualization_available=True,
        pose_snapshot_available=True,
        active_side_count=2,
    )
    assert episode_countdown_controls(countdown, events) == [
        "q",
        "d",
        "h",
        "episode_stop",
    ]
    assert current_active_controls(countdown, events) == episode_countdown_controls(countdown, events)

    held_single = GuiControlState(
        teleop_started=True,
        hold_active=True,
        episode_recording_available=True,
        fix_arm_available=True,
        init_arm_available=True,
        active_side_count=1,
    )
    assert hold_controls(held_single, events) == [
        "p",
        "d",
        "fix_arm_select",
        "init_arm_select",
        "init_arm_qpos_speed_set",
        "init_arm_move_to_qpos",
        "init_arm_move_both_to_qpos",
        "w",
        "episode_stop",
        "episode_delete",
        "q",
        "calibrate_vive_coordinate_profile",
        "save_vive_calibration_profile",
        "load_vive_calibration_profile",
        "movement_multiple_down",
        "movement_multiple_up",
        "movement_multiple_reset",
        "movement_multiple_set",
        "task_select",
        "round_trip_toggle",
        "gripper_force_set",
        "gripper_force_persist",
    ]


def test_device_retry_is_only_available_before_home_or_in_regular_hold() -> None:
    events = GuiControlEvents()
    assert "device_retry" in current_active_controls(GuiControlState(), events)
    assert "device_retry" not in current_active_controls(
        GuiControlState(teleop_started=True),
        events,
    )
    assert "device_retry" in current_active_controls(
        GuiControlState(
            teleop_started=True,
            hold_active=True,
            hardware_hold_confirmed=True,
        ),
        events,
    )
    assert "device_retry" not in current_active_controls(
        GuiControlState(
            teleop_started=True,
            hold_active=True,
            hardware_hold_confirmed=False,
        ),
        events,
    )


def test_visualization_toggles_are_disabled_for_every_episode_busy_phase() -> None:
    events = GuiControlEvents()
    for busy_state in (
        {"episode_countdown_active": True},
        {"episode_recording_active": True},
        {"episode_recording_active": True, "recording_paused": True},
        {"episode_finalizing_active": True},
    ):
        controls = current_active_controls(
            GuiControlState(
                teleop_started=True,
                camera_visualization_available=True,
                tactile_visualization_available=True,
                **busy_state,
            ),
            events,
        )
        assert "toggle_camera_vis" not in controls
        assert "toggle_tactile_vis" not in controls


def test_terminal_poll_bounds_continuous_keyboard_input(monkeypatch) -> None:
    class KeyboardSource:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool]] = []

        def set_button(self, name, value, **_kwargs) -> None:
            self.calls.append((str(name), bool(value)))

    fake_stdin = SimpleNamespace(read=lambda _count: "z", isatty=lambda: True)
    monkeypatch.setattr(terminal_input.sys, "stdin", fake_stdin)
    monkeypatch.setattr(
        terminal_input.select,
        "select",
        lambda *_args, **_kwargs: ([fake_stdin], [], []),
    )
    keyboard = KeyboardSource()
    terminal = terminal_input.TerminalClutchInput(keyboard)
    terminal.enabled = True

    terminal.poll()

    assert len(keyboard.calls) == terminal._MAX_CHARS_PER_POLL * 2


def test_exact_qpos_move_is_available_during_episode_recording_for_labeled_collection() -> None:
    state = GuiControlState(
        teleop_started=True,
        episode_recording_available=True,
        episode_recording_active=True,
        init_arm_available=True,
        init_arm_qpos_move_available=True,
        active_side_count=2,
    )

    controls = current_active_controls(state, GuiControlEvents())

    assert "init_arm_move_to_qpos" in controls
    assert "init_arm_toggle" in controls
    assert "init_arm_qpos_speed_set" not in controls
    assert "episode_stop" in controls


def test_exact_qpos_move_is_available_in_regular_hold_without_episode_start() -> None:
    state = GuiControlState(
        teleop_started=True,
        hold_active=True,
        episode_recording_available=True,
        init_arm_available=True,
        init_arm_qpos_move_available=True,
        active_side_count=2,
    )

    controls = current_active_controls(state, GuiControlEvents())

    assert "init_arm_move_to_qpos" in controls
    assert "init_arm_move_both_to_qpos" in controls
    assert "episode_start" not in controls


def test_exact_qpos_operation_keeps_home_recapture_hold_quit_and_debug_controls() -> None:
    state = GuiControlState(
        teleop_started=True,
        camera_visualization_available=True,
        tactile_visualization_available=True,
        init_arm_available=True,
        init_arm_qpos_active=True,
        active_side_count=2,
    )

    assert current_active_controls(state, GuiControlEvents()) == [
        "q",
        "d",
        "h",
        "init_qpos_emergency_stop",
        "g",
        "init_arm_toggle",
        "toggle_camera_vis",
        "toggle_tactile_vis",
    ]


def test_startup_exact_qpos_hides_runtime_hold_but_keeps_operation_stop() -> None:
    state = GuiControlState(
        teleop_started=False,
        init_arm_available=True,
        init_arm_qpos_active=True,
        startup_qpos_maintenance_enabled=True,
        active_side_count=2,
    )

    controls = current_active_controls(state, GuiControlEvents())

    assert "h" not in controls
    assert "init_qpos_emergency_stop" in controls
    assert "q" in controls


def test_impedance_gui_exposes_exact_qpos_only_with_capable_dual_backends() -> None:
    class CapableBackend:
        def begin_exact_qpos_transit(self, _operation_id: str) -> None:
            pass

    control = legacy_entrypoint.RealTeleopControlInput.__new__(
        legacy_entrypoint.RealTeleopControlInput
    )
    control.args = SimpleNamespace(
        arm_control_mode="impedance",
        no_collection=False,
        no_episode_recording=False,
    )
    control.components = SimpleNamespace(
        active_sides=("left", "right"),
        fix_arm_controller=None,
        init_arm_controller=SimpleNamespace(qpos_active=False),
        collision_guard_requested=True,
        dual_arm_retargeter=object(),
        left_command_backend=CapableBackend(),
        right_command_backend=CapableBackend(),
    )
    control.foot_pad_input = None

    assert control._gui_control_state().init_arm_qpos_move_available

    control.components.right_command_backend = object()
    assert not control._gui_control_state().init_arm_qpos_move_available

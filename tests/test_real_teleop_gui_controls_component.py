from __future__ import annotations

from types import SimpleNamespace

from teleop_runtime.real_teleop.gui_controls import (
    GuiControlEvents,
    GuiControlState,
    current_active_controls,
    episode_countdown_controls,
    hold_controls,
    pre_home_controls,
    teleop_controls,
)
from teleop_runtime.debug import run_vive_rm65_real_teleop as legacy_entrypoint


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
    control._episode_recording_ui_active = False
    control._episode_countdown_ui_active = False
    control._episode_finalizing_ui_active = False
    control._pose_snapshot_recorder = object()

    state = GuiControlState(
        teleop_started=True,
        hold_active=True,
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
        "record_pose_snapshot",
        "episode_start",
        "episode_stop",
        "task_select",
        "round_trip_toggle",
        "gripper_force_set",
        "toggle_camera_vis",
        "toggle_tactile_vis",
        "save_vive_calibration_profile",
    ]

    no_recording = GuiControlState()
    assert "task_select" in pre_home_controls(no_recording, events)
    assert "round_trip_toggle" not in pre_home_controls(no_recording, events)
    assert "gripper_force_set" in pre_home_controls(no_recording, events)


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
        "toggle_camera_vis",
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
        "w",
        "episode_start",
        "episode_stop",
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
    ]

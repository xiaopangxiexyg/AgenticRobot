"""Pure GUI control-availability policy for real teleoperation.

This module maps an immutable runtime/UI snapshot to the controls that may be
shown as active.  It neither consumes events nor invokes robot, GUI, recorder,
or calibration operations; those stateful responsibilities remain in the
legacy real-teleop control-input coordinator.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuiControlEvents:
    episode_events: tuple[str, ...] = ("episode_start", "episode_stop")
    pose_snapshot_event: str = "record_pose_snapshot"
    gripper_force_event: str = "gripper_force_set"
    vive_calibration_events: tuple[str, ...] = (
        "calibrate_vive_coordinate_profile",
        "save_vive_calibration_profile",
        "load_vive_calibration_profile",
    )
    movement_multiple_events: tuple[str, ...] = (
        "movement_multiple_down",
        "movement_multiple_up",
        "movement_multiple_reset",
        "movement_multiple_set",
    )


@dataclass(frozen=True)
class GuiControlState:
    teleop_started: bool = False
    hold_active: bool = False
    episode_recording_available: bool = False
    episode_recording_active: bool = False
    episode_countdown_active: bool = False
    episode_finalizing_active: bool = False
    camera_visualization_available: bool = False
    tactile_visualization_available: bool = False
    fix_arm_available: bool = False
    init_arm_available: bool = False
    pose_snapshot_available: bool = False
    active_side_count: int = 0
    special_arm_control_mode: str = "manual"


def episode_busy(state: GuiControlState) -> bool:
    return bool(
        state.episode_recording_active
        or state.episode_countdown_active
        or state.episode_finalizing_active
    )


def visualization_controls(state: GuiControlState) -> list[str]:
    controls: list[str] = []
    if state.camera_visualization_available:
        controls.append("toggle_camera_vis")
    if state.tactile_visualization_available:
        controls.append("toggle_tactile_vis")
    return controls


def vive_calibration_controls(
    events: GuiControlEvents,
    *,
    allow_load: bool,
) -> list[str]:
    if not allow_load:
        return [events.vive_calibration_events[1]]
    return [
        events.vive_calibration_events[0],
        events.vive_calibration_events[1],
        events.vive_calibration_events[2],
    ]


def movement_multiple_controls(
    state: GuiControlState,
    events: GuiControlEvents,
) -> list[str]:
    if episode_busy(state):
        return []
    return list(events.movement_multiple_events)


def collection_task_controls(state: GuiControlState) -> list[str]:
    if episode_busy(state):
        return []
    controls = ["task_select"]
    if state.episode_recording_available:
        controls.append("round_trip_toggle")
    return controls


def gripper_force_controls(
    state: GuiControlState,
    events: GuiControlEvents,
) -> list[str]:
    """Allow direct gripper force changes only outside episode collection."""

    return [] if episode_busy(state) else [events.gripper_force_event]


def episode_controls(state: GuiControlState, events: GuiControlEvents) -> list[str]:
    if not state.episode_recording_available:
        return []
    if episode_busy(state):
        return ["episode_stop"] if (
            state.episode_recording_active or state.episode_countdown_active
        ) else []
    return list(events.episode_events)


def fix_init_allowed_in_controls(state: GuiControlState) -> bool:
    return bool(
        state.teleop_started
        and not state.hold_active
        and not state.episode_countdown_active
        and not state.episode_finalizing_active
    )


def fix_arm_controls(state: GuiControlState) -> list[str]:
    return ["fix_arm_select"] if (
        state.fix_arm_available and state.special_arm_control_mode == "manual"
    ) else []


def init_arm_controls(state: GuiControlState) -> list[str]:
    return ["init_arm_select"] if (
        state.init_arm_available and state.special_arm_control_mode == "manual"
    ) else []


def manual_special_arm_controls(state: GuiControlState) -> bool:
    return state.special_arm_control_mode == "manual"


def pose_snapshot_controls(state: GuiControlState, events: GuiControlEvents) -> list[str]:
    return [events.pose_snapshot_event] if state.pose_snapshot_available else []


def recording_controls(
    state: GuiControlState,
    events: GuiControlEvents,
    *,
    hold: bool = False,
) -> list[str]:
    controls = ["q", "d", "episode_stop"]
    if hold:
        controls.append("p")
        if int(state.active_side_count) == 1:
            controls.append("w")
    else:
        controls.append("h")
        if (
            state.fix_arm_available
            and manual_special_arm_controls(state)
            and fix_init_allowed_in_controls(state)
        ):
            controls.append("c")
            controls.append("fix_arm_toggle")
            controls.extend(fix_arm_controls(state))
        if (
            state.init_arm_available
            and manual_special_arm_controls(state)
            and fix_init_allowed_in_controls(state)
        ):
            controls.append("g")
            controls.append("init_arm_toggle")
            controls.extend(init_arm_controls(state))
    controls.extend(visualization_controls(state))
    return controls


def finalizing_controls(state: GuiControlState) -> list[str]:
    controls = ["q", "d"]
    if state.teleop_started and not state.hold_active:
        controls.append("h")
    controls.extend(visualization_controls(state))
    return controls


def teleop_controls(state: GuiControlState, events: GuiControlEvents) -> list[str]:
    controls = ["q", "d", "h"]
    if (
        state.fix_arm_available
        and manual_special_arm_controls(state)
        and fix_init_allowed_in_controls(state)
    ):
        controls.append("c")
        controls.append("fix_arm_toggle")
        controls.extend(fix_arm_controls(state))
    if (
        state.init_arm_available
        and manual_special_arm_controls(state)
        and fix_init_allowed_in_controls(state)
    ):
        controls.append("g")
        controls.append("init_arm_toggle")
        controls.extend(init_arm_controls(state))
    if not episode_busy(state):
        controls.extend(pose_snapshot_controls(state, events))
    if state.episode_recording_available:
        controls.extend(episode_controls(state, events))
    controls.extend(collection_task_controls(state))
    controls.extend(gripper_force_controls(state, events))
    controls.extend(visualization_controls(state))
    if not episode_busy(state):
        controls.extend(vive_calibration_controls(events, allow_load=False))
    return controls


def pre_home_controls(state: GuiControlState, events: GuiControlEvents) -> list[str]:
    controls = ["p", "w", "d", "q"]
    controls.extend(fix_arm_controls(state))
    controls.extend(init_arm_controls(state))
    controls.extend(visualization_controls(state))
    controls.extend(vive_calibration_controls(events, allow_load=True))
    controls.extend(movement_multiple_controls(state, events))
    controls.extend(collection_task_controls(state))
    controls.extend(gripper_force_controls(state, events))
    return controls


def hold_controls(state: GuiControlState, events: GuiControlEvents) -> list[str]:
    controls = ["p", "d"]
    if not episode_busy(state):
        controls.extend(fix_arm_controls(state))
        controls.extend(init_arm_controls(state))
    if int(state.active_side_count) == 1:
        controls.append("w")
    controls.extend(episode_controls(state, events))
    controls.append("q")
    controls.extend(visualization_controls(state))
    if not episode_busy(state):
        controls.extend(vive_calibration_controls(events, allow_load=True))
    controls.extend(movement_multiple_controls(state, events))
    controls.extend(collection_task_controls(state))
    controls.extend(gripper_force_controls(state, events))
    return controls


def episode_countdown_controls(state: GuiControlState, events: GuiControlEvents) -> list[str]:
    return [
        control
        for control in teleop_controls(state, events)
        if control not in {
            "episode_start",
            events.pose_snapshot_event,
            events.gripper_force_event,
        }
    ]


def current_active_controls(state: GuiControlState, events: GuiControlEvents) -> list[str]:
    if state.teleop_started and state.hold_active:
        return hold_controls(state, events)
    if state.episode_finalizing_active:
        return finalizing_controls(state)
    if state.episode_recording_active:
        return recording_controls(state, events, hold=state.hold_active)
    if state.episode_countdown_active:
        return episode_countdown_controls(state, events)
    if state.teleop_started and not state.hold_active:
        return teleop_controls(state, events)
    if state.teleop_started:
        return hold_controls(state, events)
    return pre_home_controls(state, events)


def single_arm_tracker_swap_allowed_in_hold(state: GuiControlState) -> bool:
    return bool(state.hold_active and int(state.active_side_count) == 1)


__all__ = [
    "GuiControlEvents",
    "GuiControlState",
    "collection_task_controls",
    "current_active_controls",
    "episode_busy",
    "episode_countdown_controls",
    "episode_controls",
    "finalizing_controls",
    "fix_arm_controls",
    "fix_init_allowed_in_controls",
    "gripper_force_controls",
    "hold_controls",
    "init_arm_controls",
    "manual_special_arm_controls",
    "movement_multiple_controls",
    "pose_snapshot_controls",
    "pre_home_controls",
    "recording_controls",
    "single_arm_tracker_swap_allowed_in_hold",
    "teleop_controls",
    "vive_calibration_controls",
    "visualization_controls",
]

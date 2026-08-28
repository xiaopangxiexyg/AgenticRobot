"""Pure GUI control-availability policy for real teleoperation.

This module maps an immutable runtime/UI snapshot to the controls that may be
shown as active. It neither consumes events nor invokes robot, GUI, recorder,
or calibration operations; those stateful responsibilities remain behind the
real-teleop application boundary.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuiControlEvents:
    episode_events: tuple[str, ...] = (
        "episode_start",
        "episode_stop",
        "episode_delete",
    )
    pose_snapshot_event: str = "record_pose_snapshot"
    gripper_force_event: str = "gripper_force_set"
    gripper_force_persist_event: str = "gripper_force_persist"
    emergency_stop_event: str = "init_qpos_emergency_stop"
    init_arm_qpos_speed_event: str = "init_arm_qpos_speed_set"
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
    # ``None`` preserves the historical policy contract for thin/offline
    # callers that only model the armed lifecycle.  The production real-teleop
    # adapter always supplies a strict per-frame dispatch projection.
    teleop_dispatch_confirmed: bool | None = None
    hold_active: bool = False
    episode_recording_available: bool = False
    episode_delete_available: bool = True
    dataset_reconfiguration_available: bool = True
    episode_recording_active: bool = False
    recording_paused: bool = False
    episode_countdown_active: bool = False
    episode_finalizing_active: bool = False
    camera_visualization_available: bool = False
    tactile_visualization_available: bool = False
    tactile_solving_available: bool = False
    tactile_solving_active: bool = False
    fix_arm_available: bool = False
    init_arm_available: bool = False
    init_arm_qpos_active: bool = False
    fix_hold_active: bool = False
    init_arm_qpos_move_available: bool = True
    startup_qpos_maintenance_enabled: bool = False
    pose_snapshot_available: bool = False
    active_side_count: int = 0
    special_arm_control_mode: str = "manual"
    foot_pad_available: bool = False
    foot_pad_count: int = 0
    foot_pad_test_active: bool = False
    foot_pad_test_completed: bool = False


def episode_busy(state: GuiControlState) -> bool:
    return bool(
        state.episode_recording_active
        or state.episode_countdown_active
        or state.episode_finalizing_active
    )


def teleop_dispatch_active(state: GuiControlState) -> bool:
    """Separate armed lifecycle ownership from live arm-control availability."""

    projected = state.teleop_dispatch_confirmed
    return bool(
        state.teleop_started
        and (True if projected is None else projected)
        and not state.hold_active
    )


def _apply_foot_pad_test_controls(
    state: GuiControlState,
    controls: list[str],
) -> list[str]:
    """Add pedal diagnostics without taking ownership of the GUI.

    Pedal presses remain diagnostic-only while the test is active, but explicit
    GUI and keyboard controls retain their normal, robot-state-based
    availability.
    """

    if not state.foot_pad_available:
        return controls
    if state.foot_pad_test_active:
        if "foot_pad_test_stop" not in controls:
            controls.append("foot_pad_test_stop")
        return controls
    if episode_busy(state) or (state.teleop_started and not state.hold_active):
        return controls
    if "foot_pad_test_start" not in controls:
        controls.append("foot_pad_test_start")
    return controls


def visualization_controls(state: GuiControlState) -> list[str]:
    controls: list[str] = []
    if state.camera_visualization_available and not episode_busy(state):
        controls.append("toggle_camera_vis")
    if state.tactile_visualization_available and not episode_busy(state):
        controls.append("toggle_tactile_vis")
    if (
        state.tactile_solving_available
        and not state.tactile_solving_active
        and not episode_busy(state)
        and (not state.teleop_started or state.hold_active)
    ):
        controls.append("enable_tactile_solving")
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
    controls: list[str] = []
    if state.dataset_reconfiguration_available:
        controls.append("task_select")
    if state.episode_recording_available:
        controls.append("round_trip_toggle")
    return controls


def gripper_force_controls(
    state: GuiControlState,
    events: GuiControlEvents,
) -> list[str]:
    """Allow direct gripper force changes only outside episode collection."""

    return [] if episode_busy(state) else [
        events.gripper_force_event,
        events.gripper_force_persist_event,
    ]


def episode_controls(state: GuiControlState, events: GuiControlEvents) -> list[str]:
    if not state.episode_recording_available:
        return []
    if episode_busy(state):
        return ["episode_stop"] if (
            state.episode_recording_active or state.episode_countdown_active
        ) else []
    controls = [
        event
        for event in events.episode_events
        if event != "episode_delete" or state.episode_delete_available
    ]
    if not teleop_dispatch_active(state):
        controls = [event for event in controls if event != "episode_start"]
    return controls


def fix_init_allowed_in_controls(state: GuiControlState) -> bool:
    return bool(
        teleop_dispatch_active(state)
        and not state.episode_countdown_active
        and not state.episode_finalizing_active
    )


def fix_arm_controls(state: GuiControlState) -> list[str]:
    return ["fix_arm_select"] if (
        state.fix_arm_available and manual_fix_arm_controls(state)
    ) else []


def init_arm_controls(state: GuiControlState) -> list[str]:
    return ["init_arm_select"] if state.init_arm_available else []


def init_arm_move_controls(state: GuiControlState) -> list[str]:
    """Expose the explicit GUI move button without enabling the G shortcut.

    Four-pedal mode routes pedals 3/4 to this same exact-qpos operation. Foot
    pads are additional inputs and never remove the GUI operation.
    """

    if not state.init_arm_available or not state.init_arm_qpos_move_available:
        return []
    if (
        state.episode_countdown_active
        or state.episode_finalizing_active
        or state.init_arm_qpos_active
    ):
        return []
    return ["init_arm_move_to_qpos", "init_arm_move_both_to_qpos"]


def init_arm_qpos_speed_controls(
    state: GuiControlState,
    events: GuiControlEvents,
) -> list[str]:
    if (
        not state.init_arm_available
        or state.init_arm_qpos_active
        or episode_busy(state)
    ):
        return []
    return [events.init_arm_qpos_speed_event]


def manual_special_arm_controls(state: GuiControlState) -> bool:
    del state
    return True


def manual_fix_arm_controls(state: GuiControlState) -> bool:
    """Whether keyboard/GUI fix-arm controls are available.

    Foot pads supplement these controls; they never own them exclusively.
    """

    return manual_special_arm_controls(state)


def manual_init_arm_controls(state: GuiControlState) -> bool:
    """Whether keyboard/GUI init-arm controls are available.

    Foot pads supplement these controls; every supported topology retains
    GUI/keyboard Init operations.
    """

    return manual_special_arm_controls(state)


def pose_snapshot_controls(state: GuiControlState, events: GuiControlEvents) -> list[str]:
    return [events.pose_snapshot_event] if (
        state.pose_snapshot_available and teleop_dispatch_active(state)
    ) else []


def teleop_suspended_controls(state: GuiControlState) -> list[str]:
    """Expose only fail-safe lifecycle controls without a live arm dispatch."""

    controls = ["q", "d", "h"]
    if state.episode_recording_active or state.episode_countdown_active:
        controls.append("episode_stop")
    return _apply_foot_pad_test_controls(state, controls)


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
            and manual_fix_arm_controls(state)
            and fix_init_allowed_in_controls(state)
        ):
            controls.append("c")
            controls.append("fix_arm_toggle")
            controls.extend(fix_arm_controls(state))
        if (
            state.init_arm_available
            and manual_init_arm_controls(state)
            and fix_init_allowed_in_controls(state)
        ):
            controls.append("g")
            controls.append("init_arm_toggle")
            controls.extend(init_arm_controls(state))
        if state.init_arm_available and fix_init_allowed_in_controls(state):
            controls.extend(
                control for control in init_arm_controls(state) if control not in controls
            )
            controls.extend(init_arm_move_controls(state))
    controls.extend(visualization_controls(state))
    return _apply_foot_pad_test_controls(state, controls)


def finalizing_controls(state: GuiControlState) -> list[str]:
    controls = ["q", "d"]
    if state.teleop_started and not state.hold_active:
        controls.append("h")
    controls.extend(visualization_controls(state))
    return _apply_foot_pad_test_controls(state, controls)


def paused_recording_controls(state: GuiControlState) -> list[str]:
    controls = ["q", "d", "h", "episode_stop"]
    controls.extend(visualization_controls(state))
    return _apply_foot_pad_test_controls(state, controls)


def fix_hold_controls(state: GuiControlState) -> list[str]:
    """Expose only release/stop operations while both sides are fixed-like."""

    controls = ["q", "d", "h"]
    if state.episode_recording_active or state.episode_countdown_active:
        controls.append("episode_stop")
    if state.fix_arm_available and manual_fix_arm_controls(state):
        controls.extend(("c", "fix_arm_toggle"))
        controls.extend(fix_arm_controls(state))
    if state.init_arm_available and manual_init_arm_controls(state):
        controls.extend(("g", "init_arm_toggle"))
        controls.extend(init_arm_controls(state))
    controls.extend(visualization_controls(state))
    return _apply_foot_pad_test_controls(state, list(dict.fromkeys(controls)))


def qpos_active_controls(state: GuiControlState) -> list[str]:
    controls = ["q", "d"]
    # request_hold() is a runtime-teleop transition and is intentionally a
    # no-op before the first Home.  Do not expose a dead Hold button during
    # startup exact-qpos maintenance; the explicit operation emergency stop
    # is the authoritative pre-Home stop control.
    if state.teleop_started:
        controls.append("h")
    controls.append("init_qpos_emergency_stop")
    if state.episode_recording_active or state.episode_countdown_active:
        controls.append("episode_stop")
    if state.init_arm_available and manual_init_arm_controls(state):
        controls.extend(("g", "init_arm_toggle"))
    if state.fix_hold_active and state.fix_arm_available and manual_fix_arm_controls(state):
        controls.extend(("c", "fix_arm_toggle"))
        controls.extend(fix_arm_controls(state))
    controls.extend(visualization_controls(state))
    return _apply_foot_pad_test_controls(state, list(dict.fromkeys(controls)))


def teleop_controls(state: GuiControlState, events: GuiControlEvents) -> list[str]:
    if not teleop_dispatch_active(state):
        return teleop_suspended_controls(state)
    controls = ["q", "d", "h"]
    if (
        state.fix_arm_available
        and manual_fix_arm_controls(state)
        and fix_init_allowed_in_controls(state)
    ):
        controls.append("c")
        controls.append("fix_arm_toggle")
        controls.extend(fix_arm_controls(state))
    if (
        state.init_arm_available
        and manual_init_arm_controls(state)
        and fix_init_allowed_in_controls(state)
    ):
        controls.append("g")
        controls.append("init_arm_toggle")
        controls.extend(init_arm_controls(state))
    if state.init_arm_available and fix_init_allowed_in_controls(state):
        controls.extend(
            control for control in init_arm_controls(state) if control not in controls
        )
        controls.extend(init_arm_qpos_speed_controls(state, events))
        controls.extend(init_arm_move_controls(state))
    if not episode_busy(state):
        controls.extend(pose_snapshot_controls(state, events))
    if state.episode_recording_available:
        controls.extend(
            control
            for control in episode_controls(state, events)
            if control != "episode_delete"
        )
    controls.extend(collection_task_controls(state))
    controls.extend(gripper_force_controls(state, events))
    controls.extend(visualization_controls(state))
    if not episode_busy(state):
        controls.extend(vive_calibration_controls(events, allow_load=False))
    return _apply_foot_pad_test_controls(state, controls)


def pre_home_controls(state: GuiControlState, events: GuiControlEvents) -> list[str]:
    controls = ["p", "w", "d", "q"]
    controls.extend(fix_arm_controls(state))
    controls.extend(init_arm_controls(state))
    controls.extend(init_arm_qpos_speed_controls(state, events))
    if state.startup_qpos_maintenance_enabled:
        controls.extend(init_arm_move_controls(state))
    controls.extend(visualization_controls(state))
    controls.extend(vive_calibration_controls(events, allow_load=True))
    controls.extend(movement_multiple_controls(state, events))
    controls.extend(collection_task_controls(state))
    controls.extend(gripper_force_controls(state, events))
    return _apply_foot_pad_test_controls(state, controls)


def hold_controls(state: GuiControlState, events: GuiControlEvents) -> list[str]:
    controls = ["p", "d"]
    if not episode_busy(state):
        controls.extend(fix_arm_controls(state))
        controls.extend(init_arm_controls(state))
        controls.extend(init_arm_qpos_speed_controls(state, events))
        controls.extend(init_arm_move_controls(state))
    if int(state.active_side_count) == 1:
        controls.append("w")
    controls.extend(
        control
        for control in episode_controls(state, events)
        if control != "episode_start"
    )
    controls.append("q")
    controls.extend(visualization_controls(state))
    if not episode_busy(state):
        controls.extend(vive_calibration_controls(events, allow_load=True))
    controls.extend(movement_multiple_controls(state, events))
    controls.extend(collection_task_controls(state))
    controls.extend(gripper_force_controls(state, events))
    return _apply_foot_pad_test_controls(state, controls)


def episode_countdown_controls(state: GuiControlState, events: GuiControlEvents) -> list[str]:
    return [
        control
        for control in teleop_controls(state, events)
        if control not in {
            "episode_start",
            events.pose_snapshot_event,
            events.gripper_force_event,
            events.gripper_force_persist_event,
            "task_select",
            "round_trip_toggle",
        }
    ]


def current_active_controls(state: GuiControlState, events: GuiControlEvents) -> list[str]:
    controls: list[str]
    if state.init_arm_qpos_active:
        controls = qpos_active_controls(state)
    elif state.fix_hold_active:
        controls = fix_hold_controls(state)
    elif state.teleop_started and state.hold_active:
        controls = hold_controls(state, events)
    elif state.episode_finalizing_active:
        controls = finalizing_controls(state)
    elif state.episode_recording_active:
        if state.recording_paused:
            controls = paused_recording_controls(state)
        else:
            controls = recording_controls(state, events, hold=state.hold_active)
    elif state.episode_countdown_active:
        controls = episode_countdown_controls(state, events)
    elif state.teleop_started and not state.hold_active:
        controls = teleop_controls(state, events)
    elif state.teleop_started:
        controls = hold_controls(state, events)
    else:
        controls = pre_home_controls(state, events)
    if (
        not episode_busy(state)
        and not state.init_arm_qpos_active
        and not state.fix_hold_active
        and (not state.teleop_started or state.hold_active)
        and "device_retry" not in controls
    ):
        controls.append("device_retry")
    return controls


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
    "init_arm_move_controls",
    "init_arm_qpos_speed_controls",
    "manual_special_arm_controls",
    "manual_fix_arm_controls",
    "manual_init_arm_controls",
    "movement_multiple_controls",
    "pose_snapshot_controls",
    "pre_home_controls",
    "recording_controls",
    "single_arm_tracker_swap_allowed_in_hold",
    "teleop_dispatch_active",
    "teleop_controls",
    "teleop_suspended_controls",
    "vive_calibration_controls",
    "visualization_controls",
]

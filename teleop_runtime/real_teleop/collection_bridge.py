"""Adapters from the real-teleop runtime to generic collection contracts.

These helpers deliberately accept structural component objects instead of
importing the entrypoint's ``RealTeleopComponents`` dataclass.  The collection
package therefore remains independent of real-hardware/GUI assembly details.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from robot_motion.runtime import TeleopCommand, merge_teleop_commands

from ..session import (
    TRACKER_HOLD_AUTOMATIC,
    TRACKER_HOLD_GRACE,
    SessionStepResult,
)


def collection_backends_from_components(components: object) -> dict[str, Any]:
    backends: dict[str, Any] = {}
    left_arm = getattr(components, "left_state_backend", None) or getattr(
        components,
        "left_command_backend",
        None,
    )
    right_arm = getattr(components, "right_state_backend", None) or getattr(
        components,
        "right_command_backend",
        None,
    )
    if left_arm is not None:
        backends["left_arm"] = left_arm
    if right_arm is not None:
        backends["right_arm"] = right_arm
    left_hand = getattr(components, "left_hand_backend", None)
    right_hand = getattr(components, "right_hand_backend", None)
    if left_hand is not None:
        backends["left_hand"] = left_hand
    if right_hand is not None:
        backends["right_hand"] = right_hand
    return backends


def collection_sources_from_components(
    components: object,
    camera_sources: Mapping[str, Any],
) -> dict[str, Any]:
    keyboard_source = getattr(components, "keyboard_source")
    sources: dict[str, Any] = {keyboard_source.name: keyboard_source}
    for camera_key, source in camera_sources.items():
        sources[f"camera.{camera_key}"] = source
    return sources


def collection_step_result(
    step_result: SessionStepResult,
    *,
    active_command: TeleopCommand | None,
    gripper_command: TeleopCommand | None,
) -> SessionStepResult:
    command = active_command
    if gripper_command is not None:
        command = (
            gripper_command
            if command is None
            else merge_teleop_commands((command, gripper_command))
        )
    motion_hold_active = step_result.motion_hold_mode in {
        TRACKER_HOLD_GRACE,
        TRACKER_HOLD_AUTOMATIC,
    }
    return SessionStepResult(
        timestamp=float(step_result.timestamp),
        state=step_result.state,
        control_hz=float(step_result.control_hz),
        channels=dict(step_result.channels),
        raw_channels=dict(step_result.raw_channels),
        observation=step_result.observation,
        command=command,
        command_dispatched=False
        if motion_hold_active
        else bool(
            step_result.command_dispatched
            or active_command is not None
            or gripper_command is not None
        ),
        action_record=None,
        faults=list(step_result.faults),
        motion_hold_mode=step_result.motion_hold_mode,
        motion_hold_reason=step_result.motion_hold_reason,
        tracker_loss_duration_sec=float(step_result.tracker_loss_duration_sec),
        tracker_loss_channels=tuple(step_result.tracker_loss_channels),
        home_recalibration_required=bool(step_result.home_recalibration_required),
    )


def copy_teleop_command(command: TeleopCommand | None) -> TeleopCommand | None:
    if command is None:
        return None
    copied = TeleopCommand(timestamp=float(command.timestamp), metadata=dict(command.metadata))
    for effector_command in command.commands.values():
        copied.add_command(effector_command.copy())
    return copied


def collection_sample_step_result(
    step_result: SessionStepResult,
    *,
    dispatched_command: TeleopCommand,
) -> SessionStepResult:
    """Bind a current observation to the latest command known to be dispatched."""

    command = copy_teleop_command(dispatched_command)
    assert command is not None
    return SessionStepResult(
        timestamp=float(step_result.timestamp),
        state=step_result.state,
        control_hz=float(step_result.control_hz),
        channels=dict(step_result.channels),
        raw_channels=dict(step_result.raw_channels),
        observation=step_result.observation,
        command=command,
        command_dispatched=True,
        action_record=None,
        faults=list(step_result.faults),
        motion_hold_mode=step_result.motion_hold_mode,
        motion_hold_reason=step_result.motion_hold_reason,
        tracker_loss_duration_sec=float(step_result.tracker_loss_duration_sec),
        tracker_loss_channels=tuple(step_result.tracker_loss_channels),
        home_recalibration_required=bool(step_result.home_recalibration_required),
    )


__all__ = [
    "collection_backends_from_components",
    "collection_sources_from_components",
    "collection_sample_step_result",
    "collection_step_result",
    "copy_teleop_command",
]

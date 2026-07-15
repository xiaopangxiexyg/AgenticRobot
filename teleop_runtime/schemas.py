"""Compatibility re-export for legacy teleop_runtime runtime contracts.

New common code should import from robot_motion.runtime directly.
"""

from __future__ import annotations

from robot_motion.runtime.contracts import (
    EffectorCommand,
    MotionCommand,
    MotionObservation,
    ObservationChannel,
    TeleopCommand,
    TeleopObservation,
    copy_runtime_value,
    latest_channel_timestamp,
    merge_teleop_commands,
)

_copy_value = copy_runtime_value

__all__ = [
    "EffectorCommand",
    "MotionCommand",
    "MotionObservation",
    "ObservationChannel",
    "TeleopCommand",
    "TeleopObservation",
    "_copy_value",
    "copy_runtime_value",
    "latest_channel_timestamp",
    "merge_teleop_commands",
]

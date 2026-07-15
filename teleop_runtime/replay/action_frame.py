"""Compatibility re-export for legacy teleop_runtime.replay.action_frame imports.

New common replay code should import from robot_motion.replay.action_frame directly.
"""

from __future__ import annotations

from robot_motion.replay.action_frame import (
    COMMON_REPLAY_MIGRATION_NOTE,
    annotate_replay_command_metadata,
    replay_action_metadata,
    replay_end_effector_targets_at_index,
    replay_robot_action_at_index,
    replay_teleop_command_at_index,
)

__all__ = [
    "COMMON_REPLAY_MIGRATION_NOTE",
    "annotate_replay_command_metadata",
    "replay_action_metadata",
    "replay_end_effector_targets_at_index",
    "replay_robot_action_at_index",
    "replay_teleop_command_at_index",
]

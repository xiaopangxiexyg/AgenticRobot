"""Compatibility re-export for legacy teleop_runtime.action.arm_spec imports.

New common code should import from robot_motion.action.arm_spec directly.
"""

from __future__ import annotations

from robot_motion.action.arm_spec import (
    ArmActionSpec,
    arm_action_specs_from_metadata,
    arm_action_specs_metadata,
    normalize_arm_action_specs,
)

__all__ = [
    "ArmActionSpec",
    "arm_action_specs_from_metadata",
    "arm_action_specs_metadata",
    "normalize_arm_action_specs",
]

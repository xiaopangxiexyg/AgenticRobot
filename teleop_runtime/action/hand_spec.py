"""Compatibility re-export for legacy teleop_runtime.action.hand_spec imports.

New common code should import from robot_motion.action.hand_spec directly.
"""

from __future__ import annotations

from robot_motion.action.hand_spec import (
    DEFAULT_HAND_SLOT_SEMANTICS,
    HandActionSpec,
    hand_action_specs_from_metadata,
    hand_action_specs_metadata,
    normalize_hand_action_specs,
)

__all__ = [
    "DEFAULT_HAND_SLOT_SEMANTICS",
    "HandActionSpec",
    "hand_action_specs_from_metadata",
    "hand_action_specs_metadata",
    "normalize_hand_action_specs",
]

"""Compatibility re-export for legacy teleop_runtime.action.vector_pipeline imports.

New common code should import from robot_motion.action.vector_pipeline directly.
"""

from __future__ import annotations

from robot_motion.action.vector_pipeline import (
    VectorActionLayout,
    VectorRobotActionBuildResult,
    robot_action_from_vector_action,
)

__all__ = [
    "VectorActionLayout",
    "VectorRobotActionBuildResult",
    "robot_action_from_vector_action",
]

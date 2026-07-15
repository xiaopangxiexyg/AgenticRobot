"""Compatibility re-export for legacy teleop_runtime.action.sources imports.

New common code should import from robot_motion.action.sources directly.
"""

from __future__ import annotations

from robot_motion.action.sources import CallableActionSource, RobotActionSource, StaticActionSource

__all__ = ["CallableActionSource", "RobotActionSource", "StaticActionSource"]

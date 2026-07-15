"""Simulation-only task environments built on the MuJoCo scene facade."""

from .cube_lift import CubeLiftEnv, CubeLiftTaskConfig

__all__ = ["CubeLiftEnv", "CubeLiftTaskConfig"]

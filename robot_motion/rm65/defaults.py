"""Stable RM65 motion-model defaults shared by sim and real runtime."""

from __future__ import annotations

import numpy as np


DEFAULT_LEFT_ARM_QPOS = np.array(
    [1.57, -0.47, 1.94, 0.0, 0.70, -1.57],
    dtype=float,
)
DEFAULT_RIGHT_ARM_QPOS = np.array(
    [-1.57, -0.47, 1.94, 0.0, -0.70, -1.57],
    dtype=float,
)

# These values are URDF-valid preview/solver seeds, not a calibrated station
# home.  Real startup/recovery must prefer an explicit alignment profile or
# fresh measured state.  In particular, the historical right-J4 ~= pi
# visualizer fallback is outside the active RM65 URDF and must never be
# silently clipped into an execution seed.

# Position execution envelope validated by the current 60 Hz accepted-qref /
# 120 Hz command-stream teleop path. These are shared motion defaults rather
# than frontend CLI constants so deploy, replay and simulation can use the
# same contract without importing teleop_runtime.
RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S = (
    5.6,
    5.6,
    7.2,
    12.0,
    12.0,
    16.0,
)
RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2 = (
    56.0,
    56.0,
    72.0,
    120.0,
    120.0,
    160.0,
)
RM65_POSITION_TRACKING_MAX_COMMAND_STEP_DEG = 0.675
RM65_POSITION_TRACKING_FOLLOW_ALPHA = 0.9


__all__ = [
    "DEFAULT_LEFT_ARM_QPOS",
    "DEFAULT_RIGHT_ARM_QPOS",
    "RM65_POSITION_TRACKING_FOLLOW_ALPHA",
    "RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2",
    "RM65_POSITION_TRACKING_MAX_COMMAND_STEP_DEG",
    "RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S",
]

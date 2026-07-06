"""MuJoCo real2sim utilities for DexTeleop.

The package is intentionally split into small modules so it can grow from an
offline camera/qpos checker into a reusable simulation backend for teleoperation.
"""

from .config import ArmSpec, CameraMount, GripperSpec, Real2SimConfig
from .scene import MujocoScene

__all__ = [
    "ArmSpec",
    "CameraMount",
    "GripperSpec",
    "MujocoScene",
    "Real2SimConfig",
]

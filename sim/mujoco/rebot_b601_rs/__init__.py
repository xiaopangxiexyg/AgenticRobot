"""B601-RS MuJoCo visualization plant and position driver.

This package consumes the model-only B601 contract and the generic MuJoCo
scene infrastructure.  It intentionally exposes no hardware backend.
"""

from .driver import (
    B601JointPositionCommand,
    B601PositionDriver,
    B601PositionTick,
    default_b601_position_timing,
)
from .scene import (
    B601_RS_DEFAULT_ARM_QPOS,
    B601_RS_GRIPPER_OPEN_QPOS,
    B601_RS_SIM_ARM_KEY,
    build_b601_mjcf_xml,
    build_b601_mujoco_scene,
)

__all__ = [
    "B601_RS_DEFAULT_ARM_QPOS",
    "B601_RS_GRIPPER_OPEN_QPOS",
    "B601_RS_SIM_ARM_KEY",
    "B601JointPositionCommand",
    "B601PositionDriver",
    "B601PositionTick",
    "build_b601_mjcf_xml",
    "build_b601_mujoco_scene",
    "default_b601_position_timing",
]

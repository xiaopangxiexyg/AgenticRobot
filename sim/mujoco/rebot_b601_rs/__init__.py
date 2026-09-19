"""B601-RS MuJoCo position and ideal-effort pre-hardware backends.

This package consumes the model-only B601 contract and the generic MuJoCo
scene infrastructure.  It intentionally exposes no hardware backend.
"""

from .contact import (
    B601_RS_CONTACT_SCENE_PROFILE_NAME,
    B601_RS_CONTACT_SENSOR_NAME,
    B601MujocoContactObservation,
    b601_contact_scene_spec,
    read_b601_contact_observation,
)
from .driver import (
    B601_RS_DEFAULT_SIM_TARGET_MAX_VELOCITY_RAD_S,
    B601JointPositionCommand,
    B601PositionDriver,
    B601PositionTick,
    default_b601_position_timing,
)
from .mit_driver import (
    B601EffortTick,
    B601MujocoEffortDriver,
    default_b601_mit_profile,
    default_b601_mit_timing,
)
from .real2sim import B601SimFaultPolicy
from .scene import (
    B601_RS_DEFAULT_ARM_QPOS,
    B601_RS_GRIPPER_OPEN_QPOS,
    B601_RS_SIM_ARM_KEY,
    build_b601_mjcf_xml,
    build_b601_mujoco_scene,
)

__all__ = [
    "B601_RS_CONTACT_SCENE_PROFILE_NAME",
    "B601_RS_CONTACT_SENSOR_NAME",
    "B601_RS_DEFAULT_ARM_QPOS",
    "B601_RS_DEFAULT_SIM_TARGET_MAX_VELOCITY_RAD_S",
    "B601_RS_GRIPPER_OPEN_QPOS",
    "B601_RS_SIM_ARM_KEY",
    "B601EffortTick",
    "B601JointPositionCommand",
    "B601MujocoContactObservation",
    "B601MujocoEffortDriver",
    "B601PositionDriver",
    "B601PositionTick",
    "B601SimFaultPolicy",
    "b601_contact_scene_spec",
    "build_b601_mjcf_xml",
    "build_b601_mujoco_scene",
    "default_b601_mit_profile",
    "default_b601_mit_timing",
    "default_b601_position_timing",
    "read_b601_contact_observation",
]

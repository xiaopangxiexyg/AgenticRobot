from typing import TYPE_CHECKING

from .gripper_profile import (
    GripperActuationProfile,
    GripperContactTuning,
    cad_bimanual_dmtac_gripper_profile,
    cube_lift_contact_dmtac_gripper_profile,
    identified_bimanual_dmtac_gripper_profile,
    strong_cad_bimanual_dmtac_gripper_profile,
    temporary_bimanual_dmtac_gripper_profile,
)
from sim.rm65.actuation import (
    PositionActuationProfile,
    rm65_bimanual_udp_feedback_profile,
    rm65_bimanual_udp_feedback_profile_v1,
    rm65_bimanual_udp_feedback_profile_v2,
    rm65_bimanual_udp_feedback_profile_v3,
)

if TYPE_CHECKING:
    from .executor import ActuatedPositionTick, MujocoActuatedPositionExecutor
    from .gripper_executor import GripperControlTick, MujocoDmtacGripperExecutor

__all__ = [
    "ActuatedPositionTick",
    "GripperActuationProfile",
    "GripperContactTuning",
    "GripperControlTick",
    "MujocoActuatedPositionExecutor",
    "MujocoDmtacGripperExecutor",
    "PositionActuationProfile",
    "cad_bimanual_dmtac_gripper_profile",
    "cube_lift_contact_dmtac_gripper_profile",
    "identified_bimanual_dmtac_gripper_profile",
    "strong_cad_bimanual_dmtac_gripper_profile",
    "rm65_bimanual_udp_feedback_profile",
    "rm65_bimanual_udp_feedback_profile_v1",
    "rm65_bimanual_udp_feedback_profile_v2",
    "rm65_bimanual_udp_feedback_profile_v3",
    "temporary_bimanual_dmtac_gripper_profile",
]


def __getattr__(name: str):
    """Keep MJCF/profile imports independent from the runtime executor graph."""

    if name in {"ActuatedPositionTick", "MujocoActuatedPositionExecutor"}:
        from .executor import ActuatedPositionTick, MujocoActuatedPositionExecutor

        return {
            "ActuatedPositionTick": ActuatedPositionTick,
            "MujocoActuatedPositionExecutor": MujocoActuatedPositionExecutor,
        }[name]
    if name in {"GripperControlTick", "MujocoDmtacGripperExecutor"}:
        from .gripper_executor import GripperControlTick, MujocoDmtacGripperExecutor

        return {
            "GripperControlTick": GripperControlTick,
            "MujocoDmtacGripperExecutor": MujocoDmtacGripperExecutor,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

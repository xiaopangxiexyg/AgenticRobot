"""B601 real-observation to RM65 MuJoCo gripper mapping adapter.

The source is intentionally injected as a normalized observation.  The B601
J7 hardware reader is not part of the six-joint B601 arm owner; a future
read-only J7 source must calibrate through
``B601RM65GripperMappingProfile.b601_position_sample`` before calling this
adapter.
"""

from __future__ import annotations

from robot_motion.rebot_b601_rs.gripper_mapping import (
    B601RM65GripperMappingProfile,
    NormalizedGripperSample,
)
from sim.mujoco.rebot_b601_rs.gripper import (
    GripperBridgeStep,
    NormalizedGripperSimBridge,
)
from sim.mujoco.scene import MujocoScene


class B601ToRM65GripperSimBridge(NormalizedGripperSimBridge):
    """Apply a B601 normalized close sample to an RM65 MuJoCo gripper."""

    direction = "b601_to_rm65"

    def __init__(
        self,
        scene: MujocoScene,
        mapping: B601RM65GripperMappingProfile,
        *,
        target_gripper_name: str | None = None,
    ) -> None:
        super().__init__(
            scene,
            mapping,
            direction="b601_to_rm65",
            target_gripper_name=(
                mapping.rm65_gripper_name
                if target_gripper_name is None
                else target_gripper_name
            ),
        )

    def apply_b601_sample(
        self,
        sample: NormalizedGripperSample,
        *,
        now_timestamp_sec: float | None = None,
    ) -> GripperBridgeStep:
        return self.apply(sample, now_timestamp_sec=now_timestamp_sec)


__all__ = ["B601ToRM65GripperSimBridge"]

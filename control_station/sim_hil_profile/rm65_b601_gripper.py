"""RM65 real-observation to B601 MuJoCo gripper mapping adapter.

The RealMan/DMTAC backend remains the source owner.  This adapter consumes a
normalized read-only sample and updates an opt-in B601 MuJoCo gripper scene;
it never opens a serial port and never sends a RobStride command.
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


class RM65ToB601GripperSimBridge(NormalizedGripperSimBridge):
    """Apply an RM65 normalized close sample to a B601 MuJoCo gripper."""

    direction = "rm65_to_b601"

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
            direction="rm65_to_b601",
            target_gripper_name=(
                mapping.b601_gripper_name
                if target_gripper_name is None
                else target_gripper_name
            ),
        )

    def apply_rm65_sample(
        self,
        sample: NormalizedGripperSample,
        *,
        now_timestamp_sec: float | None = None,
    ) -> GripperBridgeStep:
        return self.apply(sample, now_timestamp_sec=now_timestamp_sec)


__all__ = ["RM65ToB601GripperSimBridge"]

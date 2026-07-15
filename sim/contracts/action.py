from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from robot_motion.action.schema import (
    ARM_POSE_DOF,
    INSPIRE_HAND_DOF,
    default_arm_controlled_frame,
    default_arm_frame,
)
from robot_motion.deploy.schema import (
    ACTION_DIM,
    DEPLOY_ACTION_VECTOR_ACTION_SPACE,
    DEPLOY_ACTION_VECTOR_LAYOUT_VERSION,
)

from .versions import ACTION_SPEC_VERSION, require_version


DEFAULT_ARM_REFERENCE_FRAMES = (
    ("left", default_arm_frame("left")),
    ("right", default_arm_frame("right")),
)


@dataclass(frozen=True)
class ActionSpec:
    """Version pin for the existing canonical ``RobotAction`` deploy layout."""

    schema_version: str = ACTION_SPEC_VERSION
    action_space: str = DEPLOY_ACTION_VECTOR_ACTION_SPACE
    vector_layout_version: str = DEPLOY_ACTION_VECTOR_LAYOUT_VERSION
    vector_dim: int = ACTION_DIM
    dtype: str = "float32"
    arm_representation: str = "xyz_quat_xyzw"
    arm_pose_dof: int = ARM_POSE_DOF
    arm_controlled_frame: str = default_arm_controlled_frame()
    arm_reference_frames: tuple[tuple[str, str], ...] = DEFAULT_ARM_REFERENCE_FRAMES
    hand_slot_count: int = INSPIRE_HAND_DOF
    gripper_close_semantics: str = "0=open,1=closed"

    def __post_init__(self) -> None:
        require_version(self.schema_version, ACTION_SPEC_VERSION, field="action spec version")
        if self.action_space != DEPLOY_ACTION_VECTOR_ACTION_SPACE:
            raise ValueError("ActionSpec.action_space disagrees with canonical RobotAction.")
        if self.vector_layout_version != DEPLOY_ACTION_VECTOR_LAYOUT_VERSION:
            raise ValueError("ActionSpec.vector_layout_version disagrees with deploy schema.")
        if int(self.vector_dim) != ACTION_DIM or int(self.arm_pose_dof) != ARM_POSE_DOF:
            raise ValueError("ActionSpec dimensions disagree with deploy schema.")
        if int(self.hand_slot_count) != INSPIRE_HAND_DOF:
            raise ValueError("ActionSpec.hand_slot_count disagrees with the q6 deploy layout.")
        if self.dtype != "float32" or self.arm_representation != "xyz_quat_xyzw":
            raise ValueError("ActionSpec v1 requires float32 xyz_quat_xyzw arms.")
        if self.arm_controlled_frame != default_arm_controlled_frame():
            raise ValueError("ActionSpec v1 controls the canonical tool_tcp frame.")
        if self.gripper_close_semantics != "0=open,1=closed":
            raise ValueError("ActionSpec v1 requires normalized close semantics.")
        expected_frames = {
            "left": default_arm_frame("left"),
            "right": default_arm_frame("right"),
        }
        frames = {str(side): str(frame) for side, frame in self.arm_reference_frames}
        if frames != expected_frames:
            raise ValueError(f"ActionSpec reference frames must be {expected_frames}.")
        object.__setattr__(self, "arm_reference_frames", tuple(expected_frames.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "action_space": self.action_space,
            "vector_layout_version": self.vector_layout_version,
            "vector_dim": int(self.vector_dim),
            "dtype": self.dtype,
            "arm_representation": self.arm_representation,
            "arm_pose_dof": int(self.arm_pose_dof),
            "arm_controlled_frame": self.arm_controlled_frame,
            "arm_reference_frames": dict(self.arm_reference_frames),
            "hand_slot_count": int(self.hand_slot_count),
            "gripper_close_semantics": self.gripper_close_semantics,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionSpec":
        frames = payload.get("arm_reference_frames")
        return cls(
            schema_version=str(payload.get("schema_version", ACTION_SPEC_VERSION)),
            action_space=str(payload.get("action_space", DEPLOY_ACTION_VECTOR_ACTION_SPACE)),
            vector_layout_version=str(
                payload.get("vector_layout_version", DEPLOY_ACTION_VECTOR_LAYOUT_VERSION)
            ),
            vector_dim=int(payload.get("vector_dim", ACTION_DIM)),
            dtype=str(payload.get("dtype", "float32")),
            arm_representation=str(payload.get("arm_representation", "xyz_quat_xyzw")),
            arm_pose_dof=int(payload.get("arm_pose_dof", ARM_POSE_DOF)),
            arm_controlled_frame=str(
                payload.get("arm_controlled_frame", default_arm_controlled_frame())
            ),
            arm_reference_frames=(
                tuple((str(side), str(frame)) for side, frame in dict(frames).items())
                if isinstance(frames, Mapping)
                else DEFAULT_ARM_REFERENCE_FRAMES
            ),
            hand_slot_count=int(payload.get("hand_slot_count", INSPIRE_HAND_DOF)),
            gripper_close_semantics=str(
                payload.get("gripper_close_semantics", "0=open,1=closed")
            ),
        )


__all__ = ["ActionSpec", "DEFAULT_ARM_REFERENCE_FRAMES"]

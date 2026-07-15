"""Compatibility re-export for legacy teleop_runtime.action.schema imports.

New common code should import from robot_motion.action.schema directly.
"""

from __future__ import annotations

from robot_motion.action.schema import (
    ARM_POSE_DOF,
    BIMANUAL_EE_POSE_HAND16_V1,
    BIMANUAL_EE_POSE_V1,
    BIMANUAL_MIXED_HAND16_V1,
    BIMANUAL_MIXED_V1,
    BIMANUAL_QPOS_HAND16_V1,
    BIMANUAL_QPOS_V1,
    HAND16_ACTION_SPACES,
    HAND16_SLOT_DOF,
    INSPIRE_HAND_DOF,
    RM65_ARM_DOF,
    VALID_SIDES,
    ActionExecutionResult,
    ActionRecord,
    EndEffectorTarget,
    RobotAction,
    default_arm_controlled_frame,
    default_arm_frame,
    effector_name,
    hand_dof_for_action_space,
    normalize_side,
    split_effector_name,
)

__all__ = [
    "ARM_POSE_DOF",
    "BIMANUAL_EE_POSE_HAND16_V1",
    "BIMANUAL_EE_POSE_V1",
    "BIMANUAL_MIXED_HAND16_V1",
    "BIMANUAL_MIXED_V1",
    "BIMANUAL_QPOS_HAND16_V1",
    "BIMANUAL_QPOS_V1",
    "HAND16_ACTION_SPACES",
    "HAND16_SLOT_DOF",
    "INSPIRE_HAND_DOF",
    "RM65_ARM_DOF",
    "VALID_SIDES",
    "ActionExecutionResult",
    "ActionRecord",
    "EndEffectorTarget",
    "RobotAction",
    "default_arm_controlled_frame",
    "default_arm_frame",
    "effector_name",
    "hand_dof_for_action_space",
    "normalize_side",
    "split_effector_name",
]

"""Compatibility re-export for legacy teleop_runtime.action.codec imports.

New common code should import from robot_motion.action.codec directly.
"""

from __future__ import annotations

from robot_motion.action.codec import (
    EE_POSE_HAND16_SLICES,
    EE_POSE_HAND16_VECTOR_DIM,
    EE_POSE_SLICES,
    EE_POSE_VECTOR_DIM,
    QPOS_HAND16_SLICES,
    QPOS_HAND16_VECTOR_DIM,
    QPOS_SLICES,
    QPOS_VECTOR_DIM,
    ROBOT_ACTION_VECTOR_HAND16_LAYOUT_VERSION,
    ROBOT_ACTION_VECTOR_LAYOUT_VERSION,
    ActionVectorEncoding,
    decode_robot_action,
    encode_robot_action,
)

__all__ = [
    "EE_POSE_HAND16_SLICES",
    "EE_POSE_HAND16_VECTOR_DIM",
    "EE_POSE_SLICES",
    "EE_POSE_VECTOR_DIM",
    "QPOS_HAND16_SLICES",
    "QPOS_HAND16_VECTOR_DIM",
    "QPOS_SLICES",
    "QPOS_VECTOR_DIM",
    "ROBOT_ACTION_VECTOR_HAND16_LAYOUT_VERSION",
    "ROBOT_ACTION_VECTOR_LAYOUT_VERSION",
    "ActionVectorEncoding",
    "decode_robot_action",
    "encode_robot_action",
]

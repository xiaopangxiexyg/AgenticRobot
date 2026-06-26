from __future__ import annotations

from typing import Any

import numpy as np

from .tcp import HAND_BASE_TCP_FRAME
from .transforms import as_pose_matrix, invert_pose


TOOL_TCP_FRAME = "tool_tcp"
ARM_ENDPOINT_FRAME = "arm_endpoint"
RM65_ARM_ENDPOINT_URDF_FRAME = "arm_link6"
ARM_ENDPOINT_TO_HAND_BASE_Z_M = 0.009

TOOL_TCP_SEMANTICS = (
    "Arm pose actions control the active tool TCP. The TCP is loaded from the "
    "arm/tool calibration profile at execution time; RM65 backends convert this "
    "software TCP target to the internal IK/control frame."
)

ARM_ENDPOINT_SEMANTICS = (
    "Compatibility frame for the mechanical arm endpoint, not the active tool TCP. "
    "On RM65 this maps to URDF frame arm_link6. Its orientation matches the historical hand_base "
    "frame; hand_base is translated +0.009m along local z from arm_endpoint."
)


def default_arm_controlled_frame() -> str:
    return TOOL_TCP_FRAME


def arm_endpoint_to_hand_base_transform() -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[2, 3] = ARM_ENDPOINT_TO_HAND_BASE_Z_M
    return transform


def hand_base_to_arm_endpoint_transform() -> np.ndarray:
    return invert_pose(arm_endpoint_to_hand_base_transform())


def canonical_arm_controlled_frame(frame: str | None = None) -> str:
    text = "" if frame is None else str(frame).strip()
    if text in {"", "tcp", TOOL_TCP_FRAME}:
        return TOOL_TCP_FRAME
    if text in {ARM_ENDPOINT_FRAME, RM65_ARM_ENDPOINT_URDF_FRAME}:
        return ARM_ENDPOINT_FRAME
    if text == HAND_BASE_TCP_FRAME:
        return HAND_BASE_TCP_FRAME
    return text


def rm65_urdf_frame_for_controlled_frame(frame: str | None = None) -> str:
    controlled = canonical_arm_controlled_frame(frame)
    if controlled == ARM_ENDPOINT_FRAME:
        return RM65_ARM_ENDPOINT_URDF_FRAME
    if controlled == TOOL_TCP_FRAME:
        return HAND_BASE_TCP_FRAME
    return controlled


def convert_pose_between_arm_control_frames(
    pose: Any,
    *,
    from_frame: str | None,
    to_frame: str | None,
) -> np.ndarray:
    matrix = as_pose_matrix(pose)
    source = canonical_arm_controlled_frame(from_frame)
    target = canonical_arm_controlled_frame(to_frame)
    if source == target:
        return matrix
    if TOOL_TCP_FRAME in {source, target}:
        raise ValueError(
            "tool_tcp conversions require a loaded arm/tool TCP profile and must be handled "
            "by the execution backend."
        )
    if source == HAND_BASE_TCP_FRAME and target == ARM_ENDPOINT_FRAME:
        return matrix @ hand_base_to_arm_endpoint_transform()
    if source == ARM_ENDPOINT_FRAME and target == HAND_BASE_TCP_FRAME:
        return matrix @ arm_endpoint_to_hand_base_transform()
    raise ValueError(f"Unsupported arm controlled-frame conversion: {source!r} -> {target!r}.")

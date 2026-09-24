"""Optional physical writing-tip geometry above the existing right tool TCP.

The command frame remains the configured 14 cm tool_tcp.  Only the writing
height reference is translated to the measured 20 cm fixture tip.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from robot_motion.relative_action import xyzquat_to_pose_matrix

RIGHT_TOOL_TCP_FLANGE_Z_M = 0.14
RIGHT_CONTACT_FLANGE_Z_M = 0.20
RIGHT_CONTACT_OFFSET_TOOL_TCP_M = RIGHT_CONTACT_FLANGE_Z_M - RIGHT_TOOL_TCP_FLANGE_Z_M
RIGHT_CONTACT_FRAME = "right_fixture_tip_flange_plus_z_0p20m"


def require_right_tool_tcp_profile(state: Mapping[str, Any]) -> None:
    """Fail closed if the optional fixture geometry is not what was measured."""

    try:
        pose = np.asarray(state["tool_tcp_pose_flange_m_rad"], dtype=float).reshape(6)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("--correct-tcp requires the measured right 14 cm tool TCP profile") from exc
    expected = np.array([0.0, 0.0, RIGHT_TOOL_TCP_FLANGE_Z_M, 0.0, 0.0, 0.0])
    if not np.all(np.isfinite(pose)) or not np.allclose(pose, expected, rtol=0.0, atol=1e-6):
        raise RuntimeError(
            "--correct-tcp requires right tool_tcp at flange +Z 0.14 m "
            "with zero lateral offset and rotation"
        )


def contact_height_from_tool_tcp(pose_canonical_desk: np.ndarray) -> float:
    pose = np.asarray(pose_canonical_desk, dtype=float).reshape(7)
    if not np.all(np.isfinite(pose)):
        raise ValueError("right tool TCP pose must be finite")
    rotation = xyzquat_to_pose_matrix(pose)[:3, :3]
    return float(pose[2] + RIGHT_CONTACT_OFFSET_TOOL_TCP_M * rotation[2, 2])


def tool_tcp_z_for_contact_height(
    pose_canonical_desk: np.ndarray, contact_z_m: float
) -> float:
    if not np.isfinite(float(contact_z_m)):
        raise ValueError("right contact height must be finite")
    pose = np.asarray(pose_canonical_desk, dtype=float).reshape(7)
    return float(pose[2] + float(contact_z_m) - contact_height_from_tool_tcp(pose))

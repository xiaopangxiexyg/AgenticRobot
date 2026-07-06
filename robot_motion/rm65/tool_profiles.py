from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from robot_motion.pose import pose6_to_matrix

from .constants import ARM_DOF, POSITION_ENDPOINT_PROFILE_NAME, SIDES


def _calibration_profiles_module():
    # The JSON profiles remain in hardware/rm65/utils/configs/calibrations for now,
    # but their TCP/tool semantics are shared by RM65 real and sim paths.
    from hardware.rm65 import calibration_profiles

    return calibration_profiles


def default_tool_profile_dir() -> Path:
    return Path(_calibration_profiles_module().PROFILE_DIR)


def load_rm65_tool_profile(name_or_path: str | Path, *, side: str | None = None) -> dict[str, Any]:
    arm = None if side is None else _normalize_optional_side(side)
    return _calibration_profiles_module().load_profile(name_or_path, arm=arm)


def list_rm65_tool_profiles() -> list[Path]:
    return list(_calibration_profiles_module().list_profiles())


def is_position_endpoint_profile(profile: dict[str, Any] | Any) -> bool:
    return bool(_calibration_profiles_module().is_position_endpoint_profile(profile))


def tool_tcp_pose_flange_m_rad(profile: dict[str, Any] | Any) -> np.ndarray:
    if not isinstance(profile, dict):
        raise ValueError("RM65 tool profile must be a mapping.")
    tcp = profile.get("tcp")
    if not isinstance(tcp, dict):
        raise ValueError("RM65 tool profile is missing tcp mapping.")
    try:
        pose = np.asarray(tcp.get("pose_flange_m_rad"), dtype=float).reshape(-1)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("RM65 tool profile tcp.pose_flange_m_rad must be numeric.") from exc
    if pose.shape != (ARM_DOF,) or not np.all(np.isfinite(pose)):
        raise ValueError("RM65 tool profile tcp.pose_flange_m_rad must contain 6 finite values.")
    return pose.astype(float, copy=True)


def endpoint_to_tool_transform(profile: dict[str, Any] | Any) -> np.ndarray:
    return pose6_to_matrix(tool_tcp_pose_flange_m_rad(profile))


def _normalize_optional_side(side: str) -> str:
    value = str(side).strip().lower()
    if value not in SIDES:
        raise ValueError(f"RM65 side must be one of {SIDES}, got {side!r}.")
    return value


__all__ = [
    "POSITION_ENDPOINT_PROFILE_NAME",
    "default_tool_profile_dir",
    "endpoint_to_tool_transform",
    "is_position_endpoint_profile",
    "list_rm65_tool_profiles",
    "load_rm65_tool_profile",
    "tool_tcp_pose_flange_m_rad",
]

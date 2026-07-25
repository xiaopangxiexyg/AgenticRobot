"""Arm mode, tool-profile, and saved-alignment argument projections."""

from __future__ import annotations

import argparse

import numpy as np

from ..initial_alignment import TeleopInitialAlignment
from ..log_utils import log_event as _log

def _impedance_mode_arg(args: argparse.Namespace) -> str:
    return str(getattr(args, "impedance_mode", "pose"))


def _strict_tool_profile_arg(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "strict_tool_profile", True))


def _force_tare_sec_arg(args: argparse.Namespace) -> float:
    return float(getattr(args, "force_tare_sec", 3.0))


def _tool_profile_arg(args: argparse.Namespace, side: str) -> str | None:
    return getattr(args, f"{side}_tool_profile", "") or None


def _tool_tcp_pose_flange_arg(args: argparse.Namespace, side: str) -> tuple[float, ...] | None:
    profile_arg = _tool_profile_arg(args, side)
    if not profile_arg:
        return None
    try:
        from hardware.rm65.calibration_profiles import load_profile

        profile = load_profile(profile_arg, arm=side)
        tcp = profile.get("tcp") if isinstance(profile.get("tcp"), dict) else {}
        pose = tcp.get("pose_flange_m_rad")
        if pose is None:
            return None
        values = np.asarray(pose, dtype=float).reshape(6)
        if not np.all(np.isfinite(values)):
            return None
        return tuple(float(value) for value in values)
    except Exception as exc:
        _log("TCP", f"WARNING: could not load {side} tool TCP for collision solver: {exc}")
        return None


def _alignment_qpos_pair(alignment: TeleopInitialAlignment | None) -> dict[str, np.ndarray] | None:
    if alignment is None:
        return None
    qpos_pair: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        side_state = alignment.get_side(side)
        if side_state is None or side_state.arm_qpos is None:
            return None
        qpos_pair[side] = np.asarray(side_state.arm_qpos, dtype=float).reshape(6).copy()
    return qpos_pair


def _start_state_summary(checks: dict[str, object]) -> str:
    parts: list[str] = []
    for side in ("left", "right"):
        if side not in checks:
            continue
        check = checks[side]
        parts.append(
            f"{side}: max_abs={check.max_abs_joint_error_deg:.2f}deg "
            f"weighted={check.weighted_joint_error_deg:.2f}deg "
            f"within_limits={check.within_limits}"
        )
    return "; ".join(parts)


__all__ = [
    "_alignment_qpos_pair",
    "_force_tare_sec_arg",
    "_impedance_mode_arg",
    "_start_state_summary",
    "_strict_tool_profile_arg",
    "_tool_profile_arg",
    "_tool_tcp_pose_flange_arg",
]

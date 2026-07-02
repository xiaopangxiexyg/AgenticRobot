from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

from hardware.rm65.calibration_profiles import ARM_ENDPOINT_TCP_POSE_FLANGE_M_RAD, load_profile

from ..deploy.schema import ACTION_ARM_POSE_SLICES, ARM_JOINT_DOF, xyzquat_to_pose_matrix
from .trajectory import ReplayTrajectoryGroup


@dataclass(frozen=True)
class ReplayCheckResult:
    name: str
    ok: bool
    level: str = "error"
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": bool(self.ok),
            "level": self.level,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class ReplayPreflightReport:
    checks: tuple[ReplayCheckResult, ...]

    @property
    def ok(self) -> bool:
        return all(check.ok or check.level != "error" for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "checks": [check.to_dict() for check in self.checks],
        }


def _finite_vector(value: Any, *, length: int) -> np.ndarray | None:
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if vector.shape != (int(length),) or not np.all(np.isfinite(vector)):
        return None
    return vector.astype(float, copy=True)


def _as_pose_matrix(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        matrix = np.asarray(value, dtype=float).reshape(4, 4)
    except Exception:
        return None
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        return None
    return matrix


def _pose_delta(current: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    current_pose = np.asarray(current, dtype=float).reshape(4, 4)
    target_pose = np.asarray(target, dtype=float).reshape(4, 4)
    xyz_m = float(np.linalg.norm(target_pose[:3, 3] - current_pose[:3, 3]))
    rot = R.from_matrix(target_pose[:3, :3]) * R.from_matrix(current_pose[:3, :3]).inv()
    return xyz_m, float(rot.magnitude())


def _metadata_mapping(group: ReplayTrajectoryGroup) -> Mapping[str, Any]:
    source = group.metadata.get("source_metadata")
    if isinstance(source, Mapping):
        episode_metadata = source.get("episode_metadata")
        if isinstance(episode_metadata, Mapping):
            return episode_metadata
    return {}


def _nested_get(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _dataset_tool_profile_path(group: ReplayTrajectoryGroup, side: str) -> str | None:
    metadata = _metadata_mapping(group)
    side_key = f"{side}_tool_profile"
    candidates = [
        metadata.get(side_key),
        _nested_get(metadata, ("calibration_snapshot", "metadata", side_key)),
        _nested_get(metadata, ("calibration_snapshot", "arms", f"{side}_arm", "metadata", side_key)),
        _nested_get(metadata, ("calibration_snapshot", "arms", f"{side}_arm", "metadata", "tool_profile")),
    ]
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return text
    return None


def tcp_pose_from_tool_profile(path_text: str | None, *, side: str) -> np.ndarray:
    text = str(path_text or "").strip()
    if not text:
        return np.asarray(ARM_ENDPOINT_TCP_POSE_FLANGE_M_RAD, dtype=float).reshape(6)
    profile = load_profile(text, arm=side)
    tcp = profile.get("tcp") if isinstance(profile.get("tcp"), Mapping) else {}
    pose = _finite_vector(tcp.get("pose_flange_m_rad"), length=6)
    if pose is None:
        raise ValueError(f"tool_profile {text!r} does not contain a finite tcp.pose_flange_m_rad.")
    return pose


def dataset_tcp_by_side(
    group: ReplayTrajectoryGroup,
    active_sides: Sequence[str],
) -> tuple[dict[str, np.ndarray], list[ReplayCheckResult]]:
    values: dict[str, np.ndarray] = {}
    checks: list[ReplayCheckResult] = []
    for side in active_sides:
        path = _dataset_tool_profile_path(group, str(side))
        if not path:
            checks.append(
                ReplayCheckResult(
                    name=f"{side}_dataset_tcp_metadata",
                    ok=False,
                    level="error",
                    message=(
                        f"{side} dataset has no tool profile metadata; cannot verify TCP equality."
                    ),
                )
            )
            continue
        try:
            values[str(side)] = tcp_pose_from_tool_profile(path, side=str(side))
            checks.append(
                ReplayCheckResult(
                    name=f"{side}_dataset_tcp_metadata",
                    ok=True,
                    level="info",
                    message=f"{side} dataset TCP loaded",
                    details={"tool_profile": path},
                )
            )
        except Exception as exc:
            checks.append(
                ReplayCheckResult(
                    name=f"{side}_dataset_tcp_metadata",
                    ok=False,
                    level="error",
                    message=str(exc),
                    details={"tool_profile": path},
                )
            )
    return values, checks


def run_replay_preflight(
    *,
    group: ReplayTrajectoryGroup,
    active_sides: Sequence[str],
    current_qpos_by_side: Mapping[str, Any],
    current_tcp_pose_by_side: Mapping[str, Any],
    current_tcp_flange_by_side: Mapping[str, Any],
    tcp_pos_tolerance_m: float = 1e-4,
    tcp_rot_tolerance_rad: float = 1e-3,
    max_qpos_init_delta_rad: float = float(np.pi),
) -> ReplayPreflightReport:
    sides = tuple(str(side) for side in active_sides)
    checks: list[ReplayCheckResult] = []

    if group.frame_count <= 0 or group.actions.size <= 0:
        checks.append(
            ReplayCheckResult(
                name="trajectory_nonempty",
                ok=False,
                message="Replay trajectory has no actions.",
            )
        )
        return ReplayPreflightReport(tuple(checks))
    checks.append(
        ReplayCheckResult(
            name="trajectory_nonempty",
            ok=True,
            level="info",
            message=f"trajectory frames={group.frame_count} hz={group.action_hz:.3f}",
        )
    )

    dataset_tcp, dataset_tcp_checks = dataset_tcp_by_side(group, sides)
    checks.extend(dataset_tcp_checks)

    first_action = group.actions[0]
    for side in sides:
        targets = group.arm_joint_targets.get(side)
        if targets is None:
            checks.append(
                ReplayCheckResult(
                    name=f"{side}_qpos_targets_present",
                    ok=False,
                    message=(
                        f"{side} replay requires action.replay.{side}_arm_joint_target; "
                        "true hardware replay is disabled for this episode."
                    ),
                )
            )
            continue
        if np.asarray(targets).ndim != 2 or np.asarray(targets).shape[1] != ARM_JOINT_DOF:
            checks.append(
                ReplayCheckResult(
                    name=f"{side}_qpos_targets_shape",
                    ok=False,
                    message=f"{side} qpos targets must have shape (N, 6).",
                    details={"shape": list(np.asarray(targets).shape)},
                )
            )
            continue
        checks.append(
            ReplayCheckResult(
                name=f"{side}_qpos_targets_present",
                ok=True,
                level="info",
                message=f"{side} qpos targets present",
                details={"frame_count": int(np.asarray(targets).shape[0])},
            )
        )

        current = _finite_vector(current_qpos_by_side.get(side), length=ARM_JOINT_DOF)
        if current is None:
            checks.append(
                ReplayCheckResult(
                    name=f"{side}_current_qpos",
                    ok=False,
                    message=f"{side} current hardware qpos is unavailable.",
                )
            )
        else:
            target = np.asarray(targets[0], dtype=float).reshape(ARM_JOINT_DOF)
            diff = np.abs(target - current)
            max_diff = float(np.max(diff))
            unsafe = np.flatnonzero(diff > float(max_qpos_init_delta_rad))
            checks.append(
                ReplayCheckResult(
                    name=f"{side}_qpos_start_delta",
                    ok=unsafe.size == 0,
                    message=(
                        f"{side} qpos start delta max={np.rad2deg(max_diff):.2f}deg"
                    ),
                    details={
                        "max_delta_rad": max_diff,
                        "max_delta_deg": float(np.rad2deg(max_diff)),
                        "unsafe_joint_indices": [int(index) for index in unsafe],
                    },
                )
            )

        current_tcp = _finite_vector(current_tcp_flange_by_side.get(side), length=6)
        expected_tcp = dataset_tcp.get(side)
        if current_tcp is None:
            checks.append(
                ReplayCheckResult(
                    name=f"{side}_current_tcp_config",
                    ok=False,
                    message=f"{side} current configured TCP is unavailable.",
                )
            )
        elif expected_tcp is not None:
            pos_error = float(np.linalg.norm(current_tcp[:3] - expected_tcp[:3]))
            rot_error = float(np.linalg.norm(current_tcp[3:6] - expected_tcp[3:6]))
            checks.append(
                ReplayCheckResult(
                    name=f"{side}_tcp_numeric_match",
                    ok=pos_error <= float(tcp_pos_tolerance_m)
                    and rot_error <= float(tcp_rot_tolerance_rad),
                    message=(
                        f"{side} TCP numeric delta pos={pos_error:.6g}m rot={rot_error:.6g}rad"
                    ),
                    details={
                        "position_error_m": pos_error,
                        "rotation_error_rad": rot_error,
                        "dataset_tcp_pose_flange_m_rad": expected_tcp.tolist(),
                        "current_tcp_pose_flange_m_rad": current_tcp.tolist(),
                    },
                )
            )

        current_pose = _as_pose_matrix(current_tcp_pose_by_side.get(side))
        target_pose = xyzquat_to_pose_matrix(first_action[ACTION_ARM_POSE_SLICES[side]])
        if current_pose is None or target_pose is None:
            checks.append(
                ReplayCheckResult(
                    name=f"{side}_current_to_start_pose_delta",
                    ok=True,
                    level="warning",
                    message=f"{side} current TCP pose or first action pose is unavailable.",
                )
            )
        else:
            xyz_m, rot_rad = _pose_delta(current_pose, target_pose)
            checks.append(
                ReplayCheckResult(
                    name=f"{side}_current_to_start_pose_delta",
                    ok=True,
                    level="info",
                    message=f"{side} current->start pose delta xyz={xyz_m:.4f}m rot={np.rad2deg(rot_rad):.2f}deg",
                    details={
                        "xyz_m": xyz_m,
                        "rot_rad": rot_rad,
                        "rot_deg": float(np.rad2deg(rot_rad)),
                    },
                )
            )

    return ReplayPreflightReport(tuple(checks))

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

from hardware.rm65.calibration_profiles import ARM_ENDPOINT_TCP_POSE_FLANGE_M_RAD, load_profile

from robot_motion.deploy.schema import ACTION_ARM_POSE_SLICES, ARM_JOINT_DOF, xyzquat_to_pose_matrix
from robot_motion.replay.action_frame import replay_action_metadata, replay_end_effector_targets_at_index
from robot_motion.replay.trajectory import (
    QPOS_INIT_AUTO,
    REPLAY_TRAJECTORY_ACTION_POSE,
    REPLAY_TRAJECTORY_QPOS_TARGET,
    ReplayTrajectoryGroup,
    normalize_qpos_init_source,
    normalize_replay_trajectory_source,
    select_replay_qpos_init_targets,
)
from .pose_frames import (
    replay_action_pose_frame,
    replay_action_poses_in_hardware_base_at_index,
)


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


def _qpos_trajectory_checks(
    group: ReplayTrajectoryGroup,
    *,
    side: str,
) -> tuple[np.ndarray | None, list[ReplayCheckResult]]:
    checks: list[ReplayCheckResult] = []
    targets = group.arm_joint_targets.get(side)
    if targets is None:
        checks.append(
            ReplayCheckResult(
                name=f"{side}_qpos_targets_present",
                ok=False,
                message=(
                    f"{side} qpos trajectory mode requires "
                    f"action.replay.{side}_arm_joint_target for every replay frame."
                ),
            )
        )
        return None, checks
    target_array = np.asarray(targets, dtype=float)
    if target_array.ndim != 2 or target_array.shape[1] != ARM_JOINT_DOF:
        checks.append(
            ReplayCheckResult(
                name=f"{side}_qpos_targets_shape",
                ok=False,
                message=f"{side} qpos targets must have shape (N, 6).",
                details={"shape": list(target_array.shape)},
            )
        )
        return None, checks
    if target_array.shape[0] != int(group.frame_count):
        checks.append(
            ReplayCheckResult(
                name=f"{side}_qpos_targets_frame_count",
                ok=False,
                message=(
                    f"{side} qpos target frame count {target_array.shape[0]} "
                    f"does not match replay frame count {int(group.frame_count)}."
                ),
                details={"shape": list(target_array.shape), "frame_count": int(group.frame_count)},
            )
        )
        return None, checks
    if not np.all(np.isfinite(target_array)):
        invalid = np.argwhere(~np.isfinite(target_array))
        checks.append(
            ReplayCheckResult(
                name=f"{side}_qpos_targets_finite",
                ok=False,
                message=f"{side} qpos targets contain non-finite values.",
                details={
                    "first_invalid": invalid[0].astype(int).tolist() if invalid.size else None,
                },
            )
        )
        return None, checks
    checks.append(
        ReplayCheckResult(
            name=f"{side}_qpos_targets_present",
            ok=True,
            level="info",
            message=f"{side} qpos targets present",
            details={"frame_count": int(target_array.shape[0])},
        )
    )
    return target_array, checks


def _action_pose_trajectory_checks(
    group: ReplayTrajectoryGroup,
    *,
    side: str,
) -> list[ReplayCheckResult]:
    actions = np.asarray(group.actions, dtype=np.float32)
    required_width = int(ACTION_ARM_POSE_SLICES[side].stop or 0)
    if actions.ndim != 2 or actions.shape[1] < required_width:
        return [
            ReplayCheckResult(
                name=f"{side}_action_pose_targets_shape",
                ok=False,
                message=(
                    f"{side} action pose replay requires action shape (N, >={required_width}); "
                    f"got {actions.shape}."
                ),
            )
        ]
    poses = actions[:, ACTION_ARM_POSE_SLICES[side]]
    if poses.shape[0] != int(group.frame_count) or poses.shape[1] != 7:
        return [
            ReplayCheckResult(
                name=f"{side}_action_pose_targets_frame_count",
                ok=False,
                message=(
                    f"{side} action pose frame count {poses.shape[0]} "
                    f"does not match replay frame count {int(group.frame_count)}."
                ),
                details={"shape": list(poses.shape), "frame_count": int(group.frame_count)},
            )
        ]
    if not np.all(np.isfinite(poses)):
        invalid = np.argwhere(~np.isfinite(poses))
        return [
            ReplayCheckResult(
                name=f"{side}_action_pose_targets_finite",
                ok=False,
                message=f"{side} action pose targets contain non-finite values.",
                details={
                    "first_invalid": invalid[0].astype(int).tolist() if invalid.size else None,
                },
            )
        ]
    for frame_index, pose in enumerate(poses):
        if xyzquat_to_pose_matrix(pose) is None:
            return [
                ReplayCheckResult(
                    name=f"{side}_action_pose_targets_valid",
                    ok=False,
                    message=f"{side} action pose at frame {frame_index} cannot be converted to a pose matrix.",
                    details={"frame_index": int(frame_index)},
                )
            ]
    return [
        ReplayCheckResult(
            name=f"{side}_action_pose_targets_present",
            ok=True,
            level="info",
            message=f"{side} action pose targets present",
            details={"frame_count": int(poses.shape[0])},
        )
    ]


def _end_effector_replay_checks(
    group: ReplayTrajectoryGroup,
    *,
    side: str,
    backend_present: bool,
    active_sides: Sequence[str],
) -> list[ReplayCheckResult]:
    checks: list[ReplayCheckResult] = [
        ReplayCheckResult(
            name=f"{side}_end_effector_backend_present",
            ok=bool(backend_present),
            message=(
                f"{side} replay end-effector backend "
                f"{'configured' if backend_present else 'is missing'}."
            ),
        )
    ]
    try:
        for frame_index in range(int(group.frame_count)):
            replay_end_effector_targets_at_index(
                group,
                (side,),
                index=frame_index,
            )
    except Exception as exc:  # noqa: BLE001
        checks.append(
            ReplayCheckResult(
                name=f"{side}_end_effector_targets_valid",
                ok=False,
                message=str(exc),
            )
        )
        return checks
    checks.append(
        ReplayCheckResult(
            name=f"{side}_end_effector_targets_present",
            ok=True,
            level="info",
            message=f"{side} end-effector replay targets present",
            details={"frame_count": int(group.frame_count)},
        )
    )
    metadata = replay_action_metadata(group, active_sides=active_sides)
    hand_specs = metadata.get("hand_action_specs_by_side")
    has_spec = isinstance(hand_specs, Mapping) and side in hand_specs
    checks.append(
        ReplayCheckResult(
            name=f"{side}_end_effector_action_spec",
            ok=True,
            level="info" if has_spec else "warning",
            message=(
                f"{side} end-effector HandActionSpec loaded"
                if has_spec
                else f"{side} end-effector HandActionSpec missing; resolver will use q6 default semantics."
            ),
            details={"hand_action_spec": hand_specs.get(side) if isinstance(hand_specs, Mapping) else None},
        )
    )
    return checks


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
    trajectory_source: str = REPLAY_TRAJECTORY_ACTION_POSE,
    qpos_init_source: str = QPOS_INIT_AUTO,
    end_effectors_enabled: bool = False,
    end_effector_backend_sides: Sequence[str] = (),
) -> ReplayPreflightReport:
    sides = tuple(str(side) for side in active_sides)
    trajectory_source = normalize_replay_trajectory_source(trajectory_source)
    qpos_init_source = normalize_qpos_init_source(qpos_init_source)
    end_effector_backend_side_set = {str(side) for side in end_effector_backend_sides}
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

    try:
        action_pose_frame = replay_action_pose_frame(group)
        first_hardware_poses = replay_action_poses_in_hardware_base_at_index(
            group, sides, index=0
        )
        checks.append(
            ReplayCheckResult(
                name="action_pose_frame_contract",
                ok=True,
                level="info",
                message=(
                    f"replay action poses resolved from {action_pose_frame} "
                    "to RM65 controller base"
                ),
                details={"source_frame": action_pose_frame, "hardware_frame": "arm_base"},
            )
        )
    except Exception as exc:  # noqa: BLE001
        first_hardware_poses = {}
        checks.append(
            ReplayCheckResult(
                name="action_pose_frame_contract",
                ok=False,
                message=str(exc),
            )
        )

    for side in sides:
        if trajectory_source == REPLAY_TRAJECTORY_QPOS_TARGET:
            _, qpos_checks = _qpos_trajectory_checks(group, side=side)
            checks.extend(qpos_checks)
        else:
            checks.extend(_action_pose_trajectory_checks(group, side=side))

        qpos_init_targets, qpos_init_sources = select_replay_qpos_init_targets(
            group,
            (side,),
            index=0,
            source=qpos_init_source,
        )
        target = qpos_init_targets.get(side)
        target_source = qpos_init_sources.get(side)
        if target is None:
            checks.append(
                ReplayCheckResult(
                    name=f"{side}_qpos_init_target_present",
                    ok=False,
                    message=(
                        f"{side} replay start requires a finite first-frame qpos init target "
                        f"from source={qpos_init_source!r}."
                    ),
                    details={
                        "qpos_init_source": qpos_init_source,
                        "available_sources": sorted(
                            select_replay_qpos_init_targets(
                                group,
                                (side,),
                                index=0,
                                source=QPOS_INIT_AUTO,
                            )[1].values()
                        ),
                    },
                )
            )
        else:
            checks.append(
                ReplayCheckResult(
                    name=f"{side}_qpos_init_target_present",
                    ok=True,
                    level="info",
                    message=f"{side} qpos init target selected from {target_source}",
                    details={"qpos_init_source": target_source},
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
        elif target is not None:
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
                        "qpos_init_source": target_source,
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
        target_pose = xyzquat_to_pose_matrix(first_hardware_poses.get(side))
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

        if bool(end_effectors_enabled):
            checks.extend(
                _end_effector_replay_checks(
                    group,
                    side=side,
                    backend_present=side in end_effector_backend_side_set,
                    active_sides=sides,
                )
            )

    return ReplayPreflightReport(tuple(checks))

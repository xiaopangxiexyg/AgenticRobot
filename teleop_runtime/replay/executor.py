from __future__ import annotations

import argparse
import inspect
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from robot_motion.deploy.schema import (
    ARM_JOINT_DOF,
    pose_matrix_to_xyzquat,
    xyzquat_to_pose_matrix,
)
from robot_motion.position_control import (
    DEFAULT_POSITION_COMMAND_HZ,
    DEFAULT_POSITION_REFERENCE_HZ,
    DEFAULT_POSITION_SAMPLE_HZ,
    PositionKeyframeSafetyConfig,
    TimestampedMotionTarget,
    limit_joint_keyframes,
    limit_pose_keyframes,
    resample_motion_targets,
)
from robot_motion.pose import pose6_to_matrix
from robot_motion.replay.action_frame import replay_teleop_command_at_index
from robot_motion.replay.proposal_source import (
    ReplaySourceFrameEmission,
    build_replay_source_frame_emission,
)
from robot_motion.replay.trajectory import (
    QPOS_INIT_AUTO,
    REPLAY_TRAJECTORY_ACTION_POSE,
    REPLAY_TRAJECTORY_QPOS_TARGET,
    ReplayGroupSpec,
    ReplayTrajectoryGroup,
    normalize_qpos_init_source,
    normalize_replay_trajectory_source,
    select_replay_qpos_init_targets,
)
from robot_motion.rm65 import (
    DEFAULT_COLLISION_SCENE,
    DEFAULT_DUAL_ARM_COLLISION_DISTANCE_EPSILON_M,
    DEFAULT_DUAL_ARM_COLLISION_MIN_DISTANCE_M,
    LEGACY_POSITION_CONTROL_PROFILE_NAME,
    RM65_BIMANUAL_POSITION_V1,
    RM65_BIMANUAL_POSITION_V1_NAME,
    RM65_POSE_LIFT_V4_QP_CONSTRAINED_NAME,
    RM65_ABSOLUTE_Q_MAX_RAD,
    RM65_ABSOLUTE_Q_MIN_RAD,
    RM65PositionCollisionPlanner,
    RM65PositionCollisionPlannerConfig,
    build_rm65_pose_lift_ik_tuning,
    resolve_rm65_pose_lift_profile,
)
from robot_motion.joint_limits import load_joint_velocity_limit_profiles
from robot_motion.runtime.lifecycle import (
    LifecycleFailure,
    collect_lifecycle_failures,
    raise_lifecycle_failures,
)

from ..execution_mode_contract import (
    DatasetArmControlModeError,
    validate_dataset_arm_control_mode,
)
from .arm_init.qpos_execute import (
    DEFAULT_QPOS_INIT_BRANCH_SEARCH,
    DEFAULT_QPOS_INIT_BRANCH_SEARCH_MAX_BRANCHES,
    DEFAULT_QPOS_INIT_COLLISION_SCENE,
    DEFAULT_QPOS_INIT_COLLISION_HOLD_DISTANCE_M,
    DEFAULT_QPOS_INIT_COLLISION_SLOW_DISTANCE_M,
    DEFAULT_QPOS_INIT_COLLISION_STOP_DISTANCE_M,
    DEFAULT_QPOS_INIT_JOINT_HOLD_MARGIN_DEG,
    DEFAULT_QPOS_INIT_JOINT_SLOW_MARGIN_DEG,
    DEFAULT_QPOS_INIT_MAX_JOINT_ABS_DEG,
    DEFAULT_SINGLE_ARM_MAX_QPOS_DELTA_DEG,
    DEFAULT_QPOS_TARGET_PATH,
    build_rm65_qpos_init_plan,
    load_qpos_target,
    normalize_qpos_init_max_angvel,
    _require_plan_completed,
)
from .checks import ReplayCheckResult, ReplayPreflightReport, run_replay_preflight
from .dataset import LoadedReplayEpisode
from .hardware import (
    DEFAULT_POSITION_COLLISION_STATE_MAX_AGE_SEC,
    DEFAULT_POSITION_COLLISION_STATE_PAIR_MAX_SKEW_SEC,
    RM65_ARM_CONTROL_MODE_POSITION,
    ReplayHardwareConfig,
    ReplayHardwareRig,
    normalize_active_sides,
    normalize_rm65_arm_control_mode,
)
from .source_binding import ReplayGuiSourceBinding, build_replay_gui_source_binding
from .pose_frames import (
    replay_action_poses_in_hardware_base_at_index,
    replay_group_in_hardware_base,
)
from .trajectory import build_replay_group_for_episode


REPLAY_POSITION_IK_SOLVE_MIN_ITERATIONS = (
    RM65_BIMANUAL_POSITION_V1.ik_execution.solve_min_iterations
)
REPLAY_POSITION_IK_SOLVE_MAX_ITERATIONS = (
    RM65_BIMANUAL_POSITION_V1.ik_execution.solve_max_iterations
)
REPLAY_POSITION_IK_ORIENTATION_WEIGHT = (
    RM65_BIMANUAL_POSITION_V1.ik_execution.orientation_weight
)
REPLAY_POSITION_IK_DAMPING = RM65_BIMANUAL_POSITION_V1.ik_execution.damping
REPLAY_POSITION_IK_ITERATION_RESERVE_MS = (
    RM65_BIMANUAL_POSITION_V1.ik_execution.iteration_reserve_ms
)
REPLAY_POSITION_JOINT_VELOCITY_LIMIT_PROFILE = (
    RM65_BIMANUAL_POSITION_V1.ik_execution.joint_velocity_limit_profile
)


def _replay_position_joint_velocity_limit_rad_s() -> tuple[float, ...]:
    profile = load_joint_velocity_limit_profiles()[
        REPLAY_POSITION_JOINT_VELOCITY_LIMIT_PROFILE
    ]
    values = np.deg2rad(np.asarray(profile.max_velocity_deg_s, dtype=float))
    return tuple(float(value) for value in values)


REPLAY_POSITION_COLLISION_MAX_JOINT_VELOCITY_RAD_S = (
    _replay_position_joint_velocity_limit_rad_s()
)


def _replay_position_ik_tuning(pose_lift_profile_name: str):
    """Build the formal replay objective without mutating shared IK defaults."""

    return build_rm65_pose_lift_ik_tuning(
        position_control_profile=RM65_BIMANUAL_POSITION_V1_NAME,
        pose_lift_profile=pose_lift_profile_name,
        orientation_weight=REPLAY_POSITION_IK_ORIENTATION_WEIGHT,
    )


_POSITION_COLLISION_EVENT_CAPACITY = 512
_POSITION_TRANSPORT_HEALTH_STAGES = (
    "before_arm_reference",
    "after_arm_reference_before_hand",
    "immediately_before_end_effector_dispatch",
)


def _serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


def _empty_transport_health_slots() -> dict[str, dict[str, Any]]:
    return {
        stage: {
            "status": (
                "not_applicable"
                if stage == "immediately_before_end_effector_dispatch"
                else "pending"
            ),
            "observations": [],
        }
        for stage in _POSITION_TRANSPORT_HEALTH_STAGES
    }


def _rate_label(value: float) -> str:
    rounded = round(float(value))
    if np.isclose(float(value), float(rounded), rtol=0.0, atol=1e-9):
        return str(int(rounded))
    return f"{float(value):g}"


def _print_worker_fault(stage: str, exc: BaseException) -> None:
    print(f"[ReplayHardware][ERROR] {stage} failed: {exc}", file=sys.stderr, flush=True)


def _episode_index_or_zero(episode: LoadedReplayEpisode) -> int:
    if episode.summary.episode_index is None:
        return 0
    return int(episode.summary.episode_index)


def _payload_value(payload: Mapping[str, Any], keys: Sequence[str], default: Any) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return default


def _float_tuple3(value: Any, *, name: str) -> tuple[float, float, float]:
    if isinstance(value, str):
        items = [item.strip() for item in value.replace(",", " ").split() if item.strip()]
    else:
        items = list(value)
    if len(items) != 3:
        raise ValueError(f"{name} must contain three values")
    parsed = tuple(float(item) for item in items)
    if not np.all(np.isfinite(parsed)):
        raise ValueError(f"{name} must contain finite values")
    return parsed  # type: ignore[return-value]


def _float_tuple6_or_scalar(value: Any, *, name: str) -> tuple[float, ...]:
    if isinstance(value, str):
        items = [item.strip() for item in value.replace(",", " ").split() if item.strip()]
    elif np.isscalar(value):
        items = [value]
    else:
        items = list(value)
    if len(items) == 1:
        items = items * ARM_JOINT_DOF
    if len(items) != ARM_JOINT_DOF:
        raise ValueError(f"{name} must contain one value or {ARM_JOINT_DOF} values")
    parsed = tuple(float(item) for item in items)
    if not np.all(np.isfinite(parsed)) or np.any(np.asarray(parsed) <= 0.0):
        raise ValueError(f"{name} must contain positive finite values")
    return parsed


def _optional_float_tuple6_or_scalar(value: Any, *, name: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return _float_tuple6_or_scalar(value, name=name)


def _bool_value(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean value")


def _episode_group_args(
    episode: LoadedReplayEpisode,
    *,
    active_sides: Sequence[str],
    action_column: str,
) -> argparse.Namespace:
    episode_index = episode.summary.episode_index
    use_dataset = episode_index is not None and episode.dataset_root.exists()
    return argparse.Namespace(
        replay_dataset=str(episode.dataset_root) if use_dataset else "",
        replay_parquet="" if use_dataset else str(episode.summary.parquet_path),
        replay_episode=_episode_index_or_zero(episode),
        replay_action_column=str(action_column or "action"),
        replay_all_episodes=False,
        active_sides=tuple(active_sides),
    )


def build_episode_replay_group(
    episode: LoadedReplayEpisode,
    *,
    active_sides: Sequence[str],
    action_column: str = "action",
) -> ReplayTrajectoryGroup:
    sides = normalize_active_sides(tuple(active_sides))
    episode_index = _episode_index_or_zero(episode)
    args = _episode_group_args(episode, active_sides=sides, action_column=action_column)
    group = build_replay_group_for_episode(
        args,
        episode_index=episode_index,
        group_spec=ReplayGroupSpec(
            key="g",
            start_index=0,
            end_index=-1,
            episode_range=str(episode_index),
        ),
        balanced_episode_count=1,
        require_arm_joint_targets=False,
    )
    source_metadata = dict(group.metadata.get("source_metadata") or {})
    if episode.summary.metadata and not isinstance(source_metadata.get("episode_metadata"), Mapping):
        source_metadata["episode_metadata"] = dict(episode.summary.metadata)
    group.metadata["source_metadata"] = source_metadata
    group.metadata["loaded_episode"] = {
        "episode_id": episode.summary.episode_id,
        "episode_index": episode.summary.episode_index,
        "parquet_path": str(episode.summary.parquet_path),
        "dataset_root": str(episode.dataset_root),
    }
    return group


def _supports_keyword(callable_obj: Any, keyword: str) -> bool:
    """Return whether a callable accepts a keyword without invoking it twice."""

    try:
        parameters = inspect.signature(callable_obj).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        or parameter.name == str(keyword)
        for parameter in parameters
    )


def _send_joint_reference(
    rig: Any,
    qpos_by_side: Mapping[str, Any],
    *,
    source: str,
    metadata: Mapping[str, Any],
) -> None:
    sender = rig.send_joint_targets
    if _supports_keyword(sender, "metadata"):
        sender(qpos_by_side, source=str(source), metadata=dict(metadata))
    else:
        sender(qpos_by_side, source=str(source))


def _send_pose_reference(
    rig: Any,
    pose_by_side: Mapping[str, Any],
    *,
    source: str,
    metadata: Mapping[str, Any],
) -> None:
    sender = rig.send_action_poses
    if _supports_keyword(sender, "metadata"):
        sender(pose_by_side, source=str(source), metadata=dict(metadata))
    else:
        sender(pose_by_side, source=str(source))


def _position_keyframe_safety_metadata(result: Any) -> dict[str, Any]:
    return {
        "modified": bool(result.modified),
        "modified_by_side": {
            str(side): bool(value)
            for side, value in result.modified_by_side.items()
        },
        "reasons_by_side": {
            str(side): [str(reason) for reason in reasons]
            for side, reasons in result.reasons_by_side.items()
        },
        "details_by_side": {
            str(side): dict(details)
            for side, details in result.metadata_by_side.items()
        },
        "temporal_smoothing_applied": False,
    }


def _position_collision_result_metadata(result: Any, *, debug: bool) -> dict[str, Any]:
    planner_metadata = dict(getattr(result, "metadata", {}) or {})
    endpoint_checked = bool(planner_metadata.get("endpoint_collision_checked", False))
    endpoint_passed = bool(planner_metadata.get("endpoint_collision_passed", False))
    summary: dict[str, Any] = {
        "planner": str(planner_metadata.get("planner", "rm65_coupled_position_collision")),
        "stage": "reference_pose_to_collision_aware_qpos",
        "sequence": int(getattr(result, "sequence", 0)),
        "status": str(getattr(result, "status", "unknown")),
        "valid": bool(getattr(result, "valid", False)),
        "hold": bool(getattr(result, "hold", True)),
        "modification_reasons": [
            str(reason) for reason in tuple(getattr(result, "modification_reasons", ()))
        ],
        "scene_profile": planner_metadata.get("scene_profile"),
        "collision_pair_count": int(planner_metadata.get("collision_pair_count", 0)),
        "collision_distance_m": planner_metadata.get("collision_distance_m"),
        "collision_min_distance_m": planner_metadata.get("collision_min_distance_m"),
        "collision_distance_epsilon_m": planner_metadata.get(
            "collision_distance_epsilon_m"
        ),
        "endpoint_collision_checked": endpoint_checked,
        "endpoint_collision_passed": endpoint_passed,
        "segment_collision_checked": False,
        "segment_collision_certified": False,
        "collision_certification_status": "reference_endpoint_only_segment_not_certified",
        "temporal_smoothing_applied": False,
        "solve_ms": planner_metadata.get("solve_ms"),
        "frame_total_ms": planner_metadata.get("frame_total_ms"),
        "solve_time_budget_exceeded": bool(
            planner_metadata.get("solve_time_budget_exceeded", False)
        ),
    }
    if debug:
        summary["debug"] = {
            "q_ik_raw_by_side": _serializable(
                dict(getattr(result, "q_ik_raw_by_side", {}) or {})
            ),
            "q_ref_by_side": _serializable(
                dict(getattr(result, "q_ref_by_side", {}) or {})
            ),
            "target_endpoint_pose_world_by_side": _serializable(
                dict(getattr(result, "target_endpoint_pose_world_by_side", {}) or {})
            ),
            "realized_endpoint_pose_world_by_side": _serializable(
                dict(getattr(result, "realized_endpoint_pose_world_by_side", {}) or {})
            ),
            "planner_metadata": _serializable(planner_metadata),
        }
    return summary


def _safe_pose_keyframes(
    rig: Any,
    group: ReplayTrajectoryGroup,
    *,
    active_sides: Sequence[str],
    config: PositionKeyframeSafetyConfig,
) -> list[TimestampedMotionTarget]:
    reference_by_side: dict[str, np.ndarray] = {}
    current_pose_getter = getattr(rig, "current_tcp_pose_by_side", None)
    if callable(current_pose_getter):
        current = current_pose_getter()
        if isinstance(current, Mapping):
            for side in active_sides:
                xyzquat = pose_matrix_to_xyzquat(current.get(str(side)))
                if xyzquat is not None:
                    reference_by_side[str(side)] = np.asarray(xyzquat, dtype=float).reshape(7)

    samples: list[TimestampedMotionTarget] = []
    for index in range(int(group.frame_count)):
        raw = replay_action_poses_in_hardware_base_at_index(
            group,
            active_sides,
            index=int(index),
        )
        result = limit_pose_keyframes(
            raw,
            reference_by_side=reference_by_side or None,
            config=config,
        )
        if not result.valid:
            raise RuntimeError(
                "position replay pose keyframe safety constraints are infeasible: "
                + ", ".join(result.faults)
            )
        safe_xyzquat = {
            str(side): np.asarray(value, dtype=float).reshape(7)
            for side, value in result.values_by_side.items()
        }
        pose_by_side: dict[str, np.ndarray] = {}
        for side, xyzquat in safe_xyzquat.items():
            matrix = xyzquat_to_pose_matrix(xyzquat)
            if matrix is None:
                raise RuntimeError(f"{side} safe replay keyframe could not be converted to a pose matrix.")
            pose_by_side[side] = np.asarray(matrix, dtype=float).reshape(4, 4)
        samples.append(
            TimestampedMotionTarget(
                timestamp=group.relative_time_sec(index),
                source_index=int(index),
                pose_by_side=pose_by_side,
                metadata={
                    "source_frame_index": int(index),
                    "position_keyframe_safety": _position_keyframe_safety_metadata(result),
                },
            )
        )
        reference_by_side = {
            side: value.copy()
            for side, value in safe_xyzquat.items()
        }
    return samples


def _safe_qpos_keyframes(
    rig: Any,
    group: ReplayTrajectoryGroup,
    *,
    active_sides: Sequence[str],
    config: PositionKeyframeSafetyConfig,
) -> list[TimestampedMotionTarget]:
    reference_by_side: dict[str, np.ndarray] = {}
    current_qpos_getter = getattr(rig, "current_qpos_by_side", None)
    if callable(current_qpos_getter):
        current = current_qpos_getter()
        if isinstance(current, Mapping):
            for side in active_sides:
                value = current.get(str(side))
                if value is None:
                    continue
                qpos = np.asarray(value, dtype=float).reshape(-1)
                if qpos.shape == (ARM_JOINT_DOF,) and np.all(np.isfinite(qpos)):
                    reference_by_side[str(side)] = qpos.copy()

    samples: list[TimestampedMotionTarget] = []
    for index in range(int(group.frame_count)):
        raw = {
            str(side): np.asarray(group.arm_joint_targets[str(side)][int(index)], dtype=float).reshape(
                ARM_JOINT_DOF
            )
            for side in active_sides
        }
        result = limit_joint_keyframes(
            raw,
            reference_by_side=reference_by_side or None,
            config=config,
        )
        if not result.valid:
            raise RuntimeError(
                "position replay qpos keyframe safety constraints are infeasible: "
                + ", ".join(result.faults)
            )
        safe_qpos = {
            str(side): np.asarray(value, dtype=float).reshape(ARM_JOINT_DOF)
            for side, value in result.values_by_side.items()
        }
        samples.append(
            TimestampedMotionTarget(
                timestamp=group.relative_time_sec(index),
                source_index=int(index),
                qpos_by_side=safe_qpos,
                metadata={
                    "source_frame_index": int(index),
                    "position_keyframe_safety": _position_keyframe_safety_metadata(result),
                },
            )
        )
        reference_by_side = {
            side: value.copy()
            for side, value in safe_qpos.items()
        }
    return samples


@dataclass(frozen=True)
class ReplayExecutionSettings:
    send_actions_enabled: bool = False
    action_column: str = "action"
    trajectory_source: str = REPLAY_TRAJECTORY_ACTION_POSE
    qpos_init_source: str = QPOS_INIT_AUTO
    end_effectors_enabled: bool = False
    qpos_reached_tolerance_rad: float = 0.1
    move_to_start_timeout_s: float = 30.0
    tcp_pos_tolerance_m: float = 1e-4
    tcp_rot_tolerance_rad: float = 1e-3
    max_qpos_init_delta_rad: float = float(np.pi)
    qpos_init_dt_sec: float = 0.05
    qpos_init_joint_velocity_profile: str = "rm65_init_slow"
    qpos_init_max_angvel_rad_s: tuple[float, ...] | None = None
    qpos_init_max_steps_per_side: int = 10000
    qpos_init_collision_check: bool = True
    qpos_init_collision_scene_profile: str = DEFAULT_QPOS_INIT_COLLISION_SCENE
    qpos_init_collision_slow_distance_m: float = DEFAULT_QPOS_INIT_COLLISION_SLOW_DISTANCE_M
    qpos_init_collision_hold_distance_m: float = DEFAULT_QPOS_INIT_COLLISION_HOLD_DISTANCE_M
    qpos_init_collision_stop_distance_m: float = DEFAULT_QPOS_INIT_COLLISION_STOP_DISTANCE_M
    qpos_init_max_joint_abs_deg: float = DEFAULT_QPOS_INIT_MAX_JOINT_ABS_DEG
    qpos_init_joint_slow_margin_deg: float = DEFAULT_QPOS_INIT_JOINT_SLOW_MARGIN_DEG
    qpos_init_joint_hold_margin_deg: float = DEFAULT_QPOS_INIT_JOINT_HOLD_MARGIN_DEG
    qpos_init_single_arm_max_qpos_delta_deg: float = DEFAULT_SINGLE_ARM_MAX_QPOS_DELTA_DEG
    qpos_init_branch_search: bool = DEFAULT_QPOS_INIT_BRANCH_SEARCH
    qpos_init_branch_search_max_branches: int = DEFAULT_QPOS_INIT_BRANCH_SEARCH_MAX_BRANCHES
    position_workspace_min: tuple[float, float, float] = (-0.60, -0.80, 0.00)
    position_workspace_max: tuple[float, float, float] = (0.60, 0.80, 0.90)
    position_max_xyz_step_m: float = 0.10
    position_max_rot_step_deg: float = 15.0
    position_max_joint_step_rad: float = float(np.deg2rad(6.0))
    position_control_profile_name: str = RM65_BIMANUAL_POSITION_V1_NAME
    pose_lift_profile_name: str = RM65_POSE_LIFT_V4_QP_CONSTRAINED_NAME
    position_collision_required: bool = True
    position_collision_scene_profile: str = DEFAULT_COLLISION_SCENE
    position_collision_min_distance_m: float = DEFAULT_DUAL_ARM_COLLISION_MIN_DISTANCE_M
    position_collision_distance_epsilon_m: float = (
        DEFAULT_DUAL_ARM_COLLISION_DISTANCE_EPSILON_M
    )
    position_collision_max_joint_velocity_rad_s: tuple[float, ...] = (
        REPLAY_POSITION_COLLISION_MAX_JOINT_VELOCITY_RAD_S
    )
    position_collision_max_joint_step_rad: tuple[float, ...] | None = None
    position_collision_debug: bool = False

    def __post_init__(self) -> None:
        trajectory_source = normalize_replay_trajectory_source(
            self.trajectory_source
        )
        object.__setattr__(self, "trajectory_source", trajectory_source)
        profile_name = str(self.position_control_profile_name).strip()
        if profile_name not in {
            RM65_BIMANUAL_POSITION_V1_NAME,
            LEGACY_POSITION_CONTROL_PROFILE_NAME,
        }:
            raise ValueError(
                f"unsupported position_control_profile_name: {profile_name!r}"
            )
        object.__setattr__(self, "position_control_profile_name", profile_name)
        pose_lift_profile = resolve_rm65_pose_lift_profile(
            self.pose_lift_profile_name
        )
        object.__setattr__(self, "pose_lift_profile_name", pose_lift_profile.name)
        if (
            bool(self.send_actions_enabled)
            and trajectory_source == REPLAY_TRAJECTORY_ACTION_POSE
            and not pose_lift_profile.hardware_dispatch_allowed
        ):
            raise ValueError(
                f"pose-lift profile {pose_lift_profile.name!r} forbids replay "
                "arm output."
            )
        workspace_min = np.asarray(self.position_workspace_min, dtype=float).reshape(3)
        workspace_max = np.asarray(self.position_workspace_max, dtype=float).reshape(3)
        if not np.all(np.isfinite(workspace_min)) or not np.all(np.isfinite(workspace_max)):
            raise ValueError("position replay workspace bounds must be finite.")
        if np.any(workspace_min > workspace_max):
            raise ValueError("position replay workspace_min must not exceed workspace_max.")
        for name in (
            "position_max_xyz_step_m",
            "position_max_rot_step_deg",
            "position_max_joint_step_rad",
            "position_collision_min_distance_m",
            "position_collision_distance_epsilon_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if not str(self.position_collision_scene_profile).strip():
            raise ValueError("position_collision_scene_profile must be non-empty.")
        if (
            float(self.position_collision_distance_epsilon_m)
            > float(self.position_collision_min_distance_m)
        ):
            raise ValueError(
                "position_collision_distance_epsilon_m must not exceed "
                "position_collision_min_distance_m."
            )
        max_velocity = _float_tuple6_or_scalar(
            self.position_collision_max_joint_velocity_rad_s,
            name="position_collision_max_joint_velocity_rad_s",
        )
        max_step = _optional_float_tuple6_or_scalar(
            self.position_collision_max_joint_step_rad,
            name="position_collision_max_joint_step_rad",
        )
        object.__setattr__(
            self,
            "position_collision_scene_profile",
            str(self.position_collision_scene_profile).strip(),
        )
        object.__setattr__(
            self,
            "position_collision_max_joint_velocity_rad_s",
            max_velocity,
        )
        object.__setattr__(self, "position_collision_max_joint_step_rad", max_step)
        object.__setattr__(
            self,
            "position_collision_required",
            _bool_value(
                self.position_collision_required,
                name="position_collision_required",
            ),
        )
        object.__setattr__(
            self,
            "position_collision_debug",
            _bool_value(self.position_collision_debug, name="position_collision_debug"),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], current: "ReplayExecutionSettings") -> "ReplayExecutionSettings":
        send_actions_value = (
            payload.get("send_actions")
            if "send_actions" in payload
            else payload.get("sendActions", current.send_actions_enabled)
        )
        return cls(
            send_actions_enabled=_bool_value(
                send_actions_value,
                name="send_actions",
            ),
            action_column=str(payload.get("action_column") or payload.get("actionColumn") or current.action_column),
            trajectory_source=normalize_replay_trajectory_source(
                payload.get("trajectory_source")
                or payload.get("trajectorySource")
                or current.trajectory_source
            ),
            qpos_init_source=normalize_qpos_init_source(
                payload.get("qpos_init_source")
                or payload.get("qposInitSource")
                or current.qpos_init_source
            ),
            end_effectors_enabled=bool(
                payload.get("end_effectors_enabled")
                if "end_effectors_enabled" in payload
                else payload.get(
                    "endEffectorsEnabled",
                    payload.get(
                        "hands_enabled",
                        payload.get("handsEnabled", current.end_effectors_enabled),
                    ),
                )
            ),
            qpos_reached_tolerance_rad=float(
                payload.get("qpos_reached_tolerance_rad")
                or payload.get("qposReachedToleranceRad")
                or current.qpos_reached_tolerance_rad
            ),
            move_to_start_timeout_s=float(
                payload.get("move_to_start_timeout_s")
                or payload.get("moveToStartTimeoutS")
                or current.move_to_start_timeout_s
            ),
            tcp_pos_tolerance_m=float(
                payload.get("tcp_pos_tolerance_m")
                or payload.get("tcpPosToleranceM")
                or current.tcp_pos_tolerance_m
            ),
            tcp_rot_tolerance_rad=float(
                payload.get("tcp_rot_tolerance_rad")
                or payload.get("tcpRotToleranceRad")
                or current.tcp_rot_tolerance_rad
            ),
            max_qpos_init_delta_rad=float(
                payload.get("max_qpos_init_delta_rad")
                or payload.get("maxQposInitDeltaRad")
                or current.max_qpos_init_delta_rad
            ),
            qpos_init_dt_sec=float(
                payload.get("qpos_init_dt_sec")
                or payload.get("qposInitDtSec")
                or current.qpos_init_dt_sec
            ),
            qpos_init_joint_velocity_profile=str(
                payload.get("qpos_init_joint_velocity_profile")
                or payload.get("qposInitJointVelocityProfile")
                or current.qpos_init_joint_velocity_profile
            ),
            qpos_init_max_angvel_rad_s=normalize_qpos_init_max_angvel(
                _payload_value(
                    payload,
                    (
                        "qpos_init_max_angvel_rad_s",
                        "qposInitMaxAngvelRadS",
                        "qpos_init_arm_max_angvel",
                        "qposInitArmMaxAngvel",
                    ),
                    current.qpos_init_max_angvel_rad_s,
                )
            ),
            qpos_init_max_steps_per_side=int(
                payload.get("qpos_init_max_steps_per_side")
                or payload.get("qposInitMaxStepsPerSide")
                or current.qpos_init_max_steps_per_side
            ),
            qpos_init_collision_check=bool(
                payload.get("qpos_init_collision_check")
                if "qpos_init_collision_check" in payload
                else payload.get("qposInitCollisionCheck", current.qpos_init_collision_check)
            ),
            qpos_init_collision_scene_profile=str(
                payload.get("qpos_init_collision_scene_profile")
                or payload.get("qposInitCollisionSceneProfile")
                or current.qpos_init_collision_scene_profile
            ),
            qpos_init_collision_slow_distance_m=float(
                payload.get("qpos_init_collision_slow_distance_m")
                or payload.get("qposInitCollisionSlowDistanceM")
                or current.qpos_init_collision_slow_distance_m
            ),
            qpos_init_collision_hold_distance_m=float(
                payload.get("qpos_init_collision_hold_distance_m")
                or payload.get("qposInitCollisionHoldDistanceM")
                or current.qpos_init_collision_hold_distance_m
            ),
            qpos_init_collision_stop_distance_m=float(
                payload.get("qpos_init_collision_stop_distance_m")
                or payload.get("qposInitCollisionStopDistanceM")
                or current.qpos_init_collision_stop_distance_m
            ),
            qpos_init_max_joint_abs_deg=float(
                payload.get("qpos_init_max_joint_abs_deg")
                or payload.get("qposInitMaxJointAbsDeg")
                or current.qpos_init_max_joint_abs_deg
            ),
            qpos_init_joint_slow_margin_deg=float(
                payload.get("qpos_init_joint_slow_margin_deg")
                or payload.get("qposInitJointSlowMarginDeg")
                or current.qpos_init_joint_slow_margin_deg
            ),
            qpos_init_joint_hold_margin_deg=float(
                payload.get("qpos_init_joint_hold_margin_deg")
                or payload.get("qposInitJointHoldMarginDeg")
                or current.qpos_init_joint_hold_margin_deg
            ),
            qpos_init_single_arm_max_qpos_delta_deg=float(
                payload.get("qpos_init_single_arm_max_qpos_delta_deg")
                or payload.get("qposInitSingleArmMaxQposDeltaDeg")
                or current.qpos_init_single_arm_max_qpos_delta_deg
            ),
            qpos_init_branch_search=bool(
                payload.get("qpos_init_branch_search")
                if "qpos_init_branch_search" in payload
                else payload.get("qposInitBranchSearch", current.qpos_init_branch_search)
            ),
            qpos_init_branch_search_max_branches=int(
                payload.get("qpos_init_branch_search_max_branches")
                or payload.get("qposInitBranchSearchMaxBranches")
                or current.qpos_init_branch_search_max_branches
            ),
            position_workspace_min=_float_tuple3(
                _payload_value(
                    payload,
                    ("position_workspace_min", "positionWorkspaceMin"),
                    current.position_workspace_min,
                ),
                name="position_workspace_min",
            ),
            position_workspace_max=_float_tuple3(
                _payload_value(
                    payload,
                    ("position_workspace_max", "positionWorkspaceMax"),
                    current.position_workspace_max,
                ),
                name="position_workspace_max",
            ),
            position_max_xyz_step_m=float(
                _payload_value(
                    payload,
                    ("position_max_xyz_step_m", "positionMaxXyzStepM"),
                    current.position_max_xyz_step_m,
                )
            ),
            position_max_rot_step_deg=float(
                _payload_value(
                    payload,
                    ("position_max_rot_step_deg", "positionMaxRotStepDeg"),
                    current.position_max_rot_step_deg,
                )
            ),
            position_max_joint_step_rad=float(
                _payload_value(
                    payload,
                    ("position_max_joint_step_rad", "positionMaxJointStepRad"),
                    current.position_max_joint_step_rad,
                )
            ),
            position_control_profile_name=str(
                _payload_value(
                    payload,
                    ("position_control_profile_name", "positionControlProfileName"),
                    current.position_control_profile_name,
                )
            ),
            pose_lift_profile_name=str(
                _payload_value(
                    payload,
                    ("pose_lift_profile_name", "poseLiftProfileName"),
                    current.pose_lift_profile_name,
                )
            ),
            position_collision_required=_bool_value(
                _payload_value(
                    payload,
                    ("position_collision_required", "positionCollisionRequired"),
                    current.position_collision_required,
                ),
                name="position_collision_required",
            ),
            position_collision_scene_profile=str(
                _payload_value(
                    payload,
                    (
                        "position_collision_scene_profile",
                        "positionCollisionSceneProfile",
                    ),
                    current.position_collision_scene_profile,
                )
            ),
            position_collision_min_distance_m=float(
                _payload_value(
                    payload,
                    (
                        "position_collision_min_distance_m",
                        "positionCollisionMinDistanceM",
                    ),
                    current.position_collision_min_distance_m,
                )
            ),
            position_collision_distance_epsilon_m=float(
                _payload_value(
                    payload,
                    (
                        "position_collision_distance_epsilon_m",
                        "positionCollisionDistanceEpsilonM",
                    ),
                    current.position_collision_distance_epsilon_m,
                )
            ),
            position_collision_max_joint_velocity_rad_s=_float_tuple6_or_scalar(
                _payload_value(
                    payload,
                    (
                        "position_collision_max_joint_velocity_rad_s",
                        "positionCollisionMaxJointVelocityRadS",
                    ),
                    current.position_collision_max_joint_velocity_rad_s,
                ),
                name="position_collision_max_joint_velocity_rad_s",
            ),
            position_collision_max_joint_step_rad=_optional_float_tuple6_or_scalar(
                _payload_value(
                    payload,
                    (
                        "position_collision_max_joint_step_rad",
                        "positionCollisionMaxJointStepRad",
                    ),
                    current.position_collision_max_joint_step_rad,
                ),
                name="position_collision_max_joint_step_rad",
            ),
            position_collision_debug=_bool_value(
                _payload_value(
                    payload,
                    ("position_collision_debug", "positionCollisionDebug"),
                    current.position_collision_debug,
                ),
                name="position_collision_debug",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "send_actions_enabled": bool(self.send_actions_enabled),
            "action_column": str(self.action_column),
            "trajectory_source": normalize_replay_trajectory_source(self.trajectory_source),
            "qpos_init_source": normalize_qpos_init_source(self.qpos_init_source),
            "end_effectors_enabled": bool(self.end_effectors_enabled),
            "qpos_reached_tolerance_rad": float(self.qpos_reached_tolerance_rad),
            "move_to_start_timeout_s": float(self.move_to_start_timeout_s),
            "tcp_pos_tolerance_m": float(self.tcp_pos_tolerance_m),
            "tcp_rot_tolerance_rad": float(self.tcp_rot_tolerance_rad),
            "max_qpos_init_delta_rad": float(self.max_qpos_init_delta_rad),
            "qpos_init_dt_sec": float(self.qpos_init_dt_sec),
            "qpos_init_joint_velocity_profile": str(self.qpos_init_joint_velocity_profile),
            "qpos_init_max_angvel_rad_s": (
                None
                if self.qpos_init_max_angvel_rad_s is None
                else [float(value) for value in self.qpos_init_max_angvel_rad_s]
            ),
            "qpos_init_max_steps_per_side": int(self.qpos_init_max_steps_per_side),
            "qpos_init_collision_check": bool(self.qpos_init_collision_check),
            "qpos_init_collision_scene_profile": str(self.qpos_init_collision_scene_profile),
            "qpos_init_collision_slow_distance_m": float(self.qpos_init_collision_slow_distance_m),
            "qpos_init_collision_hold_distance_m": float(self.qpos_init_collision_hold_distance_m),
            "qpos_init_collision_stop_distance_m": float(self.qpos_init_collision_stop_distance_m),
            "qpos_init_max_joint_abs_deg": float(self.qpos_init_max_joint_abs_deg),
            "qpos_init_joint_slow_margin_deg": float(self.qpos_init_joint_slow_margin_deg),
            "qpos_init_joint_hold_margin_deg": float(self.qpos_init_joint_hold_margin_deg),
            "qpos_init_single_arm_max_qpos_delta_deg": float(self.qpos_init_single_arm_max_qpos_delta_deg),
            "qpos_init_branch_search": bool(self.qpos_init_branch_search),
            "qpos_init_branch_search_max_branches": int(self.qpos_init_branch_search_max_branches),
            "position_workspace_min": [float(value) for value in self.position_workspace_min],
            "position_workspace_max": [float(value) for value in self.position_workspace_max],
            "position_max_xyz_step_m": float(self.position_max_xyz_step_m),
            "position_max_rot_step_deg": float(self.position_max_rot_step_deg),
            "position_max_joint_step_rad": float(self.position_max_joint_step_rad),
            "position_control_profile_name": self.position_control_profile_name,
            "pose_lift_profile_name": self.pose_lift_profile_name,
            "position_collision_required": bool(self.position_collision_required),
            "position_collision_scene_profile": str(self.position_collision_scene_profile),
            "position_collision_min_distance_m": float(
                self.position_collision_min_distance_m
            ),
            "position_collision_distance_epsilon_m": float(
                self.position_collision_distance_epsilon_m
            ),
            "position_collision_max_joint_velocity_rad_s": [
                float(value)
                for value in self.position_collision_max_joint_velocity_rad_s
            ],
            "position_collision_max_joint_step_rad": (
                None
                if self.position_collision_max_joint_step_rad is None
                else [
                    float(value)
                    for value in self.position_collision_max_joint_step_rad
                ]
            ),
            "position_collision_debug": bool(self.position_collision_debug),
        }


class ReplayExecutionController:
    def __init__(
        self,
        *,
        hardware_config: ReplayHardwareConfig | None = None,
        settings: ReplayExecutionSettings | None = None,
        position_collision_planner_factory: Callable[..., Any] | None = None,
        action_observer_queue: queue.Queue[ReplaySourceFrameEmission] | None = None,
    ):
        self._lock = threading.RLock()
        self._dispatch_lock = threading.RLock()
        self._hardware_config = hardware_config or ReplayHardwareConfig()
        self._settings = settings or ReplayExecutionSettings()
        self._position_collision_planner_factory = position_collision_planner_factory
        if action_observer_queue is not None and not (
            int(getattr(action_observer_queue, "maxsize", 0)) > 0
            or bool(getattr(action_observer_queue, "nonblocking_bounded", False))
        ):
            raise ValueError(
                "action_observer_queue must be explicitly bounded or expose an "
                "explicit nonblocking_bounded sink contract."
            )
        self._action_observer_queue = action_observer_queue
        self._action_observer_offer_count = 0
        self._action_observer_drop_count = 0
        self._action_observer_error_count = 0
        self._action_observer_last_error = ""
        self._rig: ReplayHardwareRig | None = None
        self._group: ReplayTrajectoryGroup | None = None
        self._preflight_report: ReplayPreflightReport | None = None
        self._source_binding: ReplayGuiSourceBinding | None = None
        self._source_binding_error = ""
        self._state = "configured"
        self._frame_index = 0
        self._last_error = ""
        self._last_message = ""
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._qpos_init_plan_summary: dict[str, Any] | None = None
        self._qpos_init_step_index = 0
        self._qpos_init_step_count = 0
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._position_collision_events: deque[dict[str, Any]] = deque(
            maxlen=_POSITION_COLLISION_EVENT_CAPACITY
        )
        self._position_collision_event_drop_count = 0
        self._position_collision_plan_count = 0
        self._position_collision_tracking_count = 0
        self._position_collision_limited_count = 0
        self._position_collision_backoff_count = 0
        self._position_collision_hold_count = 0
        self._position_collision_last: dict[str, Any] | None = None
        self._position_collision_reset: dict[str, Any] | None = None

    def close(self) -> None:
        failures = list(
            collect_lifecycle_failures((("abort", lambda: self.abort(wait=True)),))
        )
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                self._state = "fault"
                self._last_error = (
                    self._last_error
                    or "hardware replay worker did not stop; rig was intentionally left open"
                )
                failures.append(
                    LifecycleFailure(
                        label="worker.join",
                        error=RuntimeError(self._last_error),
                    )
                )
                rig = None
            else:
                rig = self._rig
        if rig is not None:
            rig_failures = collect_lifecycle_failures((("rig", rig.close),))
            failures.extend(rig_failures)
            if not rig_failures:
                with self._lock:
                    if self._rig is rig:
                        self._rig = None
        with self._lock:
            if not failures:
                self._state = "closed"
        raise_lifecycle_failures("replay_execution_controller.close", failures)

    def configure(
        self,
        *,
        hardware_config: ReplayHardwareConfig | None = None,
        settings: ReplayExecutionSettings | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._ensure_idle_locked()
            config_changed = hardware_config is not None and hardware_config != self._hardware_config
            if self._rig is not None and config_changed:
                self._rig.close()
                self._rig = None
            if hardware_config is not None:
                self._hardware_config = hardware_config
            if settings is not None:
                self._settings = settings
            self._group = None
            self._preflight_report = None
            self._source_binding = None
            self._source_binding_error = ""
            self._frame_index = 0
            self._clear_qpos_init_status_locked()
            self._reset_position_collision_diagnostics_locked()
            self._action_observer_offer_count = 0
            self._action_observer_drop_count = 0
            self._action_observer_error_count = 0
            self._action_observer_last_error = ""
            self._last_error = ""
            self._last_message = "hardware replay configured"
            self._state = "connected" if self._rig is not None and self._rig.connected else "configured"
            return self.status()

    def _offer_replay_source_frame(
        self,
        group: ReplayTrajectoryGroup,
        *,
        index: int,
        observed_at: float,
        schedule_start_at: float,
        target_effective_at: float | None = None,
        effective_until_at: float | None = None,
    ) -> None:
        """Best-effort nonblocking passive tap; never changes replay ownership."""

        observer_queue = self._action_observer_queue
        if observer_queue is None:
            return
        try:
            emission = build_replay_source_frame_emission(
                group,
                active_sides=tuple(self._hardware_config.active_sides),
                index=int(index),
                include_arms=True,
                include_end_effectors=bool(self._settings.end_effectors_enabled),
                observed_at=float(observed_at),
                schedule_start_at=float(schedule_start_at),
                target_effective_at=target_effective_at,
                effective_until_at=effective_until_at,
                clock_domain_id="local_monotonic",
            )
            observer_queue.put_nowait(emission)
            with self._lock:
                self._action_observer_offer_count += 1
        except queue.Full:
            with self._lock:
                self._action_observer_drop_count += 1
        except Exception as exc:  # noqa: BLE001 - passive tap cannot fault legacy replay.
            with self._lock:
                self._action_observer_error_count += 1
                self._action_observer_last_error = f"{type(exc).__name__}: {exc}"

    def configure_from_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            current_config = self._hardware_config
            current_settings = self._settings
        payload_dict = dict(payload)
        hardware_payload = {
            **current_config.to_dict(),
            **payload_dict,
        }
        mode_requested = "arm_control_mode" in payload_dict or "armControlMode" in payload_dict
        hz_requested = "arm_control_hz" in payload_dict or "armControlHz" in payload_dict
        follow_requested = "arm_canfd_follow" in payload_dict or "armCanfdFollow" in payload_dict
        if mode_requested and not hz_requested:
            hardware_payload.pop("arm_control_hz", None)
            hardware_payload.pop("armControlHz", None)
        if mode_requested and not follow_requested:
            hardware_payload.pop("arm_canfd_follow", None)
            hardware_payload.pop("armCanfdFollow", None)
        hardware_config = ReplayHardwareConfig.from_payload(
            hardware_payload
        )
        settings = ReplayExecutionSettings.from_payload(payload, current_settings)
        return self.configure(hardware_config=hardware_config, settings=settings)

    def set_send_actions(self, enabled: bool) -> dict[str, Any]:
        with self._lock:
            self._ensure_idle_locked(allow_ready=True)
            self._settings = replace(self._settings, send_actions_enabled=bool(enabled))
            self._last_message = f"arm output {'enabled' if enabled else 'disabled'}"
            return self.status()

    def clear_preflight(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_idle_locked(allow_ready=True)
            self._group = None
            self._preflight_report = None
            self._source_binding = None
            self._source_binding_error = ""
            self._frame_index = 0
            self._clear_qpos_init_status_locked()
            self._reset_position_collision_diagnostics_locked()
            self._last_error = ""
            self._last_message = "preflight cleared"
            if self._rig is not None and self._rig.connected:
                self._state = "connected"
            else:
                self._state = "configured"
            return self.status()

    def connect(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_idle_locked(allow_ready=True)
            if self._rig is not None:
                self._rig.close()
            self._rig = ReplayHardwareRig(self._hardware_config)
            rig = self._rig
            self._state = "connecting"
            self._last_error = ""
        try:
            rig.connect()
        except Exception as exc:
            try:
                rig.close()
            except Exception:
                pass
            with self._lock:
                if self._rig is rig:
                    self._rig = None
                self._state = "fault"
                self._last_error = str(exc)
                self._last_message = "hardware connect failed"
            raise
        with self._lock:
            self._state = "connected"
            self._last_message = "hardware connected"
            return self.status()

    def validate_episode_arm_control_mode(
        self,
        episode: LoadedReplayEpisode,
    ) -> dict[str, Any]:
        """Validate one episode before selecting or connecting its execution branch."""

        with self._lock:
            selected_mode = self._hardware_config.arm_control_mode
            real_hardware_context = bool(self._hardware_config.connect_hardware)
        summary = getattr(episode, "summary", None)
        episode_metadata = getattr(summary, "metadata", {})
        contract = validate_dataset_arm_control_mode(
            episode_metadata if isinstance(episode_metadata, Mapping) else {},
            selected_mode=selected_mode,
            hardware_output=real_hardware_context,
        )
        return contract.to_dict()

    def preflight(self, episode: LoadedReplayEpisode) -> dict[str, Any]:
        with self._lock:
            rig = self._require_rig_locked()
            sides = self._hardware_config.active_sides
            settings = self._settings
            self._require_position_collision_contract_locked()
        try:
            mode_contract = self.validate_episode_arm_control_mode(episode)
        except DatasetArmControlModeError as exc:
            mode_check = ReplayCheckResult(
                name="dataset_arm_control_mode",
                ok=False,
                level="error",
                message=str(exc),
                details={
                    "selected_mode": str(self._hardware_config.arm_control_mode),
                    "real_hardware_context": bool(
                        self._hardware_config.connect_hardware
                    ),
                },
            )
            group = None
            report = ReplayPreflightReport(checks=(mode_check,))
        else:
            mode_check = ReplayCheckResult(
                name="dataset_arm_control_mode",
                ok=True,
                level="warning" if mode_contract.get("warning") else "info",
                message=(
                    str(mode_contract["warning"])
                    if mode_contract.get("warning")
                    else "dataset arm_control_mode matches selected replay mode"
                ),
                details=dict(mode_contract),
            )
            group = build_episode_replay_group(
                episode,
                active_sides=sides,
                action_column=settings.action_column,
            )
            base_report = run_replay_preflight(
                group=group,
                active_sides=sides,
                current_qpos_by_side=rig.current_qpos_by_side(),
                current_tcp_pose_by_side=rig.current_tcp_pose_by_side(),
                current_tcp_flange_by_side=rig.current_tcp_flange_by_side(),
                tcp_pos_tolerance_m=float(settings.tcp_pos_tolerance_m),
                tcp_rot_tolerance_rad=float(settings.tcp_rot_tolerance_rad),
                max_qpos_init_delta_rad=float(settings.max_qpos_init_delta_rad),
                trajectory_source=settings.trajectory_source,
                qpos_init_source=settings.qpos_init_source,
                end_effectors_enabled=bool(settings.end_effectors_enabled),
                end_effector_backend_sides=rig.end_effector_backend_sides(),
            )
            report = ReplayPreflightReport(
                checks=(mode_check, *base_report.checks)
            )
        if (
            report.ok
            and normalize_rm65_arm_control_mode(self._hardware_config.arm_control_mode)
            == RM65_ARM_CONTROL_MODE_POSITION
            and bool(settings.position_collision_required)
        ):
            try:
                planner, reset_metadata = self._build_position_collision_planner(
                    rig,
                    reference_hz=float(self._hardware_config.position_reference_hz),
                )
                collision_check = ReplayCheckResult(
                    name="position_collision_planner_ready",
                    ok=True,
                    level="info",
                    message="dual-arm collision planner initialized from measured qpos",
                    details={
                        "collision_pair_count": int(
                            getattr(planner, "collision_pair_count", 0)
                        ),
                        "scene_profile": str(settings.position_collision_scene_profile),
                        "reference_hz": float(
                            self._hardware_config.position_reference_hz
                        ),
                        "reset": _serializable(dict(reset_metadata)),
                        "endpoint_collision_checked": True,
                        "segment_collision_certified": False,
                    },
                )
            except Exception as exc:  # noqa: BLE001 - preflight must report setup failures.
                collision_check = ReplayCheckResult(
                    name="position_collision_planner_ready",
                    ok=False,
                    level="error",
                    message=f"dual-arm collision planner initialization failed: {exc}",
                    details={
                        "scene_profile": str(settings.position_collision_scene_profile),
                        "reference_hz": float(
                            self._hardware_config.position_reference_hz
                        ),
                        "endpoint_collision_checked": False,
                        "segment_collision_certified": False,
                    },
                )
            report = ReplayPreflightReport(checks=(*report.checks, collision_check))
        source_binding: ReplayGuiSourceBinding | None = None
        source_binding_error = ""
        if report.ok and group is not None:
            try:
                source_binding = build_replay_gui_source_binding(
                    group,
                    episode_id=episode.summary.episode_id,
                    active_sides=tuple(self._hardware_config.active_sides),
                    include_end_effectors=bool(settings.end_effectors_enabled),
                    trajectory_source=settings.trajectory_source,
                )
            except Exception as exc:  # passive typed evidence cannot alter legacy preflight.
                source_binding_error = f"{type(exc).__name__}: {exc}"
        with self._lock:
            self._group = group
            self._preflight_report = report
            self._source_binding = source_binding
            self._source_binding_error = source_binding_error
            self._frame_index = 0
            self._clear_qpos_init_status_locked()
            self._last_message = "preflight ok" if report.ok else "preflight failed"
            self._last_error = "" if report.ok else "preflight failed"
            self._state = "preflight_ok" if report.ok else "preflight_failed"
            return self.status()

    def move_to_start(self) -> dict[str, Any]:
        with self._lock:
            self._require_output_enabled_locked()
            self._require_position_collision_contract_locked()
            rig = self._require_rig_locked()
            group = self._require_checked_group_locked()
            self._ensure_no_worker_locked()
            target_qpos = self._first_qpos_targets_locked()
            self._stop_event.clear()
            self._state = "moving_to_start"
            self._frame_index = 0
            self._clear_qpos_init_status_locked()
            self._last_error = ""
            self._started_at = time.time()
            self._finished_at = None
            worker = threading.Thread(
                target=self._move_to_start_worker,
                args=(rig, group, target_qpos),
                name="replay-move-to-start",
                daemon=True,
            )
            self._worker = worker
            worker.start()
            return self.status()

    def start(self) -> dict[str, Any]:
        with self._lock:
            self._require_output_enabled_locked()
            self._require_position_collision_contract_locked()
            rig = self._require_rig_locked()
            group = self._require_checked_group_locked()
            if self._state != "start_ready":
                raise RuntimeError("Move to the replay start before starting hardware replay.")
            self._ensure_no_worker_locked()
            self._stop_event.clear()
            self._state = "replaying"
            self._frame_index = 0
            self._last_error = ""
            self._started_at = time.time()
            self._finished_at = None
            worker = threading.Thread(
                target=self._replay_worker,
                args=(rig, group),
                name="replay-trajectory",
                daemon=True,
            )
            self._worker = worker
            worker.start()
            return self.status()

    def abort(self, *, wait: bool = False) -> dict[str, Any]:
        worker: threading.Thread | None = None
        with self._lock:
            self._stop_event.set()
            worker = self._worker
            if self._state in {"moving_to_start", "start_ready", "replaying", "connecting"}:
                self._state = "aborted"
                self._last_message = "hardware replay aborted"
            rig = self._rig
        hold_error = ""
        if rig is not None:
            # Serialize hold with every worker dispatch.  Once stop_event is
            # set, a worker waiting on this barrier rechecks it and cannot send
            # a late command after hold returns.
            with self._dispatch_lock:
                try:
                    rig.hold_motion("hardware replay abort")
                except Exception as exc:  # noqa: BLE001
                    hold_error = str(exc)
                finally:
                    rig.restore_replay_speed()
        if wait and worker is not None and worker.is_alive():
            worker.join(timeout=5.0)
            if worker.is_alive() and not hold_error:
                hold_error = "hardware replay worker did not stop after abort"
        if hold_error:
            with self._lock:
                self._state = "fault"
                self._last_error = hold_error
                self._last_message = "hardware replay abort hold failed"
        return self.status()

    def _dispatch_unless_stopped(self, callback: Callable[[], None]) -> bool:
        """Run one hardware dispatch behind the same barrier used by abort()."""

        with self._dispatch_lock:
            if self._stop_event.is_set():
                return False
            callback()
            return True

    def _require_position_transport_healthy(
        self,
        rig: Any,
        *,
        stage: str,
    ) -> Mapping[str, Any]:
        checker = getattr(rig, "assert_position_stream_healthy", None)
        if callable(checker):
            result = checker(stage=str(stage))
            return dict(result or {})
        if (
            self._settings.send_actions_enabled
            and self._hardware_config.connect_hardware
        ):
            raise RuntimeError(
                f"{stage}: real position replay rig has no arm-stream health gate."
            )
        return {}

    def _capture_position_transport_health(
        self,
        rig: Any,
        *,
        reference_sequence: int,
        stage: str,
        end_effector_index: int | None = None,
    ) -> Mapping[str, Any]:
        snapshot: dict[str, Any] = {
            "stage": str(stage),
            "observed_wall_time": time.time(),
            "observed_monotonic_time": time.monotonic(),
            "ok": False,
        }
        if end_effector_index is not None:
            snapshot["end_effector_index"] = int(end_effector_index)
        try:
            health = self._require_position_transport_healthy(
                rig,
                stage=stage,
            )
        except Exception as exc:
            snapshot.update(
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            with self._lock:
                self._append_position_transport_health_snapshot_locked(
                    reference_sequence=int(reference_sequence),
                    stage=str(stage),
                    snapshot=snapshot,
                )
            raise
        snapshot.update(
            {
                "ok": True,
                "health_by_side": _serializable(dict(health)),
            }
        )
        with self._lock:
            self._append_position_transport_health_snapshot_locked(
                reference_sequence=int(reference_sequence),
                stage=str(stage),
                snapshot=snapshot,
            )
        return health

    def status(self) -> dict[str, Any]:
        with self._lock:
            last_collision = (
                None
                if self._position_collision_last is None
                else dict(self._position_collision_last)
            )
            endpoint_collision_certified = bool(
                self._state in {"replaying", "finished"}
                and last_collision is not None
                and last_collision.get("endpoint_collision_checked", False)
                and last_collision.get("endpoint_collision_passed", False)
                and not last_collision.get("hold", True)
                and last_collision.get("valid", False)
            )
            group_payload: dict[str, Any] | None = None
            if self._group is not None:
                group_payload = {
                    "key": self._group.key,
                    "label": self._group.label,
                    "source_path": self._group.source_path,
                    "source_episode_index": int(self._group.source_episode_index),
                    "start_index": int(self._group.start_index),
                    "end_index": int(self._group.end_index),
                    "frame_count": int(self._group.frame_count),
                    "action_hz": float(self._group.action_hz),
                    "arm_joint_target_sides": sorted(self._group.arm_joint_targets),
                    "arm_observation_current_qpos_sides": sorted(self._group.arm_observation_current_qpos),
                    "arm_observation_actual_qpos_sides": sorted(self._group.arm_observation_actual_qpos),
                    "trajectory_source": normalize_replay_trajectory_source(self._settings.trajectory_source),
                    "qpos_init_source": normalize_qpos_init_source(self._settings.qpos_init_source),
                    "end_effectors_enabled": bool(self._settings.end_effectors_enabled),
                    "position_pipeline": (
                        {
                            "keyframe_hz": float(self._group.action_hz),
                            "reference_hz": float(self._hardware_config.position_reference_hz),
                            "command_hz": float(
                                self._hardware_config.arm_control_hz
                                if (
                                    self._hardware_config.background_streaming_enabled
                                    and self._hardware_config.arm_canfd_follow
                                )
                                else self._hardware_config.position_reference_hz
                            ),
                            "keyframe_safety": True,
                            "temporal_smoothing": False,
                            "collision_required": bool(
                                self._settings.position_collision_required
                            ),
                            "collision_scene_profile": str(
                                self._settings.position_collision_scene_profile
                            ),
                            "endpoint_collision_certified": (
                                endpoint_collision_certified
                            ),
                            "segment_collision_certified": False,
                            "collision_certification_status": (
                                "legacy_collision_disabled"
                                if not self._settings.position_collision_required
                                else (
                                    "reference_endpoint_only_segment_not_certified"
                                    if endpoint_collision_certified
                                    else (
                                        "not_certified_replay_fault"
                                        if self._state == "fault"
                                        else "pending_successful_endpoint_check"
                                    )
                                )
                            ),
                            "source_timestamp_mode": str(
                                dict(self._group.metadata.get("frame_timing", {})).get(
                                    "source_timestamp_mode",
                                    "provided"
                                    if self._group.frame_times_sec
                                    else "nominal_fallback",
                                )
                            ),
                            "pose_interpolation": (
                                "xyz_linear+quaternion_shortest_path_slerp"
                                if normalize_replay_trajectory_source(
                                    self._settings.trajectory_source
                                )
                                != REPLAY_TRAJECTORY_QPOS_TARGET
                                else None
                            ),
                            "qpos_interpolation": (
                                "linear"
                                if normalize_replay_trajectory_source(
                                    self._settings.trajectory_source
                                )
                                == REPLAY_TRAJECTORY_QPOS_TARGET
                                else None
                            ),
                        }
                        if normalize_rm65_arm_control_mode(self._hardware_config.arm_control_mode)
                        == RM65_ARM_CONTROL_MODE_POSITION
                        else None
                    ),
                }
            return {
                "state": self._state,
                "active_sides": list(self._hardware_config.active_sides),
                "connected": bool(self._rig is not None and self._rig.connected),
                "settings": self._settings.to_dict(),
                "hardware_config": self._hardware_config.to_dict(),
                "frame_index": int(self._frame_index),
                "group": group_payload,
                "preflight": None
                if self._preflight_report is None
                else self._preflight_report.to_dict(),
                "replay_source_binding": (
                    {
                        "available": False,
                        "authoritative": False,
                        "cursor_owned": False,
                        "error": self._source_binding_error,
                    }
                    if self._source_binding is None
                    else self._source_binding.to_status()
                ),
                "qpos_init": {
                    "plan_summary": self._qpos_init_plan_summary,
                    "step_index": int(self._qpos_init_step_index),
                    "step_count": int(self._qpos_init_step_count),
                },
                "position_collision_diagnostics": {
                    "debug_enabled": bool(
                        self._settings.position_collision_debug
                    ),
                    "event_capacity": int(_POSITION_COLLISION_EVENT_CAPACITY),
                    "event_count": int(len(self._position_collision_events)),
                    "event_drop_count": int(
                        self._position_collision_event_drop_count
                    ),
                    "plan_count": int(self._position_collision_plan_count),
                    "tracking_count": int(
                        self._position_collision_tracking_count
                    ),
                    "limited_count": int(self._position_collision_limited_count),
                    "backoff_count": int(
                        self._position_collision_backoff_count
                    ),
                    "hold_count": int(self._position_collision_hold_count),
                    "endpoint_collision_certified": (
                        endpoint_collision_certified
                    ),
                    "segment_collision_certified": False,
                    "last_reset": self._position_collision_reset,
                    "last": last_collision,
                },
                "action_observer": {
                    "enabled": self._action_observer_queue is not None,
                    "offer_count": int(self._action_observer_offer_count),
                    "drop_count": int(self._action_observer_drop_count),
                    "error_count": int(self._action_observer_error_count),
                    "last_error": self._action_observer_last_error,
                    "authoritative": False,
                },
                "last_error": self._last_error,
                "last_message": self._last_message,
                "started_at": self._started_at,
                "finished_at": self._finished_at,
                "worker_alive": bool(self._worker is not None and self._worker.is_alive()),
            }

    def position_collision_debug_snapshot(
        self,
        *,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Return a bounded copy of recorded collision events for debugging."""

        normalized_limit = int(limit)
        if isinstance(limit, bool) or normalized_limit <= 0:
            raise ValueError("position collision debug limit must be positive.")
        normalized_limit = min(
            normalized_limit,
            _POSITION_COLLISION_EVENT_CAPACITY,
        )
        with self._lock:
            events = list(self._position_collision_events)[-normalized_limit:]
            return {
                "debug_enabled": bool(self._settings.position_collision_debug),
                "capacity": int(_POSITION_COLLISION_EVENT_CAPACITY),
                "event_count": int(len(self._position_collision_events)),
                "event_drop_count": int(
                    self._position_collision_event_drop_count
                ),
                "events": _serializable(events),
            }

    def _reset_position_collision_diagnostics_locked(self) -> None:
        self._position_collision_events.clear()
        self._position_collision_event_drop_count = 0
        self._position_collision_plan_count = 0
        self._position_collision_tracking_count = 0
        self._position_collision_limited_count = 0
        self._position_collision_backoff_count = 0
        self._position_collision_hold_count = 0
        self._position_collision_last = None
        self._position_collision_reset = None

    def _record_position_collision_locked(
        self,
        *,
        reference_sequence: int,
        reference_timestamp: float,
        collision: Mapping[str, Any],
        record_transport_health: bool = False,
    ) -> None:
        summary = _serializable(dict(collision))
        transport_slots = (
            _empty_transport_health_slots() if record_transport_health else None
        )
        event = {
            "observed_wall_time": time.time(),
            "observed_monotonic_time": time.monotonic(),
            "reference_sequence": int(reference_sequence),
            "reference_timestamp": float(reference_timestamp),
            "collision": summary,
        }
        if transport_slots is not None:
            event["transport_health_snapshots"] = transport_slots
        if len(self._position_collision_events) >= _POSITION_COLLISION_EVENT_CAPACITY:
            self._position_collision_event_drop_count += 1
        self._position_collision_events.append(event)
        self._position_collision_plan_count += 1
        status = str(summary.get("status") or "unknown")
        if bool(summary.get("hold", False)):
            self._position_collision_hold_count += 1
        elif status == "collision_backoff":
            self._position_collision_backoff_count += 1
        elif status == "limited":
            self._position_collision_limited_count += 1
        else:
            self._position_collision_tracking_count += 1
        self._position_collision_last = dict(summary)
        if transport_slots is not None:
            self._position_collision_last["transport_health_snapshots"] = _serializable(
                transport_slots
            )

    def _append_position_transport_health_snapshot_locked(
        self,
        *,
        reference_sequence: int,
        stage: str,
        snapshot: Mapping[str, Any],
    ) -> None:
        if stage not in _POSITION_TRANSPORT_HEALTH_STAGES:
            raise ValueError(f"unknown position transport-health stage: {stage!r}")
        matching_event: dict[str, Any] | None = None
        for event in reversed(self._position_collision_events):
            if int(event.get("reference_sequence", -1)) == int(reference_sequence):
                matching_event = event
                break
        if matching_event is None:
            return
        slots = matching_event.get("transport_health_snapshots")
        if not isinstance(slots, dict):
            return
        slot = slots.get(stage)
        if not isinstance(slot, dict):
            return
        observations = slot.get("observations")
        if not isinstance(observations, list):
            observations = []
            slot["observations"] = observations
        normalized = _serializable(dict(snapshot))
        observations.append(normalized)
        slot["status"] = "checked" if bool(normalized.get("ok")) else "failed"
        if (
            self._position_collision_events
            and matching_event is self._position_collision_events[-1]
            and self._position_collision_last is not None
        ):
            self._position_collision_last["transport_health_snapshots"] = _serializable(
                slots
            )

    def _move_to_start_worker(
        self,
        rig: ReplayHardwareRig,
        group: ReplayTrajectoryGroup,
        target_qpos: dict[str, np.ndarray],
    ) -> None:
        del group
        deadline = time.monotonic() + float(self._settings.move_to_start_timeout_s)
        period = float(self._settings.qpos_init_dt_sec)
        try:
            set_command_dt = getattr(rig, "set_arm_command_dt_sec", None)
            if callable(set_command_dt):
                set_command_dt(period)
            rig.begin_replay_speed()
            plan = self._build_qpos_init_plan(rig, target_qpos)
            steps = tuple(getattr(plan, "steps", ()))
            summary = plan.summary() if hasattr(plan, "summary") else {"type": type(plan).__name__}
            with self._lock:
                self._qpos_init_plan_summary = _serializable(summary)
                self._qpos_init_step_count = len(steps)
                self._qpos_init_step_index = 0
                self._last_message = f"qpos init plan ready steps={len(steps)}"
            next_time = time.monotonic()
            for step in steps:
                if self._stop_event.is_set():
                    with self._lock:
                        self._state = "aborted"
                        self._finished_at = time.time()
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Timed out while moving to replay start qpos; "
                        "verify robot state and replay_arm_max_angvel."
                    )
                if not self._dispatch_unless_stopped(
                    lambda step=step: rig.send_joint_targets(
                        step.qpos_by_side,
                        source="replay_qpos_init",
                    )
                ):
                    with self._lock:
                        self._state = "aborted"
                        self._finished_at = time.time()
                    return
                with self._lock:
                    self._qpos_init_step_index = int(getattr(step, "index", self._qpos_init_step_index + 1))
                next_time += max(0.001, period)
                sleep_s = next_time - time.monotonic()
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
            while not self._stop_event.is_set():
                if rig.qpos_reached(
                    target_qpos,
                    tolerance=float(self._settings.qpos_reached_tolerance_rad),
                ):
                    with self._lock:
                        self._state = "start_ready"
                        self._finished_at = time.time()
                        self._last_message = "replay start qpos reached"
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Timed out while waiting for replay start qpos after qpos init plan."
                    )
                if not self._dispatch_unless_stopped(
                    lambda: rig.send_joint_targets(
                        target_qpos,
                        source="replay_qpos_init_final_hold",
                    )
                ):
                    break
                time.sleep(max(0.001, min(period, 0.05)))
            with self._lock:
                self._state = "aborted"
                self._finished_at = time.time()
        except Exception as exc:  # noqa: BLE001
            try:
                rig.hold_motion("hardware replay move_to_start fault")
            except Exception as hold_exc:  # noqa: BLE001
                exc = RuntimeError(f"{exc}; hold failed: {hold_exc}")
            _print_worker_fault("move_to_start", exc)
            with self._lock:
                self._state = "fault"
                self._last_error = str(exc)
                self._finished_at = time.time()
        finally:
            rig.restore_replay_speed()

    def _replay_worker(self, rig: ReplayHardwareRig, group: ReplayTrajectoryGroup) -> None:
        if (
            normalize_rm65_arm_control_mode(self._hardware_config.arm_control_mode)
            == RM65_ARM_CONTROL_MODE_POSITION
        ):
            self._position_replay_worker(rig, group)
            return
        self._impedance_replay_worker(rig, group)

    def _build_position_collision_planner(
        self,
        rig: Any,
        *,
        reference_hz: float,
    ) -> tuple[Any, Mapping[str, Any]]:
        active_sides = tuple(self._hardware_config.active_sides)
        if set(active_sides) != {"left", "right"} or len(active_sides) != 2:
            raise RuntimeError(
                "collision-required position pose replay requires active_sides=both; "
                f"got active_sides={active_sides}."
            )
        qpos_getter = getattr(rig, "current_qpos_by_side", None)
        tcp_getter = getattr(rig, "current_tcp_flange_by_side", None)
        if not callable(qpos_getter) or not callable(tcp_getter):
            raise RuntimeError(
                "collision-required position replay needs measured dual-arm qpos and "
                "the active endpoint-to-tool TCP transforms from the replay rig."
            )
        require_fresh_hardware_state = bool(
            self._settings.send_actions_enabled
            and self._hardware_config.connect_hardware
        )
        if require_fresh_hardware_state:
            fresh_getter = getattr(rig, "fresh_hardware_qpos_by_side", None)
            if not callable(fresh_getter):
                raise RuntimeError(
                    "real collision-required replay needs a fresh measured-qpos API; "
                    "cached qpos is not accepted for planner reset."
                )
            current_qpos = dict(
                fresh_getter(
                    max_age_sec=DEFAULT_POSITION_COLLISION_STATE_MAX_AGE_SEC,
                    pair_max_skew_sec=(
                        DEFAULT_POSITION_COLLISION_STATE_PAIR_MAX_SKEW_SEC
                    ),
                )
                or {}
            )
            qpos_state_source = "fresh_hardware_measurement"
        else:
            current_qpos = dict(qpos_getter() or {})
            qpos_state_source = "non_motion_or_local_authoritative_state"
        tcp_pose_flange = dict(tcp_getter() or {})
        normalized_qpos: dict[str, np.ndarray] = {}
        normalized_tcp: dict[str, np.ndarray] = {}
        invalid_qpos: list[str] = []
        invalid_tcp: list[str] = []
        for side in ("left", "right"):
            try:
                qpos = np.asarray(current_qpos.get(side), dtype=float).reshape(-1)
            except (TypeError, ValueError):
                qpos = np.empty(0, dtype=float)
            if qpos.shape != (ARM_JOINT_DOF,) or not np.all(np.isfinite(qpos)):
                invalid_qpos.append(side)
            else:
                normalized_qpos[side] = qpos.copy()
            try:
                tcp = np.asarray(tcp_pose_flange.get(side), dtype=float).reshape(-1)
            except (TypeError, ValueError):
                tcp = np.empty(0, dtype=float)
            if tcp.shape != (ARM_JOINT_DOF,) or not np.all(np.isfinite(tcp)):
                invalid_tcp.append(side)
            else:
                normalized_tcp[side] = tcp.copy()
        if invalid_qpos or invalid_tcp:
            raise RuntimeError(
                "collision-required position replay requires complete dual-arm state; "
                f"invalid_qpos={invalid_qpos}, invalid_tcp={invalid_tcp}."
            )
        endpoint_to_tool_by_side = {
            side: pose6_to_matrix(
                normalized_tcp[side]
            )
            for side in ("left", "right")
        }
        planner_config = RM65PositionCollisionPlannerConfig(
            reference_hz=float(reference_hz),
            scene_profile=str(self._settings.position_collision_scene_profile),
            collision_min_distance_m=float(
                self._settings.position_collision_min_distance_m
            ),
            collision_distance_epsilon_m=float(
                self._settings.position_collision_distance_epsilon_m
            ),
            max_joint_velocity_rad_s=tuple(
                float(value)
                for value in self._settings.position_collision_max_joint_velocity_rad_s
            ),
            max_joint_step_rad=(
                None
                if self._settings.position_collision_max_joint_step_rad is None
                else tuple(
                    float(value)
                    for value in self._settings.position_collision_max_joint_step_rad
                )
            ),
            damping=REPLAY_POSITION_IK_DAMPING,
            solve_max_iterations=REPLAY_POSITION_IK_SOLVE_MAX_ITERATIONS,
            solve_min_iterations=REPLAY_POSITION_IK_SOLVE_MIN_ITERATIONS,
            solve_adaptive_iterations_enabled=True,
            solve_iteration_reserve_ms=REPLAY_POSITION_IK_ITERATION_RESERVE_MS,
            pose_lift_profile=self._settings.pose_lift_profile_name,
            debug_profile=bool(self._settings.position_collision_debug),
            ik_tuning=_replay_position_ik_tuning(
                self._settings.pose_lift_profile_name
            ),
        )
        factory = (
            self._position_collision_planner_factory
            if self._position_collision_planner_factory is not None
            else RM65PositionCollisionPlanner
        )
        planner = factory(
            config=planner_config,
            endpoint_to_tool_by_side=endpoint_to_tool_by_side,
        )
        collision_pair_count = int(getattr(planner, "collision_pair_count", 0))
        if collision_pair_count <= 0:
            raise RuntimeError(
                "collision-required position replay refuses a scene with no collision pairs."
            )
        reset_metadata = dict(planner.reset(normalized_qpos) or {})
        reset_metadata.update(
            {
                "replay_position_ik_profile": {
                    "profile": self._settings.position_control_profile_name,
                    "pose_lift_profile": self._settings.pose_lift_profile_name,
                    "main_action_contract": "accepted_qref_fk_pose",
                    "source_shaping_reapplied": False,
                    "solve_min_iterations": REPLAY_POSITION_IK_SOLVE_MIN_ITERATIONS,
                    "solve_max_iterations": REPLAY_POSITION_IK_SOLVE_MAX_ITERATIONS,
                    "solve_iteration_reserve_ms": (
                        REPLAY_POSITION_IK_ITERATION_RESERVE_MS
                    ),
                    "orientation_weight": REPLAY_POSITION_IK_ORIENTATION_WEIGHT,
                    "damping": REPLAY_POSITION_IK_DAMPING,
                    "joint_velocity_limit_profile": (
                        (
                            "rm65_position_2x"
                            if self._settings.position_control_profile_name
                            == LEGACY_POSITION_CONTROL_PROFILE_NAME
                            else REPLAY_POSITION_JOINT_VELOCITY_LIMIT_PROFILE
                        )
                        if np.allclose(
                            np.asarray(
                                self._settings.position_collision_max_joint_velocity_rad_s,
                                dtype=float,
                            ),
                            np.asarray(
                                REPLAY_POSITION_COLLISION_MAX_JOINT_VELOCITY_RAD_S,
                                dtype=float,
                            ),
                            rtol=0.0,
                            atol=1e-12,
                        )
                        else "explicit_velocity_override"
                    ),
                },
                "qpos_state_source": qpos_state_source,
                "fresh_hardware_state_required": require_fresh_hardware_state,
                "state_max_age_sec": (
                    DEFAULT_POSITION_COLLISION_STATE_MAX_AGE_SEC
                    if require_fresh_hardware_state
                    else None
                ),
                "state_pair_max_skew_sec": (
                    DEFAULT_POSITION_COLLISION_STATE_PAIR_MAX_SKEW_SEC
                    if require_fresh_hardware_state
                    else None
                ),
            }
        )
        return planner, reset_metadata

    def _impedance_replay_worker(self, rig: ReplayHardwareRig, group: ReplayTrajectoryGroup) -> None:
        """Preserve the established action-rate -> 40 Hz impedance path."""

        hardware_group = replay_group_in_hardware_base(
            group, self._hardware_config.active_sides
        )
        period = 1.0 / max(1e-6, float(group.action_hz))
        next_time = time.monotonic()
        trajectory_source = normalize_replay_trajectory_source(self._settings.trajectory_source)
        current_index: int | None = None
        try:
            set_command_dt = getattr(rig, "set_arm_command_dt_sec", None)
            if callable(set_command_dt):
                set_command_dt(period)
            rig.begin_replay_speed()
            for index in range(int(group.frame_count)):
                current_index = int(index)
                if self._stop_event.is_set():
                    with self._lock:
                        self._state = "aborted"
                        self._finished_at = time.time()
                    return
                if trajectory_source == REPLAY_TRAJECTORY_QPOS_TARGET:
                    qpos_targets = self._qpos_targets_at_index(group, index)
                    dispatched = self._dispatch_unless_stopped(
                        lambda: rig.send_joint_targets(
                            qpos_targets,
                            source="replay_qpos_trajectory",
                        )
                    )
                else:
                    self._offer_replay_source_frame(
                        group,
                        index=index,
                        observed_at=time.monotonic(),
                        schedule_start_at=next_time,
                        target_effective_at=next_time,
                        effective_until_at=next_time + period,
                    )
                    replay_command = replay_teleop_command_at_index(
                        hardware_group,
                        self._hardware_config.active_sides,
                        index=index,
                        include_arms=True,
                        include_end_effectors=bool(self._settings.end_effectors_enabled),
                        source="replay_action_pose",
                        resolver=rig.action_resolver,
                    )
                    dispatched = self._dispatch_unless_stopped(
                        lambda: rig.send_commands(
                            replay_command,
                        )
                    )
                if not dispatched:
                    with self._lock:
                        self._state = "aborted"
                        self._finished_at = time.time()
                    return
                with self._lock:
                    self._frame_index = int(index)
                next_time += period
                sleep_s = next_time - time.monotonic()
                if sleep_s > 0.0 and self._stop_event.wait(sleep_s):
                    with self._lock:
                        self._state = "aborted"
                        self._finished_at = time.time()
                    return
            with self._lock:
                self._state = "finished"
                self._frame_index = max(0, int(group.frame_count) - 1)
                self._finished_at = time.time()
                self._last_message = "hardware replay finished"
        except Exception as exc:  # noqa: BLE001
            try:
                rig.hold_motion("hardware replay fault")
            except Exception as hold_exc:  # noqa: BLE001
                exc = RuntimeError(f"{exc}; hold failed: {hold_exc}")
            if current_index is not None:
                exc = RuntimeError(
                    f"replay frame_index={current_index} "
                    f"trajectory_source={trajectory_source} "
                    f"source={group.source_path}: {exc}"
                )
            _print_worker_fault("replay", exc)
            with self._lock:
                self._state = "fault"
                self._last_error = str(exc)
                self._finished_at = time.time()
        finally:
            rig.restore_replay_speed()

    def _position_replay_worker(self, rig: ReplayHardwareRig, group: ReplayTrajectoryGroup) -> None:
        """Execute the explicit safe-keyframe -> reference -> command path.

        The first stage applies only explicit keyframe limits.  It does not run
        a temporal filter.  Offline timestamp resampling then linearly
        interpolates translation/qpos and uses shortest-path quaternion SLERP.
        When collision checking is required (the production default), each
        high-rate tool pose is converted by one coupled dual-arm collision IK
        planner into an explicitly limited qpos endpoint.  Only that qpos is
        passed to the sim/real-neutral reference->command stage; per-arm backend
        Cartesian IK is not used.  Segment collision certification between
        adjacent high-rate endpoints remains explicitly out of scope.
        """

        with self._lock:
            self._reset_position_collision_diagnostics_locked()
        trajectory_source = normalize_replay_trajectory_source(self._settings.trajectory_source)
        reference_hz = float(self._hardware_config.position_reference_hz)
        reference_period = 1.0 / max(1e-6, reference_hz)
        keyframe_hz = float(group.action_hz)
        background_stream_active = bool(
            self._hardware_config.background_streaming_enabled
            and self._hardware_config.arm_canfd_follow
        )
        effective_command_hz = (
            float(self._hardware_config.arm_control_hz)
            if background_stream_active
            else reference_hz
        )
        execution_chain = (
            f"safe_{_rate_label(keyframe_hz)}hz"
            f"_to_reference_{_rate_label(reference_hz)}hz"
            f"_to_command_{_rate_label(effective_command_hz)}hz"
        )
        safety_config = PositionKeyframeSafetyConfig(
            workspace_min=tuple(float(value) for value in self._settings.position_workspace_min),
            workspace_max=tuple(float(value) for value in self._settings.position_workspace_max),
            max_translation_step_m=float(self._settings.position_max_xyz_step_m),
            max_rotation_step_deg=float(self._settings.position_max_rot_step_deg),
            max_joint_step_rad=float(self._settings.position_max_joint_step_rad),
            joint_lower_rad=tuple(float(value) for value in RM65_ABSOLUTE_Q_MIN_RAD),
            joint_upper_rad=tuple(float(value) for value in RM65_ABSOLUTE_Q_MAX_RAD),
        )
        collision_required = bool(self._settings.position_collision_required)
        collision_planner: Any | None = None
        collision_reset_metadata: dict[str, Any] = {}
        current_index: int | None = None
        try:
            if reference_hz + 1e-9 < keyframe_hz:
                raise ValueError(
                    "position replay reference_hz must be >= source keyframe rate; "
                    f"got reference_hz={reference_hz:g}, keyframe_hz={keyframe_hz:g}"
                )
            if collision_required:
                collision_planner, reset_metadata = self._build_position_collision_planner(
                    rig,
                    reference_hz=reference_hz,
                )
                collision_reset_metadata = dict(reset_metadata)
                with self._lock:
                    self._position_collision_reset = _serializable(
                        collision_reset_metadata
                    )
            set_command_dt = getattr(rig, "set_arm_command_dt_sec", None)
            if callable(set_command_dt):
                set_command_dt(reference_period)
            rig.begin_replay_speed()
            if trajectory_source == REPLAY_TRAJECTORY_QPOS_TARGET:
                samples = _safe_qpos_keyframes(
                    rig,
                    group,
                    active_sides=self._hardware_config.active_sides,
                    config=safety_config,
                )
            else:
                samples = _safe_pose_keyframes(
                    rig,
                    group,
                    active_sides=self._hardware_config.active_sides,
                    config=safety_config,
                )
            references = resample_motion_targets(
                samples,
                reference_hz=reference_hz,
                include_final=True,
            )
            first_timestamp = float(references[0].timestamp)
            started_monotonic = time.monotonic()
            next_end_effector_index = 0
            last_observed_source_index = -1
            for reference in references:
                deadline = started_monotonic + max(
                    0.0,
                    float(reference.timestamp) - first_timestamp,
                )
                wait_sec = deadline - time.monotonic()
                if wait_sec > 0.0 and self._stop_event.wait(wait_sec):
                    with self._lock:
                        self._state = "aborted"
                        self._finished_at = time.time()
                    return
                if self._stop_event.is_set():
                    with self._lock:
                        self._state = "aborted"
                        self._finished_at = time.time()
                    return

                if trajectory_source != REPLAY_TRAJECTORY_QPOS_TARGET:
                    # Emit each source keyframe once when it becomes causal;
                    # right_source_index may be future lookahead used only by
                    # offline interpolation and must not advance the observer.
                    due_source_index = int(reference.left_source_index)
                    while last_observed_source_index < due_source_index:
                        last_observed_source_index += 1
                        self._offer_replay_source_frame(
                            group,
                            index=last_observed_source_index,
                            observed_at=time.monotonic(),
                            schedule_start_at=started_monotonic - first_timestamp,
                        )

                current_index = int(reference.right_source_index)
                parent = samples[current_index]
                reference_metadata = {
                    "position_pipeline": execution_chain,
                    "position_default_profile_contract": (
                        f"safe_{_rate_label(DEFAULT_POSITION_SAMPLE_HZ)}hz"
                        f"_to_reference_{_rate_label(DEFAULT_POSITION_REFERENCE_HZ)}hz"
                        f"_to_command_{_rate_label(DEFAULT_POSITION_COMMAND_HZ)}hz"
                    ),
                    "position_keyframe_hz": keyframe_hz,
                    "position_source_timestamp_mode": str(
                        dict(group.metadata.get("frame_timing", {})).get(
                            "source_timestamp_mode",
                            "nominal_fallback" if not group.frame_times_sec else "provided",
                        )
                    ),
                    "position_source_frame_time_sec": float(parent.timestamp),
                    "position_reference_hz": reference_hz,
                    "position_command_hz": effective_command_hz,
                    "position_background_streaming_active": background_stream_active,
                    "position_reference_sequence": int(reference.sequence),
                    "left_source_index": int(reference.left_source_index),
                    "right_source_index": int(reference.right_source_index),
                    "left_source_timestamp": float(reference.left_source_timestamp),
                    "right_source_timestamp": float(reference.right_source_timestamp),
                    "interpolation_alpha": float(reference.interpolation_alpha),
                    "resampling": dict(reference.metadata),
                    "position_keyframe_safety": dict(
                        parent.metadata.get("position_keyframe_safety", {})
                    ),
                    "position_collision_required": collision_required,
                    "position_collision_reset": _serializable(collision_reset_metadata),
                    "position_endpoint_collision_checked": False,
                    "position_endpoint_collision_passed": False,
                    "position_segment_collision_checked": False,
                    "position_segment_collision_certified": False,
                    "position_collision_certification_status": (
                        "reference_endpoint_only_segment_not_certified"
                        if collision_required
                        else "legacy_collision_disabled"
                    ),
                    "temporal_smoothing_applied": False,
                }
                if (
                    collision_required
                    and trajectory_source == REPLAY_TRAJECTORY_QPOS_TARGET
                ):
                    assert collision_planner is not None
                    admission = collision_planner.admit_qpos_reference(
                        reference.qpos_by_side,
                        source=(
                            f"replay_qpos_reference_{_rate_label(reference_hz)}hz"
                        ),
                    )
                    admission_metadata = dict(
                        getattr(admission, "metadata", {}) or {}
                    )
                    endpoint_checked = bool(
                        admission_metadata.get("endpoint_collision_checked", False)
                    )
                    endpoint_passed = bool(
                        admission_metadata.get("endpoint_collision_passed", False)
                    )
                    reference_metadata.update(
                        {
                            "position_collision": admission_metadata,
                            "position_endpoint_collision_checked": endpoint_checked,
                            "position_endpoint_collision_passed": endpoint_passed,
                            "position_collision_certification_status": (
                                "qpos_endpoint_only_segment_not_certified"
                            ),
                            "replay_qpos_strict": {
                                "enabled": True,
                                "ik_applied": False,
                                "target_modified": False,
                            },
                        }
                    )
                    if (
                        not bool(getattr(admission, "safe", False))
                        or not endpoint_checked
                        or not endpoint_passed
                    ):
                        raise RuntimeError(
                            "strict QPOS replay failed coupled endpoint admission: "
                            f"reason={getattr(admission, 'reason', None)}, "
                            f"endpoint_checked={endpoint_checked}, "
                            f"endpoint_passed={endpoint_passed}."
                        )
                    qpos_reference = {
                        str(side): np.asarray(
                            admission.qpos_by_side[str(side)], dtype=float
                        ).reshape(ARM_JOINT_DOF)
                        for side in self._hardware_config.active_sides
                    }
                    dispatched = self._dispatch_unless_stopped(
                        lambda: _send_joint_reference(
                            rig,
                            qpos_reference,
                            source=(
                                f"replay_qpos_reference_{_rate_label(reference_hz)}hz"
                            ),
                            metadata=reference_metadata,
                        )
                    )
                elif collision_required:
                    assert collision_planner is not None
                    plan_result = collision_planner.plan_pose_reference(
                        reference.pose_by_side,
                        target_reference_frame="arm_base",
                        target_controlled_frame="tool_tcp",
                        source_timestamp=float(reference.timestamp),
                        source=f"replay_action_pose_reference_{_rate_label(reference_hz)}hz",
                        metadata={
                            "position_reference_sequence": int(reference.sequence),
                            "left_source_index": int(reference.left_source_index),
                            "right_source_index": int(reference.right_source_index),
                            "interpolation_alpha": float(reference.interpolation_alpha),
                            "pose_action_interval_velocity": dict(
                                reference.metadata.get(
                                    "pose_action_interval_velocity",
                                    {},
                                )
                                or {}
                            ),
                            "temporal_smoothing_applied": False,
                        },
                    )
                    collision_metadata = _position_collision_result_metadata(
                        plan_result,
                        debug=bool(self._settings.position_collision_debug),
                    )
                    reference_metadata.update(
                        {
                            "position_collision": collision_metadata,
                            "position_endpoint_collision_checked": bool(
                                collision_metadata["endpoint_collision_checked"]
                            ),
                            "position_endpoint_collision_passed": bool(
                                collision_metadata["endpoint_collision_passed"]
                            ),
                        }
                    )
                    with self._lock:
                        self._record_position_collision_locked(
                            reference_sequence=int(reference.sequence),
                            reference_timestamp=float(reference.timestamp),
                            collision=collision_metadata,
                            record_transport_health=bool(
                                self._settings.position_collision_debug
                            ),
                        )
                    if (
                        not bool(getattr(plan_result, "valid", False))
                        or bool(getattr(plan_result, "hold", True))
                        or not bool(collision_metadata["endpoint_collision_checked"])
                        or not bool(collision_metadata["endpoint_collision_passed"])
                    ):
                        failure = dict(getattr(plan_result, "metadata", {}) or {}).get(
                            "failure"
                        )
                        raise RuntimeError(
                            "collision planner failed closed before arm/hand dispatch: "
                            f"status={getattr(plan_result, 'status', 'unknown')}, "
                            f"valid={getattr(plan_result, 'valid', False)}, "
                            f"hold={getattr(plan_result, 'hold', True)}, failure={failure}."
                        )
                    qpos_reference = {
                        str(side): np.asarray(
                            plan_result.q_ref_by_side[str(side)], dtype=float
                        ).reshape(ARM_JOINT_DOF)
                        for side in self._hardware_config.active_sides
                    }
                    replay_source = (
                        f"replay_collision_qpos_reference_{_rate_label(reference_hz)}hz"
                    )
                    def dispatch_collision_joint_reference() -> None:
                        self._capture_position_transport_health(
                            rig,
                            reference_sequence=int(reference.sequence),
                            stage="before_arm_reference",
                        )
                        _send_joint_reference(
                            rig,
                            qpos_reference,
                            source=replay_source,
                            metadata=reference_metadata,
                        )
                        self._capture_position_transport_health(
                            rig,
                            reference_sequence=int(reference.sequence),
                            stage="after_arm_reference_before_hand",
                        )

                    dispatched = self._dispatch_unless_stopped(
                        dispatch_collision_joint_reference
                    )
                elif trajectory_source == REPLAY_TRAJECTORY_QPOS_TARGET:
                    dispatched = self._dispatch_unless_stopped(
                        lambda: _send_joint_reference(
                            rig,
                            reference.qpos_by_side,
                            source=f"replay_qpos_reference_{_rate_label(reference_hz)}hz",
                            metadata=reference_metadata,
                        )
                    )
                else:
                    xyzquat_by_side: dict[str, np.ndarray] = {}
                    for side, pose in reference.pose_by_side.items():
                        xyzquat = pose_matrix_to_xyzquat(pose)
                        if xyzquat is None:
                            raise RuntimeError(
                                f"{side} interpolated replay reference is not a valid pose matrix."
                            )
                        xyzquat_by_side[str(side)] = np.asarray(xyzquat, dtype=np.float32).reshape(7)
                    dispatched = self._dispatch_unless_stopped(
                        lambda: _send_pose_reference(
                            rig,
                            xyzquat_by_side,
                            source=f"replay_action_pose_reference_{_rate_label(reference_hz)}hz",
                            metadata=reference_metadata,
                        )
                    )
                if not dispatched:
                    with self._lock:
                        self._state = "aborted"
                        self._finished_at = time.time()
                    return

                # Hands remain discrete source-rate commands.  Arm planning
                # and dispatch happen first so an invalid/hold collision result
                # cannot leak a hand command from the same reference tick.
                while (
                    bool(self._settings.end_effectors_enabled)
                    and next_end_effector_index < int(group.frame_count)
                    and (
                        group.relative_time_sec(next_end_effector_index)
                        <= float(reference.timestamp) + 1e-9
                    )
                ):
                    end_effector_command = replay_teleop_command_at_index(
                        group,
                        self._hardware_config.active_sides,
                        index=int(next_end_effector_index),
                        include_arms=False,
                        include_end_effectors=True,
                        source=f"replay_action_end_effector_{_rate_label(keyframe_hz)}hz",
                        resolver=rig.action_resolver,
                    )
                    def dispatch_end_effector_after_arm_health() -> None:
                        if collision_required:
                            self._capture_position_transport_health(
                                rig,
                                reference_sequence=int(reference.sequence),
                                stage="immediately_before_end_effector_dispatch",
                                end_effector_index=int(next_end_effector_index),
                            )
                        rig.send_commands(end_effector_command)

                    if not self._dispatch_unless_stopped(
                        dispatch_end_effector_after_arm_health
                    ):
                        with self._lock:
                            self._state = "aborted"
                            self._finished_at = time.time()
                        return
                    next_end_effector_index += 1
                with self._lock:
                    self._frame_index = int(current_index)

            with self._lock:
                self._state = "finished"
                self._frame_index = max(0, int(group.frame_count) - 1)
                self._finished_at = time.time()
                self._last_message = "hardware replay finished"
        except Exception as exc:  # noqa: BLE001
            try:
                rig.hold_motion("hardware replay fault")
            except Exception as hold_exc:  # noqa: BLE001
                exc = RuntimeError(f"{exc}; hold failed: {hold_exc}")
            if current_index is not None:
                exc = RuntimeError(
                    f"replay frame_index={current_index} "
                    f"trajectory_source={trajectory_source} "
                    f"source={group.source_path}: {exc}"
                )
            _print_worker_fault("replay", exc)
            with self._lock:
                self._state = "fault"
                self._last_error = str(exc)
                self._finished_at = time.time()
        finally:
            rig.restore_replay_speed()

    def _ensure_idle_locked(self, *, allow_ready: bool = False) -> None:
        busy = {"moving_to_start", "replaying", "connecting"}
        if self._state in busy:
            raise RuntimeError(f"Cannot reconfigure while replay state is {self._state!r}.")
        if not allow_ready and self._state == "start_ready":
            raise RuntimeError("Abort or start the ready replay before reconfiguring.")

    def _ensure_no_worker_locked(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            raise RuntimeError(f"Replay worker is already active in state {self._state!r}.")

    def _require_rig_locked(self) -> ReplayHardwareRig:
        if self._rig is None:
            raise RuntimeError("Connect hardware before hardware replay preflight.")
        if not self._rig.connected:
            raise RuntimeError("Hardware replay rig is not ready.")
        return self._rig

    def _require_checked_group_locked(self) -> ReplayTrajectoryGroup:
        if self._group is None:
            raise RuntimeError("Run preflight before hardware replay.")
        if self._preflight_report is None or not self._preflight_report.ok:
            raise RuntimeError("Hardware replay preflight has not passed.")
        return self._group

    def checked_source_binding(self) -> ReplayGuiSourceBinding:
        """Return the typed binding for the exact preflighted replay group."""

        with self._lock:
            if self._preflight_report is None or not self._preflight_report.ok:
                raise RuntimeError("Replay source binding requires a successful preflight.")
            if self._source_binding is None:
                detail = self._source_binding_error or "binding was not produced"
                raise RuntimeError(f"Replay source binding is unavailable: {detail}")
            return self._source_binding

    def _require_output_enabled_locked(self) -> None:
        if not self._settings.send_actions_enabled:
            raise RuntimeError("Arm output is disabled. Enable send_actions before robot motion.")

    def _require_position_collision_contract_locked(self) -> None:
        if (
            normalize_rm65_arm_control_mode(self._hardware_config.arm_control_mode)
            != RM65_ARM_CONTROL_MODE_POSITION
        ):
            return
        if not bool(self._settings.position_collision_required):
            if bool(self._settings.send_actions_enabled):
                raise RuntimeError(
                    "real position replay output requires position_collision_required=true; "
                    "the legacy unchecked path is non-motion dry-run only"
                )
            return
        active_sides = tuple(self._hardware_config.active_sides)
        if set(active_sides) != {"left", "right"} or len(active_sides) != 2:
            raise RuntimeError(
                "collision-required position replay requires active_sides=both "
                "before preflight or motion; "
                f"got active_sides={active_sides}."
            )

    def _clear_qpos_init_status_locked(self) -> None:
        self._qpos_init_plan_summary = None
        self._qpos_init_step_index = 0
        self._qpos_init_step_count = 0

    def _default_branch_neutral_qpos(self) -> dict[str, np.ndarray] | None:
        sides = tuple(self._hardware_config.active_sides)
        if sides != ("left", "right") or not bool(self._settings.qpos_init_branch_search):
            return None
        target = load_qpos_target(DEFAULT_QPOS_TARGET_PATH).qpos_by_side()
        return {side: np.asarray(target[side], dtype=float).reshape(ARM_JOINT_DOF) for side in sides}

    def _build_qpos_init_plan(
        self,
        rig: ReplayHardwareRig,
        target_qpos: Mapping[str, np.ndarray],
    ) -> Any:
        current_raw = rig.current_qpos_by_side()
        current: dict[str, np.ndarray] = {}
        for side in self._hardware_config.active_sides:
            qpos = current_raw.get(side)
            if qpos is None:
                raise RuntimeError(f"{side} current hardware qpos is unavailable for qpos init.")
            current[side] = np.asarray(qpos, dtype=float).reshape(ARM_JOINT_DOF)
        plan = build_rm65_qpos_init_plan(
            start_qpos_by_side=current,
            target_qpos_by_side=target_qpos,
            active_sides=self._hardware_config.active_sides,
            qpos_dt_sec=float(self._settings.qpos_init_dt_sec),
            joint_velocity_profile=str(self._settings.qpos_init_joint_velocity_profile),
            qpos_init_max_angvel_rad_s=self._settings.qpos_init_max_angvel_rad_s,
            max_steps_per_side=int(self._settings.qpos_init_max_steps_per_side),
            collision_check=bool(self._settings.qpos_init_collision_check),
            collision_scene_profile=str(self._settings.qpos_init_collision_scene_profile),
            collision_slow_distance_m=float(self._settings.qpos_init_collision_slow_distance_m),
            collision_hold_distance_m=float(self._settings.qpos_init_collision_hold_distance_m),
            collision_stop_distance_m=float(self._settings.qpos_init_collision_stop_distance_m),
            max_joint_abs_deg=float(self._settings.qpos_init_max_joint_abs_deg),
            joint_slow_margin_deg=float(self._settings.qpos_init_joint_slow_margin_deg),
            joint_hold_margin_deg=float(self._settings.qpos_init_joint_hold_margin_deg),
            single_arm_max_qpos_delta_deg=float(self._settings.qpos_init_single_arm_max_qpos_delta_deg),
            branch_search=bool(self._settings.qpos_init_branch_search),
            branch_search_max_branches=int(self._settings.qpos_init_branch_search_max_branches),
            branch_neutral_qpos_by_side=self._default_branch_neutral_qpos(),
        )
        _require_plan_completed(plan)
        return plan

    def _first_qpos_targets_locked(self) -> dict[str, np.ndarray]:
        group = self._require_checked_group_locked()
        targets, sources = select_replay_qpos_init_targets(
            group,
            self._hardware_config.active_sides,
            index=0,
            source=self._settings.qpos_init_source,
        )
        missing = [side for side in self._hardware_config.active_sides if side not in targets]
        if missing:
            raise RuntimeError(
                "Replay start is missing finite qpos init target(s) for "
                f"{missing}; qpos_init_source={self._settings.qpos_init_source!r}."
            )
        self._last_message = (
            "qpos init targets selected: "
            + ", ".join(f"{side}={sources.get(side)}" for side in self._hardware_config.active_sides)
        )
        return {side: target.astype(float, copy=True) for side, target in targets.items()}

    def _qpos_targets_at_index(
        self,
        group: ReplayTrajectoryGroup,
        index: int,
    ) -> dict[str, np.ndarray]:
        targets: dict[str, np.ndarray] = {}
        for side in self._hardware_config.active_sides:
            values = group.arm_joint_targets.get(side)
            if values is None or np.asarray(values).shape[0] <= int(index):
                raise RuntimeError(f"{side} replay is missing arm joint targets.")
            target = np.asarray(values[int(index)], dtype=float).reshape(-1)
            if target.shape != (ARM_JOINT_DOF,) or not np.all(np.isfinite(target)):
                raise RuntimeError(f"{side} replay qpos target at frame {int(index)} is invalid.")
            targets[side] = target.astype(float, copy=True)
        return targets

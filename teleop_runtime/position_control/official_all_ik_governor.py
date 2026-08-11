"""Production adapter for source-rate official all-IK and qref governance.

This module is intentionally an application adapter.  The vendor candidate
producer lives in :mod:`hardware.rm65`, while branch selection and the
finite-horizon governor are backend-neutral :mod:`robot_motion` components.
The adapter binds those pieces to the existing canonical RM65 collision
planner without owning dispatch or policy queue authority.
"""

from __future__ import annotations

import hashlib
import math
import multiprocessing as mp
import os
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from hardware.rm65.official_all_ik import RM65OfficialAllIK
from robot_motion.position_control import (
    JointHorizonBatchedWorkspace,
    JointHorizonLimits,
    JointHorizonPlan,
    JointHorizonState,
    PositionPlanSpliceDynamicsConfig,
    PositionPlanSpliceDecision,
    OfflineJointWaypointTrajectory,
    OfflineWaypointTrajectoryInfeasibleError,
    ResampledMotionReference,
    build_offline_quintic_waypoint_trajectory,
    initial_jerk_regularization_for_position_plan_splice,
    jerk_delta_regularization_for_position_plan_splice,
    plan_jerk_limited_joint_horizon,
    initial_state_for_position_plan_splice,
)
from robot_motion.rm65 import (
    RM65_ABSOLUTE_Q_MAX_RAD,
    RM65_ABSOLUTE_Q_MIN_RAD,
    RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
    RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S,
    RM65PositionCollisionPlanner,
    RM65PositionCollisionPlannerConfig,
    RM65PositionCollisionPlanResult,
    RM65PositionPlannerProtocol,
    select_nearest_collision_safe_rm65_ik_pair,
)
from robot_motion.runtime import bind_current_thread_to_cpu


OFFICIAL_SOURCE_ALL_IK_GOVERNOR_BACKEND = (
    "official_source_all_ik_causal_governor"
)
PINK_POSE_LIFT_BACKEND = "pink_pose_lift"
POSITION_PATH_LIFT_BACKEND_CHOICES = (
    OFFICIAL_SOURCE_ALL_IK_GOVERNOR_BACKEND,
    PINK_POSE_LIFT_BACKEND,
)
CONTINUOUS_WAYPOINT_WORKER_THREAD = "thread"
CONTINUOUS_WAYPOINT_WORKER_PROCESS = "process"
CONTINUOUS_WAYPOINT_WORKER_ISOLATION_CHOICES = (
    CONTINUOUS_WAYPOINT_WORKER_THREAD,
    CONTINUOUS_WAYPOINT_WORKER_PROCESS,
)


class RM65AllIKCandidateProvider(Protocol):
    def solve_candidates(
        self,
        *,
        seed_rad: Sequence[float],
        target_endpoint: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        ...


@dataclass(frozen=True)
class _SourcePose:
    timestamp: float
    pose_by_side: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class _ContinuousWaypointBuildRequest:
    key: str
    epoch: int
    interval_index: int
    horizon_last_index: int
    interval_timestamp: float
    source_indices: tuple[int, ...]
    timestamps: np.ndarray
    dual_waypoints: np.ndarray
    initial_velocity_rad_s: np.ndarray
    initial_acceleration_rad_s2: np.ndarray
    terminal_velocity_rad_s: np.ndarray
    sample_timestamps: np.ndarray
    submitted_monotonic: float


@dataclass(frozen=True)
class _ContinuousWaypointBuildResult:
    request: _ContinuousWaypointBuildRequest
    shadow: OfflineJointWaypointTrajectory | None
    status: str
    compute_ms: float
    error_type: str | None = None
    error: str | None = None
    pre_admitted: bool = False
    discrete_qd_rad_s: np.ndarray | None = None
    discrete_qdd_rad_s2: np.ndarray | None = None
    admission_metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RM65ContinuousWaypointProcessSpec:
    """Pure construction inputs for a spawn-isolated C2 proposal process."""

    collision_planner_config: RM65PositionCollisionPlannerConfig
    endpoint_to_tool_by_side: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        transforms = {
            side: np.asarray(self.endpoint_to_tool_by_side[side], dtype=float)
            .reshape(4, 4)
            .copy()
            for side in ("left", "right")
        }
        if any(not np.all(np.isfinite(value)) for value in transforms.values()):
            raise ValueError(
                "endpoint_to_tool_by_side must contain finite transforms"
            )
        object.__setattr__(self, "endpoint_to_tool_by_side", transforms)


@dataclass(frozen=True)
class _ContinuousWaypointWireTrajectory:
    timestamps: np.ndarray
    q_rad: np.ndarray
    qd_rad_s: np.ndarray
    qdd_rad_s2: np.ndarray
    jerk_rad_s3: np.ndarray
    source_waypoint_timestamps: np.ndarray
    waypoint_timestamps: np.ndarray
    waypoint_q_rad: np.ndarray
    knot_qd_rad_s: np.ndarray
    knot_qdd_rad_s2: np.ndarray
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class _ContinuousWaypointWireResult:
    request: _ContinuousWaypointBuildRequest
    shadow: _ContinuousWaypointWireTrajectory | None
    status: str
    compute_ms: float
    error_type: str | None = None
    error: str | None = None
    pre_admitted: bool = False
    discrete_qd_rad_s: np.ndarray | None = None
    discrete_qdd_rad_s2: np.ndarray | None = None
    admission_metadata: Mapping[str, Any] | None = None


def _wire_trajectory(
    trajectory: OfflineJointWaypointTrajectory,
) -> _ContinuousWaypointWireTrajectory:
    return _ContinuousWaypointWireTrajectory(
        timestamps=np.asarray(trajectory.timestamps, dtype=float).copy(),
        q_rad=np.asarray(trajectory.q_rad, dtype=float).copy(),
        qd_rad_s=np.asarray(trajectory.qd_rad_s, dtype=float).copy(),
        qdd_rad_s2=np.asarray(trajectory.qdd_rad_s2, dtype=float).copy(),
        jerk_rad_s3=np.asarray(trajectory.jerk_rad_s3, dtype=float).copy(),
        source_waypoint_timestamps=np.asarray(
            trajectory.source_waypoint_timestamps,
            dtype=float,
        ).copy(),
        waypoint_timestamps=np.asarray(
            trajectory.waypoint_timestamps,
            dtype=float,
        ).copy(),
        waypoint_q_rad=np.asarray(trajectory.waypoint_q_rad, dtype=float).copy(),
        knot_qd_rad_s=np.asarray(trajectory.knot_qd_rad_s, dtype=float).copy(),
        knot_qdd_rad_s2=np.asarray(
            trajectory.knot_qdd_rad_s2,
            dtype=float,
        ).copy(),
        metadata=dict(trajectory.metadata),
    )


def _unwire_trajectory(
    trajectory: _ContinuousWaypointWireTrajectory,
) -> OfflineJointWaypointTrajectory:
    return OfflineJointWaypointTrajectory(
        timestamps=trajectory.timestamps,
        q_rad=trajectory.q_rad,
        qd_rad_s=trajectory.qd_rad_s,
        qdd_rad_s2=trajectory.qdd_rad_s2,
        jerk_rad_s3=trajectory.jerk_rad_s3,
        source_waypoint_timestamps=trajectory.source_waypoint_timestamps,
        waypoint_timestamps=trajectory.waypoint_timestamps,
        waypoint_q_rad=trajectory.waypoint_q_rad,
        knot_qd_rad_s=trajectory.knot_qd_rad_s,
        knot_qdd_rad_s2=trajectory.knot_qdd_rad_s2,
        metadata=dict(trajectory.metadata),
    )


def _wire_result(
    result: _ContinuousWaypointBuildResult,
) -> _ContinuousWaypointWireResult:
    return _ContinuousWaypointWireResult(
        request=result.request,
        shadow=None if result.shadow is None else _wire_trajectory(result.shadow),
        status=result.status,
        compute_ms=float(result.compute_ms),
        error_type=result.error_type,
        error=result.error,
        pre_admitted=bool(result.pre_admitted),
        discrete_qd_rad_s=(
            None
            if result.discrete_qd_rad_s is None
            else np.asarray(result.discrete_qd_rad_s, dtype=float).copy()
        ),
        discrete_qdd_rad_s2=(
            None
            if result.discrete_qdd_rad_s2 is None
            else np.asarray(result.discrete_qdd_rad_s2, dtype=float).copy()
        ),
        admission_metadata=dict(result.admission_metadata or {}),
    )


def _unwire_result(
    result: _ContinuousWaypointWireResult,
) -> _ContinuousWaypointBuildResult:
    return _ContinuousWaypointBuildResult(
        request=result.request,
        shadow=None if result.shadow is None else _unwire_trajectory(result.shadow),
        status=result.status,
        compute_ms=float(result.compute_ms),
        error_type=result.error_type,
        error=result.error,
        pre_admitted=bool(result.pre_admitted),
        discrete_qd_rad_s=result.discrete_qd_rad_s,
        discrete_qdd_rad_s2=result.discrete_qdd_rad_s2,
        admission_metadata=dict(result.admission_metadata or {}),
    )


def _certify_continuous_waypoint_cut_impl(
    shadow: OfflineJointWaypointTrajectory,
    *,
    initial_state: JointHorizonState,
    collision_planner: RM65PositionPlannerProtocol,
    commands_per_source: int,
    command_hz: float,
    dual_limits: JointHorizonLimits,
) -> tuple[bool, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    """Certify one complete C2 source cut without execution-state mutation."""

    required_last_row = int(commands_per_source)
    if int(shadow.q_rad.shape[0]) <= required_last_row:
        return False, None, None, {
            "admission_status": "formal_fallback_insufficient_cut_coverage",
            "authoritative": False,
            "dispatch_selected": False,
            "formal_qcmd_unchanged": True,
            "required_last_row": required_last_row,
            "available_last_row": int(shadow.q_rad.shape[0] - 1),
        }

    dt = 1.0 / float(command_hz)
    discrete_qd = np.zeros_like(shadow.q_rad)
    discrete_qdd = np.zeros_like(shadow.q_rad)
    discrete_qd[0] = np.asarray(initial_state.qd_rad_s, dtype=float)
    discrete_qdd[0] = np.asarray(initial_state.qdd_rad_s2, dtype=float)
    tolerance = 1e-7
    for row in range(1, required_last_row + 1):
        discrete_qd[row] = (
            np.asarray(shadow.q_rad[row], dtype=float)
            - np.asarray(shadow.q_rad[row - 1], dtype=float)
        ) / dt
        discrete_qdd[row] = (discrete_qd[row] - discrete_qd[row - 1]) / dt
        discrete_jerk = (discrete_qdd[row] - discrete_qdd[row - 1]) / dt
        violations = (
            ("velocity", np.abs(discrete_qd[row]), dual_limits.max_velocity_rad_s),
            (
                "acceleration",
                np.abs(discrete_qdd[row]),
                dual_limits.max_acceleration_rad_s2,
            ),
            ("jerk", np.abs(discrete_jerk), dual_limits.max_jerk_rad_s3),
        )
        for quantity, observed, limit in violations:
            failing = np.flatnonzero(observed > limit + tolerance)
            if failing.size:
                joint = int(failing[0])
                return False, None, None, {
                    "admission_status": "formal_fallback_discrete_dynamic_rejected",
                    "authoritative": False,
                    "dispatch_selected": False,
                    "formal_qcmd_unchanged": True,
                    "discrete_command_certificate_passed": False,
                    "discrete_rejected_row": row,
                    "discrete_rejected_joint_index": joint + 1,
                    "discrete_rejected_quantity": quantity,
                    "discrete_rejected_observed": float(observed[joint]),
                    "discrete_rejected_limit": float(limit[joint]),
                }

    previous = np.asarray(initial_state.q_rad, dtype=float).reshape(12)
    min_distance_m = math.inf
    checked_count = 0
    for row in range(1, required_last_row + 1):
        candidate = np.asarray(shadow.q_rad[row], dtype=float).reshape(12)
        checks = ((1.0, candidate),) + tuple(
            (alpha, (1.0 - alpha) * previous + alpha * candidate)
            for alpha in (0.25, 0.5, 0.75)
        )
        for alpha, sample in checks:
            checked_count += 1
            validation = collision_planner.validate_qpos_reference(
                {"left": sample[:6], "right": sample[6:]},
                source=(
                    "deploy_continuous_waypoint_admission_endpoint"
                    if alpha == 1.0
                    else "deploy_continuous_waypoint_admission_interval_sample"
                ),
            )
            if validation.collision_distance_m is not None:
                min_distance_m = min(
                    min_distance_m,
                    float(validation.collision_distance_m),
                )
            if not validation.safe:
                return False, None, None, {
                    "admission_status": "formal_fallback_collision_rejected",
                    "authoritative": False,
                    "dispatch_selected": False,
                    "formal_qcmd_unchanged": True,
                    "collision_checked": True,
                    "collision_admitted": False,
                    "collision_check_count": checked_count,
                    "collision_rejected_row": row,
                    "collision_rejected_alpha": alpha,
                    "collision_rejected_reason": validation.reason,
                    "collision_min_distance_m": (
                        None
                        if not math.isfinite(min_distance_m)
                        else min_distance_m
                    ),
                }
        previous = candidate

    return True, discrete_qd, discrete_qdd, {
        "admission_status": "admitted_for_complete_source_cut",
        "authoritative": True,
        "dispatch_selected": True,
        "formal_qcmd_unchanged": False,
        "collision_checked": True,
        "collision_admitted": True,
        "collision_check_count": checked_count,
        "collision_min_distance_m": (
            None if not math.isfinite(min_distance_m) else min_distance_m
        ),
        "committed_cut_rows": required_last_row,
        "discrete_command_certificate_passed": True,
        "discrete_command_certificate_scope": "complete_source_cut",
        "selection_scope": "complete_source_cut",
        "selection_switching": "forbidden_within_source_cut",
    }


def _build_continuous_waypoint_request_impl(
    request: _ContinuousWaypointBuildRequest,
    *,
    command_hz: float,
    dual_limits: JointHorizonLimits,
    commands_per_source: int,
    collision_planner: RM65PositionPlannerProtocol | None,
) -> _ContinuousWaypointBuildResult:
    started = time.perf_counter()
    try:
        shadow = build_offline_quintic_waypoint_trajectory(
            request.timestamps,
            request.dual_waypoints,
            command_hz=float(command_hz),
            limits=dual_limits,
            initial_velocity_rad_s=request.initial_velocity_rad_s,
            initial_acceleration_rad_s2=request.initial_acceleration_rad_s2,
            final_velocity_rad_s=request.terminal_velocity_rad_s,
            final_acceleration_rad_s2=np.zeros(12, dtype=float),
            sample_timestamps=request.sample_timestamps,
        )
    except OfflineWaypointTrajectoryInfeasibleError as exc:
        return _ContinuousWaypointBuildResult(
            request=request,
            shadow=None,
            status="infeasible",
            compute_ms=(time.perf_counter() - started) * 1000.0,
            error_type=type(exc).__name__,
            error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 - proposal is fail-open.
        return _ContinuousWaypointBuildResult(
            request=request,
            shadow=None,
            status="error",
            compute_ms=(time.perf_counter() - started) * 1000.0,
            error_type=type(exc).__name__,
            error=str(exc),
        )
    if collision_planner is None:
        return _ContinuousWaypointBuildResult(
            request=request,
            shadow=shadow,
            status="ready",
            compute_ms=(time.perf_counter() - started) * 1000.0,
        )
    try:
        admitted, discrete_qd, discrete_qdd, admission_metadata = (
            _certify_continuous_waypoint_cut_impl(
                shadow,
                initial_state=JointHorizonState(
                    q_rad=request.dual_waypoints[0],
                    qd_rad_s=request.initial_velocity_rad_s,
                    qdd_rad_s2=request.initial_acceleration_rad_s2,
                ),
                collision_planner=collision_planner,
                commands_per_source=int(commands_per_source),
                command_hz=float(command_hz),
                dual_limits=dual_limits,
            )
        )
    except Exception as exc:  # noqa: BLE001 - proposal is fail-open.
        admitted = False
        discrete_qd = None
        discrete_qdd = None
        admission_metadata = {
            "admission_status": "formal_fallback_precompute_check_error",
            "precompute_check_error_type": type(exc).__name__,
            "precompute_check_error": str(exc),
        }
    return _ContinuousWaypointBuildResult(
        request=request,
        shadow=shadow,
        status="ready",
        compute_ms=(time.perf_counter() - started) * 1000.0,
        pre_admitted=bool(admitted),
        discrete_qd_rad_s=discrete_qd,
        discrete_qdd_rad_s2=discrete_qdd,
        admission_metadata=admission_metadata,
    )


def _continuous_waypoint_process_main(
    request_queue: Any,
    result_queue: Any,
    *,
    process_spec: RM65ContinuousWaypointProcessSpec,
    worker_cpu: int | None,
    command_hz: float,
    commands_per_source: int,
    dual_limits: JointHorizonLimits,
) -> None:
    """Spawn target owning only proposal collision and trajectory state."""

    try:
        affinity = bind_current_thread_to_cpu(
            worker_cpu,
            role="continuous_c2_precompute_process",
        )
        collision_planner = RM65PositionCollisionPlanner(
            process_spec.collision_planner_config,
            endpoint_to_tool_by_side=process_spec.endpoint_to_tool_by_side,
        )
        result_queue.put(
            {
                "kind": "startup",
                "pid": os.getpid(),
                "affinity": dict(affinity),
                "collision_pair_count": int(collision_planner.collision_pair_count),
            }
        )
    except BaseException as exc:  # noqa: BLE001 - report startup failure.
        result_queue.put(
            {
                "kind": "startup_error",
                "pid": os.getpid(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return

    while True:
        try:
            message = request_queue.get()
        except (EOFError, OSError):
            return
        kind = str(message.get("kind", ""))
        if kind == "shutdown":
            return
        if kind == "parent_queue_warmup":
            result_queue.put(
                {
                    "kind": "parent_queue_warmup",
                    "token": str(message.get("token", "")),
                }
            )
            continue
        if kind == "reset":
            token = str(message.get("token", ""))
            try:
                receipt = collision_planner.reset(message["qpos_by_side"])
            except BaseException as exc:  # noqa: BLE001
                result_queue.put(
                    {
                        "kind": "reset_error",
                        "token": token,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            else:
                result_queue.put(
                    {
                        "kind": "reset",
                        "token": token,
                        "receipt": dict(receipt),
                    }
                )
            continue
        if kind != "build":
            continue
        request = message.get("request")
        if not isinstance(request, _ContinuousWaypointBuildRequest):
            result_queue.put(
                {
                    "kind": "worker_error",
                    "error_type": "TypeError",
                    "error": "process build request has invalid type",
                }
            )
            continue
        result = _build_continuous_waypoint_request_impl(
            request,
            command_hz=float(command_hz),
            dual_limits=dual_limits,
            commands_per_source=int(commands_per_source),
            collision_planner=collision_planner,
        )
        result_queue.put({"kind": "result", "result": _wire_result(result)})


class RM65OfficialAllIKGovernorPlanner:
    """Lift source Pose keyframes, then emit certified jerk-limited qcmd."""

    def __init__(
        self,
        collision_planner: RM65PositionPlannerProtocol,
        *,
        continuous_waypoint_precompute_collision_planner: (
            RM65PositionPlannerProtocol | None
        ) = None,
        endpoint_to_tool_by_side: Mapping[str, np.ndarray],
        candidate_provider: RM65AllIKCandidateProvider | None = None,
        source_hz: float = 15.0,
        command_hz: float = 120.0,
        lookahead_source_steps: int = 2,
        jerk_ramp_sec: float = 0.05,
        batched_solver_tolerance: float = 1e-6,
        continuous_waypoint_shadow_enabled: bool = False,
        continuous_waypoint_admission_enabled: bool = False,
        continuous_waypoint_async_precompute_enabled: bool = False,
        continuous_waypoint_worker_cpu: int | None = None,
        continuous_waypoint_worker_isolation: str = (
            CONTINUOUS_WAYPOINT_WORKER_THREAD
        ),
        continuous_waypoint_process_spec: (
            RM65ContinuousWaypointProcessSpec | None
        ) = None,
        plan_splice_dynamics_config: PositionPlanSpliceDynamicsConfig | None = None,
        max_velocity_rad_s: Sequence[float] = (
            RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S
        ),
        max_acceleration_rad_s2: Sequence[float] = (
            RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2
        ),
    ) -> None:
        if not isinstance(lookahead_source_steps, int) or lookahead_source_steps < 2:
            raise ValueError("lookahead_source_steps must be an integer >= 2.")
        self.source_hz = float(source_hz)
        self.command_hz = float(command_hz)
        self.lookahead_source_steps = int(lookahead_source_steps)
        self.jerk_ramp_sec = float(jerk_ramp_sec)
        self.batched_solver_tolerance = float(batched_solver_tolerance)
        self.continuous_waypoint_admission_enabled = bool(
            continuous_waypoint_admission_enabled
        )
        self.continuous_waypoint_async_precompute_enabled = bool(
            continuous_waypoint_async_precompute_enabled
        )
        if continuous_waypoint_worker_cpu is not None and (
            isinstance(continuous_waypoint_worker_cpu, bool)
            or int(continuous_waypoint_worker_cpu) != continuous_waypoint_worker_cpu
            or int(continuous_waypoint_worker_cpu) < 0
        ):
            raise ValueError(
                "continuous_waypoint_worker_cpu must be a non-negative "
                "integer or None"
            )
        self.continuous_waypoint_worker_cpu = (
            None
            if continuous_waypoint_worker_cpu is None
            else int(continuous_waypoint_worker_cpu)
        )
        worker_isolation = str(continuous_waypoint_worker_isolation).strip().lower()
        if worker_isolation not in CONTINUOUS_WAYPOINT_WORKER_ISOLATION_CHOICES:
            raise ValueError(
                "continuous_waypoint_worker_isolation must be one of "
                f"{CONTINUOUS_WAYPOINT_WORKER_ISOLATION_CHOICES}"
            )
        self.continuous_waypoint_worker_isolation = worker_isolation
        self._continuous_waypoint_process_spec = continuous_waypoint_process_spec
        if (
            self.continuous_waypoint_worker_cpu is not None
            and not self.continuous_waypoint_async_precompute_enabled
        ):
            raise ValueError(
                "continuous_waypoint_worker_cpu requires async precompute"
            )
        if (
            self.continuous_waypoint_async_precompute_enabled
            and not self.continuous_waypoint_admission_enabled
        ):
            raise ValueError(
                "continuous waypoint async precompute requires experimental "
                "continuous waypoint admission."
            )
        self.continuous_waypoint_shadow_enabled = bool(
            continuous_waypoint_shadow_enabled
            or self.continuous_waypoint_admission_enabled
        )
        if (
            not math.isfinite(self.source_hz)
            or not math.isfinite(self.command_hz)
            or self.source_hz <= 0.0
            or self.command_hz <= 0.0
            or not math.isclose(
                self.command_hz / self.source_hz,
                round(self.command_hz / self.source_hz),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("command_hz/source_hz must be a positive integer ratio.")
        if not math.isfinite(self.jerk_ramp_sec) or self.jerk_ramp_sec <= 0.0:
            raise ValueError("jerk_ramp_sec must be finite and positive.")
        self.commands_per_source = int(round(self.command_hz / self.source_hz))
        self._collision_planner = collision_planner
        self._continuous_waypoint_precompute_collision_planner = (
            continuous_waypoint_precompute_collision_planner
        )
        self._candidate_provider = candidate_provider or RM65OfficialAllIK()
        self.plan_splice_dynamics_config = (
            PositionPlanSpliceDynamicsConfig()
            if plan_splice_dynamics_config is None
            else plan_splice_dynamics_config
        )
        if not isinstance(
            self.plan_splice_dynamics_config,
            PositionPlanSpliceDynamicsConfig,
        ):
            raise TypeError(
                "plan_splice_dynamics_config must be a "
                "PositionPlanSpliceDynamicsConfig or None."
            )
        self._inverse_endpoint_to_tool = {
            side: np.linalg.inv(
                np.asarray(endpoint_to_tool_by_side[side], dtype=float).reshape(4, 4)
            )
            for side in ("left", "right")
        }
        max_acceleration = np.asarray(max_acceleration_rad_s2, dtype=float).reshape(6)
        self._dual_limits = JointHorizonLimits(
            min_position_rad=np.tile(
                np.asarray(RM65_ABSOLUTE_Q_MIN_RAD, dtype=float),
                2,
            ),
            max_position_rad=np.tile(
                np.asarray(RM65_ABSOLUTE_Q_MAX_RAD, dtype=float),
                2,
            ),
            max_velocity_rad_s=np.tile(
                np.asarray(max_velocity_rad_s, dtype=float).reshape(6),
                2,
            ),
            max_acceleration_rad_s2=np.tile(max_acceleration, 2),
            max_jerk_rad_s3=np.tile(
                max_acceleration / self.jerk_ramp_sec,
                2,
            ),
        )
        self._governor_workspace = JointHorizonBatchedWorkspace(
            limits=self._dual_limits,
            dt_sec=1.0 / self.command_hz,
            horizon_transitions=(
                self.lookahead_source_steps * self.commands_per_source
            ),
            solver_tolerance=self.batched_solver_tolerance,
        )
        hold_restart_jerk_weight = float(
            self.plan_splice_dynamics_config
            .hold_restart_initial_jerk_delta_regularization
        )
        hold_restart_jerk_delta_weight = float(
            self.plan_splice_dynamics_config
            .hold_restart_jerk_delta_regularization
        )
        self._hold_restart_governor_workspace = (
            None
            if (
                hold_restart_jerk_weight <= 0.0
                and hold_restart_jerk_delta_weight <= 0.0
            )
            else JointHorizonBatchedWorkspace(
                limits=self._dual_limits,
                dt_sec=1.0 / self.command_hz,
                horizon_transitions=(
                    self.lookahead_source_steps * self.commands_per_source
                ),
                solver_tolerance=self.batched_solver_tolerance,
                jerk_delta_regularization=hold_restart_jerk_delta_weight,
                initial_jerk_delta_regularization=hold_restart_jerk_weight,
            )
        )
        self._source_pose: dict[int, _SourcePose] = {}
        self._source_qpos: dict[int, dict[str, np.ndarray]] = {}
        self._plan_splice_by_source: dict[int, PositionPlanSpliceDecision] = {}
        self._last_selected_source_index: int | None = None
        self._committed_source_index: int | None = None
        self._reset_anchor: dict[str, np.ndarray] | None = None
        self._selection_anchor: dict[str, np.ndarray] | None = None
        self._state: dict[str, JointHorizonState] | None = None
        self._cached_interval_index: int | None = None
        self._cached_interval_timestamp: float | None = None
        self._cached_plan: Any | None = None
        self._cached_qref: dict[str, np.ndarray] | None = None
        self._prepared_initial_interval_pending = False
        self._cached_continuous_waypoint_shadow: (
            OfflineJointWaypointTrajectory | None
        ) = None
        self._cached_continuous_waypoint_shadow_metadata: dict[str, Any] = {}
        self._cached_continuous_waypoint_admitted = False
        self._cached_continuous_waypoint_discrete_qd: np.ndarray | None = None
        self._cached_continuous_waypoint_discrete_qdd: np.ndarray | None = None
        self._continuous_precompute_epoch = 0
        self._continuous_precompute_condition = threading.Condition()
        self._continuous_precompute_pending: (
            _ContinuousWaypointBuildRequest | None
        ) = None
        self._continuous_precompute_running_key: str | None = None
        self._continuous_precompute_results: OrderedDict[
            str, _ContinuousWaypointBuildResult
        ] = OrderedDict()
        self._continuous_precompute_stop = False
        self._continuous_precompute_thread: threading.Thread | None = None
        self._continuous_precompute_process: Any | None = None
        self._continuous_precompute_process_request_queue: Any | None = None
        self._continuous_precompute_process_result_queue: Any | None = None
        self._continuous_precompute_process_pending_key: str | None = None
        self._continuous_precompute_process_error: str | None = None
        self._continuous_precompute_process_pid: int | None = None
        self._continuous_precompute_affinity_ready = threading.Event()
        self._continuous_precompute_affinity_receipt: dict[str, Any] | None = None
        self._continuous_precompute_affinity_error: str | None = None
        self._continuous_precompute_next_anchor: (
            tuple[int, JointHorizonState] | None
        ) = None
        self._continuous_precompute_submitted_count = 0
        self._continuous_precompute_completed_count = 0
        self._continuous_precompute_superseded_count = 0
        self._continuous_precompute_miss_count = 0
        if self.continuous_waypoint_async_precompute_enabled:
            if (
                self.continuous_waypoint_worker_isolation
                == CONTINUOUS_WAYPOINT_WORKER_PROCESS
            ):
                if self._continuous_waypoint_process_spec is None:
                    raise ValueError(
                        "process-isolated continuous waypoint precompute "
                        "requires continuous_waypoint_process_spec"
                    )
                if self._continuous_waypoint_precompute_collision_planner is not None:
                    raise ValueError(
                        "process-isolated continuous waypoint precompute must "
                        "construct its proposal planner inside the child process"
                    )
                self._start_continuous_waypoint_process()
            else:
                if self._continuous_waypoint_precompute_collision_planner is None:
                    raise ValueError(
                        "continuous waypoint async precompute requires an isolated "
                        "proposal collision planner."
                    )
                self._continuous_precompute_thread = threading.Thread(
                    target=self._continuous_waypoint_precompute_loop,
                    name="rm65-continuous-waypoint-precompute",
                    daemon=True,
                )
                self._continuous_precompute_thread.start()
                if not self._continuous_precompute_affinity_ready.wait(timeout=1.0):
                    self.shutdown_background_workers(timeout_sec=1.0)
                    raise RuntimeError(
                        "continuous waypoint precompute worker did not publish its "
                        "CPU-affinity receipt within 1.000s"
                    )
                if self._continuous_precompute_affinity_error is not None:
                    error = self._continuous_precompute_affinity_error
                    self.shutdown_background_workers(timeout_sec=1.0)
                    raise RuntimeError(
                        "continuous waypoint precompute CPU-affinity setup failed: "
                        f"{error}"
                    )
        self._sequence = 0

    @property
    def collision_pair_count(self) -> int:
        return int(self._collision_planner.collision_pair_count)

    def _start_continuous_waypoint_process(self) -> None:
        process_spec = self._continuous_waypoint_process_spec
        assert process_spec is not None
        context = mp.get_context("spawn")
        request_queue = context.Queue(maxsize=1)
        result_queue = context.Queue(maxsize=4)
        process = context.Process(
            target=_continuous_waypoint_process_main,
            kwargs={
                "request_queue": request_queue,
                "result_queue": result_queue,
                "process_spec": process_spec,
                "worker_cpu": self.continuous_waypoint_worker_cpu,
                "command_hz": self.command_hz,
                "commands_per_source": self.commands_per_source,
                "dual_limits": self._dual_limits,
            },
            name="rm65-continuous-waypoint-precompute-process",
            daemon=True,
        )
        self._continuous_precompute_process_request_queue = request_queue
        self._continuous_precompute_process_result_queue = result_queue
        self._continuous_precompute_process = process
        process.start()
        try:
            startup = result_queue.get(timeout=15.0)
        except queue.Empty as exc:
            self.shutdown_background_workers(timeout_sec=1.0)
            raise RuntimeError(
                "continuous waypoint proposal process did not publish startup "
                "within 15.000s"
            ) from exc
        if str(startup.get("kind", "")) != "startup":
            self._continuous_precompute_process_error = (
                f"{startup.get('error_type', 'RuntimeError')}: "
                f"{startup.get('error', 'unknown process startup failure')}"
            )
            self.shutdown_background_workers(timeout_sec=1.0)
            raise RuntimeError(
                "continuous waypoint proposal process startup failed: "
                f"{self._continuous_precompute_process_error}"
            )
        receipt = dict(startup.get("affinity") or {})
        receipt.update(
            {
                "worker_isolation": CONTINUOUS_WAYPOINT_WORKER_PROCESS,
                "process_pid": int(startup["pid"]),
                "collision_pair_count": int(startup["collision_pair_count"]),
            }
        )
        self._continuous_precompute_process_pid = int(startup["pid"])
        self._continuous_precompute_affinity_receipt = receipt
        warmup_token = f"parent-request-feeder:{self._continuous_precompute_epoch}"
        request_queue.put(
            {
                "kind": "parent_queue_warmup",
                "token": warmup_token,
            }
        )
        try:
            warmup = result_queue.get(timeout=2.0)
        except queue.Empty as exc:
            self.shutdown_background_workers(timeout_sec=1.0)
            raise RuntimeError(
                "continuous waypoint proposal process did not acknowledge "
                "the parent request-queue feeder warmup within 2.000s"
            ) from exc
        if (
            str(warmup.get("kind", "")) != "parent_queue_warmup"
            or str(warmup.get("token", "")) != warmup_token
        ):
            self.shutdown_background_workers(timeout_sec=1.0)
            raise RuntimeError(
                "continuous waypoint proposal process returned an invalid "
                "parent request-queue feeder warmup receipt"
            )
        self._continuous_precompute_affinity_receipt[
            "parent_request_queue_feeder_warmed"
        ] = True
        self._continuous_precompute_affinity_ready.set()

    def _store_continuous_waypoint_process_result(
        self,
        wire_result: _ContinuousWaypointWireResult,
    ) -> None:
        result = _unwire_result(wire_result)
        request = result.request
        if request.key == self._continuous_precompute_process_pending_key:
            self._continuous_precompute_process_pending_key = None
        if (
            self._continuous_precompute_stop
            or request.epoch != self._continuous_precompute_epoch
        ):
            return
        self._continuous_precompute_results[request.key] = result
        self._continuous_precompute_results.move_to_end(request.key)
        while len(self._continuous_precompute_results) > 2:
            self._continuous_precompute_results.popitem(last=False)
        self._continuous_precompute_completed_count += 1

    def _drain_continuous_waypoint_process_results(self) -> None:
        result_queue = self._continuous_precompute_process_result_queue
        if result_queue is None:
            return
        while True:
            try:
                message = result_queue.get_nowait()
            except queue.Empty:
                return
            kind = str(message.get("kind", ""))
            if kind == "result":
                wire_result = message.get("result")
                if isinstance(wire_result, _ContinuousWaypointWireResult):
                    self._store_continuous_waypoint_process_result(wire_result)
                else:
                    self._continuous_precompute_process_error = (
                        "TypeError: process result payload has invalid type"
                    )
            elif kind in {"worker_error", "startup_error"}:
                self._continuous_precompute_process_error = (
                    f"{message.get('error_type', 'RuntimeError')}: "
                    f"{message.get('error', 'unknown proposal process failure')}"
                )

    def _reset_continuous_waypoint_process(
        self,
        qpos_by_side: Mapping[str, np.ndarray],
        *,
        timeout_sec: float = 5.0,
    ) -> Mapping[str, Any]:
        request_queue = self._continuous_precompute_process_request_queue
        result_queue = self._continuous_precompute_process_result_queue
        process = self._continuous_precompute_process
        if request_queue is None or result_queue is None or process is None:
            raise RuntimeError("continuous waypoint proposal process is unavailable")
        if not process.is_alive():
            raise RuntimeError("continuous waypoint proposal process exited")
        token = f"reset:{self._continuous_precompute_epoch}:{time.monotonic_ns()}"
        request_queue.put(
            {
                "kind": "reset",
                "token": token,
                "qpos_by_side": {
                    side: np.asarray(qpos_by_side[side], dtype=float).copy()
                    for side in ("left", "right")
                },
            },
            timeout=float(timeout_sec),
        )
        deadline = time.monotonic() + float(timeout_sec)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(
                    "continuous waypoint proposal process reset timed out"
                )
            try:
                message = result_queue.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError(
                    "continuous waypoint proposal process reset timed out"
                ) from exc
            kind = str(message.get("kind", ""))
            if kind == "result":
                wire_result = message.get("result")
                if isinstance(wire_result, _ContinuousWaypointWireResult):
                    self._store_continuous_waypoint_process_result(wire_result)
                continue
            if str(message.get("token", "")) != token:
                continue
            if kind == "reset":
                return dict(message.get("receipt") or {})
            if kind == "reset_error":
                raise RuntimeError(
                    "continuous waypoint proposal process reset failed: "
                    f"{message.get('error_type', 'RuntimeError')}: "
                    f"{message.get('error', 'unknown reset failure')}"
                )

    def reset(self, current_qpos_by_side: Mapping[str, Any]) -> Mapping[str, Any]:
        qpos = {
            side: np.asarray(current_qpos_by_side[side], dtype=float).reshape(6).copy()
            for side in ("left", "right")
        }
        receipt = self._collision_planner.reset(qpos)
        proposal_reset_error: str | None = None
        if self.continuous_waypoint_async_precompute_enabled:
            try:
                if (
                    self.continuous_waypoint_worker_isolation
                    == CONTINUOUS_WAYPOINT_WORKER_PROCESS
                ):
                    self._reset_continuous_waypoint_process(qpos)
                else:
                    proposal_planner = (
                        self._continuous_waypoint_precompute_collision_planner
                    )
                    assert proposal_planner is not None
                    proposal_planner.reset(qpos)
            except Exception as exc:  # noqa: BLE001 - proposal path is optional.
                proposal_reset_error = f"{type(exc).__name__}: {exc}"
        self._source_pose.clear()
        self._source_qpos.clear()
        self._plan_splice_by_source.clear()
        self._last_selected_source_index = None
        self._committed_source_index = None
        self._reset_anchor = {
            side: value.copy() for side, value in qpos.items()
        }
        self._selection_anchor = qpos
        self._state = {
            side: JointHorizonState(
                q_rad=qpos[side],
                qd_rad_s=np.zeros(6, dtype=float),
                qdd_rad_s2=np.zeros(6, dtype=float),
            )
            for side in ("left", "right")
        }
        self._cached_interval_index = None
        self._cached_interval_timestamp = None
        self._cached_plan = None
        self._cached_qref = None
        self._prepared_initial_interval_pending = False
        self._cached_continuous_waypoint_shadow = None
        self._cached_continuous_waypoint_shadow_metadata = {}
        self._cached_continuous_waypoint_admitted = False
        self._cached_continuous_waypoint_discrete_qd = None
        self._cached_continuous_waypoint_discrete_qdd = None
        self._continuous_precompute_next_anchor = None
        self._continuous_precompute_process_pending_key = None
        with self._continuous_precompute_condition:
            self._continuous_precompute_epoch += 1
            self._continuous_precompute_pending = None
            self._continuous_precompute_results.clear()
            self._continuous_precompute_condition.notify_all()
        self._sequence = 0
        return {
            **dict(receipt),
            "path_lift_backend": OFFICIAL_SOURCE_ALL_IK_GOVERNOR_BACKEND,
            "governor_state_reset": True,
            "governor_max_acceleration_rad_s2": (
                self._dual_limits.max_acceleration_rad_s2[:6].tolist()
            ),
            "governor_max_jerk_rad_s3": (
                self._dual_limits.max_jerk_rad_s3[:6].tolist()
            ),
            "governor_jerk_ramp_sec": self.jerk_ramp_sec,
            "continuous_waypoint_precompute_collision_reset_error": (
                proposal_reset_error
            ),
            "continuous_waypoint_worker_cpu": self.continuous_waypoint_worker_cpu,
            "continuous_waypoint_worker_isolation": (
                self.continuous_waypoint_worker_isolation
            ),
            "continuous_waypoint_process_pid": (
                self._continuous_precompute_process_pid
            ),
            "continuous_waypoint_process_error": (
                self._continuous_precompute_process_error
            ),
            "continuous_waypoint_worker_affinity_receipt": (
                None
                if self._continuous_precompute_affinity_receipt is None
                else dict(self._continuous_precompute_affinity_receipt)
            ),
        }

    def shutdown_background_workers(self, *, timeout_sec: float = 1.0) -> None:
        """Stop the optional proposal worker without touching robot state."""

        process = self._continuous_precompute_process
        if process is not None:
            self._continuous_precompute_stop = True
            request_queue = self._continuous_precompute_process_request_queue
            if request_queue is not None:
                try:
                    request_queue.put_nowait({"kind": "shutdown"})
                except queue.Full:
                    try:
                        request_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        request_queue.put_nowait({"kind": "shutdown"})
                    except queue.Full:
                        pass
            process.join(timeout=float(timeout_sec))
            if process.is_alive():
                process.terminate()
                process.join(timeout=float(timeout_sec))
            if process.is_alive():
                raise RuntimeError(
                    "continuous waypoint proposal process did not stop within "
                    f"{float(timeout_sec):.3f}s"
                )
            for process_queue in (
                self._continuous_precompute_process_request_queue,
                self._continuous_precompute_process_result_queue,
            ):
                if process_queue is not None:
                    process_queue.cancel_join_thread()
                    process_queue.close()
            self._continuous_precompute_process = None
            self._continuous_precompute_process_request_queue = None
            self._continuous_precompute_process_result_queue = None
            self._continuous_precompute_process_pending_key = None
            return

        thread = self._continuous_precompute_thread
        if thread is None:
            return
        with self._continuous_precompute_condition:
            self._continuous_precompute_stop = True
            self._continuous_precompute_pending = None
            self._continuous_precompute_condition.notify_all()
        if thread is not threading.current_thread():
            thread.join(timeout=float(timeout_sec))
            if thread.is_alive():
                raise RuntimeError(
                    "continuous waypoint precompute worker did not stop within "
                    f"{float(timeout_sec):.3f}s"
                )
        self._continuous_precompute_thread = None

    def close(self, *, timeout_sec: float = 1.0) -> None:
        self.shutdown_background_workers(timeout_sec=timeout_sec)

    def stage_source_pose(
        self,
        *,
        source_index: int,
        source_timestamp: float,
        pose_by_side: Mapping[str, Any],
        plan_splice_decision: PositionPlanSpliceDecision | None = None,
    ) -> None:
        index = int(source_index)
        timestamp = float(source_timestamp)
        poses = {
            side: np.asarray(pose_by_side[side], dtype=float).reshape(4, 4).copy()
            for side in ("left", "right")
        }
        if not math.isfinite(timestamp) or any(
            not np.all(np.isfinite(pose)) for pose in poses.values()
        ):
            raise ValueError("staged source pose and timestamp must be finite.")
        if plan_splice_decision is not None and not isinstance(
            plan_splice_decision,
            PositionPlanSpliceDecision,
        ):
            raise TypeError(
                "plan_splice_decision must be a PositionPlanSpliceDecision or None."
            )
        previous = self._source_pose.get(index)
        if previous is not None:
            equivalent = bool(
                math.isclose(
                    previous.timestamp,
                    timestamp,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                and all(
                    np.allclose(
                        previous.pose_by_side[side],
                        poses[side],
                        rtol=0.0,
                        atol=1e-8,
                    )
                    for side in ("left", "right")
                )
            )
            if equivalent:
                # Chunk lookahead restages the same future source from adjacent
                # submissions. Preserve the first accepted bytes so an exact
                # asynchronous admission hash cannot be invalidated by
                # sub-nanosecond floating-point re-association.
                existing_splice = self._plan_splice_by_source.get(index)
                if (
                    plan_splice_decision is not None
                    and existing_splice is not None
                    and existing_splice != plan_splice_decision
                ):
                    raise RuntimeError(
                        "an equivalent staged source Pose changed its plan-splice decision."
                    )
                if plan_splice_decision is not None:
                    self._plan_splice_by_source[index] = plan_splice_decision
                return
            # A replaced policy chunk may invalidate a speculative successor.
            # It is legal only if it has not been selected.
            if (
                self._committed_source_index is not None
                and index <= self._committed_source_index
            ):
                raise RuntimeError(
                    "an admitted source Pose cannot be replaced in place."
                )
            for stale_index in tuple(
                key for key in self._source_qpos if key >= index
            ):
                self._source_qpos.pop(stale_index, None)
            for stale_index in tuple(
                key for key in self._plan_splice_by_source if key >= index
            ):
                self._plan_splice_by_source.pop(stale_index, None)
            remaining = sorted(self._source_qpos)
            self._last_selected_source_index = (
                None if not remaining else remaining[-1]
            )
            if remaining:
                self._selection_anchor = {
                    side: self._source_qpos[remaining[-1]][side].copy()
                    for side in ("left", "right")
                }
            elif self._reset_anchor is not None:
                self._selection_anchor = {
                    side: self._reset_anchor[side].copy()
                    for side in ("left", "right")
                }
            if self._prepared_initial_interval_pending:
                self._cached_interval_index = None
                self._cached_interval_timestamp = None
                self._cached_plan = None
                self._cached_qref = None
                self._prepared_initial_interval_pending = False
        self._source_pose[index] = _SourcePose(timestamp=timestamp, pose_by_side=poses)
        if plan_splice_decision is not None:
            self._plan_splice_by_source[index] = plan_splice_decision

    def prepare_staged_source_horizon(self) -> Mapping[str, Any]:
        """Select the staged initial horizon before its reference clock starts.

        The first 120 Hz emission otherwise enumerates and collision-filters IK
        branches for the current source and all initial successors at once.
        Later cuts solve only one new source, so charging this one-time work to
        the first command tick creates an avoidable cold-start deadline miss.

        This calls the same authoritative selector as normal planning. It does
        not commit a source index or advance emitted motion state, and a true
        speculative replacement still invalidates the cached selections.
        """

        if self._state is None:
            raise RuntimeError("official all-IK governor was not reset.")
        if not self._source_pose:
            return {
                "prepared": False,
                "reason": "no_staged_source_pose",
                "last_source_index": None,
                "prepare_ms": 0.0,
            }
        started = time.perf_counter()
        last_source_index = max(self._source_pose)
        selection = self._select_source_qpos_through(last_source_index)
        interval_index = min(self._source_pose)
        horizon_last_index = min(
            interval_index + self.lookahead_source_steps,
            last_source_index,
        )
        interval_timestamp = self._source_pose[interval_index].timestamp
        dt = 1.0 / self.command_hz
        horizon_count = self.lookahead_source_steps * self.commands_per_source
        qref = {
            side: np.asarray(
                [
                    self._qref_at(interval_timestamp + offset * dt)[side]
                    for offset in range(horizon_count + 1)
                ],
                dtype=float,
            )
            for side in ("left", "right")
        }
        dual_state = JointHorizonState(
            q_rad=np.concatenate(
                (
                    np.asarray(self._state["left"].q_rad, dtype=float),
                    np.asarray(self._state["right"].q_rad, dtype=float),
                )
            ),
            qd_rad_s=np.concatenate(
                (
                    np.asarray(self._state["left"].qd_rad_s, dtype=float),
                    np.asarray(self._state["right"].qd_rad_s, dtype=float),
                )
            ),
            qdd_rad_s2=np.concatenate(
                (
                    np.asarray(self._state["left"].qdd_rad_s2, dtype=float),
                    np.asarray(self._state["right"].qdd_rad_s2, dtype=float),
                )
            ),
        )
        dual_state, splice_metadata = self._initial_state_for_source_cut(
            interval_index=interval_index,
            committed_state=dual_state,
        )
        plan_splice_decision = self._plan_splice_by_source.get(interval_index)
        warm_solve_started = time.perf_counter()
        prepared_plan = self._solve_formal_horizon(
            np.concatenate((qref["left"], qref["right"]), axis=1),
            initial_state=dual_state,
            commit_count=min(self.commands_per_source, horizon_count),
            plan_splice_decision=plan_splice_decision,
        )
        warm_solve_ms = (time.perf_counter() - warm_solve_started) * 1000.0
        self._cached_interval_index = int(interval_index)
        self._cached_interval_timestamp = float(interval_timestamp)
        self._cached_plan = prepared_plan
        self._cached_qref = {
            side: values.copy() for side, values in qref.items()
        }
        self._prepared_initial_interval_pending = True
        return {
            "prepared": True,
            "reason": "initial_source_horizon_selected_before_reference_clock",
            "last_source_index": int(last_source_index),
            "formal_horizon_last_source_index": int(horizon_last_index),
            "formal_workspace_warm_solve_ms": float(warm_solve_ms),
            "prepare_ms": (time.perf_counter() - started) * 1000.0,
            "plan_splice_initial_jerk_regularization_applied": bool(
                prepared_plan.metadata.get(
                    "plan_splice_initial_jerk_regularization_applied",
                    False,
                )
            ),
            "plan_splice_initial_jerk_regularization_fallback": bool(
                prepared_plan.metadata.get(
                    "plan_splice_initial_jerk_regularization_fallback",
                    False,
                )
            ),
            **splice_metadata,
            **selection,
        }

    def _select_source_qpos_through(self, last_index: int) -> dict[str, Any]:
        if self._selection_anchor is None:
            raise RuntimeError("official all-IK governor was not reset.")
        start = (
            min(self._source_pose)
            if self._last_selected_source_index is None
            else self._last_selected_source_index + 1
        )
        solved = 0
        candidate_counts: dict[int, dict[str, int]] = {}
        rejection_counts = 0
        for source_index in sorted(
            index
            for index in self._source_pose
            if start <= index <= int(last_index)
        ):
            source = self._source_pose[source_index]
            candidate_lists = {
                side: self._candidate_provider.solve_candidates(
                    seed_rad=self._selection_anchor[side],
                    target_endpoint=(
                        source.pose_by_side[side]
                        @ self._inverse_endpoint_to_tool[side]
                    ),
                )
                for side in ("left", "right")
            }
            receipt = select_nearest_collision_safe_rm65_ik_pair(
                candidate_lists,
                anchor_qpos_by_side=self._selection_anchor,
                validate_pair=lambda pair: (
                    self._collision_planner.validate_qpos_reference(
                        pair,
                        source="deploy_official_source_all_ik",
                    )
                ),
            )
            candidate_counts[source_index] = {
                side: int(receipt.candidate_count_by_side[side])
                for side in ("left", "right")
            }
            rejection_counts += int(receipt.collision_rejection_count)
            if not receipt.accepted or receipt.qpos_by_side is None:
                raise RuntimeError(
                    "official all-IK source selection failed closed: "
                    f"source_index={source_index} reason={receipt.reason} "
                    f"candidate_counts={dict(receipt.candidate_count_by_side)} "
                    f"rejections={dict(receipt.rejection_reason_counts)}"
                )
            selected = {
                side: np.asarray(
                    receipt.qpos_by_side[side],
                    dtype=float,
                ).copy()
                for side in ("left", "right")
            }
            self._source_qpos[source_index] = selected
            self._selection_anchor = selected
            self._last_selected_source_index = source_index
            solved += 1
        if int(last_index) not in self._source_qpos:
            raise RuntimeError(
                f"source Pose horizon is incomplete through index {last_index}."
            )
        return {
            "new_source_solutions": solved,
            "candidate_counts_by_source": candidate_counts,
            "collision_rejection_count": rejection_counts,
        }

    def _qref_at(self, timestamp: float) -> dict[str, np.ndarray]:
        selected_indices = sorted(self._source_qpos)
        if not selected_indices:
            raise RuntimeError("no selected source qpos is available.")
        target_time = float(timestamp)
        if target_time <= self._source_pose[selected_indices[0]].timestamp:
            index = selected_indices[0]
            return {
                side: self._source_qpos[index][side].copy()
                for side in ("left", "right")
            }
        for left_index, right_index in zip(
            selected_indices[:-1],
            selected_indices[1:],
        ):
            left_time = self._source_pose[left_index].timestamp
            right_time = self._source_pose[right_index].timestamp
            if target_time <= right_time:
                span = max(1e-9, right_time - left_time)
                alpha = float(np.clip((target_time - left_time) / span, 0.0, 1.0))
                return {
                    side: (
                        (1.0 - alpha) * self._source_qpos[left_index][side]
                        + alpha * self._source_qpos[right_index][side]
                    )
                    for side in ("left", "right")
                }
        index = selected_indices[-1]
        return {
            side: self._source_qpos[index][side].copy()
            for side in ("left", "right")
        }

    def _continuous_waypoint_base_metadata(
        self,
        *,
        interval_index: int,
        horizon_last_index: int,
    ) -> dict[str, Any]:
        return {
            "enabled": bool(self.continuous_waypoint_shadow_enabled),
            "admission_requested": bool(
                self.continuous_waypoint_admission_enabled
            ),
            "authoritative": False,
            "dispatch_selected": False,
            "formal_qcmd_unchanged": True,
            "collision_checked": False,
            "collision_admitted": False,
            "anchor_source": "formal_governor_admitted_state",
            "future_source_contract": "already_staged_same_chunk_only",
            "interval_source_index": int(interval_index),
            "horizon_last_source_index": int(horizon_last_index),
            "precompute_mode": (
                "bounded_async_next_source_cut"
                if self.continuous_waypoint_async_precompute_enabled
                else "synchronous_current_source_cut"
            ),
        }

    def _make_continuous_waypoint_build_request(
        self,
        *,
        interval_index: int,
        horizon_last_index: int,
        interval_timestamp: float,
        initial_state: JointHorizonState,
    ) -> tuple[_ContinuousWaypointBuildRequest | None, dict[str, Any]]:
        base_metadata: dict[str, Any] = {
            **self._continuous_waypoint_base_metadata(
                interval_index=interval_index,
                horizon_last_index=horizon_last_index,
            )
        }
        if not self.continuous_waypoint_shadow_enabled:
            return None, {
                **base_metadata,
                "status": "disabled",
            }

        future_indices = [
            index
            for index in range(int(interval_index) + 1, int(horizon_last_index) + 1)
            if index in self._source_qpos and index in self._source_pose
        ]
        if len(future_indices) < 2:
            return None, {
                **base_metadata,
                "status": "insufficient_lookahead",
                "required_future_source_steps": 2,
                "available_future_source_steps": len(future_indices),
            }

        timestamps = np.asarray(
            [float(interval_timestamp)]
            + [float(self._source_pose[index].timestamp) for index in future_indices],
            dtype=float,
        )
        dual_waypoints = np.asarray(
            [np.asarray(initial_state.q_rad, dtype=float)]
            + [
                np.concatenate(
                    (
                        self._source_qpos[index]["left"],
                        self._source_qpos[index]["right"],
                    )
                )
                for index in future_indices
            ],
            dtype=float,
        )
        if np.any(np.diff(timestamps) <= 1e-12):
            return None, {
                **base_metadata,
                "status": "invalid_staged_timing",
                "source_timestamps": timestamps.tolist(),
            }

        # A non-zero terminal tangent is essential for sustained motion.  Use
        # only the two already-available final source steps, suppress a true
        # reversal, and cap the observational tangent by the formal qd limit.
        segment_velocity = np.diff(dual_waypoints, axis=0) / np.diff(
            timestamps
        ).reshape(-1, 1)
        terminal_velocity = segment_velocity[-1].copy()
        if segment_velocity.shape[0] >= 2:
            direction_consistent = (
                segment_velocity[-2] * segment_velocity[-1] > 0.0
            )
            terminal_velocity[~direction_consistent] = 0.0
        terminal_velocity = np.clip(
            terminal_velocity,
            -self._dual_limits.max_velocity_rad_s,
            self._dual_limits.max_velocity_rad_s,
        )
        sample_count = (
            (int(horizon_last_index) - int(interval_index))
            * self.commands_per_source
            + 1
        )
        sample_timestamps = float(interval_timestamp) + np.arange(
            sample_count,
            dtype=float,
        ) / self.command_hz
        digest = hashlib.sha256()
        digest.update(str(int(self._continuous_precompute_epoch)).encode("ascii"))
        digest.update(str(int(interval_index)).encode("ascii"))
        digest.update(str(int(horizon_last_index)).encode("ascii"))
        for array in (
            timestamps,
            dual_waypoints,
            np.asarray(initial_state.qd_rad_s, dtype=float),
            np.asarray(initial_state.qdd_rad_s2, dtype=float),
            terminal_velocity,
            sample_timestamps,
        ):
            contiguous = np.ascontiguousarray(array, dtype=np.float64)
            digest.update(str(contiguous.shape).encode("ascii"))
            digest.update(contiguous.tobytes())
        request = _ContinuousWaypointBuildRequest(
            key=digest.hexdigest(),
            epoch=int(self._continuous_precompute_epoch),
            interval_index=int(interval_index),
            horizon_last_index=int(horizon_last_index),
            interval_timestamp=float(interval_timestamp),
            source_indices=tuple([int(interval_index), *future_indices]),
            timestamps=timestamps.copy(),
            dual_waypoints=dual_waypoints.copy(),
            initial_velocity_rad_s=np.asarray(
                initial_state.qd_rad_s,
                dtype=float,
            ).copy(),
            initial_acceleration_rad_s2=np.asarray(
                initial_state.qdd_rad_s2,
                dtype=float,
            ).copy(),
            terminal_velocity_rad_s=terminal_velocity.copy(),
            sample_timestamps=sample_timestamps.copy(),
            submitted_monotonic=time.monotonic(),
        )
        return request, base_metadata

    def _build_continuous_waypoint_request(
        self,
        request: _ContinuousWaypointBuildRequest,
    ) -> _ContinuousWaypointBuildResult:
        proposal_planner = (
            self._continuous_waypoint_precompute_collision_planner
            if self.continuous_waypoint_async_precompute_enabled
            else None
        )
        return _build_continuous_waypoint_request_impl(
            request,
            command_hz=self.command_hz,
            dual_limits=self._dual_limits,
            commands_per_source=self.commands_per_source,
            collision_planner=proposal_planner,
        )

    def _install_continuous_waypoint_build_result(
        self,
        result: _ContinuousWaypointBuildResult,
        *,
        formal_plan: Any,
        base_metadata: Mapping[str, Any],
        consumed_async: bool,
    ) -> None:
        request = result.request
        self._cached_continuous_waypoint_shadow = result.shadow
        if consumed_async:
            self._cached_continuous_waypoint_admitted = bool(result.pre_admitted)
            self._cached_continuous_waypoint_discrete_qd = (
                None
                if result.discrete_qd_rad_s is None
                else np.asarray(result.discrete_qd_rad_s, dtype=float).copy()
            )
            self._cached_continuous_waypoint_discrete_qdd = (
                None
                if result.discrete_qdd_rad_s2 is None
                else np.asarray(result.discrete_qdd_rad_s2, dtype=float).copy()
            )
        common = {
            **dict(base_metadata),
            "status": result.status,
            "required_future_source_steps": 2,
            "available_future_source_steps": len(request.source_indices) - 1,
            "source_indices": list(request.source_indices),
            "source_timestamps": request.timestamps.tolist(),
            "compute_ms": float(result.compute_ms),
            "precompute_request_key": request.key,
            "precompute_epoch": int(request.epoch),
            "precompute_consumed_async": bool(consumed_async),
            "precompute_age_ms": (
                max(0.0, time.monotonic() - request.submitted_monotonic) * 1000.0
            ),
            "error_type": result.error_type,
            "error": result.error,
            "precompute_candidate_certified": bool(result.pre_admitted),
            "precompute_collision_check_isolated": bool(consumed_async),
            **dict(result.admission_metadata or {}),
        }
        shadow = result.shadow
        if shadow is None or result.status != "ready":
            self._cached_continuous_waypoint_shadow_metadata = common
            return

        formal_q = np.asarray(formal_plan.q_path_rad, dtype=float)
        common_count = min(formal_q.shape[0], shadow.q_rad.shape[0])
        position_delta = shadow.q_rad[:common_count] - formal_q[:common_count]
        formal_jerk = np.diff(
            np.asarray(formal_plan.qdd_path_rad_s2, dtype=float),
            axis=0,
        ) / float(formal_plan.dt_sec)
        self._cached_continuous_waypoint_shadow_metadata = {
            **common,
            "planner": shadow.metadata.get("planner"),
            "continuous_hard_limit_certificate": shadow.metadata.get(
                "continuous_hard_limit_certificate"
            ),
            "terminal_velocity_rad_s": request.terminal_velocity_rad_s.tolist(),
            "max_abs_path_delta_deg": float(
                np.max(np.abs(np.rad2deg(position_delta)))
            ),
            "rms_path_delta_deg": float(
                np.sqrt(np.mean(np.square(np.rad2deg(position_delta))))
            ),
            "formal_max_abs_sampled_jerk_deg_s3": float(
                np.max(np.abs(np.rad2deg(formal_jerk)))
            ),
            "shadow_max_abs_sampled_jerk_deg_s3": float(
                np.max(np.abs(np.rad2deg(shadow.jerk_rad_s3)))
            ),
        }

    def _rebuild_continuous_waypoint_shadow(
        self,
        *,
        interval_index: int,
        horizon_last_index: int,
        interval_timestamp: float,
        initial_state: JointHorizonState,
        formal_plan: Any,
    ) -> None:
        """Synchronously build the original debug/admission candidate path."""

        self._cached_continuous_waypoint_shadow = None
        request, metadata = self._make_continuous_waypoint_build_request(
            interval_index=interval_index,
            horizon_last_index=horizon_last_index,
            interval_timestamp=interval_timestamp,
            initial_state=initial_state,
        )
        if request is None:
            self._cached_continuous_waypoint_shadow_metadata = metadata
            return
        result = self._build_continuous_waypoint_request(request)
        self._install_continuous_waypoint_build_result(
            result,
            formal_plan=formal_plan,
            base_metadata=metadata,
            consumed_async=False,
        )

    def _continuous_waypoint_precompute_loop(self) -> None:
        try:
            receipt = bind_current_thread_to_cpu(
                self.continuous_waypoint_worker_cpu,
                role="continuous_c2_precompute_worker",
            )
        except Exception as exc:  # noqa: BLE001 - affinity is fail-closed.
            with self._continuous_precompute_condition:
                self._continuous_precompute_affinity_error = (
                    f"{type(exc).__name__}: {exc}"
                )
                self._continuous_precompute_affinity_ready.set()
                self._continuous_precompute_condition.notify_all()
            return
        with self._continuous_precompute_condition:
            self._continuous_precompute_affinity_receipt = dict(receipt)
            self._continuous_precompute_affinity_ready.set()
            self._continuous_precompute_condition.notify_all()
        while True:
            with self._continuous_precompute_condition:
                while (
                    self._continuous_precompute_pending is None
                    and not self._continuous_precompute_stop
                ):
                    self._continuous_precompute_condition.wait()
                if self._continuous_precompute_stop:
                    return
                request = self._continuous_precompute_pending
                self._continuous_precompute_pending = None
                assert request is not None
                self._continuous_precompute_running_key = request.key
            result = self._build_continuous_waypoint_request(request)
            with self._continuous_precompute_condition:
                self._continuous_precompute_running_key = None
                if (
                    not self._continuous_precompute_stop
                    and request.epoch == self._continuous_precompute_epoch
                ):
                    self._continuous_precompute_results[request.key] = result
                    self._continuous_precompute_results.move_to_end(request.key)
                    while len(self._continuous_precompute_results) > 2:
                        self._continuous_precompute_results.popitem(last=False)
                    self._continuous_precompute_completed_count += 1
                self._continuous_precompute_condition.notify_all()

    def _submit_continuous_waypoint_precompute(
        self,
        request: _ContinuousWaypointBuildRequest,
    ) -> None:
        if (
            self.continuous_waypoint_worker_isolation
            == CONTINUOUS_WAYPOINT_WORKER_PROCESS
        ):
            self._drain_continuous_waypoint_process_results()
            process = self._continuous_precompute_process
            request_queue = self._continuous_precompute_process_request_queue
            if (
                self._continuous_precompute_stop
                or process is None
                or request_queue is None
            ):
                return
            if not process.is_alive():
                self._continuous_precompute_process_error = (
                    "RuntimeError: continuous waypoint proposal process exited"
                )
                return
            if (
                request.key == self._continuous_precompute_process_pending_key
                or request.key in self._continuous_precompute_results
            ):
                return
            try:
                request_queue.put_nowait({"kind": "build", "request": request})
            except queue.Full:
                # Proposal preparation must never wait on the authoritative
                # 60 Hz reference thread.  The next tick can submit a fresher
                # anchor while this cut uses the formal governor fallback.
                self._continuous_precompute_superseded_count += 1
                return
            self._continuous_precompute_process_pending_key = request.key
            self._continuous_precompute_submitted_count += 1
            return

        with self._continuous_precompute_condition:
            if (
                request.key == self._continuous_precompute_running_key
                or request.key in self._continuous_precompute_results
                or (
                    self._continuous_precompute_pending is not None
                    and request.key == self._continuous_precompute_pending.key
                )
            ):
                return
            if self._continuous_precompute_pending is not None:
                self._continuous_precompute_superseded_count += 1
            self._continuous_precompute_pending = request
            self._continuous_precompute_submitted_count += 1
            self._continuous_precompute_condition.notify_all()

    def _try_submit_next_continuous_waypoint_precompute(self) -> None:
        next_anchor = self._continuous_precompute_next_anchor
        if next_anchor is None or self._continuous_precompute_stop:
            return
        interval_index, initial_state = next_anchor
        horizon_last_index = interval_index + self.lookahead_source_steps
        if any(
            index not in self._source_pose
            for index in range(interval_index, horizon_last_index + 1)
        ):
            return
        try:
            self._select_source_qpos_through(horizon_last_index)
            request, _metadata = self._make_continuous_waypoint_build_request(
                interval_index=interval_index,
                horizon_last_index=horizon_last_index,
                interval_timestamp=self._source_pose[interval_index].timestamp,
                initial_state=initial_state,
            )
        except Exception:
            # This is speculative preparation only.  The canonical source
            # selection repeats fail-closed when the cut actually becomes due.
            return
        if request is not None:
            self._submit_continuous_waypoint_precompute(request)

    def _consume_continuous_waypoint_precompute(
        self,
        *,
        interval_index: int,
        horizon_last_index: int,
        interval_timestamp: float,
        initial_state: JointHorizonState,
        formal_plan: Any,
    ) -> None:
        self._cached_continuous_waypoint_shadow = None
        self._cached_continuous_waypoint_admitted = False
        self._cached_continuous_waypoint_discrete_qd = None
        self._cached_continuous_waypoint_discrete_qdd = None
        request, metadata = self._make_continuous_waypoint_build_request(
            interval_index=interval_index,
            horizon_last_index=horizon_last_index,
            interval_timestamp=interval_timestamp,
            initial_state=initial_state,
        )
        if request is None:
            self._cached_continuous_waypoint_shadow_metadata = metadata
            return
        if (
            self.continuous_waypoint_worker_isolation
            == CONTINUOUS_WAYPOINT_WORKER_PROCESS
        ):
            self._drain_continuous_waypoint_process_results()
            result = self._continuous_precompute_results.pop(request.key, None)
            pending_key = self._continuous_precompute_process_pending_key
            running_key = None
        else:
            with self._continuous_precompute_condition:
                result = self._continuous_precompute_results.pop(request.key, None)
                pending_key = (
                    None
                    if self._continuous_precompute_pending is None
                    else self._continuous_precompute_pending.key
                )
                running_key = self._continuous_precompute_running_key
        if result is None:
            self._continuous_precompute_miss_count += 1
            reason = (
                "process_error"
                if self._continuous_precompute_process_error is not None
                else "process_pending_or_running"
                if (
                    self.continuous_waypoint_worker_isolation
                    == CONTINUOUS_WAYPOINT_WORKER_PROCESS
                    and request.key == pending_key
                )
                else "running"
                if request.key == running_key
                else "pending"
                if request.key == pending_key
                else "not_submitted_or_stale_anchor"
            )
            self._cached_continuous_waypoint_shadow_metadata = {
                **metadata,
                "status": "precompute_not_ready",
                "precompute_request_key": request.key,
                "precompute_miss_reason": reason,
                "precompute_submitted_count": (
                    self._continuous_precompute_submitted_count
                ),
                "precompute_completed_count": (
                    self._continuous_precompute_completed_count
                ),
                "precompute_miss_count": self._continuous_precompute_miss_count,
                "precompute_superseded_count": (
                    self._continuous_precompute_superseded_count
                ),
                "precompute_worker_isolation": (
                    self.continuous_waypoint_worker_isolation
                ),
                "precompute_process_pid": self._continuous_precompute_process_pid,
                "precompute_process_error": (
                    self._continuous_precompute_process_error
                ),
            }
            return
        self._install_continuous_waypoint_build_result(
            result,
            formal_plan=formal_plan,
            base_metadata=metadata,
            consumed_async=True,
        )

    def _certify_continuous_waypoint_cut(
        self,
        shadow: OfflineJointWaypointTrajectory,
        *,
        initial_state: JointHorizonState,
        collision_planner: RM65PositionPlannerProtocol,
    ) -> tuple[bool, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
        """Certify one complete cut without mutating execution state."""

        required_last_row = int(self.commands_per_source)
        if int(shadow.q_rad.shape[0]) <= required_last_row:
            return False, None, None, {
                "admission_status": "formal_fallback_insufficient_cut_coverage",
                "authoritative": False,
                "dispatch_selected": False,
                "formal_qcmd_unchanged": True,
                "required_last_row": required_last_row,
                "available_last_row": int(shadow.q_rad.shape[0] - 1),
            }

        # Position control ultimately dispatches q samples, not the analytic
        # derivative fields of the quintic.  Certify the finite-difference
        # state seen by the 120 Hz command stream so a later fallback to the
        # semi-implicit formal governor cannot introduce an unmodelled state
        # discontinuity at a source-cut boundary.
        dt = 1.0 / self.command_hz
        discrete_qd = np.zeros_like(shadow.q_rad)
        discrete_qdd = np.zeros_like(shadow.q_rad)
        discrete_qd[0] = np.asarray(initial_state.qd_rad_s, dtype=float)
        discrete_qdd[0] = np.asarray(initial_state.qdd_rad_s2, dtype=float)
        tolerance = 1e-7
        for row in range(1, required_last_row + 1):
            discrete_qd[row] = (
                np.asarray(shadow.q_rad[row], dtype=float)
                - np.asarray(shadow.q_rad[row - 1], dtype=float)
            ) / dt
            discrete_qdd[row] = (
                discrete_qd[row] - discrete_qd[row - 1]
            ) / dt
            discrete_jerk = (
                discrete_qdd[row] - discrete_qdd[row - 1]
            ) / dt
            violations = (
                (
                    "velocity",
                    np.abs(discrete_qd[row]),
                    self._dual_limits.max_velocity_rad_s,
                ),
                (
                    "acceleration",
                    np.abs(discrete_qdd[row]),
                    self._dual_limits.max_acceleration_rad_s2,
                ),
                (
                    "jerk",
                    np.abs(discrete_jerk),
                    self._dual_limits.max_jerk_rad_s3,
                ),
            )
            for quantity, observed, limit in violations:
                failing = np.flatnonzero(observed > limit + tolerance)
                if failing.size:
                    joint = int(failing[0])
                    return False, None, None, {
                        "admission_status": (
                            "formal_fallback_discrete_dynamic_rejected"
                        ),
                        "authoritative": False,
                        "dispatch_selected": False,
                        "formal_qcmd_unchanged": True,
                        "discrete_command_certificate_passed": False,
                        "discrete_rejected_row": row,
                        "discrete_rejected_joint_index": joint + 1,
                        "discrete_rejected_quantity": quantity,
                        "discrete_rejected_observed": float(observed[joint]),
                        "discrete_rejected_limit": float(limit[joint]),
                    }

        previous = np.asarray(initial_state.q_rad, dtype=float).reshape(12)
        min_distance_m = math.inf
        checked_count = 0
        for row in range(1, required_last_row + 1):
            candidate = np.asarray(shadow.q_rad[row], dtype=float).reshape(12)
            checks = ((1.0, candidate),) + tuple(
                (
                    alpha,
                    (1.0 - alpha) * previous + alpha * candidate,
                )
                for alpha in (0.25, 0.5, 0.75)
            )
            for alpha, sample in checks:
                checked_count += 1
                validation = collision_planner.validate_qpos_reference(
                    {
                        "left": sample[:6],
                        "right": sample[6:],
                    },
                    source=(
                        "deploy_continuous_waypoint_admission_endpoint"
                        if alpha == 1.0
                        else "deploy_continuous_waypoint_admission_interval_sample"
                    ),
                )
                if validation.collision_distance_m is not None:
                    min_distance_m = min(
                        min_distance_m,
                        float(validation.collision_distance_m),
                    )
                if not validation.safe:
                    return False, None, None, {
                        "admission_status": "formal_fallback_collision_rejected",
                        "authoritative": False,
                        "dispatch_selected": False,
                        "formal_qcmd_unchanged": True,
                        "collision_checked": True,
                        "collision_admitted": False,
                        "collision_check_count": checked_count,
                        "collision_rejected_row": row,
                        "collision_rejected_alpha": alpha,
                        "collision_rejected_reason": validation.reason,
                        "collision_min_distance_m": (
                            None
                            if not math.isfinite(min_distance_m)
                            else min_distance_m
                        ),
                    }
            previous = candidate

        return True, discrete_qd, discrete_qdd, {
            "admission_status": "admitted_for_complete_source_cut",
            "authoritative": True,
            "dispatch_selected": True,
            "formal_qcmd_unchanged": False,
            "collision_checked": True,
            "collision_admitted": True,
            "collision_check_count": checked_count,
            "collision_min_distance_m": (
                None if not math.isfinite(min_distance_m) else min_distance_m
            ),
            "committed_cut_rows": required_last_row,
            "discrete_command_certificate_passed": True,
            "discrete_command_certificate_scope": "complete_source_cut",
            "selection_scope": "complete_source_cut",
            "selection_switching": "forbidden_within_source_cut",
        }

    def _pre_admit_continuous_waypoint_cut(
        self,
        *,
        initial_state: JointHorizonState,
    ) -> None:
        """Select the experimental C2 path once for a whole source cut."""

        metadata = dict(self._cached_continuous_waypoint_shadow_metadata)
        if not self.continuous_waypoint_admission_enabled:
            self._cached_continuous_waypoint_admitted = False
            self._cached_continuous_waypoint_discrete_qd = None
            self._cached_continuous_waypoint_discrete_qdd = None
            self._cached_continuous_waypoint_shadow_metadata = {
                **metadata,
                "admission_status": "disabled",
            }
            return

        if self.continuous_waypoint_async_precompute_enabled:
            # The bounded worker used an isolated, configuration-identical
            # proposal validator.  The canonical planner still revalidates the
            # emitted endpoint and three interior samples on every 120 Hz tick.
            if bool(metadata.get("precompute_candidate_certified", False)):
                self._cached_continuous_waypoint_shadow_metadata = {
                    **metadata,
                    "authoritative": True,
                    "dispatch_selected": True,
                    "formal_qcmd_unchanged": False,
                    "collision_authority": (
                        "isolated_precheck_plus_canonical_per_tick_revalidation"
                    ),
                    "whole_cut_collision_precheck_authoritative": False,
                    "canonical_runtime_revalidation_required": True,
                }
                return
            self._cached_continuous_waypoint_admitted = False
            self._cached_continuous_waypoint_discrete_qd = None
            self._cached_continuous_waypoint_discrete_qdd = None
            self._cached_continuous_waypoint_shadow_metadata = {
                **metadata,
                "admission_status": metadata.get(
                    "admission_status",
                    "formal_fallback_candidate_unavailable",
                ),
                "authoritative": False,
                "dispatch_selected": False,
                "formal_qcmd_unchanged": True,
            }
            return

        self._cached_continuous_waypoint_admitted = False
        self._cached_continuous_waypoint_discrete_qd = None
        self._cached_continuous_waypoint_discrete_qdd = None
        shadow = self._cached_continuous_waypoint_shadow
        if shadow is None or metadata.get("status") != "ready":
            self._cached_continuous_waypoint_shadow_metadata = {
                **metadata,
                "admission_status": "formal_fallback_candidate_unavailable",
                "authoritative": False,
                "dispatch_selected": False,
                "formal_qcmd_unchanged": True,
            }
            return
        admitted, discrete_qd, discrete_qdd, admission_metadata = (
            self._certify_continuous_waypoint_cut(
                shadow,
                initial_state=initial_state,
                collision_planner=self._collision_planner,
            )
        )
        self._cached_continuous_waypoint_admitted = bool(admitted)
        self._cached_continuous_waypoint_discrete_qd = discrete_qd
        self._cached_continuous_waypoint_discrete_qdd = discrete_qdd
        self._cached_continuous_waypoint_shadow_metadata = {
            **metadata,
            **admission_metadata,
        }

    def _continuous_waypoint_shadow_at(
        self,
        *,
        plan_row: int,
        formal_qcmd: Mapping[str, np.ndarray],
        source_cut_replanned: bool,
    ) -> dict[str, Any]:
        metadata = {
            **self._cached_continuous_waypoint_shadow_metadata,
            "source_cut_replanned": bool(source_cut_replanned),
            "plan_row": int(plan_row),
        }
        shadow = self._cached_continuous_waypoint_shadow
        if shadow is None or metadata.get("status") != "ready":
            return metadata
        row = min(int(plan_row), int(shadow.q_rad.shape[0] - 1))
        candidate = np.asarray(shadow.q_rad[row], dtype=float)
        discrete_state = bool(
            self._cached_continuous_waypoint_admitted
            and self._cached_continuous_waypoint_discrete_qd is not None
            and self._cached_continuous_waypoint_discrete_qdd is not None
        )
        candidate_qd = (
            np.asarray(self._cached_continuous_waypoint_discrete_qd[row], dtype=float)
            if discrete_state
            else np.asarray(shadow.qd_rad_s[row], dtype=float)
        )
        candidate_qdd = (
            np.asarray(self._cached_continuous_waypoint_discrete_qdd[row], dtype=float)
            if discrete_state
            else np.asarray(shadow.qdd_rad_s2[row], dtype=float)
        )
        if discrete_state and row > 0:
            candidate_jerk = (
                candidate_qdd
                - np.asarray(
                    self._cached_continuous_waypoint_discrete_qdd[row - 1],
                    dtype=float,
                )
            ) * self.command_hz
        else:
            candidate_jerk = np.asarray(shadow.jerk_rad_s3[row], dtype=float)
        formal = np.concatenate(
            (
                np.asarray(formal_qcmd["left"], dtype=float),
                np.asarray(formal_qcmd["right"], dtype=float),
            )
        )
        return {
            **metadata,
            "plan_row": row,
            "candidate_qpos_by_side": {
                "left": candidate[:6].tolist(),
                "right": candidate[6:].tolist(),
            },
            "candidate_qd_rad_s_by_side": {
                "left": candidate_qd[:6].tolist(),
                "right": candidate_qd[6:].tolist(),
            },
            "candidate_qdd_rad_s2_by_side": {
                "left": candidate_qdd[:6].tolist(),
                "right": candidate_qdd[6:].tolist(),
            },
            "candidate_jerk_rad_s3_by_side": {
                "left": candidate_jerk[:6].tolist(),
                "right": candidate_jerk[6:].tolist(),
            },
            "candidate_state_semantics": (
                "emitted_position_finite_difference"
                if discrete_state
                else "continuous_quintic_analytic_derivative"
            ),
            "max_abs_candidate_minus_formal_deg": float(
                np.max(np.abs(np.rad2deg(candidate - formal)))
            ),
        }

    def _initial_state_for_source_cut(
        self,
        *,
        interval_index: int,
        committed_state: JointHorizonState,
    ) -> tuple[JointHorizonState, dict[str, Any]]:
        decision = self._plan_splice_by_source.get(int(interval_index))
        if decision is None:
            return committed_state, {
                "position_plan_splice_enabled": False,
                "position_plan_splice_mode": "legacy_unspecified",
            }
        state = initial_state_for_position_plan_splice(
            decision,
            committed_state=committed_state,
        )
        return state, {
            "position_plan_splice_enabled": True,
            "position_plan_splice_mode": decision.mode,
            "position_plan_splice_preserve_committed_dynamics": bool(
                decision.preserve_committed_dynamics
            ),
            "position_plan_splice_require_zero_dynamic_restart": bool(
                decision.require_zero_dynamic_restart
            ),
            "position_plan_splice_mutable_future_replacement": bool(
                decision.mutable_future_replacement
            ),
            "position_plan_splice_cut_id": decision.cut.cut_id,
        }

    def _solve_formal_horizon(
        self,
        reference: np.ndarray,
        *,
        initial_state: JointHorizonState,
        commit_count: int,
        plan_splice_decision: PositionPlanSpliceDecision | None = None,
    ) -> JointHorizonPlan:
        """Use the reusable workspace, with one narrow certified fallback.

        The non-batched solver implements the same q/qd/qdd/jerk contract and
        was the original full-dataset validation path.  It is used only when
        OSQP returns a batched candidate that subsequently fails the hard
        certificate.  Solver failures, infeasibility and all other runtime
        errors remain fail-closed.
        """

        splice_weight = initial_jerk_regularization_for_position_plan_splice(
            plan_splice_decision,
            config=self.plan_splice_dynamics_config,
        )
        splice_delta_weight = jerk_delta_regularization_for_position_plan_splice(
            plan_splice_decision,
            config=self.plan_splice_dynamics_config,
        )
        splice_failure: RuntimeError | None = None
        if splice_weight > 0.0 or splice_delta_weight > 0.0:
            workspace = self._hold_restart_governor_workspace
            if workspace is None:
                raise RuntimeError(
                    "hold-restart jerk workspace is missing for a nonzero "
                    "splice regularization weight."
                )
            try:
                candidate = workspace.solve(
                    reference,
                    initial_state=initial_state,
                    commit_count=int(commit_count),
                    previous_jerk_rad_s3=np.zeros(
                        self._dual_limits.dimension,
                        dtype=float,
                    ),
                )
                return JointHorizonPlan(
                    q_path_rad=candidate.q_path_rad,
                    qd_path_rad_s=candidate.qd_path_rad_s,
                    qdd_path_rad_s2=candidate.qdd_path_rad_s2,
                    commit_count=candidate.commit_count,
                    dt_sec=candidate.dt_sec,
                    metadata={
                        **dict(candidate.metadata),
                        "plan_splice_initial_jerk_regularization_applied": True,
                        "plan_splice_initial_jerk_regularization_weight": (
                            splice_weight
                        ),
                        "plan_splice_jerk_delta_regularization_weight": (
                            splice_delta_weight
                        ),
                        "plan_splice_initial_jerk_regularization_fallback": False,
                    },
                )
            except RuntimeError as exc:
                # This optional post-Hold candidate never weakens the existing
                # hard constraints.  Failure atomically selects the exact
                # baseline workspace for the whole cut.
                splice_failure = exc

        try:
            baseline = self._governor_workspace.solve(
                reference,
                initial_state=initial_state,
                commit_count=int(commit_count),
            )
            if splice_failure is None:
                return baseline
            return JointHorizonPlan(
                q_path_rad=baseline.q_path_rad,
                qd_path_rad_s=baseline.qd_path_rad_s,
                qdd_path_rad_s2=baseline.qdd_path_rad_s2,
                commit_count=baseline.commit_count,
                dt_sec=baseline.dt_sec,
                metadata={
                    **dict(baseline.metadata),
                    "plan_splice_initial_jerk_regularization_applied": False,
                    "plan_splice_initial_jerk_regularization_weight": (
                        splice_weight
                    ),
                    "plan_splice_jerk_delta_regularization_weight": (
                        splice_delta_weight
                    ),
                    "plan_splice_initial_jerk_regularization_fallback": True,
                    "plan_splice_initial_jerk_regularization_failure_type": (
                        type(splice_failure).__name__
                    ),
                    "plan_splice_initial_jerk_regularization_failure": str(
                        splice_failure
                    ),
                },
            )
        except RuntimeError as exc:
            message = str(exc)
            if not message.startswith(
                "reused batched joint horizon candidate failed certification"
            ):
                raise
            started = time.perf_counter()
            fallback = plan_jerk_limited_joint_horizon(
                reference,
                initial_state=initial_state,
                limits=self._dual_limits,
                dt_sec=1.0 / self.command_hz,
                commit_count=int(commit_count),
                solver_tolerance=1e-7,
            )
            return JointHorizonPlan(
                q_path_rad=fallback.q_path_rad,
                qd_path_rad_s=fallback.qd_path_rad_s,
                qdd_path_rad_s2=fallback.qdd_path_rad_s2,
                commit_count=fallback.commit_count,
                dt_sec=fallback.dt_sec,
                metadata={
                    **dict(fallback.metadata),
                    "batched_workspace_fallback": True,
                    "batched_workspace_failure_type": type(exc).__name__,
                    "batched_workspace_failure": message,
                    "batched_workspace_fallback_solve_ms": (
                        time.perf_counter() - started
                    )
                    * 1000.0,
                },
            )

    def plan_pose_reference_horizon(
        self,
        pose_reference: ResampledMotionReference,
        *,
        source_timestamp: float,
        source: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> RM65PositionCollisionPlanResult:
        """Emit one 120 Hz control cut using all currently staged successors."""

        if self._state is None:
            raise RuntimeError("official all-IK governor was not reset.")
        started = time.perf_counter()
        elapsed_source_indices = [
            index
            for index, staged in self._source_pose.items()
            if staged.timestamp <= float(source_timestamp) + 1e-9
        ]
        interval_index = (
            int(pose_reference.left_source_index)
            if not elapsed_source_indices
            else max(elapsed_source_indices)
        )
        self._committed_source_index = max(
            interval_index,
            (
                -1
                if self._committed_source_index is None
                else self._committed_source_index
            ),
        )
        available = sorted(index for index in self._source_pose if index >= interval_index)
        desired_last_index = interval_index + self.lookahead_source_steps
        horizon_last_index = min(
            desired_last_index,
            max(available, default=interval_index),
        )
        selection_started = time.perf_counter()
        selection_metadata = self._select_source_qpos_through(horizon_last_index)
        selection_ms = (time.perf_counter() - selection_started) * 1000.0
        horizon_plan_started = time.perf_counter()
        dt = 1.0 / self.command_hz
        horizon_count = self.lookahead_source_steps * self.commands_per_source
        interval_timestamp = self._source_pose[interval_index].timestamp
        new_interval_plan = (
            self._prepared_initial_interval_pending
            or
            self._cached_interval_index != interval_index
            or self._cached_plan is None
            or self._cached_qref is None
        )
        if new_interval_plan:
            dual_state = JointHorizonState(
                q_rad=np.concatenate(
                    (
                        np.asarray(self._state["left"].q_rad, dtype=float),
                        np.asarray(self._state["right"].q_rad, dtype=float),
                    )
                ),
                qd_rad_s=np.concatenate(
                    (
                        np.asarray(self._state["left"].qd_rad_s, dtype=float),
                        np.asarray(self._state["right"].qd_rad_s, dtype=float),
                    )
                ),
                qdd_rad_s2=np.concatenate(
                    (
                        np.asarray(self._state["left"].qdd_rad_s2, dtype=float),
                        np.asarray(self._state["right"].qdd_rad_s2, dtype=float),
                    )
                ),
            )
            dual_state, plan_splice_metadata = (
                self._initial_state_for_source_cut(
                    interval_index=interval_index,
                    committed_state=dual_state,
                )
            )
            use_prepared_initial_plan = bool(
                self._prepared_initial_interval_pending
                and self._cached_interval_index == interval_index
                and self._cached_interval_timestamp is not None
                and math.isclose(
                    self._cached_interval_timestamp,
                    interval_timestamp,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and self._cached_plan is not None
                and self._cached_qref is not None
            )
            if use_prepared_initial_plan:
                plan = self._cached_plan
                qref = self._cached_qref
            else:
                qref = {
                    side: np.asarray(
                        [
                            self._qref_at(interval_timestamp + offset * dt)[side]
                            for offset in range(horizon_count + 1)
                        ],
                        dtype=float,
                    )
                    for side in ("left", "right")
                }
                dual_reference = np.concatenate(
                    (qref["left"], qref["right"]),
                    axis=1,
                )
                plan = self._solve_formal_horizon(
                    dual_reference,
                    initial_state=dual_state,
                    commit_count=min(
                        self.commands_per_source,
                        horizon_count,
                    ),
                    plan_splice_decision=self._plan_splice_by_source.get(
                        interval_index
                    ),
                )
            if self.continuous_waypoint_shadow_enabled:
                try:
                    if self.continuous_waypoint_async_precompute_enabled:
                        self._consume_continuous_waypoint_precompute(
                            interval_index=interval_index,
                            horizon_last_index=horizon_last_index,
                            interval_timestamp=interval_timestamp,
                            initial_state=dual_state,
                            formal_plan=plan,
                        )
                    else:
                        self._rebuild_continuous_waypoint_shadow(
                            interval_index=interval_index,
                            horizon_last_index=horizon_last_index,
                            interval_timestamp=interval_timestamp,
                            initial_state=dual_state,
                            formal_plan=plan,
                        )
                except Exception as exc:  # noqa: BLE001 - shadow is fail-open.
                    self._cached_continuous_waypoint_shadow = None
                    self._cached_continuous_waypoint_shadow_metadata = {
                        "enabled": True,
                        "status": "error",
                        "authoritative": False,
                        "dispatch_selected": False,
                        "formal_qcmd_unchanged": True,
                        "collision_checked": False,
                        "collision_admitted": False,
                        "anchor_source": "formal_governor_admitted_state",
                        "future_source_contract": "already_staged_same_chunk_only",
                        "interval_source_index": int(interval_index),
                        "horizon_last_source_index": int(horizon_last_index),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                self._pre_admit_continuous_waypoint_cut(
                    initial_state=dual_state,
                )
            self._cached_interval_index = interval_index
            self._cached_interval_timestamp = interval_timestamp
            self._cached_plan = plan
            self._cached_qref = qref
            self._prepared_initial_interval_pending = False
        else:
            plan = self._cached_plan
            qref = self._cached_qref
            plan_splice_metadata = {
                "position_plan_splice_enabled": bool(
                    interval_index in self._plan_splice_by_source
                ),
                "position_plan_splice_mode": (
                    "cached_source_cut"
                    if interval_index in self._plan_splice_by_source
                    else "legacy_unspecified"
                ),
            }
        assert plan is not None and qref is not None
        horizon_plan_ms = (
            time.perf_counter() - horizon_plan_started
        ) * 1000.0
        command_offset = int(
            np.clip(
                round((float(source_timestamp) - interval_timestamp) * self.command_hz),
                0,
                self.commands_per_source - 1,
            )
        )
        plan_row = min(
            command_offset + 1,
            int(plan.q_path_rad.shape[0] - 1),
        )
        continuous_waypoint_selected = bool(
            self.continuous_waypoint_admission_enabled
            and self._cached_continuous_waypoint_admitted
            and self._cached_continuous_waypoint_shadow is not None
        )
        execution_q = (
            self._cached_continuous_waypoint_shadow.q_rad
            if continuous_waypoint_selected
            else plan.q_path_rad
        )
        execution_qd = (
            self._cached_continuous_waypoint_discrete_qd
            if continuous_waypoint_selected
            else plan.qd_path_rad_s
        )
        execution_qdd = (
            self._cached_continuous_waypoint_discrete_qdd
            if continuous_waypoint_selected
            else plan.qdd_path_rad_s2
        )
        if continuous_waypoint_selected and (
            execution_qd is None or execution_qdd is None
        ):
            raise RuntimeError(
                "admitted continuous waypoint cut is missing its discrete "
                "command-state certificate."
            )
        if (
            new_interval_plan
            and self.continuous_waypoint_async_precompute_enabled
        ):
            next_anchor_row = min(
                int(self.commands_per_source),
                int(execution_q.shape[0] - 1),
            )
            next_anchor_state = JointHorizonState(
                q_rad=np.asarray(
                    execution_q[next_anchor_row],
                    dtype=float,
                ).copy(),
                qd_rad_s=np.asarray(
                    execution_qd[next_anchor_row],
                    dtype=float,
                ).copy(),
                qdd_rad_s2=np.asarray(
                    execution_qdd[next_anchor_row],
                    dtype=float,
                ).copy(),
            )
            self._continuous_precompute_next_anchor = (
                int(interval_index) + 1,
                next_anchor_state,
            )
        execution_row = min(plan_row, int(execution_q.shape[0] - 1))
        qcmd = {
            "left": np.asarray(execution_q[execution_row, :6], dtype=float).copy(),
            "right": np.asarray(execution_q[execution_row, 6:], dtype=float).copy(),
        }
        collision_validation_started = time.perf_counter()
        validation = self._collision_planner.validate_qpos_reference(
            qcmd,
            source="deploy_official_governor_endpoint",
        )
        if not validation.safe:
            raise RuntimeError(
                "official all-IK governor endpoint failed collision admission: "
                f"{validation.reason}"
            )
        previous = {
            side: np.asarray(self._state[side].q_rad, dtype=float)
            for side in ("left", "right")
        }
        for alpha in (0.25, 0.5, 0.75):
            sample = {
                side: (1.0 - alpha) * previous[side] + alpha * qcmd[side]
                for side in ("left", "right")
            }
            sampled = self._collision_planner.validate_qpos_reference(
                sample,
                source="deploy_official_governor_interval_sample",
            )
            if not sampled.safe:
                raise RuntimeError(
                    "official all-IK governor interval sample failed collision "
                    f"admission at alpha={alpha}: {sampled.reason}"
                )
        collision_validation_ms = (
            time.perf_counter() - collision_validation_started
        ) * 1000.0
        self._state = {
            side: JointHorizonState(
                q_rad=(
                    execution_q[execution_row, :6]
                    if side == "left"
                    else execution_q[execution_row, 6:]
                ),
                qd_rad_s=(
                    execution_qd[execution_row, :6]
                    if side == "left"
                    else execution_qd[execution_row, 6:]
                ),
                qdd_rad_s2=(
                    execution_qdd[execution_row, :6]
                    if side == "left"
                    else execution_qdd[execution_row, 6:]
                ),
            )
            for side in ("left", "right")
        }
        endpoint_fk_started = time.perf_counter()
        target_world = self._collision_planner.observe_endpoint_poses_world(
            {side: qref[side][plan_row] for side in ("left", "right")}
        )
        realized_world = self._collision_planner.observe_endpoint_poses_world(qcmd)
        endpoint_fk_ms = (time.perf_counter() - endpoint_fk_started) * 1000.0
        limited = any(
            not np.allclose(
                qcmd[side],
                qref[side][plan_row],
                rtol=0.0,
                atol=1e-8,
            )
            for side in ("left", "right")
        )
        sequence = self._sequence
        self._sequence += 1
        canonical_frame_ms = (time.perf_counter() - started) * 1000.0
        shadow_metadata_started = time.perf_counter()
        continuous_waypoint_shadow = (
            self._continuous_waypoint_shadow_at(
                plan_row=plan_row,
                formal_qcmd=qcmd,
                source_cut_replanned=new_interval_plan,
            )
            if self.continuous_waypoint_shadow_enabled
            else None
        )
        shadow_metadata_ms = (
            time.perf_counter() - shadow_metadata_started
        ) * 1000.0
        precompute_submit_ms = 0.0
        if self.continuous_waypoint_async_precompute_enabled:
            # Submit only after this tick's canonical collision/FK work.  The
            # worker may then use the inter-tick slack without competing with
            # the current command's authoritative hot path.
            precompute_submit_started = time.perf_counter()
            self._try_submit_next_continuous_waypoint_precompute()
            precompute_submit_ms = (
                time.perf_counter() - precompute_submit_started
            ) * 1000.0
        frame_total_ms = (time.perf_counter() - started) * 1000.0
        plan_metadata = {
            **dict(metadata or {}),
            **selection_metadata,
            **plan_splice_metadata,
            "planner": OFFICIAL_SOURCE_ALL_IK_GOVERNOR_BACKEND,
            "source_hz": self.source_hz,
            "command_hz": self.command_hz,
            "lookahead_source_steps_requested": self.lookahead_source_steps,
            "lookahead_source_steps_available": max(
                0,
                horizon_last_index - interval_index,
            ),
            "horizon_source_indices": [
                index
                for index in sorted(self._source_qpos)
                if int(pose_reference.left_source_index)
                <= index
                <= horizon_last_index
            ],
            "jerk_ramp_sec": self.jerk_ramp_sec,
            "source_cut_replanned": new_interval_plan,
            "source_cut_command_offset": command_offset,
            "governor_execution_mode": (
                "continuous_c2_waypoint"
                if continuous_waypoint_selected
                else "formal_third_order_horizon"
            ),
            "governor_execution_mode_legacy": (
                "experimental_continuous_c2_waypoint"
                if continuous_waypoint_selected
                else "formal_third_order_horizon"
            ),
            "continuous_waypoint_async_precompute_enabled": bool(
                self.continuous_waypoint_async_precompute_enabled
            ),
            "continuous_waypoint_worker_cpu": self.continuous_waypoint_worker_cpu,
            "continuous_waypoint_worker_isolation": (
                self.continuous_waypoint_worker_isolation
            ),
            "continuous_waypoint_process_pid": (
                self._continuous_precompute_process_pid
            ),
            "continuous_waypoint_process_error": (
                self._continuous_precompute_process_error
            ),
            "continuous_waypoint_worker_affinity_receipt": (
                None
                if self._continuous_precompute_affinity_receipt is None
                else dict(self._continuous_precompute_affinity_receipt)
            ),
            "continuous_waypoint_worker_affinity_error": (
                self._continuous_precompute_affinity_error
            ),
            "continuous_waypoint_precompute_submitted_count": int(
                self._continuous_precompute_submitted_count
            ),
            "continuous_waypoint_precompute_completed_count": int(
                self._continuous_precompute_completed_count
            ),
            "continuous_waypoint_precompute_miss_count": int(
                self._continuous_precompute_miss_count
            ),
            "continuous_waypoint_precompute_superseded_count": int(
                self._continuous_precompute_superseded_count
            ),
            "governor_solver": plan.metadata.get("planner"),
            "governor_solver_tolerance": plan.metadata.get(
                "solver_tolerance"
            ),
            "governor_batched_workspace_fallback": bool(
                plan.metadata.get("batched_workspace_fallback", False)
            ),
            "governor_batched_workspace_failure": plan.metadata.get(
                "batched_workspace_failure"
            ),
            "governor_batched_workspace_fallback_solve_ms": plan.metadata.get(
                "batched_workspace_fallback_solve_ms"
            ),
            "plan_splice_initial_jerk_regularization_applied": bool(
                plan.metadata.get(
                    "plan_splice_initial_jerk_regularization_applied",
                    False,
                )
            ),
            "plan_splice_initial_jerk_regularization_weight": plan.metadata.get(
                "plan_splice_initial_jerk_regularization_weight",
                0.0,
            ),
            "plan_splice_jerk_delta_regularization_weight": plan.metadata.get(
                "plan_splice_jerk_delta_regularization_weight",
                0.0,
            ),
            "plan_splice_initial_jerk_regularization_fallback": bool(
                plan.metadata.get(
                    "plan_splice_initial_jerk_regularization_fallback",
                    False,
                )
            ),
            "endpoint_collision_checked": True,
            "endpoint_collision_passed": True,
            "segment_collision_checked": True,
            "segment_collision_certified": False,
            "segment_collision_method": "three_interior_samples",
            "segment_collision_limitation": (
                "sampled_interval_not_continuous_swept_certificate"
            ),
            "collision_distance_m": validation.collision_distance_m,
            "collision_pair_count": self.collision_pair_count,
            "solve_ms": canonical_frame_ms,
            "frame_total_ms": frame_total_ms,
            "solve_time_budget_exceeded": (
                frame_total_ms > (1000.0 / self.command_hz)
            ),
            "selection_ms": float(selection_ms),
            "horizon_plan_ms": float(horizon_plan_ms),
            "collision_validation_ms": float(collision_validation_ms),
            "endpoint_fk_ms": float(endpoint_fk_ms),
            "continuous_waypoint_shadow_metadata_ms": float(
                shadow_metadata_ms
            ),
            "continuous_waypoint_precompute_submit_ms": float(
                precompute_submit_ms
            ),
            "temporal_smoothing": (
                "continuous_c2_quintic_waypoint"
                if continuous_waypoint_selected
                else "third_order_jerk_limited_horizon"
            ),
            **(
                {}
                if continuous_waypoint_shadow is None
                else {
                    "continuous_waypoint_shadow": continuous_waypoint_shadow,
                }
            ),
        }
        return RM65PositionCollisionPlanResult(
            sequence=sequence,
            source_timestamp=float(source_timestamp),
            source=str(source),
            status="limited" if limited else "tracking",
            valid=True,
            hold=False,
            active_sides=("left", "right"),
            q_ik_raw_by_side={
                side: qref[side][plan_row] for side in ("left", "right")
            },
            q_ref_by_side=qcmd,
            target_endpoint_pose_world_by_side=target_world,
            realized_endpoint_pose_world_by_side=realized_world,
            modification_reasons=(
                ("official_all_ik_source_lift",)
                + (
                    ("continuous_c2_waypoint",)
                    if continuous_waypoint_selected
                    else ()
                )
                + (
                    ("third_order_horizon_governor",)
                    if limited and not continuous_waypoint_selected
                    else ()
                )
            ),
            metadata=plan_metadata,
        )

    def plan_pose_reference(
        self,
        target_pose_by_side: Mapping[str, Any],
        **_: Any,
    ) -> RM65PositionCollisionPlanResult:
        raise RuntimeError(
            "official all-IK governor requires staged source lineage; "
            "call plan_pose_reference_horizon()."
        )

    def validate_qpos_reference(self, *args: Any, **kwargs: Any) -> Any:
        return self._collision_planner.validate_qpos_reference(*args, **kwargs)

    def admit_qpos_reference(self, *args: Any, **kwargs: Any) -> Any:
        return self._collision_planner.admit_qpos_reference(*args, **kwargs)


__all__ = [
    "CONTINUOUS_WAYPOINT_WORKER_ISOLATION_CHOICES",
    "CONTINUOUS_WAYPOINT_WORKER_PROCESS",
    "CONTINUOUS_WAYPOINT_WORKER_THREAD",
    "OFFICIAL_SOURCE_ALL_IK_GOVERNOR_BACKEND",
    "PINK_POSE_LIFT_BACKEND",
    "POSITION_PATH_LIFT_BACKEND_CHOICES",
    "RM65ContinuousWaypointProcessSpec",
    "RM65OfficialAllIKGovernorPlanner",
]

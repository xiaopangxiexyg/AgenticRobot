"""Fail-closed initial-alignment planning for real rule-based collection.

The rule plan names one immutable replay qpos target.  This module turns the
current causal arm state and that target into a bounded, coupled transition.
It deliberately reuses the already-constructed dual-arm collision solver and
its Active TCP envelope; it does not construct a second safety authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robot_motion.arm_init.constrained_transition import (
    ArmConstrainedTransitionConfig,
)
from robot_motion.rm65.constants import (
    RM65_ABSOLUTE_Q_MAX_RAD,
    RM65_ABSOLUTE_Q_MIN_RAD,
)
from robot_motion.rm65.defaults import (
    RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
    RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S,
)
from robot_motion.rm65.constrained_transition import (
    plan_dual_arm_constrained_qpos_transition,
)
from robot_motion.position_control.horizon_governor import JointHorizonLimits
from robot_motion.position_control.offline_waypoint_trajectory import (
    OFFLINE_WAYPOINT_TIMING_SAFE_RETIME_GLOBAL,
    build_offline_quintic_waypoint_trajectory,
)

from ..diagnostics import FaultEvent, FaultSeverity
from ..schemas import EffectorCommand, TeleopCommand


@dataclass(frozen=True)
class RuleBasedInitialAlignmentConfig:
    """Conservative first-real-run transition and certification limits."""

    waypoint_dt_sec: float = 0.05
    reference_hz: float = 60.0
    # Certified full-scale initialization rate: 1 degree per 50 ms waypoint
    # (20 deg/s).  The operator-facing speed_scale remains in (0, 1], so the
    # whole profile is still conveniently reducible without exceeding this
    # bounded coupled-path certificate.
    max_joint_step_deg: float = 1.0
    edge_check_resolution_deg: float = 0.25
    arrival_tolerance_deg: float = 0.15
    start_drift_tolerance_deg: float = 0.35
    # Intermediate certified qrefs are streamed at ``waypoint_dt_sec`` instead
    # of forcing the robot to stop and settle at every waypoint.  Bound how
    # far the current qref may lead causal measured state; when the arm falls
    # behind, the source holds the same qref until tracking recovers.
    streaming_max_lag_waypoints: int = 4
    max_steps: int = 4000
    planning_timeout_sec: float = 30.0
    collision_slow_margin_m: float = 0.02
    collision_hold_margin_m: float = 0.005
    # A value below one reduces the certified joint increment per control
    # cycle.  Values above one are deliberately rejected: the normal RM65
    # position/velocity limits remain the upper safety bound.
    speed_scale: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "waypoint_dt_sec",
            "reference_hz",
            "max_joint_step_deg",
            "edge_check_resolution_deg",
            "arrival_tolerance_deg",
            "start_drift_tolerance_deg",
            "planning_timeout_sec",
            "speed_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            if name == "speed_scale" and value > 1.0:
                raise ValueError("speed_scale must be in (0, 1].")
            object.__setattr__(self, name, value)
        if int(self.max_steps) < 1:
            raise ValueError("max_steps must be positive.")
        object.__setattr__(self, "max_steps", int(self.max_steps))
        if int(self.streaming_max_lag_waypoints) < 1:
            raise ValueError("streaming_max_lag_waypoints must be positive.")
        object.__setattr__(
            self,
            "streaming_max_lag_waypoints",
            int(self.streaming_max_lag_waypoints),
        )
        for name in ("collision_slow_margin_m", "collision_hold_margin_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, value)
        if self.collision_hold_margin_m > self.collision_slow_margin_m:
            raise ValueError(
                "collision_hold_margin_m must not exceed collision_slow_margin_m."
            )


@dataclass(frozen=True)
class RuleBasedInitialAlignmentSafetyCheck:
    safe: bool
    reason: str
    collision_distance_m: float | None
    required_collision_distance_m: float | None
    envelope_modified_sides: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuleBasedInitialAlignmentPlan:
    start_qpos_by_side: Mapping[str, np.ndarray]
    target_qpos_by_side: Mapping[str, np.ndarray]
    waypoints: tuple[Mapping[str, np.ndarray], ...]
    reference_timestamps_sec: tuple[float, ...]
    semantic_content_hash: str
    minimum_collision_distance_m: float
    dense_check_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "start_qpos_by_side",
            _qpos_pair(self.start_qpos_by_side, name="start_qpos_by_side"),
        )
        object.__setattr__(
            self,
            "target_qpos_by_side",
            _qpos_pair(self.target_qpos_by_side, name="target_qpos_by_side"),
        )
        object.__setattr__(
            self,
            "waypoints",
            tuple(_qpos_pair(value, name="waypoint") for value in self.waypoints),
        )
        timestamps = tuple(float(value) for value in self.reference_timestamps_sec)
        if len(timestamps) != len(self.waypoints) or not timestamps:
            raise ValueError(
                "reference_timestamps_sec must match the non-empty waypoint sequence."
            )
        if not np.all(np.isfinite(timestamps)) or any(
            right <= left for left, right in zip(timestamps, timestamps[1:])
        ):
            raise ValueError(
                "reference_timestamps_sec must be finite and strictly increasing."
            )
        if not math.isclose(timestamps[0], 0.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("reference_timestamps_sec must start at zero.")
        object.__setattr__(self, "reference_timestamps_sec", timestamps)


class RuleBasedInitialAlignmentPlanningError(RuntimeError):
    """A pre-dispatch transition could not be certified."""

    def __init__(
        self,
        message: str,
        *,
        status: str | None = None,
        blocked_reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.blocked_reason = blocked_reason


class _RuntimeTransitionSafetyModel:
    def __init__(self, retargeter: Any, target_filter: Any) -> None:
        self.retargeter = retargeter
        self.target_filter = target_filter
        self.last_details: dict[str, Any] = {}

    def fk_by_side(
        self,
        qpos_by_side: Mapping[str, np.ndarray],
    ) -> Mapping[str, np.ndarray]:
        qpos = _qpos_pair(qpos_by_side, name="qpos_by_side")
        return self.retargeter.tool_tcp_world_targets_for_qpos_pair(
            left_qpos=qpos["left"],
            right_qpos=qpos["right"],
        )

    def min_collision_distance(
        self,
        qpos_by_side: Mapping[str, np.ndarray],
    ) -> float | None:
        check = self.certify(qpos_by_side)
        self.last_details = dict(check.metadata)
        self.last_details.update(
            {
                "reason": check.reason,
                "envelope_modified_sides": list(check.envelope_modified_sides),
            }
        )
        if check.envelope_modified_sides:
            return -1.0
        return check.collision_distance_m

    def collision_distance_details(self) -> Mapping[str, Any]:
        return dict(self.last_details)

    def singularity_margin_by_side(
        self,
        qpos_by_side: Mapping[str, np.ndarray],
    ) -> Mapping[str, Any]:
        # This transition is executed directly in joint space; it does not
        # invert a Cartesian target, so an IK singularity margin is not an
        # applicable path constraint.
        del qpos_by_side
        return {}

    def certify(
        self,
        qpos_by_side: Mapping[str, np.ndarray],
    ) -> RuleBasedInitialAlignmentSafetyCheck:
        qpos = _qpos_pair(qpos_by_side, name="qpos_by_side")
        checker = getattr(self.retargeter, "check_qpos_pair_collision", None)
        if not callable(checker):
            return RuleBasedInitialAlignmentSafetyCheck(
                safe=False,
                reason="collision_checker_unavailable",
                collision_distance_m=None,
                required_collision_distance_m=None,
            )
        try:
            collision = checker(
                left_qpos=qpos["left"],
                right_qpos=qpos["right"],
            )
        except Exception as exc:  # noqa: BLE001 - safety is fail closed.
            return RuleBasedInitialAlignmentSafetyCheck(
                safe=False,
                reason=f"collision_check_exception:{type(exc).__name__}",
                collision_distance_m=None,
                required_collision_distance_m=None,
                metadata={"error": str(exc)},
            )
        distance = _optional_finite(getattr(collision, "min_distance_m", None))
        required = _optional_finite(
            getattr(collision, "required_min_distance_m", None)
        )
        collision_safe = bool(getattr(collision, "safe", False))
        collision_reason = str(
            getattr(collision, "reason", "collision_check_failed")
            or "collision_check_failed"
        )
        envelope_modified: tuple[str, ...] = ()
        envelope_error: str | None = None
        try:
            raw_targets = self.retargeter.solver_world_targets_for_qpos_pair(
                left_qpos=qpos["left"],
                right_qpos=qpos["right"],
            )
            effective_targets = self.target_filter(
                {
                    side: np.asarray(raw_targets[side], dtype=float).copy()
                    for side in ("left", "right")
                }
            )
            if effective_targets is None or set(effective_targets) != {"left", "right"}:
                raise RuntimeError("Active TCP envelope returned an incomplete target pair")
            envelope_modified = tuple(
                side
                for side in ("left", "right")
                if not np.allclose(
                    np.asarray(raw_targets[side], dtype=float),
                    np.asarray(effective_targets[side], dtype=float),
                    atol=1e-10,
                    rtol=0.0,
                )
            )
        except Exception as exc:  # noqa: BLE001 - safety is fail closed.
            envelope_error = f"{type(exc).__name__}: {exc}"

        safe = bool(collision_safe and not envelope_modified and envelope_error is None)
        reason = (
            "safe"
            if safe
            else "tcp_envelope_error"
            if envelope_error is not None
            else "tcp_envelope_projection_required"
            if envelope_modified
            else collision_reason
        )
        return RuleBasedInitialAlignmentSafetyCheck(
            safe=safe,
            reason=reason,
            collision_distance_m=distance,
            required_collision_distance_m=required,
            envelope_modified_sides=envelope_modified,
            metadata={
                "collision_pair_count": getattr(
                    collision, "collision_pair_count", None
                ),
                "collision_reason": collision_reason,
                "envelope_error": envelope_error,
            },
        )


class RuleBasedInitialAlignmentPlanner:
    """Plan and densely certify one runtime-current-to-replay transition."""

    def __init__(
        self,
        *,
        dual_arm_retargeter: Any,
        target_filter: Any,
        config: RuleBasedInitialAlignmentConfig | None = None,
    ) -> None:
        self.config = config or RuleBasedInitialAlignmentConfig()
        for name in (
            "check_qpos_pair_collision",
            "solver_world_targets_for_qpos_pair",
            "tool_tcp_world_targets_for_qpos_pair",
        ):
            if not callable(getattr(dual_arm_retargeter, name, None)):
                raise TypeError(f"dual-arm retargeter lacks required method {name}().")
        if not callable(target_filter) or not bool(
            getattr(target_filter, "apply_to_target", False)
        ):
            raise TypeError("initial alignment requires the callable Active TCP envelope.")
        self._model = _RuntimeTransitionSafetyModel(
            dual_arm_retargeter,
            target_filter,
        )

    @property
    def arrival_tolerance_rad(self) -> float:
        return float(np.deg2rad(self.config.arrival_tolerance_deg))

    @property
    def start_drift_tolerance_rad(self) -> float:
        return float(np.deg2rad(self.config.start_drift_tolerance_deg))

    @property
    def streaming_max_lag_rad(self) -> float:
        step_deg = float(self.config.max_joint_step_deg)
        lag_deg = max(
            2.0 * float(self.config.arrival_tolerance_deg),
            float(self.config.streaming_max_lag_waypoints) * step_deg,
        )
        return float(np.deg2rad(lag_deg))

    def certify_qpos_pair(
        self,
        qpos_by_side: Mapping[str, Any],
    ) -> RuleBasedInitialAlignmentSafetyCheck:
        return self._model.certify(_qpos_pair(qpos_by_side, name="qpos_by_side"))

    def plan(
        self,
        *,
        start_qpos_by_side: Mapping[str, Any],
        target_qpos_by_side: Mapping[str, Any],
    ) -> RuleBasedInitialAlignmentPlan:
        start = _qpos_pair(start_qpos_by_side, name="start_qpos_by_side")
        target = _qpos_pair(target_qpos_by_side, name="target_qpos_by_side")
        initial_check = self._model.certify(start)
        target_check = self._model.certify(target)
        for label, check in (("start", initial_check), ("target", target_check)):
            if not check.safe:
                raise RuleBasedInitialAlignmentPlanningError(
                    f"initial alignment {label} state failed safety certification: "
                    f"{check.reason}"
                )
        required_distance = max(
            value
            for value in (
                initial_check.required_collision_distance_m,
                target_check.required_collision_distance_m,
                0.0,
            )
            if value is not None
        )
        transition_config = ArmConstrainedTransitionConfig(
            dt_sec=float(self.config.waypoint_dt_sec),
            max_joint_step_deg=float(self.config.max_joint_step_deg),
            target_tolerance_rad=1e-6,
            max_steps=int(self.config.max_steps),
            max_joint_abs_deg=178.0,
            joint_lower_rad=tuple(float(value) for value in RM65_ABSOLUTE_Q_MIN_RAD),
            joint_upper_rad=tuple(float(value) for value in RM65_ABSOLUTE_Q_MAX_RAD),
            joint_slow_margin_deg=8.0,
            joint_hold_margin_deg=1.0,
            # The shared transition planner permits only a certified inward
            # escape from a joint-limit hold.  Outward, hard-limit, collision,
            # and TCP-envelope violations remain blocked.
            allow_joint_limit_escape=True,
            max_endpoint_excursion_m=0.5,
            collision_slow_distance_m=(
                required_distance + float(self.config.collision_slow_margin_m)
            ),
            collision_hold_distance_m=(
                required_distance + float(self.config.collision_hold_margin_m)
            ),
            collision_stop_distance_m=required_distance,
            require_kinematic_model=True,
            require_collision_check=True,
            force_slow_n=None,
            force_soft_n=None,
            force_hard_n=None,
            torque_slow_nm=None,
            torque_soft_nm=None,
            torque_hard_nm=None,
        )
        raw_plan = plan_dual_arm_constrained_qpos_transition(
            start_qpos_by_side=start,
            target_qpos_by_side=target,
            model=self._model,
            config=transition_config,
        )
        if not raw_plan.completed:
            held_joints = []
            held_collision = []
            assessment = raw_plan.metadata.get("blocked_current_assessment", {})
            for event in assessment.get("events", ()):
                if event.get("code") == "collision_soft":
                    distance = event.get("value")
                    threshold = event.get("threshold")
                    if distance is not None and threshold is not None:
                        held_collision.append(
                            f"distance_m={float(distance):.6f} "
                            f"hold_threshold_m={float(threshold):.6f}"
                        )
                    continue
                if event.get("code") != "joint_limit_hold_margin":
                    continue
                joint = event.get("metadata", {}).get("joint_index")
                side = event.get("side")
                if side not in target or not isinstance(joint, int) or not 1 <= joint <= 6:
                    continue
                held_joints.append(
                    f"{side}.j{joint} distance_to_limit_deg="
                    f"{math.degrees(float(event['value'])):.3f} "
                    f"current_deg={math.degrees(float(event['metadata']['qpos_rad'])):.3f} "
                    f"target_deg={math.degrees(float(target[side][joint - 1])):.3f}"
                )
            joint_detail = f" joints={'; '.join(held_joints)}" if held_joints else ""
            collision_detail = (
                f" collision={'; '.join(held_collision)}" if held_collision else ""
            )
            raise RuleBasedInitialAlignmentPlanningError(
                "initial alignment direct coupled transition was rejected: "
                f"status={raw_plan.status} reason={raw_plan.blocked_reason}"
                f"{joint_detail}{collision_detail}",
                status=raw_plan.status,
                blocked_reason=raw_plan.blocked_reason,
            )
        transition_waypoints = tuple(
            _qpos_pair(step.qpos_by_side, name="planned_waypoint")
            for step in raw_plan.steps
        )
        # Begin every certified plan with an exact, command-inert copy of the
        # accepted runtime-Home/start state.  This keeps the first rule frame
        # inside the planner's own pre-dispatch boundary; the original coupled
        # transition and all of its velocity/collision bounds then follow
        # unchanged.
        certified_waypoints = (
            _qpos_pair(start, name="startup_home_continuity_waypoint"),
            *transition_waypoints,
        )
        if len(certified_waypoints) == 1:
            # The shared C2 builder requires a real interval even when the
            # measured arms already equal the selected first frame.  The
            # duplicate is deliberately command-inert.
            certified_waypoints = (
                certified_waypoints[0],
                _qpos_pair(start, name="already_aligned_terminal_waypoint"),
            )
        dense_count, minimum_distance = self._certify_edges(
            start,
            certified_waypoints,
        )
        # The constrained planner above remains the geometric and collision
        # authority. Its direct path is then time-parameterized by the shared
        # backend-neutral C2 trajectory implementation used by replay.
        source_timestamps = np.arange(
            len(certified_waypoints),
            dtype=float,
        ) * float(self.config.waypoint_dt_sec)
        source_q = np.asarray(
            [
                np.concatenate(
                    (
                        np.asarray(waypoint["left"], dtype=float),
                        np.asarray(waypoint["right"], dtype=float),
                    )
                )
                for waypoint in certified_waypoints
            ],
            dtype=float,
        )
        base_velocity = min(
            math.radians(float(self.config.max_joint_step_deg))
            / float(self.config.waypoint_dt_sec),
            min(
                float(value)
                for value in RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S
            ),
        )
        acceleration = np.tile(
            np.asarray(RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2, dtype=float),
            2,
        )
        limits = JointHorizonLimits(
            min_position_rad=np.tile(RM65_ABSOLUTE_Q_MIN_RAD, 2),
            max_position_rad=np.tile(RM65_ABSOLUTE_Q_MAX_RAD, 2),
            max_velocity_rad_s=np.full(12, base_velocity, dtype=float),
            max_acceleration_rad_s2=acceleration,
            max_jerk_rad_s3=acceleration / 0.05,
        )
        base_continuous = build_offline_quintic_waypoint_trajectory(
            source_timestamps,
            source_q,
            command_hz=float(self.config.reference_hz),
            limits=limits,
            timing_mode=OFFLINE_WAYPOINT_TIMING_SAFE_RETIME_GLOBAL,
            max_time_scale=8.0,
        )
        continuous = (
            base_continuous
            if math.isclose(
                float(self.config.speed_scale),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            else build_offline_quintic_waypoint_trajectory(
                np.asarray(base_continuous.waypoint_timestamps, dtype=float)
                / float(self.config.speed_scale),
                source_q,
                command_hz=float(self.config.reference_hz),
                limits=limits,
            )
        )
        waypoints = tuple(
            {
                "left": np.asarray(row[:6], dtype=float).copy(),
                "right": np.asarray(row[6:], dtype=float).copy(),
            }
            for row in continuous.q_rad
        )
        reference_timestamps = tuple(
            float(value - continuous.timestamps[0])
            for value in continuous.timestamps
        )
        reference_dense_count, reference_minimum_distance = self._certify_edges(
            start,
            waypoints,
        )
        dense_count += reference_dense_count
        minimum_distance = min(minimum_distance, reference_minimum_distance)
        payload = {
            "schema": "dexteleop.rule_based_initial_alignment_plan.v2",
            "start_qpos_by_side": _lists(start),
            "target_qpos_by_side": _lists(target),
            "waypoints": [_lists(value) for value in waypoints],
            "reference_timestamps_sec": list(reference_timestamps),
            "max_joint_step_deg": float(self.config.max_joint_step_deg),
            "speed_scale": float(self.config.speed_scale),
            "edge_check_resolution_deg": float(
                self.config.edge_check_resolution_deg
            ),
            "required_collision_distance_m": required_distance,
        }
        semantic_hash = "sha256:" + hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        return RuleBasedInitialAlignmentPlan(
            start_qpos_by_side=start,
            target_qpos_by_side=target,
            waypoints=waypoints,
            reference_timestamps_sec=reference_timestamps,
            semantic_content_hash=semantic_hash,
            minimum_collision_distance_m=minimum_distance,
            dense_check_count=dense_count,
            metadata={
                "schema": payload["schema"],
                "planner": "direct_coupled_qpos_transition",
                "reference_trajectory": dict(continuous.metadata),
                "reference_hz": float(self.config.reference_hz),
                "waypoint_dt_sec": float(self.config.waypoint_dt_sec),
                "max_joint_step_deg": float(self.config.max_joint_step_deg),
                "speed_scale": float(self.config.speed_scale),
                "streaming_max_lag_waypoints": int(
                    self.config.streaming_max_lag_waypoints
                ),
                "streaming_max_lag_deg": float(
                    np.rad2deg(self.streaming_max_lag_rad)
                ),
                "edge_check_resolution_deg": float(
                    self.config.edge_check_resolution_deg
                ),
                "required_collision_distance_m": required_distance,
                "startup_home_continuity_waypoint": True,
                "transition_waypoint_count": len(transition_waypoints),
                "certified_waypoint_count": len(certified_waypoints),
                "reference_dense_check_count": int(reference_dense_count),
                "raw_plan_summary": raw_plan.summary(),
            },
        )

    def _certify_edges(
        self,
        start: Mapping[str, np.ndarray],
        waypoints: Sequence[Mapping[str, np.ndarray]],
    ) -> tuple[int, float]:
        previous = _qpos_pair(start, name="start")
        resolution = float(np.deg2rad(self.config.edge_check_resolution_deg))
        dense_count = 0
        minimum_distance = float("inf")
        for waypoint_index, raw_waypoint in enumerate(waypoints, start=1):
            waypoint = _qpos_pair(raw_waypoint, name="waypoint")
            max_delta = max(
                float(np.max(np.abs(waypoint[side] - previous[side])))
                for side in ("left", "right")
            )
            sample_count = max(1, int(math.ceil(max_delta / resolution)))
            for sample_index in range(1, sample_count + 1):
                alpha = float(sample_index) / float(sample_count)
                candidate = {
                    side: previous[side] + (waypoint[side] - previous[side]) * alpha
                    for side in ("left", "right")
                }
                check = self._model.certify(candidate)
                dense_count += 1
                if check.collision_distance_m is not None:
                    minimum_distance = min(
                        minimum_distance,
                        float(check.collision_distance_m),
                    )
                if not check.safe:
                    raise RuleBasedInitialAlignmentPlanningError(
                        "initial alignment dense edge certification failed: "
                        f"waypoint={waypoint_index} sample={sample_index}/"
                        f"{sample_count} reason={check.reason}"
                    )
            previous = waypoint
        if not waypoints:
            check = self._model.certify(previous)
            dense_count = 1
            if not check.safe:
                raise RuleBasedInitialAlignmentPlanningError(
                    "stationary initial alignment state failed certification: "
                    f"{check.reason}"
                )
            if check.collision_distance_m is not None:
                minimum_distance = float(check.collision_distance_m)
        if not math.isfinite(minimum_distance):
            raise RuleBasedInitialAlignmentPlanningError(
                "initial alignment collision distance was unavailable."
            )
        return dense_count, minimum_distance


def rule_based_initial_alignment_command_fault(
    command: TeleopCommand,
    *,
    planner: RuleBasedInitialAlignmentPlanner,
    expected_plan_hash: str | None,
    now: float,
) -> FaultEvent | None:
    metadata = command.metadata if isinstance(command.metadata, Mapping) else {}
    if str(metadata.get("rule_based_directive_kind", "")) != "initialize":
        return None
    actual_hash = str(
        metadata.get("rule_based_initial_alignment_plan_hash", "") or ""
    )
    if not expected_plan_hash or actual_hash != str(expected_plan_hash):
        return _fault(
            now,
            reason="uncertified_or_stale_plan",
            message="initial-alignment command lacks the active certified plan hash",
            metadata={"expected_plan_hash": expected_plan_hash, "actual_plan_hash": actual_hash},
        )
    commands = command.commands if isinstance(command.commands, Mapping) else {}
    qpos: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        value = _qpos_from_effector(commands.get(f"{side}_arm"))
        if value is None:
            return _fault(
                now,
                reason="missing_final_qpos",
                message=f"initial-alignment command has no concrete {side} qpos",
            )
        qpos[side] = value
    check = planner.certify_qpos_pair(qpos)
    if check.safe:
        return None
    return _fault(
        now,
        reason=check.reason,
        message=(
            "initial-alignment final command failed coupled collision/TCP-envelope "
            f"certification: {check.reason}"
        ),
        metadata={
            "collision_distance_m": check.collision_distance_m,
            "required_collision_distance_m": check.required_collision_distance_m,
            "envelope_modified_sides": list(check.envelope_modified_sides),
        },
    )


def rule_based_coupled_pose_command_fault(
    command: TeleopCommand,
    *,
    planner: RuleBasedInitialAlignmentPlanner,
    now: float,
) -> FaultEvent | None:
    """Recheck the concrete coupled qpos emitted for every rule pose action."""

    metadata = command.metadata if isinstance(command.metadata, Mapping) else {}
    if not bool(metadata.get("rule_based_action_has_pose", False)):
        return None
    if not bool(metadata.get("rule_based_coupled_pose_safety", False)):
        return _fault(
            now,
            reason="coupled_pose_solver_bypassed",
            message="rule pose command did not pass through the coupled IK/collision owner",
        )
    commands = command.commands if isinstance(command.commands, Mapping) else {}
    qpos: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        value = _qpos_from_effector(commands.get(f"{side}_arm"))
        if value is None:
            return _fault(
                now,
                reason="missing_coupled_pose_qpos",
                message=f"coupled rule pose command has no concrete {side} qpos",
            )
        qpos[side] = value
    check = planner.certify_qpos_pair(qpos)
    if check.safe:
        return None
    return _fault(
        now,
        reason=check.reason,
        message=(
            "coupled rule pose command failed final collision/TCP-envelope "
            f"certification: {check.reason}"
        ),
        metadata={
            "collision_distance_m": check.collision_distance_m,
            "required_collision_distance_m": check.required_collision_distance_m,
            "envelope_modified_sides": list(check.envelope_modified_sides),
        },
    )


def _fault(
    now: float,
    *,
    reason: str,
    message: str,
    metadata: Mapping[str, Any] | None = None,
) -> FaultEvent:
    return FaultEvent(
        code="rule_based_initial_alignment_safety_rejected",
        message=str(message),
        severity=FaultSeverity.ERROR,
        timestamp=float(now),
        component="rule_based_initial_alignment",
        metadata={"reason": str(reason), **dict(metadata or {})},
    )


def _qpos_from_effector(command: EffectorCommand | None) -> np.ndarray | None:
    if command is None:
        return None
    metadata = command.metadata if isinstance(command.metadata, Mapping) else {}
    for key in ("backend_final_target_qpos", "solver_output_qpos"):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            qpos = np.asarray(value, dtype=float).reshape(6)
        except (TypeError, ValueError):
            continue
        if np.all(np.isfinite(qpos)):
            return qpos.copy()
    if str(command.command_type) != "joint_vector":
        return None
    try:
        qpos = np.asarray(command.value, dtype=float).reshape(6)
    except (TypeError, ValueError):
        return None
    return qpos.copy() if np.all(np.isfinite(qpos)) else None


def _qpos_pair(value: Mapping[str, Any], *, name: str) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping) or set(value) != {"left", "right"}:
        raise ValueError(f"{name} must contain exactly left and right.")
    result: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        qpos = np.asarray(value[side], dtype=float).reshape(-1)
        if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
            raise ValueError(f"{name}.{side} must contain six finite values.")
        result[side] = qpos.copy()
    return result


def _lists(value: Mapping[str, np.ndarray]) -> dict[str, list[float]]:
    return {
        side: [float(item) for item in np.asarray(value[side], dtype=float)]
        for side in ("left", "right")
    }


def _optional_finite(value: Any) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if math.isfinite(resolved) else None


__all__ = [
    "RuleBasedInitialAlignmentConfig",
    "RuleBasedInitialAlignmentPlan",
    "RuleBasedInitialAlignmentPlanner",
    "RuleBasedInitialAlignmentPlanningError",
    "RuleBasedInitialAlignmentSafetyCheck",
    "rule_based_coupled_pose_command_fault",
    "rule_based_initial_alignment_command_fault",
]

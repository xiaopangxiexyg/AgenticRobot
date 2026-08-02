"""Production adapter for source-rate official all-IK and qref governance.

This module is intentionally an application adapter.  The vendor candidate
producer lives in :mod:`hardware.rm65`, while branch selection and the
finite-horizon governor are backend-neutral :mod:`robot_motion` components.
The adapter binds those pieces to the existing canonical RM65 collision
planner without owning dispatch or policy queue authority.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from hardware.rm65.official_all_ik import RM65OfficialAllIK
from robot_motion.position_control import (
    JointHorizonBatchedWorkspace,
    JointHorizonLimits,
    JointHorizonState,
    ResampledMotionReference,
)
from robot_motion.rm65 import (
    RM65_ABSOLUTE_Q_MAX_RAD,
    RM65_ABSOLUTE_Q_MIN_RAD,
    RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
    RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S,
    RM65PositionCollisionPlanResult,
    RM65PositionPlannerProtocol,
    select_nearest_collision_safe_rm65_ik_pair,
)


OFFICIAL_SOURCE_ALL_IK_GOVERNOR_BACKEND = (
    "official_source_all_ik_causal_governor"
)
PINK_POSE_LIFT_BACKEND = "pink_pose_lift"
POSITION_PATH_LIFT_BACKEND_CHOICES = (
    OFFICIAL_SOURCE_ALL_IK_GOVERNOR_BACKEND,
    PINK_POSE_LIFT_BACKEND,
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


class RM65OfficialAllIKGovernorPlanner:
    """Lift source Pose keyframes, then emit certified jerk-limited qcmd."""

    def __init__(
        self,
        collision_planner: RM65PositionPlannerProtocol,
        *,
        endpoint_to_tool_by_side: Mapping[str, np.ndarray],
        candidate_provider: RM65AllIKCandidateProvider | None = None,
        source_hz: float = 15.0,
        command_hz: float = 120.0,
        lookahead_source_steps: int = 2,
        jerk_ramp_sec: float = 0.05,
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
        self._candidate_provider = candidate_provider or RM65OfficialAllIK()
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
            solver_tolerance=1e-4,
        )
        self._source_pose: dict[int, _SourcePose] = {}
        self._source_qpos: dict[int, dict[str, np.ndarray]] = {}
        self._last_selected_source_index: int | None = None
        self._committed_source_index: int | None = None
        self._reset_anchor: dict[str, np.ndarray] | None = None
        self._selection_anchor: dict[str, np.ndarray] | None = None
        self._state: dict[str, JointHorizonState] | None = None
        self._cached_interval_index: int | None = None
        self._cached_interval_timestamp: float | None = None
        self._cached_plan: Any | None = None
        self._cached_qref: dict[str, np.ndarray] | None = None
        self._sequence = 0

    @property
    def collision_pair_count(self) -> int:
        return int(self._collision_planner.collision_pair_count)

    def reset(self, current_qpos_by_side: Mapping[str, Any]) -> Mapping[str, Any]:
        qpos = {
            side: np.asarray(current_qpos_by_side[side], dtype=float).reshape(6).copy()
            for side in ("left", "right")
        }
        receipt = self._collision_planner.reset(qpos)
        self._source_pose.clear()
        self._source_qpos.clear()
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
        self._sequence = 0
        return {
            **dict(receipt),
            "path_lift_backend": OFFICIAL_SOURCE_ALL_IK_GOVERNOR_BACKEND,
            "governor_state_reset": True,
        }

    def stage_source_pose(
        self,
        *,
        source_index: int,
        source_timestamp: float,
        pose_by_side: Mapping[str, Any],
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
        previous = self._source_pose.get(index)
        if previous is not None:
            if not math.isclose(
                previous.timestamp,
                timestamp,
                rel_tol=0.0,
                abs_tol=1e-9,
            ) or any(
                not np.allclose(
                    previous.pose_by_side[side],
                    poses[side],
                    rtol=0.0,
                    atol=1e-8,
                )
                for side in ("left", "right")
            ):
                # A replaced policy chunk may invalidate a speculative second
                # successor.  It is legal only if it has not been selected.
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
        self._source_pose[index] = _SourcePose(timestamp=timestamp, pose_by_side=poses)

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
        selection_metadata = self._select_source_qpos_through(horizon_last_index)
        dt = 1.0 / self.command_hz
        horizon_count = self.lookahead_source_steps * self.commands_per_source
        interval_timestamp = self._source_pose[interval_index].timestamp
        new_interval_plan = (
            self._cached_interval_index != interval_index
            or self._cached_plan is None
            or self._cached_qref is None
        )
        if new_interval_plan:
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
            plan = self._governor_workspace.solve(
                dual_reference,
                initial_state=dual_state,
                commit_count=min(
                    self.commands_per_source,
                    horizon_count,
                ),
            )
            self._cached_interval_index = interval_index
            self._cached_interval_timestamp = interval_timestamp
            self._cached_plan = plan
            self._cached_qref = qref
        else:
            plan = self._cached_plan
            qref = self._cached_qref
        assert plan is not None and qref is not None
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
        qcmd = {
            "left": np.asarray(plan.q_path_rad[plan_row, :6], dtype=float).copy(),
            "right": np.asarray(plan.q_path_rad[plan_row, 6:], dtype=float).copy(),
        }
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
        self._state = {
            side: JointHorizonState(
                q_rad=(
                    plan.q_path_rad[plan_row, :6]
                    if side == "left"
                    else plan.q_path_rad[plan_row, 6:]
                ),
                qd_rad_s=(
                    plan.qd_path_rad_s[plan_row, :6]
                    if side == "left"
                    else plan.qd_path_rad_s[plan_row, 6:]
                ),
                qdd_rad_s2=(
                    plan.qdd_path_rad_s2[plan_row, :6]
                    if side == "left"
                    else plan.qdd_path_rad_s2[plan_row, 6:]
                ),
            )
            for side in ("left", "right")
        }
        target_world = self._collision_planner.observe_endpoint_poses_world(
            {side: qref[side][plan_row] for side in ("left", "right")}
        )
        realized_world = self._collision_planner.observe_endpoint_poses_world(qcmd)
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
        frame_ms = (time.perf_counter() - started) * 1000.0
        plan_metadata = {
            **dict(metadata or {}),
            **selection_metadata,
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
            "governor_solver": plan.metadata.get("planner"),
            "governor_solver_tolerance": plan.metadata.get(
                "solver_tolerance"
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
            "solve_ms": frame_ms,
            "frame_total_ms": frame_ms,
            "solve_time_budget_exceeded": frame_ms > (1000.0 / self.command_hz),
            "temporal_smoothing": "third_order_jerk_limited_horizon",
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
                + (("third_order_horizon_governor",) if limited else ())
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
    "OFFICIAL_SOURCE_ALL_IK_GOVERNOR_BACKEND",
    "PINK_POSE_LIFT_BACKEND",
    "POSITION_PATH_LIFT_BACKEND_CHOICES",
    "RM65OfficialAllIKGovernorPlanner",
]

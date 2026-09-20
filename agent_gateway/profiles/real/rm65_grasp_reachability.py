"""RM65 profile adapter for optional point-grasp endpoint-IK annotations."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from robot_motion.coordinate_frames import CoordinateFrameGraph
from teleop_runtime.skill_deploy.motion_port import RealAgentMotionPort
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import (
    RM65GraspJointTrajectoryPlanner,
)

from .station_point_grasp_deployment import (
    StationPointGraspArmReachabilityProfile,
)


class RM65PointGraspReachabilityAdapter:
    """Evaluate Station candidates from one fresh measured dual-arm qpos cut."""

    def __init__(
        self,
        *,
        profile: StationPointGraspArmReachabilityProfile,
        planner: RM65GraspJointTrajectoryPlanner,
        motion_port: RealAgentMotionPort,
        frame_graph: CoordinateFrameGraph,
        candidate_reference_frame_id: str,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not isinstance(profile, StationPointGraspArmReachabilityProfile):
            raise TypeError("profile must be typed.")
        if not isinstance(planner, RM65GraspJointTrajectoryPlanner):
            raise TypeError("planner must be typed.")
        if not isinstance(motion_port, RealAgentMotionPort):
            raise TypeError("motion_port must be typed.")
        if not isinstance(frame_graph, CoordinateFrameGraph):
            raise TypeError("frame_graph must be typed.")
        if not callable(clock) or not callable(monotonic):
            raise TypeError("clock and monotonic must be callable.")
        candidate_frame = str(candidate_reference_frame_id).strip()
        if candidate_frame not in frame_graph.frame_by_id:
            raise ValueError("Candidate reference frame is absent from the graph.")
        if profile.solver_reference_frame_id not in frame_graph.frame_by_id:
            raise ValueError("Solver reference frame is absent from the graph.")
        if not frame_graph.connected(
            candidate_frame, profile.solver_reference_frame_id
        ):
            raise ValueError("Candidate and solver reference frames are disconnected.")
        if frame_graph.semantic_content_hash != profile.frame_graph_hash:
            raise ValueError("Reachability profile frame graph identity changed.")
        solver_frame = frame_graph.frame_by_id[profile.solver_reference_frame_id]
        if (
            solver_frame.semantic_content_hash
            != profile.solver_reference_frame_spec_hash
        ):
            raise ValueError("Reachability solver frame identity changed.")
        config = planner.reachability_config
        if config.semantic_content_hash != profile.evaluator_profile_hash:
            raise ValueError("Reachability evaluator profile identity changed.")
        expected = {
            "time_budget_ms": config.time_budget_ms,
            "max_solve_cycles": config.max_solve_cycles,
            "standard_position_tolerance_m": (config.standard_position_tolerance_m),
            "standard_orientation_tolerance_deg": (
                config.standard_orientation_tolerance_deg
            ),
            "basic_position_tolerance_m": config.basic_position_tolerance_m,
            "basic_orientation_tolerance_deg": (config.basic_orientation_tolerance_deg),
        }
        if any(getattr(profile, name) != value for name, value in expected.items()):
            raise ValueError("Reachability evaluator bounds changed.")
        self.profile = profile
        self.planner = planner
        self.motion_port = motion_port
        self.frame_graph = frame_graph
        self.candidate_reference_frame_id = candidate_frame
        self.clock = clock
        self.monotonic = monotonic

    def __call__(
        self,
        candidate_reference_from_tool_tcp: tuple[np.ndarray, ...],
        *,
        measured_qpos_cut: Any | None = None,
        measurement_reference_time: float | None = None,
    ) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
        if not 1 <= len(candidate_reference_from_tool_tcp) <= 5:
            raise ValueError("Reachability requires one to five candidates.")
        qpos_cut = (
            measured_qpos_cut
            if measured_qpos_cut is not None
            else self.motion_port.latest_measured_qpos()
        )
        if qpos_cut is None:
            raise RuntimeError("Arm reachability measured qpos is unavailable.")
        try:
            measured_at = float(qpos_cut.measured_at)
            reference_time = (
                float(self.clock())
                if measurement_reference_time is None
                else float(measurement_reference_time)
            )
            age_offset_ms = (reference_time - measured_at) * 1_000.0
        except (TypeError, ValueError, OverflowError):
            raise RuntimeError("Arm reachability measured qpos age is invalid.")
        if not math.isfinite(age_offset_ms):
            raise RuntimeError("Arm reachability measured qpos age is invalid.")
        if measured_qpos_cut is None:
            if age_offset_ms < 0.0:
                raise RuntimeError("Arm reachability measured qpos age is invalid.")
            age_ms = age_offset_ms
        else:
            if abs(age_offset_ms) > self.profile.max_measured_qpos_age_ms:
                raise RuntimeError(
                    "Arm reachability measured qpos is not camera-matched."
                )
            age_ms = max(0.0, age_offset_ms)
        if age_ms > self.profile.max_measured_qpos_age_ms:
            raise RuntimeError("Arm reachability measured qpos is stale.")

        results = []
        for candidate in candidate_reference_from_tool_tcp:
            reference_pose = np.asarray(candidate, dtype=float)
            if reference_pose.shape != (4, 4) or not np.all(
                np.isfinite(reference_pose)
            ):
                raise ValueError("Reachability candidate must be one finite 4x4 pose.")
            solver_pose, resolution = self.frame_graph.transform_pose_matrix(
                reference_pose,
                source_reference_frame_id=self.candidate_reference_frame_id,
                target_reference_frame_id=self.profile.solver_reference_frame_id,
            )
            if resolution.frame_graph_hash != self.profile.frame_graph_hash:
                raise RuntimeError("Reachability frame resolution identity changed.")
            result = self.planner.assess_endpoint_ik(
                qpos_by_side=qpos_cut.qpos_by_side,
                side=self.profile.arm_side,
                target_tool_tcp_world=solver_pose,
                monotonic=self.monotonic,
            )
            results.append(result.to_dict())
        context = {
            "arm_side": self.profile.arm_side,
            "evaluator_profile_id": self.profile.evaluator_profile_id,
            "evaluator_profile_hash": self.profile.evaluator_profile_hash,
            "candidate_reference_frame_id": self.candidate_reference_frame_id,
            "solver_reference_frame_id": self.profile.solver_reference_frame_id,
            "solver_reference_frame_spec_hash": (
                self.profile.solver_reference_frame_spec_hash
            ),
            "frame_graph_hash": self.profile.frame_graph_hash,
            "time_budget_ms": self.profile.time_budget_ms,
            "max_solve_cycles": self.profile.max_solve_cycles,
            "max_measured_qpos_age_ms": self.profile.max_measured_qpos_age_ms,
            "standard_position_tolerance_m": (
                self.profile.standard_position_tolerance_m
            ),
            "standard_orientation_tolerance_deg": (
                self.profile.standard_orientation_tolerance_deg
            ),
            "basic_position_tolerance_m": (self.profile.basic_position_tolerance_m),
            "basic_orientation_tolerance_deg": (
                self.profile.basic_orientation_tolerance_deg
            ),
            "measured_qpos_revision": qpos_cut.revision,
            "measured_qpos_at": qpos_cut.measured_at,
            "measured_qpos_age_ms": age_ms,
            "evaluation_scope": ("finite_time_endpoint_ik_not_path_or_grasp_success"),
            "candidate_order_effect": "annotation_only_order_unchanged",
        }
        return context, tuple(results)


__all__ = ["RM65PointGraspReachabilityAdapter"]

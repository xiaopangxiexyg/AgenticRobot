"""RM65 profile adapter for optional point-grasp endpoint-IK annotations."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from deployments.perception.sam2_graspgenx_point_grasp import (
    GraspGenXCandidateSelection,
)
from robot_motion.coordinate_frames import CoordinateFrameGraph
from teleop_runtime.skill_deploy.motion_port import RealAgentMotionPort
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import (
    RM65GraspJointTrajectoryPlanner,
    RM65GraspTrajectoryPlanningError,
)

from .station_point_grasp_deployment import (
    StationPointGraspArmReachabilityProfile,
)

_ARM_SHORTLIST_TOTAL_BUDGET_SEC = 8.0
_ARM_SHORTLIST_CANDIDATE_BUDGET_SEC = 2.0
_ARM_SHORTLIST_RESCUE_BUDGET_SEC = 8.0
_ARM_SHORTLIST_PROBE_TOTAL_BUDGET_SEC = 4.0
_ARM_SHORTLIST_PROBE_CANDIDATE_BUDGET_SEC = 0.35
_ARM_SHORTLIST_PROBE_MAX_COUNT = 24
_ARM_SHORTLIST_PROBE_PRIORITY = {
    "clearance_safe": 0,
    # A bounded probe's timeout or missing evaluation is not negative evidence.
    # Explore unknown candidates before spending the shortlist on known clashes.
    "unresolved": 1,
    "not_probed": 1,
    "clearance_blocked": 2,
    "collision_blocked": 3,
    "check_unavailable": 4,
}


def arm_base_origin_in_candidate_frame(
    *,
    scene_world_from_candidate: np.ndarray,
    scene_world_from_arm_base: np.ndarray,
) -> tuple[float, float, float]:
    """Resolve one arm-base origin in the point-grasp candidate frame."""

    world_from_candidate = np.asarray(
        scene_world_from_candidate, dtype=np.float64
    )
    world_from_arm_base = np.asarray(
        scene_world_from_arm_base, dtype=np.float64
    )
    if (
        world_from_candidate.shape != (4, 4)
        or world_from_arm_base.shape != (4, 4)
        or not np.all(np.isfinite(world_from_candidate))
        or not np.all(np.isfinite(world_from_arm_base))
    ):
        raise ValueError(
            "Arm-base candidate-frame transforms must be finite 4x4 matrices."
        )
    candidate_from_arm_base = (
        np.linalg.inv(world_from_candidate) @ world_from_arm_base
    )
    return tuple(float(value) for value in candidate_from_arm_base[:3, 3])


class RM65PointGraspReachabilityAdapter:
    """Evaluate Station candidates from one fresh measured dual-arm qpos cut."""

    def __init__(
        self,
        *,
        profile: StationPointGraspArmReachabilityProfile,
        planner: RM65GraspJointTrajectoryPlanner,
        motion_port: RealAgentMotionPort | None,
        frame_graph: CoordinateFrameGraph,
        candidate_reference_frame_id: str,
        arm_base_origin_candidate_m: tuple[float, float, float],
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not isinstance(profile, StationPointGraspArmReachabilityProfile):
            raise TypeError("profile must be typed.")
        if not isinstance(planner, RM65GraspJointTrajectoryPlanner):
            raise TypeError("planner must be typed.")
        if motion_port is not None and not isinstance(motion_port, RealAgentMotionPort):
            raise TypeError("motion_port must be typed or None for frozen replay.")
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
        arm_origin = np.asarray(arm_base_origin_candidate_m, dtype=np.float64)
        if arm_origin.shape != (3,) or not np.all(np.isfinite(arm_origin)):
            raise ValueError("arm_base_origin_candidate_m must be three finite values.")
        self.profile = profile
        self.planner = planner
        self.motion_port = motion_port
        self.frame_graph = frame_graph
        self.candidate_reference_frame_id = candidate_frame
        self.arm_base_origin_candidate_m = tuple(float(value) for value in arm_origin)
        self.clock = clock
        self.monotonic = monotonic

    def _resolve_measured_qpos(
        self,
        *,
        measured_qpos_cut: Any | None,
        measurement_reference_time: float | None,
    ) -> tuple[Any, float]:
        if measured_qpos_cut is None and self.motion_port is None:
            raise RuntimeError("Frozen reachability requires capture-matched measured qpos.")
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
        return qpos_cut, age_ms

    def _solver_pose(self, candidate_reference_from_tool_tcp: np.ndarray) -> np.ndarray:
        reference_pose = np.asarray(candidate_reference_from_tool_tcp, dtype=float)
        if reference_pose.shape != (4, 4) or not np.all(np.isfinite(reference_pose)):
            raise ValueError("Reachability candidate must be one finite 4x4 pose.")
        solver_pose, resolution = self.frame_graph.transform_pose_matrix(
            reference_pose,
            source_reference_frame_id=self.candidate_reference_frame_id,
            target_reference_frame_id=self.profile.solver_reference_frame_id,
        )
        if resolution.frame_graph_hash != self.profile.frame_graph_hash:
            raise RuntimeError("Reachability frame resolution identity changed.")
        return solver_pose

    def select_candidates(
        self,
        candidates: tuple[Any, ...],
        target_point_reference_m: tuple[float, float, float],
        maximum: int,
        *,
        measured_qpos_cut: Any | None = None,
        measurement_reference_time: float | None = None,
        endpoint_seeds: dict[tuple[float, ...], tuple[float, ...]] | None = None,
    ) -> GraspGenXCandidateSelection:
        """Admit candidates whose production local grasp chain is feasible."""

        if not candidates:
            return GraspGenXCandidateSelection(
                selected_indices=(),
                diagnostics={
                    "arm_shortlist_status": "empty_physical_candidate_pool",
                    "arm_shortlist_pool_count": 0,
                    "arm_shortlist_selected_count": 0,
                },
            )
        if isinstance(maximum, bool) or not 1 <= int(maximum) <= 5:
            raise ValueError("maximum must be in [1, 5].")
        target = np.asarray(target_point_reference_m, dtype=np.float64)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("target_point_reference_m must be three finite values.")
        qpos_cut, _age_ms = self._resolve_measured_qpos(
            measured_qpos_cut=measured_qpos_cut,
            measurement_reference_time=measurement_reference_time,
        )
        static_check = getattr(self.planner, "require_static_grasp_scene", None)
        if callable(static_check):
            try:
                static_check(qpos_by_side=qpos_cut.qpos_by_side, side=self.profile.arm_side)
            except RM65GraspTrajectoryPlanningError as error:
                return GraspGenXCandidateSelection(selected_indices=(), diagnostics={
                    "arm_shortlist_status": "static_scene_blocked",
                    "arm_shortlist_pool_count": len(candidates),
                    "arm_shortlist_selected_count": 0,
                    "arm_shortlist_local_chain_assessment_count": 0,
                    "arm_shortlist_rescue_activated": False,
                    "arm_shortlist_rejections_json": json.dumps({error.reason_code: 1}),
                    "arm_shortlist_static_scene_check_json": json.dumps({
                        "reason_code": error.reason_code, **error.details,
                    }),
                })
        desired_horizontal = target[:2] - np.asarray(
            self.arm_base_origin_candidate_m[:2], dtype=np.float64
        )
        desired_norm = float(np.linalg.norm(desired_horizontal))
        if desired_norm < 1e-6:
            raise RuntimeError("Arm base and grasp target have no horizontal separation.")
        desired_horizontal /= desired_norm

        candidate_evidence: list[dict[str, Any]] = []
        camera_axis = getattr(self.profile, "wrist_camera_side_axis_tool", None)
        for index, candidate in enumerate(candidates):
            reference_pose = np.asarray(candidate.base_from_tcp, dtype=np.float64).reshape(
                4, 4
            )
            solver_pose = self._solver_pose(reference_pose)
            approach = reference_pose[:3, 2]
            top_approach = -float(approach[2]) >= 0.7
            horizontal = approach[:2]
            horizontal_norm = float(np.linalg.norm(horizontal))
            if top_approach:
                side_angle_deg = 25.0
                side_admissible = True
            elif horizontal_norm < 1e-6:
                side_angle_deg = 180.0
                side_admissible = False
            else:
                # This slice aliases the pose's approach column.  Normalizing
                # it in place corrupts both the rotation and 3-D diversity dots.
                horizontal = horizontal / horizontal_norm
                side_angle_deg = math.degrees(
                    math.acos(
                        float(
                            np.clip(
                                np.dot(horizontal, desired_horizontal), -1.0, 1.0
                            )
                        )
                    )
                )
                side_admissible = side_angle_deg <= 85.0
            score = float(candidate.score)
            heuristic_score = score * math.exp(-side_angle_deg / 90.0)
            camera_side_up = (
                None if camera_axis is None else float(reference_pose[2, :3] @ camera_axis)
            )
            # Only side/inclined approaches have an unambiguous upper radial
            # side. Do not apply a wrist-roll preference to a vertical grasp.
            camera_upright = (
                camera_side_up is None or abs(float(approach[2])) >= 0.7
                or camera_side_up > 1e-6
            )
            candidate_evidence.append(
                {
                    "index": index,
                    "reference_pose": reference_pose,
                    "solver_pose": solver_pose,
                    "approach": approach,
                    "side_angle_deg": side_angle_deg,
                    "side_admissible": side_admissible,
                    "heuristic_score": heuristic_score,
                    "score": score,
                    "camera_side_up": camera_side_up,
                    "camera_upright": camera_upright,
                    "pose_source": str(
                        getattr(candidate, "pose_source", "model_scored")
                    ),
                    "half_turn": (
                        getattr(candidate, "orientation_variant_axis", None) == "z"
                        and abs(getattr(candidate, "orientation_variant_angle_deg", 0.0)) == 180.0
                    ),
                    "symmetry_parent_index": getattr(candidate, "symmetry_parent_index", None),
                }
            )

        side_eligible = [item for item in candidate_evidence if item["side_admissible"]]
        inverted = [item for item in side_eligible if not item["camera_upright"]]
        eligible = [item for item in side_eligible if item["camera_upright"]]
        symmetry_pairs = {}
        for item in eligible:
            parent = item["symmetry_parent_index"]
            if parent is not None:
                if (isinstance(parent, bool) or not isinstance(parent, int)
                        or not 0 <= parent < len(candidates) or parent == item["index"]):
                    raise ValueError("Symmetry lineage must reference another pool candidate.")
                symmetry_pairs[int(parent)] = int(item["index"])
                symmetry_pairs[int(item["index"])] = int(parent)
        eligible.sort(
            key=lambda item: (
                -float(item["heuristic_score"]),
                float(item["side_angle_deg"]),
                -float(item["score"]),
                int(item["index"]),
            )
        )
        diverse: list[dict[str, Any]] = []
        deferred: list[dict[str, Any]] = []
        diversity_cosine = math.cos(math.radians(20.0))
        for item in eligible:
            approach = np.asarray(item["approach"], dtype=np.float64)
            if any(
                float(np.dot(approach, np.asarray(previous["approach"])))
                > diversity_cosine
                for previous in diverse
            ):
                deferred.append(item)
            else:
                diverse.append(item)
        # A pose a few millimetres away can avoid the table or wrist-camera
        # collision even when its approach direction is nearly identical.
        # Give a few such variants a chance before spending the entire budget
        # on distinct but remote IK basins.
        base_evaluation = (
            diverse[:4]
            + deferred[:2]
            + diverse[4:8]
            + deferred[2:4]
            + diverse[8:]
            + deferred[4:]
        )
        # World-Z support refinements exist specifically to escape table and
        # wrist-camera clearance failures that a visually valid nominal pose
        # cannot reveal. Give a bounded set an early terminal-IK probe instead
        # of leaving all of them behind dozens of higher model scores.
        support_priority = [
            item
            for item in eligible
            if item["pose_source"] == "dmtac_support_clearance_variant_v1"
        ][:_ARM_SHORTLIST_PROBE_MAX_COUNT // 6]
        half_turn_priority = [item for item in eligible if item["half_turn"]][
            :_ARM_SHORTLIST_PROBE_MAX_COUNT // 6]
        priority = support_priority + half_turn_priority
        priority_indices = {int(item["index"]) for item in priority}
        evaluation = priority + [
            item
            for item in base_evaluation
            if int(item["index"]) not in priority_indices
        ]

        # Model score and approach direction alone can hide a low-scored side
        # grasp whose palm clears the stationary hand. This cheap native query
        # covers the entire eligible pool before the bounded IK probes. It has
        # no arm IK, pose-tolerance or path authority: keep every candidate.
        rigid_query = getattr(self.planner, "check_grasp_rigid_endpoint_clearance", None)
        rigid_started = float(self.monotonic())
        rigid_counts = {"safe": 0, "blocked": 0, "unavailable": 0}
        if callable(rigid_query):
            for item in evaluation:
                check = rigid_query(qpos_by_side=qpos_cut.qpos_by_side,
                    side=self.profile.arm_side, target_tool_tcp_world=item["solver_pose"])
                if check is None or not check.collision_check_available:
                    rigid_counts["unavailable"] += 1
                    continue
                rigid_counts["safe" if check.safe else "blocked"] += 1
                item["rigid_endpoint_check"] = {
                    "safe": check.safe, "minimum_distance_m": check.minimum_distance_m,
                    "closest_geometry_names": check.metadata.get("closest_geometry_names"),
                }
            # Unknown results retain their chance, and a blocked exact nominal
            # pose can still have a valid tolerance-region IK branch below.
            evaluation.sort(key=lambda item: not item.get("rigid_endpoint_check", {"safe": True})["safe"])
        rigid_elapsed_ms = max(0., (float(self.monotonic()) - rigid_started) * 1000.)

        # A bounded endpoint probe orders, but never removes, candidates before
        # the expensive local approach validation.  This gives every physical
        # family a cheap chance to expose a clearance-safe or positive-gap IK
        # branch and prevents the first few high model scores from consuming the
        # entire full-chain budget.  A missed probe remains eligible below.
        probe = getattr(self.planner, "probe_grasp_endpoint_clearance", None)
        endpoint_probe_attempts: list[dict[str, Any]] = []
        endpoint_probe_elapsed_ms = 0.0
        endpoint_probe_budget_exhausted = False
        if callable(probe):
            probe_started = float(self.monotonic())
            probe_deadline = probe_started + _ARM_SHORTLIST_PROBE_TOTAL_BUDGET_SEC
            for item in evaluation[:_ARM_SHORTLIST_PROBE_MAX_COUNT]:
                remaining_sec = probe_deadline - float(self.monotonic())
                if remaining_sec <= 0.0:
                    endpoint_probe_budget_exhausted = True
                    break
                try:
                    result = probe(
                        qpos_by_side=qpos_cut.qpos_by_side,
                        side=self.profile.arm_side,
                        target_tool_tcp_world=item["solver_pose"],
                        time_budget_sec=min(
                            _ARM_SHORTLIST_PROBE_CANDIDATE_BUDGET_SEC,
                            remaining_sec,
                        ),
                        monotonic=self.monotonic,
                    )
                except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                    classification = "check_unavailable"
                    evidence = {
                        "classification": classification,
                        "elapsed_ms": 0.0,
                        "reason_code": "endpoint_probe_exception",
                        "best_minimum_distance_m": None,
                        "attempted_seed_count": 0,
                        "exception_type": type(error).__name__,
                    }
                else:
                    classification = str(result.classification)
                    if classification not in _ARM_SHORTLIST_PROBE_PRIORITY:
                        classification = "check_unavailable"
                    evidence = {
                        "classification": classification,
                        "elapsed_ms": float(result.elapsed_ms),
                        "reason_code": result.reason_code,
                        "best_minimum_distance_m": (
                            result.best_minimum_distance_m
                        ),
                        "attempted_seed_count": int(
                            result.attempted_seed_count
                        ),
                    }
                item["endpoint_probe"] = evidence
                endpoint_probe_attempts.append(
                    {"index": int(item["index"]), **evidence}
                )
            endpoint_probe_elapsed_ms = max(
                0.0,
                (float(self.monotonic()) - probe_started) * 1_000.0,
            )
            evaluation.sort(
                key=lambda item: _ARM_SHORTLIST_PROBE_PRIORITY[
                    str(
                        item.get("endpoint_probe", {}).get(
                            "classification", "not_probed"
                        )
                    )
                ]
            )

        local_chain_assessment_count = 0
        local_chain_elapsed_ms = 0.0
        shortlist_started = float(self.monotonic())
        shortlist_deadline = shortlist_started + _ARM_SHORTLIST_TOTAL_BUDGET_SEC
        shortlist_budget_exhausted = False
        rescue_activated = False
        rescue_assessment_count = 0
        feasible: list[dict[str, Any]] = []
        shortlist_attempts: list[dict[str, Any]] = []
        rejection_counts: dict[str, int] = {}
        feasible_limit = int(maximum) * (2 if half_turn_priority else 1)
        assessed_indices = set()
        for evaluation_index, item in enumerate(evaluation):
            remaining_sec = shortlist_deadline - float(self.monotonic())
            if (
                remaining_sec < _ARM_SHORTLIST_CANDIDATE_BUDGET_SEC
                and not feasible
                and not rescue_activated
            ):
                # Only a zero-solution shortlist spends a second bounded window.
                # Do not start a nearly-zero-budget IK and mistake its timeout
                # for evidence that this candidate is physically unreachable.
                shortlist_deadline += _ARM_SHORTLIST_RESCUE_BUDGET_SEC
                rescue_activated = True
                remaining_sec = shortlist_deadline - float(self.monotonic())
            if remaining_sec <= 0.0:
                shortlist_budget_exhausted = True
                break
            local_chain = self.planner.assess_local_grasp_chain(
                qpos_by_side=qpos_cut.qpos_by_side,
                side=self.profile.arm_side,
                target_tool_tcp_world=item["solver_pose"],
                time_budget_sec=min(
                    _ARM_SHORTLIST_CANDIDATE_BUDGET_SEC,
                    remaining_sec,
                ),
                monotonic=self.monotonic,
            )
            local_chain_assessment_count += 1
            assessed_indices.add(int(item["index"]))
            if rescue_activated:
                rescue_assessment_count += 1
            local_chain_elapsed_ms += float(local_chain.elapsed_ms)
            attempt_reasons = tuple(
                str(attempt.get("reason_code"))
                for attempt in local_chain.attempts
                if attempt.get("reason_code") is not None
            )
            reason = str(local_chain.reason_code or "feasible")
            shortlist_attempts.append(
                {
                    "index": int(item["index"]),
                    "score": float(item["score"]),
                    "side_angle_deg": float(item["side_angle_deg"]),
                    "elapsed_ms": float(local_chain.elapsed_ms),
                    "reason_code": reason,
                    "branch_reasons": attempt_reasons,
                    "branch_attempts": tuple(
                        dict(attempt) for attempt in local_chain.attempts
                    ),
                    "endpoint_probe": item.get("endpoint_probe"),
                    "rigid_endpoint_check": item.get("rigid_endpoint_check"),
                    "joint_travel_rad": local_chain.joint_travel_rad,
                }
            )
            if not local_chain.feasible:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                shortlist_budget_exhausted = shortlist_budget_exhausted or (
                    local_chain.reason_code
                    == "path_planning_time_budget_exhausted"
                )
                continue
            feasible.append(
                {
                    **item,
                    "local_chain": local_chain,
                    "pregrasp_distance_m": (
                        local_chain.selected_pregrasp_distance_m
                    ),
                }
            )
            paired_index = symmetry_pairs.get(int(item["index"]))
            if paired_index is not None and paired_index not in assessed_indices:
                for next_index in range(evaluation_index + 1, len(evaluation)):
                    if int(evaluation[next_index]["index"]) == paired_index:
                        evaluation.insert(evaluation_index + 1, evaluation.pop(next_index))
                        break
            # A lower-scored finger-exchange proposal must be compared, not
            # hidden by the first full display page. Same total time budget.
            if len(feasible) >= feasible_limit:
                break

        feasible.sort(
            key=lambda item: (
                float(np.max(item["local_chain"].joint_travel_rad[:3])),
                float(np.linalg.norm(item["local_chain"].joint_travel_rad)),
                float(item["side_angle_deg"]),
                -float(item["pregrasp_distance_m"]),
                -float(item["score"]),
                int(item["index"]),
            )
        )
        selected: list[dict[str, Any]] = []
        deferred = []
        represented = set()
        for item in feasible:
            index = int(item["index"])
            if index in represented:
                continue
            represented.add(index)
            paired_index = symmetry_pairs.get(index)
            if paired_index is not None:
                represented.add(paired_index)
            approach = np.asarray(item["approach"], dtype=np.float64)
            if any(
                float(np.dot(approach, np.asarray(previous["approach"])))
                > diversity_cosine
                for previous in selected
            ):
                deferred.append(item)
            else:
                selected.append(item)
        selected = (selected + deferred)[: int(maximum)]
        if endpoint_seeds is not None:
            for item in selected:
                terminal_qpos = getattr(item["local_chain"], "terminal_qpos", None)
                if terminal_qpos is not None:
                    endpoint_seeds[tuple(item["solver_pose"].reshape(-1))] = terminal_qpos
        return GraspGenXCandidateSelection(
            selected_indices=tuple(int(item["index"]) for item in selected),
            diagnostics={
                "arm_shortlist_status": (
                    "production_local_grasp_chain_admitted"
                    if selected
                    else "no_production_local_grasp_chain_candidate"
                ),
                "arm_shortlist_pool_count": len(candidates),
                "arm_shortlist_side_admissible_count": len(side_eligible),
                "arm_shortlist_wrist_orientation_policy": (
                    "side_grasp_camera_above_v1" if camera_axis is not None else "unbound"
                ),
                "arm_shortlist_wrist_orientation_rejected_count": len(inverted),
                "arm_shortlist_wrist_orientation_rejections_json": json.dumps([
                    {"index": int(item["index"]), "camera_side_up": item["camera_side_up"],
                     "reason": "side_grasp_wrist_camera_not_above"}
                    for item in inverted[:_ARM_SHORTLIST_PROBE_MAX_COUNT]
                ], separators=(",", ":")),
                "arm_shortlist_evaluated_count": local_chain_assessment_count,
                "arm_shortlist_local_chain_assessment_count": (
                    local_chain_assessment_count
                ),
                "arm_shortlist_local_chain_feasible_count": len(feasible),
                "arm_shortlist_motion_cost_policy": "proximal_max_then_joint_travel_v1",
                "arm_shortlist_local_chain_elapsed_ms": (
                    local_chain_elapsed_ms
                ),
                "arm_shortlist_budget_exhausted": shortlist_budget_exhausted,
                "arm_shortlist_rescue_activated": rescue_activated,
                "arm_shortlist_rescue_assessment_count": rescue_assessment_count,
                "arm_shortlist_budget_sec": (
                    _ARM_SHORTLIST_TOTAL_BUDGET_SEC
                    + (_ARM_SHORTLIST_RESCUE_BUDGET_SEC if rescue_activated else 0.0)
                ),
                "arm_shortlist_endpoint_probe_count": len(
                    endpoint_probe_attempts
                ),
                "arm_shortlist_rigid_endpoint_counts_json": json.dumps(rigid_counts, separators=(",", ":")),
                "arm_shortlist_rigid_endpoint_elapsed_ms": rigid_elapsed_ms,
                "arm_shortlist_rigid_endpoint_scope": "advisory_only_not_ik_or_path_admission",
                "arm_shortlist_endpoint_probe_elapsed_ms": (
                    endpoint_probe_elapsed_ms
                ),
                "arm_shortlist_endpoint_probe_budget_exhausted": (
                    endpoint_probe_budget_exhausted
                ),
                "arm_shortlist_endpoint_probe_attempts_json": json.dumps(
                    endpoint_probe_attempts, separators=(",", ":")
                ),
                "arm_shortlist_attempts_json": json.dumps(
                    shortlist_attempts, separators=(",", ":")
                ),
                "arm_shortlist_rejections_json": json.dumps(
                    rejection_counts, sort_keys=True, separators=(",", ":")
                ),
                "arm_shortlist_selected_count": len(selected),
                "arm_shortlist_policy": (
                    "arm_side_family_then_rigid_endpoint_advisory_then_"
                    "advisory_endpoint_probe_then_production_local_grasp_chain_motion_cost_v5"
                ),
            },
        )

    def __call__(
        self,
        candidate_reference_from_tool_tcp: tuple[np.ndarray, ...],
        *,
        measured_qpos_cut: Any | None = None,
        measurement_reference_time: float | None = None,
        endpoint_seeds: Mapping[tuple[float, ...], tuple[float, ...]] | None = None,
    ) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
        if not 1 <= len(candidate_reference_from_tool_tcp) <= 5:
            raise ValueError("Reachability requires one to five candidates.")
        qpos_cut, age_ms = self._resolve_measured_qpos(
            measured_qpos_cut=measured_qpos_cut,
            measurement_reference_time=measurement_reference_time,
        )

        results = []
        for candidate in candidate_reference_from_tool_tcp:
            solver_pose = self._solver_pose(candidate)
            initial_seed = None if endpoint_seeds is None else endpoint_seeds.get(tuple(solver_pose.reshape(-1)))
            result = self.planner.assess_endpoint_ik(
                qpos_by_side=qpos_cut.qpos_by_side,
                side=self.profile.arm_side,
                target_tool_tcp_world=solver_pose,
                monotonic=self.monotonic,
                **({"initial_seed_qpos": np.asarray(initial_seed)} if initial_seed is not None else {}),
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

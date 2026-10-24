"""First-path selection through the existing RM65 planner, with no dispatch."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable

import numpy as np

from deployments.perception.grasp_first_path import (
    FirstGraspPathPolicy,
    GraspPathQueryRejected,
    find_first_grasp_path,
    grasp_capture_height_acceptable,
    grasp_capture_support_cost_components,
    grasp_contact_evidence,
    grasp_search_preference,
    original_grasp_neighbors,
    refine_first_grasp_path,
    refined_grasp_search_order,
)
from deployments.perception.sam2_graspgenx_point_grasp import (
    GraspGenXCandidateSelection,
)
from robot_motion.grasp_visible_scene import VisibleGraspScene
from robot_motion.pose import pose_delta_metrics
from teleop_runtime.skill_deploy.motion_port import RealAgentJointTrajectory
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import (
    RM65_GRASP_CANDIDATE_REJECTION_REASONS,
    RM65GraspTrajectoryPlanningError,
)

_LOGGER = logging.getLogger(__name__)
# A selector belongs to one frozen request. Cap advisory reuse independently
# of raw proposal/variant counts; a full cache simply recomputes on misses.
_PREFERENCE_CACHE_MAX_ENTRIES = 4096


class RM65FirstGraspPathSelector:
    """Cheap fixed preferences followed by complete, first-acceptable planning.

    The pose-priority callback may also order the generator's refinement
    sources. It is only a search preference, never an IK or collision proof.
    """

    def __init__(
        self,
        *,
        adapter,
        qpos_cut,
        visible_scene: Callable,
        speed_scale: float,
        deadline: float,
        policy=None,
        monotonic: Callable[[], float] = time.perf_counter,
        cancelled: Callable[[], bool] = lambda: False,
    ):
        self.adapter = adapter
        self.planner = adapter.planner
        self.qpos_cut = qpos_cut
        self.visible_scene = visible_scene
        self.speed_scale = speed_scale
        self.deadline = deadline
        self.policy = FirstGraspPathPolicy() if policy is None else policy
        self.monotonic = monotonic
        self.cancelled = cancelled
        self.result = None
        self.contact_evidence = {}
        self.pool_count = 0
        self.native_attempts = []
        self._preference_cache = {}
        self._preference_bindings = None
        self._preference_cache_hits = self._preference_cache_misses = 0
        self.planner.require_static_grasp_scene(
            qpos_by_side=qpos_cut.qpos_by_side, side=adapter.profile.arm_side
        )
        self.current_tcp = self.planner.tool_tcp_poses_for_qpos_pair(
            qpos_by_side=qpos_cut.qpos_by_side
        )[adapter.profile.arm_side]

    def _evidence(self, reference_pose, score, target):
        pose = np.asarray(reference_pose, dtype=float).reshape(4, 4)
        # Always run the canonical frame conversion and its identity/rigidity
        # checks, even on a cache hit. Never memoize an admission or path proof.
        solver = self.adapter._solver_pose(pose)
        bindings = (self.adapter.profile, self.planner,
            getattr(self.planner, "config", None), getattr(self.planner, "target_builder", None))
        if self._preference_bindings is None or any(
            old is not new for old, new in zip(self._preference_bindings, bindings)
        ):
            # Strong references make identity changes explicit; bare integer
            # ids could be recycled after a binding is replaced and collected.
            self._preference_cache.clear()
            self._preference_bindings = bindings
        target_array = np.asarray(target, dtype=float)
        key = None
        if target_array.shape == (3,) and np.isfinite(target_array).all():
            # Exact float64 bytes, not rounded pose buckets. Include the frozen
            # qpos/current TCP and replaceable bindings, so a changed context
            # cannot inherit a geometric necessary-condition result.
            key = (
                pose.tobytes(), np.asarray(solver, dtype=float).tobytes(), target_array.tobytes(),
                np.asarray(self.current_tcp, dtype=float).tobytes(),
                tuple((s, np.asarray(q, dtype=float).tobytes())
                    for s, q in sorted(self.qpos_cut.qpos_by_side.items())),
                np.asarray(getattr(self.adapter, "arm_base_origin_candidate_m", ()), dtype=float).tobytes(),
            )
        geometry = None if key is None else self._preference_cache.get(key)
        if geometry is None:
            self._preference_cache_misses += 1
            angle, side_admissible, _camera_side_up, upright = (
                self.adapter.approach_preference(pose, target)
            )
            reach = self.planner.check_grasp_geometric_reach(
                qpos_by_side=self.qpos_cut.qpos_by_side,
                side=self.adapter.profile.arm_side,
                target_tool_tcp_world=solver,
            )
            outside = reach is not None and reach.outside
            distance, rotation = pose_delta_metrics(self.current_tcp, solver)
            geometry = (angle, side_admissible, upright, outside, distance, rotation)
            if key is not None and len(self._preference_cache) < _PREFERENCE_CACHE_MAX_ENTRIES:
                self._preference_cache[key] = geometry
        else:
            self._preference_cache_hits += 1
            angle, side_admissible, upright, outside, distance, rotation = geometry
        # Model confidence and policy weights are deliberately fresh. Source
        # allocation and final ordering share geometry, not a stale ranking.
        p = self.policy
        cost = (
            p.translation_weight * distance / 0.5
            + p.orientation_weight * rotation / 180.0
            + p.arm_side_weight * angle / 90.0
            + p.model_score_weight * (1.0 - max(0.0, min(1.0, float(score))))
        )
        return side_admissible and upright and not outside, cost, solver

    def priority(self, reference_pose, score, target):
        if self.cancelled():
            raise RuntimeError(
                "Fast grasp planning was cancelled during proposal preparation."
            )
        if self.monotonic() >= self.deadline:
            raise RM65GraspTrajectoryPlanningError("grasp_fast_search_budget_exhausted")
        eligible, cost, _ = self._evidence(reference_pose, score, target)
        return (0 if eligible else 1, cost)

    def __call__(self, candidates, target, maximum):
        if type(maximum) is not int or maximum < 1:
            raise ValueError("Fast grasp selection requires a positive output bound.")
        self.pool_count = len(candidates)
        eligible = []
        for index, candidate in enumerate(candidates):
            if self.cancelled():
                raise RuntimeError("Fast grasp planning was cancelled.")
            if self.monotonic() >= self.deadline:
                break
            valid, cost, _ = self._evidence(
                candidate.base_from_tcp, candidate.score, target
            )
            if (candidate.visible_collision_free and valid
                    and grasp_capture_height_acceptable(candidate, policy=self.policy)):
                eligible.append((
                    *grasp_search_preference(candidate, cost, policy=self.policy),
                    index,
                ))
        eligible.sort()
        scene = self.visible_scene()
        if not isinstance(scene, VisibleGraspScene):
            raise TypeError(
                "Fast grasp planning requires the exact filtered observed scene."
            )
        solver_from_reference = self.adapter._solver_pose(np.eye(4))
        scene = VisibleGraspScene(
            points=scene.points @ solver_from_reference[:3, :3].T
            + solver_from_reference[:3, 3],
            target_mask=scene.target_mask,
            geometry=scene.geometry,
            capture_qpos_by_side=scene.capture_qpos_by_side,
            capture_self_observation_evidence=scene.capture_self_observation_evidence,
            depth_preprocessing_evidence=scene.depth_preprocessing_evidence,
        )
        candidate_indices = {id(candidate): index for index, candidate in enumerate(candidates)}
        probe_seeds = {}
        probe_records = []

        def endpoint_probe(candidate, budget):
            if self.cancelled():
                raise RuntimeError("Fast grasp planning was cancelled during endpoint probing.")
            result = self.planner.probe_grasp_endpoint_clearance(
                qpos_by_side=self.qpos_cut.qpos_by_side,
                side=self.adapter.profile.arm_side,
                target_tool_tcp_world=self.adapter._solver_pose(
                    np.asarray(candidate.base_from_tcp).reshape(4, 4)),
                time_budget_sec=budget, monotonic=self.monotonic)
            if self.cancelled():
                raise RuntimeError("Fast grasp planning was cancelled during endpoint probing.")
            if result.classification == "check_unavailable":
                raise RM65GraspTrajectoryPlanningError(result.reason_code or "endpoint_collision_check_unavailable")
            if (result.reason_code is not None
                    and result.reason_code not in RM65_GRASP_CANDIDATE_REJECTION_REASONS):
                raise RM65GraspTrajectoryPlanningError(result.reason_code)
            if result.classification not in {"clearance_safe", "clearance_blocked", "collision_blocked", "unresolved"}:
                raise ValueError("Unknown native endpoint-probe classification.")
            probe_records.append({"index": candidate_indices[id(candidate)],
                "classification": result.classification, "reason_code": result.reason_code,
                "elapsed_ms": getattr(result, "elapsed_ms", None),
                "attempted_seed_count": getattr(result, "attempted_seed_count", None)})
            if result.classification == "clearance_safe":
                seed = getattr(result, "seed_qpos", None)
                if seed is not None:
                    probe_seeds[id(candidate)] = np.asarray(seed, dtype=float).copy()
                return True
            return False

        def plan(candidate, budget, refinement_deadline=None):
            if self.cancelled():
                raise RuntimeError("Fast grasp planning was cancelled.")
            target = self.adapter._solver_pose(
                np.asarray(candidate.base_from_tcp).reshape(4, 4))
            if refinement_deadline is not None:
                budget = min(budget, refinement_deadline - self.monotonic())
                if budget <= 0:
                    raise GraspPathQueryRejected("path_planning_time_budget_exhausted")
            try:
                trajectory = self.planner(
                    qpos_by_side=self.qpos_cut.qpos_by_side,
                    side=self.adapter.profile.arm_side,
                    target_tool_tcp_world=target,
                    speed_scale=self.speed_scale,
                    visible_scene=scene,
                    require_grasp_approach=True,
                    time_budget_sec=budget,
                    monotonic=self.monotonic,
                    **({"initial_seed_qpos": probe_seeds[id(candidate)]}
                       if id(candidate) in probe_seeds else {}),
                )
            except RM65GraspTrajectoryPlanningError as error:
                # Keep request-local native facts in the replay funnel. The
                # public cross-target history remains its compact I/O contract.
                self.native_attempts.append({
                    "index": candidate_indices[id(candidate)],
                    "allocated_budget_sec": float(budget),
                    "reason_code": error.reason_code,
                    **{key: error.details[key] for key in (
                        "planning_phase", "cycles", "endpoint_seed_count",
                        "best_position_residual_m", "best_orientation_residual_deg",
                    ) if key in error.details},
                })
                if error.reason_code not in RM65_GRASP_CANDIDATE_REJECTION_REASONS:
                    raise
                raise GraspPathQueryRejected(error.reason_code) from error
            if not isinstance(trajectory, RealAgentJointTrajectory):
                raise TypeError("The native planner did not return a typed trajectory.")
            if self.cancelled():
                raise RuntimeError("Fast grasp planning was cancelled.")
            if refinement_deadline is not None and self.monotonic() >= refinement_deadline:
                raise GraspPathQueryRejected("path_planning_time_budget_exhausted")
            return trajectory

        order = refined_grasp_search_order(
            candidates,
            tuple(index for _, _, index in eligible),
            policy=self.policy,
        )
        self.result = find_first_grasp_path(
            candidates=candidates,
            ordered_indices=order,
            plan=plan,
            policy=self.policy,
            deadline=self.deadline,
            original_neighbors=original_grasp_neighbors(candidates, order, policy=self.policy),
            endpoint_probe=(endpoint_probe if callable(
                getattr(self.planner, "probe_grasp_endpoint_clearance", None)) else None),
            monotonic=self.monotonic,
        )
        self.result = refine_first_grasp_path(self.result, candidates=candidates,
            eligible_indices=tuple(index for _, _, index in eligible), plan=plan,
            policy=self.policy, deadline=self.deadline, monotonic=self.monotonic,
            preference_costs={index: total for total, _, index in eligible},
            endpoint_probe=(endpoint_probe if callable(
                getattr(self.planner, "probe_grasp_endpoint_clearance", None)) else None))
        if self.result.selected_index is not None:
            self.contact_evidence = grasp_contact_evidence(candidates[self.result.selected_index])
        if _LOGGER.isEnabledFor(logging.INFO):
            # Keep bounded counterfactual evidence in the owner log, not in the
            # Agent's result payload. No extra solver/model queries are issued.
            costs = {index: (total, motion) for total, motion, index in eligible}
            indices = dict.fromkeys(
                [index for _, _, index in eligible[:8]]
                + [attempt["index"] for attempt in self.result.attempts]
                + ([] if self.result.selected_index is None else [self.result.selected_index])
            )
            records = []
            for index in indices:
                candidate = candidates[index]
                total, motion = costs[index]
                records.append({
                    "index": index,
                    "pose_source": str(getattr(candidate, "pose_source", "unknown"))[:128],
                    "target_tcp_reference": np.asarray(candidate.base_from_tcp).reshape(-1).tolist(),
                    "ranking_score": float(candidate.score),
                    "model_score": (
                        float(candidate.model_score) if hasattr(candidate, "model_score") else None
                    ),
                    "motion_cost": float(motion),
                    "support_costs": grasp_capture_support_cost_components(candidate, policy=self.policy),
                    "observed_contact": {
                        name: getattr(candidate, name, None) for name in (
                            "surface_alignment_score", "surface_alignment_observed_sides",
                            "capture_support_score", "opposed_capture_support_score",
                            "opposed_capture_support_observed_sides",
                            "surface_opposition_score", "contact_height_support_score",
                            "contact_height_center_shift_m", "contact_height_evidence_weight",
                            "height_shift_m", "height_shift_basis")
                    },
                    "total_cost": float(total),
                })
            _LOGGER.info("Fast-grasp search evidence: %s", json.dumps({
                "reference_frame_id": self.adapter.candidate_reference_frame_id,
                "pool_count": len(candidates),
                "eligible_count": len(eligible),
                "lowest_cost_indices": [index for _, _, index in eligible[:8]],
                "search_order_prefix": list(order[:8]),
                "selected_index": self.result.selected_index,
                "attempts": self.result.attempts,
                "endpoint_probes": self.result.endpoint_probes,
                "native_endpoint_probes": probe_records,
                "advisory_geometry_reuse": {
                    "hits": self._preference_cache_hits,
                    "misses": self._preference_cache_misses,
                    "entries": len(self._preference_cache),
                    "entry_limit": _PREFERENCE_CACHE_MAX_ENTRIES,
                    "scope": "request_local_preference_not_ik_collision_or_path",
                },
                "candidates": records,
            }, sort_keys=True))
        return GraspGenXCandidateSelection(
            () if self.result.selected_index is None else (self.result.selected_index,),
            {
                "arm_shortlist_status": self.result.reason_code,
                "arm_shortlist_pool_count": len(candidates),
                "arm_shortlist_pad_height_rejected_count": sum(
                    not grasp_capture_height_acceptable(c, policy=self.policy) for c in candidates),
                "arm_shortlist_selected_count": int(
                    self.result.selected_index is not None
                ),
                "arm_shortlist_full_path_query_count": len(self.result.attempts),
                "arm_shortlist_endpoint_probe_count": len(probe_records),
                "arm_shortlist_endpoint_probe_attempts_json": json.dumps(probe_records),
                "arm_shortlist_full_path_attempts_json": json.dumps(
                    self.result.attempts
                ),
                "arm_shortlist_native_path_attempts_json": json.dumps(self.native_attempts),
            },
        )

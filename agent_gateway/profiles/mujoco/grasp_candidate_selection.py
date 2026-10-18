"""Read-only GraspGenX selection through the existing native motion planner.

The caller passes this selector to Sam2GraspGenXPointGraspPlanner.plan. Thus
finger-exchange poses are generated and visibly admitted by their existing
owner, not invented or silently substituted by a motion Skill.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from dataset_contracts.perception_artifacts import (
    PARALLEL_JAW_FINGER_EXCHANGE_POSE_SOURCE_V1,
)
from deployments.perception.sam2_graspgenx_point_grasp import (
    GraspGenXCandidateSelection,
)
from robot_motion.object_transport import rigid_pose
from robot_motion.path_planning import JointConfigurationCheck
from robot_motion.path_planning.motion_quality import (
    DEFAULT_PATH_MOTION_QUALITY_LIMITS,
    sampled_path_motion_quality,
)
from robot_motion.relative_action import xyzquat_to_pose_matrix
from sim.mujoco.deploy.planned_tcp_motion import MujocoTcpPlanRejected
from sim.mujoco.transforms import pose_to_xyz_quat_xyzw


@dataclass(frozen=True)
class MujocoGraspSelectionConfig:
    pregrasp_distance_m: float = 0.07
    near_distance_m: float = 0.025
    terminal_offset_m: float = 0.0
    maximum_candidates_to_check: int = 24
    time_budget_sec: float = 15.0

    def __post_init__(self):
        if not (0 < self.near_distance_m < self.pregrasp_distance_m):
            raise ValueError("Selection requires ordered positive approach distances.")
        if not math.isfinite(self.pregrasp_distance_m) or not math.isfinite(self.terminal_offset_m):
            raise ValueError("Grasp selection distances must be finite.")
        if not math.isfinite(self.time_budget_sec) or self.time_budget_sec <= 0:
            raise ValueError("Selection time budget must be positive and finite.")
        if type(self.maximum_candidates_to_check) is not int or self.maximum_candidates_to_check < 1:
            raise ValueError("Selection candidate budget must be a positive integer.")


class MujocoGraspCandidateSelector:
    """Preview a bounded full approach chain, without activating any plan.

    Candidate base_from_tcp is interpreted ONLY via the explicit caller-supplied
    world_from_candidate_reference. Execution must submit the selected exact
    poses afresh; a selection is not a motion permit or a grasp-success result.
    """

    def __init__(self, *, application, contract_bundle,
                 world_from_candidate_reference, config=None):
        self.application = application
        self.bundle = contract_bundle
        self.config = config = MujocoGraspSelectionConfig() if config is None else config
        if not isinstance(config, MujocoGraspSelectionConfig):
            raise TypeError("config must be MujocoGraspSelectionConfig.")
        self.world_from_candidate_reference = rigid_pose(
            world_from_candidate_reference, name="world_from_candidate_reference")
        self.last_attempts = ()
        if contract_bundle.grasp_approach_contract is None or contract_bundle.grasp_contact_contract is None:
            raise ValueError("Full grasp selection requires approach and target-bound contact Routes.")
        binding = contract_bundle.grasp_contact_contract.contact_binding
        if not 0 < config.near_distance_m + config.terminal_offset_m <= binding.max_approach_m:
            raise ValueError("Near pose is outside the bound terminal contact corridor.")

    def _constraint(self, contract):
        safety = self.application.tcp_pose_envelope_runtime
        entry = next(item for item in self.bundle.route_manifest.skill_routes
            if item.route.semantic_content_hash == contract.evaluation_plan.route_hash)
        refs = entry.deployment_binding.safety_envelope_refs
        if len(refs) != 1:
            raise ValueError("Grasp preview requires one exact deployment envelope.")
        planner = self.application.target.planned_tcp_motion

        def check(model_qpos, pair):
            pose = xyzquat_to_pose_matrix(planner.pose_for_configuration(model_qpos, pair))
            result = safety.preview_for_envelope({"right": pose},
                deployment_envelope_hash=refs[0].content_hash)
            return JointConfigurationCheck(safe=not result.modified_sides, minimum_distance_m=None,
                reason=None if not result.modified_sides else "tcp_envelope_blocked",
                metadata=result.to_payload())

        return check, refs[0].content_hash

    def __call__(self, candidates, target_point, maximum):
        del target_point  # Geometry/target membership was already admitted upstream.
        self.last_attempts = ()
        if type(maximum) is not int or maximum < 1:
            raise ValueError("Requested shortlist size must be a positive integer.")
        app = self.application
        planner = app.target.planned_tcp_motion
        if not app.held or planner.active:
            raise RuntimeError("Read-only grasp selection requires confirmed stationary HOLD.")
        capture = lambda: app.runtime.rig.capture_scene(lambda scene: scene.data.qpos.copy())
        source = capture()
        base_from_world = np.linalg.inv(app.scene.body_pose("right_arm_base_link"))
        approach, contact = self.bundle.grasp_approach_contract, self.bundle.grasp_contact_contract
        constraints = {id(contract): self._constraint(contract) for contract in (approach, contact)}
        deadline = time.perf_counter() + self.config.time_budget_sec
        attempts, accepted = [], []
        originals = [i for i, c in enumerate(candidates)
            if c.pose_source != PARALLEL_JAW_FINGER_EXCHANGE_POSE_SOURCE_V1]
        exchanges = [i for i, c in enumerate(candidates)
            if c.pose_source == PARALLEL_JAW_FINGER_EXCHANGE_POSE_SOURCE_V1]
        # A score penalty must not put every symmetry alternative beyond the
        # finite preview budget. Each group retains its upstream score order.
        order = []
        for i in range(max(len(originals), len(exchanges))):
            order.extend(group[i] for group in (originals, exchanges) if i < len(group))
        for index in order[:self.config.maximum_candidates_to_check]:
            candidate = candidates[index]
            if time.perf_counter() >= deadline:
                break
            row = {"candidate_index": index, "pose_source": candidate.pose_source, "stages": []}
            attempts.append(row)
            self.last_attempts = tuple(attempts)
            if not candidate.visible_collision_free:
                row.update(status="rejected", reason="candidate_not_visibly_admitted")
                continue
            model_qpos = source.copy()
            candidate_pose = rigid_pose(np.asarray(candidate.base_from_tcp).reshape(4, 4), name="candidate pose")
            grasp = base_from_world @ self.world_from_candidate_reference @ candidate_pose
            duration = 0.0
            chain_q, chain_xyz = [], []
            try:
                for name, offset, contract in (
                    ("pregrasp", -self.config.pregrasp_distance_m, approach),
                    ("near", -self.config.near_distance_m, approach),
                    ("terminal", self.config.terminal_offset_m, contact),
                ):
                    goal = grasp.copy()
                    goal[:3, 3] += grasp[:3, 2] * offset
                    constraint, identity = constraints[id(contract)]
                    plan = planner.plan(tuple(pose_to_xyz_quat_xyzw(goal)),
                        translation_tolerance_m=contract.motion_profile.translation_tolerance_m,
                        rotation_tolerance_rad=contract.motion_profile.rotation_tolerance_rad,
                        cancel_requested=lambda: time.perf_counter() >= deadline,
                        configuration_constraint=constraint, constraint_identity=identity,
                        path_search_time_budget_sec=min(contract.path_search_time_budget_sec,
                            max(0.001, deadline - time.perf_counter())),
                        contact_binding=contract.contact_binding if contract is contact else None,
                        preview_model_qpos=model_qpos)
                    quality = dict(plan.path.metadata["motion_quality"])
                    row["stages"].append({"phase": name, "motion_quality": quality,
                        "trajectory_duration_sec": float(plan.trajectory.timestamps[-1]),
                        "trajectory_hash": plan.trajectory_hash})
                    duration += float(plan.trajectory.timestamps[-1])
                    chain_q.extend(plan.trajectory.q_rad)
                    world_from_base = app.scene.body_pose("right_arm_base_link")
                    chain_xyz.extend((world_from_base @ xyzquat_to_pose_matrix(pose))[:3, 3]
                        for pose in plan.reference_poses)
                    for side, q in plan.path.target_qpos_by_component.items():
                        addresses = [app.scene.joint_qpos_addr(n) for n in app.scene.arm_specs[side].joint_names]
                        model_qpos[addresses] = q
            except MujocoTcpPlanRejected as error:
                row.update(status="rejected", reason=error.reason_code, details=error.details)
                continue
            quality = sampled_path_motion_quality(np.asarray(chain_q), np.asarray(chain_xyz))
            if DEFAULT_PATH_MOTION_QUALITY_LIMITS.exceeded(quality):
                row.update(status="rejected", reason="grasp_chain_motion_quality_exceeded", motion_quality=quality)
                continue
            row.update(status="approach_chain_found", trajectory_duration_sec=duration, motion_quality=quality)
            accepted.append((duration, index))
        self.last_attempts = tuple(attempts)
        if not app.held or planner.active or not np.array_equal(source, capture()):
            raise RuntimeError("MuJoCo scene changed during read-only candidate selection.")
        accepted.sort()
        return GraspGenXCandidateSelection(tuple(index for _, index in accepted[:maximum]), {
            "arm_shortlist_checked": len(attempts), "arm_shortlist_chain_found": len(accepted),
            "arm_shortlist_budget_exhausted": int(time.perf_counter() >= deadline),
            "arm_shortlist_pool_size": len(candidates),
            "arm_shortlist_not_checked": len(candidates) - len(attempts),
            "arm_shortlist_candidate_budget_exhausted": int(len(candidates) > self.config.maximum_candidates_to_check),
            "arm_shortlist_scope": "frozen_native_approach_chain_not_close_lift_or_execution_authority",
        })

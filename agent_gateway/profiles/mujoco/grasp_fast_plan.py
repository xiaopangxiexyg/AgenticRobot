"""Frozen GraspGenX -> first complete native chain -> exact paired execution."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from PIL import Image

from agent_gateway.profiles.grasp_observation import read_retained_grasp_observation
from agent_gateway.runtime.passive_read_executor import (
    PassiveReadError,
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from agent_gateway.runtime.station_operation_worker import (
    StationLifecycleActivationEvidence,
)
from agent_gateway.skills.perception.grasp_fast_plan_schemas import (
    grasp_path_attempts_json,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.grasp_rgbd_snapshot import decode_grasp_rgbd_snapshot
from deployments.perception.grasp_first_path import (
    GraspPathQueryRejected,
    find_first_grasp_path,
    grasp_capture_height_acceptable,
    grasp_search_preference,
    original_grasp_neighbors,
    refine_first_grasp_path,
    refined_grasp_search_order,
)
from deployments.perception.grasp_multiview import GraspRgbdView, fuse_grasp_views
from deployments.perception.point_grasp_failure import PointGraspPlanningError
from deployments.perception.sam2_graspgenx_point_grasp import (
    GraspGenXCandidateSelection,
    Sam2GraspGenXPointGraspPlanner,
)
from robot_motion.object_transport import rigid_pose
from robot_motion.relative_action import xyzquat_to_pose_matrix
from sim.mujoco.deploy.planned_tcp_motion import (
    MujocoTcpPlanGoal,
    MujocoTcpPlanRejected,
)
from sim.mujoco.transforms import pose_to_xyz_quat_xyzw

from .grasp_candidate_selection import mujoco_grasp_configuration_constraint
from .motion_skill import MujocoAgentPreparedTcpPoseSkillLifecycle

_GOAL_REJECTIONS = frozenset(
    {
        "safe_tcp_target_blocked",
        "safe_tcp_endpoint_search_exhausted",
        "safe_tcp_path_search_exhausted",
        "safe_tcp_motion_quality_exceeded",
        "safe_tcp_retimed_path_blocked",
        "safe_tcp_trajectory_budget_exhausted",
        "safe_tcp_planning_budget_exhausted",
    }
)
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MujocoRetainedGraspPlan:
    native_plan: object
    planning_operation_id: str
    planning_request_hash: str
    observation_operation_id: str
    observation_request_hash: str
    observation_result_hash: str
    observation_archive_sha256: str
    source_scene_hash: str
    station_revision: int
    profile_hash: str
    runtime_session_id: str
    created_at: float
    expires_at: float
    content_hash: str

    @property
    def plan_id(self):
        return "mujoco-grasp-approach-plan:" + self.content_hash


def _scene_hash(state):
    return canonical_content_hash(
        state, domain="dexteleop.mujoco.frozen_grasp_scene.v1"
    )


class MujocoFastGraspPlanProvider:
    """One native pending handle, not a second completion or reference owner.

    The service owns operation completion; the native planner owns pending /
    activation / single consumption. This profile keeps only the immutable
    provenance and TTL for the one currently pending plan.
    """

    def __init__(
        self,
        *,
        application,
        bundle,
        model,
        observation_executor,
        artifact_store,
        operation_lookup,
        current_station_revision,
        clock,
        monotonic=time.perf_counter,
    ):
        if not isinstance(model, Sam2GraspGenXPointGraspPlanner):
            raise TypeError(
                "Fast native planning requires the existing typed GraspGenX planner."
            )
        self.application, self.bundle, self.model = application, bundle, model
        self.contract = bundle.fast_grasp_contract
        if (
            self.contract is None
            or self.contract.model_profile_hash != model.semantic_content_hash
        ):
            raise ValueError(
                "Fast grasp model does not match its published deployment."
            )
        self.observation_executor, self.artifact_store = (
            observation_executor,
            artifact_store,
        )
        self.operation_lookup, self.current_station_revision = (
            operation_lookup,
            current_station_revision,
        )
        self.clock, self.monotonic = clock, monotonic
        self._lock = threading.RLock()
        self._retained = None
        self._closed = False

    def close(self):
        with self._lock:
            self._closed = True
            if self._retained is not None:
                self.application.target.planned_tcp_motion.discard_pending_plan(
                    self._retained.native_plan
                )
            self._retained = None

    def _source(self, identifier, revision):
        views, metadata = read_retained_grasp_observation(
            executor=self.observation_executor,
            operation_id=identifier,
            artifact_store=self.artifact_store,
            runtime_session_id=self.application.runtime.config.runtime_session_id,
            expected_station_revision=revision,
            frame_binding_hash=self.contract.frame_binding_hash,
        )
        operation = self.operation_lookup(identifier)
        completion = self.observation_executor.completion_for(identifier)
        if (
            operation is None
            or operation.status != "completed"
            or operation.execution_outcome != "completed"
            or completion is None
            or completion.operation.request_hash != operation.request_hash
            or completion.operation.execution_result != operation.execution_result
            or completion.operation.output_artifact_bindings
            != operation.output_artifact_bindings
        ):
            raise ValueError(
                "Frozen observation producer is not authoritatively completed."
            )
        binding = next(
            b
            for b in operation.output_artifact_bindings
            if b.slot_id == "grasp_observation"
        )
        result_hash = canonical_content_hash(
            operation.execution_result,
            domain="dexteleop.agent_gateway.passive_read_result.v1",
        )
        state = metadata.get("native_scene_state")
        if (
            not isinstance(state, dict)
            or set(state) != {"model_qpos", "model_qvel", "simulation_time_sec"}
            or state != self.application.capture_grasp_scene_state()
        ):
            raise PassiveReadError(
                "Frozen native scene no longer matches the robot, hands or objects.",
                reason_code="grasp_fast_capture_scene_changed",
                diagnostic={"source_operation_id": identifier},
            )
        return operation, views, state, result_hash, binding.artifact.sha256

    def _check_scene(self, state, revision):
        app = self.application
        if (
            self._closed
            or not app.started
            or not app.held
            or app.active_operation_id is not None
            or self.current_station_revision() != revision
            or app.capture_grasp_scene_state() != state
        ):
            raise PassiveReadError(
                "MuJoCo frozen grasp scene changed during planning.",
                reason_code="grasp_fast_capture_scene_changed",
                diagnostic={"station_revision": revision},
            )

    def _selector(self, *, parameters, deadline, cancelled):
        app, bundle, c = self.application, self.bundle, self.contract
        native = app.target.planned_tcp_motion
        profile = c.execution_contract.motion_profile
        current = xyzquat_to_pose_matrix(
            app.transform_tcp_pose_xyzquat(
                app.capture_tcp_pose_by_component()["right_arm"],
                source_frame_id=profile.execution_reference_frame_id,
                target_frame_id=profile.reference_frame_id,
            )
        )
        origin = xyzquat_to_pose_matrix(
            app.transform_tcp_pose_xyzquat(
                (0, 0, 0, 0, 0, 0, 1),
                source_frame_id=profile.execution_reference_frame_id,
                target_frame_id=profile.reference_frame_id,
            )
        )[:3, 3]
        contracts = (bundle.grasp_approach_contract, bundle.grasp_contact_contract)
        constraints = [
            mujoco_grasp_configuration_constraint(app, bundle, v) for v in contracts
        ]
        result_box = {
            "result": None,
            "candidate": None,
            "pool_count": 0,
            "query_details": [],
        }

        def priority(reference_pose, score, target):
            if cancelled():
                raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
            if self.monotonic() >= deadline:
                raise MujocoTcpPlanRejected("safe_tcp_planning_budget_exhausted")
            pose = rigid_pose(
                np.asarray(reference_pose).reshape(4, 4), name="candidate pose"
            )
            p = c.policy
            distance = float(np.linalg.norm(pose[:3, 3] - current[:3, 3]))
            rotation = np.degrees(
                np.arccos(
                    np.clip((np.trace(current[:3, :3].T @ pose[:3, :3]) - 1) / 2, -1, 1)
                )
            )
            direction = np.asarray(target) - origin
            norm = np.linalg.norm(direction)
            angle = (
                0
                if norm < 1e-9
                else np.degrees(
                    np.arccos(np.clip(np.dot(pose[:3, 2], direction / norm), -1, 1))
                )
            )
            return (
                p.translation_weight * distance / 0.5
                + p.orientation_weight * rotation / 180
                + p.arm_side_weight * angle / 90
                + p.model_score_weight * (1 - np.clip(float(score), 0, 1))
            )

        def select(candidates, target, maximum):
            del maximum
            result_box["pool_count"] = len(candidates)
            costs = [
                (
                    *grasp_search_preference(v,
                        priority(v.base_from_tcp, v.score, target), policy=c.policy),
                    i,
                )
                for i, v in enumerate(candidates)
                if v.visible_collision_free and grasp_capture_height_acceptable(v, policy=c.policy)
            ]
            costs.sort()
            order = refined_grasp_search_order(
                candidates,
                [i for _, _, i in costs],
                policy=c.policy,
            )

            pending = [None]

            def validate_executable(plan):
                if (len(plan.reference_poses) + profile.stable_measurement_count >= profile.max_control_cycles
                        or float(plan.trajectory.timestamps[-1]) >= c.execute.entry.route.max_operation_duration_ms / 1000.0):
                    raise GraspPathQueryRejected("grasp_fast_path_exceeds_execution_budget")

            def query(candidate, budget, *, original_plan=None, refinement_deadline=None):
                query_detail = {}
                result_box["query_details"].append(query_detail)
                if cancelled():
                    raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
                pose = rigid_pose(
                    np.asarray(candidate.base_from_tcp).reshape(4, 4),
                    name="candidate pose",
                )
                goals = []
                config = c.approach_config
                for offset, contract_index, speed in (
                    (-config["pregrasp_distance_m"], 0, parameters["speed_scale"]),
                    (-config["near_distance_m"], 0, parameters["speed_scale"]),
                    (
                        config["terminal_offset_m"],
                        1,
                        parameters["speed_scale"] * config["terminal_speed_scale"],
                    ),
                ):
                    target_pose = pose.copy()
                    target_pose[:3, 3] += pose[:3, 2] * offset
                    base_pose = app.transform_tcp_pose_xyzquat(
                        tuple(pose_to_xyz_quat_xyzw(target_pose)),
                        source_frame_id=profile.reference_frame_id,
                        target_frame_id=profile.execution_reference_frame_id,
                    )
                    stage_contract = contracts[contract_index]
                    constraint, identity = constraints[contract_index]
                    goals.append(
                        MujocoTcpPlanGoal(
                            base_pose,
                            stage_contract.motion_profile.translation_tolerance_m,
                            stage_contract.motion_profile.rotation_tolerance_rad,
                            configuration_constraint=constraint,
                            constraint_identity=identity,
                            contact_binding=stage_contract.contact_binding,
                            speed_scale=speed,
                        )
                    )
                try:
                    query_kwargs = {"cancel_requested": cancelled, "time_budget_sec": budget,
                        "monotonic": self.monotonic}
                    if original_plan is None:
                        plan = native.plan_chain(goals, **query_kwargs)
                        try:
                            validate_executable(plan)
                        except GraspPathQueryRejected:
                            native.discard_pending_plan(plan)
                            raise
                    else:
                        budget = min(budget, refinement_deadline - self.monotonic())
                        if budget <= 0:
                            raise GraspPathQueryRejected("safe_tcp_planning_budget_exhausted")
                        plan = native.refine_pending_chain(original_plan, goals,
                            validate_plan=validate_executable, deadline_at=refinement_deadline,
                            **{**query_kwargs, "time_budget_sec": budget})
                    pending[0] = plan
                    return plan
                except MujocoTcpPlanRejected as error:
                    query_detail.update({
                        "failed_stage": error.details.get("chain_stage_index"),
                        "lower_reason": error.details.get("reason"),
                        "endpoint_attempt_count": len(
                            error.details.get("endpoint_attempts", ())
                        ),
                    })
                    candidate_start = (
                        error.reason_code == "safe_tcp_start_blocked"
                        and error.details.get("chain_stage_index", 0) > 0
                    )
                    if error.reason_code in _GOAL_REJECTIONS or candidate_start:
                        raise GraspPathQueryRejected(error.reason_code) from error
                    raise

            result = find_first_grasp_path(
                candidates=candidates,
                ordered_indices=order,
                plan=query,
                policy=c.policy,
                deadline=deadline,
                original_neighbors=original_grasp_neighbors(candidates, order, policy=c.policy),
                monotonic=self.monotonic,
            )
            original_plan = result.trajectory
            try:
                result = refine_first_grasp_path(result, candidates=candidates,
                    eligible_indices=tuple(i for _, _, i in costs),
                    plan=lambda candidate, budget, refinement_deadline: query(candidate, budget,
                        original_plan=original_plan, refinement_deadline=refinement_deadline),
                    policy=c.policy, deadline=deadline, monotonic=self.monotonic)
            except BaseException:
                if pending[0] is not None:
                    native.discard_pending_plan(pending[0])
                raise
            if result.trajectory is None and pending[0] is not None:
                native.discard_pending_plan(pending[0])
            result_box["result"] = result
            if result.selected_index is not None:
                result_box["candidate"] = candidates[result.selected_index]
            return GraspGenXCandidateSelection(
                () if result.selected_index is None else (result.selected_index,),
                {"arm_shortlist_full_path_query_count": len(result.attempts)},
            )

        return select, priority, result_box

    def plan_request(self, request, operation, *, cancelled=lambda: False):
        started = self.monotonic()
        p, c, app = request.parameters, self.contract, self.application
        c.fast.payload_validator.validate_parameters(p)
        if request.input_artifact_bindings:
            raise ValueError(
                "Fast planning uses only the retained observation reference."
            )
        with self._lock:
            budget = min(
                float(p["planning_budget_sec"]),
                c.policy.total_budget_sec,
                max(0.0, operation.deadline_at - float(self.clock())),
            )
            deadline = started + budget
            if self.model.semantic_content_hash != c.model_profile_hash:
                raise ValueError("GraspGenX deployment configuration changed.")
            source, encoded, state, result_hash, archive_hash = self._source(
                p["observation_operation_id"], request.expected_station_revision
            )
            self._check_scene(state, request.expected_station_revision)
            decoded = [decode_grasp_rgbd_snapshot(data) for data in encoded]
            selected = next(
                ((a, m) for a, m in decoded if m["role"] == p["role"]), None
            )
            if selected is None:
                raise ValueError("Requested role is not in this frozen observation.")
            arrays, metadata = selected
            u, v = p["target_pixel_uv"]
            if (
                not 0 <= u < arrays["rgb"].shape[1]
                or not 0 <= v < arrays["rgb"].shape[0]
            ):
                raise ValueError("Target pixel is outside the retained native image.")
            reference_from_camera = rigid_pose(
                metadata["T_reference_from_camera"], name="captured optical transform"
            )
            camera_from_reference = np.linalg.inv(reference_from_camera)
            views = {
                m["role"]: GraspRgbdView(
                    m["role"],
                    a["rgb"],
                    a["depth_m"],
                    m["camera_intrinsics"],
                    camera_from_reference
                    @ rigid_pose(
                        m["T_reference_from_camera"], name="captured optical transform"
                    ),
                    m["acquired_at"],
                )
                for a, m in decoded
            }

            def preparation_guard():
                if cancelled():
                    raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
                if self.monotonic() >= deadline:
                    raise MujocoTcpPlanRejected("safe_tcp_planning_budget_exhausted")

            def segment_auxiliary(*args, **kwargs):
                preparation_guard()
                result = self.model.point_segmenter.segment_point(*args, **kwargs)
                preparation_guard()
                return result

            def visible_scene(*, depth_m, target_mask, intrinsics, geometry):
                del depth_m, intrinsics
                return fuse_grasp_views(
                    anchor=views[p["role"]],
                    anchor_mask=target_mask,
                    auxiliary=(
                        tuple(view for role, view in views.items() if role != p["role"])
                        if p["view_policy"] == "auto"
                        else ()
                    ),
                    anchor_pixel_uv=(u, v),
                    geometry=geometry,
                    segment_point=segment_auxiliary,
                ).scene

            selector, priority, box = self._selector(
                parameters=p, deadline=deadline, cancelled=cancelled
            )
            native_plan = None
            reason = "grasp_fast_search_budget_exhausted"
            try:
                if budget > 0 and self.monotonic() < deadline and not cancelled():
                    try:
                        self.model.plan(
                            Image.fromarray(arrays["rgb"]),
                            arrays["depth_m"],
                            pixel_uv=(u, v),
                            intrinsics=metadata["camera_intrinsics"],
                            base_from_camera=reference_from_camera,
                            max_candidates=1,
                            candidate_selector=selector,
                            proposal_priority=priority,
                            visible_scene_factory=visible_scene,
                            use_visible_target_points=True,
                            refine_grasp_centers=True,
                            refine_grasp_axial_depth=True,
                            preparation_guard=preparation_guard,
                        )
                    except PointGraspPlanningError as error:
                        reason = error.reason_code
                    except MujocoTcpPlanRejected as error:
                        if error.reason_code != "safe_tcp_planning_budget_exhausted":
                            raise
                        reason = "grasp_fast_search_budget_exhausted"
                search = box["result"]
                if search is not None:
                    native_plan, reason = search.trajectory, search.reason_code
                self._check_scene(state, request.expected_station_revision)
                if cancelled():
                    raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
                if self.monotonic() >= deadline:
                    if native_plan is not None:
                        app.target.planned_tcp_motion.discard_pending_plan(native_plan)
                    native_plan, reason = None, "grasp_fast_search_budget_exhausted"
                profile = c.execution_contract.motion_profile
                result = {
                    "planning_status": "rejected",
                    "reason_code": reason,
                    "component_id": profile.component_id,
                    "side": "right",
                    "observation_operation_id": source.operation_id,
                    "observation_content_hash": result_hash,
                    "role": p["role"],
                    "target_pixel_uv": list(p["target_pixel_uv"]),
                    **{
                        n: getattr(profile, n)
                        for n in (
                            "reference_frame_id",
                            "reference_frame_spec_hash",
                            "frame_graph_hash",
                            "tool_config_hash",
                            "calibration_hash",
                        )
                    },
                    "source_station_revision": request.expected_station_revision,
                    **{
                        n: None
                        for n in (
                            "T_reference_from_tool_tcp",
                            "gripper_width_m",
                            "grasp_approach_plan_id",
                            "grasp_approach_plan_hash",
                            "trajectory_hash",
                            "trajectory_sample_count",
                            "trajectory_duration_sec",
                            "plan_created_at",
                            "plan_expires_at",
                        )
                    },
                    "eligible_for_execution": False,
                    "execution_requires_explicit_motion_skill": True,
                    "motion_or_hold_performed": False,
                    "planning_policy": "first_acceptable_v1",
                    "diagnostics": {
                        "pool_count": box["pool_count"],
                        "path_query_count": (
                            0 if search is None else len(search.attempts)
                        ),
                        "elapsed_sec": max(0.0, self.monotonic() - started),
                        "attempts_json": grasp_path_attempts_json(
                            () if search is None else search.attempts
                        ),
                    },
                }
                if search is not None:
                    _LOGGER.info(
                        "Native fast grasp path attempts: %s",
                        json.dumps({
                            "operation_id": operation.operation_id,
                            "reason_code": search.reason_code,
                            "attempts": [
                                dict(row, **detail)
                                for row, detail in zip(
                                    search.attempts, box["query_details"], strict=True
                                )
                            ],
                        }, separators=(",", ":")),
                    )
                if native_plan is not None:
                    created = float(self.clock())
                    expires = min(created + 30.0, operation.deadline_at)
                    terminal = app.transform_tcp_pose_xyzquat(
                        native_plan.reference_poses[-1],
                        source_frame_id=profile.execution_reference_frame_id,
                        target_frame_id=profile.reference_frame_id,
                    )
                    identity = {
                        "session": source.runtime_session_id,
                        "profile": c.profile_hash,
                        "planning_operation_id": operation.operation_id,
                        "planning_request_hash": request.semantic_content_hash,
                        "observation_operation_id": source.operation_id,
                        "observation_request_hash": source.request_hash,
                        "observation_result_hash": result_hash,
                        "observation_archive_sha256": archive_hash,
                        "source_scene_hash": _scene_hash(state),
                        "station_revision": request.expected_station_revision,
                        "trajectory_hash": native_plan.trajectory_hash,
                        "created_at": created,
                        "expires_at": expires,
                    }
                    plan_hash = canonical_content_hash(
                        identity, domain="dexteleop.mujoco.retained_grasp_plan.v1"
                    )
                    retained = MujocoRetainedGraspPlan(
                        native_plan,
                        operation.operation_id,
                        request.semantic_content_hash,
                        source.operation_id,
                        source.request_hash,
                        result_hash,
                        archive_hash,
                        _scene_hash(state),
                        request.expected_station_revision,
                        c.profile_hash,
                        source.runtime_session_id,
                        created,
                        expires,
                        plan_hash,
                    )
                    result.update(
                        planning_status="path_found",
                        reason_code="path_found",
                        eligible_for_execution=True,
                        T_reference_from_tool_tcp=xyzquat_to_pose_matrix(terminal)
                        .ravel()
                        .tolist(),
                        gripper_width_m=float(box["candidate"].gripper_width_m),
                        grasp_approach_plan_id=retained.plan_id,
                        grasp_approach_plan_hash=plan_hash,
                        trajectory_hash=native_plan.trajectory_hash,
                        trajectory_sample_count=len(native_plan.reference_poses),
                        trajectory_duration_sec=float(
                            native_plan.trajectory.timestamps[-1]
                        ),
                        plan_created_at=created,
                        plan_expires_at=expires,
                    )
                    self._check_scene(state, request.expected_station_revision)
                    if (
                        cancelled()
                        or self.monotonic() >= deadline
                        or float(self.clock()) >= expires
                    ):
                        raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
                    self._retained = retained
                c.fast.payload_validator.validate_execution_result(result)
                c.fast.payload_validator.validate_execution_result_for_parameters(
                    p, result
                )
                return PassiveReadResult(
                    execution_result=result,
                    reason_code=result["reason_code"],
                    evidence_refs=(
                        "mujoco-first-grasp-plan:" + operation.operation_id,
                    ),
                )
            except BaseException:
                if native_plan is None and box["result"] is not None:
                    native_plan = box["result"].trajectory
                if native_plan is not None:
                    app.target.planned_tcp_motion.discard_pending_plan(native_plan)
                raise

    def require_plan(self, request):
        """Resolve only a completed, same-scene, live native pending handle."""
        with self._lock:
            c, plan = self.contract, self._retained
            c.execute.payload_validator.validate_parameters(request.parameters)
            if (
                self._closed
                or plan is None
                or request.runtime_session_id != plan.runtime_session_id
                or request.expected_station_revision != plan.station_revision
                or request.parameters["grasp_approach_plan_id"] != plan.plan_id
                or request.parameters["grasp_approach_plan_hash"] != plan.content_hash
                or plan.profile_hash != c.profile_hash
            ):
                raise MujocoTcpPlanRejected("grasp_path_plan_binding_mismatch")
            now = float(self.clock())
            if not plan.created_at <= now < plan.expires_at:
                raise MujocoTcpPlanRejected("grasp_path_plan_expired")
            producer = self.operation_lookup(plan.planning_operation_id)
            if (
                producer is None
                or producer.status != "completed"
                or producer.execution_outcome != "completed"
                or producer.runtime_session_id != plan.runtime_session_id
                or producer.request_hash != plan.planning_request_hash
                or not isinstance(producer.execution_result, Mapping)
                or producer.execution_result.get("grasp_approach_plan_id")
                != plan.plan_id
                or producer.execution_result.get("grasp_approach_plan_hash")
                != plan.content_hash
                or producer.execution_result.get("eligible_for_execution") is not True
            ):
                raise MujocoTcpPlanRejected("grasp_path_plan_not_committed")
            try:
                source, _, state, result_hash, archive_hash = self._source(
                    plan.observation_operation_id, plan.station_revision
                )
            except (ValueError, FileNotFoundError, PassiveReadError) as error:
                raise MujocoTcpPlanRejected(
                    "grasp_path_plan_source_unavailable",
                    {
                        "source_operation_id": plan.observation_operation_id,
                        "source_error": str(error)[:512],
                    },
                ) from error
            if (
                source.request_hash != plan.observation_request_hash
                or result_hash != plan.observation_result_hash
                or archive_hash != plan.observation_archive_sha256
                or _scene_hash(state) != plan.source_scene_hash
                or self.current_station_revision() != plan.station_revision
            ):
                raise MujocoTcpPlanRejected("grasp_path_plan_source_changed")
            if not self.application.target.planned_tcp_motion.owns_pending_plan(
                plan.native_plan
            ):
                raise MujocoTcpPlanRejected("grasp_path_plan_no_longer_pending")
            return plan


class _CheckedPreparedGraspLifecycle(MujocoAgentPreparedTcpPoseSkillLifecycle):
    """Recheck immediately before the native pending handle is activated."""

    def __init__(self, *, plan_provider, **kwargs):
        self._plan_provider = plan_provider
        self._retained_plan = plan_provider.require_plan(kwargs["request"])
        super().__init__(prepared_plan=self._retained_plan.native_plan, **kwargs)

    def _prepare_plan(self, **kwargs):
        if self._plan_provider.require_plan(self.request) is not self._retained_plan:
            raise MujocoTcpPlanRejected("grasp_path_plan_changed")
        return super()._prepare_plan(**kwargs)

    def _activate_plan(self, plan):
        self._planner.activate(
            plan,
            deadline_at=self._retained_plan.expires_at,
            clock=self._plan_provider.clock,
        )


class MujocoPairedGraspPlanLifecycle:
    """Resolve provenance in the worker, so rejection gets a terminal receipt.

    Construction cannot fail due to a stale/consumed/expired reference before
    the worker exists. Once resolved, the existing prepared lifecycle owns all
    execution, evaluation and lower HOLD; this adapter owns no motion state.
    """

    def __init__(self, *, plan_provider, **kwargs):
        self.executor_id = kwargs["deployment_binding"].executor_id
        self._provider, self._kwargs = plan_provider, kwargs
        self._delegate = None
        self._lock = threading.RLock()

    def activate(self, *, request, operation, cancel_requested):
        with self._lock:
            try:
                if cancel_requested():
                    raise MujocoTcpPlanRejected("grasp_path_execution_cancelled")
                self._delegate = _CheckedPreparedGraspLifecycle(
                    plan_provider=self._provider, **self._kwargs
                )
            except MujocoTcpPlanRejected as error:
                hold = self.request_hold(operation=operation, reason=error.reason_code)
                return StationLifecycleActivationEvidence(
                    succeeded=False,
                    status=error.reason_code,
                    lower_operation_id=hold.lower_operation_id,
                    evidence_refs=hold.evidence_refs,
                    lower_hold_confirmed=hold.lower_hold_confirmed,
                )
            return self._delegate.activate(
                request=request, operation=operation, cancel_requested=cancel_requested
            )

    def run(self, **kwargs):
        if self._delegate is None:
            raise RuntimeError("A rejected immutable plan cannot run.")
        return self._delegate.run(**kwargs)

    def cancel(self, *, request, operation, reason):
        if (
            request.semantic_content_hash
            != self._kwargs["request"].semantic_content_hash
        ):
            raise ValueError("Retained grasp cancellation request changed.")
        return self.request_hold(operation=operation, reason=reason)

    def request_hold(self, *, operation, reason):
        with self._lock:
            if self._delegate is not None:
                return self._delegate.request_hold(operation=operation, reason=reason)
            return self._kwargs["domain_hold"](operation=operation, reason=reason)

    def close(self):
        with self._lock:
            if self._delegate is not None:
                self._delegate.close()


def build_mujoco_fast_grasp_executor(
    *, provider, current_station_revision, clock, max_retained_operations
):
    executor = None

    def read(request, operation):
        return provider.plan_request(
            request,
            operation,
            cancelled=lambda: executor.cancel_requested(operation.operation_id),
        )

    executor = PassiveReadSkillExecutor(
        port_id="agent-gateway:mujoco-grasp-fast:"
        + provider.application.runtime.config.runtime_session_id,
        executor_id=provider.contract.fast.executor_id,
        read=read,
        current_station_revision=current_station_revision,
        clock=clock,
        max_retained_operations=max_retained_operations,
    )
    return executor

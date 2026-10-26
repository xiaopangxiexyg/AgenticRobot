"""MuJoCo-only passive feasibility projection, with no motion/lifecycle port."""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from agent_gateway.profiles.query_geometry import validate_query_selection
from agent_gateway.profiles.query_geometry import (
    relation_candidate_poses,
    region_candidate_poses,
    validate_relation_continuation_geometry_parameters,
    validate_relation_continuation_path,
    validate_region_geometry_parameters,
    validate_region_continuation_path,
)
from agent_gateway.profiles.spatial_goal import SpatialGoalRegistration
from agent_gateway.protocol.skill_catalog import SkillRouteManifestEntry
from agent_gateway.protocol.skill_io_contracts import inline_only_skill_io_contract
from agent_gateway.protocol.skill_registration import SkillRegistrationTemplate
from agent_gateway.runtime.passive_read_executor import (
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorSet
from agent_gateway.skills.perception.object_goal_query import (
    RANKING,
    RELATION_FIELDS,
    SKILL_ID,
    object_goal_query_definition,
    object_goal_query_schema,
    validate_query_parameters,
    validate_query_result,
)
from agent_gateway.skills.perception.schemas import validate_no_task_evaluation_v1
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.path_planning import JointConfigurationCheck
from robot_motion.relative_action import xyzquat_to_pose_matrix
from robot_motion.spatial_goals import (
    generate_object_above_goals,
    generate_relation_goals,
)
from sim.mujoco.deploy.planned_tcp_motion import (
    CANDIDATE_SCHEDULING_POLICY,
    MujocoJointSceneOracle,
    MujocoPlannedTcpMotion,
    MujocoTcpPlanRejected,
)


def _candidate_diagnostic(item, pose):
    """Keep exact proposals/path evidence and a bounded endpoint-search summary."""
    row = {**item, "pose_target_xyz_quat_xyzw": list(pose)}
    details = row.get("details", {})
    search = details.get("endpoint_search_diagnostics")
    if search is not None:
        # The canonical solver retains its per-seed methods, iterations and
        # joint vectors below. Repeating them across 64 inline query rows can
        # exceed the existing result bound. Keep every actual residual pair,
        # the tolerances and coverage; this does not certify a missing path.
        row["details"] = {
            "endpoint_error_pairs_m_rad": [
                [attempt["position_error_m"], attempt["rotation_error_rad"]]
                for attempt in search["attempts"]
            ],
            **{key: search[key] for key in (
                "seed_count", "attempted_seed_count", "deadline_exceeded",
                "translation_tolerance_m", "rotation_tolerance_rad")},
            **({"endpoint_check_reasons": list(dict.fromkeys(details["endpoint_check_reasons"]))}
               if details.get("endpoint_check_reasons") else {}),
            **({"motion_quality_limits": details["motion_quality_limits"]}
               if item["reason_code"] == "safe_tcp_motion_quality_exceeded" else {}),
        }
    if "scheduling_preview" in row:
        # A failed preview has no cost estimate. Repeating two null estimates
        # for every candidate can overflow the existing dense-query budget.
        row["scheduling_preview"] = {
            key: value for key, value in row["scheduling_preview"].items()
            if value is not None or key not in {
                "estimated_max_joint_excursion_rad", "estimated_total_joint_excursion_rad"}
        }
    return row


def build_object_goal_query_registration(
    *, base_route, base_binding, reference_frame_id, planning_profile_hash
):
    if base_route.control_effect_class != "read_only":
        raise ValueError("Object goal queries require a passive base route.")
    identity = canonical_content_hash(
        {"frame": reference_frame_id, "planning_profile": planning_profile_hash,
         "candidate_scheduling": CANDIDATE_SCHEDULING_POLICY},
        domain="agenticrobot.object_goal_query.profile.v1",
    )
    definition = object_goal_query_definition()
    schema = object_goal_query_schema(
        definition_hash=definition.semantic_content_hash,
        profile_hash=identity,
        reference_frame_id=reference_frame_id,
    )
    io = inline_only_skill_io_contract(
        skill_id=SKILL_ID,
        skill_version="1",
        skill_definition_hash=definition.semantic_content_hash,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
    )
    route = replace(
        base_route,
        skill_id=SKILL_ID,
        skill_version="1",
        skill_definition_hash=definition.semantic_content_hash,
        route_id=base_route.route_id + ".object_goal_query",
        route_version="1",
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema.parameter_profiles,
        skill_io_contract_hash=io.semantic_content_hash,
        max_operation_duration_ms=40_000,
    )
    binding = base_binding.to_ref() if hasattr(base_binding, "to_ref") else base_binding
    binding = replace(
        binding,
        deployment_binding_id="object-goal-query:" + binding.deployment_binding_id,
        deployment_binding_hash=canonical_content_hash(
            {
                "base": binding.deployment_binding_hash,
                "route": route.semantic_content_hash,
            },
            domain="agenticrobot.object_goal_query.binding.v1",
        ),
    )

    def parameters(value):
        validate_query_parameters(value)
        generate_object_above_goals(current_tcp_pose_xyz_quat_xyzw=[0., 0., 0., 0., 0., 0., 1.],
            **{key: value[key] for key in ("anchor_xyz_m", "heights_m", "tool_yaw_offsets_rad")})
        if (
            value["component_id"] != "right_arm"
            or value["reference_frame_id"] != reference_frame_id
        ):
            raise ValueError(
                "Query component/frame differs from the deployment binding."
            )

    def result_for(value, result):
        parameters(value)
        validate_query_result(result)
        candidates = generate_object_above_goals(
            current_tcp_pose_xyz_quat_xyzw=result["source_tcp_pose_xyz_quat_xyzw"],
            **{
                key: value[key]
                for key in ("anchor_xyz_m", "heights_m", "tool_yaw_offsets_rad")
            },
        )
        validate_query_selection(value, result, candidates)

    validator = AgentSkillPayloadValidatorSet(
        validator_id="object-goal-query:" + identity,
        skill_id=SKILL_ID,
        skill_version="1",
        route_id=route.route_id,
        route_version="1",
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        validate_parameters=parameters,
        validate_execution_result=validate_query_result,
        validate_execution_result_for_parameters=result_for,
        validate_task_evaluation=validate_no_task_evaluation_v1,
    )
    return SpatialGoalRegistration(
        SkillRegistrationTemplate(definition, schema, io, (route,)),
        SkillRouteManifestEntry(
            route=route,
            deployment_binding=binding,
            evaluation_plan_hash=None,
            availability_hash=identity,
        ),
        validator,
    )


class MujocoObjectGoalQuery:
    """Only snapshot, FK/frame and prepared safety-preview capabilities enter here."""

    def __init__(
        self,
        *,
        scene,
        rig_config,
        task_config,
        pose_projector,
        reference_hz,
        capture_scene,
        transform_pose,
        prepare_projection,
        profile,
        clearance_padding_m,
    ):
        self._scene = scene
        self._rig_config = rig_config
        self._task_config = task_config
        self._pose_projector = pose_projector
        self._reference_hz = reference_hz
        self._capture_scene = capture_scene
        self._transform_pose = transform_pose
        self._prepare_projection = prepare_projection
        self._profile = profile
        self._padding = clearance_padding_m
        self._lock = threading.Lock()

    def query(self, parameters, *, cancel_requested):
        return self._query(parameters, cancel_requested=cancel_requested,
            candidates_for=lambda initial: generate_object_above_goals(
                current_tcp_pose_xyz_quat_xyzw=initial,
                **{key: parameters[key] for key in ("anchor_xyz_m", "heights_m", "tool_yaw_offsets_rad")}))

    def query_relations(self, parameters, *, cancel_requested):
        """General relation samples use exactly the same frozen-scene planning owner."""
        return self._query(parameters, cancel_requested=cancel_requested,
            candidates_for=lambda initial: generate_relation_goals(
                current_tcp_pose_xyz_quat_xyzw=initial, anchor_xyz_m=parameters["anchor_xyz_m"],
                **{key: parameters[key] for key in RELATION_FIELDS
                   if key not in {"component_id", "reference_frame_id", "time_budget_sec"}}))
    def query_continuations(self, parameters, *, cancel_requested):
        from agent_gateway.skills.perception.tcp_continuation_query import (
            validate_continuation_parameters,
        )

        validate_continuation_parameters(parameters)
        return self._query_continuations(parameters, cancel_requested=cancel_requested)

    def query_relation_continuations(self, parameters, *, cancel_requested):
        """Generate endpoints only after capturing the same scene used for both paths."""
        from agent_gateway.skills.perception.relation_continuation_query import (
            explicit_continuation_parameters,
            validate_relation_continuation_parameters,
        )

        request = {key: value for key, value in parameters.items() if key != "anchor_xyz_m"}
        validate_relation_continuation_parameters(request)
        validate_relation_continuation_geometry_parameters(request)
        result = self._query_continuations(
            explicit_continuation_parameters(request, []), cancel_requested=cancel_requested,
            candidates_for=lambda initial: relation_candidate_poses(request, initial=initial,
                anchor_xyz_m=parameters["anchor_xyz_m"]),
        )
        validate_relation_continuation_path(request, result, anchor_xyz_m=parameters["anchor_xyz_m"])
        return result

    def query_region_continuation_extrema(self, parameters, *, cancel_requested):
        """Explicit V5 sampler; preserve the same frozen-scene planning owner."""
        return self.query_region_continuations(parameters, cancel_requested=cancel_requested, version="5")

    def query_region_continuations(self, parameters, *, cancel_requested, version="4"):
        """Sample an allowed region inside the same frozen-scene query boundary."""
        from agent_gateway.skills.perception.region_continuation_query import (
            validate_region_continuation_parameters,
        )
        from agent_gateway.skills.perception.relation_continuation_query import (
            explicit_continuation_parameters,
        )

        request = {key: value for key, value in parameters.items() if key != "anchor_xyz_m"}
        validate_region_continuation_parameters(request, version=version)
        validate_region_geometry_parameters(request, version=version)
        result = self._query_continuations(
            explicit_continuation_parameters(request, []), cancel_requested=cancel_requested,
            candidates_for=lambda initial: region_candidate_poses(request, initial=initial,
                anchor_xyz_m=parameters["anchor_xyz_m"], version=version),
        )
        validate_region_continuation_path(request, result, anchor_xyz_m=parameters["anchor_xyz_m"], version=version)
        return result

    def _query_continuations(self, parameters, *, cancel_requested, candidates_for=None):
        """Check two paths on one frozen scene, propagating the first planned qpos.

        Only private planning contexts are constructed. No prediction is promoted
        to a measured observation, pending executable plan, or motion authority.
        """
        from agent_gateway.skills.perception.tcp_continuation_query import (
            validate_continuation_parameters,
            validate_continuation_result_for_parameters,
        )
        from robot_motion.path_constraints import ToolAxisCone
        from robot_motion.spatial_goals import compute_spatial_goal

        if (parameters["component_id"] != "right_arm"
                or parameters["reference_frame_id"] != self._profile.reference_frame_id):
            raise ValueError("Continuation query differs from its bound component/frame.")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Object path query is already running.")
        planners = []
        started = time.perf_counter()
        deadline = started + parameters["time_budget_sec"]

        def cancelled():
            return cancel_requested() or time.perf_counter() >= deadline

        def planner_for(qpos):
            planner = MujocoPlannedTcpMotion(
                rig=SimpleNamespace(scene=self._scene, config=self._rig_config, capture_scene=self._capture_scene),
                task_config=self._task_config, pose_projector=self._pose_projector,
                reference_hz=self._reference_hz, planning_clearance_padding_m=self._padding,
                source_model_qpos=qpos)
            planners.append(planner)
            return planner

        def public(pose):
            return self._transform_pose(pose,
                source_frame_id=self._profile.execution_reference_frame_id,
                target_frame_id=self._profile.reference_frame_id)

        def execution(pose):
            return self._transform_pose(pose,
                source_frame_id=self._profile.reference_frame_id,
                target_frame_id=self._profile.execution_reference_frame_id)

        try:
            if cancel_requested():
                raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
            project, envelope_identity = self._prepare_projection()
            qpos, sim_time = self._capture_scene(
                lambda scene: (scene.data.qpos.copy(), float(scene.data.time)))
            planner = planner_for(qpos)
            addresses = {side: [self._scene.joint_qpos_addr(name)
                for name in self._scene.arm_specs[side].joint_names] for side in ("left", "right")}
            pair = {side: qpos[indices].copy() for side, indices in addresses.items()}
            initial = public(planner.pose_for_configuration(qpos, pair))
            if candidates_for is not None:
                parameters = {**parameters, "first_pose_candidates_xyz_quat_xyzw": candidates_for(initial)}
            validate_continuation_parameters(parameters)

            def constraint_for(owner, cone=None):
                def constraint(model_qpos, current):
                    pose = owner.pose_for_configuration(model_qpos, current)
                    projection = project({"right": xyzquat_to_pose_matrix(pose)})
                    if projection.modified_sides:
                        return JointConfigurationCheck(safe=False, reason="tcp_envelope_blocked",
                            metadata=projection.to_payload())
                    if cone is not None:
                        # Match safe TCP v2's exact float32 publication check.
                        published = owner.pose_for_configuration(model_qpos, {
                            side: np.asarray(q, dtype=np.float32) for side, q in current.items()})
                        deviation = max(cone.deviation_rad(pose), cone.deviation_rad(published))
                        if deviation > cone.max_deviation_rad + 1e-9:
                            return JointConfigurationCheck(safe=False, reason="tcp_tool_axis_cone_blocked",
                                metadata={"deviation_rad": deviation,
                                          "max_deviation_rad": cone.max_deviation_rad})
                    return JointConfigurationCheck(safe=True)
                return constraint

            first_constraint = constraint_for(planner)
            scene_hash = MujocoJointSceneOracle(self._scene, qpos,
                clearance_padding_m=self._padding, configuration_constraint=first_constraint,
                constraint_identity=envelope_identity).scene_hash
            diagnostics, best = [], None
            ordinary_rejections = {
                "safe_tcp_endpoint_search_exhausted", "safe_tcp_target_blocked",
                "safe_tcp_path_search_exhausted", "safe_tcp_retimed_path_blocked",
                "safe_tcp_path_exceeds_control_budget", "safe_tcp_target_envelope_blocked",
                "safe_tcp_target_exceeds_profile", "safe_tcp_candidate_rank_pruned",
                "safe_tcp_motion_quality_exceeded",
            }
            for index, target in enumerate(parameters["first_pose_candidates_xyz_quat_xyzw"]):
                if cancel_requested():
                    raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
                if time.perf_counter() >= deadline:
                    break
                candidate_started = time.perf_counter()
                stage_diagnostics, plans = [], []
                predicted, followup = None, None
                failed_stage, reason = None, "path_found"
                stage_planner, source, start_pose = planner, qpos, initial
                constraint, identity = first_constraint, envelope_identity
                for stage_index in (0, 1):
                    stage_started = time.perf_counter()
                    try:
                        relative = compute_spatial_goal(current_tcp_pose_xyz_quat_xyzw=start_pose,
                            anchor_xyz_m=target[:3], offset_xyz_m=[0, 0, 0],
                            offset_frame="reference", orientation_mode="preserve")
                        rotation = xyzquat_to_pose_matrix(start_pose)[:3, :3].T @ xyzquat_to_pose_matrix(target)[:3, :3]
                        angle = float(np.arccos(np.clip((np.trace(rotation)-1)/2, -1, 1)))
                        if (relative.translation_distance_m > self._profile.max_translation_delta_m
                                or angle > self._profile.max_rotation_delta_rad):
                            raise MujocoTcpPlanRejected("safe_tcp_target_exceeds_profile")
                        # Only the first stage shares q0 with the full-chain
                        # cost. A second-stage local bound would be unsound.
                        rank_bound = ({"incumbent_max_joint_excursion_rad": best[0][0][0]}
                            if stage_index == 0 and best is not None else {})
                        plan = stage_planner.plan(tuple(execution(target)),
                            translation_tolerance_m=self._profile.translation_tolerance_m,
                            rotation_tolerance_rad=self._profile.rotation_tolerance_rad,
                            normalize_ik_objective=self._profile.normalize_ik_objective,
                            cancel_requested=cancelled, deadline_monotonic=deadline,
                            path_search_time_budget_sec=min(3., max(1e-9, deadline-time.perf_counter())),
                            configuration_constraint=constraint, constraint_identity=identity,
                            **rank_bound)
                        if cancelled():
                            raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
                        if (len(plan.reference_poses) + self._profile.stable_measurement_count
                                >= self._profile.max_control_cycles):
                            raise MujocoTcpPlanRejected("safe_tcp_path_exceeds_control_budget")
                    except MujocoTcpPlanRejected as error:
                        if error.reason_code == "safe_tcp_planning_cancelled":
                            if cancel_requested():
                                raise
                            reason = "query_budget_exhausted"
                        elif (error.reason_code in ordinary_rejections
                              or stage_index == 1 and error.reason_code == "safe_tcp_start_blocked"):
                            reason = error.reason_code
                        else:
                            raise
                        failed_stage = "first" if stage_index == 0 else "continuation"
                        stage_diagnostics.append({"stage_index": stage_index, "reason_code": reason,
                            "wall_sec": time.perf_counter()-stage_started,
                            "details": dict(error.details)})
                        break
                    plans.append(plan)
                    stage_diagnostics.append({"stage_index": stage_index, "reason_code": "path_found",
                        "wall_sec": time.perf_counter()-stage_started,
                        "trajectory_hash": plan.trajectory_hash, "reference_count": len(plan.reference_poses),
                        "duration_sec": float(plan.trajectory.timestamps[-1]-plan.trajectory.timestamps[0])})
                    timing = getattr(plan, "phase_timing_sec", None)
                    details = {}
                    if timing is not None:
                        details["phase_timing_sec"] = dict(timing)
                    metadata = getattr(getattr(plan, "path", None), "metadata", None)
                    if metadata is not None:
                        details["path_search"] = {key: metadata[key] for key in (
                            "direct_path", "search_iterations", "search_node_count",
                            "unshortened_waypoint_count", "waypoint_count") if key in metadata}
                    if details:
                        stage_diagnostics[-1]["details"] = details
                    if stage_index == 0:
                        source = qpos.copy()
                        source[addresses["right"]] = plan.trajectory.q_rad[-1]
                        # Preserve all scene/object/other-arm coordinates exactly.
                        end_pair = {side: source[indices] for side, indices in addresses.items()}
                        predicted_execution = planner.pose_for_configuration(source, end_pair)
                        predicted = list(public(predicted_execution))
                        followup = [*map(float, np.asarray(predicted[:3]) + parameters["continuation_delta_xyz_m"]),
                                    *predicted[3:]]
                        target, start_pose = followup, predicted
                        stage_planner = planner_for(source)
                        policy = parameters["path_constraint"]
                        cone = None if policy is None else ToolAxisCone.from_pose(predicted_execution,
                            tool_axis=policy["tool_axis"], max_deviation_rad=policy["max_deviation_rad"])
                        constraint = constraint_for(stage_planner, cone)
                        identity = canonical_content_hash({"envelope": envelope_identity,
                            "cone": None if cone is None else cone.to_payload()},
                            domain="agenticrobot.tcp_continuation.constraint.v1")
                row = {"candidate_index": index, "reason_code": reason, "failed_stage": failed_stage,
                    "wall_sec": time.perf_counter()-candidate_started, "stage_diagnostics": stage_diagnostics}
                if len(plans) == 2:
                    q = np.concatenate([plan.trajectory.q_rad for plan in plans])
                    cost = (float(np.max(np.abs(q-q[0]))),
                            sum(item["duration_sec"] for item in stage_diagnostics),
                            float(np.sum(np.abs(np.diff(q, axis=0)))))
                    row["ranking_cost"] = list(cost)
                    if best is None or (cost, index) < best[0]:
                        best = ((cost, index), index, predicted, followup)
                diagnostics.append(row)
                if reason == "query_budget_exhausted":
                    break
            if cancel_requested():
                raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
            if not first_constraint(qpos, pair).safe:
                raise MujocoTcpPlanRejected("safe_tcp_query_envelope_changed")
            selected = None if best is None else best[1]
            result = {"component_id": "right_arm", "reference_frame_id": self._profile.reference_frame_id,
                "source_tcp_pose_xyz_quat_xyzw": list(initial), "scene_hash": scene_hash,
                "source_qpos_hash": canonical_content_hash(qpos.tolist(), domain="object-goal-query.qpos.v1"),
                "source_sim_time_sec": sim_time, "query_status": "no_path_found" if best is None else "path_found",
                "selected_index": selected,
                "first_pose_target_xyz_quat_xyzw": None if best is None else list(parameters["first_pose_candidates_xyz_quat_xyzw"][selected]),
                "predicted_first_tcp_pose_xyz_quat_xyzw": None if best is None else best[2],
                "continuation_pose_target_xyz_quat_xyzw": None if best is None else best[3],
                "candidate_count": len(parameters["first_pose_candidates_xyz_quat_xyzw"]),
                "attempted_count": len(diagnostics),
                "all_candidates_attempted": len(diagnostics) == len(parameters["first_pose_candidates_xyz_quat_xyzw"]),
                "candidate_diagnostics": diagnostics, "planning_wall_sec": time.perf_counter()-started,
                "ranking_criteria": RANKING, "continuation_source": "planned_first_endpoint",
                "constraint_scope": "frozen_scene_two_stage_sampled_paths",
                "motion_executed": False, "execution_authorized": False}
            validate_continuation_result_for_parameters(parameters, result)
            if candidates_for is not None:
                result["first_pose_candidates_xyz_quat_xyzw"] = parameters["first_pose_candidates_xyz_quat_xyzw"]
            return result
        finally:
            for planner in planners:
                planner.clear()
            self._lock.release()

    def _query(self, parameters, *, cancel_requested, candidates_for):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Object path query is already running.")
        planner = None
        try:
            started = time.perf_counter()
            project, constraint_hash = self._prepare_projection()
            qpos, sim_time = self._capture_scene(
                lambda scene: (scene.data.qpos.copy(), float(scene.data.time))
            )
            # This private planning context has no live rig, dispatch, HOLD or activation capability.
            planner = MujocoPlannedTcpMotion(
                rig=SimpleNamespace(scene=self._scene, config=self._rig_config, capture_scene=self._capture_scene),
                task_config=self._task_config,
                pose_projector=self._pose_projector,
                reference_hz=self._reference_hz,
                planning_clearance_padding_m=self._padding,
                source_model_qpos=qpos,
            )
            pair = {
                side: np.array(
                    [
                        qpos[self._scene.joint_qpos_addr(name)]
                        for name in self._scene.arm_specs[side].joint_names
                    ]
                )
                for side in ("left", "right")
            }
            initial = self._transform_pose(
                planner.pose_for_configuration(qpos, pair),
                source_frame_id=self._profile.execution_reference_frame_id,
                target_frame_id=self._profile.reference_frame_id,
            )
            candidates = candidates_for(initial)
            goals = [
                self._transform_pose(
                    candidate.goal.pose_xyz_quat_xyzw,
                    source_frame_id=self._profile.reference_frame_id,
                    target_frame_id=self._profile.execution_reference_frame_id,
                )
                for candidate in candidates
            ]

            def constraint(model_qpos, current):
                pose = planner.pose_for_configuration(model_qpos, current)
                answer = project({"right": xyzquat_to_pose_matrix(pose)})
                return JointConfigurationCheck(
                    safe=not answer.modified_sides,
                    reason="tcp_envelope_blocked" if answer.modified_sides else None,
                    metadata=answer.to_payload(),
                )

            eligible = [
                index
                for index, candidate in enumerate(candidates)
                if candidate.goal.translation_distance_m
                <= self._profile.max_translation_delta_m
                and candidate.goal.rotation_distance_rad
                <= self._profile.max_rotation_delta_rad
            ]
            diagnostics = [
                {
                    "candidate_index": index,
                    "reason_code": "safe_tcp_target_exceeds_profile",
                    "wall_sec": 0.0,
                }
                for index in range(len(candidates))
                if index not in eligible
            ]
            selection = None
            scene_hash = MujocoJointSceneOracle(
                self._scene,
                qpos,
                clearance_padding_m=self._padding,
                configuration_constraint=constraint,
                constraint_identity=constraint_hash,
            ).scene_hash
            remaining = parameters["time_budget_sec"] - (time.perf_counter() - started)
            try:
                if eligible and remaining > 0:
                    selection = planner.plan_candidates(
                        [goals[index] for index in eligible],
                        prioritize_current_seed=True,
                        translation_tolerance_m=self._profile.translation_tolerance_m,
                        rotation_tolerance_rad=self._profile.rotation_tolerance_rad,
                        normalize_ik_objective=self._profile.normalize_ik_objective,
                        cancel_requested=cancel_requested,
                        time_budget_sec=remaining,
                        max_reference_count=self._profile.max_control_cycles
                        - self._profile.stable_measurement_count,
                        configuration_constraint=constraint,
                        constraint_identity=constraint_hash,
                    )
                    diagnostics.extend(
                        {**item, "candidate_index": eligible[item["candidate_index"]]}
                        for item in selection.diagnostics
                    )
            except MujocoTcpPlanRejected as error:
                if error.reason_code != "safe_tcp_no_candidate_path":
                    raise
                diagnostics.extend(
                    {**item, "candidate_index": eligible[item["candidate_index"]]}
                    for item in error.details["candidates"]
                )
            if cancel_requested():
                raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
            index = None if selection is None else eligible[selection.candidate_index]
            # Preserve generated alternatives for later multi-stage queries. A
            # proposal on a cost-pruned/rejected row is not a checked path.
            diagnostics = [_candidate_diagnostic(item,
                candidates[item["candidate_index"]].goal.pose_xyz_quat_xyzw) for item in diagnostics]
            # Revalidate the captured safety reference before publishing frozen-scene evidence.
            constraint(qpos, pair)
            return {
                "component_id": "right_arm",
                "reference_frame_id": self._profile.reference_frame_id,
                "anchor_xyz_m": list(parameters["anchor_xyz_m"]),
                "source_tcp_pose_xyz_quat_xyzw": list(initial),
                "query_status": "no_path_found" if selection is None else "path_found",
                "scene_hash": scene_hash,
                "source_qpos_hash": canonical_content_hash(
                    qpos.tolist(), domain="object-goal-query.qpos.v1"
                ),
                "source_sim_time_sec": sim_time,
                "pose_targets_by_component": {}
                if index is None
                else {"right_arm": list(candidates[index].goal.pose_xyz_quat_xyzw)},
                "candidate_count": len(candidates),
                "checked_count": len(diagnostics),
                "all_candidates_checked": len(diagnostics) == len(candidates),
                "selected_index": index,
                "candidate_diagnostics": diagnostics,
                "planning_wall_sec": time.perf_counter() - started,
                "ranking_criteria": RANKING,
                "motion_executed": False,
                "execution_authorized": False,
                "input_source": "caller_anchor_and_frozen_measured_scene",
                "constraint_scope": "frozen_scene_full_path",
            }
        finally:
            if planner is not None:
                planner.clear()
            self._lock.release()


def build_object_goal_query_executor(
    *,
    registration,
    query_port,
    current_station_revision,
    clock,
    max_retained_operations,
):
    def read(request, operation):
        registration.payload_validator.validate_parameters(request.parameters)
        result = query_port.query(
            request.parameters,
            cancel_requested=lambda: (
                executor.cancel_requested(operation.operation_id)
                or clock() >= operation.deadline_at
            ),
        )
        return PassiveReadResult(
            result,
            ("object-goal-query:" + operation.operation_id,),
            reason_code="object_goal_query_completed",
        )

    executor = PassiveReadSkillExecutor(
        port_id="agent-gateway:"
        + registration.entry.deployment_binding.deployment_binding_id,
        executor_id="passive.object-goal-query.v1",
        read=read,
        current_station_revision=current_station_revision,
        clock=clock,
        max_retained_operations=max_retained_operations,
    )
    return executor

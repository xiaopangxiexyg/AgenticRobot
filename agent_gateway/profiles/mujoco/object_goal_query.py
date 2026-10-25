"""MuJoCo-only passive feasibility projection, with no motion/lifecycle port."""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from agent_gateway.profiles.query_geometry import validate_query_selection
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

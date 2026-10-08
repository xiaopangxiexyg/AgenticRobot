"""Opt-in, read-only object transport Skill assembly.

The caller cannot assert attachment or remove obstacles. A bound observation
owner supplies grasp verification, correlated object/tool poses and a complete
identified scene. No real profile advertises this without that provider.
"""
from __future__ import annotations

import io
import json
from contextlib import nullcontext
from dataclasses import dataclass, replace
from collections.abc import Mapping

import numpy as np

from agent_gateway.protocol.artifact_contracts import AgentArtifactBinding
from agent_gateway.protocol.skill_catalog import SkillDeploymentBindingRef
from agent_gateway.protocol.skill_io_contracts import SkillArtifactSlotSpec, SkillIOContract
from agent_gateway.protocol.skill_registration import SkillRegistrationTemplate
from agent_gateway.runtime.artifact_producer import artifact_ref_for_bytes, commit_artifact_bytes
from agent_gateway.runtime.passive_read_executor import PassiveReadResult, PassiveReadSkillExecutor
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorSet
from agent_gateway.skills.perception.object_transport import (
    SKILL_ID, PLAN_SCHEMA, object_transport_definition, object_transport_schema,
    validate_object_transport_parameters, validate_object_transport_result,
    validate_object_transport_result_for_parameters,
)
from agent_gateway.skills.perception.schemas import validate_no_task_evaluation_v1
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.object_transport import GraspedObjectObservation, rigid_pose
from robot_motion.scene_obstacles import ObstacleSceneSnapshot
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import RM65GraspTrajectoryPlanningError


@dataclass(frozen=True)
class ObjectTransportRegistration:
    skill: SkillRegistrationTemplate
    binding: SkillDeploymentBindingRef
    payload_validator: AgentSkillPayloadValidatorSet
    executor_id: str = "real.passive.object_transport.v1"


def build_object_transport_registration(*, base_registration, observation_profile_hash: str):
    """Bind to an existing passive route/session; never upgrade its authority."""
    base_route = base_registration.skill.routes[0]
    if base_route.control_effect_class != "read_only":
        raise ValueError("Object transport planning needs a passive base route")
    if not observation_profile_hash:
        raise ValueError("A verified held-object observation profile is required")
    definition = object_transport_definition()
    schema = object_transport_schema(definition_hash=definition.semantic_content_hash, profile_hash=observation_profile_hash)
    io_contract = SkillIOContract(skill_id=SKILL_ID, skill_version="1", skill_definition_hash=definition.semantic_content_hash,
        parameter_schema_hash=schema.parameter_schema_hash, execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash, input_artifact_slots=(),
        output_artifact_slots=(SkillArtifactSlotSpec(slot_id="object_transport_plan", direction="output", min_count=0, max_count=1,
            allowed_media_types=("application/x-npz",), allowed_schema_ids=(PLAN_SCHEMA,),
            allowed_encodings=("numpy-npz",), allowed_compressions=("zip",),
            max_artifact_bytes=16*1024*1024, max_total_bytes=16*1024*1024),))
    route = replace(base_route, skill_id=SKILL_ID, skill_version="1", skill_definition_hash=definition.semantic_content_hash,
        route_id="real.rm65.perception.plan_object_move", route_version="1",
        parameter_schema_hash=schema.parameter_schema_hash, execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash, allowed_parameter_profiles=schema.parameter_profiles,
        skill_io_contract_hash=io_contract.semantic_content_hash, max_operation_duration_ms=60_000)
    binding = base_registration.binding
    if hasattr(binding, "to_ref"):
        binding = binding.to_ref()
    binding = replace(binding, deployment_binding_id="object-transport:"+binding.deployment_binding_id,
        deployment_binding_hash=canonical_content_hash({"source": binding.deployment_binding_hash,
            "route": route.semantic_content_hash, "observation_profile": observation_profile_hash},
            domain="agenticrobot.object_transport.binding.v1"))
    validator = AgentSkillPayloadValidatorSet(validator_id="object-transport:"+observation_profile_hash,
        skill_id=SKILL_ID, skill_version="1", route_id=route.route_id, route_version="1",
        parameter_schema_hash=schema.parameter_schema_hash, execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        validate_parameters=validate_object_transport_parameters, validate_execution_result=validate_object_transport_result,
        validate_execution_result_for_parameters=validate_object_transport_result_for_parameters,
        validate_task_evaluation=validate_no_task_evaluation_v1)
    return ObjectTransportRegistration(SkillRegistrationTemplate(definition, schema, io_contract, (route,)), binding, validator)


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_object_transport_reader(*, observe_held_object, planner_factory, artifact_store,
                                  runtime_session_id, current_station_revision, clock,
                                  non_motion_read_window=nullcontext):
    """No lifecycle/dispatch/HOLD port is accepted at this boundary."""
    def read(request, operation):
        parameters = request.parameters
        validate_object_transport_parameters(parameters)
        goal = rigid_pose(np.asarray(parameters["T_reference_from_object_goal"]).reshape(4, 4), name="object goal")
        revision = current_station_revision()
        if revision != request.expected_station_revision:
            raise ValueError("Object transport scene revision changed")
        with non_motion_read_window():
            observation, scene = observe_held_object(object_id=parameters["object_id"], side=parameters["side"],
                                                      station_revision=revision)
            if not isinstance(observation, GraspedObjectObservation) or not isinstance(scene, ObstacleSceneSnapshot):
                raise TypeError("A correlated held-object observation and identified scene are required")
            observation.require_current(now=clock(), scene_revision=revision, object_id=parameters["object_id"],
                side=parameters["side"], reference_frame_id=parameters["reference_frame_id"])
            result = {"status": "rejected", "reason_code": "not_planned", "object_id": observation.object_id,
                "side": observation.side, "reference_frame_id": observation.reference_frame_id, "scene_revision": revision,
                "T_reference_from_object_goal": goal.reshape(-1).tolist(),
                "T_reference_from_tool_goal": observation.tool_goal(goal).reshape(-1).tolist(),
                "T_tool_from_object": observation.tool_from_object.reshape(-1).tolist(),
                "trajectory_hash": None, "duration_sec": 0., "sample_count": 0,
                "motion_executed": False, "execution_authorized": False}
            try:
                plan = planner_factory().plan_object_move(observation=observation, object_goal=goal,
                    world_scene=scene, speed_scale=parameters["speed_scale"])
            except RM65GraspTrajectoryPlanningError as exc:
                result["reason_code"] = exc.reason_code
                return PassiveReadResult(result, (observation.pose_evidence_ref, observation.grasp_evidence_ref),
                                         reason_code="object_transport_rejected")
        if current_station_revision() != revision:
            raise ValueError("Scene changed during object transport planning")
        trajectory = plan.trajectory
        result.update(status="path_found", reason_code="path_found", trajectory_hash=trajectory.semantic_content_hash,
                      duration_sec=float(trajectory.time_from_start_sec[-1]), sample_count=len(trajectory.time_from_start_sec))
        data = io.BytesIO()
        np.savez_compressed(data, time_from_start_sec=trajectory.time_from_start_sec,
            left_qpos=trajectory.qpos_by_side["left"], right_qpos=trajectory.qpos_by_side["right"],
            metadata_json=np.frombuffer(json.dumps({"schema": PLAN_SCHEMA, "result": result,
                "details": _plain(plan.details), "planner_evidence_refs": trajectory.planner_evidence_refs},
                allow_nan=False, sort_keys=True).encode(), dtype=np.uint8))
        payload = data.getvalue()
        if len(payload) > 16*1024*1024:
            raise ValueError("Object transport artifact exceeds the registered byte limit")
        artifact = artifact_ref_for_bytes(payload, media_type="application/x-npz", schema_id=PLAN_SCHEMA,
                                          encoding="numpy-npz", compression="zip")
        artifact = commit_artifact_bytes(store=artifact_store, data=payload, artifact=artifact,
            producer_id="agent_gateway.object_transport", runtime_session_id=runtime_session_id,
            idempotency_key="object-transport:"+operation.operation_id, now=clock())
        return PassiveReadResult(result, trajectory.evidence_refs,
            output_artifact_bindings=(AgentArtifactBinding(slot_id="object_transport_plan", ordinal=0, artifact=artifact,
                production_receipt_ref="object-transport:"+operation.operation_id),), reason_code="object_transport_planned")
    return read


def build_object_transport_executor(*, registration, max_retained_operations, clock, **kwargs):
    return PassiveReadSkillExecutor(port_id="agent-gateway:object-transport:"+kwargs["runtime_session_id"],
        executor_id=registration.executor_id, current_station_revision=kwargs["current_station_revision"],
        read=build_object_transport_reader(clock=clock, **kwargs), max_retained_operations=max_retained_operations,
        clock=clock)

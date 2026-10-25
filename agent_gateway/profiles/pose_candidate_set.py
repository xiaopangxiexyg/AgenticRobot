"""Authenticated proposal handoff over the existing CAS and passive query owners."""

from __future__ import annotations

import json
from dataclasses import replace

from agent_gateway.profiles.spatial_goal import SpatialGoalRegistration
from agent_gateway.protocol._contract_validation import thaw_json
from agent_gateway.protocol.artifact_contracts import AgentArtifactBinding
from agent_gateway.protocol.skill_catalog import SkillRouteManifestEntry
from agent_gateway.protocol.skill_io_contracts import (
    SkillArtifactSlotSpec,
    SkillIOContract,
)
from agent_gateway.protocol.skill_registration import SkillRegistrationTemplate
from agent_gateway.runtime.artifact_producer import (
    artifact_ref_for_bytes,
    commit_artifact_bytes,
)
from agent_gateway.runtime.passive_read_executor import (
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorSet
from agent_gateway.skills.perception.object_goal_query import (
    RELATION_FIELDS,
    RELATION_SKILL_ID,
)
from agent_gateway.skills.perception.pose_candidate_set import (
    MAX_CANDIDATE_BYTES,
    POSE_CANDIDATES_SCHEMA_ID,
    POSE_CANDIDATES_SLOT,
    candidate_continuation_definition,
    candidate_continuation_schema,
    validate_index_parameters,
    validate_index_result,
    validate_index_result_for_parameters,
)
from agent_gateway.skills.perception.schemas import validate_no_task_evaluation_v1
from agent_gateway.skills.perception.tcp_continuation_query import (
    SKILL_ID as CONTINUATION_SKILL_ID,
)
from agent_gateway.skills.perception.tcp_continuation_query import (
    validate_continuation_result_for_parameters,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.spatial_goals import generate_relation_goals


def _bytes(value):
    return json.dumps(thaw_json(value), ensure_ascii=False, allow_nan=False,
        sort_keys=True, separators=(",", ":")).encode("utf-8")


def candidate_slot(direction):
    return SkillArtifactSlotSpec(slot_id=POSE_CANDIDATES_SLOT, direction=direction,
        min_count=1, max_count=1, allowed_media_types=("application/json",),
        allowed_schema_ids=(POSE_CANDIDATES_SCHEMA_ID,), allowed_encodings=("utf-8",),
        allowed_compressions=("none",), max_artifact_bytes=MAX_CANDIDATE_BYTES,
        max_total_bytes=MAX_CANDIDATE_BYTES)


def _registration(base, definition, schema, *, inputs=(), outputs=(), parameters, result, result_for):
    version = definition.skill_version
    io = SkillIOContract(skill_id=definition.skill_id, skill_version=version,
        skill_definition_hash=definition.semantic_content_hash, parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        input_artifact_slots=inputs, output_artifact_slots=outputs)
    route = replace(base.entry.route, skill_version=version,
        route_id=base.entry.route.route_id + ".candidate_artifact.v" + version, route_version=version,
        skill_definition_hash=definition.semantic_content_hash, parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema.parameter_profiles, skill_io_contract_hash=io.semantic_content_hash)
    binding = replace(base.entry.deployment_binding,
        deployment_binding_id="candidate-artifact:" + base.entry.deployment_binding.deployment_binding_id,
        deployment_binding_hash=canonical_content_hash(
            {"base": base.entry.deployment_binding.deployment_binding_hash, "route": route.semantic_content_hash},
            domain="agenticrobot.pose_candidate_set.binding.v1"))
    validator = AgentSkillPayloadValidatorSet(validator_id="candidate-artifact:" + route.semantic_content_hash,
        skill_id=definition.skill_id, skill_version=version, route_id=route.route_id, route_version=version,
        parameter_schema_hash=schema.parameter_schema_hash, execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash, validate_parameters=parameters,
        validate_execution_result=result, validate_execution_result_for_parameters=result_for,
        validate_task_evaluation=validate_no_task_evaluation_v1)
    return SpatialGoalRegistration(SkillRegistrationTemplate(definition, schema, io, (route,)),
        SkillRouteManifestEntry(route=route, deployment_binding=binding, evaluation_plan_hash=None,
            availability_hash=schema.parameter_profiles[0].parameter_profile_hash), validator)


def build_relation_candidate_registration(*, relation_registration):
    base = relation_registration
    if (base.skill.definition.skill_id != RELATION_SKILL_ID or base.skill.definition.skill_version != "2"
            or base.entry.route.control_effect_class != "read_only"):
        raise ValueError("Candidate Artifacts require the explicit passive relation v2 producer.")
    definition = replace(base.skill.definition, skill_version="3",
        parameter_semantics_id="artifact_surface_relation_query.v3",
        execution_result_semantics_id="anchored_relation_candidate_artifact.v3",
        summary=("Generate relation proposals and publish one immutable pose_candidates output Artifact. "
            "Use this exact Artifact and candidate_indices in query_tcp_continuation_paths@2 instead of "
            "transcribing XYZ/quaternions. Consume one authenticated surface_anchor v2 from this "
            "client/session/current Station revision; do not supply anchor XYZ or current TCP. Offsets use "
            "reference or original input-tool axes. Preserve orientation (zero turns only), or align a "
            "tool axis and sample rotations about that axis in radians. At most 64 offset/rotation samples "
            "and 0.1..30 seconds for the whole query; each path search gets at most 3 seconds and the "
            "remaining budget. Existing collision/envelope checks and joint-excursion ranking apply to "
            "completed paths in the frozen measured scene. Source indices enumerate every generated proposal, including "
            "unattempted samples; consult path_query diagnostics separately for checked path evidence. "
            "Relations constrain endpoints only, not transit orientation or custom corridors. Budget "
            "exhaustion or rank pruning is not collision proof or global infeasibility. No execution "
            "permission; submit the chosen target separately through live safe motion."))
    identity = canonical_content_hash({"base_route": base.entry.route.semantic_content_hash,
        "candidate_schema": POSE_CANDIDATES_SCHEMA_ID}, domain="agenticrobot.relation_candidate.profile.v1")
    schema = replace(base.skill.schema_entry, skill_version="3", skill_definition_hash=definition.semantic_content_hash,
        parameter_profiles=(replace(base.skill.schema_entry.parameter_profiles[0],
            parameter_profile_id=RELATION_SKILL_ID + ".v3", parameter_profile_hash=identity),))
    validator = base.payload_validator
    return _registration(base, definition, schema, inputs=base.skill.io_contract.input_artifact_slots,
        outputs=(candidate_slot("output"),), parameters=validator.validate_parameters,
        result=validator.validate_execution_result, result_for=validator.validate_execution_result_for_parameters)


def build_candidate_continuation_registration(*, continuation_registration, relation_registration):
    base = continuation_registration
    if (base.skill.definition.skill_id != CONTINUATION_SKILL_ID or base.skill.definition.skill_version != "1"
            or base.entry.route.control_effect_class != "read_only"
            or relation_registration.skill.definition.skill_id != RELATION_SKILL_ID
            or relation_registration.skill.definition.skill_version != "3"
            or any(getattr(base.entry.deployment_binding, key) != getattr(relation_registration.entry.deployment_binding, key)
                for key in ("target_binding_id", "execution_manifest_hash", "target_kind"))):
        raise ValueError("Candidate continuation must share its producer deployment.")
    for key in ("component_id", "reference_frame_id"):
        if base.skill.schema_entry.parameter_schema["properties"][key]["const"] != relation_registration.skill.schema_entry.parameter_schema["properties"][key]["const"]:
            raise ValueError("Candidate producer and consumer component/frame bindings differ.")
    definition = candidate_continuation_definition()
    identity = canonical_content_hash({"base_route": base.entry.route.semantic_content_hash,
        "producer_route": relation_registration.entry.route.semantic_content_hash},
        domain="agenticrobot.candidate_continuation.profile.v1")
    schema = candidate_continuation_schema(base_schema=base.skill.schema_entry, definition=definition, profile_hash=identity)

    def parameters(value):
        validate_index_parameters(value)
        for key in ("component_id", "reference_frame_id"):
            if value[key] != schema.parameter_schema["properties"][key]["const"]:
                raise ValueError("Candidate continuation component/frame differs from its deployment.")

    def result_for(value, output):
        parameters(value)
        validate_index_result_for_parameters(value, output)

    return _registration(base, definition, schema, inputs=(candidate_slot("input"),),
        parameters=parameters, result=validate_index_result, result_for=result_for)


def _candidate_payload(*, request, operation_id, result, registration, clock_domain_id):
    registration.payload_validator.validate_execution_result_for_parameters(request.parameters, result)
    path = result["path_query"]
    proposals = generate_relation_goals(current_tcp_pose_xyz_quat_xyzw=path["source_tcp_pose_xyz_quat_xyzw"],
        anchor_xyz_m=result["surface_anchor"]["measurement"]["world_xyz_m"],
        **{key: request.parameters[key] for key in RELATION_FIELDS
            if key not in {"component_id", "reference_frame_id", "time_budget_sec"}})
    return {"schema": POSE_CANDIDATES_SCHEMA_ID, "runtime_session_id": request.runtime_session_id,
        "station_revision": request.expected_station_revision, "clock_domain_id": clock_domain_id,
        "source_operation_id": operation_id, "source_request_hash": request.semantic_content_hash,
        "source_route_hash": registration.entry.route.semantic_content_hash,
        "source_profile_hash": registration.skill.schema_entry.parameter_profiles[0].parameter_profile_hash,
        **{key: path[key] for key in ("component_id", "reference_frame_id", "scene_hash", "source_qpos_hash",
            "source_sim_time_sec", "source_tcp_pose_xyz_quat_xyzw")},
        "anchor_artifact_id": result["anchor_artifact_id"],
        "candidate_poses_xyz_quat_xyzw": [list(item.goal.pose_xyz_quat_xyzw) for item in proposals],
        "scope": "generated_endpoint_proposals_not_path_certificates",
        "motion_executed": False, "execution_authorized": False}


def publish_candidate_set(*, request, operation, result, registration, store, clock, clock_domain_id):
    payload = _candidate_payload(request=request, operation_id=operation.operation_id, result=result,
        registration=registration, clock_domain_id=clock_domain_id)
    data = _bytes(payload)
    if len(data) > MAX_CANDIDATE_BYTES:
        raise ValueError("Candidate proposals exceed the bounded Artifact contract.")
    artifact = artifact_ref_for_bytes(data, media_type="application/json", schema_id=POSE_CANDIDATES_SCHEMA_ID,
        encoding="utf-8", compression="none")
    commit_artifact_bytes(store=store, data=data, artifact=artifact, producer_id=RELATION_SKILL_ID,
        runtime_session_id=request.runtime_session_id, idempotency_key=operation.operation_id + ":pose-candidates", now=clock())
    return AgentArtifactBinding(slot_id=POSE_CANDIDATES_SLOT, ordinal=0, artifact=artifact,
        production_receipt_ref="pose-candidates-production:" + operation.operation_id)


def resolve_candidate_set(*, request, registration, store, lookup_operation, current_station_revision, clock_domain_id):
    bindings = request.input_artifact_bindings
    if len(bindings) != 1 or bindings[0].slot_id != POSE_CANDIDATES_SLOT or bindings[0].ordinal != 0:
        raise ValueError("Exactly one pose_candidates input Artifact is required.")
    artifact = bindings[0].artifact
    if (artifact.schema_id != POSE_CANDIDATES_SCHEMA_ID or artifact.media_type != "application/json"
            or artifact.encoding != "utf-8" or artifact.compression != "none"
            or artifact.byte_length > MAX_CANDIDATE_BYTES):
        raise ValueError("Incompatible candidate Artifact descriptor.")
    data = store.read_bytes(artifact, maximum=MAX_CANDIDATE_BYTES)
    payload = json.loads(data)
    if not isinstance(payload, dict) or not isinstance(payload.get("source_operation_id"), str):
        raise TypeError("Candidate Artifact has no producer identity.")
    retained = lookup_operation(payload["source_operation_id"])
    original, source = (None, None) if retained is None else retained
    if (source is None or original is None or source.status != "completed" or source.execution_outcome != "completed"
            or source.execution_result is None or original.skill_id != RELATION_SKILL_ID or original.skill_version != "3"
            or original.route_hash != registration.entry.route.semantic_content_hash
            or original.client_id != request.client_id or original.runtime_session_id != request.runtime_session_id
            or source.runtime_session_id != request.runtime_session_id
            or original.skill_route_manifest_hash != request.skill_route_manifest_hash
            or original.semantic_content_hash != source.request_hash
            or original.expected_station_revision != request.expected_station_revision
            or current_station_revision() != request.expected_station_revision
            or not any(item.slot_id == POSE_CANDIDATES_SLOT and item.ordinal == 0
                and item.artifact == artifact for item in source.output_artifact_bindings)):
        raise ValueError("Candidate Artifact is not a completed output of this client/session/current revision.")
    expected = _candidate_payload(request=original, operation_id=source.operation_id,
        result=source.execution_result, registration=registration, clock_domain_id=clock_domain_id)
    if data != _bytes(expected):
        raise ValueError("Candidate Artifact differs from its authenticated producer evidence.")
    if any(payload[key] != request.parameters[key] for key in ("component_id", "reference_frame_id")):
        raise ValueError("Candidate Artifact component/frame differs from the request.")
    return payload, artifact.artifact_id


def build_candidate_continuation_executor(*, registration, relation_registration, query_port, store,
        lookup_operation, current_station_revision, clock, clock_domain_id, max_retained_operations):
    """Resolve exact generated geometry, then reuse the unchanged two-stage query port."""
    registration.payload_validator.validate_route(registration.entry.route)
    if (registration.skill.definition.skill_id, registration.skill.definition.skill_version) != (CONTINUATION_SKILL_ID, "2"):
        raise ValueError("Candidate continuation requires its exact v2 registration.")
    if not callable(getattr(query_port, "query_continuations", None)):
        raise TypeError("The passive port must expose query_continuations.")

    def read(request, operation):
        registration.payload_validator.validate_parameters(request.parameters)
        payload, artifact_id = resolve_candidate_set(request=request, registration=relation_registration,
            store=store, lookup_operation=lookup_operation, current_station_revision=current_station_revision,
            clock_domain_id=clock_domain_id)
        indices = request.parameters["candidate_indices"]
        poses = payload["candidate_poses_xyz_quat_xyzw"]
        if any(index >= len(poses) for index in indices):
            raise ValueError("Candidate index exceeds the authenticated proposal set.")
        expanded = {key: value for key, value in request.parameters.items() if key != "candidate_indices"}
        expanded["first_pose_candidates_xyz_quat_xyzw"] = [poses[index] for index in indices]
        result = query_port.query_continuations(expanded, cancel_requested=lambda: (
            executor.cancel_requested(operation.operation_id) or clock() >= operation.deadline_at))
        validate_continuation_result_for_parameters(expanded, result)
        if current_station_revision() != request.expected_station_revision:
            raise RuntimeError("Station revision changed during candidate continuation query.")
        selected = result["selected_index"]
        answer = {**result, "candidate_set_artifact_id": artifact_id, "candidate_indices": list(indices),
            "selected_source_index": None if selected is None else indices[selected]}
        registration.payload_validator.validate_execution_result_for_parameters(request.parameters, answer)
        return PassiveReadResult(answer, ("pose-candidates-source:" + payload["source_operation_id"],
            "tcp-continuation-query:" + operation.operation_id), reason_code="candidate_continuation_query_completed")

    executor = PassiveReadSkillExecutor(port_id=registration.entry.route.route_id,
        executor_id="passive.tcp-continuation-query.v2", read=read,
        current_station_revision=current_station_revision, clock=clock, max_retained_operations=max_retained_operations)
    return executor

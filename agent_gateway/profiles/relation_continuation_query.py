"""Authenticated surface-anchor v2 input over the existing passive query owner."""

from collections.abc import Mapping

from agent_gateway.profiles.query_geometry import (
    validate_relation_continuation_geometry_parameters,
    validate_relation_continuation_path,
)
from agent_gateway.profiles.surface_anchor import (
    _registration,
    require_matching_anchor_schema,
    resolve_surface_anchor,
)
from agent_gateway.runtime.passive_read_executor import (
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from agent_gateway.skills.perception.relation_continuation_query import (
    POSES_KEY,
    explicit_continuation_parameters,
    relation_continuation_definition,
    relation_continuation_schema,
    validate_relation_continuation_parameters,
)
from agent_gateway.skills.perception.tcp_continuation_query import (
    validate_continuation_parameters,
)
from dataset_contracts.canonical_encoding import canonical_content_hash


def build_relation_continuation_registration(*, capture, relation, continuation):
    for item in (capture, relation, continuation):
        if (item.entry.route.target_kind != "sim" or item.entry.route.control_effect_class != "read_only"
                or item.entry.route.cancellation_policy != "cancel_only"):
            raise ValueError("Relation continuation queries require passive simulation owners.")
        if any(getattr(item.entry.deployment_binding, key) != getattr(continuation.entry.deployment_binding, key)
                for key in ("target_binding_id", "execution_manifest_hash")):
            raise ValueError("Relation continuation owners must share one target and execution manifest.")
    if (capture.skill.definition.skill_id, capture.skill.definition.skill_version) != (
            "perception.capture_surface_anchor", "2"):
        raise ValueError("Relation continuation v3 requires its exact surface anchor v2 producer.")
    if (relation.skill.definition.skill_id, relation.skill.definition.skill_version) != (
            "perception.query_relation_paths", "2") or (
            continuation.skill.definition.skill_id, continuation.skill.definition.skill_version) != (
                "perception.query_tcp_continuation_paths", "1"):
        raise ValueError("Relation continuation requires its existing relation v2 and continuation v1 owners.")
    require_matching_anchor_schema(relation, capture)
    definition = relation_continuation_definition()
    identity = canonical_content_hash({"capture": capture.entry.route.semantic_content_hash,
        "relation": relation.entry.route.semantic_content_hash,
        "continuation": continuation.entry.route.semantic_content_hash},
        domain="agenticrobot.relation_continuation.profile.v3")
    frame = continuation.skill.schema_entry.parameter_schema["properties"]["reference_frame_id"]["const"]
    component = continuation.skill.schema_entry.parameter_schema["properties"]["component_id"]["const"]
    relation_identity = relation.skill.schema_entry.parameter_schema["properties"]
    measured_frame = capture.skill.schema_entry.execution_result_schema["properties"]["measurement"]["properties"]["reference_frame_id"]
    if (relation_identity["component_id"].get("const") != component
            or relation_identity["reference_frame_id"].get("const") != frame
            or measured_frame.get("const") != frame):
        raise ValueError("Relation continuation owners must share the same component and public frame.")

    def parameters(value):
        validate_relation_continuation_parameters(value)
        validate_relation_continuation_geometry_parameters(value)
        relation.payload_validator.validate_parameters({key: value[key]
            for key in relation.skill.schema_entry.parameter_schema["required"]})
        if value["component_id"] != component or value["reference_frame_id"] != frame:
            raise ValueError("Relation continuation component/frame differs from its deployment binding.")

    def result(value):
        if (not isinstance(value, Mapping) or set(value) != {
                "surface_anchor", "anchor_artifact_id", "path_query", "relation_constraint_scope"}
                or value["relation_constraint_scope"] != "endpoint_only"
                or not isinstance(value["anchor_artifact_id"], str) or not value["anchor_artifact_id"]):
            raise ValueError("Relation continuation must retain its anchor and endpoint-only relation scope.")
        capture.payload_validator.validate_execution_result(value["surface_anchor"])
        path = value["path_query"]
        if not isinstance(path, Mapping) or POSES_KEY not in path:
            raise ValueError("Relation continuation requires the generated candidate poses.")
        continuation.payload_validator.validate_execution_result({key: item for key, item in path.items()
            if key != POSES_KEY})
        validate_continuation_parameters(explicit_continuation_parameters({**path,
            "continuation_delta_xyz_m": [0, 0, 0], "path_constraint": None, "time_budget_sec": .1}, path[POSES_KEY]))
        if path["candidate_count"] != len(path[POSES_KEY]):
            raise ValueError("Relation continuation candidate accounting differs from its generated proposals.")
        if path["selected_index"] is not None and any(abs(a - b) > 1e-8 for a, b in zip(
                path["first_pose_target_xyz_quat_xyzw"], path[POSES_KEY][path["selected_index"]], strict=True)):
            raise ValueError("The selected first target differs from its generated candidate.")
        if value["surface_anchor"]["measurement"]["reference_frame_id"] != frame:
            raise ValueError("Relation continuation changed its anchor frame.")

    def result_for(value, answer):
        parameters(value)
        result(answer)
        validate_relation_continuation_path(value, answer["path_query"],
            anchor_xyz_m=answer["surface_anchor"]["measurement"]["world_xyz_m"])

    return _registration(definition=definition, schema=relation_continuation_schema(
        definition=definition, capture=capture.skill.schema_entry, relation=relation.skill.schema_entry,
        continuation=continuation.skill.schema_entry, profile_hash=identity), base_entry=continuation.entry,
        direction="input", namespace="relation_continuation_query.v3",
        parameters=parameters, result=result, result_for=result_for)


def build_relation_continuation_executor(*, registration, capture_registration, query_port, store,
        lookup_operation, current_station_revision, clock, clock_domain_id, max_retained_operations):
    return _build_anchored_continuation_executor(registration=registration,
        capture_registration=capture_registration, query_port=query_port, store=store,
        lookup_operation=lookup_operation, current_station_revision=current_station_revision,
        clock=clock, clock_domain_id=clock_domain_id, max_retained_operations=max_retained_operations,
        version="3", query_method="query_relation_continuations", namespace="relation-continuation-query")


def _build_anchored_continuation_executor(*, registration, capture_registration, query_port, store,
        lookup_operation, current_station_revision, clock, clock_domain_id, max_retained_operations,
        version, query_method, namespace):
    """Keep provenance, revision, cancellation and retention under one adapter."""
    registration.payload_validator.validate_route(registration.entry.route)
    if (registration.skill.definition.skill_id, registration.skill.definition.skill_version) != (
            "perception.query_tcp_continuation_paths", version) or (
            capture_registration.skill.definition.skill_id, capture_registration.skill.definition.skill_version) != (
                "perception.capture_surface_anchor", "2"):
        raise ValueError(f"Continuation requires its exact v{version} registration and v2 anchor producer.")
    require_matching_anchor_schema(registration, capture_registration)
    query = getattr(query_port, query_method, None)
    if not callable(query):
        raise TypeError(f"The passive port must expose {query_method}.")

    def read(request, operation):
        registration.payload_validator.validate_parameters(request.parameters)
        anchor, artifact_id = resolve_surface_anchor(request=request, store=store,
            lookup_operation=lookup_operation, capture_registration=capture_registration,
            current_station_revision=current_station_revision, clock_domain_id=clock_domain_id)
        path = query({**request.parameters,
            "anchor_xyz_m": anchor["measurement"]["world_xyz_m"]},
            cancel_requested=lambda: executor.cancel_requested(operation.operation_id)
                or clock() >= operation.deadline_at)
        if current_station_revision() != request.expected_station_revision:
            raise RuntimeError("Station revision changed during relation continuation query; recapture the anchor.")
        result = {"surface_anchor": anchor, "anchor_artifact_id": artifact_id, "path_query": path,
            "relation_constraint_scope": "endpoint_only"}
        registration.payload_validator.validate_execution_result_for_parameters(request.parameters, result)
        return PassiveReadResult(result, ("surface-anchor-source:" + anchor["source_operation_id"],
            namespace + ":" + operation.operation_id),
            reason_code=namespace.replace("-", "_") + "_completed")

    executor = PassiveReadSkillExecutor(port_id=registration.entry.route.route_id,
        executor_id=f"passive.{namespace}.v{version}", read=read,
        current_station_revision=current_station_revision, clock=clock,
        max_retained_operations=max_retained_operations)
    return executor

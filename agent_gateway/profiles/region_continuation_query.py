"""Reuse authenticated anchors and the passive two-stage owner for region queries."""

from agent_gateway.profiles.query_geometry import (
    validate_region_continuation_path,
    validate_region_geometry_parameters,
)
from agent_gateway.profiles.relation_continuation_query import (
    _build_anchored_continuation_executor,
    build_relation_continuation_registration,
)
from agent_gateway.profiles.surface_anchor import _registration
from agent_gateway.skills.perception.region_continuation_query import (
    SAMPLING_METHODS,
    region_continuation_definition,
    region_continuation_schema,
    validate_region_continuation_parameters,
    validate_region_continuation_version,
)
from dataset_contracts.canonical_encoding import canonical_content_hash


def build_region_continuation_registration(*, capture, relation, continuation, version="4"):
    validate_region_continuation_version(version)
    base = build_relation_continuation_registration(
        capture=capture, relation=relation, continuation=continuation)
    definition = region_continuation_definition(version=version)
    identity = canonical_content_hash({"source": base.entry.route.semantic_content_hash,
        "sampling_method": SAMPLING_METHODS[version]}, domain=f"agenticrobot.region_continuation.profile.v{version}")
    component = base.skill.schema_entry.parameter_schema["properties"]["component_id"]["const"]
    frame = base.skill.schema_entry.parameter_schema["properties"]["reference_frame_id"]["const"]

    def parameters(value):
        validate_region_continuation_parameters(value, version=version)
        validate_region_geometry_parameters(value, version=version)
        if value["component_id"] != component or value["reference_frame_id"] != frame:
            raise ValueError("Region continuation component/frame differs from its deployment binding.")

    def result_for(value, answer):
        parameters(value)
        base.payload_validator.validate_execution_result(answer)
        validate_region_continuation_path(value, answer["path_query"],
            anchor_xyz_m=answer["surface_anchor"]["measurement"]["world_xyz_m"], version=version)

    return _registration(definition=definition, schema=region_continuation_schema(
        definition=definition, relation_schema=base.skill.schema_entry, profile_hash=identity),
        base_entry=continuation.entry, direction="input", namespace=f"region_continuation_query.v{version}",
        parameters=parameters, result=base.payload_validator.validate_execution_result,
        result_for=result_for)


def build_region_continuation_executor(**kwargs):
    version = kwargs["registration"].skill.definition.skill_version
    validate_region_continuation_version(version)
    method = "query_region_continuations" if version == "4" else "query_region_continuation_extrema"
    return _build_anchored_continuation_executor(**kwargs, version=version,
        query_method=method, namespace="region-continuation-query")

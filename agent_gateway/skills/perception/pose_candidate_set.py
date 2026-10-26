"""Portable generated pose proposals and explicit Artifact-index query semantics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from agent_gateway.protocol._contract_validation import nonnegative_integer
from agent_gateway.protocol.skill_catalog import (
    SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)
from agent_gateway.skills.perception.tcp_continuation_query import (
    PARAMETER_KEYS,
    RESULT_KEYS,
    continuation_query_definition,
    validate_continuation_parameters,
    validate_continuation_result,
)

POSE_CANDIDATES_SCHEMA_ID = "agenticrobot.generated_pose_candidates.v1"
POSE_CANDIDATES_SLOT = "pose_candidates"
MAX_CANDIDATE_BYTES = 32_768
INDEX_PARAMETER_KEYS = PARAMETER_KEYS - {"first_pose_candidates_xyz_quat_xyzw"} | {"candidate_indices"}
INDEX_RESULT_KEYS = RESULT_KEYS | {"candidate_set_artifact_id", "candidate_indices", "selected_source_index"}


def validate_candidate_indices(value):
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 8:
        raise ValueError("Select 1..8 unique candidate indices.")
    for index in value:
        nonnegative_integer(index, field_name="candidate_indices", maximum=63)
    if len(set(value)) != len(value):
        raise ValueError("Candidate indices must be unique.")


def validate_index_parameters(value):
    if not isinstance(value, Mapping) or set(value) != INDEX_PARAMETER_KEYS:
        raise ValueError("Artifact continuation parameters require indices, not caller poses.")
    validate_candidate_indices(value["candidate_indices"])
    expanded = {key: item for key, item in value.items() if key != "candidate_indices"}
    expanded["first_pose_candidates_xyz_quat_xyzw"] = [[0, 0, 0, 0, 0, 0, 1]] * len(value["candidate_indices"])
    validate_continuation_parameters(expanded)


def validate_index_result(value):
    if not isinstance(value, Mapping) or set(value) != INDEX_RESULT_KEYS:
        raise ValueError("Artifact continuation result must retain its exact source mapping.")
    validate_continuation_result({key: value[key] for key in RESULT_KEYS})
    validate_candidate_indices(value["candidate_indices"])
    artifact_id = value["candidate_set_artifact_id"]
    if (not isinstance(artifact_id, str) or not artifact_id.startswith("artifact:")
            or len(artifact_id) != 73 or any(c not in "0123456789abcdef" for c in artifact_id[9:])):
        raise ValueError("Continuation result requires an immutable candidate Artifact identity.")
    if value["candidate_count"] != len(value["candidate_indices"]):
        raise ValueError("Continuation candidate count differs from its source index mapping.")
    selected = value["selected_index"]
    expected = None if selected is None else value["candidate_indices"][selected]
    if (value["selected_source_index"] != expected
            or expected is not None and type(value["selected_source_index"]) is not int):
        raise ValueError("Selected source index differs from the local continuation index.")


def validate_index_result_for_parameters(parameters, result):
    validate_index_parameters(parameters)
    validate_index_result(result)
    if (any(parameters[key] != result[key] for key in ("component_id", "reference_frame_id"))
            or list(parameters["candidate_indices"]) != list(result["candidate_indices"])):
        raise ValueError("Continuation result changed its requested identities or source indices.")
    if result["selected_index"] is not None:
        predicted = result["predicted_first_tcp_pose_xyz_quat_xyzw"]
        expected = [predicted[i] + parameters["continuation_delta_xyz_m"][i] for i in range(3)] + list(predicted[3:])
        if any(abs(a - b) > 1e-8 for a, b in zip(
                result["continuation_pose_target_xyz_quat_xyzw"], expected, strict=True)):
            raise ValueError("Continuation target differs from predicted first FK plus requested delta.")


def candidate_continuation_definition():
    base = continuation_query_definition()
    return replace(base, skill_version="2",
        parameter_semantics_id="artifact_index_tcp_continuation_query.v2",
        execution_result_semantics_id="artifact_index_two_stage_path_evidence.v2",
        summary=("Query two-stage paths from one authenticated pose_candidates input Artifact produced by "
            "query_relation_paths@3 in this client/session/current Station revision. Select 1..8 unique "
            "candidate_indices (zero-based source indices); do not transcribe poses. The bounded passive "
            "planner recomputes both paths in one fresh frozen scene, propagating first endpoint qpos/FK "
            "into the second stage. Add continuation_delta_xyz_m in the public frame and preserve the "
            "predicted quaternion. path_constraint applies only to the second path's sampled references. "
            "The whole query budget is 0.1..30 seconds. Rank complete chains by maximum joint excursion, "
            "duration, total joint travel, then subset index. selected_index addresses the requested "
            "subset; selected_source_index addresses the original Artifact. A proposal, including an "
            "unattempted or rank-pruned proposal, is not a path certificate. The result retains the exact "
            "selected first target for independent safe motion. No execution or authorization; query "
            "plans cannot activate. Results are bounded sampled-path evidence, not global impossibility "
            "or continuous-path guarantees. Actual moves replan with fresh measurements; derive a relative "
            "followup from actual arrival, not predicted FK."))


def candidate_continuation_schema(*, base_schema, definition, profile_hash):
    parameters = dict(base_schema.parameter_schema)
    properties = {key: value for key, value in parameters["properties"].items()
        if key != "first_pose_candidates_xyz_quat_xyzw"}
    indices = {"type": "array", "minItems": 1, "maxItems": 8, "uniqueItems": True,
        "items": {"type": "integer", "minimum": 0, "maximum": 63}}
    properties["candidate_indices"] = indices
    parameters.update(properties=properties, required=sorted(INDEX_PARAMETER_KEYS))
    result = dict(base_schema.execution_result_schema)
    result.update(properties={**result["properties"], "candidate_indices": indices,
        "candidate_set_artifact_id": {"type": "string", "pattern": "^artifact:[0-9a-f]{64}$"},
        "selected_source_index": {"anyOf": [{"type": "null"}, {"type": "integer", "minimum": 0, "maximum": 63}]}},
        required=sorted(INDEX_RESULT_KEYS))
    return replace(base_schema, skill_version="2", skill_definition_hash=definition.semantic_content_hash,
        parameter_schema=parameters, execution_result_schema=result,
        parameter_profiles=(SkillParameterProfileRef(parameter_profile_id="tcp-continuation-path-query-v2",
            parameter_profile_hash=profile_hash,
            parameter_schema_hash=agent_skill_schema_document_hash(parameters, schema_kind="parameter")),))

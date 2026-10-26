"""Target-independent, passive evidence for bounded two-stage TCP queries."""

from __future__ import annotations

from collections.abc import Mapping

from agent_gateway.protocol._contract_validation import (
    freeze_json,
    nonnegative_integer,
    positive_integer,
    required_text,
    sha256_digest,
)
from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry,
    SkillDefinition,
    SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)
from agent_gateway.skills.motion.path_constraint import (
    _constraint_schema,
    _finite,
    validate_path_constraint,
)
from agent_gateway.skills.motion.schemas import _pose

SKILL_ID = "perception.query_tcp_continuation_paths"
RANKING = "max_joint_excursion_rad_then_duration_sec_then_total_joint_travel_rad"
PARAMETER_KEYS = frozenset({
    "component_id", "reference_frame_id", "first_pose_candidates_xyz_quat_xyzw",
    "continuation_delta_xyz_m", "path_constraint", "time_budget_sec",
})
RESULT_KEYS = frozenset({
    "component_id", "reference_frame_id", "source_tcp_pose_xyz_quat_xyzw",
    "scene_hash", "source_qpos_hash", "source_sim_time_sec", "query_status",
    "selected_index", "first_pose_target_xyz_quat_xyzw",
    "continuation_pose_target_xyz_quat_xyzw", "predicted_first_tcp_pose_xyz_quat_xyzw",
    "candidate_count", "attempted_count", "all_candidates_attempted",
    "candidate_diagnostics", "planning_wall_sec", "ranking_criteria",
    "continuation_source", "constraint_scope", "motion_executed", "execution_authorized",
})
_CANDIDATE_KEYS = frozenset({
    "candidate_index", "reason_code", "failed_stage", "wall_sec", "stage_diagnostics",
})
_STAGE_KEYS = frozenset({"stage_index", "reason_code", "wall_sec"})
_PATH_KEYS = frozenset({"trajectory_hash", "reference_count", "duration_sec"})
_TARGET_KEYS = (
    "first_pose_target_xyz_quat_xyzw", "continuation_pose_target_xyz_quat_xyzw",
    "predicted_first_tcp_pose_xyz_quat_xyzw",
)


def continuation_query_definition():
    return SkillDefinition(
        skill_id=SKILL_ID,
        skill_version="1",
        skill_kind="inspection",
        summary=(
            "Query 1..8 explicit TCP candidates and their continuations in one frozen measured scene. "
            "The first path uses ordinary collision/envelope checks only. Propagate its planned endpoint "
            "qpos and measured-FK TCP into the second planning stage; add continuation_delta_xyz_m in "
            "the same public reference frame, preserving that predicted endpoint quaternion. The optional "
            "safe TCP v2 preserve_start_tool_axis cone applies only to sampled references of the second "
            "path, relative to its predicted starting TCP. Budget 0.1..30 seconds covers the whole query. "
            "Rank completed two-path candidates by maximum joint excursion, duration, then total joint "
            "travel across the joined trajectory, breaking ties by caller index. Attempted candidates "
            "include budget exhaustion, which is not proof of rejection. No claim that all IK branches "
            "were searched, no global impossibility or continuous-path guarantee. This default-off "
            "read-only query neither executes nor authorizes motion, grasping or attachment. Replan each "
            "actual safe move from fresh measurements; query plans cannot be activated."
        ),
        intended_target_kind="target_agnostic",
        cross_target_deployment_policy="not_applicable",
        controlled_component_ids=(),
        parameter_semantics_id="tcp_continuation_path_query.v1",
        execution_result_semantics_id="frozen_scene_two_stage_path_evidence.v1",
        task_outcome_semantics_id="no_task_outcome.v1",
        entry_predicate_ids=(),
        task_outcome_predicate_ids=("bounded_query_without_motion.v1",),
    )


def _vector(value, size, *, field_name):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{field_name} must contain exactly {size} numbers.")
    return tuple(_finite(x, field_name=field_name) for x in value)


def _tcp_pose(value, *, field_name):
    return _pose(_vector(value, 7, field_name=field_name), field_name=field_name)


def _nonnegative(value, *, field_name):
    number = _finite(value, field_name=field_name)
    if number < 0:
        raise ValueError(f"{field_name} must be nonnegative.")
    return number


def _identity(value):
    for key in ("component_id", "reference_frame_id"):
        required_text(value[key], field_name=key)


def validate_continuation_parameters(value):
    if not isinstance(value, Mapping) or set(value) != PARAMETER_KEYS:
        raise ValueError("Continuation parameters must match the exact schema.")
    _identity(value)
    candidates = value["first_pose_candidates_xyz_quat_xyzw"]
    if not isinstance(candidates, (list, tuple)) or not 1 <= len(candidates) <= 8:
        raise ValueError("A continuation query requires 1..8 first-pose candidates.")
    for pose in candidates:
        _tcp_pose(pose, field_name="first_pose_candidates_xyz_quat_xyzw")
    _vector(value["continuation_delta_xyz_m"], 3, field_name="continuation_delta_xyz_m")
    validate_path_constraint(value["path_constraint"])
    if not 0.1 <= _finite(value["time_budget_sec"], field_name="time_budget_sec") <= 30:
        raise ValueError("Query time budget must be 0.1..30 seconds.")


def _diagnostic(value, required, success_keys=frozenset()):
    if not isinstance(value, Mapping):
        raise TypeError("A diagnostic must be an object.")
    reason = required_text(value.get("reason_code"), field_name="reason_code")
    expected = required | (success_keys if reason == "path_found" else frozenset())
    if not expected <= set(value) or set(value) - expected - {"details"}:
        raise ValueError("Diagnostic fields differ from its success/failure contract.")
    _nonnegative(value["wall_sec"], field_name="wall_sec")
    if "details" in value:
        if not isinstance(value["details"], Mapping):
            raise ValueError("Diagnostic details must be an optional JSON object.")
        freeze_json(value["details"], field_name="details")
    return reason


def _candidate(value, count):
    reason = _diagnostic(value, _CANDIDATE_KEYS, {"ranking_cost"})
    index = nonnegative_integer(value["candidate_index"], field_name="candidate_index", maximum=count - 1)
    failed = value["failed_stage"]
    if failed not in (None, "first", "continuation"):
        raise ValueError("failed_stage must be null, first or continuation.")
    stages = value["stage_diagnostics"]
    if not isinstance(stages, (list, tuple)) or len(stages) > 2:
        raise ValueError("A candidate has at most two stage diagnostics.")
    stage_reasons = []
    for expected_index, stage in enumerate(stages):
        stage_reason = _diagnostic(stage, _STAGE_KEYS, _PATH_KEYS)
        stage_index = nonnegative_integer(stage["stage_index"], field_name="stage_index", maximum=1)
        if stage_index != expected_index or (stage_reasons and stage_reasons[-1] != "path_found"):
            raise ValueError("Stages must follow the successful first path in order.")
        if stage_reason == "path_found":
            sha256_digest(stage["trajectory_hash"], field_name="trajectory_hash")
            positive_integer(stage["reference_count"], field_name="reference_count")
            _nonnegative(stage["duration_sec"], field_name="duration_sec")
        stage_reasons.append(stage_reason)
    if reason == "path_found":
        if failed is not None or stage_reasons != ["path_found", "path_found"]:
            raise ValueError("Found candidates require two successful stages and no failed stage.")
        cost = _vector(value["ranking_cost"], 3, field_name="ranking_cost")
        if any(x < 0 for x in cost):
            raise ValueError("Ranking costs must be nonnegative.")
        return index, cost
    if stage_reasons == ["path_found", "path_found"]:
        raise ValueError("A failed candidate cannot claim two completed paths.")
    if stage_reasons and stage_reasons[-1] != "path_found":
        if failed != ("first" if len(stages) == 1 else "continuation"):
            raise ValueError("failed_stage must identify the unsuccessful stage.")
    elif (failed == "first" and stages) or (failed == "continuation" and len(stages) != 1):
        raise ValueError("failed_stage conflicts with the completed stage prefix.")
    return index, None


def validate_continuation_result(result):
    if not isinstance(result, Mapping) or set(result) != RESULT_KEYS:
        raise ValueError("Continuation result must match the exact schema.")
    _identity(result)
    if (result["motion_executed"] is not False or result["execution_authorized"] is not False
            or result["ranking_criteria"] != RANKING
            or result["continuation_source"] != "planned_first_endpoint"
            or result["constraint_scope"] != "frozen_scene_two_stage_sampled_paths"):
        raise ValueError("Continuation evidence cannot claim motion, authority or a different scope.")
    _tcp_pose(result["source_tcp_pose_xyz_quat_xyzw"], field_name="source_tcp_pose_xyz_quat_xyzw")
    for key in ("scene_hash", "source_qpos_hash"):
        sha256_digest(result[key], field_name=key)
    for key in ("source_sim_time_sec", "planning_wall_sec"):
        _nonnegative(result[key], field_name=key)
    count = positive_integer(result["candidate_count"], field_name="candidate_count", maximum=8)
    attempted = nonnegative_integer(result["attempted_count"], field_name="attempted_count", maximum=count)
    diagnostics = result["candidate_diagnostics"]
    if (type(result["all_candidates_attempted"]) is not bool
            or result["all_candidates_attempted"] != (attempted == count)
            or not isinstance(diagnostics, (list, tuple)) or len(diagnostics) != attempted):
        raise ValueError("Candidate attempt accounting is inconsistent.")
    seen, found = set(), {}
    for diagnostic in diagnostics:
        index, cost = _candidate(diagnostic, count)
        if index in seen:
            raise ValueError("Candidate diagnostics must use unique caller indices.")
        seen.add(index)
        if cost is not None:
            found[index] = cost
    index = result["selected_index"]
    if result["query_status"] == "no_path_found":
        if index is not None or found or any(result[key] is not None for key in _TARGET_KEYS):
            raise ValueError("No-path evidence cannot select or expose successful targets.")
    elif result["query_status"] == "path_found":
        nonnegative_integer(index, field_name="selected_index", maximum=count - 1)
        if not found or index != min(found, key=lambda i: (found[i], i)):
            raise ValueError("Selected index must minimize completed ranking cost, then caller index.")
        for key in _TARGET_KEYS:
            _tcp_pose(result[key], field_name=key)
        if any(abs(a - b) > 1e-8 for a, b in zip(
            result["continuation_pose_target_xyz_quat_xyzw"][3:],
            result["predicted_first_tcp_pose_xyz_quat_xyzw"][3:], strict=True,
        )):
            raise ValueError("The continuation must preserve its predicted starting quaternion.")
    else:
        raise ValueError("query_status must be path_found or no_path_found.")


def validate_continuation_result_for_parameters(parameters, result):
    validate_continuation_parameters(parameters)
    validate_continuation_result(result)
    if any(parameters[key] != result[key] for key in ("component_id", "reference_frame_id")):
        raise ValueError("Continuation evidence changed the requested component/frame identity.")
    if result["candidate_count"] != len(parameters["first_pose_candidates_xyz_quat_xyzw"]):
        raise ValueError("Continuation evidence changed the caller candidate set.")
    if result["selected_index"] is None:
        return
    expected_first = parameters["first_pose_candidates_xyz_quat_xyzw"][result["selected_index"]]
    predicted = result["predicted_first_tcp_pose_xyz_quat_xyzw"]
    expected_continuation = [
        predicted[i] + parameters["continuation_delta_xyz_m"][i] for i in range(3)
    ] + list(predicted[3:])
    _tcp_pose(expected_continuation, field_name="computed_continuation_pose")
    for key, expected in (
        ("first_pose_target_xyz_quat_xyzw", expected_first),
        ("continuation_pose_target_xyz_quat_xyzw", expected_continuation),
    ):
        if any(abs(a - b) > 1e-8 for a, b in zip(result[key], expected, strict=True)):
            raise ValueError("Selected targets differ from caller candidate or predicted first FK plus delta.")


def continuation_query_schema(*, definition_hash, profile_hash, reference_frame_id, component_id):
    required_text(reference_frame_id, field_name="reference_frame_id")
    required_text(component_id, field_name="component_id")

    def vector(size):
        return {"type": "array", "minItems": size, "maxItems": size, "items": {"type": "number"}}

    def exact(properties, required=None):
        return {"type": "object", "additionalProperties": False, "properties": properties,
                "required": sorted(properties if required is None else required)}

    nonnegative = {"type": "number", "minimum": 0}
    digest = {"type": "string", "pattern": "^[0-9a-f]{64}$", "minLength": 64, "maxLength": 64}
    identity = {"component_id": {"type": "string", "const": component_id},
                "reference_frame_id": {"type": "string", "const": reference_frame_id}}
    parameters = exact({
        **identity,
        "first_pose_candidates_xyz_quat_xyzw": {
            "type": "array", "minItems": 1, "maxItems": 8, "items": vector(7)},
        "continuation_delta_xyz_m": vector(3),
        "path_constraint": {"anyOf": [{"type": "null"}, _constraint_schema()]},
        "time_budget_sec": {"type": "number", "minimum": 0.1, "maximum": 30},
    })
    # Keep discovery within the wire depth cap. Semantic validators enforce
    # conditional evidence fields, stage order and candidate accounting.
    stage = {"stage_index": {"type": "integer", "minimum": 0, "maximum": 1},
             "reason_code": {"type": "string", "minLength": 1}, "wall_sec": nonnegative,
             "details": {"type": "object"}, "trajectory_hash": digest,
             "reference_count": {"type": "integer", "minimum": 1}, "duration_sec": nonnegative}
    candidate = {
        "candidate_index": {"type": "integer", "minimum": 0, "maximum": 7},
        "reason_code": {"type": "string", "minLength": 1},
        "failed_stage": {"enum": [None, "first", "continuation"]},
        "wall_sec": nonnegative, "details": {"type": "object"},
        "ranking_cost": {**vector(3), "items": nonnegative},
        "stage_diagnostics": {"type": "array", "maxItems": 2,
                              "items": exact(stage, _STAGE_KEYS)},
    }
    properties = {
        **identity,
        "source_tcp_pose_xyz_quat_xyzw": vector(7),
        "scene_hash": digest, "source_qpos_hash": digest, "source_sim_time_sec": nonnegative,
        "query_status": {"enum": ["path_found", "no_path_found"]},
        "selected_index": {"anyOf": [{"type": "null"}, {"type": "integer", "minimum": 0, "maximum": 7}]},
        **{key: {"anyOf": [{"type": "null"}, vector(7)]} for key in _TARGET_KEYS},
        "candidate_count": {"type": "integer", "minimum": 1, "maximum": 8},
        "attempted_count": {"type": "integer", "minimum": 0, "maximum": 8},
        "all_candidates_attempted": {"type": "boolean"},
        "candidate_diagnostics": {"type": "array", "maxItems": 8,
                                  "items": exact(candidate, _CANDIDATE_KEYS)},
        "planning_wall_sec": nonnegative, "ranking_criteria": {"const": RANKING},
        "continuation_source": {"const": "planned_first_endpoint"},
        "constraint_scope": {"const": "frozen_scene_two_stage_sampled_paths"},
        "motion_executed": {"const": False}, "execution_authorized": {"const": False},
    }
    return AgentSkillSchemaEntry(
        skill_id=SKILL_ID, skill_version="1", skill_definition_hash=definition_hash,
        parameter_schema=parameters, execution_result_schema=exact(properties),
        task_evaluation_schema=exact({}),
        parameter_profiles=(SkillParameterProfileRef(
            parameter_profile_id="tcp-continuation-path-query-v1", parameter_profile_hash=profile_hash,
            parameter_schema_hash=agent_skill_schema_document_hash(parameters, schema_kind="parameter"),
        ),),
    )

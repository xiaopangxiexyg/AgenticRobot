"""Portable measured object-state feedback, not task or grasp evaluation."""

from collections.abc import Mapping
import math

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry, SkillDefinition, SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)

SKILL_ID = "state.read_object_state"
IDENTITY_FIELDS = ("object_id", "reference_frame_id", "reference_frame_spec_hash",
                   "frame_graph_hash", "clock_domain_id", "observation_source", "source_binding_hash")
RELEASE_POLICY = {"duration_s": .6, "min_measurements": 60, "max_sample_gap_s": .05,
                  "position_tolerance_m": .002, "rotation_tolerance_rad": .05}
RELEASE_STATUSES = ("not_observed", "not_released", "not_supported", "unexpected_contact",
    "contact_evidence_unavailable", "insufficient_window", "moving", "stable_supported_release",
    "stale", "current_cut_unobserved")


def object_state_definition():
    return SkillDefinition(
        skill_id=SKILL_ID, skill_version="1", skill_kind="inspection",
        summary=("Read a source-bound object body pose, linear/angular velocity and measured contact flags. "
            "Null means unobserved, not absent. Optional release-window evidence is explicitly unavailable "
            "without its deployed producer. A single stationary cut, open command or completed motion "
            "does not prove stable placement, grasping, task success or motion authorization."),
        intended_target_kind="target_agnostic", cross_target_deployment_policy="not_applicable",
        controlled_component_ids=(), parameter_semantics_id="bound_object.reference_frame.freshness.v1",
        execution_result_semantics_id="measured_object_pose.velocity.contact.release_evidence.v1",
        task_outcome_semantics_id="passive_object_state_not_task_evaluation.v1",
        entry_predicate_ids=("bound_object_observation_available.v1",),
        task_outcome_predicate_ids=("object_state_read.v1",))


def _nullable(schema):
    return {"anyOf": [schema, {"type": "null"}]}


def object_state_schema(*, definition_hash, identity, profile_hash, release_supported):
    vector = {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}}
    pose = {**vector, "minItems": 7, "maxItems": 7}
    number = {"type": "number", "minimum": 0}
    parameters = {"type": "object", "additionalProperties": False, "properties": {
        "object_id": {"const": identity["object_id"]},
        "reference_frame_id": {"const": identity["reference_frame_id"]},
        "max_measurement_age_ms": {"type": "integer", "minimum": 1, "maximum": 50},
    }, "required": ["object_id", "reference_frame_id", "max_measurement_age_ms"]}
    release = {
        "status": {"enum": list(RELEASE_STATUSES)}, "stable_supported_release": {"type": "boolean"},
        "measured_at": _nullable(number), "evidence_age_s": _nullable(number),
        "evidence_fresh": {"type": "boolean"}, "window_start_s": _nullable(number),
        "observed_duration_s": number, "measurement_count": {"type": "integer", "minimum": 0, "maximum": 4096},
        "max_observed_sample_gap_s": _nullable(number),
        "max_position_drift_m": number, "max_rotation_drift_rad": {**number, "maximum": math.pi},
        "window_reason": {"enum": ["not_observed", "unexpected_contact", "not_released", "not_supported",
            "contact_evidence_unavailable", "collecting", "observation_gap", "complete"]},
        "policy": {"const": RELEASE_POLICY},
    }
    release_schema = {"type": "object", "additionalProperties": False,
                      "properties": release, "required": list(release)}
    properties = {**{name: {"const": identity[name]} for name in IDENTITY_FIELDS},
        "measured_at": number, "cutoff_at": number,
        "pose_xyz_quat_xyzw": _nullable(pose),
        "linear_velocity_xyz_m_s": _nullable(vector),
        "angular_velocity_xyz_rad_s": _nullable(vector),
        **{name: _nullable({"type": "boolean"}) for name in (
            "support_contact", "gripper_contact", "other_contact")},
        "release_observation": _nullable(release_schema) if release_supported else {"const": None}}
    return AgentSkillSchemaEntry(
        skill_id=SKILL_ID, skill_version="1", skill_definition_hash=definition_hash,
        parameter_schema=parameters,
        execution_result_schema={"type": "object", "additionalProperties": False,
            "properties": properties, "required": list(properties)},
        task_evaluation_schema={"type": "object", "additionalProperties": False, "properties": {}, "required": []},
        parameter_profiles=(SkillParameterProfileRef(parameter_profile_id="bound-object-state-v1",
            parameter_profile_hash=profile_hash,
            parameter_schema_hash=agent_skill_schema_document_hash(parameters, schema_kind="parameter")),))


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Expected finite nonnegative object-state measurement")
    return float(value)


def validate_object_state_parameters(payload, *, identity):
    if (not isinstance(payload, Mapping)
            or set(payload) != {"object_id", "reference_frame_id", "max_measurement_age_ms"}
            or any(payload[key] != identity[key] for key in ("object_id", "reference_frame_id"))):
        raise ValueError("Object query must match the deployed object and reference frame")
    age = payload["max_measurement_age_ms"]
    if isinstance(age, bool) or not isinstance(age, int) or not 1 <= age <= 50:
        raise ValueError("Object query freshness must be 1..50ms")


def validate_object_state_result(payload, *, identity, fields, release_supported):
    if (not isinstance(payload, Mapping) or set(payload) != set(fields)
            or any(payload[key] != identity[key] for key in IDENTITY_FIELDS)):
        raise ValueError("Object result differs from its bound source/frame identity")
    measured, cutoff = _number(payload["measured_at"]), _number(payload["cutoff_at"])
    if not 0 <= cutoff - measured <= .05 + 1e-9:
        raise ValueError("Object result is stale or from the future")
    for name, size in (("pose_xyz_quat_xyzw", 7), ("linear_velocity_xyz_m_s", 3),
                       ("angular_velocity_xyz_rad_s", 3)):
        values = payload[name]
        if values is None:
            continue
        if (not isinstance(values, (tuple, list)) or len(values) != size
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) for v in values)):
            raise ValueError("Object pose/velocity must be measured finite vectors or null")
        if size == 7 and not math.isclose(math.hypot(*values[3:]), 1., rel_tol=0., abs_tol=1e-6):
            raise ValueError("Object orientation must be a measured unit quaternion")
    for name in ("support_contact", "gripper_contact", "other_contact"):
        if payload[name] is not None and type(payload[name]) is not bool:
            raise ValueError("Object contact must be measured bool or null")
    release = payload["release_observation"]
    if release is None:
        return
    expected = {"status", "stable_supported_release", "measured_at", "evidence_age_s", "evidence_fresh",
        "window_start_s", "observed_duration_s", "measurement_count", "max_position_drift_m",
        "max_rotation_drift_rad", "max_observed_sample_gap_s", "window_reason", "policy"}
    if not release_supported or not isinstance(release, Mapping) or set(release) != expected:
        raise ValueError("Release evidence requires its explicit deployed producer")
    if (release["status"] not in RELEASE_STATUSES or release["policy"] != RELEASE_POLICY
            or type(release["evidence_fresh"]) is not bool
            or type(release["stable_supported_release"]) is not bool
            or release["stable_supported_release"] != (release["status"] == "stable_supported_release")):
        raise ValueError("Invalid release status or policy")
    if (not isinstance(release["policy"], Mapping)
            or any(isinstance(release["policy"][key], bool) for key in RELEASE_POLICY)
            or release["window_reason"] not in {"not_observed", "unexpected_contact", "not_released",
                "not_supported", "contact_evidence_unavailable", "collecting", "observation_gap", "complete"}):
        raise ValueError("Invalid release policy or window reason")
    for name in ("observed_duration_s", "max_position_drift_m", "max_rotation_drift_rad"):
        _number(release[name])
    if release["max_rotation_drift_rad"] > math.pi:
        raise ValueError("Invalid release rotation magnitude")
    for name in ("measured_at", "evidence_age_s", "window_start_s", "max_observed_sample_gap_s"):
        if release[name] is not None:
            _number(release[name])
    if release["measured_at"] is not None:
        if (release["evidence_age_s"] is None
                or not math.isclose(measured - release["measured_at"], release["evidence_age_s"],
                                    abs_tol=1e-9, rel_tol=0.)
                or release["evidence_fresh"] != (release["evidence_age_s"] <= .05 + 1e-9)):
            raise ValueError("Release age/freshness differs from its measured source clock")
    elif release["evidence_age_s"] is not None or release["evidence_fresh"]:
        raise ValueError("Unobserved release has no age or freshness")
    if release["window_start_s"] is not None and (
            release["measured_at"] is None or not math.isclose(
                release["measured_at"] - release["window_start_s"], release["observed_duration_s"],
                abs_tol=1e-9, rel_tol=0.)):
        raise ValueError("Release duration differs from its measured window")
    count = release["measurement_count"]
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 4096:
        raise ValueError("Invalid release measurement count")
    max_gap = release["max_observed_sample_gap_s"]
    if max_gap is not None and (count < 2 or
            max_gap * (count - 1) + 1e-9 < release["observed_duration_s"]):
        raise ValueError("Release sample gap contradicts its measured count/duration")
    if release["stable_supported_release"] and (
            not release["evidence_fresh"] or release["measured_at"] != measured
            or release["window_start_s"] is None or release["window_reason"] != "complete"
            or max_gap is None or max_gap > RELEASE_POLICY["max_sample_gap_s"]
            or release["evidence_age_s"] != 0 or release["observed_duration_s"] + 1e-9 < .6
            or count < 60 or release["max_position_drift_m"] > .002
            or release["max_rotation_drift_rad"] > .05 or payload["pose_xyz_quat_xyzw"] is None
            or payload["support_contact"] is not True or payload["gripper_contact"] is not False
            or payload["other_contact"] is not False):
        raise ValueError("Stable release conflicts with its measured window/current cut")


def validate_object_state_result_for_parameters(parameters, result):
    if ((result["object_id"], result["reference_frame_id"]) != (
            parameters["object_id"], parameters["reference_frame_id"])
            or result["cutoff_at"] - result["measured_at"] > parameters["max_measurement_age_ms"] / 1000 + 1e-9):
        raise ValueError("Object result does not satisfy requested identity/freshness")

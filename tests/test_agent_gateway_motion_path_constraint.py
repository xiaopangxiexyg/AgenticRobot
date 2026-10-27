from __future__ import annotations

import inspect
import math
from copy import deepcopy
from types import MappingProxyType

import pytest

from agent_gateway.protocol.skill_catalog import agent_skill_schema_document_hash
from agent_gateway.skills.motion.definition import (
    safe_tcp_pose_target_skill_definition_v1,
)
from agent_gateway.skills.motion.path_constraint import (
    SAFE_TCP_POSE_TARGET_SKILL_ID_V2,
    safe_tcp_pose_target_schema_entry_v2,
    safe_tcp_pose_target_skill_definition_v2,
    validate_path_constraint,
    validate_path_constraint_evidence,
    validate_safe_tcp_pose_parameters_v2,
)
from agent_gateway.skills.motion.schemas import tcp_pose_target_schema_entry_v1

pytestmark = pytest.mark.contract


def _constraint(axis="z", bound=0.2):
    return {"kind": "preserve_start_tool_axis", "tool_axis": axis, "max_deviation_rad": bound}


def _evidence():
    return {
        **_constraint(),
        "activation_axis_in_reference_frame": [0.6, -0.8, 0.0],
        "constraint_identity": "aB09" * 16,
        "planned_max_deviation_rad": 0.1,
        "validated_reference_count": 7,
        "scope": "sampled_planned_and_live_references",
    }


def _parameters():
    return {
        "pose_targets_by_component": {"arm": [0.4, -0.1, 0.5, 0.0, 0.0, 0.0, 1.0]},
        "reference_frame_id": "public_reference",
        "reference_frame_spec_hash": "frame-spec",
        "controlled_frame_id": "tool_tcp",
        "frame_graph_hash": "frame-graph",
        "tool_config_hash": "tool-config",
        "calibration_hash": "calibration",
        "path_constraint": _constraint(),
    }


def _schema_kwargs(**changes):
    definition = safe_tcp_pose_target_skill_definition_v2(controlled_component_ids=("arm",))
    return {
        "skill_definition_hash": definition.semantic_content_hash,
        "parameter_profile_id": "caller-safe-tcp-profile-v2",
        "parameter_profile_hash": "caller-profile-hash",
        "allowed_component_ids": ("arm",),
        "allowed_reference_frame_ids": ("public_reference",),
        "max_requested_duration_ms": 1000,
        "workspace_bounds_by_component": {"arm": (-1.0, 1.0, -1.0, 1.0, 0.0, 1.0)},
        "min_speed_scale": 0.05,
        "max_speed_scale": 1.0,
        **changes,
    }


def test_v2_reuses_safe_identity_and_preserves_v1_definition():
    v1 = safe_tcp_pose_target_skill_definition_v1(controlled_component_ids=("left_arm", "right_arm"))
    before = v1.to_wire()
    v2 = safe_tcp_pose_target_skill_definition_v2(controlled_component_ids=("right_arm", "left_arm"))
    assert v2.skill_id == v1.skill_id == SAFE_TCP_POSE_TARGET_SKILL_ID_V2 == "motion.move_tcp_to_pose_safe"
    assert v2.skill_version == "2"
    assert v1.skill_version == "1"
    assert v2.intended_target_kind == "target_agnostic"
    assert v2.controlled_component_ids == v1.controlled_component_ids
    assert v2.parameter_semantics_id.endswith(".v2")
    assert v2.execution_result_semantics_id.endswith(".v2")
    assert v2.semantic_content_hash != v1.semantic_content_hash
    assert v2.task_outcome_semantics_id == v1.task_outcome_semantics_id
    assert v2.entry_predicate_ids == v1.entry_predicate_ids
    assert v2.task_outcome_predicate_ids == v1.task_outcome_predicate_ids
    assert v1.to_wire() == before
    # Freeze the accepted mainline v1, including its goal-policy semantics.
    assert v1.semantic_content_hash == "dcf19df2fc402065a3f40f1e7383120b233fbea9514eb06a0b83ddd546fc5a12"


@pytest.mark.parametrize("components", [(), ("arm", "arm"), (" arm",), (True,), ["arm"]])
def test_v2_definition_preserves_component_validation(components):
    with pytest.raises((TypeError, ValueError)):
        safe_tcp_pose_target_skill_definition_v2(controlled_component_ids=components)


def test_v2_schema_signature_preserves_v1_options_and_fixes_skill_identity():
    v1 = inspect.signature(tcp_pose_target_schema_entry_v1).parameters
    v2 = inspect.signature(safe_tcp_pose_target_schema_entry_v2).parameters
    assert set(v2) == set(v1) - {"skill_id"}
    for key in v2:
        assert v2[key].kind == v1[key].kind
        assert v2[key].default == v1[key].default
    with pytest.raises(TypeError):
        safe_tcp_pose_target_schema_entry_v2(**_schema_kwargs(), skill_id="motion.other")


@pytest.mark.parametrize("description", [None, "Caller-bound optional parameter semantics."])
@pytest.mark.parametrize("max_target_components", [None, 1])
def test_v2_schema_clones_v1_and_rebinds_caller_profile_to_new_hash(description, max_target_components):
    kwargs = _schema_kwargs(parameter_description=description, max_target_components=max_target_components)
    original_kwargs = deepcopy(kwargs)
    v1 = tcp_pose_target_schema_entry_v1(**kwargs, skill_id=SAFE_TCP_POSE_TARGET_SKILL_ID_V2)
    before = v1.to_wire()
    v2 = safe_tcp_pose_target_schema_entry_v2(**kwargs)
    wire = v2.to_wire()
    assert v2.skill_version == "2"
    assert v2.skill_id == v1.skill_id
    assert v2.skill_definition_hash == kwargs["skill_definition_hash"]
    assert kwargs == original_kwargs
    assert v1.to_wire() == before

    parameters = wire["parameter_schema"]
    assert parameters["additionalProperties"] is False
    policy_schema = parameters["properties"].pop("path_constraint")
    parameters["required"].remove("path_constraint")
    assert parameters == before["parameter_schema"]
    assert policy_schema["anyOf"][0] == {"type": "null"}
    policy_object = policy_schema["anyOf"][1]
    assert policy_object["additionalProperties"] is False
    assert set(policy_object["required"]) == set(_constraint())
    assert set(policy_object["properties"]) == set(_constraint())
    assert policy_object["properties"]["tool_axis"]["enum"] == ["x", "y", "z"]
    assert policy_object["properties"]["max_deviation_rad"] == {
        "type": "number", "exclusiveMinimum": 0.0, "maximum": math.pi,
    }

    result = wire["execution_result_schema"]
    assert result["additionalProperties"] is False
    evidence_schema = result["properties"].pop("path_constraint_evidence")
    result["required"].remove("path_constraint_evidence")
    assert result == before["execution_result_schema"]
    assert evidence_schema["anyOf"][0] == {"type": "null"}
    evidence_object = evidence_schema["anyOf"][1]
    assert evidence_object["additionalProperties"] is False
    assert set(evidence_object["required"]) == set(_evidence())
    assert set(evidence_object["properties"]) == set(_evidence())
    assert evidence_object["properties"]["scope"] == {
        "const": "sampled_planned_and_live_references",
    }
    assert wire["task_evaluation_schema"] == before["task_evaluation_schema"]
    assert v2.task_evaluation_schema_hash == v1.task_evaluation_schema_hash
    assert v2.execution_result_schema_hash != v1.execution_result_schema_hash
    assert v2.parameter_schema_hash != v1.parameter_schema_hash
    profile, = v2.parameter_profiles
    assert profile.parameter_profile_id == kwargs["parameter_profile_id"]
    assert profile.parameter_profile_hash == kwargs["parameter_profile_hash"]
    assert profile.parameter_schema_hash == v2.parameter_schema_hash
    assert profile.parameter_schema_hash == agent_skill_schema_document_hash(
        v2.parameter_schema, schema_kind="parameter",
    )
    assert safe_tcp_pose_target_schema_entry_v2(**kwargs).to_wire() == v2.to_wire()
    with pytest.raises(TypeError):
        v2.parameter_schema["properties"]["path_constraint"] = None


def test_v2_schema_preserves_absent_optional_profile_fields():
    entry = safe_tcp_pose_target_schema_entry_v2(**_schema_kwargs(
        max_requested_duration_ms=None, min_speed_scale=None, max_speed_scale=None,
        workspace_bounds_by_component=None,
    ))
    properties = entry.parameter_schema["properties"]
    assert "requested_duration_ms" not in properties
    assert "speed_scale" not in properties
    assert "path_constraint" in entry.parameter_schema["required"]
    assert "path_constraint_evidence" in entry.execution_result_schema["required"]


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("bound", [math.nextafter(0.0, 1.0), 0.1, 1, math.pi])
def test_valid_policy_is_accepted_without_transformation(axis, bound):
    policy = _constraint(axis, bound)
    before = deepcopy(policy)
    assert validate_path_constraint(MappingProxyType(policy)) is None
    assert policy == before
    assert validate_path_constraint(None) is None


@pytest.mark.parametrize("value", [True, False, [], "null", 0, {},
    {"kind": "preserve_start_tool_axis", "tool_axis": "z"},
    {**_constraint(), "relax": True},
    {**_constraint(), "kind": "other"},
    {**_constraint(), "kind": True},
    {**_constraint(), "tool_axis": "-z"},
    {**_constraint(), "tool_axis": "+z"},
    {**_constraint(), "tool_axis": True},
])
def test_policy_rejects_wrong_shape_kind_or_axis(value):
    with pytest.raises((TypeError, ValueError)):
        validate_path_constraint(value)


@pytest.mark.parametrize("bound", [True, False, None, "0.2", -0.1, 0,
    math.nextafter(math.pi, math.inf), math.nan, math.inf, -math.inf, 10**400,
])
def test_policy_rejects_invalid_numeric_bound(bound):
    with pytest.raises((TypeError, ValueError)):
        validate_path_constraint(_constraint(bound=bound))


def test_evidence_validates_standalone_or_exact_request_without_mutation():
    evidence = _evidence()
    before = deepcopy(evidence)
    assert validate_path_constraint_evidence(MappingProxyType(evidence)) is None
    assert validate_path_constraint_evidence(evidence, constraint=_constraint()) is None
    assert evidence == before
    assert validate_path_constraint_evidence(None) is None
    assert validate_path_constraint_evidence(None, constraint=None) is None
    evidence["activation_axis_in_reference_frame"] = (-1.0, 0.0, 0.0)
    validate_path_constraint_evidence(evidence)


@pytest.mark.parametrize("evidence, constraint", [
    (None, _constraint()), (_evidence(), None),
    (_evidence(), _constraint(axis="x")), (_evidence(), _constraint(bound=0.3)),
    (None, {}), (_evidence(), {**_constraint(), "extra": True}),
])
def test_evidence_enforces_explicit_null_and_request_matching(evidence, constraint):
    with pytest.raises((TypeError, ValueError)):
        validate_path_constraint_evidence(evidence, constraint=constraint)


@pytest.mark.parametrize("field", sorted(_evidence()))
def test_evidence_requires_each_exact_field(field):
    evidence = _evidence()
    evidence.pop(field)
    with pytest.raises(ValueError):
        validate_path_constraint_evidence(evidence)


@pytest.mark.parametrize("evidence", [True, [], "null", 0, {**_evidence(), "live_max": 0.1}])
def test_evidence_rejects_wrong_shape_or_additional_fields(evidence):
    with pytest.raises(ValueError):
        validate_path_constraint_evidence(evidence)


@pytest.mark.parametrize("field, value", [
    ("kind", "other"), ("tool_axis", "-z"), ("max_deviation_rad", True),
    ("max_deviation_rad", 0), ("max_deviation_rad", math.pi + 0.1),
    ("max_deviation_rad", math.nan),
    ("activation_axis_in_reference_frame", [0, 0]),
    ("activation_axis_in_reference_frame", [0, 0, 1, 0]),
    ("activation_axis_in_reference_frame", [0, 0, 0]),
    ("activation_axis_in_reference_frame", [0, 0, 2]),
    ("activation_axis_in_reference_frame", [True, 0, 0]),
    ("activation_axis_in_reference_frame", [math.nan, 0, 1]),
    ("activation_axis_in_reference_frame", [math.inf, 0, 1]),
    ("activation_axis_in_reference_frame", [0, 0, "1"]),
    ("activation_axis_in_reference_frame", [0, 0, 1 - 2e-9]),
    ("constraint_identity", "a" * 63), ("constraint_identity", "a" * 65),
    ("constraint_identity", "g" * 64), ("constraint_identity", "a" * 64 + "\n"),
    ("constraint_identity", True),
    ("planned_max_deviation_rad", True), ("planned_max_deviation_rad", "0.1"),
    ("planned_max_deviation_rad", -1e-12), ("planned_max_deviation_rad", 0.2 + 2e-9),
    ("planned_max_deviation_rad", math.nan), ("planned_max_deviation_rad", math.inf),
    ("planned_max_deviation_rad", -math.inf),
    ("validated_reference_count", True), ("validated_reference_count", 0),
    ("validated_reference_count", -1), ("validated_reference_count", 1.0),
    ("validated_reference_count", "1"), ("scope", "continuous"),
    ("scope", "sampled_planned_references"),
])
def test_evidence_rejects_invalid_values(field, value):
    evidence = {**_evidence(), field: value}
    with pytest.raises((TypeError, ValueError)):
        validate_path_constraint_evidence(evidence)


@pytest.mark.parametrize("maximum", [0, 0.2, 0.2 + 1e-9])
def test_evidence_accepts_planned_deviation_bound_and_tolerance(maximum):
    validate_path_constraint_evidence({**_evidence(), "planned_max_deviation_rad": maximum})


@pytest.mark.parametrize("policy", [None, _constraint()])
def test_parameters_preserve_absolute_pose_and_accept_explicit_policy(policy):
    parameters = {**_parameters(), "path_constraint": policy,
        "requested_duration_ms": 100, "speed_scale": 0.3}
    before = deepcopy(parameters)
    assert validate_safe_tcp_pose_parameters_v2(MappingProxyType(parameters), ("arm",)) is None
    assert parameters == before


@pytest.mark.parametrize("field", sorted(_parameters()))
def test_parameters_require_all_v1_fields_and_path_constraint(field):
    parameters = _parameters()
    parameters.pop(field)
    with pytest.raises(ValueError):
        validate_safe_tcp_pose_parameters_v2(parameters, ("arm",))


@pytest.mark.parametrize("field, value", [
    ("path_constraint", {**_constraint(), "relax": True}),
    ("reference_frame_id", ""), ("reference_frame_id", True),
    ("reference_frame_spec_hash", " frame"), ("frame_graph_hash", None),
    ("tool_config_hash", ""), ("calibration_hash", "calibration "),
    ("controlled_frame_id", "endpoint"),
    ("requested_duration_ms", True), ("requested_duration_ms", 0),
    ("requested_duration_ms", 1.0), ("requested_duration_ms", math.inf),
    ("speed_scale", True), ("speed_scale", 0), ("speed_scale", 1.1),
    ("speed_scale", math.nan), ("speed_scale", math.inf),
    ("pose_targets_by_component", {}),
    ("pose_targets_by_component", {"other": [0, 0, 0, 0, 0, 0, 1]}),
    ("pose_targets_by_component", {"arm": [0, 0, 0, 0, 0, 0, 2]}),
    ("pose_targets_by_component", {"arm": [True, 0, 0, 0, 0, 0, 1]}),
    ("pose_targets_by_component", {"arm": [math.nan, 0, 0, 0, 0, 0, 1]}),
    ("pose_targets_by_component", {"arm": [math.inf, 0, 0, 0, 0, 0, 1]}),
    ("joint_targets", [0]),
])
def test_parameters_reject_invalid_pose_identity_policy_or_extra_fields(field, value):
    with pytest.raises((TypeError, ValueError)):
        validate_safe_tcp_pose_parameters_v2({**_parameters(), field: value}, ("arm",))


def test_parameters_reject_non_mapping():
    with pytest.raises(TypeError):
        validate_safe_tcp_pose_parameters_v2(None, ("arm",))

"""Offline contracts for explicit state.read_tcp_pose V2, without a backend."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import replace

import pytest

from agent_gateway.protocol.contracts import (
    AGENT_SKILL_EXECUTION_OUTCOMES_V1,
    AGENT_SKILL_TASK_OUTCOMES_V1,
    AGENT_SKILL_TERMINAL_OPERATION_STATUSES_V1,
)
from agent_gateway.protocol.state_schemas import (
    tcp_pose_state_definition_v1,
    tcp_pose_state_schema_entry_v1,
    validate_agent_state_parameters_v1,
    validate_agent_state_result_v1,
)
from agent_gateway.skills.state import (
    tcp_pose_comparison_definition_v2,
    tcp_pose_comparison_schema_entry_v2,
    validate_tcp_pose_comparison_parameters_v2,
    validate_tcp_pose_comparison_result_shape_v2,
    validate_tcp_pose_comparison_result_v2,
)

pytestmark = pytest.mark.contract
COMPONENTS = ("left_arm", "right_arm")
FRAMES = ("robot_base",)
TARGET = "opaque:retained-safe-operation"


def _base_schema():
    definition = tcp_pose_state_definition_v1(controlled_component_ids=COMPONENTS)
    return tcp_pose_state_schema_entry_v1(
        skill_definition_hash=definition.semantic_content_hash,
        parameter_profile_id="test-tcp-state-v1", parameter_profile_hash="a" * 64,
        allowed_component_ids=COMPONENTS, allowed_reference_frame_ids=FRAMES,
        max_measurement_age_ms=500,
    )


def _parameters(target=TARGET, components=("right_arm",)):
    return {"component_ids": list(components), "reference_frame_id": "robot_base",
                "controlled_frame_id": "tool_tcp", "max_measurement_age_ms": 250,
                "target_operation_id": target}


def _result(target=TARGET, components=("right_arm",), *, target_components=None):
    compared = components if target_components is None else target_components
    result = {
        "observation_cut_id": "cut:12", "observation_cut_hash": "1" * 64,
        "cutoff_at": 12.0, "cutoff_clock_domain_id": "owner-clock",
        "component_ids": list(components), "reference_frame_id": "robot_base",
        "reference_frame_spec_hash": "2" * 64, "controlled_frame_id": "tool_tcp",
        "frame_graph_hash": "3" * 64, "tool_config_hash": "4" * 64, "calibration_hash": "5" * 64,
        "measurement_id": "measurement:latest", "measurement_observation_hash": "6" * 64,
        "source_measurement_hash": "7" * 64, "measured_at": 11.9, "available_at": 11.91,
        "clock_domain_id": "owner-clock", "observation_binding_id": "observation:fixed",
        "observation_plan_hash": "8" * 64,
        "pose_by_component": {c: [.2, -.1, .3, 0, 0, 0, 1] for c in components},
        "pose_schema_hash_by_component": {c: "9" * 64 for c in components},
        "representation_by_component": {c: "cartesian_pose" for c in components},
    }
    result["target_comparison"] = None if target is None else {
        "target_operation_id": target, "target_request_hash": "a" * 64,
        "target_route_hash": "b" * 64, "target_skill_version": "2", "target_status": "failed",
        "target_execution_outcome": "completed", "target_task_outcome": "failed",
        "target_pose_by_component": {c: [.21, -.1, .3, 0, 0, 0, 1] for c in compared},
        "translation_error_m_by_component": {c: .01 for c in compared},
        "rotation_error_rad_by_component": {c: 0.0 for c in compared},
    }
    return result


def _validate_parameters(parameters):
    validate_tcp_pose_comparison_parameters_v2(
        parameters, allowed_component_ids=COMPONENTS,
        allowed_reference_frame_ids=FRAMES, max_measurement_age_ms=500,
    )


def _validate_result(result, parameters=None):
    validate_tcp_pose_comparison_result_v2(
        _parameters() if parameters is None else parameters, result,
        allowed_component_ids=COMPONENTS,
    )


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def test_definition_reuses_v1_identity_and_does_not_select_a_target_or_outcome():
    v1 = tcp_pose_state_definition_v1(controlled_component_ids=COMPONENTS)
    before = _bytes(v1.to_wire())
    v2 = tcp_pose_comparison_definition_v2(controlled_component_ids=COMPONENTS)
    assert v2.skill_id == v1.skill_id == "state.read_tcp_pose"
    assert v2.skill_version == "2" and v1.skill_version == "1"
    assert v2.skill_kind == "inspection" and v2.intended_target_kind == "target_agnostic"
    assert v2.cross_target_deployment_policy == "not_applicable"
    assert v2.controlled_component_ids == COMPONENTS
    assert v2.task_outcome_semantics_id == v1.task_outcome_semantics_id
    assert v2.task_outcome_predicate_ids == v1.task_outcome_predicate_ids == ()
    assert v2.semantic_content_hash != v1.semantic_content_hash
    assert _bytes(v1.to_wire()) == before


def test_schema_extends_detached_v1_wire_without_mutation_or_new_source_fields():
    base = _base_schema(); before = _bytes(base.to_wire())
    definition = tcp_pose_comparison_definition_v2(controlled_component_ids=COMPONENTS)
    entry = tcp_pose_comparison_schema_entry_v2(
        base_schema=base, definition_hash=definition.semantic_content_hash, profile_hash="c" * 64,
    )
    wire = entry.to_wire(); old = base.to_wire()
    assert _bytes(base.to_wire()) == before
    assert entry.skill_version == "2" and entry.skill_definition_hash == definition.semantic_content_hash
    assert entry.parameter_profiles[0].parameter_profile_id == "tcp-pose-comparison-v2"
    assert entry.parameter_profiles[0].parameter_profile_hash == "c" * 64
    assert entry.parameter_profiles[0].parameter_schema_hash == entry.parameter_schema_hash
    for name, extra in (("parameter_schema", "target_operation_id"), ("execution_result_schema", "target_comparison")):
        extended = deepcopy(wire[name])
        assert set(extended["properties"]) == set(old[name]["properties"]) | {extra}
        assert set(extended["required"]) == set(old[name]["required"]) | {extra}
        extended["properties"].pop(extra); extended["required"].remove(extra)
        assert _bytes(extended) == _bytes(old[name])
    assert entry.task_evaluation_schema_hash == base.task_evaluation_schema_hash
    assert wire["task_evaluation_schema"] == old["task_evaluation_schema"]
    assert entry.parameter_schema_hash != base.parameter_schema_hash
    assert entry.execution_result_schema_hash != base.execution_result_schema_hash
    comparison = wire["execution_result_schema"]["properties"]["target_comparison"]["anyOf"][0]
    assert set(comparison["properties"]) == set(comparison["required"]) == set(_result()["target_comparison"])
    assert "target_updated_at" not in comparison["properties"]
    assert "measurement_after_target_terminal" not in comparison["properties"]
    wire["execution_result_schema"]["properties"]["pose_by_component"]["properties"].clear()
    assert _bytes(base.to_wire()) == before
    assert entry.to_wire()["execution_result_schema"]["properties"]["pose_by_component"]["properties"]


@pytest.mark.parametrize("wrong", [None, "schema", replace(_base_schema(), skill_version="2"), replace(_base_schema(), skill_id="state.read_tcp_pose_history")])
def test_schema_rejects_non_v1_pose_base(wrong):
    with pytest.raises((TypeError, ValueError)):
        tcp_pose_comparison_schema_entry_v2(base_schema=wrong, definition_hash="d" * 64, profile_hash="e" * 64)


@pytest.mark.parametrize("components", [("right_arm",), ("left_arm",), COMPONENTS, tuple(reversed(COMPONENTS))])
@pytest.mark.parametrize("target", [None, TARGET])
def test_explicit_null_and_comparison_are_valid_without_mutating_inputs(components, target):
    parameters = _parameters(target, components); result = _result(target, components)
    before = deepcopy((parameters, result))
    _validate_parameters(parameters); validate_tcp_pose_comparison_result_shape_v2(result)
    _validate_result(result, parameters)
    assert (parameters, result) == before


def test_old_four_parameters_remain_v1_and_are_never_defaulted_to_v2_null():
    parameters = _parameters(None); parameters.pop("target_operation_id")
    validate_agent_state_parameters_v1(schema_kind="tcp_pose", parameters=parameters, allowed_component_ids=COMPONENTS)
    v1_result = _result(None); v1_result.pop("target_comparison")
    validate_agent_state_result_v1(schema_kind="tcp_pose", parameters=parameters, result=v1_result, allowed_component_ids=COMPONENTS)
    with pytest.raises(ValueError, match="explicit target_operation_id"):
        _validate_parameters(parameters)
    with pytest.raises(ValueError):
        _validate_result(_result(None), parameters)
    assert "target_operation_id" not in parameters
    with pytest.raises(ValueError, match="target_comparison"):
        _validate_result(v1_result)


@pytest.mark.parametrize("target", ["", " ", " leading", "trailing ", "x" * 129, True, 1, [], {}])
def test_invalid_opaque_target_references_fail(target):
    with pytest.raises((TypeError, ValueError)):
        _validate_parameters(_parameters(target))


@pytest.mark.parametrize("target", ["x", "x" * 128, "opaque/../value", "目标:original"])
def test_operation_reference_is_opaque_and_bounded_not_a_path_or_hash(target):
    _validate_parameters(_parameters(target)); _validate_result(_result(target), _parameters(target))


@pytest.mark.parametrize("key,value", [
    ("reference_frame_id", "foreign"), ("reference_frame_id", True), ("controlled_frame_id", "flange"),
    ("max_measurement_age_ms", 501), ("max_measurement_age_ms", True), ("max_measurement_age_ms", 0),
    ("max_measurement_age_ms", 1.5), ("component_ids", []), ("component_ids", ["right_arm", "right_arm"]),
    ("component_ids", ["foreign"]), ("component_ids", [True]),
])
def test_request_preserves_v1_component_frame_and_age_rejections(key, value):
    parameters = _parameters(); parameters[key] = value
    with pytest.raises((TypeError, ValueError)):
        _validate_parameters(parameters)


@pytest.mark.parametrize("extra", ["target_pose_by_component", "tolerance", "permission", "source_binding_id"])
def test_caller_cannot_supply_target_threshold_source_or_authority(extra):
    parameters = _parameters(); parameters[extra] = {}
    with pytest.raises(ValueError): _validate_parameters(parameters)
    with pytest.raises(ValueError): _validate_result(_result(), parameters)


@pytest.mark.parametrize("target,comparison", [(None, {}), (None, _result()["target_comparison"]), (TARGET, None), (TARGET, {})])
def test_request_and_result_null_must_match(target, comparison):
    result = _result(); result["target_comparison"] = comparison
    with pytest.raises((TypeError, ValueError)):
        _validate_result(result, _parameters(target))


@pytest.mark.parametrize("field", sorted(_result()["target_comparison"]))
def test_every_comparison_field_is_required(field):
    result = _result(); result["target_comparison"].pop(field)
    with pytest.raises(ValueError): validate_tcp_pose_comparison_result_shape_v2(result)


@pytest.mark.parametrize("extra", ["tolerance", "verdict", "task_evaluation", "source_binding_id", "target_skill_id", "units"])
def test_comparison_is_closed_without_tolerance_verdict_evaluation_or_source(extra):
    result = _result(); result["target_comparison"][extra] = None
    with pytest.raises(ValueError): validate_tcp_pose_comparison_result_shape_v2(result)


@pytest.mark.parametrize("key,value", [
    ("target_skill_version", "3"), ("target_skill_version", 2), ("target_skill_version", True),
    ("target_status", "running"), ("target_status", "succeeded"), ("target_status", None),
    ("target_execution_outcome", "succeeded"), ("target_execution_outcome", True),
    ("target_task_outcome", "completed"), ("target_task_outcome", []),
    ("target_request_hash", "request:opaque"), ("target_request_hash", "A" * 64),
    ("target_route_hash", "b" * 63), ("target_route_hash", None), ("target_operation_id", "x" * 129),
])
def test_comparison_metadata_rejects_bad_hash_status_version_and_outcome(key, value):
    result = _result(); result["target_comparison"][key] = value
    with pytest.raises((TypeError, ValueError)):
        validate_tcp_pose_comparison_result_shape_v2(result)


@pytest.mark.parametrize("status", AGENT_SKILL_TERMINAL_OPERATION_STATUSES_V1)
@pytest.mark.parametrize("execution", AGENT_SKILL_EXECUTION_OUTCOMES_V1)
@pytest.mark.parametrize("task", AGENT_SKILL_TASK_OUTCOMES_V1)
def test_original_outcomes_are_enums_not_rewritten_from_terminal_status(status, execution, task):
    result = _result(); result["target_comparison"].update(target_status=status, target_execution_outcome=execution, target_task_outcome=task)
    before = deepcopy(result["target_comparison"])
    _validate_result(result)
    assert result["target_comparison"] == before


@pytest.mark.parametrize("removed,value", [
    ("target_updated_at", 11.8), ("measurement_after_target_terminal", True),
])
def test_removed_historical_timing_fields_are_not_optional_or_ignored(removed, value):
    result = _result(); result["target_comparison"][removed] = value
    with pytest.raises(ValueError, match="exact non-null target_comparison fields"):
        validate_tcp_pose_comparison_result_shape_v2(result)
    with pytest.raises(ValueError, match="exact non-null target_comparison fields"):
        _validate_result(result)


@pytest.mark.parametrize("mapping", ["target_pose_by_component", "translation_error_m_by_component", "rotation_error_rad_by_component"])
@pytest.mark.parametrize("keys", [(), ("left_arm",), COMPONENTS, ("foreign",)])
def test_comparison_mappings_reject_missing_or_unknown_keys(mapping, keys):
    result = _result(); values = result["target_comparison"][mapping]
    result["target_comparison"][mapping] = {key: deepcopy(values["right_arm"]) for key in keys}
    with pytest.raises(ValueError, match="[Cc]omparison components"):
        validate_tcp_pose_comparison_result_shape_v2(result)


def test_one_bimanual_cut_can_compare_only_the_authenticated_right_arm_target():
    parameters = _parameters(components=COMPONENTS)
    result = _result(components=COMPONENTS, target_components=("right_arm",))
    before = deepcopy(result)
    _validate_parameters(parameters)
    validate_tcp_pose_comparison_result_shape_v2(result)
    _validate_result(result, parameters)
    assert set(result["pose_by_component"]) == set(COMPONENTS)
    for name in ("target_pose_by_component", "translation_error_m_by_component", "rotation_error_rad_by_component"):
        assert set(result["target_comparison"][name]) == {"right_arm"}
    assert result == before


@pytest.mark.parametrize("targets", [(), ("foreign",), ("right_arm", "foreign")])
def test_all_three_maps_cannot_agree_on_empty_or_unknown_components(targets):
    result = _result(components=COMPONENTS, target_components=targets)
    with pytest.raises(ValueError, match="non-empty subset"):
        validate_tcp_pose_comparison_result_shape_v2(result)


@pytest.mark.parametrize("field", ["target_pose_by_component", "translation_error_m_by_component", "rotation_error_rad_by_component"])
def test_comparison_maps_cannot_disagree_even_within_measured_components(field):
    result = _result(components=COMPONENTS, target_components=("right_arm",))
    mapping = result["target_comparison"][field]
    mapping["left_arm"] = deepcopy(mapping["right_arm"])
    with pytest.raises(ValueError, match="comparison components"):
        _validate_result(result, _parameters(components=COMPONENTS))


@pytest.mark.parametrize("pose", [
    [], [0] * 6, [0] * 8, [0] * 7, [0, 0, 0, 0, 0, 0, 2],
    [False, 0, 0, 0, 0, 0, 1], [0, 0, 0, False, 0, 0, 1],
    [float("nan"), 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, float("inf"), 1],
    [0, 0, 0, 0, 0, 0, "1"],
])
def test_target_pose_requires_finite_xyz_and_unit_quaternion(pose):
    result = _result(); result["target_comparison"]["target_pose_by_component"]["right_arm"] = pose
    with pytest.raises((TypeError, ValueError)):
        validate_tcp_pose_comparison_result_shape_v2(result)


def test_target_quaternion_sign_and_v1_unit_norm_roundoff_are_allowed_without_repair():
    for q in (-1.0, 1.0, 1.0 + 5e-6):
        result = _result(); pose = result["target_comparison"]["target_pose_by_component"]["right_arm"]
        pose[6] = q; before = deepcopy(pose)
        _validate_result(result)
        assert pose == before


@pytest.mark.parametrize("field", ["translation_error_m_by_component", "rotation_error_rad_by_component"])
@pytest.mark.parametrize("error", [-1, True, float("nan"), float("inf"), "1 mm", None])
def test_metric_errors_must_be_finite_nonnegative_numbers(field, error):
    result = _result(); result["target_comparison"][field]["right_arm"] = error
    with pytest.raises((TypeError, ValueError)):
        validate_tcp_pose_comparison_result_shape_v2(result)


def test_rotation_is_radians_with_only_numeric_roundoff_beyond_pi():
    for angle in (0.0, math.pi, math.pi + 5e-13):
        result = _result(); result["target_comparison"]["rotation_error_rad_by_component"]["right_arm"] = angle
        _validate_result(result)
    for angle in (math.pi + 2e-12, 180.0):
        result = _result(); result["target_comparison"]["rotation_error_rad_by_component"]["right_arm"] = angle
        with pytest.raises(ValueError): _validate_result(result)


@pytest.mark.parametrize("key,value", [
    ("measured_at", 11.0), ("available_at", 12.1), ("clock_domain_id", "foreign-clock"),
    ("reference_frame_id", "foreign-frame"), ("controlled_frame_id", "flange"),
    ("component_ids", ["left_arm"]), ("measured_at", True), ("cutoff_at", float("nan")),
])
@pytest.mark.parametrize("target", [None, TARGET])
def test_contextual_validation_keeps_v1_freshness_causality_frame_and_component_gates(key, value, target):
    result = _result(target); result[key] = value
    with pytest.raises((TypeError, ValueError)):
        _validate_result(result, _parameters(target))


def test_contextual_validation_checks_target_id_and_all_requested_components():
    with pytest.raises(ValueError, match="does not match"):
        _validate_result(_result("another-operation"))
    with pytest.raises(ValueError, match="component_ids"):
        _validate_result(_result(components=COMPONENTS))
    result = _result(); result["source_binding_id"] = "unrequested-source-projection"
    with pytest.raises(ValueError, match="exactly the V1"):
        _validate_result(result)


def test_bare_validator_uses_no_fabricated_parameters_or_freshness_policy():
    validate_tcp_pose_comparison_result_shape_v2({"target_comparison": None})
    result = _result(); result.pop("measured_at"); result.pop("cutoff_at")
    validate_tcp_pose_comparison_result_shape_v2(result)
    with pytest.raises(ValueError): _validate_result(result)


def test_shape_does_not_authenticate_targets_or_recompute_profile_geometry():
    result = _result(); result["target_comparison"]["translation_error_m_by_component"]["right_arm"] = 123.0
    _validate_result(result)
    # The fixed assembly/profile must authenticate the target and recompute errors.
    assert result["target_comparison"]["translation_error_m_by_component"]["right_arm"] == 123.0

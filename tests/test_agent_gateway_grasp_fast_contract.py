"""Pure fast-grasp contracts; no models, devices, provider or execution owner."""

from __future__ import annotations

from copy import deepcopy

import pytest

from agent_gateway.protocol.skill_catalog import agent_skill_schema_document_hash
from agent_gateway.skills.perception.grasp_fast_plan_schemas import (
    fast_grasp_plan_schema_entry_v1,
    fast_grasp_plan_skill_definition_v1,
    validate_fast_grasp_plan_parameters_v1,
    validate_fast_grasp_plan_result_for_parameters_v1,
    validate_fast_grasp_plan_result_v1,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


@pytest.fixture
def entry():
    definition = fast_grasp_plan_skill_definition_v1(
        controlled_component_ids=("right_arm",)
    )
    return fast_grasp_plan_schema_entry_v1(
        skill_definition_hash=definition.semantic_content_hash,
        parameter_profile_id="fast-grasp:test",
        parameter_profile_hash="a" * 64,
        component_id="right_arm",
        side="right",
        roles=("main", "wrist_right"),
        min_speed_scale=0.1,
        max_speed_scale=0.5,
        reference_frame_id="station",
        reference_frame_spec_hash="b" * 64,
        frame_graph_hash="c" * 64,
        tool_config_hash="d" * 64,
        calibration_hash="e" * 64,
    )


def _parameters():
    return {
        "component_id": "right_arm",
        "observation_operation_id": "observation:test",
        "role": "main",
        "target_pixel_uv": [123, 234],
        "view_policy": "auto",
        "speed_scale": 0.3,
        "planning_budget_sec": 10,
    }


def _result(*, success=True):
    return {
        "planning_status": "path_found",
        "reason_code": "path_found",
        "component_id": "right_arm",
        "side": "right",
        "observation_operation_id": "observation:test",
        "observation_content_hash": "f" * 64,
        "role": "main",
        "target_pixel_uv": [123, 234],
        "reference_frame_id": "station",
        "reference_frame_spec_hash": "b" * 64,
        "frame_graph_hash": "c" * 64,
        "tool_config_hash": "d" * 64,
        "calibration_hash": "e" * 64,
        "source_station_revision": 7,
        "T_reference_from_tool_tcp": [
            1,
            0,
            0,
            0.4,
            0,
            1,
            0,
            0.1,
            0,
            0,
            1,
            0.2,
            0,
            0,
            0,
            1,
        ]
        if success
        else None,
        "gripper_width_m": 0.04 if success else None,
        "grasp_approach_plan_id": "real-grasp-approach-plan:" + "1" * 64
        if success
        else None,
        "grasp_approach_plan_hash": "1" * 64 if success else None,
        "trajectory_hash": "2" * 64 if success else None,
        "trajectory_sample_count": 200 if success else None,
        "trajectory_duration_sec": 2.5 if success else None,
        "plan_created_at": 10.0 if success else None,
        "plan_expires_at": 30.0 if success else None,
        "eligible_for_execution": success,
        "execution_requires_explicit_motion_skill": True,
        "motion_or_hold_performed": False,
        "planning_policy": "first_acceptable_v1",
        "diagnostics": {
            "pool_count": 10,
            "path_query_count": 2,
            "elapsed_sec": 1.5,
            "attempts_json": '[{"reason_code":"path_found"}]',
        },
    } | (
        {}
        if success
        else {"planning_status": "rejected", "reason_code": "budget_exhausted"}
    )


def test_definition_and_closed_schemas_bind_exact_semantics_without_execution(entry):
    definition = fast_grasp_plan_skill_definition_v1(
        controlled_component_ids=("right_arm",)
    )
    assert definition.skill_id == entry.skill_id == "perception.plan_grasp_fast"
    assert definition.skill_version == entry.skill_version == "1"
    assert definition.skill_kind == "inspection"
    assert "not physical grasp success" in definition.summary
    assert set(entry.parameter_schema["required"]) == set(_parameters())
    assert set(entry.execution_result_schema["required"]) == set(_result())
    assert "candidate_ordinal" not in entry.parameter_schema["properties"]
    assert "point_grasp_operation_id" not in entry.execution_result_schema["properties"]
    assert entry.parameter_profiles[
        0
    ].parameter_schema_hash == agent_skill_schema_document_hash(
        entry.parameter_schema,
        schema_kind="parameter",
    )
    for schema in (
        entry.parameter_schema,
        entry.execution_result_schema,
        entry.task_evaluation_schema,
    ):
        assert schema["additionalProperties"] is False
    with pytest.raises(TypeError):
        entry.parameter_schema["properties"]["component_id"]["const"] = "left_arm"


@pytest.mark.parametrize("view,budget,speed", [("single", 1, 0.1), ("auto", 30, 0.5)])
def test_parameters_accept_exact_deployment_and_budget_edges(
    entry, view, budget, speed
):
    payload = {
        **_parameters(),
        "view_policy": view,
        "planning_budget_sec": budget,
        "speed_scale": speed,
    }
    before = deepcopy(payload)
    validate_fast_grasp_plan_parameters_v1(payload, entry.parameter_schema)
    assert payload == before


@pytest.mark.parametrize(
    "changes",
    [
        {"component_id": "left_arm"},
        {"observation_operation_id": ""},
        {"observation_operation_id": " "},
        {"observation_operation_id": "x" * 257},
        {"role": "wrist_left"},
        {"target_pixel_uv": [1.0, 2]},
        {"target_pixel_uv": [True, 2]},
        {"target_pixel_uv": [-1, 2]},
        {"target_pixel_uv": [1, 2, 3]},
        {"view_policy": "multi"},
        {"speed_scale": 0.09},
        {"speed_scale": 0.51},
        {"speed_scale": True},
        {"speed_scale": float("nan")},
        {"planning_budget_sec": 0.99},
        {"planning_budget_sec": 30.01},
        {"planning_budget_sec": float("inf")},
        {"planning_budget_sec": True},
        {"candidate_ordinal": 0},
        {"point_grasp_operation_id": "fabricated"},
        {"planning_policy": "best_score"},
    ],
)
def test_parameters_reject_wrong_source_shape_scope_or_bounds(entry, changes):
    with pytest.raises((TypeError, ValueError)):
        validate_fast_grasp_plan_parameters_v1(
            {**_parameters(), **changes}, entry.parameter_schema
        )


@pytest.mark.parametrize("missing", list(_parameters()))
def test_all_parameters_are_required(entry, missing):
    payload = _parameters()
    del payload[missing]
    with pytest.raises(ValueError, match="fields"):
        validate_fast_grasp_plan_parameters_v1(payload, entry.parameter_schema)


@pytest.mark.parametrize("success", [True, False])
def test_success_and_rejection_preserve_frozen_lineage_without_mutating_results(
    entry, success
):
    result = _result(success=success)
    original = deepcopy(result)
    validate_fast_grasp_plan_result_v1(result, entry.execution_result_schema)
    validate_fast_grasp_plan_result_for_parameters_v1(_parameters(), result)
    assert result == original
    # Results may be read back from the immutable operation cache as tuples.
    result["target_pixel_uv"] = tuple(result["target_pixel_uv"])
    validate_fast_grasp_plan_result_for_parameters_v1(_parameters(), result)
    result["diagnostics"] = {}
    validate_fast_grasp_plan_result_v1(result, entry.execution_result_schema)


@pytest.mark.parametrize(
    "field",
    [
        "T_reference_from_tool_tcp",
        "gripper_width_m",
        "grasp_approach_plan_id",
        "grasp_approach_plan_hash",
        "trajectory_hash",
        "trajectory_sample_count",
        "trajectory_duration_sec",
        "plan_created_at",
        "plan_expires_at",
    ],
)
@pytest.mark.parametrize("success", [True, False])
def test_all_nullable_plan_fields_follow_the_success_boundary(entry, field, success):
    result = _result(success=success)
    result[field] = None if success else _result()[field]
    with pytest.raises(ValueError, match="nullable plan fields"):
        validate_fast_grasp_plan_result_v1(result, entry.execution_result_schema)


@pytest.mark.parametrize(
    "changes",
    [
        {"planning_status": "unknown"},
        {"eligible_for_execution": False},
        {"eligible_for_execution": 1},
        {"observation_content_hash": "not-a-hash"},
        {"observation_content_hash": "A" * 64},
        {"reason_code": ""},
        {"component_id": "left_arm"},
        {"side": "left"},
        {"role": "unregistered"},
        {"source_station_revision": True},
        {"target_pixel_uv": [1.0, 2]},
        {"T_reference_from_tool_tcp": [0] * 15},
        {"T_reference_from_tool_tcp": [True] * 16},
        {"gripper_width_m": -0.1},
        {"grasp_approach_plan_id": ""},
        {"grasp_approach_plan_id": "x" * 257},
        {"trajectory_hash": ""},
        {"trajectory_sample_count": 1},
        {"trajectory_sample_count": 2.0},
        {"trajectory_duration_sec": 0},
        {"plan_expires_at": 10},
        {"plan_created_at": float("nan")},
        {"reference_frame_id": "another-station"},
        {"frame_graph_hash": "wrong"},
        {"tool_config_hash": "wrong"},
        {"calibration_hash": "wrong"},
        {"reference_frame_spec_hash": "wrong"},
        {"planning_policy": "best_score"},
        {"motion_or_hold_performed": True},
        {"execution_requires_explicit_motion_skill": False},
        {"motion_or_hold_performed": 0},
        {"point_grasp_operation_id": "fabricated"},
        {"diagnostics": {"candidate_pose": [0] * 16}},
        {"diagnostics": {"pool_count": 65537}},
        {"diagnostics": {"path_query_count": True}},
        {"diagnostics": {"elapsed_sec": 3601}},
        {"diagnostics": {"attempts_json": "x" * 8193}},
    ],
)
def test_results_reject_invalid_plans_lineage_constants_and_unbounded_diagnostics(
    entry, changes
):
    with pytest.raises((TypeError, ValueError)):
        validate_fast_grasp_plan_result_v1(
            {**_result(), **changes}, entry.execution_result_schema
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"component_id": "left_arm"},
        {"observation_operation_id": "another-observation"},
        {"role": "wrist_right"},
        {"target_pixel_uv": [123, 235]},
    ],
)
def test_contextual_validation_rejects_substituted_component_or_frozen_pixel(changes):
    with pytest.raises(ValueError, match="differs from the request"):
        validate_fast_grasp_plan_result_for_parameters_v1(
            _parameters(), {**_result(), **changes}
        )


@pytest.mark.parametrize(
    "plan_id",
    [
        "real-grasp-approach-plan:" + "1" * 64,
        "sim-grasp-approach-plan:" + "1" * 64,
        "mujoco-plan:immutable-1",
    ],
)
def test_public_plan_identity_is_target_neutral(entry, plan_id):
    result = {**_result(), "grasp_approach_plan_id": plan_id}
    validate_fast_grasp_plan_result_v1(result, entry.execution_result_schema)
    plan_schema = entry.execution_result_schema["properties"]["grasp_approach_plan_id"][
        "anyOf"
    ][0]
    assert plan_schema == {"type": "string", "minLength": 1, "maxLength": 256}


@pytest.mark.parametrize(
    "options",
    [
        {"side": "either"},
        {"roles": ()},
        {"roles": ("main", "main")},
        {"component_id": ""},
        {"min_speed_scale": 0},
        {"min_speed_scale": 0.6},
        {"max_speed_scale": float("inf")},
    ],
)
def test_schema_builder_rejects_ambiguous_deployment_configuration(entry, options):
    values = {
        "skill_definition_hash": entry.skill_definition_hash,
        "parameter_profile_id": "fast-grasp:test",
        "parameter_profile_hash": "a" * 64,
        "component_id": "right_arm",
        "side": "right",
        "roles": ("main",),
        "min_speed_scale": 0.1,
        "max_speed_scale": 0.5,
        "reference_frame_id": "station",
        "reference_frame_spec_hash": "b" * 64,
        "frame_graph_hash": "c" * 64,
        "tool_config_hash": "d" * 64,
        "calibration_hash": "e" * 64,
    }
    with pytest.raises((TypeError, ValueError)):
        fast_grasp_plan_schema_entry_v1(**{**values, **options})


def test_complete_sixty_four_query_history_fits_existing_inline_contract(entry):
    import json
    from agent_gateway.skills.perception.grasp_fast_plan_schemas import grasp_path_attempts_json

    attempts = tuple({
        "index": 1000000 - i,
        "reason_code": "safe_tcp_trajectory_budget_exhausted",
        "elapsed_sec": 0.03123456789012345,
        "failed_stage": 2,
        "endpoint_attempt_count": 8,
        "lower_reason": "collision",
    } for i in range(64))
    encoded = grasp_path_attempts_json(attempts)
    decoded = json.loads(encoded)
    assert len(encoded) <= 8192 and len(decoded) == 64
    assert [row["index"] for row in decoded] == [row["index"] for row in attempts]
    assert [row["reason_code"] for row in decoded] == [row["reason_code"] for row in attempts]
    assert all(row["elapsed_sec"] == 0.031235 for row in decoded)
    assert attempts[0]["elapsed_sec"] == 0.03123456789012345
    result = _result()
    result["diagnostics"].update(path_query_count=64, attempts_json=encoded)
    validate_fast_grasp_plan_result_v1(result, entry.execution_result_schema)


def test_attempt_summary_never_silently_truncates_reasons():
    from agent_gateway.skills.perception.grasp_fast_plan_schemas import grasp_path_attempts_json

    with pytest.raises(ValueError, match="exceeds"):
        grasp_path_attempts_json([{"index": 0, "reason_code": "x" * 8192, "elapsed_sec": 0.1}])


@pytest.mark.parametrize("elapsed", (True, -1.0, float("nan"), float("inf")))
def test_attempt_summary_rejects_invalid_diagnostic_time(elapsed):
    from agent_gateway.skills.perception.grasp_fast_plan_schemas import grasp_path_attempts_json

    with pytest.raises(ValueError, match="finite"):
        grasp_path_attempts_json([{"index": 0, "reason_code": "blocked", "elapsed_sec": elapsed}])

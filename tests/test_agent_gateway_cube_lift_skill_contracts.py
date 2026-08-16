from __future__ import annotations

from dataclasses import replace

import pytest

from agent_gateway.protocol.skill_catalog import (
    SkillRouteManifestEntry,
)
from agent_gateway.skills.cube_lift.definition import (
    CUBE_LIFT_PARAMETER_PROFILE_ID_V1,
    cube_lift_schema_entry_v1,
    cube_lift_skill_definition_v1,
    validate_cube_lift_execution_result_v1,
    validate_cube_lift_parameters_v1,
)
from agent_gateway.profiles.mujoco.skills.cube_lift.sim_route import (
    CUBE_LIFT_ENVIRONMENT_EVALUATOR_ID_V1,
    CubeLiftSimDeploymentBinding,
    CubeLiftTerminalEvaluatorConfig,
    cube_lift_evaluation_plan_v1,
    cube_lift_evaluator_input_schema_hash_v1,
    cube_lift_evaluator_input_schema_v1,
    cube_lift_payload_validator_set_v1,
    cube_lift_sim_route_v1,
    validate_cube_lift_sim_route_v1,
    validate_cube_lift_task_evaluation_v1,
)

TASK_CONFIG_HASH = (
    "e3450cc3cc3d58c191095d82eb87f0a2569c0df6561ba778b7451220a1add170"
)
CHECKPOINT_HASH = (
    "86dc170d1556219ed9c8b5a192d4c636559721fb06e0c2b5544dd35a0eeec60e"
)


def _contracts():
    definition = cube_lift_skill_definition_v1()
    schema = cube_lift_schema_entry_v1(
        skill_definition_hash=definition.semantic_content_hash
    )
    route = cube_lift_sim_route_v1(definition=definition, schema_entry=schema)
    return definition, schema, route


def _evaluator_config() -> CubeLiftTerminalEvaluatorConfig:
    return CubeLiftTerminalEvaluatorConfig(
        task_config_hash=TASK_CONFIG_HASH,
        object_body_name="debug_cube",
        left_finger_body_name="right_gripper_left_finger",
        right_finger_body_name="right_gripper_right_finger",
        table_body_name="table",
        lift_delta_m=0.04,
        stable_hold_physics_ticks=100,
        physics_hz=1000.0,
    )


def _binding(
    *,
    route_hash: str,
    evaluator_hash: str,
) -> CubeLiftSimDeploymentBinding:
    return CubeLiftSimDeploymentBinding(
        deployment_binding_id="sim.mujoco.cube_lift.policy_only.binding",
        deployment_binding_version="1",
        route_hash=route_hash,
        runtime_session_id="sim-hil:test-session:1",
        control_domain_id="mujoco-sim-hil",
        target_binding_id="mujoco-rm65-sim-hil",
        station_manifest_hash="station-manifest:test",
        execution_manifest_hash="execution-manifest:test",
        execution_domain_binding_hash="execution-domain-binding:test",
        target_runtime_action_contract_hash="target-action-contract:test",
        component_target_capability_hash="component-capability:test",
        action_schema_hash="component-action-schema:test",
        observation_plan_hash="execution-observation-plan:test",
        frame_graph_hash="frame-graph:test",
        tool_config_hash="tool-config:test",
        calibration_hash="calibration:sim-ground-truth",
        task_config_hash=TASK_CONFIG_HASH,
        policy_checkpoint_hash=CHECKPOINT_HASH,
        evaluator_configuration_hash=evaluator_hash,
        trace_requirement_profile_id="trace:sim-hil-agent-skill:v1",
        executor_id="control-station.cube-lift-operation-worker",
        explicit_exclusions=(
            "hardware_backend",
            "human_input_attachment",
            "raw_action_submission",
            "real_target",
            "replay_authority",
            "second_dispatch_owner",
        ),
    )


def _execution_result() -> dict[str, object]:
    string_values = {
        "runtime_session_id": "sim-hil:test-session:1",
        "control_domain_id": "mujoco-sim-hil",
        "execution_evidence_chain_ref": "execution-chain:1",
        "source_proposal_chain_hash": "source-chain:1",
        "authority_decision_chain_hash": "authority-chain:1",
        "safety_decision_chain_hash": "safety-chain:1",
        "target_request_chain_hash": "target-request-chain:1",
        "target_submission_receipt_chain_hash": "submission-chain:1",
        "target_completion_receipt_chain_hash": "completion-chain:1",
        "resolved_component_target_chain_hash": "resolved-chain:1",
        "component_admission_receipt_chain_hash": "admission-chain:1",
        "dispatch_receipt_chain_hash": "dispatch-chain:1",
        "measurement_lineage_chain_hash": "measurement-chain:1",
        "first_cycle_receipt_ref": "cycle:1",
        "last_cycle_receipt_ref": "cycle:12",
        "terminal_hold_receipt_ref": "hold:1",
        "terminal_state_hash": "terminal-state:1",
        "trace_terminal_receipt_ref": "trace-terminal:1",
        "trace_commit_receipt_ref": "trace-commit:1",
    }
    return {
        "schema_version": (
            "dexteleop.agent_skill.cube_lift_execution_result.v1"
        ),
        **string_values,
        "cycle_count": 12,
        "dispatched_cycle_count": 10,
        "measurement_count": 10,
    }


def _task_evaluation(
    config: CubeLiftTerminalEvaluatorConfig,
) -> dict[str, object]:
    return {
        "schema_version": (
            "dexteleop.agent_skill.cube_lift_task_evaluation.v1"
        ),
        "verdict": "succeeded",
        "reason_codes": ["cube_lift_terminal_stable_hold_satisfied"],
        "evaluated_cut_id": "cut:terminal:1",
        "environment_evidence_hash": "environment-evidence:1",
        "evaluator_configuration_hash": config.semantic_content_hash,
        "terminal_state_hash": "terminal-state:1",
        "baseline_resting_center_z_m": 0.026,
        "required_lift_delta_m": 0.04,
        "minimum_verified_cube_center_z_m": 0.067,
        "final_cube_center_z_m": 0.068,
        "stable_hold_required_physics_ticks": 100,
        "stable_hold_observed_physics_ticks": 100,
        "finite_state_all_ticks": True,
        "bilateral_contact_all_ticks": True,
        "table_contact_any_tick": False,
        "terminal_hold_confirmed": True,
        "execution_lineage_complete": True,
    }


def test_cube_lift_route_is_server_bound_t1_and_holds_on_every_stop() -> None:
    definition, schema, route = _contracts()
    validate_cube_lift_sim_route_v1(route=route, schema_entry=schema)
    assert definition.controlled_component_ids == (
        "right_arm",
        "right_gripper",
    )
    assert schema.to_wire()["parameter_schema"]["properties"] == {}
    assert route.allowed_parameter_profile_ids == (
        CUBE_LIFT_PARAMETER_PROFILE_ID_V1,
    )
    assert route.cancellation_policy == (
        "hold_on_cancel_disconnect_or_timeout"
    )
    assert route.may_move_robot and route.may_open_publication
    assert route.requires_task_evaluation
    assert "evaluation_plan_hash" not in route.to_wire()

    with pytest.raises(ValueError, match="cancellation_policy"):
        validate_cube_lift_sim_route_v1(
            route=replace(route, cancellation_policy="cancel_only")
        )


def test_cube_lift_executable_validators_bind_exact_route_schema_hashes() -> None:
    _, schema, route = _contracts()
    config = _evaluator_config()
    validators = cube_lift_payload_validator_set_v1(
        schema_entry=schema,
        evaluator_config=config,
    )
    validators.validate_route(route)
    validate_cube_lift_parameters_v1({})
    validators.validate_execution_result(_execution_result())
    validators.validate_task_evaluation(_task_evaluation(config))

    with pytest.raises(ValueError, match="unknown"):
        validate_cube_lift_parameters_v1({"tcp_pose": [0.0] * 7})
    with pytest.raises(ValueError, match="schema_hash"):
        validators.validate_route(
            replace(route, execution_result_schema_hash="other-result-schema")
        )


def test_evaluation_plan_is_bound_after_route_and_deployment_hashes_exist() -> None:
    definition, schema, route = _contracts()
    config = _evaluator_config()
    binding = _binding(
        route_hash=route.semantic_content_hash,
        evaluator_hash=config.semantic_content_hash,
    )
    plan = cube_lift_evaluation_plan_v1(
        definition=definition,
        route=route,
        deployment_binding=binding,
        evaluator_config=config,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
    )
    assert plan.route_hash == route.semantic_content_hash
    assert plan.deployment_binding_hash == binding.semantic_content_hash
    assert plan.evaluator_specs[0].evaluator_id == (
        CUBE_LIFT_ENVIRONMENT_EVALUATOR_ID_V1
    )
    entry = SkillRouteManifestEntry(
        route=route,
        deployment_binding=binding.to_ref(),
        evaluation_plan_hash=plan.semantic_content_hash,
        availability_hash="availability:cube-lift-ready",
    )
    assert entry.route.semantic_content_hash == route.semantic_content_hash

    with pytest.raises(ValueError, match="must bind an exact"):
        SkillRouteManifestEntry(
            route=route,
            deployment_binding=binding.to_ref(),
            evaluation_plan_hash=None,
            availability_hash="availability:cube-lift-ready",
        )


def test_terminal_evaluator_input_is_exactly_one_100_tick_hold_window() -> None:
    config = _evaluator_config()
    schema = cube_lift_evaluator_input_schema_v1(config)
    samples = schema["properties"]["samples"]
    assert samples["minItems"] == samples["maxItems"] == 100
    assert samples["items"]["additionalProperties"] is False
    assert len(cube_lift_evaluator_input_schema_hash_v1(config)) == 64


def test_execution_result_requires_dispatch_and_measurement_lineage() -> None:
    result = _execution_result()
    validate_cube_lift_execution_result_v1(result)
    with pytest.raises(ValueError, match="cannot exceed"):
        validate_cube_lift_execution_result_v1(
            {**result, "dispatched_cycle_count": 13}
        )
    with pytest.raises(ValueError, match="measurement coverage"):
        validate_cube_lift_execution_result_v1(
            {**result, "measurement_count": 9}
        )
    with pytest.raises(ValueError, match="unknown"):
        validate_cube_lift_execution_result_v1({**result, "raw_qpos": []})


def test_success_requires_stable_measured_lift_under_confirmed_hold() -> None:
    config = _evaluator_config()
    evaluation = _task_evaluation(config)
    validate_cube_lift_task_evaluation_v1(
        evaluation=evaluation,
        config=config,
    )
    for field_name, value in (
        ("bilateral_contact_all_ticks", False),
        ("table_contact_any_tick", True),
        ("terminal_hold_confirmed", False),
        ("execution_lineage_complete", False),
    ):
        with pytest.raises(ValueError, match="stable terminal lift"):
            validate_cube_lift_task_evaluation_v1(
                evaluation={**evaluation, field_name: value},
                config=config,
            )
    with pytest.raises(ValueError, match="stable terminal lift"):
        validate_cube_lift_task_evaluation_v1(
            evaluation={
                **evaluation,
                "minimum_verified_cube_center_z_m": 0.065,
            },
            config=config,
        )


def test_non_success_verdict_preserves_evidence_without_claiming_success() -> None:
    config = _evaluator_config()
    evaluation = {
        **_task_evaluation(config),
        "verdict": "inconclusive",
        "reason_codes": ["terminal_hold_window_incomplete"],
        "stable_hold_observed_physics_ticks": 42,
        "bilateral_contact_all_ticks": False,
    }
    validate_cube_lift_task_evaluation_v1(
        evaluation=evaluation,
        config=config,
    )

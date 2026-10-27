"""Offline contract and passive executor tests; no simulator is constructed."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import threading
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_gateway.profiles.mujoco.tcp_continuation_query import (
    build_tcp_continuation_query_executor,
    build_tcp_continuation_query_registration,
)
from agent_gateway.profiles.tcp_continuation_query import (
    build_tcp_continuation_query_registration as build_neutral_registration,
)
from agent_gateway.protocol._contract_validation import freeze_json
from agent_gateway.protocol.contracts import AgentObservationSnapshot, AgentSkillRequest
from agent_gateway.protocol.skill_catalog import (
    SkillDeploymentBindingRef,
    SkillExecutionRoute,
    SkillParameterProfileRef,
    SkillRouteManifest,
)
from agent_gateway.protocol.skill_registration import build_skill_registration_catalogs
from agent_gateway.runtime.access_policy import AgentSkillAccessPolicy
from agent_gateway.runtime.executor_registry import SkillExecutorRegistry
from agent_gateway.runtime.passive_read_executor import PassiveReadSkillExecutor
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorRegistry
from agent_gateway.runtime.service import AgentGatewayService
from agent_gateway.skills.motion.path_constraint import _constraint_schema
from agent_gateway.skills.perception.tcp_continuation_query import (
    PARAMETER_KEYS,
    RANKING,
    RESULT_KEYS,
    SKILL_ID,
    continuation_query_definition,
    continuation_query_schema,
    validate_continuation_parameters,
    validate_continuation_result,
    validate_continuation_result_for_parameters,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _parameters():
    return {
        "component_id": "right_arm", "reference_frame_id": "public_station",
        "first_pose_candidates_xyz_quat_xyzw": [
            [-.17339, -.028584, .211701, -.72606921196, .68762165308, 0., 0.],
            [-.17339, -.068584, .171701, -.72606921196, .68762165308, 0., 0.],
            [-.17339, -.028584, .171701, -.72606921196, .68762165308, 0., 0.],
        ],
        "continuation_delta_xyz_m": [0., 0., .03],
        "path_constraint": {"kind": "preserve_start_tool_axis", "tool_axis": "z",
                            "max_deviation_rad": math.radians(10)},
        "time_budget_sec": 30.,
    }


def _stage(index, reason="path_found"):
    result = {"stage_index": index, "reason_code": reason, "wall_sec": .1}
    if reason == "path_found":
        result.update(trajectory_hash=("a" if index == 0 else "b") * 64,
                      reference_count=5, duration_sec=.3)
    return result


def _success(index, cost=(.4, .6, .7)):
    return {"candidate_index": index, "reason_code": "path_found", "failed_stage": None,
            "wall_sec": .2, "ranking_cost": list(cost), "stage_diagnostics": [_stage(0), _stage(1)]}


def _select(result, parameters, index):
    result["selected_index"] = index
    first = list(parameters["first_pose_candidates_xyz_quat_xyzw"][index])
    predicted = list(first)
    # The planned endpoint FK need not equal the caller's ideal first target.
    predicted[0] += .001
    predicted[2] -= .0002
    result["first_pose_target_xyz_quat_xyzw"] = first
    result["predicted_first_tcp_pose_xyz_quat_xyzw"] = predicted
    result["continuation_pose_target_xyz_quat_xyzw"] = [
        predicted[i] + parameters["continuation_delta_xyz_m"][i] for i in range(3)
    ] + predicted[3:]


def _result(parameters=None, *, no_path=False):
    parameters = _parameters() if parameters is None else parameters
    result = {
        "component_id": parameters["component_id"], "reference_frame_id": parameters["reference_frame_id"],
        "source_tcp_pose_xyz_quat_xyzw": [.1, .2, .3, 0., 0., 0., 1.],
        "scene_hash": "c" * 64, "source_qpos_hash": "d" * 64, "source_sim_time_sec": 1.,
        "query_status": "no_path_found" if no_path else "path_found", "selected_index": None,
        "first_pose_target_xyz_quat_xyzw": None, "predicted_first_tcp_pose_xyz_quat_xyzw": None,
        "continuation_pose_target_xyz_quat_xyzw": None,
        "candidate_count": len(parameters["first_pose_candidates_xyz_quat_xyzw"]),
        "attempted_count": 0, "all_candidates_attempted": False, "candidate_diagnostics": [],
        "planning_wall_sec": .5, "ranking_criteria": RANKING,
        "continuation_source": "planned_first_endpoint",
        "constraint_scope": "frozen_scene_two_stage_sampled_paths",
        "motion_executed": False, "execution_authorized": False,
    }
    if not no_path:
        _select(result, parameters, 1)
        result["attempted_count"] = 2
        result["all_candidates_attempted"] = result["candidate_count"] == 2
        result["candidate_diagnostics"] = [
            {"candidate_index": 0, "reason_code": "safe_tcp_target_blocked", "failed_stage": "first",
             "wall_sec": .1, "stage_diagnostics": [_stage(0, "safe_tcp_target_blocked")]},
            _success(1),
        ]
    return result


def _base():
    route = SkillExecutionRoute(
        skill_id="state.read_tcp_pose", skill_version="1", skill_definition_hash="state-definition",
        route_id="mujoco.test.tcp_state", route_version="1", parameter_schema_hash="state-parameters",
        execution_result_schema_hash="state-result", task_evaluation_schema_hash="state-evaluation",
        allowed_parameter_profiles=(SkillParameterProfileRef(parameter_profile_id="state-profile",
            parameter_profile_hash="state-profile-hash", parameter_schema_hash="state-parameters"),),
        control_effect_class="read_only", privilege_tier="t0_read_only", supported_modes=("sim_active",),
        target_kind="sim", may_move_robot=False, may_open_publication=False,
        requires_operator_confirmation=False, requires_deadman=False,
        max_operation_duration_ms=1000, cancellation_policy="cancel_only",
    )
    binding = SkillDeploymentBindingRef(deployment_binding_id="state-binding",
        deployment_binding_hash="state-binding-hash", target_binding_id="test-target", target_kind="sim",
        execution_manifest_hash="test-execution")
    return route, binding


def _registration(**changes):
    route, binding = _base()
    return build_tcp_continuation_query_registration(**{
        "base_route": route, "base_binding": binding, "reference_frame_id": "public_station",
        "planning_profile_hash": "planning-profile", **changes,
    })


def test_definition_schema_and_frozen_catalog_fit_the_unchanged_wire_depth_cap():
    definition = continuation_query_definition()
    schema = continuation_query_schema(definition_hash=definition.semantic_content_hash,
        profile_hash="profile-hash", reference_frame_id="public_station", component_id="right_arm")
    assert definition.skill_id == SKILL_ID == "perception.query_tcp_continuation_paths"
    assert definition.skill_version == schema.skill_version == "1"
    assert definition.intended_target_kind == "target_agnostic"
    assert definition.controlled_component_ids == ()
    assert definition.skill_kind == "inspection"
    for document, keys in ((schema.parameter_schema, PARAMETER_KEYS),
                           (schema.execution_result_schema, RESULT_KEYS)):
        assert document["additionalProperties"] is False
        assert set(document["required"]) == set(document["properties"]) == keys
        assert document["properties"]["component_id"]["const"] == "right_arm"
        assert document["properties"]["reference_frame_id"]["const"] == "public_station"
    assert schema.to_wire()["parameter_schema"]["properties"]["path_constraint"]["anyOf"][1] == _constraint_schema()
    assert schema.parameter_profiles[0].parameter_schema_hash == schema.parameter_schema_hash
    registration = _registration()
    catalogs = build_skill_registration_catalogs((registration.skill,))
    # RPent is an independent consumer, not a dependency of these owner tests.
    # Include the actual schema catalog and an RPC result envelope in the cap.
    wrapper = json.loads(json.dumps({"result": catalogs.schema_catalog.to_wire()}))

    def depth(value):
        children = value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
        return max((1 + depth(child) for child in children), default=0)

    assert depth(wrapper) <= 16
    diagnostic = schema.execution_result_schema["properties"]["candidate_diagnostics"]["items"]
    stage = diagnostic["properties"]["stage_diagnostics"]["items"]
    assert diagnostic["additionalProperties"] is stage["additionalProperties"] is False
    assert "anyOf" not in diagnostic and "anyOf" not in stage
    with pytest.raises(TypeError):
        schema.parameter_schema["properties"]["component_id"] = {}


def test_schema_component_is_bound_by_the_caller_without_changing_neutral_semantics():
    definition = continuation_query_definition()
    schema = continuation_query_schema(definition_hash=definition.semantic_content_hash,
        profile_hash="another-profile", reference_frame_id="another_public_frame", component_id="another_arm")
    for document in (schema.parameter_schema, schema.execution_result_schema):
        assert document["properties"]["component_id"]["const"] == "another_arm"
        assert document["properties"]["reference_frame_id"]["const"] == "another_public_frame"
    parameters = {**_parameters(), "component_id": "another_arm", "reference_frame_id": "another_public_frame"}
    validate_continuation_result_for_parameters(parameters, _result(parameters))
    with pytest.raises(ValueError):
        _registration().payload_validator.validate_parameters(parameters)


@pytest.mark.parametrize("constraint", [None, {"kind": "preserve_start_tool_axis", "tool_axis": "x", "max_deviation_rad": math.pi}])
@pytest.mark.parametrize("count,budget", [(1, .1), (8, 30)])
def test_valid_parameters_are_accepted_without_mutating_or_deduplicating(constraint, count, budget):
    parameters = _parameters()
    parameters.update(path_constraint=constraint, time_budget_sec=budget,
        first_pose_candidates_xyz_quat_xyzw=[parameters["first_pose_candidates_xyz_quat_xyzw"][0]] * count)
    before = deepcopy(parameters)
    validate_continuation_parameters(parameters)
    assert parameters == before


@pytest.mark.parametrize("key", sorted(PARAMETER_KEYS))
def test_parameters_require_every_field_and_reject_extra_fields(key):
    parameters = _parameters()
    del parameters[key]
    with pytest.raises(ValueError):
        validate_continuation_parameters(parameters)
    with pytest.raises(ValueError):
        validate_continuation_parameters({**_parameters(), "caller_qpos": []})


@pytest.mark.parametrize("key,value", [
    ("component_id", ""), ("component_id", True), ("reference_frame_id", " frame"),
    ("first_pose_candidates_xyz_quat_xyzw", []), ("first_pose_candidates_xyz_quat_xyzw", [[0] * 7] * 9),
    ("first_pose_candidates_xyz_quat_xyzw", "poses"), ("first_pose_candidates_xyz_quat_xyzw", [[0] * 6]),
    ("first_pose_candidates_xyz_quat_xyzw", [[0] * 7]),
    ("first_pose_candidates_xyz_quat_xyzw", [[0, 0, 0, 0, 0, 0, 2]]),
    ("first_pose_candidates_xyz_quat_xyzw", [[True, 0, 0, 0, 0, 0, 1]]),
    ("first_pose_candidates_xyz_quat_xyzw", [[math.nan, 0, 0, 0, 0, 0, 1]]),
    ("first_pose_candidates_xyz_quat_xyzw", [[0, 0, 0, math.inf, 0, 0, 1]]),
    ("first_pose_candidates_xyz_quat_xyzw", [[10**400, 0, 0, 0, 0, 0, 1]]),
    ("continuation_delta_xyz_m", [0, 0]), ("continuation_delta_xyz_m", [0, 0, 0, 0]),
    ("continuation_delta_xyz_m", [0, False, 0]), ("continuation_delta_xyz_m", [0, math.inf, 0]),
    ("continuation_delta_xyz_m", [0, math.nan, 0]), ("continuation_delta_xyz_m", [0, 10**400, 0]),
    ("time_budget_sec", .099), ("time_budget_sec", 30.01), ("time_budget_sec", True),
    ("time_budget_sec", "1"), ("time_budget_sec", math.inf), ("time_budget_sec", math.nan),
    ("path_constraint", {}), ("path_constraint", {"kind": "other", "tool_axis": "z", "max_deviation_rad": .1}),
    ("path_constraint", {"kind": "preserve_start_tool_axis", "tool_axis": "-z", "max_deviation_rad": .1}),
    ("path_constraint", {"kind": "preserve_start_tool_axis", "tool_axis": "z", "max_deviation_rad": 0}),
    ("path_constraint", {"kind": "preserve_start_tool_axis", "tool_axis": "z", "max_deviation_rad": True}),
])
def test_parameters_reject_malformed_geometry_budget_and_policy(key, value):
    with pytest.raises((TypeError, ValueError)):
        validate_continuation_parameters({**_parameters(), key: value})


def test_results_use_predicted_fk_for_delta_and_preserve_request_identity():
    parameters, result = _parameters(), _result()
    before = deepcopy(result)
    validate_continuation_result_for_parameters(parameters, result)
    assert result == before
    assert result["predicted_first_tcp_pose_xyz_quat_xyzw"] != result["first_pose_target_xyz_quat_xyzw"]
    for key in ("component_id", "reference_frame_id"):
        forged = {**result, key: "another_identity"}
        with pytest.raises(ValueError):
            validate_continuation_result_for_parameters(parameters, forged)
    for key in ("first_pose_target_xyz_quat_xyzw", "continuation_pose_target_xyz_quat_xyzw"):
        forged = deepcopy(result)
        forged[key][0] += .001
        with pytest.raises(ValueError):
            validate_continuation_result_for_parameters(parameters, forged)
    ideal = parameters["first_pose_candidates_xyz_quat_xyzw"][result["selected_index"]]
    forged = deepcopy(result)
    forged["continuation_pose_target_xyz_quat_xyzw"] = [
        ideal[i] + parameters["continuation_delta_xyz_m"][i] for i in range(3)
    ] + ideal[3:]
    with pytest.raises(ValueError):
        validate_continuation_result_for_parameters(parameters, forged)


@pytest.mark.parametrize("first,second,winner", [
    ((.3, 8., 9.), (.4, .1, .1), 0), ((.4, .5, 8.), (.4, .6, .1), 0),
    ((.4, .6, .8), (.4, .6, .7), 1), ((.4, .6, .7), (.4, .6, .7), 0),
])
def test_selection_minimizes_joined_cost_then_original_index(first, second, winner):
    parameters, result = _parameters(), _result()
    result["candidate_diagnostics"] = [_success(1, second), _success(0, first)]
    _select(result, parameters, winner)
    validate_continuation_result_for_parameters(parameters, result)
    _select(result, parameters, 1 - winner)
    with pytest.raises(ValueError):
        validate_continuation_result_for_parameters(parameters, result)


@pytest.mark.parametrize("failed_stage", ["first", "continuation"])
def test_budget_attempts_are_not_proven_rejection_and_failure_details_are_allowed(failed_stage):
    parameters = _parameters()
    parameters["first_pose_candidates_xyz_quat_xyzw"] = parameters["first_pose_candidates_xyz_quat_xyzw"][:1]
    result = _result(parameters, no_path=True)
    stage_index = 0 if failed_stage == "first" else 1
    failure = _stage(stage_index, "query_budget_exhausted")
    failure["details"] = {"phase": "search", "elapsed_sec": .1}
    result.update(attempted_count=1, all_candidates_attempted=True, candidate_diagnostics=[{
        "candidate_index": 0, "reason_code": "query_budget_exhausted", "failed_stage": failed_stage,
        "wall_sec": .3, "stage_diagnostics": ([_stage(0)] if stage_index else []) + [failure],
    }])
    validate_continuation_result_for_parameters(parameters, result)
    failure["trajectory_hash"] = "a" * 64
    with pytest.raises(ValueError):
        validate_continuation_result(result)


def test_empty_partial_and_no_path_results_have_exact_attempt_accounting():
    parameters, result = _parameters(), _result(no_path=True)
    validate_continuation_result_for_parameters(parameters, result)
    result.update(attempted_count=1, candidate_diagnostics=[{
        "candidate_index": 2, "reason_code": "query_budget_exhausted", "failed_stage": None,
        "wall_sec": 0., "stage_diagnostics": [],
    }])
    validate_continuation_result_for_parameters(parameters, result)
    result["candidate_diagnostics"][0]["ranking_cost"] = [0., 0., 0.]
    with pytest.raises(ValueError):
        validate_continuation_result(result)


@pytest.mark.parametrize("key,value", [
    ("motion_executed", True), ("execution_authorized", 0), ("continuation_source", "caller_first_pose"),
    ("constraint_scope", "continuous_path"), ("ranking_criteria", "duration"),
    ("scene_hash", "a" * 63), ("source_qpos_hash", "g" * 64),
    ("source_tcp_pose_xyz_quat_xyzw", [0] * 7), ("source_sim_time_sec", -1),
    ("planning_wall_sec", math.nan), ("planning_wall_sec", math.inf), ("planning_wall_sec", True),
    ("candidate_count", True), ("candidate_count", 0), ("candidate_count", 9),
    ("attempted_count", True), ("attempted_count", 4), ("all_candidates_attempted", 0),
    ("all_candidates_attempted", True), ("candidate_diagnostics", []), ("candidate_diagnostics", {}),
    ("selected_index", True), ("selected_index", 1.), ("selected_index", 0), ("selected_index", 3),
    ("first_pose_target_xyz_quat_xyzw", None), ("predicted_first_tcp_pose_xyz_quat_xyzw", [0] * 7),
    ("continuation_pose_target_xyz_quat_xyzw", [0, 0, 0, 0, 0, 0, 1]),
    ("query_status", "all_ik_branches_impossible"),
])
def test_result_rejects_forged_authority_evidence_geometry_and_counts(key, value):
    with pytest.raises((TypeError, ValueError)):
        validate_continuation_result({**_result(), key: value})


@pytest.mark.parametrize("key", sorted(RESULT_KEYS))
def test_result_requires_every_key(key):
    result = _result()
    del result[key]
    with pytest.raises(ValueError):
        validate_continuation_result(result)


@pytest.mark.parametrize("mutation", ["extra", "missing_cost", "negative_cost", "bool_cost", "nan_cost",
    "duplicate", "bool_index", "missing_stage", "stage_order", "stage_hash", "stage_count",
    "stage_duration", "failure_as_success", "failed_stage", "extra_stage_field", "invalid_details"])
def test_candidate_and_stage_evidence_cannot_be_forged(mutation):
    result = _result()
    candidate = result["candidate_diagnostics"][1]
    stage = candidate["stage_diagnostics"][1]
    if mutation == "extra":
        result["proof_of_unreachability"] = True
    elif mutation == "missing_cost":
        del candidate["ranking_cost"]
    elif mutation in {"negative_cost", "bool_cost", "nan_cost"}:
        candidate["ranking_cost"][0] = {"negative_cost": -1, "bool_cost": True, "nan_cost": math.nan}[mutation]
    elif mutation == "duplicate":
        result["candidate_diagnostics"][0]["candidate_index"] = 1
    elif mutation == "bool_index":
        candidate["candidate_index"] = True
    elif mutation == "missing_stage":
        candidate["stage_diagnostics"].pop()
    elif mutation == "stage_order":
        stage["stage_index"] = 0
    elif mutation == "stage_hash":
        stage["trajectory_hash"] = "f" * 65
    elif mutation == "stage_count":
        stage["reference_count"] = False
    elif mutation == "stage_duration":
        stage["duration_sec"] = -1
    elif mutation == "failure_as_success":
        stage["reason_code"] = "query_budget_exhausted"
    elif mutation == "failed_stage":
        candidate["failed_stage"] = "continuation"
    elif mutation == "extra_stage_field":
        stage["execution_authorized"] = True
    elif mutation == "invalid_details":
        stage["details"] = {"bad": math.inf}
    with pytest.raises((TypeError, ValueError)):
        validate_continuation_result(result)


def test_registration_closes_hashes_and_inherits_only_the_passive_binding():
    base_route, base_binding = _base()
    registration = _registration()
    skill, entry, validator = registration.skill, registration.entry, registration.payload_validator
    assert entry.route == skill.routes[0]
    validator.validate_route(entry.route)
    assert entry.route.control_effect_class == "read_only"
    assert entry.route.privilege_tier == "t0_read_only"
    assert entry.route.cancellation_policy == "cancel_only"
    assert entry.route.may_move_robot is entry.route.may_open_publication is False
    assert entry.route.supported_modes == base_route.supported_modes
    assert entry.route.resource_claims == base_route.resource_claims
    assert entry.route.max_operation_duration_ms == 40_000
    assert entry.deployment_binding.target_binding_id == base_binding.target_binding_id
    assert entry.deployment_binding.execution_manifest_hash == base_binding.execution_manifest_hash
    assert entry.deployment_binding.deployment_binding_hash != base_binding.deployment_binding_hash
    assert entry.route.route_id != base_route.route_id
    assert skill.io_contract.input_artifact_slots == skill.io_contract.output_artifact_slots == ()
    assert skill.io_contract.skill_definition_hash == skill.definition.semantic_content_hash
    assert skill.io_contract.parameter_schema_hash == skill.schema_entry.parameter_schema_hash
    assert skill.io_contract.execution_result_schema_hash == skill.schema_entry.execution_result_schema_hash
    assert entry.route.skill_io_contract_hash == skill.io_contract.semantic_content_hash
    assert _registration().entry == entry
    for changes in ({"reference_frame_id": "other_frame"}, {"planning_profile_hash": "other_profile"},
                    {"base_binding": replace(base_binding, deployment_binding_hash="other_binding")}):
        assert _registration(**changes).entry.deployment_binding.deployment_binding_hash != entry.deployment_binding.deployment_binding_hash
    assert _registration(base_binding=SimpleNamespace(to_ref=lambda: base_binding)).entry == entry


@pytest.mark.parametrize("key,value", [("component_id", "left_arm"), ("reference_frame_id", "arm_base")])
def test_profile_rejects_wrong_identity_in_parameters_and_standalone_results(key, value):
    validator = _registration().payload_validator
    with pytest.raises(ValueError):
        validator.validate_parameters({**_parameters(), key: value})
    with pytest.raises(ValueError):
        validator.validate_execution_result({**_result(), key: value})
    validator.validate_execution_result_for_parameters(_parameters(), _result())
    with pytest.raises(ValueError):
        validator.validate_task_evaluation({"task_succeeded": True})


def test_registration_rejects_effectful_and_real_routes():
    base_route, binding = _base()
    motion = replace(base_route, control_effect_class="motion", privilege_tier="t1_sim_or_no_motion",
        may_move_robot=True, cancellation_policy="hold_on_disconnect_or_timeout")
    with pytest.raises(ValueError):
        _registration(base_route=motion)
    with pytest.raises(ValueError):
        _registration(base_route=replace(base_route, target_kind="real", supported_modes=("real_observe",)),
                      base_binding=replace(binding, target_kind="real"))


def test_shared_registration_preserves_the_existing_mujoco_contract_identity():
    route, binding = _base()
    neutral = build_neutral_registration(base_route=route, base_binding=binding,
        component_id="right_arm", reference_frame_id="public_station", planning_profile_hash="planning-profile")
    existing = _registration()
    assert neutral.skill == existing.skill
    assert neutral.entry == existing.entry
    assert neutral.payload_validator.parameter_schema_hash == existing.payload_validator.parameter_schema_hash
    # Independently frozen from e2888a90 with _base() and this test's inputs.
    assert {
        "parameter_schema": existing.skill.schema_entry.parameter_schema_hash,
        "result_schema": existing.skill.schema_entry.execution_result_schema_hash,
        "io": existing.skill.io_contract.semantic_content_hash,
        "route": existing.entry.route.semantic_content_hash,
        "binding": existing.entry.deployment_binding.deployment_binding_hash,
        "availability": existing.entry.availability_hash,
    } == {
        "parameter_schema": "61c934e542383327ab8a9dbfd96a8b0aa6062e5dea9a2b60095fb66197a56fdb",
        "result_schema": "7c210d6e54ca3220584f7f3cf95f12904dedb475ff3bccc60e8849dad9f97e9a",
        "io": "8f8cb49d536fe9c8f82e3351d98bf8e4b695c86e5fd8da602e65f2751c204e69",
        "route": "3cc4d62a7fa25b2c97e4476f77c6834ad031b510247b7422a9ffbcad6cb16d7d",
        "binding": "325eda3d5c497c98732160e6086e5c6f1e2042ecdf315199785aea59c79effa2",
        "availability": "1dcf2bf2658415fb15afcf9ec4ca33f5ee72fc057678777e9f549c08412d533a",
    }


@pytest.mark.parametrize("component,frame", [("panda_arm", "world"), ("left_arm", "table_world")])
def test_other_simulation_bindings_use_the_same_passive_contract_and_runtime(component, frame):
    route, binding = _base()
    registration = build_neutral_registration(base_route=route, base_binding=binding,
        component_id=component, reference_frame_id=frame, planning_profile_hash="native-planner-profile")
    parameters = {**_parameters(), "component_id": component, "reference_frame_id": frame}
    assert registration.skill.definition == _registration().skill.definition
    assert registration.entry.route.control_effect_class == "read_only"
    assert registration.entry.route.semantic_content_hash != _registration().entry.route.semantic_content_hash
    with pytest.raises(ValueError):
        registration.payload_validator.validate_parameters(_parameters())
    with pytest.raises(ValueError):
        registration.payload_validator.validate_execution_result(_result())
    calls = []

    def query(value, *, cancel_requested):
        assert not cancel_requested()
        calls.append(value)
        return _result(value)

    service, executor, domain, _ = _runtime(SimpleNamespace(query_continuations=query), registration=registration)
    admitted = _submit(service, parameters=parameters)
    assert admitted.accepted, admitted.reason_code
    assert service.launch_operation(admitted.operation.operation_id).accepted
    assert executor.wait(admitted.operation.operation_id, timeout_sec=2.)
    operation = service.operation_store.get(admitted.operation.operation_id)
    assert operation.status == "completed", executor.failure_diagnostics
    validate_continuation_result_for_parameters(parameters, operation.execution_result)
    assert calls == [freeze_json(parameters, field_name="parameters")] and not domain.calls
    assert all(not receipt.proves_motion_or_hold for receipt in executor.port_receipts)


@pytest.mark.parametrize("component", ["", " padded", None])
def test_shared_registration_requires_an_explicit_valid_component(component):
    route, binding = _base()
    with pytest.raises((TypeError, ValueError)):
        build_neutral_registration(base_route=route, base_binding=binding,
            component_id=component, reference_frame_id="world", planning_profile_hash="native-profile")


def test_shared_registration_imports_without_any_simulator_or_upper_framework():
    code = """
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
blocked = ('mujoco', 'sim', 'torch', 'hardware', 'teleop_runtime', 'control_station', 'rpent')
class NoBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('Backend imported by passive registration: ' + fullname)
sys.meta_path.insert(0, NoBackend())
from agent_gateway.profiles.tcp_continuation_query import build_tcp_continuation_query_registration
assert callable(build_tcp_continuation_query_registration)
assert not any(name.split('.')[0] in blocked for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-I", "-c", code, str(Path(__file__).resolve().parents[1])],
        capture_output=True, text=True, timeout=15., check=False)
    assert result.returncode == 0, result.stderr


class _Clock:
    value = 10.

    def __call__(self):
        return self.value


class _ForbiddenDomainControl:
    no_io = False
    port_id = "forbidden-domain-control"

    def __init__(self):
        self.calls = []

    def request_hold(self, **kwargs):
        self.calls.append(kwargs)
        raise AssertionError("A passive query must never request domain HOLD.")


def _runtime(port, *, registration=None):
    clock = _Clock()
    registration = _registration() if registration is None else registration
    catalogs = build_skill_registration_catalogs((registration.skill,))
    manifest = SkillRouteManifest(runtime_session_id="test-session", control_domain_id="test-domain",
        station_manifest_hash="test-station", execution_manifest_hash="test-execution",
        skill_definition_catalog_hash=catalogs.definition_catalog.semantic_content_hash,
        skill_schema_catalog_hash=catalogs.schema_catalog.semantic_content_hash,
        skill_io_catalog_hash=catalogs.io_catalog.semantic_content_hash,
        target_binding_id="test-target", target_kind="sim", current_mode="sim_active",
        supported_transport_features=("poll_status_v1", "idempotency_v1"), skill_routes=(registration.entry,))
    executor = build_tcp_continuation_query_executor(registration=registration, query_port=port,
        current_station_revision=lambda: 4, clock=clock, max_retained_operations=2)
    domain = _ForbiddenDomainControl()
    service = AgentGatewayService(manifest=manifest,
        access_policy=AgentSkillAccessPolicy(access_policy_id="test-policy", allowed_client_ids=("test-client",),
            allowed_modes=("sim_active",), max_request_timeout_ms=40_000, max_grant_lifetime_sec=30.),
        executor_registry=SkillExecutorRegistry.bind_all_routes_to_port(manifest=manifest, execution_port=executor),
        domain_control_port=domain,
        payload_validators=AgentSkillPayloadValidatorRegistry(registry_id="test-validators",
            validators=(registration.payload_validator,)), clock=clock)
    snapshot = AgentObservationSnapshot(snapshot_id="test-snapshot", runtime_session_id=manifest.runtime_session_id,
        skill_route_manifest_hash=manifest.semantic_content_hash, station_revision=4,
        captured_at=9.8, available_at=9.9, clock_domain_id="agent-gateway-monotonic",
        authority_epoch=0, execution_epoch=0, dispatch_owner_generation=0,
        lifecycle_state="hold", execution_gate="hold", active_operation_id=None, source_binding_id=None,
        producer_epoch=None, target_binding_id=manifest.target_binding_id,
        execution_manifest_hash=manifest.execution_manifest_hash, frame_graph_hash="test-frames",
        tool_config_hash="test-tool", calibration_hash="test-calibration", measurement_summary={},
        measurement_freshness_ms=0, measurement_valid=True)
    assert service.publish_observation(snapshot=snapshot).accepted
    return service, executor, domain, clock


def _submit(service, sequence=1, parameters=None):
    entry = service.manifest.skill_routes[0]
    route, profile = entry.route, entry.route.allowed_parameter_profiles[0]
    request = AgentSkillRequest(client_id="test-client", request_id=f"test-request-{sequence}",
        idempotency_key=f"test-key-{sequence}", client_sequence=sequence,
        runtime_session_id=service.manifest.runtime_session_id,
        skill_route_manifest_hash=service.manifest.semantic_content_hash, expected_station_revision=4,
        skill_id=SKILL_ID, skill_version="1", skill_definition_hash=route.skill_definition_hash,
        route_id=route.route_id, route_version="1", route_hash=route.semantic_content_hash,
        deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
        parameter_profile_id=profile.parameter_profile_id, parameter_profile_hash=profile.parameter_profile_hash,
        parameters=_parameters() if parameters is None else parameters,
        observation_snapshot_id="test-snapshot", requested_mode="sim_active", requested_timeout_ms=1000)
    return service.submit_request(request=request, current_station_revision=4)


def test_executor_uses_only_the_query_port_is_reusable_and_emits_read_evidence():
    calls = []

    def query(parameters, *, cancel_requested):
        assert cancel_requested() is False
        calls.append(parameters)
        return _result(parameters)

    service, executor, domain, _ = _runtime(SimpleNamespace(query_continuations=query))
    assert isinstance(executor, PassiveReadSkillExecutor)
    assert executor.control_effect_class == "read_only"
    for sequence in (1, 2):
        admitted = _submit(service, sequence)
        assert admitted.accepted, admitted.reason_code
        operation_id = admitted.operation.operation_id
        assert service.launch_operation(operation_id).accepted
        assert executor.wait(operation_id, timeout_sec=2.)
        operation = service.operation_store.get(operation_id)
        assert operation.status == "completed", executor.failure_diagnostics
        validate_continuation_result_for_parameters(_parameters(), operation.execution_result)
        receipt = executor.execution_receipts[-1]
        assert receipt.reason_code == "tcp_continuation_query_completed"
        assert receipt.evidence_refs == ("tcp-continuation-query:" + operation_id,)
        assert receipt.receipt_id in operation.execution_receipt_refs
    assert len(calls) == 2 and domain.calls == []
    assert all(not receipt.proves_motion_or_hold for receipt in executor.port_receipts)


@pytest.mark.parametrize("termination", ["cancel", "deadline", "deadline_error"])
def test_executor_propagates_cancellation_and_deadline_without_late_publication_or_hold(termination):
    entered, release = threading.Event(), threading.Event()
    cancellations = []

    def query(parameters, *, cancel_requested):
        cancellations.append(cancel_requested())
        entered.set()
        assert release.wait(timeout=2.)
        cancellations.append(cancel_requested())
        if termination == "deadline_error":
            raise RuntimeError("Cooperative query stopped at its deadline.")
        return _result(parameters)

    service, executor, domain, clock = _runtime(SimpleNamespace(query_continuations=query))
    admitted = _submit(service)
    assert admitted.accepted, admitted.reason_code
    operation_id = admitted.operation.operation_id
    assert service.launch_operation(operation_id).accepted
    try:
        assert entered.wait(timeout=2.)
        if termination == "cancel":
            outcome = service.cancel_operation(operation_id=operation_id, reason="test-cancel",
                current_station_revision=4)
            expected = "cancelled"
        else:
            clock.value = admitted.operation.deadline_at
            outcome = service.expire_due_operation(current_station_revision=4, now=clock())
            expected = "timed_out"
        assert outcome.accepted
    finally:
        release.set()
        assert executor.wait(operation_id, timeout_sec=2.)
    operation = service.operation_store.get(operation_id)
    assert operation.status == expected
    assert operation.execution_result is None
    assert cancellations == [False, True]
    assert executor.execution_receipts == ()
    if termination == "deadline_error":
        diagnostic, = executor.failure_diagnostics
        assert diagnostic["operation_id"] == operation_id
        assert diagnostic["exception_type"] == "RuntimeError"
        assert diagnostic["message"] == "Cooperative query stopped at its deadline."
    else:
        assert executor.failure_diagnostics == ()
    assert domain.calls == []
    assert all(not receipt.proves_motion_or_hold for receipt in executor.port_receipts)


def test_executor_rejects_forged_port_output_and_invalid_parameters_before_query():
    calls = []

    def query(parameters, *, cancel_requested):
        calls.append(parameters)
        result = _result(parameters)
        result["continuation_pose_target_xyz_quat_xyzw"][0] += .01
        return result

    service, executor, domain, _ = _runtime(SimpleNamespace(query_continuations=query))
    rejected = _submit(service, parameters={**_parameters(), "reference_frame_id": "arm_base"})
    assert not rejected.accepted and calls == []
    admitted = _submit(service, sequence=2)
    assert admitted.accepted, admitted.reason_code
    operation_id = admitted.operation.operation_id
    assert service.launch_operation(operation_id).accepted
    assert executor.wait(operation_id, timeout_sec=2.)
    operation = service.operation_store.get(operation_id)
    assert operation.status == "failed" and operation.execution_result is None
    assert executor.failure_diagnostics[0]["exception_type"] == "ValueError"
    assert len(calls) == 1 and domain.calls == []


@pytest.fixture
def continuation_owner_with_fake_planners(monkeypatch):
    """Exercise the real query owner without constructing a native scene or planner."""
    import numpy as np

    from agent_gateway.profiles.mujoco import object_goal_query as owner_module

    state = SimpleNamespace(outcomes=[], planners=[], calls=[],
        details={"motion_quality": {"joint_total_travel_deg": [181., 0., 0., 0., 0., 0.]}})

    class Planner:
        def __init__(self, *, source_model_qpos, **kwargs):
            self.source = source_model_qpos.copy()
            self.clear_calls = 0
            state.planners.append(self)

        def pose_for_configuration(self, model_qpos, pair):
            return [*map(float, pair["right"][:3]), 0., 0., 0., 1.]

        def plan(self, target, *, cancel_requested, **kwargs):
            assert not cancel_requested()
            state.calls.append((self.source.copy(), target))
            outcome = state.outcomes.pop(0)
            if outcome is not None:
                raise owner_module.MujocoTcpPlanRejected(outcome, deepcopy(state.details))
            endpoint = np.array([*target[:3], 0., 0., 0.])
            return SimpleNamespace(reference_poses=(target, target),
                trajectory_hash=f"{len(state.calls):064x}",
                trajectory=SimpleNamespace(
                    q_rad=np.array([self.source[6:12], endpoint]), timestamps=(0., .1)))

        def clear(self):
            self.clear_calls += 1

    monkeypatch.setattr(owner_module, "MujocoPlannedTcpMotion", Planner)
    monkeypatch.setattr(owner_module, "MujocoJointSceneOracle",
        lambda *args, **kwargs: SimpleNamespace(scene_hash="f" * 64))
    scene = SimpleNamespace(data=SimpleNamespace(
        qpos=np.r_[np.linspace(.1, .6, 6), np.zeros(6), .7], time=1.),
        arm_specs={side: SimpleNamespace(joint_names=[f"{side}{i}" for i in range(6)])
                   for side in ("left", "right")},
        joint_qpos_addr=lambda name: (0 if name.startswith("left") else 6) + int(name[-1]))
    profile = SimpleNamespace(reference_frame_id="public_station",
        execution_reference_frame_id="public_station", max_translation_delta_m=1.,
        max_rotation_delta_rad=math.pi, translation_tolerance_m=.003,
        rotation_tolerance_rad=.03, normalize_ik_objective=False,
        stable_measurement_count=3, max_control_cycles=100)
    owner = owner_module.MujocoObjectGoalQuery(scene=scene, rig_config=None, task_config=None,
        pose_projector=None, reference_hz=50., clearance_padding_m=0., profile=profile,
        capture_scene=lambda callback: callback(scene), transform_pose=lambda pose, **kwargs: pose,
        prepare_projection=lambda: (
            lambda poses: SimpleNamespace(modified_sides=()), "a" * 64))
    parameters = {**_parameters(), "first_pose_candidates_xyz_quat_xyzw": [
        [.04, 0., 0., 0., 0., 0., 1.], [.08, 0., 0., 0., 0., 0., 1.]]}
    return owner, scene, parameters, state


@pytest.mark.parametrize("rejected_stage", [0, 1], ids=["first", "continuation"])
@pytest.mark.parametrize("next_candidate_succeeds", [False, True], ids=["all_rejected", "next_found"])
def test_owner_motion_quality_rejection_is_diagnostic_and_cleans_up(
    continuation_owner_with_fake_planners, rejected_stage, next_candidate_succeeds,
):
    import numpy as np

    owner, scene, parameters, state = continuation_owner_with_fake_planners
    reason = "safe_tcp_motion_quality_exceeded"
    rejected = [None] * rejected_stage + [reason]
    state.outcomes = rejected + ([None, None] if next_candidate_succeeds else rejected)
    original_qpos, original_time = scene.data.qpos.copy(), scene.data.time
    try:
        result = owner.query_continuations(parameters, cancel_requested=lambda: False)
    finally:
        assert state.planners and all(planner.clear_calls == 1 for planner in state.planners)
        assert owner._lock.acquire(blocking=False)
        owner._lock.release()
        np.testing.assert_array_equal(scene.data.qpos, original_qpos)
        assert scene.data.time == original_time
    validate_continuation_result_for_parameters(parameters, result)
    assert not state.outcomes
    assert result["attempted_count"] == 2 and result["all_candidates_attempted"] is True
    assert result["motion_executed"] is result["execution_authorized"] is False
    assert result["query_status"] == ("path_found" if next_candidate_succeeds else "no_path_found")
    assert result["selected_index"] == (1 if next_candidate_succeeds else None)
    failed_stage = "first" if rejected_stage == 0 else "continuation"
    for row in result["candidate_diagnostics"][:1 if next_candidate_succeeds else 2]:
        assert row["reason_code"] == reason and row["failed_stage"] == failed_stage
        assert [item["reason_code"] for item in row["stage_diagnostics"]] == [
            *(["path_found"] * rejected_stage), reason]
        assert row["stage_diagnostics"][-1]["details"] == state.details
        assert "ranking_cost" not in row
    if next_candidate_succeeds:
        assert result["candidate_diagnostics"][1]["reason_code"] == "path_found"
        assert result["first_pose_target_xyz_quat_xyzw"] == parameters["first_pose_candidates_xyz_quat_xyzw"][1]
    offset = 0
    for index, stages in enumerate((rejected, [None, None] if next_candidate_succeeds else rejected)):
        first_target = parameters["first_pose_candidates_xyz_quat_xyzw"][index]
        for stage in range(len(stages)):
            source, target = state.calls[offset + stage]
            expected = list(first_target)
            if stage == 1:
                expected[2] += parameters["continuation_delta_xyz_m"][2]
                np.testing.assert_array_equal(source[6:9], first_target[:3])
            else:
                np.testing.assert_array_equal(source, original_qpos)
            assert list(target) == expected
            np.testing.assert_array_equal(source[:6], original_qpos[:6])
            np.testing.assert_array_equal(source[12:], original_qpos[12:])
        offset += len(stages)
    assert len(state.calls) == offset

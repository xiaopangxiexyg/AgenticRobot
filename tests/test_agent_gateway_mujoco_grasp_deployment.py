"""Pure native registration closure; no scene, model, devices or motion start."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_agent_gateway_mujoco_tcp_pose import _build_no_model_contract_bundle_fixture

from agent_gateway.profiles.mujoco.grasp_deployment import (
    MUJOCO_EXECUTE_GRASP_APPROACH_PLAN_ROUTE_ID_V1,
    build_mujoco_fast_grasp_contract,
)
from agent_gateway.profiles.mujoco.motion_deployment import (
    build_mujoco_agent_tcp_pose_contract_bundle,
)
from agent_gateway.profiles.mujoco.motion_factory import (
    _bind_grippers,
    _build_standard_tcp_pose_envelope_runtime,
)
from agent_gateway.protocol.skill_catalog import validate_skill_route_catalog_closure
from deployments.perception.grasp_first_path import FirstGraspPathPolicy
from robot_motion.target_safety import TCPPoseEnvelopeSafetyRuntime
from sim.mujoco.deploy.grasp_contact import MujocoGraspContactBinding

pytestmark = [pytest.mark.unit, pytest.mark.contract]

MODEL_HASH = "a" * 64
APPROACH_CONFIG = {
    "pregrasp_distance_m": 0.07,
    "near_distance_m": 0.025,
    "terminal_offset_m": 0.0,
    "terminal_speed_scale": 0.25,
}


def _native_bundle(
    monkeypatch,
    *,
    allow_execute=True,
    source_id="front",
    source_width=640,
    session_id=None,
    contact_target="blue_cube",
    calibration_hash=None,
):
    application, runtime, _source, _bundle, _args = (
        _build_no_model_contract_bundle_fixture(monkeypatch)
    )
    application.visual_source_catalog = lambda: (
        {"source_id": source_id, "native_width": source_width, "native_height": 480},
    )
    if session_id is not None:
        runtime.config.runtime_session_id = session_id
    if calibration_hash is not None:
        application.calibration_hash = calibration_hash
    application.target = SimpleNamespace(
        planned_tcp_motion=SimpleNamespace(planning_clearance_padding_m=0.002)
    )
    contact = MujocoGraspContactBinding(contact_target, ("left_pad", "right_pad"))
    safety = _build_standard_tcp_pose_envelope_runtime(
        prepared=application,
        target_config=runtime.config,
        enable_grasp_approach=True,
        grasp_contact_binding=contact,
    )
    entries = []
    for entry in safety.catalog_entries:
        routes = tuple(
            route
            for route in entry.allowed_route_ids
            if route != MUJOCO_EXECUTE_GRASP_APPROACH_PLAN_ROUTE_ID_V1
        )
        if allow_execute and "sim.mujoco.motion.approach_grasp_pose.v1" in routes:
            routes = (*routes, MUJOCO_EXECUTE_GRASP_APPROACH_PLAN_ROUTE_ID_V1)
        entries.append(replace(entry, allowed_route_ids=routes))
    application.tcp_pose_envelope_runtime = TCPPoseEnvelopeSafetyRuntime(
        catalog_entries=tuple(entries),
        target_frame_binding_hash=safety.target_frame_binding_hash,
        tool_frame_binding_hash_by_side=safety.tool_frame_binding_hash_by_side,
        reference_frame_by_side=safety.reference_frame_by_side,
        clock_domain_id=safety.clock_domain_id,
    )
    bundle = build_mujoco_agent_tcp_pose_contract_bundle(
        application=application,
        enable_artifact_perception=True,
        enable_grasp_approach=True,
        grasp_contact_binding=contact,
    )
    return _bind_grippers(
        bundle,
        {
            component: SimpleNamespace(port_binding_hash=component + ":native-port")
            for component in ("left_gripper", "right_gripper")
        },
    )


@pytest.fixture
def native(monkeypatch):
    return _native_bundle(monkeypatch)


def _build(native, *, model_hash=MODEL_HASH, config=None):
    return build_mujoco_fast_grasp_contract(
        native,
        model_profile_hash=model_hash,
        approach_config=APPROACH_CONFIG if config is None else config,
    )


def test_complete_native_closure_preserves_existing_definitions_and_safety(native):
    before = native.route_manifest.semantic_content_hash
    updated = _build(native)
    contract = updated.fast_grasp_contract
    validate_skill_route_catalog_closure(
        definition_catalog=updated.definition_catalog,
        schema_catalog=updated.schema_catalog,
        io_catalog=updated.io_catalog,
        route_manifest=updated.route_manifest,
    )
    assert native.route_manifest.semantic_content_hash == before
    assert native.fast_grasp_contract is None
    assert (
        len(updated.route_manifest.skill_routes)
        == len(native.route_manifest.skill_routes) + 3
    )
    assert {
        item.skill_id: item
        for item in updated.definition_catalog.skill_definitions
        if item in native.definition_catalog.skill_definitions
    } == {item.skill_id: item for item in native.definition_catalog.skill_definitions}
    for previous in native.route_manifest.skill_routes:
        assert (
            updated.route_manifest.route(
                previous.route.skill_id, previous.route.skill_version
            )
            == previous
        )
        assert updated.payload_validators.validator_for(
            previous.route
        ) is native.payload_validators.validator_for(previous.route)
    assert contract.roles == ("main",)
    assert contract.source_by_role == {"main": "front"}
    assert contract.model_profile_hash == MODEL_HASH
    assert contract.policy == FirstGraspPathPolicy()
    assert contract.approach_config == APPROACH_CONFIG
    for registration, identity in (
        (contract.observation, "perception.capture_grasp_observation"),
        (contract.fast, "perception.plan_grasp_fast"),
        (contract.execute, "motion.execute_grasp_approach_plan"),
    ):
        assert registration.skill.definition.skill_id == identity
        assert registration.skill.definition.skill_version == "1"
        assert updated.route_manifest.route(identity, "1") == registration.entry
        assert (
            updated.payload_validators.validator_for(registration.entry.route)
            is registration.payload_validator
        )
    for registration in (contract.observation, contract.fast):
        route = registration.entry.route
        assert (
            route.control_effect_class,
            route.privilege_tier,
            route.cancellation_policy,
        ) == ("read_only", "t0_read_only", "cancel_only")
        assert not any(
            (
                route.may_move_robot,
                route.may_open_publication,
                route.requires_deadman,
                route.requires_operator_confirmation,
                route.requires_task_evaluation,
            )
        )
        assert registration.entry.evaluation_plan_hash is None
        assert not registration.entry.deployment_binding.safety_envelope_refs
    slots = {
        slot.slot_id: slot
        for slot in contract.observation.skill.io_contract.output_artifact_slots
    }
    assert set(slots) == {"grasp_observation", "rgb_frames"}
    assert slots["grasp_observation"].allowed_compressions == ("zip",)
    assert contract.fast.skill.io_contract.output_artifact_slots == ()
    previous = native.grasp_approach_contract
    execution = contract.execution_contract
    assert execution.motion_profile is previous.motion_profile
    assert execution.contact_binding is previous.contact_binding
    assert (
        execution.deployment_binding.safety_envelope_refs
        == previous.deployment_binding.safety_envelope_refs
    )
    assert (
        execution.evaluation_plan.evaluator_specs
        == previous.evaluation_plan.evaluator_specs
    )
    assert (
        contract.execute.entry.deployment_binding
        == execution.deployment_binding.to_ref()
    )
    assert (
        execution.evaluation_plan.route_hash
        == contract.execute.entry.route.semantic_content_hash
    )
    assert (
        execution.evaluation_plan.deployment_binding_hash
        == execution.deployment_binding.semantic_content_hash
    )
    assert (
        contract.execute.entry.route.cancellation_policy
        == "hold_on_cancel_disconnect_or_timeout"
    )


def test_source_and_approach_config_are_copied_immutable_and_hashes_deterministic(
    native,
):
    config = dict(APPROACH_CONFIG)
    first = _build(native, config=config).fast_grasp_contract
    config["terminal_offset_m"] = 0.01
    assert first.approach_config == APPROACH_CONFIG
    with pytest.raises(TypeError):
        first.approach_config["terminal_offset_m"] = 0.01
    with pytest.raises(TypeError):
        first.source_by_role["main"] = "other"
    assert _build(native).fast_grasp_contract.profile_hash == first.profile_hash
    changed = _build(native, model_hash="b" * 64).fast_grasp_contract
    assert changed.model_profile_hash == "b" * 64
    assert changed.profile_hash != first.profile_hash
    assert changed.frame_binding_hash == first.frame_binding_hash
    for name, value in (
        ("pregrasp_distance_m", 0.08),
        ("near_distance_m", 0.02),
        ("terminal_offset_m", 0.01),
        ("terminal_speed_scale", 0.3),
    ):
        assert (
            _build(
                native, config=APPROACH_CONFIG | {name: value}
            ).fast_grasp_contract.profile_hash
            != first.profile_hash
        )


def test_rgb_source_catalog_is_bound_to_both_hashes(monkeypatch):
    first = _build(_native_bundle(monkeypatch)).fast_grasp_contract
    changed = _build(
        _native_bundle(monkeypatch, source_id="right_wrist")
    ).fast_grasp_contract
    assert changed.source_by_role == {"wrist_right": "right_wrist"}
    assert changed.roles == ("wrist_right",)
    assert changed.frame_binding_hash != first.frame_binding_hash
    assert changed.profile_hash != first.profile_hash


@pytest.mark.parametrize(
    "change,frame_changes",
    [
        ({"source_width": 800}, True),
        ({"session_id": "runtime:another-native-session"}, True),
        ({"contact_target": "another_bound_target"}, False),
        ({"calibration_hash": "scene-calibration:another-native-scene"}, True),
    ],
)
def test_hashes_bind_native_source_session_target_and_calibration(
    monkeypatch, change, frame_changes
):
    first = _build(_native_bundle(monkeypatch)).fast_grasp_contract
    changed = _build(_native_bundle(monkeypatch, **change)).fast_grasp_contract
    assert changed.profile_hash != first.profile_hash
    assert (changed.frame_binding_hash != first.frame_binding_hash) is frame_changes


def test_execute_accepts_only_native_exact_plan_reference_and_reuses_result_evaluation(
    native,
):
    contract = _build(native).fast_grasp_contract
    validator = contract.execute.payload_validator
    parameters = {
        "component_id": "right_arm",
        "grasp_approach_plan_id": "mujoco-grasp-approach-plan:" + "c" * 64,
        "grasp_approach_plan_hash": "d" * 64,
    }
    validator.validate_parameters(parameters)
    schema = contract.execute.skill.schema_entry.parameter_schema
    assert set(schema["required"]) == set(parameters)
    assert schema["additionalProperties"] is False
    for bad in (
        parameters | {"extra": True},
        {name: value for name, value in parameters.items() if name != "component_id"},
        parameters | {"component_id": "left_arm"},
        parameters | {"grasp_approach_plan_id": "real-grasp-approach-plan:" + "c" * 64},
        parameters
        | {"grasp_approach_plan_id": "mujoco-grasp-approach-plan:" + "C" * 64},
        parameters | {"grasp_approach_plan_hash": "C" * 64},
    ):
        with pytest.raises(ValueError):
            validator.validate_parameters(bad)
    validator.validate_execution_result(
        {
            "execution_receipt_refs": ["native:receipt"],
            "final_measured_state_hash": "native:state",
        }
    )
    validator.validate_task_evaluation(
        {
            "aggregate_evaluation_receipt_ref": "native:evaluation",
            "task_outcome": "succeeded",
        }
    )
    with pytest.raises(ValueError):
        validator.validate_execution_result(
            {"execution_receipt_refs": [], "final_measured_state_hash": "native:state"}
        )
    with pytest.raises(ValueError):
        validator.validate_task_evaluation(
            {
                "aggregate_evaluation_receipt_ref": "native:evaluation",
                "task_outcome": "completed",
            }
        )


def test_fast_and_observation_validators_use_registered_roles_and_exact_frames(native):
    contract = _build(native).fast_grasp_contract
    observation = contract.observation.payload_validator
    observation.validate_parameters({"selected_roles": ["main"]})
    with pytest.raises(ValueError):
        observation.validate_parameters({"selected_roles": ["unregistered"]})
    parameters = {
        "component_id": "right_arm",
        "observation_operation_id": "observation:test",
        "role": "main",
        "target_pixel_uv": [10, 20],
        "view_policy": "auto",
        "speed_scale": 0.05,
        "planning_budget_sec": 20,
    }
    contract.fast.payload_validator.validate_parameters(parameters)
    contract.fast.payload_validator.validate_parameters(
        parameters | {"speed_scale": 1.0}
    )
    with pytest.raises(ValueError):
        contract.fast.payload_validator.validate_parameters(
            parameters | {"speed_scale": 0.049}
        )
    with pytest.raises(ValueError):
        contract.fast.payload_validator.validate_parameters(
            parameters | {"role": "unregistered"}
        )
    profile = native.grasp_approach_contract.motion_profile
    properties = contract.fast.skill.schema_entry.execution_result_schema["properties"]
    for name in (
        "reference_frame_id",
        "reference_frame_spec_hash",
        "frame_graph_hash",
        "tool_config_hash",
        "calibration_hash",
    ):
        assert properties[name]["const"] == getattr(profile, name)
    with pytest.raises(ValueError):
        contract.fast.payload_validator.validate_task_evaluation(
            {"task_outcome": "succeeded"}
        )


@pytest.mark.parametrize(
    "field",
    [
        "rgb_capture",
        "grasp_approach_contract",
        "grasp_contact_contract",
        "gripper_contract",
    ],
)
def test_missing_native_prerequisites_fail_before_registration(native, field):
    with pytest.raises(ValueError):
        _build(replace(native, **{field: None}))


def test_native_owner_mismatch_is_rejected_even_with_same_public_target(native):
    approach = native.grasp_approach_contract
    corrupt = replace(
        approach,
        deployment_binding=replace(
            approach.deployment_binding, runtime_session_id="runtime:other"
        ),
    )
    # Exercise the public construction boundary with an inconsistent input;
    # no native execution or provider is instantiated by this fixture.
    supplied = SimpleNamespace(**(vars(native) | {"grasp_approach_contract": corrupt}))
    with pytest.raises(ValueError, match="native owner mismatch: runtime_session_id"):
        _build(supplied)


def test_execute_must_be_allowed_by_existing_native_envelope(monkeypatch):
    with pytest.raises(ValueError, match="envelope must already allow"):
        _build(_native_bundle(monkeypatch, allow_execute=False))


@pytest.mark.parametrize("model_hash", ["", "A" * 64, "a" * 63, "model-profile:alias"])
def test_model_profile_hash_is_an_exact_digest(native, model_hash):
    with pytest.raises(ValueError):
        _build(native, model_hash=model_hash)


@pytest.mark.parametrize(
    "changes",
    [
        {"pregrasp_distance_m": 0.01},
        {"near_distance_m": 0},
        {"terminal_offset_m": float("nan")},
        {"terminal_offset_m": 0.02},
        {"terminal_speed_scale": 0},
        {"terminal_speed_scale": True},
        {"unbound_option": 1},
    ],
)
def test_invalid_or_unbound_approach_settings_are_rejected(native, changes):
    with pytest.raises((TypeError, ValueError)):
        _build(native, config=APPROACH_CONFIG | changes)

"""Real profile/neutral-contract closure, with device and inference fakes."""

import copy

import pytest
from test_agent_gateway_real_station_point_grasp import _provider_stack, _request

from agent_gateway.profiles.real.point_grasp_finger_exchange import (
    upgrade_finger_exchange_point_grasp_registration,
)
from agent_gateway.profiles.real.station_point_grasp_deployment import (
    build_station_bound_point_grasp_registration,
)
from agent_gateway.skills.perception.point_grasp_finger_exchange import (
    FINGER_EXCHANGE_POSE_SOURCE,
    finger_exchange_point_grasp_contract,
)


def exchange_stack(*, frozen=False):
    # No annotation is requested; the callable only closes the embodied profile.
    _, _, _planner, old, provider = _provider_stack(
        backend_id="graspgenx",
        arm_reachability_evaluator=lambda *_a, **_kw: pytest.fail(
            "No annotation requested"
        ),
    )
    p, b = old.profile, old.binding
    registration = build_station_bound_point_grasp_registration(
        **{
            name: getattr(p, name)
            for name in (
                "source_binding_hash",
                "frame_binding_hash",
                "frame_graph_hash",
                "source_frame_id",
                "source_frame_spec_hash",
                "reference_frame_id",
                "reference_frame_spec_hash",
                "backend_id",
                "model_profile_id",
                "model_profile_hash",
                "inference_implementation_hash",
            )
        },
        **{
            name: getattr(b, name)
            for name in (
                "runtime_session_id",
                "control_domain_id",
                "target_binding_id",
                "station_manifest_hash",
                "execution_manifest_hash",
            )
        },
        aligned_rgbd_source_catalog=tuple(
            {"source_id": sid, "role": role, "native_width": w, "native_height": h}
            for sid, role, w, h in p.aligned_rgbd_sources
        ),
        arm_reachability_profile=p.arm_reachability_profile,
        frozen_observation=frozen,
        finger_exchange=True,
    )
    payload = copy.deepcopy(
        provider.plan_request(_request(include_arm_reachability=False)).execution_result
    )
    for candidate in payload["candidates"]:
        candidate.update(
            pose_source=FINGER_EXCHANGE_POSE_SOURCE,
            orientation_variant_axis="z",
            orientation_variant_angle_deg=180.0,
        )
    parameters = dict(_request(include_arm_reachability=False).parameters)
    if frozen:
        parameters.update(
            observation_operation_id="observation:test", view_policy="single"
        )
        payload["view_fusion"] = {
            "observation_operation_id": "observation:test",
            "view_policy": "single",
            "diagnostics_json": '[{"role":"main","used":true}]',
        }
    return old, registration, parameters, payload


@pytest.mark.parametrize("frozen", [False, True])
def test_v5_closes_live_or_frozen_contract_without_mutating_published_candidates(
    frozen,
):
    old, registration, parameters, payload = exchange_stack(frozen=frozen)
    skill = registration.skill
    route = skill.routes[0]
    assert skill.definition.skill_version == route.skill_version == "5"
    assert (
        registration.payload_validator.skill_version
        == registration.binding.deployment_binding_version
        == "5"
    )
    assert (
        route.skill_definition_hash
        == skill.io_contract.skill_definition_hash
        == skill.definition.semantic_content_hash
    )
    assert route.skill_io_contract_hash == skill.io_contract.semantic_content_hash
    assert (
        route.execution_result_schema_hash
        == registration.payload_validator.execution_result_schema_hash
    )
    assert route.control_effect_class == "read_only" and not route.may_move_robot
    before = copy.deepcopy(payload)
    registration.payload_validator.validate_parameters(parameters)
    registration.payload_validator.validate_execution_result(payload)
    registration.payload_validator.validate_execution_result_for_parameters(
        parameters, payload
    )
    assert payload == before
    assert old.skill.definition.skill_version == "3"
    assert (
        old.skill.schema_entry.execution_result_schema["properties"]["candidates"][
            "items"
        ]["properties"]["orientation_variant_angle_deg"]["maximum"]
        == 45
    )
    assert (registration.observation_registration is not None) == frozen


@pytest.mark.parametrize(
    "change",
    ["axis", "angle", "local_label", "clearance", "extra_field", "matrix", "count"],
)
def test_v5_rejects_invalid_exchange_and_preserves_common_checks(change):
    _, registration, _, payload = exchange_stack()
    candidate = payload["candidates"][0]
    if change == "axis":
        candidate["orientation_variant_axis"] = "x"
    elif change == "angle":
        candidate["orientation_variant_angle_deg"] = 179.0
    elif change == "local_label":
        candidate["pose_source"] = "dmtac_local_orientation_variant_v1"
    elif change == "clearance":
        candidate["support_clearance_world_z_m"] = 0.005
    elif change == "extra_field":
        candidate["caller_override"] = True
    elif change == "matrix":
        candidate["T_reference_from_tool_tcp"][-1] = 2.0
    elif change == "count":
        payload["candidate_count"] += 1
    with pytest.raises((TypeError, ValueError)):
        registration.payload_validator.validate_execution_result(payload)


def test_old_version_rejects_half_turn_and_upgrade_does_not_change_old_hashes():
    old, _, _, payload = exchange_stack()
    hashes = (
        old.skill.definition.semantic_content_hash,
        old.skill.schema_entry.semantic_content_hash,
        old.skill.io_contract.semantic_content_hash,
        old.skill.routes[0].semantic_content_hash,
    )
    with pytest.raises(ValueError):
        old.payload_validator.validate_execution_result(payload)
    finger_exchange_point_grasp_contract(
        old.skill.definition,
        old.skill.schema_entry,
        profile_hash=old.profile.semantic_content_hash,
    )
    with pytest.raises(ValueError, match="explicitly enable"):
        upgrade_finger_exchange_point_grasp_registration(old)
    assert hashes == (
        old.skill.definition.semantic_content_hash,
        old.skill.schema_entry.semantic_content_hash,
        old.skill.io_contract.semantic_content_hash,
        old.skill.routes[0].semantic_content_hash,
    )


def test_v5_is_deployment_bound_not_a_caller_request_to_relax_safety():
    _, registration, parameters, _ = exchange_stack()
    with pytest.raises(ValueError):
        registration.payload_validator.validate_parameters(
            {**parameters, "finger_exchange": True}
        )

"""Portable feedback validation; synthetic ports are not physical evidence."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agent_gateway.profiles.object_state import (
    ObjectStateProfile, build_object_state_executor, build_object_state_registration,
)
from agent_gateway.skills.state.object_state import (
    RELEASE_POLICY, object_state_definition, object_state_schema,
    validate_object_state_parameters, validate_object_state_result,
    validate_object_state_result_for_parameters,
)

pytestmark = pytest.mark.unit


def _profile(**changes):
    return ObjectStateProfile(object_id="target", reference_frame_id="station",
        reference_frame_spec_hash="frame-spec", frame_graph_hash="graph", clock_domain_id="native-clock",
        observation_source="synthetic-unit-fixture", source_binding_hash="source-binding", **changes)


def _parameters():
    return {"object_id": "target", "reference_frame_id": "station", "max_measurement_age_ms": 50}


def _state():
    return {**_profile().identity, "measured_at": 1., "cutoff_at": 1.,
        "pose_xyz_quat_xyzw": [0., 0., .1, 0., 0., 0., 1.],
        "linear_velocity_xyz_m_s": [0., 0., 0.], "angular_velocity_xyz_rad_s": [0., 0., 0.],
        "support_contact": True, "gripper_contact": False, "other_contact": False,
        "release_observation": None}


def _validate(state, *, release_supported=False):
    definition = object_state_definition()
    profile = _profile(release_supported=release_supported)
    schema = object_state_schema(definition_hash=definition.semantic_content_hash, identity=profile.identity,
        profile_hash=profile.semantic_content_hash, release_supported=release_supported)
    validate_object_state_result(state, identity=profile.identity,
        fields=schema.execution_result_schema["properties"], release_supported=release_supported)
    validate_object_state_result_for_parameters(_parameters(), state)
    return schema


def _release():
    return {"status": "stable_supported_release", "stable_supported_release": True,
        "measured_at": 1., "evidence_age_s": 0., "evidence_fresh": True,
        "window_start_s": .4, "observed_duration_s": .6, "measurement_count": 61,
        "max_position_drift_m": 0., "max_rotation_drift_rad": 0.,
        "window_reason": "complete", "max_observed_sample_gap_s": .01,
        "policy": dict(RELEASE_POLICY)}


def test_object_state_is_passive_portable_and_missing_evidence_remains_unknown():
    definition = object_state_definition()
    assert definition.skill_kind == "inspection" and definition.controlled_component_ids == ()
    state = _state()
    schema = _validate(state)
    assert schema.execution_result_schema["properties"]["release_observation"]["const"] is None
    for key in ("pose_xyz_quat_xyzw", "linear_velocity_xyz_m_s", "angular_velocity_xyz_rad_s",
                "support_contact", "gripper_contact", "other_contact"):
        state[key] = None
    _validate(state)


@pytest.mark.parametrize("change", [
    {"object_id": "other"}, {"reference_frame_id": "world"}, {"clock_domain_id": "wall"},
    {"source_binding_hash": "another-source"}, {"observation_source": "guessed"},
    {"cutoff_at": 1.051}, {"measured_at": 1.01}, {"measured_at": True},
    {"linear_velocity_xyz_m_s": [float("nan"), 0, 0]}, {"support_contact": 1},
    {"pose_xyz_quat_xyzw": [0, 0, 0, 0, 0, 0, 2]}, {"extra": "ignored"},
])
def test_invalid_measurements_and_identity_cannot_become_success(change):
    with pytest.raises(ValueError):
        _validate({**_state(), **change})


@pytest.mark.parametrize("age", [True, 0, 51, "50"])
def test_query_freshness_is_explicit_and_bounded(age):
    with pytest.raises(ValueError):
        validate_object_state_parameters({**_parameters(), "max_measurement_age_ms": age}, identity=_profile().identity)


def test_requested_stricter_freshness_is_not_replaced_by_the_profile_maximum():
    state = {**_state(), "cutoff_at": 1.02}
    _validate(state)
    with pytest.raises(ValueError, match="requested"):
        validate_object_state_result_for_parameters({**_parameters(), "max_measurement_age_ms": 10}, state)


def test_stable_release_requires_the_declared_source_and_current_supported_cut():
    state = {**_state(), "release_observation": _release()}
    with pytest.raises(ValueError, match="producer"):
        _validate(state)
    _validate(state, release_supported=True)
    for key, value in (("gripper_contact", True), ("support_contact", None),
                       ("other_contact", True), ("pose_xyz_quat_xyzw", None)):
        with pytest.raises(ValueError, match="Stable release"):
            _validate({**state, key: value}, release_supported=True)


@pytest.mark.parametrize("change", [
    {"measurement_count": 2}, {"observed_duration_s": .1}, {"evidence_fresh": False},
    {"max_position_drift_m": .003}, {"max_rotation_drift_rad": .06},
    {"window_start_s": 1.1}, {"measured_at": 1.01}, {"evidence_age_s": -.1},
    {"measurement_count": True}, {"window_reason": "guessed"}, {"policy": {}},
    {"window_start_s": None}, {"window_reason": "collecting"},
    {"max_observed_sample_gap_s": None}, {"max_observed_sample_gap_s": .051},
    {"max_observed_sample_gap_s": .001},
])
def test_release_history_cannot_fabricate_stability(change):
    state = {**_state(), "release_observation": {**_release(), **change}}
    with pytest.raises(ValueError):
        _validate(state, release_supported=True)


def test_registration_inherits_no_motion_authority_and_reads_exactly_once():
    from test_agent_gateway_real_motion_primitives import _inputs
    from agent_gateway.profiles.real.motion_primitive_profiles import build_real_agent_primitive_contract_bundle

    attachment, config, graph, envelope = _inputs()
    base = build_real_agent_primitive_contract_bundle(attachment=attachment, port_config=config,
        frame_graph=graph, primitive_reference_frame_id="canonical_station", safety_envelope_refs=(envelope,))
    entry = base.tcp_pose_observe.route_manifest.route("state.read_tcp_pose", "1")
    reg = build_object_state_registration(base_entry=entry, profile=_profile())
    assert reg.entry.route.control_effect_class == "read_only"
    assert not reg.entry.route.may_move_robot and not reg.entry.route.requires_task_evaluation
    assert reg.entry.deployment_binding.deployment_binding_hash != entry.deployment_binding.deployment_binding_hash
    reads = []
    executor = build_object_state_executor(registration=reg,
        capture=lambda: reads.append(1) or deepcopy(_state()),
        current_station_revision=lambda: 0, clock=lambda: 1., max_retained_operations=2)
    result = executor._read(SimpleNamespace(parameters=_parameters()), SimpleNamespace())
    assert reads == [1] and result.execution_result["release_observation"] is None
    assert result.reason_code == "passive_object_state_read_completed"
    with pytest.raises(ValueError):
        build_object_state_registration(base_entry=replace(entry,
            route=replace(entry.route, control_effect_class="motion")), profile=_profile())

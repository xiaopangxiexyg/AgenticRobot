from __future__ import annotations

from dataclasses import replace

import pytest

from agent_gateway.profiles.real.gripper_state_composition import (
    assemble_real_gripper_state_gateway_session,
    build_real_gripper_state_contract_bundle,
)
from agent_gateway.profiles.real.gripper_state_deployment import (
    REAL_GRIPPER_STATE_EXECUTOR_ID_V1,
    REAL_GRIPPER_STATE_ROUTE_ID_V1,
    build_real_gripper_state_registration,
)
from agent_gateway.profiles.real.gripper_state_gateway import (
    PublishedGripperState,
    PublishedGripperStateBuffer,
    build_real_gripper_state_executor,
    read_published_gripper_state,
)
from agent_gateway.protocol.contracts import AgentSkillRequest
from agent_gateway.skills.state.definition import READ_GRIPPER_STATE_SKILL_ID_V1

COMPONENTS = ("gripper:left", "gripper:right")


def _registration(*, current_mode: str = "real_active"):
    return build_real_gripper_state_registration(
        component_ids=COMPONENTS,
        observation_binding_id="dmtac-runtime-telemetry:v1",
        observation_binding_hash="sha256:dmtac-runtime-telemetry",
        clock_domain_id="runtime-monotonic",
        runtime_session_id="real-session:gripper-test",
        control_domain_id="real-control:test",
        target_binding_id="real-target:dual-rm65",
        station_manifest_hash="station-manifest:test",
        execution_manifest_hash="execution-manifest:test",
        maximum_measurement_age_ms=200,
        current_mode=current_mode,
    )


def _state(*, measured_at: float = 9.95, available_at: float = 9.96):
    return PublishedGripperState(
        measurement_id="gripper-measurement:17",
        clock_domain_id="runtime-monotonic",
        close_fraction_by_component={
            "gripper:left": 0.25,
            "gripper:right": 0.75,
        },
        measured_at_by_component={
            "gripper:left": measured_at,
            "gripper:right": measured_at + 0.001,
        },
        state_source_by_component={
            "gripper:left": "runtime_backend",
            "gripper:right": "runtime_backend",
        },
        available_at=available_at,
        evidence_refs=("dmtac-telemetry:left:17", "dmtac-telemetry:right:17"),
    )


def test_registration_is_strictly_read_only_and_inline() -> None:
    registration = _registration()
    definition = registration.skill.definition
    route = registration.skill.routes[0]

    assert definition.skill_id == READ_GRIPPER_STATE_SKILL_ID_V1
    assert definition.skill_kind == "inspection"
    assert route.route_id == REAL_GRIPPER_STATE_ROUTE_ID_V1
    assert route.control_effect_class == "read_only"
    assert route.privilege_tier == "t0_read_only"
    assert route.cancellation_policy == "cancel_only"
    assert route.may_move_robot is False
    assert route.may_open_publication is False
    assert route.requires_operator_confirmation is False
    assert route.requires_deadman is False
    assert registration.binding.executor_id == REAL_GRIPPER_STATE_EXECUTOR_ID_V1
    assert registration.skill.io_contract.input_artifact_slots == ()
    assert registration.skill.io_contract.output_artifact_slots == ()
    close_schema = registration.skill.schema_entry.execution_result_schema[
        "properties"
    ]["close_fraction_by_component"]
    assert close_schema["minProperties"] == 1
    assert "required" not in close_schema


def test_read_projects_requested_measured_subset_without_hardware_io() -> None:
    registration = _registration()
    source = PublishedGripperStateBuffer()
    source.publish(_state())

    read = read_published_gripper_state(
        registration=registration,
        source=source,
        parameters={
            "component_ids": ["gripper:right"],
            "max_measurement_age_ms": 100,
        },
        station_revision=8,
        now=10.0,
    )

    assert source.no_io is True
    assert read.execution_result["component_ids"] == ["gripper:right"]
    assert read.execution_result["close_fraction_by_component"] == {
        "gripper:right": 0.75
    }
    assert read.execution_result["station_revision"] == 8
    assert read.reason_code == "passive_real_gripper_state_read_completed"
    assert "dmtac-telemetry:right:17" in read.evidence_refs


def test_read_fails_closed_for_stale_or_assumed_state() -> None:
    registration = _registration()
    source = PublishedGripperStateBuffer()
    source.publish(_state(measured_at=9.0, available_at=9.01))

    with pytest.raises(RuntimeError, match="stale"):
        read_published_gripper_state(
            registration=registration,
            source=source,
            parameters={
                "component_ids": list(COMPONENTS),
                "max_measurement_age_ms": 200,
            },
            station_revision=1,
            now=10.0,
        )

    with pytest.raises(ValueError, match="measured, not assumed"):
        replace(
            _state(),
            state_source_by_component={
                "gripper:left": "configured_disconnected_assumption",
                "gripper:right": "runtime_backend",
            },
        )


def test_buffer_rejects_regression_and_executor_requires_no_io_source() -> None:
    registration = _registration()
    source = PublishedGripperStateBuffer()
    source.publish(_state())
    with pytest.raises(ValueError, match="regressed"):
        source.publish(_state(measured_at=8.0, available_at=8.1))

    executor = build_real_gripper_state_executor(
        registration=registration,
        source=source,
        current_station_revision=lambda: 3,
        clock=lambda: 10.0,
    )
    assert executor.control_effect_class == "read_only"
    assert executor.executor_id == REAL_GRIPPER_STATE_EXECUTOR_ID_V1

    class _HardwareReadingSource:
        no_io = False

        def latest_gripper_state(self):
            raise AssertionError("must not be called")

    with pytest.raises(TypeError, match="no-I/O"):
        build_real_gripper_state_executor(
            registration=registration,
            source=_HardwareReadingSource(),
            current_station_revision=lambda: 3,
        )


def test_gateway_composition_discovers_and_executes_gripper_state() -> None:
    registration = _registration(current_mode="real_observe")
    bundle = build_real_gripper_state_contract_bundle(registration)
    source = PublishedGripperStateBuffer()
    source.publish(_state())
    session = assemble_real_gripper_state_gateway_session(
        contract_bundle=bundle,
        source=source,
        lower_station_revision=lambda: 8,
        allowed_client_ids=("rpent:gripper-test",),
        clock=lambda: 10.0,
    )
    adapter = session.wire_adapter(authenticated_client_id="rpent:gripper-test")

    assert (
        adapter.get_skill_definition_catalog()["skill_definitions"][0]["skill_id"]
        == "state.read_gripper_state"
    )
    assert (
        adapter.get_skill_route_manifest()["skill_routes"][0]["route"]["route_id"]
        == REAL_GRIPPER_STATE_ROUTE_ID_V1
    )
    assert (
        adapter.get_skill_io_catalog()["io_contracts"][0]["input_artifact_slots"] == []
    )

    published = session.observation_producer.capture_and_publish(session.service)
    assert published.accepted
    snapshot = session.service.get_observation()
    assert snapshot is not None
    entry = bundle.route_manifest.route("state.read_gripper_state", "1")
    route = entry.route
    profile = route.allowed_parameter_profiles[0]
    request = AgentSkillRequest(
        client_id="rpent:gripper-test",
        request_id="request:gripper-state:1",
        idempotency_key="idem:gripper-state:1",
        client_sequence=1,
        runtime_session_id=bundle.route_manifest.runtime_session_id,
        skill_route_manifest_hash=bundle.route_manifest.semantic_content_hash,
        expected_station_revision=snapshot.station_revision,
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        skill_definition_hash=route.skill_definition_hash,
        route_id=route.route_id,
        route_version=route.route_version,
        route_hash=route.semantic_content_hash,
        deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
        parameter_profile_id=profile.parameter_profile_id,
        parameter_profile_hash=profile.parameter_profile_hash,
        parameters={
            "component_ids": ["gripper:left", "gripper:right"],
            "max_measurement_age_ms": 100,
        },
        observation_snapshot_id=snapshot.snapshot_id,
        requested_mode="real_observe",
        requested_timeout_ms=500,
    )

    submitted = session.service.submit_request(
        request=request,
        current_station_revision=snapshot.station_revision,
        now=10.0,
    )
    assert submitted.accepted
    assert submitted.operation is not None
    operation_id = submitted.operation.operation_id
    launched = session.service.launch_operation(operation_id)
    assert launched.accepted
    assert session.state_executor.wait(operation_id, timeout_sec=1.0)
    completed = session.service.get_operation(operation_id)
    assert completed is not None
    assert completed.status == "completed"
    assert completed.execution_result is not None
    assert completed.execution_result["close_fraction_by_component"] == {
        "gripper:left": 0.25,
        "gripper:right": 0.75,
    }
    assert session.close().accepted

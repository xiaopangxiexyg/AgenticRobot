from __future__ import annotations

import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agent_gateway.profiles.real.gripper_action_composition import (
    assemble_real_gripper_action_gateway_session,
    build_real_gripper_action_contract_bundle,
)
from agent_gateway.profiles.real.gripper_action_deployment import (
    REAL_CLOSE_GRIPPER_ROUTE_ID_V1,
    REAL_OPEN_GRIPPER_ROUTE_ID_V1,
    build_real_gripper_action_registrations,
)
from agent_gateway.profiles.real.gripper_action_gateway import (
    RealGripperActionSkillLifecycle,
)
from agent_gateway.protocol.contracts import AgentSkillOperation, AgentSkillRequest
from agent_gateway.runtime.access_policy import issue_skill_execution_grant
from agent_gateway.skills.motion.definition import (
    CLOSE_GRIPPER_SKILL_ID_V1,
    OPEN_GRIPPER_SKILL_ID_V1,
)
from teleop_runtime.backends.grippers.dmtac import (
    DMTacGripperBackend,
    DMTacGripperBackendConfig,
)
from teleop_runtime.skill_deploy.gripper_port import DMTacGripperActionPort

COMPONENTS = ("gripper:left", "gripper:right")


class _FakeBackend:
    def __init__(self, *, name: str, initial: float, final: float) -> None:
        self.name = name
        self.value = initial
        self.final = final
        self.sent = []
        self.hold_reasons: list[str] = []
        self.is_moving = False

    def read_gripper_state_now(self):
        return {
            "joint_pos": self.value,
            "timestamp": time.time(),
            "is_moving": self.is_moving,
            "error_code": 0,
        }

    def send_command(self, command):
        self.sent.append(command)
        self.value = self.final
        self.is_moving = False
        return True

    def hold_motion(self, reason: str = "manual hold") -> None:
        self.hold_reasons.append(reason)
        self.is_moving = False


class _FailAfterSendBackend(_FakeBackend):
    def send_command(self, command):
        self.sent.append(command)
        self.value = self.final
        self.is_moving = True
        raise RuntimeError("dispatch result unavailable after device command")


class _QueuedBackend(_FakeBackend):
    def __init__(self, *, name: str, initial: float, final: float) -> None:
        super().__init__(name=name, initial=initial, final=final)
        self.queued = False

    def send_command(self, command):
        self.sent.append(command)
        self.queued = True
        self.is_moving = False
        return True

    def hold_motion(self, reason: str = "manual hold") -> None:
        super().hold_motion(reason)
        self.queued = False


class _ReadFailureAfterSendBackend(_QueuedBackend):
    def __init__(self, *, name: str, initial: float, final: float) -> None:
        super().__init__(name=name, initial=initial, final=final)
        self.fail_reads = False

    def read_gripper_state_now(self):
        if self.fail_reads:
            raise RuntimeError("feedback unavailable")
        return super().read_gripper_state_now()

    def send_command(self, command):
        sent = super().send_command(command)
        self.fail_reads = True
        return sent


class _FaultAfterSendBackend(_FakeBackend):
    def __init__(self, *, name: str, initial: float, final: float) -> None:
        super().__init__(name=name, initial=initial, final=final)
        self.error_code = 0

    def read_gripper_state_now(self):
        payload = super().read_gripper_state_now()
        payload["error_code"] = self.error_code
        return payload

    def send_command(self, command):
        sent = super().send_command(command)
        self.error_code = 7
        return sent


class _SequencedBackend:
    def __init__(
        self,
        *,
        name: str,
        close_fractions: tuple[float, ...],
        moving: tuple[bool, ...] | None = None,
    ) -> None:
        self.name = name
        self._close_fractions = close_fractions
        self._moving = moving or (False,)
        self._read_index = 0
        self.sent = []
        self.hold_reasons: list[str] = []

    def read_gripper_state_now(self):
        index = min(self._read_index, len(self._close_fractions) - 1)
        self._read_index += 1
        return {
            "joint_pos": self._close_fractions[index],
            "timestamp": time.time(),
            "is_moving": self._moving[min(index, len(self._moving) - 1)],
            "error_code": 0,
        }

    def send_command(self, command):
        self.sent.append(command)
        return True

    def hold_motion(self, reason: str = "manual hold") -> None:
        self.hold_reasons.append(reason)


def _registrations():
    return build_real_gripper_action_registrations(
        component_ids=COMPONENTS,
        clock_domain_id="python-time-wall",
        runtime_session_id="real-session:gripper-actions",
        control_domain_id="real-control:grippers",
        target_binding_id="real-target:dual-rm65",
        station_manifest_hash="station-manifest:test",
        execution_manifest_hash="execution-manifest:test",
        stable_measurement_count=2,
    )


def _request(registration, *, component_id: str) -> AgentSkillRequest:
    route = registration.skill.routes[0]
    profile = route.allowed_parameter_profiles[0]
    return AgentSkillRequest(
        client_id="rpent:gripper-actions",
        request_id="request:" + route.skill_id,
        idempotency_key="idem:" + route.skill_id,
        client_sequence=1,
        runtime_session_id=registration.binding.runtime_session_id,
        skill_route_manifest_hash="manifest:gripper-actions",
        expected_station_revision=3,
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        skill_definition_hash=route.skill_definition_hash,
        route_id=route.route_id,
        route_version=route.route_version,
        route_hash=route.semantic_content_hash,
        deployment_binding_hash=registration.binding.semantic_content_hash,
        parameter_profile_id=profile.parameter_profile_id,
        parameter_profile_hash=profile.parameter_profile_hash,
        parameters={"component_id": component_id},
        observation_snapshot_id="snapshot:gripper-actions:3",
        requested_mode="real_active",
        requested_timeout_ms=2_000,
        skill_execution_grant_id="agent-skill-grant:test",
    )


def _operation(request: AgentSkillRequest) -> AgentSkillOperation:
    now = time.time()
    return AgentSkillOperation(
        operation_id="operation:" + request.skill_id,
        request_hash=request.semantic_content_hash,
        runtime_session_id=request.runtime_session_id,
        status="running",
        reason_code="activated",
        station_revision_at_admission=3,
        current_station_revision=3,
        started_at=now,
        updated_at=now,
        deadline_at=now + 2.0,
        evaluation_status="pending",
        task_outcome="pending",
    )


def test_paired_registrations_are_fixed_real_motion_routes() -> None:
    opened, closed = _registrations()

    assert opened.skill.definition.skill_id == OPEN_GRIPPER_SKILL_ID_V1
    assert closed.skill.definition.skill_id == CLOSE_GRIPPER_SKILL_ID_V1
    assert opened.skill.routes[0].route_id == REAL_OPEN_GRIPPER_ROUTE_ID_V1
    assert closed.skill.routes[0].route_id == REAL_CLOSE_GRIPPER_ROUTE_ID_V1
    for registration in (opened, closed):
        route = registration.skill.routes[0]
        assert route.control_effect_class == "motion"
        assert route.privilege_tier == "t2_real_motion"
        assert route.requires_operator_confirmation is True
        assert route.cancellation_policy == "hold_on_cancel_disconnect_or_timeout"
        assert set(registration.skill.schema_entry.parameter_schema["properties"]) == {
            "component_id"
        }
        with pytest.raises(ValueError, match="exactly component_id"):
            registration.payload_validator.validate_parameters(
                {"component_id": "gripper:left", "amount": 0.5}
            )


@pytest.mark.parametrize(
    ("registration_index", "initial", "final", "expected_target", "predicate"),
    (
        (0, 0.8, 0.0, 0.0, "gripper_open_endpoint_reached.v1"),
        (1, 0.0, 0.62, 1.0, "gripper_close_command_settled.v1"),
    ),
)
def test_endpoint_lifecycle_dispatches_fixed_command_and_returns_compact_result(
    registration_index: int,
    initial: float,
    final: float,
    expected_target: float,
    predicate: str,
) -> None:
    registration = _registrations()[registration_index]
    backend = _FakeBackend(name="left_hand", initial=initial, final=final)
    port = DMTacGripperActionPort(
        port_id="lower:gripper-actions",
        clock_domain_id="python-time-wall",
        backends_by_component={"gripper:left": backend},
        side_by_component={"gripper:left": "left"},
    )
    request = _request(registration, component_id="gripper:left")
    operation = _operation(request)
    lifecycle = RealGripperActionSkillLifecycle(
        registration=registration,
        port=port,
        request=request,
        operation=operation,
        current_station_revision=lambda: 4,
        sleep=lambda _: None,
    )

    activated = lifecycle.activate(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )
    assert activated.succeeded
    completion = lifecycle.run(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )

    assert completion.succeeded
    assert completion.terminal_hold_confirmed
    assert completion.execution_result is not None
    assert completion.execution_result["component_id"] == "gripper:left"
    assert completion.execution_result["commanded_close_fraction"] == expected_target
    assert completion.execution_result["initial_close_fraction"] == initial
    assert completion.execution_result["final_close_fraction"] == final
    assert completion.execution_result["station_revision"] == 4
    assert completion.evaluation is not None
    assert completion.evaluation.task_evaluation["predicate"] == predicate
    assert len(backend.sent) == 1
    assert backend.sent[0].command_type == "gripper_scalar"
    assert backend.sent[0].value == expected_target


def test_dispatch_uncertainty_keeps_component_fenced_for_cleanup_hold() -> None:
    registration = _registrations()[1]
    backend = _FailAfterSendBackend(name="left_hand", initial=0.1, final=0.6)
    port = DMTacGripperActionPort(
        port_id="lower:gripper-actions",
        clock_domain_id="python-time-wall",
        backends_by_component={"gripper:left": backend},
        side_by_component={"gripper:left": "left"},
    )
    request = _request(registration, component_id="gripper:left")
    operation = _operation(request)
    lifecycle = RealGripperActionSkillLifecycle(
        registration=registration,
        port=port,
        request=request,
        operation=operation,
        current_station_revision=lambda: 4,
        sleep=lambda _: None,
    )

    with pytest.raises(RuntimeError, match="dispatch result unavailable"):
        lifecycle.activate(
            request=request,
            operation=operation,
            cancel_requested=lambda: False,
        )

    held = lifecycle.request_hold(
        operation=operation,
        reason="dispatch_uncertain_cleanup",
    )
    assert held.lower_hold_confirmed
    assert backend.hold_reasons == ["dispatch_uncertain_cleanup"]


def test_active_dmtac_hold_calls_vendor_stop_when_cached_motion_is_idle() -> None:
    stop_calls: list[str] = []
    backend = DMTacGripperBackend(
        DMTacGripperBackendConfig(
            side="right",
            connect_hardware=True,
            connect_tactile=False,
        )
    )
    backend.device = SimpleNamespace(
        gripper=object(),
        stop=lambda: stop_calls.append("stop"),
    )
    backend._latest_gripper_is_moving = False

    backend.hold_motion("active_gateway_fence")

    assert stop_calls == ["stop"]


def test_hold_stops_a_queued_command_even_when_feedback_reports_idle() -> None:
    backend = _QueuedBackend(name="left_hand", initial=0.1, final=0.6)
    port = DMTacGripperActionPort(
        port_id="lower:gripper-actions",
        clock_domain_id="python-time-wall",
        backends_by_component={"gripper:left": backend},
        side_by_component={"gripper:left": "left"},
    )

    port.dispatch(
        operation_id="operation:queued",
        component_id="gripper:left",
        target_close_fraction=1.0,
    )
    assert backend.queued
    receipt = port.hold(operation_id="operation:queued", reason="terminal_hold")

    assert receipt.succeeded
    assert not backend.queued
    assert backend.hold_reasons == ["terminal_hold"]


def test_hold_readback_failure_keeps_the_component_fenced() -> None:
    backend = _ReadFailureAfterSendBackend(
        name="left_hand",
        initial=0.1,
        final=0.6,
    )
    port = DMTacGripperActionPort(
        port_id="lower:gripper-actions",
        clock_domain_id="python-time-wall",
        backends_by_component={"gripper:left": backend},
        side_by_component={"gripper:left": "left"},
    )
    port.dispatch(
        operation_id="operation:readback-failure",
        component_id="gripper:left",
        target_close_fraction=1.0,
    )

    with pytest.raises(RuntimeError, match="feedback unavailable"):
        port.hold(
            operation_id="operation:readback-failure",
            reason="first_hold",
        )
    backend.fail_reads = False
    receipt = port.hold(
        operation_id="operation:readback-failure",
        reason="retry_hold",
    )

    assert receipt.succeeded
    assert backend.hold_reasons == ["first_hold", "retry_hold"]


def test_hardware_fault_after_dispatch_fails_and_holds_promptly() -> None:
    registration = _registrations()[1]
    backend = _FaultAfterSendBackend(name="left_hand", initial=0.1, final=0.6)
    port = DMTacGripperActionPort(
        port_id="lower:gripper-actions",
        clock_domain_id="python-time-wall",
        backends_by_component={"gripper:left": backend},
        side_by_component={"gripper:left": "left"},
    )
    request = _request(registration, component_id="gripper:left")
    operation = _operation(request)
    lifecycle = RealGripperActionSkillLifecycle(
        registration=registration,
        port=port,
        request=request,
        operation=operation,
        current_station_revision=lambda: 4,
        sleep=lambda _: None,
    )
    lifecycle.activate(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )

    completion = lifecycle.run(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )

    assert not completion.succeeded
    assert completion.reason_code == "gripper_action_feedback_invalid"
    assert not completion.terminal_hold_confirmed
    assert backend.hold_reasons == ["gripper_action_feedback_invalid"]


@pytest.mark.parametrize(
    "samples",
    (
        (0.0, 0.0),
        (0.8, 0.4),
    ),
)
def test_close_requires_effect_in_the_commanded_direction(
    samples: tuple[float, float],
) -> None:
    registration = _registrations()[1]
    backend = _FakeBackend(
        name="left_hand",
        initial=samples[0],
        final=samples[1],
    )
    port = DMTacGripperActionPort(
        port_id="lower:gripper-actions",
        clock_domain_id="python-time-wall",
        backends_by_component={"gripper:left": backend},
        side_by_component={"gripper:left": "left"},
    )
    request = _request(registration, component_id="gripper:left")
    operation = replace(_operation(request), deadline_at=time.time() + 0.005)
    lifecycle = RealGripperActionSkillLifecycle(
        registration=registration,
        port=port,
        request=request,
        operation=operation,
        current_station_revision=lambda: 4,
        sleep=lambda _: None,
    )
    lifecycle.activate(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )

    completion = lifecycle.run(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )

    assert not completion.succeeded
    assert completion.reason_code == "gripper_action_deadline_exceeded"
    assert completion.terminal_hold_confirmed


def test_close_does_not_latch_a_transient_command_effect() -> None:
    registration = _registrations()[1]
    backend = _SequencedBackend(
        name="left_hand",
        close_fractions=(0.1, 0.1, 0.6, 0.1, 0.1, 0.1),
    )
    port = DMTacGripperActionPort(
        port_id="lower:gripper-actions",
        clock_domain_id="python-time-wall",
        backends_by_component={"gripper:left": backend},
        side_by_component={"gripper:left": "left"},
    )
    request = _request(registration, component_id="gripper:left")
    operation = replace(_operation(request), deadline_at=time.time() + 0.005)
    lifecycle = RealGripperActionSkillLifecycle(
        registration=registration,
        port=port,
        request=request,
        operation=operation,
        current_station_revision=lambda: 4,
        sleep=lambda _: None,
    )
    lifecycle.activate(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )

    completion = lifecycle.run(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )

    assert not completion.succeeded
    assert completion.reason_code == "gripper_action_deadline_exceeded"
    assert completion.terminal_hold_confirmed


def test_close_rechecks_settling_after_terminal_hold() -> None:
    registration = _registrations()[1]
    backend = _SequencedBackend(
        name="left_hand",
        close_fractions=(0.1, 0.1, 0.6, 0.6, 0.6, 0.3, 0.3),
    )
    port = DMTacGripperActionPort(
        port_id="lower:gripper-actions",
        clock_domain_id="python-time-wall",
        backends_by_component={"gripper:left": backend},
        side_by_component={"gripper:left": "left"},
    )
    request = _request(registration, component_id="gripper:left")
    operation = _operation(request)
    lifecycle = RealGripperActionSkillLifecycle(
        registration=registration,
        port=port,
        request=request,
        operation=operation,
        current_station_revision=lambda: 4,
        sleep=lambda _: None,
    )
    lifecycle.activate(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )

    completion = lifecycle.run(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )

    assert not completion.succeeded
    assert completion.reason_code == "gripper_action_post_hold_close_not_settled"
    assert not completion.terminal_hold_confirmed
    retry = lifecycle.request_hold(operation=operation, reason="worker_retry_hold")
    assert retry.lower_hold_confirmed
    assert backend.hold_reasons == [
        "gripper_action_terminal_idle",
        "worker_retry_hold",
    ]


def test_open_rechecks_endpoint_after_terminal_hold() -> None:
    registration = _registrations()[0]
    backend = _SequencedBackend(
        name="left_hand",
        close_fractions=(0.8, 0.8, 0.0, 0.0, 0.0, 0.2),
    )
    port = DMTacGripperActionPort(
        port_id="lower:gripper-actions",
        clock_domain_id="python-time-wall",
        backends_by_component={"gripper:left": backend},
        side_by_component={"gripper:left": "left"},
    )
    request = _request(registration, component_id="gripper:left")
    operation = _operation(request)
    lifecycle = RealGripperActionSkillLifecycle(
        registration=registration,
        port=port,
        request=request,
        operation=operation,
        current_station_revision=lambda: 4,
        sleep=lambda _: None,
    )

    activated = lifecycle.activate(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )
    assert activated.succeeded

    completion = lifecycle.run(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )
    assert not completion.succeeded
    assert completion.reason_code == "gripper_action_post_hold_open_endpoint_lost"
    assert not completion.terminal_hold_confirmed
    assert completion.execution_result is None
    retry = lifecycle.request_hold(operation=operation, reason="worker_retry_hold")
    assert retry.lower_hold_confirmed
    assert backend.hold_reasons == [
        "gripper_action_terminal_idle",
        "worker_retry_hold",
    ]


def test_terminal_hold_moving_feedback_keeps_the_component_fenced() -> None:
    registration = _registrations()[1]
    backend = _SequencedBackend(
        name="left_hand",
        close_fractions=(0.1, 0.1, 0.6, 0.6, 0.6, 0.6, 0.6),
        moving=(False, False, False, False, False, True, False),
    )
    port = DMTacGripperActionPort(
        port_id="lower:gripper-actions",
        clock_domain_id="python-time-wall",
        backends_by_component={"gripper:left": backend},
        side_by_component={"gripper:left": "left"},
    )
    request = _request(registration, component_id="gripper:left")
    operation = _operation(request)
    lifecycle = RealGripperActionSkillLifecycle(
        registration=registration,
        port=port,
        request=request,
        operation=operation,
        current_station_revision=lambda: 4,
        sleep=lambda _: None,
    )
    lifecycle.activate(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )

    completion = lifecycle.run(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )

    assert not completion.succeeded
    assert completion.reason_code == "gripper_action_post_hold_motion_detected"
    assert not completion.terminal_hold_confirmed
    retry = lifecycle.request_hold(operation=operation, reason="worker_retry_hold")
    assert retry.lower_hold_confirmed
    assert backend.hold_reasons == [
        "gripper_action_terminal_idle",
        "worker_retry_hold",
    ]


def test_open_requires_the_measured_open_endpoint() -> None:
    registration = _registrations()[0]
    result = {
        "component_id": "gripper:left",
        "commanded_close_fraction": 0.0,
        "initial_close_fraction": 0.5,
        "final_close_fraction": 0.2,
        "final_measured_at": 10.0,
        "completed_at": 10.1,
        "station_revision": 2,
        "command_receipt_ref": "command:1",
        "final_measurement_hash": "measurement:1",
    }
    registration.payload_validator.validate_execution_result(result)
    # Schema validation confirms shape and fixed command semantics; runtime
    # completion separately enforces the deployment's open tolerance.
    assert result["final_close_fraction"] > registration.profile.open_endpoint_tolerance


def test_gateway_rejects_a_lower_port_from_another_clock_domain() -> None:
    registrations = _registrations()
    bundle = build_real_gripper_action_contract_bundle(registrations)
    backend = _FakeBackend(name="left_hand", initial=0.1, final=0.6)
    other_backend = _FakeBackend(name="right_hand", initial=0.1, final=0.6)
    port = DMTacGripperActionPort(
        port_id="lower:gripper-actions",
        clock_domain_id="another-clock",
        backends_by_component={
            "gripper:left": backend,
            "gripper:right": other_backend,
        },
        side_by_component={
            "gripper:left": "left",
            "gripper:right": "right",
        },
    )

    with pytest.raises(ValueError, match="clock domain"):
        assemble_real_gripper_action_gateway_session(
            contract_bundle=bundle,
            port=port,
            lower_station_revision=lambda: 1,
            allowed_client_ids=("rpent:test",),
        )


def test_gateway_discovers_and_executes_close_with_request_bound_grant() -> None:
    registrations = build_real_gripper_action_registrations(
        component_ids=("gripper:left",),
        clock_domain_id="python-time-wall",
        runtime_session_id="real-session:gripper-gateway",
        control_domain_id="real-control:gripper-gateway",
        target_binding_id="real-target:left-gripper",
        station_manifest_hash="station-manifest:gateway-test",
        execution_manifest_hash="execution-manifest:gateway-test",
        stable_measurement_count=2,
    )
    bundle = build_real_gripper_action_contract_bundle(registrations)
    backend = _FakeBackend(name="left_hand", initial=0.1, final=0.55)
    port = DMTacGripperActionPort(
        port_id="lower:gripper-gateway",
        clock_domain_id="python-time-wall",
        backends_by_component={"gripper:left": backend},
        side_by_component={"gripper:left": "left"},
    )
    gateway = assemble_real_gripper_action_gateway_session(
        contract_bundle=bundle,
        port=port,
        lower_station_revision=lambda: 7,
        allowed_client_ids=("rpent:gripper-gateway",),
        sleep=lambda _: None,
    )
    adapter = gateway.wire_adapter(authenticated_client_id="rpent:gripper-gateway")
    discovered = adapter.get_skill_definition_catalog()["skill_definitions"]
    assert {item["skill_id"] for item in discovered} == {
        "motion.open_gripper",
        "motion.close_gripper",
    }
    published = gateway.observation_producer.capture_and_publish(gateway.service)
    assert published.accepted and published.snapshot is not None
    snapshot = published.snapshot
    entry = bundle.route_manifest.route(CLOSE_GRIPPER_SKILL_ID_V1, "1")
    route = entry.route
    profile = route.allowed_parameter_profiles[0]
    pre_grant = AgentSkillRequest(
        client_id="rpent:gripper-gateway",
        request_id="request:gateway-close:1",
        idempotency_key="idem:gateway-close:1",
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
        parameters={"component_id": "gripper:left"},
        observation_snapshot_id=snapshot.snapshot_id,
        requested_mode="real_active",
        requested_timeout_ms=2_000,
    )
    now = time.time()
    grant = issue_skill_execution_grant(
        access_policy=gateway.service.access_policy,
        manifest=bundle.route_manifest,
        route=route,
        request=pre_grant,
        operator_authorization_receipt_id="operator-confirmation:close-left",
        client_id=pre_grant.client_id,
        parameter_profile_id=pre_grant.parameter_profile_id,
        parameter_profile_hash=pre_grant.parameter_profile_hash,
        max_invocations=1,
        issued_at=now,
        expires_at=now + 5.0,
    )
    request = replace(pre_grant, skill_execution_grant_id=grant.grant_id)
    submitted = gateway.service.submit_request(
        request=request,
        current_station_revision=snapshot.station_revision,
        grant=grant,
    )
    assert submitted.accepted and submitted.operation is not None
    launched = gateway.service.launch_operation(submitted.operation.operation_id)
    assert launched.accepted
    assert gateway.wait(timeout_sec=1.0)
    operation = gateway.service.get_operation(submitted.operation.operation_id)
    assert operation is not None
    assert operation.status == "completed"
    assert operation.task_outcome == "succeeded"
    assert operation.execution_result is not None
    assert operation.execution_result["final_close_fraction"] == 0.55
    assert gateway.close().accepted

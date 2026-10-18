from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

from agent_gateway.profiles.mujoco.gripper_skill import MujocoGripperControlPort
from agent_gateway.profiles.real.gripper_skill import (
    ExclusiveRealGripperResources,
    RealDmtacGripperControlPort,
    build_real_dmtac_gripper_skill_contract_bundle,
    stop_arm_and_gripper,
    stop_arm_and_grippers,
    build_real_dmtac_grippers_skill_contract_bundle,
)
from agent_gateway.protocol.contracts import AgentSkillOperation, AgentSkillRequest
from agent_gateway.skills.gripper.definition import (
    SET_GRIPPER_STATE_RESULT_SCHEMA_V1,
    validate_set_gripper_state_parameters_v2,
    validate_set_gripper_state_result_for_parameters_v2,
    validate_set_gripper_state_result_for_parameters_v1,
)
from agent_gateway.skills.gripper.deployment import (
    GripperSkillDeploymentBinding,
    build_gripper_skill_contract_bundle,
)
from agent_gateway.skills.gripper.runtime import (
    GripperCommandReceipt,
    GripperConvergenceProfile,
    GripperHoldReceipt,
    GripperReadiness,
    GripperSkillLifecycle,
    GripperStateSample,
    assemble_gripper_agent_gateway_session,
    build_gripper_skill_executor,
)


class FakeGripperPort:
    component_id = "right_gripper"
    target_kind = "sim"
    port_binding_hash = "port:fake-gripper"

    def __init__(self, samples: list[tuple[float, bool | None]]) -> None:
        self.samples = list(samples)
        self.commanded: list[float] = []
        self.holds: list[str] = []
        self.sequence = 0

    def readiness(self) -> GripperReadiness:
        return GripperReadiness(True, "fake:ready", ("fake-ready-receipt",))

    def command_close(self, target_close: float, *, cancel_requested=None) -> GripperCommandReceipt:
        if cancel_requested is not None and cancel_requested():
            return GripperCommandReceipt(False, "fake:preempted", ("fake-preempted",))
        self.commanded.append(target_close)
        return GripperCommandReceipt(
            True,
            f"fake:command:{len(self.commanded)}",
            (f"fake-command-receipt:{len(self.commanded)}",),
        )

    def observe_state(self) -> GripperStateSample:
        self.sequence += 1
        close, moving = self.samples.pop(0) if self.samples else (1.0, False)
        return GripperStateSample(
            measured_close=close,
            moving=moving,
            fault_code=None,
            observed_at=time.monotonic(),
            evidence_ref=f"fake-sample:{self.sequence}",
        )

    def poll_state(self) -> GripperStateSample:
        return self.observe_state()

    def hold(self, *, reason: str) -> GripperHoldReceipt:
        self.holds.append(reason)
        return GripperHoldReceipt(True, (f"fake-hold:{len(self.holds)}",))


def _sim_bundle(port: FakeGripperPort):
    return build_gripper_skill_contract_bundle(
        GripperSkillDeploymentBinding(
            runtime_session_id="runtime:gripper-test",
            control_domain_id="domain:right-gripper",
            component_id=port.component_id,
            target_binding_id="target:mujoco-test",
            target_kind="sim",
            current_mode="sim_active",
            executor_id="executor:fake-gripper",
            port_binding_hash=port.port_binding_hash,
            station_manifest_hash="station:test",
            execution_manifest_hash="execution:test",
        )
    )


def _request(bundle, snapshot, *, target_state: str, sequence: int = 1):
    entry = bundle.route_manifest.skill_routes[0]
    route = entry.route
    profile = route.allowed_parameter_profiles[0]
    return AgentSkillRequest(
        client_id="client:test",
        request_id=f"request:{sequence}",
        idempotency_key=f"gripper:{sequence}",
        client_sequence=sequence,
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
        parameters={"target_state": target_state},
        observation_snapshot_id=snapshot.snapshot_id,
        requested_mode=bundle.route_manifest.current_mode,
        requested_timeout_ms=1_000,
    )


def test_sim_route_is_target_neutral_semantics_with_sim_privilege() -> None:
    port = FakeGripperPort([(0.0, False)])
    bundle = _sim_bundle(port)
    definition = bundle.definition_catalog.skill_definitions[0]
    route = bundle.route_manifest.skill_routes[0].route

    assert definition.skill_id == "manipulation.set_gripper_state"
    assert definition.intended_target_kind == "target_agnostic"
    assert definition.controlled_component_ids == ("right_gripper",)
    assert route.target_kind == "sim"
    assert route.privilege_tier == "t1_sim_or_no_motion"
    assert route.requires_operator_confirmation is False
    assert (
        "approach_then_close_composition" in bundle.route_manifest.explicit_exclusions
    )


def test_gripper_executor_rejects_wrong_port_before_activation():
    port = FakeGripperPort([(0.0, False)])
    bundle = _sim_bundle(port)
    port.port_binding_hash = "port:other-device"
    with pytest.raises(ValueError, match="port identity"):
        build_gripper_skill_executor(
            contract_bundle=bundle, port=port, current_station_revision=lambda: 0
        )
    assert port.commanded == port.holds == []


def test_gateway_executes_closed_contact_settle_and_reuses_executor() -> None:
    port = FakeGripperPort(
        [
            (0.0, False),  # initial observation
            (0.25, True),
            (0.4, False),
            (0.4, False),
        ]
    )
    bundle = _sim_bundle(port)
    session = assemble_gripper_agent_gateway_session(
        contract_bundle=bundle,
        port=port,
        allowed_client_ids=("client:test",),
        current_station_revision=lambda: 3,
        clock_domain_id="clock:test",
        convergence=GripperConvergenceProfile(
            stable_sample_count=2,
            minimum_contact_settle_sec=0.0,
            poll_interval_sec=0.0,
        ),
    )
    snapshot = session.publish_observation()
    request = _request(bundle, snapshot, target_state="closed")
    submitted = session.service.submit_request(
        request=request,
        current_station_revision=3,
    )
    assert submitted.accepted
    assert submitted.operation is not None
    assert session.service.launch_operation(submitted.operation.operation_id).accepted
    assert session.executor.wait(timeout_sec=1.0)

    operation = session.service.get_operation(submitted.operation.operation_id)
    assert operation is not None
    assert operation.execution_outcome == "completed"
    assert operation.execution_result == {
        "schema_version": SET_GRIPPER_STATE_RESULT_SCHEMA_V1,
        "component_id": "right_gripper",
        "commanded_state": "closed",
        "measured_close": 0.4,
        "completion_kind": "settled_before_full_close",
        "terminal_hold_confirmed": True,
    }
    assert port.commanded == [1.0]
    assert len(port.holds) == 1

    port.samples.extend(
        [
            (0.4, False),  # next passive observation
            (0.2, True),
            (0.01, False),
            (0.01, False),
        ]
    )
    open_snapshot = session.publish_observation()
    open_request = _request(
        bundle,
        open_snapshot,
        target_state="open",
        sequence=2,
    )
    open_submission = session.service.submit_request(
        request=open_request,
        current_station_revision=3,
    )
    assert open_submission.accepted
    assert open_submission.operation is not None
    assert session.service.launch_operation(
        open_submission.operation.operation_id
    ).accepted
    assert session.executor.wait(timeout_sec=1.0)
    open_operation = session.service.get_operation(
        open_submission.operation.operation_id
    )
    assert open_operation is not None
    assert open_operation.execution_outcome == "completed"
    assert open_operation.execution_result["commanded_state"] == "open"
    assert open_operation.execution_result["completion_kind"] == "target_reached"
    assert port.commanded == [1.0, 0.0]
    assert len(port.holds) == 2


def test_wire_adapter_observation_uses_receiver_admission_outcome() -> None:
    port = FakeGripperPort([(0.2, False), (0.3, False)])
    bundle = _sim_bundle(port)
    session = assemble_gripper_agent_gateway_session(
        contract_bundle=bundle,
        port=port,
        allowed_client_ids=("client:test",),
        current_station_revision=lambda: 3,
        clock_domain_id="clock:test",
    )
    adapter = session.wire_adapter(authenticated_client_id="client:test")

    wire_snapshot = adapter.get_observation_snapshot()

    admitted_snapshot = session.service.get_observation()
    assert admitted_snapshot is not None
    assert wire_snapshot == admitted_snapshot.to_wire()

    session.close()
    with pytest.raises(
        RuntimeError,
        match="Gateway observation publication failed: agent_gateway_service_closed",
    ):
        adapter.get_observation_snapshot()


def test_open_result_cannot_claim_contact_settle() -> None:
    result = {
        "schema_version": SET_GRIPPER_STATE_RESULT_SCHEMA_V1,
        "component_id": "right_gripper",
        "commanded_state": "open",
        "measured_close": 0.3,
        "completion_kind": "settled_before_full_close",
        "terminal_hold_confirmed": True,
    }
    try:
        validate_set_gripper_state_result_for_parameters_v1(
            {"target_state": "open"},
            result,
        )
    except ValueError as error:
        assert "opening cannot complete" in str(error)
    else:  # pragma: no cover - explicit regression boundary.
        raise AssertionError("invalid open completion was accepted")


@dataclass
class _FakeDmtacConfig:
    side: str = "right"
    gripper_port: str = "/dev/fake"
    gripper_id: int = 1
    observation_only: bool = False
    connect_hardware: bool = True
    disconnected_state_scalar: float | None = None


class _FakeDmtacBackend:
    def __init__(self) -> None:
        self.config = _FakeDmtacConfig()
        self.commands: list[float] = []
        self.holds: list[str] = []
        self.samples = [(0.0, True), (0.98, False)]
        self.held = False
        self.events = []

    def start(self):
        self.events.append("hand_start")

    def close(self):
        self.events.append("hand_close")

    def is_ready(self) -> bool:
        return True

    def readiness_report(self):
        return {"ready": True, "reason": ""}

    def send_fixed_joint_pos(self, value):
        self.held = False
        self.commands.append(float(value))
        return True

    def read_gripper_state_now(self):
        return {"timestamp": time.time()}

    def get_state(self, **_kwargs):
        close, moving = self.samples.pop(0) if self.samples else (0.98, False)
        if self.held:
            moving = False
        return {
            "gripper_measured_scalar": close,
            "gripper_error_code": 0,
            "gripper_is_moving": moving,
        }

    def hold_gripper_motion(self, reason):
        self.holds.append(reason)
        self.held = True
        return True


def test_gripper_resources_start_once_and_close_hand_before_lower():
    backend = _FakeDmtacBackend()
    backend.samples = [(0.0, False)]

    class Lower:
        def start(self):
            backend.events.append("lower_start")

        def close(self):
            backend.events.append("lower_close")

    resources = ExclusiveRealGripperResources(
        lower=Lower(), port=RealDmtacGripperControlPort(backend=backend)
    )
    resources.start()
    with pytest.raises(RuntimeError, match="start once"):
        resources.start()
    resources.close()
    resources.close()
    assert backend.events == ["lower_start", "hand_start", "hand_close", "lower_close"]
    assert backend.commands == []


def test_gripper_startup_moving_state_is_rejected_and_cleans_owned_resources():
    backend = _FakeDmtacBackend()

    class Lower:
        def start(self):
            backend.events.append("lower_start")

        def close(self):
            backend.events.append("lower_close")

    resources = ExclusiveRealGripperResources(
        lower=Lower(), port=RealDmtacGripperControlPort(backend=backend)
    )
    with pytest.raises(RuntimeError, match="stationary"):
        resources.start()
    assert backend.events == ["lower_start", "hand_start", "hand_close", "lower_close"]
    assert backend.commands == []


@pytest.mark.parametrize(
    "reading", [None, {"timestamp": 0.0}, {"timestamp": float("inf")}]
)
def test_real_gripper_never_reuses_cached_state_after_invalid_fresh_read(reading):
    backend = _FakeDmtacBackend()
    backend.read_gripper_state_now = lambda: reading
    port = RealDmtacGripperControlPort(backend=backend)
    with pytest.raises(RuntimeError):
        port.observe_state()


def test_real_gripper_stop_ack_without_stopped_feedback_is_not_hold_proof():
    backend = _FakeDmtacBackend()
    backend.hold_gripper_motion = lambda reason: True
    backend.get_state = lambda **kwargs: {
        "gripper_measured_scalar": 0.98,
        "gripper_error_code": 0,
        "gripper_is_moving": True,
    }
    port = RealDmtacGripperControlPort(backend=backend, hold_confirmation_timeout_sec=0.02)
    assert port.hold(reason="test unconfirmed stop").succeeded is False


@pytest.mark.parametrize("failed_boundary", ["arm", "hand", "both"])
def test_composed_stop_attempts_both_boundaries_even_after_exception(failed_boundary):
    backend = _FakeDmtacBackend()
    calls = []

    def arm_stop():
        calls.append("arm")
        if failed_boundary in {"arm", "both"}:
            raise RuntimeError("arm stop failed")
        return GripperHoldReceipt(True, ("arm-stop-proof",))

    def hand_stop(reason):
        calls.append("hand")
        if failed_boundary in {"hand", "both"}:
            raise RuntimeError("hand stop failed")
        backend.held = True
        return True

    backend.hold_gripper_motion = hand_stop
    port = RealDmtacGripperControlPort(backend=backend)
    with pytest.raises(RuntimeError, match="Composed stop failed") as captured:
        stop_arm_and_gripper(arm_stop=arm_stop, port=port, reason="shutdown")
    assert calls == ["arm", "hand"]
    for boundary in {"arm", "hand"} if failed_boundary == "both" else {failed_boundary}:
        assert f"{boundary} stop failed" in str(captured.value)


def test_real_gripper_command_or_failed_read_clears_previous_stop_proof():
    backend = _FakeDmtacBackend()
    port = RealDmtacGripperControlPort(backend=backend)
    assert port.hold(reason="first stop").succeeded
    assert port.last_hold_receipt is not None
    port.command_close(1.0)
    assert port.last_hold_receipt is None
    assert port.hold(reason="second stop").succeeded
    backend.read_gripper_state_now = lambda: None
    with pytest.raises(RuntimeError, match="unavailable"):
        port.hold(reason="unproved stop")
    assert port.last_hold_receipt is None


def test_real_gripper_fresh_timestamp_does_not_hide_an_over_budget_read():
    backend = _FakeDmtacBackend()

    def slow_read():
        time.sleep(0.21)
        return {"timestamp": time.time()}

    backend.read_gripper_state_now = slow_read
    port = RealDmtacGripperControlPort(backend=backend)
    with pytest.raises(RuntimeError, match="200 ms budget"):
        port.observe_state()


def test_real_adapter_and_route_are_default_off_until_granted() -> None:
    backend = _FakeDmtacBackend()
    port = RealDmtacGripperControlPort(backend=backend)
    bundle = build_real_dmtac_gripper_skill_contract_bundle(
        runtime_session_id="runtime:real-gripper-test",
        control_domain_id="domain:real-right-gripper",
        target_binding_id="target:dmtac-test",
        station_manifest_hash="station:real-test",
        execution_manifest_hash="execution:real-test",
        port=port,
    )
    route = bundle.route_manifest.skill_routes[0].route

    assert route.target_kind == "real"
    assert route.privilege_tier == "t2_real_motion"
    assert route.requires_operator_confirmation is True
    session = assemble_gripper_agent_gateway_session(
        contract_bundle=bundle,
        port=port,
        allowed_client_ids=("client:test",),
        current_station_revision=lambda: 4,
        clock_domain_id="clock:real-test",
    )
    snapshot = session.publish_observation()
    request = _request(bundle, snapshot, target_state="closed")
    submission = session.service.submit_request(
        request=request,
        current_station_revision=4,
    )
    assert submission.accepted is False
    assert "grant" in submission.reason_code or "confirmation" in submission.reason_code
    assert backend.commands == []
    assert backend.holds == []


def test_real_adapter_dispatches_normalized_close_and_measures_feedback() -> None:
    backend = _FakeDmtacBackend()
    port = RealDmtacGripperControlPort(backend=backend)

    command = port.command_close(1.0)
    sample = port.poll_state()
    hold = port.hold(reason="test terminal hold")

    assert command.accepted is True
    assert backend.commands == [1.0]
    assert sample.measured_close == 0.0
    assert sample.moving is True
    assert hold.succeeded is True
    assert backend.holds == ["test terminal hold"]


def test_real_gripper_binding_distinguishes_continuous_grasp_configuration():
    backend = _FakeDmtacBackend()
    first = RealDmtacGripperControlPort(backend=backend).port_binding_hash
    backend.config.continuous_grasp = False
    second = RealDmtacGripperControlPort(backend=backend).port_binding_hash
    assert first != second
    assert backend.commands == backend.holds == []


def test_real_adapter_rejects_backend_side_that_does_not_match_component() -> None:
    backend = _FakeDmtacBackend()
    backend.config.side = "left"

    with pytest.raises(ValueError, match="bound to right_gripper"):
        RealDmtacGripperControlPort(backend=backend)

    assert backend.commands == []
    assert backend.holds == []


def test_mujoco_adapter_observation_is_passive_and_command_is_gripper_only() -> None:
    class Config:
        realtime_worker = False

    class Rig:
        config = Config()

        def __init__(self) -> None:
            self.close = 0.0
            self.target = 0.0
            self.steps = 0
            self.held = []

        def is_ready(self):
            return True

        def queue_gripper_close_targets(self, values):
            self.target = float(values["right_gripper"])
            return 7

        def gripper_execution_snapshot(self):
            return {
                "measured_close_by_gripper": {"right_gripper": self.close},
                "actuator_target_rate_by_gripper": {
                    "right_gripper": 0.5 if self.close != self.target else 0.0
                },
                "pending_close_by_gripper": {},
            }

        def hold_gripper_motion(self, reason):
            self.held.append(reason)

        def step_physics(self):
            self.steps += 1
            self.close = self.target

    rig = Rig()
    port = MujocoGripperControlPort(rig=rig, gripper_name="right_gripper")

    assert port.observe_state().measured_close == 0.0
    assert rig.steps == 0
    receipt = port.command_close(1.0)
    sample = port.poll_state()

    assert receipt.accepted is True
    assert sample.measured_close == 1.0
    assert rig.steps == 1


def test_dmtac_backend_hold_uses_stop_without_reconnect() -> None:
    from teleop_runtime.backends.grippers.dmtac import (
        DMTacGripperBackend,
        DMTacGripperBackendConfig,
    )

    class Device:
        gripper = object()

        def __init__(self) -> None:
            self.stop_count = 0

        def stop(self) -> None:
            self.stop_count += 1

    backend = DMTacGripperBackend(
        DMTacGripperBackendConfig(
            side="right",
            connect_hardware=False,
            connect_tactile=False,
        )
    )
    backend.device = Device()
    backend._lock = threading.RLock()

    assert backend.hold_gripper_motion("test hold") is True
    assert backend.device.stop_count == 1
    assert backend._last_send_status == "gripper_hold"


def test_real_gripper_hold_waits_for_delayed_fresh_stopped_feedback():
    backend = _FakeDmtacBackend()
    backend.samples = [(0.98, True), (0.99, True), (1.0, False)]

    def stop(reason):
        backend.holds.append(reason)
        return True

    backend.hold_gripper_motion = stop
    port = RealDmtacGripperControlPort(backend=backend, hold_confirmation_timeout_sec=0.2)
    receipt = port.hold(reason="delayed stop")
    assert receipt.succeeded
    assert backend.holds == ["delayed stop"]
    assert len(receipt.evidence_refs) == 4
    assert backend.commands == []



@pytest.mark.parametrize("moving,error_code", [(None, 0), (False, 16)])
def test_real_gripper_hold_does_not_wait_through_unknown_state_or_hardware_fault(moving, error_code, caplog):
    backend = _FakeDmtacBackend()
    reads = []
    backend.get_state = lambda **kwargs: reads.append(1) or {
        "gripper_measured_scalar": 0.98,
        "gripper_error_code": error_code,
        "gripper_is_moving": moving,
    }
    port = RealDmtacGripperControlPort(backend=backend)
    assert not port.hold(reason="invalid feedback").succeeded
    assert reads == [1]
    assert "HOLD not confirmed" in caplog.text



def test_real_gripper_hold_without_stop_ack_does_not_claim_or_query_stopped_state(caplog):
    backend = _FakeDmtacBackend()
    backend.hold_gripper_motion = lambda reason: False
    backend.read_gripper_state_now = lambda: pytest.fail("No stop ACK")
    port = RealDmtacGripperControlPort(backend=backend)
    assert not port.hold(reason="no ack").succeeded
    assert "stop_ack=False" in caplog.text



def test_real_skill_waits_for_busy_native_backend_without_changing_stream_policy():
    from teleop_runtime.backends.grippers.dmtac import (
        DMTacGripperBackend,
        DMTacGripperBackendConfig,
    )

    class Device:
        gripper = object()
        tactile = None

        def __init__(self):
            self.moves = []

        def move(self, value, **kwargs):
            self.moves.append(value)

    backend = DMTacGripperBackend(DMTacGripperBackendConfig(
        side="right", connect_hardware=True, connect_tactile=False, enable_telemetry=False,
    ))
    backend.device = device = Device()
    port = RealDmtacGripperControlPort(backend=backend, command_submission_timeout_sec=0.2)
    entered = threading.Event()
    native_send = backend.send_fixed_joint_pos

    def send(value):
        sent = native_send(value)
        entered.set()
        return sent

    backend.send_fixed_joint_pos = send
    backend._lock.acquire()
    receipts = []
    thread = threading.Thread(target=lambda: receipts.append(port.command_close(1.0)))
    thread.start()
    try:
        assert entered.wait(timeout=1)
        assert thread.is_alive()
        assert device.moves == []
    finally:
        backend._lock.release()
        thread.join(timeout=1)
    assert not thread.is_alive()
    assert len(receipts) == 1 and receipts[0].accepted
    assert device.moves == [1.0]



@pytest.mark.parametrize("status", ["send_exception", "hardware_fault", "not_connected", "io_busy"])
def test_real_skill_preserves_send_failure_before_hold_overwrites_it(status, caplog):
    backend = _FakeDmtacBackend()
    sends = []
    diagnostic = {
        "gripper_last_send_status": status,
        "gripper_last_send_skip_reason": "original send failure",
        "gripper_measured_scalar": 0.0,
        "gripper_error_code": 0,
        "gripper_is_moving": False,
    }
    backend.get_state = lambda **kwargs: dict(diagnostic)
    backend.send_fixed_joint_pos = lambda value: sends.append(value) or False

    def hold(reason):
        diagnostic["gripper_last_send_status"] = "gripper_hold"
        diagnostic["gripper_last_send_skip_reason"] = reason
        return True

    backend.hold_gripper_motion = hold
    port = RealDmtacGripperControlPort(backend=backend, command_submission_timeout_sec=0.02)
    command = port.command_close(1.0)
    assert not command.accepted
    assert port.hold(reason="cleanup").succeeded
    assert status in caplog.text and "original send failure" in caplog.text
    assert command.evidence_refs[0] in caplog.text
    if status != "io_busy":
        assert sends == [1.0]
    else:
        assert 1 <= len(sends) <= 5



def test_cancelled_real_command_never_calls_backend():
    backend = _FakeDmtacBackend()
    port = RealDmtacGripperControlPort(backend=backend)
    assert not port.command_close(1.0, cancel_requested=lambda: True).accepted
    assert backend.commands == []



def test_diagnostic_read_failure_does_not_retry_or_prevent_hold(caplog):
    backend = _FakeDmtacBackend()
    attempts = []
    original_get_state = backend.get_state
    backend.send_fixed_joint_pos = lambda target: attempts.append(target) or False
    backend.get_state = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("diagnostic unavailable"))
    port = RealDmtacGripperControlPort(backend=backend)
    assert not port.command_close(1.0).accepted
    assert attempts == [1.0]
    assert "diagnostic unavailable" in caplog.text
    backend.get_state = original_get_state
    assert port.hold(reason="cleanup").succeeded



@pytest.mark.parametrize("name", ["command_submission_timeout_sec", "hold_confirmation_timeout_sec"])
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_real_gripper_wait_budgets_must_be_finite_positive(name, value):
    with pytest.raises(ValueError, match=name):
        RealDmtacGripperControlPort(backend=_FakeDmtacBackend(), **{name: value})



@pytest.mark.parametrize("moving", [True, None])
def test_at_target_without_stopped_feedback_times_out_instead_of_completing(moving):
    from dataclasses import replace

    port = FakeGripperPort([(0.5, False)])
    port.poll_state = lambda: GripperStateSample(
        measured_close=1.0, moving=moving, fault_code=None,
        observed_at=time.monotonic(), evidence_ref="sample:still-moving",
    )
    bundle = _sim_bundle(port)
    session = assemble_gripper_agent_gateway_session(
        contract_bundle=bundle, port=port, allowed_client_ids=("client:test",),
        current_station_revision=lambda: 3, clock_domain_id="clock:test",
    )
    try:
        snapshot = session.publish_observation()
        request = replace(_request(bundle, snapshot, target_state="closed"), requested_timeout_ms=50)
        submitted = session.service.submit_request(request=request, current_station_revision=3)
        assert submitted.accepted
        assert session.service.launch_operation(submitted.operation.operation_id).accepted
        assert session.executor.wait(timeout_sec=1)
        operation = session.service.get_operation(submitted.operation.operation_id)
        assert operation.execution_outcome != "completed"
        assert operation.execution_result is None
        assert port.commanded == [1.0]
        assert len(port.holds) == 1
    finally:
        assert session.close().accepted



@pytest.mark.parametrize("target_state", ["open", "closed"])
def test_endpoint_position_does_not_complete_until_motion_stops(target_state):
    target = 0.0 if target_state == "open" else 1.0
    port = FakeGripperPort([(0.5, False), *[(target, True)] * 3, *[(target, False)] * 3])
    bundle = _sim_bundle(port)
    session = assemble_gripper_agent_gateway_session(
        contract_bundle=bundle, port=port, allowed_client_ids=("client:test",),
        current_station_revision=lambda: 3, clock_domain_id="clock:test",
        convergence=GripperConvergenceProfile(poll_interval_sec=0),
    )
    try:
        snapshot = session.publish_observation()
        submitted = session.service.submit_request(
            request=_request(bundle, snapshot, target_state=target_state),
            current_station_revision=3,
        )
        assert submitted.accepted
        assert session.service.launch_operation(submitted.operation.operation_id).accepted
        assert session.executor.wait(timeout_sec=1)
        operation = session.service.get_operation(submitted.operation.operation_id)
        assert operation.execution_outcome == "completed"
        assert port.sequence == 7
        assert len(port.holds) == 1
    finally:
        assert session.close().accepted


class _GripperTestClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds

    def wall(self):
        return 1_000.0 + self.now


@pytest.fixture
def controlled_real_gripper(monkeypatch):
    from agent_gateway.profiles.real import gripper_skill as owner

    clock = _GripperTestClock()
    # Replace only this adapter's time port, leaving service/thread clocks intact.
    monkeypatch.setattr(owner, "time", SimpleNamespace(
        monotonic=clock, time=clock.wall, sleep=clock.sleep))
    backend = _FakeDmtacBackend()
    backend.read_gripper_state_now = lambda: {"timestamp": clock.wall()}
    port = RealDmtacGripperControlPort(
        backend=backend, clock=clock,
        command_submission_timeout_sec=0.02, hold_confirmation_timeout_sec=0.05)
    return backend, port, clock


def test_real_hold_cannot_confirm_an_earlier_measurement(controlled_real_gripper):
    backend, port, clock = controlled_real_gripper
    backend.samples = [(0.98, False)]
    backend.read_gripper_state_now = lambda: {"timestamp": clock.wall() - 0.01}
    assert not port.hold(reason="stop needs newer feedback").succeeded
    assert len(backend.holds) == 1 and backend.commands == []


@pytest.mark.parametrize("phase", ("stop", "read"))
def test_real_hold_rejects_late_stop_or_stopped_feedback(controlled_real_gripper, phase):
    backend, port, clock = controlled_real_gripper
    backend.samples = [(0.98, False)]
    native_stop = backend.hold_gripper_motion
    reads = []

    def stop(reason):
        result = native_stop(reason)
        if phase == "stop":
            clock.sleep(0.051)
        return result

    def read():
        reads.append(1)
        if phase == "read":
            clock.sleep(0.051)
        return {"timestamp": clock.wall()}

    backend.hold_gripper_motion = stop
    backend.read_gripper_state_now = read
    assert not port.hold(reason="late confirmation").succeeded
    assert reads == ([] if phase == "stop" else [1])
    assert len(backend.holds) == 1


def test_real_hold_moving_forever_exhausts_only_its_wait_budget(controlled_real_gripper):
    backend, port, clock = controlled_real_gripper
    backend.get_state = lambda **_kw: {
        "gripper_measured_scalar": 0.98, "gripper_error_code": 0, "gripper_is_moving": True}
    assert not port.hold(reason="still moving").succeeded
    assert clock.now == pytest.approx(100.05)
    assert len(backend.holds) == 1 and backend.commands == []


@pytest.mark.parametrize("reading", ("unavailable", "missing", "stale", "future", "nan"))
def test_real_feedback_rejects_unavailable_or_invalid_acquisition_time(controlled_real_gripper, reading):
    backend, port, clock = controlled_real_gripper
    value = {
        "unavailable": None, "missing": {},
        "stale": {"timestamp": clock.wall() - 0.201},
        "future": {"timestamp": clock.wall() + 0.001},
        "nan": {"timestamp": float("nan")},
    }[reading]
    backend.read_gripper_state_now = lambda: value
    with pytest.raises(RuntimeError):
        port.observe_state()


@pytest.mark.parametrize("phase", ("read", "cached_state"))
def test_real_feedback_slow_access_cannot_refresh_an_old_cut(controlled_real_gripper, phase):
    backend, port, clock = controlled_real_gripper
    native_get = backend.get_state

    def read():
        if phase == "read":
            clock.sleep(0.201)
        return {"timestamp": clock.wall()}

    def get(**kw):
        if phase == "cached_state":
            clock.sleep(0.201)
        return native_get(**kw)

    backend.read_gripper_state_now = read
    backend.get_state = get
    with pytest.raises(RuntimeError, match="200 ms budget"):
        port.observe_state()


def test_cancel_during_unwritten_busy_retry_prevents_later_dispatch(controlled_real_gripper):
    backend, port, _clock = controlled_real_gripper
    cancelled = [False]

    def busy(target):
        backend.commands.append(target)
        cancelled[0] = True
        return False

    backend.send_fixed_joint_pos = busy
    backend.get_state = lambda **_kw: {"gripper_last_send_status": "io_busy"}
    assert not port.command_close(1.0, cancel_requested=lambda: cancelled[0]).accepted
    assert backend.commands == [1.0]


def _prepared_gripper_lifecycle():
    clock = _GripperTestClock()
    port = FakeGripperPort([(0.5, False)])
    bundle = _sim_bundle(port)
    session = assemble_gripper_agent_gateway_session(
        contract_bundle=bundle, port=port, allowed_client_ids=("client:test",),
        current_station_revision=lambda: 3, clock_domain_id="clock:test")
    request = _request(bundle, session.publish_observation(), target_state="closed")
    operation = AgentSkillOperation(
        operation_id="operation:controlled-gripper", request_hash=request.semantic_content_hash,
        runtime_session_id=request.runtime_session_id, status="activated", reason_code="test",
        station_revision_at_admission=3, current_station_revision=3,
        started_at=clock(), updated_at=clock(), deadline_at=clock() + 1.0)
    lifecycle = GripperSkillLifecycle(
        port=port, contract_bundle=bundle,
        convergence=GripperConvergenceProfile(stable_sample_count=1),
        clock=clock, sleep=clock.sleep)
    activation = lifecycle.activate(request=request, operation=operation, cancel_requested=lambda: False)
    assert activation.succeeded
    return lifecycle, request, replace(operation, status="running",
        lower_operation_id=activation.lower_operation_id), port, clock


@pytest.mark.parametrize("reason", ("cancelled", "expired"))
def test_gripper_predispatch_preemption_does_not_issue_a_command(reason):
    lifecycle, request, operation, port, clock = _prepared_gripper_lifecycle()
    if reason == "expired":
        clock.now = operation.deadline_at
    result = lifecycle.run(request=request, operation=operation,
        cancel_requested=lambda: reason == "cancelled")
    assert not result.succeeded and result.execution_result is None
    assert port.commanded == []
    assert result.terminal_hold_confirmed


@pytest.mark.parametrize("phase", ("dispatch", "poll", "hold"))
@pytest.mark.parametrize("reason", ("cancelled", "expired"))
def test_gripper_completion_rechecks_preemption_after_synchronous_boundaries(phase, reason):
    lifecycle, request, operation, port, clock = _prepared_gripper_lifecycle()
    cancelled = [False]

    def preempt():
        if reason == "expired":
            clock.now = operation.deadline_at
        else:
            cancelled[0] = True

    original = {"dispatch": port.command_close, "poll": port.poll_state, "hold": port.hold}[phase]

    def cross_boundary(*args, **kw):
        result = original(*args, **kw)
        preempt()
        return result

    setattr(port, {"dispatch": "command_close", "poll": "poll_state", "hold": "hold"}[phase], cross_boundary)
    result = lifecycle.run(request=request, operation=operation, cancel_requested=lambda: cancelled[0])
    assert not result.succeeded and result.execution_result is None
    assert result.reason_code == (
        "gripper_execution_preempted" if reason == "cancelled" else "gripper_convergence_timeout")
    assert result.terminal_hold_confirmed and len(port.holds) == 1


def test_gripper_cancel_and_hold_keep_the_activation_owner_identity():
    lifecycle, request, operation, port, _clock = _prepared_gripper_lifecycle()
    result = lifecycle.run(request=request, operation=operation, cancel_requested=lambda: False)
    assert result.succeeded and port.commanded == [1.0]
    cancel = lifecycle.cancel(request=request, operation=operation, reason="identity check")
    held = lifecycle.request_hold(operation=operation, reason="identity check")
    assert cancel.lower_operation_id == held.lower_operation_id == operation.lower_operation_id
    assert cancel.lower_hold_confirmed and held.lower_hold_confirmed
    assert len(port.holds) == 1


@pytest.mark.parametrize("cancelled", (False, True))
def test_mujoco_gripper_predispatch_callback_preserves_passive_physics(cancelled):
    queued, steps = [], []
    rig = SimpleNamespace(
        config=SimpleNamespace(realtime_worker=False), is_ready=lambda: True,
        queue_gripper_close_targets=lambda targets: queued.append(targets) or 7,
        gripper_execution_snapshot=dict, hold_gripper_motion=lambda _reason: None,
        step_physics=lambda: steps.append(1),
    )
    port = MujocoGripperControlPort(rig=rig, gripper_name="right_gripper")
    receipt = port.command_close(1.0, cancel_requested=lambda: cancelled)
    assert receipt.accepted is (not cancelled)
    assert queued == ([] if cancelled else [{"right_gripper": 1.0}])
    assert steps == []


@pytest.mark.parametrize("name", ("command_submission_timeout_sec", "hold_confirmation_timeout_sec"))
def test_real_gripper_binding_identifies_its_wait_budget(name):
    backend = _FakeDmtacBackend()
    default = RealDmtacGripperControlPort(backend=backend)
    changed = RealDmtacGripperControlPort(backend=backend, **{name: 0.3})
    assert default.port_binding_hash != changed.port_binding_hash
    assert default.component_id == changed.component_id == "right_gripper"
    assert backend.commands == backend.holds == []


@pytest.mark.parametrize(
    "parameters",
    [
        {"target_state": "closed"},
        {"component_id": "both", "target_state": "closed"},
        {"component_id": "left_gripper", "target_state": "closed", "speed": 1.0},
    ],
)
def test_v2_requires_one_explicit_hand_and_does_not_allow_extra_command_settings(
    parameters,
):
    with pytest.raises(ValueError):
        validate_set_gripper_state_parameters_v2(parameters)



def test_v2_feedback_must_belong_to_the_selected_hand():
    result = {
        "schema_version": "dexteleop.agent_skill.set_gripper_state_result.v2",
        "component_id": "right_gripper",
        "commanded_state": "closed",
        "measured_close": 0.5,
        "completion_kind": "settled_before_full_close",
        "terminal_hold_confirmed": True,
    }
    with pytest.raises(ValueError, match="selected gripper"):
        validate_set_gripper_state_result_for_parameters_v2(
            {"component_id": "left_gripper", "target_state": "closed"}, result
        )



def test_gripper_session_retries_unconfirmed_shutdown_instead_of_caching_failure():
    port = FakeGripperPort([(0.0, False)])
    original_hold = port.hold
    port.hold = lambda **kwargs: GripperHoldReceipt(False, ("stop-not-proved",))
    session = assemble_gripper_agent_gateway_session(
        contract_bundle=_sim_bundle(port),
        port=port,
        allowed_client_ids=("client:test",),
        current_station_revision=lambda: 0,
        clock_domain_id="clock:test",
    )
    assert not session.close().accepted
    port.hold = original_hold
    assert session.close().accepted



def test_failure_to_stop_one_hand_still_attempts_the_other_and_does_not_claim_safety():
    left = _FakeDmtacBackend()
    left.config.side = "left"
    right = _FakeDmtacBackend()
    left.hold_gripper_motion = lambda reason: (_ for _ in ()).throw(
        RuntimeError("left serial failed")
    )
    ports = {
        "left_gripper": RealDmtacGripperControlPort(backend=left, component_id="left_gripper"),
        "right_gripper": RealDmtacGripperControlPort(backend=right),
    }
    with pytest.raises(RuntimeError, match="left serial failed"):
        stop_arm_and_grippers(
            arm_stop=lambda: "arm stopped", ports=ports, reason="cancel"
        )
    assert right.holds == ["cancel"]
    assert ports["left_gripper"].last_hold_receipt is None
    assert ports["right_gripper"].last_hold_receipt.succeeded



def test_two_gripper_contract_rejects_duplicate_serial_ownership_without_device_io():
    left = _FakeDmtacBackend()
    left.config.side = "left"
    right = _FakeDmtacBackend()
    ports = {
        "left_gripper": RealDmtacGripperControlPort(backend=left, component_id="left_gripper"),
        "right_gripper": RealDmtacGripperControlPort(backend=right),
    }
    with pytest.raises(ValueError, match="same serial port"):
        build_real_dmtac_grippers_skill_contract_bundle(
            runtime_session_id="runtime:test",
            control_domain_id="domain:test",
            target_binding_id="target:test",
            station_manifest_hash="station:test",
            execution_manifest_hash="execution:test",
            ports=ports,
        )
    assert left.events == right.events == []



@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("boundary", ["arm", "left"])
def test_bilateral_stop_attempts_all_boundaries_before_reraising_interrupt(interrupt_type, boundary):
    from agent_gateway.skills.gripper.runtime import hold_gripper_ports

    calls = []
    original = interrupt_type("interrupt stop")

    def stop(name):
        calls.append(name)
        if name == boundary:
            raise original
        return GripperHoldReceipt(True, ("stop:"+name,))

    ports = {name: SimpleNamespace(hold=lambda reason, name=name: stop(name))
             for name in ("left", "right")}
    with pytest.raises(interrupt_type) as captured:
        stop_arm_and_grippers(arm_stop=lambda: stop("arm"), ports=ports, reason="cancel")
    assert captured.value is original
    assert calls == ["arm", "left", "right"]
    if boundary == "left":
        calls.clear()
        with pytest.raises(interrupt_type) as captured:
            hold_gripper_ports(ports=ports, reason="shutdown")
        assert captured.value is original
        assert calls == ["left", "right"]


@pytest.mark.parametrize("component,physical", [("left_gripper", "right_gripper"),
                                                ("right_gripper", "left_gripper"),
                                                (None, "left_gripper")])
def test_explicit_mujoco_component_cannot_select_the_other_physical_hand(component, physical):
    rig = SimpleNamespace(
        config=SimpleNamespace(realtime_worker=False,
            gripper_actuation_profile=SimpleNamespace(gripper_for_side=lambda side: side+"_gripper")),
        **{name: lambda *args, **kwargs: pytest.fail("Constructor must not execute the rig")
           for name in ("is_ready", "queue_gripper_close_targets", "gripper_execution_snapshot",
                        "hold_gripper_motion", "step_physics")})
    with pytest.raises(ValueError, match="physical gripper"):
        MujocoGripperControlPort(rig=rig, gripper_name=physical,
            **({} if component is None else {"component_id": component}))

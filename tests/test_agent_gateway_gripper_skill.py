from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import pytest

from agent_gateway.profiles.mujoco.gripper_skill import MujocoGripperControlPort
from agent_gateway.profiles.real.gripper_skill import (
    RealDmtacGripperControlPort,
    build_real_dmtac_gripper_skill_contract_bundle,
)
from agent_gateway.protocol.contracts import AgentSkillRequest
from agent_gateway.skills.gripper.definition import (
    SET_GRIPPER_STATE_RESULT_SCHEMA_V1,
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
    GripperStateSample,
    assemble_gripper_agent_gateway_session,
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

    def command_close(self, target_close: float) -> GripperCommandReceipt:
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

    def is_ready(self) -> bool:
        return True

    def readiness_report(self):
        return {"ready": True, "reason": ""}

    def send_fixed_joint_pos(self, value):
        self.commands.append(float(value))
        return True

    def read_gripper_state_now(self):
        return {}

    def get_state(self, **_kwargs):
        close, moving = self.samples.pop(0) if self.samples else (0.98, False)
        return {
            "gripper_measured_scalar": close,
            "gripper_error_code": 0,
            "gripper_is_moving": moving,
        }

    def hold_gripper_motion(self, reason):
        self.holds.append(reason)
        return True


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

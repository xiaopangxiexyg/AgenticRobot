from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from agent_gateway.profiles.real.exclusive_application import (
    ExclusiveRealSkillApplication,
    ExclusiveRealSkillApplicationConfig,
    ExclusiveRealSkillApplicationState,
)
from agent_gateway.profiles.real.exclusive_factory import (
    assemble_exclusive_real_agent_primitive_deployment,
)
from agent_gateway.profiles.real.motion_primitive_profiles import (
    real_agent_motion_route_ids_v1,
)
from agent_gateway.profiles.real.observation_gateway import (
    RealTcpPoseObserveAttachmentSpec,
)
from agent_gateway.protocol.skill_catalog import SkillDeploymentSafetyEnvelopeRef
from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
    RigidFrameTransformSpec,
)
from robot_motion.coordinate_profiles import STATION_RETARGET_FRAME_CONTRACT_V1
from robot_motion.runtime import EffectorCommand, TeleopCommand, TeleopObservation
from teleop_runtime.deploy.dispatch import PositionDispatchNotReady
from teleop_runtime.skill_deploy import (
    RealAgentMotionPortConfig,
    RealSkillMotionDispatchReceipt,
    RealSkillMotionHoldReceipt,
    RealSkillMotionObservationCut,
)


class _TargetBuilder:
    def __init__(self) -> None:
        self.qpos_by_side = {
            "left": np.zeros(6),
            "right": np.zeros(6),
        }

    def synchronize(self, qpos_by_side) -> None:
        self.qpos_by_side = {
            side: np.asarray(value, dtype=float).copy()
            for side, value in qpos_by_side.items()
        }

    def current_tool_tcp_world_targets(self):
        return {"left": np.eye(4), "right": np.eye(4)}

    def solver_world_target_from_tool_tcp_world_target(
        self,
        *,
        side,
        tool_tcp_world_target,
    ):
        del side
        return np.asarray(tool_tcp_world_target, dtype=float).copy()

    def build_command_for_target_pair(self, target):
        command = TeleopCommand(timestamp=target.timestamp)
        command.add_command(
            EffectorCommand(
                effector_name="right_arm",
                side="right",
                command_type="joint_vector",
                value=np.zeros(6),
            )
        )
        return command

    def tool_tcp_world_targets_for_qpos_pair(self, *, left_qpos, right_qpos):
        del left_qpos, right_qpos
        return self.current_tool_tcp_world_targets()


def _assembly_inputs():
    execution_frame = CoordinateFrameSpec(frame_id="dual_arm_solver_world")
    station_frame = CoordinateFrameSpec(
        frame_id="canonical_station",
        semantic_parameters={"axis_contract": STATION_RETARGET_FRAME_CONTRACT_V1},
    )
    frame_graph = CoordinateFrameGraph(
        graph_id="exclusive-real-skill-test-graph",
        frames=(execution_frame, station_frame),
        transforms=(
            RigidFrameTransformSpec(
                transform_id="solver-world-to-station",
                source_frame_id=execution_frame.frame_id,
                target_frame_id=station_frame.frame_id,
                matrix_target_source=tuple(tuple(row) for row in np.eye(4)),
            ),
        ),
        graph_version="1",
    )
    attachment = RealTcpPoseObserveAttachmentSpec(
        runtime_session_id="real-skill-session:test",
        control_domain_id="real-skill-domain:test",
        embodiment_instance_id="dual-rm65:test",
        target_binding_id="real-skill-target:test",
        component_ids=("right_arm",),
        station_manifest_hash="station-manifest:test",
        execution_manifest_hash="execution-manifest:test",
        backend_inventory_hash="backend-inventory:test",
        source_binding_id="source-binding:test",
        source_binding_hash="source-binding-hash:test",
        observation_binding_id="observation-binding:test",
        observation_binding_hash="observation-binding-hash:test",
        observation_plan_hash="observation-plan-hash:test",
        observation_capability_hash="observation-capability-hash:test",
        pose_schema_hash="pose-schema-hash:test",
        reference_frame_id=execution_frame.frame_id,
        reference_frame_spec_hash=execution_frame.semantic_content_hash,
        frame_graph_hash=frame_graph.semantic_content_hash,
        tool_config_hash="tool-config-hash:test",
        calibration_hash="calibration-hash:test",
        clock_domain_id="real-owner-wall-clock",
        producer_epoch=1,
    )
    config = RealAgentMotionPortConfig(
        active_sides=("right",),
        component_id_by_side={"right": "right_arm"},
        execution_frame_id=execution_frame.frame_id,
        execution_frame_spec_hash=execution_frame.semantic_content_hash,
        exclusive_command_claim=True,
    )
    route_ids = real_agent_motion_route_ids_v1()
    envelope = SkillDeploymentSafetyEnvelopeRef(
        object_id="real-envelope:test",
        schema="dexteleop.deployment_safety_envelope.v1",
        content_hash="real-envelope-hash:test",
        constraint_set_hash="real-constraint-hash:test",
        allowed_route_ids=route_ids,
    )
    return frame_graph, station_frame, attachment, config, envelope


def _observation_cut() -> RealSkillMotionObservationCut:
    return RealSkillMotionObservationCut(
        observation=TeleopObservation(timestamp=2.0),
        qpos_by_side={"left": np.zeros(6), "right": np.zeros(6)},
        measured_at=1.9,
        available_at=1.95,
        evidence_refs=("hardware-qpos-cut:test",),
    )


def _fresh_observation_cut() -> RealSkillMotionObservationCut:
    now = time.time()
    return RealSkillMotionObservationCut(
        observation=TeleopObservation(timestamp=now),
        qpos_by_side={"left": np.zeros(6), "right": np.zeros(6)},
        measured_at=now - 0.001,
        available_at=now,
        evidence_refs=(f"hardware-qpos-cut:{now:.9f}",),
    )


class _Resources:
    def __init__(self) -> None:
        self.starts = 0
        self.closes = 0

    def start(self) -> None:
        self.starts += 1

    def close(self) -> None:
        self.closes += 1


class _Server:
    def __init__(self) -> None:
        self.shutdown_event = threading.Event()
        self.closed = 0

    def serve_forever(self, poll_interval=0.5) -> None:
        self.shutdown_event.wait(timeout=max(1.0, 4.0 * poll_interval))

    def shutdown(self) -> None:
        self.shutdown_event.set()

    def server_close(self) -> None:
        self.closed += 1


def _deployment(*, observation_provider, hold_reasons, gripper_port=None):
    frame_graph, station_frame, attachment, config, envelope = _assembly_inputs()
    target_builder = _TargetBuilder()
    return assemble_exclusive_real_agent_primitive_deployment(
        gripper_port=gripper_port,
        target_builder=target_builder,
        port_config=config,
        observation_attachment=attachment,
        frame_graph=frame_graph,
        primitive_reference_frame_id=station_frame.frame_id,
        safety_envelope_refs=(envelope,),
        allowed_client_ids=("rpent:test",),
        observation_provider=observation_provider,
        synchronize_target_builder=target_builder.synchronize,
        command_admission=lambda command, observation, now: True,
        command_dispatch=lambda command: RealSkillMotionDispatchReceipt(
            succeeded=True,
            evidence_refs=("rm65-dispatch:test",),
        ),
        hold_controller=lambda reason: (
            hold_reasons.append(reason)
            or RealSkillMotionHoldReceipt(
                succeeded=True,
                evidence_refs=("rm65-hold:test",),
            )
        ),
        clock=time.time,
        sleep=time.sleep,
    )


def test_composed_application_starts_hand_once_and_publishes_confirmed_hold():
    from agent_gateway.profiles.real.gripper_skill import (
        ExclusiveRealGripperResources,
        RealDmtacGripperControlPort,
    )
    from teleop_runtime.skill_deploy.motion_port import RealAgentMotionPhase
    from tests.test_agent_gateway_gripper_skill import _FakeDmtacBackend

    backend = _FakeDmtacBackend()
    backend.samples = [(0.0, False)]
    hand = RealDmtacGripperControlPort(backend=backend, clock=time.time)
    holds = []
    deployment = _deployment(
        observation_provider=_fresh_observation_cut,
        hold_reasons=holds,
        gripper_port=hand,
    )
    lower_resources = _Resources()
    resources = ExclusiveRealGripperResources(lower=lower_resources, port=hand)
    server = _Server()
    application = ExclusiveRealSkillApplication(
        deployment=deployment,
        resources=resources,
        startup_admission=lambda: "fixture:lower-startup-admission",
        server_factory=lambda: server,
    )
    try:
        application.start()
        snapshot = deployment.gateway_session.observation_producer.capture_and_publish(
            deployment.gateway_session.service
        ).snapshot
        assert snapshot.lifecycle_state == "hold"
        assert snapshot.execution_gate == "hold"
        assert (
            deployment.gateway_session.port.snapshot().phase
            == RealAgentMotionPhase.HELD
        )
        assert backend.commands == []
    finally:
        application.close()
    assert backend.events == ["hand_start", "hand_close"]
    assert lower_resources.starts == lower_resources.closes == 1


def test_composed_startup_failure_keeps_resources_when_hand_stop_is_unproved():
    from agent_gateway.profiles.real.gripper_skill import (
        ExclusiveRealGripperResources,
        RealDmtacGripperControlPort,
    )
    from tests.test_agent_gateway_gripper_skill import _FakeDmtacBackend

    backend = _FakeDmtacBackend()
    backend.samples = [(0.0, False)]
    hand = RealDmtacGripperControlPort(backend=backend, clock=time.time)
    deployment = _deployment(
        observation_provider=_fresh_observation_cut,
        hold_reasons=[],
        gripper_port=hand,
    )
    lower_resources = _Resources()
    resources = ExclusiveRealGripperResources(lower=lower_resources, port=hand)
    original_stop = backend.hold_gripper_motion
    backend.hold_gripper_motion = lambda reason: False

    def reject_startup():
        raise RuntimeError("injected admission failure")

    def no_server():
        raise AssertionError("Failed startup must not expose the transport.")

    application = ExclusiveRealSkillApplication(
        deployment=deployment,
        resources=resources,
        startup_admission=reject_startup,
        server_factory=no_server,
    )
    with pytest.raises(RuntimeError, match="injected admission failure.*startup_close"):
        application.start()
    assert application.state == ExclusiveRealSkillApplicationState.FAULT
    assert lower_resources.closes == 0
    assert backend.events == ["hand_start"]
    assert backend.commands == []
    backend.hold_gripper_motion = original_stop
    application.close()
    assert application.state == ExclusiveRealSkillApplicationState.CLOSED
    assert backend.events == ["hand_start", "hand_close"]
    assert lower_resources.closes == 1


def test_factory_assembles_one_gateway_and_independent_lower_owner() -> None:
    frame_graph, station_frame, attachment, config, envelope = _assembly_inputs()
    target_builder = _TargetBuilder()
    deployment = assemble_exclusive_real_agent_primitive_deployment(
        target_builder=target_builder,
        port_config=config,
        observation_attachment=attachment,
        frame_graph=frame_graph,
        primitive_reference_frame_id=station_frame.frame_id,
        safety_envelope_refs=(envelope,),
        allowed_client_ids=("rpent:test",),
        observation_provider=_observation_cut,
        synchronize_target_builder=target_builder.synchronize,
        command_admission=lambda command, observation, now: True,
        command_dispatch=lambda command: RealSkillMotionDispatchReceipt(
            succeeded=True,
            evidence_refs=("rm65-dispatch:test",),
        ),
        hold_controller=lambda reason: RealSkillMotionHoldReceipt(
            succeeded=True,
            evidence_refs=("rm65-hold:test",),
        ),
        clock=lambda: 2.0,
        sleep=lambda duration: None,
    )

    assert (
        deployment.control_runtime.motion_port
        is deployment.gateway_session.port
    )
    deployment.control_runtime.start()
    deployment.control_runtime.step(now=2.0)
    snapshot = deployment.gateway_session.observation_producer.capture_snapshot(
        active_operation_id=None,
    )

    assert snapshot.runtime_session_id == attachment.runtime_session_id
    assert snapshot.lifecycle_state == "running"
    assert snapshot.execution_gate == "ready"
    assert snapshot.measurement_valid is True


def test_exclusive_factory_import_has_no_realteleop_or_composition_dependency() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import json
import sys
import agent_gateway.profiles.real.exclusive_factory
import agent_gateway.profiles.real.exclusive_application
print(json.dumps(sorted(
    name for name in sys.modules
    if name == 'control_composition'
    or name.startswith('control_composition.')
    or name == 'teleop_runtime.real_teleop'
    or name.startswith('teleop_runtime.real_teleop.')
)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_application_owns_control_loop_and_holds_before_resource_close() -> None:
    holds: list[str] = []
    resources = _Resources()
    server = _Server()
    deployment = _deployment(
        observation_provider=_fresh_observation_cut,
        hold_reasons=holds,
    )
    application = ExclusiveRealSkillApplication(
        deployment=deployment,
        resources=resources,
        startup_admission=lambda: {"startup": "admitted"},
        server_factory=lambda: server,
        config=ExclusiveRealSkillApplicationConfig(control_hz=100.0),
    )

    assert application.control_fault_reason == "application_not_running:created"
    application.start()
    assert server.control_fault_reader() is None
    time.sleep(0.03)
    application.close()

    assert application.state == ExclusiveRealSkillApplicationState.CLOSED
    assert server.control_fault_reader() == "application_not_running:closed"
    assert resources.starts == 1
    assert resources.closes == 1
    assert application.startup_admission_receipt == {"startup": "admitted"}
    assert server.closed == 1
    assert holds
    assert deployment.control_runtime.last_hold_receipt is not None
    assert deployment.control_runtime.last_hold_receipt.succeeded is True


@pytest.mark.parametrize("message", ["", " \n\t ", "a" * 497 + " " + "tail" * 20])
def test_application_fault_text_satisfies_health_contract(message):
    holds = []
    deployment = _deployment(observation_provider=_fresh_observation_cut, hold_reasons=holds)
    application = ExclusiveRealSkillApplication(deployment=deployment, resources=_Resources(),
        startup_admission=lambda: {}, server_factory=_Server)
    application._control_fault = RuntimeError(message)
    reason = application.control_fault_reason
    assert reason.startswith("RuntimeError") and reason == reason.strip()
    assert 0 < len(reason) <= 512
    if not message.strip():
        assert reason == "RuntimeError"
    assert holds == []


def test_application_fault_reader_binding_failure_closes_the_same_server():
    class BoundServer(_Server):
        @property
        def control_fault_reader(self):
            return lambda: "another-owner"

        @control_fault_reader.setter
        def control_fault_reader(self, value):
            raise RuntimeError("control fault reader is already bound")

    resources = _Resources()
    server = BoundServer()
    holds = []
    deployment = _deployment(observation_provider=_fresh_observation_cut, hold_reasons=holds)
    application = ExclusiveRealSkillApplication(deployment=deployment, resources=resources,
        startup_admission=lambda: {"startup": "admitted"}, server_factory=lambda: server)
    with pytest.raises(RuntimeError, match="already bound"):
        application.start()
    assert application.state == ExclusiveRealSkillApplicationState.FAULT
    assert server.closed == 1 and resources.starts == resources.closes == 1
    assert holds
    application.close()
    assert server.closed == 1 and resources.closes == 1


def test_application_close_failure_is_retryable_and_retains_resources(
    monkeypatch,
) -> None:
    holds: list[str] = []
    resources = _Resources()
    server = _Server()
    deployment = _deployment(
        observation_provider=_fresh_observation_cut,
        hold_reasons=holds,
    )
    application = ExclusiveRealSkillApplication(
        deployment=deployment,
        resources=resources,
        startup_admission=lambda: {"startup": "admitted"},
        server_factory=lambda: server,
        config=ExclusiveRealSkillApplicationConfig(control_hz=100.0),
    )
    original_close = deployment.control_runtime.close
    close_calls = 0

    def fail_once() -> None:
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise RuntimeError("injected lower HOLD failure")
        original_close()

    monkeypatch.setattr(deployment.control_runtime, "close", fail_once)
    application.start()
    time.sleep(0.03)

    with pytest.raises(RuntimeError, match="injected lower HOLD failure"):
        application.close()

    assert application.state == ExclusiveRealSkillApplicationState.FAULT
    assert resources.closes == 0
    assert server.closed == 1

    application.close()

    assert close_calls == 2
    assert application.state == ExclusiveRealSkillApplicationState.CLOSED
    assert resources.closes == 1
    assert server.closed == 1


def test_application_control_fault_reaches_hold_without_another_read() -> None:
    calls = 0

    def fail_after_initial_cut():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("dual RM65 state read failed")
        return _fresh_observation_cut()

    holds: list[str] = []
    resources = _Resources()
    deployment = _deployment(
        observation_provider=fail_after_initial_cut,
        hold_reasons=holds,
    )
    application = ExclusiveRealSkillApplication(
        deployment=deployment,
        resources=resources,
        startup_admission=lambda: {"startup": "admitted"},
        server_factory=_Server,
        config=ExclusiveRealSkillApplicationConfig(control_hz=100.0),
    )

    application.start()
    deadline = time.monotonic() + 1.0
    while (
        application.state != ExclusiveRealSkillApplicationState.FAULT
        and time.monotonic() < deadline
    ):
        time.sleep(0.005)

    assert application.state == ExclusiveRealSkillApplicationState.FAULT
    assert isinstance(application.control_fault, RuntimeError)
    count_before_health = calls
    assert "dual RM65 state read failed" in application.control_fault_reason
    assert calls == count_before_health
    assert holds
    assert deployment.control_runtime.last_hold_receipt is not None
    assert deployment.control_runtime.last_hold_receipt.succeeded is True
    application.close()


def test_application_fails_closed_before_server_when_startup_admission_fails() -> None:
    holds: list[str] = []
    resources = _Resources()
    server_built = False
    deployment = _deployment(
        observation_provider=_fresh_observation_cut,
        hold_reasons=holds,
    )

    def reject_startup():
        raise RuntimeError("target safety anchor unavailable")

    def build_server():
        nonlocal server_built
        server_built = True
        return _Server()

    application = ExclusiveRealSkillApplication(
        deployment=deployment,
        resources=resources,
        startup_admission=reject_startup,
        server_factory=build_server,
    )

    with pytest.raises(RuntimeError, match="target safety anchor unavailable"):
        application.start()

    assert application.state == ExclusiveRealSkillApplicationState.FAULT
    assert application.startup_admission_receipt is None
    assert server_built is False
    assert resources.starts == 1
    assert resources.closes == 1
    assert len(holds) == 1
    assert "exclusive_skill_application_start_failed" in holds[0]


def test_application_ready_hook_and_shutdown_request_preserve_close_owner() -> None:
    holds: list[str] = []
    resources = _Resources()
    server = _Server()
    deployment = _deployment(
        observation_provider=_fresh_observation_cut,
        hold_reasons=holds,
    )
    application = ExclusiveRealSkillApplication(
        deployment=deployment,
        resources=resources,
        startup_admission=lambda: {"startup": "admitted"},
        server_factory=lambda: server,
        config=ExclusiveRealSkillApplicationConfig(control_hz=100.0),
    )
    ready = threading.Event()
    thread = threading.Thread(
        target=lambda: application.serve_forever(ready_callback=ready.set),
    )

    thread.start()
    assert ready.wait(timeout=1.0)
    assert application.request_shutdown() is True
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert application.state == ExclusiveRealSkillApplicationState.CLOSED
    assert deployment.control_runtime.last_hold_receipt is not None
    assert deployment.control_runtime.last_hold_receipt.succeeded is True
    assert resources.closes == 1
    assert holds


@pytest.mark.parametrize("error_type,expected_prefix", [
    (PositionDispatchNotReady, "lower_backend_not_ready"),
    (RuntimeError, "exclusive_skill_control_cycle_failed"),
])
def test_control_fault_retains_typed_cause_without_misclassifying_other_faults(
    monkeypatch, error_type, expected_prefix,
):
    deployment = _deployment(observation_provider=_fresh_observation_cut, hold_reasons=[])
    application = ExclusiveRealSkillApplication(
        deployment=deployment, resources=_Resources(),
        startup_admission=lambda: {"startup": "admitted"}, server_factory=_Server,
    )
    holds = []

    def failed_step():
        if error_type is PositionDispatchNotReady:
            raise error_type("native cause with\nnew line", health={})
        raise error_type("native cause with\nnew line")

    monkeypatch.setattr(deployment.control_runtime, "step", failed_step)
    monkeypatch.setattr(deployment.control_runtime, "fail_closed", holds.append)
    application._run_control_loop()
    assert holds == [expected_prefix + ": native cause with new line"]
    assert isinstance(application.control_fault, error_type)


@pytest.mark.parametrize("retained", [False, None, "raises"])
def test_failed_transport_fence_keeps_original_shutdown_path(monkeypatch, retained):
    deployment = _deployment(observation_provider=_fresh_observation_cut, hold_reasons=[])
    server = _Server()

    def retain(reason):
        if retained == "raises":
            raise RuntimeError("injected transport fence failure")
        return retained

    server.retain_control_fault = retain
    application = ExclusiveRealSkillApplication(
        deployment=deployment, resources=_Resources(), startup_admission=lambda: "fixture",
        server_factory=lambda: server)
    application._server = server
    application._serving.set()
    monkeypatch.setattr(deployment.control_runtime, "step", lambda: (_ for _ in ()).throw(RuntimeError("fault")))
    monkeypatch.setattr(deployment.control_runtime, "fail_closed", lambda reason: None)
    application._run_control_loop()
    assert server.shutdown_event.is_set()
    assert application.state == ExclusiveRealSkillApplicationState.FAULT


def test_native_unix_application_retains_fault_queries_and_fences_new_work(
    monkeypatch, short_unix_socket_path_factory,
):
    from tests.test_agent_gateway_unix_transport import _raw_call, _wait_for
    from teleop_runtime.skill_deploy.motion_port import RealAgentMotionPhase

    holds = []
    resources = _Resources()
    deployment = _deployment(observation_provider=_fresh_observation_cut, hold_reasons=holds)
    path = short_unix_socket_path_factory("fault")
    server = deployment.gateway_session.build_unix_json_server(
        socket_path=str(path), client_id_by_uid={os.getuid(): "rpent:test"})
    application = ExclusiveRealSkillApplication(
        deployment=deployment, resources=resources, startup_admission=lambda: "fixture:admitted",
        server_factory=lambda: server)
    step = deployment.control_runtime.step
    inject = threading.Event()

    def maybe_fault():
        if inject.is_set():
            raise PositionDispatchNotReady("injected SDK send fault", health={})
        return step()

    monkeypatch.setattr(deployment.control_runtime, "step", maybe_fault)
    ready, failures = threading.Event(), []

    def serve():
        try:
            application.serve_forever(ready_callback=ready.set)
        except Exception as exc:
            failures.append(exc)

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        assert ready.wait(2)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(2)
            sock.connect(str(path))
            health = _raw_call(sock, request_id="before", method="healthz", payload=None)
            assert health["result"]["control_fault_reason"] is None
            inject.set()
            _wait_for(lambda: application.state == ExclusiveRealSkillApplicationState.FAULT)
            assert thread.is_alive() and resources.closes == 0
            assert deployment.gateway_session.port.snapshot().phase == RealAgentMotionPhase.HELD
            health = _raw_call(sock, request_id="after", method="healthz", payload=None)
            assert health["result"]["control_fault_reason"] == "lower_backend_not_ready"
            assert health["result"]["domain_control_state"]["emergency_stop_confirmed"] is False
            assert _raw_call(sock, request_id="catalog", method="get_skill_route_manifest", payload=None)["ok"]
            for method in ("submit_skill_request", "submit_read_only_skill_request"):
                response = _raw_call(sock, request_id=method, method=method, payload={})
                assert response["error"]["code"] == "control_fault_latched"
            manifest = deployment.gateway_session.service.manifest
            query = {"client_id": "rpent:test", "query_id": "fault-query",
                     "runtime_session_id": manifest.runtime_session_id,
                     "skill_route_manifest_hash": manifest.semantic_content_hash,
                     "operation_id": "missing-operation"}
            # A negative ownership query still returns normally on the same channel.
            assert _raw_call(sock, request_id="query", method="get_own_operation", payload=query)["ok"]
            assert _raw_call(sock, request_id="still", method="healthz", payload=None)["ok"]
            assert deployment.gateway_session.port.snapshot().dispatch_count == 0
    finally:
        application.request_shutdown()
        thread.join(3)
        if thread.is_alive():
            application.close()
            thread.join(3)
    assert not thread.is_alive()
    assert resources.closes == 1 and holds
    assert len(failures) == 1 and "control loop failed" in str(failures[0])


@pytest.mark.parametrize("hand_stop_proved", [True, False])
def test_arm_fault_still_closes_composed_hand_with_an_accepted_command(
    monkeypatch, hand_stop_proved,
):
    from agent_gateway.profiles.real.gripper_skill import ExclusiveRealGripperResources, RealDmtacGripperControlPort
    from tests.test_agent_gateway_gripper_skill import _FakeDmtacBackend

    backend = _FakeDmtacBackend()
    backend.samples = [(0., False)]
    hand = RealDmtacGripperControlPort(backend=backend, clock=time.time)
    lower = _Resources()
    deployment = _deployment(observation_provider=_fresh_observation_cut, hold_reasons=[], gripper_port=hand)
    server, retain_calls = _Server(), []
    server.retain_control_fault = lambda reason: retain_calls.append(reason) or True
    app = ExclusiveRealSkillApplication(deployment=deployment,
        resources=ExclusiveRealGripperResources(lower=lower, port=hand),
        startup_admission=lambda: "fixture:admitted", server_factory=lambda: server)
    inject, ready = threading.Event(), threading.Event()
    failures = []
    step = deployment.control_runtime.step

    def maybe_fault():
        if inject.is_set():
            raise RuntimeError("injected arm failure")
        return step()

    monkeypatch.setattr(deployment.control_runtime, "step", maybe_fault)

    def serve():
        try:
            app.serve_forever(ready_callback=ready.set)
        except Exception as error:
            failures.append(error)

    thread = threading.Thread(target=serve)
    thread.start()
    real_stop = backend.hold_gripper_motion
    try:
        assert ready.wait(2)
        assert hand.command_close(.5).accepted
        assert backend.commands == [.5] and not backend.held
        if not hand_stop_proved:
            backend.hold_gripper_motion = lambda reason: backend.holds.append(reason) or False
        inject.set()
        thread.join(3)
        assert not thread.is_alive() and failures
        assert retain_calls == [] and server.shutdown_event.is_set()
        assert backend.holds
        if hand_stop_proved:
            assert backend.held and lower.closes == 1
        else:
            assert app.state == ExclusiveRealSkillApplicationState.FAULT
            assert lower.closes == 0 and "hand_close" not in backend.events
    finally:
        backend.hold_gripper_motion = real_stop
        app.request_shutdown()
        thread.join(3)
        app.close()
    assert lower.closes == 1 and "hand_close" in backend.events


@pytest.mark.parametrize("with_grippers", [False, True])
@pytest.mark.parametrize("with_udp", [False, True])
@pytest.mark.parametrize("guard_failures", [3, 4])
@pytest.mark.parametrize("entry", ["start", "serve_forever"])
def test_application_partial_resource_start_failure_retries_the_same_owner(with_grippers, with_udp, guard_failures, entry):
    from types import SimpleNamespace
    from agent_gateway.profiles.real.gripper_skill import ExclusiveRealGripperResources, RealDmtacGripperControlPort
    from teleop_runtime.skill_deploy.control_runtime import RealSkillControlLifecycle
    from tests.test_agent_gateway_gripper_skill import _FakeDmtacBackend
    from tests.test_real_skill_deploy_rm65_ports import _ports

    lower, backends, _ = _ports(force_guard_ready=False, force_guard_close_failures=guard_failures)
    guard = lower.position_force_guard
    receiver_closes = []
    receiver = SimpleNamespace(start=lambda: None, wait_ready=lambda **kwargs: None,
        close=lambda: receiver_closes.append("closed"), diagnostics=lambda: {"transport": "udp"})
    if with_udp:
        lower.udp_feedback = receiver
    hands = {}
    if with_grippers:
        backend = _FakeDmtacBackend()
        hands["right_gripper"] = RealDmtacGripperControlPort(backend=backend)
    resources = ExclusiveRealGripperResources(lower=lower, port=hands["right_gripper"]) if with_grippers else lower
    holds = []
    deployment = _deployment(observation_provider=_fresh_observation_cut, hold_reasons=holds,
        gripper_port=hands.get("right_gripper"))
    def no_server():
        pytest.fail("partial startup must never publish transport")
    app = ExclusiveRealSkillApplication(deployment=deployment, resources=resources,
        startup_admission=lambda: pytest.fail("partial startup must not reach admission"),
        server_factory=no_server)
    with pytest.raises(RuntimeError, match="startup rollback incomplete"):
        getattr(app, entry)()
    assert app.state == ExclusiveRealSkillApplicationState.FAULT
    assert lower.resources_closed is False
    assert guard.closes >= 2  # The application retains and retries partial acquisition.
    assert all(backend.closes == 0 for backend in backends.values())
    assert receiver_closes == []
    assert app.resources is resources
    assert lower.position_force_guard is guard
    assert deployment.control_runtime.lifecycle == RealSkillControlLifecycle.CREATED
    assert deployment.control_runtime.last_hold_receipt is None
    assert holds == []
    with pytest.raises(RuntimeError, match="starts once"):
        app.start()
    while guard.closes < guard_failures:
        with pytest.raises(RuntimeError, match="force_guard"):
            app.close()
        assert app.state == ExclusiveRealSkillApplicationState.FAULT
        assert all(backend.closes == 0 for backend in backends.values())
        assert receiver_closes == []
    app.close()
    app.close()
    assert app.state == ExclusiveRealSkillApplicationState.CLOSED
    assert lower.resources_closed is True
    assert guard.starts == 1 and guard.closes == guard_failures + 1
    assert all(backend.closes == 1 for backend in backends.values())
    assert all(len(backend.holds) == 2 for backend in backends.values())
    assert receiver_closes == (["closed"] if with_udp else [])
    assert holds == []
    assert deployment.control_runtime.last_hold_receipt is None
    assert all(port.backend.events == port.backend.commands == [] for port in hands.values())



def test_application_close_waits_for_the_inflight_resource_start():
    entered, release, start_returned = threading.Event(), threading.Event(), threading.Event()
    close_entered, close_finished = threading.Event(), threading.Event()
    class Resources(_Resources):
        def start(self):
            entered.set()
            assert release.wait(3.)
            super().start()
            start_returned.set()
        def close(self):
            assert start_returned.is_set()
            super().close()
    resources = Resources()
    deployment = _deployment(observation_provider=_fresh_observation_cut, hold_reasons=[])
    app = ExclusiveRealSkillApplication(deployment=deployment, resources=resources,
        startup_admission=lambda: {"startup": "admitted"}, server_factory=_Server)
    errors = []
    def start():
        try:
            app.start()
        except BaseException as error:
            errors.append(error)
    def close():
        close_entered.set()
        try:
            app.close()
        except BaseException as error:
            errors.append(error)
        finally:
            close_finished.set()
    starter, closer = threading.Thread(target=start), threading.Thread(target=close)
    starter.start()
    try:
        assert entered.wait(2.)
        closer.start()
        assert close_entered.wait(2.)
        assert not close_finished.wait(.05)
        assert resources.closes == 0
    finally:
        release.set()
        starter.join(timeout=5.)
        if closer.ident is not None:
            closer.join(timeout=5.)
        if not starter.is_alive() and not closer.is_alive():
            app.close()
    assert not starter.is_alive() and not closer.is_alive()
    assert errors == []
    assert app.state == ExclusiveRealSkillApplicationState.CLOSED
    assert resources.starts == resources.closes == 1

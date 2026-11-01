"""Hardware generations use real application shutdown with fake device ports."""

from dataclasses import replace
from types import SimpleNamespace
import threading
import time
import os
import socket

import pytest

from agent_gateway.profiles.real.exclusive_application import ExclusiveRealSkillApplication
from agent_gateway.profiles.real.exclusive_factory import assemble_exclusive_real_agent_primitive_deployment
from agent_gateway.profiles.real.gripper_skill import ExclusiveRealGripperResources, RealDmtacGripperControlPort
from agent_gateway.profiles.real.hardware_connection import (
    GenerationServer, RealHardwareConnectionOwner, ResidentPerceptionResources,
)
from agent_gateway.profiles.real.realsense_perception import ExclusiveRealPerceptionResources
from teleop_runtime.skill_deploy import RealSkillMotionDispatchReceipt, RealSkillMotionHoldReceipt
from teleop_runtime.skill_deploy.control_runtime import RealSkillMotionEmergencyStopReceipt
from tests.test_agent_gateway_gripper_skill import _FakeDmtacBackend
from tests.test_agent_gateway_real_exclusive_factory import _assembly_inputs, _TargetBuilder, _fresh_observation_cut
from tests.test_agent_gateway_unix_transport import _raw_call, _wait_for
from agent_gateway.transport.unix_transport import AgentGatewayUnixJsonServer, AgentGatewayUnixPeerClientMap


class Resource:
    def __init__(self):
        self.starts = self.closes = 0

    def start(self):
        self.starts += 1

    def close(self):
        self.closes += 1


def harness(monkeypatch):
    cameras, model = Resource(), Resource()
    monkeypatch.setattr("agent_gateway.profiles.real.realsense_perception.start_realsense_sources", lambda sources: cameras.start())
    monkeypatch.setattr("agent_gateway.profiles.real.realsense_perception._formal_close_camera_sources", lambda sources: cameras.close())
    resident = ResidentPerceptionResources()
    resident.source_binding = SimpleNamespace(camera_sources={"main": cameras})
    lease = resident.model(model)
    generations = []

    def build():
        graph, frame, attachment, config, envelope = _assembly_inputs()
        attachment = replace(attachment, runtime_session_id=f"runtime:generation-{len(generations)}")
        hands = {}
        for side in ("left", "right"):
            backend = _FakeDmtacBackend()
            backend.config.side = side
            backend.config.gripper_port = f"/dev/fake-{side}"
            backend.samples = [(0.0, False)]
            hands[f"{side}_gripper"] = RealDmtacGripperControlPort(backend=backend, component_id=f"{side}_gripper", clock=time.time)
        builder = _TargetBuilder()
        holds = []
        stops = []
        deployment = assemble_exclusive_real_agent_primitive_deployment(
            gripper_ports=hands, target_builder=builder, port_config=config,
            observation_attachment=attachment, frame_graph=graph,
            primitive_reference_frame_id=frame.frame_id, safety_envelope_refs=(envelope,),
            allowed_client_ids=("operator",), observation_provider=_fresh_observation_cut,
            synchronize_target_builder=builder.synchronize,
            command_admission=lambda *args: True,
            command_dispatch=lambda *args: RealSkillMotionDispatchReceipt(succeeded=True, evidence_refs=("dispatch:test",)),
            hold_controller=lambda reason: holds.append(reason) or RealSkillMotionHoldReceipt(succeeded=True, evidence_refs=("hold:test",)),
            emergency_stop_controller=lambda reason: stops.append(reason) or RealSkillMotionEmergencyStopReceipt(
                succeeded=True, evidence_refs=("stop:left", "stop:right"),
                status_by_side={"left": "confirmed", "right": "confirmed"}),
        )
        arms = {side: Resource() for side in ("left", "right")}
        class ArmResources:
            def start(self):
                for resource in arms.values():
                    resource.start()

            def close(self):
                for resource in arms.values():
                    resource.close()
        perception = ExclusiveRealPerceptionResources(lower=ArmResources(),
            camera_sources=resident.source_binding.camera_sources,
            additional_resources=(lease,), start_cameras=resident.start_cameras,
            admit_cameras=lambda sources: None, close_cameras=resident.retain_cameras)
        resources = ExclusiveRealGripperResources(lower=perception, ports=hands)
        app = ExclusiveRealSkillApplication(deployment=deployment, resources=resources,
            startup_admission=lambda: "lower:admitted", server_factory=GenerationServer)
        result = SimpleNamespace(application=app, deployment=deployment,
            lower=SimpleNamespace(ports=SimpleNamespace(last_emergency_stop_receipt=None)),
            authorization=SimpleNamespace(grant_resolver=lambda _: None), hands=hands, arms=arms, holds=holds, stops=stops)
        generations.append(result)
        return result
    owner = RealHardwareConnectionOwner(initial=build(), rebuild=build, resident=resident,
        operator_client_ids=frozenset({"operator"}), server_factory=lambda _: None)
    return owner, generations, cameras, model


def settle(owner):
    assert owner._transition.join_and_confirm(timeout=3.0)
    return owner.snapshot("operator")


def test_release_and_reconnect_replace_control_owners_but_retain_cameras_and_models(monkeypatch):
    owner, generations, cameras, model = harness(monkeypatch)
    first = owner.current
    first.application.start()
    initial = owner.snapshot("operator")
    old_facade = owner.facade("operator")
    try:
        owner.request("operator", {"action": "disconnect", "expected_generation": 0})
        assert settle(owner)["state"] == "disconnected"
        assert first.application.state.value == "closed"
        assert all(resource.starts == resource.closes == 1 for resource in first.arms.values())
        assert all(port.backend.events == ["hand_start", "hand_close"] for port in first.hands.values())
        assert first.holds
        held_before = list(first.holds)
        hold = first.deployment.gateway_session.service.request_hold(
            reason="late-upper-cancel", current_station_revision=0)
        assert hold.accepted and hold.port_receipt.proves_motion_or_hold
        assert first.holds == held_before
        assert owner.facade("operator", "request_hold_own_domain", old_facade) is old_facade
        assert cameras.starts == model.starts == 1
        assert cameras.closes == model.closes == 0
        with pytest.raises(RuntimeError, match="disconnected"):
            owner.facade("operator", "submit_skill_request")
        owner.request("operator", {"action": "reconnect", "expected_generation": 0})
        state = settle(owner)
        assert state["state"] == "connected" and state["generation"] == 1
        assert state["runtime_session_id"] != initial["runtime_session_id"]
        assert state["route_manifest_hash"] != initial["route_manifest_hash"]
        assert owner.current is not first and len(generations) == 2
        assert owner.facade("operator", "get_own_operation", old_facade) is old_facade
        assert owner.facade("operator", "healthz", old_facade) is old_facade
        with pytest.raises(RuntimeError, match="generation changed"):
            owner.facade("operator", "submit_skill_request", old_facade)
        assert owner.facade("operator", "get_skill_definition_catalog", old_facade).service is owner.current.deployment.gateway_session.service
        assert cameras.starts == model.starts == 1
        assert cameras.closes == model.closes == 0
        assert all(not port.backend.commands for gen in generations for port in gen.hands.values())
        with pytest.raises(RuntimeError, match="binding changed"):
            owner.request("operator", {"action": "disconnect", "expected_generation": 0})
    finally:
        owner.close()
    assert cameras.closes == model.closes == 1
    assert all(resource.closes == 1 for resource in owner.current.arms.values())


def test_incomplete_release_retains_owner_and_blocks_replacement(monkeypatch):
    owner, generations, cameras, model = harness(monkeypatch)
    owner.current.application.start()
    arm = owner.current.arms["right"]
    close = arm.close
    arm.close = lambda: (_ for _ in ()).throw(RuntimeError("arm close incomplete"))
    try:
        owner.request("operator", {"action": "disconnect", "expected_generation": 0})
        state = settle(owner)
        assert state["state"] == "faulted" and "close incomplete" in state["error"]
        assert not state["reconnect_allowed"] and len(generations) == 1
        assert cameras.closes == model.closes == 0
        with pytest.raises(RuntimeError, match="complete release"):
            owner.request("operator", {"action": "reconnect", "expected_generation": 0})
        arm.close = close
        owner.request("operator", {"action": "disconnect", "expected_generation": 0})
        assert settle(owner)["state"] == "disconnected"
    finally:
        arm.close = close
        owner.close()


def test_double_click_and_shutdown_wait_for_the_same_release_worker(monkeypatch):
    owner, generations, cameras, model = harness(monkeypatch)
    owner.current.application.start()
    entered, release = threading.Event(), threading.Event()
    close = owner.current.application.close
    def blocked():
        entered.set()
        assert release.wait(3)
        close()
    owner.current.application.close = blocked
    try:
        owner.request("operator", {"action": "disconnect", "expected_generation": 0})
        assert entered.wait(1)
        assert owner.snapshot("operator")["state"] == "disconnecting"
        with pytest.raises(RuntimeError, match="already pending"):
            owner.request("operator", {"action": "disconnect", "expected_generation": 0})
        with pytest.raises(PermissionError):
            owner.request("readonly", {"action": "disconnect", "expected_generation": 0})
    finally:
        release.set()
        settle(owner)
        owner.current.application.close = close
        owner.close()
    assert len(generations) == 1


def test_emergency_stop_cannot_be_cleared_through_reconnect(monkeypatch):
    owner, generations, _, _ = harness(monkeypatch)
    owner.current.application.start()
    try:
        owner.facade("operator", "request_emergency_stop_own_domain")
        owner.request("operator", {"action": "disconnect", "expected_generation": 0})
        assert settle(owner)["emergency_latched"] is True
        with pytest.raises(RuntimeError, match="unlatched"):
            owner.request("operator", {"action": "reconnect", "expected_generation": 0})
        assert len(generations) == 1
    finally:
        owner.close()


def test_lower_stop_published_while_draining_close_remains_latched(monkeypatch):
    owner, generations, _, _ = harness(monkeypatch)
    owner.current.application.start()
    close = owner.current.application.close
    def close_with_stop():
        close()
        owner.current.lower.ports.last_emergency_stop_receipt = object()
    owner.current.application.close = close_with_stop
    try:
        owner.request("operator", {"action": "disconnect", "expected_generation": 0})
        state = settle(owner)
        assert state["state"] == "disconnected" and state["emergency_latched"]
        assert state["reconnect_allowed"] is False
        with pytest.raises(RuntimeError, match="unlatched"):
            owner.request("operator", {"action": "reconnect", "expected_generation": 0})
        assert len(generations) == 1
    finally:
        owner.close()


def test_failed_reconnect_keeps_failed_generation_until_explicit_release(monkeypatch):
    owner, generations, cameras, model = harness(monkeypatch)
    owner.current.application.start()
    build = owner._rebuild
    def failed():
        candidate = build()
        candidate.hands["right_gripper"].backend.is_ready = lambda: False
        return candidate
    try:
        owner.request("operator", {"action": "disconnect", "expected_generation": 0})
        settle(owner)
        owner._rebuild = failed
        owner.request("operator", {"action": "reconnect", "expected_generation": 0})
        assert settle(owner)["state"] == "faulted"
        assert owner.current is generations[1]
        with pytest.raises(RuntimeError, match="complete release"):
            owner.request("operator", {"action": "reconnect", "expected_generation": 1})
        owner.request("operator", {"action": "disconnect", "expected_generation": 1})
        assert settle(owner)["state"] == "disconnected"
        owner._rebuild = build
        owner.request("operator", {"action": "reconnect", "expected_generation": 1})
        assert settle(owner)["state"] == "connected"
        assert len(generations) == 3 and cameras.starts == model.starts == 1
    finally:
        owner.close()


def test_stop_during_reconnect_prevents_ready_publication(monkeypatch):
    owner, generations, _, _ = harness(monkeypatch)
    owner.current.application.start()
    build = owner._rebuild
    entered, release = threading.Event(), threading.Event()
    def rebuild():
        candidate = build()
        start = candidate.application.start
        def paused():
            entered.set()
            assert release.wait(3)
            start()
        candidate.application.start = paused
        return candidate
    try:
        owner.request("operator", {"action": "disconnect", "expected_generation": 0})
        settle(owner)
        owner._rebuild = rebuild
        owner.request("operator", {"action": "reconnect", "expected_generation": 0})
        assert entered.wait(1)
        owner.facade("operator", "request_emergency_stop_own_domain")
        release.set()
        assert settle(owner)["state"] == "faulted"
        assert owner.current.application.state.value == "closed"
        assert owner.snapshot("operator")["emergency_latched"]
    finally:
        release.set()
        owner.close()


def test_resident_model_partial_start_is_retained_and_never_reported_ready():
    resident = ResidentPerceptionResources()
    model = Resource()
    def fail():
        model.starts += 1
        raise RuntimeError("partial model startup")
    model.start = fail
    lease = resident.model(model)
    with pytest.raises(RuntimeError, match="partial model"):
        lease.start()
    with pytest.raises(RuntimeError, match="did not complete"):
        lease.start()
    assert model.starts == 1
    lease.close()
    assert model.closes == 0
    resident.close()
    assert model.closes == 1


def test_persistent_transport_controls_real_application_generations(monkeypatch, short_unix_socket_path_factory):
    owner, generations, cameras, model = harness(monkeypatch)
    path = short_unix_socket_path_factory("hardware-generation")
    def server_factory(current):
        return AgentGatewayUnixJsonServer(socket_path=path,
            peer_client_map=AgentGatewayUnixPeerClientMap({os.getuid(): "operator"}),
            facade_factory=current.facade, hardware_connection_control=current,
            close_gateway=lambda: current.current.deployment.gateway_session.close())
    owner._server_factory = server_factory
    ready = threading.Event()
    errors = []
    def serve():
        try:
            owner.serve_forever(ready_callback=ready.set)
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=serve)
    thread.start()
    channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        assert ready.wait(3)
        channel.settimeout(3)
        channel.connect(str(path))
        def call(method, payload=None):
            return _raw_call(channel, request_id="connection:test", method=method, payload=payload)
        assert call("healthz")["result"]["hardware_connection_control"] == "agent_gateway.hardware_connection.v1"
        old = call("get_skill_route_manifest")["result"]
        assert call("request_hardware_connection", {"action": "disconnect", "expected_generation": 0})["ok"]
        _wait_for(lambda: owner.snapshot("operator")["state"] == "disconnected")
        state = call("read_hardware_connection_state")["result"]
        assert state["state"] == "disconnected"
        blocked = call("get_observation_snapshot")
        assert blocked["ok"] is False
        # A rejected ordinary call must not kill the operator control socket.
        assert call("read_hardware_connection_state")["result"]["reconnect_allowed"]
        assert call("request_hardware_connection", {"action": "reconnect", "expected_generation": 0})["ok"]
        _wait_for(lambda: owner.snapshot("operator")["state"] == "connected")
        new = call("get_skill_route_manifest")["result"]
        assert old["runtime_session_id"] != new["runtime_session_id"]
        assert cameras.starts == model.starts == 1
    finally:
        channel.close()
        owner.request_shutdown()
        thread.join(timeout=5)
    assert not thread.is_alive() and not errors
    assert cameras.closes == model.closes == 1


def _gripper(owner, component="right_gripper", target="closed", generation=0, client="operator"):
    return owner.operator_gripper(client, {"component_id": component, "target_state": target,
                                           "expected_generation": generation})


def test_operator_gripper_commands_either_hand_through_its_skill_port_and_holds(monkeypatch):
    owner, _, _, _ = harness(monkeypatch)
    owner.current.application.start()
    try:
        right = owner.current.hands["right_gripper"].backend
        right.samples = [(0.4, True), (0.97, False), (0.97, False), (0.97, False)]
        before = owner.current.hands["right_gripper"].last_command_attempt_ref
        closed = _gripper(owner)
        assert closed["status"] == "target_reached" and closed["hold_confirmed"] is True
        assert closed["measured_close"] == pytest.approx(0.97) and closed["moving"] is False
        assert closed["component_id"] == "right_gripper" and closed["generation"] == 0
        assert right.commands == [1.0] and right.holds[-1] == "operator_gripper_settled"
        # A new command attempt: a held object's retained grip is no longer trusted.
        assert owner.current.hands["right_gripper"].last_command_attempt_ref not in (None, before)
        left = owner.current.hands["left_gripper"].backend
        left.samples = [(0.3, True), (0.01, False), (0.01, False), (0.01, False)]
        opened = _gripper(owner, component="left_gripper", target="open")
        assert opened["status"] == "target_reached" and left.commands == [0.0]
        # Stopped on an object: a close that settles before full closure.
        right.samples = [(0.5, True)] + [(0.6, False)] * 40
        contact = _gripper(owner)
        assert contact["status"] == "settled_before_full_close"
        assert owner.snapshot("operator")["operator_gripper_busy"] is None
    finally:
        owner.close()


def test_operator_gripper_refusals_send_no_command(monkeypatch):
    owner, generations, _, _ = harness(monkeypatch)
    owner.current.application.start()
    hands = owner.current.hands
    try:
        with pytest.raises(PermissionError):
            _gripper(owner, client="agent")
        with pytest.raises(ValueError, match="Invalid operator gripper"):
            _gripper(owner, component="right_arm")
        with pytest.raises(ValueError, match="Invalid operator gripper"):
            _gripper(owner, target="half")
        with pytest.raises(RuntimeError, match="binding changed"):
            _gripper(owner, generation=1)
        store = owner.current.deployment.gateway_session.service.operation_store
        monkeypatch.setattr(store, "active_operation", lambda: object())
        with pytest.raises(RuntimeError, match="Skill operation is active"):
            _gripper(owner)
        monkeypatch.undo()
        owner._gripper_command = "left_gripper"
        with pytest.raises(RuntimeError, match="still running"):
            _gripper(owner)
        with pytest.raises(RuntimeError, match="operator gripper command is running"):
            owner.facade("operator", "submit_skill_request")
        owner._gripper_command = None
        owner.request("operator", {"action": "disconnect", "expected_generation": 0})
        assert settle(owner)["state"] == "disconnected"
        with pytest.raises(RuntimeError, match="disconnected"):
            _gripper(owner)
        assert all(not port.backend.commands for port in hands.values())
        assert len(generations) == 1
    finally:
        owner.close()


def test_operator_gripper_is_refused_after_an_emergency_stop(monkeypatch):
    owner, _, _, _ = harness(monkeypatch)
    owner.current.application.start()
    try:
        owner.facade("operator", "request_emergency_stop_own_domain")
        with pytest.raises(RuntimeError, match="Emergency stop is latched"):
            _gripper(owner)
        assert not owner.current.hands["right_gripper"].backend.commands
    finally:
        owner.close()


def test_operator_gripper_reports_a_timeout_and_still_holds(monkeypatch):
    from agent_gateway.profiles.real import hardware_connection

    clock = iter(float(index) for index in range(1000))
    port = SimpleNamespace(component_id="right_gripper", calls=[])

    def command_close(target):
        port.calls.append(("command", target))
        return SimpleNamespace(accepted=True, evidence_refs=("command:1",))

    def poll_state():
        return SimpleNamespace(measured_close=0.5, moving=True, fault_code=None, evidence_ref="sample")

    def hold(*, reason):
        port.calls.append(("hold", reason))
        return SimpleNamespace(succeeded=True, evidence_refs=("hold:1",))

    port.command_close, port.poll_state, port.hold = command_close, poll_state, hold
    result = hardware_connection.run_operator_gripper_command(
        port, "open", timeout_sec=3.0, clock=lambda: next(clock), sleep=lambda _: None)
    assert result["status"] == "timeout" and result["hold_confirmed"] is True
    assert port.calls == [("command", 0.0), ("hold", "operator_gripper_timeout")]


def test_operator_gripper_over_the_persistent_transport(monkeypatch, short_unix_socket_path_factory):
    owner, _, _, _ = harness(monkeypatch)
    path = short_unix_socket_path_factory("operator-gripper")
    owner._server_factory = lambda current: AgentGatewayUnixJsonServer(socket_path=path,
        peer_client_map=AgentGatewayUnixPeerClientMap({os.getuid(): "operator"}),
        facade_factory=current.facade, hardware_connection_control=current,
        close_gateway=lambda: current.current.deployment.gateway_session.close())
    ready, errors = threading.Event(), []

    def serve():
        try:
            owner.serve_forever(ready_callback=ready.set)
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=serve)
    thread.start()
    channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        assert ready.wait(3)
        channel.settimeout(5)
        channel.connect(str(path))
        def call(method, payload=None):
            return _raw_call(channel, request_id="gripper:test", method=method, payload=payload)
        assert call("healthz")["result"]["operator_gripper_control"] == "agent_gateway.operator_gripper.v1"
        owner.current.hands["left_gripper"].backend.samples = [(0.02, False)] * 3
        reply = call("request_operator_gripper",
                     {"component_id": "left_gripper", "target_state": "open", "expected_generation": 0})
        assert reply["ok"] and reply["result"]["status"] == "target_reached"
        refused = call("request_operator_gripper",
                       {"component_id": "left_gripper", "target_state": "open", "expected_generation": 7})
        assert refused["ok"] is False
        assert call("read_hardware_connection_state")["result"]["state"] == "connected"
    finally:
        channel.close()
        owner.request_shutdown()
        thread.join(timeout=5)
    assert not thread.is_alive() and not errors

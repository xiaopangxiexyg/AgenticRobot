from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from agent_gateway.profiles.real.exclusive_factory import (
    assemble_exclusive_real_agent_primitive_deployment,
)
from agent_gateway.profiles.real.exclusive_application import (
    ExclusiveRealSkillApplication,
    ExclusiveRealSkillApplicationConfig,
    ExclusiveRealSkillApplicationState,
)
from agent_gateway.profiles.real.motion_primitive_profiles import (
    REAL_AGENT_TCP_CARDINAL_STEP_ROUTE_ID_V1,
    REAL_AGENT_TCP_FIXED_ORIENTATION_ROUTE_ID_BY_SKILL_ID_V1,
)
from agent_gateway.profiles.real.observation_gateway import (
    RealTcpPoseObserveAttachmentSpec,
)
from agent_gateway.protocol.skill_catalog import SkillDeploymentSafetyEnvelopeRef
from agent_gateway.skills.motion.definition import (
    FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1,
)
from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
    RigidFrameTransformSpec,
)
from robot_motion.coordinate_profiles import STATION_RETARGET_FRAME_CONTRACT_V1
from robot_motion.runtime import EffectorCommand, TeleopCommand, TeleopObservation
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
    route_ids = (
        REAL_AGENT_TCP_CARDINAL_STEP_ROUTE_ID_V1,
        *tuple(
            REAL_AGENT_TCP_FIXED_ORIENTATION_ROUTE_ID_BY_SKILL_ID_V1[skill_id]
            for skill_id in FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1
        ),
    )
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


def _deployment(*, observation_provider, hold_reasons):
    frame_graph, station_frame, attachment, config, envelope = _assembly_inputs()
    target_builder = _TargetBuilder()
    return assemble_exclusive_real_agent_primitive_deployment(
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

    application.start()
    time.sleep(0.03)
    application.close()

    assert application.state == ExclusiveRealSkillApplicationState.CLOSED
    assert resources.starts == 1
    assert resources.closes == 1
    assert application.startup_admission_receipt == {"startup": "admitted"}
    assert server.closed == 1
    assert holds
    assert deployment.control_runtime.last_hold_receipt is not None
    assert deployment.control_runtime.last_hold_receipt.succeeded is True


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

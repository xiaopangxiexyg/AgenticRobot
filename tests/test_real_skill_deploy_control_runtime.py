from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from robot_motion.runtime import EffectorCommand, TeleopCommand, TeleopObservation
from teleop_runtime.skill_deploy import (
    ExclusiveRealSkillControlRuntime,
    RealAgentMotionActivation,
    RealAgentMotionOwnerIdentity,
    RealAgentMotionPhase,
    RealAgentMotionPort,
    RealAgentMotionPortConfig,
    RealSkillControlLifecycle,
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

    def build_command_for_target_pair(self, target):
        command = TeleopCommand(timestamp=target.timestamp)
        command.add_command(
            EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="joint_vector",
                value=np.zeros(6),
            )
        )
        return command

    def tool_tcp_world_targets_for_qpos_pair(self, *, left_qpos, right_qpos):
        del left_qpos, right_qpos
        return self.current_tool_tcp_world_targets()


def _port(builder: _TargetBuilder) -> RealAgentMotionPort:
    return RealAgentMotionPort(
        config=RealAgentMotionPortConfig(
            active_sides=("left",),
            component_id_by_side={"left": "left_arm"},
            execution_frame_id="solver_world",
            execution_frame_spec_hash="frame-spec:test",
            exclusive_command_claim=True,
        ),
        target_builder=builder,
        owner_identity=RealAgentMotionOwnerIdentity(
            runtime_session_id="runtime:test",
            control_domain_id="real-skill:test",
            embodiment_instance_id="embodiment:test",
            target_binding_id="target:test",
            execution_manifest_hash="execution:test",
            backend_inventory_hash="backend:test",
            clock_domain_id="clock:test",
        ),
    )


def _activation() -> RealAgentMotionActivation:
    target = np.eye(4)
    target[0, 3] = 0.03
    return RealAgentMotionActivation(
        operation_id="operation:test",
        request_id="request:test",
        activation_grant_id="grant:test",
        activation_grant_hash="grant-hash:test",
        component_id="left_arm",
        side="left",
        primitive_id="tcp_step_forward",
        primitive_hash="primitive:test",
        target_pose_world=target,
        execution_frame_id="solver_world",
        execution_frame_spec_hash="frame-spec:test",
        measured_state_revision=0,
        measured_state_hash="measured:test",
        submitted_at=1.0,
        expires_at=10.0,
    )


def _observation_cut() -> RealSkillMotionObservationCut:
    return RealSkillMotionObservationCut(
        observation=TeleopObservation(timestamp=2.0),
        qpos_by_side={"left": np.zeros(6), "right": np.zeros(6)},
        measured_at=1.9,
        available_at=1.95,
        evidence_refs=("hardware-qpos-cut:test",),
    )


def test_independent_runtime_owns_dispatch_and_lower_confirmed_hold() -> None:
    builder = _TargetBuilder()
    port = _port(builder)
    dispatched: list[TeleopCommand] = []
    hold_reasons: list[str] = []

    runtime = ExclusiveRealSkillControlRuntime(
        motion_port=port,
        observation_provider=_observation_cut,
        synchronize_target_builder=builder.synchronize,
        command_admission=lambda command, observation, now: True,
        command_dispatch=lambda command: (
            dispatched.append(command)
            or RealSkillMotionDispatchReceipt(
                succeeded=True,
                evidence_refs=("rm65-dispatch:test",),
            )
        ),
        hold_controller=lambda reason: (
            hold_reasons.append(reason)
            or RealSkillMotionHoldReceipt(
                succeeded=True,
                evidence_refs=("rm65-hold:test",),
            )
        ),
        clock=lambda: 3.0,
    )

    attachment = runtime.start()
    port.submit_activation(_activation())
    active = runtime.step(now=2.0)

    assert attachment.startswith("real-agent-motion-exclusive-attachment:")
    assert active.lifecycle == RealSkillControlLifecycle.RUNNING
    assert active.dispatch_succeeded is True
    assert active.dispatch_evidence_refs == ("rm65-dispatch:test",)
    assert len(dispatched) == 1
    assert active.port_snapshot.phase == RealAgentMotionPhase.ACTIVE
    assert active.port_snapshot.dispatch_count == 1
    owner = port.latest_owner_state()
    assert owner is not None
    assert owner.owner_kind == "exclusive_real_skill_motion_port"

    port.request_hold(
        operation_id="operation:test",
        reason="skill_completed",
        requested_at=2.1,
    )
    held = runtime.step(now=2.2)

    assert held.lifecycle == RealSkillControlLifecycle.HOLD
    assert held.port_snapshot.phase == RealAgentMotionPhase.HELD
    assert held.port_snapshot.hold_succeeded is True
    assert hold_reasons == [
        "real Agent motion operation operation:test: skill_completed"
    ]

    runtime.close()
    assert runtime.lifecycle == RealSkillControlLifecycle.CLOSED


def test_failed_dispatch_fails_closed_through_same_hold_owner() -> None:
    builder = _TargetBuilder()
    port = _port(builder)
    holds: list[str] = []
    runtime = ExclusiveRealSkillControlRuntime(
        motion_port=port,
        observation_provider=_observation_cut,
        synchronize_target_builder=builder.synchronize,
        command_admission=lambda command, observation, now: True,
        command_dispatch=lambda command: RealSkillMotionDispatchReceipt(
            succeeded=False,
            evidence_refs=("rm65-dispatch-rejected:test",),
        ),
        hold_controller=lambda reason: (
            holds.append(reason)
            or RealSkillMotionHoldReceipt(
                succeeded=True,
                evidence_refs=("rm65-hold-after-reject:test",),
            )
        ),
    )
    runtime.start()
    port.submit_activation(_activation())

    step = runtime.step(now=2.0)

    assert step.dispatch_attempted is True
    assert step.dispatch_succeeded is False
    assert step.lifecycle == RealSkillControlLifecycle.HOLD
    assert step.port_snapshot.phase == RealAgentMotionPhase.HELD
    assert holds == [
        "real Agent motion operation operation:test: lower_dispatch_failed"
    ]


def test_realteleop_motion_port_path_is_object_identical_compatibility_export() -> None:
    canonical = importlib.import_module("teleop_runtime.skill_deploy.motion_port")
    compatibility = importlib.import_module(
        "teleop_runtime.real_teleop.agent_motion_port"
    )

    assert compatibility.RealAgentMotionPort is canonical.RealAgentMotionPort
    assert (
        compatibility.RealAgentMotionActivation
        is canonical.RealAgentMotionActivation
    )


def test_skill_deploy_import_does_not_load_realteleop_or_composition() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import json
import sys
import teleop_runtime.skill_deploy
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

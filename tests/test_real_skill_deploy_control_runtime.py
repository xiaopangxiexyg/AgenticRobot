from __future__ import annotations

import importlib
import json
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

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


def _activation(
    *,
    suffix: str = "test",
    submitted_at: float = 1.0,
    expires_at: float = 10.0,
) -> RealAgentMotionActivation:
    target = np.eye(4)
    target[0, 3] = 0.03
    return RealAgentMotionActivation(
        operation_id=f"operation:{suffix}",
        request_id=f"request:{suffix}",
        activation_grant_id=f"grant:{suffix}",
        activation_grant_hash=f"grant-hash:{suffix}",
        component_id="left_arm",
        side="left",
        primitive_id="tcp_step_forward",
        primitive_hash="primitive:test",
        target_pose_world=target,
        execution_frame_id="solver_world",
        execution_frame_spec_hash="frame-spec:test",
        measured_state_revision=0,
        measured_state_hash="measured:test",
        submitted_at=submitted_at,
        expires_at=expires_at,
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


def test_confirmed_hold_reopens_only_for_a_new_pending_activation() -> None:
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
                evidence_refs=(f"rm65-dispatch:{len(dispatched)}",),
            )
        ),
        hold_controller=lambda reason: (
            hold_reasons.append(reason)
            or RealSkillMotionHoldReceipt(
                succeeded=True,
                evidence_refs=(f"rm65-hold:{len(hold_reasons)}",),
            )
        ),
    )
    runtime.start()
    port.submit_activation(_activation(suffix="first"))
    first_active = runtime.step(now=2.0)
    assert first_active.port_snapshot.phase == RealAgentMotionPhase.ACTIVE

    port.request_hold(
        operation_id="operation:first",
        reason="skill_completed",
        requested_at=2.1,
    )
    first_held = runtime.step(now=2.2)
    assert first_held.lifecycle == RealSkillControlLifecycle.HOLD
    assert first_held.port_snapshot.phase == RealAgentMotionPhase.HELD
    first_receipt = runtime.last_hold_receipt
    assert first_receipt is not None and first_receipt.succeeded

    idle_held = runtime.step(now=2.3)
    assert idle_held.lifecycle == RealSkillControlLifecycle.HOLD
    assert idle_held.port_snapshot.phase == RealAgentMotionPhase.HELD
    assert runtime.last_hold_receipt is first_receipt

    port.submit_activation(
        _activation(suffix="second", submitted_at=2.4, expires_at=10.0)
    )
    second_active = runtime.step(now=2.5)

    assert second_active.lifecycle == RealSkillControlLifecycle.RUNNING
    assert second_active.port_snapshot.phase == RealAgentMotionPhase.ACTIVE
    assert second_active.port_snapshot.operation_id == "operation:second"
    assert second_active.dispatch_succeeded is True
    assert runtime.last_hold_receipt is None
    assert len(dispatched) == 2

    port.request_hold(
        operation_id="operation:second",
        reason="skill_completed",
        requested_at=2.6,
    )
    second_held = runtime.step(now=2.7)
    assert second_held.lifecycle == RealSkillControlLifecycle.HOLD
    assert second_held.port_snapshot.phase == RealAgentMotionPhase.HELD
    assert len(hold_reasons) == 2


def test_failed_lower_hold_never_becomes_a_rearmable_session_boundary() -> None:
    builder = _TargetBuilder()
    port = _port(builder)
    runtime = ExclusiveRealSkillControlRuntime(
        motion_port=port,
        observation_provider=_observation_cut,
        synchronize_target_builder=builder.synchronize,
        command_admission=lambda command, observation, now: True,
        command_dispatch=lambda command: RealSkillMotionDispatchReceipt(
            succeeded=True,
            evidence_refs=("rm65-dispatch:test",),
        ),
        hold_controller=lambda reason: RealSkillMotionHoldReceipt(
            succeeded=False,
            evidence_refs=("rm65-hold-failed:test",),
        ),
    )
    runtime.start()
    port.submit_activation(_activation())
    runtime.step(now=2.0)
    port.request_hold(
        operation_id="operation:test",
        reason="skill_completed",
        requested_at=2.1,
    )

    failed_hold = runtime.step(now=2.2)

    assert failed_hold.lifecycle == RealSkillControlLifecycle.FAULT
    assert failed_hold.port_snapshot.phase == RealAgentMotionPhase.FAULTED
    assert runtime.last_hold_receipt is not None
    assert runtime.last_hold_receipt.succeeded is False
    with pytest.raises(RuntimeError, match="busy in phase faulted"):
        port.submit_activation(
            _activation(suffix="second", submitted_at=2.3, expires_at=10.0)
        )


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


def test_fail_closed_does_not_require_another_observation() -> None:
    builder = _TargetBuilder()
    port = _port(builder)
    holds: list[str] = []
    runtime = ExclusiveRealSkillControlRuntime(
        motion_port=port,
        observation_provider=lambda: (_ for _ in ()).throw(
            RuntimeError("camera-independent arm read failed")
        ),
        synchronize_target_builder=builder.synchronize,
        command_admission=lambda command, observation, now: True,
        command_dispatch=lambda command: RealSkillMotionDispatchReceipt(
            succeeded=True,
            evidence_refs=("unused-dispatch:test",),
        ),
        hold_controller=lambda reason: (
            holds.append(reason)
            or RealSkillMotionHoldReceipt(
                succeeded=True,
                evidence_refs=("rm65-hold-after-read-failure:test",),
            )
        ),
        clock=lambda: 2.0,
    )
    runtime.start()
    port.submit_activation(_activation())

    with pytest.raises(RuntimeError, match="arm read failed"):
        runtime.step(now=2.0)
    receipt = runtime.fail_closed("control_cycle_failed", now=2.0)

    assert receipt.succeeded is True
    assert runtime.lifecycle == RealSkillControlLifecycle.HOLD
    assert port.snapshot().phase == RealAgentMotionPhase.HELD
    assert holds == [
        "real Agent motion operation operation:test: control_cycle_failed"
    ]


def test_fail_closed_is_a_dispatch_barrier_for_an_inflight_cycle() -> None:
    builder = _TargetBuilder()
    port = _port(builder)
    admission_entered = threading.Event()
    release_admission = threading.Event()
    hold_called = threading.Event()
    order: list[str] = []
    errors: list[BaseException] = []

    def lower_admission(command, observation, now):
        del command, observation, now
        order.append("admission")
        admission_entered.set()
        assert release_admission.wait(timeout=1.0)
        return True

    def dispatch(command):
        del command
        order.append("dispatch")
        return RealSkillMotionDispatchReceipt(
            succeeded=True,
            evidence_refs=("rm65-dispatch:barrier",),
        )

    def hold(reason):
        del reason
        order.append("hold")
        hold_called.set()
        return RealSkillMotionHoldReceipt(
            succeeded=True,
            evidence_refs=("rm65-hold:barrier",),
        )

    runtime = ExclusiveRealSkillControlRuntime(
        motion_port=port,
        observation_provider=_observation_cut,
        synchronize_target_builder=builder.synchronize,
        command_admission=lower_admission,
        command_dispatch=dispatch,
        hold_controller=hold,
    )
    runtime.start()
    port.submit_activation(_activation())

    def run_step() -> None:
        try:
            runtime.step(now=2.0)
        except BaseException as exc:  # noqa: BLE001 - assert thread outcome.
            errors.append(exc)

    def run_hold() -> None:
        try:
            runtime.fail_closed("shutdown", now=2.1)
        except BaseException as exc:  # noqa: BLE001 - assert thread outcome.
            errors.append(exc)

    step_thread = threading.Thread(target=run_step)
    hold_thread = threading.Thread(target=run_hold)
    step_thread.start()
    assert admission_entered.wait(timeout=1.0)
    hold_thread.start()

    assert not hold_called.wait(timeout=0.05)
    release_admission.set()
    step_thread.join(timeout=1.0)
    hold_thread.join(timeout=1.0)

    assert not step_thread.is_alive()
    assert not hold_thread.is_alive()
    assert errors == []
    assert order == ["admission", "dispatch", "hold"]
    assert runtime.lifecycle == RealSkillControlLifecycle.HOLD


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

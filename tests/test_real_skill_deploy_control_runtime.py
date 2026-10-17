from __future__ import annotations

import importlib
import json
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robot_motion.runtime import EffectorCommand, TeleopCommand, TeleopObservation
from teleop_runtime.skill_deploy import (
    REAL_AGENT_MOTION_COMMAND_METADATA_KEY,
    ExclusiveRealSkillControlRuntime,
    RealAgentJointTrajectory,
    RealAgentMotionActivation,
    RealAgentMotionOwnerIdentity,
    RealAgentMotionPhase,
    RealAgentMotionPort,
    RealAgentMotionPortConfig,
    RealSkillControlLifecycle,
    RealSkillMotionDispatchReceipt,
    RealSkillMotionHoldReceipt,
    RealSkillMotionObservationCut,
    RealSkillMotionObservationUnavailable,
)


class _TargetBuilder:
    def __init__(self) -> None:
        self.qpos_by_side = {
            "left": np.zeros(6),
            "right": np.zeros(6),
        }
        self.sync_count = 0

    def synchronize(self, qpos_by_side) -> None:
        self.sync_count += 1
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


def test_startup_hold_uses_lower_owner_clock_without_dispatch_or_resume():
    builder = _TargetBuilder()
    port = _port(builder)
    dispatched = []
    runtime = ExclusiveRealSkillControlRuntime(
        motion_port=port,
        observation_provider=_observation_cut,
        synchronize_target_builder=builder.synchronize,
        command_admission=lambda command, observation, now: True,
        command_dispatch=lambda command: dispatched.append(command),
        hold_controller=lambda reason: RealSkillMotionHoldReceipt(
            succeeded=True, evidence_refs=("fixture:startup-hold",)
        ),
        clock=lambda: 2.0,
    )
    with pytest.raises(RuntimeError, match="started lower owner"):
        runtime.request_startup_hold()
    runtime.start()
    runtime.request_startup_hold()
    assert port.snapshot().hold_requested_at == 2.0
    result = runtime.step()
    assert result.port_snapshot.phase == RealAgentMotionPhase.HELD
    assert result.lifecycle == RealSkillControlLifecycle.HOLD
    assert result.port_snapshot.hold_succeeded is True
    assert not dispatched
    with pytest.raises(RuntimeError):
        runtime.request_startup_hold()
    runtime.close()


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
    operation_trace = dispatched[0].metadata[
        REAL_AGENT_MOTION_COMMAND_METADATA_KEY
    ]
    assert operation_trace["target_pose_semantics"] == "actual_tool_tcp_world"
    np.testing.assert_allclose(
        operation_trace["target_tool_tcp_pose_world"],
        _activation().target_pose_world,
    )
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


def test_preplanned_grasp_path_dispatch_never_enters_dynamic_target_builder() -> None:
    builder = _TargetBuilder()

    def reject_dynamic_target(_target):
        raise AssertionError("preplanned grasp path entered dynamic target solving")

    builder.build_command_for_target_pair = reject_dynamic_target
    port = _port(builder)
    dispatched: list[TeleopCommand] = []
    trajectory = RealAgentJointTrajectory(
        time_from_start_sec=np.array([0.0, 0.25], dtype=float),
        qpos_by_side={
            "left": np.array(
                [[0.0] * 6, [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]],
                dtype=float,
            ),
            "right": np.full((2, 6), 0.2, dtype=float),
        },
        planner_evidence_refs=("rm65-grasp-path:runtime-test",),
    )
    runtime = ExclusiveRealSkillControlRuntime(
        motion_port=port,
        observation_provider=lambda: RealSkillMotionObservationCut(
            observation=TeleopObservation(timestamp=2.0),
            qpos_by_side={
                "left": np.zeros(6, dtype=float),
                "right": np.full(6, 0.2, dtype=float),
            },
            measured_at=1.9,
            available_at=1.95,
            evidence_refs=("hardware-qpos-cut:preplanned-grasp",),
        ),
        synchronize_target_builder=builder.synchronize,
        command_admission=lambda command, observation, now: True,
        command_dispatch=lambda command: (
            dispatched.append(command)
            or RealSkillMotionDispatchReceipt(
                succeeded=True,
                evidence_refs=("rm65-dispatch:preplanned-grasp",),
            )
        ),
        hold_controller=lambda reason: RealSkillMotionHoldReceipt(
            succeeded=True,
            evidence_refs=("rm65-hold:preplanned-grasp",),
        ),
        max_measurement_age_sec=1.0,
        clock=lambda: 2.0,
    )

    runtime.start()
    port.submit_activation(replace(_activation(), joint_trajectory=trajectory))
    step = runtime.step(now=2.0)

    assert step.dispatch_succeeded is True
    assert len(dispatched) == 1
    assert set(dispatched[0].commands) == {"left_arm", "right_arm"}
    np.testing.assert_allclose(dispatched[0].commands["right_arm"].value, 0.2)
    metadata = dispatched[0].metadata[REAL_AGENT_MOTION_COMMAND_METADATA_KEY]
    assert metadata["joint_trajectory_hash"] == trajectory.semantic_content_hash


def test_preplanned_path_clock_rebases_after_non_dispatch_gap() -> None:
    class _ReadyOwner:
        def skill_motion_owner_state(self) -> tuple[str, str]:
            return ("running", "ready")

        def hold_skill_motion(self, reason: str) -> bool:
            del reason
            return True

    trajectory = RealAgentJointTrajectory(
        time_from_start_sec=np.array([0.0, 1.0, 2.0], dtype=float),
        qpos_by_side={
            "left": np.array(
                [[0.0] * 6, [0.1] * 6, [0.2] * 6],
                dtype=float,
            ),
            "right": np.zeros((3, 6), dtype=float),
        },
        planner_evidence_refs=("rm65-grasp-path:clock-rebase",),
    )
    port = _port(_TargetBuilder())
    initial = {"left": np.zeros(6), "right": np.zeros(6)}
    port.publish_measured_qpos_pair(
        qpos_by_side=initial,
        measured_at=1.0,
        available_at=1.0,
        source_evidence_refs=("hardware-qpos-cut:clock-rebase",),
    )
    port.submit_activation(
        replace(_activation(suffix="clock-rebase"), joint_trajectory=trajectory)
    )
    port.apply_control_owner_requests(owner=_ReadyOwner(), now=1.0)

    first = port.command_for_observation(
        TeleopObservation(timestamp=1.05),
        1.05,
    )
    assert isinstance(first, TeleopCommand)
    port.observe_session_step(command=first, command_dispatched=False)

    delayed = port.command_for_observation(
        TeleopObservation(timestamp=2.75),
        2.75,
    )
    assert isinstance(delayed, TeleopCommand)
    np.testing.assert_allclose(delayed.commands["left_arm"].value, 0.01)
    operation = delayed.metadata[REAL_AGENT_MOTION_COMMAND_METADATA_KEY]
    assert operation["source"] == "replay_qpos_trajectory"
    assert operation["singularity_replay_slowdown_preapplied"] is True
    assert operation["singularity_replay_slowdown_collision_rechecked"] is True
    effector_operation = delayed.commands["left_arm"].metadata
    assert effector_operation["source"] == "replay_qpos_trajectory"
    assert effector_operation["singularity_replay_slowdown_preapplied"] is True
    assert operation["trajectory_elapsed_sec"] == pytest.approx(0.1)
    assert operation["trajectory_clock_advance_capped"] is True

    snapshot = port.observe_session_step(
        command=delayed,
        command_dispatched=True,
    )
    assert snapshot.trajectory_clock_rebased_count == 1
    assert snapshot.trajectory_elapsed_sec == pytest.approx(0.1)
    resumed = port.command_for_observation(
        TeleopObservation(timestamp=2.85),
        2.85,
    )
    assert isinstance(resumed, TeleopCommand)
    np.testing.assert_allclose(resumed.commands["left_arm"].value, 0.02)



def test_unverified_preplanned_path_does_not_claim_singularity_replay() -> None:
    class _ReadyOwner:
        def skill_motion_owner_state(self) -> tuple[str, str]:
            return ("running", "ready")

        def hold_skill_motion(self, reason: str) -> bool:
            del reason
            return True

    trajectory = RealAgentJointTrajectory(
        time_from_start_sec=np.array([0.0, 1.0], dtype=float),
        qpos_by_side={
            "left": np.array([[0.0] * 6, [0.1] * 6], dtype=float),
            "right": np.zeros((2, 6), dtype=float),
        },
        planner_evidence_refs=("external-path:test",),
    )
    port = _port(_TargetBuilder())
    port.publish_measured_qpos_pair(
        qpos_by_side={"left": np.zeros(6), "right": np.zeros(6)},
        measured_at=1.0,
        available_at=1.0,
        source_evidence_refs=("hardware-qpos-cut:unverified",),
    )
    port.submit_activation(
        replace(_activation(suffix="unverified"), joint_trajectory=trajectory)
    )
    port.apply_control_owner_requests(owner=_ReadyOwner(), now=1.0)

    command = port.command_for_observation(
        TeleopObservation(timestamp=1.1),
        1.1,
    )
    assert isinstance(command, TeleopCommand)
    operation = command.metadata[REAL_AGENT_MOTION_COMMAND_METADATA_KEY]
    assert operation["source"] == "joint_vector"
    assert operation["singularity_replay_slowdown_preapplied"] is False
    assert operation["singularity_replay_slowdown_collision_rechecked"] is False
    effector_operation = command.commands["left_arm"].metadata
    assert effector_operation["source"] == "joint_vector"
    assert effector_operation["singularity_replay_slowdown_preapplied"] is False


def test_measured_qpos_projection_uses_post_synchronization_availability() -> None:
    builder = _TargetBuilder()
    port = _port(builder)
    observed: list[tuple[RealSkillMotionObservationCut, float]] = []
    source_cut = RealSkillMotionObservationCut(
        observation=TeleopObservation(timestamp=2.0),
        qpos_by_side={"left": np.ones(6), "right": np.full(6, 2.0)},
        measured_at=1.9,
        available_at=1.95,
        evidence_refs=("hardware-qpos-cut:projection",),
    )

    def observe(cut, *, derived_available_at):
        assert np.array_equal(builder.qpos_by_side["left"], cut.qpos_by_side["left"])
        assert np.array_equal(builder.qpos_by_side["right"], cut.qpos_by_side["right"])
        observed.append((cut, derived_available_at))

    runtime = ExclusiveRealSkillControlRuntime(
        motion_port=port,
        observation_provider=lambda: source_cut,
        synchronize_target_builder=builder.synchronize,
        measured_qpos_observer=observe,
        command_admission=lambda command, observation, now: True,
        command_dispatch=lambda command: RealSkillMotionDispatchReceipt(
            succeeded=True,
            evidence_refs=("unused-dispatch:projection",),
        ),
        hold_controller=lambda reason: RealSkillMotionHoldReceipt(
            succeeded=True,
            evidence_refs=("unused-hold:projection",),
        ),
    )
    runtime.start()

    runtime.step(now=2.25)

    assert observed == [(source_cut, 2.25)]
    measured = port.latest_measured_pose()
    assert measured is not None
    assert measured.measured_at == 1.9
    assert measured.available_at == 2.25


def test_motion_trace_failure_isolated_from_control_cycle() -> None:
    builder = _TargetBuilder()
    port = _port(builder)
    runtime = ExclusiveRealSkillControlRuntime(
        motion_port=port,
        observation_provider=_observation_cut,
        synchronize_target_builder=builder.synchronize,
        command_admission=lambda command, observation, now: True,
        command_dispatch=lambda command: RealSkillMotionDispatchReceipt(
            succeeded=True,
            evidence_refs=("unused-dispatch:trace",),
        ),
        hold_controller=lambda reason: RealSkillMotionHoldReceipt(
            succeeded=True,
            evidence_refs=("unused-hold:trace",),
        ),
    )

    def failing_trace(_sample) -> None:
        raise RuntimeError("synthetic trace failure")

    runtime.set_motion_trace_sink(failing_trace)
    runtime.start()

    step = runtime.step(now=2.0)

    assert step.command_present is False
    timing = runtime.control_cycle_snapshot()
    assert timing["motion_trace_error_count"] == 1
    assert "synthetic trace failure" in timing["last_motion_trace_error"]


def test_transient_observation_loss_is_no_dispatch_skip_when_inactive_or_held() -> None:
    builder = _TargetBuilder()
    port = _port(builder)
    dispatched: list[TeleopCommand] = []
    outcomes: list[object] = [
        RealSkillMotionObservationUnavailable("one slow coherent read"),
        _observation_cut(),
        RealSkillMotionObservationUnavailable("one slow read while held"),
    ]

    def observe() -> RealSkillMotionObservationCut:
        value = outcomes.pop(0)
        if isinstance(value, BaseException):
            raise value
        assert isinstance(value, RealSkillMotionObservationCut)
        return value

    runtime = ExclusiveRealSkillControlRuntime(
        motion_port=port,
        observation_provider=observe,
        synchronize_target_builder=builder.synchronize,
        command_admission=lambda command, observation, now: True,
        command_dispatch=lambda command: (
            dispatched.append(command)
            or RealSkillMotionDispatchReceipt(
                succeeded=True,
                evidence_refs=("unexpected-dispatch:transient",),
            )
        ),
        hold_controller=lambda reason: RealSkillMotionHoldReceipt(
            succeeded=True,
            evidence_refs=("rm65-hold:transient",),
        ),
    )
    runtime.start()

    inactive_skip = runtime.step(now=2.0)
    recovered = runtime.step(now=2.01)
    runtime.fail_closed("candidate_review_hold", now=2.02)
    held_skip = runtime.step(now=2.03)

    assert inactive_skip.port_snapshot.phase == RealAgentMotionPhase.INACTIVE
    assert held_skip.port_snapshot.phase == RealAgentMotionPhase.HELD
    assert recovered.command_present is False
    assert dispatched == []
    assert builder.sync_count == 1
    timing = runtime.control_cycle_snapshot()
    assert timing["transient_observation_skip_count"] == 2
    assert timing["consecutive_transient_observation_skips"] == 1
    assert timing["last_transient_observation_error"] == (
        "RealSkillMotionObservationUnavailable: one slow read while held"
    )


def test_transient_observation_loss_propagates_before_pending_activation() -> None:
    builder = _TargetBuilder()
    port = _port(builder)
    dispatched: list[TeleopCommand] = []
    runtime = ExclusiveRealSkillControlRuntime(
        motion_port=port,
        observation_provider=lambda: (_ for _ in ()).throw(
            RealSkillMotionObservationUnavailable("pending cut unavailable")
        ),
        synchronize_target_builder=builder.synchronize,
        command_admission=lambda command, observation, now: True,
        command_dispatch=lambda command: (
            dispatched.append(command)
            or RealSkillMotionDispatchReceipt(
                succeeded=True,
                evidence_refs=("unexpected-dispatch:pending",),
            )
        ),
        hold_controller=lambda reason: RealSkillMotionHoldReceipt(
            succeeded=True,
            evidence_refs=("rm65-hold:pending",),
        ),
    )
    runtime.start()
    port.submit_activation(_activation(suffix="pending-transient"))

    with pytest.raises(
        RealSkillMotionObservationUnavailable,
        match="pending cut unavailable",
    ):
        runtime.step(now=2.0)

    assert dispatched == []
    assert port.snapshot().phase == RealAgentMotionPhase.PENDING_ACTIVATION


def test_transient_observation_loss_propagates_during_active_motion() -> None:
    builder = _TargetBuilder()
    port = _port(builder)
    dispatched: list[TeleopCommand] = []
    observations: list[object] = [
        _observation_cut(),
        RealSkillMotionObservationUnavailable("active cut unavailable"),
    ]

    def observe() -> RealSkillMotionObservationCut:
        value = observations.pop(0)
        if isinstance(value, BaseException):
            raise value
        assert isinstance(value, RealSkillMotionObservationCut)
        return value

    runtime = ExclusiveRealSkillControlRuntime(
        motion_port=port,
        observation_provider=observe,
        synchronize_target_builder=builder.synchronize,
        command_admission=lambda command, observation, now: True,
        command_dispatch=lambda command: (
            dispatched.append(command)
            or RealSkillMotionDispatchReceipt(
                succeeded=True,
                evidence_refs=("rm65-dispatch:active-transient",),
            )
        ),
        hold_controller=lambda reason: RealSkillMotionHoldReceipt(
            succeeded=True,
            evidence_refs=("rm65-hold:active-transient",),
        ),
    )
    runtime.start()
    port.submit_activation(_activation(suffix="active-transient"))
    first = runtime.step(now=2.0)

    with pytest.raises(
        RealSkillMotionObservationUnavailable,
        match="active cut unavailable",
    ):
        runtime.step(now=2.01)

    assert first.port_snapshot.phase == RealAgentMotionPhase.ACTIVE
    assert len(dispatched) == 1


def test_reused_hardware_source_sequence_does_not_publish_fake_measurements() -> None:
    builder = _TargetBuilder()
    port = _port(builder)
    sequence = {"left": 4, "right": 5}

    def cut() -> RealSkillMotionObservationCut:
        return RealSkillMotionObservationCut(
            observation=TeleopObservation(timestamp=2.0),
            qpos_by_side={"left": np.zeros(6), "right": np.zeros(6)},
            measured_at=1.99,
            available_at=2.0,
            evidence_refs=(f"hardware-qpos-cut:{sequence}",),
            source_sequence_by_side=sequence,
        )

    runtime = ExclusiveRealSkillControlRuntime(
        motion_port=port,
        observation_provider=cut,
        synchronize_target_builder=builder.synchronize,
        command_admission=lambda command, observation, now: True,
        command_dispatch=lambda command: RealSkillMotionDispatchReceipt(
            succeeded=True,
            evidence_refs=("unused-dispatch:sequence",),
        ),
        hold_controller=lambda reason: RealSkillMotionHoldReceipt(
            succeeded=True,
            evidence_refs=("unused-hold:sequence",),
        ),
    )
    runtime.start()

    runtime.step(now=2.0)
    first = port.latest_measured_pose()
    runtime.step(now=2.05)
    duplicate = port.latest_measured_pose()
    sequence["left"] += 1
    sequence["right"] += 1
    runtime.step(now=2.1)
    advanced = port.latest_measured_pose()

    assert first is not None
    assert duplicate is first
    assert advanced is not None and advanced.revision == first.revision + 1
    assert builder.sync_count == 2


def test_reference_computed_past_measurement_deadline_is_not_dispatched() -> None:
    class _SlowFirstTargetBuilder(_TargetBuilder):
        def __init__(self) -> None:
            super().__init__()
            self.slow = True

        def build_command_for_target_pair(self, target):
            if self.slow:
                self.slow = False
                time.sleep(0.22)
            return super().build_command_for_target_pair(target)

    builder = _SlowFirstTargetBuilder()
    port = _port(builder)
    dispatched: list[TeleopCommand] = []
    source_sequence = 1

    def cut() -> RealSkillMotionObservationCut:
        now = time.time()
        return RealSkillMotionObservationCut(
            observation=TeleopObservation(timestamp=now),
            qpos_by_side={"left": np.zeros(6), "right": np.zeros(6)},
            measured_at=now,
            available_at=now,
            evidence_refs=(f"hardware-qpos-cut:{source_sequence}",),
            source_sequence_by_side={
                "left": source_sequence,
                "right": source_sequence,
            },
        )

    runtime = ExclusiveRealSkillControlRuntime(
        motion_port=port,
        observation_provider=cut,
        synchronize_target_builder=builder.synchronize,
        command_admission=lambda command, observation, now: True,
        command_dispatch=lambda command: (
            dispatched.append(command)
            or RealSkillMotionDispatchReceipt(
                succeeded=True,
                evidence_refs=("rm65-dispatch:fresh-only",),
            )
        ),
        hold_controller=lambda reason: RealSkillMotionHoldReceipt(
            succeeded=True,
            evidence_refs=("rm65-hold:slow-ik",),
        ),
        clock=time.time,
    )
    runtime.start()
    now = time.time()
    port.submit_activation(
        _activation(
            suffix="slow-ik",
            submitted_at=now,
            expires_at=now + 10.0,
        )
    )

    expired = runtime.step()
    source_sequence += 1
    fresh = runtime.step()

    assert expired.command_present is True
    assert expired.dispatch_attempted is False
    assert expired.dispatch_succeeded is False
    assert fresh.dispatch_succeeded is True
    assert len(dispatched) == 1
    timing = runtime.control_cycle_snapshot()
    assert timing["cycle_count"] == 2
    assert timing["expired_reference_skip_count"] == 1
    assert timing["consecutive_expired_reference_skips"] == 0
    assert timing["max_cycle_sec"] >= 0.2


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
        max_measurement_age_sec=10.0,
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
    assert holds == ["real Agent motion operation operation:test: control_cycle_failed"]


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
        compatibility.RealAgentMotionActivation is canonical.RealAgentMotionActivation
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

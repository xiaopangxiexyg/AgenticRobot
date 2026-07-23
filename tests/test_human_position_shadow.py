from __future__ import annotations

import numpy as np
import pytest

from robot_motion.position_control.audit import PositionCommandAuditEvent
from robot_motion.runtime import EffectorCommand, TeleopCommand
from robot_motion.runtime.dispatch import CommandDispatchReceipt
from teleop_runtime.action.human_position_shadow import (
    bind_human_position_backend_evidence,
)
from teleop_runtime.action.human_proposal import (
    HumanCommandEmission,
    HumanLegacyCommandObservation,
)


def observation(*, target_sequence: int = 7) -> HumanLegacyCommandObservation:
    qpos = np.arange(6, dtype=float) * 0.1
    command = TeleopCommand(timestamp=1.0)
    command.add_command(
        EffectorCommand(
            effector_name="left_arm",
            side="left",
            command_type="joint_vector",
            value=qpos,
            metadata={"position_transport_target_sequence": target_sequence},
        )
    )
    emission = HumanCommandEmission.capture(
        emission_sequence=3,
        emitted_at=1.0,
        contributions=(("position-retargeter", command),),
        merged_command=command,
    )
    return HumanLegacyCommandObservation.capture(
        emission=emission,
        observed_at=1.01,
        outcome="dispatched",
        command_dispatched=True,
        run_state="RUNNING",
        command=command,
        dispatch_receipts={
            "left_arm": CommandDispatchReceipt(
                effector_name="left_arm",
                accepted=True,
                operation="queue_background_joint_target",
                status="queued",
                timestamp=1.005,
                metadata={"position_transport_target_sequence": target_sequence},
            )
        },
    )


def audit_event(
    *,
    target_sequence: int = 7,
    qpos: np.ndarray | None = None,
    accepted: bool | None = None,
) -> PositionCommandAuditEvent:
    target = np.arange(6, dtype=float) * 0.1 if qpos is None else qpos
    return PositionCommandAuditEvent(
        side="left",
        tick_sequence=11,
        timestamp_wall=2.0,
        timestamp_monotonic=1.02,
        scheduled_monotonic_time=1.015,
        wakeup_monotonic_time=1.016,
        deadline_monotonic_time=1.025,
        wakeup_lag_sec=0.001,
        raw_wakeup_lag_sec=0.001,
        skipped_tick_count=0,
        deadline_missed=False,
        target_active=True,
        target_sequence=target_sequence,
        target_source="human:vive:canonical_gui",
        target_timestamp_wall=2.0,
        target_timestamp_monotonic=1.005,
        target_age_sec=0.015,
        segment_start_qpos=np.zeros(6),
        segment_target_qpos=target,
        segment_start_monotonic_time=1.005,
        segment_duration_sec=0.05,
        interpolation_alpha=0.25,
        q_cmd=target * 0.25,
        sdk_attempted=accepted is not None,
        sdk_result_code=(0 if accepted else None),
        sdk_result_text=("OK" if accepted else None),
        sdk_accepted=accepted,
    )


def test_human_position_binding_progresses_without_claiming_canonical_admission() -> None:
    target_only = bind_human_position_backend_evidence(observation())
    assert len(target_only) == 1
    assert target_only[0].status == "target_admitted"
    assert not target_only[0].planner_observed
    assert not target_only[0].anchor_commit_eligible

    planner = bind_human_position_backend_evidence(
        observation(),
        audit_events=(audit_event(),),
    )[0]
    assert planner.status == "planner_observed"
    assert planner.planner_observed
    assert not planner.backend_accepted_observed
    assert planner.non_authoritative

    backend = bind_human_position_backend_evidence(
        observation(),
        audit_events=(audit_event(accepted=True),),
    )[0]
    assert backend.status == "backend_accepted_observed"
    assert backend.backend_accepted_observed
    assert backend.planner_event_hashes
    assert not backend.anchor_commit_eligible


def test_human_position_binding_rejects_sequence_or_target_mismatch() -> None:
    source = observation()
    command = source.to_command()
    emission = HumanCommandEmission.capture(
        emission_sequence=3,
        emitted_at=1.0,
        contributions=(("position-retargeter", command),),
        merged_command=command,
    )
    bad = HumanLegacyCommandObservation.capture(
        emission=emission,
        observed_at=1.01,
        outcome="dispatched",
        command_dispatched=True,
        run_state="RUNNING",
        command=command,
        dispatch_receipts={
            "left_arm": CommandDispatchReceipt(
                effector_name="left_arm",
                accepted=True,
                operation="queue_background_joint_target",
                status="queued",
                metadata={"position_transport_target_sequence": 8},
            )
        },
    )
    with pytest.raises(ValueError, match="sequences disagree"):
        bind_human_position_backend_evidence(bad)
    with pytest.raises(ValueError, match="does not match"):
        bind_human_position_backend_evidence(
            observation(),
            audit_events=(audit_event(qpos=np.ones(6)),),
        )

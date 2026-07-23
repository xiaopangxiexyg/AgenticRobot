from __future__ import annotations

from dataclasses import replace

import numpy as np

from robot_motion.action import BIMANUAL_EE_POSE_V1, RobotAction
from robot_motion.action.arbitration_input import capture_arbitration_input
from robot_motion.action.arbitration_runtime import DecisionScheduleSpec, DecisionScheduler
from robot_motion.action.authority import AuthorityPolicySpec, AuthorityStore
from robot_motion.action.components import ActionComponentCatalog, ActionComponentSpec
from robot_motion.action.proposal import ActionProposalDraft, ProposalBuffer
from robot_motion.action.snapshot import (
    ActionEmissionScheduleSpec,
    build_proposal_snapshot,
    project_proposals,
)
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.execution_gate import ExecutionGateStore
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry


class Clock:
    def __init__(self, value: float = 5.0):
        self.value = value

    def __call__(self) -> float:
        return float(self.value)


def build_runtime(*, proposal_capacity: int = 32):
    clock = Clock()
    coordinator = ControlAdmissionCoordinator(
        "robot-0",
        clock_domain=ClockDomainDescriptor(
            clock_domain_id="control.monotonic",
            runtime_session_id="session-0",
            boot_id="boot-0",
            reset_epoch=0,
        ),
        clock=clock,
    )
    epochs = ProducerEpochRegistry(runtime_session_id="session-0", coordinator=coordinator)
    epochs.activate(
        producer_id="policy-main",
        producer_epoch=1,
        produced_at=1.0,
        produced_clock_domain_id="policy.clock",
        reason="startup",
    )
    catalog = ActionComponentCatalog(
        catalog_id="left-arm-only-v1",
        action_space=BIMANUAL_EE_POSE_V1,
        components=(ActionComponentSpec("left_arm", "left_arm", "left_arm"),),
    )
    proposals = ProposalBuffer(
        runtime_session_id="session-0",
        coordinator=coordinator,
        producer_epochs=epochs,
        capacity=proposal_capacity,
    )
    authority = AuthorityStore(
        runtime_session_id="session-0",
        coordinator=coordinator,
        policy=AuthorityPolicySpec(
            policy_id="policy-first-v1",
            priority_by_group={"left_arm": {"kind:policy": 10, "kind:human": 20}},
            fallback_by_group={"left_arm": "no_action"},
        ),
    )
    gate = ExecutionGateStore(runtime_session_id="session-0", coordinator=coordinator)
    return clock, coordinator, epochs, catalog, proposals, authority, gate


def draft(catalog, *, sequence: int, effective_from: float, effective_until: float):
    return ActionProposalDraft(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        embodiment_instance_id="rm65-pair-0",
        producer_kind="policy",
        producer_id="policy-main",
        producer_epoch=1,
        sequence=sequence,
        action=RobotAction.bimanual_ee_pose(
            left_arm=np.asarray(
                [0.3 + 0.01 * sequence, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0],
                dtype=np.float32,
            ),
            source="policy",
        ),
        produced_at=4.0,
        produced_clock_domain_id="policy.clock",
        effective_from=effective_from,
        effective_until=effective_until,
        target_effective_at=(effective_from + effective_until) / 2.0,
        effective_clock_domain_id="control.monotonic",
        component_schema_hash=catalog.semantic_content_hash,
        component_coverage=("left_arm",),
        selection_groups=("left_arm",),
    )


def test_action_emission_intervals_are_half_open_and_not_pop_time_based() -> None:
    spec = ActionEmissionScheduleSpec(
        spec_id="two-at-30hz",
        start_at=10.0,
        keyframe_period_sec=1.0 / 30.0,
        keyframe_count=2,
        horizon_sec=2.0 / 30.0,
        maximum_hold_sec=1.0 / 30.0,
    )

    assert spec.interval_for(0) == (10.0, 10.0 + 1.0 / 30.0)
    assert spec.interval_for(1) == (10.0 + 1.0 / 30.0, 10.0 + 2.0 / 30.0)


def test_snapshot_uses_target_interval_and_is_input_order_independent() -> None:
    clock, coordinator, epochs, catalog, proposals, _, _ = build_runtime()
    older = proposals.admit(draft(catalog, sequence=0, effective_from=10.0, effective_until=11.0)).proposal
    newer = proposals.admit(draft(catalog, sequence=1, effective_from=10.2, effective_until=11.0)).proposal
    future = proposals.admit(draft(catalog, sequence=2, effective_from=11.0, effective_until=12.0)).proposal
    assert older and newer and future
    clock.value = 10.5
    cut = coordinator.capture_cut(cutoff_at=10.5).cut
    projection = project_proposals(proposals, cut)
    epoch_snapshot = epochs.snapshot(cut)

    first = build_proposal_snapshot(
        projection,
        producer_epochs=epoch_snapshot,
        target_effective_at=10.5,
        component_schema_hash=catalog.semantic_content_hash,
        active_selection_groups=("left_arm",),
    )
    second = build_proposal_snapshot(
        replace(projection, proposals=tuple(reversed(projection.proposals))),
        producer_epochs=epoch_snapshot,
        target_effective_at=10.5,
        component_schema_hash=catalog.semantic_content_hash,
        active_selection_groups=("left_arm",),
    )

    assert first.semantic_content_hash == second.semantic_content_hash
    assert [item.proposal.proposal_id for item in first.candidates] == [newer.proposal_id]
    dispositions = {(item.proposal_id, item.disposition) for item in first.dispositions}
    assert (older.proposal_id, "superseded_before_snapshot") in dispositions
    assert (future.proposal_id, "effective_interval_missed") in dispositions


def test_complete_capture_freezes_one_committed_cut() -> None:
    clock, coordinator, epochs, catalog, proposals, authority, gate = build_runtime()
    proposals.admit(draft(catalog, sequence=0, effective_from=10.0, effective_until=11.0))
    clock.value = 10.5
    scheduler = DecisionScheduler(
        DecisionScheduleSpec(spec_id="30hz", decision_hz=30.0), clock=clock
    )

    snapshot = capture_arbitration_input(
        coordinator=coordinator,
        proposal_buffer=proposals,
        producer_epochs=epochs,
        authority_store=authority,
        execution_gate_store=gate,
        scheduler=scheduler,
        embodiment_instance_id="rm65-pair-0",
        component_catalog=catalog,
        active_selection_groups=("left_arm",),
    )

    assert snapshot is not None
    cutoff = snapshot.decision_schedule_tick.cutoff_admission_sequence
    assert snapshot.proposal_snapshot.cutoff_admission_sequence == cutoff
    assert snapshot.authority_state_snapshot.cutoff_admission_sequence == cutoff
    assert snapshot.execution_gate_snapshot.cutoff_admission_sequence == cutoff
    assert snapshot.producer_epoch_snapshot.cutoff_admission_sequence == cutoff


def test_high_rate_proposals_are_causally_reduced_without_promising_one_to_one_selection() -> None:
    clock, coordinator, epochs, catalog, proposals, _, _ = build_runtime(
        proposal_capacity=600
    )
    for sequence in range(500):
        assert proposals.admit(
            draft(
                catalog,
                sequence=sequence,
                effective_from=10.0,
                effective_until=11.0,
            )
        ).accepted
    clock.value = 10.5
    cut = coordinator.capture_cut(cutoff_at=10.5).cut

    snapshot = build_proposal_snapshot(
        project_proposals(proposals, cut),
        producer_epochs=epochs.snapshot(cut),
        target_effective_at=10.5,
        component_schema_hash=catalog.semantic_content_hash,
        active_selection_groups=("left_arm",),
    )

    assert len(snapshot.candidates) == 1
    assert snapshot.candidates[0].proposal.sequence == 499
    assert sum(
        item.disposition == "superseded_before_snapshot"
        for item in snapshot.dispositions
    ) == 499

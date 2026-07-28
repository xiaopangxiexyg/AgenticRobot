from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

import numpy as np
import pytest

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import (
    ActionComponentCatalog,
    ActionComponentSpec,
    ArmActionSpec,
    BIMANUAL_EE_POSE_V1,
    RobotAction,
    build_robot_action_v1_runtime_schema,
    project_robot_action_v1_to_component_action,
)
from robot_motion.action.proposal import (
    ACTION_PROPOSAL_SEMANTIC_HASH_DOMAIN_V3,
    ActionProposalDraft,
    ProposalBuffer,
    action_proposal_semantic_payload,
)
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry


class _Clock:
    def __init__(self, value: float = 10.0):
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


def _runtime(*, capacity: int = 8) -> tuple[_Clock, ControlAdmissionCoordinator, ProducerEpochRegistry, ProposalBuffer]:
    clock = _Clock()
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
        produced_clock_domain_id="policy.monotonic",
        reason="startup",
    )
    buffer = ProposalBuffer(
        runtime_session_id="session-0",
        coordinator=coordinator,
        producer_epochs=epochs,
        capacity=capacity,
    )
    return clock, coordinator, epochs, buffer


def _action(*, metadata: dict[str, Any] | None = None) -> RobotAction:
    arm_spec = ArmActionSpec.tool_tcp("left", reference_frame="left_arm_base")
    return RobotAction.bimanual_ee_pose(
        left_arm=np.asarray([0.3, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        timestamp=2.0,
        source="policy",
        metadata=(
            {"arm_action_specs_by_side": {"left": arm_spec.to_metadata()}}
            if metadata is None
            else metadata
        ),
    )


def _action_schema():
    arm_spec = ArmActionSpec.tool_tcp("left", reference_frame="left_arm_base")
    return build_robot_action_v1_runtime_schema(
        component_catalog=ActionComponentCatalog(
            catalog_id="proposal-contract-left-arm",
            action_space=BIMANUAL_EE_POSE_V1,
            components=(
                ActionComponentSpec(
                    "left_arm",
                    "left_arm",
                    "left_arm",
                ),
            ),
        ),
        arm_action_specs_by_side={"left": arm_spec},
    )


def _draft(*, sequence: int, action: RobotAction | None = None, audit: dict[str, Any] | None = None, epoch: int = 1) -> ActionProposalDraft:
    source_action = _action() if action is None else action
    return ActionProposalDraft(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        embodiment_instance_id="rm65-pair-0",
        producer_kind="policy",
        producer_id="policy-main",
        producer_epoch=epoch,
        sequence=sequence,
        action=project_robot_action_v1_to_component_action(
            source_action,
            action_schema=_action_schema(),
        ),
        produced_at=3.0,
        produced_clock_domain_id="policy.monotonic",
        effective_from=4.0 + sequence,
        effective_until=5.0 + sequence,
        target_effective_at=4.5 + sequence,
        effective_clock_domain_id="control.monotonic",
        policy_node_id="policy-main",
        observation_cut_id="cut-7",
        inference_id="inference-2",
        policy_output_id="output-2",
        chunk_id="chunk-2",
        chunk_keyframe_index=sequence,
        audit_metadata={} if audit is None else audit,
    )


def test_receiver_admission_detaches_and_deep_freezes_action() -> None:
    _, coordinator, _, buffer = _runtime()
    producer_action = _action()
    receipt = buffer.admit(_draft(sequence=0, action=producer_action, audit={"debug": [1, 2]}))

    assert receipt.accepted
    assert receipt.available_at == 10.0
    assert receipt.available_clock_domain_id == "control.monotonic"
    assert receipt.control_admission_sequence == 2  # epoch activation is the first domain fact
    assert receipt.proposal is not None
    admitted = receipt.proposal
    producer_action.arm_pose_by_side["left"][0] = 99.0
    producer_action.metadata["arm_action_specs_by_side"]["left"]["reference_frame"] = "mutated"
    assert admitted.action.value("left_arm").values[0] == pytest.approx(0.3)
    assert (
        admitted.action.value("left_arm")
        .payload_schema.semantic_parameters["arm_action_spec"]["reference_frame"]
        == "left_arm_base"
    )
    with pytest.raises(ValueError):
        admitted.action.value("left_arm").values[0] = 1.0
    with pytest.raises(ValueError):
        admitted.action.value("left_arm").values.setflags(write=True)
    with pytest.raises(TypeError):
        admitted.action.values_by_component["new"] = "not allowed"
    assert coordinator.committed_high_watermark == 2


def test_draft_cannot_claim_receiver_availability_or_persisted_record_integrity() -> None:
    draft_fields = {field.name for field in fields(ActionProposalDraft)}
    assert "available_at" not in draft_fields
    _, _, _, buffer = _runtime()
    proposal = buffer.admit(_draft(sequence=0)).proposal
    assert proposal is not None
    assert proposal.semantic_hash_domain == ACTION_PROPOSAL_SEMANTIC_HASH_DOMAIN_V3
    assert not hasattr(proposal, "record_integrity_hash")


def test_audit_metadata_does_not_change_semantic_hash() -> None:
    _, _, _, first_buffer = _runtime()
    _, _, _, second_buffer = _runtime()
    first = first_buffer.admit(_draft(sequence=0, audit={"logger": "one"}))
    second = second_buffer.admit(_draft(sequence=0, audit={"logger": "two"}))
    assert first.accepted and second.accepted
    assert first.semantic_content_hash == second.semantic_content_hash


def test_runtime_and_dataset_projection_share_one_semantic_hash() -> None:
    _, _, _, buffer = _runtime()
    proposal = buffer.admit(_draft(sequence=0)).proposal
    assert proposal is not None
    dataset_like_payload = action_proposal_semantic_payload(proposal)
    dataset_like_payload["action"] = dict(dataset_like_payload["action"])
    assert canonical_content_hash(
        dataset_like_payload,
        domain=ACTION_PROPOSAL_SEMANTIC_HASH_DOMAIN_V3,
    ) == proposal.semantic_content_hash


def test_non_action_v1_metadata_does_not_enter_production_proposal() -> None:
    _, coordinator, _, buffer = _runtime()
    action = _action(metadata={"selected_owner": "policy"})
    receipt = buffer.admit(_draft(sequence=0, action=action))
    assert receipt.accepted
    assert receipt.proposal is not None
    assert "selected_owner" not in (
        receipt.proposal.action.action_schema.semantic_parameters
    )
    assert coordinator.committed_high_watermark == 2
    assert len(buffer) == 1


def test_effective_time_must_use_receiver_control_clock_domain() -> None:
    _, coordinator, _, buffer = _runtime()
    receipt = buffer.admit(
        replace(_draft(sequence=0), effective_clock_domain_id="remote-policy.monotonic")
    )
    assert not receipt.accepted
    assert receipt.rejection_code == "clock_domain_mismatch"
    assert receipt.available_at is None
    assert coordinator.committed_high_watermark == 1


def test_buffer_fails_closed_on_duplicate_rollback_capacity_and_inactive_epoch() -> None:
    _, coordinator, epochs, buffer = _runtime(capacity=2)
    first = buffer.admit(_draft(sequence=2))
    duplicate = buffer.admit(_draft(sequence=2))
    rollback = buffer.admit(_draft(sequence=1))
    second = buffer.admit(_draft(sequence=3))
    full = buffer.admit(_draft(sequence=4))

    assert first.accepted and second.accepted
    assert duplicate.rejection_code == "duplicate"
    assert rollback.rejection_code == "sequence_rollback"
    assert full.rejection_code == "capacity_exhausted"
    assert coordinator.committed_high_watermark == 3

    epochs.activate(
        producer_id="policy-main",
        producer_epoch=2,
        produced_at=8.0,
        produced_clock_domain_id="policy.monotonic",
        reason="restart",
    )
    old_epoch = buffer.admit(_draft(sequence=5, epoch=1))
    assert old_epoch.rejection_code == "inactive_producer_epoch"
    cut = coordinator.capture_cut(cutoff_at=10.0).cut
    assert buffer.records_for_active_epochs(cut) == ()


def test_each_chunk_keyframe_has_independent_fact_identity_and_shared_lineage() -> None:
    _, coordinator, _, buffer = _runtime()
    first = buffer.admit(_draft(sequence=0)).proposal
    second = buffer.admit(_draft(sequence=1)).proposal
    assert first is not None and second is not None
    assert first.proposal_id != second.proposal_id
    assert first.chunk_id == second.chunk_id == "chunk-2"
    assert first.inference_id == second.inference_id == "inference-2"
    assert (first.chunk_keyframe_index, second.chunk_keyframe_index) == (0, 1)
    cut = coordinator.capture_cut(cutoff_at=10.0).cut
    assert buffer.records_at_or_before(cut) == (first, second)

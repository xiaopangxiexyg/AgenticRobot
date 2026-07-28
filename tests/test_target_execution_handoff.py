from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from robot_motion.action import (
    ActionAdaptationResult,
    ComponentExecutionTarget,
    RobotAction,
    RobotActionV1RepresentationAdapter,
    TargetExecutionRequest,
    TargetExecutionCompletionReceipt,
    TargetProcessingFactReference,
    admit_component_execution_target,
    build_action_adaptation_request,
    build_component_execution_target,
    build_component_execution_target_activation_receipt,
    build_target_execution_completion_receipt,
    build_target_execution_status_snapshot,
    build_robot_action_v1_target_capability,
    build_target_execution_submission_receipt,
    project_robot_action_v1_to_component_action,
    ROBOT_ACTION_V1_TARGET_ADAPTER_CONTRACT_HASH,
    ROBOT_ACTION_V1_TARGET_ADAPTER_CONTRACT_ID,
)


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _adaptation() -> ActionAdaptationResult:
    action = project_robot_action_v1_to_component_action(
        RobotAction(
            action_space="bimanual_ee_pose_v1",
            arm_pose_by_side={
                "right": np.asarray(
                    [0.3, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0],
                    dtype=np.float32,
                ),
            },
            active_effectors=("right_arm",),
            frame_by_effector={"right_arm": "robot_base"},
            controlled_frame_by_effector={"right_arm": "tool_tcp"},
            source="fixture",
        )
    )
    capability = build_robot_action_v1_target_capability(
        capability_id="fixture-target",
        action_schema=action.action_schema,
        handler_contract_id=ROBOT_ACTION_V1_TARGET_ADAPTER_CONTRACT_ID,
        handler_contract_hash=ROBOT_ACTION_V1_TARGET_ADAPTER_CONTRACT_HASH,
        output_target_schema_hash="fixture-adapted-schema",
    )
    request = build_action_adaptation_request(
        authoritative_decision_id="decision-1",
        authoritative_decision_hash="decision-hash-1",
        arbitration_result_id="arbitration-result-1",
        candidate_action=action,
        component_target_capability=capability,
        target_effective_at=12.5,
        supported_until=13.0,
        authority_sequence=3,
        execution_gate_sequence=4,
        execution_fence_sequence=5,
        requested_at=12.0,
        clock_domain_id="control.monotonic",
    )
    return RobotActionV1RepresentationAdapter(
        receiver_id="fixture-adapter",
        component_target_capability=capability,
        clock=lambda: 12.1,
    ).adapt(request)


def _request() -> TargetExecutionRequest:
    return TargetExecutionRequest(
        control_domain_id="robot-0",
        target_handler_id="fixture-target-handler",
        adaptation_result=_adaptation(),
        submitted_at=12.2,
        clock_domain_id="control.monotonic",
    )


def _submission(request: TargetExecutionRequest):
    return build_target_execution_submission_receipt(
        request,
        received_at=12.21,
        accepted_for_processing=True,
        processing_token="work-17",
    )


def test_target_request_binds_authority_adaptation_and_fences() -> None:
    request = _request()
    adaptation = request.adaptation_result

    assert request.adapted_candidate is adaptation.adapted_candidate
    assert request.clock_domain_id == adaptation.request.clock_domain_id
    assert request.supported_until == 13.0
    assert request.request_id.startswith("target-execution-request:")


def test_rejected_adaptation_cannot_be_handed_to_target() -> None:
    accepted = _adaptation()
    rejected_receipt = replace(
        accepted.receipt,
        accepted=False,
        rejection_code="representation_rejected",
        adapted_candidate_id=None,
        adapted_candidate_hash=None,
    )
    rejected = replace(
        accepted,
        adapted_candidate=None,
        receipt=rejected_receipt,
    )

    with pytest.raises(ValueError, match="accepted adaptation"):
        TargetExecutionRequest(
            control_domain_id="robot-0",
            target_handler_id="fixture-target-handler",
            adaptation_result=rejected,
            submitted_at=12.2,
            clock_domain_id="control.monotonic",
        )


def test_submission_receipt_is_not_target_admission_or_motion_permission() -> None:
    request = _request()
    receipt = build_target_execution_submission_receipt(
        request,
        received_at=12.21,
        accepted_for_processing=True,
        processing_token="work-17",
    )

    assert receipt.accepted_for_processing is True
    assert receipt.processing_token == "work-17"
    assert not hasattr(receipt, "target_id")
    assert not hasattr(receipt, "dispatchable")
    assert not hasattr(receipt, "hold")


def test_expired_target_request_cannot_be_accepted_or_scheduled() -> None:
    request = _request()
    with pytest.raises(ValueError, match="cannot be accepted"):
        build_target_execution_submission_receipt(
            request,
            received_at=request.supported_until,
            accepted_for_processing=True,
            processing_token="late-work",
        )

    submission = _submission(request)
    with pytest.raises(ValueError, match="scheduled after support"):
        build_component_execution_target(
            request,
            submission,
            component_dependencies=("right_arm",),
            output_target_schema_hash="fixture-qpos-target-schema",
            target_kind="fixture.qpos_target",
            resolved_at=12.3,
            temporal_semantics="point_target",
            scheduled_for=request.supported_until,
            scheduled_until=None,
            payload={},
        )
    with pytest.raises(ValueError, match="cannot exceed upstream support"):
        build_component_execution_target(
            request,
            submission,
            component_dependencies=("right_arm",),
            output_target_schema_hash="fixture-qpos-target-schema",
            target_kind="fixture.qpos_target",
            resolved_at=12.3,
            temporal_semantics="scheduled_interval",
            scheduled_for=12.5,
            scheduled_until=13.1,
            payload={},
        )


def test_target_requires_the_accepted_receipt_for_its_exact_request() -> None:
    request = _request()
    rejected = build_target_execution_submission_receipt(
        request,
        received_at=12.21,
        accepted_for_processing=False,
        rejection_code="handler_busy",
    )

    with pytest.raises(ValueError, match="accepted receipt"):
        build_component_execution_target(
            request,
            rejected,
            component_dependencies=("right_arm",),
            output_target_schema_hash="fixture-qpos-target-schema",
            target_kind="fixture.qpos_target",
            resolved_at=12.3,
            temporal_semantics="point_target",
            scheduled_for=12.5,
            scheduled_until=None,
            payload={},
        )


def test_accepted_submission_has_terminal_failure_receipt() -> None:
    request = _request()
    submission = _submission(request)
    completion = build_target_execution_completion_receipt(
        request,
        submission,
        status="failed",
        completed_at=12.4,
        available_at=12.41,
        outcome_code="ik_timeout",
    )

    assert isinstance(completion, TargetExecutionCompletionReceipt)
    assert completion.status == "failed"
    assert completion.target_id is None
    assert completion.processing_token == submission.processing_token


def test_resolved_completion_requires_the_exact_target() -> None:
    request = _request()
    submission = _submission(request)
    target = build_component_execution_target(
        request,
        submission,
        component_dependencies=("right_arm",),
        output_target_schema_hash="fixture-qpos-target-schema",
        target_kind="fixture.qpos_target",
        resolved_at=12.3,
        temporal_semantics="point_target",
        scheduled_for=12.5,
        scheduled_until=None,
        payload={},
    )
    completion = build_target_execution_completion_receipt(
        request,
        submission,
        status="resolved",
        completed_at=12.31,
        available_at=12.32,
        target=target,
    )

    assert completion.status == "resolved"
    assert completion.target_id == target.target_id
    assert completion.target_hash == target.semantic_content_hash


def test_component_target_has_no_universal_safe_action_claim() -> None:
    request = _request()
    safety_fact = TargetProcessingFactReference(
        stage="fixture.private_workspace_check",
        fact_id="workspace-check-1",
        fact_hash="workspace-check-hash-1",
        status="accepted",
        processing_started_at=12.22,
        processing_completed_at=12.23,
        clock_domain_id="control.monotonic",
    )
    target = build_component_execution_target(
        request,
        _submission(request),
        component_dependencies=("right_arm",),
        output_target_schema_hash="fixture-qpos-target-schema",
        target_kind="fixture.qpos_target",
        resolved_at=12.3,
        temporal_semantics="point_target",
        scheduled_for=12.5,
        scheduled_until=None,
        payload={"right_arm": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
        processing_fact_references=(safety_fact,),
    )

    assert isinstance(target, ComponentExecutionTarget)
    assert target.adapted_candidate_id == request.adapted_candidate.adapted_candidate_id
    assert target.execution_fence_sequence == 5
    assert target.component_dependencies == ("right_arm",)
    assert target.processing_fact_references == (safety_fact,)
    assert not hasattr(target, "safe_action")
    assert not hasattr(target, "source_safety_result_id")


def test_target_dependencies_must_be_covered_by_authoritative_candidate() -> None:
    request = _request()
    with pytest.raises(ValueError, match="covered by the candidate"):
        build_component_execution_target(
            request,
            _submission(request),
            component_dependencies=("other_arm",),
            output_target_schema_hash="fixture-qpos-target-schema",
            target_kind="fixture.qpos_target",
            resolved_at=12.3,
            temporal_semantics="point_target",
            scheduled_for=12.5,
            scheduled_until=None,
            payload={},
        )


def test_scheduled_interval_and_target_admission_remain_separate() -> None:
    request = _request()
    target = build_component_execution_target(
        request,
        _submission(request),
        component_dependencies=("right_arm",),
        output_target_schema_hash="fixture-trajectory-schema",
        target_kind="fixture.trajectory",
        resolved_at=12.3,
        temporal_semantics="scheduled_interval",
        scheduled_for=12.5,
        scheduled_until=12.6,
        payload={"trajectory_id": "trajectory-1"},
    )
    admission = admit_component_execution_target(
        target,
        receiver_id="fixture-target-mailbox",
        admission_sequence=8,
        admitted_at=12.31,
        available_at=12.32,
        accepted=True,
        admission_token="target-slot-8",
    )

    assert admission.target_id == target.target_id
    assert admission.execution_fence_sequence == target.execution_fence_sequence
    assert admission.accepted is True
    assert not hasattr(admission, "activated_at")
    assert not hasattr(admission, "dispatch_receipt_id")


def test_receiver_may_reject_without_inventing_hold_or_safety() -> None:
    request = _request()
    target = build_component_execution_target(
        request,
        _submission(request),
        component_dependencies=("right_arm",),
        output_target_schema_hash="fixture-qpos-target-schema",
        target_kind="fixture.qpos_target",
        resolved_at=12.3,
        temporal_semantics="point_target",
        scheduled_for=12.5,
        scheduled_until=None,
        payload={},
    )
    admission = admit_component_execution_target(
        target,
        receiver_id="fixture-target-mailbox",
        admission_sequence=9,
        admitted_at=12.31,
        available_at=12.31,
        accepted=False,
        rejection_code="receiver_busy",
    )

    assert admission.accepted is False
    assert admission.rejection_code == "receiver_busy"
    assert not hasattr(admission, "hold")
    assert not hasattr(admission, "safe")


def test_activation_receipt_is_separate_from_target_admission() -> None:
    request = _request()
    target = build_component_execution_target(
        request,
        _submission(request),
        component_dependencies=("right_arm",),
        output_target_schema_hash="fixture-qpos-target-schema",
        target_kind="fixture.qpos_target",
        resolved_at=12.3,
        temporal_semantics="point_target",
        scheduled_for=12.5,
        scheduled_until=None,
        payload={},
    )
    admission = admit_component_execution_target(
        target,
        receiver_id="fixture-target-mailbox",
        admission_sequence=10,
        admitted_at=12.31,
        available_at=12.31,
        accepted=True,
        admission_token="target-slot-10",
    )
    activation = build_component_execution_target_activation_receipt(
        target,
        admission,
        receiver_id="fixture-target-worker",
        activation_sequence=11,
        activation_status="activated",
        activation_at=12.5,
        available_at=12.501,
        effective_from=12.5,
        effective_until=12.6,
    )

    assert activation.target_admission_receipt_id == admission.receipt_id
    assert activation.effective_from == 12.5
    assert activation.effective_until == 12.6
    assert not hasattr(activation, "dispatch_receipt_id")


def test_activation_rejection_has_no_effective_interval() -> None:
    request = _request()
    target = build_component_execution_target(
        request,
        _submission(request),
        component_dependencies=("right_arm",),
        output_target_schema_hash="fixture-qpos-target-schema",
        target_kind="fixture.qpos_target",
        resolved_at=12.3,
        temporal_semantics="point_target",
        scheduled_for=12.5,
        scheduled_until=None,
        payload={},
    )
    admission = admit_component_execution_target(
        target,
        receiver_id="fixture-target-mailbox",
        admission_sequence=12,
        admitted_at=12.31,
        available_at=12.31,
        accepted=True,
        admission_token="target-slot-12",
    )
    activation = build_component_execution_target_activation_receipt(
        target,
        admission,
        receiver_id="fixture-target-worker",
        activation_sequence=13,
        activation_status="rejected",
        activation_at=12.5,
        available_at=12.51,
        rejection_code="stale_execution_fence",
    )

    assert activation.activation_status == "rejected"
    assert activation.effective_from is None


def test_status_snapshot_keeps_one_request_chain_and_exact_stage_ids() -> None:
    request = _request()
    submission = _submission(request)
    target = build_component_execution_target(
        request,
        submission,
        component_dependencies=("right_arm",),
        output_target_schema_hash="fixture-qpos-target-schema",
        target_kind="fixture.qpos_target",
        resolved_at=12.3,
        temporal_semantics="point_target",
        scheduled_for=12.5,
        scheduled_until=None,
        payload={},
    )
    completion = build_target_execution_completion_receipt(
        request,
        submission,
        status="resolved",
        completed_at=12.31,
        available_at=12.32,
        target=target,
    )
    admission = admit_component_execution_target(
        target,
        receiver_id="fixture-target-mailbox",
        admission_sequence=14,
        admitted_at=12.33,
        available_at=12.34,
        accepted=True,
        admission_token="target-slot-14",
    )
    activation = build_component_execution_target_activation_receipt(
        target,
        admission,
        receiver_id="fixture-target-worker",
        activation_sequence=15,
        activation_status="activated",
        activation_at=12.5,
        available_at=12.501,
        effective_from=12.5,
        effective_until=12.6,
    )

    pending = build_target_execution_status_snapshot(
        request,
        activation_mode="required",
        captured_at=12.4,
        submission_receipt=submission,
        completion_receipt=completion,
        target_admission=admission,
    )
    assert pending.request_id == request.request_id
    assert pending.request.status == "submitted"
    assert pending.request.record_id == request.request_id
    assert pending.submission.status == "accepted_for_processing"
    assert pending.submission.record_id == submission.receipt_id
    assert pending.completion.status == "resolved"
    assert pending.completion.record_id == completion.receipt_id
    assert pending.admission.status == "accepted"
    assert pending.admission.record_id == admission.receipt_id
    assert pending.activation.status == "pending"
    assert pending.activation.record_id is None

    activated = build_target_execution_status_snapshot(
        request,
        activation_mode="required",
        captured_at=12.51,
        submission_receipt=submission,
        completion_receipt=completion,
        target_admission=admission,
        target_activation=activation,
    )
    assert activated.target_id == target.target_id
    assert activated.activation.status == "activated"
    assert activated.activation.record_id == activation.activation_id
    assert activated.snapshot_id != pending.snapshot_id


def test_status_snapshot_marks_rejected_submission_tail_not_applicable() -> None:
    request = _request()
    submission = build_target_execution_submission_receipt(
        request,
        received_at=12.21,
        accepted_for_processing=False,
        rejection_code="handler_busy",
    )

    snapshot = build_target_execution_status_snapshot(
        request,
        activation_mode="required",
        captured_at=12.3,
        submission_receipt=submission,
    )

    assert snapshot.submission.status == "rejected"
    assert snapshot.completion.status == "not_applicable"
    assert snapshot.admission.status == "not_applicable"
    assert snapshot.activation.status == "not_applicable"


def test_status_snapshot_marks_omitted_activation_not_applicable() -> None:
    request = _request()

    snapshot = build_target_execution_status_snapshot(
        request,
        activation_mode="omitted",
        captured_at=12.3,
    )

    assert snapshot.submission.status == "pending"
    assert snapshot.completion.status == "pending"
    assert snapshot.admission.status == "pending"
    assert snapshot.activation.status == "not_applicable"

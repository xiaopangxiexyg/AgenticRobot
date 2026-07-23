from __future__ import annotations

import numpy as np
import pytest

from robot_motion.position_control import ResampledMotionReference
from teleop_runtime.position_control.reference_observation import (
    LEGACY_POSITION_GENERATION_NAMESPACE_V1,
    LegacyPositionReferenceObservationBatch,
    LegacyPositionReferenceObserverRuntime,
    LegacyPositionReferenceResolutionFailureObservation,
    LegacyPositionReferenceResolutionObservation,
    LegacyPositionReferenceSegmentTerminalObservation,
    LegacyReferenceDerivation,
    LegacyReferenceGenerationInvalidation,
    LegacyReferenceResolutionDerivation,
    LegacyReferenceResolutionFailureDerivation,
    LegacyReferenceSourceFact,
    build_execution_context_binding_evidence,
    build_legacy_reference_observation_batch,
    build_legacy_reference_parent_provenance,
    build_legacy_reference_resolution_observation,
    build_legacy_reference_resolution_failure_observation,
    build_legacy_reference_segment_terminal_observation,
    project_legacy_reference_segment,
)


def _source(index: int, timestamp: float, value: float) -> LegacyReferenceSourceFact:
    return LegacyReferenceSourceFact(
        source_index=index,
        source_timestamp=timestamp,
        source_stage=(
            "current_observation_anchor"
            if index == 0
            else "safe_15hz_keyframe"
        ),
        target_payload_hash=f"payload:{value}",
    )


def _reference(
    *,
    sequence: int = 0,
    left_index: int = 0,
    right_index: int = 1,
    left_time: float = 10.0,
    right_time: float = 10.1,
    alpha: float = 0.25,
    mode: str = "causal_delayed_interpolation",
) -> ResampledMotionReference:
    return ResampledMotionReference(
        sequence=sequence,
        timestamp=left_time + alpha * (right_time - left_time),
        left_source_index=left_index,
        right_source_index=right_index,
        left_source_timestamp=left_time,
        right_source_timestamp=right_time,
        interpolation_alpha=alpha,
        qpos_by_side={"left": np.full(6, alpha)},
        metadata={"resampling_mode": mode},
    )


def _derivation(*, sequence: int = 0, alpha: float = 0.25) -> LegacyReferenceDerivation:
    return LegacyReferenceDerivation(
        legacy_worker_generation=3,
        modality="qpos",
        scheduled_for=10.0 + sequence / 60.0,
        emitted_at=10.01 + sequence / 60.0,
        reference=_reference(sequence=sequence, alpha=alpha),
        start_source_fact=_source(0, 10.0, 0.0),
        end_source_fact=_source(1, 10.1, 1.0),
        interpolation_contract_hash="linear-qpos-15-to-60-v1",
    )


def test_legacy_observation_is_non_authoritative_and_exact_hold_is_not_segment() -> None:
    interpolated = build_legacy_reference_observation_batch(
        _derivation(),
        worker_instance_id="worker:0",
        executor_instance_id="executor:0",
        include_segment_observation=True,
    )
    segment = interpolated.segment_observation
    assert segment is not None
    assert segment.legacy_generation_namespace == (
        LEGACY_POSITION_GENERATION_NAMESPACE_V1
    )
    assert segment.observation_authority == "legacy_observed"
    assert segment.non_authoritative
    assert not segment.anchor_commit_eligible
    assert tuple(item.role for item in segment.endpoint_observations) == (
        "start",
        "end",
    )
    assert interpolated.frame_observation.source_segment_observation_id == (
        segment.observation_id
    )
    assert interpolated.frame_observation.observation_stage == (
        "resampled_pre_resolver"
    )

    exact_source = _source(1, 10.1, 1.0)
    exact = LegacyReferenceDerivation(
        legacy_worker_generation=3,
        modality="qpos",
        scheduled_for=10.1,
        emitted_at=10.11,
        reference=_reference(
            left_index=1,
            right_index=1,
            left_time=10.1,
            right_time=10.1,
            alpha=0.0,
            mode="causal_exact",
        ),
        start_source_fact=exact_source,
        end_source_fact=exact_source,
        interpolation_contract_hash="linear-qpos-15-to-60-v1",
    )
    exact_batch = build_legacy_reference_observation_batch(
        exact,
        worker_instance_id="worker:0",
        executor_instance_id="executor:0",
        include_segment_observation=False,
    )
    assert exact_batch.segment_observation is None
    assert exact_batch.frame_observation.derivation_kind == "exact_keyframe"
    assert exact_batch.frame_observation.source_segment_observation_id is None

    with pytest.raises(ValueError, match="cannot create a segment"):
        build_legacy_reference_observation_batch(
            exact,
            worker_instance_id="worker:0",
            executor_instance_id="executor:0",
            include_segment_observation=True,
        )


def test_observer_emits_one_segment_per_parent_pair_and_all_frames() -> None:
    batches = []
    observer = LegacyPositionReferenceObserverRuntime(
        worker_instance_id="worker:0",
        executor_instance_id="executor:0",
        sink=batches.append,
        queue_capacity=8,
    )
    try:
        assert observer.submit_nowait(_derivation(sequence=0, alpha=0.25))
        assert observer.submit_nowait(_derivation(sequence=1, alpha=0.5))
        observer.flush()
    finally:
        observer.close()

    assert len(batches) == 2
    assert batches[0].segment_observation is not None
    assert batches[1].segment_observation is None
    assert (
        batches[0].frame_observation.source_segment_observation_id
        == batches[1].frame_observation.source_segment_observation_id
    )
    assert observer.drop_count == 0
    assert observer.error_count == 0


def test_bound_projection_keeps_context_only_distinct_from_enforcement() -> None:
    raw = build_legacy_reference_observation_batch(
        _derivation(),
        worker_instance_id="worker:0",
        executor_instance_id="executor:0",
        include_segment_observation=True,
    ).segment_observation
    assert raw is not None
    context = build_execution_context_binding_evidence(
        control_domain_id="robot:0",
        selection_group_id="arms",
        execution_epoch=8,
        group_segment_generation=12,
        binding_source="legacy-frontend-shadow-v1",
        bound_at=10.02,
        enforcement_level="context_only",
    )
    parents = tuple(
        build_legacy_reference_parent_provenance(
            role=item.role,
            modality=item.modality,
            source_index=item.source_index,
            source_kind=(
                "measured_rebase"
                if item.source_stage == "current_observation_anchor"
                else "legacy_post_safety_keyframe"
            ),
            source_fact_id=f"legacy-keyframe:{item.source_index}",
            action_content_hash=item.target_payload_hash,
            action_schema_hash="bimanual-qpos-v1",
            frame_schema_hash="joint-order-v1",
            target_effective_at=item.source_timestamp,
        )
        for item in raw.endpoint_observations
    )
    bound = project_legacy_reference_segment(
        raw,
        execution_context=context,
        parent_provenance=parents,
    )
    assert bound.execution_context_enforcement == "context_only"
    assert bound.admission_semantics == "observed_legacy"
    assert bound.admission_receipt_id is None
    assert bound.non_authoritative
    assert not bound.anchor_commit_eligible

    mismatched = list(parents)
    mismatched[1] = build_legacy_reference_parent_provenance(
        role="end",
        modality="qpos",
        source_index=99,
        source_kind="legacy_post_safety_keyframe",
        source_fact_id="legacy-keyframe:99",
        action_content_hash=parents[1].action_content_hash,
        action_schema_hash="bimanual-qpos-v1",
        frame_schema_hash="joint-order-v1",
        target_effective_at=parents[1].target_effective_at,
    )
    with pytest.raises(ValueError, match="source index mismatch"):
        project_legacy_reference_segment(
            raw,
            execution_context=context,
            parent_provenance=tuple(mismatched),
        )


def test_checked_binding_requires_guard_evidence() -> None:
    with pytest.raises(ValueError, match="guard evidence"):
        build_execution_context_binding_evidence(
            control_domain_id="robot:0",
            selection_group_id="arms",
            execution_epoch=8,
            group_segment_generation=12,
            binding_source="legacy-frontend-shadow-v1",
            bound_at=10.02,
            enforcement_level="pre_backend_checked",
        )


def _resolution(
    *,
    derivation: LegacyReferenceDerivation | None = None,
    planner: bool = False,
) -> LegacyReferenceResolutionDerivation:
    return LegacyReferenceResolutionDerivation(
        reference_derivation=_derivation() if derivation is None else derivation,
        resolved_at=10.02,
        resolver_evidence_hash="resolver-hash",
        planner_evidence_status=(
            "legacy_collision_planner_accepted" if planner else "not_applicable"
        ),
        planner_input_hash=("planner-input" if planner else None),
        planner_output_hash=("planner-output" if planner else None),
        planner_evidence_hash=("planner-evidence" if planner else None),
        measured_state_evidence_status=(
            "legacy_planner_state_not_typed" if planner else "not_applicable"
        ),
        measured_state_evidence_ids=(),
        command_intent_hash="command-intent-hash",
        local_pre_backend_generation_check_passed=True,
        local_pre_backend_check_observed_at=10.02,
    )


def test_post_resolver_observation_does_not_claim_m1c_or_backend_acceptance() -> None:
    resolution = build_legacy_reference_resolution_observation(
        _resolution(planner=True),
        worker_instance_id="worker:0",
        executor_instance_id="executor:0",
    )
    assert resolution.observation_stage == "resolved_pre_dispatch"
    assert resolution.planner_evidence_status == (
        "legacy_collision_planner_accepted"
    )
    assert resolution.measured_state_evidence_status == (
        "legacy_planner_state_not_typed"
    )
    assert resolution.execution_context_enforcement == (
        "executor_local_generation_only"
    )
    assert not resolution.backend_accepted
    assert resolution.admission_receipt_id is None
    assert resolution.non_authoritative
    assert not resolution.anchor_commit_eligible


def test_resolution_job_backfills_same_frame_when_pre_job_was_not_submitted() -> None:
    records = []
    observer = LegacyPositionReferenceObserverRuntime(
        worker_instance_id="worker:0",
        executor_instance_id="executor:0",
        sink=records.append,
        queue_capacity=8,
    )
    try:
        assert observer.submit_resolution_nowait(_resolution())
        observer.flush()
    finally:
        observer.close()

    assert len(records) == 2
    assert isinstance(records[0], LegacyPositionReferenceObservationBatch)
    assert isinstance(records[1], LegacyPositionReferenceResolutionObservation)
    assert (
        records[0].frame_observation.frame_observation_id
        == records[1].source_frame_observation_id
    )


def test_resolution_failure_is_explicit_non_authoritative_and_backfills_frame() -> None:
    failure = LegacyReferenceResolutionFailureDerivation(
        reference_derivation=_derivation(),
        failed_at=10.02,
        failure_stage="planner",
        error_type="RuntimeError",
        error_message="collision planner rejected endpoint",
        failure_evidence_hash="planner-failure-evidence",
    )
    pure = build_legacy_reference_resolution_failure_observation(
        failure,
        worker_instance_id="worker:0",
        executor_instance_id="executor:0",
    )
    assert pure.failure_stage == "planner"
    assert pure.execution_context_enforcement == (
        "legacy_generation_captured_at_reference_start"
    )
    assert not pure.backend_accepted
    assert pure.admission_receipt_id is None
    assert pure.non_authoritative
    assert not pure.anchor_commit_eligible

    records = []
    observer = LegacyPositionReferenceObserverRuntime(
        worker_instance_id="worker:0",
        executor_instance_id="executor:0",
        sink=records.append,
        queue_capacity=8,
    )
    try:
        assert observer.submit_resolution_failure_nowait(failure)
        observer.flush()
    finally:
        observer.close()

    assert len(records) == 2
    assert isinstance(records[0], LegacyPositionReferenceObservationBatch)
    assert isinstance(
        records[1], LegacyPositionReferenceResolutionFailureObservation
    )
    assert (
        records[0].frame_observation.frame_observation_id
        == records[1].source_frame_observation_id
    )


def test_generation_invalidation_emits_only_legacy_terminal_without_stop_ack() -> None:
    records = []
    observer = LegacyPositionReferenceObserverRuntime(
        worker_instance_id="worker:0",
        executor_instance_id="executor:0",
        sink=records.append,
        queue_capacity=8,
    )
    try:
        assert observer.submit_nowait(_derivation())
        assert observer.submit_generation_invalidation_nowait(
            LegacyReferenceGenerationInvalidation(
                legacy_worker_generation=3,
                invalidated_at=10.2,
                reason="stream_clear",
                terminal_status="canceled",
            )
        )
        observer.flush()
    finally:
        observer.close()

    terminal = next(
        item
        for item in records
        if isinstance(item, LegacyPositionReferenceSegmentTerminalObservation)
    )
    segment = next(
        item.segment_observation
        for item in records
        if isinstance(item, LegacyPositionReferenceObservationBatch)
        and item.segment_observation is not None
    )
    assert terminal.source_segment_observation_id == segment.observation_id
    assert terminal.terminal_status == "canceled"
    assert terminal.execution_context_enforcement == (
        "executor_local_generation_only"
    )
    assert not terminal.backend_stop_acknowledged
    assert terminal.non_authoritative
    assert not terminal.anchor_commit_eligible

    with pytest.raises(ValueError, match="generation does not match"):
        build_legacy_reference_segment_terminal_observation(
            segment,
            LegacyReferenceGenerationInvalidation(
                legacy_worker_generation=4,
                invalidated_at=10.2,
                reason="wrong-generation",
                terminal_status="canceled",
            ),
        )

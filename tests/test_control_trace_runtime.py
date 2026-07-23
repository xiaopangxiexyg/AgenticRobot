from __future__ import annotations

import threading
import time

import pytest

from dataset_contracts.completeness import (
    TraceRequirementProfile,
    trace_requirement_profile,
)
from dataset_contracts.control_trace import ControlTraceFact
from teleop_runtime.control_trace.lifecycle import (
    EpisodeTraceMembership,
    EpisodeTraceMembershipRelease,
    RuntimeTraceSession,
)
from teleop_runtime.control_trace.runtime import ControlTraceRuntime
from teleop_runtime.control_trace.runtime import summarize_control_trace_admission_latency


def session(profile_id: str = "plain_dry_run_v1") -> RuntimeTraceSession:
    return RuntimeTraceSession(
        trace_session_id="trace-1",
        runtime_session_id="runtime-1",
        control_domain_id="robot-0",
        profile_id=profile_id,
        started_at=1.0,
        clock_domain_id="local_monotonic",
    )


def fact(event_type: str, index: int, *, runtime_session_id: str = "runtime-1") -> ControlTraceFact:
    if event_type == "runtime_trace_terminal":
        payload = {
            "terminal_status": "completed",
            "reason": "test_complete",
            "occurred_at": float(index),
        }
    elif event_type == "source_validation":
        payload = {
            "validation_id": f"validation-{index}",
            "source_event_id": f"source-{index}",
            "accepted": True,
            "reasons": (),
        }
    else:
        raise ValueError(f"Unsupported test fact type: {event_type}")
    return ControlTraceFact.create(
        fact_id=f"fact:{event_type}:{index}",
        event_type=event_type,
        runtime_session_id=runtime_session_id,
        control_domain_id="robot-0",
        embodiment_instance_id="rm65-pair-0",
        occurred_at=float(index),
        clock_domain_id="local_monotonic",
        payload=payload,
    )


def test_runtime_is_subordinate_to_external_episode_membership_owner() -> None:
    runtime = ControlTraceRuntime(session())
    bind_receipt = runtime.bind_episode(
        EpisodeTraceMembership(
            episode_uuid="episode-1",
            lifecycle_owner_id="gui-recorder",
            binding_event_id="bind-1",
            membership_epoch=1,
            bound_at=2.0,
        )
    )
    receipt = runtime.issue(fact("source_validation", 1))

    assert bind_receipt.accepted and receipt.accepted
    assert runtime.records()[0].fact.event_type == "episode_membership_transition"
    assert runtime.records()[1].episode_uuid == "episode-1"
    assert runtime.records()[1].episode_membership is not None
    assert runtime.records()[1].episode_membership.lifecycle_owner_id == "gui-recorder"
    assert not hasattr(runtime, "begin_episode")
    assert not hasattr(runtime, "finalize_episode")
    assert not hasattr(runtime, "commit_episode")
    with pytest.raises(ValueError, match="different episode lifecycle owner"):
        runtime.bind_episode(
            EpisodeTraceMembership(
                episode_uuid="episode-2",
                lifecycle_owner_id="another-owner",
                binding_event_id="bind-2",
                membership_epoch=2,
                bound_at=3.0,
            )
        )

    release = runtime.clear_episode_membership(
        EpisodeTraceMembershipRelease(
            episode_uuid="episode-1",
            lifecycle_owner_id="gui-recorder",
            unbinding_event_id="unbind-1",
            membership_epoch=2,
            unbound_at=4.0,
            reason="episode_finalized",
        )
    )
    assert release is not None and release.accepted
    assert runtime.episode_membership is None
    assert runtime.records()[-1].fact.payload["transition"] == "unbind"


def test_rejection_consumes_issued_sequence_and_preserves_gap_evidence() -> None:
    base = trace_requirement_profile("plain_dry_run_v1")
    profile = TraceRequirementProfile(
        profile_id="reject-source-validation-v1",
        runtime_mode="test",
        required_event_types=base.required_event_types,
        optional_event_types=base.optional_event_types - {"source_validation"},
        not_applicable_event_types=(
            base.not_applicable_event_types | {"source_validation"}
        ),
        receipt_policy=base.receipt_policy,
        measurement_policy=base.measurement_policy,
        lineage_policy_id=base.lineage_policy_id,
    )
    runtime = ControlTraceRuntime(
        session("reject-source-validation-v1"),
        profile=profile,
    )
    rejected = runtime.issue(fact("source_validation", 1))
    accepted_after_gap = runtime.issue(fact("runtime_trace_terminal", 2))
    snapshot = runtime.snapshot()
    report = runtime.completeness()

    assert not rejected.accepted
    assert rejected.rejection_code == "event_not_applicable"
    assert accepted_after_gap.accepted and accepted_after_gap.sink_accepted_sequence == 2
    assert snapshot.journal_issued_sequence == 2
    assert snapshot.accepted_sequences == (2,)
    assert snapshot.sink_accepted_high_watermark == 0
    assert report.admission_gap_ranges == ((1, 1),)
    assert not report.in_memory_complete
    assert not report.capture_commit_eligible


def test_bounded_runtime_rejects_immediately_without_overwriting_records() -> None:
    runtime = ControlTraceRuntime(session(), capacity=1)

    assert runtime.issue(fact("source_validation", 1)).accepted
    overflow = runtime.issue(fact("source_validation", 2))

    assert not overflow.accepted
    assert overflow.rejection_code == "journal_capacity_exhausted"
    assert len(runtime.records()) == 1
    assert runtime.snapshot().rejected_sequences == (2,)


def test_required_profile_can_be_complete_in_memory_but_never_durable_in_m4a() -> None:
    profile = trace_requirement_profile("plain_dry_run_v1")
    runtime = ControlTraceRuntime(session(), capacity=1)
    assert runtime.issue(fact("runtime_trace_terminal", 1)).accepted

    report = runtime.completeness()

    assert report.missing_required_event_types == frozenset()
    assert report.terminal_observed
    assert report.lineage_complete
    assert report.in_memory_complete
    assert not report.durability_supported
    assert not report.durable_complete
    assert not report.capture_commit_eligible


def test_terminal_closes_admission_but_does_not_claim_episode_commit() -> None:
    runtime = ControlTraceRuntime(session())
    assert runtime.issue(fact("runtime_trace_terminal", 1)).accepted
    late = runtime.issue(fact("source_validation", 2))

    assert not late.accepted
    assert late.rejection_code == "terminal_already_accepted"
    assert runtime.completeness().capture_commit_eligible is False


def test_planned_formal_profile_cannot_be_reported_complete_from_event_presence() -> None:
    runtime = ControlTraceRuntime(session("authoritative_real_v1"))
    assert runtime.issue(fact("runtime_trace_terminal", 1)).accepted

    report = runtime.completeness()

    assert not report.lineage_complete
    assert not report.in_memory_complete
    assert {item.code for item in report.lineage_issues} == {
        "profile_not_implemented"
    }


def test_control_facing_admission_is_bounded_and_defers_envelope_encoding() -> None:
    trace = ControlTraceRuntime(session(), capacity=512)
    facts = tuple(fact("source_validation", index) for index in range(1, 257))
    stop = threading.Event()

    def contend() -> None:
        while not stop.is_set():
            trace.snapshot()

    worker = threading.Thread(target=contend, daemon=True)
    worker.start()
    durations: list[float] = []
    try:
        for item in facts:
            started = time.perf_counter()
            receipt = trace.issue(item)
            durations.append(time.perf_counter() - started)
            assert receipt.accepted
            assert receipt.record_integrity_hash is None
    finally:
        stop.set()
        worker.join(timeout=1.0)

    report = summarize_control_trace_admission_latency(
        durations,
        deadline_sec=0.05,
    )
    assert report.sample_count == len(facts)
    assert report.p99_sec < 0.02
    assert report.deadline_miss_count == 0
    # Encoding is available to the storage worker, outside issue().
    assert trace.records()[0].record_integrity_hash

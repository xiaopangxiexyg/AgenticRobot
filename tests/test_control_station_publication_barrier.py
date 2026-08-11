from __future__ import annotations

import threading

import pytest

from control_station import (
    StationOperationResultDraft,
    StationPublicationBarrier,
    admit_station_operation_result,
)
from control_station.publication_barrier import (
    StationPublicationBarrierObserverPort,
)


def _authority_result(*, succeeded: bool = True):
    return admit_station_operation_result(
        StationOperationResultDraft(
            source_intent_id="intent:authority",
            operation_kind="request_motion_authority",
            subject_id="human-primary",
            succeeded=succeeded,
            completed_at=1.0,
            completed_clock_domain_id="adapter-monotonic",
            status="bound" if succeeded else "failed",
            evidence_refs=("evidence:binding",),
        ),
        runtime_session_id="session:station",
        control_domain_id="domain:arms",
        available_at=1.1,
        available_clock_domain_id="control-monotonic",
        admission_sequence=1,
    )


def _human_takeover_result(*, status: str = "human_takeover_ready"):
    return admit_station_operation_result(
        StationOperationResultDraft(
            source_intent_id="intent:human-takeover",
            operation_kind="request_human_takeover",
            subject_id="human-primary",
            succeeded=True,
            completed_at=2.0,
            completed_clock_domain_id="adapter-monotonic",
            status=status,
            evidence_refs=("evidence:human-takeover-aggregate",),
        ),
        runtime_session_id="session:station",
        control_domain_id="domain:arms",
        available_at=2.1,
        available_clock_domain_id="control-monotonic",
        admission_sequence=2,
    )


def test_publication_barrier_requires_evidenced_authority_completion() -> None:
    barrier = StationPublicationBarrier(source_slot_id="human-primary")
    assert not barrier.snapshot().enabled
    with pytest.raises(ValueError, match="successful, evidenced"):
        barrier.enable_from_authority_result(_authority_result(succeeded=False))

    enabled = barrier.enable_from_authority_result(_authority_result())
    assert enabled.enabled
    assert enabled.authority_operation_result_id is not None
    assert barrier.execute_if_enabled(lambda: "admitted") == "admitted"

    disabled = barrier.disable(reason="release_requested")
    assert not disabled.enabled
    assert disabled.generation == enabled.generation + 1
    with pytest.raises(RuntimeError, match="disabled"):
        barrier.execute_if_enabled(lambda: "must-not-run")


def test_human_takeover_barrier_requires_exact_ready_terminal_result() -> None:
    barrier = StationPublicationBarrier(source_slot_id="human-primary")
    with pytest.raises(ValueError, match="successful, evidenced"):
        barrier.enable_from_human_takeover_result(
            _human_takeover_result(status="completed")
        )
    enabled = barrier.enable_from_human_takeover_result(_human_takeover_result())
    assert enabled.enabled
    assert enabled.transition_reason == "typed_human_takeover_completion"


def test_release_linearizes_after_inflight_publication() -> None:
    barrier = StationPublicationBarrier(source_slot_id="human-primary")
    barrier.enable_from_authority_result(_authority_result())
    publication_entered = threading.Event()
    allow_publication_to_finish = threading.Event()
    publication_done = threading.Event()
    release_done = threading.Event()

    def publish() -> None:
        def operation() -> None:
            publication_entered.set()
            assert allow_publication_to_finish.wait(timeout=1.0)

        barrier.execute_if_enabled(operation)
        publication_done.set()

    def release() -> None:
        barrier.disable(reason="release_requested")
        release_done.set()

    publication_thread = threading.Thread(target=publish)
    release_thread = threading.Thread(target=release)
    publication_thread.start()
    assert publication_entered.wait(timeout=1.0)
    release_thread.start()
    assert not release_done.wait(timeout=0.05)
    allow_publication_to_finish.set()
    publication_thread.join(timeout=1.0)
    release_thread.join(timeout=1.0)
    assert publication_done.is_set()
    assert release_done.is_set()
    assert not barrier.snapshot().enabled


def test_publication_barrier_observer_port_preserves_delivery_policy() -> None:
    snapshot = StationPublicationBarrier(
        source_slot_id="human-primary"
    ).snapshot()
    observed = []
    port = StationPublicationBarrierObserverPort(
        suppress_callback_errors=True
    )
    port.bind(lambda item, **metadata: observed.append((item, metadata)))
    metadata = {
        "causal_station_fact_id": "intent:observe",
        "occurred_at": 3.0,
        "clock_domain_id": "control-monotonic",
    }
    port.observe(snapshot, **metadata)
    port.observe(snapshot, **metadata)
    assert len(observed) == 1
    assert port.failure_count == 0

    failing = StationPublicationBarrierObserverPort(
        suppress_callback_errors=True
    )

    def fail(*_args, **_kwargs) -> None:
        raise RuntimeError("observer unavailable")

    failing.bind(fail)
    failing.observe(snapshot, **metadata)
    assert failing.failure_count == 1

    strict = StationPublicationBarrierObserverPort(
        suppress_callback_errors=False
    )
    strict.bind(fail)
    with pytest.raises(RuntimeError, match="observer unavailable"):
        strict.observe(snapshot, **metadata)

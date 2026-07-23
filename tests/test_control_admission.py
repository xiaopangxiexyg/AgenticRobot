from __future__ import annotations

import threading
from dataclasses import dataclass

import pytest

from robot_motion.runtime.control_admission import (
    ControlAdmissionCoordinator,
    admitted_at_or_before,
)
from robot_motion.runtime.identity import ClockDomainDescriptor, RuntimeFactIdentity
from robot_motion.runtime.producer_epoch import ProducerEpochError, ProducerEpochRegistry


@dataclass(frozen=True)
class _Fact:
    control_domain_id: str
    available_at: float
    control_admission_sequence: int
    value: str


class _Store:
    def __init__(self) -> None:
        self.records: list[_Fact] = []

    def _insert_admitted(self, record: _Fact) -> None:
        self.records.append(record)


class _FailingStore(_Store):
    def _insert_admitted(self, record: _Fact) -> None:
        raise RuntimeError("store failed")


def _clock_descriptor() -> ClockDomainDescriptor:
    return ClockDomainDescriptor(
        clock_domain_id="control.monotonic",
        runtime_session_id="session-0",
        boot_id="boot-0",
        reset_epoch=0,
    )


def test_commit_advances_only_after_synchronous_store_insert() -> None:
    coordinator = ControlAdmissionCoordinator(
        "arm-cell", clock_domain=_clock_descriptor(), clock=lambda: 10.0
    )
    store = _Store()

    with pytest.raises(RuntimeError, match="store failed"):
        coordinator.commit(
            store=_FailingStore(),
            build=lambda available_at, sequence: _Fact(
                "arm-cell", available_at, sequence, "rejected"
            ),
        )
    assert coordinator.committed_high_watermark == 0

    fact = coordinator.commit(
        store=store,
        build=lambda available_at, sequence: _Fact(
            "arm-cell", available_at, sequence, "accepted"
        ),
    )
    assert fact.control_admission_sequence == 1
    assert store.records == [fact]
    assert coordinator.committed_high_watermark == 1


def test_cut_captures_one_time_and_sequence_committed_prefix() -> None:
    coordinator = ControlAdmissionCoordinator(
        "arm-cell", clock_domain=_clock_descriptor(), clock=lambda: 5.0
    )
    store = _Store()
    coordinator.commit(
        store=store,
        build=lambda available_at, sequence: _Fact(
            "arm-cell", available_at, sequence, "first"
        ),
    )
    capture = coordinator.capture_cut(
        cutoff_at=5.0,
        projectors={
            "facts": lambda cut: tuple(
                record.value for record in store.records if admitted_at_or_before(record, cut)
            )
        },
    )
    coordinator.commit(
        store=store,
        build=lambda available_at, sequence: _Fact(
            "arm-cell", available_at, sequence, "second"
        ),
    )

    assert capture.cut.cutoff_admission_sequence == 1
    assert capture.projections["facts"] == ("first",)


def test_concurrent_same_timestamp_admissions_form_one_gapless_prefix() -> None:
    coordinator = ControlAdmissionCoordinator(
        "arm-cell", clock_domain=_clock_descriptor(), clock=lambda: 8.0
    )
    store = _Store()
    barrier = threading.Barrier(9)

    def worker(index: int) -> None:
        barrier.wait()
        coordinator.commit(
            store=store,
            build=lambda available_at, sequence: _Fact(
                "arm-cell", available_at, sequence, str(index)
            ),
        )

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=2.0)
        assert not thread.is_alive()

    assert sorted(record.control_admission_sequence for record in store.records) == list(range(1, 9))
    assert coordinator.committed_high_watermark == 8
    cut = coordinator.capture_cut(cutoff_at=8.0).cut
    assert all(admitted_at_or_before(record, cut) for record in store.records)


def test_clock_domain_and_fact_identity_separate_boot_reset_and_fact_namespace() -> None:
    descriptor = ClockDomainDescriptor(
        clock_domain_id="control.monotonic",
        runtime_session_id="runtime-session",
        boot_id="host-boot-7",
        reset_epoch=2,
        resolution_sec=1e-9,
    )
    assert descriptor.reset_epoch == 2
    proposal_id = RuntimeFactIdentity(
        runtime_session_id="runtime-session",
        control_domain_id="robot-0",
        producer_id="policy/main",
        producer_epoch=3,
        sequence=4,
        namespace="action_proposal",
    ).fact_id
    activation_id = RuntimeFactIdentity(
        runtime_session_id="runtime-session",
        control_domain_id="robot-0",
        producer_id="policy/main",
        producer_epoch=3,
        sequence=4,
        namespace="producer_epoch_activation",
    ).fact_id
    assert proposal_id != activation_id
    assert "episode" not in proposal_id


def test_producer_epoch_activation_is_receiver_admitted_and_cut_replayable() -> None:
    coordinator = ControlAdmissionCoordinator(
        "robot-0", clock_domain=_clock_descriptor(), clock=lambda: 20.0
    )
    registry = ProducerEpochRegistry(runtime_session_id="session-0", coordinator=coordinator)
    first = registry.activate(
        producer_id="policy-main",
        producer_epoch=1,
        produced_at=10.0,
        produced_clock_domain_id="policy.monotonic",
        reason="startup",
    )
    first_cut = coordinator.capture_cut(cutoff_at=20.0).cut
    second = registry.activate(
        producer_id="policy-main",
        producer_epoch=2,
        produced_at=11.0,
        produced_clock_domain_id="policy.monotonic",
        reason="restart",
    )

    assert first.available_at == 20.0
    assert first.reason == "startup"
    assert first.activation_id != second.activation_id
    assert registry.snapshot(first_cut).active_epoch_by_producer == {"policy-main": 1}
    assert registry.current_epoch("policy-main") == 2
    watermark = coordinator.committed_high_watermark
    with pytest.raises(ProducerEpochError, match="must increase"):
        registry.activate(
            producer_id="policy-main",
            producer_epoch=2,
            produced_at=12.0,
            produced_clock_domain_id="policy.monotonic",
            reason="duplicate",
        )
    assert coordinator.committed_high_watermark == watermark

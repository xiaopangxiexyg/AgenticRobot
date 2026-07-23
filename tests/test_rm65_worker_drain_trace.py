from __future__ import annotations

from robot_motion.runtime import aggregate_worker_epoch_drains
from teleop_runtime.backends.arms.rm65_worker_drain import (
    RM65ArmsOnlyWorkerDrainResult,
)
from teleop_runtime.control_trace.rm65_worker_drain import (
    admit_rm65_worker_drain_trace,
)
from tests.test_worker_epoch_drain import (
    ManualClock,
    _cancellation,
    _drained,
    _participants,
)


def _result():
    clock = ManualClock()
    cancellation = _cancellation(clock)
    participants = _participants()
    clock.value = cancellation.effective_at + 0.01
    receipts = tuple(
        _drained(cancellation, participant, acknowledged_at=clock.value)
        for participant in participants
    )
    aggregate = aggregate_worker_epoch_drains(
        cancellation=cancellation,
        participants=participants,
        receipts=receipts,
        aggregated_at=clock.value,
    )
    return cancellation, RM65ArmsOnlyWorkerDrainResult(
        plan_id="fixture-plan",
        cancellation_id=cancellation.cancellation_id,
        receipts=receipts,
        aggregate=aggregate,
    )


def test_trace_failure_is_reported_without_mutating_completed_drain() -> None:
    cancellation, result = _result()
    observed: list[str] = []

    def sink(fact):
        observed.append(fact.fact_id)
        if fact.event_type == "execution_worker_drain_aggregate":
            raise OSError("spool unavailable")

    report = admit_rm65_worker_drain_trace(
        result,
        cancellation=cancellation,
        embodiment_instance_id="rm65-bimanual-0",
        fact_sink=sink,
    )

    assert result.aggregate.complete
    assert len(observed) == 3
    assert not report.complete
    assert report.accepted_fact_ids == tuple(
        receipt.receipt_id for receipt in result.receipts
    )
    assert report.failed_fact_ids == (result.aggregate.aggregate_id,)
    assert "OSError:spool unavailable" in report.failure_messages[0]


def test_complete_trace_report_requires_every_receipt_and_aggregate() -> None:
    cancellation, result = _result()
    facts = []
    report = admit_rm65_worker_drain_trace(
        result,
        cancellation=cancellation,
        embodiment_instance_id="rm65-bimanual-0",
        fact_sink=facts.append,
    )

    assert report.complete
    assert report.failed_fact_ids == ()
    assert tuple(fact.fact_id for fact in facts) == report.attempted_fact_ids

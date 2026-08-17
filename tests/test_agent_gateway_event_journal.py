from __future__ import annotations

import pytest

from agent_gateway.runtime.event_journal import (
    BoundedAgentGatewayEventJournal,
)


def test_journal_assigns_monotonic_events_and_reports_retention_gaps() -> None:
    journal = BoundedAgentGatewayEventJournal(
        runtime_session_id="runtime:test",
        max_events=2,
    )
    first = journal.append(
        event_kind="operation_updated",
        emitted_at=1.0,
        operation_id="operation:one",
        operation_hash="hash:one",
        payload={"status": "running"},
    )
    second = journal.append(
        event_kind="operation_progress",
        emitted_at=2.0,
        operation_id="operation:one",
        operation_hash="hash:two",
        payload={"phase": "capture"},
    )
    third = journal.append(
        event_kind="operation_terminal",
        emitted_at=3.0,
        operation_id="operation:one",
        operation_hash="hash:three",
        payload={"status": "completed"},
    )
    assert (first.sequence, second.sequence, third.sequence) == (1, 2, 3)
    assert journal.oldest_available_sequence == 2

    gap = journal.replay(after_sequence=0, max_events=2)
    assert gap.gap_detected
    assert tuple(event.sequence for event in gap.events) == (2, 3)

    resumed = journal.replay(after_sequence=2, max_events=2)
    assert not resumed.gap_detected
    assert tuple(event.sequence for event in resumed.events) == (3,)


def test_empty_journal_has_a_non_gap_cursor_zero_snapshot() -> None:
    journal = BoundedAgentGatewayEventJournal(
        runtime_session_id="runtime:test",
        max_events=4,
    )
    batch = journal.replay(after_sequence=0, max_events=4)
    assert not batch.gap_detected
    assert batch.oldest_available_sequence == 1
    assert batch.latest_sequence == 0
    assert not batch.events
    with pytest.raises(ValueError, match="latest_sequence"):
        journal.replay(after_sequence=1, max_events=1)

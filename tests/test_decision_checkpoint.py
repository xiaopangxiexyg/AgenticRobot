"""Decision grid, publication spacing and pending evidence survive restore."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from robot_motion.action.arbitration_runtime import (
    ArbitrationRuntime,
    DecisionScheduler,
    DecisionScheduleSpec,
)
from robot_motion.action.decision_checkpoint import (
    capture_decision_owners,
    restore_decision_owners,
)
from tests.test_action_arbitration_runtime import Clock, computation

pytestmark = pytest.mark.unit


def test_grid_pending_trigger_truncation_and_future_ticks_match_without_catchup():
    clock = Clock(1.0)
    spec = DecisionScheduleSpec(
        spec_id="event",
        schedule_mode="event_limited",
        decision_hz=30,
        max_pending_trigger_evidence=3,
    )
    scheduler, publication = (
        DecisionScheduler(spec, clock=clock),
        ArbitrationRuntime(spec),
    )
    scheduler.trigger("initial", "start")
    assert scheduler.make_tick(cutoff_admission_sequence=7) is not None
    for i in range(10):
        scheduler.trigger(str(i), "observation")
    state = capture_decision_owners(scheduler, publication)
    assert state.pending_trigger_count == 10
    assert state.pending_triggers == (
        ("7", "observation"),
        ("8", "observation"),
        ("9", "observation"),
    )
    candidate_clock = Clock(clock.value)
    restored = restore_decision_owners(state, spec=spec, clock=candidate_clock)
    assert restored.scheduler.make_tick(cutoff_admission_sequence=8) is None
    assert scheduler.make_tick(cutoff_admission_sequence=8) is None
    for at in (1.1, 10.0, 10.001, 10.04):
        clock.value = candidate_clock.value = at
        scheduler.trigger("next", "observation")
        restored.scheduler.trigger("next", "observation")
        actual = scheduler.make_tick(cutoff_admission_sequence=9)
        expected = restored.scheduler.make_tick(cutoff_admission_sequence=9)
        assert actual == expected
    assert capture_decision_owners(
        restored.scheduler, restored.publication
    ) == capture_decision_owners(scheduler, publication)
    restored.scheduler.trigger("branch-only", "extra")
    assert restored.scheduler._pending_trigger_count != scheduler._pending_trigger_count


def test_publication_spacing_rejection_retains_last_successful_gate(monkeypatch):
    clock = Clock(1.0)
    spec = DecisionScheduleSpec(spec_id="publish", decision_hz=30)
    scheduler, publication = (
        DecisionScheduler(spec, clock=clock),
        ArbitrationRuntime(spec),
    )
    gate = SimpleNamespace(status="ready", gate_epoch=0)
    result = computation(
        result_id="first", target=1.0, schedule_hash=spec.semantic_content_hash
    )
    assert publication.publish(result, attempted_at=1.0, live_gate_snapshot=gate)[
        0
    ].accepted
    state = capture_decision_owners(scheduler, publication)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Recovery cannot tick or publish.")

    with monkeypatch.context() as patch:
        patch.setattr(DecisionScheduler, "make_tick", forbidden)
        patch.setattr(ArbitrationRuntime, "publish", forbidden)
        restored = restore_decision_owners(state, spec=spec, clock=Clock(1.0))
    for at in (1.001, 1.04):
        result = computation(
            result_id=str(at), target=at, schedule_hash=spec.semantic_content_hash
        )
        old = publication.publish(result, attempted_at=at, live_gate_snapshot=gate)
        new = restored.publication.publish(
            result, attempted_at=at, live_gate_snapshot=gate
        )
        assert old[0] == new[0]
        assert (old[1] is None) == (new[1] is None)
        assert old[0].accepted == (at == 1.04)


def test_initial_and_invalid_decision_cuts_preserve_explicit_binding():
    clock = Clock(1.0)
    spec = DecisionScheduleSpec(spec_id="fresh")
    scheduler, publication = (
        DecisionScheduler(spec, clock=clock),
        ArbitrationRuntime(spec),
    )
    state = capture_decision_owners(scheduler, publication)
    restored = restore_decision_owners(state, spec=spec, clock=Clock(1.0))
    assert restored.scheduler.next_due_at is None
    assert restored.publication._last_published_at is None
    for bad in (
        replace(state, tick_index=1),
        replace(state, pending_trigger_count=1),
        replace(state, last_published_at=1.0),
        replace(state, captured_at=float("nan")),
    ):
        with pytest.raises(ValueError):
            restore_decision_owners(bad, spec=spec, clock=clock)
    with pytest.raises(ValueError, match="binding"):
        restore_decision_owners(state, spec=replace(spec, decision_hz=20), clock=clock)
    with pytest.raises(ValueError, match="clock"):
        restore_decision_owners(state, spec=spec, clock=Clock(2.0))


@pytest.mark.parametrize("count", [2, 10])
def test_pending_trigger_evidence_cannot_be_missing_below_or_above_truncation(count):
    clock = Clock(1.0)
    spec = DecisionScheduleSpec(spec_id="evidence", max_pending_trigger_evidence=3)
    scheduler, publication = (
        DecisionScheduler(spec, clock=clock),
        ArbitrationRuntime(spec),
    )
    for index in range(count):
        scheduler.trigger(str(index), "event")
    state = capture_decision_owners(scheduler, publication)
    with pytest.raises(ValueError, match="checkpoint"):
        restore_decision_owners(
            replace(state, pending_triggers=state.pending_triggers[1:]),
            spec=spec,
            clock=clock,
        )

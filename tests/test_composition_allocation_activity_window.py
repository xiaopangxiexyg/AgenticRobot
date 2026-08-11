from __future__ import annotations

from dataclasses import replace

import pytest

from control_composition.allocation import (
    SourceActionActivitySample,
    SourceActionActivityWindow,
)
from tests.test_composition_input_batch_admission import _fixture


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _sample(action, *, index: int, sampled_at: float):
    return SourceActionActivitySample(
        sampled_at=sampled_at,
        clock_domain_id="control.monotonic",
        action=action,
        source_fact_id=f"human-residual:{index}",
        source_fact_hash=f"human-residual-hash:{index}",
    )


def test_window_prunes_by_time_instead_of_record_count() -> None:
    _clock, _coordinator, _store, action = _fixture()
    window = SourceActionActivityWindow(window_duration_sec=0.5)
    for index, sampled_at in enumerate((9.0, 9.4, 9.8)):
        assert window.offer(
            _sample(action, index=index, sampled_at=sampled_at)
        )

    snapshot = window.snapshot(cutoff_at=9.8)

    assert tuple(sample.sampled_at for sample in snapshot.samples) == (9.4, 9.8)
    assert snapshot.ready
    assert snapshot.dropped_sample_count == 1


def test_window_memory_tracks_declared_duration_at_high_input_rate() -> None:
    _clock, _coordinator, _store, action = _fixture()
    window = SourceActionActivityWindow(window_duration_sec=0.2)
    for index in range(100):
        window.offer(
            _sample(
                action,
                index=index,
                sampled_at=10.0 + 0.01 * index,
            )
        )

    snapshot = window.snapshot(cutoff_at=10.99)

    assert 20 <= len(snapshot.samples) <= 21
    assert snapshot.samples[0].sampled_at >= 10.79 - 1e-12
    assert snapshot.dropped_sample_count >= 79


def test_duplicate_is_idempotent_but_time_or_clock_rollback_is_rejected() -> None:
    _clock, _coordinator, _store, action = _fixture()
    window = SourceActionActivityWindow(window_duration_sec=1.0)
    first = _sample(action, index=0, sampled_at=10.0)
    assert window.offer(first)
    assert window.offer(first) is False
    with pytest.raises(ValueError, match="increasing time order"):
        window.offer(_sample(action, index=1, sampled_at=9.9))
    with pytest.raises(ValueError, match="clock domain changed"):
        window.offer(
            replace(
                _sample(action, index=2, sampled_at=10.1),
                clock_domain_id="another.clock",
            )
        )

    assert tuple(
        sample.sample_id for sample in window.snapshot(cutoff_at=10.1).samples
    ) == (first.sample_id,)


def test_snapshot_cut_does_not_include_future_samples() -> None:
    _clock, _coordinator, _store, action = _fixture()
    window = SourceActionActivityWindow(window_duration_sec=1.0)
    window.offer(_sample(action, index=0, sampled_at=10.0))
    window.offer(_sample(action, index=1, sampled_at=10.2))

    snapshot = window.snapshot(cutoff_at=10.1)

    assert tuple(sample.sampled_at for sample in snapshot.samples) == (10.0,)
    assert snapshot.ready is False

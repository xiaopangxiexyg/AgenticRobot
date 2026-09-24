from __future__ import annotations

import pytest

from dataset_interface.temporal_sampling import (
    TemporalSamplingError,
    select_fixed_cadence_window_starts,
)


def test_regular_capture_selects_row_aligned_windows() -> None:
    times = [100.0 + index / 15.0 for index in range(8)]

    starts = select_fixed_cadence_window_starts(times, target_fps=15.0, horizon=4)

    assert starts == tuple(range(5))


def test_sparse_capture_is_not_relabelled_as_nominal_15hz() -> None:
    times = [100.0 + index / 10.0 for index in range(12)]
    times[8:] = [value + 1.0 for value in times[8:]]

    fifteen_hz = select_fixed_cadence_window_starts(times, target_fps=15.0, horizon=4)
    ten_hz = select_fixed_cadence_window_starts(times, target_fps=10.0, horizon=4)

    assert fifteen_hz == ()
    assert 0 in ten_hz
    assert 8 in ten_hz
    assert all(not (start < 8 <= start + 3) for start in ten_hz)


def test_segment_boundary_excludes_crossing_windows() -> None:
    times = [index / 15.0 for index in range(10)]
    segments = [0] * 6 + [1] * 4

    starts = select_fixed_cadence_window_starts(
        times,
        target_fps=15.0,
        horizon=3,
        continuity_segments=segments,
    )

    assert starts == (0, 1, 2, 3, 6, 7)


def test_cumulative_drift_rejects_consistently_slow_chunk() -> None:
    times = [index * (1.2 / 15.0) for index in range(10)]

    assert select_fixed_cadence_window_starts(times, target_fps=15.0, horizon=5) == ()


@pytest.mark.parametrize(
    "times,kwargs",
    [
        ([1.0, 1.0], {}),
        ([1.0, float("nan")], {}),
        ([1.0, 0.9], {}),
        ([1.0, 2.0], {"target_fps": 0.0}),
        ([1.0, 2.0], {"continuity_segments": [0]}),
    ],
)
def test_ambiguous_clock_fails_closed(times, kwargs) -> None:
    with pytest.raises(TemporalSamplingError):
        select_fixed_cadence_window_starts(
            times,
            target_fps=kwargs.get("target_fps", 10.0),
            horizon=2,
            continuity_segments=kwargs.get("continuity_segments"),
        )

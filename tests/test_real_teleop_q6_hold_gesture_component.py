from __future__ import annotations

import numpy as np

from teleop_runtime.debug import run_vive_rm65_real_teleop as legacy_entrypoint
from teleop_runtime.real_teleop.q6_hold_gesture import (
    Q6HoldRecalibrationGesture,
    Q6HoldRecalibrationGestureConfig,
)


def _q6(tail: tuple[float, float, float]) -> np.ndarray:
    return np.asarray([0.0, 0.0, 0.0, *tail], dtype=float)


def test_extracted_q6_gesture_matches_legacy_close_release_close_contract() -> None:
    assert legacy_entrypoint.Q6HoldRecalibrationGesture is Q6HoldRecalibrationGesture
    assert (
        legacy_entrypoint.Q6HoldRecalibrationGestureConfig
        is Q6HoldRecalibrationGestureConfig
    )
    legacy = legacy_entrypoint.Q6HoldRecalibrationGesture(
        legacy_entrypoint.Q6HoldRecalibrationGestureConfig(enabled=True, hold_sec=1.0)
    )
    extracted = Q6HoldRecalibrationGesture(
        Q6HoldRecalibrationGestureConfig(enabled=True, hold_sec=1.0)
    )
    closed = _q6((1.0, 1.0, 1.0))
    opened = _q6((0.5, 0.5, 0.5))
    samples = (
        (closed, closed, False, 0.0),
        (closed, closed, False, 1.01),
        (opened, opened, True, 2.0),
        (opened, opened, True, 3.01),
        (closed, closed, True, 4.0),
        (closed, closed, True, 5.01),
    )

    for left_q6, right_q6, hold_active, now in samples:
        actual = extracted.update(
            left_q6=left_q6,
            right_q6=right_q6,
            teleop_started=True,
            hold_active=hold_active,
            now=now,
        )
        expected = legacy.update(
            left_q6=left_q6,
            right_q6=right_q6,
            teleop_started=True,
            hold_active=hold_active,
            now=now,
        )
        assert actual == expected

    assert actual == Q6HoldRecalibrationGesture.EVENT_RESTART

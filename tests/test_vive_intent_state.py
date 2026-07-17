from __future__ import annotations

import numpy as np

from teleop_runtime.sources.pose.vive_motion import (
    TrackerMotionIntentConfig,
    TrackerMotionIntentStateMachine,
    summarize_tracker_motion_window,
)


def _pose(x: float = 0.0, yaw_rad: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    cosine = float(np.cos(yaw_rad))
    sine = float(np.sin(yaw_rad))
    pose[:3, :3] = [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    pose[0, 3] = x
    return pose


def _sample(
    sample_time: float,
    *,
    x: float = 0.0,
    yaw_rad: float = 0.0,
    linear_speed: float = 0.005,
    angular_speed: float = 0.02,
) -> dict[str, object]:
    return {
        "history_time": sample_time,
        "history_clock_source": "pose_sample_monotonic_time",
        "pose": _pose(x, yaw_rad),
        "valid": True,
        "velocity_valid": True,
        "linear_velocity_m_s": [linear_speed, 0.0, 0.0],
        "angular_velocity_rad_s": [0.0, 0.0, angular_speed],
    }


def _quiet_metrics(sample_time: float) -> dict[str, object]:
    return {
        "quality_valid": True,
        "newest_history_time": sample_time,
        "linear_speed_p90_m_s": 0.01,
        "angular_speed_p90_rad_s": 0.05,
        "translation_span_m": 0.001,
        "rotation_span_rad": 0.01,
    }


def test_motion_window_uses_elapsed_time_with_irregular_samples() -> None:
    history = [
        _sample(1.000, x=-1.0),
        _sample(1.890, x=-0.5),
        _sample(1.905, x=0.0000),
        _sample(1.950, x=0.0005),
        _sample(2.000, x=0.0010),
    ]

    metrics = summarize_tracker_motion_window(history, window_sec=0.1)

    assert metrics["sample_count"] == 3
    assert np.isclose(metrics["window_span_sec"], 0.095)
    assert np.isclose(metrics["translation_span_m"], 0.001)
    assert metrics["quality_valid"] is True
    assert metrics["speed_source"] == "openvr_validated"


def test_motion_window_requires_sufficient_time_and_velocity_quality() -> None:
    history = [_sample(1.000), _sample(1.010)]
    history[-1]["velocity_valid"] = False

    metrics = summarize_tracker_motion_window(history, window_sec=0.1)

    assert metrics["quality_valid"] is False
    assert "insufficient_time_span" in metrics["quality_reasons"]
    assert "velocity_valid_ratio_low" in metrics["quality_reasons"]


def test_motion_intent_enters_inactive_only_after_time_dwell() -> None:
    state_machine = TrackerMotionIntentStateMachine()

    first = state_machine.update(
        motion_window=_quiet_metrics(10.0), tracker_pose=_pose()
    )
    before_dwell = state_machine.update(
        motion_window=_quiet_metrics(10.299), tracker_pose=_pose()
    )
    after_dwell = state_machine.update(
        motion_window=_quiet_metrics(10.300), tracker_pose=_pose()
    )

    assert first["state"] == "QUIETING"
    assert before_dwell["state"] == "QUIETING"
    assert before_dwell["hold_weight"] == 0.0
    assert after_dwell["state"] == "INACTIVE"
    assert after_dwell["hold_weight"] == 1.0


def test_tightened_motion_intent_requires_all_quiet_evidence_below_scaled_thresholds() -> None:
    config = TrackerMotionIntentConfig(
        enter_linear_speed_m_s=0.015,
        enter_angular_speed_rad_s=0.09,
        enter_translation_span_m=0.00125,
        enter_rotation_span_rad=0.0125,
    )
    state_machine = TrackerMotionIntentStateMachine(config)
    metrics = _quiet_metrics(1.0)
    metrics["linear_speed_p90_m_s"] = 0.020

    first = state_machine.update(motion_window=metrics, tracker_pose=_pose())
    metrics = dict(metrics, newest_history_time=1.4)
    later = state_machine.update(motion_window=metrics, tracker_pose=_pose())

    assert first["state"] == "ACTIVE"
    assert later["state"] == "ACTIVE"
    assert later["evidence"]["quiet"] is False


def test_motion_intent_exit_inherits_anchor_and_requires_debounce() -> None:
    state_machine = TrackerMotionIntentStateMachine()
    state_machine.update(motion_window=_quiet_metrics(1.0), tracker_pose=_pose())
    state_machine.update(motion_window=_quiet_metrics(1.3), tracker_pose=_pose())

    moving_metrics = _quiet_metrics(1.31)
    moving_metrics["translation_span_m"] = 0.003
    first_exit = state_machine.update(
        motion_window=moving_metrics,
        tracker_pose=_pose(x=0.005),
    )
    moving_metrics = dict(moving_metrics, newest_history_time=1.36)
    debounced_exit = state_machine.update(
        motion_window=moving_metrics,
        tracker_pose=_pose(x=0.005),
    )

    assert first_exit["state"] == "INACTIVE"
    assert debounced_exit["state"] == "ACTIVE"
    assert debounced_exit["hold_weight"] == 0.0


def test_motion_intent_does_not_enter_inactive_during_slow_monotonic_motion() -> None:
    state_machine = TrackerMotionIntentStateMachine()
    state_machine.update(motion_window=_quiet_metrics(1.0), tracker_pose=_pose())
    state_machine.update(
        motion_window=_quiet_metrics(1.15),
        tracker_pose=_pose(x=0.0015),
    )

    result = state_machine.update(
        motion_window=_quiet_metrics(1.3),
        tracker_pose=_pose(x=0.0030),
    )

    assert result["state"] == "ACTIVE"
    assert result["hold_weight"] == 0.0
    assert result["transition_reason"] == "quiet_dwell_motion"
    assert result["evidence"]["quiet"] is True
    assert result["evidence"]["quiet_anchor_translation_m"] > 0.0025


def test_motion_intent_accumulates_slow_motion_from_fixed_inactive_anchor() -> None:
    state_machine = TrackerMotionIntentStateMachine()
    state_machine.update(motion_window=_quiet_metrics(1.0), tracker_pose=_pose())
    state_machine.update(motion_window=_quiet_metrics(1.3), tracker_pose=_pose())

    first_exit = state_machine.update(
        motion_window=_quiet_metrics(2.10),
        tracker_pose=_pose(x=0.0041),
    )
    result = state_machine.update(
        motion_window=_quiet_metrics(2.15),
        tracker_pose=_pose(x=0.0043),
    )

    assert first_exit["state"] == "INACTIVE"
    assert first_exit["evidence"]["quiet"] is True
    assert first_exit["evidence"]["anchor_exit"] is True
    assert result["state"] == "ACTIVE"
    assert result["hold_weight"] == 0.0


def test_motion_intent_tolerates_short_quality_dropout() -> None:
    state_machine = TrackerMotionIntentStateMachine()
    state_machine.update(motion_window=_quiet_metrics(1.0), tracker_pose=_pose())
    state_machine.update(motion_window=_quiet_metrics(1.3), tracker_pose=_pose())

    result = state_machine.update(
        motion_window={"quality_valid": False, "newest_history_time": 1.35},
        tracker_pose=_pose(),
    )

    assert result["state"] == "INACTIVE"
    assert result["transition_reason"] == "motion_window_quality_grace"


def test_motion_intent_invalid_history_and_clutch_fail_active_safe() -> None:
    state_machine = TrackerMotionIntentStateMachine()
    state_machine.update(motion_window=_quiet_metrics(1.0), tracker_pose=_pose())
    state_machine.update(motion_window=_quiet_metrics(1.3), tracker_pose=_pose())

    invalid = state_machine.update(
        motion_window={"quality_valid": False, "newest_history_time": 1.401},
        tracker_pose=_pose(),
    )
    clutch = state_machine.update(
        motion_window=_quiet_metrics(1.5),
        tracker_pose=_pose(),
        clutch_pressed=True,
    )

    assert invalid["state"] == "UNKNOWN"
    assert invalid["hold_weight"] == 0.0
    assert clutch["state"] == "UNKNOWN"
    assert clutch["transition_reason"] == "clutch_pressed"

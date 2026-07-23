from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sim.rm65.real2sim.gripper_position_fit import (
    FIT_SCHEMA,
    MotionSegment,
    build_fit_artifact,
    fit_direction,
    load_motion_segments,
    predict_step_response,
)


def _segment(
    *,
    trial_id: str,
    initial: float,
    target: float,
    delay: float = 0.08,
    rate: float = 1.3,
    acceleration: float = 20.0,
) -> MotionSegment:
    times = np.linspace(0.01, 1.5, 100)
    positions = predict_step_response(
        times,
        initial_position=initial,
        target_position=target,
        delay_sec=delay,
        max_rate_per_sec=rate,
        acceleration_per_sec2=acceleration,
    )
    return MotionSegment(
        source_trace="synthetic.jsonl",
        side="left",
        trial_id=trial_id,
        segment_index=0,
        stage="motion_initial",
        target_position=target,
        initial_position=initial,
        time_sec=times,
        measured_position=positions,
        command_duration_sec=0.03,
        force_setting=0.2,
        speed_setting=1.0,
        continuous_grasp=True,
        command_mode="runtime-equivalent",
        trace_schema="dmtac_gripper_actuator_trace_v2",
    )


def test_step_response_supports_close_open_and_exact_settle() -> None:
    times = np.asarray([0.0, 0.05, 0.10, 1.5])
    close = predict_step_response(
        times,
        initial_position=0.0,
        target_position=1.0,
        delay_sec=0.08,
        max_rate_per_sec=1.3,
        acceleration_per_sec2=20.0,
    )
    opened = predict_step_response(
        times,
        initial_position=1.0,
        target_position=0.0,
        delay_sec=0.08,
        max_rate_per_sec=1.3,
        acceleration_per_sec2=20.0,
    )

    assert close[0] == close[1] == 0.0
    assert opened[0] == opened[1] == 1.0
    assert close[-1] == 1.0
    assert opened[-1] == 0.0
    assert np.allclose(opened, 1.0 - close)


def test_robust_fit_recovers_synthetic_position_parameters() -> None:
    segments = [
        _segment(trial_id="close_1", initial=0.0, target=1.0),
        _segment(trial_id="close_2", initial=0.2, target=0.8),
        _segment(trial_id="close_3", initial=0.4, target=0.9),
    ]

    fitted = fit_direction(segments)

    assert fitted["status"] == "identified"
    assert fitted["delay_sec"] == pytest.approx(0.08, abs=0.008)
    assert fitted["max_rate_per_sec"] == pytest.approx(1.3, rel=0.08)
    assert fitted["acceleration_per_sec2"] == pytest.approx(20.0, rel=0.20)
    assert fitted["segment_rmse_max"] < 0.01


def _write_v2_trace(root: Path, *, continuous: bool = True) -> None:
    metadata = {
        "schema": "dmtac_gripper_actuator_trace_v2",
        "endpoint": {"side": "left"},
        "speed_fixed": 1.0,
        "trials": [
            {
                "trial_id": "trial_001",
                "protocol": "free",
                "force": 0.2,
                "continuous_grasp": continuous,
            }
        ],
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    command_end_ns = 1_000_000_000
    times = np.linspace(0.01, 1.2, 80)
    positions = predict_step_response(
        times,
        initial_position=0.0,
        target_position=1.0,
        delay_sec=0.08,
        max_rate_per_sec=1.3,
        acceleration_per_sec2=20.0,
    )
    rows = [
        {
            "schema": "dmtac_gripper_actuator_trace_v2",
            "event": "command_end",
            "trial_id": "trial_001",
            "stage": "motion_initial",
            "segment_index": 0,
            "target_position": 1.0,
            "monotonic_time_ns": command_end_ns,
            "command_end_monotonic_ns": command_end_ns,
            "command_begin_monotonic_ns": command_end_ns - 30_000_000,
            "command_mode": "runtime-equivalent",
        }
    ]
    for sequence, (time_sec, position) in enumerate(zip(times, positions), start=1):
        midpoint_ns = command_end_ns + int(round(float(time_sec) * 1e9))
        rows.append(
            {
                "schema": "dmtac_gripper_actuator_trace_v2",
                "event": "state",
                "trial_id": "trial_001",
                "stage": "motion_and_hold",
                "segment_index": 0,
                "sequence": sequence,
                "monotonic_time_ns": midpoint_ns,
                "read_midpoint_monotonic_ns": midpoint_ns,
                "state": {
                    "joint_pos": float(position),
                    "error_code": 0,
                    "temperature_c": 30,
                },
            }
        )
    (root / "gripper_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_v2_loader_and_artifact_keep_position_only_claims(tmp_path: Path) -> None:
    _write_v2_trace(tmp_path)

    segments, evidence = load_motion_segments(tmp_path)
    artifact = build_fit_artifact(
        [tmp_path],
        expected_speed=1.0,
        expected_force=0.2,
        expected_continuous=True,
    )

    assert len(segments) == 1
    assert segments[0].direction == "close"
    assert segments[0].command_duration_sec == pytest.approx(0.03)
    assert segments[0].time_sec[0] == pytest.approx(0.04)
    assert evidence["loaded_segment_count"] == 1
    assert artifact["schema"] == FIT_SCHEMA
    assert artifact["fit_by_side"]["left"]["close"]["status"] == "provisional"
    assert artifact["fidelity_claims"]["contact_force_newtons"] == "unavailable"
    assert artifact["fidelity_claims"]["friction_and_slip"] == "unmodeled"
    assert artifact["fit_clock"] == "command_begin_to_state_read_midpoint_monotonic"
    assert len(artifact["payload_sha256"]) == 64


def test_artifact_rejects_source_settings_that_do_not_match_profile(
    tmp_path: Path,
) -> None:
    _write_v2_trace(tmp_path, continuous=False)

    with pytest.raises(ValueError, match="source condition mismatch"):
        build_fit_artifact(
            [tmp_path],
            expected_speed=1.0,
            expected_force=0.2,
            expected_continuous=True,
        )

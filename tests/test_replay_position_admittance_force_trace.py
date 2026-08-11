from __future__ import annotations

import json
from pathlib import Path

import pytest

from teleop_runtime.debug.replay_position_admittance_force_trace import (
    replay_recorded_control_force,
)


def _write_force_log(
    path: Path,
    *,
    reverse_time: bool = False,
    contact_force_n: float = 2.0,
) -> Path:
    rows: list[dict[str, object]] = [
        {
            "event": "position_admittance_config",
            "payload": {
                "enabled": True,
                "requested_mode": "shadow",
                "schema_version": "position-admittance-v3",
                "active_sides": ["left", "right"],
                "config": {
                    "enabled_axes": [True, True, True],
                    "control_frame": "solver_world",
                    "response_model": "profile_compliance",
                    "virtual_mass_kg": [2.0, 2.0, 2.0],
                    "damping_n_s_m": [50.0, 50.0, 50.0],
                    "stiffness_n_m": [250.0, 250.0, 250.0],
                    "force_deadband_n": [0.45, 0.45, 0.45],
                    "force_filter_tau_sec": 0.010857362,
                    "force_sign": [1.0, 1.0, 1.0],
                    "max_offset_m": [0.005, 0.005, 0.005],
                    "max_velocity_m_s": [0.01, 0.01, 0.01],
                    "max_return_velocity_m_s": [0.01, 0.01, 0.01],
                    "max_output_slew_m_s": [0.01, 0.01, 0.01],
                    "return_force_deadband_n": [0.45, 0.45, 0.45],
                    "return_rate_s": 1.0,
                    "response_reference_hz": 40.0,
                    "stale_after_sec": 0.15,
                    "max_force_dt_sec": 0.15,
                },
            },
        }
    ]
    for index in range(90):
        observed = 1.0 + index / 15.0
        source = observed - 0.02
        if reverse_time and index == 45:
            source -= 0.2
        force_x = contact_force_n if 15 <= index < 45 else 0.0
        rows.append(
            {
                "event": "frame",
                "position_admittance": {
                    "sides": {
                        "left": {
                            "observed_at_monotonic_sec": observed,
                            "source_time_sec": source,
                            "filtered_force_control_n": [force_x, 0.0, 0.0],
                        }
                    }
                },
            }
        )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_saved_force_trace_stays_inside_current_c0_envelope(
    tmp_path: Path,
) -> None:
    log = _write_force_log(tmp_path / "action_debug_force.jsonl")

    report = replay_recorded_control_force(log)

    assert report["passed"] is True
    assert report["recorded_event_count"] == 90
    assert report["output_offset_norm_m"]["max"] <= 0.002 + 1e-9
    assert report["output_slew_norm_m_s"]["max"] <= 0.005 + 1e-9
    assert report["requested_velocity_norm_m_s"]["max"] <= 0.005 + 1e-9
    assert report["final_output_norm_m"] <= 1e-5
    assert report["force_output_direction_cosine"]["min"] == pytest.approx(1.0)


def test_force_replay_preserves_distinct_normal_profile_speed_caps(
    tmp_path: Path,
) -> None:
    log = _write_force_log(
        tmp_path / "action_debug_normal_profile.jsonl",
        contact_force_n=30.0,
    )

    report = replay_recorded_control_force(
        log,
        max_offset_m=0.08,
        max_speed_m_s=0.06,
        max_return_speed_m_s=0.03,
        max_output_slew_m_s=0.06,
        release_tail_sec=8.0,
    )

    assert report["passed"] is True
    config = report["candidate_config"]
    assert config["max_offset_m"] == pytest.approx([0.08] * 3)
    assert config["max_velocity_m_s"] == pytest.approx([0.06] * 3)
    assert config["max_return_velocity_m_s"] == pytest.approx([0.03] * 3)
    assert config["max_output_slew_m_s"] == pytest.approx([0.06] * 3)
    assert report["output_offset_norm_m"]["max"] == pytest.approx(0.08)


def test_saved_force_trace_rejects_nonmonotonic_source_time(
    tmp_path: Path,
) -> None:
    log = _write_force_log(
        tmp_path / "action_debug_bad_clock.jsonl",
        reverse_time=True,
    )

    with pytest.raises(ValueError, match="source time must increase"):
        replay_recorded_control_force(log)


def test_saved_force_trace_preserves_runtime_duplicate_sample_noop(
    tmp_path: Path,
) -> None:
    log = _write_force_log(tmp_path / "action_debug_force.jsonl")
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    duplicate = json.loads(json.dumps(rows[21]))
    duplicate["position_admittance"]["sides"]["left"][
        "observed_at_monotonic_sec"
    ] += 0.01
    rows.insert(22, duplicate)
    log.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    report = replay_recorded_control_force(log)

    assert report["passed"] is True
    assert report["recorded_event_count"] == 91


def test_saved_force_trace_rejects_changed_force_at_duplicate_source_time(
    tmp_path: Path,
) -> None:
    log = _write_force_log(tmp_path / "action_debug_force.jsonl")
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    duplicate = json.loads(json.dumps(rows[21]))
    payload = duplicate["position_admittance"]["sides"]["left"]
    payload["observed_at_monotonic_sec"] += 0.01
    payload["filtered_force_control_n"][0] += 1.0
    rows.insert(22, duplicate)
    log.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must preserve the same force"):
        replay_recorded_control_force(log)

from __future__ import annotations

import json
from pathlib import Path

from teleop_runtime.debug.replay_position_admittance_active_log import (
    replay_active_translation_gate,
)


def _write_incident_log(
    path: Path,
    *,
    nominal_residual_m: float,
    anchor_m: float,
) -> Path:
    config = {
        "event": "position_admittance_config",
        "payload": {
            "enabled": True,
            "requested_mode": "active",
            "schema_version": "position-admittance-v3",
            "apply_sides": ["left"],
            "config": {
                "enabled_axes": [True, True, True],
                "max_offset_m": [0.002, 0.002, 0.002],
                "stale_after_sec": 0.15,
            },
        },
    }
    zero_frame = {
        "event": "frame",
        "step_index": 10,
        "sample_timestamp": 100.0,
        "position_admittance": {
            "sides": {
                "left": {
                    "observed_at_monotonic_sec": 1.00,
                    "output_offset_m": [0.0, 0.0, 0.0],
                    "anchor_offset_m": [0.0, 0.0, 0.0],
                    "applied_total_offset_m": [0.0, 0.0, 0.0],
                    "applied_target_translation": [0.0, 0.0, 0.0],
                    "admission_feedback": {
                        "at_monotonic_sec": 1.01,
                        "admitted_target_translation": [
                            nominal_residual_m,
                            0.0,
                            0.0,
                        ],
                    },
                }
            }
        },
    }
    contact_frame = {
        "event": "frame",
        "step_index": 11,
        "sample_timestamp": 100.066,
        "position_admittance": {
            "sides": {
                "left": {
                    "observed_at_monotonic_sec": 1.06,
                    "output_offset_m": [0.0001, 0.0, 0.0],
                    "anchor_offset_m": [anchor_m, 0.0, 0.0],
                    "applied_total_offset_m": [anchor_m + 0.0001, 0.0, 0.0],
                    "applied_target_translation": [
                        anchor_m + 0.0001,
                        0.0,
                        0.0,
                    ],
                    "admission_feedback": {
                        "at_monotonic_sec": 1.07,
                        "admitted_target_translation": [
                            nominal_residual_m,
                            0.0,
                            0.0,
                        ],
                    },
                }
            }
        },
    }
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in (config, zero_frame, contact_frame)),
        encoding="utf-8",
    )
    return path


def test_replay_predicts_incident_rejection_before_anchor_violation(
    tmp_path: Path,
) -> None:
    log = _write_incident_log(
        tmp_path / "action_debug_incident.jsonl",
        nominal_residual_m=0.140,
        anchor_m=0.140,
    )

    report = replay_active_translation_gate(log)

    assert report["passed"] is True
    assert report["max_transient_m"] == 0.0001
    assert report["max_anchor_m"] == 0.140
    assert (
        report["first_predicted_rejection"]["rejection_reason"]
        == "nominal_translation_residual"
    )
    assert report["first_predicted_rejection"]["frame_number"] == 2
    assert report["first_old_total_violation"]["frame_number"] == 2


def test_replay_does_not_claim_prevention_without_nominal_residual(
    tmp_path: Path,
) -> None:
    log = _write_incident_log(
        tmp_path / "action_debug_unexplained_anchor.jsonl",
        nominal_residual_m=0.0,
        anchor_m=0.140,
    )

    report = replay_active_translation_gate(log)

    assert report["passed"] is False
    assert report["first_predicted_rejection"] is None
    gate = next(
        check
        for check in report["checks"]
        if check["name"] == "current_gate_rejects_before_recorded_violation"
    )
    assert gate["passed"] is False

from __future__ import annotations

import json
from pathlib import Path

from teleop_runtime.debug.analyze_position_admittance_shadow_ab import analyze_pair


def _write_log(
    root: Path,
    *,
    shadow: bool,
    applied_offset: float = 0.0,
    shadow_fixed_side: str | None = None,
) -> Path:
    root.mkdir()
    path = root / "action_debug_20260806_000000_000.jsonl"
    rates = {"sample_hz": 15.0, "reference_hz": 60.0, "ik_hz": 60.0, "command_hz": 120.0}
    rows = [
        {
            "event": "config",
            "payload": {
                "arm_execution_profile": {"position_rates": rates},
                "position_control_profile": {"name": "rm65_bimanual_position_v1"},
                "position_ik_step_mode": "single_step_singularity_aware",
            },
        }
    ]
    if shadow:
        rows.append(
            {
                "event": "position_admittance_config",
                "payload": {
                    "enabled": True,
                    "requested_mode": "shadow",
                    "active_sides": ["left", "right"],
                    "coupled_pre_ik_target_filter_available": True,
                },
            }
        )
    for index in range(8):
        fixed_side = (
            shadow_fixed_side
            if shadow and shadow_fixed_side is not None
            else "left" if index < 4 else "right"
        )
        frame = {
            "event": "frame",
            "active_sides": ["left", "right"],
            "faults": [],
            "fix_arm": {
                "available": True,
                "active": True,
                "selected_side": fixed_side,
                "fixed_side": fixed_side,
            },
            "tracker_channels": {},
            "command_by_side": {},
            "jitter_focus_by_side": {},
            "ik_timing": {"time_budget_exceeded": False},
        }
        for side_index, side in enumerate(("left", "right")):
            frame["tracker_channels"][f"{side}_wrist_pose"] = {
                "pose_stage": "observation_channel_before_retarget",
                "pose": {
                    "xyz_m": [index * 0.001, side_index * 0.1, 0.2],
                    "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                },
                "motion_window": {"sample_count": 10},
            }
            frame["command_by_side"][side] = {
                "value_qpos": {
                    "deg": [
                        0.0
                        if shadow and side == shadow_fixed_side
                        else float(index)
                    ]
                    * 6
                }
            }
            frame["jitter_focus_by_side"][side] = {
                "position_background_streaming": {
                    "failed_send_count": 0,
                    "fault": None,
                    "healthy": True,
                }
            }
        if shadow:
            frame["position_admittance"] = {
                "mode": "shadow",
                "shadow_only": True,
                "applied_to_target": False,
                "target_identity_preserved": True,
                "last_error": None,
                "sides": {
                    side: {
                        "active": side != shadow_fixed_side,
                        "activity_reason": (
                            "fixed_side"
                            if side == shadow_fixed_side
                            else "running_live_side"
                        ),
                        "applied_offset_m": [applied_offset, 0.0, 0.0],
                        "sample_age_sec": 0.02,
                        "new_sample_integrated": side != shadow_fixed_side,
                        "frozen": side == shadow_fixed_side,
                        "deadbanded_force_control_n": (
                            [0.0, 0.0, 0.0]
                            if side == shadow_fixed_side
                            else [1.0, 0.0, 0.0]
                        ),
                        "filtered_force_control_n": (
                            [0.0, 0.0, 0.0]
                            if side == shadow_fixed_side
                            else [1.0, 0.0, 0.0]
                        ),
                        "requested_offset_m": (
                            [0.0, 0.0, 0.0]
                            if side == shadow_fixed_side
                            else [0.001, 0.0, 0.0]
                        ),
                        "output_offset_m": (
                            [0.0, 0.0, 0.0]
                            if side == shadow_fixed_side
                            else [0.001, 0.0, 0.0]
                        ),
                        "velocity_m_s": (
                            [0.0, 0.0, 0.0]
                            if side == shadow_fixed_side
                            else [0.001, 0.0, 0.0]
                        ),
                    }
                    for side in ("left", "right")
                },
            }
        rows.append(frame)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_analyzer_accepts_matched_dual_arm_baseline_and_identity_shadow(tmp_path: Path) -> None:
    baseline = _write_log(tmp_path / "baseline", shadow=False)
    shadow = _write_log(tmp_path / "shadow", shadow=True)

    report = analyze_pair(
        baseline,
        shadow,
        min_frames=8,
        min_fixed_frames_per_side=4,
        min_tracker_pose_coverage=1.0,
    )

    assert report["passed"] is True
    assert report["baseline"]["qref"]["left"]["fixed_frame_count"] == 4
    assert report["shadow"]["shadow"]["max_abs_applied_offset_m"] == {
        "left": 0.0,
        "right": 0.0,
    }
    assert report["shadow"]["admittance"]["left"]["active_frame_count"] == 8
    assert report["shadow"]["admittance"]["left"]["sample_age_ms"]["n"] == 8


def test_analyzer_rejects_shadow_that_changes_the_target(tmp_path: Path) -> None:
    baseline = _write_log(tmp_path / "baseline", shadow=False)
    shadow = _write_log(tmp_path / "shadow", shadow=True, applied_offset=0.001)

    report = analyze_pair(
        baseline,
        shadow,
        min_frames=8,
        min_fixed_frames_per_side=4,
        min_tracker_pose_coverage=1.0,
    )

    assert report["passed"] is False
    check = next(
        item
        for item in report["shadow"]["checks"]
        if item["name"] == "shadow_never_changes_target"
    )
    assert check["passed"] is False


def test_analyzer_can_require_the_promoted_position_profile(tmp_path: Path) -> None:
    baseline = _write_log(tmp_path / "baseline", shadow=False)
    shadow = _write_log(tmp_path / "shadow", shadow=True)

    report = analyze_pair(
        baseline,
        shadow,
        min_frames=8,
        min_fixed_frames_per_side=4,
        min_tracker_pose_coverage=1.0,
        expected_position_profile="rm65_bimanual_position_v2_balanced",
    )

    assert report["passed"] is False
    check = next(
        item
        for item in report["baseline"]["checks"]
        if item["name"] == "expected_position_profile"
    )
    assert check == {
        "name": "expected_position_profile",
        "passed": False,
        "detail": "rm65_bimanual_position_v1",
    }


def test_analyzer_accepts_explicit_left_live_right_fix_topology(tmp_path: Path) -> None:
    baseline = _write_log(tmp_path / "baseline", shadow=False)
    shadow = _write_log(
        tmp_path / "shadow",
        shadow=True,
        shadow_fixed_side="right",
    )

    report = analyze_pair(
        baseline,
        shadow,
        min_frames=8,
        min_fixed_frames_per_side=4,
        min_tracker_pose_coverage=1.0,
        expected_shadow_live_side="left",
        expected_shadow_fixed_side="right",
    )

    assert report["passed"] is True
    topology = report["shadow"]["expected_shadow_topology"]
    assert topology["expected_fix_active_frame_count"] == 8
    assert not any(
        check["name"] == "right_shadow_force_response_observed"
        for check in report["shadow"]["checks"]
    )


def test_analyzer_rejects_selected_but_inactive_fix_topology(tmp_path: Path) -> None:
    baseline = _write_log(tmp_path / "baseline", shadow=False)
    shadow = _write_log(tmp_path / "shadow", shadow=True)

    rows = [json.loads(line) for line in shadow.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row.get("event") != "frame":
            continue
        row["fix_arm"] = {
            "available": True,
            "active": False,
            "selected_side": "right",
            "fixed_side": "",
        }
    shadow.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    report = analyze_pair(
        baseline,
        shadow,
        min_frames=8,
        min_fixed_frames_per_side=4,
        min_tracker_pose_coverage=1.0,
        expected_shadow_live_side="left",
        expected_shadow_fixed_side="right",
    )

    assert report["passed"] is False
    check = next(
        item
        for item in report["shadow"]["checks"]
        if item["name"] == "expected_single_arm_shadow_topology"
    )
    assert check["passed"] is False
    assert check["detail"]["expected_fix_active_frame_count"] == 0

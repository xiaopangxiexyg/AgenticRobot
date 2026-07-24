from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from teleop_runtime.calibration import TeleopCalibration
from teleop_runtime.gui import calibration_web_ui
from teleop_runtime.real_teleop.tracker_side_check import (
    evaluate_tracker_side_assignment,
)


def _pose(*, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [x, y, z]
    return pose


def _yaw_pose(yaw_deg: float) -> np.ndarray:
    angle = np.deg2rad(yaw_deg)
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    return pose


def _channel(
    pose: np.ndarray,
    *,
    valid: bool = True,
    timestamp: float | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        value=pose,
        valid=valid,
        timestamp=timestamp,
        metadata={},
    )


def _calibration(*, post_yaw_deg: float) -> TeleopCalibration:
    calibration = TeleopCalibration()
    calibration.set_world_to_desk(_yaw_pose(post_yaw_deg))
    calibration.metadata["virtual_desk_post_yaw_deg"] = post_yaw_deg
    return calibration


def test_tracker_side_check_accepts_left_on_positive_operator_y() -> None:
    result = evaluate_tracker_side_assignment(
        {
            "left_wrist_pose": _channel(_pose(y=0.25)),
            "right_wrist_pose": _channel(_pose(y=-0.20)),
        },
        calibration=_calibration(post_yaw_deg=0.0),
        active_sides=("left", "right"),
    )

    assert result["status"] == "correct"
    assert result["warning_required"] is False
    assert result["lateral_gap_m"] == 0.45


def test_tracker_side_check_removes_default_post_yaw_before_comparison() -> None:
    result = evaluate_tracker_side_assignment(
        {
            "left_wrist_pose": _channel(_pose(y=0.25)),
            "right_wrist_pose": _channel(_pose(y=-0.20)),
        },
        calibration=_calibration(post_yaw_deg=180.0),
        active_sides=("left", "right"),
    )

    assert result["status"] == "correct"
    assert result["left_y_m"] == 0.25
    assert result["right_y_m"] == -0.2
    assert result["post_yaw_deg"] == 180.0


def test_tracker_side_check_warns_when_logical_trackers_are_swapped() -> None:
    result = evaluate_tracker_side_assignment(
        {
            "left_wrist_pose": _channel(_pose(y=-0.20)),
            "right_wrist_pose": _channel(_pose(y=0.25)),
        },
        calibration=_calibration(post_yaw_deg=180.0),
        active_sides=("left", "right"),
    )

    assert result["status"] == "swapped"
    assert result["warning_required"] is True
    assert "当前逻辑映射" in str(result["message"])


def test_tracker_side_check_warns_when_lateral_gap_is_ambiguous() -> None:
    result = evaluate_tracker_side_assignment(
        {
            "left_wrist_pose": _channel(_pose(y=0.02)),
            "right_wrist_pose": _channel(_pose(y=0.0)),
        },
        calibration=_calibration(post_yaw_deg=0.0),
        active_sides=("left", "right"),
    )

    assert result["status"] == "ambiguous"
    assert result["warning_required"] is True


def test_tracker_side_check_does_not_warn_until_both_channels_are_valid() -> None:
    result = evaluate_tracker_side_assignment(
        {
            "left_wrist_pose": _channel(_pose(y=-0.20), valid=False),
            "right_wrist_pose": _channel(_pose(y=0.25)),
        },
        calibration=_calibration(post_yaw_deg=180.0),
        active_sides=("left", "right"),
    )

    assert result["status"] == "unavailable"
    assert result["warning_required"] is False


def test_tracker_side_check_rejects_stale_valid_channels() -> None:
    result = evaluate_tracker_side_assignment(
        {
            "left_wrist_pose": _channel(_pose(y=0.25), timestamp=9.0),
            "right_wrist_pose": _channel(_pose(y=-0.20), timestamp=9.9),
        },
        calibration=_calibration(post_yaw_deg=180.0),
        active_sides=("left", "right"),
        now=10.0,
    )

    assert result["status"] == "unavailable"
    assert result["warning_required"] is False
    assert "数据不新鲜" in str(result["message"])


def test_tracker_side_check_never_classifies_non_finite_pose_as_correct() -> None:
    invalid_pose = _pose(y=0.25)
    invalid_pose[1, 3] = np.nan
    result = evaluate_tracker_side_assignment(
        {
            "left_wrist_pose": _channel(invalid_pose),
            "right_wrist_pose": _channel(_pose(y=-0.20)),
        },
        calibration=_calibration(post_yaw_deg=180.0),
        active_sides=("left", "right"),
    )

    assert result["status"] == "unavailable"
    assert result["warning_required"] is False
    assert "非有限" in str(result["message"])


def test_calibration_web_ui_exposes_tracker_side_confirmation_contract() -> None:
    ui = calibration_web_ui.CalibrationWebUi(open_browser=False)
    ui.publish(
        tracker_side_check={
            "status": "swapped",
            "confirmation_required": True,
            "confirmation_token": "tracker-side-1",
        }
    )

    assert ui.snapshot()["tracker_side_check"]["status"] == "swapped"
    assert "tracker_side_confirm_continue" in calibration_web_ui._HTML_PAGE
    assert "tracker_side_confirm_cancel" in calibration_web_ui._HTML_PAGE
    assert "window.confirm(message)" in calibration_web_ui._HTML_PAGE
    assert "tracker_side_confirm_continue" in calibration_web_ui._FALLBACK_SCRIPT


def test_calibration_web_ui_accepts_only_active_tracker_side_confirmation() -> None:
    ui = calibration_web_ui.CalibrationWebUi(open_browser=False)
    assert not ui.push_event("tracker_side_confirm_continue")

    ui.publish(
        active_controls=[
            "tracker_side_confirm_continue",
            "tracker_side_confirm_cancel",
        ]
    )

    assert ui.push_event("tracker_side_confirm_continue")
    assert ui.consume_events() == ["tracker_side_confirm_continue"]

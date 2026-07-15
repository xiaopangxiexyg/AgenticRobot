from __future__ import annotations

from teleop_runtime.gui import calibration_web_ui


def test_runtime_home_capture_progress_is_exposed_as_gui_diagnostic() -> None:
    ui = calibration_web_ui.CalibrationWebUi(open_browser=False)
    ui.publish(
        capture_stage="runtime_home",
        capture_phase="capturing",
        capture_left_samples=10,
        capture_right_samples=1,
        capture_min_samples=20,
        capture_left_valid=True,
        capture_right_valid=False,
        capture_elapsed_sec=0.3,
        capture_duration_sec=10.0,
        capture_waiting_for_samples=True,
    )

    diagnostic = ui.snapshot()["runtime_home_diagnostic"]

    assert diagnostic["visible"] is True
    assert diagnostic["level"] == "capturing"
    assert diagnostic["title"] == "Runtime Home 正在采样"
    assert "左臂 10/20；右臂 1/20" in diagnostic["detail"]
    assert "左臂 有效；右臂 等待有效 tracker" in diagnostic["detail"]
    assert "采样时间：0.3/10.0s" in diagnostic["detail"]
    assert "无效帧不会计入进度" in diagnostic["detail"]


def test_runtime_home_joint_limit_failure_is_rendered_with_actionable_context() -> None:
    failure = (
        "runtime home 标定失败: ValueError: left_arm local qpos joint "
        "3=2.355008rad exceeds absolute limits [-2.355000, 2.355000]rad.\n"
        "请检查 VIVE sender/tracker_role 后重新选择操作。"
    )
    ui = calibration_web_ui.CalibrationWebUi(open_browser=False)
    ui.publish(
        capture_stage="runtime_home",
        capture_phase="ready",
        control_hint=failure,
    )

    snapshot = ui.snapshot()
    diagnostic = snapshot["runtime_home_diagnostic"]

    assert diagnostic["visible"] is True
    assert diagnostic["level"] == "error"
    assert diagnostic["title"] == "Runtime Home 标定失败：左臂关节 3 到达绝对限位"
    assert "当前位置：2.355008 rad" in diagnostic["detail"]
    assert "允许范围：[-2.355000, 2.355000] rad" in diagnostic["detail"]
    assert "点击“开始遥操”重新采集 Runtime Home" in diagnostic["detail"]

    initial_html = calibration_web_ui._render_initial_html(snapshot)
    assert 'id="runtimeHomeDiagnostic"' in initial_html
    assert "runtime-home-diagnostic error" in initial_html
    assert "左臂关节 3 到达绝对限位" in initial_html


def test_runtime_home_diagnostic_is_hidden_outside_capture_or_failure() -> None:
    diagnostic = calibration_web_ui._runtime_home_diagnostic_payload(
        {
            "capture_stage": "teleop",
            "capture_phase": "teleop",
            "control_hint": "遥操已开始",
        }
    )

    assert diagnostic == {
        "visible": False,
        "level": "",
        "title": "",
        "detail": "",
    }


def test_all_gui_refresh_paths_render_runtime_home_diagnostic() -> None:
    assert "renderRuntimeHomeDiagnosticLegacy(state)" in calibration_web_ui._HTML_PAGE
    assert "renderRuntimeHomeDiagnostic(state)" in calibration_web_ui._HTML_PAGE
    assert "renderRuntimeHomeDiagnosticFallback(state)" in calibration_web_ui._FALLBACK_SCRIPT

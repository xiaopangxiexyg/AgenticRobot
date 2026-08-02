from __future__ import annotations

from pathlib import Path

from control_station.web import calibration_web_ui
from teleop_runtime.collection.umi.cli import build_parser as build_umi_parser
from teleop_runtime.real_teleop.cli import build_real_teleop_argument_parser


def _render_page(*, enabled: bool) -> str:
    ui = calibration_web_ui.CalibrationWebUi(
        open_browser=False,
        resizable_sidebars=enabled,
    )
    return calibration_web_ui._render_initial_html(
        ui.snapshot(),
        resizable_sidebars=enabled,
    )


def test_resizable_sidebars_default_preserves_established_two_column_markup() -> None:
    page = _render_page(enabled=False)

    assert "<main>" in page
    assert '<main data-resizable-sidebars="true">' not in page
    assert 'id="sidebarResizeHandle"' not in page
    assert '<script src="/sidebar-resize.js"></script>' not in page


def test_resizable_sidebars_opt_in_adds_only_presentation_resources() -> None:
    page = _render_page(enabled=True)

    assert '<main data-resizable-sidebars="true">' in page
    assert 'id="sidebarResizeHandle"' in page
    assert 'role="separator"' in page
    assert '<script src="/sidebar-resize.js"></script>' in page
    assert page.index('id="sidebarResizeHandle"') < page.index(
        '<header><h2>控制</h2></header>'
    )


def test_real_teleop_resizable_sidebars_cli_is_default_on_and_reversible() -> None:
    parser = build_real_teleop_argument_parser()

    assert parser.parse_args([]).gui_resizable_sidebars is True
    assert parser.parse_args(["--gui-resizable-sidebars"]).gui_resizable_sidebars is True
    assert parser.parse_args(["--no-gui-resizable-sidebars"]).gui_resizable_sidebars is False


def test_umi_resizable_sidebars_cli_preserves_config_unless_explicit() -> None:
    parser = build_umi_parser()

    assert parser.parse_args([]).gui_resizable_sidebars is None
    assert parser.parse_args(["--gui-resizable-sidebars"]).gui_resizable_sidebars is True
    assert parser.parse_args(["--no-gui-resizable-sidebars"]).gui_resizable_sidebars is False


def test_sidebar_resize_script_clamps_width_and_supports_keyboard_reset() -> None:
    script = Path(__file__).parents[1] / "control_station/web/sidebar_resize.js"
    source = script.read_text(encoding="utf-8")

    assert "MIN_LEFT_PX" in source
    assert "MIN_RIGHT_PX" in source
    assert 'event.key === "ArrowLeft"' in source
    assert 'event.key === "ArrowRight"' in source
    assert 'event.key === "Home"' in source


def test_input_activity_indicators_are_default_on_and_reversible() -> None:
    parser = build_real_teleop_argument_parser()

    assert parser.parse_args([]).gui_input_activity_indicators is True
    assert (
        parser.parse_args(
            ["--no-gui-input-activity-indicators"]
        ).gui_input_activity_indicators
        is False
    )

    enabled = calibration_web_ui._render_initial_html({})
    disabled = calibration_web_ui._render_initial_html(
        {},
        input_activity_indicators=False,
    )
    assert 'id="inputActivityStrip"' in enabled
    assert 'data-input-activity="pedal_5"' in enabled
    assert '<script src="/input-activity.js"></script>' in enabled
    assert 'id="inputActivityStrip"' not in disabled
    assert '<script src="/input-activity.js"></script>' not in disabled


def test_input_activity_state_is_presentation_only_and_monotonic() -> None:
    ui = calibration_web_ui.CalibrationWebUi(open_browser=False)

    ui.publish_input_activity(
        source="foot_pad",
        event="fix_arm_right",
        pedal_count=2,
    )
    first = ui.snapshot()["input_activity"]
    ui.publish_input_activity(
        source="foot_pad",
        event="progress_undo",
        pedal_count=5,
    )
    second = ui.snapshot()["input_activity"]

    assert first["sequence"] == 1
    assert first["events"] == [{
        "sequence": 1,
        "source": "foot_pad",
        "event": "fix_arm_right",
        "pedal_index": 2,
        "pedal_count": 2,
    }]
    assert second["sequence"] == 2
    assert [item["pedal_index"] for item in second["events"]] == [2, 5]


def test_umi_input_activity_cli_preserves_config_unless_explicit() -> None:
    parser = build_umi_parser()

    assert parser.parse_args([]).gui_input_activity_indicators is None
    assert (
        parser.parse_args(
            ["--gui-input-activity-indicators"]
        ).gui_input_activity_indicators
        is True
    )
    assert (
        parser.parse_args(
            ["--no-gui-input-activity-indicators"]
        ).gui_input_activity_indicators
        is False
    )

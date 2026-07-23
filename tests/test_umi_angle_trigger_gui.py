from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hardware.angle_trigger import angle_trigger
from hardware.angle_trigger.angle_trigger import AngleTriggerError, SensorCandidate
from teleop_runtime.collection.umi import angle_trigger_calibration as gui_calibration
from teleop_runtime.configs.dmtac_hardware import (
    prepare_dmtac_hardware_assignment_before_gui,
    _unique_dmtac_gripper_completion,
    _usb_assignment_module,
)
from teleop_runtime.gui.calibration_web_ui import _HTML_PAGE


def _sensor(name: str, *, location: str, servo_id: int) -> SensorCandidate:
    return SensorCandidate(
        device=f"/dev/{name}",
        servo_id=servo_id,
        location=location,
        serial_number="",
        vid="1a86",
        pid="7523",
        description="test",
    )


def test_noninteractive_reader_loads_valid_saved_config(tmp_path, monkeypatch):
    sensors = [
        _sensor("left", location="1-1", servo_id=1),
        _sensor("right", location="1-2", servo_id=2),
    ]
    angle_trigger.save_assignment_config(
        {"left": sensors[0], "right": sensors[1]}, config_root=tmp_path
    )
    angle_trigger.save_angle_calibration_config(
        {
            "left": {"zero_deg": 10.0, "one_deg": 30.0},
            "right": {"zero_deg": 40.0, "one_deg": 15.0},
        },
        config_root=tmp_path,
    )
    monkeypatch.setattr(angle_trigger, "discover_sensors", lambda **_kwargs: sensors)

    reader = angle_trigger.load_configured_reader(config_root=tmp_path)

    assert reader.mapping == {"left": sensors[0], "right": sensors[1]}
    assert reader.calibration["profiles"]["left"]["one_deg"] == 30.0


def test_noninteractive_reader_reports_stale_assignment_without_terminal_fallback(
    tmp_path, monkeypatch
):
    saved = [
        _sensor("left", location="old-1", servo_id=1),
        _sensor("right", location="old-2", servo_id=2),
    ]
    current = [
        _sensor("left-new", location="new-1", servo_id=1),
        _sensor("right-new", location="new-2", servo_id=2),
    ]
    angle_trigger.save_assignment_config(
        {"left": saved[0], "right": saved[1]}, config_root=tmp_path
    )
    angle_trigger.save_angle_calibration_config(
        {
            "left": {"zero_deg": 0.0, "one_deg": 20.0},
            "right": {"zero_deg": 0.0, "one_deg": 20.0},
        },
        config_root=tmp_path,
    )
    monkeypatch.setattr(angle_trigger, "discover_sensors", lambda **_kwargs: current)

    with pytest.raises(AngleTriggerError, match="does not match connected sensors"):
        angle_trigger.load_configured_reader(config_root=tmp_path)


def test_angle_assignment_uniquely_completes_only_one_unmatched_side():
    saved = [
        _sensor("left", location="stable-left", servo_id=1),
        _sensor("right-old", location="old-right", servo_id=2),
    ]
    current = [
        saved[0],
        _sensor("right-new", location="new-right", servo_id=2),
    ]
    assignment = angle_trigger.build_assignment(
        {"left": saved[0], "right": saved[1]}
    )

    mapping, inferred, errors = angle_trigger.resolve_assignment_with_unique_completion(
        assignment, current
    )

    assert mapping == {"left": current[0], "right": current[1]}
    assert inferred == ("right", current[1])
    assert errors == ["Could not uniquely resolve the configured right sensor."]

    both_changed = [
        _sensor("left-new", location="new-left", servo_id=1),
        current[1],
    ]
    mapping, inferred, _ = angle_trigger.resolve_assignment_with_unique_completion(
        assignment, both_changed
    )
    assert mapping is None
    assert inferred is None


class _FakeConnection:
    def __init__(self, values):
        self._values = iter(values)
        self._last = 0.0

    def read_angle_deg(self):
        self._last = float(next(self._values, self._last))
        return self._last


class _FakeUi:
    def __init__(self, confirm_calls):
        self.confirm_calls = set(confirm_calls)
        self.consume_count = 0
        self.states = []

    def publish(self, **state):
        self.states.append(state)

    def consume_events(self):
        self.consume_count += 1
        if self.consume_count in self.confirm_calls:
            return [gui_calibration.ANGLE_NEXT_EVENT]
        return []


def test_gui_side_assignment_selects_the_sensor_moved_in_each_step():
    sensors = [
        _sensor("first", location="1-1", servo_id=1),
        _sensor("second", location="1-2", servo_id=2),
    ]
    ui = _FakeUi(confirm_calls={4, 8})
    connections = {
        "sensor_1": _FakeConnection([0, 1, 2, 4, 4, 4, 4, 4]),
        "sensor_2": _FakeConnection([0, 0, 0, 0, 0, 1, 2, 4]),
    }

    mapping = gui_calibration._assign_sides_in_gui(
        ui,
        candidates=connections,
        sensors=sensors,
        sample_hz=10000.0,
        min_motion_deg=2.0,
        reason="missing config",
    )

    assert mapping == {"left": sensors[0], "right": sensors[1]}
    assert any("左侧" in state["stage"] for state in ui.states)
    assert any("右侧" in state["stage"] for state in ui.states)


def test_umi_web_gui_exposes_angle_calibration_controls():
    assert 'data-event="angle_recalibrate"' in _HTML_PAGE
    assert 'data-event="angle_assignment_confirm"' in _HTML_PAGE
    assert 'data-event="angle_calibration_next"' in _HTML_PAGE
    assert 'data-event="device_retry"' in _HTML_PAGE


def test_dmtac_uniquely_completes_one_missing_side_from_one_reachable_extra():
    usb = _usb_assignment_module()
    old_scan = {
        "grippers": [
            {
                "index": 0,
                "port": "/dev/old-left",
                "preferred_port": "/dev/old-left",
                "serial_number": "OLD_LEFT",
                "probe": {"reachable": True},
            },
            {
                "index": 1,
                "port": "/dev/right",
                "preferred_port": "/dev/right",
                "serial_number": "RIGHT",
                "probe": {"reachable": True},
            },
        ],
        "tactile": [],
    }
    assignment = usb.build_assignment_from_tokens(
        old_scan,
        left_gripper=0,
        right_gripper=1,
    )
    current_scan = {
        "grippers": [
            {
                "index": 0,
                "port": "/dev/new-left",
                "preferred_port": "/dev/new-left",
                "serial_number": "NEW_LEFT",
                "probe": {"reachable": True},
            },
            old_scan["grippers"][1],
        ],
        "tactile": [],
    }
    comparison = usb.compare_assignment_to_scan(assignment, current_scan)

    completion = _unique_dmtac_gripper_completion(
        usb,
        assignment=assignment,
        scan=current_scan,
        comparison=comparison,
    )

    assert completion is not None
    side, current, completed = completion
    assert side == "left"
    assert current["port"] == "/dev/new-left"
    verified = usb.compare_assignment_to_scan(completed, current_scan)
    assert verified["missing"] == []
    assert verified["extra"] == []


def test_dmtac_unique_completion_is_confirmed_and_saved_before_gui(
    tmp_path, monkeypatch
):
    usb = _usb_assignment_module()
    old_scan = {
        "grippers": [
            {
                "index": 0,
                "port": "/dev/old-left",
                "preferred_port": "/dev/old-left",
                "serial_number": "OLD_LEFT",
                "probe": {"reachable": True},
            },
            {
                "index": 1,
                "port": "/dev/right",
                "preferred_port": "/dev/right",
                "serial_number": "RIGHT",
                "probe": {"reachable": True},
            },
        ],
        "tactile": [],
    }
    assignment = usb.build_assignment_from_tokens(
        old_scan, left_gripper=0, right_gripper=1
    )
    usb.save_assignment(assignment, config_dir=tmp_path)
    current_scan = {
        "grippers": [
            {
                "index": 0,
                "port": "/dev/new-left",
                "preferred_port": "/dev/new-left",
                "serial_number": "NEW_LEFT",
                "probe": {"reachable": True},
            },
            old_scan["grippers"][1],
        ],
        "tactile": [],
    }
    monkeypatch.setattr(usb, "scan_usb_devices", lambda **_kwargs: current_scan)
    prompts = []

    effective = prepare_dmtac_hardware_assignment_before_gui(
        tmp_path / "latest.json",
        input_func=lambda prompt: prompts.append(prompt) or "YES",
    )

    assert effective["assignments"]["left"]["gripper"]["port"] == "/dev/new-left"
    assert "Infer left=/dev/new-left" in prompts[0]
    saved = usb.load_assignment(tmp_path / "latest.json")
    assert saved["assignments"]["left"]["gripper"]["port"] == "/dev/new-left"


def test_umi_starts_web_gui_before_attempting_dmtac_preflight(monkeypatch):
    from teleop_runtime.collection.umi import app

    events = []

    class FakeUi:
        def __init__(self, **_kwargs):
            events.append("ui_construct")

        def publish(self, **_kwargs):
            return None

        def start(self):
            events.append("ui_start")

        def close(self):
            events.append("ui_close")

    config = SimpleNamespace(
        dmtac_assignment="latest",
        gui_host="127.0.0.1",
        gui_port=0,
        open_browser=False,
        angle_trigger_config_root=Path("angle"),
        angle_trigger_rate_hz=30.0,
        policy_io_layout=Path("layout.json"),
    )
    monkeypatch.setattr(app, "load_policy_io_layout", lambda _path: object())
    monkeypatch.setattr(
        app,
        "prepare_dmtac_hardware_assignment_before_gui",
        lambda _config: events.append("terminal_dmtac_preflight") or {"assignments": {}},
    )
    monkeypatch.setattr(app, "CalibrationWebUi", FakeUi)
    monkeypatch.setattr(
        app,
        "_ensure_angle_trigger_from_gui",
        lambda *_args, **_kwargs: events.append("angle_trigger_attempt") or False,
    )

    assert app.run_umi_collection(config) == 0
    assert events == [
        "ui_construct",
        "ui_start",
        "terminal_dmtac_preflight",
        "angle_trigger_attempt",
        "ui_close",
    ]


def test_umi_dmtac_failure_does_not_skip_angle_trigger_attempt(monkeypatch):
    from teleop_runtime.collection.umi import app

    events = []

    class FakeUi:
        def __init__(self, **_kwargs):
            pass

        def publish(self, **_kwargs):
            return None

        def start(self):
            events.append("ui_start")

        def close(self):
            events.append("ui_close")

    config = SimpleNamespace(
        dmtac_assignment="latest",
        gui_host="127.0.0.1",
        gui_port=0,
        open_browser=False,
        angle_trigger_config_root=Path("angle"),
        angle_trigger_rate_hz=30.0,
        policy_io_layout=Path("layout.json"),
    )
    monkeypatch.setattr(app, "load_policy_io_layout", lambda _path: object())

    def fail_dmtac(_config):
        events.append("dmtac_failed")
        raise RuntimeError("DMTac unavailable")

    monkeypatch.setattr(app, "prepare_dmtac_hardware_assignment_before_gui", fail_dmtac)
    monkeypatch.setattr(app, "CalibrationWebUi", FakeUi)
    monkeypatch.setattr(
        app,
        "_ensure_angle_trigger_from_gui",
        lambda *_args, **_kwargs: events.append("angle_attempt") or False,
    )

    assert app.run_umi_collection(config) == 0
    assert events == ["ui_start", "dmtac_failed", "angle_attempt", "ui_close"]

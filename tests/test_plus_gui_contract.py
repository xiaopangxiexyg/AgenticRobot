from __future__ import annotations

import json

from teleop_runtime.gui.calibration_web_ui import CalibrationWebUi, _encode_http_event_payload


def test_plus_gui_event_payload_is_explicit_and_bounded() -> None:
    start = _encode_http_event_payload({"event": "episode_start", "plus_variant_id": "left"})
    assert start.startswith("episode_start:")
    payload = json.loads(start.split(":", 1)[1])
    assert payload["plus_variant_id"] == "left"
    save = _encode_http_event_payload(
        {
            "event": "plus_curation_save",
            "episode_index": "4",
            "pending_token": "pending-4",
            "disposition": "exclude",
            "reason_code": "task_failure",
        }
    )
    assert json.loads(save.split(":", 1)[1]) == {
        "episode_index": 4,
        "pending_token": "pending-4",
        "disposition": "exclude",
        "reason_code": "task_failure",
    }


def test_task_selection_preserves_plus_variant_when_present() -> None:
    task_select = _encode_http_event_payload(
        {"event": "task_select", "task_key": "fixture", "plus_variant_id": "left"}
    )
    payload = json.loads(task_select.split(":", 1)[1])
    assert payload["plus_variant_id"] == "left"


def test_task_selection_without_plus_keeps_legacy_wire_shape() -> None:
    task_select = _encode_http_event_payload(
        {"event": "task_select", "task_key": "fixture"}
    )
    payload = json.loads(task_select.split(":", 1)[1])
    assert "plus_variant_id" not in payload


def test_old_plus_save_wire_shape_remains_accepted_by_encoder() -> None:
    save = _encode_http_event_payload(
        {
            "event": "plus_curation_save",
            "episode_index": "4",
            "disposition": "unknown",
            "reason_code": "unknown",
        }
    )
    assert "pending_token" not in json.loads(save.split(":", 1)[1])


def test_plus_save_is_only_accepted_when_runtime_publishes_control() -> None:
    ui = CalibrationWebUi(open_browser=False)
    event = 'plus_curation_save:{"episode_index":4,"disposition":"unknown","reason_code":"unknown"}'
    assert not ui.push_event(event)
    ui.publish(active_controls=["plus_curation_save"])
    assert ui.push_event(event)
    assert ui.consume_events() == [event]


def test_plus_state_defaults_to_off_and_is_not_written_by_publish() -> None:
    ui = CalibrationWebUi(open_browser=False)
    state = ui.snapshot()
    assert state["plus_curation_enabled"] is False
    assert state["plus_curation_pending"] is False
    assert state["plus_curation_disposition"] == "unknown"
    assert state["plus_curation_pending_token"] == ""
    ui.publish(plus_curation_enabled=True, plus_curation_pending=True)
    assert ui.snapshot()["plus_curation_enabled"] is True

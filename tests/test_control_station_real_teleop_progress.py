from __future__ import annotations

from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from control_station.real_teleop.progress import (
    ProgressArmControlModeSelectionIntent,
    ProgressControlAvailability,
    ProgressEntryStepSelectionIntent,
    ProgressOperatorIntent,
    ProgressProfileOption,
    ProgressProfileSelectionIntent,
    available_progress_controls,
    build_progress_ui_payload,
    build_real_teleop_progress_ui_payload,
    progress_profile_selection_intent_from_event,
    progress_arm_control_mode_selection_intent_from_event,
    progress_entry_step_selection_intent_from_event,
    progress_intent_from_event,
)
from control_station.real_teleop_application import (
    build_real_teleop_progress_application,
)
from control_station.web.calibration_web_ui import CalibrationWebUi, _HTML_PAGE
from teleop_runtime.real_teleop.operator_control_runtime import (
    RealTeleopControlInput,
)
from teleop_runtime.real_teleop.progress_contracts import (
    ProgressArmControlModeSelectionCommand,
    ProgressEntryStepSelectionCommand,
    ProgressProfileSelectionCommand,
)
import teleop_runtime.real_teleop.operator_control_runtime as operator_runtime


def test_progress_controls_require_active_attempt() -> None:
    assert available_progress_controls(ProgressControlAvailability()) == ()
    assert available_progress_controls(
        ProgressControlAvailability(
            enabled=True,
            recording_active=True,
            phase="PAUSED_NO_ACTIVE_ATTEMPT",
            allowed_next_step_ids=("place",),
        )
    ) == ()

    assert available_progress_controls(
        ProgressControlAvailability(
            enabled=True,
            recording_active=True,
            phase="ACTIVE_ATTEMPT",
            allowed_next_step_ids=("place",),
            can_undo_last_advance=True,
        )
    ) == (
        "progress_failure",
        "progress_advance",
        "progress_undo",
    )
    assert available_progress_controls(
        ProgressControlAvailability(
            enabled=True,
            recording_active=True,
            phase="ACTIVE_ATTEMPT",
            allowed_next_step_ids=(),
        )
    ) == ("progress_failure",)
    assert available_progress_controls(
        ProgressControlAvailability(
            enabled=True,
            recording_active=True,
            phase="ACTIVE_ATTEMPT",
            allowed_next_step_ids=("place",),
            can_undo_last_advance=True,
            application_held=True,
        )
    ) == ()
    assert available_progress_controls(
        ProgressControlAvailability(
            enabled=True,
            recording_active=True,
            phase="ACTIVE_ATTEMPT",
            allowed_next_step_ids=("place",),
            can_undo_last_advance=True,
            application_held=True,
            failure_promotes_existing_hold=True,
        )
    ) == ("progress_failure",)


def test_progress_event_mapping_is_typed_and_dataset_neutral() -> None:
    assert progress_intent_from_event(
        "progress_advance:regrasp",
        intent_id="intent-1",
    ) == ProgressOperatorIntent(
        intent_id="intent-1",
        kind="ADVANCE_PROGRESS",
        target_progress_step_id="regrasp",
    )
    assert progress_intent_from_event(
        "progress_failure",
        intent_id="intent-2",
    ) == ProgressOperatorIntent(
        intent_id="intent-2",
        kind="REPORT_FAILURE",
    )
    assert progress_intent_from_event("episode_stop", intent_id="intent-3") is None
    assert progress_intent_from_event(
        "progress_advance:",
        intent_id="intent-empty-target",
    ) is None
    assert progress_profile_selection_intent_from_event(
        "progress_profile_select:fixture-profile",
        intent_id="select-1",
    ) == ProgressProfileSelectionIntent(
        intent_id="select-1",
        profile_id="fixture-profile",
    )
    assert progress_profile_selection_intent_from_event(
        "progress_profile_select:",
        intent_id="select-empty",
    ) is None
    assert progress_entry_step_selection_intent_from_event(
        "progress_entry_step_select:write",
        intent_id="entry-1",
    ) == ProgressEntryStepSelectionIntent(
        intent_id="entry-1",
        progress_step_id="write",
    )
    assert progress_entry_step_selection_intent_from_event(
        "progress_entry_step_select:",
        intent_id="entry-empty",
    ) is None
    assert progress_arm_control_mode_selection_intent_from_event(
        "progress_arm_control_mode_select:strict_profile",
        intent_id="arm-mode-1",
    ) == ProgressArmControlModeSelectionIntent(
        intent_id="arm-mode-1",
        mode="strict_profile",
    )
    assert progress_arm_control_mode_selection_intent_from_event(
        "progress_arm_control_mode_select:invalid",
        intent_id="arm-mode-invalid",
    ) is None


def test_non_advance_intent_cannot_smuggle_target() -> None:
    with pytest.raises(ValueError, match="Only ADVANCE_PROGRESS"):
        ProgressOperatorIntent(
            intent_id="intent-invalid",
            kind="REPORT_FAILURE",
            target_progress_step_id="other",
        )


def test_progress_ui_payload_is_ephemeral_and_default_off() -> None:
    disabled = build_progress_ui_payload(
        gui_enabled=False,
        configured=True,
        availability=ProgressControlAvailability(
            enabled=True,
            recording_active=True,
            phase="ACTIVE_ATTEMPT",
            allowed_next_step_ids=("place",),
        ),
        profile_id="fixture",
        current_progress_step_id="pick",
        current_progress_step_display_name="抓取",
        next_steps=(("place", "放置"),),
    )
    enabled = build_progress_ui_payload(
        gui_enabled=True,
        configured=True,
        availability=ProgressControlAvailability(
            enabled=True,
            recording_active=True,
            phase="ACTIVE_ATTEMPT",
            allowed_next_step_ids=("place",),
            can_undo_last_advance=True,
        ),
        profile_id="fixture",
        current_progress_step_id="pick",
        current_progress_step_display_name="抓取",
        next_steps=(("place", "放置"),),
        current_step_left_arm_control="fixed_hold",
        current_step_right_arm_control="teleop",
        current_step_fixed_scope="arm_and_end_effector",
    )

    assert disabled["enabled"] is False
    assert disabled["active_controls"] == []
    assert enabled["enabled"] is True
    assert enabled["next_steps"] == [
        {"progress_step_id": "place", "display_name": "放置"}
    ]
    assert enabled["active_controls"] == [
        "progress_failure",
        "progress_advance",
        "progress_undo",
    ]
    assert enabled["current_step_arm_control"] == {
        "left": "fixed_hold",
        "right": "teleop",
        "fixed_scope": "arm_and_end_effector",
    }
    promoted = build_progress_ui_payload(
        gui_enabled=True,
        configured=True,
        availability=ProgressControlAvailability(
            enabled=True,
            recording_active=True,
            phase="ACTIVE_ATTEMPT",
            application_held=True,
            failure_promotes_existing_hold=True,
        ),
    )
    assert promoted["active_controls"] == ["progress_failure"]
    assert promoted["application_held"] is True
    assert promoted["failure_promotes_existing_hold"] is True


def test_profile_selector_is_available_before_configuration_and_lock() -> None:
    option = ProgressProfileOption(
        profile_id="fixture-progress",
        profile_version="1",
        task_id="fixture-task",
        display_name="fixture-progress",
        step_count=3,
    )
    payload = build_progress_ui_payload(
        gui_enabled=True,
        configured=False,
        availability=ProgressControlAvailability(),
        available_profiles=(option,),
        profile_selection_enabled=True,
    )

    assert payload["enabled"] is True
    assert payload["configured"] is False
    assert payload["active_controls"] == ["progress_profile_select"]
    assert payload["available_profiles"] == [option.to_payload()]

    locked = build_progress_ui_payload(
        gui_enabled=True,
        configured=True,
        availability=ProgressControlAvailability(),
        available_profiles=(option,),
        profile_selection_enabled=True,
        profile_selection_locked=True,
        profile_binding_source="dataset_binding",
    )
    assert locked["active_controls"] == []
    assert locked["profile_selection_locked"] is True


def test_entry_step_selector_is_idle_only_and_independent_of_profile_lock() -> None:
    payload = build_progress_ui_payload(
        gui_enabled=True,
        configured=True,
        availability=ProgressControlAvailability(),
        entry_steps=(("acquire", "取笔"), ("write", "书写")),
        selected_entry_step_id="write",
        entry_step_selection_enabled=True,
        profile_selection_locked=True,
    )

    assert payload["entry_steps"] == [
        {"progress_step_id": "acquire", "display_name": "取笔"},
        {"progress_step_id": "write", "display_name": "书写"},
    ]
    assert payload["selected_entry_step_id"] == "write"
    assert payload["active_controls"] == ["progress_entry_step_select"]

    busy = build_progress_ui_payload(
        gui_enabled=True,
        configured=True,
        availability=ProgressControlAvailability(),
        entry_steps=(("acquire", "取笔"), ("write", "书写")),
        selected_entry_step_id="write",
        entry_step_selection_enabled=False,
    )
    assert busy["active_controls"] == []
    assert busy["entry_step_selection_enabled"] is False


def test_busy_profile_projection_explains_that_selection_returns_after_idle() -> None:
    option = ProgressProfileOption(
        profile_id="fixture-progress",
        profile_version="1",
        task_id="fixture-task",
        display_name="fixture-progress",
        step_count=3,
    )
    payload = build_real_teleop_progress_ui_payload(
        gui_enabled=True,
        profile=None,
        episode=None,
        availability=ProgressControlAvailability(),
        available_profiles=(option,),
        profile_selection_enabled=False,
    )

    assert payload["active_controls"] == []
    assert "回到空闲后" in str(payload["message"])


def test_web_transport_accepts_only_active_progress_events() -> None:
    ui = CalibrationWebUi(open_browser=False)
    ui.publish(
        progress_ui={
            "enabled": True,
            "phase": "ACTIVE_ATTEMPT",
            "next_steps": [
                {
                    "progress_step_id": "place",
                    "display_name": "放置",
                }
            ],
        },
        active_controls=[
            "progress_advance",
            "progress_failure",
            "progress_undo",
        ],
    )

    assert not ui.push_event("progress_advance:")
    assert not ui.push_event("progress_advance:unknown")
    assert ui.push_event("progress_advance:place")
    assert not ui.push_event("progress_failure")
    assert not ui.push_event("progress_advance:place")
    assert ui.consume_events() == ["progress_advance:place"]
    ui.publish(
        active_controls=[
            "progress_advance",
            "progress_failure",
            "progress_undo",
        ]
    )
    assert ui.push_event("progress_failure")
    assert ui.consume_events() == ["progress_failure"]
    ui.publish(
        active_controls=[
            "progress_advance",
            "progress_failure",
            "progress_undo",
        ]
    )
    assert ui.push_event("progress_undo")
    assert ui.consume_events() == ["progress_undo"]
    ui.publish(active_controls=["q"])
    assert not ui.push_event("progress_advance:place")
    assert ui.snapshot()["progress_ui"]["phase"] == "ACTIVE_ATTEMPT"
    assert "/real-teleop-progress.js" in _HTML_PAGE
    assert "将当前 Hold 记为阶段失败" in Path(
        "control_station/web/real_teleop_progress.js"
    ).read_text(encoding="utf-8")


def test_operator_controls_are_grouped_into_presentation_only_sortable_modules() -> None:
    module_ids = (
        "quick_status",
        "calibration_motion",
        "teleop_flow",
        "arm_modes",
        "dataset_configuration",
        "episode_collection",
        "long_horizon_progress",
        "devices_visualization",
    )
    for module_id in module_ids:
        assert (
            _HTML_PAGE.count(
                f'data-operator-module-id="{module_id}"'
            )
            == 1
        )
    assert "/operator-module-layout.js" in _HTML_PAGE
    dataset_chunk = _HTML_PAGE.split(
        'data-operator-module-id="dataset_configuration"',
        maxsplit=1,
    )[1].split(
        'data-operator-module-id="episode_collection"',
        maxsplit=1,
    )[0]
    assert 'id="datasetConfigurationApplyButton"' in dataset_chunk
    collection_chunk = _HTML_PAGE.split(
        'data-operator-module-id="episode_collection"',
        maxsplit=1,
    )[1].split(
        'data-operator-module-id="long_horizon_progress"',
        maxsplit=1,
    )[0]
    assert 'id="episodeStatus"' in collection_chunk
    progress_chunk = _HTML_PAGE.split(
        'data-operator-module-id="long_horizon_progress"',
        maxsplit=1,
    )[1].split(
        'data-operator-module-id="devices_visualization"',
        maxsplit=1,
    )[0]
    assert 'id="longHorizonProgressMount"' in progress_chunk

    script = (
        Path(__file__).resolve().parents[1]
        / "control_station"
        / "web"
        / "operator_module_layout.js"
    ).read_text(encoding="utf-8")
    assert "data-operator-module-id" in script
    assert "dexteleop_operator_module_order_v2" in script
    assert "ArrowUp" in script
    assert "ArrowDown" in script
    assert 'fetch("/event"' not in script
    assert "data-event" not in script


def test_progress_failure_is_immediate_and_does_not_open_confirmation() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "control_station"
        / "web"
        / "real_teleop_progress.js"
    ).read_text(encoding="utf-8")

    assert '() => submitEvent("progress_failure")' in script
    assert "window.confirm" not in script
    assert 'document.getElementById("longHorizonProgressMount")' in script
    assert "item.task_id" in script
    assert "Keep the operator's local draft while polling" in script
    assert 'let renderedDatasetRoot = "";' in script
    assert "const datasetChanged = currentDatasetRoot !== renderedDatasetRoot;" in script
    assert "!datasetChanged" in script
    assert "!Boolean(progress.profile_selection_locked)" in script
    assert "renderedDatasetRoot = currentDatasetRoot;" in script
    assert "当前阶段（本次采集）" in script
    assert "下一阶段（完成当前后进入）" in script
    assert "renderNextStagePreview" in script
    assert 'entryStepSelect.id = "progressEntryStepSelect"' in script
    assert "progress_entry_step_select:${progressStepId}" in script
    assert "前置子任务不会伪造为已采集" in script
    assert script.index(
        "String(item.profile_id) === priorProfileSelection"
    ) < script.index("String(item.profile_id) === selectedProfileId")


def test_web_transport_validates_profile_selection_against_published_catalog() -> None:
    ui = CalibrationWebUi(open_browser=False)
    ui.publish(
        progress_ui={
            "enabled": True,
            "available_profiles": [
                {"profile_id": "fixture-progress"},
            ],
            "profile_selection_locked": False,
        },
        active_controls=["progress_profile_select"],
    )

    assert not ui.push_event("progress_profile_select:")
    assert not ui.push_event("progress_profile_select:unknown")
    assert ui.push_event("progress_profile_select:fixture-progress")
    assert not ui.push_event("progress_profile_select:fixture-progress")
    assert ui.consume_events() == [
        "progress_profile_select:fixture-progress"
    ]

    ui.publish(
        progress_ui={
            "enabled": True,
            "available_profiles": [
                {"profile_id": "fixture-progress"},
            ],
            "profile_selection_locked": True,
        },
        active_controls=["progress_profile_select"],
    )
    assert not ui.push_event("progress_profile_select:fixture-progress")


def test_web_transport_validates_progress_arm_control_mode_selection() -> None:
    ui = CalibrationWebUi(open_browser=False)
    ui.publish(
        progress_ui={
            "enabled": True,
            "arm_control_modes": [
                "strict_profile",
                "profile",
                "ignore_profile",
            ],
            "arm_control_mode_selection_enabled": True,
        },
        active_controls=["progress_arm_control_mode_select"],
    )

    assert not ui.push_event("progress_arm_control_mode_select:invalid")
    assert ui.push_event("progress_arm_control_mode_select:strict_profile")
    assert ui.consume_events() == [
        "progress_arm_control_mode_select:strict_profile"
    ]


def test_web_transport_validates_next_episode_entry_step_selection() -> None:
    ui = CalibrationWebUi(open_browser=False)
    ui.publish(
        progress_ui={
            "enabled": True,
            "entry_steps": [
                {"progress_step_id": "acquire", "display_name": "取笔"},
                {"progress_step_id": "write", "display_name": "书写"},
            ],
            "entry_step_selection_enabled": True,
        },
        active_controls=["progress_entry_step_select"],
    )

    assert not ui.push_event("progress_entry_step_select:")
    assert not ui.push_event("progress_entry_step_select:unknown")
    assert ui.push_event("progress_entry_step_select:write")
    assert not ui.push_event("progress_entry_step_select:acquire")
    assert ui.consume_events() == ["progress_entry_step_select:write"]


def test_concurrent_progress_requests_admit_exactly_one_mutation() -> None:
    ui = CalibrationWebUi(open_browser=False)
    ui.publish(
        progress_ui={
            "enabled": True,
            "phase": "ACTIVE_ATTEMPT",
            "next_steps": [],
        },
        active_controls=["progress_failure"],
    )
    worker_count = 8
    barrier = threading.Barrier(worker_count)
    accepted: list[bool] = []
    result_lock = threading.Lock()

    def submit() -> None:
        barrier.wait()
        result = ui.push_event("progress_failure")
        with result_lock:
            accepted.append(result)

    workers = [threading.Thread(target=submit) for _ in range(worker_count)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=2.0)

    assert all(not worker.is_alive() for worker in workers)
    assert accepted.count(True) == 1
    assert accepted.count(False) == worker_count - 1
    assert ui.consume_events() == ["progress_failure"]


def test_bare_linear_advance_requires_exactly_one_published_target() -> None:
    ui = CalibrationWebUi(open_browser=False)
    ui.publish(
        progress_ui={
            "enabled": True,
            "phase": "ACTIVE_ATTEMPT",
            "next_steps": [
                {
                    "progress_step_id": "place",
                    "display_name": "放置",
                }
            ],
        },
        active_controls=["progress_advance"],
    )
    assert ui.push_event("progress_advance")
    assert ui.consume_events() == ["progress_advance"]

    ui.publish(
        progress_ui={
            "enabled": True,
            "phase": "ACTIVE_ATTEMPT",
            "next_steps": [
                {"progress_step_id": "left", "display_name": "左分支"},
                {"progress_step_id": "right", "display_name": "右分支"},
            ],
        },
        active_controls=["progress_advance"],
    )
    assert not ui.push_event("progress_advance")
    assert ui.push_event("progress_advance:right")


def test_entity_adapter_translates_gui_event_without_running_effects() -> None:
    control = RealTeleopControlInput.__new__(RealTeleopControlInput)
    control._progress_application = build_real_teleop_progress_application(
        SimpleNamespace()
    )
    control._progress_operator_intents = __import__("collections").deque()
    control._progress_profile_selection_intents = (
        __import__("collections").deque()
    )
    control._progress_entry_step_selection_intents = (
        __import__("collections").deque()
    )
    control._progress_operator_intent_sequence = 0

    assert control.submit_progress_ui_event("progress_advance")
    assert not control.submit_progress_ui_event("progress_advance:place")
    assert not control.submit_progress_ui_event("progress_failure")
    first = control.consume_progress_operator_intents()
    assert len(first) == 1
    assert first[0].kind == "ADVANCE_PROGRESS"
    assert first[0].target_progress_step_id is None

    assert control.submit_progress_ui_event("progress_advance:place")
    second = control.consume_progress_operator_intents()
    assert len(second) == 1
    assert second[0].target_progress_step_id == "place"

    assert control.submit_progress_ui_event("progress_failure")
    assert not control.submit_progress_ui_event("episode_stop")

    intents = control.consume_progress_operator_intents()
    assert [intent.kind for intent in intents] == ["REPORT_FAILURE"]

    assert control.submit_progress_profile_selection_ui_event(
        "progress_profile_select:fixture-progress"
    )
    assert not control.submit_progress_profile_selection_ui_event(
        "progress_profile_select:other-progress"
    )
    selections = control.consume_progress_profile_selection_intents()
    assert selections == (
        ProgressProfileSelectionCommand(
            command_id="real-teleop-progress-profile-gui:7",
            profile_id="fixture-progress",
        ),
    )

    assert control.submit_progress_entry_step_selection_ui_event(
        "progress_entry_step_select:write"
    )
    assert not control.submit_progress_entry_step_selection_ui_event(
        "progress_entry_step_select:return"
    )
    assert control.consume_progress_entry_step_selection_intents() == (
        ProgressEntryStepSelectionCommand(
            command_id="real-teleop-progress-entry-step-gui:9",
            progress_step_id="write",
        ),
    )

    assert control.submit_progress_arm_control_mode_selection_ui_event(
        "progress_arm_control_mode_select:strict_profile"
    )
    arm_mode_selections = (
        control.consume_progress_arm_control_mode_selection_intents()
    )
    assert arm_mode_selections == (
        ProgressArmControlModeSelectionCommand(
            command_id="real-teleop-progress-arm-mode-gui:11",
            mode="strict_profile",
        ),
    )


def test_strict_profile_blocks_manual_fix_and_single_arm_pedal_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = RealTeleopControlInput.__new__(RealTeleopControlInput)
    control._progress_application = build_real_teleop_progress_application(
        SimpleNamespace()
    )
    control._progress_ui_payload = {
        "configured": True,
        "arm_control_mode": "strict_profile",
    }
    messages: list[str] = []
    control._foot_pad_mode_enabled = lambda: True
    control._foot_pad_count = lambda: 4
    control._two_pedal_mode_active = lambda: True
    control._handle_progress_pedal_event = lambda _event, *, now: False
    control._publish_progress_pedal_message = (
        lambda message, *, now: messages.append(str(message))
    )
    control.components = SimpleNamespace(
        fix_arm_controller=SimpleNamespace(
            select_side=lambda _side: pytest.fail(
                "strict mode must not reach the manual Fix controller"
            )
        )
    )

    assert not control._manual_fix_arm_controls_enabled()
    monkeypatch.setattr(
        operator_runtime,
        "_gui_current_active_controls",
        lambda _state, _events: [
            "fix_arm_select",
            "fix_arm_toggle",
            "episode_stop",
        ],
        raising=False,
    )
    control._gui_control_state = lambda: object()
    control._gui_events = lambda: object()
    control._progress_ui_controls = ()
    control._plus_curation_save_in_flight = lambda: False
    control._teleop_started = True
    assert control._current_active_controls() == [
        "episode_stop",
        "two_pedal_c",
    ]
    control._progress_operator_intent_sequence = 0
    assert control.submit_progress_arm_control_mode_selection_ui_event(
        "progress_arm_control_mode_select:profile"
    )
    assert not control._manual_fix_arm_controls_enabled()
    assert len(control.consume_progress_arm_control_mode_selection_intents()) == 1
    control._handle_foot_pad_special_arm_event("fix_arm_left", now=10.0)
    control._handle_two_pedal_b_event(now=11.0)

    assert len(messages) == 2
    assert all("严格 Profile" in message for message in messages)

    control._progress_ui_payload = {
        "configured": True,
        "arm_control_mode": "profile",
    }
    assert control._manual_fix_arm_controls_enabled()

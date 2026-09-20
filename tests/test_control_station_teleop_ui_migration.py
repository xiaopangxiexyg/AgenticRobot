from __future__ import annotations

import importlib
from pathlib import Path
import threading

import pytest

from control_station.real_teleop.control_availability import (
    GuiControlEvents,
    GuiControlState,
    current_active_controls,
)
from control_station.real_teleop.input_mapping import prioritize_input_batch
from control_station.umi.control_availability import UmiControlState, active_controls
from control_station.umi.input_mapping import (
    UmiOperatorEvent,
    foot_pad_test_batch_active,
    merge_operator_events,
)


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_obsolete_station_facades_are_absent() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    for relative_path in (
        "teleop_runtime/debug/calibration_web_ui.py",
        "teleop_runtime/gui/calibration_web_ui.py",
        "teleop_runtime/gui/startup_progress.py",
        "teleop_runtime/real_teleop/gui_controls.py",
    ):
        assert not (repository_root / relative_path).exists()


def test_legacy_gui_status_payload_path_delegates_to_real_teleop_owner() -> None:
    canonical = importlib.import_module("teleop_runtime.real_teleop.status_payloads")
    legacy = importlib.import_module("teleop_runtime.gui.status_payloads")

    for name in (
        "_build_camera_status",
        "_build_tactile_camera_status",
        "_build_ik_warning_payload",
        "_build_pose_quality_display_payload",
    ):
        assert getattr(legacy, name) is getattr(canonical, name)


def test_legacy_real_teleop_control_path_is_exact_runtime_module_alias() -> None:
    canonical = importlib.import_module(
        "teleop_runtime.real_teleop.operator_control_runtime"
    )
    legacy = importlib.import_module("teleop_runtime.gui.real_teleop_control")

    assert legacy is canonical
    assert legacy.RealTeleopControlInput is canonical.RealTeleopControlInput


def test_real_teleop_recording_controls_keep_stop_and_hold_without_start() -> None:
    controls = current_active_controls(
        GuiControlState(
            teleop_started=True,
            episode_recording_available=True,
            episode_recording_active=True,
            active_side_count=2,
        ),
        GuiControlEvents(),
    )

    assert "episode_stop" in controls
    assert "episode_start" not in controls
    assert "h" in controls


def test_real_teleop_foot_pad_is_additive_to_manual_controls() -> None:
    controls = current_active_controls(
        GuiControlState(
            teleop_started=True,
            hold_active=True,
            fix_arm_available=True,
            init_arm_available=True,
            special_arm_control_mode="foot-pad",
            foot_pad_available=True,
            foot_pad_count=4,
            active_side_count=2,
        ),
        GuiControlEvents(),
    )

    assert "fix_arm_select" in controls
    assert "init_arm_select" in controls
    assert "foot_pad_test_start" in controls


@pytest.mark.parametrize(
    ("state", "required", "forbidden"),
    (
        (
            UmiControlState(),
            {"episode_start", "episode_delete", "device_retry", "task_select", "q"},
            {"episode_stop"},
        ),
        (
            UmiControlState(recording=True, foot_pad_online=True),
            {"episode_stop", "q", "c", "g"},
            {
                "episode_start",
                "episode_delete",
                "device_retry",
                "task_select",
                "foot_pad_test_start",
            },
        ),
        (
            UmiControlState(calibration_expired=True, plus_save_available=True),
            {
                "q",
                "task_select",
                "umi_recalibrate",
                "angle_recalibrate",
                "plus_curation_save",
            },
            {"episode_start", "episode_delete"},
        ),
        (
            UmiControlState(calibration_expired=True, recording=True),
            {"q", "episode_stop"},
            {"umi_recalibrate", "angle_recalibrate", "episode_start"},
        ),
        (
            UmiControlState(
                collection_blocked=True,
                foot_pad_online=True,
                foot_pad_test_active=True,
                plus_save_in_flight=True,
            ),
            {"device_retry", "task_select", "foot_pad_test_stop"},
            {"episode_start", "episode_delete", "foot_pad_test_start"},
        ),
        (
            UmiControlState(dataset_reconfiguration_available=False),
            {"episode_start", "episode_delete", "device_retry", "q"},
            {"task_select"},
        ),
        (
            UmiControlState(episode_deletion_available=False),
            {"episode_start", "task_select", "device_retry", "q"},
            {"episode_delete"},
        ),
    ),
)
def test_umi_control_availability_corner_cases(
    state: UmiControlState,
    required: set[str],
    forbidden: set[str],
) -> None:
    controls = active_controls(state)

    assert required.issubset(controls)
    assert forbidden.isdisjoint(controls)
    assert len(controls) == len(set(controls))


def test_umi_control_availability_rejects_mutable_or_untyped_state() -> None:
    with pytest.raises(TypeError, match="UmiControlState"):
        active_controls({})  # type: ignore[arg-type]


def test_real_teleop_input_batch_preserves_order_and_duplicates_without_barrier() -> None:
    result = prioritize_input_batch(
        ["episode_start", "w", "w", "fix_arm_toggle"],
        terminal_quit_requested=False,
        teleop_started=True,
        two_pedal_mode_active=True,
    )

    assert result.events == ("episode_start", "w", "w", "fix_arm_toggle")
    assert result.barrier == "none"
    assert result.suppress_foot_pad_events is False


def test_rule_home_retry_is_not_silently_dropped_with_same_cycle_hold() -> None:
    result = prioritize_input_batch(
        ["h", "runtime_home_retry"],
        terminal_quit_requested=False,
        teleop_started=True,
        two_pedal_mode_active=False,
    )

    assert result.events == ("h", "runtime_home_retry")
    assert result.barrier == "hold"


@pytest.mark.parametrize(
    ("events", "terminal_quit", "expected_events", "expected_barrier"),
    (
        (["h", "q", "episode_stop"], False, ("q",), "ui_quit"),
        (
            ["h", "init_qpos_emergency_stop", "episode_stop"],
            False,
            ("init_qpos_emergency_stop",),
            "init_emergency_stop",
        ),
        (
            ["episode_stop", "h", "w"],
            False,
            ("episode_stop", "h"),
            "hold",
        ),
        (["episode_start", "w"], True, (), "terminal_quit"),
    ),
)
def test_real_teleop_input_barriers_match_legacy_precedence(
    events: list[str],
    terminal_quit: bool,
    expected_events: tuple[str, ...],
    expected_barrier: str,
) -> None:
    result = prioritize_input_batch(
        events,
        terminal_quit_requested=terminal_quit,
        teleop_started=True,
        two_pedal_mode_active=True,
    )

    assert result.events == expected_events
    assert result.barrier == expected_barrier
    assert result.suppress_foot_pad_events is True


def test_real_teleop_two_pedal_hold_requires_started_active_mode() -> None:
    not_started = prioritize_input_batch(
        ["two_pedal_c", "episode_start"],
        terminal_quit_requested=False,
        teleop_started=False,
        two_pedal_mode_active=True,
    )
    not_two_pedal = prioritize_input_batch(
        ["two_pedal_c", "episode_start"],
        terminal_quit_requested=False,
        teleop_started=True,
        two_pedal_mode_active=False,
    )

    assert not_started.barrier == "none"
    assert not_two_pedal.barrier == "none"


def test_umi_event_merge_preserves_web_first_order_and_one_cycle_timestamp() -> None:
    result = merge_operator_events(
        ["episode_start", "fix_arm_select:right"],
        ["fix_arm_left", "fix_arm_right"],
        timestamp=12.5,
    )

    assert result == (
        UmiOperatorEvent("episode_start", "web_gui", 12.5),
        UmiOperatorEvent("fix_arm_select:right", "web_gui", 12.5),
        UmiOperatorEvent("fix_arm_left", "foot_pad", 12.5),
        UmiOperatorEvent("fix_arm_right", "foot_pad", 12.5),
    )


@pytest.mark.parametrize("barrier", ("foot_pad_test_start", "foot_pad_test_stop"))
def test_umi_test_barrier_keeps_same_cycle_pedal_edges_diagnostic(barrier: str) -> None:
    batch = merge_operator_events(
        [barrier],
        ["fix_arm_left"],
        timestamp=20.0,
    )

    assert foot_pad_test_batch_active(batch, test_already_active=False) is True


def test_umi_non_test_batch_does_not_consume_real_pedal_as_diagnostic() -> None:
    batch = merge_operator_events([], ["fix_arm_left"], timestamp=20.0)

    assert foot_pad_test_batch_active(batch, test_already_active=False) is False


def test_web_ui_concurrent_episode_start_is_admitted_exactly_once() -> None:
    from control_station.web.calibration_web_ui import CalibrationWebUi

    ui = CalibrationWebUi(open_browser=False)
    ui.publish(
        active_controls=[
            "q",
            "episode_start",
            "episode_delete",
            "device_retry",
            "task_select",
        ]
    )
    worker_count = 8
    start_barrier = threading.Barrier(worker_count)
    accepted: list[bool] = []
    result_lock = threading.Lock()

    def submit() -> None:
        start_barrier.wait()
        result = ui.push_event("episode_start")
        with result_lock:
            accepted.append(result)

    threads = [threading.Thread(target=submit) for _ in range(worker_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)

    assert all(not thread.is_alive() for thread in threads)
    assert accepted.count(True) == 1
    assert accepted.count(False) == worker_count - 1
    assert ui.consume_events() == ["episode_start"]
    snapshot = ui.snapshot()
    assert snapshot["episode_event_status"] == "preparing"
    assert "episode_start" not in snapshot["active_controls"]
    assert "episode_delete" not in snapshot["active_controls"]
    assert "device_retry" not in snapshot["active_controls"]
    assert "task_select" not in snapshot["active_controls"]


def test_web_ui_concurrent_dataset_rebind_is_admitted_exactly_once() -> None:
    from control_station.web.calibration_web_ui import CalibrationWebUi

    ui = CalibrationWebUi(open_browser=False)
    ui.publish(
        active_controls=[
            "q",
            "task_select",
            "episode_start",
            "episode_delete",
            "device_retry",
        ]
    )
    event = 'task_select:{"dataset_base_root":"/tmp/next"}'
    worker_count = 8
    start_barrier = threading.Barrier(worker_count)
    accepted: list[bool] = []
    result_lock = threading.Lock()

    def submit() -> None:
        start_barrier.wait()
        result = ui.push_event(event)
        with result_lock:
            accepted.append(result)

    threads = [threading.Thread(target=submit) for _ in range(worker_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)

    assert all(not thread.is_alive() for thread in threads)
    assert accepted.count(True) == 1
    assert accepted.count(False) == worker_count - 1
    assert ui.consume_events() == [event]
    snapshot = ui.snapshot()
    assert "task_select" not in snapshot["active_controls"]
    assert "episode_start" not in snapshot["active_controls"]
    assert "episode_delete" not in snapshot["active_controls"]
    assert "device_retry" not in snapshot["active_controls"]


def test_web_ui_dataset_rebind_and_episode_start_share_one_pending_slot() -> None:
    from control_station.web.calibration_web_ui import CalibrationWebUi

    ui = CalibrationWebUi(open_browser=False)
    ui.publish(
        active_controls=[
            "q",
            "task_select",
            "episode_start",
            "episode_delete",
            "device_retry",
        ]
    )
    events = (
        'task_select:{"dataset_base_root":"/tmp/next"}',
        "episode_start",
    )
    start_barrier = threading.Barrier(len(events))
    results: list[tuple[str, bool]] = []
    result_lock = threading.Lock()

    def submit(event: str) -> None:
        start_barrier.wait()
        accepted = ui.push_event(event)
        with result_lock:
            results.append((event, accepted))

    threads = [
        threading.Thread(target=submit, args=(event,)) for event in events
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)

    assert all(not thread.is_alive() for thread in threads)
    assert sum(accepted for _, accepted in results) == 1
    queued = ui.consume_events()
    assert len(queued) == 1
    assert queued[0] in events
    snapshot = ui.snapshot()
    assert "task_select" not in snapshot["active_controls"]
    assert "episode_start" not in snapshot["active_controls"]
    assert "episode_delete" not in snapshot["active_controls"]
    assert "device_retry" not in snapshot["active_controls"]


def test_station_web_ui_satisfies_lower_operator_ui_port_shape() -> None:
    from control_station.web.calibration_web_ui import CalibrationWebUi

    ui = CalibrationWebUi(open_browser=False)
    assert isinstance(ui.url, str)
    for method_name in (
        "start",
        "close",
        "consume_events",
        "publish",
        "append_message",
    ):
        assert callable(getattr(ui, method_name))

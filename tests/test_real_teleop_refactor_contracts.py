from __future__ import annotations

import argparse
from dataclasses import fields
import inspect
import re
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from teleop_runtime.debug import run_vive_rm65_real_teleop as legacy_entrypoint
from teleop_runtime.gui import collection_tasks as gui_collection_tasks
from teleop_runtime.gui import status_payloads
from teleop_runtime.real_teleop import (
    action_debug_payload,
    cli_value_parsers,
    components,
    dataset_finalize,
    dataset_layout,
    debug_payload_primitives,
    debug_ik_payload,
    debug_timing_payload,
)


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "teleop_runtime/debug/run_vive_rm65_real_teleop.py"
EXTRACTED_ROOT = ROOT / "teleop_runtime/real_teleop"

# Temporary staged-refactor measurements.  Update these values deliberately at
# the start/end of every extraction or in-place slimming batch; do not silently
# let the baseline drift.
BASELINE_ENTRYPOINT_LINES = 14_536
BASELINE_EXTRACTED_MODULE_COUNT = 25
BASELINE_EXTRACTED_LINES = 5_648
BASELINE_GUI_CONTROL_LINES = 2_297
BASELINE_DIRECT_REAL_TELEOP_IMPORTS = 26
BASELINE_BRIDGE_MARKERS = 17


def test_real_teleop_light_dedup_measurements_are_frozen() -> None:
    entrypoint_lines = len(ENTRYPOINT.read_text(encoding="utf-8").splitlines())
    module_paths = sorted(EXTRACTED_ROOT.glob("*.py"))
    extracted_lines = sum(
        len(path.read_text(encoding="utf-8").splitlines()) for path in module_paths
    )
    gui_control_lines = len(
        (ROOT / "teleop_runtime/gui/real_teleop_control.py")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    source = ENTRYPOINT.read_text(encoding="utf-8")

    assert entrypoint_lines == BASELINE_ENTRYPOINT_LINES
    assert len(module_paths) == BASELINE_EXTRACTED_MODULE_COUNT
    assert extracted_lines == BASELINE_EXTRACTED_LINES
    assert gui_control_lines == BASELINE_GUI_CONTROL_LINES
    assert len(re.findall(r"^from \.\.real_teleop", source, flags=re.MULTILINE)) == (
        BASELINE_DIRECT_REAL_TELEOP_IMPORTS
    )
    assert len(
        re.findall(r"^# (?:MIGRATION|GUI batch|Compatibility bridge)", source, flags=re.MULTILINE)
    ) == BASELINE_BRIDGE_MARKERS


def test_gui_and_headless_frontends_share_writer_runtime_boundary() -> None:
    debug_source = ENTRYPOINT.read_text(encoding="utf-8")
    collection_runtime_source = (
        ROOT / "teleop_runtime/collection/runtime.py"
    ).read_text(encoding="utf-8")
    writer_runtime_source = (
        ROOT / "teleop_runtime/collection/writer_runtime.py"
    ).read_text(encoding="utf-8")

    assert "from ..collection.writer_runtime import CollectionWriterRuntime" in debug_source
    assert "self.writer_runtime = CollectionWriterRuntime(" in debug_source
    assert "self.writer_runtime = CollectionWriterRuntime(" in collection_runtime_source
    assert "run_vive_rm65_real_teleop" not in writer_runtime_source
    assert "teleop_runtime.debug" not in writer_runtime_source


def test_legacy_runtime_entrypoint_signatures_are_frozen() -> None:
    assert list(inspect.signature(legacy_entrypoint.parse_args).parameters) == []
    assert list(inspect.signature(legacy_entrypoint.main).parameters) == []
    assert list(inspect.signature(legacy_entrypoint._build_components).parameters) == ["args"]
    control_parameters = list(
        inspect.signature(legacy_entrypoint.RealTeleopControlInput.__init__).parameters
    )
    assert control_parameters == [
        "self",
        "terminal",
        "ui",
        "args",
        "components",
        "pose_snapshot_recorder",
        "foot_pad_input",
    ]

    recorder_methods = (
        "start",
        "start_episode",
        "record_step",
        "record_step_async",
        "pause_episode",
        "resume_episode",
        "stop_episode",
        "poll_finalize_tasks",
        "wait_finalize_tasks",
        "close",
    )
    for method_name in recorder_methods:
        assert callable(getattr(legacy_entrypoint.RealTeleopDatasetRecorder, method_name))


def test_batch_one_payload_modules_preserve_legacy_names_and_do_not_reverse_import() -> None:
    assert legacy_entrypoint._json_safe_value is status_payloads._json_safe_value
    assert legacy_entrypoint._build_tactile_camera_status is status_payloads._build_tactile_camera_status
    assert legacy_entrypoint._action_debug_jitter_focus is action_debug_payload._action_debug_jitter_focus
    assert (
        legacy_entrypoint._action_debug_backend_snapshot_from_state
        is action_debug_payload._action_debug_backend_snapshot_from_state
    )

    for path in (
        ROOT / "teleop_runtime/gui/status_payloads.py",
        ROOT / "teleop_runtime/real_teleop/action_debug_payload.py",
        ROOT / "teleop_runtime/real_teleop/cli_value_parsers.py",
        ROOT / "teleop_runtime/real_teleop/components.py",
        ROOT / "teleop_runtime/real_teleop/dataset_finalize.py",
        ROOT / "teleop_runtime/real_teleop/debug_payload_primitives.py",
        ROOT / "teleop_runtime/real_teleop/debug_ik_payload.py",
        ROOT / "teleop_runtime/real_teleop/debug_timing_payload.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "run_vive_rm65_real_teleop" not in source
        assert "teleop_runtime.debug" not in source


def test_batch_two_pure_modules_preserve_legacy_identity_and_contracts() -> None:
    assert legacy_entrypoint.RealTeleopComponents is components.RealTeleopComponents
    assert legacy_entrypoint.RealTeleopComponents.__module__ == legacy_entrypoint.__name__
    component_fields = tuple(item.name for item in fields(components.RealTeleopComponents))
    assert component_fields[:4] == (
        "session",
        "active_sides",
        "left_command_backend",
        "right_command_backend",
    )
    assert component_fields[-3:] == (
        "fix_arm_controller",
        "init_arm_controller",
        "position_reference_publisher",
    )

    for name in (
        "_non_negative_float",
        "_canfd_trajectory_mode",
        "_canfd_radio",
        "_movement_multiple_arg",
        "_local_step_clip_scale_arg",
        "_positive_int",
        "_dmtac_worker_cpu_affinity",
        "_dmtac_worker_nice",
        "_parse_dmtac_tactile_fields",
    ):
        assert getattr(legacy_entrypoint, name) is getattr(cli_value_parsers, name)

    assert legacy_entrypoint.RealDatasetFinalizeTask is dataset_finalize.RealDatasetFinalizeTask
    assert legacy_entrypoint.RealDatasetFinalizeTask.__module__ == legacy_entrypoint.__name__
    task = dataset_finalize.RealDatasetFinalizeTask(
        episode_index=3,
        frame_count=4,
        reason="test",
        queued_at=1.0,
    )
    assert task.pending and not task.running
    task.thread = threading.Thread(target=lambda: None)
    assert task.running
    task.mark_started()
    task.record_stage_timing("write", 0.0125)
    task.mark_done(success=True)
    assert task.done and task.success and not task.pending and not task.running
    assert task.stage_timings_ms == {"write": 12.5}

    for name in (
        "_pose_matrix_debug",
        "_qpos_debug",
        "_compact_numeric_list",
        "_compact_debug_value",
        "_pose_compact_debug",
        "_qpos_compact_debug",
        "_pose_delta_compact_debug",
    ):
        assert getattr(legacy_entrypoint, name) is getattr(debug_payload_primitives, name)


def test_dataset_layout_module_preserves_legacy_adapters_and_import_boundary() -> None:
    for legacy_name, module_name in (
        ("_expand_cli_path", "expand_cli_path"),
        ("_sanitize_dataset_name_for_path", "sanitize_dataset_name_for_path"),
        ("_default_task_id_from_name", "default_task_id_from_name"),
    ):
        assert getattr(legacy_entrypoint, legacy_name) is getattr(dataset_layout, module_name)

    assert list(inspect.signature(legacy_entrypoint._load_task_registry).parameters) == ["args"]
    assert list(inspect.signature(legacy_entrypoint._task_definition_for_key).parameters) == [
        "args",
        "key",
    ]
    assert list(inspect.signature(legacy_entrypoint._collection_task_ui_state).parameters) == ["args"]

    assert dataset_layout.expand_cli_path("~/dexteleop").name == "dexteleop"
    assert dataset_layout.sanitize_dataset_name_for_path("Pick  chips") == "Pick_chips"
    assert dataset_layout.default_task_id_from_name("Pick  chips") == "pick_chips"
    layout_args = SimpleNamespace(
        dataset_root="/tmp/dexteleop-layout-contract",
        dataset_name="",
        task_config="",
        collection_task_key="",
        collection_task_id="",
        gripper_task="",
        collection_task_type="",
        collection_task_name="",
        collection_task_name_zh="",
        collection_task_description="",
    )
    layout = dataset_layout.resolve_formal_dataset_layout(layout_args, date_prefix="2026-07-15")
    assert isinstance(layout, dataset_layout.FormalDatasetLayout)
    assert layout.dataset_dir_name == "2026-07-15_vive_rm65_real_teleop"
    assert layout.dataset_root == Path("/tmp/dexteleop-layout-contract") / layout.dataset_dir_name
    assert gui_collection_tasks.FormalDatasetLayout is dataset_layout.FormalDatasetLayout
    assert gui_collection_tasks.resolve_formal_dataset_layout is dataset_layout.resolve_formal_dataset_layout
    assert gui_collection_tasks.toggle_round_trip_phase is dataset_layout.toggle_round_trip_phase
    for path in (
        ROOT / "teleop_runtime/real_teleop/dataset_layout.py",
        ROOT / "teleop_runtime/gui/collection_tasks.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "run_vive_rm65_real_teleop" not in source
        assert "teleop_runtime.debug" not in source


def test_debug_ik_payload_module_preserves_legacy_identity_and_import_boundary() -> None:
    for name in (
        "_ik_profile_compact_debug",
        "_execution_step_compact_debug",
        "_candidate_selection_compact_debug",
        "_command_metadata_compact_debug",
        "_teleop_command_metadata_compact_debug",
        "_command_compact_debug_payload",
        "_impedance_record_compact_debug",
        "_backend_ik_metadata_compact_debug",
        "_backend_compact_debug_payload",
        "_observation_pose_compact_debug",
        "_command_debug_payload",
        "_teleop_command_metadata_debug",
        "_impedance_record_debug",
        "_backend_debug_payload",
        "_observation_pose_debug",
    ):
        assert getattr(legacy_entrypoint, name) is getattr(debug_ik_payload, name)
    for name in (
        "_IK_PROFILE_COMPACT_KEYS",
        "_IMPEDANCE_RECORD_BASE_KEYS",
        "_IMPEDANCE_RECORD_BACKEND_TARGET_KEYS",
        "_IMPEDANCE_RECORD_FORCE_KEYS",
        "_IMPEDANCE_RECORD_TARGET_OFFSET_KEYS",
        "_IMPEDANCE_RECORD_REALIZATION_KEYS",
        "_IMPEDANCE_RECORD_REALIZATION_TAIL_KEYS",
        "_IMPEDANCE_RECORD_RETURN_LIMIT_KEYS",
        "_IMPEDANCE_RECORD_JOINT_LIMIT_PROFILE_KEYS",
        "_IMPEDANCE_RECORD_COMMON_TAIL_KEYS",
        "_IMPEDANCE_RECORD_POSE_TRACKING_KEYS",
        "_IMPEDANCE_RECORD_COMPACT_KEYS",
        "_IMPEDANCE_RECORD_DEBUG_KEYS",
    ):
        assert getattr(legacy_entrypoint, name) is getattr(debug_ik_payload, name)
    source = (ROOT / "teleop_runtime/real_teleop/debug_ik_payload.py").read_text(
        encoding="utf-8"
    )
    assert "run_vive_rm65_real_teleop" not in source
    assert "teleop_runtime.debug" not in source


def test_batch_two_cli_parsers_and_debug_primitives_are_pure() -> None:
    assert cli_value_parsers._non_negative_float("0.5") == 0.5
    assert cli_value_parsers._canfd_trajectory_mode("2") == 2
    assert cli_value_parsers._canfd_radio("100") == 100
    assert cli_value_parsers._movement_multiple_arg("1.25") == 1.25
    assert cli_value_parsers._local_step_clip_scale_arg("1") == 1.0
    assert cli_value_parsers._positive_int("3") == 3
    assert cli_value_parsers._dmtac_worker_cpu_affinity("2, 1 2") == (2, 1)
    assert cli_value_parsers._dmtac_worker_nice("19") == 19
    with pytest.raises(argparse.ArgumentTypeError):
        cli_value_parsers._canfd_radio("101")
    with pytest.raises(argparse.ArgumentTypeError):
        cli_value_parsers._movement_multiple_arg("nan")
    with pytest.raises(argparse.ArgumentTypeError):
        cli_value_parsers._positive_int("0")

    pose = np.eye(4)
    pose[:3, 3] = (0.1, 0.2, 0.3)
    compact = debug_payload_primitives._pose_compact_debug(pose)
    assert compact == {"xyz_m": [0.1, 0.2, 0.3], "rpy_deg": [0.0, 0.0, 0.0]}
    assert debug_payload_primitives._pose_delta_compact_debug(pose, pose) == {
        "translation_m": 0.0,
        "rotation_deg": 0.0,
    }
    assert debug_payload_primitives._compact_debug_value(
        {"values": np.arange(20), "ik_resolver_init": "omitted"},
        max_items=4,
    )["values"]["omitted"] is True


def test_batch_two_ik_payloads_match_for_metadata_snapshots() -> None:
    command = SimpleNamespace(
        metadata={
            "status": "failed",
            "ik_debug_profile": {
                "frame_total_ms": 8.5,
                "solve_total_ms": 6.0,
                "time_budget_ms": 10.0,
                "time_budget_exceeded": False,
            },
            "singularity_gate_status": "hold_entry",
        },
        commands={},
    )
    payload = status_payloads._build_ik_timing_payload(
        active_command=command,
        left_command_state={"ik_metadata": {"status": "hold_last_safe"}},
    )
    assert payload == legacy_entrypoint._build_ik_timing_payload(
        active_command=command,
        left_command_state={"ik_metadata": {"status": "hold_last_safe"}},
    )
    assert payload["available"] is True
    assert payload["ik_warning"]["active"] is True
    assert payload["ik_warning"]["codes"] == ["failed", "hold_entry", "hold_last_safe"]
    assert status_payloads._build_ik_timing_payload(active_command=None)["available"] is False


def test_batch_three_debug_timing_payloads_preserve_snapshot_outputs() -> None:
    for name in (
        "_fault_signature",
        "_phase_timings_log_payload",
        "_latency_items_log_payload",
        "_latency_log_payload",
        "_faults_log_payload",
        "_channels_log_payload",
    ):
        assert getattr(legacy_entrypoint, name) is getattr(debug_timing_payload, name)

    fault = SimpleNamespace(
        code="hold_last_safe",
        channel_name="left_arm",
        component="ik",
        severity=SimpleNamespace(value="error"),
        message="test fault",
        timestamp=9.0,
        metadata={"score": np.asarray([1.0, 2.0])},
    )
    channel = SimpleNamespace(
        valid=True,
        critical=True,
        capability="pose",
        side="left",
        frame_name="desk",
        source_name="vive",
        timestamp=9.5,
        metadata={"seq": 7, "sender_timestamp": 9.25, "ignored": "x"},
    )
    result = SimpleNamespace(faults=(fault,), channels={"left_wrist_pose": channel})
    assert debug_timing_payload._fault_signature(result.faults) == ("hold_last_safe:left_arm",)
    assert debug_timing_payload._faults_log_payload(result) == legacy_entrypoint._faults_log_payload(result)
    assert debug_timing_payload._channels_log_payload(result, now=10.0) == legacy_entrypoint._channels_log_payload(
        result,
        now=10.0,
    )
    assert debug_timing_payload._phase_timings_log_payload(
        [("slow", 0.02), ("fast", 0.001)]
    )["top"][0]["name"] == "slow"
    assert debug_timing_payload._latency_log_payload(None, now=10.0) == {
        "reference_time": 10.0,
        "step_age_ms": None,
        "observation_age_ms": None,
        "command_age_ms": None,
        "channel_age": [],
        "receiver_age": [],
        "sender_age": [],
    }


def test_in_place_impedance_debug_catalogs_preserve_order_and_special_handling() -> None:
    compact_keys = legacy_entrypoint._IMPEDANCE_RECORD_COMPACT_KEYS
    debug_keys = legacy_entrypoint._IMPEDANCE_RECORD_DEBUG_KEYS

    assert len(compact_keys) == 130
    assert len(debug_keys) == 123
    assert compact_keys[:11] == debug_keys[:11]
    assert compact_keys[11:24] == (
        "backend_target_sequence",
        "backend_target_timestamp",
        "backend_target_present",
        "backend_target_joint_hint_present",
        "backend_target_dispatched_this_step",
        "backend_last_dispatched_target_sequence",
        "backend_target_snapshot_wall_time",
        "backend_controller_step_start_wall_time",
        "backend_controller_step_end_wall_time",
        "backend_controller_step_wall_duration_sec",
        "backend_target_age_sec_at_snapshot",
        "backend_target_age_sec_at_step_start",
        "backend_target_age_sec_at_step_end",
    )
    assert debug_keys[11:20] == compact_keys[24:33]
    assert "ik_metadata" not in compact_keys
    assert debug_keys[-3:] == (
        "ik_metadata",
        "pose_tracking_linear_m",
        "pose_tracking_angular_rad",
    )

    record = {
        key: key
        for key in set(compact_keys) | set(debug_keys) | {"ik_metadata"}
    }
    record["ik_metadata"] = {"status": "ok"}
    compact = legacy_entrypoint._impedance_record_compact_debug(record)
    debug = legacy_entrypoint._impedance_record_debug(record)

    assert tuple(compact) == compact_keys + ("ik_metadata",)
    assert compact["ik_metadata"] == {"status": "ok"}
    assert tuple(debug) == debug_keys
    assert debug["ik_metadata"] is record["ik_metadata"]


def test_batch_one_action_debug_keeps_backend_read_in_legacy_seam() -> None:
    class CountingBackend:
        def __init__(self) -> None:
            self.reads = 0

        def get_state(self) -> dict[str, object]:
            self.reads += 1
            return {
                "last_command_timestamp": 1.0,
                "current_tcp_pose": None,
                "last_target_tcp_pose": None,
            }

    backend = CountingBackend()
    legacy_entrypoint._action_debug_backend_snapshot(
        backend,
        side="left",
        role="command_backend",
        sample_timestamp=2.0,
        record_wall_time=2.1,
    )
    assert backend.reads == 1

    state = {
        "last_command_timestamp": 1.0,
        "current_tcp_pose": None,
        "last_target_tcp_pose": None,
    }
    legacy_entrypoint._action_debug_backend_snapshot_from_state(
        state,
        backend=backend,
        side="left",
        role="command_backend",
        sample_timestamp=2.0,
        record_wall_time=2.1,
    )
    assert backend.reads == 1


def test_batch_one_status_payloads_match_for_snapshot_inputs() -> None:
    camera_status = {
        "left_wrist": {
            "status_level": "warn",
            "status_label": "帧延迟",
            "status_message": "last frame age 0.8s",
            "depth_required": True,
            "depth_status_level": "ok",
        },
        "right_wrist": {
            "status_level": "ok",
            "depth_required": True,
            "depth_status_level": "error",
            "depth_status_label": "Wrist Depth 失效",
            "depth_status_message": "missing depth",
        },
    }
    assert status_payloads._camera_status_blocking_message(
        camera_status,
        require_ok=False,
    ) == legacy_entrypoint._camera_status_blocking_message(
        camera_status,
        require_ok=False,
    )

    tactile_input = {
        "left": {
            "configured": True,
            "connect_tactile": True,
            "is_connected": True,
            "tactile_devices": ["dev0"],
            "tactile_ready": {"dev0": True},
            "tactile_frame_ids": {"dev0": 7},
            "tactile_timestamp": 9.5,
            "tactile_read_hz": 15.0,
        }
    }
    assert status_payloads._build_tactile_camera_status(
        tactile_input,
        now=10.0,
    ) == legacy_entrypoint._build_tactile_camera_status(
        tactile_input,
        now=10.0,
    )

def test_control_input_poll_keeps_input_and_publish_order_without_hardware() -> None:
    events: list[str] = []

    class FakeTerminal:
        @property
        def should_quit(self) -> bool:
            events.append("terminal.should_quit")
            return False

        def poll(self) -> None:
            events.append("terminal.poll")

        def consume_calibration_request(self) -> bool:
            events.append("terminal.consume_calibration")
            return False

    class FakeUi:
        def consume_events(self) -> list[str]:
            events.append("ui.consume_events")
            return ["h"]

    class FakeFootPad:
        def consume_events(self) -> list[str]:
            events.append("foot_pad.consume_events")
            return ["fix_arm_left"]

    control = legacy_entrypoint.RealTeleopControlInput.__new__(
        legacy_entrypoint.RealTeleopControlInput
    )
    control.terminal = FakeTerminal()
    control.ui = FakeUi()
    control.foot_pad_input = FakeFootPad()
    control.should_quit = False
    control._teleop_started = True
    control._hold_active = False
    control._calibration_requested = False
    control._calibration_request_from_hold = False
    control._hold_requested = False
    control._camera_visualization_toggle_requested = False
    control._tactile_visualization_toggle_requested = False
    control._ik_debug_log_toggle_requested = False
    control._q6_hold_recalibration_gesture = SimpleNamespace()
    control._handle_foot_pad_special_arm_event = (
        lambda event, *, now: events.append(f"foot_pad.route:{event}")
    )
    control.request_hold = lambda *, reason="manual": events.append(f"request_hold:{reason}")
    control._poll_q6_hold_recalibration_gesture = lambda *, now: events.append("q6.poll")
    control.publish_raw_tracker_pose_state = lambda *, now: events.append("publish.raw_tracker")
    control.publish_gripper_status = lambda *, now: events.append("publish.gripper")

    control.poll()

    assert events == [
        "terminal.poll",
        "foot_pad.consume_events",
        "foot_pad.route:fix_arm_left",
        "terminal.should_quit",
        "terminal.consume_calibration",
        "ui.consume_events",
        "request_hold:manual",
        "q6.poll",
        "publish.raw_tracker",
        "publish.gripper",
    ]

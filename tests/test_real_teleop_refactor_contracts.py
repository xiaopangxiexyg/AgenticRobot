from __future__ import annotations

import argparse
from dataclasses import fields
import inspect
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from teleop_runtime import calibration_flow as shared_calibration_flow
from teleop_runtime import cli_value_parsers as shared_cli_value_parsers
from teleop_runtime import log_utils as shared_log_utils
from teleop_runtime.debug import calibration_flow as legacy_calibration_flow
from teleop_runtime.debug import run_vive_rm65_real_teleop as legacy_entrypoint
from teleop_runtime.debug import run_vive_sapien_virtual as virtual_entrypoint
from teleop_runtime.debug import log_utils as legacy_log_utils
from teleop_runtime.gui import collection_tasks as gui_collection_tasks
from teleop_runtime.gui import status_payloads
from teleop_runtime.real_teleop import (
    action_debug_payload,
    arm_runtime,
    arm_state_sync,
    arm_tool_config,
    camera_runtime,
    cli,
    cli_resolution,
    cli_runtime_config,
    cli_value_parsers,
    command_filters,
    component_builder,
    components,
    control_loop,
    dataset_recorder as dataset_recorder_module,
    dataset_finalize,
    dataset_layout,
    dataset_selection,
    debug_payload_primitives,
    debug_ik_payload,
    debug_log_frames,
    debug_timing_payload,
    defaults,
    diagnostic_setup,
    device_retry_runtime,
    dmtac_config,
    episode_runtime,
    episode_recording_runtime,
    gripper_control,
    gripper_policy,
    gripper_recovery,
    gripper_status,
    gripper_warmup_runtime,
    hand_config,
    hand_cli,
    hand_runtime,
    impedance_force_runtime,
    movement,
    operator_control_runtime,
    position_force_guard,
    position_episode_debug,
    position_tuning,
    plus_curation_runtime,
    recording_admission,
    runtime,
    runtime_debug,
    runtime_health,
    runtime_metadata,
    session_recording,
    shutdown_runtime,
    special_arm_runtime,
    startup_runtime,
    tactile_visualization,
    vive_calibration_runtime,
    vive_calibration_profiles,
    visualization_runtime,
)


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "control_station/real_teleop_entrypoint.py"


def test_control_step_timestamp_refreshes_after_control_plane_stall() -> None:
    assert control_loop._fresh_control_step_timestamp(
        10.0,
        clock=lambda: 11.25,
    ) == pytest.approx(11.25)
    assert control_loop._fresh_control_step_timestamp(
        10.0,
        clock=lambda: 9.5,
    ) == pytest.approx(10.0)


def test_real_teleop_start_and_resume_separate_freshness_from_overall_timeout() -> None:
    assert defaults.runtime_home_fresh_data_timeout_sec(0.25) == pytest.approx(0.25)
    assert defaults.runtime_home_fresh_data_timeout_sec(1.0) == pytest.approx(1.0)
    assert defaults.runtime_home_fresh_data_timeout_sec(10.0) == pytest.approx(1.0)

    runtime_source = (ROOT / "teleop_runtime/real_teleop/runtime.py").read_text(
        encoding="utf-8"
    )
    control_loop_source = (
        ROOT / "teleop_runtime/real_teleop/control_loop.py"
    ).read_text(encoding="utf-8")
    assert "timeout_sec=args.calibration_timeout" in runtime_source
    assert "timeout_sec=args.calibration_timeout" in control_loop_source
    freshness_call = "fresh_data_timeout_sec=runtime_home_fresh_data_timeout_sec("
    assert freshness_call in runtime_source
    assert freshness_call in control_loop_source


def test_formal_writer_ready_handshake_is_not_global_startup_work() -> None:
    runtime_source = (ROOT / "teleop_runtime/real_teleop/runtime.py").read_text(
        encoding="utf-8"
    )

    assert "prepare_startup_dataset_recorder(" not in runtime_source
    assert "formal dataset writer deferred until collection start" in runtime_source
    assert "正在启动隔离 writer 并等待 ready 握手" not in runtime_source


def test_unchanged_dataset_reconfigure_does_not_rescan_finalized_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recorder = object.__new__(dataset_recorder_module.RealTeleopDatasetRecorder)
    recorder.args = SimpleNamespace(
        dataset_root=tmp_path,
        ignore_camera=(),
        ignore_tactile=(),
    )
    recorder.components = SimpleNamespace(active_sides=())
    recorder.dataset_root = tmp_path
    recorder.gripper_force_contract = None
    recorder.configured_for_current_args = lambda: True
    calls: list[str] = []
    recorder.reload_finalized_dataset_state = lambda: calls.append("reload")
    recorder.write_camera_device_config = lambda: calls.append("camera_config")
    monkeypatch.setattr(
        dataset_recorder_module,
        "_gripper_force_dataset_contract_from_args",
        lambda _args: None,
    )

    recorder.reconfigure_for_current_args()

    assert calls == []


def test_transactional_dataset_selection_defers_live_state_and_start_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Ui:
        def __init__(self) -> None:
            self.published: list[dict[str, object]] = []
            self.messages: list[tuple[str, str]] = []

        def publish(self, **state: object) -> None:
            self.published.append(dict(state))

        def append_message(self, title: str, message: str) -> None:
            self.messages.append((title, message))

    args = SimpleNamespace(
        collection_task_name="old",
        collection_task_id="old-id",
        collection_task_name_zh="旧任务",
        dataset_root="/tmp/old-dataset",
        ignore_camera=[],
        ignore_tactile=[],
    )
    plus = SimpleNamespace(
        enabled=True,
        variant_id="old-variant",
        task_profile=SimpleNamespace(
            require_variant=lambda value: (
                value
                if value == "new-variant"
                else (_ for _ in ()).throw(ValueError(value))
            )
        ),
    )
    plus.set_variant = lambda value: setattr(plus, "variant_id", str(value))
    plus.ui_state = lambda: {"plus_curation_variant_id": plus.variant_id}
    control = operator_control_runtime.RealTeleopControlInput.__new__(
        operator_control_runtime.RealTeleopControlInput
    )
    control.args = args
    control.ui = Ui()
    control.components = SimpleNamespace(active_sides=(), gripper_policy={})
    control.plus_curation = plus
    control._dataset_selection_pending = False
    control._episode_start_requested = False
    control._dataset_reconfigure_transactional = True
    control._episode_busy = lambda: False
    control._current_active_controls = list
    policy_calls: list[str] = []
    control._apply_current_gripper_task_policy = lambda: policy_calls.append(
        str(args.collection_task_id)
    )
    callbacks: dict[str, object] = {}

    def apply_selection(selected_args: object, _text: str, **kwargs: object):
        assert kwargs["bind_dataset_contract"] is False
        selected_args.collection_task_name = "new"
        selected_args.collection_task_id = "new-id"
        selected_args.collection_task_name_zh = "新任务"
        selected_args.dataset_root = "/tmp/new-dataset"
        return {"plus_variant_id": "new-variant"}

    def schedule_selection(
        snapshot: dict[str, object],
        **kwargs: object,
    ) -> str:
        callbacks["snapshot"] = snapshot
        callbacks.update(kwargs)
        return "pending"

    monkeypatch.setattr(
        operator_control_runtime,
        "apply_real_teleop_dataset_selection",
        apply_selection,
    )
    monkeypatch.setattr(
        operator_control_runtime,
        "_collection_task_ui_state",
        lambda selected_args: {
            "collection_task_id": selected_args.collection_task_id,
            "dataset_root": str(selected_args.dataset_root),
        },
    )
    control._dataset_reconfigure_callback = schedule_selection

    assert control._apply_episode_start_payload(
        "{}",
        bind_dataset_contract=True,
        start_after_commit=True,
    )
    assert args.collection_task_id == "old-id"
    assert args.dataset_root == "/tmp/old-dataset"
    assert plus.variant_id == "old-variant"
    assert policy_calls == []
    assert control._dataset_selection_pending is True
    assert control._episode_start_requested is False
    assert callbacks["bind_dataset_contract"] is True

    first_snapshot = callbacks["snapshot"]
    assert not control._apply_episode_start_payload("{}")
    assert callbacks["snapshot"] is first_snapshot
    assert args.collection_task_id == "old-id"
    assert args.dataset_root == "/tmp/old-dataset"
    assert control._dataset_selection_pending is True
    assert control._episode_start_requested is False
    assert control.ui.published[-1][
        "dataset_configuration_transition_active"
    ] is True

    callbacks["on_commit"](callbacks["snapshot"])
    assert args.collection_task_id == "new-id"
    assert args.dataset_root == "/tmp/new-dataset"
    assert plus.variant_id == "new-variant"
    assert policy_calls == ["new-id"]
    assert control._dataset_selection_pending is True
    assert control._episode_start_requested is False

    callbacks["on_complete"]()
    assert control._dataset_selection_pending is False
    assert control._episode_start_requested is True
    assert control.ui.published[-1][
        "dataset_configuration_transition_active"
    ] is False
    assert any(
        "数采任务已更新" in message
        for title, message in control.ui.messages
        if title == "Dataset"
    )


def test_transactional_dataset_selection_rolls_back_side_effects_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(
        collection_task_name="old",
        collection_task_id="old-id",
        collection_task_name_zh="旧任务",
        dataset_root="/tmp/old-dataset",
        ignore_camera=[],
        ignore_tactile=[],
    )
    ui = SimpleNamespace(
        messages=[],
        published=[],
    )
    ui.append_message = lambda title, message: ui.messages.append((title, message))
    ui.publish = lambda **state: ui.published.append(dict(state))
    plus = SimpleNamespace(
        enabled=True,
        variant_id="old-variant",
        task_profile=SimpleNamespace(require_variant=lambda value: value),
    )
    plus.set_variant = lambda value: setattr(plus, "variant_id", str(value))
    plus.ui_state = lambda: {"plus_curation_variant_id": plus.variant_id}
    control = operator_control_runtime.RealTeleopControlInput.__new__(
        operator_control_runtime.RealTeleopControlInput
    )
    control.args = args
    control.ui = ui
    control.components = SimpleNamespace(active_sides=(), gripper_policy={})
    control.plus_curation = plus
    control._dataset_selection_pending = False
    control._episode_start_requested = False
    control._dataset_reconfigure_transactional = True
    control._episode_busy = lambda: False
    control._current_active_controls = list
    policy_calls: list[str] = []
    control._apply_current_gripper_task_policy = lambda: policy_calls.append(
        str(args.collection_task_id)
    )
    callbacks: dict[str, object] = {}

    def apply_selection(selected_args: object, _text: str, **_kwargs: object):
        selected_args.collection_task_name = "new"
        selected_args.collection_task_id = "new-id"
        selected_args.dataset_root = "/tmp/new-dataset"
        return {"plus_variant_id": "new-variant"}

    def schedule_selection(
        snapshot: dict[str, object],
        **kwargs: object,
    ) -> str:
        callbacks["snapshot"] = snapshot
        callbacks.update(kwargs)
        return "pending"

    monkeypatch.setattr(
        operator_control_runtime,
        "apply_real_teleop_dataset_selection",
        apply_selection,
    )
    monkeypatch.setattr(
        operator_control_runtime,
        "_collection_task_ui_state",
        lambda selected_args: {
            "collection_task_id": selected_args.collection_task_id,
            "dataset_root": str(selected_args.dataset_root),
        },
    )
    control._dataset_reconfigure_callback = schedule_selection

    assert control._apply_episode_start_payload("{}")
    callbacks["on_commit"](callbacks["snapshot"])
    callbacks["on_failure"](RuntimeError("publish failed"))

    assert args.collection_task_id == "old-id"
    assert args.dataset_root == "/tmp/old-dataset"
    assert plus.variant_id == "old-variant"
    assert policy_calls == ["new-id", "old-id"]
    assert control._dataset_selection_pending is False
    assert ui.published[-1]["dataset_configuration_transition_active"] is False
    assert any("原配置保持不变" in message for _title, message in ui.messages)


def test_pending_dataset_selection_rejects_progress_configuration_intents() -> None:
    control = operator_control_runtime.RealTeleopControlInput.__new__(
        operator_control_runtime.RealTeleopControlInput
    )
    control._dataset_selection_pending = True
    control._progress_profile_selection_intents = __import__("collections").deque()
    control._progress_arm_control_mode_selection_intents = (
        __import__("collections").deque()
    )
    control._progress_entry_step_selection_intents = (
        __import__("collections").deque()
    )
    control._progress_ui_payload = {}
    control._progress_ui_controls = ()
    control._progress_ui_transport_revision = 0

    assert not control.submit_progress_profile_selection_intent(
        operator_control_runtime.ProgressProfileSelectionCommand(
            command_id="profile",
            profile_id="detail",
        )
    )
    assert not control.submit_progress_arm_control_mode_selection_intent(
        operator_control_runtime.ProgressArmControlModeSelectionCommand(
            command_id="mode",
            mode="profile",
        )
    )
    assert not control.submit_progress_entry_step_selection_intent(
        operator_control_runtime.ProgressEntryStepSelectionCommand(
            command_id="entry",
            progress_step_id="write",
        )
    )
    assert not control._progress_profile_selection_intents
    assert not control._progress_arm_control_mode_selection_intents
    assert not control._progress_entry_step_selection_intents


def test_gui_and_headless_frontends_share_writer_runtime_boundary() -> None:
    debug_source = ENTRYPOINT.read_text(encoding="utf-8")
    real_recorder_source = (
        ROOT / "teleop_runtime/real_teleop/dataset_recorder.py"
    ).read_text(encoding="utf-8")
    collection_runtime_source = (
        ROOT / "teleop_runtime/collection/runtime.py"
    ).read_text(encoding="utf-8")
    writer_runtime_source = (
        ROOT / "teleop_runtime/collection/writer_runtime.py"
    ).read_text(encoding="utf-8")

    assert (
        "from teleop_runtime.collection.writer_runtime import CollectionWriterRuntime"
        in debug_source
    )
    assert "from ..collection.writer_runtime import CollectionWriterRuntime" in real_recorder_source
    assert "self.writer_runtime = CollectionWriterRuntime(" in real_recorder_source
    assert "self.writer_runtime = CollectionWriterRuntime(" in collection_runtime_source
    assert "from teleop_runtime.debug" not in real_recorder_source
    assert "from ..debug" not in real_recorder_source
    assert "run_vive_rm65_real_teleop" not in writer_runtime_source
    assert "teleop_runtime.debug" not in writer_runtime_source


def test_legacy_runtime_entrypoint_signatures_are_frozen() -> None:
    assert list(inspect.signature(legacy_entrypoint.parse_args).parameters) == []
    assert list(inspect.signature(cli_resolution.parse_real_teleop_args).parameters) == []
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
    from teleop_runtime.real_teleop.dataset_recorder import RealTeleopDatasetRecorder

    assert legacy_entrypoint.RealTeleopDatasetRecorder is RealTeleopDatasetRecorder
    assert legacy_entrypoint.RealTeleopDatasetRecorder.__module__ == (
        "teleop_runtime.debug.run_vive_rm65_real_teleop"
    )
    cli_resolution_source = (
        ROOT / "teleop_runtime/real_teleop/cli_resolution.py"
    ).read_text(encoding="utf-8")
    assert "from teleop_runtime.debug" not in cli_resolution_source
    assert "from ..debug" not in cli_resolution_source


def test_batch_one_payload_modules_preserve_legacy_names_and_do_not_reverse_import() -> None:
    assert legacy_entrypoint._json_safe_value is status_payloads._json_safe_value
    assert legacy_entrypoint._build_tactile_camera_status is status_payloads._build_tactile_camera_status
    assert legacy_entrypoint._camera_status_for_source is status_payloads._camera_status_for_source
    assert legacy_entrypoint._build_camera_status is status_payloads._build_camera_status
    assert legacy_entrypoint._wrist_depth_requested is status_payloads._wrist_depth_requested
    assert (
        legacy_entrypoint._with_expected_wrist_depth_camera_statuses
        is status_payloads._with_expected_wrist_depth_camera_statuses
    )
    assert legacy_entrypoint.WRIST_DEPTH_CAMERA_KEYS is status_payloads.WRIST_DEPTH_CAMERA_KEYS
    assert legacy_entrypoint._action_debug_jitter_focus is action_debug_payload._action_debug_jitter_focus
    assert (
        legacy_entrypoint._action_debug_backend_snapshot_from_state
        is action_debug_payload._action_debug_backend_snapshot_from_state
    )
    for name in (
        "_action_debug_backend_snapshot",
        "_preloaded_action_debug_backend_snapshot",
        "_dataset_storage_debug_text",
        "_dataset_recording_debug_text",
        "_debug_timing_report",
        "_debug_timing_maybe_report",
    ):
        assert getattr(legacy_entrypoint, name) is getattr(runtime_debug, name)
    assert legacy_entrypoint._DEBUG_TIMING_RED == gripper_status._DEBUG_TIMING_RED
    assert legacy_entrypoint._DEBUG_TIMING_RESET == gripper_status._DEBUG_TIMING_RESET
    for path in (
        ROOT / "teleop_runtime/gui/status_payloads.py",
        ROOT / "teleop_runtime/real_teleop/action_debug_payload.py",
        ROOT / "teleop_runtime/real_teleop/cli_value_parsers.py",
        ROOT / "teleop_runtime/real_teleop/components.py",
        ROOT / "teleop_runtime/real_teleop/dataset_finalize.py",
        ROOT / "teleop_runtime/real_teleop/debug_payload_primitives.py",
        ROOT / "teleop_runtime/real_teleop/debug_ik_payload.py",
        ROOT / "teleop_runtime/real_teleop/debug_timing_payload.py",
        ROOT / "teleop_runtime/real_teleop/runtime_debug.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "run_vive_rm65_real_teleop" not in source
        assert "teleop_runtime.debug" not in source


def test_dataset_recording_debug_uses_process_writer_stats() -> None:
    process_worker = SimpleNamespace(
        stats=lambda: SimpleNamespace(
            submitted_samples=12,
            completed_samples=11,
            rejected_busy=1,
            sample_rejections=0,
            queue_capacity=4,
            queue_high_water=2,
            slots_available=3,
            slot_count=4,
            worker_alive=True,
            worker_error=None,
            sample_process_ms_p50=10.0,
            sample_process_ms_p95=20.0,
        )
    )
    recorder = SimpleNamespace(
        process_worker=process_worker,
        writer_pool=None,
        _storage_queue_stats_payload=lambda: {"enabled": False},
    )

    text = runtime_debug._dataset_recording_debug_text(
        dataset_recorder=recorder,
        components=SimpleNamespace(active_sides=()),
        camera_sources={},
        camera_previewer=None,
        now=1.0,
    )

    assert "writer=mode:process submitted:12 completed:11" in text
    assert "writer=status_error" not in text


def test_extracted_debug_timing_report_keeps_legacy_ansi_frame() -> None:
    messages: list[str] = []
    state = legacy_entrypoint.RealDebugTimingState(
        enabled=True,
        console_writer=SimpleNamespace(submit=messages.append),
    )

    legacy_entrypoint._debug_timing_report(
        state,
        reason="contract",
        elapsed_sec=0.1,
        phases=(),
        step_index=3,
        result=None,
        active_command=None,
        left_hw_state={},
        right_hw_state={},
        left_command_state={},
        right_command_state={},
        target_step_scale=1.0,
        force=True,
    )

    assert len(messages) == 1
    assert messages[0].startswith(legacy_entrypoint._DEBUG_TIMING_RED)
    assert messages[0].endswith(legacy_entrypoint._DEBUG_TIMING_RESET)


def test_batch_two_pure_modules_preserve_legacy_identity_and_contracts() -> None:
    assert legacy_entrypoint.RealTeleopComponents is components.RealTeleopComponents
    assert legacy_entrypoint.RealTeleopComponents.__module__ == (
        "teleop_runtime.debug.run_vive_rm65_real_teleop"
    )
    component_fields = tuple(item.name for item in fields(components.RealTeleopComponents))
    assert component_fields[:4] == (
        "session",
        "active_sides",
        "left_command_backend",
        "right_command_backend",
    )
    assert component_fields[-4:] == (
        "fix_arm_controller",
        "init_arm_controller",
        "position_reference_publisher",
        "startup_home_continuity_gate",
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
    assert legacy_entrypoint.RealDatasetFinalizeTask.__module__ == (
        "teleop_runtime.debug.run_vive_rm65_real_teleop"
    )
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


def test_hand_and_gripper_config_modules_preserve_legacy_identity_and_boundaries() -> None:
    for legacy_name, module_name in (
        ("_hand_backend_mode", "hand_backend_mode"),
        ("_hand_backend_mode_for_side", "hand_backend_mode_for_side"),
        ("_dmtac_sides", "dmtac_sides"),
    ):
        assert getattr(legacy_entrypoint, legacy_name) is getattr(hand_config, module_name)

    for name in (
        "HAND_BACKEND_INSPIRE",
        "HAND_BACKEND_DMTAC_GRIPPER",
        "HAND_BACKEND_NONE",
        "HAND_BACKEND_CHOICES",
    ):
        assert getattr(legacy_entrypoint, name) is getattr(hand_config, name)

    for legacy_name, module_name in (
        ("_profile_value", "profile_value"),
        ("_strip_side_gripper_overrides", "strip_side_gripper_overrides"),
        ("_side_gripper_overrides_from_profile", "side_gripper_overrides_from_profile"),
        ("_merge_gripper_profile", "merge_gripper_profile"),
        ("_load_gripper_task_profile", "load_gripper_task_profile"),
        ("_resolved_gripper_policy", "resolved_gripper_policy"),
        ("_gripper_side_override", "gripper_side_override"),
        ("_resolved_gripper_policy_for_side", "resolved_gripper_policy_for_side"),
        ("_resolved_gripper_policies_by_side", "resolved_gripper_policies_by_side"),
        (
            "_current_gripper_force_dataset_contract",
            "current_gripper_force_dataset_contract",
        ),
        (
            "_gripper_force_dataset_contract_from_args",
            "gripper_force_dataset_contract_from_args",
        ),
        ("_gripper_force_dataset_metadata", "gripper_force_dataset_metadata"),
        ("_clear_gripper_policy_cache", "clear_gripper_policy_cache"),
    ):
        assert getattr(legacy_entrypoint, legacy_name) is getattr(gripper_policy, module_name)
    assert legacy_entrypoint.GRIPPER_POLICY_DEFAULTS is gripper_policy.GRIPPER_POLICY_DEFAULTS
    assert (
        legacy_entrypoint._dmtac_tactile_enabled
        is dmtac_config.dmtac_tactile_enabled
    )
    assert (
        legacy_entrypoint._apply_gripper_policy_to_dmtac_config
        is dmtac_config.apply_gripper_policy_to_dmtac_config
    )

    args = SimpleNamespace(
        hand_backend="inspire",
        left_hand_backend="none",
        right_hand_backend="dmtac-gripper",
    )
    assert hand_config.hand_backend_mode(args) == "inspire"
    assert hand_config.dmtac_sides(args, ("left", "right")) == ("right",)
    with pytest.raises(ValueError, match="Unsupported hand backend side"):
        hand_config.hand_backend_mode_for_side(args, "center")

    for path in (
        ROOT / "teleop_runtime/real_teleop/dmtac_config.py",
        ROOT / "teleop_runtime/real_teleop/hand_config.py",
        ROOT / "teleop_runtime/real_teleop/hand_cli.py",
        ROOT / "teleop_runtime/real_teleop/gripper_policy.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "run_vive_rm65_real_teleop" not in source
        assert "teleop_runtime.debug" not in source


def test_shared_cli_value_parsers_and_hand_argument_groups_preserve_contracts() -> None:
    for name in (
        "_active_sides_from_args",
        "_alpha_float",
        "_non_negative_int",
        "_parse_active_sides",
        "_parse_arm_max_angvel",
        "_parse_joint_order",
        "_positive_float",
    ):
        shared = getattr(shared_cli_value_parsers, name)
        assert getattr(legacy_entrypoint, name) is shared
        assert getattr(virtual_entrypoint, name) is shared

    parser = argparse.ArgumentParser()
    hand_cli.add_hand_backend_arguments(parser)
    hand_cli.add_dmtac_arguments(parser)
    hand_cli.add_gripper_policy_arguments(parser)
    hand_cli.add_inspire_hand_arguments(parser)
    args = parser.parse_args([])
    assert args.hand_backend == "inspire"
    assert args.q6_config == "latest"
    assert args.dmtac_tactile_backend == "cpu"
    assert args.gripper_warmup_probe is True
    assert args.hand_joint_order == legacy_entrypoint.DEFAULT_INSPIRE_HAND_JOINT_ORDER
    assert legacy_entrypoint.DEFAULT_TASKS_CONFIG_PATH is hand_cli.DEFAULT_TASKS_CONFIG_PATH

    source = (ROOT / "teleop_runtime/cli_value_parsers.py").read_text(encoding="utf-8")
    assert "teleop_runtime.debug" not in source
    assert "run_vive" not in source


def test_real_teleop_parser_builder_preserves_registration_contract_and_boundary() -> None:
    parser = cli.build_real_teleop_argument_parser()
    assert len(parser._actions) == 450
    assert sum(len(action.option_strings) for action in parser._actions) == 510
    assert parser._actions[0].dest == "help"
    assert parser._actions[-1].dest == "human_action_shadow_wal_max_mb"
    assert parser.parse_args([]).artifact_layout == "legacy"
    assert parser.parse_args([]).v25p_native_shadow is False
    assert parser.parse_args([]).v25p_native_storage_layout == "record_cjson_v1"
    assert parser.parse_args([]).tactile_bundle_root == ""
    assert parser.parse_args([]).tactile_bundle_min_free_gb == 0.0
    assert parser.parse_args([]).collection_process_startup_timeout_sec == (
        defaults.DEFAULT_COLLECTION_PROCESS_STARTUP_TIMEOUT_SEC
    )
    assert parser.parse_args([]).storage_queue_backpressure_policy == "block"
    assert parser.parse_args([]).position_ik_branch_continuity_anchor is False
    assert parser.parse_args([]).position_joint_limit_guard_mode == "legacy"
    assert parser.parse_args([]).runtime_home_atomic_arm_snapshot is True
    assert parser.parse_args([]).runtime_home_late_weighted_vive is True
    assert parser.parse_args([]).position_admittance_shadow is False
    assert parser.parse_args([]).position_admittance_active is False
    assert parser.parse_args([]).position_admittance_combined_active is False
    assert parser.parse_args([]).position_admittance_active_side is None
    assert parser.parse_args([]).position_admittance_active_envelope == "cautious"
    assert parser.parse_args([]).position_admittance_active_max_tool_excursion_m == 0.08
    assert parser.parse_args([]).position_admittance_translation_response_weight == 1.0
    assert parser.parse_args([]).position_admittance_rotation_response_weight == 1.0
    assert parser.parse_args([]).position_admittance_rotation_filter_tau_sec == pytest.approx(
        0.010857362
    )
    assert (
        parser.parse_args(
            ["--position-admittance-shadow"]
        ).position_admittance_shadow
        is True
    )
    assert (
        parser.parse_args(
            ["--position-admittance-active"]
        ).position_admittance_active
        is True
    )
    assert parser.parse_args([]).idle_tactile_compression is False
    assert (
        parser.parse_args(["--idle-tactile-compression"]).idle_tactile_compression
        is True
    )
    assert (
        parser.parse_args(
            ["--no-idle-tactile-compression"]
        ).idle_tactile_compression
        is False
    )
    assert parser.parse_args([]).tactile_artifact_integrity is False
    assert (
        parser.parse_args(
            ["--tactile-artifact-integrity"]
        ).tactile_artifact_integrity
        is True
    )
    assert (
        parser.parse_args(
            ["--no-tactile-artifact-integrity"]
        ).tactile_artifact_integrity
        is False
    )
    rollback_args = parser.parse_args(
        [
            "--no-runtime-home-atomic-arm-snapshot",
            "--no-runtime-home-late-weighted-vive",
        ]
    )
    assert rollback_args.runtime_home_atomic_arm_snapshot is False
    assert rollback_args.runtime_home_late_weighted_vive is False
    assert parser.parse_args(["--v25p-native-shadow"]).v25p_native_shadow is True
    assert (
        parser.parse_args(
            [
                "--v25p-native-shadow",
                "--v25p-native-storage-layout",
                "parquet_stream_chunks_v1",
            ]
        ).v25p_native_storage_layout
        == "parquet_stream_chunks_v1"
    )

    for name in (
        "DEFAULT_CAMERA_CONFIG_PATH",
        "DEFAULT_DATASET_ROOT",
        "DEFAULT_INIT_ARM_PROFILE_PATH",
        "DEFAULT_INIT_ARM_QPOS_SPEED_SCALE",
        "DEFAULT_POSITION_MAX_ACCELERATION_RAD_S2",
        "DEFAULT_REAL_TELEOP_MAX_TRACKER_ROTATION_JUMP_DEG",
        "DEFAULT_REAL_TELEOP_MAX_TRACKER_TRANSLATION_JUMP_M",
        "INIT_ARM_QPOS_SPEED_BASELINE_PROFILE",
    ):
        assert getattr(legacy_entrypoint, name) is getattr(defaults, name)

    source = (ROOT / "teleop_runtime/real_teleop/cli.py").read_text(encoding="utf-8")
    assert "parse_args(" not in source
    assert "parse_known_args(" not in source
    assert "run_vive_rm65_real_teleop" not in source
    assert "teleop_runtime.debug" not in source


def test_runtime_metadata_and_movement_modules_preserve_legacy_contracts() -> None:
    assert (
        legacy_entrypoint._arm_execution_profile_metadata
        is runtime_metadata.arm_execution_profile_metadata
    )
    assert (
        legacy_entrypoint._position_ik_iteration_metadata
        is runtime_metadata.position_ik_iteration_metadata
    )
    assert legacy_entrypoint._movement_multiple_value is movement.movement_multiple_value
    assert legacy_entrypoint._movement_multiple_ui_state is movement.movement_multiple_ui_state
    assert legacy_entrypoint._apply_movement_multiple is movement.apply_movement_multiple
    assert (
        legacy_entrypoint._resolve_movement_multiple_args
        is movement.resolve_movement_multiple_args
    )
    for name in (
        "MOVEMENT_MULTIPLE_DEFAULT",
        "MOVEMENT_MULTIPLE_GUI_EVENTS",
        "MOVEMENT_MULTIPLE_GUI_MAX",
        "MOVEMENT_MULTIPLE_GUI_MIN",
        "MOVEMENT_MULTIPLE_GUI_STEP",
    ):
        assert getattr(legacy_entrypoint, name) is getattr(movement, name)

    args = SimpleNamespace(movement_multiple=None, translation_scale=1.25)
    assert movement.movement_multiple_value(args) == 1.25
    assert movement.movement_multiple_ui_state(args) == {
        "movement_multiple": 1.25,
        "movement_multiple_min": 0.05,
        "movement_multiple_max": 10.0,
        "movement_multiple_step": 0.1,
    }

    for path in (
        ROOT / "teleop_runtime/real_teleop/movement.py",
        ROOT / "teleop_runtime/real_teleop/runtime_metadata.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "run_vive_rm65_real_teleop" not in source
        assert "teleop_runtime.debug" not in source


def test_component_configuration_helpers_preserve_legacy_identity_and_boundaries() -> None:
    for name in (
        "_alignment_qpos_pair",
        "_force_tare_sec_arg",
        "_impedance_mode_arg",
        "_start_state_summary",
        "_strict_tool_profile_arg",
        "_tool_profile_arg",
        "_tool_tcp_pose_flange_arg",
    ):
        assert getattr(legacy_entrypoint, name) is getattr(arm_tool_config, name)

    for name in (
        "_compose_real_command_filter",
        "_compose_real_command_guard",
        "_compose_real_fault_filter",
        "_filter_manual_real_command",
        "_fix_arm_required_effectors_by_side",
    ):
        assert getattr(legacy_entrypoint, name) is getattr(command_filters, name)

    for name in (
        "_impedance_bimanual_ik_safety_profile",
        "_position_ik_tuning_for_args",
        "_position_tracker_motion_intent_config",
    ):
        assert getattr(legacy_entrypoint, name) is getattr(position_tuning, name)

    for name in (
        "_command_backend_health_signature",
        "_command_backend_status_text",
        "_short_status_text",
        "_temporary_safety_health_signature",
    ):
        assert getattr(legacy_entrypoint, name) is getattr(runtime_health, name)

    for name in (
        "_session_record_root_dir",
        "_session_recording_config",
        "_session_recording_enabled",
    ):
        assert getattr(legacy_entrypoint, name) is getattr(session_recording, name)
    assert (
        legacy_entrypoint._position_force_guard_config
        is position_force_guard.position_force_guard_config
    )

    for path in (
        ROOT / "teleop_runtime/real_teleop/arm_tool_config.py",
        ROOT / "teleop_runtime/real_teleop/command_filters.py",
        ROOT / "teleop_runtime/real_teleop/position_tuning.py",
        ROOT / "teleop_runtime/real_teleop/runtime_health.py",
        ROOT / "teleop_runtime/real_teleop/session_recording.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "run_vive_rm65_real_teleop" not in source
        assert "teleop_runtime.debug" not in source
        assert "from ..debug" not in source

    assert legacy_log_utils.log_event is shared_log_utils.log_event
    assert legacy_log_utils.log_stage is shared_log_utils.log_stage
    assert legacy_log_utils.ChangeGate is shared_log_utils.ChangeGate


def test_runtime_setup_helpers_live_in_non_authoritative_modules() -> None:
    for name in (
        "_attach_position_audit_sink",
        "_attach_actuator_command_trace_sink",
        "_attach_singularity_transition_trace_sink",
        "_new_actuator_command_trace_recorder",
        "_new_position_debug_recorder",
    ):
        assert getattr(legacy_entrypoint, name) is getattr(diagnostic_setup, name)
    assert (
        legacy_entrypoint._build_tactile_visualizer
        is tactile_visualization._build_tactile_visualizer
    )

    for path in (
        ROOT / "teleop_runtime/real_teleop/diagnostic_setup.py",
        ROOT / "teleop_runtime/real_teleop/tactile_visualization.py",
        ROOT / "teleop_runtime/real_teleop/cli_runtime_config.py",
        ROOT / "teleop_runtime/real_teleop/shutdown_runtime.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "from teleop_runtime.debug" not in source
        assert "from ..debug" not in source

    assert list(
        inspect.signature(cli_runtime_config._run_gui_device_preflight).parameters
    ) == ["args", "active_sides", "parser"]
    assert list(
        inspect.signature(cli_runtime_config._discover_runtime_q6_ports).parameters
    ) == ["args", "active_sides"]
    assert not hasattr(cli_runtime_config, "bind_legacy_support")
    assert not hasattr(legacy_entrypoint, "_refresh_cli_runtime_config_support")


def test_shutdown_runtime_preserves_cleanup_registration_order() -> None:
    labels: list[str] = []

    def closeable() -> SimpleNamespace:
        return SimpleNamespace(close=lambda: None)

    runtime_state = {
        "args": SimpleNamespace(),
        "startup_progress": closeable(),
        "position_force_guard_monitor": None,
        "episode_recorder": closeable(),
        "clutch_input": closeable(),
        "dataset_recorder": closeable(),
        "camera_previewer": closeable(),
        "camera_sources": {"camera": object()},
        "tactile_visualizer": closeable(),
        "actuator_trace_recorder": None,
        "components": SimpleNamespace(
            human_shadow_frontend=closeable(),
            session=closeable(),
            arm_state_read_executor=SimpleNamespace(shutdown=lambda **kwargs: None),
            right_hand_backend=closeable(),
            right_hand_control_via_rig=False,
            left_hand_backend=closeable(),
            left_hand_control_via_rig=False,
            right_state_backend=closeable(),
            left_state_backend=closeable(),
        ),
        "position_audit_fanout": None,
        "position_debug_session_recorder": None,
        "position_debug_close_threads": [],
        "position_episode_debug_runtime": None,
        "original_stdout": None,
        "silenced_stdout": None,
        "update_control_debug_runtime_metadata": lambda **kwargs: None,
    }
    support = {
        "_arm_control_mode_arg": lambda args: "position",
            "_attach_actuator_command_trace_sink": lambda *args: None,
            "_attach_ik_shadow_trace_sink": lambda *args: None,
            "_attach_position_audit_sink": lambda *args: None,
        "_attach_singularity_transition_trace_sink": lambda *args: None,
        "_close_camera_sources": lambda sources: None,
        "_log": lambda *args: None,
        "_position_force_guard_collision_summary": lambda fault: {},
        "_terminal_force_collision_summary": lambda *args, **kwargs: {},
        "run_best_effort_cleanup": (
            lambda _name, steps: labels.extend(label for label, _action in steps)
        ),
    }

    shutdown_runtime.shutdown_real_teleop_runtime(
        runtime_state,
        terminal_error=None,
        support=support,
    )

    assert labels == [
        "startup_progress",
        "episode_recorder",
        "clutch_input",
        "dataset_recorder",
        "camera_previewer",
        "camera_sources",
        "tactile_visualizer",
        "position_audit.detach",
        "position_audit.metadata_detach",
        "human_action_shadow",
        "teleop_session",
        "arm_state_read_executor",
        "right_hand_backend",
        "left_hand_backend",
        "right_state_backend",
        "left_state_backend",
    ]


def test_episode_recording_shutdown_commits_force_guard_reason_before_resources() -> None:
    close_calls: list[dict[str, object]] = []

    class Fault:
        def to_dict(self) -> dict[str, object]:
            return {"code": "joint_force_limit"}

    episode_runtime = SimpleNamespace(
        close=lambda **kwargs: close_calls.append(dict(kwargs)) or True
    )
    result = runtime._close_episode_recording_runtime_before_resources(
        {
            "args": SimpleNamespace(),
            "episode_recording_runtime": episode_runtime,
            "position_force_guard_monitor": SimpleNamespace(fault=Fault()),
        },
        terminal_error=RuntimeError("force guard"),
        support={
            "_arm_control_mode_arg": lambda _args: "position",
            "_position_force_guard_collision_summary": (
                lambda _fault: {"peak_force_n": 42.0}
            ),
            "_terminal_force_collision_summary": (
                lambda *_args, **_kwargs: pytest.fail(
                    "force-guard summary must remain authoritative"
                )
            ),
            "time": SimpleNamespace(time=lambda: 123.5),
        },
    )

    assert result is True
    assert close_calls == [
        {
            "now": 123.5,
            "end_reason": "position_force_guard",
            "summary": {
                "reason": "position_force_guard",
                "position_force_guard_fault": {
                    "code": "joint_force_limit"
                },
                "peak_force_n": 42.0,
            },
            "timeout_sec": 15.0,
        }
    ]


def test_real_component_builder_preserves_legacy_wrapper_and_import_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_args = SimpleNamespace()
    sentinel_components = object()
    sentinel_dmtac_resolver = object()
    captured: dict[str, object] = {}

    def fake_builder(
        args: argparse.Namespace,
        *,
        dmtac_backend_configs,
    ):
        captured["args"] = args
        captured["dmtac_backend_configs"] = dmtac_backend_configs
        return sentinel_components

    monkeypatch.setattr(
        legacy_entrypoint,
        "_dmtac_backend_configs",
        sentinel_dmtac_resolver,
    )
    monkeypatch.setattr(
        legacy_entrypoint,
        "build_real_teleop_components",
        fake_builder,
    )

    assert legacy_entrypoint._build_components(sentinel_args) is sentinel_components
    assert captured == {
        "args": sentinel_args,
        "dmtac_backend_configs": sentinel_dmtac_resolver,
    }
    assert list(
        inspect.signature(component_builder.build_real_teleop_components).parameters
    ) == ["args", "dmtac_backend_configs"]

    source = (
        ROOT / "teleop_runtime/real_teleop/component_builder.py"
    ).read_text(encoding="utf-8")
    assert "run_vive_rm65_real_teleop" in source
    assert "from teleop_runtime.debug" not in source
    assert "from ..debug" not in source


def test_component_builder_uses_active_sides_for_arm_ip_admission() -> None:
    args = SimpleNamespace(
        swap_arm_sides=False,
        left_arm_ip="192.168.101.21",
        right_arm_ip="192.168.1.20",
    )

    assert component_builder._active_arm_ip_by_side(args, ("left",)) == {
        "left": ("left_arm_ip", "192.168.101.21"),
    }


def test_extracted_application_runtime_preserves_entrypoint_support_contract() -> None:
    runtime.bind_legacy_support(vars(legacy_entrypoint))
    assert list(inspect.signature(runtime.run_real_teleop).parameters) == [
        "args",
        "progress_application",
        "input_batch_prioritizer",
    ]
    assert list(inspect.signature(legacy_entrypoint.main).parameters) == []

    source = (ROOT / "teleop_runtime/real_teleop/runtime.py").read_text(
        encoding="utf-8"
    )
    assert "progress_application: ProgressApplicationPort | None = None" in source
    assert "input_batch_prioritizer: RealTeleopInputBatchPrioritizer | None" in source
    assert "from teleop_runtime.debug" not in source
    assert "from ..debug" not in source
    assert set(runtime._REQUIRED_SUPPORT_NAMES).issubset(vars(legacy_entrypoint))


def test_extracted_control_loop_preserves_quit_frame_order_without_hardware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime.bind_legacy_support(vars(legacy_entrypoint))
    events: list[str] = []

    class FakeWatchdog:
        def start(self) -> None:
            events.append("watchdog.start")

        def beat(self, stage: str, step_index: int) -> None:
            events.append(f"watchdog.beat:{stage}:{step_index}")

    class FakeClutch:
        should_quit = True

        def poll(self) -> None:
            events.append("clutch.poll")

        def consume_plus_curation_request(self) -> None:
            return None

        def consume_ik_debug_log_toggle_request(self) -> bool:
            return False

        def consume_device_retry_request(self) -> bool:
            return False

        def consume_camera_visualization_toggle_request(self) -> bool:
            return False

        def consume_tactile_solving_enable_request(self) -> bool:
            return False

        def consume_tactile_visualization_toggle_request(self) -> bool:
            return False

    noop = lambda *args, **kwargs: None
    values = {
        item.name: noop
        for item in fields(control_loop.RealTeleopControlLoopContext)
    }
    values.update(
        {
            "action_debug_log_recorder": SimpleNamespace(active=False),
            "args": SimpleNamespace(),
            "camera_sources": {},
            "clutch_input": FakeClutch(),
            "components": SimpleNamespace(),
            "control_ui": None,
            "debug_timing_log_recorder": SimpleNamespace(),
            "debug_timing_modules": SimpleNamespace(
                needs_loop_phase_timer=False,
            ),
            "debug_timing_state": SimpleNamespace(),
            "episode_recorder": SimpleNamespace(),
            "ik_debug_log_recorder": SimpleNamespace(active=False),
            "ik_debug_profile_base_enabled": False,
            "main_loop_watchdog": FakeWatchdog(),
            "outer_control_hz": 15.0,
            "position_audit_fanout": None,
            "position_force_guard_monitor": None,
            "preview_camera_sources": {},
            "recording_active": lambda: False,
            "dataset_recorder": lambda: None,
            "visualization_state": lambda: (None, False, None),
            "episode_start_countdown_deadline": lambda: None,
            "episode_last_state": lambda: None,
            "step_index": lambda: 0,
            "sync_foot_pad_runtime_status": lambda: events.append("foot_pad.sync"),
            "poll_dataset_finalize": lambda now: events.append("dataset.poll_finalize"),
            "cancel_recording_countdown": (
                lambda now, *, reason: events.append(f"countdown.cancel:{reason}")
            ),
        }
    )
    monkeypatch.setattr(
        control_loop,
        "_emergency_stop_real_arms",
        lambda components, *, reason, now: events.append(f"estop:{reason}"),
    )

    control_loop.run_real_teleop_control_loop(
        control_loop.RealTeleopControlLoopContext(**values)
    )

    assert events == [
        "watchdog.start",
        "watchdog.beat:loop.start:0",
        "clutch.poll",
        "watchdog.beat:loop.control_input_poll:0",
        "foot_pad.sync",
        "dataset.poll_finalize",
        "estop:quit_requested",
        "countdown.cancel:quit_requested",
    ]


def test_device_retry_runtime_keeps_optional_foot_pad_status_dynamic() -> None:
    errors = {"foot_pad": "offline", "camera.overhead": "camera failed"}
    foot_pad_slot = [None]
    published: list[dict[str, object]] = []
    args = SimpleNamespace(
        foot_pad_requested=True,
        foot_pad_startup_error="offline",
        ignore_camera=(),
    )
    clutch = SimpleNamespace(
        _current_active_controls=lambda: (),
        _foot_pad_test_ui_state=lambda: {
            "foot_pad_available": bool(
                foot_pad_slot[0] is not None
                and getattr(foot_pad_slot[0], "is_online", False)
            ),
            "foot_pad_status": "online"
            if foot_pad_slot[0] is not None
            and getattr(foot_pad_slot[0], "is_online", False)
            else "unavailable",
        },
    )
    ui = SimpleNamespace(
        publish=lambda **kwargs: published.append(kwargs),
    )
    component_snapshot = SimpleNamespace(
        left_hand_backend=None,
        right_hand_backend=None,
    )
    retry_runtime = device_retry_runtime.build_device_retry_runtime(
        args=args,
        components=component_snapshot,
        control_ui=ui,
        clutch_input=clutch,
        camera_sources={"overhead": object()},
        vive_source=None,
        startup_device_errors=errors,
        position_force_guard_requested=False,
        dataset_recorder_provider=lambda: None,
        set_dataset_recorder=lambda value: None,
        foot_pad_input_provider=lambda: foot_pad_slot[0],
        set_foot_pad_input=lambda value: foot_pad_slot.__setitem__(0, value),
        dataset_recorder_type=object,
        dmtac_gripper_backend_type=object,
        foot_pad_input_type=SimpleNamespace,
        foot_pad_control_mode="foot_pad",
        has_dmtac_gripper_backend=lambda components: False,
        promote_angle_trigger_gripper_input_to_q6=lambda *args, **kwargs: (
            False,
            "",
        ),
        log=lambda *args, **kwargs: None,
    )

    foot_pad_status = retry_runtime.hardware_status_item("foot_pad")
    assert foot_pad_status["status_level"] == "warn"
    assert foot_pad_status["requirement"] == "optional_input"
    assert retry_runtime.collection_startup_blockers() == {
        "camera.overhead": "camera failed"
    }

    foot_pad_slot[0] = SimpleNamespace(is_online=True)
    retry_runtime.sync_foot_pad_runtime_status()
    assert "foot_pad" not in errors
    assert args.foot_pad_startup_error == ""
    assert published
    assert any(
        update.get("foot_pad_available") is True
        and update.get("foot_pad_status") == "online"
        for update in published
    )


def test_device_retry_runtime_immediately_refreshes_camera_projection() -> None:
    errors = {"camera.front": "initial startup failure"}
    camera_calls: list[str] = []
    camera_status_refreshes: list[str] = []

    class Camera:
        def close(self) -> None:
            camera_calls.append("close")

        def start(self) -> None:
            camera_calls.append("start")

    published: list[dict[str, object]] = []
    ui = SimpleNamespace(
        publish=lambda **kwargs: published.append(kwargs),
        append_message=lambda *_args: None,
    )
    clutch = SimpleNamespace(
        _current_active_controls=lambda: ("device_retry",),
        device_retry_admission=lambda: (True, ""),
        _episode_busy=lambda: False,
        _teleop_started=False,
        hold_active=lambda: True,
    )
    components = SimpleNamespace(
        gripper_input_source="",
        left_state_backend=None,
        right_state_backend=None,
        left_hand_backend=None,
        right_hand_backend=None,
    )
    runtime = device_retry_runtime.build_device_retry_runtime(
        args=SimpleNamespace(foot_pad_requested=False, ignore_camera=()),
        components=components,
        control_ui=ui,
        clutch_input=clutch,
        camera_sources={"front": Camera()},
        vive_source=None,
        startup_device_errors=errors,
        position_force_guard_requested=False,
        dataset_recorder_provider=lambda: None,
        set_dataset_recorder=lambda value: None,
        foot_pad_input_provider=lambda: None,
        set_foot_pad_input=lambda value: None,
        dataset_recorder_type=object,
        dmtac_gripper_backend_type=object,
        foot_pad_input_type=object,
        foot_pad_control_mode="foot-pad",
        has_dmtac_gripper_backend=lambda _components: False,
        promote_angle_trigger_gripper_input_to_q6=lambda *args, **kwargs: (
            False,
            "",
        ),
        log=lambda *args, **kwargs: None,
        publish_camera_status=lambda: camera_status_refreshes.append("refresh"),
    )

    runtime.retry_failed_devices()

    assert camera_calls == ["close", "start"]
    assert camera_status_refreshes == ["refresh"]
    assert "camera.front" not in errors
    assert published[-1]["hardware_status"]["camera.front"]["status_level"] == "ok"


def test_generic_device_retry_never_starts_or_reuses_dataset_writer() -> None:
    errors = {"dataset.writer": "initial writer startup failure"}
    published: list[dict[str, object]] = []
    messages: list[tuple[str, str]] = []

    class FailedRecorder:
        def __init__(self) -> None:
            self.start_calls = 0
            self.closed = False

        def start(self) -> None:
            self.start_calls += 1
            raise AssertionError("cannot start a process twice")

        def close(self) -> None:
            self.closed = True

    failed_recorder = FailedRecorder()
    current_recorder: dict[str, object | None] = {"value": failed_recorder}

    class Recorder:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError(
                "generic device retry must not construct a dataset writer"
            )

    clutch = SimpleNamespace(
        _current_active_controls=lambda: ("device_retry",),
        device_retry_admission=lambda: (True, ""),
        _episode_busy=lambda: False,
        _teleop_started=False,
        hold_active=lambda: True,
    )
    components = SimpleNamespace(
        gripper_input_source="",
        left_state_backend=None,
        right_state_backend=None,
        left_hand_backend=None,
        right_hand_backend=None,
    )
    runtime = device_retry_runtime.build_device_retry_runtime(
        args=SimpleNamespace(
            foot_pad_requested=False,
            ignore_camera=(),
            q6_config_required=False,
        ),
        components=components,
        control_ui=SimpleNamespace(
            publish=lambda **kwargs: published.append(kwargs),
            append_message=lambda tag, message: messages.append((tag, message)),
        ),
        clutch_input=clutch,
        camera_sources={},
        vive_source=None,
        startup_device_errors=errors,
        position_force_guard_requested=False,
        dataset_recorder_provider=lambda: current_recorder["value"],
        set_dataset_recorder=lambda value: current_recorder.__setitem__(
            "value", value
        ),
        foot_pad_input_provider=lambda: None,
        set_foot_pad_input=lambda _value: None,
        dataset_recorder_type=Recorder,
        dmtac_gripper_backend_type=object,
        foot_pad_input_type=object,
        foot_pad_control_mode="foot-pad",
        has_dmtac_gripper_backend=lambda _components: False,
        promote_angle_trigger_gripper_input_to_q6=lambda *_args, **_kwargs: (
            False,
            "",
        ),
        log=lambda *_args, **_kwargs: None,
    )

    runtime.retry_failed_devices()
    assert errors == {"dataset.writer": "initial writer startup failure"}
    assert current_recorder["value"] is failed_recorder
    assert failed_recorder.closed is False
    assert failed_recorder.start_calls == 0
    assert messages[-1][0] == "Dataset"
    assert "再次点击“开始数采”" in messages[-1][1]
    writer_status = published[-1]["hardware_status"]["dataset.writer"]
    assert writer_status["status_level"] == "error"
    assert writer_status["status_label"] == "writer 准备失败"
    assert "全新 writer" in writer_status["error"]


def test_visualization_runtime_mirrors_preview_state_without_opening_devices() -> None:
    outer_states: list[tuple[object | None, bool, object | None]] = []
    visualization_updates: list[dict[str, object]] = []

    class FakeVisualizer:
        def close(self) -> None:
            visualization_updates.append({"closed": True})

    visualizer = FakeVisualizer()
    clutch = SimpleNamespace(
        set_visualization_state=lambda **kwargs: visualization_updates.append(
            kwargs
        ),
    )
    preview = visualization_runtime.build_visualization_runtime(
        args=SimpleNamespace(tactile_visualization_available=False),
        components=SimpleNamespace(active_sides=()),
        clutch_input=clutch,
        control_ui=None,
        preview_camera_sources={},
        camera_preview_rate=SimpleNamespace(reset=lambda: None),
        tactile_preview_rate=SimpleNamespace(reset=lambda: None),
        initial_camera_previewer=None,
        initial_camera_preview_active=False,
        initial_tactile_visualizer=visualizer,
        set_outer_state=lambda camera, active, tactile: outer_states.append(
            (camera, active, tactile)
        ),
        retry_failed_devices=lambda: None,
        start_camera_sources=lambda sources: None,
        camera_previewer_type=object,
        camera_preview_config_type=object,
        build_tactile_visualizer=lambda *args, **kwargs: None,
        enable_gui_tactile_solving=lambda *args, **kwargs: 0,
        tactile_solving_active=lambda *args, **kwargs: False,
        tactile_visualization_available=lambda *args, **kwargs: False,
        dmtac_visualizer_types=(type("D1", (), {}), type("D2", (), {})),
        dmtac_visualization_snapshot=lambda *args, **kwargs: ({}, {}),
        tactile_snapshot=lambda *args, **kwargs: {},
        gripper_backend_for_side=lambda *args, **kwargs: None,
        hand_backend_for_side=lambda *args, **kwargs: None,
        log=lambda *args, **kwargs: None,
    )

    assert preview.current_state() == (None, False, visualizer)
    preview.open_camera_preview(1.0)
    assert preview.current_state() == (None, False, visualizer)
    preview.close_tactile_preview("closed")
    assert preview.current_state() == (None, False, None)
    assert outer_states[-1] == (None, False, None)
    assert {"closed": True} in visualization_updates


def test_episode_recording_runtime_refreshes_external_recorder_slot() -> None:
    dataset_slot: list[object | None] = [None]
    episode_state_slot: list[str | None] = [None]
    step_slot = [0]
    position_sidecar = SimpleNamespace(
        start=lambda *args, **kwargs: False,
        stop=lambda *args, **kwargs: None,
        start_close=lambda action: None,
        current_recorder=lambda: None,
    )
    episode_recorder_snapshot = SimpleNamespace(
        is_recording=False,
        frame_count=2,
        episode_dir=None,
    )
    support = {
        name: getattr(legacy_entrypoint, name)
        for name in (
            "CaptureModalityContract",
            "DEFAULT_EPISODE_START_COUNTDOWN_SEC",
            "RealTeleopDatasetRecorder",
            "_arm_control_mode_arg",
            "_build_episode_metadata",
            "_dataset_recording_debug_text",
            "_episode_root_dir",
            "_episode_start_countdown_ready_to_record",
            "_episode_start_countdown_remaining",
            "_log",
            "_start_episode_recording",
            "_stop_episode_recording",
            "bind_capture_modality_contract",
        )
    }
    recording = episode_recording_runtime.build_episode_recording_runtime(
        args=SimpleNamespace(collection_warning_hz=15.0),
        components=SimpleNamespace(),
        clutch_input=SimpleNamespace(),
        episode_recorder=episode_recorder_snapshot,
        camera_sources={},
        control_ui=None,
        startup_device_errors={},
        fix_hold_coordinator=SimpleNamespace(),
        position_episode_debug_runtime=position_sidecar,
        update_control_debug_runtime_metadata=lambda **kwargs: None,
        publish_runtime_hardware_status=lambda: None,
        collection_startup_blockers=lambda: {},
        recording_warmup_status=lambda *args, **kwargs: (True, "ready"),
        reset_gripper_warmup_probe=lambda reason: None,
        prepare_dataset_transition=lambda reason: None,
        dataset_recorder_provider=lambda: dataset_slot[0],
        set_dataset_recorder=lambda value: dataset_slot.__setitem__(0, value),
        camera_previewer_provider=lambda: None,
        episode_last_state_provider=lambda: episode_state_slot[0],
        set_episode_last_state=lambda value: episode_state_slot.__setitem__(0, value),
        step_index_provider=lambda: step_slot[0],
        support=support,
    )

    assert recording.recording_active() is False
    assert recording.recording_frame_count() == 2
    dataset_slot[0] = SimpleNamespace(
        is_recording=True,
        is_paused=False,
        frame_count=7,
        dataset_root=Path("/tmp/dexteleop-recording-slot"),
        collection_frequency_status=lambda: {"sample_hz": 15.0},
    )
    assert recording.recording_active() is True
    assert recording.recording_frame_count() == 7
    assert recording.recording_path() == "/tmp/dexteleop-recording-slot"
    assert recording.recording_frequency_status() == {"sample_hz": 15.0}
    assert recording.cancel_recording_countdown(1.0, reason="unit_test") is False


def test_camera_episode_and_special_arm_modules_preserve_legacy_seams() -> None:
    for name in (
        "_build_camera_sources",
        "_close_camera_sources",
        "_select_preview_camera_sources",
        "_start_camera_sources",
        "_start_camera_sources_best_effort",
    ):
        assert getattr(legacy_entrypoint, name) is getattr(camera_runtime, name)
    assert (
        legacy_entrypoint.CAMERA_START_STAGGER_SEC
        is camera_runtime.CAMERA_START_STAGGER_SEC
    )

    for name in (
        "_build_episode_metadata",
        "_debug_episode_recording_enabled",
        "_episode_root_dir",
        "_fix_arm_pose_snapshot_recorder",
        "_start_episode_recording",
        "_stop_episode_recording",
    ):
        assert getattr(legacy_entrypoint, name) is getattr(episode_runtime, name)

    for name in (
        "_cancel_special_arm_modes_for_hold",
        "_disengage_position_backends_for_manual_interaction",
        "_fresh_gui_init_qpos_by_side",
        "_init_arm_qpos_speed_scale_for_side",
        "_normalize_init_arm_qpos_speed_scale",
        "_release_gui_qpos_init_transit",
    ):
        assert getattr(legacy_entrypoint, name) is getattr(special_arm_runtime, name)
    assert list(inspect.signature(legacy_entrypoint._start_gui_qpos_init).parameters) == [
        "args",
        "components",
        "now",
    ]

    for path in (
        ROOT / "teleop_runtime/real_teleop/camera_runtime.py",
        ROOT / "teleop_runtime/real_teleop/episode_runtime.py",
        ROOT / "teleop_runtime/real_teleop/special_arm_runtime.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "from teleop_runtime.debug" not in source
        assert "from ..debug" not in source


def test_startup_runtime_preserves_entrypoint_functions_and_patch_seams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "_run_camera_startup_self_check",
        "_update_init_arm_arrival_from_hardware",
        "_release_runtime_state_readers",
        "_log_dual_arm_readiness",
        "_startup_gui_qpos_seed_command",
        "_finish_startup_gui_qpos_terminal",
        "_advance_startup_gui_qpos_maintenance",
        "_wait_for_real_teleop_start_request",
        "_runtime_home_failure_message",
        "_restore_init_arm_after_foot_pad_fallback",
        "_arm_after_start_calibration",
    ):
        assert getattr(legacy_entrypoint, name).__module__ == startup_runtime.__name__

    calls: list[str] = []
    monkeypatch.setattr(
        legacy_entrypoint,
        "_advance_startup_gui_qpos_maintenance",
        lambda **kwargs: calls.append("advance") or False,
    )
    monkeypatch.setattr(
        legacy_entrypoint,
        "_handle_ik_debug_log_toggle_request",
        lambda **kwargs: calls.append("debug"),
    )
    monkeypatch.setattr(
        legacy_entrypoint,
        "_impedance_force_control_not_ready_sides",
        lambda args, components: [],
    )
    startup_runtime.bind_legacy_support(vars(legacy_entrypoint))
    clutch = SimpleNamespace(
        enabled=True,
        args=SimpleNamespace(),
        poll=lambda: calls.append("poll"),
        should_quit=True,
        set_startup_qpos_maintenance_enabled=lambda enabled: calls.append(
            f"maintenance:{enabled}"
        ),
        consume_init_qpos_emergency_stop_request=lambda: False,
    )
    assert legacy_entrypoint._wait_for_real_teleop_start_request(
        SimpleNamespace(),
        clutch,
        arm_mode="position",
        args=SimpleNamespace(),
        control_ui=None,
        ik_debug_log_recorder=SimpleNamespace(),
        ik_debug_profile_base_enabled=False,
    ) is None
    assert calls == [
        "maintenance:True",
        "poll",
        "advance",
        "debug",
        "maintenance:False",
    ]

    source = (
        ROOT / "teleop_runtime/real_teleop/startup_runtime.py"
    ).read_text(encoding="utf-8")
    assert "from teleop_runtime.debug" not in source
    assert "from ..debug" not in source


def test_tcp_pose_envelope_home_rebind_precedes_real_hardware_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        startup_runtime,
        "_reset_runtime_motion_state_from_real_qpos",
        lambda *args, **kwargs: events.append("motion_reset") or True,
    )
    monkeypatch.setattr(
        startup_runtime,
        "_enable_impedance_force_control_after_runtime_home",
        lambda *args, **kwargs: events.append("force_enable"),
    )
    monkeypatch.setattr(startup_runtime, "_log", lambda *args, **kwargs: None)

    class SafetyFilter:
        execution_epoch = 0
        profile = SimpleNamespace(anchor_required_sides=("left", "right"))

        def rebind_from_calibration_home(self, **kwargs):
            events.append(f"rebind:{kwargs['execution_epoch']}")
            return SimpleNamespace(semantic_content_hash="anchor-hash")

    session = SimpleNamespace(
        mark_home_recalibrated=lambda reason: events.append("mark_home"),
        set_calibration_ready=lambda: events.append("calibration_ready"),
        mark_debug_passed=lambda: events.append("debug_passed"),
        arm_hardware=lambda: events.append("arm_hardware"),
        hold_hardware=lambda reason: events.append("hold_hardware"),
    )
    components = SimpleNamespace(
        active_sides=("left", "right"),
        dual_arm_retargeter=SimpleNamespace(
            pre_ik_target_safety_filter=SafetyFilter()
        ),
        position_reference_publisher=SimpleNamespace(
            reset_epoch=lambda: events.append("epoch_reset") or 4
        ),
        position_admittance_endpoint_guard=None,
        session=session,
    )
    clutch = SimpleNamespace(
        publish_calibration_waiting=lambda message: events.append("waiting"),
        publish_teleop_running=lambda **kwargs: events.append("teleop_running"),
    )

    startup_runtime._arm_after_start_calibration(
        SimpleNamespace(),
        components,
        clutch,
        home_captured=True,
        runtime_home_qpos_snapshot={
            "left": np.zeros(6),
            "right": np.zeros(6),
        },
    )

    assert events.index("rebind:4") < events.index("arm_hardware")
    assert "hold_hardware" not in events
    assert events[-1] == "teleop_running"


def test_startup_home_continuity_gate_arms_before_real_hardware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    home_qpos = {"left": np.zeros(6), "right": np.zeros(6)}
    monkeypatch.setattr(
        startup_runtime,
        "_reset_runtime_motion_state_from_real_qpos",
        lambda *args, **kwargs: {"left": True, "right": True},
    )
    monkeypatch.setattr(
        startup_runtime,
        "_enable_impedance_force_control_after_runtime_home",
        lambda *args, **kwargs: events.append("force_enable"),
    )
    monkeypatch.setattr(startup_runtime, "_log", lambda *args, **kwargs: None)

    class ContinuityGate:
        def arm_from_home(self, qpos_by_side):
            assert qpos_by_side is home_qpos
            events.append("continuity_armed")
            return {"status": "waiting_for_candidate"}

    session = SimpleNamespace(
        mark_home_recalibrated=lambda reason: events.append("mark_home"),
        set_calibration_ready=lambda: events.append("calibration_ready"),
        mark_debug_passed=lambda: events.append("debug_passed"),
        arm_hardware=lambda: events.append("arm_hardware"),
        hold_hardware=lambda reason: events.append("hold_hardware"),
    )
    components = SimpleNamespace(
        active_sides=("left", "right"),
        dual_arm_retargeter=None,
        position_reference_publisher=SimpleNamespace(reset_epoch=lambda: 8),
        position_admittance_endpoint_guard=None,
        startup_home_continuity_gate=ContinuityGate(),
        session=session,
    )
    clutch = SimpleNamespace(
        publish_calibration_waiting=lambda message: events.append("waiting"),
        publish_teleop_running=lambda **kwargs: events.append("teleop_running"),
    )

    startup_runtime._arm_after_start_calibration(
        SimpleNamespace(),
        components,
        clutch,
        home_captured=True,
        runtime_home_qpos_snapshot=home_qpos,
    )

    assert events.index("continuity_armed") < events.index("arm_hardware")
    assert "hold_hardware" not in events


def test_relative_tcp_pose_envelope_refuses_arm_without_runtime_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(startup_runtime, "_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        startup_runtime,
        "_enable_impedance_force_control_after_runtime_home",
        lambda *args, **kwargs: events.append("force_enable"),
    )
    safety_filter = SimpleNamespace(
        profile=SimpleNamespace(anchor_required_sides=("left",)),
        execution_epoch=0,
        apply_to_target=True,
    )
    components = SimpleNamespace(
        active_sides=("left",),
        dual_arm_retargeter=SimpleNamespace(
            pre_ik_target_safety_filter=safety_filter
        ),
        position_reference_publisher=SimpleNamespace(reset_epoch=lambda: 1),
        session=SimpleNamespace(
            hold_hardware=lambda reason: events.append("hold_hardware"),
            set_calibration_ready=lambda: events.append("calibration_ready"),
            mark_debug_passed=lambda: events.append("debug_passed"),
            arm_hardware=lambda: events.append("arm_hardware"),
        ),
    )
    clutch = SimpleNamespace(
        publish_calibration_waiting=lambda message: events.append("waiting"),
        publish_teleop_running=lambda **kwargs: events.append("teleop_running"),
    )

    startup_runtime._arm_after_start_calibration(
        SimpleNamespace(),
        components,
        clutch,
        home_captured=False,
    )

    assert events == ["hold_hardware", "waiting"]


def test_shadow_relative_tcp_pose_envelope_never_blocks_arm_without_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(startup_runtime, "_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        startup_runtime,
        "_enable_impedance_force_control_after_runtime_home",
        lambda *args, **kwargs: events.append("force_enable"),
    )
    safety_filter = SimpleNamespace(
        profile=SimpleNamespace(anchor_required_sides=("left",)),
        execution_epoch=0,
        apply_to_target=False,
    )
    components = SimpleNamespace(
        active_sides=("left",),
        dual_arm_retargeter=SimpleNamespace(
            pre_ik_target_safety_filter=safety_filter
        ),
        position_reference_publisher=SimpleNamespace(reset_epoch=lambda: 1),
        session=SimpleNamespace(
            hold_hardware=lambda reason: events.append("hold_hardware"),
            set_calibration_ready=lambda: events.append("calibration_ready"),
            mark_debug_passed=lambda: events.append("debug_passed"),
            arm_hardware=lambda: events.append("arm_hardware"),
        ),
    )
    clutch = SimpleNamespace(
        publish_calibration_waiting=lambda message: events.append("waiting"),
        publish_teleop_running=lambda **kwargs: events.append("teleop_running"),
    )

    startup_runtime._arm_after_start_calibration(
        SimpleNamespace(),
        components,
        clutch,
        home_captured=False,
    )

    assert "hold_hardware" not in events
    assert "waiting" not in events
    assert events == [
        "calibration_ready",
        "debug_passed",
        "force_enable",
        "arm_hardware",
        "teleop_running",
    ]


def test_active_tcp_pose_envelope_fault_has_explicit_pre_dispatch_hold_path() -> None:
    source = (
        ROOT / "teleop_runtime/real_teleop/control_loop.py"
    ).read_text(encoding="utf-8")
    run_step_marker = "step_result = components.session.run_step("
    active_fault_marker = "except TCPPoseEnvelopeRuntimeError as exc:"
    generic_fault_marker = (
        "except Exception as exc:  # noqa: BLE001 - only safety races are consumed here."
    )

    run_step_index = source.index(run_step_marker)
    active_fault_index = source.index(active_fault_marker, run_step_index)
    generic_fault_index = source.index(generic_fault_marker, active_fault_index)
    hold_index = source.index(
        "require_home_recalibration=True",
        active_fault_index,
    )
    request_index = source.index(
        'reason="tcp_pose_envelope_active_pre_ik_fault"',
        active_fault_index,
    )

    assert run_step_index < active_fault_index < generic_fault_index
    assert active_fault_index < hold_index < generic_fault_index
    assert active_fault_index < request_index < generic_fault_index


def test_startup_home_wait_quiesces_idle_maintenance_during_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        legacy_entrypoint,
        "_advance_startup_gui_qpos_maintenance",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        legacy_entrypoint,
        "_handle_ik_debug_log_toggle_request",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        legacy_entrypoint,
        "_impedance_force_control_not_ready_sides",
        lambda args, components: [],
    )
    startup_runtime.bind_legacy_support(vars(legacy_entrypoint))
    clutch = SimpleNamespace(
        enabled=True,
        poll=lambda: None,
        should_quit=False,
        set_startup_qpos_maintenance_enabled=lambda enabled: None,
        consume_init_qpos_emergency_stop_request=lambda: False,
        consume_calibration_request=lambda: True,
        before_calibration_start_delay=lambda *args, **kwargs: (
            calls.append("delay") or True
        ),
    )

    result = legacy_entrypoint._wait_for_real_teleop_start_request(
        SimpleNamespace(init_arm_controller=None),
        clutch,
        arm_mode="position",
        args=SimpleNamespace(),
        control_ui=None,
        ik_debug_log_recorder=SimpleNamespace(),
        ik_debug_profile_base_enabled=False,
        begin_home_transition_handler=lambda: calls.append("begin"),
        require_home_transition_ready_handler=lambda: calls.append("require"),
    )

    assert result == "home"
    assert calls == ["begin", "delay", "require"]


def test_arm_force_gripper_and_debug_frame_modules_preserve_legacy_seams() -> None:
    for name in (
        "_acceleration_limited_position_execution_active",
        "_active_command_backend",
        "_active_impedance_force_backends",
        "_apply_active_arm_command_to_preview_real",
        "_command_backend_for_side",
        "_estimate_impedance_force_control_warmup_sec",
        "_impedance_force_control_warmup_backends",
        "_runtime_arm_backend_state",
        "_sync_dual_arm_retargeter_for_real_control",
    ):
        assert getattr(legacy_entrypoint, name) is getattr(arm_runtime, name)

    for name in (
        "_reset_runtime_motion_state_from_real_qpos",
        "_run_initial_alignment_start_check",
        "_sync_preview_from_real_state_real",
        "_sync_real_execution_step_limits",
    ):
        assert getattr(legacy_entrypoint, name) is getattr(arm_state_sync, name)

    for name in (
        "_emergency_stop_real_arms",
        "_enable_impedance_force_control_after_runtime_home",
        "_exception_chain_messages",
        "_force_collision_recording_summary",
        "_force_guard_recording_stop_context",
        "_handle_init_qpos_emergency_stop",
        "_impedance_force_control_issue",
        "_position_force_guard_collision_summary",
        "_terminal_force_collision_summary",
        "_warm_start_impedance_force_control_before_home",
    ):
        assert getattr(legacy_entrypoint, name) is getattr(
            impedance_force_runtime,
            name,
        )

    for name in (
        "_dispatch_q6_gripper_control",
        "_probe_gripper_backend_motion",
        "_run_gripper_warmup_probe_for_side",
    ):
        assert getattr(legacy_entrypoint, name) is getattr(gripper_control, name)

    assert list(
        inspect.signature(legacy_entrypoint._promote_angle_trigger_gripper_input_to_q6).parameters
    ) == ["args", "components", "ready_timeout_sec"]
    assert callable(gripper_recovery.promote_angle_trigger_gripper_input_to_q6)

    for name in (
        "_command_for_side",
        "_build_ik_debug_log_frame",
        "_camera_status_log_payload",
        "_backend_timing_log_payload",
        "_build_debug_timing_log_frame",
    ):
        assert getattr(legacy_entrypoint, name) is getattr(debug_log_frames, name)
    assert not hasattr(debug_log_frames, "bind_legacy_support")
    assert (
        legacy_entrypoint._build_pose_quality_display_payload
        is status_payloads._build_pose_quality_display_payload
    )

    for path in (
        ROOT / "teleop_runtime/real_teleop/arm_runtime.py",
        ROOT / "teleop_runtime/real_teleop/arm_state_sync.py",
        ROOT / "teleop_runtime/real_teleop/impedance_force_runtime.py",
        ROOT / "teleop_runtime/real_teleop/gripper_control.py",
        ROOT / "teleop_runtime/real_teleop/gripper_recovery.py",
        ROOT / "teleop_runtime/real_teleop/gripper_warmup_runtime.py",
        ROOT / "teleop_runtime/real_teleop/recording_admission.py",
        ROOT / "teleop_runtime/real_teleop/position_episode_debug.py",
        ROOT / "teleop_runtime/real_teleop/device_retry_runtime.py",
        ROOT / "teleop_runtime/real_teleop/control_loop.py",
        ROOT / "teleop_runtime/real_teleop/visualization_runtime.py",
        ROOT / "teleop_runtime/real_teleop/episode_recording_runtime.py",
        ROOT / "teleop_runtime/real_teleop/debug_log_frames.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "from teleop_runtime.debug" not in source
        assert "from ..debug" not in source


def test_gripper_warmup_runtime_keeps_probe_state_inside_one_closure() -> None:
    probe_called = threading.Event()
    logs: list[tuple[str, str, float | None]] = []
    backend = SimpleNamespace(config=SimpleNamespace(connect_hardware=True))
    args = SimpleNamespace(
        gripper_warmup_probe=True,
        gripper_warmup_probe_tolerance=0.18,
        gripper_warmup_probe_step_timeout_sec=0.01,
        gripper_warmup_probe_settle_sec=0.0,
        gripper_warmup_probe_poll_sec=0.01,
    )
    components_snapshot = SimpleNamespace(active_sides=("left",))

    def run_probe(side: str, candidate: object, **kwargs: object) -> dict[str, object]:
        assert side == "left"
        assert candidate is backend
        assert kwargs["targets"] == (0.0, 0.5, 0.0)
        probe_called.set()
        return {"ok": True, "reconnected": False}

    probe = gripper_warmup_runtime.build_gripper_warmup_probe_runtime(
        args,
        components_snapshot,
        gripper_backend_for_side=lambda _components, side: (
            backend if side == "left" else None
        ),
        run_probe_for_side=run_probe,
        safe_float_value=lambda value: (
            None if value is None else float(value)
        ),
        log=lambda stage, message, *, now=None: logs.append((stage, message, now)),
    )

    assert probe.enabled()
    assert probe.start(1.0, reason="unit_test") == (
        False,
        "等待 gripper warmup probe 完成。",
    )
    assert probe_called.wait(1.0)
    for _ in range(100):
        ready, message = probe.status(1.1, reason="unit_test")
        if ready:
            break
        threading.Event().wait(0.005)
    assert ready
    assert message == "gripper warmup probe 通过：open->half->open state matched。"
    assert any(stage == "GripperWarmup" for stage, _message, _now in logs)

    probe.reset("next_episode")
    assert probe.status(2.0, reason="next_episode")[0] is False


def test_gripper_warmup_runtime_never_moves_profile_fixed_side() -> None:
    probe_called = threading.Event()
    called_sides: list[str] = []
    backends = {
        side: SimpleNamespace(config=SimpleNamespace(connect_hardware=True))
        for side in ("left", "right")
    }
    components_snapshot = SimpleNamespace(
        active_sides=("left", "right"),
        fix_arm_controller=SimpleNamespace(fixed_sides=("left",)),
    )
    args = SimpleNamespace(
        gripper_warmup_probe=True,
        gripper_warmup_probe_tolerance=0.18,
        gripper_warmup_probe_step_timeout_sec=0.01,
        gripper_warmup_probe_settle_sec=0.0,
        gripper_warmup_probe_poll_sec=0.01,
    )

    def run_probe(side: str, _backend: object, **_kwargs: object) -> dict[str, object]:
        called_sides.append(side)
        probe_called.set()
        return {"ok": True, "reconnected": False}

    probe = gripper_warmup_runtime.build_gripper_warmup_probe_runtime(
        args,
        components_snapshot,
        gripper_backend_for_side=lambda _components, side: backends[side],
        run_probe_for_side=run_probe,
        safe_float_value=lambda value: None if value is None else float(value),
        log=lambda *_args, **_kwargs: None,
    )

    assert probe.enabled()
    assert probe.start(1.0, reason="unit_test")[0] is False
    assert probe_called.wait(1.0)
    for _ in range(100):
        ready, _message = probe.status(1.1, reason="unit_test")
        if ready:
            break
        threading.Event().wait(0.005)
    assert ready
    assert called_sides == ["right"]


def test_gripper_warmup_runtime_keeps_partial_entry_grippers_passive_by_default() -> None:
    called_sides: list[str] = []
    backend = SimpleNamespace(config=SimpleNamespace(connect_hardware=True))
    args = SimpleNamespace(
        gripper_warmup_probe=True,
        _gripper_warmup_probe_explicit=False,
        collection_progress_profile=SimpleNamespace(
            initial_step_id="right_acquire_brush",
        ),
        collection_progress_entry_step="right_write",
        gripper_warmup_probe_tolerance=0.18,
        gripper_warmup_probe_step_timeout_sec=0.01,
        gripper_warmup_probe_settle_sec=0.0,
        gripper_warmup_probe_poll_sec=0.01,
    )
    components_snapshot = SimpleNamespace(active_sides=("left", "right"))

    def run_probe(side: str, _backend: object, **_kwargs: object) -> dict[str, object]:
        called_sides.append(side)
        return {"ok": True, "reconnected": False}

    probe = gripper_warmup_runtime.build_gripper_warmup_probe_runtime(
        args,
        components_snapshot,
        gripper_backend_for_side=lambda _components, _side: backend,
        run_probe_for_side=run_probe,
        safe_float_value=lambda value: None if value is None else float(value),
        log=lambda *_args, **_kwargs: None,
    )

    assert not probe.enabled()
    assert probe.start(1.0, reason="partial_entry") == (
        True,
        "gripper warmup probe 未启用。",
    )
    assert called_sides == []

    args.collection_progress_entry_step = "right_acquire_brush"
    assert probe.enabled()

    args.collection_progress_entry_step = "right_write"
    args._gripper_warmup_probe_explicit = True
    assert probe.enabled()


def test_recording_admission_keeps_fail_closed_check_order() -> None:
    args = SimpleNamespace(
        no_camera=True,
        ignore_camera=(),
        ignore_tactile=(),
    )
    components_snapshot = SimpleNamespace(active_sides=())
    clutch = SimpleNamespace(hold_active=lambda: False)
    common = {
        "args": args,
        "components": components_snapshot,
        "clutch_input": clutch,
        "dataset_recorder": None,
        "camera_sources": {},
        "preview_camera_sources": {},
        "camera_previewer": None,
        "recording_auto_stop_reason_fn": recording_admission.recording_auto_stop_reason,
        "gripper_warmup_probe_status": lambda now, *, reason: (True, "ready"),
        "camera_status_blocking_message": lambda status: None,
    }

    assert recording_admission.recording_warmup_status(
        10.0,
        step_result=None,
        **common,
    ) == (False, "等待首个控制帧。")

    fault_step = SimpleNamespace(
        state=legacy_entrypoint.TeleopRunState.FAULT,
        observation=object(),
        command=object(),
        channels={},
    )
    assert recording_admission.recording_warmup_status(
        10.0,
        step_result=fault_step,
        **common,
    ) == (False, "等待恢复，当前状态会自动停止数采：runtime_fault。")

    running_command = SimpleNamespace(
        commands={"left_arm": object()},
    )
    running_step = SimpleNamespace(
        state=legacy_entrypoint.TeleopRunState.RUNNING,
        observation=object(),
        command=running_command,
        command_dispatched=True,
        channels={},
    )
    assert recording_admission.recording_warmup_status(
        10.0,
        step_result=running_step,
        **common,
    ) == (True, "warmup 输入稳定。")

    running_step.command_dispatched = False
    assert recording_admission.recording_warmup_status(
        10.0,
        step_result=running_step,
        **common,
    ) == (False, "等待机械臂命令通过最终 admission 并实际派发。")

    running_step.command_dispatched = True
    running_step.command = SimpleNamespace(commands={"left_hand": object()})
    assert recording_admission.recording_warmup_status(
        10.0,
        step_result=running_step,
        **common,
    ) == (False, "等待机械臂命令通过最终 admission 并实际派发。")


def test_recording_admission_blocks_nonzero_dmtac_hardware_fault() -> None:
    backend = legacy_entrypoint.DMTacGripperBackend(
        legacy_entrypoint.DMTacGripperBackendConfig(
            side="right",
            connect_hardware=True,
            connect_tactile=False,
        )
    )
    backend._apply_gripper_state(
        SimpleNamespace(
            joint_pos=0.993,
            timestamp=10.0,
            error_code=16,
            status_code=5,
            status_text="正在张开",
            is_moving=True,
            current=0,
            temperature_c=36,
        )
    )
    args = SimpleNamespace(
        no_camera=True,
        ignore_camera=(),
        ignore_tactile=(),
        hand_backend="none",
    )
    components_snapshot = SimpleNamespace(
        active_sides=("right",),
        left_hand_backend=None,
        right_hand_backend=backend,
    )
    running_step = SimpleNamespace(
        state=legacy_entrypoint.TeleopRunState.RUNNING,
        observation=object(),
        command=SimpleNamespace(commands={"right_arm": object()}),
        command_dispatched=True,
        channels={},
    )

    ready, message = recording_admission.recording_warmup_status(
        10.0,
        step_result=running_step,
        args=args,
        components=components_snapshot,
        clutch_input=SimpleNamespace(hold_active=lambda: False),
        dataset_recorder=None,
        camera_sources={},
        preview_camera_sources={},
        camera_previewer=None,
        recording_auto_stop_reason_fn=recording_admission.recording_auto_stop_reason,
        gripper_warmup_probe_status=lambda now, *, reason: (True, "ready"),
        camera_status_blocking_message=lambda status: None,
    )

    assert not ready
    assert message == "等待 right 夹爪故障恢复：error_code=16，status=5:正在张开。"


def test_position_episode_debug_detaches_before_deferred_close(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    metadata_updates: list[dict[str, object]] = []

    class FakeRecorder:
        output_dir = tmp_path / "episode-sidecar"

        def close(self, *, timeout_sec: float) -> None:
            events.append(f"close:{timeout_sec}")

        def snapshot(self) -> dict[str, object]:
            return {
                "closed": True,
                "enqueue_drop_count": 0,
                "writer_error": None,
            }

    recorder = FakeRecorder()

    class FakeFanout:
        def add_sink(self, candidate: object) -> bool:
            assert candidate is recorder
            events.append("add")
            return True

        def remove_sink(self, candidate: object) -> bool:
            assert candidate is recorder
            events.append("remove")
            return True

    runtime_snapshot = position_episode_debug.build_position_episode_debug_runtime(
        SimpleNamespace(
            control_debug_config=SimpleNamespace(enabled=True),
            control_debug_episode_sidecar=True,
        ),
        SimpleNamespace(),
        position_audit_fanout=FakeFanout(),
        position_debug_close_threads=[],
        update_runtime_metadata=lambda **kwargs: metadata_updates.append(kwargs),
        new_position_debug_recorder=lambda *args, **kwargs: recorder,
        log=lambda *args, **kwargs: None,
    )
    dataset = SimpleNamespace(dataset_root=tmp_path)

    assert runtime_snapshot.start(3, dataset_recorder=dataset)
    assert runtime_snapshot.current_recorder() is recorder
    close_action = runtime_snapshot.stop(reason="unit_test")
    assert close_action is not None
    assert runtime_snapshot.current_recorder() is None
    assert events == ["add", "remove"]

    result = close_action()
    assert result is not None and result["closed"] is True
    assert events == ["add", "remove", "close:5.0"]
    assert metadata_updates[-1]["episode_sidecar_detach_reason"] == "unit_test"


def test_vive_calibration_runtime_and_shared_math_preserve_legacy_contracts() -> None:
    for name in (
        "_capture_dual_wrist_offsets_from_gui",
        "_capture_single_vive_channel_from_gui",
        "_channel_for_tracker_role",
        "_channel_sample_key",
        "_clear_runtime_home_state",
        "_consume_vive_capture_events",
        "_default_gui_vive_calibration_side",
        "_format_vive_source_rates",
        "_format_vive_sweep_quality",
        "_format_wall_time",
        "_publish_vive_capture_state",
        "_resolve_virtual_desk_channel_name",
        "_select_gui_vive_coordinate_sweep_hand",
        "_source_debug_stats",
        "_tracker_role_for_channel",
        "_virtual_desk_post_rotation_pose",
    ):
        assert getattr(legacy_entrypoint, name) is getattr(
            vive_calibration_runtime,
            name,
        )
    assert list(
        inspect.signature(
            legacy_entrypoint._calibrate_gui_vive_coordinate_profile
        ).parameters
    ) == ["args", "components", "ui"]
    for name in (
        "_calibration_profile_latest_sort_key",
        "_calibration_profile_payload_without_workspace",
        "_runtime_home_was_available",
    ):
        assert getattr(legacy_entrypoint, name) is getattr(
            vive_calibration_profiles,
            name,
        )
    expected_profile_parameters = {
        "_latest_vive_calibration_profile_path": [],
        "_gui_vive_calibration_load_path": ["args"],
        "_save_gui_vive_calibration_profile": ["args", "components"],
        "_load_gui_vive_calibration_profile": ["args", "components"],
    }
    for name, expected_parameters in expected_profile_parameters.items():
        assert list(
            inspect.signature(getattr(legacy_entrypoint, name)).parameters
        ) == expected_parameters
        assert list(
            inspect.signature(getattr(vive_calibration_profiles, name)).parameters
        ) == [*expected_parameters, "profile_dir"]
    assert not hasattr(vive_calibration_profiles, "bind_legacy_support")

    for name in (
        "AxisFitQuality",
        "SignedAxisFit",
        "VirtualDeskCalibrationResult",
        "average_pose",
        "compose_tracker_offset",
        "euler_xyz_deg_to_matrix",
        "fit_signed_axis_from_poses",
        "fit_virtual_desk_frame",
        "pose_from_rotation_translation",
        "tracker_offset_from_neutral_pose",
    ):
        assert getattr(legacy_calibration_flow, name) is getattr(
            shared_calibration_flow,
            name,
        )

    for path in (
        ROOT / "teleop_runtime/calibration_flow.py",
        ROOT / "teleop_runtime/real_teleop/vive_calibration_profiles.py",
        ROOT / "teleop_runtime/real_teleop/vive_calibration_runtime.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "from teleop_runtime.debug" not in source
        assert "from ..debug" not in source


def test_hand_runtime_and_gripper_status_preserve_legacy_identity_and_boundaries() -> None:
    for name in (
        "_build_runtime_hand_command_for_recovery",
        "_dmtac_command_contract_configured",
        "_dmtac_tactile_backends",
        "_emit_tactile_sum_only_status",
        "_enable_gui_tactile_solving",
        "_extract_hand_command_from_command",
        "_gripper_backend_for_side",
        "_gripper_gui_available",
        "_gripper_input_channel_name",
        "_gripper_input_source_from_components",
        "_gripper_retargeter_for_side",
        "_gripper_state_scalar_from_payload",
        "_hand_backend_for_side",
        "_hand_q6_source_from_components",
        "_hand_tactile_sum",
        "_has_dmtac_gripper_backend",
        "_has_q6_gripper_control",
        "_hold_requested_pause_reason",
        "_merge_runtime_hand_command",
        "_parse_hand_tactile_regions",
        "_read_gripper_backend_scalar_now",
        "_tactile_solving_active",
        "_tactile_solving_available",
        "_tactile_visualization_available",
    ):
        assert getattr(legacy_entrypoint, name) is getattr(hand_runtime, name)

    for name in (
        "_build_gripper_status",
        "_gripper_runtime_health_signature",
        "_gripper_runtime_status_signature",
        "_gripper_status_for_side",
        "_gripper_status_signature",
        "_hand_channel_status",
        "_q6_channel_status",
        "_q6_retarget_status",
    ):
        assert getattr(legacy_entrypoint, name) is getattr(gripper_status, name)

    for path in (
        ROOT / "teleop_runtime/real_teleop/hand_runtime.py",
        ROOT / "teleop_runtime/real_teleop/gripper_status.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "run_vive_rm65_real_teleop" not in source
        assert "teleop_runtime.debug" not in source


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


def test_dataset_selection_owns_pending_contract_admission_without_ui_or_devices() -> None:
    selection_source = (
        ROOT / "teleop_runtime/real_teleop/dataset_selection.py"
    ).read_text(encoding="utf-8")
    coordinator_source = (
        ROOT / "teleop_runtime/real_teleop/operator_control_runtime.py"
    ).read_text(encoding="utf-8")

    assert callable(dataset_selection.apply_real_teleop_dataset_selection)
    assert "inspect_dataset_progress_profile" in selection_source
    assert "inspect_dataset_task_stage_profile" in selection_source
    assert "inspect_dataset_progress_profile" not in coordinator_source
    assert "inspect_dataset_task_stage_profile" not in coordinator_source
    assert "apply_real_teleop_dataset_selection(" in coordinator_source
    assert "control_station" not in selection_source
    assert "teleop_runtime.debug" not in selection_source
    assert "hardware" not in selection_source


def test_teleop_running_projection_requires_an_actual_arm_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Gesture:
        config = SimpleNamespace(enabled=False)

        def reset_for_running(self) -> None:
            return None

        def mark_runtime_restarted(self) -> None:
            return None

    session_state = SimpleNamespace(value="ARMED")
    control = operator_control_runtime.RealTeleopControlInput.__new__(
        operator_control_runtime.RealTeleopControlInput
    )
    control.components = SimpleNamespace(
        active_sides=("left", "right"),
        session=SimpleNamespace(
            state_machine=SimpleNamespace(state=session_state),
        ),
    )
    control.ui = None
    control._hold_active = False
    control._teleop_started = False
    control._teleop_dispatch_confirmed = False
    control._teleop_dispatch_projection = "idle"
    control._gripper_control_enabled = False
    control._last_calibration_started_from_hold = False
    control._q6_hold_recalibration_gesture = Gesture()
    control._dataset_selection_pending = False

    monkeypatch.setattr(
        operator_control_runtime,
        "_has_q6_gripper_control",
        lambda _components: True,
    )

    control.publish_teleop_running(home_ready=True)
    assert control._teleop_started is True
    assert control._teleop_dispatch_confirmed is False
    assert control.gripper_control_enabled() is False
    assert control.episode_start_admission() == (
        False,
        "机械臂命令尚未实际派发，不能开始数采。",
    )

    running_step = SimpleNamespace(
        state=SimpleNamespace(value="RUNNING"),
        command_dispatched=True,
        command=SimpleNamespace(
            commands={"left_arm": object(), "right_arm": object()},
            metadata={},
        ),
        motion_hold_reason="",
    )
    assert control_loop._project_session_dispatch_state(
        running_step,
        clutch_input=control,
    )
    assert control._teleop_dispatch_confirmed is True
    assert control.gripper_control_enabled() is True

    frozen_step = SimpleNamespace(
        state=SimpleNamespace(value="FROZEN"),
        command_dispatched=False,
        command=None,
        motion_hold_reason="source timing fault",
    )
    assert not control_loop._project_session_dispatch_state(
        frozen_step,
        clutch_input=control,
    )
    assert control._teleop_dispatch_confirmed is False
    assert control.gripper_control_enabled() is False


def test_two_pedal_transition_cannot_bypass_missing_dispatch_receipt() -> None:
    transitions: list[str] = []
    alerts: list[str] = []
    controller = SimpleNamespace(
        fixed_sides=(),
        select_side=lambda side: transitions.append(str(side)),
    )
    control = operator_control_runtime.RealTeleopControlInput.__new__(
        operator_control_runtime.RealTeleopControlInput
    )
    control.components = SimpleNamespace(
        session=SimpleNamespace(state_machine=SimpleNamespace()),
        fix_arm_controller=controller,
    )
    control._teleop_started = True
    control._teleop_dispatch_confirmed = False
    control._hold_active = False
    control._two_pedal_mode_active = lambda: True
    control._manual_single_arm_switch_enabled = lambda: True
    control._reject_two_pedal_switch_during_init = lambda *, now: False
    control._publish_two_pedal_alert = (
        lambda *, message, **_kwargs: alerts.append(str(message))
    )

    control._request_two_pedal_fixed_side(
        "right",
        now=10.0,
        source="pedal_1",
    )

    assert transitions == []
    assert alerts and "没有已确认的机械臂命令下发" in alerts[-1]


def test_episode_lifecycle_state_is_authoritative_without_a_gui() -> None:
    control = operator_control_runtime.RealTeleopControlInput.__new__(
        operator_control_runtime.RealTeleopControlInput
    )
    control.ui = None
    control._episode_recording_ui_active = False
    control._episode_recording_ui_paused = False
    control._episode_countdown_ui_active = False
    control._episode_finalizing_ui_active = False
    control._episode_last_episode_index = None
    control._episode_last_frame_count = None

    control.set_episode_start_countdown_state(remaining_sec=2.0)
    assert control._episode_countdown_ui_active is True
    assert control._episode_recording_ui_active is False

    control.set_episode_recording_state(
        active=True,
        frame_count=4,
        event_status="recording",
        episode_index=3,
    )
    assert control._episode_recording_ui_active is True
    assert control._episode_countdown_ui_active is False
    assert control._episode_last_episode_index == 3
    assert control._episode_last_frame_count == 4

    control.set_episode_recording_state(
        active=False,
        frame_count=9,
        event_status="finalizing",
        episode_index=3,
    )
    assert control._episode_finalizing_ui_active is True

    control.set_episode_finalize_state(
        status="saved",
        message="saved",
        episode_index=3,
        frame_count=9,
    )
    assert control._episode_finalizing_ui_active is False
    assert control._episode_recording_ui_active is False
    assert control._episode_last_frame_count == 9


def test_plus_curation_runtime_owns_request_and_writer_thread_lifecycle() -> None:
    runtime_source = (
        ROOT / "teleop_runtime/real_teleop/plus_curation_runtime.py"
    ).read_text(encoding="utf-8")
    coordinator_source = (
        ROOT / "teleop_runtime/real_teleop/operator_control_runtime.py"
    ).read_text(encoding="utf-8")

    assert callable(plus_curation_runtime.RealTeleopPlusCurationRuntime)
    assert "threading.Thread(" in runtime_source
    assert "_pending_request" in runtime_source
    assert "threading.Thread(" not in coordinator_source
    assert "_plus_curation_save_thread" not in coordinator_source
    assert "self._plus_curation_request" not in coordinator_source
    assert "control_station" not in runtime_source
    assert "hardware" not in runtime_source


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
    assert cli_value_parsers._positive_int is shared_cli_value_parsers._positive_int
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
    control._tracker_side_request_pending = False
    control._tracker_side_request_kind = "normal"
    control._tracker_side_request_deadline = 0.0
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
        # UI safety events are sampled before foot-pad routing.  The terminal
        # quit latch is checked first, then the foot-pad fd is drained without
        # routing because this batch contains Hold.
        "ui.consume_events",
        "terminal.should_quit",
        "foot_pad.consume_events",
        "terminal.consume_calibration",
        "request_hold:manual",
        "q6.poll",
        "publish.raw_tracker",
        "publish.gripper",
    ]

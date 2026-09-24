"""Offline recording-runtime fixtures shared by lifecycle contract tests."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from teleop_runtime.real_teleop import (
    episode_recording_runtime as episode_recording_module,
)
from teleop_runtime.real_teleop.episode_recording_runtime import (
    build_episode_recording_runtime,
)


class _CaptureModalityContract:
    def __init__(
        self,
        *,
        ignored_camera_keys: tuple[str, ...],
        ignored_tactile_keys: tuple[str, ...],
    ) -> None:
        self.ignored_camera_keys = ignored_camera_keys
        self.ignored_tactile_keys = ignored_tactile_keys


class _Clutch:
    def __init__(self) -> None:
        self.countdown_states: list[dict[str, object]] = []
        self.recording_states: list[dict[str, object]] = []
        self.finalize_states: list[dict[str, object]] = []
        self.restore_calls = 0
        self.round_trip_advances = 0
        self.hold_reasons: list[str] = []
        self.plus_episode_indices: list[int] = []

    def episode_start_admission(self) -> tuple[bool, str]:
        return True, ""

    def set_episode_start_countdown_state(self, **state: object) -> None:
        self.countdown_states.append(dict(state))

    def set_episode_recording_state(self, **state: object) -> None:
        self.recording_states.append(dict(state))

    def set_episode_finalize_state(self, **state: object) -> None:
        self.finalize_states.append(dict(state))

    def _restore_gripper_task_policy_for_collection(self) -> None:
        self.restore_calls += 1

    def advance_round_trip_task_after_saved_episode(self) -> None:
        self.round_trip_advances += 1

    def request_hold(self, *, reason: str) -> None:
        self.hold_reasons.append(reason)

    def begin_plus_curation_episode(self, episode_index: int) -> None:
        self.plus_episode_indices.append(int(episode_index))

    def set_plus_curation_pending(
        self,
        *,
        dataset_root: Path,
        episode_index: int,
    ) -> None:
        del dataset_root
        self.plus_episode_indices.append(int(episode_index))

    def _current_active_controls(self) -> list[str]:
        return []


def _episode_runtime(
    *,
    tmp_path: Path,
    recorder_type: type,
    dataset_slot: list[object | None],
    clutch: _Clutch,
    transition_calls: list[str],
    control_ui: object | None = None,
    debug_episode_enabled: bool = False,
    start_episode_recording=lambda *_args, **_kwargs: None,
    stop_episode_recording=lambda *_args, **_kwargs: None,
    episode_recorder_override: object | None = None,
    publish_runtime_hardware_status=lambda: None,
    bind_capture_modality_contract=lambda *_args, **_kwargs: None,
    position_episode_debug_runtime_override: object | None = None,
    update_control_debug_runtime_metadata=lambda **_kwargs: None,
    discard_episode_on_stop=None,
):
    logs: list[tuple[str, str]] = []
    episode_state = [None]
    episode_recorder = (
        SimpleNamespace(
            config=SimpleNamespace(enabled=bool(debug_episode_enabled)),
            is_recording=False,
            frame_count=0,
            episode_dir=None,
            record_event=lambda *_args, **_kwargs: None,
        )
        if episode_recorder_override is None
        else episode_recorder_override
    )
    args = SimpleNamespace(
        no_collection=False,
        dataset_root=tmp_path,
        dataset_base_root=tmp_path,
        dataset_dir_name=tmp_path.name,
        dataset_name="unit-dataset",
        collection_task_id="unit-task",
        collection_task_name="Unit task",
        collection_task_name_zh="单元任务",
        collection_task_description="Unit test task",
        ignore_camera=(),
        ignore_tactile=(),
        no_camera=True,
        camera_config=tmp_path / "camera.json",
        write_video=False,
        tactile_storage="inline",
        storage_queue_size=4,
        storage_queue_backpressure_policy="block",
        collection_warning_hz=15.0,
        dataset_startup_error="",
        arm_control_mode="position",
        control_debug_config=SimpleNamespace(enabled=False),
    )
    dependency_overrides = {
        "_CaptureModalityContract": _CaptureModalityContract,
        "_RealTeleopDatasetRecorder": recorder_type,
        "_canonical_arm_control_mode_arg": lambda _args: "position",
        "_canonical_build_episode_metadata": lambda *_args, **_kwargs: {},
        "_canonical_dataset_recording_debug_text": lambda **_kwargs: "",
        "_canonical_episode_root_dir": lambda *_args: tmp_path,
        "_canonical_log": (
            lambda tag, message, **_kwargs: logs.append((tag, message))
        ),
        "_canonical_start_episode_recording": start_episode_recording,
        "_canonical_stop_episode_recording": stop_episode_recording,
        "_bind_capture_modality_contract": bind_capture_modality_contract,
    }
    with pytest.MonkeyPatch.context() as dependency_patch:
        for name, value in dependency_overrides.items():
            dependency_patch.setattr(episode_recording_module, name, value)
        runtime = build_episode_recording_runtime(
            args=args,
            components=SimpleNamespace(),
            clutch_input=clutch,
            episode_recorder=episode_recorder,
            camera_sources={},
            control_ui=control_ui,
            startup_device_errors={},
            fix_hold_coordinator=SimpleNamespace(active=False),
            position_episode_debug_runtime=(
                SimpleNamespace(
                    start=lambda *_args, **_kwargs: False,
                    stop=lambda *_args, **_kwargs: None,
                    start_close=lambda _action: None,
                    current_recorder=lambda: None,
                )
                if position_episode_debug_runtime_override is None
                else position_episode_debug_runtime_override
            ),
            update_control_debug_runtime_metadata=update_control_debug_runtime_metadata,
            publish_runtime_hardware_status=publish_runtime_hardware_status,
            collection_startup_blockers=dict,
            recording_warmup_status=lambda *_args, **_kwargs: (True, "ready"),
            reset_gripper_warmup_probe=lambda _reason: None,
            prepare_dataset_transition=transition_calls.append,
            dataset_recorder_provider=lambda: dataset_slot[0],
            set_dataset_recorder=lambda value: dataset_slot.__setitem__(0, value),
            camera_previewer_provider=lambda: None,
            episode_last_state_provider=lambda: episode_state[0],
            set_episode_last_state=lambda value: episode_state.__setitem__(0, value),
            step_index_provider=lambda: 0,
            discard_episode_on_stop=discard_episode_on_stop,
        )
    return runtime, args, logs


def poll_recording_transition(runtime, *, timeout_sec: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while runtime.collection_transition_active() and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)
    assert not runtime.collection_transition_active()

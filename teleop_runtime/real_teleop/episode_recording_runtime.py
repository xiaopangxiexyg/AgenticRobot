"""Episode recording, countdown, pause/resume, and finalize coordination."""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .collection_control_plane import (
    CollectionControlOperationBusy,
    CollectionControlOperationResult,
    CollectionControlPlane,
)


@dataclass(frozen=True)
class _PreparedDatasetRecorder:
    recorder: Any
    owned_candidate: bool


@dataclass(frozen=True)
class _PreparedDatasetSelection:
    recorder: Any | None
    args_snapshot: Mapping[str, Any]
    previous_recorder: Any | None


@dataclass(frozen=True)
class _DatasetOpenCompletion:
    recorder: Any | None
    episode_index: int | None
    planned_episode_index: int | None
    debug_started: bool


@dataclass(frozen=True)
class _DatasetStopCompletion:
    recorder: Any | None
    task: Any | None
    formal_stopped: bool
    debug_stopped: bool
    frame_count: int


@dataclass(frozen=True)
class EpisodeRecordingRuntime:
    recording_active: Callable[[], bool]
    recording_paused: Callable[[], bool]
    recording_pause_reason: Callable[[], str]
    recording_frame_count: Callable[[], int]
    recording_path: Callable[[], str]
    recording_frequency_status: Callable[[], dict[str, object]]
    publish_collection_frequency_warning: Callable[..., None]
    start_recording: Callable[..., bool]
    schedule_recording_countdown: Callable[..., bool]
    cancel_recording_countdown: Callable[..., bool]
    advance_recording_countdown: Callable[..., None]
    stop_recording: Callable[..., bool]
    pause_recording: Callable[..., bool]
    resume_recording_if_ready: Callable[..., bool]
    sync_fix_hold: Callable[[float], None]
    poll_dataset_finalize: Callable[[float], None]
    schedule_dataset_selection: Callable[..., str]
    countdown_deadline: Callable[[], float | None]
    collection_transition_active: Callable[[], bool]
    close: Callable[..., bool]


def build_episode_recording_runtime(
    *,
    args: Any,
    components: Any,
    clutch_input: Any,
    episode_recorder: Any,
    camera_sources: Mapping[str, Any],
    control_ui: Any,
    startup_device_errors: dict[str, str],
    fix_hold_coordinator: Any,
    position_episode_debug_runtime: Any,
    update_control_debug_runtime_metadata: Callable[..., Any],
    publish_runtime_hardware_status: Callable[[], None],
    collection_startup_blockers: Callable[[], dict[str, str]],
    recording_warmup_status: Callable[..., tuple[bool, str]],
    reset_gripper_warmup_probe: Callable[[str], None],
    prepare_dataset_transition: Callable[[str], None],
    dataset_recorder_provider: Callable[[], Any],
    set_dataset_recorder: Callable[[Any], None],
    camera_previewer_provider: Callable[[], Any],
    episode_last_state_provider: Callable[[], str | None],
    set_episode_last_state: Callable[[str | None], None],
    step_index_provider: Callable[[], int],
    support: Mapping[str, Any],
) -> EpisodeRecordingRuntime:
    CaptureModalityContract = support["CaptureModalityContract"]
    DEFAULT_EPISODE_START_COUNTDOWN_SEC = support[
        "DEFAULT_EPISODE_START_COUNTDOWN_SEC"
    ]
    RealTeleopDatasetRecorder = support["RealTeleopDatasetRecorder"]
    _arm_control_mode_arg = support["_arm_control_mode_arg"]
    _build_episode_metadata = support["_build_episode_metadata"]
    _dataset_recording_debug_text = support["_dataset_recording_debug_text"]
    _episode_root_dir = support["_episode_root_dir"]
    _episode_start_countdown_ready_to_record = support[
        "_episode_start_countdown_ready_to_record"
    ]
    _episode_start_countdown_remaining = support[
        "_episode_start_countdown_remaining"
    ]
    _log = support["_log"]
    _start_episode_recording = support["_start_episode_recording"]
    _stop_episode_recording = support["_stop_episode_recording"]
    bind_capture_modality_contract = support["bind_capture_modality_contract"]

    def construct_started_dataset_recorder(
        previous: Any | None = None,
        *,
        args_override: Any | None = None,
    ) -> Any:
        """Build one writer candidate and publish nothing until it is ready.

        A process-backed collection worker owns a one-shot
        ``multiprocessing.Process``.  If construction or the ready handshake
        fails, that candidate is permanently unusable; retrying it would
        surface ``AssertionError: cannot start a process twice``.  Keep the
        candidate private through both start and device-config publication,
        close it on every failure, and let the caller publish only a fully
        usable recorder.
        """

        candidate = RealTeleopDatasetRecorder(
            args=args if args_override is None else args_override,
            components=components,
            camera_sources=camera_sources,
        )
        try:
            if previous is not None:
                previous_root = Path(previous.dataset_root).expanduser().resolve()
                candidate_root = Path(candidate.dataset_root).expanduser().resolve()
                trace_sink = getattr(previous, "control_trace_episode_sink", None)
                v25p_store = getattr(previous, "v25p_sidecar_store", None)
                if trace_sink is not None:
                    if candidate_root != previous_root:
                        raise RuntimeError(
                            "Formal ControlTrace capture cannot move dataset_root within "
                            "one runtime session; restart the frontend for the new root."
                        )
                    candidate.attach_control_trace_episode_sink(trace_sink)
                if v25p_store is not None:
                    if candidate_root != previous_root:
                        raise RuntimeError(
                            "v2.5-p sidecars cannot move dataset_root within one runtime "
                            "session; restart the frontend for the new root."
                        )
                    candidate.attach_v25p_sidecar_store(v25p_store)
            ensure_started = getattr(candidate, "ensure_started", None)
            if callable(ensure_started):
                ensure_started()
            else:
                candidate.start()
                candidate.write_camera_device_config()
        except BaseException:
            try:
                candidate.close()
            except BaseException:
                pass
            raise
        return candidate

    dataset_recorder = dataset_recorder_provider()
    camera_previewer = camera_previewer_provider()
    episode_last_state = episode_last_state_provider()
    step_index = int(step_index_provider())
    last_collection_frequency_warning_active = False
    last_collection_frequency_debug_at = 0.0
    episode_start_countdown_deadline: float | None = None
    episode_start_countdown_reason = "manual_key_start"
    episode_start_warmup_ready_since: float | None = None
    episode_start_warmup_last_message: str | None = None
    episode_start_failure_message: str | None = None
    episode_recording_paused = False
    episode_recording_pause_reason = ""
    episode_resume_warmup_ready_count = 0
    episode_resume_warmup_required_frames = 2
    collection_control_plane = CollectionControlPlane(
        name="real-teleop-collection-control"
    )
    runtime_closing = False
    episode_prepare_generation: int | None = None
    episode_prepare_reason = "manual_key_start"
    episode_prepare_cancel_reason = ""
    episode_prepare_previous: Any | None = None
    episode_prepare_previous_retirement_started = False
    episode_open_generation: int | None = None
    episode_open_recorder: Any | None = None
    episode_open_start_attempted = False
    episode_open_formal_started = False
    episode_open_start_reason = "manual_key_start"
    episode_stop_generation: int | None = None
    episode_stop_recorder: Any | None = None
    episode_stop_frame_count = 0
    episode_stop_reason = ""
    dataset_selection_generation: int | None = None
    dataset_selection_previous: Any | None = None
    dataset_selection_on_commit: Callable[[Mapping[str, Any]], None] | None = None
    dataset_selection_on_complete: Callable[[], None] | None = None
    dataset_selection_on_failure: Callable[[BaseException], None] | None = None
    dataset_selection_cleanup_generation: int | None = None
    dataset_selection_cleanup_on_complete: Callable[[], None] | None = None
    dataset_selection_cleanup_kind = ""
    finalize_failure_count = 0

    def start_position_episode_debug(
        episode_index: int,
        *,
        recorder: Any | None = None,
    ) -> bool:
        return position_episode_debug_runtime.start(
            episode_index,
            dataset_recorder=(dataset_recorder if recorder is None else recorder),
        )

    stop_position_episode_debug = position_episode_debug_runtime.stop
    start_position_episode_debug_close = position_episode_debug_runtime.start_close

    def refresh_runtime_views() -> None:
        nonlocal dataset_recorder, camera_previewer, episode_last_state, step_index
        dataset_recorder = dataset_recorder_provider()
        camera_previewer = camera_previewer_provider()
        episode_last_state = episode_last_state_provider()
        step_index = int(step_index_provider())

    def collection_transition_active() -> bool:
        return bool(
            episode_prepare_generation is not None
            or episode_open_generation is not None
            or episode_stop_generation is not None
            or dataset_selection_generation is not None
            or dataset_selection_cleanup_generation is not None
        )

    def recording_active() -> bool:
        refresh_runtime_views()
        if episode_open_generation is not None or episode_stop_generation is not None:
            # Open/stop owns the episode mutation.  Do not let the control loop
            # enqueue a sample or debug frame before that mutation has been
            # published back on the control thread.
            return False
        return bool(
            (dataset_recorder is not None and dataset_recorder.is_recording)
            or episode_recorder.is_recording
        )


    def recording_paused() -> bool:
        refresh_runtime_views()
        return bool(
            (dataset_recorder is not None and dataset_recorder.is_paused)
            or episode_recording_paused
        )


    def recording_pause_reason() -> str:
        refresh_runtime_views()
        if dataset_recorder is not None and dataset_recorder.is_paused:
            return dataset_recorder.pause_reason
        return str(episode_recording_pause_reason or "")


    def recording_frame_count() -> int:
        refresh_runtime_views()
        if dataset_recorder is not None and dataset_recorder.is_recording:
            return int(dataset_recorder.frame_count)
        return int(episode_recorder.frame_count)


    def recording_path() -> str:
        refresh_runtime_views()
        if dataset_recorder is not None:
            return str(dataset_recorder.dataset_root)
        return "" if episode_recorder.episode_dir is None else str(episode_recorder.episode_dir)


    def recording_frequency_status() -> dict[str, object]:
        refresh_runtime_views()
        if dataset_recorder is not None and dataset_recorder.is_recording:
            return dataset_recorder.collection_frequency_status()
        return {
            "sample_hz": None,
            "target_hz": float(getattr(args, "collection_warning_hz", 15.0)),
            "warning": {"active": False, "message": ""},
        }


    def publish_collection_frequency_warning(
        frequency_status: Mapping[str, object],
        *,
        now: float,
    ) -> None:
        nonlocal last_collection_frequency_debug_at, last_collection_frequency_warning_active
        refresh_runtime_views()
        warning = frequency_status.get("warning")
        warning_payload = warning if isinstance(warning, Mapping) else {}
        warning_active = bool(warning_payload.get("active", False))
        should_emit_debug = bool(
            warning_active
            and dataset_recorder is not None
            and dataset_recorder.is_recording
            and (
                not last_collection_frequency_warning_active
                or float(now) - float(last_collection_frequency_debug_at) >= 5.0
            )
        )
        if warning_active and not last_collection_frequency_warning_active:
            message = str(warning_payload.get("message") or "WARNING: 数采实际频率低于目标 15 Hz。")
            _log("DatasetFreq", message, now=now)
            if control_ui is not None:
                control_ui.append_message("DatasetFreq", message)
        if should_emit_debug:
            assert dataset_recorder is not None
            last_collection_frequency_debug_at = float(now)
            _log(
                "DatasetFreqDebug",
                _dataset_recording_debug_text(
                    dataset_recorder=dataset_recorder,
                    components=components,
                    camera_sources=camera_sources,
                    camera_previewer=camera_previewer,
                    now=now,
                ),
                now=now,
            )
        last_collection_frequency_warning_active = warning_active


    def start_recording(now: float, *, start_reason: str) -> bool:
        nonlocal episode_open_generation, episode_open_recorder
        nonlocal episode_open_start_attempted, episode_open_formal_started
        nonlocal episode_open_start_reason
        nonlocal episode_start_failure_message, dataset_recorder
        refresh_runtime_views()
        episode_start_failure_message = None
        if collection_transition_active():
            episode_start_failure_message = "数采 lifecycle 仍在过渡中。"
            return False
        admitted, admission_message = clutch_input.episode_start_admission()
        if not admitted:
            episode_start_failure_message = admission_message
            _log("Dataset", f"recording start rejected: {admission_message}", now=now)
            clutch_input._publish_episode_start_rejection(admission_message)
            return False
        if recording_active():
            episode_start_failure_message = "数采已处于运行状态。"
            return False
        if (
            dataset_recorder is None
            and not bool(getattr(args, "no_collection", False))
        ):
            message = (
                "正式 dataset writer 尚未完成异步预备；"
                "请重新点击“开始数采”并等待预备完成。"
            )
            episode_start_failure_message = message
            _log("Dataset", message, now=now)
            if control_ui is not None:
                control_ui.append_message("Dataset", message)
            return False
        if (
            bool(getattr(args, "no_collection", False))
            and dataset_recorder is None
            and not episode_recorder.config.enabled
        ):
            message = "正式 dataset collection 和 debug trajectory recording 都不可用。"
            episode_start_failure_message = message
            _log("Dataset", message, now=now)
            if control_ui is not None:
                control_ui.append_message("Dataset", message)
            return False

        formal_recorder = dataset_recorder
        captured_step_index = int(step_index)
        start_wall_time = float(now)

        def open_episode_lifecycle() -> _DatasetOpenCompletion:
            nonlocal episode_open_start_attempted, episode_open_formal_started
            planned_episode_index: int | None = None
            episode_index: int | None = None
            position_debug_started = False
            debug_started = False
            try:
                if formal_recorder is not None:
                    # Configuration inspection and the process-writer
                    # start_episode RPC may touch disk or wait for another
                    # process.  The recorder is detached while this operation
                    # owns it, so the realtime loop cannot observe a partially
                    # opened episode.
                    formal_recorder.require_configured_for_current_args()
                    planned_episode_index = int(
                        formal_recorder.writer.next_episode_index
                    )
                    position_debug_started = start_position_episode_debug(
                        planned_episode_index,
                        recorder=formal_recorder,
                    )
                    metadata = _build_episode_metadata(
                        args,
                        components,
                        start_reason=start_reason,
                    )
                    metadata.update(
                        {
                            "formal_dataset": True,
                            "dataset_root": str(formal_recorder.dataset_root),
                            "dataset_base_root": str(
                                getattr(args, "dataset_base_root", "")
                            ),
                            "dataset_dir_name": str(
                                getattr(args, "dataset_dir_name", "")
                            ),
                            "dataset_name": str(args.dataset_name),
                            "task_id": str(args.collection_task_id),
                            "task_name": str(args.collection_task_name),
                            "task_name_zh": str(
                                getattr(args, "collection_task_name_zh", "")
                            ),
                            "task_description": str(
                                args.collection_task_description
                            ),
                            "camera_keys": sorted(formal_recorder.camera_sources),
                            "camera_config_path": (
                                None
                                if args.no_camera
                                else str(args.camera_config)
                            ),
                            "write_video": bool(
                                args.write_video and not args.no_camera
                            ),
                            "tactile_storage": str(args.tactile_storage),
                            "tactile_compression": str(
                                getattr(args, "tactile_compression", "none")
                            ),
                            "storage_queue_size": int(args.storage_queue_size),
                            "storage_queue_backpressure_policy": str(
                                getattr(
                                    args,
                                    "storage_queue_backpressure_policy",
                                    "block",
                                )
                            ),
                        }
                    )
                    episode_open_start_attempted = True
                    episode_index = int(
                        formal_recorder.start_episode(
                            metadata,
                            now=start_wall_time,
                        )
                    )
                    episode_open_formal_started = True
                if episode_recorder.config.enabled:
                    episode_recorder.root_dir = _episode_root_dir(args, components)
                    _start_episode_recording(
                        episode_recorder,
                        args,
                        components,
                        now=start_wall_time,
                        start_reason=start_reason,
                    )
                    debug_started = True
                return _DatasetOpenCompletion(
                    recorder=formal_recorder,
                    episode_index=episode_index,
                    planned_episode_index=planned_episode_index,
                    debug_started=debug_started,
                )
            except BaseException:
                if debug_started or episode_recorder.is_recording:
                    try:
                        _stop_episode_recording(
                            episode_recorder,
                            now=start_wall_time,
                            end_reason="episode_open_failed",
                            summary={},
                        )
                    except BaseException:
                        pass
                if formal_recorder is not None and (
                    episode_open_start_attempted or runtime_closing
                ):
                    try:
                        if formal_recorder.is_recording:
                            formal_recorder.stop_episode(
                                reason="episode_open_failed",
                                step_index=captured_step_index,
                                last_state=None,
                                summary={},
                            )
                    except BaseException:
                        pass
                    try:
                        formal_recorder.close()
                    except BaseException:
                        pass
                if position_debug_started or (
                    position_episode_debug_runtime.current_recorder() is not None
                ):
                    try:
                        audit_close = stop_position_episode_debug(
                            reason="dataset_start_failed"
                        )
                        if audit_close is not None:
                            audit_close()
                    except BaseException:
                        pass
                raise

        def discard_open_completion(value: Any) -> None:
            if not isinstance(value, _DatasetOpenCompletion):
                return
            if value.debug_started and episode_recorder.is_recording:
                try:
                    _stop_episode_recording(
                        episode_recorder,
                        now=start_wall_time,
                        end_reason="episode_open_cancelled",
                        summary={},
                    )
                except BaseException:
                    pass
            if value.recorder is not None:
                if value.episode_index is not None:
                    try:
                        value.recorder.stop_episode(
                            reason="episode_open_cancelled",
                            step_index=captured_step_index,
                            last_state=None,
                            summary={},
                        )
                    except BaseException:
                        pass
                try:
                    value.recorder.close()
                except BaseException:
                    pass

        episode_open_recorder = formal_recorder
        episode_open_start_attempted = False
        episode_open_formal_started = False
        episode_open_start_reason = str(start_reason)
        try:
            episode_open_generation = collection_control_plane.submit(
                "open_episode",
                open_episode_lifecycle,
                discard_value=discard_open_completion,
            )
        except CollectionControlOperationBusy as exc:
            episode_start_failure_message = str(exc)
            return False
        if formal_recorder is not None:
            set_dataset_recorder(None)
            dataset_recorder = None
        message = "正在打开正式 dataset episode；遥操控制保持运行。"
        clutch_input.set_episode_recording_state(
            active=False,
            episode_dir="",
            frame_count=0,
            event_status="starting",
            event_message=message,
        )
        _log(
            "Dataset",
            f"episode open submitted generation={episode_open_generation} "
            f"reason={start_reason}",
            now=now,
        )
        if control_ui is not None:
            control_ui.append_message("Dataset", message)
        return True


    def begin_recording_countdown(now: float, *, start_reason: str) -> None:
        nonlocal episode_start_countdown_deadline, episode_start_countdown_reason
        nonlocal episode_start_warmup_ready_since, episode_start_warmup_last_message
        clutch_input._restore_gripper_task_policy_for_collection()
        countdown_sec = float(DEFAULT_EPISODE_START_COUNTDOWN_SEC)
        episode_start_countdown_deadline = float(now) + countdown_sec
        episode_start_countdown_reason = str(start_reason)
        episode_start_warmup_ready_since = None
        episode_start_warmup_last_message = None
        reset_gripper_warmup_probe(f"countdown:{start_reason}")
        message = (
            f"准备开始数采：warmup 与 {countdown_sec:.1f} 秒倒计时同时进行；"
            "两者都完成后开始写入。"
        )
        clutch_input.set_episode_start_countdown_state(
            remaining_sec=countdown_sec,
            message=message,
        )
        _log(
            "Dataset",
            f"recording warmup/countdown started delay={countdown_sec:.1f}s "
            f"reason={start_reason}",
            now=now,
        )
        if control_ui is not None:
            control_ui.append_message("Dataset", message)

    def discard_prepared_dataset_recorder(value: Any) -> None:
        if not isinstance(value, _PreparedDatasetRecorder):
            return
        if not value.owned_candidate and not runtime_closing:
            return
        try:
            value.recorder.close()
        except BaseException:
            pass

    def discard_prepared_dataset_selection(value: Any) -> None:
        if not isinstance(value, _PreparedDatasetSelection):
            return
        if value.recorder is not None:
            try:
                value.recorder.close()
            except BaseException:
                pass
        if runtime_closing and value.previous_recorder is not None:
            try:
                value.previous_recorder.close()
            except BaseException:
                pass

    def schedule_dataset_selection_cleanup(
        recorder: Any,
        *,
        cleanup_kind: str,
        on_complete: Callable[[], None] | None,
    ) -> None:
        nonlocal dataset_selection_cleanup_generation
        nonlocal dataset_selection_cleanup_on_complete
        nonlocal dataset_selection_cleanup_kind

        def close_recorder() -> None:
            recorder.close()

        generation = collection_control_plane.submit(
            str(cleanup_kind),
            close_recorder,
        )
        dataset_selection_cleanup_on_complete = on_complete
        dataset_selection_cleanup_kind = str(cleanup_kind)
        dataset_selection_cleanup_generation = generation

    def schedule_dataset_selection(
        selected_args_snapshot: Mapping[str, Any],
        *,
        bind_dataset_contract: bool,
        on_commit: Callable[[Mapping[str, Any]], None],
        on_complete: Callable[[], None],
        on_failure: Callable[[BaseException], None],
    ) -> str:
        """Prepare and publish one idle dataset selection transaction.

        Validation has already produced ``selected_args_snapshot`` without
        committing it to the live argparse namespace. A live process writer
        is replaced by a fresh private candidate; the old recorder remains
        recoverable and detached until that candidate is ready. Publication
        and the small configuration commit run on the control thread, while
        candidate startup and old-recorder retirement stay on the collection
        control plane.
        """

        nonlocal dataset_recorder, dataset_selection_generation
        nonlocal dataset_selection_previous, dataset_selection_on_commit
        nonlocal dataset_selection_on_complete, dataset_selection_on_failure
        refresh_runtime_views()
        if collection_transition_active():
            active_kind = collection_control_plane.active_kind or "collection_transition"
            raise CollectionControlOperationBusy(
                "collection control-plane operation already pending: "
                f"{active_kind}"
            )
        if episode_start_countdown_deadline is not None or recording_active():
            raise RuntimeError(
                "dataset selection requires an idle collection lifecycle"
            )
        if dataset_recorder is not None and bool(dataset_recorder.finalizing):
            raise RuntimeError(
                "dataset selection requires the current recorder to finish saving"
            )
        if not isinstance(selected_args_snapshot, Mapping):
            raise TypeError("selected dataset args snapshot must be a mapping")
        for callback, label in (
            (on_commit, "on_commit"),
            (on_complete, "on_complete"),
            (on_failure, "on_failure"),
        ):
            if not callable(callback):
                raise TypeError(f"dataset selection {label} must be callable")

        staged_args = copy.copy(args)
        vars(staged_args).clear()
        vars(staged_args).update(dict(selected_args_snapshot))
        selected_root = Path(staged_args.dataset_root)
        selected_modality_contract = CaptureModalityContract(
            ignored_camera_keys=tuple(
                getattr(staged_args, "ignore_camera", ()) or ()
            ),
            ignored_tactile_keys=tuple(
                getattr(staged_args, "ignore_tactile", ()) or ()
            ),
        )
        existing_recorder = dataset_recorder

        def prepare_selection() -> _PreparedDatasetSelection:
            try:
                prepare_dataset_transition("dataset_selection_reconfigure")
                if bind_dataset_contract:
                    bind_capture_modality_contract(
                        selected_root,
                        selected_modality_contract,
                    )
                candidate = (
                    None
                    if existing_recorder is None
                    else construct_started_dataset_recorder(
                        existing_recorder,
                        args_override=staged_args,
                    )
                )
                return _PreparedDatasetSelection(
                    recorder=candidate,
                    args_snapshot=dict(vars(staged_args)),
                    previous_recorder=existing_recorder,
                )
            except BaseException:
                if runtime_closing:
                    try:
                        existing_recorder.close()
                    except BaseException:
                        pass
                raise

        dataset_selection_previous = existing_recorder
        dataset_selection_on_commit = on_commit
        dataset_selection_on_complete = on_complete
        dataset_selection_on_failure = on_failure
        dataset_selection_generation = collection_control_plane.submit(
            "prepare_selection",
            prepare_selection,
            discard_value=discard_prepared_dataset_selection,
        )
        # A live old recorder remains recoverable but is no longer observable
        # by finalize/progress/status consumers while the candidate owns the
        # selection transaction.  When there is no recorder, selection only
        # prepares the dataset contract; writer construction stays lazy until
        # the next formal episode start.
        set_dataset_recorder(None)
        dataset_recorder = None
        _log(
            "Dataset",
            "dataset selection preparation submitted "
            f"generation={dataset_selection_generation}",
        )
        return "pending"

    def schedule_recording_countdown(now: float, *, start_reason: str) -> bool:
        nonlocal episode_prepare_generation, episode_prepare_reason
        nonlocal episode_prepare_cancel_reason, episode_start_failure_message
        nonlocal episode_prepare_previous
        nonlocal episode_prepare_previous_retirement_started, dataset_recorder
        refresh_runtime_views()
        if recording_active():
            return False
        if collection_transition_active() or episode_start_countdown_deadline is not None:
            return False
        admitted, admission_message = clutch_input.episode_start_admission()
        if not admitted:
            _log("Dataset", f"recording countdown rejected: {admission_message}", now=now)
            clutch_input._publish_episode_start_rejection(admission_message)
            return False
        if dataset_recorder is not None and dataset_recorder.finalizing:
            message = "上一个正式 dataset episode 仍在后台保存，暂不开始新的数采。"
            _log("Dataset", message, now=now)
            if control_ui is not None:
                control_ui.append_message("Dataset", message)
            return False
        hardware_blockers = collection_startup_blockers()
        if hardware_blockers:
            message = (
                "数采所需硬件仍未就绪："
                + ", ".join(sorted(hardware_blockers))
                + "。请重连，或仅对允许缺失的 camera/tactile 先设置数据集级忽视规则。"
            )
            _log("Dataset", message, now=now)
            if control_ui is not None:
                control_ui.append_message("Dataset", message)
            clutch_input.set_episode_recording_state(
                active=False,
                episode_dir=recording_path(),
                frame_count=recording_frame_count(),
                event_status="error",
                event_message=message,
            )
            return False
        if (
            bool(getattr(args, "no_collection", False))
            and not episode_recorder.config.enabled
        ):
            message = "正式 dataset collection 和 debug trajectory recording 都不可用。"
            _log("Dataset", message, now=now)
            if control_ui is not None:
                control_ui.append_message("Dataset", message)
            return False
        if bool(getattr(args, "no_collection", False)):
            begin_recording_countdown(time.time(), start_reason=start_reason)
            return True

        existing_recorder = dataset_recorder
        dataset_root = Path(args.dataset_root)
        modality_contract = CaptureModalityContract(
            ignored_camera_keys=tuple(getattr(args, "ignore_camera", ()) or ()),
            ignored_tactile_keys=tuple(getattr(args, "ignore_tactile", ()) or ()),
        )

        def prepare_writer() -> _PreparedDatasetRecorder:
            nonlocal episode_prepare_previous_retirement_started
            try:
                prepare_dataset_transition("formal_episode_start")
                bind_capture_modality_contract(dataset_root, modality_contract)
                if existing_recorder is not None:
                    configured = bool(existing_recorder.configured_for_current_args())
                    started = bool(getattr(existing_recorder, "writer_started", False))
                    if configured and started:
                        existing_recorder.write_camera_device_config()
                        return _PreparedDatasetRecorder(
                            recorder=existing_recorder,
                            owned_candidate=False,
                        )
                candidate = construct_started_dataset_recorder(existing_recorder)
                if existing_recorder is not None:
                    # From this point the old recorder is no longer recoverable:
                    # close() may partially release a one-shot process even when
                    # it raises.  The control thread must not republish it on a
                    # failed completion.
                    episode_prepare_previous_retirement_started = True
                    try:
                        existing_recorder.close()
                    except BaseException:
                        try:
                            candidate.close()
                        except BaseException:
                            pass
                        raise
                return _PreparedDatasetRecorder(
                    recorder=candidate,
                    owned_candidate=True,
                )
            except BaseException:
                if (
                    runtime_closing
                    and existing_recorder is not None
                    and not episode_prepare_previous_retirement_started
                ):
                    try:
                        existing_recorder.close()
                    except BaseException:
                        pass
                raise

        episode_prepare_reason = str(start_reason)
        episode_prepare_cancel_reason = ""
        episode_prepare_previous = existing_recorder
        episode_prepare_previous_retirement_started = False
        episode_start_failure_message = None
        try:
            episode_prepare_generation = collection_control_plane.submit(
                "prepare_writer",
                prepare_writer,
                discard_value=discard_prepared_dataset_recorder,
            )
        except CollectionControlOperationBusy as exc:
            _log("Dataset", str(exc), now=now)
            return False
        if existing_recorder is not None:
            # The worker now has exclusive lifecycle ownership.  Detaching the
            # provider prevents finalize/progress/status code from observing
            # the recorder while it is checked, reused, or retired.
            set_dataset_recorder(None)
            dataset_recorder = None
        message = "正在准备正式 dataset writer；遥操控制保持运行，ready 后开始倒计时。"
        clutch_input.set_episode_start_countdown_state(
            remaining_sec=0.0,
            message=message,
        )
        _log(
            "Dataset",
            f"writer preparation submitted generation={episode_prepare_generation} "
            f"reason={start_reason}",
            now=now,
        )
        if control_ui is not None:
            control_ui.append_message("Dataset", message)
        return True


    def cancel_recording_countdown(now: float, *, reason: str) -> bool:
        nonlocal episode_start_countdown_deadline, episode_start_warmup_ready_since
        nonlocal episode_start_warmup_last_message
        nonlocal episode_prepare_cancel_reason
        refresh_runtime_views()
        if episode_prepare_generation is not None:
            episode_prepare_cancel_reason = str(reason)
            collection_control_plane.cancel(episode_prepare_generation)
            episode_start_warmup_ready_since = None
            episode_start_warmup_last_message = None
            message = f"数采 writer 预备正在取消并回收：{reason}。"
            _log("Dataset", message, now=now)
            if control_ui is not None:
                control_ui.append_message("Dataset", message)
            clutch_input.set_episode_start_countdown_state(
                remaining_sec=0.0,
                message=message,
            )
            return True
        if episode_start_countdown_deadline is None:
            return False
        episode_start_countdown_deadline = None
        episode_start_warmup_ready_since = None
        episode_start_warmup_last_message = None
        reset_gripper_warmup_probe(f"countdown_cancelled:{reason}")
        message = f"数采倒计时已取消：{reason}。"
        _log("Dataset", message, now=now)
        if control_ui is not None:
            control_ui.append_message("Dataset", message)
        clutch_input.set_episode_recording_state(
            active=False,
            episode_dir=recording_path(),
            frame_count=recording_frame_count(),
            event_status="idle",
            event_message="数采倒计时已取消；点击“开始数采”或按 A 重新开始。",
        )
        return True


    def advance_recording_countdown(
        now: float,
        *,
        step_result: SessionStepResult | None,
    ) -> None:
        nonlocal episode_start_countdown_deadline, episode_start_warmup_ready_since
        nonlocal episode_start_warmup_last_message
        refresh_runtime_views()
        if episode_start_countdown_deadline is None:
            return
        if recording_active():
            episode_start_countdown_deadline = None
            episode_start_warmup_ready_since = None
            episode_start_warmup_last_message = None
            return
        admitted, admission_message = clutch_input.episode_start_admission()
        if not admitted:
            cancel_recording_countdown(
                now,
                reason=f"start admission changed: {admission_message}",
            )
            return
        ready, warmup_message = recording_warmup_status(now, step_result=step_result)
        remaining_sec = _episode_start_countdown_remaining(now, episode_start_countdown_deadline)
        if not ready:
            episode_start_warmup_ready_since = None
            if warmup_message != episode_start_warmup_last_message:
                episode_start_warmup_last_message = warmup_message
                _log("Dataset", f"recording warmup waiting: {warmup_message}", now=now)
            countdown_text = (
                f"倒计时还需 {remaining_sec:.1f}s。"
                if remaining_sec > 0.0
                else "倒计时已完成，warmup 完成后立即开始。"
            )
            clutch_input.set_episode_start_countdown_state(
                remaining_sec=remaining_sec,
                message=f"数采 warmup 等待中：{warmup_message}；{countdown_text}",
            )
            return
        if episode_start_warmup_ready_since is None:
            episode_start_warmup_ready_since = float(now)
            _log("Dataset", "recording warmup ready; waiting for countdown if needed", now=now)
        if not _episode_start_countdown_ready_to_record(
            now=now,
            deadline=episode_start_countdown_deadline,
            warmup_ready=True,
        ):
            clutch_input.set_episode_start_countdown_state(
                remaining_sec=remaining_sec,
                message=f"数采准备中：warmup 已就绪，倒计时还需 {remaining_sec:.1f}s。",
            )
            return
        start_reason = episode_start_countdown_reason
        episode_start_countdown_deadline = None
        episode_start_warmup_ready_since = None
        episode_start_warmup_last_message = None
        _log("Dataset", "recording warmup finished; starting collection", now=now)
        if control_ui is not None:
            control_ui.append_message("Dataset", "warmup 完成，正式开始数采写入。")
        if not start_recording(now, start_reason=start_reason):
            clutch_input.set_episode_recording_state(
                active=False,
                episode_dir=recording_path(),
                frame_count=recording_frame_count(),
                event_status="error",
                event_message=(
                    episode_start_failure_message
                    or "数采启动失败；请查看 Dataset 日志后重新点击“开始数采”。"
                ),
            )


    def stop_recording(
        now: float,
        *,
        end_reason: str,
        summary: Mapping[str, object] | None = None,
    ) -> bool:
        nonlocal episode_stop_generation, episode_stop_recorder
        nonlocal episode_stop_frame_count, episode_stop_reason, dataset_recorder
        refresh_runtime_views()
        if collection_transition_active():
            return False
        formal_recorder = (
            dataset_recorder
            if dataset_recorder is not None and dataset_recorder.is_recording
            else None
        )
        debug_was_recording = bool(episode_recorder.is_recording)
        if formal_recorder is None and not debug_was_recording:
            return False
        frame_count = recording_frame_count()
        audit_close = None
        impedance_finalize_deferred_launch = False
        if formal_recorder is not None:
            audit_close = stop_position_episode_debug(reason=end_reason)
            impedance_finalize_deferred_launch = _arm_control_mode_arg(args) == "impedance"
            if audit_close is not None and not impedance_finalize_deferred_launch:
                # Preserve the prior position-mode ordering and behavior:
                # diagnostic close remains independent of its writer path.
                start_position_episode_debug_close(audit_close)

        captured_last_state = episode_last_state
        captured_step_index = int(step_index)
        captured_summary = {} if summary is None else dict(summary)
        stop_wall_time = float(now)

        def stop_episode_lifecycle() -> _DatasetStopCompletion:
            task = None
            debug_stopped = False
            try:
                if formal_recorder is not None:
                    task = formal_recorder.stop_episode(
                        reason=end_reason,
                        step_index=captured_step_index,
                        last_state=captured_last_state,
                        summary=captured_summary,
                        defer_finalize_start=impedance_finalize_deferred_launch,
                        before_finalize=(
                            audit_close
                            if impedance_finalize_deferred_launch
                            else None
                        ),
                    )
                    if (
                        audit_close is not None
                        and task is None
                        and impedance_finalize_deferred_launch
                    ):
                        audit_close()
                if debug_was_recording:
                    _stop_episode_recording(
                        episode_recorder,
                        now=stop_wall_time,
                        end_reason=end_reason,
                        summary=captured_summary,
                    )
                    debug_stopped = True
                return _DatasetStopCompletion(
                    recorder=formal_recorder,
                    task=task,
                    formal_stopped=formal_recorder is not None,
                    debug_stopped=debug_stopped,
                    frame_count=int(frame_count),
                )
            except BaseException:
                if debug_was_recording and not debug_stopped:
                    try:
                        _stop_episode_recording(
                            episode_recorder,
                            now=stop_wall_time,
                            end_reason=f"{end_reason}:formal_stop_error",
                            summary=captured_summary,
                        )
                    except BaseException:
                        pass
                if formal_recorder is not None:
                    try:
                        formal_recorder.close()
                    except BaseException:
                        pass
                raise

        def discard_stop_completion(value: Any) -> None:
            if not isinstance(value, _DatasetStopCompletion):
                return
            if value.recorder is not None:
                try:
                    value.recorder.close()
                except BaseException:
                    pass

        try:
            episode_stop_generation = collection_control_plane.submit(
                "stop_episode",
                stop_episode_lifecycle,
                discard_value=discard_stop_completion,
            )
        except CollectionControlOperationBusy:
            return False
        episode_stop_recorder = formal_recorder
        episode_stop_frame_count = int(frame_count)
        episode_stop_reason = str(end_reason)
        if formal_recorder is not None:
            # Stop now exclusively owns the recorder through async sample drain
            # and episode mutation.  Detach it until completion publication so
            # no finalize/progress/status caller races that lifecycle.
            set_dataset_recorder(None)
            dataset_recorder = None
        message = "正在停止数采并排空最后一个已接纳样本；遥操控制保持运行。"
        clutch_input.set_episode_recording_state(
            active=True,
            episode_dir=recording_path(),
            frame_count=frame_count,
            event_status="stopping",
            event_message=message,
            episode_index=(
                int(formal_recorder.active_episode_index)
                if formal_recorder is not None
                and getattr(formal_recorder, "active_episode_index", None) is not None
                else None
            ),
        )
        _log(
            "Dataset",
            f"episode stop submitted generation={episode_stop_generation} "
            f"reason={end_reason}",
            now=now,
        )
        if control_ui is not None:
            control_ui.append_message("Dataset", message)
        return True


    def pause_recording(now: float, *, reason: str) -> bool:
        nonlocal episode_recording_paused, episode_recording_pause_reason
        nonlocal last_collection_frequency_warning_active
        nonlocal episode_resume_warmup_ready_count
        refresh_runtime_views()
        if not recording_active():
            return False
        episode_resume_warmup_ready_count = 0
        # A runtime Hold must not restart the episode-start motion probe.
        # Modality health is rechecked while resuming, using the probe's
        # existing ready/error result.
        newly_paused = False
        if dataset_recorder is not None and dataset_recorder.is_recording:
            newly_paused = bool(
                dataset_recorder.pause_episode(
                    reason=reason,
                    now=now,
                    step_index=step_index,
                    last_state=episode_last_state,
                )
            ) or newly_paused
        if episode_recorder.is_recording and not episode_recording_paused:
            episode_recorder.record_event(
                "recording_paused",
                payload={
                    "reason": str(reason),
                    "step_index": int(step_index),
                    "last_state": episode_last_state,
                    "frame_count": int(episode_recorder.frame_count),
                },
                timestamp=now,
            )
            episode_recording_paused = True
            episode_recording_pause_reason = str(reason)
            newly_paused = True
        if not recording_paused():
            return False
        last_collection_frequency_warning_active = False
        message = (
            f"数采已暂停：{reason}；当前 episode 保持打开，恢复遥操且输入稳定后会继续写入。"
        )
        if newly_paused:
            _log("Dataset", message, now=now)
            if control_ui is not None:
                control_ui.append_message("Dataset", message)
        clutch_input.set_episode_recording_state(
            active=True,
            episode_dir=recording_path(),
            frame_count=recording_frame_count(),
            frequency_status=recording_frequency_status(),
            event_status="paused",
            event_message=message,
            episode_index=(
                int(dataset_recorder.active_episode_index)
                if dataset_recorder is not None
                and getattr(dataset_recorder, "active_episode_index", None) is not None
                else None
            ),
        )
        if control_ui is not None:
            control_ui.publish(
                active_controls=clutch_input._current_active_controls()
            )
        return True


    def resume_recording_if_ready(
        now: float,
        *,
        step_result: SessionStepResult,
    ) -> bool:
        nonlocal episode_recording_paused, episode_recording_pause_reason
        nonlocal episode_resume_warmup_ready_count
        refresh_runtime_views()
        if not recording_paused():
            return True
        if fix_hold_coordinator.active:
            episode_resume_warmup_ready_count = 0
            return False
        ready, warmup_message = recording_warmup_status(now, step_result=step_result)
        if not ready:
            episode_resume_warmup_ready_count = 0
            pause_reason = recording_pause_reason()
            clutch_input.set_episode_recording_state(
                active=True,
                episode_dir=recording_path(),
                frame_count=recording_frame_count(),
                frequency_status=recording_frequency_status(),
                event_status="resume_pending",
                event_message=(
                    f"数采暂停中：{pause_reason or 'paused'}；"
                    f"等待恢复写入：{warmup_message}"
                ),
                episode_index=(
                    int(dataset_recorder.active_episode_index)
                    if dataset_recorder is not None
                    and getattr(dataset_recorder, "active_episode_index", None) is not None
                    else None
                ),
            )
            return False
        episode_resume_warmup_ready_count += 1
        if episode_resume_warmup_ready_count < episode_resume_warmup_required_frames:
            pause_reason = recording_pause_reason()
            clutch_input.set_episode_recording_state(
                active=True,
                episode_dir=recording_path(),
                frame_count=recording_frame_count(),
                frequency_status=recording_frequency_status(),
                event_status="resume_pending",
                event_message=(
                    f"数采暂停中：{pause_reason or 'paused'}；"
                    "遥操输入已恢复，等待连续稳定帧 "
                    f"{episode_resume_warmup_ready_count}/{episode_resume_warmup_required_frames}。"
                ),
                episode_index=(
                    int(dataset_recorder.active_episode_index)
                    if dataset_recorder is not None
                    and getattr(dataset_recorder, "active_episode_index", None) is not None
                    else None
                ),
            )
            return False
        resumed_any = False
        pause_reason = recording_pause_reason()
        if dataset_recorder is not None and dataset_recorder.is_paused:
            try:
                dataset_resumed = dataset_recorder.resume_episode(
                    reason="runtime_running_warmup_ready",
                    now=now,
                    step_index=step_index,
                )
            except Exception as exc:  # noqa: BLE001 - keep motion live and data paused.
                episode_resume_warmup_ready_count = 0
                message = (
                    "数采时序 segment 重置失败，为避免跨暂停窗口已保持暂停："
                    f"{type(exc).__name__}: {exc}"
                )
                _log("Dataset", message, now=now)
                if control_ui is not None:
                    control_ui.append_message("Dataset", message)
                clutch_input.set_episode_recording_state(
                    active=True,
                    episode_dir=recording_path(),
                    frame_count=recording_frame_count(),
                    frequency_status=recording_frequency_status(),
                    event_status="resume_pending",
                    event_message=message,
                    episode_index=(
                        int(dataset_recorder.active_episode_index)
                        if dataset_recorder.active_episode_index is not None
                        else None
                    ),
                )
                return False
            resumed_any = bool(dataset_resumed) or resumed_any
        if episode_recording_paused:
            episode_recorder.record_event(
                "recording_resumed",
                payload={
                    "reason": "runtime_running_warmup_ready",
                    "pause_reason": pause_reason,
                    "step_index": int(step_index),
                    "state": step_result.state.value,
                    "frame_count": int(episode_recorder.frame_count),
                },
                timestamp=now,
            )
            episode_recording_paused = False
            episode_recording_pause_reason = ""
            resumed_any = True
        if resumed_any:
            episode_resume_warmup_ready_count = 0
            message = "数采已恢复：遥操 RUNNING 且输入/相机 warmup 已稳定，继续写入同一个 episode。"
            _log("Dataset", message, now=now)
            if control_ui is not None:
                control_ui.append_message("Dataset", message)
        clutch_input.set_episode_recording_state(
            active=True,
            episode_dir=recording_path(),
            frame_count=recording_frame_count(),
            frequency_status=recording_frequency_status(),
            event_status="recording",
            event_message="正式数采正在写入；结束前请确认图像、触觉和机械臂状态正常。",
            episode_index=(
                int(dataset_recorder.active_episode_index)
                if dataset_recorder is not None
                and getattr(dataset_recorder, "active_episode_index", None) is not None
                else None
            ),
        )
        return True


    def sync_fix_hold(now: float) -> None:
        refresh_runtime_views()
        transition = fix_hold_coordinator.update(
            fix_controller=getattr(components, "fix_arm_controller", None),
            init_controller=getattr(components, "init_arm_controller", None),
        )
        if not transition.changed:
            return
        source_text = ", ".join(
            f"{side}={source}"
            for side, source in sorted(transition.source_by_side.items())
        )
        if transition.entered:
            message = (
                "左右臂均进入 Fix/Init-fixed，已进入 Fix-Hold；"
                "双臂保持、当前 episode 暂停。解除任一侧后自动恢复；"
                "再次按 Hold 将升级为常规 Hold。"
                f" ({source_text})"
            )
            clutch_input.set_fix_hold_active(True, message=message)
            cancel_recording_countdown(
                now,
                reason="fix_hold:both_sides_fixed_like",
            )
            pause_recording(now, reason="fix_hold:both_sides_fixed_like")
            if episode_recorder.is_recording:
                episode_recorder.record_event(
                    "fix_hold_started",
                    payload={
                        "step_index": int(step_index),
                        "locked_sides": list(transition.locked_sides),
                        "source_by_side": dict(transition.source_by_side),
                    },
                    timestamp=now,
                )
            _log("FixHold", message, now=now)
        elif transition.exited:
            message = (
                "Fix-Hold 已解除：至少一侧特殊固定已完成释放；"
                "遥操输入稳定后自动恢复当前 episode。"
            )
            clutch_input.set_fix_hold_active(False, message=message)
            if episode_recorder.is_recording:
                episode_recorder.record_event(
                    "fix_hold_released",
                    payload={
                        "step_index": int(step_index),
                        "remaining_locked_sides": list(transition.locked_sides),
                        "source_by_side": dict(transition.source_by_side),
                    },
                    timestamp=now,
                )
            _log("FixHold", message, now=now)


    def publish_prepare_completion(
        result: CollectionControlOperationResult,
        *,
        now: float,
    ) -> None:
        nonlocal dataset_recorder, episode_prepare_generation
        nonlocal episode_prepare_cancel_reason, episode_start_failure_message
        nonlocal episode_prepare_previous
        nonlocal episode_prepare_previous_retirement_started
        previous_recorder = episode_prepare_previous
        previous_recoverable = bool(
            previous_recorder is not None
            and not episode_prepare_previous_retirement_started
        )
        episode_prepare_generation = None
        episode_prepare_previous = None
        episode_prepare_previous_retirement_started = False
        if result.cancelled:
            if previous_recoverable:
                dataset_recorder = previous_recorder
                set_dataset_recorder(previous_recorder)
            reason = episode_prepare_cancel_reason or "operator_cancelled"
            episode_prepare_cancel_reason = ""
            message = f"数采 writer 预备已取消：{reason}。"
            _log("Dataset", message, now=now)
            if control_ui is not None:
                control_ui.append_message("Dataset", message)
            clutch_input.set_episode_recording_state(
                active=False,
                episode_dir=recording_path(),
                frame_count=recording_frame_count(),
                event_status="idle",
                event_message="writer 预备已取消；可重新开始数采。",
            )
            return
        if not result.succeeded or not isinstance(
            result.value,
            _PreparedDatasetRecorder,
        ):
            failure = result.error or result.discard_error or RuntimeError(
                "writer preparation returned an invalid completion"
            )
            if previous_recoverable:
                dataset_recorder = previous_recorder
                set_dataset_recorder(previous_recorder)
            detail = f"{type(failure).__name__}: {failure}"
            args.dataset_startup_error = detail
            startup_device_errors["dataset.writer"] = detail
            publish_runtime_hardware_status()
            message = f"正式 dataset 预备失败: {detail}"
            episode_start_failure_message = message
            _log("Dataset", message, now=now)
            if control_ui is not None:
                control_ui.append_message("Dataset", message)
            clutch_input.set_episode_recording_state(
                active=False,
                episode_dir=recording_path(),
                frame_count=recording_frame_count(),
                event_status="error",
                event_message=message,
            )
            return

        prepared = result.value
        dataset_recorder = prepared.recorder
        set_dataset_recorder(dataset_recorder)
        args.dataset_startup_error = ""
        startup_device_errors.pop("dataset.writer", None)
        try:
            publish_runtime_hardware_status()
        except BaseException as exc:
            _log(
                "Dataset",
                "selected writer is active but hardware-status refresh failed: "
                f"{type(exc).__name__}: {exc}",
                now=now,
            )
        _log(
            "DatasetTiming",
            "writer preparation completed outside the realtime loop "
            f"elapsed={result.elapsed_sec * 1000.0:.1f}ms "
            f"generation={result.generation}",
            now=now,
        )
        begin_recording_countdown(time.time(), start_reason=episode_prepare_reason)

    def publish_dataset_selection_completion(
        result: CollectionControlOperationResult,
        *,
        now: float,
    ) -> None:
        nonlocal dataset_recorder, dataset_selection_generation
        nonlocal dataset_selection_previous, dataset_selection_on_commit
        nonlocal dataset_selection_on_complete, dataset_selection_on_failure
        previous_recorder = dataset_selection_previous
        commit_callback = dataset_selection_on_commit
        complete_callback = dataset_selection_on_complete
        failure_callback = dataset_selection_on_failure
        dataset_selection_generation = None
        dataset_selection_previous = None
        dataset_selection_on_commit = None
        dataset_selection_on_complete = None
        dataset_selection_on_failure = None

        def republish_previous() -> None:
            nonlocal dataset_recorder
            dataset_recorder = previous_recorder
            if previous_recorder is not None:
                set_dataset_recorder(previous_recorder)

        def publish_failure(failure: BaseException) -> None:
            republish_previous()
            if failure_callback is not None:
                try:
                    failure_callback(failure)
                except BaseException as callback_error:
                    _log(
                        "Dataset",
                        "dataset selection rollback callback failed: "
                        f"{type(callback_error).__name__}: {callback_error}",
                        now=now,
                    )
            _log(
                "Dataset",
                "dataset selection writer replacement failed; previous "
                "recorder/configuration remain authoritative: "
                f"{type(failure).__name__}: {failure}",
                now=now,
            )

        if result.cancelled:
            publish_failure(RuntimeError("dataset selection was cancelled"))
            return
        if not result.succeeded or not isinstance(
            result.value,
            _PreparedDatasetSelection,
        ):
            publish_failure(
                result.error
                or result.discard_error
                or RuntimeError("dataset selection returned an invalid completion")
            )
            return

        prepared = result.value
        try:
            if commit_callback is None:
                raise RuntimeError("dataset selection commit callback is unavailable")
            commit_callback(prepared.args_snapshot)
            if prepared.recorder is not None:
                # The recorder was constructed from a private staged namespace
                # so its ready handshake could not expose draft configuration.
                # Future consistency checks must observe the now-committed
                # live namespace.
                prepared.recorder.args = args
            dataset_recorder = prepared.recorder
            set_dataset_recorder(prepared.recorder)
        except BaseException as exc:
            publish_failure(exc)
            if prepared.recorder is not None:
                try:
                    schedule_dataset_selection_cleanup(
                        prepared.recorder,
                        cleanup_kind="discard_selection_candidate",
                        on_complete=None,
                    )
                except BaseException as cleanup_submit_error:
                    _log(
                        "Dataset",
                        "failed to submit rejected selection candidate cleanup: "
                        f"{type(cleanup_submit_error).__name__}: "
                        f"{cleanup_submit_error}",
                        now=now,
                    )
            return

        writer_ready = prepared.recorder is not None
        if writer_ready:
            args.dataset_startup_error = ""
            startup_device_errors.pop("dataset.writer", None)
        try:
            publish_runtime_hardware_status()
        except Exception as exc:  # noqa: BLE001 - status is observational only.
            _log(
                "Dataset",
                "selected dataset state is active but hardware-status refresh failed: "
                f"{type(exc).__name__}: {exc}",
                now=now,
            )
        _log(
            "DatasetTiming",
            "dataset selection preparation completed outside the realtime loop "
            f"writer_ready={writer_ready} "
            f"elapsed={result.elapsed_sec * 1000.0:.1f}ms "
            f"generation={result.generation}",
            now=now,
        )
        if previous_recorder is None:
            if complete_callback is not None:
                complete_callback()
            return
        try:
            schedule_dataset_selection_cleanup(
                previous_recorder,
                cleanup_kind="retire_selection_previous",
                on_complete=complete_callback,
            )
        except BaseException as exc:
            # The new candidate is already ready and authoritative. Failure to
            # schedule old-recorder retirement must not roll back onto an
            # object whose lifecycle ownership has already been transferred.
            _log(
                "Dataset",
                "selected writer is active but previous recorder retirement "
                f"could not be submitted: {type(exc).__name__}: {exc}",
                now=now,
            )
            if complete_callback is not None:
                complete_callback()

    def publish_dataset_selection_cleanup(
        result: CollectionControlOperationResult,
        *,
        now: float,
    ) -> None:
        nonlocal dataset_selection_cleanup_generation
        nonlocal dataset_selection_cleanup_on_complete
        nonlocal dataset_selection_cleanup_kind
        completion_callback = dataset_selection_cleanup_on_complete
        cleanup_kind = dataset_selection_cleanup_kind
        dataset_selection_cleanup_generation = None
        dataset_selection_cleanup_on_complete = None
        dataset_selection_cleanup_kind = ""
        if not result.succeeded:
            failure = result.error or result.discard_error or RuntimeError(
                f"{cleanup_kind or 'dataset selection cleanup'} was cancelled"
            )
            message = (
                "dataset selection cleanup did not complete; the selected "
                "ready writer remains authoritative: "
                f"{type(failure).__name__}: {failure}"
            )
            _log("Dataset", message, now=now)
            if control_ui is not None:
                control_ui.append_message("Dataset", message)
        if completion_callback is not None:
            try:
                completion_callback()
            except BaseException as exc:
                _log(
                    "Dataset",
                    "dataset selection completion callback failed: "
                    f"{type(exc).__name__}: {exc}",
                    now=now,
                )

    def publish_open_completion(
        result: CollectionControlOperationResult,
        *,
        now: float,
    ) -> None:
        nonlocal dataset_recorder, episode_open_generation, episode_open_recorder
        nonlocal episode_open_start_attempted, episode_open_formal_started
        nonlocal episode_start_failure_message, episode_last_state
        nonlocal last_collection_frequency_debug_at
        nonlocal last_collection_frequency_warning_active
        opened_recorder = episode_open_recorder
        start_attempted = bool(episode_open_start_attempted)
        episode_open_generation = None
        episode_open_recorder = None
        episode_open_start_attempted = False
        episode_open_formal_started = False

        if not result.succeeded or not isinstance(
            result.value,
            _DatasetOpenCompletion,
        ):
            failure = result.error or result.discard_error or RuntimeError(
                "episode open returned an invalid completion"
            )
            if opened_recorder is not None and not start_attempted:
                # Configuration rejection happened before the process writer
                # was asked to mutate episode state, so the ready recorder is
                # still safe to reuse after the operator corrects the profile.
                dataset_recorder = opened_recorder
                set_dataset_recorder(opened_recorder)
            else:
                dataset_recorder = None
            detail = f"{type(failure).__name__}: {failure}"
            if opened_recorder is not None and start_attempted:
                args.dataset_startup_error = detail
                startup_device_errors["dataset.writer"] = detail
                publish_runtime_hardware_status()
            if bool(getattr(getattr(args, "control_debug_config", None), "enabled", False)):
                update_control_debug_runtime_metadata(
                    episode_sidecar_error=f"dataset_start_failed:{detail}"
                )
            message = f"正式 dataset episode 启动失败: {detail}"
            episode_start_failure_message = message
            _log("Dataset", message, now=now)
            if control_ui is not None:
                control_ui.append_message("Dataset", message)
            clutch_input.set_episode_recording_state(
                active=False,
                episode_dir="",
                frame_count=0,
                event_status="error",
                event_message=message,
            )
            return

        completion = result.value
        dataset_recorder = completion.recorder
        if dataset_recorder is not None:
            set_dataset_recorder(dataset_recorder)
            args.dataset_startup_error = ""
            startup_device_errors.pop("dataset.writer", None)
            publish_runtime_hardware_status()
            try:
                clutch_input.begin_plus_curation_episode(
                    int(completion.episode_index)
                )
            except (OSError, TypeError, ValueError, RuntimeError) as exc:
                # Plus remains subordinate to raw collection.  The formal
                # episode is already open and must keep recording.
                message = (
                    "Plus episode 状态初始化失败；正式原始数采继续："
                    f"{type(exc).__name__}: {exc}"
                )
                _log("Plus", message, now=now)
                if control_ui is not None:
                    control_ui.append_message("Plus", message)
            if completion.episode_index != completion.planned_episode_index:
                mismatch = (
                    "dataset episode index changed after audit attachment: "
                    f"planned={completion.planned_episode_index} "
                    f"actual={completion.episode_index}"
                )
                update_control_debug_runtime_metadata(
                    episode_sidecar_error=mismatch,
                )
                _log("ControlDebug", f"WARNING: {mismatch}", now=now)
            _log(
                "Dataset",
                "recording started "
                f"episode={completion.episode_index} "
                f"root={dataset_recorder.dataset_root}",
                now=now,
            )
            if control_ui is not None:
                control_ui.append_message(
                    "Dataset",
                    "正式数采开始："
                    f"episode {completion.episode_index} -> "
                    f"{dataset_recorder.dataset_root}",
                )
        last_collection_frequency_warning_active = False
        last_collection_frequency_debug_at = 0.0
        episode_last_state = None
        set_episode_last_state(None)
        clutch_input.set_episode_recording_state(
            active=True,
            episode_dir=recording_path(),
            frame_count=recording_frame_count(),
            frequency_status=recording_frequency_status(),
            event_status="recording",
            event_message=(
                "正式数采正在写入；结束前请确认图像、触觉和机械臂状态正常。"
                if dataset_recorder is not None
                else "Debug trajectory 正在写入。"
            ),
            episode_index=(
                int(completion.episode_index)
                if completion.episode_index is not None
                else None
            ),
        )
        _log(
            "DatasetTiming",
            "episode open completed outside the realtime loop "
            f"elapsed={result.elapsed_sec * 1000.0:.1f}ms "
            f"generation={result.generation} "
            f"reason={episode_open_start_reason}",
            now=now,
        )

    def publish_stop_completion(
        result: CollectionControlOperationResult,
        *,
        now: float,
    ) -> None:
        nonlocal dataset_recorder, episode_stop_generation, episode_stop_recorder
        nonlocal episode_stop_frame_count, episode_stop_reason, episode_last_state
        nonlocal last_collection_frequency_warning_active
        nonlocal episode_recording_paused, episode_recording_pause_reason
        nonlocal episode_resume_warmup_ready_count
        stopped_recorder = episode_stop_recorder
        stopped_frame_count = int(episode_stop_frame_count)
        stopped_reason = str(episode_stop_reason or "control_plane_completion")
        episode_stop_generation = None
        episode_stop_recorder = None
        episode_stop_frame_count = 0
        episode_stop_reason = ""
        if not result.succeeded or not isinstance(
            result.value,
            _DatasetStopCompletion,
        ):
            failure = result.error or result.discard_error or RuntimeError(
                "episode stop returned an invalid completion"
            )
            dataset_recorder = None
            detail = f"{type(failure).__name__}: {failure}"
            args.dataset_startup_error = detail
            startup_device_errors["dataset.writer"] = detail
            publish_runtime_hardware_status()
            message = (
                "正式 dataset 停止/排空失败；已隔离该 recorder 并保持 Hold，"
                f"下次开始将创建全新 writer：{detail}"
            )
            _log("Dataset", message, now=now)
            if control_ui is not None:
                control_ui.append_message("Dataset", message)
            try:
                clutch_input.request_hold(reason="dataset_stop_failure")
            except BaseException:
                pass
            clutch_input.set_episode_recording_state(
                active=False,
                episode_dir="",
                frame_count=stopped_frame_count,
                event_status="error",
                event_message=message,
            )
        else:
            completion = result.value
            if completion.recorder is not None:
                dataset_recorder = completion.recorder
                set_dataset_recorder(dataset_recorder)
            task = completion.task
            event_status = "idle"
            event_message = "数采已停止。"
            stopped_episode_index: int | None = None
            if completion.formal_stopped:
                if task is None:
                    event_status = "discarded"
                    event_message = "正式数采已停止：当前 episode 没有样本，已丢弃。"
                else:
                    event_status = "finalizing"
                    event_message = (
                        f"正式数采已停止：episode {task.episode_index} "
                        f"{task.frame_count} 帧，正在后台保存。"
                    )
                    stopped_episode_index = int(task.episode_index)
                    stopped_frame_count = int(task.frame_count)
                _log("Dataset", event_message, now=now)
                if control_ui is not None:
                    control_ui.append_message("Dataset", event_message)
            clutch_input.set_episode_recording_state(
                active=False,
                episode_dir=recording_path(),
                frame_count=stopped_frame_count,
                event_status=event_status,
                event_message=event_message,
                episode_index=stopped_episode_index,
            )
        episode_last_state = None
        set_episode_last_state(None)
        last_collection_frequency_warning_active = False
        episode_recording_paused = False
        episode_recording_pause_reason = ""
        episode_resume_warmup_ready_count = 0
        reset_gripper_warmup_probe(f"recording_stopped:{stopped_reason}")

    def poll_dataset_finalize(now: float) -> None:
        nonlocal finalize_failure_count
        if dataset_selection_generation is not None:
            result = collection_control_plane.poll(dataset_selection_generation)
            if result is not None:
                publish_dataset_selection_completion(result, now=now)
        elif dataset_selection_cleanup_generation is not None:
            result = collection_control_plane.poll(
                dataset_selection_cleanup_generation
            )
            if result is not None:
                publish_dataset_selection_cleanup(result, now=now)
        elif episode_prepare_generation is not None:
            result = collection_control_plane.poll(episode_prepare_generation)
            if result is not None:
                publish_prepare_completion(result, now=now)
        elif episode_open_generation is not None:
            result = collection_control_plane.poll(episode_open_generation)
            if result is not None:
                publish_open_completion(result, now=now)
        elif episode_stop_generation is not None:
            result = collection_control_plane.poll(episode_stop_generation)
            if result is not None:
                publish_stop_completion(result, now=now)
        if collection_transition_active():
            # The active worker exclusively owns its detached recorder.  In
            # particular, do not poll finalize/status state concurrently with
            # prepare, episode open, or sample drain/stop.
            return
        refresh_runtime_views()
        if dataset_recorder is None:
            return
        for task in dataset_recorder.poll_finalize_tasks():
            started_at = task.started_at if task.started_at is not None else task.queued_at
            elapsed = 0.0 if task.completed_at is None else task.completed_at - started_at
            queue_delay = max(0.0, started_at - task.queued_at)
            stage_text = " ".join(
                f"{name}={elapsed_ms:.1f}ms"
                for name, elapsed_ms in sorted(task.stage_timings_ms.items())
            )
            if task.success:
                message = (
                    f"正式数据集 episode {task.episode_index} 已保存 "
                    f"({task.frame_count} 帧，后台收尾 {elapsed:.2f}s，"
                    f"启动延后 {queue_delay:.3f}s"
                    f"{('，' + stage_text) if stage_text else ''})。"
                )
                _log("Dataset", message, now=now)
                if control_ui is not None:
                    control_ui.append_message("Dataset", message)
                clutch_input.set_episode_finalize_state(
                    status="saved",
                    message=message,
                    episode_index=int(task.episode_index),
                    frame_count=int(task.frame_count),
                )
                clutch_input.advance_round_trip_task_after_saved_episode()
                clutch_input.set_plus_curation_pending(
                    dataset_root=dataset_recorder.dataset_root,
                    episode_index=int(task.episode_index),
                )
                if task.v25p_error:
                    sidecar_message = (
                        f"episode {task.episode_index} 的旧格式数据已保存，但 "
                        f"v2.5-p sidecar 未完成：{task.v25p_error}"
                    )
                    _log("Dataset", sidecar_message, now=now)
                    if control_ui is not None:
                        control_ui.append_message("Dataset", sidecar_message)
                if task.v25p_native_error:
                    native_message = (
                        f"episode {task.episode_index} 的旧格式数据已保存，但 "
                        f"v2.5-p native shadow 未完成：{task.v25p_native_error}"
                    )
                    _log("Dataset", native_message, now=now)
                    if control_ui is not None:
                        control_ui.append_message("Dataset", native_message)
                if task.progress_error:
                    progress_message = (
                        f"episode {task.episode_index} 的正式数据已保存，但 "
                        f"collection progress 未完整封存：{task.progress_error}"
                    )
                    _log("Dataset", progress_message, now=now)
                    if control_ui is not None:
                        control_ui.append_message("Dataset", progress_message)
            else:
                finalize_failure_count += 1
                message = f"正式数据集 episode {task.episode_index} 保存失败: {task.error}"
                _log("Dataset", message, now=now)
                if control_ui is not None:
                    control_ui.append_message("Dataset", message)
                clutch_input.set_episode_finalize_state(
                    status="error",
                    message=message,
                    episode_index=int(task.episode_index),
                    frame_count=int(task.frame_count),
                )

    def countdown_deadline() -> float | None:
        if collection_transition_active():
            # Existing control/GUI gates already treat any non-None deadline as
            # an episode transition.  Infinity preserves that contract without
            # starting/shortening the real countdown or permitting dataset
            # maintenance while prepare/stop owns the recorder.
            return float("inf")
        return episode_start_countdown_deadline

    def close(
        *,
        now: float | None = None,
        end_reason: str = "shutdown",
        summary: Mapping[str, object] | None = None,
        timeout_sec: float = 15.0,
    ) -> bool:
        nonlocal runtime_closing
        timeout = float(timeout_sec)
        if not math.isfinite(timeout) or timeout < 0.0:
            raise ValueError(
                "episode recording shutdown timeout must be finite "
                "and non-negative"
            )
        deadline = time.monotonic() + timeout
        shutdown_now = time.time() if now is None else float(now)
        shutdown_summary = {} if summary is None else dict(summary)
        lifecycle_committed = True
        finalize_failures_before = int(finalize_failure_count)

        def remaining_sec() -> float:
            return max(0.0, deadline - time.monotonic())

        def wait_and_publish(
            generation: int,
            publish: Callable[..., None],
        ) -> bool:
            result = collection_control_plane.wait(
                generation,
                timeout_sec=remaining_sec(),
            )
            if result is None:
                return False
            publish(result, now=time.time())
            return True

        if not runtime_closing:
            # An episode-open RPC may already have crossed the writer boundary
            # when the operator exits.  Publish that exact completion first so
            # a successfully opened episode is stopped through the normal
            # drain/seal lifecycle rather than discarded as an unknown partial
            # mutation.
            if episode_open_generation is not None:
                lifecycle_committed = wait_and_publish(
                    episode_open_generation,
                    publish_open_completion,
                )
            if lifecycle_committed and episode_stop_generation is not None:
                lifecycle_committed = wait_and_publish(
                    episode_stop_generation,
                    publish_stop_completion,
                )
            if lifecycle_committed and recording_active():
                lifecycle_committed = stop_recording(
                    shutdown_now,
                    end_reason=str(end_reason),
                    summary=shutdown_summary,
                )
                if lifecycle_committed and episode_stop_generation is not None:
                    lifecycle_committed = wait_and_publish(
                        episode_stop_generation,
                        publish_stop_completion,
                    )
            # stop_episode() only hands the sample drain to the writer's
            # finalize lifecycle.  Shutdown is not complete until that task is
            # published as saved/error; otherwise round-trip/progress/Plus
            # state would be skipped even though generic recorder.close()
            # later wrote the files successfully.
            refresh_runtime_views()
            while (
                lifecycle_committed
                and dataset_recorder is not None
                and bool(getattr(dataset_recorder, "finalizing", False))
                and remaining_sec() > 0.0
            ):
                poll_dataset_finalize(time.time())
                refresh_runtime_views()
                if (
                    dataset_recorder is not None
                    and bool(getattr(dataset_recorder, "finalizing", False))
                ):
                    time.sleep(min(0.01, remaining_sec()))
            if (
                dataset_recorder is not None
                and bool(getattr(dataset_recorder, "finalizing", False))
            ):
                lifecycle_committed = False
            if finalize_failure_count > finalize_failures_before:
                lifecycle_committed = False

        runtime_closing = True
        if dataset_selection_generation is not None:
            collection_control_plane.cancel(dataset_selection_generation)
        if dataset_selection_cleanup_generation is not None:
            collection_control_plane.cancel(dataset_selection_cleanup_generation)
        if episode_prepare_generation is not None:
            collection_control_plane.cancel(episode_prepare_generation)
        if episode_open_generation is not None:
            collection_control_plane.cancel(episode_open_generation)
        if episode_stop_generation is not None:
            collection_control_plane.cancel(episode_stop_generation)
        # Prepare/open/stop detach their recorder before the worker mutates it.
        # Keep the outer shutdown path from reattaching or closing that private
        # lifecycle concurrently.  The bounded close barrier waits for its
        # cancellation cleanup before lower camera/session teardown continues.
        drained = collection_control_plane.close(timeout_sec=remaining_sec())
        if not lifecycle_committed:
            _log(
                "Dataset",
                "formal collection shutdown could not publish the normal "
                "episode stop lifecycle within the bounded timeout; pending "
                "work was cancelled and isolated",
                now=shutdown_now,
            )
        if not drained:
            _log(
                "Dataset",
                "collection control-plane worker did not drain before the "
                f"{timeout:.3f}s shutdown deadline",
                now=shutdown_now,
            )
        return bool(lifecycle_committed and drained)

    return EpisodeRecordingRuntime(
        recording_active=recording_active,
        recording_paused=recording_paused,
        recording_pause_reason=recording_pause_reason,
        recording_frame_count=recording_frame_count,
        recording_path=recording_path,
        recording_frequency_status=recording_frequency_status,
        publish_collection_frequency_warning=publish_collection_frequency_warning,
        start_recording=start_recording,
        schedule_recording_countdown=schedule_recording_countdown,
        cancel_recording_countdown=cancel_recording_countdown,
        advance_recording_countdown=advance_recording_countdown,
        stop_recording=stop_recording,
        pause_recording=pause_recording,
        resume_recording_if_ready=resume_recording_if_ready,
        sync_fix_hold=sync_fix_hold,
        poll_dataset_finalize=poll_dataset_finalize,
        schedule_dataset_selection=schedule_dataset_selection,
        countdown_deadline=countdown_deadline,
        collection_transition_active=collection_transition_active,
        close=close,
    )


__all__ = ["EpisodeRecordingRuntime", "build_episode_recording_runtime"]

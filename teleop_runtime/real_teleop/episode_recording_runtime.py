"""Episode recording, countdown, pause/resume, and finalize coordination."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any


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
    countdown_deadline: Callable[[], float | None]


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

    def start_position_episode_debug(episode_index: int) -> bool:
        return position_episode_debug_runtime.start(
            episode_index,
            dataset_recorder=dataset_recorder,
        )

    stop_position_episode_debug = position_episode_debug_runtime.stop
    start_position_episode_debug_close = position_episode_debug_runtime.start_close

    def refresh_runtime_views() -> None:
        nonlocal dataset_recorder, camera_previewer, episode_last_state, step_index
        dataset_recorder = dataset_recorder_provider()
        camera_previewer = camera_previewer_provider()
        episode_last_state = episode_last_state_provider()
        step_index = int(step_index_provider())
    def recording_active() -> bool:
        refresh_runtime_views()
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
        nonlocal episode_last_state, last_collection_frequency_debug_at, last_collection_frequency_warning_active
        nonlocal episode_start_failure_message, dataset_recorder
        refresh_runtime_views()
        episode_start_failure_message = None
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
            try:
                dataset_recorder = RealTeleopDatasetRecorder(
                    args=args,
                    components=components,
                    camera_sources=camera_sources,
                )
                set_dataset_recorder(dataset_recorder)
                dataset_recorder.start()
                dataset_recorder.write_camera_device_config()
                startup_device_errors.pop("dataset.writer", None)
                publish_runtime_hardware_status()
            except Exception as exc:  # noqa: BLE001 - keep GUI/runtime available.
                dataset_recorder = None
                set_dataset_recorder(None)
                detail = f"{type(exc).__name__}: {exc}"
                startup_device_errors["dataset.writer"] = detail
                publish_runtime_hardware_status()
                message = f"正式 dataset 当前不可用: {detail}"
                episode_start_failure_message = message
                _log("Dataset", message, now=now)
                if control_ui is not None:
                    control_ui.append_message("Dataset", message)
                return False
        last_collection_frequency_warning_active = False
        last_collection_frequency_debug_at = 0.0
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
        if dataset_recorder is not None:
            try:
                # Reconfiguration belongs before warmup/countdown.  Never
                # rebuild feedback pollers after readiness has passed.
                dataset_recorder.require_configured_for_current_args()
            except Exception as exc:  # noqa: BLE001
                message = f"正式 dataset 配置在 warmup 后发生变化: {type(exc).__name__}: {exc}"
                episode_start_failure_message = message
                _log("Dataset", message, now=now)
                if control_ui is not None:
                    control_ui.append_message("Dataset", message)
                return False
            planned_episode_index = int(dataset_recorder.writer.next_episode_index)
            # Attach before metadata is frozen and before the dataset starts
            # accepting samples.  The recorded metadata therefore describes
            # the actual sidecar state/path instead of only the request.
            start_position_episode_debug(planned_episode_index)
            metadata = _build_episode_metadata(
                args,
                components,
                start_reason=start_reason,
            )
            metadata.update(
                {
                    "formal_dataset": True,
                    "dataset_root": str(dataset_recorder.dataset_root),
                    "dataset_base_root": str(getattr(args, "dataset_base_root", "")),
                    "dataset_dir_name": str(getattr(args, "dataset_dir_name", "")),
                    "dataset_name": str(args.dataset_name),
                    "task_id": str(args.collection_task_id),
                    "task_name": str(args.collection_task_name),
                    "task_name_zh": str(getattr(args, "collection_task_name_zh", "")),
                    "task_description": str(args.collection_task_description),
                    "camera_keys": sorted(dataset_recorder.camera_sources),
                    "camera_config_path": None if args.no_camera else str(args.camera_config),
                    "write_video": bool(args.write_video and not args.no_camera),
                    "tactile_storage": str(args.tactile_storage),
                    "tactile_compression": str(getattr(args, "tactile_compression", "none")),
                    "storage_queue_size": int(args.storage_queue_size),
                    "storage_queue_backpressure_policy": str(
                        getattr(args, "storage_queue_backpressure_policy", "block")
                    ),
                }
            )
            try:
                episode_index = dataset_recorder.start_episode(metadata, now=now)
            except Exception as exc:  # noqa: BLE001
                if (
                    position_episode_debug_runtime.current_recorder()
                    is not None
                ):
                    audit_close = stop_position_episode_debug(
                        reason="dataset_start_failed"
                    )
                    if audit_close is not None:
                        start_position_episode_debug_close(audit_close)
                if bool(args.control_debug_config.enabled):
                    update_control_debug_runtime_metadata(
                        episode_sidecar_error=(
                            f"dataset_start_failed:{type(exc).__name__}: {exc}"
                        )
                    )
                message = f"正式 dataset episode 启动失败: {type(exc).__name__}: {exc}"
                episode_start_failure_message = message
                _log("Dataset", message, now=now)
                if control_ui is not None:
                    control_ui.append_message("Dataset", message)
                return False
            try:
                clutch_input.begin_plus_curation_episode(int(episode_index))
            except (OSError, TypeError, ValueError, RuntimeError) as exc:
                # Plus is deliberately subordinate to raw collection.  The
                # dataset episode is already open at this point and must keep
                # recording even if optional review state cannot initialize.
                message = (
                    "Plus episode 状态初始化失败；正式原始数采继续："
                    f"{type(exc).__name__}: {exc}"
                )
                _log("Plus", message, now=now)
                if control_ui is not None:
                    control_ui.append_message("Plus", message)
            if int(episode_index) != planned_episode_index:
                mismatch = (
                    "dataset episode index changed after audit attachment: "
                    f"planned={planned_episode_index} actual={int(episode_index)}"
                )
                update_control_debug_runtime_metadata(
                    episode_sidecar_error=mismatch,
                )
                _log("ControlDebug", f"WARNING: {mismatch}", now=now)
            _log(
                "Dataset",
                f"recording started episode={episode_index} root={dataset_recorder.dataset_root}",
                now=now,
            )
            if control_ui is not None:
                control_ui.append_message(
                    "Dataset",
                    f"正式数采开始：episode {episode_index} -> {dataset_recorder.dataset_root}",
                )
        if episode_recorder.config.enabled:
            episode_recorder.root_dir = _episode_root_dir(args, components)
            _start_episode_recording(
                episode_recorder,
                args,
                components,
                now=now,
                start_reason=start_reason,
            )
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
                int(dataset_recorder.active_episode_index)
                if dataset_recorder is not None
                and getattr(dataset_recorder, "active_episode_index", None) is not None
                else None
            ),
        )
        # The countdown probe is the admission gate for this episode.
        # Preserve its ready state after opening the episode; resetting it
        # here would make the first sample-health check move the grippers
        # again and immediately pause collection.
        return True


    def schedule_recording_countdown(now: float, *, start_reason: str) -> bool:
        nonlocal episode_start_countdown_deadline, episode_start_countdown_reason
        nonlocal episode_start_warmup_ready_since, episode_start_warmup_last_message
        nonlocal dataset_recorder
        preparation_started_at = time.perf_counter()
        refresh_runtime_views()
        if recording_active():
            return False
        if episode_start_countdown_deadline is not None:
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
        if dataset_recorder is None and not episode_recorder.config.enabled:
            message = "正式 dataset collection 和 debug trajectory recording 都不可用。"
            _log("Dataset", message, now=now)
            if control_ui is not None:
                control_ui.append_message("Dataset", message)
            return False
        if not bool(getattr(args, "no_collection", False)):
            try:
                bind_capture_modality_contract(
                    Path(args.dataset_root),
                    CaptureModalityContract(
                        ignored_camera_keys=tuple(
                            getattr(args, "ignore_camera", ()) or ()
                        ),
                        ignored_tactile_keys=tuple(
                            getattr(args, "ignore_tactile", ()) or ()
                        ),
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                message = (
                    "数采配置未通过数据集级忽视规则校验："
                    f"{type(exc).__name__}: {exc}"
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
            try:
                if dataset_recorder is None:
                    dataset_recorder = RealTeleopDatasetRecorder(
                        args=args,
                        components=components,
                        camera_sources=camera_sources,
                    )
                    set_dataset_recorder(dataset_recorder)
                    dataset_recorder.start()
                else:
                    # Task/root/modality changes rebuild the recorder and
                    # therefore replace its actual-state poller.  Perform
                    # that replacement before warmup so the countdown gate
                    # observes the exact poller that will record the episode.
                    dataset_recorder.reconfigure_for_current_args()
                dataset_recorder.write_camera_device_config()
                startup_device_errors.pop("dataset.writer", None)
                publish_runtime_hardware_status()
            except Exception as exc:  # noqa: BLE001 - keep teleop available.
                failed_recorder = dataset_recorder
                dataset_recorder = None
                set_dataset_recorder(None)
                if failed_recorder is not None:
                    try:
                        failed_recorder.close()
                    except Exception:
                        pass
                detail = f"{type(exc).__name__}: {exc}"
                startup_device_errors["dataset.writer"] = detail
                publish_runtime_hardware_status()
                message = f"正式 dataset 预备失败: {detail}"
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
        clutch_input._restore_gripper_task_policy_for_collection()
        countdown_started_at = time.time()
        preparation_elapsed_sec = max(
            0.0,
            time.perf_counter() - preparation_started_at,
        )
        if preparation_elapsed_sec >= 0.05:
            _log(
                "DatasetTiming",
                "recording countdown preparation delayed the control plane "
                f"elapsed={preparation_elapsed_sec * 1000.0:.1f}ms; "
                "the next control frame will refresh its causal timestamp",
                now=countdown_started_at,
            )
        countdown_sec = float(DEFAULT_EPISODE_START_COUNTDOWN_SEC)
        # The countdown begins only after synchronous preparation has
        # completed.  Using the request timestamp here silently shortened the
        # operator countdown by however long writer/filesystem preparation
        # took.
        episode_start_countdown_deadline = countdown_started_at + countdown_sec
        episode_start_countdown_reason = str(start_reason)
        episode_start_warmup_ready_since = None
        episode_start_warmup_last_message = None
        reset_gripper_warmup_probe(f"countdown:{start_reason}")
        message = (
            f"准备开始数采：warmup 与 {countdown_sec:.1f} 秒倒计时同时进行；"
            "两者都完成后开始写入。"
        )
        clutch_input.set_episode_start_countdown_state(remaining_sec=countdown_sec, message=message)
        _log(
            "Dataset",
            f"recording warmup/countdown started delay={countdown_sec:.1f}s reason={start_reason}",
            now=countdown_started_at,
        )
        if control_ui is not None:
            control_ui.append_message("Dataset", message)
        return True


    def cancel_recording_countdown(now: float, *, reason: str) -> bool:
        nonlocal episode_start_countdown_deadline, episode_start_warmup_ready_since
        nonlocal episode_start_warmup_last_message
        refresh_runtime_views()
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
    ) -> None:
        nonlocal episode_last_state, last_collection_frequency_warning_active
        nonlocal episode_recording_paused, episode_recording_pause_reason
        nonlocal episode_resume_warmup_ready_count
        refresh_runtime_views()
        stopped_any = False
        frame_count = recording_frame_count()
        advance_round_trip_after_stop = False
        if dataset_recorder is not None and dataset_recorder.is_recording:
            audit_close = stop_position_episode_debug(reason=end_reason)
            impedance_finalize_deferred_launch = _arm_control_mode_arg(args) == "impedance"
            if audit_close is not None and not impedance_finalize_deferred_launch:
                # Preserve the prior position-mode ordering and behavior:
                # diagnostic close remains independent of its writer path.
                start_position_episode_debug_close(audit_close)
            task = dataset_recorder.stop_episode(
                reason=end_reason,
                step_index=step_index,
                last_state=episode_last_state,
                summary=summary,
                # In force/impedance mode, launch finalization after the
                # stop-request control frame has produced its final normal
                # command. The worker remains asynchronous: recording
                # finalization must never hold tracker/IK updates.
                defer_finalize_start=impedance_finalize_deferred_launch,
                # Serializing the observational sidecar close with the
                # writer prevents the two post-stop Python finalizers from
                # competing with each other or the real-time loop.
                before_finalize=(
                    audit_close if impedance_finalize_deferred_launch else None
                ),
            )
            if (
                audit_close is not None
                and task is None
                and impedance_finalize_deferred_launch
            ):
                # Empty impedance episodes have no dataset task to own the
                # detached audit close, so retain a small standalone close.
                start_position_episode_debug_close(audit_close)
            stopped_any = True
            if task is None:
                message = "正式数采已停止：当前 episode 没有样本，已丢弃。"
            else:
                message = (
                    f"正式数采已停止：episode {task.episode_index} "
                    f"{task.frame_count} 帧，正在后台收尾。"
                )
                advance_round_trip_after_stop = True
            _log("Dataset", message, now=now)
            if control_ui is not None:
                control_ui.append_message("Dataset", message)
        if episode_recorder.is_recording:
            _stop_episode_recording(
                episode_recorder,
                now=now,
                end_reason=end_reason,
                summary={} if summary is None else dict(summary),
            )
            stopped_any = True
        if stopped_any:
            event_status = "idle"
            event_message = "数采已停止。"
            stopped_episode_index: int | None = None
            stopped_frame_count: int | None = frame_count
            if dataset_recorder is not None:
                if "task" in locals() and task is not None:
                    event_status = "finalizing"
                    event_message = (
                        f"正式数采已停止：episode {task.episode_index} "
                        f"{task.frame_count} 帧，正在后台保存。"
                    )
                    stopped_episode_index = int(task.episode_index)
                    stopped_frame_count = int(task.frame_count)
                elif "task" in locals() and task is None:
                    event_status = "discarded"
                    event_message = "正式数采已停止：当前 episode 没有样本，已丢弃。"
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
            reset_gripper_warmup_probe(f"recording_stopped:{end_reason}")
            if advance_round_trip_after_stop:
                clutch_input.advance_round_trip_task_after_saved_episode()


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


    def poll_dataset_finalize(now: float) -> None:
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
        countdown_deadline=lambda: episode_start_countdown_deadline,
    )


__all__ = ["EpisodeRecordingRuntime", "build_episode_recording_runtime"]

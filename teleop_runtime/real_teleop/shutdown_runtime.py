"""Ordered best-effort shutdown for the real-teleop application runtime.

This module owns resource release ordering only. It receives the runtime
local-state snapshot, including partial startup, and imports fixed operations
from their canonical owners; it does not start hardware or make control decisions.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Mapping
from typing import Any, Callable

from robot_motion.runtime import run_best_effort_cleanup
from teleop_runtime.log_utils import log_event as _log

from .camera_runtime import _close_camera_sources
from .control_profile import arm_control_mode as _arm_control_mode_arg
from .diagnostic_setup import (
    _attach_actuator_command_trace_sink,
    _attach_ik_shadow_trace_sink,
    _attach_position_audit_sink,
    _attach_singularity_transition_trace_sink,
)
from .impedance_force_runtime import (
    _position_force_guard_collision_summary,
    _terminal_force_collision_summary,
)


def shutdown_real_teleop_runtime(
    runtime_state: Mapping[str, Any],
    *,
    terminal_error: BaseException | None,
) -> None:
    args = runtime_state["args"]
    startup_progress = runtime_state["startup_progress"]
    position_force_guard_monitor = runtime_state["position_force_guard_monitor"]
    episode_recorder = runtime_state["episode_recorder"]
    clutch_input = runtime_state["clutch_input"]
    dataset_recorder = runtime_state["dataset_recorder"]
    camera_previewer = runtime_state["camera_previewer"]
    camera_sources = runtime_state["camera_sources"]
    tactile_visualizer = runtime_state["tactile_visualizer"]
    actuator_trace_recorder = runtime_state["actuator_trace_recorder"]
    components = runtime_state["components"]
    position_audit_fanout = runtime_state["position_audit_fanout"]
    position_debug_session_recorder = runtime_state[
        "position_debug_session_recorder"
    ]
    position_debug_close_threads = runtime_state["position_debug_close_threads"]
    position_episode_debug_runtime = runtime_state[
        "position_episode_debug_runtime"
    ]
    original_stdout = runtime_state["original_stdout"]
    silenced_stdout = runtime_state["silenced_stdout"]
    update_control_debug_runtime_metadata = runtime_state[
        "update_control_debug_runtime_metadata"
    ]

    main_loop_watchdog = runtime_state.get("main_loop_watchdog")
    debug_timing_console_writer = runtime_state.get(
        "debug_timing_console_writer"
    )
    debug_timing_log_recorder = runtime_state.get("debug_timing_log_recorder")
    action_debug_log_recorder = runtime_state.get("action_debug_log_recorder")
    ik_debug_log_recorder = runtime_state.get("ik_debug_log_recorder")
    recording_active = runtime_state.get("recording_active")
    stop_recording = runtime_state.get("stop_recording")
    idle_tactile_compression = runtime_state.get("idle_tactile_compression")

    shutdown_steps: list[tuple[str, Callable[[], None]]] = []
    position_force_guard_fault = (
        None
        if position_force_guard_monitor is None
        else position_force_guard_monitor.fault
    )
    force_collision_summary = _position_force_guard_collision_summary(
        position_force_guard_fault
    )
    if not force_collision_summary:
        force_collision_summary = _terminal_force_collision_summary(
            terminal_error,
            control_mode=_arm_control_mode_arg(args),
        )
    if startup_progress is not None:
        shutdown_steps.append(("startup_progress", startup_progress.close))
    if idle_tactile_compression is not None:
        # Revoke maintenance before any recorder/camera/device teardown.  The
        # controller joins the child and publishes its filesystem barrier
        # before returning.
        shutdown_steps.append(
            ("idle_tactile_compression", idle_tactile_compression.close)
        )
    if position_force_guard_monitor is not None:
        shutdown_steps.append(
            ("position_force_guard_monitor", position_force_guard_monitor.close)
        )
    if "main_loop_watchdog" in runtime_state:
        shutdown_steps.append(("main_loop_watchdog", main_loop_watchdog.stop))
    if "debug_timing_console_writer" in runtime_state and debug_timing_console_writer is not None:
        shutdown_steps.append(
            ("debug_timing_console_writer", debug_timing_console_writer.close)
        )
    if "debug_timing_log_recorder" in runtime_state:
        shutdown_steps.append(
            (
                "debug_timing_log_recorder",
                lambda: debug_timing_log_recorder.stop(
                    reason="finally",
                    now=time.time(),
                ),
            )
        )
    if "action_debug_log_recorder" in runtime_state:
        shutdown_steps.append(
            (
                "action_debug_log_recorder",
                lambda: action_debug_log_recorder.stop(
                    reason="finally",
                    now=time.time(),
                ),
            )
        )
    if "ik_debug_log_recorder" in runtime_state:
        shutdown_steps.append(
            (
                "ik_debug_log_recorder",
                lambda: ik_debug_log_recorder.stop(
                    reason="finally",
                    now=time.time(),
                ),
            )
        )
    if "recording_active" in runtime_state and recording_active():
        recording_end_reason = (
            "position_force_guard"
            if position_force_guard_fault is not None
            else (
                "force_collision"
                if force_collision_summary
                else "shutdown"
            )
        )
        recording_summary: dict[str, object] = {
            "reason": (
                "position_force_guard"
                if position_force_guard_fault is not None
                else (
                    "force_collision"
                    if force_collision_summary
                    else "finally"
                )
            )
        }
        if position_force_guard_fault is not None:
            recording_summary["position_force_guard_fault"] = (
                position_force_guard_fault.to_dict()
            )
        recording_summary.update(force_collision_summary)
        shutdown_steps.append(
            (
                "active_recording",
                lambda: stop_recording(
                    time.time(),
                    end_reason=recording_end_reason,
                    summary=recording_summary,
                ),
            )
        )
    shutdown_steps.extend(
        [
            ("episode_recorder", episode_recorder.close),
            ("clutch_input", clutch_input.close),
        ]
    )
    if dataset_recorder is not None:
        shutdown_steps.append(("dataset_recorder", dataset_recorder.close))
    if camera_previewer is not None:
        shutdown_steps.append(("camera_previewer", camera_previewer.close))
    shutdown_steps.append(("camera_sources", lambda: _close_camera_sources(camera_sources)))
    if tactile_visualizer is not None:
        shutdown_steps.append(("tactile_visualizer", tactile_visualizer.close))

    if actuator_trace_recorder is not None:
        shutdown_steps.append(
            (
                "actuator_trace.detach",
                lambda: _attach_actuator_command_trace_sink(components, None),
            )
        )
        shutdown_steps.append(
            (
                "actuator_trace.singularity_detach",
                lambda: _attach_singularity_transition_trace_sink(
                    components,
                    None,
                ),
            )
        )
        shutdown_steps.append(
            (
                "actuator_trace.ik_shadow_detach",
                lambda: _attach_ik_shadow_trace_sink(components, None),
            )
        )
    shutdown_steps.extend(
        [
            (
                "position_audit.detach",
                lambda: _attach_position_audit_sink(components, None),
            ),
            (
                "position_audit.metadata_detach",
                lambda: update_control_debug_runtime_metadata(
                    attached=False,
                    session_sidecar_attached=False,
                    session_sidecar_detached_wall_time=time.time(),
                    session_sidecar_detached_monotonic_time=time.monotonic(),
                ),
            ),
        ]
    )
    episode_debug_recorder_at_shutdown = (
        None
        if position_episode_debug_runtime is None
        else position_episode_debug_runtime.current_recorder()
    )
    if episode_debug_recorder_at_shutdown is not None:
        if position_audit_fanout is not None:
            shutdown_steps.append(
                (
                    "position_audit.episode.remove_sink",
                    lambda recorder=episode_debug_recorder_at_shutdown: (
                        position_audit_fanout.remove_sink(recorder)
                    ),
                )
            )
        shutdown_steps.extend(
            [
                (
                    "position_audit.episode_recorder",
                    lambda recorder=episode_debug_recorder_at_shutdown: recorder.close(
                        timeout_sec=5.0
                    ),
                ),
                (
                    "position_audit.episode_metadata",
                    lambda: update_control_debug_runtime_metadata(
                        episode_sidecar_attached=False,
                        episode_sidecars=False,
                        episode_sidecar_detached_wall_time=time.time(),
                        episode_sidecar_detached_monotonic_time=time.monotonic(),
                        episode_sidecar_detach_reason="shutdown",
                    ),
                ),
            ]
        )
    if position_audit_fanout is not None:
        shutdown_steps.append(("position_audit.fanout", position_audit_fanout.clear))
    if position_debug_session_recorder is not None:
        def close_position_debug_session(
            recorder: PositionDebugRecorder = position_debug_session_recorder,
        ) -> None:
            recorder.close(timeout_sec=5.0)
            snapshot = recorder.snapshot()
            writer_error = snapshot.get("writer_error")
            update_control_debug_runtime_metadata(
                session_sidecar_error=writer_error,
            )
            _log(
                "ControlDebug",
                "session audit closed "
                f"output={recorder.output_dir} "
                f"drops={snapshot.get('enqueue_drop_count', 0)} "
                f"fanout_errors={0 if position_audit_fanout is None else position_audit_fanout.error_count} "
                f"writer_error={writer_error}",
            )

        shutdown_steps.append(
            ("position_audit.session_recorder", close_position_debug_session)
        )
    for close_thread in position_debug_close_threads:
        def join_position_debug_thread(
            thread: threading.Thread = close_thread,
        ) -> None:
            thread.join(timeout=5.0)
            if thread.is_alive():
                raise RuntimeError(
                    f"position debug close thread {thread.name!r} did not stop within 5.000s"
                )

        shutdown_steps.append(
            (f"position_audit.close_thread[{close_thread.name}]", join_position_debug_thread)
        )
    if components.human_shadow_frontend is not None:
        shutdown_steps.append(
            ("human_action_shadow", components.human_shadow_frontend.close)
        )
    runtime_extension = runtime_state.get("runtime_extension")
    if runtime_extension is not None:
        close_before_session = getattr(
            runtime_extension,
            "close_before_session",
            None,
        )
        if not callable(close_before_session):
            raise TypeError("runtime_extension lacks close_before_session().")
        shutdown_steps.append(
            (
                "runtime_extension.before_session",
                lambda: close_before_session(
                    components=components,
                    terminal_error=terminal_error,
                ),
            )
        )
    shutdown_steps.append(("teleop_session", components.session.close))
    if actuator_trace_recorder is not None:
        def close_actuator_trace(
            recorder: ActuatorCommandTraceRecorder = actuator_trace_recorder,
        ) -> None:
            recorder.close()
            snapshot = recorder.snapshot()
            _log(
                "ActuatorTrace",
                f"capture closed output={recorder.output_dir} "
                f"rows={snapshot.get('rows_by_side')} "
                f"singularity_rows={snapshot.get('singularity_rows_by_side')} "
                f"drops={snapshot.get('drop_count_by_side')} "
                f"errors={snapshot.get('error_count_by_side')}",
            )

        shutdown_steps.append(("actuator_trace.recorder", close_actuator_trace))
    if components.arm_state_read_executor is not None:
        shutdown_steps.append(
            (
                "arm_state_read_executor",
                lambda: components.arm_state_read_executor.shutdown(wait=True),
            )
        )
    if components.right_hand_backend is not None and not components.right_hand_control_via_rig:
        shutdown_steps.append(("right_hand_backend", components.right_hand_backend.close))
    if components.left_hand_backend is not None and not components.left_hand_control_via_rig:
        shutdown_steps.append(("left_hand_backend", components.left_hand_backend.close))
    if components.right_state_backend is not None:
        shutdown_steps.append(("right_state_backend", components.right_state_backend.close))
    if components.left_state_backend is not None:
        shutdown_steps.append(("left_state_backend", components.left_state_backend.close))
    if original_stdout is not None:
        def restore_stdout() -> None:
            sys.stdout = original_stdout

        shutdown_steps.append(("stdout.restore", restore_stdout))
    if silenced_stdout is not None:
        shutdown_steps.append(("stdout.silenced_stream", silenced_stdout.close))
    try:
        run_best_effort_cleanup("run_vive_rm65_real_teleop.shutdown", shutdown_steps)
    except BaseException as cleanup_error:  # noqa: BLE001 - preserve the primary exit.
        # Ctrl+C is a normal operator exit, not a reason to replace the
        # original KeyboardInterrupt with a second lifecycle traceback.  The
        # individual failures have already been collected by
        # run_best_effort_cleanup; report them through the normal logger and
        # let the caller decide how to render the primary terminal error.
        _log(
            "Shutdown",
            "cleanup completed with recoverable failures: "
            f"{type(cleanup_error).__name__}: {cleanup_error}",
        )
        if terminal_error is None:
            raise


__all__ = ["shutdown_real_teleop_runtime"]

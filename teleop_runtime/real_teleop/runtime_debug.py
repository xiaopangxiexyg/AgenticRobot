"""Read-only runtime diagnostics for the real-teleop control path.

The helpers in this module sample existing state and format diagnostic output.
They do not dispatch commands, mutate device state, or own runtime authority.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from ..collection.diagnostics import (
    build_end_to_end_latency_breakdown,
    format_end_to_end_latency_breakdown,
    format_phase_timings,
    should_report_debug_frozen,
)
from ..collection.status_snapshot import (
    camera_debug_text as _collection_camera_debug_text,
    hand_debug_text as _collection_hand_debug_text,
    writer_debug_text as _collection_writer_debug_text,
)
from .status_payloads import (
    _debug_timing_arm_text,
    _debug_timing_fault_text,
    _debug_timing_ik_text,
)
from .action_debug_payload import _action_debug_backend_snapshot_from_state
from .gripper_status import _DEBUG_TIMING_RED, _DEBUG_TIMING_RESET
from .hand_runtime import _hand_backend_for_side


def _action_debug_backend_snapshot(
    backend: Any | None,
    *,
    side: str,
    role: str,
    sample_timestamp: float | None,
    record_wall_time: float,
) -> dict[str, Any]:
    if backend is None:
        return {
            "available": False,
            "side": str(side),
            "role": str(role),
            "reason": "missing_backend",
        }
    read_start = time.time()
    try:
        state = backend.get_state()
    except Exception as exc:  # noqa: BLE001
        read_end = time.time()
        return {
            "available": False,
            "side": str(side),
            "role": str(role),
            "backend_class": type(backend).__name__,
            "reason": "get_state_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "read_start_wall_time": float(read_start),
            "read_end_wall_time": float(read_end),
            "read_elapsed_s": float(read_end - read_start),
        }
    read_end = time.time()
    return _action_debug_backend_snapshot_from_state(
        state,
        backend=backend,
        side=side,
        role=role,
        sample_timestamp=sample_timestamp,
        record_wall_time=record_wall_time,
        read_start=read_start,
        read_end=read_end,
    )


def _preloaded_action_debug_backend_snapshot(
    backend: Any | None,
    state: Mapping[str, object] | None,
    *,
    side: str,
    role: str,
    sample_timestamp: float | None,
    record_wall_time: float,
) -> dict[str, Any]:
    if isinstance(state, Mapping):
        return _action_debug_backend_snapshot_from_state(
            state,
            backend=backend,
            side=side,
            role=role,
            sample_timestamp=sample_timestamp,
            record_wall_time=record_wall_time,
            read_start=record_wall_time,
            read_end=record_wall_time,
        )
    return _action_debug_backend_snapshot(
        backend,
        side=side,
        role=role,
        sample_timestamp=sample_timestamp,
        record_wall_time=record_wall_time,
    )


def _dataset_storage_debug_text(dataset_recorder: RealTeleopDatasetRecorder) -> str:
    stats = dataset_recorder._storage_queue_stats_payload()
    if not bool(stats.get("enabled", False)):
        return "storage_queue=off"
    routes = stats.get("routes")
    if not isinstance(routes, Mapping) or not routes:
        return (
            "storage_queue=enabled"
            f":size:{int(stats.get('storage_queue_size', 0) or 0)},routes:none"
        )
    parts: list[str] = []
    for key, route in sorted(routes.items()):
        if not isinstance(route, Mapping):
            parts.append(f"{key}=invalid")
            continue
        parts.append(
            f"{key}=q:{int(route.get('queued', 0) or 0)},"
            f"enq:{int(route.get('enqueued', 0) or 0)},"
            f"sub:{int(route.get('submitted', 0) or 0)},"
            f"drop:{int(route.get('dropped', 0) or 0)},"
            f"high:{int(route.get('high_water', 0) or 0)}/"
            f"{int(route.get('max_queue_size', 0) or 0)}"
        )
    return "storage_queue=" + "; ".join(parts)


def _dataset_recording_debug_text(
    *,
    dataset_recorder: RealTeleopDatasetRecorder,
    components: RealTeleopComponents,
    camera_sources: Mapping[str, RealSenseRGBSource],
    camera_previewer: CameraPreviewer | None,
    now: float,
) -> str:
    writer_owner = (
        dataset_recorder.process_worker
        if getattr(dataset_recorder, "process_worker", None) is not None
        else dataset_recorder.writer_pool
    )
    writer_text = _collection_writer_debug_text(writer_owner)
    storage_text = _dataset_storage_debug_text(dataset_recorder)
    camera_text = _collection_camera_debug_text(
        camera_sources,
        camera_previewer=camera_previewer,
    )
    hand_text = _collection_hand_debug_text(
        components.active_sides,
        lambda side: _hand_backend_for_side(components, side),
        now=now,
    )
    return f"{writer_text} {storage_text} {camera_text} {hand_text}"


def _debug_timing_report(
    state: RealDebugTimingState,
    *,
    reason: str,
    elapsed_sec: float,
    phases: Sequence[tuple[str, float]],
    step_index: int,
    result,
    active_command,
    left_hw_state: dict[str, object],
    right_hw_state: dict[str, object],
    left_command_state: dict[str, object],
    right_command_state: dict[str, object],
    target_step_scale: float,
    force: bool = False,
) -> None:
    if not state.enabled:
        return
    now = time.time()
    report_interval = max(0.0, float(state.report_interval_sec))
    if (
        not force
        and report_interval > 0.0
        and state.last_report_at > 0.0
        and (now - state.last_report_at) < report_interval
    ):
        state.suppressed_reports += 1
        return
    suppressed = int(state.suppressed_reports)
    state.suppressed_reports = 0
    state.last_report_at = now
    prefix = (
        f"[RealDebugTiming] slow reason={reason} elapsed={elapsed_sec * 1000.0:.1f}ms "
        f"active_hz={(1.0 / elapsed_sec) if elapsed_sec > 0.0 else 0.0:.2f} "
        f"threshold={float(state.threshold_sec) * 1000.0:.1f}ms "
        f"step={int(step_index)}"
    )
    if result is not None:
        prefix += f" control_hz={float(result.control_hz):.2f}"
    if suppressed:
        prefix += f" suppressed={suppressed}"
    lines = [
        prefix,
        "  " + format_phase_timings(phases, limit=14),
        "  "
        + format_end_to_end_latency_breakdown(
            build_end_to_end_latency_breakdown(result, reference_time=now),
            limit=6,
        ),
        "  " + _debug_timing_ik_text(active_command),
        "  "
        + _debug_timing_arm_text(
            left_hw_state=left_hw_state,
            right_hw_state=right_hw_state,
            left_command_state=left_command_state,
            right_command_state=right_command_state,
            target_step_scale=target_step_scale,
        ),
        "  " + _debug_timing_fault_text(result),
    ]
    message = _DEBUG_TIMING_RED + "\n".join(lines) + _DEBUG_TIMING_RESET
    if state.console_writer is not None:
        state.console_writer.submit(message)
    else:
        print(message, flush=True)


def _debug_timing_maybe_report(
    state: RealDebugTimingState,
    *,
    reason: str,
    elapsed_sec: float,
    phases: Sequence[tuple[str, float]],
    step_index: int,
    result,
    active_command,
    left_hw_state: dict[str, object],
    right_hw_state: dict[str, object],
    left_command_state: dict[str, object],
    right_command_state: dict[str, object],
    target_step_scale: float,
    force: bool = False,
) -> None:
    if not state.enabled:
        return
    if should_report_debug_frozen(
        elapsed_sec=elapsed_sec,
        phases=phases,
        threshold_sec=float(state.threshold_sec),
        force=force,
    ):
        _debug_timing_report(
            state,
            reason=reason,
            elapsed_sec=elapsed_sec,
            phases=phases,
            step_index=step_index,
            result=result,
            active_command=active_command,
            left_hw_state=left_hw_state,
            right_hw_state=right_hw_state,
            left_command_state=left_command_state,
            right_command_state=right_command_state,
            target_step_scale=target_step_scale,
            force=force,
        )

__all__ = [
    "_action_debug_backend_snapshot",
    "_preloaded_action_debug_backend_snapshot",
    "_dataset_storage_debug_text",
    "_dataset_recording_debug_text",
    "_debug_timing_report",
    "_debug_timing_maybe_report",
]

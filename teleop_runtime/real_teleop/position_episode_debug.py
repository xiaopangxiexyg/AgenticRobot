"""Per-episode position-control audit sidecar lifecycle."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any

from robot_motion.position_control import PositionAuditFanout

from ..position_control import PositionDebugRecorder
from .components import RealTeleopComponents


@dataclass(frozen=True)
class PositionEpisodeDebugRuntime:
    start: Callable[..., bool]
    stop: Callable[..., Callable[[], Mapping[str, Any] | None] | None]
    start_close: Callable[[Callable[[], Mapping[str, Any] | None]], None]
    current_recorder: Callable[[], PositionDebugRecorder | None]


def build_position_episode_debug_runtime(
    args: argparse.Namespace,
    components: RealTeleopComponents,
    *,
    position_audit_fanout: PositionAuditFanout | None,
    position_debug_close_threads: list[threading.Thread],
    update_runtime_metadata: Callable[..., None],
    new_position_debug_recorder: Callable[..., PositionDebugRecorder],
    log: Callable[..., None],
) -> PositionEpisodeDebugRuntime:
    """Build the historical attach/detach callbacks over one recorder slot."""

    update_control_debug_runtime_metadata = update_runtime_metadata
    _new_position_debug_recorder = new_position_debug_recorder
    _log = log
    position_debug_episode_recorder: PositionDebugRecorder | None = None

    def start_position_episode_debug(
        episode_index: int,
        *,
        dataset_recorder: object | None,
    ) -> bool:
        nonlocal position_debug_episode_recorder
        requested = bool(
            dataset_recorder is not None
            and args.control_debug_config.enabled
            and bool(getattr(args, "control_debug_episode_sidecar", True))
        )
        update_control_debug_runtime_metadata(
            episode_sidecar_requested=requested,
            episode_sidecar_attached=False,
            episode_sidecars=False,
        )
        if not requested:
            return False
        if position_audit_fanout is None:
            unavailable_error = "session position-audit fanout is unavailable"
            update_control_debug_runtime_metadata(
                episode_sidecar_error=unavailable_error,
                episode_sidecar_episode_index=int(episode_index),
            )
            return False
        if position_debug_episode_recorder is not None:
            # Defensive cleanup: normal stop removes the previous sink.
            position_audit_fanout.remove_sink(position_debug_episode_recorder)
            try:
                position_debug_episode_recorder.close(timeout_sec=1.0)
            except Exception as exc:  # noqa: BLE001 - diagnostics remain best-effort.
                _log(
                    "ControlDebug",
                    "WARNING: stale episode audit close failed before reattach: "
                    f"{type(exc).__name__}: {exc}",
                )
            position_debug_episode_recorder = None
        output_dir = (
            Path(dataset_recorder.dataset_root)
            / "meta"
            / "position_control"
            / f"episode_{int(episode_index):06d}"
        )
        recorder: PositionDebugRecorder | None = None
        try:
            recorder = _new_position_debug_recorder(
                args,
                output_dir=output_dir,
                thread_name=f"position-debug-episode-{int(episode_index):06d}",
                components=components,
                recorder_scope="episode",
                episode_index=int(episode_index),
            )
            if not position_audit_fanout.add_sink(recorder):
                raise RuntimeError("episode position-audit recorder was already attached")
            position_debug_episode_recorder = recorder
            attached_wall_time = time.time()
            attached_monotonic_time = time.monotonic()
            update_control_debug_runtime_metadata(
                episode_sidecar_requested=True,
                episode_sidecar_attached=True,
                episode_sidecar_error=None,
                episode_sidecar_output_dir=str(output_dir),
                episode_sidecar_episode_index=int(episode_index),
                episode_sidecar_attached_wall_time=attached_wall_time,
                episode_sidecar_attached_monotonic_time=attached_monotonic_time,
                episode_sidecars=True,
            )
            _log(
                "ControlDebug",
                f"episode audit attached episode={int(episode_index)} output={output_dir}",
            )
            return True
        except Exception as exc:  # noqa: BLE001 - isolate debug storage from control.
            if recorder is not None:
                position_audit_fanout.remove_sink(recorder)
                if position_debug_episode_recorder is recorder:
                    position_debug_episode_recorder = None
                try:
                    recorder.close(timeout_sec=1.0)
                except Exception:
                    pass
            setup_error = f"{type(exc).__name__}: {exc}"
            update_control_debug_runtime_metadata(
                episode_sidecar_requested=True,
                episode_sidecar_attached=False,
                episode_sidecar_error=setup_error,
                episode_sidecar_output_dir=str(output_dir),
                episode_sidecar_episode_index=int(episode_index),
                episode_sidecar_attached_wall_time=None,
                episode_sidecar_attached_monotonic_time=None,
                episode_sidecars=False,
            )
            _log(
                "ControlDebug",
                f"WARNING: episode audit setup failed: {type(exc).__name__}: {exc}",
            )
            return False

    def stop_position_episode_debug(
        *,
        reason: str,
    ) -> Callable[[], Mapping[str, Any] | None] | None:
        """Detach the episode audit immediately and return its close action.

        The caller decides where to run the close action.  In impedance
        collection it is executed by the *same* deferred finalizer as the
        dataset writer, avoiding two independently scheduled Python
        finalizers during the first post-stop control period.
        """

        nonlocal position_debug_episode_recorder
        recorder = position_debug_episode_recorder
        if recorder is None:
            return None
        position_debug_episode_recorder = None
        if position_audit_fanout is not None:
            position_audit_fanout.remove_sink(recorder)
        update_control_debug_runtime_metadata(
            episode_sidecar_attached=False,
            episode_sidecars=False,
            episode_sidecar_detached_wall_time=time.time(),
            episode_sidecar_detached_monotonic_time=time.monotonic(),
            episode_sidecar_detach_reason=str(reason),
        )

        def close_recorder() -> Mapping[str, Any] | None:
            close_started_at = time.perf_counter()
            try:
                recorder.close(timeout_sec=5.0)
                snapshot = recorder.snapshot()
                result = {
                    "closed": bool(snapshot.get("closed", False)),
                    "enqueue_drop_count": int(snapshot.get("enqueue_drop_count", 0) or 0),
                    "writer_error": snapshot.get("writer_error"),
                    "elapsed_ms": (time.perf_counter() - close_started_at) * 1000.0,
                }
                _log(
                    "ControlDebug",
                    "episode audit closed "
                    f"reason={reason} output={recorder.output_dir} "
                    f"drops={result['enqueue_drop_count']} "
                    f"writer_error={result['writer_error']} "
                    f"elapsed={result['elapsed_ms']:.1f}ms",
                )
                return result
            except Exception as exc:  # noqa: BLE001
                result = {
                    "closed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_ms": (time.perf_counter() - close_started_at) * 1000.0,
                }
                _log(
                    "ControlDebug",
                    "WARNING: episode audit close failed: "
                    f"{type(exc).__name__}: {exc} elapsed={result['elapsed_ms']:.1f}ms",
                )
                return result

        return close_recorder

    def start_position_episode_debug_close(
        close_recorder: Callable[[], Mapping[str, Any] | None],
    ) -> None:
        close_thread = threading.Thread(
            target=close_recorder,
            name="position-debug-episode-close",
            daemon=True,
        )
        position_debug_close_threads.append(close_thread)
        close_thread.start()

    return PositionEpisodeDebugRuntime(
        start=start_position_episode_debug,
        stop=stop_position_episode_debug,
        start_close=start_position_episode_debug_close,
        current_recorder=lambda: position_debug_episode_recorder,
    )


__all__ = [
    "PositionEpisodeDebugRuntime",
    "build_position_episode_debug_runtime",
]

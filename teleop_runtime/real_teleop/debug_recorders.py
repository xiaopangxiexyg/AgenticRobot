"""Asynchronous JSONL diagnostics owned by the real-teleop runtime.

The three schemas in this module describe the Vive-to-RM65 control loop only.
They intentionally stay outside the generic diagnostics package: the generic
package provides timing primitives, while these recorders own entrypoint-
specific filenames, schemas, and bounded best-effort writer behavior.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from robot_motion.runtime import run_best_effort_cleanup

from ..collection.diagnostics import should_report_debug_frozen
from project_paths import teleop_records_root


DEFAULT_IK_DEBUG_LOG_ROOT = teleop_records_root() / "ik_debug"
DEFAULT_DEBUG_TIMING_LOG_ROOT = teleop_records_root() / "debug_timing"
DEFAULT_ACTION_DEBUG_LOG_ROOT = teleop_records_root() / "action_debug"


def json_safe(value: object) -> object:
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def state_value(value: object) -> str:
    return str(getattr(value, "value", value))


def stop_debug_writer_thread(
    *,
    log_queue: queue.Queue[Any],
    writer_thread: threading.Thread | None,
    join_timeout_sec: float,
    name: str,
) -> None:
    """Guarantee a stop signal and bounded join even when the debug queue is full."""

    def send_stop_signal() -> None:
        try:
            log_queue.put(None, timeout=0.1)
            return
        except queue.Full:
            pass
        try:
            log_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            log_queue.put(None, timeout=0.1)
        except queue.Full as exc:
            raise RuntimeError(f"{name} queue stayed full while stopping") from exc

    def join_writer() -> None:
        if writer_thread is None:
            return
        writer_thread.join(timeout=max(0.0, float(join_timeout_sec)))
        if writer_thread.is_alive():
            raise RuntimeError(
                f"{name} writer did not stop within {float(join_timeout_sec):.3f}s"
            )

    run_best_effort_cleanup(
        f"{name}.stop",
        (
            ("stop_signal", send_stop_signal),
            ("worker.join", join_writer),
        ),
    )


class DebugTimingLogRecorder:
    def __init__(
        self,
        *,
        root_dir: Path | str | None = None,
        max_queue_size: int = 512,
        heartbeat_sec: float = 1.0,
        sample_hz: float = 0.0,
    ) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else DEFAULT_DEBUG_TIMING_LOG_ROOT
        self.path: Path | None = None
        self._queue: queue.Queue[Mapping[str, object] | None] | None = None
        self._writer_thread: threading.Thread | None = None
        self._max_queue_size = max(16, int(max_queue_size))
        self.heartbeat_sec = max(0.05, float(heartbeat_sec))
        self.sample_hz = max(0.0, float(sample_hz))
        self.frame_count = 0
        self.dropped_frame_count = 0
        self.dropped_event_count = 0
        self._last_logged_at = 0.0
        self._last_state: str | None = None
        self._last_fault_signature: tuple[str, ...] | None = None
        self._last_sampled_at = 0.0

    @property
    def active(self) -> bool:
        return self._queue is not None

    def start(self, *, now: float | None = None) -> Path:
        if self.active and self.path is not None:
            return self.path
        timestamp_time = time.time() if now is None else float(now)
        timestamp = (
            time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp_time))
            + f"_{int((timestamp_time % 1.0) * 1000):03d}"
        )
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.root_dir / f"debug_timing_{timestamp}.jsonl"
        log_file = self.path.open("a", encoding="utf-8")
        log_queue: queue.Queue[Mapping[str, object] | None] = queue.Queue(maxsize=self._max_queue_size)
        self._queue = log_queue
        self.frame_count = 0
        self.dropped_frame_count = 0
        self.dropped_event_count = 0
        self._last_logged_at = 0.0
        self._last_state = None
        self._last_fault_signature = None
        self._last_sampled_at = 0.0
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            args=(log_queue, log_file, self.path),
            daemon=True,
            name="debug_timing_log_writer",
        )
        self._writer_thread.start()
        self.write_event(
            "start",
            {
                "path": str(self.path),
                "heartbeat_sec": self.heartbeat_sec,
                "sample_hz": self.sample_hz,
            },
            now=now,
        )
        return self.path

    def _writer_loop(
        self,
        log_queue: queue.Queue[Mapping[str, object] | None],
        log_file,
        path: Path,
    ) -> None:
        pending_frames = 0
        try:
            while True:
                row = log_queue.get()
                if row is None:
                    break
                log_file.write(json.dumps(json_safe(row), ensure_ascii=False, separators=(",", ":")) + "\n")
                if row.get("event") == "frame":
                    pending_frames += 1
                    if pending_frames % 10 == 0:
                        log_file.flush()
                else:
                    log_file.flush()
            log_file.flush()
        except Exception as exc:  # noqa: BLE001 - diagnostics must not break teleop.
            print(f"WARNING: debug timing log writer failed for {path}: {type(exc).__name__}: {exc}")
        finally:
            try:
                log_file.close()
            except Exception:
                pass

    def select_frame_reason(
        self,
        *,
        now: float,
        elapsed_sec: float,
        phases: Sequence[tuple[str, float]],
        state: object,
        fault_codes: Sequence[object],
        threshold_sec: float,
        force: bool = False,
    ) -> str | None:
        if not self.active:
            return None
        current_state = state_value(state)
        fault_signature = tuple(str(item) for item in fault_codes)
        reasons: list[str] = []
        if should_report_debug_frozen(
            elapsed_sec=elapsed_sec,
            phases=phases,
            threshold_sec=float(threshold_sec),
            force=force,
        ):
            reasons.append("slow")
        if self._last_state != current_state:
            reasons.append("state_change")
        if self._last_fault_signature is not None and self._last_fault_signature != fault_signature:
            reasons.append("fault_change" if fault_signature else "fault_clear")
        elif self._last_fault_signature is None and fault_signature:
            reasons.append("fault_change")
        if (
            not reasons
            and self._last_logged_at > 0.0
            and (float(now) - self._last_logged_at) >= self.heartbeat_sec
        ):
            reasons.append("heartbeat")
        if not reasons:
            return None
        transition_reason = any(
            reason in {"state_change", "fault_change", "fault_clear"}
            for reason in reasons
        )
        if not transition_reason and self.sample_hz > 0.0:
            sample_period_sec = 1.0 / self.sample_hz
            if (
                self._last_sampled_at > 0.0
                and float(now) - self._last_sampled_at < sample_period_sec
            ):
                return None
        self._last_sampled_at = float(now)
        self._last_logged_at = float(now)
        self._last_state = current_state
        self._last_fault_signature = fault_signature
        return "+".join(reasons)

    def _enqueue_row(self, row: Mapping[str, object], *, kind: str) -> bool:
        log_queue = self._queue
        if log_queue is None:
            return False
        try:
            log_queue.put_nowait(dict(row))
            return True
        except queue.Full:
            if kind == "frame":
                self.dropped_frame_count += 1
            else:
                self.dropped_event_count += 1
            return False

    def write_event(
        self,
        event: str,
        payload: Mapping[str, object] | None = None,
        *,
        now: float | None = None,
    ) -> None:
        if self._queue is None:
            return
        row = {
            "schema": "dexteleop_debug_timing_v1",
            "event": str(event),
            "timestamp": float(time.time() if now is None else now),
            "payload": dict(payload or {}),
        }
        try:
            self._queue.put(row, timeout=0.02)
        except queue.Full:
            self.dropped_event_count += 1

    def record_frame(self, payload: Mapping[str, object]) -> None:
        if self._queue is None:
            return
        if self._enqueue_row(payload, kind="frame"):
            self.frame_count += 1

    def stop(
        self,
        *,
        reason: str,
        now: float | None = None,
        join_timeout_sec: float = 0.5,
    ) -> Path | None:
        log_queue = self._queue
        if log_queue is None:
            return self.path
        stop_row = {
            "schema": "dexteleop_debug_timing_v1",
            "event": "stop",
            "timestamp": float(time.time() if now is None else now),
            "payload": {
                "reason": str(reason),
                "frame_count": int(self.frame_count),
                "dropped_frame_count": int(self.dropped_frame_count),
                "dropped_event_count": int(self.dropped_event_count),
            },
        }
        try:
            log_queue.put(stop_row, timeout=0.1)
        except queue.Full:
            self.dropped_event_count += 1
        writer_thread = self._writer_thread
        try:
            stop_debug_writer_thread(
                log_queue=log_queue,
                writer_thread=writer_thread,
                join_timeout_sec=join_timeout_sec,
                name="debug_timing_log",
            )
        finally:
            self._queue = None
            if writer_thread is None or not writer_thread.is_alive():
                self._writer_thread = None
        return self.path

    def close(self) -> None:
        self.stop(reason="close")


class ActionDebugLogRecorder:
    def __init__(
        self,
        *,
        root_dir: Path | str | None = None,
        max_queue_size: int = 2048,
        flush_every: int = 1,
        sample_hz: float = 0.0,
    ) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else DEFAULT_ACTION_DEBUG_LOG_ROOT
        self.path: Path | None = None
        self._queue: queue.Queue[Mapping[str, object] | None] | None = None
        self._writer_thread: threading.Thread | None = None
        self._max_queue_size = max(16, int(max_queue_size))
        self.flush_every = max(1, int(flush_every))
        self.sample_hz = max(0.0, float(sample_hz))
        self.frame_count = 0
        self.dropped_frame_count = 0
        self.dropped_event_count = 0
        self._last_sampled_at = 0.0

    @property
    def active(self) -> bool:
        return self._queue is not None

    def start(self, *, now: float | None = None) -> Path:
        if self.active and self.path is not None:
            return self.path
        timestamp_time = time.time() if now is None else float(now)
        timestamp = (
            time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp_time))
            + f"_{int((timestamp_time % 1.0) * 1000):03d}"
        )
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.root_dir / f"action_debug_{timestamp}.jsonl"
        log_file = self.path.open("a", encoding="utf-8")
        log_queue: queue.Queue[Mapping[str, object] | None] = queue.Queue(maxsize=self._max_queue_size)
        self._queue = log_queue
        self.frame_count = 0
        self.dropped_frame_count = 0
        self.dropped_event_count = 0
        self._last_sampled_at = 0.0
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            args=(log_queue, log_file, self.path),
            daemon=True,
            name="action_debug_log_writer",
        )
        self._writer_thread.start()
        self.write_event(
            "start",
            {
                "path": str(self.path),
                "flush_every": int(self.flush_every),
                "max_queue_size": int(self._max_queue_size),
                "sample_hz": self.sample_hz,
            },
            now=now,
        )
        return self.path

    def should_record_frame(self, *, now: float) -> bool:
        """Gate payload construction as well as background JSON/file work."""

        if self._queue is None:
            return False
        timestamp = float(now)
        if self.sample_hz <= 0.0:
            self._last_sampled_at = timestamp
            return True
        sample_period_sec = 1.0 / self.sample_hz
        if (
            self._last_sampled_at > 0.0
            and timestamp - self._last_sampled_at < sample_period_sec
        ):
            return False
        self._last_sampled_at = timestamp
        return True

    def _writer_loop(
        self,
        log_queue: queue.Queue[Mapping[str, object] | None],
        log_file,
        path: Path,
    ) -> None:
        pending_frames = 0
        try:
            while True:
                row = log_queue.get()
                if row is None:
                    break
                log_file.write(json.dumps(json_safe(row), ensure_ascii=False, separators=(",", ":")) + "\n")
                if row.get("event") == "frame":
                    pending_frames += 1
                    if pending_frames % self.flush_every == 0:
                        log_file.flush()
                else:
                    log_file.flush()
            log_file.flush()
        except Exception as exc:  # noqa: BLE001 - diagnostics must not break teleop.
            print(f"WARNING: action debug log writer failed for {path}: {type(exc).__name__}: {exc}")
        finally:
            try:
                log_file.close()
            except Exception:
                pass

    def _enqueue_row(self, row: Mapping[str, object], *, kind: str) -> bool:
        log_queue = self._queue
        if log_queue is None:
            return False
        try:
            log_queue.put_nowait(dict(row))
            return True
        except queue.Full:
            if kind == "frame":
                self.dropped_frame_count += 1
            else:
                self.dropped_event_count += 1
            return False

    def write_event(
        self,
        event: str,
        payload: Mapping[str, object] | None = None,
        *,
        now: float | None = None,
    ) -> None:
        if self._queue is None:
            return
        row = {
            "schema": "dexteleop_action_debug_v1",
            "event": str(event),
            "timestamp": float(time.time() if now is None else now),
            "payload": dict(payload or {}),
        }
        try:
            self._queue.put(row, timeout=0.02)
        except queue.Full:
            self.dropped_event_count += 1

    def record_frame(self, payload: Mapping[str, object]) -> None:
        if self._queue is None:
            return
        if self._enqueue_row(payload, kind="frame"):
            self.frame_count += 1

    def stop(
        self,
        *,
        reason: str,
        now: float | None = None,
        join_timeout_sec: float = 0.5,
    ) -> Path | None:
        log_queue = self._queue
        if log_queue is None:
            return self.path
        stop_row = {
            "schema": "dexteleop_action_debug_v1",
            "event": "stop",
            "timestamp": float(time.time() if now is None else now),
            "payload": {
                "reason": str(reason),
                "frame_count": int(self.frame_count),
                "dropped_frame_count": int(self.dropped_frame_count),
                "dropped_event_count": int(self.dropped_event_count),
            },
        }
        try:
            log_queue.put(stop_row, timeout=0.1)
        except queue.Full:
            self.dropped_event_count += 1
        writer_thread = self._writer_thread
        try:
            stop_debug_writer_thread(
                log_queue=log_queue,
                writer_thread=writer_thread,
                join_timeout_sec=join_timeout_sec,
                name="action_debug_log",
            )
        finally:
            self._queue = None
            if writer_thread is None or not writer_thread.is_alive():
                self._writer_thread = None
        return self.path

    def close(self) -> None:
        self.stop(reason="close")


class IkDebugLogRecorder:
    def __init__(self, *, root_dir: Path | str | None = None, max_queue_size: int = 256) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else DEFAULT_IK_DEBUG_LOG_ROOT
        self.path: Path | None = None
        self._queue: queue.Queue[Mapping[str, object] | None] | None = None
        self._writer_thread: threading.Thread | None = None
        self._max_queue_size = max(16, int(max_queue_size))
        self.frame_count = 0
        self.dropped_frame_count = 0

    @property
    def active(self) -> bool:
        return self._queue is not None

    def start(self, *, now: float | None = None) -> Path:
        if self.active and self.path is not None:
            return self.path
        timestamp_time = time.time() if now is None else float(now)
        timestamp = (
            time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp_time))
            + f"_{int((timestamp_time % 1.0) * 1000):03d}"
        )
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.root_dir / f"ik_debug_{timestamp}.jsonl"
        log_file = self.path.open("a", encoding="utf-8")
        log_queue: queue.Queue[Mapping[str, object] | None] = queue.Queue(maxsize=self._max_queue_size)
        self._queue = log_queue
        self.frame_count = 0
        self.dropped_frame_count = 0
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            args=(log_queue, log_file, self.path),
            daemon=True,
            name="ik_debug_log_writer",
        )
        self._writer_thread.start()
        self.write_event("start", {"path": str(self.path)}, now=now)
        return self.path

    def _writer_loop(
        self,
        log_queue: queue.Queue[Mapping[str, object] | None],
        log_file,
        path: Path,
    ) -> None:
        pending_frames = 0
        try:
            while True:
                row = log_queue.get()
                if row is None:
                    break
                log_file.write(json.dumps(json_safe(row), ensure_ascii=False, separators=(",", ":")) + "\n")
                if row.get("event") == "frame":
                    pending_frames += 1
                    if pending_frames % 20 == 0:
                        log_file.flush()
                else:
                    log_file.flush()
            log_file.flush()
        except Exception as exc:  # noqa: BLE001 - logging must not break teleop.
            print(f"WARNING: IK debug log writer failed for {path}: {type(exc).__name__}: {exc}")
        finally:
            try:
                log_file.close()
            except Exception:
                pass

    def _enqueue_row(self, row: Mapping[str, object], *, kind: str) -> bool:
        log_queue = self._queue
        if log_queue is None:
            return False
        try:
            log_queue.put_nowait(dict(row))
            return True
        except queue.Full:
            if kind == "frame":
                self.dropped_frame_count += 1
            return False

    def write_event(
        self,
        event: str,
        payload: Mapping[str, object] | None = None,
        *,
        now: float | None = None,
    ) -> None:
        if self._queue is None:
            return
        row = {
            "schema": "dexteleop_ik_debug_v1",
            "event": str(event),
            "timestamp": float(time.time() if now is None else now),
            "payload": dict(payload or {}),
        }
        try:
            self._queue.put(row, timeout=0.02)
        except queue.Full:
            self.dropped_frame_count += 1

    def record_frame(self, payload: Mapping[str, object]) -> None:
        if self._queue is None:
            return
        if self._enqueue_row(payload, kind="frame"):
            self.frame_count += 1

    def stop(
        self,
        *,
        reason: str,
        now: float | None = None,
        join_timeout_sec: float = 0.5,
    ) -> Path | None:
        log_queue = self._queue
        if log_queue is None:
            return self.path
        stop_row = {
            "schema": "dexteleop_ik_debug_v1",
            "event": "stop",
            "timestamp": float(time.time() if now is None else now),
            "payload": {
                "reason": str(reason),
                "frame_count": int(self.frame_count),
                "dropped_frame_count": int(self.dropped_frame_count),
            },
        }
        try:
            log_queue.put(stop_row, timeout=0.1)
        except queue.Full:
            self.dropped_frame_count += 1
        writer_thread = self._writer_thread
        try:
            stop_debug_writer_thread(
                log_queue=log_queue,
                writer_thread=writer_thread,
                join_timeout_sec=join_timeout_sec,
                name="ik_debug_log",
            )
        finally:
            self._queue = None
            if writer_thread is None or not writer_thread.is_alive():
                self._writer_thread = None
        return self.path

    def close(self) -> None:
        self.stop(reason="close")


__all__ = [
    "ActionDebugLogRecorder",
    "DebugTimingLogRecorder",
    "DEFAULT_ACTION_DEBUG_LOG_ROOT",
    "DEFAULT_DEBUG_TIMING_LOG_ROOT",
    "DEFAULT_IK_DEBUG_LOG_ROOT",
    "IkDebugLogRecorder",
    "json_safe",
    "state_value",
    "stop_debug_writer_thread",
]

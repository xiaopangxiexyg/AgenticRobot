"""Asynchronous sidecar recorder for real position-control diagnostics."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import copy
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, IO, Mapping

from robot_motion.position_control.audit import (
    POSITION_AUDIT_SCHEMA_VERSION,
    PositionAuditEvent,
    PositionAuditMode,
    PositionAuditStatistics,
    PositionCommandAuditEvent,
    PositionReferenceAuditEvent,
    PositionTimingAuditEvent,
    normalize_position_audit_mode,
    position_audit_event_to_dict,
)


POSITION_REFERENCE_SIDECAR_NAME = "position_reference.jsonl"
POSITION_COMMAND_SIDECAR_NAME = "position_command_150.jsonl"
POSITION_TIMING_SIDECAR_NAME = "position_timing.jsonl"
POSITION_SUMMARY_NAME = "position_summary.json"


@dataclass(frozen=True)
class PositionDebugRecorderConfig:
    """Storage and backpressure policy for position-control audit data."""

    output_dir: str | Path | None
    mode: PositionAuditMode | str = PositionAuditMode.OFF
    queue_capacity: int = 16_384
    max_pending_bind_events: int = 8_192
    max_target_bindings: int = 16_384
    max_samples_per_metric: int = 100_000
    flush_every_events: int = 128
    summary_checkpoint_every_events: int = 2_048
    thread_name: str = "position-debug-recorder"
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mode = normalize_position_audit_mode(self.mode)
        object.__setattr__(self, "mode", mode)
        if mode is not PositionAuditMode.OFF and self.output_dir is None:
            raise ValueError("output_dir is required when position debug mode is not off.")
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        for name in (
            "queue_capacity",
            "max_pending_bind_events",
            "max_target_bindings",
            "max_samples_per_metric",
            "flush_every_events",
            "summary_checkpoint_every_events",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
            object.__setattr__(self, name, value)
        thread_name = str(self.thread_name).strip()
        if not thread_name:
            raise ValueError("thread_name must not be empty.")
        object.__setattr__(self, "thread_name", thread_name)
        if not isinstance(self.context, Mapping):
            raise TypeError("context must be a mapping")
        object.__setattr__(self, "context", copy.deepcopy(dict(self.context)))

    @classmethod
    def from_control_debug_config(
        cls,
        control_debug: Any,
        *,
        output_dir: str | Path | None = None,
        thread_name: str = "position-debug-recorder",
        context: Mapping[str, Any] | None = None,
    ) -> "PositionDebugRecorderConfig":
        """Adapt the CLI-facing ``ControlDebugConfig`` without coupling layers."""

        resolved_output = (
            output_dir
            if output_dir is not None
            else getattr(control_debug, "root", None)
        )
        return cls(
            output_dir=resolved_output,
            mode=getattr(control_debug, "level", PositionAuditMode.OFF.value),
            queue_capacity=int(getattr(control_debug, "queue_size", 16_384)),
            flush_every_events=int(getattr(control_debug, "flush_every", 128)),
            thread_name=thread_name,
            context={} if context is None else dict(context),
        )


class PositionDebugRecorder:
    """Bounded, non-blocking audit sink with late reference binding.

    Producers only call ``put_nowait``.  A command tick may race ahead of the
    publisher that associates its transport target with a high-rate reference; in
    full mode the writer holds that row in a second bounded buffer and patches
    ``reference_epoch/reference_sequence`` when the reference event arrives.
    Both buffers have explicit drop/eviction counters.
    """

    _STOP = object()

    def __init__(
        self,
        config: PositionDebugRecorderConfig,
        *,
        auto_start: bool = True,
    ) -> None:
        self.config = config
        self._queue: queue.Queue[PositionAuditEvent | object] = queue.Queue(
            maxsize=int(config.queue_capacity)
        )
        self._stop_requested = threading.Event()
        self._stats = PositionAuditStatistics(
            max_samples_per_metric=int(config.max_samples_per_metric)
        )
        self._lifecycle_lock = threading.RLock()
        self._counter_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._started = False
        self._accepting = False
        self._closed = False
        self._writer_error: str | None = None
        self._started_wall_time: float | None = None
        self._closed_wall_time: float | None = None
        self._enqueued_count = 0
        self._enqueue_drop_count = 0
        self._enqueue_after_close_count = 0
        self._queue_high_watermark = 0
        self._processed_count = 0
        self._pending_binding_eviction_count = 0
        self._unbound_at_close_count = 0
        self._last_reference_monotonic: float | None = None
        self._last_tick_wakeup_by_side: dict[str, float] = {}
        self._bindings: OrderedDict[tuple[str, int], tuple[int, int]] = OrderedDict()
        self._pending_commands: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._pending_ids_by_key: dict[tuple[str, int], deque[int]] = defaultdict(deque)
        self._next_pending_id = 0
        self._files: dict[str, IO[str]] = {}
        self._writes_since_flush = 0
        if auto_start:
            self.start()

    @property
    def mode(self) -> PositionAuditMode:
        return normalize_position_audit_mode(self.config.mode)

    @property
    def enabled(self) -> bool:
        return self.mode is not PositionAuditMode.OFF

    @property
    def output_dir(self) -> Path | None:
        value = self.config.output_dir
        return None if value is None else Path(value)

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._started or self._closed:
                return
            self._started = True
            self._started_wall_time = time.time()
            if not self.enabled:
                self._closed = True
                self._closed_wall_time = self._started_wall_time
                return
            output_dir = self.output_dir
            assert output_dir is not None
            output_dir.mkdir(parents=True, exist_ok=True)
            self._accepting = True
            self._thread = threading.Thread(
                target=self._writer_main,
                name=str(self.config.thread_name),
                daemon=True,
            )
            self._thread.start()

    def __call__(self, event: PositionAuditEvent) -> None:
        self.record(event)

    def record(self, event: PositionAuditEvent) -> bool:
        """Enqueue without waiting; return ``False`` when audit data is dropped."""

        if not isinstance(
            event,
            (PositionReferenceAuditEvent, PositionCommandAuditEvent, PositionTimingAuditEvent),
        ):
            raise TypeError(f"Unsupported position audit event: {type(event).__name__}")
        if not self.enabled:
            return False
        if not self._started:
            # Normally start() is called during setup (auto_start defaults to
            # true).  Refuse to start a thread from a 150 Hz producer.
            with self._counter_lock:
                self._enqueue_drop_count += 1
            return False
        if not self._accepting:
            with self._counter_lock:
                self._enqueue_after_close_count += 1
            return False
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            with self._counter_lock:
                self._enqueue_drop_count += 1
            return False
        with self._counter_lock:
            self._enqueued_count += 1
            self._queue_high_watermark = max(
                self._queue_high_watermark,
                self._queue.qsize(),
            )
        return True

    record_event = record

    def close(self, *, timeout_sec: float = 5.0) -> None:
        timeout = max(0.0, float(timeout_sec))
        with self._lifecycle_lock:
            if self._closed:
                return
            self._accepting = False
            thread = self._thread
            if thread is None or not thread.is_alive():
                self._closed = True
                self._closed_wall_time = time.time()
                return
        deadline = time.monotonic() + timeout
        self._stop_requested.set()
        try:
            # Wakes an idle consumer.  If the queue is full, the event above
            # still makes the writer exit immediately after draining it.
            self._queue.put_nowait(self._STOP)
        except queue.Full:
            pass
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if thread.is_alive() and self._writer_error is None:
            self._writer_error = "close timeout while waiting for audit writer"
        with self._lifecycle_lock:
            self._closed = not thread.is_alive()
            if self._closed_wall_time is None:
                self._closed_wall_time = time.time()

    def snapshot(self) -> dict[str, Any]:
        with self._counter_lock:
            producer = {
                "enqueued_count": int(self._enqueued_count),
                "enqueue_drop_count": int(self._enqueue_drop_count),
                "enqueue_after_close_count": int(self._enqueue_after_close_count),
                "queue_high_watermark": int(self._queue_high_watermark),
                "queue_capacity": int(self.config.queue_capacity),
                "max_pending_bind_events": int(self.config.max_pending_bind_events),
                "max_target_bindings": int(self.config.max_target_bindings),
                "flush_every_events": int(self.config.flush_every_events),
                "summary_checkpoint_every_events": int(
                    self.config.summary_checkpoint_every_events
                ),
            }
        summary = {
            "schema_version": POSITION_AUDIT_SCHEMA_VERSION,
            "mode": self.mode.value,
            "enabled": bool(self.enabled),
            "started": bool(self._started),
            "accepting": bool(self._accepting),
            "closed": bool(self._closed),
            "started_wall_time": self._started_wall_time,
            "closed_wall_time": self._closed_wall_time,
            "writer_error": self._writer_error,
            "context": copy.deepcopy(dict(self.config.context)),
            **producer,
            "processed_count": int(self._processed_count),
            "pending_binding_count": len(self._pending_commands),
            "pending_binding_eviction_count": int(self._pending_binding_eviction_count),
            "unbound_at_close_count": int(self._unbound_at_close_count),
            "target_binding_count": len(self._bindings),
            "statistics": self._stats.snapshot(),
            "files": self._file_manifest(),
        }
        return summary

    def __enter__(self) -> "PositionDebugRecorder":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()

    def _writer_main(self) -> None:
        try:
            self._open_files()
            while True:
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    if self._stop_requested.is_set():
                        break
                    continue
                try:
                    if item is self._STOP:
                        # Do not let an in-flight producer that observed the
                        # pre-close accepting flag land behind the sentinel.
                        # The stop event plus an empty-queue timeout is the
                        # authoritative termination condition.
                        continue
                    assert isinstance(
                        item,
                        (PositionReferenceAuditEvent, PositionCommandAuditEvent, PositionTimingAuditEvent),
                    )
                    self._process_event(item)
                    self._processed_count += 1
                    if (
                        self._processed_count
                        % int(self.config.summary_checkpoint_every_events)
                        == 0
                    ):
                        self._write_summary_file()
                finally:
                    self._queue.task_done()
            self._flush_pending_commands(binding_status="unbound_at_close")
            self._flush_files()
        except Exception as exc:  # noqa: BLE001 - audit must fail isolated from control.
            self._writer_error = f"{type(exc).__name__}: {exc}"
            self._accepting = False
        finally:
            self._closed_wall_time = time.time()
            self._closed = True
            try:
                self._write_summary_file()
            except Exception as exc:  # noqa: BLE001 - preserve the first useful failure.
                if self._writer_error is None:
                    self._writer_error = f"summary {type(exc).__name__}: {exc}"
            self._close_files()

    def _open_files(self) -> None:
        if self.mode is not PositionAuditMode.FULL:
            return
        output_dir = self.output_dir
        assert output_dir is not None
        for event_type, name in (
            ("position_reference", POSITION_REFERENCE_SIDECAR_NAME),
            ("position_command_150", POSITION_COMMAND_SIDECAR_NAME),
            ("position_timing", POSITION_TIMING_SIDECAR_NAME),
        ):
            self._files[event_type] = (output_dir / name).open("w", encoding="utf-8")

    def _process_event(self, event: PositionAuditEvent) -> None:
        self._observe_event(event)
        if isinstance(event, PositionReferenceAuditEvent):
            self._process_reference(event)
        elif isinstance(event, PositionCommandAuditEvent):
            self._process_command(event)
        else:
            self._write_row(event.event_type, position_audit_event_to_dict(event))

    def _process_reference(self, event: PositionReferenceAuditEvent) -> None:
        self._write_row(event.event_type, position_audit_event_to_dict(event))
        for side, target_sequence in event.transport_target_sequences.items():
            key = (str(side), int(target_sequence))
            self._bindings[key] = (int(event.epoch), int(event.reference_sequence))
            self._bindings.move_to_end(key)
            pending_ids = self._pending_ids_by_key.pop(key, deque())
            while pending_ids:
                pending_id = pending_ids.popleft()
                row = self._pending_commands.pop(pending_id, None)
                if row is None:
                    continue
                self._bind_command_row(
                    row,
                    epoch=event.epoch,
                    reference_sequence=event.reference_sequence,
                    binding_status="late_bound",
                )
                self._write_row("position_command_150", row)
                self._stats.increment("command_reference_late_bound_count")
        while len(self._bindings) > int(self.config.max_target_bindings):
            self._bindings.popitem(last=False)

    def _process_command(self, event: PositionCommandAuditEvent) -> None:
        row = position_audit_event_to_dict(event)
        if event.reference_epoch is not None and event.reference_sequence is not None:
            self._bind_command_row(
                row,
                epoch=event.reference_epoch,
                reference_sequence=event.reference_sequence,
                binding_status="emitted_bound",
            )
            self._write_row(event.event_type, row)
            return
        if event.target_sequence is None:
            row["reference_binding_status"] = "no_transport_target"
            self._write_row(event.event_type, row)
            return
        key = (str(event.side), int(event.target_sequence))
        binding = self._bindings.get(key)
        if binding is not None:
            self._bindings.move_to_end(key)
            self._bind_command_row(
                row,
                epoch=binding[0],
                reference_sequence=binding[1],
                binding_status="bound",
            )
            self._write_row(event.event_type, row)
            return
        if self.mode is not PositionAuditMode.FULL:
            # Summary mode needs no row retention; the association counters are
            # still represented by the reference/target event totals.
            return
        self._next_pending_id += 1
        pending_id = int(self._next_pending_id)
        self._pending_commands[pending_id] = row
        self._pending_ids_by_key[key].append(pending_id)
        while len(self._pending_commands) > int(self.config.max_pending_bind_events):
            self._evict_oldest_pending()

    def _evict_oldest_pending(self) -> None:
        pending_id, row = self._pending_commands.popitem(last=False)
        side = str(row.get("side"))
        target_sequence = row.get("target_sequence")
        if target_sequence is not None:
            key = (side, int(target_sequence))
            ids = self._pending_ids_by_key.get(key)
            if ids is not None:
                try:
                    ids.remove(pending_id)
                except ValueError:
                    pass
                if not ids:
                    self._pending_ids_by_key.pop(key, None)
        row["reference_binding_status"] = "pending_buffer_evicted"
        self._write_row("position_command_150", row)
        self._pending_binding_eviction_count += 1
        self._stats.increment("command_reference_binding_eviction_count")

    def _flush_pending_commands(self, *, binding_status: str) -> None:
        while self._pending_commands:
            _pending_id, row = self._pending_commands.popitem(last=False)
            row["reference_binding_status"] = str(binding_status)
            self._write_row("position_command_150", row)
            self._unbound_at_close_count += 1
            self._stats.increment("command_reference_unbound_at_close_count")
        self._pending_ids_by_key.clear()

    @staticmethod
    def _bind_command_row(
        row: dict[str, Any],
        *,
        epoch: int,
        reference_sequence: int,
        binding_status: str,
    ) -> None:
        row["reference_epoch"] = int(epoch)
        row["reference_sequence"] = int(reference_sequence)
        row["reference_binding_status"] = str(binding_status)

    def _observe_event(self, event: PositionAuditEvent) -> None:
        if isinstance(event, PositionReferenceAuditEvent):
            self._stats.increment("reference_event_count")
            if event.references and all(reference.valid for reference in event.references.values()):
                self._stats.increment("reference_valid_event_count")
            if any(reference.hold for reference in event.references.values()):
                self._stats.increment("reference_hold_event_count")
            previous = self._last_reference_monotonic
            if previous is not None:
                gap_ms = max(0.0, event.published_monotonic_time - previous) * 1000.0
                self._stats.observe("reference_publish_gap_ms", gap_ms)
                if gap_ms > 0.0:
                    self._stats.observe("reference_publish_hz", 1000.0 / gap_ms)
            self._last_reference_monotonic = event.published_monotonic_time
            return
        if isinstance(event, PositionCommandAuditEvent):
            side = str(event.side)
            command_side_prefix = f"command.{side}"
            sdk_side_prefix = f"sdk.{side}"
            self._stats.increment("command_tick_count")
            self._stats.increment("command_skipped_tick_count", event.skipped_tick_count)
            self._stats.increment(f"{command_side_prefix}.tick_count")
            self._stats.increment(
                f"{command_side_prefix}.skipped_tick_count",
                event.skipped_tick_count,
            )
            if event.deadline_missed:
                self._stats.increment("command_deadline_miss_count")
                self._stats.increment(f"{command_side_prefix}.deadline_miss_count")
            if not event.target_active:
                self._stats.increment("command_idle_tick_count")
                self._stats.increment(f"{command_side_prefix}.idle_tick_count")
            if event.stale_target_expired:
                self._stats.increment("command_stale_target_expired_count")
                self._stats.increment(
                    f"{command_side_prefix}.stale_target_expired_count"
                )
            if event.sdk_attempted:
                self._stats.increment("sdk_attempt_count")
                self._stats.increment(f"{sdk_side_prefix}.attempt_count")
            if event.sdk_accepted is True:
                self._stats.increment("sdk_accepted_count")
                self._stats.increment(f"{sdk_side_prefix}.accepted_count")
            elif event.sdk_accepted is False:
                self._stats.increment("sdk_failed_count")
                self._stats.increment(f"{sdk_side_prefix}.failed_count")
            self._stats.observe("command_wakeup_lag_ms", event.wakeup_lag_sec * 1000.0)
            self._stats.observe("command_raw_wakeup_lag_ms", event.raw_wakeup_lag_sec * 1000.0)
            self._stats.observe(
                "command_planner_lock_delay_ms",
                None
                if event.planner_lock_delay_sec is None
                else event.planner_lock_delay_sec * 1000.0,
            )
            self._stats.observe(
                f"{command_side_prefix}.wakeup_lag_ms",
                event.wakeup_lag_sec * 1000.0,
            )
            self._stats.observe(
                f"{command_side_prefix}.raw_wakeup_lag_ms",
                event.raw_wakeup_lag_sec * 1000.0,
            )
            self._stats.observe(
                f"{command_side_prefix}.planner_lock_delay_ms",
                None
                if event.planner_lock_delay_sec is None
                else event.planner_lock_delay_sec * 1000.0,
            )
            self._stats.observe(
                "command_wakeup_gap_ms",
                None if event.wakeup_gap_sec is None else event.wakeup_gap_sec * 1000.0,
            )
            self._stats.observe(
                "sdk_send_gap_ms",
                None if event.sdk_send_gap_sec is None else event.sdk_send_gap_sec * 1000.0,
            )
            self._stats.observe("sdk_call_ms", event.sdk_call_ms)
            self._stats.observe(f"{sdk_side_prefix}.call_ms", event.sdk_call_ms)
            self._stats.observe(
                "target_age_ms",
                None
                if event.target_age_sec is None
                else event.target_age_sec * 1000.0,
            )
            self._stats.observe(
                f"{command_side_prefix}.target_age_ms",
                None
                if event.target_age_sec is None
                else event.target_age_sec * 1000.0,
            )
            previous_wakeup = self._last_tick_wakeup_by_side.get(event.side)
            effective_wakeup_gap_sec = event.wakeup_gap_sec
            if event.wakeup_gap_sec is None and previous_wakeup is not None:
                effective_wakeup_gap_sec = max(
                    0.0,
                    event.wakeup_monotonic_time - previous_wakeup,
                )
                self._stats.observe(
                    "command_derived_wakeup_gap_ms",
                    effective_wakeup_gap_sec * 1000.0,
                )
            if effective_wakeup_gap_sec is not None:
                self._stats.observe(
                    f"{command_side_prefix}.wakeup_gap_ms",
                    effective_wakeup_gap_sec * 1000.0,
                )
                if effective_wakeup_gap_sec > 0.0:
                    actual_hz = 1.0 / effective_wakeup_gap_sec
                    self._stats.observe("command_wakeup_actual_hz", actual_hz)
                    self._stats.observe(
                        f"{command_side_prefix}.wakeup_actual_hz",
                        actual_hz,
                    )
            if event.sdk_send_gap_sec is not None:
                self._stats.observe(
                    f"{sdk_side_prefix}.send_gap_ms",
                    event.sdk_send_gap_sec * 1000.0,
                )
                if event.sdk_send_gap_sec > 0.0:
                    actual_hz = 1.0 / event.sdk_send_gap_sec
                    self._stats.observe("sdk_send_actual_hz", actual_hz)
                    self._stats.observe(
                        f"{sdk_side_prefix}.send_actual_hz",
                        actual_hz,
                    )
            self._last_tick_wakeup_by_side[event.side] = event.wakeup_monotonic_time
            return
        self._stats.increment("timing_event_count")
        stage_prefix = f"timing.{event.stage}"
        self._stats.increment(f"{stage_prefix}.count")
        self._stats.observe(f"{stage_prefix}.elapsed_ms", event.elapsed_ms)
        side_stage_prefix = None
        if event.side is not None:
            side_stage_prefix = f"timing.{event.side}.{event.stage}"
            self._stats.increment(f"{side_stage_prefix}.count")
            self._stats.observe(f"{side_stage_prefix}.elapsed_ms", event.elapsed_ms)
        if event.deadline_missed:
            self._stats.increment(f"{stage_prefix}.deadline_miss_count")
            if side_stage_prefix is not None:
                self._stats.increment(f"{side_stage_prefix}.deadline_miss_count")
        if bool(event.metadata.get("fault_latched", False)):
            self._stats.increment(f"{stage_prefix}.fault_count")
            if side_stage_prefix is not None:
                self._stats.increment(f"{side_stage_prefix}.fault_count")

    def _write_row(self, event_type: str, row: dict[str, Any]) -> None:
        if self.mode is not PositionAuditMode.FULL:
            return
        handle = self._files.get(str(event_type))
        if handle is None:
            raise RuntimeError(f"No sidecar handle for event type {event_type!r}.")
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
        handle.write("\n")
        self._writes_since_flush += 1
        if self._writes_since_flush >= int(self.config.flush_every_events):
            self._flush_files()

    def _flush_files(self) -> None:
        for handle in self._files.values():
            handle.flush()
        self._writes_since_flush = 0

    def _close_files(self) -> None:
        for handle in self._files.values():
            try:
                handle.close()
            except Exception:
                pass
        self._files.clear()

    def _write_summary_file(self) -> None:
        if not self.enabled:
            return
        output_dir = self.output_dir
        assert output_dir is not None
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / POSITION_SUMMARY_NAME
        temp_path = output_dir / f".{POSITION_SUMMARY_NAME}.tmp"
        temp_path.write_text(
            json.dumps(self.snapshot(), ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, summary_path)

    def _file_manifest(self) -> dict[str, str | None]:
        output_dir = self.output_dir
        if output_dir is None:
            return {
                "position_reference": None,
                "position_command_150": None,
                "position_timing": None,
                "summary": None,
            }
        return {
            "position_reference": (
                str(output_dir / POSITION_REFERENCE_SIDECAR_NAME)
                if self.mode is PositionAuditMode.FULL
                else None
            ),
            "position_command_150": (
                str(output_dir / POSITION_COMMAND_SIDECAR_NAME)
                if self.mode is PositionAuditMode.FULL
                else None
            ),
            "position_timing": (
                str(output_dir / POSITION_TIMING_SIDECAR_NAME)
                if self.mode is PositionAuditMode.FULL
                else None
            ),
            "summary": str(output_dir / POSITION_SUMMARY_NAME),
        }


__all__ = [
    "POSITION_COMMAND_SIDECAR_NAME",
    "POSITION_REFERENCE_SIDECAR_NAME",
    "POSITION_SUMMARY_NAME",
    "POSITION_TIMING_SIDECAR_NAME",
    "PositionDebugRecorder",
    "PositionDebugRecorderConfig",
]

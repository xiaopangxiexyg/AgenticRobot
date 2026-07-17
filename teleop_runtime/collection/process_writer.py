"""Spawned formal-collection worker with shared-memory camera transport.

The control/frontend process freezes causal hardware state in a low-priority
thread.  Pure sample assembly, schema conversion, video/tactile handling and
all dataset writes run here in a spawned process, so they cannot hold the
control process GIL.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, replace
from multiprocessing import shared_memory
from typing import Any, Mapping, Sequence

import numpy as np

from .sample_assembler import (
    CollectorSampleBuilder,
    CollectorSampleBuilderConfig,
    CollectorSampleSnapshot,
)
from .video_frame import VideoFrame
from .writer_v21 import LeRobotV21Writer, LeRobotV21WriterConfig


_ALIGNMENT = 64


@dataclass(frozen=True)
class SharedArrayRef:
    slot_id: int
    shm_name: str
    offset: int
    shape: tuple[int, ...]
    dtype: str
    nbytes: int


@dataclass(frozen=True)
class ProcessCollectionConfig:
    queue_size: int = 4
    shared_slot_count: int = 4
    shared_slot_bytes: int = 1
    startup_timeout_sec: float = 10.0
    request_timeout_sec: float = 30.0
    worker_cpus: tuple[int, ...] = ()
    worker_nice: int = 5

    def __post_init__(self) -> None:
        if int(self.queue_size) <= 0:
            raise ValueError("queue_size must be positive.")
        if int(self.shared_slot_count) <= 0:
            raise ValueError("shared_slot_count must be positive.")
        if int(self.shared_slot_bytes) <= 0:
            raise ValueError("shared_slot_bytes must be positive.")
        if float(self.startup_timeout_sec) <= 0.0:
            raise ValueError("startup_timeout_sec must be positive.")
        if float(self.request_timeout_sec) <= 0.0:
            raise ValueError("request_timeout_sec must be positive.")


@dataclass(frozen=True)
class ProcessCollectionStats:
    submitted_samples: int
    completed_samples: int
    rejected_busy: int
    queue_capacity: int
    queue_high_water: int
    slots_available: int
    slot_count: int
    worker_alive: bool
    worker_error: str | None
    active_episode_frame_count: int
    sample_process_ms_p50: float | None
    sample_process_ms_p95: float | None
    timing_ms: dict[str, dict[str, float | int | None]]
    shared_payload_bytes: dict[str, float | int | None]


def camera_shared_slot_bytes(
    camera_sources: Mapping[str, Any],
    *,
    extra_array_bytes: int = 0,
) -> int:
    """Return a fixed slot size for camera plus non-camera ndarray payloads."""

    total = 0
    for source in camera_sources.values():
        config = getattr(source, "config", None)
        width = int(getattr(config, "width", 0) or 0)
        height = int(getattr(config, "height", 0) or 0)
        if width <= 0 or height <= 0:
            continue
        total += width * height * 3
        if bool(getattr(config, "enable_depth", False)):
            depth_width = int(getattr(config, "depth_width", width) or width)
            depth_height = int(getattr(config, "depth_height", height) or height)
            total += depth_width * depth_height * 2
        total += 2 * _ALIGNMENT
    return max(1, int(total) + max(0, int(extra_array_bytes)))


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), float(q)))


def _metric_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


class _SharedSlotPool:
    def __init__(self, *, slot_count: int, slot_bytes: int):
        self.slot_bytes = int(slot_bytes)
        self.blocks = [
            shared_memory.SharedMemory(create=True, size=self.slot_bytes)
            for _ in range(int(slot_count))
        ]
        self.available: deque[int] = deque(range(len(self.blocks)))
        self.in_use: set[int] = set()
        self.closed = False

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(block.name for block in self.blocks)

    def acquire(self) -> int | None:
        if not self.available:
            return None
        slot_id = int(self.available.popleft())
        self.in_use.add(slot_id)
        return slot_id

    def release(self, slot_id: int) -> None:
        slot = int(slot_id)
        if slot not in self.in_use:
            return
        self.in_use.remove(slot)
        self.available.append(slot)

    def pack_snapshot(
        self,
        snapshot: CollectorSampleSnapshot,
        *,
        slot_id: int,
    ) -> tuple[CollectorSampleSnapshot, int]:
        block = self.blocks[int(slot_id)]
        offset = 0

        def pack(array: Any | None) -> Any | None:
            nonlocal offset
            if array is None:
                return None
            contiguous = np.ascontiguousarray(array)
            aligned = (offset + (_ALIGNMENT - 1)) // _ALIGNMENT * _ALIGNMENT
            end = aligned + int(contiguous.nbytes)
            if end > self.slot_bytes:
                raise BufferError(
                    "collection shared-memory slot is too small: "
                    f"required={end} configured={self.slot_bytes}"
                )
            target = np.ndarray(
                contiguous.shape,
                dtype=contiguous.dtype,
                buffer=block.buf,
                offset=aligned,
            )
            np.copyto(target, contiguous, casting="no")
            offset = end
            return SharedArrayRef(
                slot_id=int(slot_id),
                shm_name=block.name,
                offset=int(aligned),
                shape=tuple(int(value) for value in contiguous.shape),
                dtype=contiguous.dtype.str,
                nbytes=int(contiguous.nbytes),
            )

        def pack_nested(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return pack(value)
            if isinstance(value, Mapping):
                return {
                    key: pack_nested(item)
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [pack_nested(item) for item in value]
            if isinstance(value, tuple):
                return tuple(pack_nested(item) for item in value)
            return value

        frames: dict[str, VideoFrame | None] = {}
        for camera_key, frame in snapshot.camera_frames.items():
            if frame is None:
                frames[str(camera_key)] = None
                continue
            frames[str(camera_key)] = VideoFrame(
                image=pack(frame.image),
                source_timestamp=float(frame.source_timestamp),
                receive_timestamp=frame.receive_timestamp,
                available_timestamp=frame.available_timestamp,
                frame_index=frame.frame_index,
                metadata=dict(frame.metadata),
                depth_image=pack(frame.depth_image),
                depth_metadata=dict(frame.depth_metadata),
            )
        packed_snapshot = replace(
            snapshot,
            camera_frames=frames,
            arm_states=pack_nested(snapshot.arm_states),
            hand_states=pack_nested(snapshot.hand_states),
            position_reference=pack_nested(snapshot.position_reference),
        )
        return packed_snapshot, int(offset)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for block in self.blocks:
            try:
                block.close()
            finally:
                try:
                    block.unlink()
                except FileNotFoundError:
                    pass


def _hydrate_snapshot(
    snapshot: CollectorSampleSnapshot,
    blocks: Mapping[str, shared_memory.SharedMemory],
) -> tuple[CollectorSampleSnapshot, int | None]:
    slot_id: int | None = None

    def hydrate(value: Any | None) -> Any | None:
        nonlocal slot_id
        if value is None or not isinstance(value, SharedArrayRef):
            return value
        if slot_id is None:
            slot_id = int(value.slot_id)
        elif slot_id != int(value.slot_id):
            raise RuntimeError("one collection sample referenced multiple shared slots")
        block = blocks[value.shm_name]
        view = np.ndarray(
            value.shape,
            dtype=np.dtype(value.dtype),
            buffer=block.buf,
            offset=int(value.offset),
        )
        if int(view.nbytes) != int(value.nbytes):
            raise RuntimeError("shared array descriptor byte count mismatch")
        # Keep a zero-copy view until build_from_snapshot() creates the
        # child-owned CollectorSample copy. The slot is acknowledged only
        # after that copy, so the parent cannot overwrite this view early.
        return view

    def hydrate_nested(value: Any) -> Any:
        if isinstance(value, SharedArrayRef):
            return hydrate(value)
        if isinstance(value, Mapping):
            return {
                key: hydrate_nested(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [hydrate_nested(item) for item in value]
        if isinstance(value, tuple):
            return tuple(hydrate_nested(item) for item in value)
        return value

    frames: dict[str, VideoFrame | None] = {}
    for camera_key, frame in snapshot.camera_frames.items():
        if frame is None:
            frames[str(camera_key)] = None
            continue
        frames[str(camera_key)] = VideoFrame(
            image=hydrate(frame.image),
            source_timestamp=float(frame.source_timestamp),
            receive_timestamp=frame.receive_timestamp,
            available_timestamp=frame.available_timestamp,
            frame_index=frame.frame_index,
            metadata=dict(frame.metadata),
            depth_image=hydrate(frame.depth_image),
            depth_metadata=dict(frame.depth_metadata),
        )
    return (
        replace(
            snapshot,
            camera_frames=frames,
            arm_states=hydrate_nested(snapshot.arm_states),
            hand_states=hydrate_nested(snapshot.hand_states),
            position_reference=hydrate_nested(snapshot.position_reference),
        ),
        slot_id,
    )


def _configure_worker_process(config: ProcessCollectionConfig) -> dict[str, Any]:
    status: dict[str, Any] = {"pid": os.getpid()}
    if config.worker_cpus and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, set(int(cpu) for cpu in config.worker_cpus))
            status["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        except Exception as exc:  # noqa: BLE001 - report but do not prevent collection.
            status["cpu_affinity_error"] = f"{type(exc).__name__}: {exc}"
    if int(config.worker_nice) != 0:
        try:
            os.nice(int(config.worker_nice))
            status["nice"] = os.nice(0)
        except Exception as exc:  # noqa: BLE001
            status["nice_error"] = f"{type(exc).__name__}: {exc}"
    return status


def _worker_main(
    writer_config: LeRobotV21WriterConfig,
    sample_builder_config: CollectorSampleBuilderConfig,
    process_config: ProcessCollectionConfig,
    shm_names: tuple[str, ...],
    input_queue: Any,
    reply_queue: Any,
    ack_queue: Any,
) -> None:
    blocks: dict[str, shared_memory.SharedMemory] = {}
    writer: LeRobotV21Writer | None = None
    try:
        process_status = _configure_worker_process(process_config)
        blocks = {
            name: shared_memory.SharedMemory(name=name, create=False)
            for name in shm_names
        }
        writer = LeRobotV21Writer(writer_config)
        detached_builder_config = replace(
            sample_builder_config,
            camera_buffers={},
            require_camera_frames=False,
            include_debug_metadata=False,
            include_action_debug_metadata=False,
            episode_index=int(writer.next_episode_index),
            sample_index_start=int(writer.next_sample_index),
        )
        builder = CollectorSampleBuilder(detached_builder_config)
        reply_queue.put(
            {
                "kind": "ready",
                "ok": True,
                "next_episode_index": int(writer.next_episode_index),
                "next_sample_index": int(writer.next_sample_index),
                "process": process_status,
            }
        )
        while True:
            message = input_queue.get()
            kind = str(message.get("kind", ""))
            if kind == "sample":
                slot_id = message.get("slot_id")
                started = time.perf_counter()
                try:
                    stage_started = started
                    snapshot, hydrated_slot = _hydrate_snapshot(
                        message["snapshot"], blocks
                    )
                    stage_ms: dict[str, float] = {
                        "worker.hydrate_shared": float(
                            (time.perf_counter() - stage_started) * 1000.0
                        )
                    }
                    if hydrated_slot is not None:
                        slot_id = int(hydrated_slot)
                    stage_started = time.perf_counter()
                    sample = builder.build_from_snapshot(snapshot)
                    stage_ms["worker.build_sample"] = float(
                        (time.perf_counter() - stage_started) * 1000.0
                    )
                    # Sample assembly deep-copies state arrays into child-owned
                    # storage. Camera frames remain referenced by the sample,
                    # so retain the slot through synchronous job handling.
                    metadata_extra = message.get("metadata_extra")
                    if isinstance(metadata_extra, Mapping):
                        sample.metadata.update(dict(metadata_extra))
                    stage_started = time.perf_counter()
                    jobs = writer.plan_sample_jobs(sample)
                    stage_ms["worker.plan_jobs"] = float(
                        (time.perf_counter() - stage_started) * 1000.0
                    )
                    for job in jobs:
                        job_started = time.perf_counter()
                        writer.handle_job(job)
                        job_key = f"worker.handle.{job.kind}"
                        stage_ms[job_key] = stage_ms.get(job_key, 0.0) + float(
                            (time.perf_counter() - job_started) * 1000.0
                        )
                    ack_queue.put({"kind": "slot_free", "slot_id": slot_id})
                    slot_id = None
                    ack_queue.put(
                        {
                            "kind": "sample_done",
                            "episode_index": int(sample.episode_index),
                            "frame_count": int(builder.frame_index),
                            "elapsed_ms": float(
                                (time.perf_counter() - started) * 1000.0
                            ),
                            "stage_ms": stage_ms,
                            "job_count": len(jobs),
                            "tactile_frame_counts": writer.tactile_frame_counts(
                                int(sample.episode_index)
                            ),
                        }
                    )
                except BaseException as exc:  # noqa: BLE001
                    if slot_id is not None:
                        ack_queue.put({"kind": "slot_free", "slot_id": slot_id})
                    ack_queue.put(
                        {
                            "kind": "worker_error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                    )
                continue

            request_id = message.get("request_id")
            try:
                if kind == "start_episode":
                    episode_index = writer.start_episode(
                        metadata=dict(message.get("metadata") or {})
                    )
                    builder.set_episode(int(episode_index))
                    result: Any = int(episode_index)
                elif kind == "write_device_config":
                    writer.write_device_config(
                        str(message["device_name"]),
                        dict(message.get("payload") or {}),
                    )
                    result = True
                elif kind == "flush":
                    result = True
                elif kind == "finalize_episode":
                    writer.finalize_episode(
                        int(message["episode_index"]),
                        summary=dict(message.get("summary") or {}),
                    )
                    result = True
                elif kind == "discard_episode":
                    result = bool(writer.discard_episode(int(message["episode_index"])))
                elif kind == "status":
                    episode_index = message.get("episode_index")
                    result = {
                        "next_episode_index": int(writer.next_episode_index),
                        "next_sample_index": int(writer.next_sample_index),
                        "frame_index": int(builder.frame_index),
                        "tactile_frame_counts": (
                            {}
                            if episode_index is None
                            else writer.tactile_frame_counts(int(episode_index))
                        ),
                    }
                elif kind == "close":
                    reply_queue.put(
                        {
                            "kind": "reply",
                            "request_id": request_id,
                            "ok": True,
                            "result": True,
                        }
                    )
                    break
                else:
                    raise ValueError(f"unknown collection worker message kind {kind!r}")
            except BaseException as exc:  # noqa: BLE001
                reply_queue.put(
                    {
                        "kind": "reply",
                        "request_id": request_id,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
            else:
                reply_queue.put(
                    {
                        "kind": "reply",
                        "request_id": request_id,
                        "ok": True,
                        "result": result,
                    }
                )
    except BaseException as exc:  # noqa: BLE001
        reply_queue.put(
            {
                "kind": "ready" if writer is None else "fatal",
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        for block in blocks.values():
            block.close()


class ProcessCollectionWorker:
    """Parent-side proxy for the spawned collection data plane."""

    def __init__(
        self,
        *,
        writer_config: LeRobotV21WriterConfig,
        sample_builder_config: CollectorSampleBuilderConfig,
        config: ProcessCollectionConfig,
    ) -> None:
        self.writer_config = writer_config
        # Never pass frontend camera buffers through multiprocessing ``spawn``.
        # NearestFrameBuffer intentionally owns a threading.Lock and is only
        # consumed while snapshotting in the parent.  The child receives fully
        # frozen camera frames through shared memory, so live buffers are both
        # unnecessary and invalid at this process boundary.
        self.sample_builder_config = replace(
            sample_builder_config,
            camera_buffers={},
            require_camera_frames=False,
        )
        self.config = config
        self._ctx = mp.get_context("spawn")
        self._input_queue = self._ctx.Queue(maxsize=int(config.queue_size))
        self._reply_queue = self._ctx.Queue(maxsize=16)
        self._ack_queue = self._ctx.Queue(maxsize=max(16, int(config.queue_size) * 4))
        self._slots = _SharedSlotPool(
            slot_count=int(config.shared_slot_count),
            slot_bytes=int(config.shared_slot_bytes),
        )
        self._process = self._ctx.Process(
            target=_worker_main,
            args=(
                writer_config,
                self.sample_builder_config,
                config,
                self._slots.names,
                self._input_queue,
                self._reply_queue,
                self._ack_queue,
            ),
            name="formal_collection_worker",
            daemon=True,
        )
        self._started = False
        self._closed = False
        self._request_lock = self._ctx.Lock()
        self._submitted_samples = 0
        self._completed_samples = 0
        self._rejected_busy = 0
        self._queue_high_water = 0
        self._active_episode_frame_count = 0
        self._worker_error: str | None = None
        self._sample_process_ms: deque[float] = deque(maxlen=2048)
        self._parent_pack_ms: deque[float] = deque(maxlen=2048)
        self._parent_queue_put_ms: deque[float] = deque(maxlen=2048)
        self._shared_payload_bytes: deque[float] = deque(maxlen=2048)
        self._worker_stage_ms: dict[str, deque[float]] = {}
        self._tactile_counts_by_episode: dict[int, dict[str, int]] = {}
        self.next_episode_index = 0
        self.next_sample_index = 0
        self.process_status: dict[str, Any] = {}

    def start(self) -> None:
        if self._started:
            return
        try:
            self._process.start()
        except BaseException:  # noqa: BLE001 - release preallocated shared memory.
            self.close(force=True)
            raise
        try:
            ready = self._reply_queue.get(
                timeout=float(self.config.startup_timeout_sec)
            )
        except queue.Empty as exc:
            self.close(force=True)
            raise RuntimeError("collection worker did not become ready") from exc
        if not isinstance(ready, Mapping) or not bool(ready.get("ok")):
            self.close(force=True)
            raise RuntimeError(
                "collection worker startup failed: "
                f"{ready.get('error') if isinstance(ready, Mapping) else ready!r}"
            )
        self.next_episode_index = int(ready.get("next_episode_index", 0))
        self.next_sample_index = int(ready.get("next_sample_index", 0))
        self.process_status = dict(ready.get("process") or {})
        self._started = True

    def _drain_acks(self) -> None:
        while True:
            try:
                message = self._ack_queue.get_nowait()
            except queue.Empty:
                break
            kind = str(message.get("kind", ""))
            if kind == "slot_free":
                slot_id = message.get("slot_id")
                if slot_id is not None:
                    self._slots.release(int(slot_id))
            elif kind == "sample_done":
                self._completed_samples += 1
                self._active_episode_frame_count = int(message.get("frame_count", 0))
                elapsed_ms = message.get("elapsed_ms")
                if elapsed_ms is not None:
                    self._sample_process_ms.append(float(elapsed_ms))
                stage_ms = message.get("stage_ms")
                if isinstance(stage_ms, Mapping):
                    for name, value in stage_ms.items():
                        if value is None:
                            continue
                        values = self._worker_stage_ms.setdefault(
                            str(name), deque(maxlen=2048)
                        )
                        values.append(float(value))
                episode_index = int(message.get("episode_index", -1))
                counts = message.get("tactile_frame_counts")
                if isinstance(counts, Mapping):
                    self._tactile_counts_by_episode[episode_index] = {
                        str(side): int(value) for side, value in counts.items()
                    }
            elif kind == "worker_error":
                self._worker_error = str(message.get("error") or "collection worker error")

    def _raise_if_failed(self) -> None:
        self._drain_acks()
        if self._worker_error:
            raise RuntimeError(self._worker_error)
        if self._started and not self._process.is_alive():
            raise RuntimeError(
                f"collection worker exited unexpectedly with code {self._process.exitcode}"
            )

    def _request(self, kind: str, **payload: Any) -> Any:
        self._raise_if_failed()
        request_id = uuid.uuid4().hex
        message = {"kind": str(kind), "request_id": request_id, **payload}
        with self._request_lock:
            self._input_queue.put(
                message,
                timeout=float(self.config.request_timeout_sec),
            )
            deadline = time.monotonic() + float(self.config.request_timeout_sec)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(f"collection worker request {kind!r} timed out")
                try:
                    reply = self._reply_queue.get(timeout=remaining)
                except queue.Empty as exc:
                    raise TimeoutError(
                        f"collection worker request {kind!r} timed out"
                    ) from exc
                if reply.get("request_id") != request_id:
                    raise RuntimeError("collection worker reply ordering mismatch")
                if not bool(reply.get("ok")):
                    raise RuntimeError(
                        f"collection worker {kind} failed: {reply.get('error')}"
                    )
                return reply.get("result")

    def submit_snapshot(
        self,
        snapshot: CollectorSampleSnapshot,
        *,
        metadata_extra: Mapping[str, Any] | None = None,
    ) -> bool:
        self._raise_if_failed()
        slot_id = self._slots.acquire()
        if slot_id is None:
            self._rejected_busy += 1
            return False
        try:
            pack_started = time.perf_counter()
            packed, payload_bytes = self._slots.pack_snapshot(
                snapshot, slot_id=slot_id
            )
            self._parent_pack_ms.append(
                float((time.perf_counter() - pack_started) * 1000.0)
            )
            self._shared_payload_bytes.append(float(payload_bytes))
            queue_started = time.perf_counter()
            self._input_queue.put_nowait(
                {
                    "kind": "sample",
                    "slot_id": int(slot_id),
                    "snapshot": packed,
                    "metadata_extra": (
                        None if metadata_extra is None else dict(metadata_extra)
                    ),
                }
            )
            self._parent_queue_put_ms.append(
                float((time.perf_counter() - queue_started) * 1000.0)
            )
        except queue.Full:
            self._slots.release(slot_id)
            self._rejected_busy += 1
            return False
        except BaseException:
            self._slots.release(slot_id)
            raise
        self._submitted_samples += 1
        try:
            self._queue_high_water = max(
                self._queue_high_water, int(self._input_queue.qsize())
            )
        except (NotImplementedError, OSError):
            pass
        return True

    def start_episode(self, metadata: Mapping[str, Any]) -> int:
        episode_index = int(
            self._request("start_episode", metadata=dict(metadata))
        )
        self.next_episode_index = episode_index + 1
        self._active_episode_frame_count = 0
        return episode_index

    def write_device_config(self, device_name: str, payload: Mapping[str, Any]) -> None:
        self._request(
            "write_device_config",
            device_name=str(device_name),
            payload=dict(payload),
        )

    def flush(self) -> None:
        self._request("flush")
        self._drain_acks()

    def finalize_episode(self, episode_index: int, *, summary: Mapping[str, Any]) -> None:
        self._request(
            "finalize_episode",
            episode_index=int(episode_index),
            summary=dict(summary),
        )

    def discard_episode(self, episode_index: int) -> bool:
        return bool(
            self._request("discard_episode", episode_index=int(episode_index))
        )

    def tactile_frame_counts(self, episode_index: int) -> dict[str, int]:
        self._drain_acks()
        return dict(self._tactile_counts_by_episode.get(int(episode_index), {}))

    @property
    def active_episode_frame_count(self) -> int:
        self._drain_acks()
        return int(self._active_episode_frame_count)

    def stats(self) -> ProcessCollectionStats:
        self._drain_acks()
        values = list(self._sample_process_ms)
        timing_ms = {
            "parent.pack_shared": _metric_summary(list(self._parent_pack_ms)),
            "parent.queue_put": _metric_summary(list(self._parent_queue_put_ms)),
            "worker.total": _metric_summary(values),
            **{
                name: _metric_summary(list(stage_values))
                for name, stage_values in sorted(self._worker_stage_ms.items())
            },
        }
        return ProcessCollectionStats(
            submitted_samples=int(self._submitted_samples),
            completed_samples=int(self._completed_samples),
            rejected_busy=int(self._rejected_busy),
            queue_capacity=int(self.config.queue_size),
            queue_high_water=int(self._queue_high_water),
            slots_available=len(self._slots.available),
            slot_count=len(self._slots.blocks),
            worker_alive=bool(self._started and self._process.is_alive()),
            worker_error=self._worker_error,
            active_episode_frame_count=int(self._active_episode_frame_count),
            sample_process_ms_p50=_percentile(values, 50),
            sample_process_ms_p95=_percentile(values, 95),
            timing_ms=timing_ms,
            shared_payload_bytes=_metric_summary(
                list(self._shared_payload_bytes)
            ),
        )

    def quick_status(self) -> dict[str, Any]:
        """Cheap status for the 60 Hz frontend; avoids percentile recomputation."""

        self._drain_acks()
        return {
            "submitted_samples": int(self._submitted_samples),
            "completed_samples": int(self._completed_samples),
            "rejected_busy": int(self._rejected_busy),
            "queue_high_water": int(self._queue_high_water),
            "slots_available": len(self._slots.available),
            "slot_count": len(self._slots.blocks),
            "worker_alive": bool(self._started and self._process.is_alive()),
            "worker_error": self._worker_error,
            "active_episode_frame_count": int(self._active_episode_frame_count),
            "parent_pack_ms_latest": (
                None if not self._parent_pack_ms else float(self._parent_pack_ms[-1])
            ),
            "parent_queue_put_ms_latest": (
                None
                if not self._parent_queue_put_ms
                else float(self._parent_queue_put_ms[-1])
            ),
            "shared_payload_bytes_latest": (
                None
                if not self._shared_payload_bytes
                else int(self._shared_payload_bytes[-1])
            ),
        }

    def close(self, *, flush: bool = True, force: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._process.is_alive() and not force:
                try:
                    if flush:
                        self.flush()
                    self._request("close")
                except Exception:
                    force = True
            if self._process.is_alive():
                self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
        finally:
            self._slots.close()
            for runtime_queue in (
                self._input_queue,
                self._reply_queue,
                self._ack_queue,
            ):
                try:
                    runtime_queue.close()
                    runtime_queue.join_thread()
                except Exception:
                    pass


__all__ = [
    "ProcessCollectionConfig",
    "ProcessCollectionStats",
    "ProcessCollectionWorker",
    "SharedArrayRef",
    "camera_shared_slot_bytes",
]

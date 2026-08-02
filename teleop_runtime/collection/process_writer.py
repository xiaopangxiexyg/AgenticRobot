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
import sys
import time
import traceback
import uuid
from multiprocessing.reduction import ForkingPickler
from collections import deque
from dataclasses import dataclass, replace
from multiprocessing import shared_memory
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.v25p_native_storage import (
    V25P_NATIVE_STORAGE_LAYOUTS_V1,
    V25P_RECORD_CJSON_LAYOUT_V1,
)
from dataset_contracts.stream_catalog import (
    NativeStreamContract,
    NativeStreamEpisodeManifest,
    NativeStreamExternalArtifactReceipt,
    NativeStreamExternalSampleIntent,
)
from robot_motion.runtime.observation_cut import ObservationSampleRef

from collection_runtime.sample_acceptance import (
    AcceptedSampleReceipt,
    SampleAcceptanceReceiptBuffer,
)
from .sample_assembler import (
    CollectorSampleBuilder,
    CollectorSampleBuilderConfig,
    CollectorSampleSnapshot,
)
from .schema_v22 import (
    StrictV22SampleRejection,
    validate_strict_v22_measured_state,
)
from collection_runtime.stream_sink import NativeStreamEpisodeWriter
from .video_frame import VideoFrame
from .writer_v21 import (
    LegacyEpisodeArtifactReport,
    LeRobotV21Writer,
    LeRobotV21WriterConfig,
)


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
    native_inline_message_max_bytes: int = 256 * 1024
    sample_acceptance_receipt_capacity: int = 0

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
        if int(self.native_inline_message_max_bytes) <= 0:
            raise ValueError("native_inline_message_max_bytes must be positive.")
        if int(self.sample_acceptance_receipt_capacity) < 0:
            raise ValueError(
                "sample_acceptance_receipt_capacity must be non-negative."
            )


@dataclass(frozen=True)
class ProcessV25pShadowConfig:
    """Minimal opt-in needed to construct the native shadow in the child.

    Stream/profile details are derived from the writer's already-authoritative
    policy-I/O and pose-frame contracts. Dataset identity and the physical
    persistence choice cross the process boundary; no second embodiment or
    component profile surface is introduced.
    """

    dataset_id: str
    pose_frame_contract_hash: str
    physical_storage_layout: str = V25P_RECORD_CJSON_LAYOUT_V1

    def __post_init__(self) -> None:
        dataset_id = str(self.dataset_id).strip()
        pose_hash = str(self.pose_frame_contract_hash).strip()
        if not dataset_id or not pose_hash:
            raise ValueError(
                "v2.5-p shadow dataset and pose-frame identity are required."
            )
        storage_layout = str(self.physical_storage_layout).strip()
        if storage_layout not in V25P_NATIVE_STORAGE_LAYOUTS_V1:
            raise ValueError("Unsupported v2.5-p native physical storage layout.")
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "pose_frame_contract_hash", pose_hash)
        object.__setattr__(self, "physical_storage_layout", storage_layout)


@dataclass(frozen=True)
class ProcessExternalNativeStreamBinding:
    """Data-driven projection from one specialized writer source."""

    stream_id: str
    source_kind: str
    source_key: str

    def __post_init__(self) -> None:
        stream_id = str(self.stream_id).strip()
        source_kind = str(self.source_kind).strip()
        source_key = str(self.source_key).strip()
        if not stream_id or not source_key:
            raise ValueError("External stream binding identity must be non-empty.")
        if source_kind != "selected_rgb":
            raise ValueError(
                "Process external stream binding currently supports only "
                "selected_rgb."
            )
        object.__setattr__(self, "stream_id", stream_id)
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "source_key", source_key)


@dataclass(frozen=True)
class ProcessLegacyFinalizeResult:
    episode_index: int
    legacy_finalized: bool
    native_stream_artifact_receipts: tuple[
        NativeStreamExternalArtifactReceipt, ...
    ] = ()
    artifact_receipt_error: str = ""
    v25p_native_shadow_report: Any | None = None
    v25p_native_shadow_error: str = ""


@dataclass(frozen=True)
class ProcessCollectionStats:
    submitted_samples: int
    completed_samples: int
    rejected_busy: int
    sample_rejections: int
    queue_capacity: int
    queue_high_water: int
    slots_available: int
    slot_count: int
    worker_alive: bool
    worker_error: str | None
    worker_traceback: str | None
    active_episode_frame_count: int
    sample_process_ms_p50: float | None
    sample_process_ms_p95: float | None
    timing_ms: dict[str, dict[str, float | int | None]]
    shared_payload_bytes: dict[str, float | int | None]
    submitted_native_stream_samples: int
    completed_native_stream_samples: int
    rejected_native_stream_samples: int


@dataclass(frozen=True)
class ProcessNativeStreamSubmissionReceipt:
    episode_uuid: str
    legacy_episode_index: int
    stream_id: str
    sample_id: str
    accepted: bool
    rejection_reason: str | None
    serialized_message_bytes: int


class ProcessNativeStreamEpisodeCapture:
    """Parent-side subordinate capture proxy for inline or external streams."""

    def __init__(
        self,
        worker: "ProcessCollectionWorker",
        *,
        episode_uuid: str,
        legacy_episode_index: int,
    ) -> None:
        self.worker = worker
        self.episode_uuid = str(episode_uuid)
        self.legacy_episode_index = int(legacy_episode_index)
        self._pending_drop_counts: dict[tuple[str, str], int] = {}
        self._closed = False
        self._frozen = False

    @property
    def requires_post_finalize_seal(self) -> bool:
        return bool(self.worker.native_external_stream_bindings)

    def append(self, **sample: Any) -> ProcessNativeStreamSubmissionReceipt:
        if self._closed:
            raise RuntimeError("Native stream process capture is closed.")
        if self._frozen:
            raise RuntimeError("Native stream process capture is frozen.")
        receipt = self.worker.submit_native_stream_sample(
            episode_uuid=self.episode_uuid,
            legacy_episode_index=self.legacy_episode_index,
            **sample,
        )
        if not receipt.accepted:
            key = (
                str(sample.get("stream_id", "")),
                str(receipt.rejection_reason or "process_queue_busy"),
            )
            self._pending_drop_counts[key] = (
                self._pending_drop_counts.get(key, 0) + 1
            )
        return receipt

    def note_drop(self, stream_id: str, *, reason: str) -> None:
        if self._closed:
            raise RuntimeError("Native stream process capture is closed.")
        if self._frozen:
            raise RuntimeError("Native stream process capture is frozen.")
        key = (str(stream_id), str(reason))
        self._pending_drop_counts[key] = self._pending_drop_counts.get(key, 0) + 1

    def append_observation_sample(
        self,
        sample: ObservationSampleRef,
        payload: Any,
        *,
        written_at: float,
        written_clock_domain_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> ProcessNativeStreamSubmissionReceipt:
        if not isinstance(sample, ObservationSampleRef):
            raise TypeError("sample must be an ObservationSampleRef.")
        return self.append(
            stream_id=sample.stream_id,
            sample_id=sample.sample_id,
            sequence=sample.sequence,
            measured_at=sample.measured_at,
            measured_clock_domain_id=sample.measured_clock_domain_id,
            available_at=sample.available_at,
            available_clock_domain_id=sample.available_clock_domain_id,
            source_payload_object_id=sample.payload_object_id,
            payload=payload,
            valid=sample.valid,
            stale=sample.stale,
            gap_reason=None,
            written_at=written_at,
            written_clock_domain_id=written_clock_domain_id,
            metadata={
                **dict(sample.content_metadata),
                **dict(metadata or {}),
            },
        )

    def seal(
        self,
        *,
        sealed_at: float,
        sealed_clock_domain_id: str,
    ) -> NativeStreamEpisodeManifest:
        if self._closed:
            raise RuntimeError("Native stream process capture is closed.")
        if self.requires_post_finalize_seal:
            raise RuntimeError(
                "External native streams must seal after legacy finalize."
            )
        manifest = self.worker.seal_native_stream_episode(
            episode_uuid=self.episode_uuid,
            legacy_episode_index=self.legacy_episode_index,
            sealed_at=sealed_at,
            sealed_clock_domain_id=sealed_clock_domain_id,
            drop_counts=self._pending_drop_counts,
        )
        self._closed = True
        return manifest

    def freeze_ingestion(self) -> None:
        if self._closed:
            raise RuntimeError("Native stream process capture is closed.")
        if not self.requires_post_finalize_seal:
            raise RuntimeError(
                "Inline native streams do not use post-finalize freeze."
            )
        if self._frozen:
            return
        self.worker.freeze_native_stream_episode(
            episode_uuid=self.episode_uuid,
            legacy_episode_index=self.legacy_episode_index,
            drop_counts=self._pending_drop_counts,
        )
        self._pending_drop_counts.clear()
        self._frozen = True

    def seal_after_legacy_finalize(
        self,
        receipts: Iterable[NativeStreamExternalArtifactReceipt],
        *,
        sealed_at: float,
        sealed_clock_domain_id: str,
    ) -> NativeStreamEpisodeManifest:
        if self._closed:
            raise RuntimeError("Native stream process capture is closed.")
        if not self._frozen:
            raise RuntimeError(
                "External native stream capture must freeze before seal."
            )
        manifest = self.worker.seal_native_stream_episode_after_legacy_finalize(
            episode_uuid=self.episode_uuid,
            legacy_episode_index=self.legacy_episode_index,
            receipts=tuple(receipts),
            sealed_at=sealed_at,
            sealed_clock_domain_id=sealed_clock_domain_id,
        )
        self._closed = True
        return manifest

    def close_unsealed(self) -> None:
        if self._closed:
            return
        self.worker.close_native_stream_episode(
            episode_uuid=self.episode_uuid,
            legacy_episode_index=self.legacy_episode_index,
        )
        self._closed = True


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


def _selected_rgb_sample_id(
    binding: ProcessExternalNativeStreamBinding,
    frame: VideoFrame,
) -> str:
    source_identity: dict[str, Any] = {
        "stream_id": binding.stream_id,
        "camera_key": binding.source_key,
        "source_frame_index": frame.frame_index,
        "source_timestamp": frame.source_timestamp,
    }
    if frame.frame_index is None:
        source_identity["payload_content_hash"] = canonical_content_hash(
            frame.image,
            domain="dexteleop.selected_rgb_source_payload.v1",
        )
    identity_hash = canonical_content_hash(
        source_identity,
        domain="dexteleop.selected_rgb_sample_identity.v1",
    )
    return f"selected-rgb:{identity_hash}"


def _selected_rgb_payload(frame: VideoFrame) -> np.ndarray:
    payload = np.asarray(frame.image)
    color_space = str(frame.metadata.get("color_space", "rgb")).strip().lower()
    if color_space == "rgb":
        return np.ascontiguousarray(payload)
    if color_space == "bgr":
        if payload.ndim != 3 or payload.shape[-1] != 3:
            raise ValueError("BGR selected frame must have exactly three channels.")
        return np.ascontiguousarray(payload[..., ::-1])
    raise ValueError(
        "selected_rgb requires source color_space 'rgb' or 'bgr', "
        f"got {color_space!r}."
    )


def _stage_selected_rgb_intents(
    sample: Any,
    *,
    native_writer: NativeStreamEpisodeWriter,
    native_stream_contracts: Mapping[str, NativeStreamContract],
    external_stream_bindings: Mapping[
        str, ProcessExternalNativeStreamBinding
    ],
    next_sequence_by_stream: dict[str, int],
    intent_and_sample_frame_by_identity: dict[
        tuple[str, str], tuple[NativeStreamExternalSampleIntent, int]
    ],
) -> None:
    for stream_id, binding in external_stream_bindings.items():
        contract = native_stream_contracts[stream_id]
        match = sample.camera_frames.get(binding.source_key)
        frame = None if match is None else match.frame
        if frame is None:
            native_writer.note_drop(
                stream_id,
                reason="selected_rgb_frame_missing",
            )
            continue
        sample_id = _selected_rgb_sample_id(binding, frame)
        identity = (stream_id, sample_id)
        if identity in intent_and_sample_frame_by_identity:
            continue
        available_at = frame.available_timestamp
        availability_fallback = "frame.available_timestamp"
        if available_at is None:
            available_at = frame.receive_timestamp
            availability_fallback = "frame.receive_timestamp"
        if available_at is None:
            available_at = sample.timestamp
            availability_fallback = "collector_sample.timestamp"
        sequence = int(next_sequence_by_stream.get(stream_id, 0))
        intent = NativeStreamExternalSampleIntent.create(
            contract=contract,
            sample_id=sample_id,
            sequence=sequence,
            measured_at=float(frame.source_timestamp),
            measured_clock_domain_id=contract.measured_clock_domain_id,
            available_at=float(available_at),
            available_clock_domain_id=contract.available_clock_domain_id,
            source_payload_object_id=(
                f"selected-rgb:{binding.source_key}:{sample_id}"
            ),
            payload=_selected_rgb_payload(frame),
            metadata={
                "source_kind": binding.source_kind,
                "source_key": binding.source_key,
                "source_frame_index": frame.frame_index,
                "collector_sample_frame_index": int(sample.frame_index),
                "availability_source": availability_fallback,
                "logical_color_space": "rgb",
                "source_color_space": str(
                    frame.metadata.get("color_space", "rgb")
                ).strip().lower(),
            },
        )
        native_writer.stage_external_sample(intent)
        next_sequence_by_stream[stream_id] = sequence + 1
        intent_and_sample_frame_by_identity[identity] = (
            intent,
            int(sample.frame_index),
        )


def _selected_rgb_artifact_receipts(
    report: LegacyEpisodeArtifactReport,
    *,
    episode_uuid: str,
    native_stream_contracts: Mapping[str, NativeStreamContract],
    external_stream_bindings: Mapping[
        str, ProcessExternalNativeStreamBinding
    ],
    intent_and_sample_frame_by_identity: Mapping[
        tuple[str, str], tuple[NativeStreamExternalSampleIntent, int]
    ],
) -> tuple[NativeStreamExternalArtifactReceipt, ...]:
    artifact_by_camera = {
        artifact.camera_key: artifact
        for artifact in report.video_artifacts
    }
    receipts: list[NativeStreamExternalArtifactReceipt] = []
    durable_at = time.time()
    for (stream_id, _sample_id), (
        intent,
        sample_frame_index,
    ) in sorted(
        intent_and_sample_frame_by_identity.items(),
        key=lambda item: (item[0][0], item[1][0].sequence),
    ):
        binding = external_stream_bindings[stream_id]
        contract = native_stream_contracts[stream_id]
        artifact = artifact_by_camera.get(binding.source_key)
        if artifact is None:
            raise RuntimeError(
                "Finalized RGB artifact report is missing camera "
                f"{binding.source_key!r}."
            )
        frame_index_by_sample = dict(
            artifact.video_frame_index_by_sample_frame_index
        )
        if sample_frame_index not in frame_index_by_sample:
            raise RuntimeError(
                "Finalized RGB artifact lacks the selected sample frame "
                f"{sample_frame_index}."
            )
        if (
            artifact.encoding != contract.encoding
            or artifact.compression != contract.compression
        ):
            raise RuntimeError(
                "Finalized RGB artifact encoding/compression differs from "
                "the native stream contract."
            )
        video_frame_index = frame_index_by_sample[sample_frame_index]
        receipts.append(
            NativeStreamExternalArtifactReceipt(
                episode_uuid=episode_uuid,
                legacy_episode_index=int(report.episode_index),
                stream_id=stream_id,
                sample_id=intent.sample_id,
                payload_content_hash=intent.payload_content_hash,
                stored_payload_object_id=(
                    f"dataset:{artifact.artifact_relative_path}"
                    f"#video_frame={video_frame_index}"
                ),
                artifact_relative_path=artifact.artifact_relative_path,
                artifact_sha256=artifact.artifact_sha256,
                artifact_byte_count=artifact.artifact_byte_count,
                encoding=artifact.encoding,
                compression=artifact.compression,
                durable_at=durable_at,
                durable_clock_domain_id="local_wall_clock",
            )
        )
    return tuple(receipts)


def _worker_main(
    writer_config: LeRobotV21WriterConfig,
    sample_builder_config: CollectorSampleBuilderConfig,
    process_config: ProcessCollectionConfig,
    v25p_shadow_config: ProcessV25pShadowConfig | None,
    native_stream_contracts: Mapping[str, NativeStreamContract],
    external_stream_bindings: Mapping[
        str, ProcessExternalNativeStreamBinding
    ],
    shm_names: tuple[str, ...],
    input_queue: Any,
    reply_queue: Any,
    ack_queue: Any,
) -> None:
    blocks: dict[str, shared_memory.SharedMemory] = {}
    writer: LeRobotV21Writer | None = None
    native_writer: NativeStreamEpisodeWriter | None = None
    v25p_native_shadow: Any | None = None
    v25p_native_shadow_identity: EpisodeCaptureIdentity | None = None
    v25p_native_shadow_error = ""
    native_ingestion_frozen = False
    next_external_sequence_by_stream: dict[str, int] = {}
    external_intent_and_sample_frame_by_identity: dict[
        tuple[str, str], tuple[NativeStreamExternalSampleIntent, int]
    ] = {}
    try:
        process_status = _configure_worker_process(process_config)
        blocks = {
            name: shared_memory.SharedMemory(name=name, create=False)
            for name in shm_names
        }
        writer = LeRobotV21Writer(writer_config)
        if v25p_shadow_config is not None:
            if writer_config.policy_io_layout is None:
                raise ValueError("v2.5-p native shadow requires a policy-I/O layout.")
            from .v25p_capture_profiles import (
                V25P_REAL_TELEOP_SOURCE,
                build_policy_io_v25p_capture_profiles,
            )
            from .v25p_dual_write_shadow import V25pDualWriteShadow
            from collection_runtime.component_capture import (
                ComponentCaptureEpisodeStore,
            )
            from .v25p_real_teleop_producer import RealTeleopV25pProducer

            bundle = build_policy_io_v25p_capture_profiles(
                source=V25P_REAL_TELEOP_SOURCE,
                layout=writer_config.policy_io_layout,
                pose_frame_contract_hash=v25p_shadow_config.pose_frame_contract_hash,
                component_sides=tuple(sample_builder_config.active_sides),
            )
            v25p_native_shadow = V25pDualWriteShadow(
                store=ComponentCaptureEpisodeStore(
                    dataset_root=writer_config.root_dir,
                    dataset_id=v25p_shadow_config.dataset_id,
                    embodiment_profile=bundle.embodiment,
                    component_stream_profile=bundle.streams,
                    compatibility_projection_profiles=(bundle.v23_projection,),
                    physical_storage_layout=(
                        v25p_shadow_config.physical_storage_layout
                    ),
                    defer_sample_file_writes=(
                        v25p_shadow_config.physical_storage_layout
                        == V25P_RECORD_CJSON_LAYOUT_V1
                    ),
                ),
                bundle=bundle,
                producer=RealTeleopV25pProducer(
                    layout=writer_config.policy_io_layout,
                    stream_profile=bundle.streams,
                    bindings=bundle.producer_bindings,
                    component_action_schema=bundle.component_action_schema,
                    component_action_projection=(
                        bundle.component_action_projection
                    ),
                ),
            )
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
                active_stage = "worker.hydrate_shared"
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
                    if (
                        bool(getattr(writer_config, "strict_v22_policy_io", False))
                        and writer_config.policy_io_layout is not None
                    ):
                        active_stage = "worker.validate_sample_contract"
                        stage_started = time.perf_counter()
                        validate_strict_v22_measured_state(
                            snapshot.hand_states,
                            layout=writer_config.policy_io_layout,
                            sample_timestamp=float(snapshot.step_result.timestamp),
                        )
                        stage_ms["worker.validate_sample_contract"] = float(
                            (time.perf_counter() - stage_started) * 1000.0
                        )
                    active_stage = "worker.build_sample"
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
                    active_stage = "worker.plan_jobs"
                    stage_started = time.perf_counter()
                    jobs = writer.plan_sample_jobs(sample)
                    stage_ms["worker.plan_jobs"] = float(
                        (time.perf_counter() - stage_started) * 1000.0
                    )
                    if (
                        v25p_native_shadow is not None
                        and v25p_native_shadow_identity is not None
                        and not v25p_native_shadow_error
                    ):
                        active_stage = "worker.v25p_native_shadow_append"
                        stage_started = time.perf_counter()
                        try:
                            from .v25p_dual_write_shadow import (
                                legacy_row_from_planned_jobs,
                            )

                            v25p_native_shadow.append(
                                v25p_native_shadow_identity,
                                source_sample=sample,
                                legacy_row=legacy_row_from_planned_jobs(jobs),
                            )
                        except BaseException as exc:
                            # The legacy capture stays authoritative.
                            v25p_native_shadow_error = (
                                f"append:{type(exc).__name__}: {exc}"
                            )
                            try:
                                v25p_native_shadow.close_unsealed(
                                    v25p_native_shadow_identity
                                )
                            except BaseException as close_exc:
                                v25p_native_shadow_error += (
                                    ";close_unsealed:"
                                    f"{type(close_exc).__name__}: {close_exc}"
                                )
                            v25p_native_shadow_identity = None
                        stage_ms["worker.v25p_native_shadow_append"] = float(
                            (time.perf_counter() - stage_started) * 1000.0
                        )
                    if native_writer is not None and external_stream_bindings:
                        active_stage = "worker.stage_external_selected_rgb"
                        stage_started = time.perf_counter()
                        _stage_selected_rgb_intents(
                            sample,
                            native_writer=native_writer,
                            native_stream_contracts=native_stream_contracts,
                            external_stream_bindings=external_stream_bindings,
                            next_sequence_by_stream=(
                                next_external_sequence_by_stream
                            ),
                            intent_and_sample_frame_by_identity=(
                                external_intent_and_sample_frame_by_identity
                            ),
                        )
                        stage_ms[
                            "worker.stage_external_selected_rgb"
                        ] = float(
                            (time.perf_counter() - stage_started) * 1000.0
                        )
                    for job in jobs:
                        active_stage = f"worker.handle.{job.kind}"
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
                            "sample_index": int(sample.index),
                            "episode_frame_index": int(sample.frame_index),
                            "sample_timestamp": float(sample.timestamp),
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
                except StrictV22SampleRejection as exc:
                    if slot_id is not None:
                        ack_queue.put({"kind": "slot_free", "slot_id": slot_id})
                        slot_id = None
                    # A strict measured-feedback rejection belongs to this
                    # sample, not to the writer process.  Keep the worker
                    # usable for the next valid snapshot.
                    ack_queue.put(
                        {
                            "kind": "sample_rejected",
                            "reason": str(exc),
                            "side": str(exc.side),
                            "rejection_reason": str(exc.reason),
                            "age_sec": exc.age_sec,
                        }
                    )
                except BaseException as exc:  # noqa: BLE001
                    if slot_id is not None:
                        ack_queue.put({"kind": "slot_free", "slot_id": slot_id})
                    ack_queue.put(
                        {
                            "kind": "worker_error",
                            "error": f"{active_stage}: {type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                    )
                continue

            if kind == "native_stream_sample":
                active_stage = "worker.native_stream_append"
                try:
                    if native_writer is None:
                        raise RuntimeError(
                            "No native stream episode is active in the process worker."
                        )
                    if (
                        str(message.get("episode_uuid", ""))
                        != native_writer.episode_uuid
                        or int(message.get("legacy_episode_index", -1))
                        != native_writer.legacy_episode_index
                    ):
                        raise ValueError(
                            "Native stream sample does not match the active episode."
                        )
                    native_writer.append(**dict(message["sample"]))
                    ack_queue.put(
                        {
                            "kind": "native_stream_sample_done",
                            "stream_id": str(message["sample"]["stream_id"]),
                            "sample_id": str(message["sample"]["sample_id"]),
                        }
                    )
                except BaseException as exc:  # noqa: BLE001
                    ack_queue.put(
                        {
                            "kind": "worker_error",
                            "error": (
                                f"{active_stage}: {type(exc).__name__}: {exc}"
                            ),
                            "traceback": traceback.format_exc(),
                        }
                    )
                continue

            request_id = message.get("request_id")
            try:
                if kind == "start_episode":
                    episode_uuid = str(message.get("episode_uuid") or "").strip()
                    identity_payload = message.get("episode_identity")
                    episode_identity = (
                        None
                        if identity_payload is None
                        else EpisodeCaptureIdentity.from_payload(identity_payload)
                    )
                    if native_stream_contracts and not episode_uuid:
                        raise ValueError(
                            "Native stream process mode requires an episode UUID."
                        )
                    if not native_stream_contracts and episode_uuid and episode_identity is None:
                        raise ValueError(
                            "episode_uuid was provided without native stream contracts."
                        )
                    if v25p_shadow_config is not None and episode_identity is None:
                        raise ValueError(
                            "v2.5-p native shadow requires the owner's episode identity."
                        )
                    if v25p_shadow_config is None and episode_identity is not None:
                        raise ValueError(
                            "episode_identity was provided without a v2.5-p native shadow."
                        )
                    if episode_identity is not None:
                        if episode_uuid and episode_uuid != episode_identity.episode_uuid:
                            raise ValueError("Episode UUID and owner identity disagree.")
                        if (
                            episode_identity.legacy_episode_index
                            != int(writer.next_episode_index)
                        ):
                            raise ValueError(
                                "Owner identity index does not match the legacy writer."
                            )
                    if native_writer is not None:
                        raise RuntimeError(
                            "Previous native stream episode is still active."
                        )
                    if v25p_native_shadow_identity is not None:
                        raise RuntimeError(
                            "Previous v2.5-p native shadow episode is still active."
                        )
                    v25p_native_shadow_error = ""
                    if v25p_native_shadow is not None and episode_identity is not None:
                        v25p_native_shadow.prepare_episode(episode_identity)
                        v25p_native_shadow_identity = episode_identity
                    try:
                        episode_index = writer.start_episode(
                            metadata=dict(message.get("metadata") or {})
                        )
                    except BaseException:
                        if (
                            v25p_native_shadow is not None
                            and v25p_native_shadow_identity is not None
                        ):
                            v25p_native_shadow.discard_episode(
                                v25p_native_shadow_identity
                            )
                            v25p_native_shadow_identity = None
                        raise
                    if (
                        episode_identity is not None
                        and int(episode_index) != episode_identity.legacy_episode_index
                    ):
                        writer.discard_episode(int(episode_index))
                        assert v25p_native_shadow is not None
                        assert v25p_native_shadow_identity is not None
                        v25p_native_shadow.discard_episode(v25p_native_shadow_identity)
                        v25p_native_shadow_identity = None
                        raise RuntimeError(
                            "Legacy episode index changed after owner identity preparation."
                        )
                    if native_stream_contracts:
                        try:
                            native_writer = NativeStreamEpisodeWriter(
                                writer_config.root_dir,
                                episode_uuid=episode_uuid,
                                legacy_episode_index=int(episode_index),
                                contracts=native_stream_contracts,
                                external_payload_stream_ids=tuple(
                                    external_stream_bindings
                                ),
                            )
                        except BaseException:
                            writer.discard_episode(int(episode_index))
                            if (
                                v25p_native_shadow is not None
                                and v25p_native_shadow_identity is not None
                            ):
                                v25p_native_shadow.discard_episode(
                                    v25p_native_shadow_identity
                                )
                                v25p_native_shadow_identity = None
                            raise
                    if (
                        v25p_native_shadow is not None
                        and v25p_native_shadow_identity is not None
                    ):
                        try:
                            v25p_native_shadow.bind_prepared_episode(
                                v25p_native_shadow_identity,
                                bound_at=time.monotonic(),
                                bound_clock_domain_id="python_monotonic",
                            )
                        except BaseException:
                            writer.discard_episode(int(episode_index))
                            v25p_native_shadow.discard_episode(
                                v25p_native_shadow_identity
                            )
                            v25p_native_shadow_identity = None
                            raise
                    native_ingestion_frozen = False
                    next_external_sequence_by_stream.clear()
                    external_intent_and_sample_frame_by_identity.clear()
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
                elif kind == "reset_temporal_history":
                    builder.reset_temporal_history()
                    result = True
                elif kind == "freeze_native_stream_episode":
                    if native_writer is None:
                        raise RuntimeError(
                            "No native stream episode is active in the process worker."
                        )
                    if not native_writer.requires_post_finalize_seal:
                        raise RuntimeError(
                            "Inline native streams do not use post-finalize freeze."
                        )
                    if (
                        str(message.get("episode_uuid", ""))
                        != native_writer.episode_uuid
                        or int(message.get("legacy_episode_index", -1))
                        != native_writer.legacy_episode_index
                    ):
                        raise ValueError(
                            "Native stream freeze does not match the active episode."
                        )
                    native_writer.freeze_ingestion()
                    native_ingestion_frozen = True
                    result = True
                elif kind == "finalize_episode":
                    if native_writer is not None:
                        if (
                            int(message["episode_index"])
                            != native_writer.legacy_episode_index
                        ):
                            raise ValueError(
                                "Legacy finalize index does not match the active "
                                "native stream episode."
                            )
                        if not native_writer.requires_post_finalize_seal:
                            raise RuntimeError(
                                "Inline native stream episode must seal before "
                                "legacy finalize."
                            )
                        if not native_ingestion_frozen:
                            raise RuntimeError(
                                "External native stream episode must freeze "
                                "before legacy finalize."
                            )
                        report = writer.finalize_episode_with_artifact_report(
                            int(message["episode_index"]),
                            summary=dict(message.get("summary") or {}),
                        )
                        try:
                            receipts = _selected_rgb_artifact_receipts(
                                report,
                                episode_uuid=native_writer.episode_uuid,
                                native_stream_contracts=(
                                    native_stream_contracts
                                ),
                                external_stream_bindings=(
                                    external_stream_bindings
                                ),
                                intent_and_sample_frame_by_identity=(
                                    external_intent_and_sample_frame_by_identity
                                ),
                            )
                            receipt_error = ""
                        except BaseException as exc:  # legacy is finalized
                            receipts = ()
                            receipt_error = f"{type(exc).__name__}: {exc}"
                        result = ProcessLegacyFinalizeResult(
                            episode_index=int(message["episode_index"]),
                            legacy_finalized=True,
                            native_stream_artifact_receipts=receipts,
                            artifact_receipt_error=receipt_error,
                        )
                    else:
                        writer.finalize_episode(
                            int(message["episode_index"]),
                            summary=dict(message.get("summary") or {}),
                        )
                        result = ProcessLegacyFinalizeResult(
                            episode_index=int(message["episode_index"]),
                            legacy_finalized=True,
                        )
                    if v25p_native_shadow is not None:
                        shadow_report = None
                        shadow_error = v25p_native_shadow_error
                        if v25p_native_shadow_identity is not None:
                            try:
                                from .v25p_dual_write_shadow import (
                                    load_finalized_legacy_episode_record,
                                )

                                shadow_report = v25p_native_shadow.finalize_after_legacy(
                                    v25p_native_shadow_identity,
                                    legacy_episode_record=(
                                        load_finalized_legacy_episode_record(
                                            writer_config.root_dir,
                                            int(message["episode_index"]),
                                        )
                                    ),
                                    completed_at=time.time(),
                                    completed_clock_domain_id="local_wall_clock",
                                )
                                if shadow_report.receipt.status != "complete":
                                    shadow_error = str(
                                        shadow_report.receipt.message
                                        or "v2.5-p compatibility projection failed"
                                    )
                            except BaseException as exc:  # legacy finalize succeeded
                                shadow_error = f"finalize:{type(exc).__name__}: {exc}"
                                if (
                                    v25p_native_shadow.active_identity
                                    == v25p_native_shadow_identity
                                ):
                                    try:
                                        v25p_native_shadow.close_unsealed(
                                            v25p_native_shadow_identity
                                        )
                                    except BaseException as close_exc:
                                        shadow_error += (
                                            ";close_unsealed:"
                                            f"{type(close_exc).__name__}: {close_exc}"
                                        )
                            finally:
                                v25p_native_shadow_identity = None
                        result = replace(
                            result,
                            v25p_native_shadow_report=shadow_report,
                            v25p_native_shadow_error=shadow_error,
                        )
                elif kind == "discard_episode":
                    if native_writer is not None:
                        if (
                            int(message["episode_index"])
                            != native_writer.legacy_episode_index
                        ):
                            raise ValueError(
                                "Discard index does not match native stream episode."
                            )
                        native_writer.close_unsealed()
                        native_writer = None
                        native_ingestion_frozen = False
                        next_external_sequence_by_stream.clear()
                        external_intent_and_sample_frame_by_identity.clear()
                    if (
                        v25p_native_shadow is not None
                        and v25p_native_shadow_identity is not None
                    ):
                        v25p_native_shadow.discard_episode(
                            v25p_native_shadow_identity
                        )
                        v25p_native_shadow_identity = None
                        v25p_native_shadow_error = ""
                    result = bool(writer.discard_episode(int(message["episode_index"])))
                elif kind == "native_stream_note_drops":
                    if native_writer is None:
                        raise RuntimeError(
                            "No native stream episode is active in the process worker."
                        )
                    if (
                        str(message.get("episode_uuid", ""))
                        != native_writer.episode_uuid
                        or int(message.get("legacy_episode_index", -1))
                        != native_writer.legacy_episode_index
                    ):
                        raise ValueError(
                            "Native stream drops do not match the active episode."
                        )
                    for item in tuple(message.get("drop_counts") or ()):
                        stream_id = str(item["stream_id"])
                        reason = str(item["reason"])
                        count = int(item["count"])
                        if count <= 0:
                            raise ValueError(
                                "Native stream drop count must be positive."
                            )
                        for _ in range(count):
                            native_writer.note_drop(stream_id, reason=reason)
                    result = True
                elif kind == "seal_native_stream_episode":
                    if native_writer is None:
                        raise RuntimeError(
                            "No native stream episode is active in the process worker."
                        )
                    if (
                        str(message.get("episode_uuid", ""))
                        != native_writer.episode_uuid
                        or int(message.get("legacy_episode_index", -1))
                        != native_writer.legacy_episode_index
                    ):
                        raise ValueError(
                            "Native stream seal does not match the active episode."
                        )
                    if native_writer.requires_post_finalize_seal:
                        raise RuntimeError(
                            "External native streams must seal after legacy finalize."
                        )
                    result = native_writer.seal(
                        sealed_at=float(message["sealed_at"]),
                        sealed_clock_domain_id=str(
                            message["sealed_clock_domain_id"]
                        ),
                    )
                    native_writer = None
                    native_ingestion_frozen = False
                    next_external_sequence_by_stream.clear()
                    external_intent_and_sample_frame_by_identity.clear()
                elif kind == "seal_native_stream_episode_after_legacy_finalize":
                    if native_writer is None:
                        raise RuntimeError(
                            "No native stream episode is active in the process worker."
                        )
                    if (
                        str(message.get("episode_uuid", ""))
                        != native_writer.episode_uuid
                        or int(message.get("legacy_episode_index", -1))
                        != native_writer.legacy_episode_index
                    ):
                        raise ValueError(
                            "Post-finalize native stream seal does not match "
                            "the active episode."
                        )
                    if (
                        not native_writer.requires_post_finalize_seal
                        or not native_ingestion_frozen
                    ):
                        raise RuntimeError(
                            "Post-finalize native stream seal requires a frozen "
                            "external capture."
                        )
                    result = native_writer.seal_after_legacy_finalize(
                        tuple(message.get("receipts") or ()),
                        sealed_at=float(message["sealed_at"]),
                        sealed_clock_domain_id=str(
                            message["sealed_clock_domain_id"]
                        ),
                    )
                    native_writer = None
                    native_ingestion_frozen = False
                    next_external_sequence_by_stream.clear()
                    external_intent_and_sample_frame_by_identity.clear()
                elif kind == "close_native_stream_episode":
                    if native_writer is not None:
                        if (
                            str(message.get("episode_uuid", ""))
                            != native_writer.episode_uuid
                            or int(message.get("legacy_episode_index", -1))
                            != native_writer.legacy_episode_index
                        ):
                            raise ValueError(
                                "Native stream close does not match the active episode."
                            )
                        native_writer.close_unsealed()
                        native_writer = None
                        native_ingestion_frozen = False
                        next_external_sequence_by_stream.clear()
                        external_intent_and_sample_frame_by_identity.clear()
                    result = True
                elif kind == "reload_finalized_dataset_state":
                    writer.reload_finalized_dataset_state()
                    builder.set_dataset_position(
                        episode_index=int(writer.next_episode_index),
                        sample_index=int(writer.next_sample_index),
                    )
                    result = {
                        "next_episode_index": int(writer.next_episode_index),
                        "next_sample_index": int(writer.next_sample_index),
                    }
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
                    if native_writer is not None:
                        native_writer.close_unsealed()
                        native_writer = None
                    native_ingestion_frozen = False
                    next_external_sequence_by_stream.clear()
                    external_intent_and_sample_frame_by_identity.clear()
                    if (
                        v25p_native_shadow is not None
                        and v25p_native_shadow_identity is not None
                    ):
                        v25p_native_shadow.close_unsealed(
                            v25p_native_shadow_identity
                        )
                        v25p_native_shadow_identity = None
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
        if native_writer is not None:
            native_writer.close_unsealed()
        if (
            v25p_native_shadow is not None
            and v25p_native_shadow_identity is not None
        ):
            v25p_native_shadow.close_unsealed(v25p_native_shadow_identity)
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
        native_stream_contracts: Mapping[str, NativeStreamContract] | None = None,
        native_external_stream_bindings: Mapping[
            str, ProcessExternalNativeStreamBinding
        ]
        | None = None,
        v25p_shadow_config: ProcessV25pShadowConfig | None = None,
    ) -> None:
        self.writer_config = writer_config
        if v25p_shadow_config is not None and not isinstance(
            v25p_shadow_config, ProcessV25pShadowConfig
        ):
            raise TypeError("v25p_shadow_config must be ProcessV25pShadowConfig.")
        self.v25p_shadow_config = v25p_shadow_config
        self.native_stream_contracts = dict(
            sorted(dict(native_stream_contracts or {}).items())
        )
        for stream_id, contract in self.native_stream_contracts.items():
            if not isinstance(contract, NativeStreamContract):
                raise TypeError(
                    "native_stream_contracts must contain NativeStreamContract values."
                )
            if str(stream_id) != contract.stream_id:
                raise ValueError(
                    "Native stream contract mapping key must match contract.stream_id."
                )
        self.native_external_stream_bindings = dict(
            sorted(dict(native_external_stream_bindings or {}).items())
        )
        seen_external_sources: set[tuple[str, str]] = set()
        for stream_id, binding in self.native_external_stream_bindings.items():
            if not isinstance(binding, ProcessExternalNativeStreamBinding):
                raise TypeError(
                    "native_external_stream_bindings must contain "
                    "ProcessExternalNativeStreamBinding values."
                )
            if str(stream_id) != binding.stream_id:
                raise ValueError(
                    "External stream binding key must match binding.stream_id."
                )
            contract = self.native_stream_contracts.get(stream_id)
            if contract is None:
                raise ValueError(
                    "External stream binding requires a matching native "
                    "stream contract."
                )
            if contract.modality != "selected_rgb":
                raise ValueError(
                    "selected_rgb binding requires contract.modality="
                    "'selected_rgb'."
                )
            if (
                tuple(contract.shape[-1:]) != (3,)
                or np.dtype(contract.dtype) != np.dtype(np.uint8)
                or contract.units != "rgb_u8"
            ):
                raise ValueError(
                    "selected_rgb contract requires three-channel uint8 "
                    "shape and units='rgb_u8'."
                )
            if (
                contract.encoding
                != f"mp4/{writer_config.video_codec}"
                or contract.compression != "none"
            ):
                raise ValueError(
                    "selected_rgb contract encoding/compression must match "
                    "the legacy video writer."
                )
            if not bool(writer_config.write_video):
                raise ValueError(
                    "selected_rgb external binding requires write_video=True."
                )
            source_identity = (binding.source_kind, binding.source_key)
            if source_identity in seen_external_sources:
                raise ValueError(
                    "One specialized writer source cannot bind multiple "
                    "external native streams."
                )
            seen_external_sources.add(source_identity)
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
                self.v25p_shadow_config,
                self.native_stream_contracts,
                self.native_external_stream_bindings,
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
        self._submitted_native_stream_samples = 0
        self._completed_native_stream_samples = 0
        self._rejected_native_stream_samples = 0
        self._active_native_episode_uuid: str | None = None
        self._active_native_episode_index: int | None = None
        self._active_native_episode_frozen = False
        self._queue_high_water = 0
        self._active_episode_frame_count = 0
        self._worker_error: str | None = None
        self._worker_traceback: str | None = None
        self._sample_rejection_count = 0
        self._pending_sample_rejections: deque[str] = deque()
        self._sample_acceptance_receipt_buffer = SampleAcceptanceReceiptBuffer(
            capacity=int(config.sample_acceptance_receipt_capacity)
        )
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
        deadline = time.monotonic() + float(self.config.startup_timeout_sec)
        ready: Any
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                self.close(force=True)
                raise RuntimeError("collection worker did not become ready")
            try:
                ready = self._reply_queue.get(timeout=min(0.05, remaining))
                break
            except queue.Empty:
                if self._process.is_alive():
                    continue
                exitcode = self._process.exitcode
                self.close(force=True)
                raise RuntimeError(
                    "collection worker exited before becoming ready "
                    f"(exitcode={exitcode})"
                )
        if not isinstance(ready, Mapping) or not bool(ready.get("ok")):
            worker_traceback = (
                str(ready.get("traceback") or "").strip()
                if isinstance(ready, Mapping)
                else ""
            )
            if worker_traceback:
                print(
                    "[CollectionWorker] startup failed:\n"
                    f"{worker_traceback}",
                    file=sys.stderr,
                    flush=True,
                )
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
                if self._sample_acceptance_receipt_buffer.capacity > 0:
                    self._sample_acceptance_receipt_buffer.append(
                        AcceptedSampleReceipt(
                            episode_index=int(message["episode_index"]),
                            sample_index=int(message["sample_index"]),
                            episode_frame_index=int(
                                message["episode_frame_index"]
                            ),
                            sample_timestamp=float(message["sample_timestamp"]),
                        )
                    )
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
            elif kind == "native_stream_sample_done":
                self._completed_native_stream_samples += 1
            elif kind == "sample_rejected":
                reason = str(message.get("reason") or "collection sample rejected")
                self._sample_rejection_count += 1
                self._pending_sample_rejections.append(reason)
            elif kind == "worker_error":
                if self._worker_error is None:
                    self._worker_error = str(
                        message.get("error") or "collection worker error"
                    )
                    worker_traceback = str(message.get("traceback") or "").strip()
                    self._worker_traceback = worker_traceback or None
                    if self._worker_traceback:
                        print(
                            "[CollectionWorker] sample processing failed:\n"
                            f"{self._worker_traceback}",
                            file=sys.stderr,
                            flush=True,
                        )

    def consume_sample_rejection(self) -> str | None:
        """Return one nonfatal sample-contract rejection, if available."""

        self._drain_acks()
        if not self._pending_sample_rejections:
            return None
        return self._pending_sample_rejections.popleft()

    def drain_sample_acceptance_receipts(
        self,
    ) -> tuple[AcceptedSampleReceipt, ...]:
        """Drain writer-terminal sample receipts in acceptance order.

        Receipt retention is explicitly enabled with
        ``sample_acceptance_receipt_capacity``.  Overflow makes the exact
        first-sample boundary unknowable, so callers fail closed rather than
        silently using a later sample.
        """

        self._drain_acks()
        return self._sample_acceptance_receipt_buffer.drain()

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
        # ``multiprocessing.Queue.put`` serializes on a background feeder
        # thread.  Without this synchronous preflight, an unpicklable runtime
        # metadata value is only printed by that thread and the caller sees a
        # misleading request timeout.  Use multiprocessing's own serializer so
        # the check has exactly the same boundary semantics as the real queue.
        try:
            ForkingPickler.dumps(message)
        except BaseException as exc:  # noqa: BLE001 - preserve serializer detail.
            raise TypeError(
                f"collection worker request {kind!r} is not serializable: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
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
                    worker_traceback = str(reply.get("traceback") or "").strip()
                    if worker_traceback:
                        print(
                            f"[CollectionWorker] request {kind!r} failed:\n"
                            f"{worker_traceback}",
                            file=sys.stderr,
                            flush=True,
                        )
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

    def start_episode(
        self,
        metadata: Mapping[str, Any],
        *,
        episode_uuid: str | None = None,
        episode_identity: EpisodeCaptureIdentity | None = None,
    ) -> int:
        normalized_uuid = str(episode_uuid or "").strip()
        if self.native_stream_contracts and not normalized_uuid:
            raise ValueError(
                "episode_uuid is required when native streams are configured."
            )
        if (
            not self.native_stream_contracts
            and normalized_uuid
            and episode_identity is None
        ):
            raise ValueError(
                "episode_uuid is only accepted when native streams are configured."
            )
        if self.v25p_shadow_config is not None and episode_identity is None:
            raise ValueError(
                "episode_identity is required when the v2.5-p shadow is configured."
            )
        if self.v25p_shadow_config is None and episode_identity is not None:
            raise ValueError(
                "episode_identity is only accepted when the v2.5-p shadow is configured."
            )
        if episode_identity is not None:
            if normalized_uuid and normalized_uuid != episode_identity.episode_uuid:
                raise ValueError("Episode UUID and owner identity disagree.")
            normalized_uuid = episode_identity.episode_uuid
        episode_index = int(
            self._request(
                "start_episode",
                metadata=dict(metadata),
                episode_uuid=normalized_uuid or None,
                episode_identity=(
                    None
                    if episode_identity is None
                    else episode_identity.to_payload()
                ),
            )
        )
        self.next_episode_index = episode_index + 1
        self._active_episode_frame_count = 0
        self._active_native_episode_uuid = (
            normalized_uuid if self.native_stream_contracts else None
        )
        self._active_native_episode_index = (
            episode_index if self.native_stream_contracts else None
        )
        self._active_native_episode_frozen = False
        return episode_index

    def create_native_stream_capture(
        self,
        prepared: Any,
    ) -> ProcessNativeStreamEpisodeCapture:
        if not self.native_stream_contracts:
            raise RuntimeError(
                "Process collection worker has no native stream contracts."
            )
        return ProcessNativeStreamEpisodeCapture(
            self,
            episode_uuid=str(prepared.episode_uuid),
            legacy_episode_index=int(prepared.legacy_episode_index),
        )

    def submit_native_stream_sample(
        self,
        *,
        episode_uuid: str,
        legacy_episode_index: int,
        stream_id: str,
        sample_id: str,
        **sample: Any,
    ) -> ProcessNativeStreamSubmissionReceipt:
        self._raise_if_failed()
        normalized_uuid = str(episode_uuid)
        normalized_index = int(legacy_episode_index)
        normalized_stream_id = str(stream_id)
        normalized_sample_id = str(sample_id)
        if (
            normalized_uuid != self._active_native_episode_uuid
            or normalized_index != self._active_native_episode_index
        ):
            raise ValueError(
                "Native stream submission does not match the active process episode."
            )
        if normalized_stream_id not in self.native_stream_contracts:
            raise KeyError(f"Unknown native stream {normalized_stream_id!r}.")
        sample_payload = {
            "stream_id": normalized_stream_id,
            "sample_id": normalized_sample_id,
            **sample,
        }
        message = {
            "kind": "native_stream_sample",
            "episode_uuid": normalized_uuid,
            "legacy_episode_index": normalized_index,
            "sample": sample_payload,
        }
        try:
            encoded = ForkingPickler.dumps(message)
        except BaseException as exc:  # noqa: BLE001
            raise TypeError(
                "native stream process sample is not serializable: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        message_bytes = len(encoded)
        if message_bytes > int(self.config.native_inline_message_max_bytes):
            self._rejected_native_stream_samples += 1
            return ProcessNativeStreamSubmissionReceipt(
                episode_uuid=normalized_uuid,
                legacy_episode_index=normalized_index,
                stream_id=normalized_stream_id,
                sample_id=normalized_sample_id,
                accepted=False,
                rejection_reason="inline_message_too_large",
                serialized_message_bytes=message_bytes,
            )
        try:
            self._input_queue.put_nowait(message)
        except queue.Full:
            self._rejected_native_stream_samples += 1
            return ProcessNativeStreamSubmissionReceipt(
                episode_uuid=normalized_uuid,
                legacy_episode_index=normalized_index,
                stream_id=normalized_stream_id,
                sample_id=normalized_sample_id,
                accepted=False,
                rejection_reason="process_queue_busy",
                serialized_message_bytes=message_bytes,
            )
        self._submitted_native_stream_samples += 1
        try:
            self._queue_high_water = max(
                self._queue_high_water, int(self._input_queue.qsize())
            )
        except (NotImplementedError, OSError):
            pass
        return ProcessNativeStreamSubmissionReceipt(
            episode_uuid=normalized_uuid,
            legacy_episode_index=normalized_index,
            stream_id=normalized_stream_id,
            sample_id=normalized_sample_id,
            accepted=True,
            rejection_reason=None,
            serialized_message_bytes=message_bytes,
        )

    def seal_native_stream_episode(
        self,
        *,
        episode_uuid: str,
        legacy_episode_index: int,
        sealed_at: float,
        sealed_clock_domain_id: str,
        drop_counts: Mapping[tuple[str, str], int] | None = None,
    ) -> NativeStreamEpisodeManifest:
        normalized_uuid = str(episode_uuid)
        normalized_index = int(legacy_episode_index)
        if (
            normalized_uuid != self._active_native_episode_uuid
            or normalized_index != self._active_native_episode_index
        ):
            raise ValueError(
                "Native stream seal does not match the active process episode."
            )
        self._submit_native_stream_drop_counts(
            episode_uuid=normalized_uuid,
            legacy_episode_index=normalized_index,
            drop_counts=drop_counts,
        )
        manifest = self._request(
            "seal_native_stream_episode",
            episode_uuid=normalized_uuid,
            legacy_episode_index=normalized_index,
            sealed_at=float(sealed_at),
            sealed_clock_domain_id=str(sealed_clock_domain_id),
        )
        if not isinstance(manifest, NativeStreamEpisodeManifest):
            raise RuntimeError(
                "Process worker returned an invalid native stream manifest."
            )
        self._active_native_episode_uuid = None
        self._active_native_episode_index = None
        self._active_native_episode_frozen = False
        self._drain_acks()
        return manifest

    def _submit_native_stream_drop_counts(
        self,
        *,
        episode_uuid: str,
        legacy_episode_index: int,
        drop_counts: Mapping[tuple[str, str], int] | None,
    ) -> None:
        normalized_drops = tuple(
            {
                "stream_id": str(stream_id),
                "reason": str(reason),
                "count": int(count),
            }
            for (stream_id, reason), count in sorted(
                dict(drop_counts or {}).items()
            )
            if int(count) > 0
        )
        if normalized_drops:
            self._request(
                "native_stream_note_drops",
                episode_uuid=str(episode_uuid),
                legacy_episode_index=int(legacy_episode_index),
                drop_counts=normalized_drops,
            )

    def freeze_native_stream_episode(
        self,
        *,
        episode_uuid: str,
        legacy_episode_index: int,
        drop_counts: Mapping[tuple[str, str], int] | None = None,
    ) -> None:
        normalized_uuid = str(episode_uuid)
        normalized_index = int(legacy_episode_index)
        if not self.native_external_stream_bindings:
            raise RuntimeError(
                "Process collection worker has no external native streams."
            )
        if (
            normalized_uuid != self._active_native_episode_uuid
            or normalized_index != self._active_native_episode_index
        ):
            raise ValueError(
                "Native stream freeze does not match the active process episode."
            )
        if self._active_native_episode_frozen:
            return
        self._submit_native_stream_drop_counts(
            episode_uuid=normalized_uuid,
            legacy_episode_index=normalized_index,
            drop_counts=drop_counts,
        )
        self._request(
            "freeze_native_stream_episode",
            episode_uuid=normalized_uuid,
            legacy_episode_index=normalized_index,
        )
        self._active_native_episode_frozen = True

    def seal_native_stream_episode_after_legacy_finalize(
        self,
        *,
        episode_uuid: str,
        legacy_episode_index: int,
        receipts: Iterable[NativeStreamExternalArtifactReceipt],
        sealed_at: float,
        sealed_clock_domain_id: str,
    ) -> NativeStreamEpisodeManifest:
        normalized_uuid = str(episode_uuid)
        normalized_index = int(legacy_episode_index)
        if (
            normalized_uuid != self._active_native_episode_uuid
            or normalized_index != self._active_native_episode_index
        ):
            raise ValueError(
                "Post-finalize native stream seal does not match the active "
                "process episode."
            )
        if not self._active_native_episode_frozen:
            raise RuntimeError(
                "External native stream episode must freeze before seal."
            )
        manifest = self._request(
            "seal_native_stream_episode_after_legacy_finalize",
            episode_uuid=normalized_uuid,
            legacy_episode_index=normalized_index,
            receipts=tuple(receipts),
            sealed_at=float(sealed_at),
            sealed_clock_domain_id=str(sealed_clock_domain_id),
        )
        if not isinstance(manifest, NativeStreamEpisodeManifest):
            raise RuntimeError(
                "Process worker returned an invalid native stream manifest."
            )
        self._active_native_episode_uuid = None
        self._active_native_episode_index = None
        self._active_native_episode_frozen = False
        self._drain_acks()
        return manifest

    def close_native_stream_episode(
        self,
        *,
        episode_uuid: str,
        legacy_episode_index: int,
    ) -> None:
        normalized_uuid = str(episode_uuid)
        normalized_index = int(legacy_episode_index)
        if self._active_native_episode_uuid is None:
            return
        if (
            normalized_uuid != self._active_native_episode_uuid
            or normalized_index != self._active_native_episode_index
        ):
            raise ValueError(
                "Native stream close does not match the active process episode."
            )
        self._request(
            "close_native_stream_episode",
            episode_uuid=normalized_uuid,
            legacy_episode_index=normalized_index,
        )
        self._active_native_episode_uuid = None
        self._active_native_episode_index = None
        self._active_native_episode_frozen = False

    def write_device_config(self, device_name: str, payload: Mapping[str, Any]) -> None:
        self._request(
            "write_device_config",
            device_name=str(device_name),
            payload=dict(payload),
        )

    def flush(self) -> None:
        self._request("flush")
        # Sample terminal acknowledgements and synchronous request replies use
        # separate multiprocessing queues.  The worker enqueues the sample ack
        # before replying to this flush request, but independent queue feeder
        # threads may make the reply visible first.  A single non-blocking
        # drain therefore does not provide flush semantics.  Wait until every
        # sample accepted by the parent has a terminal ack, while continuing
        # to surface worker death/errors fail closed.
        deadline = time.monotonic() + float(self.config.request_timeout_sec)
        while True:
            self._drain_acks()
            self._raise_if_failed()
            samples_settled = (
                self._completed_samples + self._sample_rejection_count
                >= self._submitted_samples
            )
            native_samples_settled = (
                self._completed_native_stream_samples
                >= self._submitted_native_stream_samples
            )
            if samples_settled and native_samples_settled:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(
                    "collection worker flush timed out waiting for terminal "
                    "sample acknowledgements: "
                    f"samples={self._completed_samples + self._sample_rejection_count}/"
                    f"{self._submitted_samples} native_samples="
                    f"{self._completed_native_stream_samples}/"
                    f"{self._submitted_native_stream_samples}"
                )
            time.sleep(min(0.001, remaining))

    def reset_temporal_history(self) -> None:
        self._request("reset_temporal_history")

    def finalize_episode(
        self,
        episode_index: int,
        *,
        summary: Mapping[str, Any],
    ) -> ProcessLegacyFinalizeResult:
        if self._active_native_episode_uuid is not None:
            if not self.native_external_stream_bindings:
                raise RuntimeError(
                    "Inline native stream episode must seal before legacy finalize."
                )
            if not self._active_native_episode_frozen:
                raise RuntimeError(
                    "External native stream episode must freeze before "
                    "legacy finalize."
                )
        result = self._request(
            "finalize_episode",
            episode_index=int(episode_index),
            summary=dict(summary),
        )
        if not isinstance(result, ProcessLegacyFinalizeResult):
            raise RuntimeError(
                "Process worker returned an invalid legacy finalize result."
            )
        return result

    def discard_episode(self, episode_index: int) -> bool:
        discarded = bool(
            self._request("discard_episode", episode_index=int(episode_index))
        )
        if int(episode_index) == self._active_native_episode_index:
            self._active_native_episode_uuid = None
            self._active_native_episode_index = None
            self._active_native_episode_frozen = False
        return discarded

    def reload_finalized_dataset_state(self) -> None:
        result = self._request("reload_finalized_dataset_state")
        if not isinstance(result, Mapping):
            raise RuntimeError("collection worker returned invalid dataset reload state")
        self.next_episode_index = int(result.get("next_episode_index", 0))
        self.next_sample_index = int(result.get("next_sample_index", 0))

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
            sample_rejections=int(self._sample_rejection_count),
            queue_capacity=int(self.config.queue_size),
            queue_high_water=int(self._queue_high_water),
            slots_available=len(self._slots.available),
            slot_count=len(self._slots.blocks),
            worker_alive=bool(self._started and self._process.is_alive()),
            worker_error=self._worker_error,
            worker_traceback=self._worker_traceback,
            active_episode_frame_count=int(self._active_episode_frame_count),
            sample_process_ms_p50=_percentile(values, 50),
            sample_process_ms_p95=_percentile(values, 95),
            timing_ms=timing_ms,
            shared_payload_bytes=_metric_summary(
                list(self._shared_payload_bytes)
            ),
            submitted_native_stream_samples=int(
                self._submitted_native_stream_samples
            ),
            completed_native_stream_samples=int(
                self._completed_native_stream_samples
            ),
            rejected_native_stream_samples=int(
                self._rejected_native_stream_samples
            ),
        )

    def quick_status(self) -> dict[str, Any]:
        """Cheap status for the 60 Hz frontend; avoids percentile recomputation."""

        self._drain_acks()
        status = {
            "submitted_samples": int(self._submitted_samples),
            "completed_samples": int(self._completed_samples),
            "rejected_busy": int(self._rejected_busy),
            "queue_high_water": int(self._queue_high_water),
            "slots_available": len(self._slots.available),
            "slot_count": len(self._slots.blocks),
            "worker_alive": bool(self._started and self._process.is_alive()),
            "worker_error": self._worker_error,
            "worker_traceback": self._worker_traceback,
            "sample_rejection_count": int(self._sample_rejection_count),
            "pending_sample_rejections": len(self._pending_sample_rejections),
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
        if self.native_stream_contracts:
            status.update(
                {
                    "submitted_native_stream_samples": int(
                        self._submitted_native_stream_samples
                    ),
                    "completed_native_stream_samples": int(
                        self._completed_native_stream_samples
                    ),
                    "rejected_native_stream_samples": int(
                        self._rejected_native_stream_samples
                    ),
                }
            )
        return status

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
    "ProcessExternalNativeStreamBinding",
    "ProcessLegacyFinalizeResult",
    "ProcessNativeStreamEpisodeCapture",
    "ProcessNativeStreamSubmissionReceipt",
    "ProcessV25pShadowConfig",
    "SharedArrayRef",
    "camera_shared_slot_bytes",
]

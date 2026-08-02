"""Default-off component-stream episode persistence.

The application recorder remains the only episode owner.  This subordinate
store never starts, stops, discards, or finalizes a compatibility writer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import threading
import time
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash, canonical_loads
from dataset_contracts.embodiment import (
    CaptureProjectionProfileLike,
    EmbodimentCaptureProfileLike,
)
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.task_semantics import (
    LongHorizonTaskProfile,
    SubtaskAttemptEvent,
    TaskSemanticsEpisodeManifest,
    validate_subtask_attempt_events,
)
from dataset_contracts.v25p import (
    V25P_EPISODE_IDENTITY_FILE_DOMAIN_V1,
    V25P_TASK_EVENT_FILE_DOMAIN_V1,
    V25P_TASK_MANIFEST_FILE_DOMAIN_V1,
)
from dataset_contracts.v25p_native import (
    V25P_NATIVE_BINDING_FILE_DOMAIN_V1,
    V25P_NATIVE_CAPTURE_MANIFEST_FILE_DOMAIN_V1,
    V25P_NATIVE_DATASET_MANIFEST_FILE_DOMAIN_V2,
    V25P_NATIVE_EPISODE_MANIFEST_FILE_DOMAIN_V2,
    V25P_NATIVE_EPISODE_RELATIVE_ROOT_V1,
    V25P_NATIVE_PROFILE_FILE_DOMAIN_V1,
    V25P_NATIVE_PROFILE_RELATIVE_ROOT_V1,
    V25P_NATIVE_RECORD_BUNDLE_HASH_DOMAIN_V1,
    V25P_NATIVE_RELATIVE_ROOT_V1,
    V25P_NATIVE_SAMPLE_BUNDLE_HASH_DOMAIN_V1,
    V25P_NATIVE_COMPONENT_RECORD_FILE_DOMAIN_V1,
    V25P_NATIVE_CAUSAL_SAMPLE_FILE_DOMAIN_V1,
    V25pCausalSample,
    V25pComponentRecord,
    V25pComponentStreamProfileLike,
    V25pExternalPayloadReference,
    V25pNativeCaptureManifest,
    V25pNativeDatasetManifest,
    V25pNativeEmbodimentBinding,
    V25pNativeEpisodeManifest,
)
from dataset_contracts.v25p_projection import (
    V25P_COMPATIBILITY_PROJECTION_RECEIPT_FILE_DOMAIN_V1,
    V25pCompatibilityProjectionReceipt,
)
from dataset_contracts.v25p_native_storage import (
    V25P_PARQUET_CHUNK_LAYOUT_V1,
    V25P_RECORD_CJSON_LAYOUT_V1,
)
from dataset_interface.v25p_projection_plan import (
    V25pV23ProjectionError,
    plan_v23_compatibility_projection,
    verify_complete_v23_projection_receipt,
)
from dataset_interface.v25p_component_profile import (
    validate_v25p_embodiment_stream_binding,
)

from ..immutable_canonical_store import (
    fsync_directory,
    publish_immutable_canonical,
)
from .parquet import ParquetChunkEpisodeWriter


@dataclass(frozen=True)
class ComponentStreamInput:
    values: object | None
    effective_at: float
    available_at: float
    effective_clock_domain_id: str
    available_clock_domain_id: str
    payload_reference: V25pExternalPayloadReference | None = None
    valid: bool = True
    invalid_reason: str = ""
    source_record_id: str | None = None


@dataclass(frozen=True)
class ComponentCaptureSealReport:
    episode_uuid: str
    legacy_episode_index: int
    sample_count: int
    native_capture_manifest_hash: str
    task_semantics_manifest_hash: str | None
    auto_closed_attempt_count: int
    native_storage_manifest_hash: str | None = None


@dataclass(frozen=True)
class ComponentCaptureCommitReport:
    episode_uuid: str
    legacy_episode_index: int
    native_capture_manifest_hash: str
    episode_manifest_hash: str
    compatibility_projection_receipt_hashes: tuple[str, ...]
    v24_capture_commit_hash: str | None
    episode_manifest_relative_path: str


@dataclass
class _ActiveNativeEpisode:
    identity: EpisodeCaptureIdentity
    binding: V25pNativeEmbodimentBinding | None = None
    records_by_stream: dict[str, list[V25pComponentRecord]] | None = None
    samples: list[V25pCausalSample] | None = None
    task_events: tuple[SubtaskAttemptEvent, ...] = ()
    task_manifest: TaskSemanticsEpisodeManifest | None = None
    native_manifest: V25pNativeCaptureManifest | None = None
    parquet_writer: ParquetChunkEpisodeWriter | None = None
    sample_count: int = 0
    record_count_by_stream: dict[str, int] | None = None
    latest_cut_by_clock: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.records_by_stream is None:
            self.records_by_stream = {}
        if self.samples is None:
            self.samples = []
        if self.record_count_by_stream is None:
            self.record_count_by_stream = {}
        if self.latest_cut_by_clock is None:
            self.latest_cut_by_clock = {}


def validate_component_stream_profile(
    embodiment_profile: EmbodimentCaptureProfileLike,
    stream_profile: V25pComponentStreamProfileLike,
) -> None:
    """Compatibility entrypoint for the dataset-interface binding validator."""

    validate_v25p_embodiment_stream_binding(
        embodiment_profile=embodiment_profile,
        stream_profile=stream_profile,
    )


class ComponentCaptureEpisodeStore:
    """Immutable native v2.5-p stream store driven by an external owner."""

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        dataset_id: str,
        embodiment_profile: EmbodimentCaptureProfileLike,
        component_stream_profile: V25pComponentStreamProfileLike,
        task_profile: LongHorizonTaskProfile | None = None,
        compatibility_projection_profiles: tuple[CaptureProjectionProfileLike, ...] = (),
        defer_sample_file_writes: bool = False,
        physical_storage_layout: str = V25P_RECORD_CJSON_LAYOUT_V1,
        parquet_chunk_rows: int = 128,
        parquet_compression: str = "none",
        clock=time.time,
        clock_domain_id: str = "local_wall_clock",
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.root = self.dataset_root / V25P_NATIVE_RELATIVE_ROOT_V1
        self.profile_root = self.dataset_root / V25P_NATIVE_PROFILE_RELATIVE_ROOT_V1
        self.episode_root = self.dataset_root / V25P_NATIVE_EPISODE_RELATIVE_ROOT_V1
        self.dataset_id = str(dataset_id).strip()
        self.embodiment_profile = embodiment_profile
        self.component_stream_profile = component_stream_profile
        self.task_profile = task_profile
        self.compatibility_projection_profiles = tuple(
            compatibility_projection_profiles
        )
        self.defer_sample_file_writes = bool(defer_sample_file_writes)
        self.physical_storage_layout = str(physical_storage_layout).strip()
        self.parquet_chunk_rows = int(parquet_chunk_rows)
        self.parquet_compression = str(parquet_compression).strip().lower()
        self.clock = clock
        self.clock_domain_id = str(clock_domain_id).strip()
        if not self.dataset_id or not self.clock_domain_id:
            raise ValueError("Native v2.5-p dataset and clock identity must be non-empty.")
        if self.physical_storage_layout not in {
            V25P_RECORD_CJSON_LAYOUT_V1,
            V25P_PARQUET_CHUNK_LAYOUT_V1,
        }:
            raise ValueError("Unsupported native v2.5-p physical storage layout.")
        if (
            self.physical_storage_layout == V25P_PARQUET_CHUNK_LAYOUT_V1
            and self.defer_sample_file_writes
        ):
            raise ValueError(
                "Parquet chunks already buffer writes; defer_sample_file_writes "
                "only applies to record_cjson_v1."
            )
        validate_component_stream_profile(
            embodiment_profile,
            component_stream_profile,
        )
        hashes = tuple(
            item.semantic_content_hash for item in self.compatibility_projection_profiles
        )
        if len(hashes) != len(set(hashes)):
            raise ValueError("Compatibility projection profiles must be unique.")
        self._projection_by_hash = {
            item.semantic_content_hash: item
            for item in self.compatibility_projection_profiles
        }
        self._lock = threading.RLock()
        self._active: _ActiveNativeEpisode | None = None
        self.dataset_manifest = self._bind_dataset_contracts()

    def _bind_dataset_contracts(self) -> V25pNativeDatasetManifest:
        expected = {
            "dataset_id": self.dataset_id,
            "embodiment_profile_hashes": (
                self.embodiment_profile.semantic_content_hash,
            ),
            "component_stream_profile_hashes": (
                self.component_stream_profile.semantic_content_hash,
            ),
            "long_horizon_task_profile_hashes": (
                ()
                if self.task_profile is None
                else (self.task_profile.semantic_content_hash,)
            ),
            "compatibility_projection_profile_hashes": tuple(
                item.semantic_content_hash
                for item in self.compatibility_projection_profiles
            ),
        }
        path = self.root / "dataset_manifest.cjson"
        if path.exists():
            payload = canonical_loads(
                path.read_bytes(),
                expected_domain=V25P_NATIVE_DATASET_MANIFEST_FILE_DOMAIN_V2,
            )
            if not isinstance(payload, Mapping):
                raise ValueError("Native v2.5-p dataset manifest is not a mapping.")
            manifest = V25pNativeDatasetManifest.from_payload(payload)
            for name, value in expected.items():
                if getattr(manifest, name) != value:
                    raise ValueError(
                        f"Existing native v2.5-p dataset manifest differs at {name}."
                    )
        else:
            manifest = V25pNativeDatasetManifest(
                **expected,
                created_at=float(self.clock()),
                created_clock_domain_id=self.clock_domain_id,
            )

        profiles: list[tuple[str, str, Mapping[str, Any]]] = [
            (
                "embodiment",
                self.embodiment_profile.semantic_content_hash,
                self.embodiment_profile.to_payload(),
            ),
            (
                "component_stream",
                self.component_stream_profile.semantic_content_hash,
                self.component_stream_profile.to_payload(),
            ),
        ]
        if self.task_profile is not None:
            profiles.append(
                (
                    "task",
                    self.task_profile.semantic_content_hash,
                    self.task_profile.to_payload(),
                )
            )
        profiles.extend(
            (
                "compatibility_projection",
                item.semantic_content_hash,
                item.to_payload(),
            )
            for item in self.compatibility_projection_profiles
        )
        for kind, content_hash, payload in profiles:
            publish_immutable_canonical(
                self.profile_root / kind / f"{content_hash}.cjson",
                payload,
                domain=V25P_NATIVE_PROFILE_FILE_DOMAIN_V1,
            )
        if not path.exists():
            publish_immutable_canonical(
                path,
                manifest.to_payload(),
                domain=V25P_NATIVE_DATASET_MANIFEST_FILE_DOMAIN_V2,
            )
        return manifest

    def _episode_dir(self, identity: EpisodeCaptureIdentity) -> Path:
        return self.episode_root / identity.episode_uuid

    @property
    def active_identity(self) -> EpisodeCaptureIdentity | None:
        with self._lock:
            return None if self._active is None else self._active.identity

    def prepare_episode(self, identity: EpisodeCaptureIdentity) -> None:
        with self._lock:
            if self._active is not None:
                raise RuntimeError("A native v2.5-p episode is already active.")
            episode_dir = self._episode_dir(identity)
            if episode_dir.exists():
                raise RuntimeError(
                    "A native v2.5-p episode directory already exists; audit it "
                    "before reusing the UUID."
                )
            self._active = _ActiveNativeEpisode(identity=identity)

    def bind_prepared_episode(
        self,
        identity: EpisodeCaptureIdentity,
        *,
        bound_at: float,
        bound_clock_domain_id: str,
    ) -> V25pNativeEmbodimentBinding:
        with self._lock:
            active = self._require_active(identity)
            if active.binding is not None:
                raise RuntimeError("The native v2.5-p episode is already bound.")
            if (
                bound_clock_domain_id == identity.prepared_clock_domain_id
                and float(bound_at) < identity.prepared_at
            ):
                raise ValueError("Native binding cannot precede episode preparation.")
            binding = V25pNativeEmbodimentBinding(
                episode_uuid=identity.episode_uuid,
                legacy_episode_index=identity.legacy_episode_index,
                episode_identity_hash=identity.semantic_content_hash,
                embodiment_profile_hash=self.embodiment_profile.semantic_content_hash,
                component_stream_profile_hash=(
                    self.component_stream_profile.semantic_content_hash
                ),
                bound_at=bound_at,
                bound_clock_domain_id=bound_clock_domain_id,
            )
            episode_dir = self._episode_dir(identity)
            publish_immutable_canonical(
                episode_dir / "identity.cjson",
                identity.to_payload(),
                domain=V25P_EPISODE_IDENTITY_FILE_DOMAIN_V1,
            )
            publish_immutable_canonical(
                episode_dir / "embodiment_binding.cjson",
                binding.to_payload(),
                domain=V25P_NATIVE_BINDING_FILE_DOMAIN_V1,
            )
            active.binding = binding
            active.records_by_stream = {
                item.stream_id: [] for item in self.component_stream_profile.streams
            }
            active.record_count_by_stream = {
                item.stream_id: 0 for item in self.component_stream_profile.streams
            }
            if self.physical_storage_layout == V25P_PARQUET_CHUNK_LAYOUT_V1:
                active.parquet_writer = ParquetChunkEpisodeWriter(
                    episode_dir=episode_dir,
                    identity=identity,
                    component_stream_profile=self.component_stream_profile,
                    chunk_rows=self.parquet_chunk_rows,
                    compression=self.parquet_compression,
                )
            return binding

    def append_sample(
        self,
        identity: EpisodeCaptureIdentity,
        *,
        sample_index: int,
        cut_at: float,
        cut_clock_domain_id: str,
        values_by_stream: Mapping[str, ComponentStreamInput],
    ) -> V25pCausalSample:
        with self._lock:
            active = self._require_bound_unsealed(identity)
            assert active.samples is not None
            expected_sample_index = active.sample_count
            if int(sample_index) != expected_sample_index:
                raise ValueError(
                    "Native causal sample indices must be contiguous: "
                    f"expected={expected_sample_index} got={sample_index}."
                )
            provided = dict(values_by_stream)
            known = set(self.component_stream_profile.stream_by_id)
            unknown = sorted(set(provided) - known)
            missing = sorted(
                set(self.component_stream_profile.required_stream_ids) - set(provided)
            )
            if unknown or missing:
                raise ValueError(
                    f"Native sample stream coverage mismatch unknown={unknown} missing={missing}."
                )
            sample_records: dict[str, V25pComponentRecord] = {}
            cut_value = float(cut_at)
            cut_clock = str(cut_clock_domain_id).strip()
            if not cut_clock:
                raise ValueError("cut_clock_domain_id must be non-empty.")
            assert active.binding is not None
            if (
                cut_clock == active.binding.bound_clock_domain_id
                and cut_value < active.binding.bound_at
            ):
                raise ValueError("A native sample cut cannot precede episode binding.")
            assert active.records_by_stream is not None
            assert active.record_count_by_stream is not None
            for stream in self.component_stream_profile.streams:
                value = provided.get(stream.stream_id)
                if value is None:
                    continue
                if not isinstance(value, ComponentStreamInput):
                    raise TypeError(
                        "values_by_stream entries must be ComponentStreamInput."
                    )
                if stream.required_for_sample and not value.valid:
                    raise ValueError(
                        f"Required native stream {stream.stream_id!r} cannot be invalid."
                    )
                if value.available_clock_domain_id == cut_clock and (
                    float(value.available_at) > cut_value
                ):
                    raise ValueError(
                        f"Stream {stream.stream_id!r} was not available at the causal cut."
                    )
                if value.effective_clock_domain_id == cut_clock and (
                    float(value.effective_at) > cut_value
                ):
                    raise ValueError(
                        f"Stream {stream.stream_id!r} is effective after the causal cut."
                    )
                if not value.valid:
                    if value.values is not None or value.payload_reference is not None:
                        raise ValueError(
                            f"Invalid stream {stream.stream_id!r} cannot carry payload."
                        )
                    normalized = None
                    reference = None
                elif stream.storage_kind == "inline_tensor":
                    if value.payload_reference is not None:
                        raise ValueError(
                            f"Inline stream {stream.stream_id!r} cannot carry an "
                            "external payload reference."
                        )
                    normalized = stream.normalize_values(value.values)
                    reference = None
                else:
                    if value.values is not None or not isinstance(
                        value.payload_reference,
                        V25pExternalPayloadReference,
                    ):
                        raise ValueError(
                            f"External stream {stream.stream_id!r} requires exactly "
                            "one typed payload reference."
                        )
                    normalized = None
                    reference = value.payload_reference
                record = V25pComponentRecord(
                    episode_uuid=identity.episode_uuid,
                    legacy_episode_index=identity.legacy_episode_index,
                    component_stream_profile_hash=(
                        self.component_stream_profile.semantic_content_hash
                    ),
                    stream_id=stream.stream_id,
                    component_id=stream.component_id,
                    dataset_role=stream.dataset_role,
                    value_stage=stream.value_stage,
                    payload_schema_hash=stream.payload_schema_hash,
                    record_sequence=(
                        active.record_count_by_stream[stream.stream_id] + 1
                    ),
                    sample_index=expected_sample_index,
                    effective_at=value.effective_at,
                    available_at=value.available_at,
                    effective_clock_domain_id=value.effective_clock_domain_id,
                    available_clock_domain_id=value.available_clock_domain_id,
                    values=normalized,
                    payload_reference=reference,
                    valid=value.valid,
                    invalid_reason=value.invalid_reason,
                    source_record_id=value.source_record_id,
                )
                if (
                    self.physical_storage_layout == V25P_RECORD_CJSON_LAYOUT_V1
                    and not self.defer_sample_file_writes
                ):
                    self._publish_component_record(identity, record)
                sample_records[stream.stream_id] = record
            sample = V25pCausalSample(
                episode_uuid=identity.episode_uuid,
                legacy_episode_index=identity.legacy_episode_index,
                component_stream_profile_hash=(
                    self.component_stream_profile.semantic_content_hash
                ),
                sample_index=expected_sample_index,
                cut_at=cut_value,
                cut_clock_domain_id=cut_clock,
                record_hash_by_stream={
                    stream_id: record.semantic_content_hash
                    for stream_id, record in sample_records.items()
                },
            )
            if (
                self.physical_storage_layout == V25P_RECORD_CJSON_LAYOUT_V1
                and not self.defer_sample_file_writes
            ):
                self._publish_causal_sample(identity, sample)
            if active.parquet_writer is not None:
                active.parquet_writer.append(
                    records_by_stream=sample_records,
                    sample=sample,
                )
            for stream_id, record in sample_records.items():
                active.record_count_by_stream[stream_id] += 1
                if active.parquet_writer is None:
                    active.records_by_stream[stream_id].append(record)
            if active.parquet_writer is None:
                active.samples.append(sample)
            active.sample_count += 1
            assert active.latest_cut_by_clock is not None
            active.latest_cut_by_clock[cut_clock] = max(
                cut_value,
                active.latest_cut_by_clock.get(cut_clock, cut_value),
            )
            return sample

    def begin_subtask(self, identity: EpisodeCaptureIdentity, **values: Any) -> SubtaskAttemptEvent:
        return self._append_task_event(identity, event_kind="begin", outcome=None, **values)

    def finish_subtask(
        self,
        identity: EpisodeCaptureIdentity,
        *,
        outcome: str,
        **values: Any,
    ) -> SubtaskAttemptEvent:
        return self._append_task_event(
            identity,
            event_kind="finish",
            outcome=outcome,
            **values,
        )

    def _append_task_event(
        self,
        identity: EpisodeCaptureIdentity,
        **values: Any,
    ) -> SubtaskAttemptEvent:
        with self._lock:
            active = self._require_bound_unsealed(identity)
            if self.task_profile is None:
                raise RuntimeError("No native long-horizon task profile is configured.")
            sequence = len(active.task_events) + 1
            event = SubtaskAttemptEvent(
                event_id=f"task-event:{identity.episode_uuid}:{sequence}",
                episode_uuid=identity.episode_uuid,
                legacy_episode_index=identity.legacy_episode_index,
                event_sequence=sequence,
                **values,
            )
            candidate = active.task_events + (event,)
            validate_subtask_attempt_events(
                self.task_profile,
                candidate,
                require_closed=False,
            )
            publish_immutable_canonical(
                self._episode_dir(identity)
                / "task_events"
                / f"{sequence:08d}.cjson",
                event.to_payload(),
                domain=V25P_TASK_EVENT_FILE_DOMAIN_V1,
            )
            active.task_events = candidate
            return event

    def seal_native_episode(
        self,
        identity: EpisodeCaptureIdentity,
        *,
        sealed_at: float,
        sealed_clock_domain_id: str,
        task_effective_at: float | None = None,
        task_effective_clock_domain_id: str | None = None,
    ) -> ComponentCaptureSealReport:
        with self._lock:
            active = self._require_bound_unsealed(identity)
            assert active.samples is not None
            if active.sample_count <= 0:
                raise RuntimeError("A native v2.5-p episode cannot seal without samples.")
            seal_value = float(sealed_at)
            seal_clock = str(sealed_clock_domain_id).strip()
            if not seal_clock:
                raise ValueError("sealed_clock_domain_id must be non-empty.")
            assert active.binding is not None
            if (
                seal_clock == active.binding.bound_clock_domain_id
                and seal_value < active.binding.bound_at
            ):
                raise ValueError("Native capture seal cannot precede episode binding.")
            assert active.latest_cut_by_clock is not None
            if active.latest_cut_by_clock.get(seal_clock, seal_value) > seal_value:
                raise ValueError("Native capture seal cannot precede a causal sample cut.")
            storage_manifest = None
            parquet_record_hashes: Mapping[str, tuple[str, ...]] | None = None
            parquet_sample_hashes: tuple[str, ...] | None = None
            if active.parquet_writer is not None:
                storage_manifest = active.parquet_writer.seal()
                parquet_record_hashes, parquet_sample_hashes = (
                    active.parquet_writer.logical_hash_sequences()
                )
                assert active.record_count_by_stream is not None
                if (
                    storage_manifest.sample_count != active.sample_count
                    or dict(storage_manifest.record_count_by_stream)
                    != active.record_count_by_stream
                ):
                    raise RuntimeError(
                        "Parquet storage seal disagrees with native logical counts."
                    )
            elif self.defer_sample_file_writes:
                self._publish_deferred_sample_files(identity, active)
            auto_closed = self._auto_close_open_attempts(
                identity,
                effective_at=(
                    sealed_at if task_effective_at is None else task_effective_at
                ),
                effective_clock_domain_id=(
                    sealed_clock_domain_id
                    if task_effective_clock_domain_id is None
                    else task_effective_clock_domain_id
                ),
            )
            task_manifest = None
            if self.task_profile is not None:
                task_manifest = TaskSemanticsEpisodeManifest.create(
                    profile=self.task_profile,
                    episode_uuid=identity.episode_uuid,
                    legacy_episode_index=identity.legacy_episode_index,
                    episode_identity_hash=identity.semantic_content_hash,
                    events=active.task_events,
                    sealed_at=seal_value,
                    sealed_clock_domain_id=seal_clock,
                )
                publish_immutable_canonical(
                    self._episode_dir(identity) / "task_manifest.cjson",
                    task_manifest.to_payload(),
                    domain=V25P_TASK_MANIFEST_FILE_DOMAIN_V1,
                )
            assert active.binding is not None
            assert active.records_by_stream is not None
            record_hashes = (
                {
                    stream_id: tuple(item.semantic_content_hash for item in records)
                    for stream_id, records in active.records_by_stream.items()
                }
                if parquet_record_hashes is None
                else dict(parquet_record_hashes)
            )
            sample_hashes = (
                tuple(item.semantic_content_hash for item in active.samples)
                if parquet_sample_hashes is None
                else parquet_sample_hashes
            )
            native_manifest = V25pNativeCaptureManifest(
                episode_uuid=identity.episode_uuid,
                legacy_episode_index=identity.legacy_episode_index,
                episode_identity_hash=identity.semantic_content_hash,
                native_embodiment_binding_hash=active.binding.semantic_content_hash,
                component_stream_profile_hash=(
                    self.component_stream_profile.semantic_content_hash
                ),
                sample_count=active.sample_count,
                record_count_by_stream={
                    stream_id: len(values) for stream_id, values in record_hashes.items()
                },
                record_bundle_hash_by_stream={
                    stream_id: canonical_content_hash(
                        values,
                        domain=V25P_NATIVE_RECORD_BUNDLE_HASH_DOMAIN_V1,
                    )
                    for stream_id, values in record_hashes.items()
                },
                sample_bundle_hash=canonical_content_hash(
                    sample_hashes,
                    domain=V25P_NATIVE_SAMPLE_BUNDLE_HASH_DOMAIN_V1,
                ),
                task_semantics_manifest_hash=(
                    None
                    if task_manifest is None
                    else task_manifest.semantic_content_hash
                ),
                sealed_at=seal_value,
                sealed_clock_domain_id=seal_clock,
            )
            publish_immutable_canonical(
                self._episode_dir(identity) / "native_capture_manifest.cjson",
                native_manifest.to_payload(),
                domain=V25P_NATIVE_CAPTURE_MANIFEST_FILE_DOMAIN_V1,
            )
            active.task_manifest = task_manifest
            active.native_manifest = native_manifest
            return ComponentCaptureSealReport(
                episode_uuid=identity.episode_uuid,
                legacy_episode_index=identity.legacy_episode_index,
                sample_count=active.sample_count,
                native_capture_manifest_hash=native_manifest.semantic_content_hash,
                task_semantics_manifest_hash=(
                    None
                    if task_manifest is None
                    else task_manifest.semantic_content_hash
                ),
                auto_closed_attempt_count=auto_closed,
                native_storage_manifest_hash=(
                    None
                    if storage_manifest is None
                    else storage_manifest.semantic_content_hash
                ),
            )

    def _publish_component_record(
        self,
        identity: EpisodeCaptureIdentity,
        record: V25pComponentRecord,
    ) -> None:
        publish_immutable_canonical(
            self._episode_dir(identity)
            / "streams"
            / record.stream_id
            / f"{record.record_sequence:08d}.cjson",
            record.to_payload(),
            domain=V25P_NATIVE_COMPONENT_RECORD_FILE_DOMAIN_V1,
        )

    def _publish_causal_sample(
        self,
        identity: EpisodeCaptureIdentity,
        sample: V25pCausalSample,
    ) -> None:
        publish_immutable_canonical(
            self._episode_dir(identity)
            / "samples"
            / f"{sample.sample_index:08d}.cjson",
            sample.to_payload(),
            domain=V25P_NATIVE_CAUSAL_SAMPLE_FILE_DOMAIN_V1,
        )

    def _publish_deferred_sample_files(
        self,
        identity: EpisodeCaptureIdentity,
        active: _ActiveNativeEpisode,
    ) -> None:
        assert active.records_by_stream is not None
        assert active.samples is not None
        for records in active.records_by_stream.values():
            for record in records:
                self._publish_component_record(identity, record)
        for sample in active.samples:
            self._publish_causal_sample(identity, sample)

    def commit_episode(
        self,
        identity: EpisodeCaptureIdentity,
        *,
        committed_at: float,
        committed_clock_domain_id: str,
        compatibility_projection_receipts: tuple[
            V25pCompatibilityProjectionReceipt, ...
        ] = (),
        v24_capture_commit_hash: str | None = None,
    ) -> ComponentCaptureCommitReport:
        with self._lock:
            active = self._require_active(identity)
            if active.binding is None or active.native_manifest is None:
                raise RuntimeError("Native capture must be sealed before episode commit.")
            commit_value = float(committed_at)
            commit_clock = str(committed_clock_domain_id).strip()
            if not commit_clock:
                raise ValueError("committed_clock_domain_id must be non-empty.")
            if (
                commit_clock == active.native_manifest.sealed_clock_domain_id
                and commit_value < active.native_manifest.sealed_at
            ):
                raise ValueError("Native episode commit cannot precede native sealing.")
            receipt_hashes: list[str] = []
            for receipt in compatibility_projection_receipts:
                if not isinstance(receipt, V25pCompatibilityProjectionReceipt):
                    raise TypeError(
                        "compatibility_projection_receipts must contain typed receipts."
                    )
                if (
                    receipt.episode_uuid,
                    receipt.legacy_episode_index,
                    receipt.episode_identity_hash,
                    receipt.native_capture_manifest_hash,
                ) != (
                    identity.episode_uuid,
                    identity.legacy_episode_index,
                    identity.semantic_content_hash,
                    active.native_manifest.semantic_content_hash,
                ):
                    raise ValueError("Compatibility projection receipt identity mismatch.")
                if receipt.projection_profile_hash not in self._projection_by_hash:
                    raise ValueError("Compatibility projection profile is not dataset-bound.")
                if receipt.status == "complete":
                    try:
                        plan = plan_v23_compatibility_projection(
                            stream_profile=self.component_stream_profile,
                            projection_profile=self._projection_by_hash[
                                receipt.projection_profile_hash
                            ],
                        )
                    except V25pV23ProjectionError as exc:
                        raise ValueError(
                            f"Complete compatibility receipt is not projectable: {exc}"
                        ) from exc
                    if (
                        receipt.lossy != plan.lossy
                        or receipt.omitted_stream_ids != plan.omitted_stream_ids
                    ):
                        raise ValueError(
                            "Complete compatibility receipt misstates omitted streams."
                        )
                    try:
                        verify_complete_v23_projection_receipt(
                            dataset_root=self.dataset_root,
                            identity=identity,
                            receipt=receipt,
                        )
                    except V25pV23ProjectionError as exc:
                        raise ValueError(
                            f"Complete compatibility receipt target is invalid: {exc}"
                        ) from exc
                if (
                    receipt.completed_clock_domain_id
                    == active.native_manifest.sealed_clock_domain_id
                    and receipt.completed_at < active.native_manifest.sealed_at
                ):
                    raise ValueError(
                        "Compatibility projection cannot complete before native sealing."
                    )
                publish_immutable_canonical(
                    self._episode_dir(identity)
                    / "compatibility_projections"
                    / f"{receipt.semantic_content_hash}.cjson",
                    receipt.to_payload(),
                    domain=V25P_COMPATIBILITY_PROJECTION_RECEIPT_FILE_DOMAIN_V1,
                )
                receipt_hashes.append(receipt.semantic_content_hash)
            manifest = V25pNativeEpisodeManifest(
                episode_uuid=identity.episode_uuid,
                legacy_episode_index=identity.legacy_episode_index,
                episode_identity_hash=identity.semantic_content_hash,
                native_embodiment_binding_hash=active.binding.semantic_content_hash,
                native_capture_manifest_hash=active.native_manifest.semantic_content_hash,
                compatibility_projection_receipt_hashes=tuple(receipt_hashes),
                v24_capture_commit_hash=v24_capture_commit_hash,
                committed_at=commit_value,
                committed_clock_domain_id=commit_clock,
            )
            manifest_path = self._episode_dir(identity) / "episode_manifest.cjson"
            publish_immutable_canonical(
                manifest_path,
                manifest.to_payload(),
                domain=V25P_NATIVE_EPISODE_MANIFEST_FILE_DOMAIN_V2,
            )
            report = ComponentCaptureCommitReport(
                episode_uuid=identity.episode_uuid,
                legacy_episode_index=identity.legacy_episode_index,
                native_capture_manifest_hash=active.native_manifest.semantic_content_hash,
                episode_manifest_hash=manifest.semantic_content_hash,
                compatibility_projection_receipt_hashes=tuple(receipt_hashes),
                v24_capture_commit_hash=v24_capture_commit_hash,
                episode_manifest_relative_path=manifest_path.relative_to(
                    self.dataset_root
                ).as_posix(),
            )
            self._active = None
            return report

    def _auto_close_open_attempts(
        self,
        identity: EpisodeCaptureIdentity,
        *,
        effective_at: float,
        effective_clock_domain_id: str,
    ) -> int:
        if self.task_profile is None:
            return 0
        active = self._require_bound_unsealed(identity)
        open_by_track: dict[str, SubtaskAttemptEvent] = {}
        for event in sorted(
            active.task_events,
            key=lambda item: (item.effective_at, item.event_sequence),
        ):
            if event.event_kind == "begin":
                open_by_track[event.track_id] = event
            else:
                open_by_track.pop(event.track_id, None)
        for begin in tuple(open_by_track.values()):
            if begin.effective_clock_domain_id != effective_clock_domain_id:
                raise RuntimeError(
                    "Auto-close requires an episode-end timestamp in the task clock."
                )
            finish_at = max(float(effective_at), float(begin.effective_at))
            self._append_task_event(
                identity,
                track_id=begin.track_id,
                attempt_id=begin.attempt_id,
                subtask_id=begin.subtask_id,
                event_kind="finish",
                outcome="unknown",
                effective_at=finish_at,
                available_at=finish_at,
                effective_clock_domain_id=effective_clock_domain_id,
                available_clock_domain_id=effective_clock_domain_id,
                actor_id="episode_owner:auto_close",
                frame_index=None,
                origin=begin.origin,
                source_event_ids=(begin.event_id,),
            )
        return len(open_by_track)

    def discard_episode(self, identity: EpisodeCaptureIdentity) -> None:
        with self._lock:
            active = self._require_active(identity)
            if active.native_manifest is not None:
                raise RuntimeError("A sealed native capture cannot be discarded.")
            if active.parquet_writer is not None:
                active.parquet_writer.close_unsealed()
            episode_dir = self._episode_dir(identity)
            if episode_dir.exists():
                shutil.rmtree(episode_dir)
                if self.episode_root.exists():
                    fsync_directory(self.episode_root)
            self._active = None

    def close_unsealed(self, identity: EpisodeCaptureIdentity) -> None:
        with self._lock:
            active = self._require_active(identity)
            if active.parquet_writer is not None:
                active.parquet_writer.close_unsealed()
            self._active = None

    def _require_active(self, identity: EpisodeCaptureIdentity) -> _ActiveNativeEpisode:
        active = self._active
        if active is None or active.identity != identity:
            raise ValueError("Episode identity is not the active native v2.5-p episode.")
        return active

    def _require_bound_unsealed(
        self,
        identity: EpisodeCaptureIdentity,
    ) -> _ActiveNativeEpisode:
        active = self._require_active(identity)
        if active.binding is None:
            raise RuntimeError("The active native episode is not bound.")
        if active.native_manifest is not None:
            raise RuntimeError("The native capture is already sealed.")
        return active


__all__ = [
    "V25P_RECORD_CJSON_LAYOUT_V1",
    "ComponentCaptureCommitReport",
    "ComponentCaptureEpisodeStore",
    "ComponentCaptureSealReport",
    "ComponentStreamInput",
    "validate_component_stream_profile",
]

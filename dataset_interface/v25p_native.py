"""Read-only verification for native, embodiment-neutral v2.5-p episodes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash, canonical_loads
from dataset_contracts.control_trace_episode import EPISODE_COMMIT_FILE_DOMAIN_V1
from dataset_contracts.embodiment import (
    CaptureProjectionProfileLike,
    EmbodimentCaptureProfileLike,
    capture_projection_profile_from_payload,
    embodiment_capture_profile_from_payload,
)
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.task_semantics import (
    LongHorizonTaskProfile,
    SubtaskAttemptEvent,
    TaskSemanticsEpisodeManifest,
)
from dataset_contracts.v25p import (
    V25P_EPISODE_IDENTITY_FILE_DOMAIN_V1,
    V25P_TASK_EVENT_FILE_DOMAIN_V1,
    V25P_TASK_MANIFEST_FILE_DOMAIN_V1,
)
from dataset_contracts.v25p_native import (
    V25P_NATIVE_BINDING_FILE_DOMAIN_V1,
    V25P_NATIVE_CAPTURE_MANIFEST_FILE_DOMAIN_V1,
    V25P_NATIVE_CAUSAL_SAMPLE_FILE_DOMAIN_V1,
    V25P_NATIVE_COMPONENT_RECORD_FILE_DOMAIN_V1,
    V25P_NATIVE_DATASET_MANIFEST_FILE_DOMAIN_V2,
    V25P_NATIVE_EPISODE_MANIFEST_FILE_DOMAIN_V2,
    V25P_NATIVE_EPISODE_RELATIVE_ROOT_V1,
    V25P_NATIVE_PROFILE_FILE_DOMAIN_V1,
    V25P_NATIVE_PROFILE_RELATIVE_ROOT_V1,
    V25P_NATIVE_RECORD_BUNDLE_HASH_DOMAIN_V1,
    V25P_NATIVE_RELATIVE_ROOT_V1,
    V25P_NATIVE_SAMPLE_BUNDLE_HASH_DOMAIN_V1,
    V25pCausalSample,
    V25pComponentRecord,
    V25pComponentStreamProfileLike,
    V25pNativeCaptureManifest,
    V25pNativeDatasetManifest,
    V25pNativeEmbodimentBinding,
    V25pNativeEpisodeManifest,
    component_stream_profile_from_payload,
    normalize_component_stream_stored_values,
)
from dataset_contracts.v25p_projection import (
    V25P_COMPATIBILITY_PROJECTION_RECEIPT_FILE_DOMAIN_V1,
    V25pCompatibilityProjectionReceipt,
)

from .v25p_projection_plan import (
    V25pV23ProjectionError,
    plan_v23_compatibility_projection,
    verify_complete_v23_projection_receipt,
)
from .v25p_parquet_storage import load_v25p_parquet_storage


V25P_NATIVE_READINESS_PROVISIONAL = "native_component_streams_provisional"
V25P_NATIVE_READINESS_WITH_COMPATIBILITY = "native_with_compatibility_projection"
V25P_NATIVE_READINESS_WITH_V24 = "native_with_v24_merge_candidate"


class V25pNativeReadError(ValueError):
    pass


@dataclass(frozen=True)
class V25pNativeReadiness:
    classification: str
    valid: bool
    native_v25: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class V25pLoadedNativeEpisode:
    dataset_manifest: V25pNativeDatasetManifest
    episode_manifest: V25pNativeEpisodeManifest
    native_capture_manifest: V25pNativeCaptureManifest
    identity: EpisodeCaptureIdentity
    embodiment_profile: EmbodimentCaptureProfileLike
    component_stream_profile: V25pComponentStreamProfileLike
    embodiment_binding: V25pNativeEmbodimentBinding
    records_by_stream: Mapping[str, tuple[V25pComponentRecord, ...]]
    samples: tuple[V25pCausalSample, ...]
    task_profile: LongHorizonTaskProfile | None
    task_events: tuple[SubtaskAttemptEvent, ...]
    task_manifest: TaskSemanticsEpisodeManifest | None
    compatibility_projection_profiles: Mapping[str, CaptureProjectionProfileLike]
    compatibility_projection_receipts: tuple[
        V25pCompatibilityProjectionReceipt, ...
    ]
    readiness: V25pNativeReadiness


def _mapping(path: Path, *, domain: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise V25pNativeReadError(f"Required native v2.5-p artifact is missing: {path}")
    try:
        payload = canonical_loads(path.read_bytes(), expected_domain=domain)
    except (OSError, TypeError, ValueError) as exc:
        raise V25pNativeReadError(f"Invalid native v2.5-p artifact {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise V25pNativeReadError(f"Native v2.5-p artifact is not a mapping: {path}")
    return payload


def _safe_uuid(value: object) -> str:
    result = str(value).strip()
    if not result or result in {".", ".."} or "/" in result or "\\" in result:
        raise V25pNativeReadError("episode_uuid is empty or unsafe.")
    return result


def _load_task(
    *,
    root: Path,
    episode_dir: Path,
    identity: EpisodeCaptureIdentity,
    capture_manifest: V25pNativeCaptureManifest,
) -> tuple[
    LongHorizonTaskProfile | None,
    tuple[SubtaskAttemptEvent, ...],
    TaskSemanticsEpisodeManifest | None,
]:
    if capture_manifest.task_semantics_manifest_hash is None:
        if (episode_dir / "task_manifest.cjson").exists() or (
            episode_dir / "task_events"
        ).exists():
            raise V25pNativeReadError(
                "Task artifacts exist but native capture does not declare task semantics."
            )
        return None, (), None
    task_manifest = TaskSemanticsEpisodeManifest.from_payload(
        _mapping(
            episode_dir / "task_manifest.cjson",
            domain=V25P_TASK_MANIFEST_FILE_DOMAIN_V1,
        )
    )
    if task_manifest.semantic_content_hash != capture_manifest.task_semantics_manifest_hash:
        raise V25pNativeReadError("Native capture task manifest hash mismatch.")
    task_profile = LongHorizonTaskProfile.from_payload(
        _mapping(
            root
            / V25P_NATIVE_PROFILE_RELATIVE_ROOT_V1
            / "task"
            / f"{task_manifest.task_profile_hash}.cjson",
            domain=V25P_NATIVE_PROFILE_FILE_DOMAIN_V1,
        )
    )
    event_dir = episode_dir / "task_events"
    event_paths = sorted(event_dir.glob("*.cjson")) if event_dir.exists() else []
    events = tuple(
        SubtaskAttemptEvent.from_payload(
            _mapping(path, domain=V25P_TASK_EVENT_FILE_DOMAIN_V1)
        )
        for path in event_paths
    )
    rebuilt = TaskSemanticsEpisodeManifest.create(
        profile=task_profile,
        episode_uuid=identity.episode_uuid,
        legacy_episode_index=identity.legacy_episode_index,
        episode_identity_hash=identity.semantic_content_hash,
        events=events,
        sealed_at=task_manifest.sealed_at,
        sealed_clock_domain_id=task_manifest.sealed_clock_domain_id,
    )
    if rebuilt.semantic_content_hash != task_manifest.semantic_content_hash:
        raise V25pNativeReadError("Task events do not reproduce the native task seal.")
    return task_profile, events, task_manifest


def load_v25p_native_episode(
    dataset_root: str | Path,
    episode_uuid: str,
) -> V25pLoadedNativeEpisode:
    root = Path(dataset_root).expanduser().resolve()
    uuid_text = _safe_uuid(episode_uuid)
    dataset_manifest = V25pNativeDatasetManifest.from_payload(
        _mapping(
            root / V25P_NATIVE_RELATIVE_ROOT_V1 / "dataset_manifest.cjson",
            domain=V25P_NATIVE_DATASET_MANIFEST_FILE_DOMAIN_V2,
        )
    )
    episode_dir = root / V25P_NATIVE_EPISODE_RELATIVE_ROOT_V1 / uuid_text
    identity = EpisodeCaptureIdentity.from_payload(
        _mapping(
            episode_dir / "identity.cjson",
            domain=V25P_EPISODE_IDENTITY_FILE_DOMAIN_V1,
        )
    )
    if identity.episode_uuid != uuid_text:
        raise V25pNativeReadError("Episode directory and identity UUID do not match.")
    binding = V25pNativeEmbodimentBinding.from_payload(
        _mapping(
            episode_dir / "embodiment_binding.cjson",
            domain=V25P_NATIVE_BINDING_FILE_DOMAIN_V1,
        )
    )
    embodiment = embodiment_capture_profile_from_payload(
        _mapping(
            root
            / V25P_NATIVE_PROFILE_RELATIVE_ROOT_V1
            / "embodiment"
            / f"{binding.embodiment_profile_hash}.cjson",
            domain=V25P_NATIVE_PROFILE_FILE_DOMAIN_V1,
        )
    )
    profile = component_stream_profile_from_payload(
        _mapping(
            root
            / V25P_NATIVE_PROFILE_RELATIVE_ROOT_V1
            / "component_stream"
            / f"{binding.component_stream_profile_hash}.cjson",
            domain=V25P_NATIVE_PROFILE_FILE_DOMAIN_V1,
        )
    )
    capture_manifest = V25pNativeCaptureManifest.from_payload(
        _mapping(
            episode_dir / "native_capture_manifest.cjson",
            domain=V25P_NATIVE_CAPTURE_MANIFEST_FILE_DOMAIN_V1,
        )
    )
    episode_manifest = V25pNativeEpisodeManifest.from_payload(
        _mapping(
            episode_dir / "episode_manifest.cjson",
            domain=V25P_NATIVE_EPISODE_MANIFEST_FILE_DOMAIN_V2,
        )
    )
    identity_pair = (identity.episode_uuid, identity.legacy_episode_index)
    for label, pair in (
        ("binding", (binding.episode_uuid, binding.legacy_episode_index)),
        (
            "native capture",
            (capture_manifest.episode_uuid, capture_manifest.legacy_episode_index),
        ),
        (
            "episode manifest",
            (episode_manifest.episode_uuid, episode_manifest.legacy_episode_index),
        ),
    ):
        if pair != identity_pair:
            raise V25pNativeReadError(f"{label} identity does not match the episode.")
    if (
        binding.episode_identity_hash != identity.semantic_content_hash
        or capture_manifest.episode_identity_hash != identity.semantic_content_hash
        or episode_manifest.episode_identity_hash != identity.semantic_content_hash
    ):
        raise V25pNativeReadError("Native episode identity hash binding mismatch.")
    if binding.semantic_content_hash not in {
        capture_manifest.native_embodiment_binding_hash,
        episode_manifest.native_embodiment_binding_hash,
    }:
        raise V25pNativeReadError("Native embodiment binding hash mismatch.")
    if (
        capture_manifest.native_embodiment_binding_hash
        != episode_manifest.native_embodiment_binding_hash
    ):
        raise V25pNativeReadError("Native manifests bind different embodiments.")
    if capture_manifest.semantic_content_hash != episode_manifest.native_capture_manifest_hash:
        raise V25pNativeReadError("Episode manifest binds another native capture.")
    if (
        binding.bound_clock_domain_id == identity.prepared_clock_domain_id
        and binding.bound_at < identity.prepared_at
    ):
        raise V25pNativeReadError("Native binding precedes episode preparation.")
    if (
        capture_manifest.sealed_clock_domain_id == binding.bound_clock_domain_id
        and capture_manifest.sealed_at < binding.bound_at
    ):
        raise V25pNativeReadError("Native capture seal precedes episode binding.")
    if (
        episode_manifest.committed_clock_domain_id
        == capture_manifest.sealed_clock_domain_id
        and episode_manifest.committed_at < capture_manifest.sealed_at
    ):
        raise V25pNativeReadError("Native episode commit precedes native sealing.")
    if embodiment.semantic_content_hash != binding.embodiment_profile_hash:
        raise V25pNativeReadError("Embodiment profile hash mismatch.")
    if profile.semantic_content_hash != binding.component_stream_profile_hash:
        raise V25pNativeReadError("Component stream profile hash mismatch.")
    if capture_manifest.component_stream_profile_hash != profile.semantic_content_hash:
        raise V25pNativeReadError(
            "Native capture manifest binds another component stream profile."
        )
    if profile.embodiment_profile_hash != embodiment.semantic_content_hash:
        raise V25pNativeReadError("Component stream profile binds another embodiment.")
    if embodiment.semantic_content_hash not in dataset_manifest.embodiment_profile_hashes:
        raise V25pNativeReadError("Embodiment profile is not dataset-bound.")
    if profile.semantic_content_hash not in dataset_manifest.component_stream_profile_hashes:
        raise V25pNativeReadError("Component stream profile is not dataset-bound.")
    projection_profiles: dict[str, CaptureProjectionProfileLike] = {}
    for profile_hash in dataset_manifest.compatibility_projection_profile_hashes:
        projection = capture_projection_profile_from_payload(
            _mapping(
                root
                / V25P_NATIVE_PROFILE_RELATIVE_ROOT_V1
                / "compatibility_projection"
                / f"{profile_hash}.cjson",
                domain=V25P_NATIVE_PROFILE_FILE_DOMAIN_V1,
            )
        )
        if projection.semantic_content_hash != profile_hash:
            raise V25pNativeReadError(
                "Compatibility projection profile hash mismatch."
            )
        projection_profiles[profile_hash] = projection

    parquet_records: Mapping[str, tuple[V25pComponentRecord, ...]] | None = None
    parquet_samples: tuple[V25pCausalSample, ...] | None = None
    if (episode_dir / "native_storage_manifest.cjson").exists():
        try:
            _storage_manifest, parquet_records, parquet_samples = (
                load_v25p_parquet_storage(
                    episode_dir=episode_dir,
                    identity=identity,
                    profile=profile,
                )
            )
        except (OSError, TypeError, ValueError) as exc:
            raise V25pNativeReadError(
                f"Invalid native v2.5-p Parquet storage: {exc}"
            ) from exc

    records_by_stream: dict[str, tuple[V25pComponentRecord, ...]] = {}
    record_by_hash: dict[str, V25pComponentRecord] = {}
    streams_root = episode_dir / "streams"
    if parquet_records is None:
        actual_stream_dirs = (
            {path.name for path in streams_root.iterdir() if path.is_dir()}
            if streams_root.exists()
            else set()
        )
        unknown_stream_dirs = actual_stream_dirs - set(profile.stream_by_id)
        if unknown_stream_dirs:
            raise V25pNativeReadError(
                "Native episode contains unknown component stream directories: "
                f"{sorted(unknown_stream_dirs)}."
            )
    for stream in profile.streams:
        paths: tuple[Path, ...] = ()
        if parquet_records is None:
            paths = tuple(
                sorted((episode_dir / "streams" / stream.stream_id).glob("*.cjson"))
            )
            records = tuple(
                V25pComponentRecord.from_payload(
                    _mapping(
                        path,
                        domain=V25P_NATIVE_COMPONENT_RECORD_FILE_DOMAIN_V1,
                    )
                )
                for path in paths
            )
        else:
            records = tuple(parquet_records.get(stream.stream_id, ()))
        if tuple(item.record_sequence for item in records) != tuple(
            range(1, len(records) + 1)
        ):
            raise V25pNativeReadError(
                f"Stream {stream.stream_id!r} record sequence is not contiguous."
            )
        for record_index, record in enumerate(records):
            if paths and paths[record_index].name != f"{record.record_sequence:08d}.cjson":
                raise V25pNativeReadError(
                    f"Stream {stream.stream_id!r} record filename/sequence mismatch."
                )
            if (
                record.episode_uuid,
                record.legacy_episode_index,
                record.component_stream_profile_hash,
                record.stream_id,
                record.component_id,
                record.dataset_role,
                record.value_stage,
                record.payload_schema_hash,
            ) != (
                identity.episode_uuid,
                identity.legacy_episode_index,
                profile.semantic_content_hash,
                stream.stream_id,
                stream.component_id,
                stream.dataset_role,
                stream.value_stage,
                stream.payload_schema_hash,
            ):
                raise V25pNativeReadError(
                    f"Stream {stream.stream_id!r} contains a mismatched record."
                )
            if record.valid and stream.storage_kind == "inline_tensor":
                if record.payload_reference is not None or record.values is None:
                    raise V25pNativeReadError(
                        f"Inline stream {stream.stream_id!r} has invalid storage."
                    )
                if (
                    normalize_component_stream_stored_values(
                        stream,
                        record.values,
                    )
                    != record.values
                ):
                    raise V25pNativeReadError(
                        f"Stream {stream.stream_id!r} record values are not canonical."
                    )
            elif record.valid and (
                record.values is not None or record.payload_reference is None
            ):
                raise V25pNativeReadError(
                    f"External stream {stream.stream_id!r} has invalid storage."
                )
            if record.semantic_content_hash in record_by_hash:
                raise V25pNativeReadError("Component record hashes must be unique.")
            record_by_hash[record.semantic_content_hash] = record
        expected_count = capture_manifest.record_count_by_stream.get(stream.stream_id)
        if expected_count != len(records):
            raise V25pNativeReadError(
                f"Stream {stream.stream_id!r} record count does not match the seal."
            )
        bundle = canonical_content_hash(
            tuple(item.semantic_content_hash for item in records),
            domain=V25P_NATIVE_RECORD_BUNDLE_HASH_DOMAIN_V1,
        )
        if capture_manifest.record_bundle_hash_by_stream.get(stream.stream_id) != bundle:
            raise V25pNativeReadError(
                f"Stream {stream.stream_id!r} record bundle hash mismatch."
            )
        records_by_stream[stream.stream_id] = records
    if set(capture_manifest.record_count_by_stream) != set(profile.stream_by_id):
        raise V25pNativeReadError("Native capture seal contains unknown/missing streams.")

    sample_paths: tuple[Path, ...] = ()
    if parquet_samples is None:
        sample_paths = tuple(sorted((episode_dir / "samples").glob("*.cjson")))
        samples = tuple(
            V25pCausalSample.from_payload(
                _mapping(path, domain=V25P_NATIVE_CAUSAL_SAMPLE_FILE_DOMAIN_V1)
            )
            for path in sample_paths
        )
    else:
        samples = tuple(parquet_samples)
    if tuple(item.sample_index for item in samples) != tuple(range(len(samples))):
        raise V25pNativeReadError("Native causal sample indices are not contiguous.")
    if sample_paths and any(
        path.name != f"{sample.sample_index:08d}.cjson"
        for path, sample in zip(sample_paths, samples)
    ):
        raise V25pNativeReadError("Native causal sample filename/index mismatch.")
    if len(samples) != capture_manifest.sample_count:
        raise V25pNativeReadError("Native causal sample count does not match the seal.")
    referenced: set[str] = set()
    for sample in samples:
        if (
            sample.episode_uuid,
            sample.legacy_episode_index,
            sample.component_stream_profile_hash,
        ) != (
            identity.episode_uuid,
            identity.legacy_episode_index,
            profile.semantic_content_hash,
        ):
            raise V25pNativeReadError("Native causal sample identity mismatch.")
        if (
            sample.cut_clock_domain_id == binding.bound_clock_domain_id
            and sample.cut_at < binding.bound_at
        ):
            raise V25pNativeReadError("Native causal sample precedes episode binding.")
        if (
            sample.cut_clock_domain_id == capture_manifest.sealed_clock_domain_id
            and sample.cut_at > capture_manifest.sealed_at
        ):
            raise V25pNativeReadError("Native causal sample occurs after capture sealing.")
        missing_required = set(profile.required_stream_ids) - set(
            sample.record_hash_by_stream
        )
        if missing_required:
            raise V25pNativeReadError(
                f"Native causal sample is missing required streams: {sorted(missing_required)}."
            )
        if set(sample.record_hash_by_stream) - set(profile.stream_by_id):
            raise V25pNativeReadError("Native causal sample references an unknown stream.")
        for stream_id, record_hash in sample.record_hash_by_stream.items():
            record = record_by_hash.get(record_hash)
            if record is None or record.stream_id != stream_id:
                raise V25pNativeReadError("Native causal sample record reference is invalid.")
            if record.sample_index != sample.sample_index:
                raise V25pNativeReadError("Component record/sample index mismatch.")
            if (
                record.available_clock_domain_id == sample.cut_clock_domain_id
                and record.available_at > sample.cut_at
            ):
                raise V25pNativeReadError("Component record was unavailable at the causal cut.")
            if (
                record.effective_clock_domain_id == sample.cut_clock_domain_id
                and record.effective_at > sample.cut_at
            ):
                raise V25pNativeReadError(
                    "Component record becomes effective after the causal cut."
                )
            if record_hash in referenced:
                raise V25pNativeReadError("A component record is referenced more than once.")
            referenced.add(record_hash)
    if referenced != set(record_by_hash):
        raise V25pNativeReadError("Native component records contain unreferenced evidence.")
    sample_bundle = canonical_content_hash(
        tuple(item.semantic_content_hash for item in samples),
        domain=V25P_NATIVE_SAMPLE_BUNDLE_HASH_DOMAIN_V1,
    )
    if sample_bundle != capture_manifest.sample_bundle_hash:
        raise V25pNativeReadError("Native causal sample bundle hash mismatch.")

    task_profile, task_events, task_manifest = _load_task(
        root=root,
        episode_dir=episode_dir,
        identity=identity,
        capture_manifest=capture_manifest,
    )
    if task_profile is not None and (
        task_profile.semantic_content_hash
        not in dataset_manifest.long_horizon_task_profile_hashes
    ):
        raise V25pNativeReadError("Task profile is not bound by the native dataset.")
    receipts: list[V25pCompatibilityProjectionReceipt] = []
    for receipt_hash in episode_manifest.compatibility_projection_receipt_hashes:
        receipt = V25pCompatibilityProjectionReceipt.from_payload(
            _mapping(
                episode_dir
                / "compatibility_projections"
                / f"{receipt_hash}.cjson",
                domain=V25P_COMPATIBILITY_PROJECTION_RECEIPT_FILE_DOMAIN_V1,
            )
        )
        if receipt.semantic_content_hash != receipt_hash:
            raise V25pNativeReadError("Compatibility projection receipt hash mismatch.")
        if (
            receipt.episode_uuid,
            receipt.legacy_episode_index,
            receipt.episode_identity_hash,
            receipt.native_capture_manifest_hash,
        ) != (
            identity.episode_uuid,
            identity.legacy_episode_index,
            identity.semantic_content_hash,
            capture_manifest.semantic_content_hash,
        ):
            raise V25pNativeReadError("Compatibility projection receipt identity mismatch.")
        projection_profile = projection_profiles.get(receipt.projection_profile_hash)
        if projection_profile is None:
            raise V25pNativeReadError("Compatibility projection profile is not dataset-bound.")
        if receipt.target_dataset_schema != projection_profile.base_dataset_schema:
            raise V25pNativeReadError(
                "Compatibility receipt target schema disagrees with its profile."
            )
        if receipt.status == "complete":
            try:
                plan = plan_v23_compatibility_projection(
                    stream_profile=profile,
                    projection_profile=projection_profile,
                )
            except V25pV23ProjectionError as exc:
                raise V25pNativeReadError(
                    f"Complete compatibility receipt has no valid projection: {exc}"
                ) from exc
            if (
                receipt.lossy != plan.lossy
                or receipt.omitted_stream_ids != plan.omitted_stream_ids
            ):
                raise V25pNativeReadError(
                    "Complete compatibility receipt misstates omitted native streams."
                )
            try:
                verify_complete_v23_projection_receipt(
                    dataset_root=root,
                    identity=identity,
                    receipt=receipt,
                )
            except V25pV23ProjectionError as exc:
                raise V25pNativeReadError(str(exc)) from exc
        if (
            receipt.completed_clock_domain_id
            == capture_manifest.sealed_clock_domain_id
            and receipt.completed_at < capture_manifest.sealed_at
        ):
            raise V25pNativeReadError(
                "Compatibility projection predates native capture sealing."
            )
        if (
            receipt.completed_clock_domain_id
            == episode_manifest.committed_clock_domain_id
            and receipt.completed_at > episode_manifest.committed_at
        ):
            raise V25pNativeReadError(
                "Compatibility projection completes after episode commit."
            )
        receipts.append(receipt)
    if episode_manifest.v24_capture_commit_hash is not None:
        commit = _mapping(
            root
            / "meta/v24/control_trace/episodes"
            / identity.episode_uuid
            / "capture_commit.cjson",
            domain=EPISODE_COMMIT_FILE_DOMAIN_V1,
        )
        if commit.get("semantic_content_hash") != episode_manifest.v24_capture_commit_hash:
            raise V25pNativeReadError("Referenced v2.4 capture commit hash mismatch.")
        if (
            str(commit.get("episode_uuid", "")),
            int(commit.get("legacy_episode_index", -1)),
        ) != identity_pair:
            raise V25pNativeReadError("Referenced v2.4 capture commit identity mismatch.")

    if episode_manifest.v24_capture_commit_hash is not None:
        classification = V25P_NATIVE_READINESS_WITH_V24
    elif receipts:
        classification = V25P_NATIVE_READINESS_WITH_COMPATIBILITY
    else:
        classification = V25P_NATIVE_READINESS_PROVISIONAL
    reasons = [
        "Native component metadata and causal joins are internally complete; a "
        "future formal v2.5 commit contract is still required."
    ]
    if any(
        stream.storage_kind == "external_artifact" for stream in profile.streams
    ):
        reasons.append(
            "External artifact descriptors are content-hash-bound, but payload "
            "byte/container verification remains owned by the corresponding "
            "artifact reader."
        )
    readiness = V25pNativeReadiness(
        classification=classification,
        valid=True,
        native_v25=False,
        reasons=tuple(reasons),
    )
    return V25pLoadedNativeEpisode(
        dataset_manifest=dataset_manifest,
        episode_manifest=episode_manifest,
        native_capture_manifest=capture_manifest,
        identity=identity,
        embodiment_profile=embodiment,
        component_stream_profile=profile,
        embodiment_binding=binding,
        records_by_stream=MappingProxyType(records_by_stream),
        samples=samples,
        task_profile=task_profile,
        task_events=task_events,
        task_manifest=task_manifest,
        compatibility_projection_profiles=MappingProxyType(projection_profiles),
        compatibility_projection_receipts=tuple(receipts),
        readiness=readiness,
    )
__all__ = [
    "V25P_NATIVE_READINESS_PROVISIONAL",
    "V25P_NATIVE_READINESS_WITH_COMPATIBILITY",
    "V25P_NATIVE_READINESS_WITH_V24",
    "V25pLoadedNativeEpisode",
    "V25pNativeReadError",
    "V25pNativeReadiness",
    "load_v25p_native_episode",
]

"""Build auditable, re-indexed LeRobot v2.1-style dataset derivatives.

The existing ``split-episodes`` and ``delete-episodes`` tools operate on one
dataset at a time.  This module complements them with an immutable-source
workflow that can select episodes from multiple roots, merge them into a new
dataset, assign contiguous episode indices, and apply narrowly-scoped schema
migrations.  Artifact materialization is an explicit manifest policy.  The
legacy default remains hard-link-or-copy for existing callers, while reviewed
physical consolidations can require copy-on-write reflinks and independent
inodes.  All metadata and parquet files that may be rewritten are newly
written.

The input is a JSON manifest.  See ``manifests/dexteleop_2026_07_sort.json``
for a complete example.  A build is staged beside its destination and only
published once every selected episode has been materialised successfully.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dataset_interface.dataset_task_binding import (
    DatasetTaskBindingError,
    inspect_dataset_task_binding,
)
from dataset_interface.modality_artifacts import (
    DatasetCaptureWritableError,
    assert_dataset_capture_writable,
)
from dataset_interface.progress_artifacts import (
    ProgressArtifactReadError,
    load_dataset_progress_profile,
    load_dataset_task_stage_profile,
    load_sealed_progress_episode,
)
from dataset_interface.progress_detail_annotation_artifacts import (
    DETAIL_STAGE_ANNOTATION_RELATIVE_ROOT,
    DetailStageAnnotationArtifactReadError,
    load_detail_stage_annotation_set,
)
from dataset_interface.progress_stage_annotation_artifacts import (
    STAGE_ANNOTATION_RELATIVE_ROOT,
    StageAnnotationArtifactReadError,
    load_stage_annotation_set,
)
from dataset_interface.tactile_bundle import (
    AttachedTactileBundle,
    TactileBundleAttachmentError,
    attach_tactile_bundle,
)

from .io import (
    episode_artifact_paths,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from .selection import parse_episode_selection

_EPISODE_TOKEN_RE = re.compile(r"episode_\d+")
_CHUNK_TOKEN_RE = re.compile(r"chunk-\d+")
_PATH_FIELD_RE = re.compile(r"(?:^|_)(?:path|paths|relative_dir)$")
_DMTAC_SLOT_MIGRATION = "dmtac_slot1_to_slot0"
_DMTAC_SIDES = ("left", "right")
_ARTIFACT_POLICY_HARDLINK_OR_COPY = "hardlink_or_copy"
_ARTIFACT_POLICY_COPY = "copy"
_ARTIFACT_POLICY_REFLINK_REQUIRED = "reflink_required"
_ARTIFACT_POLICIES = frozenset(
    {
        _ARTIFACT_POLICY_HARDLINK_OR_COPY,
        _ARTIFACT_POLICY_COPY,
        _ARTIFACT_POLICY_REFLINK_REQUIRED,
    }
)
_PROGRESS_POLICY_OMIT = "omit"
_PROGRESS_POLICY_PRESERVE = "preserve"
_PROGRESS_POLICIES = frozenset(
    {_PROGRESS_POLICY_OMIT, _PROGRESS_POLICY_PRESERVE}
)
_FICLONE = 0x40049409
_VECTOR_COLUMN_SPECS: tuple[tuple[str, tuple[tuple[int, int], tuple[int, int]]], ...] = (
    ("action", ((14, 20), (20, 26))),
    ("observation.last_action", ((14, 20), (20, 26))),
    ("observation.state", ((13, 19), (32, 38))),
)
_HAND_VECTOR_COLUMNS: tuple[str, ...] = (
    "action.left_hand.joint_target",
    "action.right_hand.joint_target",
    "observation.left_hand.qpos",
    "observation.right_hand.qpos",
    "observation.hand.left.joint_pos",
    "observation.hand.right.joint_pos",
    "observation.hand.left.measured_joint_pos",
    "observation.hand.right.measured_joint_pos",
    "observation.hand.left.commanded_joint_pos",
    "observation.hand.right.commanded_joint_pos",
    "action.replay.left_hand_joint_target",
    "action.replay.right_hand_joint_target",
)


@dataclass(frozen=True)
class PlannedEpisode:
    """One source episode and its deterministic output index."""

    source_root: str
    source_episode_index: int
    output_episode_index: int
    frame_count: int
    transforms: tuple[str, ...]


@dataclass(frozen=True)
class PlannedTaskRebinding:
    """One explicitly reviewed legacy source-task normalization."""

    source_root: str
    source_binding_sha256: str
    source_info_task_id: str
    source_info_task_description: str
    source_task_rows: tuple[dict[str, Any], ...]
    target_task_id: str
    target_task_description: str
    reason: str


@dataclass(frozen=True)
class PlannedExternalTactileSource:
    """One verified native tactile bundle used as an immutable source."""

    source_root: str
    tactile_storage_root: str
    dataset_uuid: str
    episode_count: int
    artifact_count: int
    payload_count: int
    payload_bytes: int
    payload_hashes_verified: bool


@dataclass(frozen=True)
class DatasetBuildPlan:
    """A no-write description of one destination dataset."""

    name: str
    output_root: str
    task_id: str
    task_description: str
    episode_count: int
    frame_count: int
    source_episode_count: int
    minimum_duration_sec: float | None
    filtered_by_minimum_duration: tuple[dict[str, Any], ...]
    missing_requested_indices: tuple[dict[str, Any], ...]
    missing_artifacts: tuple[dict[str, Any], ...]
    artifact_materialization_policy: str
    progress_artifact_policy: str
    progress_profile_hash: str | None
    task_stage_profile_hash: str | None
    task_rebindings: tuple[PlannedTaskRebinding, ...]
    external_tactile_sources: tuple[PlannedExternalTactileSource, ...]
    episodes: tuple[PlannedEpisode, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetOrganizationError(ValueError):
    """Raised when an immutable-source derived dataset cannot be built safely."""


_REVIEWED_TASK_REBINDING_SCHEMA = (
    "dexteleop.dataset_source_task_rebinding.v1"
)
_FRAME_TASK_COLUMN_NAMES = frozenset(
    {"task", "task_id", "task_index", "task_description"}
)


def _source_task_binding_snapshot(
    source_root: Path,
    source_info: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Return the narrowly scoped legacy task metadata bound by a review hash."""

    rows = read_jsonl(source_root / "meta" / "tasks.jsonl")
    payload = {
        "schema": "dexteleop.dataset_source_task_binding_snapshot.v1",
        "info_task_id": str(source_info.get("task_id", "") or "").strip(),
        "info_task_description": str(
            source_info.get("task_description", "") or ""
        ).strip(),
        "task_rows": rows,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return payload, hashlib.sha256(encoded).hexdigest()


def _task_rebinding_from_source(
    *,
    raw_source: Mapping[str, Any],
    source_root: Path,
    source_info: Mapping[str, Any],
    task_id: str,
    task_description: str,
    binding_error: DatasetTaskBindingError | None,
    observed_task_id: str | None,
) -> PlannedTaskRebinding | None:
    """Validate a manifest-pinned, metadata-only legacy task rebinding."""

    raw_rebinding = raw_source.get("reviewed_task_rebinding")
    task_matches = binding_error is None and observed_task_id == task_id
    if task_matches:
        if raw_rebinding is not None:
            raise DatasetOrganizationError(
                "reviewed_task_rebinding is unnecessary for an already canonical "
                f"source: {source_root}."
            )
        return None
    if not isinstance(raw_rebinding, Mapping):
        if binding_error is not None:
            raise DatasetOrganizationError(
                f"Source has invalid task binding and no reviewed rebinding: "
                f"{binding_error}"
            ) from binding_error
        raise DatasetOrganizationError(
            "Dataset source task_id differs from the canonical merge task and no reviewed "
            f"rebinding was supplied: source={observed_task_id!r}, "
            f"canonical={task_id!r}, root={source_root}."
        )
    if str(raw_rebinding.get("schema") or "") != _REVIEWED_TASK_REBINDING_SCHEMA:
        raise DatasetOrganizationError(
            "reviewed_task_rebinding has an unsupported schema for "
            f"{source_root}."
        )
    expected_target_id = str(raw_rebinding.get("target_task_id") or "").strip()
    expected_target_description = str(
        raw_rebinding.get("target_task_description") or ""
    ).strip()
    if (expected_target_id, expected_target_description) != (
        task_id,
        task_description,
    ):
        raise DatasetOrganizationError(
            "reviewed_task_rebinding target differs from the dataset canonical task: "
            f"root={source_root}."
        )
    reason = str(raw_rebinding.get("reason") or "").strip()
    if not reason:
        raise DatasetOrganizationError(
            f"reviewed_task_rebinding requires a non-empty reason: {source_root}."
        )
    snapshot, observed_snapshot_sha256 = _source_task_binding_snapshot(
        source_root,
        source_info,
    )
    expected_snapshot_sha256 = str(
        raw_rebinding.get("source_binding_sha256") or ""
    ).strip()
    if expected_snapshot_sha256 != observed_snapshot_sha256:
        raise DatasetOrganizationError(
            "reviewed_task_rebinding source snapshot changed: "
            f"expected={expected_snapshot_sha256!r}, "
            f"observed={observed_snapshot_sha256!r}, root={source_root}."
        )
    feature_names = {
        str(name).strip().lower()
        for name in (source_info.get("features") or {})
    }
    embedded = sorted(feature_names & _FRAME_TASK_COLUMN_NAMES)
    if embedded:
        raise DatasetOrganizationError(
            "reviewed_task_rebinding is metadata-only but the source declares "
            f"frame-level task columns {embedded}: {source_root}."
        )
    return PlannedTaskRebinding(
        source_root=str(source_root),
        source_binding_sha256=observed_snapshot_sha256,
        source_info_task_id=str(snapshot["info_task_id"]),
        source_info_task_description=str(snapshot["info_task_description"]),
        source_task_rows=tuple(dict(row) for row in snapshot["task_rows"]),
        target_task_id=task_id,
        target_task_description=task_description,
        reason=reason,
    )


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:  # pragma: no cover - dependent on runtime environment.
        raise ModuleNotFoundError(
            "organize-datasets requires pyarrow. Run it in the DexTeleop Python environment."
        ) from exc
    return pa, pq


def _assert_rebinding_has_no_frame_task_columns(
    source_root: Path,
    records_by_index: Mapping[int, Mapping[str, Any]],
) -> None:
    """Reject metadata-only rebinding if any selected frame schema owns task state."""

    _pa, pq = _require_pyarrow()
    del _pa
    for episode_index, record in records_by_index.items():
        parquet_value = record.get("parquet_path")
        if not parquet_value:
            raise DatasetOrganizationError(
                "Cannot verify metadata-only task rebinding because an Episode has "
                f"no parquet_path: root={source_root}, episode={episode_index}."
            )
        parquet_path = source_root / _safe_relative_path(
            source_root,
            str(parquet_value),
        )
        try:
            schema = pq.read_schema(parquet_path)
        except Exception as exc:
            raise DatasetOrganizationError(
                "Cannot verify frame schema before reviewed task rebinding: "
                f"{parquet_path}: {exc}"
            ) from exc
        embedded = sorted(
            name
            for name in schema.names
            if str(name).strip().lower() in _FRAME_TASK_COLUMN_NAMES
        )
        if embedded:
            raise DatasetOrganizationError(
                "reviewed_task_rebinding is metadata-only but a source Parquet "
                f"contains frame-level task columns {embedded}: {parquet_path}."
            )


def _read_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path).resolve()
    payload = read_json(manifest_path)
    if not payload:
        raise DatasetOrganizationError(f"Manifest is empty or invalid: {manifest_path}")
    if not isinstance(payload.get("datasets"), list):
        raise DatasetOrganizationError("Manifest requires a non-empty 'datasets' list.")
    return manifest_path, payload


def _selected_dataset_entries(
    manifest: Mapping[str, Any],
    dataset_names: Sequence[str] | None,
) -> list[Mapping[str, Any]]:
    entries = manifest.get("datasets")
    if not isinstance(entries, list):
        raise DatasetOrganizationError("Manifest requires a non-empty 'datasets' list.")
    if not dataset_names:
        return list(entries)
    wanted = {str(name) for name in dataset_names}
    selected = [entry for entry in entries if isinstance(entry, Mapping) and str(entry.get("name")) in wanted]
    found = {str(entry.get("name")) for entry in selected}
    missing = sorted(wanted - found)
    if missing:
        raise DatasetOrganizationError(f"Requested dataset names are absent from manifest: {missing}")
    return selected


def _resolve_manifest_path(manifest_path: Path, raw_path: Any, *, label: str) -> Path:
    value = str(raw_path or "").strip()
    if not value:
        raise DatasetOrganizationError(f"Missing {label} in manifest {manifest_path}.")
    path = Path(value)
    return (path if path.is_absolute() else manifest_path.parent / path).resolve()


def _selection_from_source(source: Mapping[str, Any]) -> tuple[str | Sequence[int], set[int]]:
    raw_selected = source.get("episode_indices", "all")
    if isinstance(raw_selected, str) and raw_selected.strip().lower() in {"", "all", "*"}:
        selected: str | Sequence[int] = "all"
    elif isinstance(raw_selected, str):
        selected = parse_episode_selection(raw_selected)
    elif isinstance(raw_selected, Sequence) and not isinstance(raw_selected, (bytes, bytearray)):
        selected = tuple(int(value) for value in raw_selected)
    else:
        raise DatasetOrganizationError("episode_indices must be 'all', a selection string, or an integer list.")
    raw_excluded = source.get("exclude_episode_indices", ())
    if isinstance(raw_excluded, str):
        excluded = set(parse_episode_selection(raw_excluded)) if raw_excluded.strip() else set()
    elif isinstance(raw_excluded, Sequence) and not isinstance(raw_excluded, (bytes, bytearray)):
        excluded = {int(value) for value in raw_excluded}
    else:
        raise DatasetOrganizationError("exclude_episode_indices must be a selection string or an integer list.")
    return selected, excluded


def _normalise_transforms(source: Mapping[str, Any]) -> tuple[str, ...]:
    raw = source.get("transforms", ())
    if isinstance(raw, str):
        values = (raw,)
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        values = tuple(str(value) for value in raw)
    else:
        raise DatasetOrganizationError("transforms must be a string or a list of strings.")
    unsupported = sorted(set(values) - {_DMTAC_SLOT_MIGRATION})
    if unsupported:
        raise DatasetOrganizationError(f"Unsupported transforms: {unsupported}")
    return tuple(dict.fromkeys(values))


def _minimum_duration_sec(
    manifest: Mapping[str, Any],
    dataset: Mapping[str, Any],
) -> float | None:
    """Return a dataset override or manifest-wide source-duration floor."""

    raw_value = dataset.get("min_duration_sec", manifest.get("min_duration_sec"))
    if raw_value is None or str(raw_value).strip() == "":
        return None
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise DatasetOrganizationError("min_duration_sec must be a non-negative finite number.") from exc
    if not math.isfinite(value) or value < 0.0:
        raise DatasetOrganizationError("min_duration_sec must be a non-negative finite number.")
    return value


def _artifact_materialization_policy(dataset: Mapping[str, Any]) -> str:
    value = str(
        dataset.get(
            "artifact_materialization_policy",
            _ARTIFACT_POLICY_HARDLINK_OR_COPY,
        )
    ).strip()
    if value not in _ARTIFACT_POLICIES:
        raise DatasetOrganizationError(
            "artifact_materialization_policy must be one of "
            f"{sorted(_ARTIFACT_POLICIES)}, got {value!r}."
        )
    return value


def _attach_manifest_tactile_bundle(
    *,
    manifest_path: Path,
    raw_source: Mapping[str, Any],
    source_root: Path,
) -> AttachedTactileBundle | None:
    """Resolve and verify an explicitly bound external tactile source.

    The field is optional so every legacy manifest keeps its original behavior.
    A populated native-bundle core, however, must never be interpreted as a
    legacy self-contained dataset: callers must state which storage root owns
    the other half of the bundle.
    """

    raw_storage_root = raw_source.get("tactile_storage_root")
    native_identity = source_root / "meta/tactile_bundle_v1/dataset_identity.cjson"
    if raw_storage_root is None or not str(raw_storage_root).strip():
        if native_identity.is_file():
            raise DatasetOrganizationError(
                "Native tactile bundle source requires an explicit "
                f"tactile_storage_root: {source_root}."
            )
        return None
    if not native_identity.is_file():
        raise DatasetOrganizationError(
            "tactile_storage_root was supplied for a source without a native "
            f"tactile bundle identity: {source_root}."
        )
    storage_root = _resolve_manifest_path(
        manifest_path,
        raw_storage_root,
        label=f"tactile_storage_root for {source_root}",
    )
    verify_payload_hashes = raw_source.get(
        "verify_tactile_payload_hashes",
        False,
    )
    if not isinstance(verify_payload_hashes, bool):
        raise DatasetOrganizationError(
            "verify_tactile_payload_hashes must be a boolean."
        )
    try:
        return attach_tactile_bundle(
            source_root,
            storage_root,
            verify_payload_hashes=verify_payload_hashes,
        )
    except TactileBundleAttachmentError as exc:
        raise DatasetOrganizationError(
            f"Cannot attach external tactile bundle for {source_root}: {exc}"
        ) from exc


def _planned_external_tactile_source(
    attached: AttachedTactileBundle,
) -> PlannedExternalTactileSource:
    report = attached.report
    return PlannedExternalTactileSource(
        source_root=str(attached.core_dataset_root),
        tactile_storage_root=str(attached.tactile_storage_root),
        dataset_uuid=report.dataset_uuid,
        episode_count=report.episode_count,
        artifact_count=report.artifact_count,
        payload_count=report.payload_count,
        payload_bytes=report.payload_bytes,
        payload_hashes_verified=report.payload_hashes_verified,
    )


def _progress_artifact_policy(dataset: Mapping[str, Any]) -> str:
    value = str(
        dataset.get("progress_artifact_policy", _PROGRESS_POLICY_OMIT)
    ).strip()
    if value not in _PROGRESS_POLICIES:
        raise DatasetOrganizationError(
            "progress_artifact_policy must be one of "
            f"{sorted(_PROGRESS_POLICIES)}, got {value!r}."
        )
    return value


def _safe_progress_episode_uuid(record: Mapping[str, Any]) -> str:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise DatasetOrganizationError(
            "Progress-preserving merge requires Episode metadata."
        )
    reference = metadata.get("progress_episode")
    if not isinstance(reference, Mapping):
        raise DatasetOrganizationError(
            "Progress-preserving merge requires metadata.progress_episode."
        )
    episode_uuid = str(reference.get("episode_uuid") or "").strip()
    if (
        not episode_uuid
        or episode_uuid in {".", ".."}
        or Path(episode_uuid).name != episode_uuid
    ):
        raise DatasetOrganizationError(
            f"Invalid progress Episode UUID {episode_uuid!r}."
        )
    return episode_uuid


def _progress_reference(record: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = record.get("metadata")
    reference = (
        metadata.get("progress_episode")
        if isinstance(metadata, Mapping)
        else None
    )
    if not isinstance(reference, Mapping):
        raise DatasetOrganizationError(
            "Progress-preserving merge requires metadata.progress_episode."
        )
    return reference


def _require_progress_reference_path(
    *,
    source_root: Path,
    reference: Mapping[str, Any],
    field: str,
    expected: Path,
) -> None:
    raw = str(reference.get(field) or "").strip()
    if not raw:
        raise DatasetOrganizationError(
            f"Progress Episode reference is missing {field}."
        )
    relative = _safe_relative_path(source_root, raw)
    if relative != expected:
        raise DatasetOrganizationError(
            f"Progress Episode {field} mismatch: expected={expected} actual={relative}."
        )


def _validate_annotation_sets(
    *,
    source_root: Path,
    episode_uuid: str,
    episode_identity_hash: str,
    progress_manifest_hash: str,
) -> None:
    roots_and_loaders = (
        (STAGE_ANNOTATION_RELATIVE_ROOT, load_stage_annotation_set),
        (
            DETAIL_STAGE_ANNOTATION_RELATIVE_ROOT,
            load_detail_stage_annotation_set,
        ),
    )
    for relative_root, loader in roots_and_loaders:
        episode_root = source_root / relative_root / episode_uuid
        if not episode_root.exists():
            continue
        if not episode_root.is_dir():
            raise DatasetOrganizationError(
                f"Progress annotation path is not a directory: {episode_root}."
            )
        for annotation_dir in sorted(episode_root.iterdir()):
            if not annotation_dir.is_dir():
                raise DatasetOrganizationError(
                    "Progress annotation Episode directory contains a non-directory: "
                    f"{annotation_dir}."
                )
            loaded = loader(annotation_dir)
            binding = loaded.binding
            if (
                binding.episode_capture_identity_hash
                != episode_identity_hash
                or binding.source_progress_manifest_hash
                != progress_manifest_hash
            ):
                raise DatasetOrganizationError(
                    "Progress annotation does not bind the selected sealed Episode: "
                    f"{annotation_dir}."
                )


def _validate_progress_episode(
    *,
    source_root: Path,
    record: Mapping[str, Any],
    expected_profile_hash: str,
    expected_stage_profile_hash: str | None,
) -> tuple[str, str]:
    reference = _progress_reference(record)
    episode_uuid = _safe_progress_episode_uuid(record)
    episode_relative = Path("meta/progress/episodes") / episode_uuid
    _require_progress_reference_path(
        source_root=source_root,
        reference=reference,
        field="manifest_ref",
        expected=episode_relative / "manifest.cjson",
    )
    _require_progress_reference_path(
        source_root=source_root,
        reference=reference,
        field="binding_ref",
        expected=episode_relative / "binding.cjson",
    )
    _require_progress_reference_path(
        source_root=source_root,
        reference=reference,
        field="profile_ref",
        expected=episode_relative / "profile.cjson",
    )
    try:
        loaded = load_sealed_progress_episode(
            source_root / episode_relative,
            expected_episode_identity_hash=str(
                reference.get("episode_identity_hash") or ""
            ),
            expected_profile_hash=expected_profile_hash,
            expected_binding_hash=str(
                reference.get("progress_episode_binding_hash") or ""
            ),
        )
    except ProgressArtifactReadError as exc:
        raise DatasetOrganizationError(
            f"Invalid selected progress Episode {episode_uuid}: {exc}"
        ) from exc
    observed_stage_hash = (
        None
        if loaded.stage_profile is None
        else loaded.stage_profile.semantic_content_hash
    )
    if observed_stage_hash != expected_stage_profile_hash:
        raise DatasetOrganizationError(
            "Selected Episode stage profile differs from the dataset merge contract: "
            f"episode_uuid={episode_uuid} expected={expected_stage_profile_hash} "
            f"actual={observed_stage_hash}."
        )
    try:
        _validate_annotation_sets(
            source_root=source_root,
            episode_uuid=episode_uuid,
            episode_identity_hash=loaded.binding.episode_capture_identity_hash,
            progress_manifest_hash=loaded.manifest.semantic_content_hash,
        )
    except (
        StageAnnotationArtifactReadError,
        DetailStageAnnotationArtifactReadError,
    ) as exc:
        raise DatasetOrganizationError(
            f"Invalid annotation set for progress Episode {episode_uuid}: {exc}"
        ) from exc
    return episode_uuid, loaded.binding.episode_capture_identity_hash


def _passes_minimum_duration(record: Mapping[str, Any], minimum_duration_sec: float | None) -> bool:
    if minimum_duration_sec is None:
        return True
    try:
        duration_sec = float(record["duration_sec"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetOrganizationError(
            "Cannot apply min_duration_sec because source episode has no valid duration_sec: "
            f"episode_index={record.get('episode_index')}"
        ) from exc
    if not math.isfinite(duration_sec) or duration_sec < 0.0:
        raise DatasetOrganizationError(
            "Cannot apply min_duration_sec because source episode has invalid duration_sec: "
            f"episode_index={record.get('episode_index')} duration_sec={record.get('duration_sec')!r}"
        )
    return duration_sec >= minimum_duration_sec


def _record_by_episode_index(dataset_root: Path) -> dict[int, dict[str, Any]]:
    records = read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    if not records:
        raise DatasetOrganizationError(f"No meta/episodes.jsonl records found in {dataset_root}.")
    indexed: dict[int, dict[str, Any]] = {}
    for record in records:
        try:
            index = int(record["episode_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetOrganizationError(f"Invalid episode_index in {dataset_root}: {record!r}") from exc
        if index in indexed:
            raise DatasetOrganizationError(f"Duplicate episode_index={index} in {dataset_root}.")
        indexed[index] = record
    return indexed


def _safe_relative_path(dataset_root: Path, artifact_path: str) -> Path:
    path = Path(str(artifact_path))
    if path.is_absolute():
        try:
            path = path.relative_to(dataset_root)
        except ValueError as exc:
            raise DatasetOrganizationError(
                f"Artifact path must be inside its source dataset: {artifact_path!r}"
            ) from exc
    if not path.parts or path.parts[0] in {"", ".", ".."} or ".." in path.parts:
        raise DatasetOrganizationError(f"Unsafe artifact path: {artifact_path!r}")
    return path


def _output_relative_path(relative_path: Path, *, output_episode_index: int, chunk_size: int) -> Path:
    output_episode = f"episode_{int(output_episode_index):06d}"
    output_chunk = f"chunk-{int(output_episode_index) // int(chunk_size):03d}"
    source_text = relative_path.as_posix()
    if _EPISODE_TOKEN_RE.search(source_text) is None:
        raise DatasetOrganizationError(
            "Episode artifact path does not contain an episode_NNNNNN token and cannot be re-indexed: "
            f"{relative_path}"
        )
    rewritten = _CHUNK_TOKEN_RE.sub(output_chunk, source_text)
    rewritten = _EPISODE_TOKEN_RE.sub(output_episode, rewritten)
    return Path(rewritten)


def _rewrite_paths_in_value(value: Any, *, output_episode_index: int, chunk_size: int) -> Any:
    """Rewrite only values that visibly contain episode/chunk path tokens."""

    if isinstance(value, str):
        # Runtime metadata also contains semantic event names such as
        # ``episode_start``.  Only numbered artifact tokens are paths that can
        # be re-indexed.
        if _EPISODE_TOKEN_RE.search(value) is None:
            return value
        return _output_relative_path(
            Path(value),
            output_episode_index=output_episode_index,
            chunk_size=chunk_size,
        ).as_posix()
    if isinstance(value, list):
        return [
            _rewrite_paths_in_value(item, output_episode_index=output_episode_index, chunk_size=chunk_size)
            for item in value
        ]
    if isinstance(value, tuple):
        return [
            _rewrite_paths_in_value(item, output_episode_index=output_episode_index, chunk_size=chunk_size)
            for item in value
        ]
    if isinstance(value, Mapping):
        return {
            str(key): _rewrite_paths_in_value(
                item,
                output_episode_index=output_episode_index,
                chunk_size=chunk_size,
            )
            for key, item in value.items()
        }
    return value


def _dmtac_specs() -> tuple[dict[str, Any], dict[str, list[bool]]]:
    # Keep this metadata description local to the offline dataset boundary.
    # Importing robot_motion here would make post-processing depend on the
    # runtime/control package and would allow a runtime schema change to alter
    # an immutable dataset migration unexpectedly.
    active_mask = [True, False, False, False, False, False]
    specs = {
        side: {
            "side": side,
            "hand_model": "dmtac_gripper",
            "slot_count": 6,
            "active_dof": 1,
            "active_mask": list(active_mask),
            "joint_names": ["gripper_close"],
            "command_space": "normalized_0_0.9",
            "slot_map": {"gripper_close": 0},
            "invalid_slot_value": "zero_padding",
            "semantics": (
                "DM-Tac gripper uses the fixed 6-wide hand slot for compatibility; "
                "only slot 0 / gripper_close is a real actuator target and the other slots are padding."
            ),
            "metadata": {"gripper_slot": 0, "gripper_action_dim": 1},
        }
        for side in _DMTAC_SIDES
    }
    masks = {side: list(active_mask) for side in _DMTAC_SIDES}
    return specs, masks


def _update_dmtac_metadata(value: Any) -> Any:
    """Correct known hand-schema metadata keys without touching source q6 values."""

    specs, masks = _dmtac_specs()
    if isinstance(value, list):
        return [_update_dmtac_metadata(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    result = {str(key): _update_dmtac_metadata(item) for key, item in value.items()}
    if "hand_action_specs_by_side" in result:
        result["hand_action_specs_by_side"] = specs
    if "hand_active_mask_by_side" in result:
        result["hand_active_mask_by_side"] = masks
    if result.get("hand_backend") in {"dmtac-gripper", "dmtac_gripper", "inspire_tac"}:
        result["hand_backend"] = "dmtac-gripper"
    for policy_key in ("gripper_policy",):
        policy = result.get(policy_key)
        if isinstance(policy, Mapping):
            rewritten = dict(policy)
            rewritten["output_dim"] = 0
            rewritten["gripper_slot"] = 0
            result[policy_key] = rewritten
    policy_by_side = result.get("gripper_policy_by_side")
    if isinstance(policy_by_side, Mapping):
        result["gripper_policy_by_side"] = {
            str(side): {**dict(policy), "output_dim": 0, "gripper_slot": 0}
            if isinstance(policy, Mapping)
            else policy
            for side, policy in policy_by_side.items()
        }
    return result


def _validate_artifacts(
    source_root: Path,
    record: Mapping[str, Any],
    *,
    tactile_bundle: AttachedTactileBundle | None = None,
) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for artifact_path in episode_artifact_paths(record):
        relative = _safe_relative_path(source_root, artifact_path)
        physical_root = _artifact_physical_source_root(
            source_root=source_root,
            relative_path=relative,
            tactile_bundle=tactile_bundle,
        )
        target = physical_root / relative
        if relative.name == "manifest.json":
            target = target.parent
        if not target.exists():
            missing.append(
                {
                    "source_dataset_root": str(source_root),
                    "episode_index": int(record["episode_index"]),
                    "artifact_path": str(relative),
                }
            )
    return missing


def _artifact_physical_source_root(
    *,
    source_root: Path,
    relative_path: Path,
    tactile_bundle: AttachedTactileBundle | None,
) -> Path:
    """Return the immutable physical root for one logical source artifact."""

    if relative_path.parts[0] != "tactile" or tactile_bundle is None:
        return source_root
    # Resolve the exact logical artifact first so a directory that merely
    # resembles the expected layout cannot bypass bundle ownership checks.
    tactile_bundle.resolve(relative_path.as_posix())
    return tactile_bundle.bundle_dataset_root


def _strip_native_tactile_bundle_binding(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert a materialized native-bundle record into legacy hot layout."""

    result = dict(value)
    result.pop("tactile_bundle", None)
    metadata = result.get("metadata")
    if isinstance(metadata, Mapping):
        metadata = dict(metadata)
        metadata.pop("tactile_bundle_episode", None)
        result["metadata"] = metadata
    return result


def _source_info_fingerprint(info: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format": info.get("format"),
        "nominal_fps": info.get("nominal_fps"),
        "features": info.get("features"),
        "chunk_size": info.get("chunk_size"),
    }


def plan_dataset_build(
    manifest_path: str | Path,
    *,
    dataset_names: Sequence[str] | None = None,
) -> list[DatasetBuildPlan]:
    """Inspect a manifest and return all planned outputs without writing files."""

    path, manifest = _read_manifest(manifest_path)
    plans: list[DatasetBuildPlan] = []
    output_roots: set[Path] = set()
    for raw_dataset in _selected_dataset_entries(manifest, dataset_names):
        if not isinstance(raw_dataset, Mapping):
            raise DatasetOrganizationError("Every manifest dataset entry must be an object.")
        name = str(raw_dataset.get("name") or "").strip()
        if not name:
            raise DatasetOrganizationError("Every manifest dataset entry requires a non-empty name.")
        output_root = _resolve_manifest_path(path, raw_dataset.get("output_root"), label=f"output_root for {name}")
        if output_root in output_roots:
            raise DatasetOrganizationError(f"Duplicate output_root in manifest: {output_root}")
        output_roots.add(output_root)
        raw_sources = raw_dataset.get("sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise DatasetOrganizationError(f"Dataset {name} requires a non-empty sources list.")
        task = raw_dataset.get("task")
        if not isinstance(task, Mapping):
            raise DatasetOrganizationError(f"Dataset {name} requires a task object.")
        task_id = str(task.get("id") or "").strip()
        task_description = str(task.get("description") or "").strip()
        if not task_id or not task_description:
            raise DatasetOrganizationError(f"Dataset {name} task requires id and description.")
        artifact_policy = _artifact_materialization_policy(raw_dataset)
        progress_policy = _progress_artifact_policy(raw_dataset)
        output_chunk_size = int(raw_dataset.get("chunk_size", 1000))
        if output_chunk_size <= 0:
            raise DatasetOrganizationError(f"Dataset {name} chunk_size must be positive.")
        missing_requested: list[dict[str, Any]] = []
        filtered_by_duration: list[dict[str, Any]] = []
        missing_artifacts: list[dict[str, Any]] = []
        selected: list[tuple[Path, dict[str, Any], tuple[str, ...]]] = []
        minimum_duration_sec = _minimum_duration_sec(manifest, raw_dataset)
        source_fingerprint: dict[str, Any] | None = None
        progress_profile_hash: str | None = None
        task_stage_profile_hash: str | None = None
        progress_identities: set[str] = set()
        progress_episode_uuids: set[str] = set()
        task_rebindings: list[PlannedTaskRebinding] = []
        external_tactile_sources: list[PlannedExternalTactileSource] = []
        for raw_source in raw_sources:
            if not isinstance(raw_source, Mapping):
                raise DatasetOrganizationError(f"Dataset {name} has an invalid source entry.")
            source_root = _resolve_manifest_path(path, raw_source.get("dataset_root"), label=f"dataset_root for {name}")
            tactile_bundle = _attach_manifest_tactile_bundle(
                manifest_path=path,
                raw_source=raw_source,
                source_root=source_root,
            )
            if tactile_bundle is not None:
                external_tactile_sources.append(
                    _planned_external_tactile_source(tactile_bundle)
                )
            if output_root == source_root or output_root.is_relative_to(source_root):
                raise DatasetOrganizationError(f"Output {output_root} must not be inside source {source_root}.")
            source_info = read_json(source_root / "meta" / "info.json")
            source_task_binding = None
            binding_error: DatasetTaskBindingError | None = None
            try:
                source_task_binding = inspect_dataset_task_binding(
                    source_root,
                    require_complete=True,
                )
            except DatasetTaskBindingError as exc:
                binding_error = exc
            rebinding = _task_rebinding_from_source(
                raw_source=raw_source,
                source_root=source_root,
                source_info=source_info,
                task_id=task_id,
                task_description=task_description,
                binding_error=binding_error,
                observed_task_id=(
                    None
                    if source_task_binding is None
                    else source_task_binding.task_id
                ),
            )
            if rebinding is not None:
                task_rebindings.append(rebinding)
            fingerprint = _source_info_fingerprint(source_info)
            if source_fingerprint is None:
                source_fingerprint = fingerprint
            elif fingerprint != source_fingerprint:
                raise DatasetOrganizationError(
                    f"Dataset {name} has incompatible source info metadata. "
                    "Split by schema or explicitly migrate the source first."
                )
            records_by_index = _record_by_episode_index(source_root)
            if rebinding is not None:
                _assert_rebinding_has_no_frame_task_columns(
                    source_root,
                    records_by_index,
                )
            source_progress_profile = None
            source_stage_profile = None
            if progress_policy == _PROGRESS_POLICY_PRESERVE:
                try:
                    source_progress_profile = load_dataset_progress_profile(
                        source_root,
                        require_explicit_binding=True,
                    )
                    if source_progress_profile is None:
                        raise DatasetOrganizationError(
                            f"Dataset has no progress profile: {source_root}."
                        )
                    source_stage_profile = load_dataset_task_stage_profile(
                        source_root,
                        progress_profile=source_progress_profile,
                    )
                except ProgressArtifactReadError as exc:
                    raise DatasetOrganizationError(
                        f"Cannot preserve progress artifacts from {source_root}: {exc}"
                    ) from exc
                observed_progress_hash = (
                    source_progress_profile.semantic_content_hash
                )
                observed_stage_hash = (
                    None
                    if source_stage_profile is None
                    else source_stage_profile.semantic_content_hash
                )
                if progress_profile_hash is None:
                    progress_profile_hash = observed_progress_hash
                    task_stage_profile_hash = observed_stage_hash
                elif (
                    observed_progress_hash != progress_profile_hash
                    or observed_stage_hash != task_stage_profile_hash
                ):
                    raise DatasetOrganizationError(
                        "Progress-preserving merge requires identical progress and "
                        "stage profiles across every source."
                    )
            selected_spec, excluded = _selection_from_source(raw_source)
            transforms = _normalise_transforms(raw_source)
            if selected_spec == "all":
                requested = tuple(sorted(records_by_index))
            else:
                requested = tuple(dict.fromkeys(int(value) for value in selected_spec))
            for episode_index in requested:
                if episode_index not in records_by_index:
                    missing_requested.append(
                        {
                            "source_dataset_root": str(source_root),
                            "episode_index": int(episode_index),
                        }
                    )
                    continue
                if episode_index in excluded:
                    continue
                record = records_by_index[episode_index]
                if not _passes_minimum_duration(record, minimum_duration_sec):
                    filtered_by_duration.append(
                        {
                            "source_dataset_root": str(source_root),
                            "episode_index": int(episode_index),
                            "duration_sec": float(record["duration_sec"]),
                            "minimum_duration_sec": minimum_duration_sec,
                        }
                    )
                    continue
                if progress_policy == _PROGRESS_POLICY_PRESERVE:
                    if source_progress_profile is None:
                        raise AssertionError("validated progress profile disappeared")
                    episode_uuid, identity_hash = _validate_progress_episode(
                        source_root=source_root,
                        record=record,
                        expected_profile_hash=(
                            source_progress_profile.semantic_content_hash
                        ),
                        expected_stage_profile_hash=(
                            None
                            if source_stage_profile is None
                            else source_stage_profile.semantic_content_hash
                        ),
                    )
                    if episode_uuid in progress_episode_uuids:
                        raise DatasetOrganizationError(
                            "Duplicate progress Episode UUID selected across sources: "
                            f"{episode_uuid}."
                        )
                    if identity_hash in progress_identities:
                        raise DatasetOrganizationError(
                            "Duplicate capture identity selected across sources: "
                            f"{identity_hash}."
                        )
                    progress_episode_uuids.add(episode_uuid)
                    progress_identities.add(identity_hash)
                missing_artifacts.extend(
                    _validate_artifacts(
                        source_root,
                        record,
                        tactile_bundle=tactile_bundle,
                    )
                )
                selected.append((source_root, record, transforms))
        if missing_requested:
            raise DatasetOrganizationError(
                f"Dataset {name} requested missing episode indices: {missing_requested}"
            )
        if missing_artifacts:
            raise DatasetOrganizationError(
                f"Dataset {name} has missing declared artifacts: {missing_artifacts[:5]}"
            )
        if not selected:
            raise DatasetOrganizationError(f"Dataset {name} selects no episodes.")
        episodes = tuple(
            PlannedEpisode(
                source_root=str(source_root),
                source_episode_index=int(record["episode_index"]),
                output_episode_index=output_index,
                frame_count=int(record.get("frame_count", 0) or 0),
                transforms=transforms,
            )
            for output_index, (source_root, record, transforms) in enumerate(selected)
        )
        plans.append(
            DatasetBuildPlan(
                name=name,
                output_root=str(output_root),
                task_id=task_id,
                task_description=task_description,
                episode_count=len(episodes),
                frame_count=sum(item.frame_count for item in episodes),
                source_episode_count=len(selected),
                minimum_duration_sec=minimum_duration_sec,
                filtered_by_minimum_duration=tuple(filtered_by_duration),
                missing_requested_indices=tuple(missing_requested),
                missing_artifacts=tuple(missing_artifacts),
                artifact_materialization_policy=artifact_policy,
                progress_artifact_policy=progress_policy,
                progress_profile_hash=progress_profile_hash,
                task_stage_profile_hash=task_stage_profile_hash,
                task_rebindings=tuple(task_rebindings),
                external_tactile_sources=tuple(external_tactile_sources),
                episodes=episodes,
            )
        )
    return plans


def _require_independent_inode(source: Path, destination: Path) -> None:
    source_stat = source.stat()
    destination_stat = destination.stat()
    if (
        source_stat.st_dev == destination_stat.st_dev
        and source_stat.st_ino == destination_stat.st_ino
    ):
        raise DatasetOrganizationError(
            f"Artifact materialization reused the source inode: {source}."
        )
    if destination_stat.st_nlink != 1:
        raise DatasetOrganizationError(
            "Independent artifact destination must have exactly one hard link: "
            f"{destination}."
        )
    if source_stat.st_size != destination_stat.st_size:
        raise DatasetOrganizationError(
            "Artifact size changed during materialization: "
            f"source={source} destination={destination}."
        )


def _files_equal(source: Path, destination: Path) -> bool:
    """Read both files back and require byte-for-byte equality."""

    with source.open("rb") as source_handle, destination.open("rb") as output:
        while True:
            source_block = source_handle.read(4 * 1024 * 1024)
            output_block = output.read(4 * 1024 * 1024)
            if source_block != output_block:
                return False
            if not source_block:
                return True


def _stable_source_signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _copy2_verified(
    source: Path,
    destination: Path,
    *,
    attempts: int = 3,
) -> None:
    """Copy an immutable artifact and verify the persisted bytes before return."""

    expected_source = _stable_source_signature(source)
    for attempt in range(1, max(1, int(attempts)) + 1):
        destination.unlink(missing_ok=True)
        shutil.copy2(source, destination)
        _require_independent_inode(source, destination)
        if _stable_source_signature(source) != expected_source:
            destination.unlink(missing_ok=True)
            raise DatasetOrganizationError(
                f"Source artifact changed during materialization: {source}."
            )
        if _files_equal(source, destination):
            return
        destination.unlink(missing_ok=True)
        if attempt == max(1, int(attempts)):
            raise DatasetOrganizationError(
                "Artifact content changed during independent copy after "
                f"{attempt} verified attempt(s): source={source} "
                f"destination={destination}."
            )


def _reflink_required(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as source_handle, destination.open("xb") as output:
            fcntl.ioctl(output.fileno(), _FICLONE, source_handle.fileno())
            output.flush()
            os.fsync(output.fileno())
        shutil.copystat(source, destination)
        _require_independent_inode(source, destination)
        if not _files_equal(source, destination):
            raise DatasetOrganizationError(
                "Reflink artifact failed byte-for-byte readback verification: "
                f"source={source} destination={destination}."
            )
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _materialize_file(
    source: Path,
    destination: Path,
    *,
    policy: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if policy == _ARTIFACT_POLICY_HARDLINK_OR_COPY:
        try:
            os.link(source, destination)
        except OSError:
            _copy2_verified(source, destination)
        return
    if policy == _ARTIFACT_POLICY_COPY:
        _copy2_verified(source, destination)
        return
    if policy == _ARTIFACT_POLICY_REFLINK_REQUIRED:
        try:
            _reflink_required(source, destination)
        except OSError as exc:
            raise DatasetOrganizationError(
                "Required reflink materialization failed; refusing an implicit "
                f"full-copy fallback for {source}: {exc}"
            ) from exc
        return
    raise AssertionError(f"unsupported artifact policy {policy!r}")


def _copy_artifact(
    source_root: Path,
    staging_root: Path,
    artifact_path: str,
    *,
    output_episode_index: int,
    chunk_size: int,
    artifact_policy: str,
) -> None:
    relative = _safe_relative_path(source_root, artifact_path)
    source = source_root / relative
    destination_relative = _output_relative_path(
        relative,
        output_episode_index=output_episode_index,
        chunk_size=chunk_size,
    )
    if relative.name == "manifest.json":
        source = source.parent
        destination_relative = destination_relative.parent
    destination = staging_root / destination_relative
    if source.is_file():
        _materialize_file(source, destination, policy=artifact_policy)
        return
    if not source.is_dir():
        raise DatasetOrganizationError(f"Declared artifact disappeared during build: {source}")
    for file_path in source.rglob("*"):
        if not file_path.is_file():
            continue
        rel_child = file_path.relative_to(source)
        target = destination / rel_child
        if file_path.name != "manifest.json":
            _materialize_file(file_path, target, policy=artifact_policy)
            continue
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            shutil.copy2(file_path, target)
            continue
        rewritten = _rewrite_paths_in_value(
            payload,
            output_episode_index=output_episode_index,
            chunk_size=chunk_size,
        )
        write_json(target, rewritten)


def _set_table_column(table: Any, name: str, values: Sequence[Any], *, field_type: Any | None = None) -> Any:
    pa, _ = _require_pyarrow()
    index = table.schema.get_field_index(name)
    target_type = field_type
    if target_type is None and index >= 0:
        target_type = table.schema.field(index).type
    if target_type is not None and pa.types.is_null(target_type):
        # Legacy collections created nullable placeholder columns for gripper
        # metadata.  A migration is entitled to give those columns their first
        # concrete type.
        target_type = None
    array = pa.array(values, type=target_type)
    if index >= 0:
        return table.set_column(index, name, array)
    return table.append_column(name, array)


def _migrate_dmtac_vector(value: Any, *, source_slot: int = 1, target_slot: int = 0) -> Any:
    if value is None:
        return None
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise DatasetOrganizationError(f"Expected a six-wide hand vector, got {value!r}") from exc
    if len(vector) != 6:
        raise DatasetOrganizationError(f"Expected a six-wide hand vector, got length={len(vector)}")
    finite = [index for index, item in enumerate(vector) if item == item and abs(item) != float("inf")]
    if not finite:
        return vector
    for index, item in enumerate(vector):
        if index == source_slot or item != item:
            continue
        if abs(item) > 1e-7:
            raise DatasetOrganizationError(
                "DM-Tac slot migration found a non-zero value outside the declared old source slot: "
                f"vector={vector} source_slot={source_slot}"
            )
    scalar = vector[source_slot]
    if not (scalar == scalar and abs(scalar) != float("inf")):
        raise DatasetOrganizationError(
            "DM-Tac slot migration found a partially-valid vector with no finite source scalar: "
            f"vector={vector} source_slot={source_slot}"
        )
    migrated = [0.0] * 6
    migrated[target_slot] = scalar
    return migrated


def _migrate_sliced_vectors(values: Sequence[Any], *, start: int, end: int) -> list[Any]:
    migrated_rows: list[Any] = []
    for value in values:
        if value is None:
            migrated_rows.append(None)
            continue
        vector = list(value)
        if len(vector) < end:
            raise DatasetOrganizationError(
                f"Expected vector length at least {end} while migrating DM-Tac slots, got {len(vector)}"
            )
        vector[start:end] = _migrate_dmtac_vector(vector[start:end])
        migrated_rows.append(vector)
    return migrated_rows


def _update_json_column(table: Any, name: str, updater: Any) -> Any:
    if table.schema.get_field_index(name) < 0:
        return table
    updated: list[Any] = []
    for raw in table[name].to_pylist():
        if raw is None:
            updated.append(None)
            continue
        try:
            payload = json.loads(str(raw))
        except json.JSONDecodeError:
            updated.append(raw)
            continue
        updated.append(json.dumps(updater(payload), ensure_ascii=True, sort_keys=True))
    return _set_table_column(table, name, updated)


def _rewrite_parquet_episode_paths(
    table: Any,
    *,
    output_episode_index: int,
    chunk_size: int,
) -> Any:
    """Rewrite per-frame artifact paths and JSON payloads after episode re-indexing."""

    pa, _ = _require_pyarrow()
    for field in tuple(table.schema):
        name = field.name
        if not pa.types.is_string(field.type):
            continue
        values = table[name].to_pylist()
        if not any(isinstance(value, str) and "episode_" in value for value in values):
            continue
        if name.endswith("_json"):
            rewritten_values: list[Any] = []
            for value in values:
                if value is None:
                    rewritten_values.append(None)
                    continue
                try:
                    payload = json.loads(str(value))
                except json.JSONDecodeError:
                    rewritten_values.append(value)
                    continue
                rewritten_values.append(
                    json.dumps(
                        _rewrite_paths_in_value(
                            payload,
                            output_episode_index=output_episode_index,
                            chunk_size=chunk_size,
                        ),
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                )
            table = _set_table_column(table, name, rewritten_values)
            continue
        if not (name.endswith(".path") or name.endswith("_path") or name == "parquet_path"):
            continue
        rewritten_values = [
            None
            if value is None
            else _rewrite_paths_in_value(
                value,
                output_episode_index=output_episode_index,
                chunk_size=chunk_size,
            )
            for value in values
        ]
        table = _set_table_column(table, name, rewritten_values)
    return table


def _apply_dmtac_slot_migration(table: Any) -> Any:
    """Move legacy DM-Tac slot 1 into the canonical slot 0 in a parquet table."""

    for column, side_slices in _VECTOR_COLUMN_SPECS:
        if table.schema.get_field_index(column) < 0:
            continue
        values = table[column].to_pylist()
        for start, end in side_slices:
            values = _migrate_sliced_vectors(values, start=start, end=end)
        table = _set_table_column(table, column, values)
    for column in _HAND_VECTOR_COLUMNS:
        if table.schema.get_field_index(column) < 0:
            continue
        values = [_migrate_dmtac_vector(value) for value in table[column].to_pylist()]
        table = _set_table_column(table, column, values)
    for side in _DMTAC_SIDES:
        output_dim_column = f"action.gripper.{side}.output_dim"
        table = _set_table_column(table, output_dim_column, [0] * table.num_rows, field_type=None)
        slot_column = f"action.gripper.{side}.slot_index"
        table = _set_table_column(table, slot_column, [0] * table.num_rows, field_type=None)
        action_dim_column = f"action.gripper.{side}.action_dim"
        table = _set_table_column(table, action_dim_column, [1] * table.num_rows, field_type=None)
    specs, masks = _dmtac_specs()
    table = _update_json_column(table, "hand_action_specs_json", lambda _payload: specs)
    table = _update_json_column(table, "hand_active_mask_by_side_json", lambda _payload: masks)
    table = _update_json_column(table, "sample_metadata_json", _update_dmtac_metadata)
    return table


def _rewrite_parquet(
    source_path: Path,
    destination_path: Path,
    *,
    output_episode_index: int,
    output_frame_offset: int,
    chunk_size: int,
    transforms: Sequence[str],
) -> None:
    pa, pq = _require_pyarrow()
    del pa
    table = pq.read_table(source_path)
    if table.schema.get_field_index("episode_index") >= 0:
        table = _set_table_column(table, "episode_index", [int(output_episode_index)] * table.num_rows)
    if table.schema.get_field_index("index") >= 0:
        table = _set_table_column(
            table,
            "index",
            [int(output_frame_offset) + offset for offset in range(table.num_rows)],
        )
    table = _rewrite_parquet_episode_paths(
        table,
        output_episode_index=output_episode_index,
        chunk_size=chunk_size,
    )
    if _DMTAC_SLOT_MIGRATION in transforms:
        table = _apply_dmtac_slot_migration(table)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination_path, compression="zstd")


def _rewrite_episode_record(
    source_record: Mapping[str, Any],
    *,
    output_episode_index: int,
    chunk_size: int,
    transforms: Sequence[str],
) -> dict[str, Any]:
    record = dict(source_record)
    record["episode_index"] = int(output_episode_index)
    record["episode_id"] = f"episode_{int(output_episode_index):06d}"
    for key in (
        "parquet_path",
        "video_paths",
        "tactile_paths",
        "wrist_depth_mask_paths",
        "wrist_depth_raw_paths",
        "summary",
    ):
        if key in record:
            record[key] = _rewrite_paths_in_value(
                record[key],
                output_episode_index=output_episode_index,
                chunk_size=chunk_size,
            )
    if _DMTAC_SLOT_MIGRATION in transforms:
        record = _update_dmtac_metadata(record)
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}
        metadata = dict(metadata)
        metadata["dmtac_slot_migration"] = {
            "schema": "dexteleop.dmtac_slot_migration.v1",
            "legacy_source_slot": 1,
            "canonical_target_slot": 0,
            "rule": "legacy six-wide DM-Tac scalar slot 1 was moved to canonical slot 0",
        }
        record["metadata"] = metadata
    return record


def _rewrite_stats_record(source_stats: Mapping[str, Any], *, output_episode_index: int) -> dict[str, Any]:
    record = dict(source_stats)
    record["episode_index"] = int(output_episode_index)
    return record


def _copy_source_devices(source_roots: Iterable[Path], staging_root: Path) -> None:
    seen: set[Path] = set()
    for source_root in source_roots:
        source_root = source_root.resolve()
        if source_root in seen:
            continue
        seen.add(source_root)
        source_devices = source_root / "meta" / "devices"
        if source_devices.is_dir():
            destination = staging_root / "meta" / "source_devices" / source_root.name
            shutil.copytree(source_devices, destination, dirs_exist_ok=False)

        # A derived dataset can itself carry the original per-session camera
        # metadata under ``meta/source_devices``.  Preserve that whole tree
        # when creating another derived/re-indexed dataset; otherwise a
        # copy-on-write cleanup would silently discard calibration provenance.
        inherited_source_devices = source_root / "meta" / "source_devices"
        if inherited_source_devices.is_dir():
            shutil.copytree(
                inherited_source_devices,
                staging_root / "meta" / "source_devices",
                dirs_exist_ok=True,
            )


def _copy_tree_independent(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        if path.is_symlink():
            raise DatasetOrganizationError(
                f"Refusing a symlink in immutable semantic artifacts: {path}."
            )
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    for source_file in source.rglob("*"):
        if not source_file.is_file():
            continue
        destination_file = destination / source_file.relative_to(source)
        _require_independent_inode(source_file, destination_file)


def _copy_progress_episode_artifacts(
    *,
    source_root: Path,
    staging_root: Path,
    source_record: Mapping[str, Any],
) -> None:
    episode_uuid = _safe_progress_episode_uuid(source_record)
    source_episode_dir = (
        source_root / "meta" / "progress" / "episodes" / episode_uuid
    )
    destination_episode_dir = (
        staging_root / "meta" / "progress" / "episodes" / episode_uuid
    )
    if destination_episode_dir.exists():
        raise DatasetOrganizationError(
            f"Duplicate progress Episode destination: {episode_uuid}."
        )
    _copy_tree_independent(source_episode_dir, destination_episode_dir)
    for relative_root in (
        STAGE_ANNOTATION_RELATIVE_ROOT,
        DETAIL_STAGE_ANNOTATION_RELATIVE_ROOT,
    ):
        source_annotation_dir = source_root / relative_root / episode_uuid
        if not source_annotation_dir.exists():
            continue
        destination_annotation_dir = (
            staging_root / relative_root / episode_uuid
        )
        if destination_annotation_dir.exists():
            raise DatasetOrganizationError(
                "Duplicate progress annotation destination: "
                f"{destination_annotation_dir}."
            )
        _copy_tree_independent(
            source_annotation_dir,
            destination_annotation_dir,
        )


def _copy_shared_metadata_file(
    *,
    source_roots: Iterable[Path],
    relative_path: Path,
    staging_root: Path,
    required: bool,
) -> None:
    existing: list[Path] = []
    missing: list[Path] = []
    for source_root in dict.fromkeys(path.resolve() for path in source_roots):
        source = source_root / relative_path
        if source.is_file():
            existing.append(source)
        else:
            missing.append(source_root)
    if required and missing:
        raise DatasetOrganizationError(
            f"Required shared metadata {relative_path} is absent from {missing}."
        )
    if not existing:
        return
    if missing:
        raise DatasetOrganizationError(
            f"Sources mix present and absent shared metadata {relative_path}."
        )
    first_bytes = existing[0].read_bytes()
    if any(path.read_bytes() != first_bytes for path in existing[1:]):
        raise DatasetOrganizationError(
            f"Sources disagree on shared metadata {relative_path}."
        )
    destination = staging_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(existing[0], destination)
    _require_independent_inode(existing[0], destination)


def _validate_output_progress(
    *,
    output_root: Path,
    output_episodes: Sequence[Mapping[str, Any]],
    expected_profile_hash: str,
    expected_stage_profile_hash: str | None,
) -> None:
    profile = load_dataset_progress_profile(
        output_root,
        require_explicit_binding=True,
    )
    if profile is None or profile.semantic_content_hash != expected_profile_hash:
        raise DatasetOrganizationError(
            "Published progress profile differs from the merge plan."
        )
    stage_profile = load_dataset_task_stage_profile(
        output_root,
        progress_profile=profile,
    )
    observed_stage_hash = (
        None
        if stage_profile is None
        else stage_profile.semantic_content_hash
    )
    if observed_stage_hash != expected_stage_profile_hash:
        raise DatasetOrganizationError(
            "Published task stage profile differs from the merge plan."
        )
    identities: set[str] = set()
    episode_uuids: set[str] = set()
    for record in output_episodes:
        episode_uuid, identity_hash = _validate_progress_episode(
            source_root=output_root,
            record=record,
            expected_profile_hash=expected_profile_hash,
            expected_stage_profile_hash=expected_stage_profile_hash,
        )
        if episode_uuid in episode_uuids or identity_hash in identities:
            raise DatasetOrganizationError(
                "Merged output contains duplicate progress identity."
            )
        episode_uuids.add(episode_uuid)
        identities.add(identity_hash)


def _readme_text(
    raw_dataset: Mapping[str, Any],
    plan: DatasetBuildPlan,
    *,
    source_summary: Sequence[Mapping[str, Any]],
) -> str:
    labels = raw_dataset.get("labels") if isinstance(raw_dataset.get("labels"), Mapping) else {}
    notes = raw_dataset.get("notes")
    note_lines = [str(note) for note in notes] if isinstance(notes, Sequence) and not isinstance(notes, str) else ([str(notes)] if notes else [])
    lines = [
        f"# {plan.name}",
        "",
        "这是由 `utils.dataset_postprocess.organize_episodes` 生成的派生 LeRobot v2.1-style 数据集。",
        "原始数据集未被移动或修改。",
        (
            "大型视频、触觉和深度二进制流采用 "
            f"`{plan.artifact_materialization_policy}` 策略物化。"
        ),
        "",
        "## 分类",
        "",
    ]
    for key, value in sorted(labels.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            f"- task_id: {plan.task_id}",
            f"- episode_count: {plan.episode_count}",
            f"- frame_count: {plan.frame_count}",
            "",
            "## 来源与重编号",
            "",
            "输出 episode 编号连续为 `0..N-1`。原始数据集路径、原始 episode 编号及任何迁移规则记录在 `meta/provenance.jsonl`。",
            (
                "Episode 顺序严格保留 manifest 中的 source 顺序及各 source "
                "内部 episode 顺序，用于复现合并前的训练输入顺序。"
            ),
            "",
        ]
    )
    if plan.minimum_duration_sec is not None:
        lines.extend(
            [
                "",
                "## 来源筛选规则",
                "",
                f"- 仅保留原始 `duration_sec >= {plan.minimum_duration_sec:g}` 的 episode。",
                f"- 因此从本输出分支排除 {len(plan.filtered_by_minimum_duration)} 个较短 episode；"
                "完整原始编号见 `meta/build_audit.json`。",
            ]
        )
    for summary in source_summary:
        lines.append(
            f"- `{summary['source_dataset_root']}`: {summary['episode_count']} episodes"
            + (f"; transforms={summary['transforms']}" if summary["transforms"] else "")
        )
    if note_lines:
        lines.extend(["", "## 使用与已知限制", ""])
        lines.extend(f"- {note}" for note in note_lines)
    lines.extend(
        [
            "",
            "## 完整性",
            "",
            "构建时已验证每个 `meta/episodes.jsonl` 声明的 parquet、视频、触觉和 wrist-depth artifact 均存在。",
            "逐 episode 来源及文件完整性结果见 `meta/build_audit.json`。",
            "",
        ]
    )
    return "\n".join(lines)


def _prepare_output(output_root: Path, *, overwrite: bool) -> Path:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}")
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output_root.parent / f".{output_root.name}.staging-{uuid.uuid4().hex}"
    staging_root.mkdir(parents=False, exist_ok=False)
    return staging_root


def _published_output(staging_root: Path, output_root: Path) -> None:
    if output_root.exists():  # Defensive check for a concurrent creator.
        raise FileExistsError(f"Output root appeared during build: {output_root}")
    os.replace(staging_root, output_root)


def build_datasets(
    manifest_path: str | Path,
    *,
    overwrite: bool = False,
    dataset_names: Sequence[str] | None = None,
) -> list[DatasetBuildPlan]:
    """Materialise every dataset in a manifest while leaving all sources immutable."""

    manifest_file, manifest = _read_manifest(manifest_path)
    raw_datasets = _selected_dataset_entries(manifest, dataset_names)
    plans = plan_dataset_build(manifest_file, dataset_names=dataset_names)
    for raw_dataset, plan in zip(raw_datasets, plans, strict=True):
        if not isinstance(raw_dataset, Mapping):  # Already validated by plan, for type narrowing.
            raise AssertionError("validated dataset entry became non-mapping")
        output_root = Path(plan.output_root)
        chunk_size = int(raw_dataset.get("chunk_size", 1000))
        artifact_policy = plan.artifact_materialization_policy
        progress_policy = plan.progress_artifact_policy
        minimum_duration_sec = _minimum_duration_sec(manifest, raw_dataset)
        task_rebinding_by_root = {
            Path(item.source_root): item for item in plan.task_rebindings
        }
        staging_root = _prepare_output(output_root, overwrite=bool(overwrite))
        try:
            source_entries: list[tuple[Path, dict[str, Any], tuple[str, ...]]] = []
            source_stats_by_key: dict[tuple[Path, int], dict[str, Any]] = {}
            source_infos: dict[Path, dict[str, Any]] = {}
            source_tactile_bundles: dict[Path, AttachedTactileBundle] = {}
            source_summary: list[dict[str, Any]] = []
            for raw_source in raw_dataset["sources"]:
                source_root = _resolve_manifest_path(manifest_file, raw_source.get("dataset_root"), label="dataset_root")
                tactile_bundle = _attach_manifest_tactile_bundle(
                    manifest_path=manifest_file,
                    raw_source=raw_source,
                    source_root=source_root,
                )
                if tactile_bundle is not None:
                    previous = source_tactile_bundles.get(source_root)
                    if (
                        previous is not None
                        and previous.tactile_storage_root
                        != tactile_bundle.tactile_storage_root
                    ):
                        raise DatasetOrganizationError(
                            "One source root was bound to multiple tactile storage "
                            f"roots: {source_root}."
                        )
                    source_tactile_bundles[source_root] = tactile_bundle
                records_by_index = _record_by_episode_index(source_root)
                stats_by_index = {
                    int(record["episode_index"]): record
                    for record in read_jsonl(source_root / "meta" / "episodes_stats.jsonl")
                    if "episode_index" in record
                }
                selected_spec, excluded = _selection_from_source(raw_source)
                requested = tuple(sorted(records_by_index)) if selected_spec == "all" else tuple(selected_spec)
                transforms = _normalise_transforms(raw_source)
                selected_count = 0
                for source_episode_index in requested:
                    if source_episode_index in excluded:
                        continue
                    record = records_by_index[source_episode_index]
                    if not _passes_minimum_duration(record, minimum_duration_sec):
                        continue
                    source_entries.append((source_root, record, transforms))
                    if source_episode_index in stats_by_index:
                        source_stats_by_key[(source_root, source_episode_index)] = stats_by_index[source_episode_index]
                    selected_count += 1
                source_infos[source_root] = read_json(source_root / "meta" / "info.json")
                summary = {
                    "source_dataset_root": str(source_root),
                    "episode_count": selected_count,
                    "transforms": list(transforms),
                }
                task_rebinding = task_rebinding_by_root.get(source_root)
                if task_rebinding is not None:
                    _snapshot, current_snapshot_sha256 = (
                        _source_task_binding_snapshot(
                            source_root,
                            source_infos[source_root],
                        )
                    )
                    if current_snapshot_sha256 != task_rebinding.source_binding_sha256:
                        raise DatasetOrganizationError(
                            "Source task metadata changed after preflight planning: "
                            f"root={source_root}."
                        )
                    summary["reviewed_task_rebinding"] = asdict(task_rebinding)
                if tactile_bundle is not None:
                    summary["external_tactile_source"] = asdict(
                        _planned_external_tactile_source(tactile_bundle)
                    )
                source_summary.append(summary)
            if len(source_entries) != plan.episode_count:
                raise DatasetOrganizationError("Build selection no longer matches its preflight plan.")
            source_roots = tuple(entry[0] for entry in source_entries)
            if progress_policy == _PROGRESS_POLICY_PRESERVE:
                if plan.progress_profile_hash is None:
                    raise AssertionError("progress-preserving plan has no profile hash")
                _copy_shared_metadata_file(
                    source_roots=source_roots,
                    relative_path=Path("meta/progress/profile.cjson"),
                    staging_root=staging_root,
                    required=True,
                )
                _copy_shared_metadata_file(
                    source_roots=source_roots,
                    relative_path=Path("meta/progress/stages.cjson"),
                    staging_root=staging_root,
                    required=plan.task_stage_profile_hash is not None,
                )
            output_episodes: list[dict[str, Any]] = []
            output_stats: list[dict[str, Any]] = []
            provenance: list[dict[str, Any]] = []
            global_frame_offset = 0
            for output_index, (source_root, source_record, transforms) in enumerate(source_entries):
                rewritten_record = _rewrite_episode_record(
                    source_record,
                    output_episode_index=output_index,
                    chunk_size=chunk_size,
                    transforms=transforms,
                )
                tactile_bundle = source_tactile_bundles.get(source_root)
                if tactile_bundle is not None:
                    rewritten_record = _strip_native_tactile_bundle_binding(
                        rewritten_record
                    )
                source_parquet_relative = _safe_relative_path(source_root, str(source_record["parquet_path"]))
                output_parquet_relative = _safe_relative_path(staging_root, str(rewritten_record["parquet_path"]))
                _rewrite_parquet(
                    source_root / source_parquet_relative,
                    staging_root / output_parquet_relative,
                    output_episode_index=output_index,
                    output_frame_offset=global_frame_offset,
                    chunk_size=chunk_size,
                    transforms=transforms,
                )
                for artifact_path in episode_artifact_paths(source_record):
                    if str(artifact_path) == str(source_record.get("parquet_path")):
                        continue
                    relative_artifact = _safe_relative_path(
                        source_root,
                        artifact_path,
                    )
                    physical_source_root = _artifact_physical_source_root(
                        source_root=source_root,
                        relative_path=relative_artifact,
                        tactile_bundle=tactile_bundle,
                    )
                    _copy_artifact(
                        physical_source_root,
                        staging_root,
                        artifact_path,
                        output_episode_index=output_index,
                        chunk_size=chunk_size,
                        artifact_policy=artifact_policy,
                    )
                if progress_policy == _PROGRESS_POLICY_PRESERVE:
                    _copy_progress_episode_artifacts(
                        source_root=source_root,
                        staging_root=staging_root,
                        source_record=source_record,
                    )
                output_episodes.append(rewritten_record)
                source_index = int(source_record["episode_index"])
                source_stats = source_stats_by_key.get((source_root, source_index))
                if source_stats is not None:
                    output_stats.append(_rewrite_stats_record(source_stats, output_episode_index=output_index))
                provenance.append(
                    {
                        "output_episode_index": output_index,
                        "output_episode_id": rewritten_record["episode_id"],
                        "source_dataset_root": str(source_root),
                        "source_episode_index": source_index,
                        "source_episode_id": source_record.get("episode_id"),
                        "frame_count": int(source_record.get("frame_count", 0) or 0),
                        "transforms": list(transforms),
                    }
                )
                global_frame_offset += int(source_record.get("frame_count", 0) or 0)
            first_source_root = source_entries[0][0]
            info = dict(source_infos[first_source_root])
            info["dataset_name"] = plan.name
            info["dataset_dir_name"] = plan.name
            info["episode_count"] = len(output_episodes)
            info["frame_count"] = global_frame_offset
            info["task_id"] = plan.task_id
            info["task_description"] = plan.task_description
            info["source_datasets"] = source_summary
            if source_tactile_bundles:
                info.pop("tactile_bundle", None)
            if any(record.get("tactile_paths") for record in output_episodes):
                # Source-side integrity inventories bind source-relative paths
                # and cannot be copied after re-indexing.  The merged payload
                # stays self-contained, but detachable-artifact integrity must
                # be rebuilt by the existing explicit offline full scan.
                info["tactile_artifact_integrity"] = False
                info["tactile_artifact_integrity_merge_status"] = (
                    "legacy_unsealed_requires_offline_full_scan"
                )
            if any(_DMTAC_SLOT_MIGRATION in entry[2] for entry in source_entries):
                info = _update_dmtac_metadata(info)
                info["dmtac_slot_migration"] = {
                    "schema": "dexteleop.dmtac_slot_migration.v1",
                    "legacy_source_slot": 1,
                    "canonical_target_slot": 0,
                    "rule": "legacy six-wide DM-Tac scalar slot 1 was moved to canonical slot 0",
                }
            history = list(info.get("postprocess_history", ())) if isinstance(info.get("postprocess_history"), list) else []
            history.append(
                {
                    "tool": "utils.dataset_postprocess.organize_episodes",
                    "updated_at": time.time(),
                    "manifest": str(manifest_file),
                    "output_root": str(output_root),
                    "episode_count": len(output_episodes),
                    "frame_count": global_frame_offset,
                    "source_datasets": source_summary,
                }
            )
            info["postprocess_history"] = history
            meta_dir = staging_root / "meta"
            write_jsonl(meta_dir / "episodes.jsonl", output_episodes)
            write_jsonl(meta_dir / "episodes_stats.jsonl", output_stats)
            write_jsonl(
                meta_dir / "tasks.jsonl",
                [{"task_id": plan.task_id, "description": plan.task_description}],
            )
            write_jsonl(meta_dir / "provenance.jsonl", provenance)
            write_json(meta_dir / "info.json", info)
            _copy_shared_metadata_file(
                source_roots=source_roots,
                relative_path=Path("meta/modality_validity.json"),
                staging_root=staging_root,
                required=False,
            )
            write_json(
                meta_dir / "build_audit.json",
                {
                    "schema": "dexteleop.dataset_organization_audit.v1",
                    "created_at": time.time(),
                    "manifest": str(manifest_file),
                    "plan": plan.to_dict(),
                    "source_summary": source_summary,
                    "artifact_policy": (
                        "parquet/metadata rewritten; unchanged "
                        "video/tactile/wrist-depth files materialized with "
                        f"{artifact_policy}"
                    ),
                    "artifact_materialization_integrity": (
                        "independent copy/reflink bytes read back and compared "
                        "before atomic publication"
                    ),
                    "progress_artifact_policy": progress_policy,
                    "progress_profile_hash": plan.progress_profile_hash,
                    "task_stage_profile_hash": plan.task_stage_profile_hash,
                    "tactile_integrity_policy": (
                        "source inventories omitted after path re-indexing; "
                        "offline full scan required before detach"
                    ),
                    "external_tactile_sources": [
                        asdict(item) for item in plan.external_tactile_sources
                    ],
                },
            )
            shutil.copy2(manifest_file, meta_dir / "sort_manifest.json")
            _copy_source_devices((entry[0] for entry in source_entries), staging_root)
            (staging_root / "README.md").write_text(
                _readme_text(raw_dataset, plan, source_summary=source_summary),
                encoding="utf-8",
            )
            if progress_policy == _PROGRESS_POLICY_PRESERVE:
                if plan.progress_profile_hash is None:
                    raise AssertionError("progress-preserving plan has no profile hash")
                _validate_output_progress(
                    output_root=staging_root,
                    output_episodes=output_episodes,
                    expected_profile_hash=plan.progress_profile_hash,
                    expected_stage_profile_hash=plan.task_stage_profile_hash,
                )
            if any(record.get("tactile_paths") for record in output_episodes):
                try:
                    assert_dataset_capture_writable(staging_root)
                except DatasetCaptureWritableError as exc:
                    raise DatasetOrganizationError(
                        "Materialized tactile output failed self-contained "
                        f"admission: {exc}"
                    ) from exc
            _published_output(staging_root, output_root)
        except BaseException:
            shutil.rmtree(staging_root, ignore_errors=True)
            raise
    return plans


__all__ = [
    "DatasetBuildPlan",
    "DatasetOrganizationError",
    "PlannedExternalTactileSource",
    "PlannedEpisode",
    "build_datasets",
    "plan_dataset_build",
]

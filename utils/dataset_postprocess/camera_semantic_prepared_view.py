"""Build camera-corrected numeric/video views without touching tactile payloads."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dataset_contracts.camera_semantic_prepared_view import (
    CAMERA_SEMANTIC_PREPARED_VIEW_RELATIVE_PATH,
    CAMERA_SEMANTIC_VIDEO_INTEGRITY_EXCLUSIONS_RELATIVE_PATH,
    CameraSemanticPreparedViewContractError,
    CameraSemanticPreparedViewFileV1,
    CameraSemanticPreparedViewManifestV1,
    CameraSemanticVideoIntegrityExclusionsV1,
    CameraSemanticVideoIntegrityIssueV1,
)
from dataset_contracts.camera_semantics import (
    CAMERA_SEMANTIC_MATERIALIZATION_HASH_DOMAIN_V1,
    CAMERA_SEMANTIC_MATERIALIZATION_SCHEMA_V1,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_interface.camera_semantic_prepared_view import (
    load_camera_semantic_prepared_view,
)
from dataset_interface.camera_semantics import (
    CAMERA_SEMANTIC_MATERIALIZATION_RELATIVE_PATH,
    DatasetCameraSemanticResolver,
    load_camera_role_binding,
    resolve_camera_snapshot,
)

from .camera_semantic_materialize import (
    CAMERA_ARTIFACT_MAP_KEYS,
    CAMERA_COLUMN_PREFIX,
    CameraSemanticMaterializationError,
    _camera_role_in_path,
    _camera_snapshot_from_binding,
    _declared_video_frame_count,
    _episode_index_in_path,
    _episode_records,
    _frame_index_mismatch_count,
    _fresh_copy,
    _fsync_directory,
    _index_values,
    _map_camera_artifact_path,
    _parquet_module,
    _resolution_maps,
    _rewrite_camera_json,
    _rewrite_episode_record,
    _rewrite_info,
    _rewrite_parquet,
    _safe_relative_path,
    _video_probe_frame_count,
    _write_json_fsync,
    _write_jsonl_fsync,
)
from .io import read_json, read_jsonl


class CameraSemanticPreparedViewError(CameraSemanticMaterializationError):
    """Raised when a non-tactile camera view cannot be built safely."""


@dataclass(frozen=True)
class InvalidCameraPreparedViewQuarantinePlan:
    source_root: str
    audit_root: str
    destination_root: str
    source_manifest_sha256: str
    claimed_manifest_hash: str
    source_inventory_hash: str
    file_count: int
    total_bytes: int
    invalid_reason: str
    plan_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_EXCLUDED_META_PREFIXES = (
    Path("meta/modality_artifacts/tactile"),
    Path("meta/prepared_views"),
    Path("meta/tactile_bundle_v1"),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _is_excluded_path(relative: Path) -> bool:
    if any(part.endswith(".lock") for part in relative.parts):
        return True
    tactile_metadata = (
        bool(relative.parts)
        and relative.parts[0] == "meta"
        and any("tactile" in part.lower() for part in relative.parts[1:])
    )
    return tactile_metadata or any(
        relative == prefix or prefix in relative.parents
        for prefix in _EXCLUDED_META_PREFIXES
    ) or relative == Path(CAMERA_SEMANTIC_PREPARED_VIEW_RELATIVE_PATH)


def _artifact_paths(value: object) -> Iterable[object]:
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _artifact_paths(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _artifact_paths(item)
    elif isinstance(value, (str, Path)) and str(value).strip():
        yield value


def _add_artifact_family(
    files: set[Path],
    *,
    root: Path,
    raw_path: object,
    label: str,
) -> None:
    relative = _safe_relative_path(root, raw_path, label=label)
    path = root / relative
    if path.is_symlink() or not path.exists():
        raise CameraSemanticPreparedViewError(
            f"Prepared-view artifact is missing or unsafe: {path}"
        )
    if path.is_dir():
        family = tuple(sorted(path.rglob("*")))
    elif path.suffix.lower() == ".json" and "wrist" in relative.as_posix():
        family = tuple(sorted(path.parent.rglob("*")))
    else:
        family = (path,)
    for member in family:
        if member.is_symlink():
            raise CameraSemanticPreparedViewError(
                f"Prepared view refuses symlinks: {member}"
            )
        if member.is_file():
            member_relative = member.relative_to(root)
            if _is_excluded_path(member_relative):
                raise CameraSemanticPreparedViewError(
                    f"Camera artifact unexpectedly resolves into tactile storage: {member_relative}"
                )
            files.add(member_relative)


def _selected_source_files(
    root: Path,
    episodes: tuple[Mapping[str, Any], ...],
) -> tuple[Path, ...]:
    """Select numeric, camera and small metadata files without walking tactile/."""

    files: set[Path] = set()
    meta_root = root / "meta"
    if not meta_root.is_dir():
        raise CameraSemanticPreparedViewError(f"Dataset meta/ is missing: {root}")
    for current, directory_names, file_names in os.walk(meta_root, topdown=True):
        current_path = Path(current)
        kept_directories: list[str] = []
        for name in sorted(directory_names):
            candidate = current_path / name
            relative = candidate.relative_to(root)
            if _is_excluded_path(relative):
                continue
            if candidate.is_symlink():
                raise CameraSemanticPreparedViewError(
                    f"Prepared view refuses metadata symlinks: {candidate}"
                )
            kept_directories.append(name)
        directory_names[:] = kept_directories
        for name in sorted(file_names):
            path = current_path / name
            relative = path.relative_to(root)
            if _is_excluded_path(relative):
                continue
            if path.is_symlink() or not path.is_file():
                raise CameraSemanticPreparedViewError(
                    f"Prepared view refuses unsafe metadata: {path}"
                )
            files.add(relative)

    for record in episodes:
        episode_index = int(record["episode_index"])
        parquet = _safe_relative_path(
            root,
            record.get("parquet_path"),
            label=f"episode {episode_index} parquet_path",
        )
        parquet_path = root / parquet
        if parquet_path.is_symlink() or not parquet_path.is_file():
            raise CameraSemanticPreparedViewError(
                f"Episode Parquet is missing or unsafe: {parquet_path}"
            )
        files.add(parquet)
        for key in CAMERA_ARTIFACT_MAP_KEYS:
            for raw_path in _artifact_paths(record.get(key)):
                _add_artifact_family(
                    files,
                    root=root,
                    raw_path=raw_path,
                    label=f"episode {episode_index} {key}",
                )
    files.discard(Path("meta/episodes.jsonl"))
    files.discard(Path("meta/episodes_stats.jsonl"))
    files.discard(Path("meta/info.json"))
    files.discard(Path("meta/devices/realsense.json"))
    files.discard(Path(CAMERA_SEMANTIC_MATERIALIZATION_RELATIVE_PATH))
    # Rewritten files are added back to the selected inventory so a stale plan
    # still detects changes to these source inputs.
    for required in (
        Path("meta/episodes.jsonl"),
        Path("meta/info.json"),
        Path("meta/devices/realsense.json"),
    ):
        path = root / required
        if not path.is_file() or path.is_symlink():
            raise CameraSemanticPreparedViewError(
                f"Required prepared-view source metadata is missing: {path}"
            )
        files.add(required)
    optional_episode_stats = Path("meta/episodes_stats.jsonl")
    if (root / optional_episode_stats).is_file():
        files.add(optional_episode_stats)
    return tuple(sorted(files))


def _selected_inventory(
    root: Path,
    episodes: tuple[Mapping[str, Any], ...],
) -> tuple[tuple[Path, ...], str, int, int]:
    files = _selected_source_files(root, episodes)
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    hardlinked = 0
    for relative in files:
        stat = (root / relative).stat()
        total_bytes += int(stat.st_size)
        hardlinked += int(stat.st_nlink > 1)
        entries.append(
            {
                "path": relative.as_posix(),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "device": int(stat.st_dev),
                "inode": int(stat.st_ino),
                "nlink": int(stat.st_nlink),
            }
        )
    return (
        files,
        canonical_content_hash(
            entries,
            domain="dexteleop.camera_semantic_prepared_source_inventory.v1",
        ),
        total_bytes,
        hardlinked,
    )


def _source_dataset_identity(root: Path) -> str:
    info = read_json(root / "meta/info.json")
    if not info:
        raise CameraSemanticPreparedViewError("Source meta/info.json is invalid.")
    return canonical_content_hash(
        info,
        domain="dexteleop.camera_semantic_prepared_source_dataset.v1",
    )


def plan_invalid_camera_prepared_view_quarantine(
    source_root: str | Path,
    audit_root: str | Path,
) -> InvalidCameraPreparedViewQuarantinePlan:
    """Plan a recoverable same-filesystem quarantine for a proven-invalid view."""

    source = Path(source_root).expanduser().resolve()
    audit = Path(audit_root).expanduser().resolve()
    if not source.is_dir() or source.is_symlink():
        raise CameraSemanticPreparedViewError(
            f"Invalid prepared-view quarantine source is missing or unsafe: {source}"
        )
    manifest_path = source / CAMERA_SEMANTIC_PREPARED_VIEW_RELATIVE_PATH
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise CameraSemanticPreparedViewError(
            "Quarantine requires an existing non-symlink prepared-view manifest: "
            f"{manifest_path}"
        )
    raw_manifest = read_json(manifest_path)
    claimed_manifest_hash = str(raw_manifest.get("manifest_hash", "")).strip()
    if len(claimed_manifest_hash) != 64 or any(
        char not in "0123456789abcdef" for char in claimed_manifest_hash
    ):
        raise CameraSemanticPreparedViewError(
            "Quarantine requires a lowercase 64-hex claimed manifest_hash."
        )
    try:
        load_camera_semantic_prepared_view(source, verify_payload_hashes=True)
    except CameraSemanticPreparedViewContractError as exc:
        invalid_reason = f"{type(exc).__name__}: {exc}"
    else:
        raise CameraSemanticPreparedViewError(
            "Refusing to quarantine a valid camera-semantic prepared view."
        )

    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise CameraSemanticPreparedViewError(
                f"Quarantine refuses prepared views containing symlinks: {path}"
            )
        if not path.is_file():
            continue
        stat = path.stat()
        total_bytes += int(stat.st_size)
        entries.append(
            {
                "path": path.relative_to(source).as_posix(),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "device": int(stat.st_dev),
                "inode": int(stat.st_ino),
                "nlink": int(stat.st_nlink),
            }
        )
    inventory_hash = canonical_content_hash(
        entries,
        domain="dexteleop.invalid_camera_prepared_view_inventory.v1",
    )
    destination = audit / f"{source.name}.invalid-{inventory_hash[:12]}"
    payload = {
        "source_root": str(source),
        "audit_root": str(audit),
        "destination_root": str(destination),
        "source_manifest_sha256": _sha256_file(manifest_path),
        "claimed_manifest_hash": claimed_manifest_hash,
        "source_inventory_hash": inventory_hash,
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "invalid_reason": invalid_reason,
    }
    return InvalidCameraPreparedViewQuarantinePlan(
        **payload,
        plan_hash=canonical_content_hash(
            payload,
            domain="dexteleop.invalid_camera_prepared_view_quarantine_plan.v1",
        ),
    )


def quarantine_invalid_camera_prepared_view(
    source_root: str | Path,
    audit_root: str | Path,
    *,
    expected_plan_hash: str,
) -> InvalidCameraPreparedViewQuarantinePlan:
    plan = plan_invalid_camera_prepared_view_quarantine(source_root, audit_root)
    if plan.plan_hash != str(expected_plan_hash).strip():
        raise CameraSemanticPreparedViewError(
            "Invalid prepared-view quarantine expected-plan-hash mismatch: "
            f"expected={expected_plan_hash} actual={plan.plan_hash}"
        )
    source = Path(plan.source_root)
    audit = Path(plan.audit_root)
    destination = Path(plan.destination_root)
    if destination.exists() or destination.is_symlink():
        raise CameraSemanticPreparedViewError(
            f"Quarantine destination already exists: {destination}"
        )
    existing_audit_parent = audit
    while not existing_audit_parent.exists():
        if existing_audit_parent.parent == existing_audit_parent:
            raise CameraSemanticPreparedViewError(
                f"Cannot resolve quarantine audit filesystem: {audit}"
            )
        existing_audit_parent = existing_audit_parent.parent
    if source.stat().st_dev != existing_audit_parent.stat().st_dev:
        raise CameraSemanticPreparedViewError(
            "Quarantine requires source and audit root on the same filesystem."
        )
    audit.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)
    _fsync_directory(destination.parent)
    _fsync_directory(source.parent)
    return plan


def _validate_all_episode_videos(
    root: Path,
    episodes: tuple[Mapping[str, Any], ...],
) -> tuple[int, int]:
    video_count = 0
    decoded_frames = 0
    for record in episodes:
        episode_index = int(record["episode_index"])
        parquet = _safe_relative_path(
            root,
            record.get("parquet_path"),
            label=f"episode {episode_index} parquet_path",
        )
        raw_videos = record.get("video_paths")
        if not isinstance(raw_videos, Mapping):
            continue
        for raw_role, raw_path in sorted(raw_videos.items()):
            if not raw_path:
                continue
            role = str(raw_role)
            relative = _safe_relative_path(
                root,
                raw_path,
                label=f"episode {episode_index} {role} video",
            )
            declared = _declared_video_frame_count(
                _index_values(root / parquet, role),
                label=f"episode={episode_index} role={role}",
            )
            decoded = _video_probe_frame_count(root / relative)
            if decoded != declared:
                raise CameraSemanticPreparedViewError(
                    "Declared/decoded video frame count mismatch: "
                    f"episode={episode_index} role={role} "
                    f"declared={declared} decoded={decoded}"
                )
            video_count += 1
            decoded_frames += decoded
    return video_count, decoded_frames


def _video_integrity_issues(
    root: Path,
    episodes: tuple[Mapping[str, Any], ...],
) -> tuple[CameraSemanticVideoIntegrityIssueV1, ...]:
    """Audit every source video and return deterministic episode exclusions.

    This is a read-only admission scan.  A failure quarantines the complete
    episode from the camera-only view; it never authorizes a partial multi-view
    episode and never changes the source dataset.
    """

    if shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None:
        raise CameraSemanticPreparedViewError(
            "Safe video-integrity audit requires ffprobe and ffmpeg."
        )
    _parquet_module()
    issues: list[CameraSemanticVideoIntegrityIssueV1] = []
    for record in episodes:
        episode_index = int(record["episode_index"])
        try:
            parquet = _safe_relative_path(
                root,
                record.get("parquet_path"),
                label=f"episode {episode_index} parquet_path",
            )
        except CameraSemanticMaterializationError as exc:
            issues.append(
                CameraSemanticVideoIntegrityIssueV1(
                    episode_index=episode_index,
                    camera_key="*",
                    relative_path="",
                    reason="video_frame_index_contract_invalid",
                    detail=str(exc),
                )
            )
            continue
        raw_videos = record.get("video_paths")
        if not isinstance(raw_videos, Mapping) or not raw_videos:
            issues.append(
                CameraSemanticVideoIntegrityIssueV1(
                    episode_index=episode_index,
                    camera_key="*",
                    relative_path="",
                    reason="video_paths_missing",
                    detail="Episode video_paths is missing or empty.",
                )
            )
            continue
        for raw_role, raw_path in sorted(raw_videos.items()):
            role = str(raw_role).strip()
            relative_text = str(raw_path or "").strip()
            try:
                relative = _safe_relative_path(
                    root,
                    raw_path,
                    label=f"episode {episode_index} {role} video",
                )
            except CameraSemanticMaterializationError as exc:
                issues.append(
                    CameraSemanticVideoIntegrityIssueV1(
                        episode_index=episode_index,
                        camera_key=role or "*",
                        relative_path="",
                        reason="video_path_invalid",
                        detail=(
                            f"Unsafe or invalid video path {relative_text!r}: {exc}"
                        ),
                    )
                )
                continue
            path = root / relative
            if path.is_symlink() or not path.is_file():
                issues.append(
                    CameraSemanticVideoIntegrityIssueV1(
                        episode_index=episode_index,
                        camera_key=role or "*",
                        relative_path=relative.as_posix(),
                        reason="video_file_missing_or_unsafe",
                        detail="Video payload is missing, not a regular file, or a symlink.",
                    )
                )
                continue
            try:
                declared = _declared_video_frame_count(
                    _index_values(root / parquet, role),
                    label=f"episode={episode_index} role={role}",
                )
            except CameraSemanticMaterializationError as exc:
                issues.append(
                    CameraSemanticVideoIntegrityIssueV1(
                        episode_index=episode_index,
                        camera_key=role or "*",
                        relative_path=relative.as_posix(),
                        reason="video_frame_index_contract_invalid",
                        detail=str(exc),
                    )
                )
                continue
            try:
                decoded = _video_probe_frame_count(path)
            except CameraSemanticMaterializationError as exc:
                issues.append(
                    CameraSemanticVideoIntegrityIssueV1(
                        episode_index=episode_index,
                        camera_key=role or "*",
                        relative_path=relative.as_posix(),
                        reason="video_probe_or_decode_failed",
                        declared_frame_count=declared,
                        detail=str(exc),
                    )
                )
                continue
            if decoded != declared:
                issues.append(
                    CameraSemanticVideoIntegrityIssueV1(
                        episode_index=episode_index,
                        camera_key=role or "*",
                        relative_path=relative.as_posix(),
                        reason="declared_decoded_frame_count_mismatch",
                        declared_frame_count=declared,
                        decoded_frame_count=decoded,
                        detail=(
                            "Parquet video-frame indices and the fully decoded "
                            "payload report different frame counts."
                        ),
                    )
                )
    return tuple(
        sorted(
            issues,
            key=lambda item: (
                item.episode_index,
                item.camera_key,
                item.relative_path,
                item.reason,
            ),
        )
    )


@dataclass(frozen=True)
class CameraSemanticPreparedViewPlan:
    source_root: str
    output_root: str
    registry_path: str
    source_dataset_name: str
    source_dataset_identity_hash: str
    binding_hash: str
    selected_source_inventory_hash: str
    selected_file_count: int
    selected_byte_count: int
    selected_hardlinked_file_count: int
    source_episode_count: int
    episode_count: int
    video_count: int
    frame_count: int
    status_counts: Mapping[str, int]
    remapped_episode_count: int
    remapped_mismatched_row_count: int
    tactile_reference_episode_count: int
    excluded_episode_count: int
    excluded_episode_indices: tuple[int, ...]
    video_integrity_issues: tuple[CameraSemanticVideoIntegrityIssueV1, ...]
    video_integrity_exclusions_hash: str | None

    @property
    def plan_hash(self) -> str:
        return canonical_content_hash(
            self.to_dict(include_hash=False),
            domain="dexteleop.camera_semantic_prepared_view_plan.v1",
        )

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "schema": "dexteleop.camera_semantic_prepared_view_plan.v1",
                "modalities": ["numeric", "video"],
                "excluded_modalities": ["tactile"],
                "tactile_payloads_inspected": False,
                "full_v23_dataset": False,
                "requires_explicit_consumer_opt_in": True,
                "source_mutated": False,
                "output_uses_fresh_inodes": True,
                "frame_gap_policy": "preserve_on_physical_camera",
            }
        )
        if include_hash:
            payload["plan_hash"] = self.plan_hash
        return payload


def plan_camera_semantic_prepared_view(
    source_root: str | Path,
    output_root: str | Path,
    registry_path: str | Path,
) -> CameraSemanticPreparedViewPlan:
    source = Path(source_root).expanduser().resolve(strict=True)
    output = Path(output_root).expanduser().resolve()
    registry = Path(registry_path).expanduser().resolve(strict=True)
    if output == source or source in output.parents or output in source.parents:
        raise CameraSemanticPreparedViewError(
            "Source and prepared-view trees must be disjoint."
        )
    binding = load_camera_role_binding(registry)
    resolver = DatasetCameraSemanticResolver(source, binding)
    episodes = _episode_records(source)
    files, inventory_hash, byte_count, hardlinked_count = _selected_inventory(
        source, episodes
    )
    source_identity = _source_dataset_identity(source)
    video_integrity_issues = _video_integrity_issues(source, episodes)
    excluded_episode_indices = tuple(
        sorted({item.episode_index for item in video_integrity_issues})
    )
    excluded_episode_set = set(excluded_episode_indices)
    selected_episodes = tuple(
        record
        for record in episodes
        if int(record["episode_index"]) not in excluded_episode_set
    )
    included_episode_indices = tuple(
        sorted(int(record["episode_index"]) for record in selected_episodes)
    )
    exclusions = (
        None
        if not video_integrity_issues
        else CameraSemanticVideoIntegrityExclusionsV1(
            source_dataset_name=source.name,
            source_dataset_identity_hash=source_identity,
            selected_source_inventory_hash=inventory_hash,
            source_episode_count=len(episodes),
            included_episode_indices=included_episode_indices,
            excluded_episode_indices=excluded_episode_indices,
            issues=video_integrity_issues,
        )
    )
    status_counts: Counter[str] = Counter()
    frame_count = 0
    remapped_episodes = 0
    mismatch_rows = 0
    tactile_reference_episodes = 0
    video_count = 0
    for record in selected_episodes:
        episode_index = int(record["episode_index"])
        resolution = resolver.resolution_for_episode(episode_index)
        _resolution_maps(resolution)
        status_counts[resolution.status] += 1
        parquet = _safe_relative_path(
            source,
            record.get("parquet_path"),
            label=f"episode {episode_index} parquet_path",
        )
        rows, mismatches = _frame_index_mismatch_count(
            source / parquet, resolution
        )
        try:
            frame_count += int(record["frame_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CameraSemanticPreparedViewError(
                f"Episode {episode_index} has no integer frame_count."
            ) from exc
        if resolution.status != "identity":
            remapped_episodes += 1
            mismatch_rows += mismatches
        tactile_reference_episodes += int(bool(record.get("tactile_paths")))
        raw_videos = record.get("video_paths")
        if isinstance(raw_videos, Mapping):
            video_count += sum(bool(value) for value in raw_videos.values())
    return CameraSemanticPreparedViewPlan(
        source_root=str(source),
        output_root=str(output),
        registry_path=str(registry),
        source_dataset_name=source.name,
        source_dataset_identity_hash=source_identity,
        binding_hash=binding.contract_hash,
        selected_source_inventory_hash=inventory_hash,
        selected_file_count=len(files),
        selected_byte_count=byte_count,
        selected_hardlinked_file_count=hardlinked_count,
        source_episode_count=len(episodes),
        episode_count=len(selected_episodes),
        video_count=video_count,
        frame_count=frame_count,
        status_counts=dict(sorted(status_counts.items())),
        remapped_episode_count=remapped_episodes,
        remapped_mismatched_row_count=mismatch_rows,
        tactile_reference_episode_count=tactile_reference_episodes,
        excluded_episode_count=len(excluded_episode_indices),
        excluded_episode_indices=excluded_episode_indices,
        video_integrity_issues=video_integrity_issues,
        video_integrity_exclusions_hash=(
            None if exclusions is None else exclusions.receipt_hash
        ),
    )


def _strip_tactile_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_tactile_fields(item)
            for key, item in value.items()
            if "tactile" not in str(key).lower()
        }
    if isinstance(value, list):
        return [_strip_tactile_fields(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_tactile_fields(item) for item in value)
    return value


def _strip_tactile_episode_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    stripped = _strip_tactile_fields(record)
    if not isinstance(stripped, dict):  # pragma: no cover - type contract
        raise CameraSemanticPreparedViewError(
            "Prepared-view episode projection must remain an object."
        )
    return stripped


def _project_modality_validity(
    value: Mapping[str, Any],
    *,
    included_episode_indices: set[int],
) -> dict[str, Any]:
    projected = _strip_tactile_fields(value)
    if not isinstance(projected, dict):  # pragma: no cover - type contract
        raise CameraSemanticPreparedViewError(
            "Prepared modality-validity projection must remain an object."
        )
    training = projected.get("training")
    if isinstance(training, Mapping):
        training_copy = dict(training)
        raw_episodes = training_copy.get("episodes")
        if isinstance(raw_episodes, Mapping):
            training_copy["episodes"] = {
                str(raw_index): raw_value
                for raw_index, raw_value in raw_episodes.items()
                if int(raw_index) in included_episode_indices
            }
        projected["training"] = training_copy
    return projected


def _prepared_info(
    info: Mapping[str, Any],
    *,
    swap_wrist_feature_declarations: bool,
    plan: CameraSemanticPreparedViewPlan,
) -> dict[str, Any]:
    rewritten = _rewrite_info(
        info, swap_wrist_feature_declarations=swap_wrist_feature_declarations
    )
    output = _strip_tactile_fields(rewritten)
    if not isinstance(output, dict):  # pragma: no cover - type contract
        raise CameraSemanticPreparedViewError(
            "Prepared-view info projection must remain an object."
        )
    for key in ("episode_count", "num_episodes", "total_episodes"):
        if key in output:
            output[key] = plan.episode_count
    for key in ("frame_count", "num_frames", "total_frames"):
        if key in output:
            output[key] = plan.frame_count
    for key in ("num_videos", "total_videos"):
        if key in output:
            output[key] = plan.video_count
    output["dexteleop_dataset_projection"] = {
        "schema": "dexteleop.camera_semantic_prepared_view_projection.v1",
        "manifest_path": CAMERA_SEMANTIC_PREPARED_VIEW_RELATIVE_PATH,
        "modalities": ["numeric", "video"],
        "excluded_modalities": ["tactile"],
        "full_v23_dataset": False,
        "requires_explicit_consumer_opt_in": True,
        "plan_hash": plan.plan_hash,
        "source_episode_count": plan.source_episode_count,
        "included_episode_count": plan.episode_count,
        "excluded_episode_count": plan.excluded_episode_count,
    }
    if plan.excluded_episode_indices:
        output["dexteleop_dataset_projection"].update(
            {
                "excluded_episode_indices": list(plan.excluded_episode_indices),
                "video_integrity_exclusions_path": (
                    CAMERA_SEMANTIC_VIDEO_INTEGRITY_EXCLUSIONS_RELATIVE_PATH
                ),
                "video_integrity_exclusions_hash": (
                    plan.video_integrity_exclusions_hash
                ),
            }
        )
    return output


def _rewrite_prepared_parquet(
    source: Path,
    destination: Path,
    *,
    stored_to_canonical: Mapping[str, str],
) -> None:
    _rewrite_parquet(
        source,
        destination,
        stored_to_canonical=stored_to_canonical,
    )
    _, pq = _parquet_module()
    table = pq.read_table(destination)
    retained = [
        name for name in table.column_names if "tactile" not in name.lower()
    ]
    if len(retained) == len(table.column_names):
        return
    projected = table.select(retained)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".projection.tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pq.write_table(projected, temporary, compression="zstd")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        reread = pq.read_table(temporary)
        if (
            reread.num_rows != projected.num_rows
            or tuple(reread.column_names) != tuple(retained)
        ):
            raise CameraSemanticPreparedViewError(
                f"Prepared Parquet projection validation failed: {destination}"
            )
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _output_file_inventory(root: Path) -> tuple[CameraSemanticPreparedViewFileV1, ...]:
    rows: list[CameraSemanticPreparedViewFileV1] = []
    manifest_relative = Path(CAMERA_SEMANTIC_PREPARED_VIEW_RELATIVE_PATH)
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CameraSemanticPreparedViewError(
                f"Prepared view refuses output symlinks: {path}"
            )
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative == manifest_relative:
            continue
        rows.append(
            CameraSemanticPreparedViewFileV1(
                relative_path=relative.as_posix(),
                size_bytes=path.stat().st_size,
                sha256=_sha256_file(path),
            )
        )
    return tuple(rows)


def materialize_camera_semantic_prepared_view(
    source_root: str | Path,
    output_root: str | Path,
    registry_path: str | Path,
    *,
    expected_plan_hash: str,
) -> dict[str, Any]:
    plan = plan_camera_semantic_prepared_view(
        source_root, output_root, registry_path
    )
    if not str(expected_plan_hash).strip() or expected_plan_hash != plan.plan_hash:
        raise CameraSemanticPreparedViewError(
            "Apply requires the exact current --expected-plan-hash."
        )
    source = Path(plan.source_root)
    output = Path(plan.output_root)
    if output.exists():
        raise CameraSemanticPreparedViewError(
            f"Output already exists; overwrite is forbidden: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.camera-view-stage-", dir=output.parent
        )
    )
    binding = load_camera_role_binding(plan.registry_path)
    resolver = DatasetCameraSemanticResolver(source, binding)
    source_episodes = _episode_records(source)
    excluded_episode_set = set(plan.excluded_episode_indices)
    episodes = tuple(
        record
        for record in source_episodes
        if int(record["episode_index"]) not in excluded_episode_set
    )
    if len(episodes) != plan.episode_count:
        raise CameraSemanticPreparedViewError(
            "Prepared-view selected episode count differs from the frozen plan."
        )
    resolutions = {
        int(record["episode_index"]): resolver.resolution_for_episode(
            int(record["episode_index"])
        )
        for record in episodes
    }
    selected_files, _, _, _ = _selected_inventory(source, episodes)
    parquet_paths = {
        _safe_relative_path(
            source,
            record.get("parquet_path"),
            label=f"episode {int(record['episode_index'])} parquet_path",
        )
        for record in episodes
    }
    rewritten_inputs = {
        Path("meta/episodes.jsonl"),
        Path("meta/episodes_stats.jsonl"),
        Path("meta/info.json"),
        Path("meta/modality_validity.json"),
        Path("meta/devices/realsense.json"),
        Path(CAMERA_SEMANTIC_MATERIALIZATION_RELATIVE_PATH),
    }
    copied_files = 0
    copied_bytes = 0
    try:
        for relative in selected_files:
            if relative in parquet_paths or relative in rewritten_inputs:
                continue
            role = _camera_role_in_path(relative)
            target_relative = relative
            stored_to_canonical: Mapping[str, str] | None = None
            if role is not None:
                episode_index = _episode_index_in_path(relative)
                if episode_index is None or episode_index not in resolutions:
                    raise CameraSemanticPreparedViewError(
                        f"Camera artifact is not bound to an episode: {relative}"
                    )
                _, stored_to_canonical = _resolution_maps(
                    resolutions[episode_index]
                )
                target_relative = _map_camera_artifact_path(
                    relative, stored_to_canonical
                )
            target = staging / target_relative
            if target.exists():
                raise CameraSemanticPreparedViewError(
                    f"Prepared-view path collision: {target_relative}"
                )
            source_path = source / relative
            if stored_to_canonical is not None and relative.suffix.lower() == ".json":
                payload = read_json(source_path)
                if not payload:
                    raise CameraSemanticPreparedViewError(
                        f"Camera artifact JSON is invalid: {source_path}"
                    )
                _write_json_fsync(
                    target,
                    _rewrite_camera_json(payload, stored_to_canonical),
                )
                copied_bytes += target.stat().st_size
            else:
                copied_bytes += _fresh_copy(source_path, target)
            copied_files += 1

        output_episodes: list[dict[str, Any]] = []
        for record in episodes:
            episode_index = int(record["episode_index"])
            _, stored_to_canonical = _resolution_maps(resolutions[episode_index])
            parquet = _safe_relative_path(
                source,
                record.get("parquet_path"),
                label=f"episode {episode_index} parquet_path",
            )
            _rewrite_prepared_parquet(
                source / parquet,
                staging / parquet,
                stored_to_canonical=stored_to_canonical,
            )
            copied_files += 1
            copied_bytes += (staging / parquet).stat().st_size
            output_episodes.append(
                _strip_tactile_episode_fields(
                    _rewrite_episode_record(record, stored_to_canonical)
                )
            )
        output_episode_tuple = tuple(output_episodes)
        _write_jsonl_fsync(
            staging / "meta/episodes.jsonl", output_episode_tuple
        )
        source_episode_stats = source / "meta/episodes_stats.jsonl"
        if source_episode_stats.is_file():
            selected_episode_indices = {
                int(record["episode_index"]) for record in episodes
            }
            output_stats: list[dict[str, Any]] = []
            for record in read_jsonl(source_episode_stats):
                try:
                    episode_index = int(record["episode_index"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise CameraSemanticPreparedViewError(
                        "Source episodes_stats.jsonl has an invalid episode_index."
                    ) from exc
                if episode_index not in selected_episode_indices:
                    continue
                _, stored_to_canonical = _resolution_maps(
                    resolutions[episode_index]
                )
                output_stats.append(
                    _strip_tactile_episode_fields(
                        _rewrite_episode_record(record, stored_to_canonical)
                    )
                )
            _write_jsonl_fsync(
                staging / "meta/episodes_stats.jsonl", tuple(output_stats)
            )

        statuses = {resolution.status for resolution in resolutions.values()}
        info = read_json(source / "meta/info.json")
        if not info:
            raise CameraSemanticPreparedViewError("Source meta/info.json is invalid.")
        features = info.get("features")
        if statuses == {"identity", "swapped"} and isinstance(features, Mapping):
            left = features.get(f"{CAMERA_COLUMN_PREFIX}left_wrist")
            right = features.get(f"{CAMERA_COLUMN_PREFIX}right_wrist")
            if left != right:
                raise CameraSemanticPreparedViewError(
                    "Mixed camera semantics require equal wrist feature declarations."
                )
        _write_json_fsync(
            staging / "meta/info.json",
            _prepared_info(
                info,
                swap_wrist_feature_declarations=statuses == {"swapped"},
                plan=plan,
            ),
        )
        projected_validity: dict[str, Any] = {}
        modality_validity_path = source / "meta/modality_validity.json"
        if modality_validity_path.is_file():
            modality_validity = read_json(modality_validity_path)
            if not modality_validity:
                raise CameraSemanticPreparedViewError(
                    "Source meta/modality_validity.json is invalid."
                )
            projected_validity = _project_modality_validity(
                modality_validity,
                included_episode_indices=set(resolutions),
            )
        if not resolutions:
            projected_validity.setdefault(
                "schema_version", "dexteleop.modality_validity.v1"
            )
            raw_training = projected_validity.get("training")
            training = (
                dict(raw_training)
                if isinstance(raw_training, Mapping)
                else {}
            )
            training.update(
                {
                    "valid": False,
                    "reason": (
                        "All source episodes were excluded because at least "
                        "one required video failed integrity validation."
                    ),
                    "episodes": {},
                }
            )
            projected_validity["training"] = training
        if projected_validity:
            _write_json_fsync(
                staging / "meta/modality_validity.json", projected_validity
            )
        video_integrity_exclusions = None
        if plan.excluded_episode_indices:
            video_integrity_exclusions = (
                CameraSemanticVideoIntegrityExclusionsV1(
                    source_dataset_name=plan.source_dataset_name,
                    source_dataset_identity_hash=plan.source_dataset_identity_hash,
                    selected_source_inventory_hash=(
                        plan.selected_source_inventory_hash
                    ),
                    source_episode_count=plan.source_episode_count,
                    included_episode_indices=tuple(sorted(resolutions)),
                    excluded_episode_indices=plan.excluded_episode_indices,
                    issues=plan.video_integrity_issues,
                )
            )
            if (
                video_integrity_exclusions.receipt_hash
                != plan.video_integrity_exclusions_hash
            ):
                raise CameraSemanticPreparedViewError(
                    "Video-integrity exclusions differ from the frozen plan."
                )
            _write_json_fsync(
                staging
                / CAMERA_SEMANTIC_VIDEO_INTEGRITY_EXCLUSIONS_RELATIVE_PATH,
                video_integrity_exclusions.to_dict(),
            )
        output_snapshot = _camera_snapshot_from_binding(binding)
        _write_json_fsync(
            staging / "meta/devices/realsense.json", output_snapshot
        )
        snapshot_resolution = resolve_camera_snapshot(
            output_snapshot, binding, source_id=output.name
        )
        camera_receipt: dict[str, Any] = {
            "schema": CAMERA_SEMANTIC_MATERIALIZATION_SCHEMA_V1,
            "complete": True,
            "source_root": str(source),
            "source_inventory_hash": plan.selected_source_inventory_hash,
            "plan_hash": plan.plan_hash,
            "binding_hash": binding.contract_hash,
            "output_snapshot_hash": snapshot_resolution.source_device_snapshot_hash,
            "episode_indices": sorted(resolutions),
            "source_resolutions": [
                {
                    "episode_index": index,
                    "resolution_hash": resolutions[index].semantic_content_hash,
                    "status": resolutions[index].status,
                    "canonical_to_stored_key": resolutions[index].canonical_to_stored_key,
                }
                for index in sorted(resolutions)
            ],
            "frame_gap_policy": "preserve_on_physical_camera",
            "source_mutated": False,
            "output_uses_fresh_inodes": True,
            "prepared_view_only": True,
        }
        camera_receipt["receipt_hash"] = canonical_content_hash(
            camera_receipt,
            domain=CAMERA_SEMANTIC_MATERIALIZATION_HASH_DOMAIN_V1,
        )
        _write_json_fsync(
            staging / CAMERA_SEMANTIC_MATERIALIZATION_RELATIVE_PATH,
            camera_receipt,
        )

        output_resolver = DatasetCameraSemanticResolver(staging, binding)
        for episode_index in sorted(resolutions):
            if output_resolver.resolution_for_episode(episode_index).status != "identity":
                raise CameraSemanticPreparedViewError(
                    f"Prepared-view episode {episode_index} is not canonical."
                )
        video_count, decoded_frames = _validate_all_episode_videos(
            staging, output_episode_tuple
        )
        files = _output_file_inventory(staging)
        manifest = CameraSemanticPreparedViewManifestV1(
            source_dataset_name=plan.source_dataset_name,
            source_dataset_identity_hash=plan.source_dataset_identity_hash,
            selected_source_inventory_hash=plan.selected_source_inventory_hash,
            camera_binding_hash=plan.binding_hash,
            camera_materialization_receipt_hash=camera_receipt["receipt_hash"],
            plan_hash=plan.plan_hash,
            episode_indices=tuple(sorted(resolutions)),
            files=files,
            excluded_episode_indices=plan.excluded_episode_indices,
            video_integrity_exclusions_hash=(
                None
                if video_integrity_exclusions is None
                else video_integrity_exclusions.receipt_hash
            ),
        )
        _write_json_fsync(
            staging / CAMERA_SEMANTIC_PREPARED_VIEW_RELATIVE_PATH,
            manifest.to_dict(),
        )
        load_camera_semantic_prepared_view(
            staging, verify_payload_hashes=True
        )
        _, final_inventory_hash, _, _ = _selected_inventory(
            source, source_episodes
        )
        if final_inventory_hash != plan.selected_source_inventory_hash:
            raise CameraSemanticPreparedViewError(
                "Selected source files changed during preparation; output was not published."
            )
        os.replace(staging, output)
        _fsync_directory(output.parent)
        return {
            **plan.to_dict(),
            "schema": "dexteleop.camera_semantic_prepared_view_result.v1",
            "applied": True,
            "output_root": str(output),
            "copied_file_count": copied_files,
            "copied_byte_count": copied_bytes,
            "validated_video_count": video_count,
            "validated_decoded_video_frame_count": decoded_frames,
            "manifest_path": str(
                output / CAMERA_SEMANTIC_PREPARED_VIEW_RELATIVE_PATH
            ),
            "manifest_hash": manifest.manifest_hash,
        }
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "CameraSemanticPreparedViewError",
    "CameraSemanticPreparedViewPlan",
    "materialize_camera_semantic_prepared_view",
    "plan_camera_semantic_prepared_view",
]

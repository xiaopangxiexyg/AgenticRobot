"""Atomic promotion and rollback for complete camera-semantic datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import fcntl
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.v23 import DEXTELEOP_DATASET_SCHEMA_V23
from dataset_interface.camera_semantics import (
    CAMERA_SEMANTIC_MATERIALIZATION_RELATIVE_PATH,
    DatasetCameraSemanticResolver,
    load_camera_role_binding,
)
from dataset_interface.modality_artifacts import (
    DETACHED_VERIFIED,
    MODALITY_ARTIFACT_SIDECAR_RELATIVE,
    SELF_CONTAINED,
)
from dataset_interface.validation import validate_v23_core_dataset, validate_v23_dataset

from .camera_semantic_materialize import (
    _column_mapping,
    _episode_records,
    _fsync_directory,
    _parquet_module,
    _resolution_maps,
    _rewrite_episode_record,
    _rewrite_info,
    _rewrite_role_text,
    _safe_relative_path,
    _source_inventory,
    _validate_dataset_videos,
    _write_json_fsync,
)
from .io import read_json
from .modality_storage import load_modality_inventory, modality_storage_status


PROMOTION_PLAN_SCHEMA = "dexteleop.camera_semantic_promotion_plan.v2"
PROMOTION_PLAN_DOMAIN = PROMOTION_PLAN_SCHEMA
PROMOTION_TRANSACTION_SCHEMA = "dexteleop.camera_semantic_promotion_transaction.v1"
PROMOTION_TRANSACTION_DOMAIN = PROMOTION_TRANSACTION_SCHEMA
ROLLBACK_PLAN_SCHEMA = "dexteleop.camera_semantic_promotion_rollback_plan.v1"
ROLLBACK_PLAN_DOMAIN = ROLLBACK_PLAN_SCHEMA
NATIVE_V23_VALIDATION_MODE = "native_v23"
LEGACY_FORMAT_PRESERVING_VALIDATION_MODE = "legacy_format_preserving"
LEGACY_FORMAT_PRESERVING_DOMAIN = (
    "dexteleop.camera_semantic_legacy_format_preserving.v1"
)


class CameraSemanticPromotionError(RuntimeError):
    """Raised when a corrected dataset cannot be promoted or rolled back safely."""


def _root(path: str | Path, *, label: str) -> Path:
    lexical = Path(path).expanduser()
    if lexical.is_symlink():
        raise CameraSemanticPromotionError(f"{label} must not be a symlink: {lexical}")
    try:
        resolved = lexical.resolve(strict=True)
    except FileNotFoundError as exc:
        raise CameraSemanticPromotionError(f"{label} is missing: {lexical}") from exc
    if not resolved.is_dir():
        raise CameraSemanticPromotionError(f"{label} is not a directory: {resolved}")
    return resolved


def _future_root(path: str | Path, *, label: str) -> Path:
    lexical = Path(path).expanduser()
    if lexical.is_symlink():
        raise CameraSemanticPromotionError(f"{label} must not be a symlink: {lexical}")
    return lexical.resolve(strict=False)


def _existing_ancestor(path: Path) -> Path:
    cursor = path
    while not cursor.exists():
        if cursor.parent == cursor:
            raise CameraSemanticPromotionError(
                f"Cannot resolve an existing filesystem ancestor for {path}"
            )
        cursor = cursor.parent
    if cursor.is_symlink() or not cursor.is_dir():
        raise CameraSemanticPromotionError(
            f"Promotion ancestor is missing or unsafe: {cursor}"
        )
    return cursor.resolve()


def _file_inodes(root: Path) -> set[tuple[int, int]]:
    output: set[tuple[int, int]] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise CameraSemanticPromotionError(
                f"Promotion refuses dataset symlinks: {path}"
            )
        if path.is_file():
            metadata = path.stat()
            output.add((int(metadata.st_dev), int(metadata.st_ino)))
    return output


def _episode_indices(root: Path) -> tuple[int, ...]:
    return tuple(int(row["episode_index"]) for row in _episode_records(root))


def _materialization_receipt(corrected: Path) -> Mapping[str, Any]:
    path = corrected / CAMERA_SEMANTIC_MATERIALIZATION_RELATIVE_PATH
    if path.is_symlink() or not path.is_file():
        raise CameraSemanticPromotionError(
            f"Corrected dataset has no safe materialization receipt: {path}"
        )
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise CameraSemanticPromotionError("Materialization receipt must be an object")
    return value


def _require_v23(
    root: Path,
    *,
    label: str,
    tactile_status: str,
    tactile_inventory_hash: str,
    tactile_detachment_scope: str,
) -> tuple[int, int, int]:
    if tactile_status == DETACHED_VERIFIED:
        inventory = load_modality_inventory(root, tactile_inventory_hash)
        report = validate_v23_core_dataset(
            root,
            verified_inventories=(inventory,),
            verified_detachment_scopes={
                inventory.modality_key: tactile_detachment_scope,
            },
        )
    else:
        report = validate_v23_dataset(root)
    if not report.ok:
        raise CameraSemanticPromotionError(
            f"{label} strict v2.3 validation failed: {report.errors[:5]}"
        )
    episodes = _episode_records(root)
    video_count, decoded_frames = _validate_dataset_videos(root, episodes)
    return int(report.frame_count), int(video_count), int(decoded_frames)


def _legacy_info(root: Path, *, label: str) -> dict[str, Any]:
    value = read_json(root / "meta" / "info.json")
    if not isinstance(value, dict):
        raise CameraSemanticPromotionError(
            f"{label} legacy meta/info.json must be an object"
        )
    if value.get("dexteleop_dataset_schema") == DEXTELEOP_DATASET_SCHEMA_V23:
        raise CameraSemanticPromotionError(
            f"{label} declares native v2.3 and cannot use legacy promotion"
        )
    dataset_format = str(value.get("format") or "").strip()
    if not dataset_format:
        raise CameraSemanticPromotionError(
            f"{label} legacy dataset has no explicit format"
        )
    return value


def _semantic_value_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(
            _semantic_value_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _semantic_value_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    try:
        if math.isnan(left) and math.isnan(right):
            return True
    except (TypeError, ValueError):
        pass
    return bool(left == right)


def _arrow_column_semantically_equal(expected: Any, corrected: Any) -> bool:
    if expected.type != corrected.type:
        return False
    if expected.equals(corrected):
        return True
    # Arrow treats equal-position NaN values as unequal. Materialization reads
    # and rewrites Parquet, so compare decoded values while admitting only that
    # one IEEE semantic equivalence; every finite/non-numeric value stays exact.
    return _semantic_value_equal(expected.to_pylist(), corrected.to_pylist())


def _require_legacy_parquet_pair(
    source: Path,
    corrected: Path,
    source_episodes: tuple[Mapping[str, Any], ...],
    corrected_episodes: tuple[Mapping[str, Any], ...],
    source_resolver: DatasetCameraSemanticResolver,
) -> tuple[int, tuple[dict[str, Any], ...]]:
    pa, pq = _parquet_module()
    corrected_by_index = {
        int(record["episode_index"]): record for record in corrected_episodes
    }
    total_rows = 0
    parquet_contracts: list[dict[str, Any]] = []
    for source_record in source_episodes:
        episode_index = int(source_record["episode_index"])
        corrected_record = corrected_by_index.get(episode_index)
        if corrected_record is None:
            raise CameraSemanticPromotionError(
                f"Corrected legacy dataset is missing episode {episode_index}"
            )
        source_relative = _safe_relative_path(
            source,
            source_record.get("parquet_path"),
            label=f"source episode {episode_index} parquet_path",
        )
        corrected_relative = _safe_relative_path(
            corrected,
            corrected_record.get("parquet_path"),
            label=f"corrected episode {episode_index} parquet_path",
        )
        if source_relative != corrected_relative:
            raise CameraSemanticPromotionError(
                f"Legacy episode {episode_index} parquet path changed"
            )
        source_path = source / source_relative
        corrected_path = corrected / corrected_relative
        if (
            source_path.is_symlink()
            or corrected_path.is_symlink()
            or not source_path.is_file()
            or not corrected_path.is_file()
        ):
            raise CameraSemanticPromotionError(
                f"Legacy episode {episode_index} parquet is missing or unsafe"
            )
        source_table = pq.read_table(source_path)
        corrected_table = pq.read_table(corrected_path)
        _, stored_to_canonical = _resolution_maps(
            source_resolver.resolution_for_episode(episode_index)
        )
        source_names = tuple(source_table.column_names)
        expected_mapping = _column_mapping(source_names, stored_to_canonical)
        expected_names = tuple(expected_mapping[name] for name in source_names)
        if tuple(corrected_table.column_names) != expected_names:
            raise CameraSemanticPromotionError(
                f"Legacy episode {episode_index} parquet columns differ from "
                "the physical-camera namespace transform"
            )
        if source_table.num_rows != corrected_table.num_rows:
            raise CameraSemanticPromotionError(
                f"Legacy episode {episode_index} parquet row count changed"
            )
        try:
            declared_rows = int(source_record.get("frame_count"))
            corrected_declared_rows = int(corrected_record.get("frame_count"))
        except (TypeError, ValueError) as exc:
            raise CameraSemanticPromotionError(
                f"Legacy episode {episode_index} has an invalid frame_count"
            ) from exc
        if (
            declared_rows != source_table.num_rows
            or corrected_declared_rows != declared_rows
        ):
            raise CameraSemanticPromotionError(
                f"Legacy episode {episode_index} frame_count/parquet rows differ"
            )
        for source_name in source_names:
            corrected_name = expected_mapping[source_name]
            source_column = source_table.column(source_name)
            if corrected_name.endswith(".path") and (
                pa.types.is_string(source_column.type)
                or pa.types.is_large_string(source_column.type)
            ):
                expected_column = pa.array(
                    [
                        None
                        if value is None
                        else _rewrite_role_text(str(value), stored_to_canonical)
                        for value in source_column.to_pylist()
                    ],
                    type=source_column.type,
                )
            else:
                expected_column = source_column.combine_chunks()
            corrected_column = corrected_table.column(corrected_name).combine_chunks()
            if not _arrow_column_semantically_equal(
                expected_column, corrected_column
            ):
                raise CameraSemanticPromotionError(
                    f"Legacy episode {episode_index} column {corrected_name!r} "
                    "differs from the exact camera-semantic transform"
                )
        total_rows += int(source_table.num_rows)
        parquet_contracts.append(
            {
                "episode_index": episode_index,
                "row_count": int(source_table.num_rows),
                "columns": expected_names,
            }
        )
    return total_rows, tuple(parquet_contracts)


def _require_legacy_format_preserving_pair(
    source: Path,
    corrected: Path,
    *,
    source_resolver: DatasetCameraSemanticResolver,
) -> tuple[int, int, int, str, str]:
    source_info = _legacy_info(source, label="source")
    corrected_info = _legacy_info(corrected, label="corrected")
    source_format = str(source_info["format"])
    corrected_format = str(corrected_info["format"])
    if source_format != corrected_format:
        raise CameraSemanticPromotionError(
            "Legacy promotion must preserve the dataset format exactly"
        )

    source_episodes = tuple(_episode_records(source))
    corrected_episodes = tuple(_episode_records(corrected))
    statuses = {
        source_resolver.resolution_for_episode(int(record["episode_index"])).status
        for record in source_episodes
    }
    expected_info = _rewrite_info(
        source_info,
        swap_wrist_feature_declarations=statuses == {"swapped"},
    )
    if corrected_info != expected_info:
        raise CameraSemanticPromotionError(
            "Legacy corrected info.json is not the exact format-preserving transform"
        )
    expected_episodes: list[dict[str, Any]] = []
    for record in source_episodes:
        episode_index = int(record["episode_index"])
        _, stored_to_canonical = _resolution_maps(
            source_resolver.resolution_for_episode(episode_index)
        )
        expected_episodes.append(
            _rewrite_episode_record(record, stored_to_canonical)
        )
    if tuple(expected_episodes) != corrected_episodes:
        raise CameraSemanticPromotionError(
            "Legacy corrected episodes.jsonl is not the exact camera-semantic transform"
        )

    frame_count, parquet_contracts = _require_legacy_parquet_pair(
        source,
        corrected,
        source_episodes,
        corrected_episodes,
        source_resolver,
    )
    try:
        source_episode_count = int(source_info.get("episode_count"))
        corrected_episode_count = int(corrected_info.get("episode_count"))
        source_frame_count = int(source_info.get("frame_count"))
        corrected_frame_count = int(corrected_info.get("frame_count"))
    except (TypeError, ValueError) as exc:
        raise CameraSemanticPromotionError(
            "Legacy info.json has invalid episode_count/frame_count"
        ) from exc
    if (
        source_episode_count != len(source_episodes)
        or corrected_episode_count != len(corrected_episodes)
        or source_frame_count != frame_count
        or corrected_frame_count != frame_count
    ):
        raise CameraSemanticPromotionError(
            "Legacy info/episode/parquet counts are inconsistent"
        )
    source_videos, source_decoded = _validate_dataset_videos(
        source, source_episodes
    )
    corrected_videos, corrected_decoded = _validate_dataset_videos(
        corrected, corrected_episodes
    )
    if source_videos != corrected_videos or source_decoded != corrected_decoded:
        raise CameraSemanticPromotionError(
            "Legacy source/corrected decoded video coverage differs"
        )
    contract_hash = canonical_content_hash(
        {
            "format": source_format,
            "info": corrected_info,
            "episodes": corrected_episodes,
            "parquet_contracts": parquet_contracts,
        },
        domain=LEGACY_FORMAT_PRESERVING_DOMAIN,
    )
    return (
        frame_count,
        source_videos,
        source_decoded,
        source_format,
        contract_hash,
    )


def _require_modality_pair(
    source: Path,
    corrected: Path,
) -> tuple[str, str, str, str]:
    source_status = modality_storage_status(source)
    corrected_status = modality_storage_status(corrected)
    source_state = str(source_status.get("status") or "")
    corrected_state = str(corrected_status.get("status") or "")
    if source_state not in {SELF_CONTAINED, DETACHED_VERIFIED}:
        raise CameraSemanticPromotionError(
            f"source tactile must be SELF_CONTAINED or DETACHED_VERIFIED: {source_status}"
        )
    if corrected_state != SELF_CONTAINED:
        raise CameraSemanticPromotionError(
            f"corrected tactile must be SELF_CONTAINED: {corrected_status}"
        )
    for label, status in (("source", source_status), ("corrected", corrected_status)):
        if bool(status.get("pending_transaction")):
            raise CameraSemanticPromotionError(
                f"{label} tactile has a pending storage transaction"
            )
    if source_state == DETACHED_VERIFIED and not bool(
        source_status.get("production_detached_verified")
    ):
        raise CameraSemanticPromotionError(
            "source tactile detach lacks production retirement evidence"
        )
    if not bool(corrected_status.get("tactile_read_eligible")):
        raise CameraSemanticPromotionError(
            "corrected tactile is not reader eligible"
        )
    if source_state == SELF_CONTAINED and not bool(
        source_status.get("tactile_read_eligible")
    ):
        raise CameraSemanticPromotionError(
            "self-contained source tactile is not reader eligible"
        )
    source_scope = str(source_status.get("detachment_scope") or "")
    source_inventory = str(source_status.get("inventory_hash") or "")
    corrected_inventory = str(corrected_status.get("inventory_hash") or "")
    if not source_inventory or source_inventory != corrected_inventory:
        raise CameraSemanticPromotionError(
            "Source and corrected datasets do not bind the same tactile inventory"
        )
    return source_inventory, source_scope, source_state, corrected_state


def _rollback_guard_inventory(root: Path) -> str:
    """Hash immutable dataset structure while ignoring tactile lifecycle state.

    Rehydrate/detach cycles intentionally add storage receipts and toggle the
    tactile tree.  Those changes are admitted separately through the exact
    tactile inventory and availability contract.  All other dataset files must
    retain their original inode/size/mtime identity before an atomic rollback.
    """

    entries: list[dict[str, Any]] = []
    sidecar = MODALITY_ARTIFACT_SIDECAR_RELATIVE.parts
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CameraSemanticPromotionError(
                f"Promotion rollback guard refuses dataset symlinks: {path}"
            )
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == "tactile":
            continue
        if relative.parts[: len(sidecar)] == sidecar:
            suffix = relative.parts[len(sidecar) :]
            if suffix and suffix[0] in {"receipts", "active_transaction.json"}:
                continue
        metadata = path.stat()
        entries.append(
            {
                "path": relative.as_posix(),
                "size": int(metadata.st_size),
                "mtime_ns": int(metadata.st_mtime_ns),
                "device": int(metadata.st_dev),
                "inode": int(metadata.st_ino),
                "nlink": int(metadata.st_nlink),
            }
        )
    return canonical_content_hash(
        entries,
        domain="dexteleop.camera_semantic_promotion_rollback_guard.v1",
    )


@dataclass(frozen=True)
class CameraSemanticPromotionPlan:
    source_root: str
    corrected_root: str
    retirement_root: str
    retired_dataset_root: str
    registry_path: str
    binding_hash: str
    materialization_receipt_hash: str
    materialization_plan_hash: str
    tactile_inventory_hash: str
    tactile_detachment_scope: str
    source_tactile_status: str
    corrected_tactile_status: str
    dataset_validation_mode: str
    source_dataset_format: str
    corrected_dataset_format: str
    legacy_format_preserving_contract_hash: str
    source_inventory_hash: str
    corrected_inventory_hash: str
    source_rollback_guard_hash: str
    corrected_rollback_guard_hash: str
    source_root_device: int
    source_root_inode: int
    corrected_root_device: int
    corrected_root_inode: int
    source_file_count: int
    corrected_file_count: int
    source_hardlinked_file_count: int
    corrected_hardlinked_file_count: int
    shared_file_inode_count: int
    episode_count: int
    frame_count: int
    source_video_count: int
    corrected_video_count: int
    source_decoded_video_frames: int
    corrected_decoded_video_frames: int

    @property
    def plan_hash(self) -> str:
        return canonical_content_hash(
            self.to_dict(include_hash=False), domain=PROMOTION_PLAN_DOMAIN
        )

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {"schema": PROMOTION_PLAN_SCHEMA, **asdict(self)}
        if include_hash:
            payload["plan_hash"] = self.plan_hash
        return payload


def plan_camera_semantic_promotion(
    source_root: str | Path,
    corrected_root: str | Path,
    retirement_root: str | Path,
    registry_path: str | Path,
    *,
    allow_legacy_format_preserving: bool = False,
) -> CameraSemanticPromotionPlan:
    source = _root(source_root, label="source_root")
    corrected = _root(corrected_root, label="corrected_root")
    retirement = _future_root(retirement_root, label="retirement_root")
    registry = Path(registry_path).expanduser().resolve(strict=True)
    if registry.is_symlink() or not registry.is_file():
        raise CameraSemanticPromotionError(f"Registry is missing or unsafe: {registry}")
    if source == corrected or source in corrected.parents or corrected in source.parents:
        raise CameraSemanticPromotionError("Source and corrected roots must be disjoint")
    if (
        retirement == source
        or retirement == corrected
        or source in retirement.parents
        or corrected in retirement.parents
        or retirement in source.parents
        or retirement in corrected.parents
    ):
        raise CameraSemanticPromotionError("Retirement root must be disjoint")

    binding = load_camera_role_binding(registry)
    receipt = _materialization_receipt(corrected)
    # Resolver construction validates receipt schema/hash/completeness and the
    # corrected output snapshot before any promotion can be planned.
    corrected_resolver = DatasetCameraSemanticResolver(corrected, binding)
    source_resolver = DatasetCameraSemanticResolver(source, binding)
    source_indices = _episode_indices(source)
    corrected_indices = _episode_indices(corrected)
    if source_indices != corrected_indices:
        raise CameraSemanticPromotionError("Source/corrected episode coverage differs")
    declared_indices = tuple(sorted(int(value) for value in receipt.get("episode_indices", ())))
    if source_indices != declared_indices:
        raise CameraSemanticPromotionError("Materialization receipt episode coverage differs")
    if Path(str(receipt.get("source_root") or "")).expanduser().resolve() != source:
        raise CameraSemanticPromotionError("Materialization receipt points to another source")

    declared_resolutions = {
        int(item["episode_index"]): item
        for item in receipt.get("source_resolutions", ())
    }
    for episode_index in source_indices:
        source_resolution = source_resolver.resolution_for_episode(episode_index)
        corrected_resolution = corrected_resolver.resolution_for_episode(episode_index)
        if source_resolution.status != "swapped":
            raise CameraSemanticPromotionError(
                f"Source episode {episode_index} is not proven swapped"
            )
        if corrected_resolution.status != "identity":
            raise CameraSemanticPromotionError(
                f"Corrected episode {episode_index} is not canonical identity"
            )
        declared = declared_resolutions.get(episode_index)
        if (
            not isinstance(declared, Mapping)
            or str(declared.get("resolution_hash") or "")
            != source_resolution.semantic_content_hash
        ):
            raise CameraSemanticPromotionError(
                f"Source episode {episode_index} differs from its materialization receipt"
            )

    tactile_inventory, tactile_scope, source_tactile_status, corrected_tactile_status = (
        _require_modality_pair(source, corrected)
    )
    source_v23_error: CameraSemanticPromotionError | None = None
    corrected_v23_error: CameraSemanticPromotionError | None = None
    try:
        source_frames, source_videos, source_decoded = _require_v23(
            source,
            label="source",
            tactile_status=source_tactile_status,
            tactile_inventory_hash=tactile_inventory,
            tactile_detachment_scope=tactile_scope,
        )
    except CameraSemanticPromotionError as exc:
        source_v23_error = exc
    try:
        corrected_frames, corrected_videos, corrected_decoded = _require_v23(
            corrected,
            label="corrected",
            tactile_status=corrected_tactile_status,
            tactile_inventory_hash=tactile_inventory,
            tactile_detachment_scope=tactile_scope,
        )
    except CameraSemanticPromotionError as exc:
        corrected_v23_error = exc

    if source_v23_error is None and corrected_v23_error is None:
        dataset_validation_mode = NATIVE_V23_VALIDATION_MODE
        source_dataset_format = DEXTELEOP_DATASET_SCHEMA_V23
        corrected_dataset_format = DEXTELEOP_DATASET_SCHEMA_V23
        legacy_contract_hash = ""
    else:
        if not allow_legacy_format_preserving:
            validation_error = source_v23_error or corrected_v23_error
            if validation_error is None:  # Defensive: both-success is handled above.
                validation_error = CameraSemanticPromotionError(
                    "strict v2.3 validation failed"
                )
            raise validation_error
        if (source_v23_error is None) != (corrected_v23_error is None):
            raise CameraSemanticPromotionError(
                "Legacy promotion refuses mixed native-v2.3 and legacy roots"
            )
        (
            source_frames,
            source_videos,
            source_decoded,
            source_dataset_format,
            legacy_contract_hash,
        ) = _require_legacy_format_preserving_pair(
            source,
            corrected,
            source_resolver=source_resolver,
        )
        corrected_frames = source_frames
        corrected_videos = source_videos
        corrected_decoded = source_decoded
        corrected_dataset_format = source_dataset_format
        dataset_validation_mode = LEGACY_FORMAT_PRESERVING_VALIDATION_MODE
    if source_frames != corrected_frames:
        raise CameraSemanticPromotionError("Source/corrected frame counts differ")
    source_files, source_inventory, _, source_hardlinks = _source_inventory(source)
    corrected_files, corrected_inventory, _, corrected_hardlinks = _source_inventory(
        corrected
    )
    shared_inodes = _file_inodes(source) & _file_inodes(corrected)
    if source_hardlinks or corrected_hardlinks or shared_inodes:
        raise CameraSemanticPromotionError(
            "Promotion requires independent single-link trees: "
            f"source_hardlinks={source_hardlinks} "
            f"corrected_hardlinks={corrected_hardlinks} "
            f"shared_inodes={len(shared_inodes)}"
        )

    source_stat = source.stat()
    corrected_stat = corrected.stat()
    ancestor = _existing_ancestor(retirement)
    if len({int(source_stat.st_dev), int(corrected_stat.st_dev), int(ancestor.stat().st_dev)}) != 1:
        raise CameraSemanticPromotionError(
            "Promotion requires source, corrected, and retirement on one filesystem"
        )
    receipt_hash = str(receipt.get("receipt_hash") or "")
    retired = retirement / f"{source.name}.swapped-{receipt_hash[:12]}"
    if retired.exists() or retired.is_symlink():
        raise CameraSemanticPromotionError(
            f"Retirement destination already exists: {retired}"
        )
    return CameraSemanticPromotionPlan(
        source_root=str(source),
        corrected_root=str(corrected),
        retirement_root=str(retirement),
        retired_dataset_root=str(retired),
        registry_path=str(registry),
        binding_hash=binding.contract_hash,
        materialization_receipt_hash=receipt_hash,
        materialization_plan_hash=str(receipt.get("plan_hash") or ""),
        tactile_inventory_hash=tactile_inventory,
        tactile_detachment_scope=tactile_scope,
        source_tactile_status=source_tactile_status,
        corrected_tactile_status=corrected_tactile_status,
        dataset_validation_mode=dataset_validation_mode,
        source_dataset_format=source_dataset_format,
        corrected_dataset_format=corrected_dataset_format,
        legacy_format_preserving_contract_hash=legacy_contract_hash,
        source_inventory_hash=source_inventory,
        corrected_inventory_hash=corrected_inventory,
        source_rollback_guard_hash=_rollback_guard_inventory(source),
        corrected_rollback_guard_hash=_rollback_guard_inventory(corrected),
        source_root_device=int(source_stat.st_dev),
        source_root_inode=int(source_stat.st_ino),
        corrected_root_device=int(corrected_stat.st_dev),
        corrected_root_inode=int(corrected_stat.st_ino),
        source_file_count=len(source_files),
        corrected_file_count=len(corrected_files),
        source_hardlinked_file_count=source_hardlinks,
        corrected_hardlinked_file_count=corrected_hardlinks,
        shared_file_inode_count=len(shared_inodes),
        episode_count=len(source_indices),
        frame_count=source_frames,
        source_video_count=source_videos,
        corrected_video_count=corrected_videos,
        source_decoded_video_frames=source_decoded,
        corrected_decoded_video_frames=corrected_decoded,
    )


def _transaction_payload(plan: CameraSemanticPromotionPlan, *, state: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": PROMOTION_TRANSACTION_SCHEMA,
        "state": state,
        "plan": plan.to_dict(),
    }
    payload["transaction_hash"] = canonical_content_hash(
        payload, domain=PROMOTION_TRANSACTION_DOMAIN
    )
    return payload


def _root_inode(path: Path) -> int | None:
    try:
        return int(path.stat().st_ino) if path.is_dir() and not path.is_symlink() else None
    except FileNotFoundError:
        return None


def _rollback_failed_apply(plan: CameraSemanticPromotionPlan) -> None:
    source = Path(plan.source_root)
    corrected = Path(plan.corrected_root)
    retired = Path(plan.retired_dataset_root)
    source_inode = _root_inode(source)
    corrected_inode = _root_inode(corrected)
    retired_inode = _root_inode(retired)
    if (
        source_inode == plan.corrected_root_inode
        and corrected_inode is None
        and retired_inode == plan.source_root_inode
    ):
        os.replace(source, corrected)
        os.replace(retired, source)
    elif (
        source_inode is None
        and corrected_inode == plan.corrected_root_inode
        and retired_inode == plan.source_root_inode
    ):
        os.replace(retired, source)


def promote_camera_semantic_dataset(
    source_root: str | Path,
    corrected_root: str | Path,
    retirement_root: str | Path,
    registry_path: str | Path,
    *,
    expected_plan_hash: str,
    confirm_no_active_readers: bool,
    allow_legacy_format_preserving: bool = False,
) -> dict[str, Any]:
    plan = plan_camera_semantic_promotion(
        source_root,
        corrected_root,
        retirement_root,
        registry_path,
        allow_legacy_format_preserving=allow_legacy_format_preserving,
    )
    if plan.plan_hash != str(expected_plan_hash):
        raise CameraSemanticPromotionError("Promotion plan hash changed")
    if not confirm_no_active_readers:
        raise CameraSemanticPromotionError(
            "Promotion requires explicit confirmation of no active readers"
        )
    retirement = Path(plan.retirement_root)
    retirement.mkdir(parents=True, exist_ok=True)
    transaction_dir = retirement / ".transactions"
    transaction_dir.mkdir(parents=True, exist_ok=True)
    lock_path = retirement / ".promotion.lock"
    transaction_path = transaction_dir / f"{plan.plan_hash}.json"
    with lock_path.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CameraSemanticPromotionError("Another promotion is active") from exc
        fresh = plan_camera_semantic_promotion(
            source_root,
            corrected_root,
            retirement_root,
            registry_path,
            allow_legacy_format_preserving=allow_legacy_format_preserving,
        )
        if fresh.plan_hash != plan.plan_hash:
            raise CameraSemanticPromotionError("Promotion plan changed after locking")
        if transaction_path.exists():
            raise CameraSemanticPromotionError(
                f"Promotion transaction already exists: {transaction_path}"
            )
        _write_json_fsync(transaction_path, _transaction_payload(plan, state="prepared"))
        source = Path(plan.source_root)
        corrected = Path(plan.corrected_root)
        retired = Path(plan.retired_dataset_root)
        try:
            os.replace(source, retired)
            _fsync_directory(source.parent)
            _fsync_directory(retired.parent)
            _write_json_fsync(
                transaction_path, _transaction_payload(plan, state="source_retired")
            )
            os.replace(corrected, source)
            _fsync_directory(corrected.parent)
            _fsync_directory(source.parent)
            if (
                _root_inode(source) != plan.corrected_root_inode
                or _root_inode(retired) != plan.source_root_inode
            ):
                raise CameraSemanticPromotionError("Post-promotion root inode mismatch")
            completed = _transaction_payload(plan, state="completed")
            _write_json_fsync(transaction_path, completed)
            return {
                **plan.to_dict(),
                "applied": True,
                "transaction_path": str(transaction_path),
                "transaction_hash": completed["transaction_hash"],
                "rollback_available": True,
                "rollback_requires_canonical_detach": bool(
                    plan.source_tactile_status == DETACHED_VERIFIED
                ),
            }
        except BaseException:
            _rollback_failed_apply(plan)
            _fsync_directory(source.parent)
            _fsync_directory(corrected.parent)
            _fsync_directory(retired.parent)
            _write_json_fsync(
                transaction_path, _transaction_payload(plan, state="rolled_back_after_failure")
            )
            raise


@dataclass(frozen=True)
class CameraSemanticPromotionRollbackPlan:
    transaction_path: str
    transaction_hash: str
    canonical_root: str
    corrected_work_root: str
    retired_dataset_root: str
    canonical_root_inode: int
    retired_root_inode: int

    @property
    def plan_hash(self) -> str:
        return canonical_content_hash(
            self.to_dict(include_hash=False), domain=ROLLBACK_PLAN_DOMAIN
        )

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {"schema": ROLLBACK_PLAN_SCHEMA, **asdict(self)}
        if include_hash:
            payload["plan_hash"] = self.plan_hash
        return payload


def plan_camera_semantic_promotion_rollback(
    transaction_path: str | Path,
) -> CameraSemanticPromotionRollbackPlan:
    path = Path(transaction_path).expanduser().resolve(strict=True)
    value = read_json(path)
    if not isinstance(value, Mapping) or value.get("schema") != PROMOTION_TRANSACTION_SCHEMA:
        raise CameraSemanticPromotionError("Unsupported promotion transaction")
    claimed_hash = str(value.get("transaction_hash") or "")
    hash_payload = dict(value)
    hash_payload.pop("transaction_hash", None)
    if claimed_hash != canonical_content_hash(
        hash_payload, domain=PROMOTION_TRANSACTION_DOMAIN
    ):
        raise CameraSemanticPromotionError("Promotion transaction hash mismatch")
    if value.get("state") != "completed":
        raise CameraSemanticPromotionError("Only a completed promotion can be rolled back")
    plan = value.get("plan")
    if not isinstance(plan, Mapping):
        raise CameraSemanticPromotionError("Promotion transaction has no plan")
    canonical = Path(str(plan["source_root"]))
    corrected = Path(str(plan["corrected_root"]))
    retired = Path(str(plan["retired_dataset_root"]))
    if corrected.exists() or corrected.is_symlink():
        raise CameraSemanticPromotionError(
            f"Corrected work root must remain absent before rollback: {corrected}"
        )
    if _root_inode(canonical) != int(plan["corrected_root_inode"]):
        raise CameraSemanticPromotionError("Canonical root no longer matches promoted data")
    if _root_inode(retired) != int(plan["source_root_inode"]):
        raise CameraSemanticPromotionError("Retired root no longer matches original data")
    expected_canonical_guard = str(plan.get("corrected_rollback_guard_hash") or "")
    expected_retired_guard = str(plan.get("source_rollback_guard_hash") or "")
    if expected_canonical_guard:
        if _rollback_guard_inventory(canonical) != expected_canonical_guard:
            raise CameraSemanticPromotionError(
                "Canonical core files changed after promotion"
            )
        if _rollback_guard_inventory(retired) != expected_retired_guard:
            raise CameraSemanticPromotionError(
                "Retired core files changed after promotion"
            )
    else:
        _, canonical_inventory, _, canonical_hardlinks = _source_inventory(canonical)
        _, retired_inventory, _, retired_hardlinks = _source_inventory(retired)
        if canonical_hardlinks or retired_hardlinks:
            raise CameraSemanticPromotionError(
                "Legacy promotion rollback refuses new hardlinks"
            )
        if canonical_inventory != str(plan["corrected_inventory_hash"]):
            raise CameraSemanticPromotionError(
                "Canonical files changed after promotion"
            )
        if retired_inventory != str(plan["source_inventory_hash"]):
            raise CameraSemanticPromotionError(
                "Retired files changed after promotion"
            )
    source_was_detached = str(plan.get("source_tactile_status") or "") == DETACHED_VERIFIED
    if source_was_detached:
        canonical_status = modality_storage_status(canonical)
        retired_status = modality_storage_status(retired)
        for label, status in (("canonical", canonical_status), ("retired", retired_status)):
            if str(status.get("status") or "") != DETACHED_VERIFIED:
                raise CameraSemanticPromotionError(
                    f"{label} tactile must be DETACHED_VERIFIED before rollback"
                )
            if bool(status.get("pending_transaction")) or not bool(
                status.get("production_detached_verified")
            ):
                raise CameraSemanticPromotionError(
                    f"{label} tactile detach is not rollback eligible"
                )
            if str(status.get("inventory_hash") or "") != str(
                plan["tactile_inventory_hash"]
            ):
                raise CameraSemanticPromotionError(
                    f"{label} tactile inventory changed after promotion"
                )
    return CameraSemanticPromotionRollbackPlan(
        transaction_path=str(path),
        transaction_hash=claimed_hash,
        canonical_root=str(canonical),
        corrected_work_root=str(corrected),
        retired_dataset_root=str(retired),
        canonical_root_inode=int(plan["corrected_root_inode"]),
        retired_root_inode=int(plan["source_root_inode"]),
    )


def rollback_camera_semantic_promotion(
    transaction_path: str | Path,
    *,
    expected_plan_hash: str,
    confirm_no_active_readers: bool,
) -> dict[str, Any]:
    plan = plan_camera_semantic_promotion_rollback(transaction_path)
    if plan.plan_hash != str(expected_plan_hash):
        raise CameraSemanticPromotionError("Rollback plan hash changed")
    if not confirm_no_active_readers:
        raise CameraSemanticPromotionError(
            "Rollback requires explicit confirmation of no active readers"
        )
    canonical = Path(plan.canonical_root)
    corrected = Path(plan.corrected_work_root)
    retired = Path(plan.retired_dataset_root)
    transaction = Path(plan.transaction_path)
    lock_path = transaction.parent.parent / ".promotion.lock"
    with lock_path.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CameraSemanticPromotionError("Another promotion is active") from exc
        fresh = plan_camera_semantic_promotion_rollback(transaction)
        if fresh.plan_hash != plan.plan_hash:
            raise CameraSemanticPromotionError("Rollback plan changed after locking")
        os.replace(canonical, corrected)
        try:
            os.replace(retired, canonical)
        except BaseException:
            os.replace(corrected, canonical)
            raise
        _fsync_directory(canonical.parent)
        _fsync_directory(corrected.parent)
        _fsync_directory(retired.parent)
        original = read_json(transaction)
        rolled_back = dict(original)
        rolled_back["state"] = "rolled_back"
        rolled_back.pop("transaction_hash", None)
        rolled_back["transaction_hash"] = canonical_content_hash(
            rolled_back, domain=PROMOTION_TRANSACTION_DOMAIN
        )
        _write_json_fsync(transaction, rolled_back)
        return {
            **plan.to_dict(),
            "applied": True,
            "state": "rolled_back",
            "transaction_hash": rolled_back["transaction_hash"],
        }


__all__ = [
    "CameraSemanticPromotionError",
    "CameraSemanticPromotionPlan",
    "CameraSemanticPromotionRollbackPlan",
    "plan_camera_semantic_promotion",
    "plan_camera_semantic_promotion_rollback",
    "promote_camera_semantic_dataset",
    "rollback_camera_semantic_promotion",
]

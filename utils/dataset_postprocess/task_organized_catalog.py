"""Organize whole dataset roots and build read-only logical episode manifests.

The organizer only renames complete directories on the same filesystem and
leaves compatibility symlinks at every historical absolute path. It never
opens dataset files for writing. Logical manifests select source episodes
without copying, renumbering, or hard-linking media.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import stat
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


LOGICAL_EPISODE_SCHEMA = "dexteleop.logical_episode_ref.v1"
LOGICAL_EXCLUSION_SCHEMA = "dexteleop.logical_episode_exclusion.v1"
ORGANIZATION_RECEIPT_SCHEMA = "dexteleop.dataset_directory_organization.v1"
CATALOG_RECEIPT_SCHEMA = "dexteleop.logical_dataset_catalog.v1"

CAMERA_BY_SLOT = {
    "base_0_rgb": "front",
    "left_wrist_0_rgb": "left_wrist",
    "right_wrist_0_rgb": "right_wrist",
}


@dataclasses.dataclass(frozen=True)
class MoveSpec:
    key: str
    source_relative: str
    target_relative: str
    classification: str


MOVE_SPECS: tuple[MoveSpec, ...] = (
    MoveSpec(
        "cube_front_success",
        "sorted_v22/new_cube_front_ready",
        "task_organized_v1/sources/cube/front_only/success/active/new_cube_front_ready",
        "cube/front_only/success/active",
    ),
    MoveSpec(
        "cube_front_success_archive_before_0725",
        "sorted_v22/new_cube_front_ready_before_0725_backup_20260725",
        "task_organized_v1/sources/cube/front_only/success/archives/"
        "new_cube_front_ready_before_0725_backup_20260725",
        "cube/front_only/success/archive",
    ),
    MoveSpec(
        "cube_front_suboptimal",
        "sorted_v22/new_cube_front_ready_suboptimal",
        "task_organized_v1/sources/cube/front_only/suboptimal/active/"
        "new_cube_front_ready_suboptimal",
        "cube/front_only/suboptimal/active",
    ),
    MoveSpec(
        "cube_wrist_success",
        "sorted_v22/bimanual-cube-new-success",
        "task_organized_v1/sources/cube/wrist_only/success/active/"
        "bimanual-cube-new-success",
        "cube/wrist_only/success/active",
    ),
    MoveSpec(
        "cube_wrist_suboptimal_new",
        "sorted_v22/bimanual-cube-new-failure-case",
        "task_organized_v1/sources/cube/wrist_only/suboptimal/new/"
        "bimanual-cube-new-failure-case",
        "cube/wrist_only/suboptimal/new",
    ),
    MoveSpec(
        "cube_wrist_suboptimal_legacy",
        "sorted_v22/bimanual-cube-old-failure-case",
        "task_organized_v1/sources/cube/wrist_only/suboptimal/legacy/"
        "bimanual-cube-old-failure-case",
        "cube/wrist_only/suboptimal/legacy",
    ),
    MoveSpec(
        "cube_excluded_old_success",
        "sorted_v22/bimanual-cube-old-success",
        "task_organized_v1/sources/cube/excluded/no_valid_task_view/"
        "bimanual-cube-old-success",
        "cube/excluded/no_valid_task_view",
    ),
    MoveSpec(
        "writing_front_active",
        "writing_phase1_grap_pen",
        "task_organized_v1/sources/writing/front_only/active/"
        "writing_phase1_grap_pen",
        "writing/front_only/active",
    ),
    MoveSpec(
        "writing_front_archive_pre_merge",
        "writing_phase1_grap_pen_pre_merge_backup_20260724",
        "task_organized_v1/sources/writing/front_only/archives/"
        "writing_phase1_grap_pen_pre_merge_backup_20260724",
        "writing/front_only/archive",
    ),
    MoveSpec(
        "writing_composite_base_plus_0712",
        "writing_phase1_grap_pen_before_20260724_append_backup",
        "task_organized_v1/sources/writing/composite_sources/base_plus_0712/"
        "writing_phase1_grap_pen_before_20260724_append_backup",
        "writing/composite_sources/base_plus_0712",
    ),
    MoveSpec(
        "writing_composite_base_plus_0724",
        "writing_phase1_grap_pen_with_0724_backup_20260725",
        "task_organized_v1/sources/writing/composite_sources/base_plus_0724/"
        "writing_phase1_grap_pen_with_0724_backup_20260725",
        "writing/composite_sources/base_plus_0724",
    ),
)


@dataclasses.dataclass(frozen=True)
class SourceSelection:
    move_key: str
    include: Callable[[Mapping[str, Any]], bool]
    source_quality: str
    generation: str = "current"


@dataclasses.dataclass(frozen=True)
class ManifestSpec:
    relative_path: str
    task: str
    view_profile: str
    quality_class: str
    camera_slots: tuple[str, ...]
    sources: tuple[SourceSelection, ...]
    expected_admitted_count: int


def _all(_provenance: Mapping[str, Any]) -> bool:
    return True


def _source_contains(fragment: str) -> Callable[[Mapping[str, Any]], bool]:
    return lambda provenance: fragment in str(
        provenance.get("source_dataset_root", "")
    )


MANIFEST_SPECS: tuple[ManifestSpec, ...] = (
    ManifestSpec(
        "cube/front_only/success.jsonl",
        "cube",
        "front_only",
        "success",
        ("base_0_rgb",),
        (SourceSelection("cube_front_success", _all, "success"),),
        725,
    ),
    ManifestSpec(
        "cube/front_only/suboptimal.jsonl",
        "cube",
        "front_only",
        "suboptimal",
        ("base_0_rgb",),
        (SourceSelection("cube_front_suboptimal", _all, "suboptimal"),),
        8,
    ),
    ManifestSpec(
        "cube/wrist_only/success.jsonl",
        "cube",
        "wrist_only",
        "success",
        ("left_wrist_0_rgb", "right_wrist_0_rgb"),
        (SourceSelection("cube_wrist_success", _all, "success"),),
        503,
    ),
    ManifestSpec(
        "cube/wrist_only/suboptimal.jsonl",
        "cube",
        "wrist_only",
        "suboptimal",
        ("left_wrist_0_rgb", "right_wrist_0_rgb"),
        (
            SourceSelection(
                "cube_wrist_suboptimal_new",
                _all,
                "failure_case",
            ),
            SourceSelection(
                "cube_wrist_suboptimal_legacy",
                _all,
                "failure_case",
                generation="legacy",
            ),
        ),
        17,
    ),
    ManifestSpec(
        "writing/front_only.jsonl",
        "writing",
        "front_only",
        "unlabeled",
        ("base_0_rgb",),
        (SourceSelection("writing_front_active", _all, "unlabeled"),),
        479,
    ),
    ManifestSpec(
        "writing/three_view_ready.jsonl",
        "writing",
        "three_view_ready",
        "unlabeled",
        ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        (
            SourceSelection(
                "writing_composite_base_plus_0712",
                _source_contains("2026-07-12_vive_rm65_real_teleop_v23_clean"),
                "unlabeled",
            ),
            SourceSelection(
                "writing_composite_base_plus_0724",
                _source_contains("2026-07-24_vive_rm65_real_teleop_v23"),
                "unlabeled",
            ),
        ),
        323,
    ),
)


def _json_lines(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        records.append(value)
    return records


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _inventory(root: Path) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"inventory root must be a real directory: {root}")
    digest = hashlib.sha256()
    file_count = 0
    directory_count = 1
    symlink_count = 0
    total_file_bytes = 0
    root_stat = root.stat()
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            kind = "directory"
            directory_count += 1
        elif stat.S_ISLNK(metadata.st_mode):
            kind = "symlink"
            symlink_count += 1
        else:
            kind = "file"
            file_count += 1
            total_file_bytes += int(metadata.st_size)
        row = (
            relative,
            kind,
            int(metadata.st_ino),
            int(metadata.st_size),
            int(metadata.st_mtime_ns),
            int(metadata.st_nlink),
        )
        digest.update(
            json.dumps(row, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        )
        digest.update(b"\n")
    return {
        "root_device": int(root_stat.st_dev),
        "root_inode": int(root_stat.st_ino),
        "file_count": file_count,
        "directory_count": directory_count,
        "symlink_count": symlink_count,
        "total_file_bytes": total_file_bytes,
        "stat_inventory_sha256": "sha256:" + digest.hexdigest(),
    }


def _move_map() -> dict[str, MoveSpec]:
    return {spec.key: spec for spec in MOVE_SPECS}


def _path_for(data_root: Path, spec: MoveSpec, *, target: bool) -> Path:
    relative = spec.target_relative if target else spec.source_relative
    return data_root / relative


def plan_organization(data_root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    devices: set[int] = set()
    for spec in MOVE_SPECS:
        source = _path_for(data_root, spec, target=False)
        target = _path_for(data_root, spec, target=True)
        if source.is_symlink() and target.is_dir():
            status = "already_organized"
            source_device = int(target.stat().st_dev)
        elif source.is_dir() and not source.is_symlink():
            status = "ready"
            source_device = int(source.stat().st_dev)
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"organization target already exists: {target}")
        else:
            raise FileNotFoundError(f"organization source is missing: {source}")
        devices.add(source_device)
        entries.append(
            {
                "key": spec.key,
                "classification": spec.classification,
                "source": str(source),
                "target": str(target),
                "status": status,
                "source_device": source_device,
            }
        )
    if len(devices) != 1 or devices != {int(data_root.stat().st_dev)}:
        raise RuntimeError(
            f"all moves must remain on dataset device {data_root.stat().st_dev}; "
            f"found {sorted(devices)}"
        )
    return {
        "schema": "dexteleop.dataset_directory_organization_plan.v1",
        "data_root": str(data_root),
        "device": next(iter(devices)),
        "entries": entries,
    }


def execute_organization(data_root: Path) -> dict[str, Any]:
    plan = plan_organization(data_root)
    receipt_path = data_root / "task_organized_v1/organization_receipt.json"
    if all(
        entry["status"] == "already_organized"
        for entry in plan["entries"]
    ) and receipt_path.is_file():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        if existing.get("schema") != ORGANIZATION_RECEIPT_SCHEMA:
            raise ValueError(
                f"existing organization receipt has unsupported schema: {receipt_path}"
            )
        return existing
    results: list[dict[str, Any]] = []
    for spec, planned in zip(MOVE_SPECS, plan["entries"], strict=True):
        source = _path_for(data_root, spec, target=False)
        target = _path_for(data_root, spec, target=True)
        if planned["status"] == "already_organized":
            results.append(
                {
                    **planned,
                    "inventory": _inventory(target),
                    "compatibility_symlink_target": os.readlink(source),
                }
            )
            continue
        before = _inventory(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(source, target)
        try:
            os.symlink(str(target), str(source), target_is_directory=True)
        except BaseException:
            if not source.exists() and not source.is_symlink():
                os.rename(target, source)
            raise
        after = _inventory(target)
        if before != after:
            raise RuntimeError(
                f"directory inventory changed across rename for {spec.key}: "
                f"before={before}, after={after}"
            )
        if source.resolve() != target.resolve():
            raise RuntimeError(f"compatibility symlink does not resolve to target: {source}")
        results.append(
            {
                **planned,
                "status": "organized",
                "inventory": after,
                "compatibility_symlink_target": os.readlink(source),
            }
        )
    receipt = {
        "schema": ORGANIZATION_RECEIPT_SCHEMA,
        "created_at_epoch_sec": time.time(),
        "data_root": str(data_root),
        "content_operation": "same_filesystem_directory_rename_only",
        "compatibility": "absolute_symlink_at_every_historical_path",
        "entries": results,
    }
    _write_new_or_same(
        receipt_path,
        json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    return receipt


def _load_source(root: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    episodes = _json_lines(root / "meta/episodes.jsonl")
    provenance_path = root / "meta/provenance.jsonl"
    provenance = (
        {
            int(record["output_episode_index"]): record
            for record in _json_lines(provenance_path)
        }
        if provenance_path.is_file()
        else {}
    )
    return episodes, provenance


def _episode_identity(
    record: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> str:
    origin = {
        "source_dataset_root": provenance.get("source_dataset_root"),
        "source_episode_index": provenance.get(
            "source_episode_index",
            record.get("episode_index"),
        ),
        "frame_count": record.get("frame_count"),
    }
    if not origin["source_dataset_root"]:
        origin["dataset_name"] = record.get("dataset_name")
        origin["parquet_path"] = record.get("parquet_path")
    return _canonical_hash(origin)


def _video_issues(
    root: Path,
    record: Mapping[str, Any],
    camera_keys: Sequence[str],
) -> list[dict[str, Any]]:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("manifest generation requires opencv-python") from exc
    episode_index = int(record["episode_index"])
    declared = int(record["frame_count"])
    paths = record.get("video_paths")
    paths = paths if isinstance(paths, Mapping) else {}
    issues: list[dict[str, Any]] = []
    for camera in camera_keys:
        relative = paths.get(camera)
        if not relative:
            issues.append({"camera": camera, "reason": "missing_video_path"})
            continue
        path = root / str(relative)
        if not path.is_file():
            issues.append({"camera": camera, "reason": "missing_video_file"})
            continue
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                issues.append({"camera": camera, "reason": "decoder_open_failed"})
                continue
            decoded = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
            if decoded != declared:
                issues.append(
                    {
                        "camera": camera,
                        "reason": "decoder_frame_count_mismatch",
                        "declared": declared,
                        "decoded": decoded,
                    }
                )
                continue
            if declared:
                capture.set(cv2.CAP_PROP_POS_FRAMES, declared - 1)
                ok, frame = capture.read()
                if not ok or frame is None:
                    issues.append(
                        {
                            "camera": camera,
                            "reason": "tail_frame_decode_failed",
                            "declared": declared,
                            "decoded": decoded,
                        }
                    )
        finally:
            capture.release()
    return issues


def _mask_receipts(
    data_root: Path,
    source_root: Path,
    episode_index: int,
    camera_keys: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for camera in camera_keys:
        if camera not in {"left_wrist", "right_wrist"}:
            continue
        path = (
            data_root
            / "masks/final_keep_mask"
            / source_root.name
            / f"episode_{episode_index:06d}"
            / camera
            / "manifest.json"
        )
        result[camera] = {
            "path": str(path),
            "exists": path.is_file(),
        }
    return result


def _logical_record(
    *,
    source_root: Path,
    episode: Mapping[str, Any],
    provenance: Mapping[str, Any],
    spec: ManifestSpec,
    selection: SourceSelection,
    mask_receipts: Mapping[str, Any],
) -> dict[str, Any]:
    episode_index = int(episode["episode_index"])
    augmentation_ready = all(
        bool(receipt.get("exists")) for receipt in mask_receipts.values()
    )
    return {
        "schema": LOGICAL_EPISODE_SCHEMA,
        "source_dataset_root": str(source_root),
        "source_episode_index": episode_index,
        "episode_identity": _episode_identity(episode, provenance),
        "task": spec.task,
        "view_profile": spec.view_profile,
        "quality_class": spec.quality_class,
        "source_quality": selection.source_quality,
        "generation": selection.generation,
        "camera_slots": list(spec.camera_slots),
        "frame_count": int(episode["frame_count"]),
        "media_integrity": {
            "status": "valid",
            "camera_keys": [CAMERA_BY_SLOT[slot] for slot in spec.camera_slots],
            "checks": ["declared_frame_count", "decoder_frame_count", "tail_decode"],
        },
        "final_keep_mask": dict(mask_receipts),
        "augmentation_ready": augmentation_ready,
        "source_provenance": dict(provenance),
    }


def _encode_jsonl(records: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8")
        + b"\n"
        for record in records
    )


def _write_new_or_same(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(
                f"immutable versioned output already exists with different content: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def generate_catalog(data_root: Path) -> dict[str, Any]:
    moves = _move_map()
    source_cache: dict[str, tuple[Path, list[dict[str, Any]], dict[int, dict[str, Any]]]] = {}
    for key, move in moves.items():
        root = _path_for(data_root, move, target=True)
        if not root.is_dir():
            raise FileNotFoundError(f"organized dataset source is missing: {root}")
        episodes, provenance = _load_source(root)
        source_cache[key] = (root, episodes, provenance)

    manifest_payloads: dict[str, bytes] = {}
    manifest_counts: dict[str, int] = {}
    excluded: list[dict[str, Any]] = []
    admitted_identities: dict[str, str] = {}
    for spec in MANIFEST_SPECS:
        admitted: list[dict[str, Any]] = []
        for selection in spec.sources:
            root, episodes, provenance_by_episode = source_cache[selection.move_key]
            camera_keys = tuple(CAMERA_BY_SLOT[slot] for slot in spec.camera_slots)
            for episode in episodes:
                episode_index = int(episode["episode_index"])
                provenance = provenance_by_episode.get(episode_index, {})
                if not selection.include(provenance):
                    continue
                issues = _video_issues(root, episode, camera_keys)
                identity = _episode_identity(episode, provenance)
                if issues:
                    excluded.append(
                        {
                            "schema": LOGICAL_EXCLUSION_SCHEMA,
                            "source_dataset_root": str(root),
                            "source_episode_index": episode_index,
                            "episode_identity": identity,
                            "task": spec.task,
                            "intended_view_profile": spec.view_profile,
                            "quality_class": spec.quality_class,
                            "reason": "required_camera_media_invalid",
                            "issues": issues,
                        }
                    )
                    continue
                mask_receipts = _mask_receipts(
                    data_root,
                    root,
                    episode_index,
                    camera_keys,
                )
                record = _logical_record(
                    source_root=root,
                    episode=episode,
                    provenance=provenance,
                    spec=spec,
                    selection=selection,
                    mask_receipts=mask_receipts,
                )
                previous = admitted_identities.get(identity)
                if previous is not None:
                    raise ValueError(
                        f"episode identity {identity} is admitted by both "
                        f"{previous} and {spec.relative_path}"
                    )
                admitted_identities[identity] = spec.relative_path
                admitted.append(record)
        admitted.sort(
            key=lambda item: (
                str(item["source_dataset_root"]),
                int(item["source_episode_index"]),
            )
        )
        if len(admitted) != spec.expected_admitted_count:
            raise RuntimeError(
                f"{spec.relative_path} admitted {len(admitted)} episodes, "
                f"expected {spec.expected_admitted_count}; refusing to publish"
            )
        manifest_payloads[spec.relative_path] = _encode_jsonl(admitted)
        manifest_counts[spec.relative_path] = len(admitted)

    excluded_root, excluded_episodes, excluded_provenance = source_cache[
        "cube_excluded_old_success"
    ]
    for episode in excluded_episodes:
        episode_index = int(episode["episode_index"])
        provenance = excluded_provenance.get(episode_index, {})
        excluded.append(
            {
                "schema": LOGICAL_EXCLUSION_SCHEMA,
                "source_dataset_root": str(excluded_root),
                "source_episode_index": episode_index,
                "episode_identity": _episode_identity(episode, provenance),
                "task": "cube",
                "intended_view_profile": None,
                "quality_class": "success",
                "reason": "no_valid_task_view",
                "issues": [
                    {"camera": "front", "reason": "task_semantics_invalid"},
                    {"camera": "left_wrist", "reason": "media_invalid"},
                    {"camera": "right_wrist", "reason": "media_invalid"},
                ],
            }
        )
    excluded.sort(
        key=lambda item: (
            str(item["task"]),
            str(item["source_dataset_root"]),
            int(item["source_episode_index"]),
        )
    )
    manifest_payloads["excluded.jsonl"] = _encode_jsonl(excluded)
    manifest_counts["excluded.jsonl"] = len(excluded)

    catalog_root = data_root / "task_organized_v1/catalogs/task_ready_v1"
    source_receipts: list[dict[str, Any]] = []
    for key, (root, episodes, _provenance) in sorted(source_cache.items()):
        info_path = root / "meta/info.json"
        episodes_path = root / "meta/episodes.jsonl"
        source_receipts.append(
            {
                "move_key": key,
                "dataset_root": str(root),
                "episode_count": len(episodes),
                "root_inode": int(root.stat().st_ino),
                "info_json_sha256": _sha256_bytes(info_path.read_bytes()),
                "episodes_jsonl_sha256": _sha256_bytes(episodes_path.read_bytes()),
            }
        )
    receipt = {
        "schema": CATALOG_RECEIPT_SCHEMA,
        "catalog_version": "task_ready_v1",
        "created_at_epoch_sec": time.time(),
        "manifest_counts": manifest_counts,
        "manifest_sha256": {
            relative: _sha256_bytes(payload)
            for relative, payload in sorted(manifest_payloads.items())
        },
        "source_receipts": source_receipts,
        "notes": [
            "No source Parquet, video, mask, tactile, or metadata file was modified.",
            "Writing composite roots remain whole; manifests select only their unique 0712 or 0724 provenance rows.",
            "Writing three_view_ready is media-ready but augmentation remains blocked where final_keep_mask receipts are absent.",
        ],
    }
    readme = """# DexTeleop task-ready logical catalog v1

This catalog is a read-only episode view over whole source dataset directories.
Source datasets are not copied, split, renumbered, or hard-linked. Historical
paths remain compatibility symlinks to the organized source directories.

- `cube/front_only`: only the front camera is admitted.
- `cube/wrist_only`: only the two wrist cameras are admitted.
- `writing/front_only`: the 2026-07-23 base source, filtered by front integrity.
- `writing/three_view_ready`: unique 2026-07-12 and 2026-07-24 provenance rows
  selected from intact composite roots.
- `excluded.jsonl`: explicit media or task-semantic exclusions.

Training must bind both a manifest SHA256 and the selected source receipts.
"""
    for relative, payload in sorted(manifest_payloads.items()):
        _write_new_or_same(catalog_root / relative, payload)
    _write_new_or_same(catalog_root / "README.md", readme.encode("utf-8"))
    _write_new_or_same(
        catalog_root / "catalog_receipt.json",
        json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Organize whole task dataset roots and build logical manifests."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/user/zc/DexTeleop/datasets"),
    )
    parser.add_argument(
        "--execute-organization",
        action="store_true",
        help="Atomically rename whole directories and create compatibility symlinks.",
    )
    parser.add_argument(
        "--generate-catalog",
        action="store_true",
        help="Validate selected media and publish immutable task_ready_v1 manifests.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    if args.execute_organization:
        organization = execute_organization(data_root)
    else:
        organization = plan_organization(data_root)
    output: dict[str, Any] = {"organization": organization}
    if args.generate_catalog:
        if not args.execute_organization and any(
            entry["status"] != "already_organized"
            for entry in organization["entries"]
        ):
            raise RuntimeError(
                "--generate-catalog requires already-organized sources or "
                "--execute-organization"
            )
        output["catalog"] = generate_catalog(data_root)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

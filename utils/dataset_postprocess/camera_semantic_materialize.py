"""Fresh-inode materialization of physical wrist-camera semantics.

The migration is deliberately namespace based.  For every episode it resolves
the physical camera role from the captured RealSense serial, then moves the
complete ``observation.images.<role>`` column and artifact family to that
canonical role.  Missing frames remain missing on the physical camera that
missed them; indices are never borrowed from the other wrist view.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Mapping

from dataset_contracts.camera_semantics import (
    CAMERA_SEMANTIC_MATERIALIZATION_HASH_DOMAIN_V1,
    CAMERA_SEMANTIC_MATERIALIZATION_SCHEMA_V1,
    CameraSemanticResolution,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.modality_artifacts import ModalityArtifactInventoryV1
from dataset_interface.camera_semantics import (
    CAMERA_SEMANTIC_MATERIALIZATION_RELATIVE_PATH,
    DatasetCameraSemanticResolver,
    load_camera_role_binding,
    resolve_camera_snapshot,
)
from dataset_interface.modality_artifacts import (
    PAYLOAD_ONLY_DETACH,
    WHOLE_PREFIX_DETACH,
    verified_detachment_scope,
)

from .io import read_json, read_jsonl


WRIST_ROLES = ("left_wrist", "right_wrist")
CAMERA_COLUMN_PREFIX = "observation.images."
CAMERA_ARTIFACT_MAP_KEYS = (
    "video_paths",
    "wrist_depth_mask_paths",
    "wrist_depth_raw_paths",
    "final_mask_paths",
    "wrist_final_mask_paths",
    "wrist_keep_mask_paths",
)
_EPISODE_RE = re.compile(r"episode_(\d+)")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TACTILE_SIDECAR_RELATIVE = Path("meta/modality_artifacts/tactile")


class CameraSemanticMaterializationError(RuntimeError):
    """Raised when a camera namespace cannot be materialized safely."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative_path(root: Path, raw_path: object, *, label: str) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        raise CameraSemanticMaterializationError(f"Missing {label}.")
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        try:
            candidate = candidate.resolve().relative_to(root)
        except ValueError as exc:
            raise CameraSemanticMaterializationError(
                f"{label} is outside the dataset root: {text}"
            ) from exc
    if not candidate.parts or ".." in candidate.parts:
        raise CameraSemanticMaterializationError(f"Unsafe {label}: {text}")
    return candidate


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fresh_copy(source: Path, destination: Path) -> int:
    if source.is_symlink() or not source.is_file():
        raise CameraSemanticMaterializationError(
            f"Source artifact is not a regular non-symlink file: {source}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    source_stat = source.stat()
    source_hash = hashlib.sha256()
    target_hash = hashlib.sha256()
    try:
        with source.open("rb") as source_handle, os.fdopen(descriptor, "wb") as target:
            while block := source_handle.read(1024 * 1024):
                source_hash.update(block)
                written = target.write(block)
                if written != len(block):
                    raise OSError(f"short write {written}/{len(block)}")
                target_hash.update(block)
            target.flush()
            os.fsync(target.fileno())
        if source_hash.digest() != target_hash.digest():
            raise CameraSemanticMaterializationError(
                f"Fresh copy hash mismatch: {source} -> {destination}"
            )
        os.replace(temporary, destination)
        destination.chmod(source_stat.st_mode & 0o777)
        target_stat = destination.stat()
        if (source_stat.st_dev, source_stat.st_ino) == (
            target_stat.st_dev,
            target_stat.st_ino,
        ):
            raise CameraSemanticMaterializationError(
                f"Materialized file reused its source inode: {source}"
            )
        _fsync_directory(destination.parent)
        return int(target_stat.st_size)
    finally:
        temporary.unlink(missing_ok=True)


def _source_inventory(root: Path) -> tuple[tuple[Path, ...], str, int, int]:
    files: list[Path] = []
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    hardlinked = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CameraSemanticMaterializationError(
                f"Dataset materialization refuses symlinks: {path}"
            )
        if not path.is_file():
            continue
        stat = path.stat()
        relative = path.relative_to(root)
        files.append(relative)
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
        tuple(files),
        canonical_content_hash(
            entries,
            domain="dexteleop.camera_semantic_source_inventory.v1",
        ),
        total_bytes,
        hardlinked,
    )


def _episode_records(root: Path) -> tuple[dict[str, Any], ...]:
    rows = tuple(read_jsonl(root / "meta" / "episodes.jsonl"))
    if not rows:
        raise CameraSemanticMaterializationError(
            f"Dataset has no meta/episodes.jsonl records: {root}"
        )
    seen: set[int] = set()
    for row in rows:
        try:
            index = int(row["episode_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CameraSemanticMaterializationError(
                "Episode record has no integer episode_index."
            ) from exc
        if index in seen:
            raise CameraSemanticMaterializationError(
                f"Duplicate episode_index={index}."
            )
        seen.add(index)
    return rows


def _episode_tactile_references(
    root: Path,
    episodes: tuple[Mapping[str, Any], ...],
) -> set[str]:
    references: set[str] = set()
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        tactile_paths = episode.get("tactile_paths", {})
        if tactile_paths is None:
            continue
        if not isinstance(tactile_paths, Mapping):
            raise CameraSemanticMaterializationError(
                f"Episode {episode_index} tactile_paths must be an object."
            )
        for side, raw_path in tactile_paths.items():
            relative = _safe_relative_path(
                root,
                raw_path,
                label=f"episode {episode_index} tactile_paths.{side}",
            )
            if not relative.parts or relative.parts[0] != "tactile":
                raise CameraSemanticMaterializationError(
                    f"Tactile manifest lies outside tactile/: {relative}"
                )
            references.add(relative.as_posix())
    return references


def _tactile_artifact_graph(payload: Mapping[str, Any]) -> tuple[
    str,
    tuple[tuple[str, str, int, str], ...],
]:
    try:
        inventory = ModalityArtifactInventoryV1.from_dict(payload)
    except Exception as exc:
        raise CameraSemanticMaterializationError(
            f"Invalid tactile inventory contract: {exc}"
        ) from exc
    inventory_hash = inventory.semantic_content_hash
    if not _SHA256_RE.fullmatch(inventory_hash):
        raise CameraSemanticMaterializationError(
            "Tactile inventory has no valid semantic_content_hash."
        )
    if not inventory.artifacts:
        raise CameraSemanticMaterializationError(
            f"Tactile inventory {inventory_hash} has no artifacts."
        )
    artifacts: list[tuple[str, str, int, str]] = []
    for artifact in inventory.artifacts:
        logical_path = artifact.logical_path
        role = artifact.role
        size_bytes = artifact.size_bytes
        sha256 = artifact.sha256
        relative = Path(logical_path)
        if (
            role not in {"manifest", "payload"}
            or size_bytes < 0
            or not _SHA256_RE.fullmatch(sha256)
            or relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.parts[0] != "tactile"
        ):
            raise CameraSemanticMaterializationError(
                f"Tactile inventory {inventory_hash} has an unsafe artifact."
            )
        artifacts.append((logical_path, role, size_bytes, sha256))
    if len({item[0] for item in artifacts}) != len(artifacts):
        raise CameraSemanticMaterializationError(
            f"Tactile inventory {inventory_hash} has duplicate artifact paths."
        )
    return inventory_hash, tuple(sorted(artifacts))


def _tactile_detach_audit(
    root: Path,
    episodes: tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    """Prove camera export will not read or duplicate cold tactile payloads."""

    references = _episode_tactile_references(root, episodes)
    if not references:
        return {
            "status": "not_referenced",
            "safe_for_camera_materialization": True,
            "inventory_hashes": [],
            "detachment_scope": None,
            "manifest_count": 0,
            "payload_count": 0,
            "present_hot_payload_count": 0,
            "cold_payloads_read": False,
        }
    inventory_dir = root / _TACTILE_SIDECAR_RELATIVE / "inventories"
    if not inventory_dir.is_dir():
        return {
            "status": "missing_detachment_inventory",
            "safe_for_camera_materialization": False,
            "inventory_hashes": [],
            "detachment_scope": None,
            "manifest_count": len(references),
            "payload_count": 0,
            "present_hot_payload_count": 0,
            "cold_payloads_read": False,
        }

    candidates: list[
        tuple[str, str, tuple[tuple[str, str, int, str], ...]]
    ] = []
    for path in sorted(inventory_dir.glob("*.json")):
        payload = read_json(path)
        if not payload:
            raise CameraSemanticMaterializationError(
                f"Tactile inventory is empty or invalid: {path}"
            )
        inventory_hash, artifacts = _tactile_artifact_graph(payload)
        scope = verified_detachment_scope(root, inventory_hash)
        if scope is not None:
            candidates.append((inventory_hash, scope, artifacts))
    if not candidates:
        return {
            "status": "no_verified_retirement",
            "safe_for_camera_materialization": False,
            "inventory_hashes": [],
            "detachment_scope": None,
            "manifest_count": len(references),
            "payload_count": 0,
            "present_hot_payload_count": 0,
            "cold_payloads_read": False,
        }

    graphs = {candidate[2] for candidate in candidates}
    if len(graphs) != 1:
        raise CameraSemanticMaterializationError(
            "Verified tactile retirements describe different artifact graphs."
        )
    artifacts = next(iter(graphs))
    manifests = {item[0]: item for item in artifacts if item[1] == "manifest"}
    payloads = {item[0]: item for item in artifacts if item[1] == "payload"}
    if set(manifests) != references:
        raise CameraSemanticMaterializationError(
            "Episode tactile_paths differ from the verified retired manifest graph."
        )
    present_payloads = sorted(
        logical_path for logical_path in payloads if (root / logical_path).exists()
    )
    actual_hot = {
        path.relative_to(root).as_posix()
        for path in (root / "tactile").rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_hot == set(manifests):
        scope = PAYLOAD_ONLY_DETACH
        expected_hot = set(manifests)
    elif not actual_hot:
        scope = WHOLE_PREFIX_DETACH
        expected_hot = set()
    else:
        scope = None
        expected_hot = set()
    matching_candidates = [
        candidate for candidate in candidates if candidate[1] == scope
    ]
    unsafe_hot = sorted(actual_hot - expected_hot)
    missing_hot = sorted(expected_hot - actual_hot)
    manifest_mismatches: list[str] = []
    if scope == PAYLOAD_ONLY_DETACH:
        for logical_path, _, size_bytes, sha256 in manifests.values():
            path = root / logical_path
            if not path.is_file() or path.is_symlink():
                manifest_mismatches.append(logical_path)
                continue
            stat = path.stat()
            if int(stat.st_nlink) != 1 or int(stat.st_size) != size_bytes:
                manifest_mismatches.append(logical_path)
                continue
            if _sha256_file(path) != sha256:
                manifest_mismatches.append(logical_path)
    safe = (
        bool(matching_candidates)
        and not present_payloads
        and not unsafe_hot
        and not missing_hot
        and not manifest_mismatches
        and scope in {PAYLOAD_ONLY_DETACH, WHOLE_PREFIX_DETACH}
    )
    return {
        "status": "detached_verified" if safe else "detached_state_mismatch",
        "safe_for_camera_materialization": safe,
        "inventory_hashes": sorted(
            candidate[0] for candidate in matching_candidates
        ),
        "detachment_scope": scope,
        "manifest_count": len(manifests),
        "payload_count": len(payloads),
        "present_hot_payload_count": len(present_payloads),
        "unexpected_hot_path_count": len(unsafe_hot),
        "missing_hot_manifest_count": len(missing_hot),
        "manifest_mismatch_count": len(manifest_mismatches),
        "cold_payloads_read": False,
    }


def _resolution_maps(
    resolution: CameraSemanticResolution,
) -> tuple[dict[str, str], dict[str, str]]:
    canonical_to_stored = resolution.canonical_to_stored_key
    if set(canonical_to_stored) != set(WRIST_ROLES):
        raise CameraSemanticMaterializationError(
            "Materialization currently requires both wrist-camera roles."
        )
    stored_to_canonical = {
        stored: canonical for canonical, stored in canonical_to_stored.items()
    }
    if len(stored_to_canonical) != len(WRIST_ROLES):
        raise CameraSemanticMaterializationError(
            "Wrist camera resolution is not bijective."
        )
    return canonical_to_stored, stored_to_canonical


def _camera_role_in_path(path: Path) -> str | None:
    matches: set[str] = set()
    for part in path.parts:
        for role in WRIST_ROLES:
            if part == role or part == f"{CAMERA_COLUMN_PREFIX}{role}":
                matches.add(role)
    if len(matches) > 1:
        raise CameraSemanticMaterializationError(
            f"One artifact path contains both wrist roles: {path}"
        )
    return next(iter(matches), None)


def _episode_index_in_path(path: Path) -> int | None:
    match = _EPISODE_RE.search(path.as_posix())
    return None if match is None else int(match.group(1))


def _rewrite_role_text(text: str, mapping: Mapping[str, str]) -> str:
    rewritten = str(text)
    placeholders = {
        role: f"__DEXTELEOP_CAMERA_ROLE_{index}__"
        for index, role in enumerate(sorted(mapping))
    }
    for source_role, placeholder in placeholders.items():
        rewritten = rewritten.replace(
            f"{CAMERA_COLUMN_PREFIX}{source_role}",
            f"{CAMERA_COLUMN_PREFIX}{placeholder}",
        ).replace(f"/{source_role}/", f"/{placeholder}/")
        if rewritten == source_role:
            rewritten = placeholder
    for source_role, placeholder in placeholders.items():
        rewritten = rewritten.replace(placeholder, mapping[source_role])
    return rewritten


def _map_camera_artifact_path(
    relative: Path,
    stored_to_canonical: Mapping[str, str],
) -> Path:
    source_role = _camera_role_in_path(relative)
    if source_role is None:
        return relative
    target_role = stored_to_canonical[source_role]
    parts = [
        (
            target_role
            if part == source_role
            else f"{CAMERA_COLUMN_PREFIX}{target_role}"
            if part == f"{CAMERA_COLUMN_PREFIX}{source_role}"
            else part
        )
        for part in relative.parts
    ]
    return Path(*parts)


def _rewrite_camera_json(value: Any, mapping: Mapping[str, str]) -> Any:
    if isinstance(value, Mapping):
        rewritten: dict[str, Any] = {}
        for key, item in value.items():
            new_key = mapping.get(str(key), str(key))
            rewritten[new_key] = _rewrite_camera_json(item, mapping)
        return rewritten
    if isinstance(value, list):
        return [_rewrite_camera_json(item, mapping) for item in value]
    if isinstance(value, str):
        return _rewrite_role_text(value, mapping)
    return value


def _write_json_fsync(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl_fsync(path: Path, rows: tuple[Mapping[str, Any], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=True, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _camera_snapshot_from_binding(binding: Any) -> dict[str, Any]:
    return {
        "schema": "dexteleop.materialized_realsense_snapshot.v1",
        "cameras": {
            role.canonical_role: {
                "camera_key": role.canonical_role,
                "serial": role.serial,
                "serial_number": role.serial,
                "semantic_role": role.canonical_role,
                "materialized": True,
            }
            for role in binding.roles
        },
    }


def _parquet_module() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise CameraSemanticMaterializationError(
            "Camera materialization requires pyarrow."
        ) from exc
    return pa, pq


def _column_mapping(
    names: tuple[str, ...],
    stored_to_canonical: Mapping[str, str],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name in names:
        mapped = name
        for stored, canonical in stored_to_canonical.items():
            prefix = f"{CAMERA_COLUMN_PREFIX}{stored}."
            if name.startswith(prefix):
                mapped = f"{CAMERA_COLUMN_PREFIX}{canonical}.{name[len(prefix):]}"
                break
        mapping[name] = mapped
    if len(set(mapping.values())) != len(mapping):
        raise CameraSemanticMaterializationError(
            "Camera column materialization would create duplicate Parquet columns."
        )
    return mapping


def _rewrite_parquet(
    source: Path,
    destination: Path,
    *,
    stored_to_canonical: Mapping[str, str],
) -> int:
    pa, pq = _parquet_module()
    table = pq.read_table(source)
    original_names = tuple(table.column_names)
    mapping = _column_mapping(original_names, stored_to_canonical)
    for index, name in enumerate(original_names):
        mapped_name = mapping[name]
        if not mapped_name.endswith(".path"):
            continue
        column = table.column(index)
        if not (pa.types.is_string(column.type) or pa.types.is_large_string(column.type)):
            continue
        values = []
        for value in column.to_pylist():
            if value is None:
                values.append(None)
                continue
            values.append(_rewrite_role_text(str(value), stored_to_canonical))
        table = table.set_column(index, name, pa.array(values, type=column.type))
    table = table.rename_columns([mapping[name] for name in original_names])
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination, compression="zstd")
    with destination.open("rb") as handle:
        os.fsync(handle.fileno())
    reread = pq.read_table(destination)
    if reread.num_rows != table.num_rows or tuple(reread.column_names) != tuple(
        table.column_names
    ):
        raise CameraSemanticMaterializationError(
            f"Rewritten Parquet validation failed: {destination}"
        )
    return int(table.num_rows)


def _index_values(parquet_path: Path, role: str) -> list[int | None]:
    _, pq = _parquet_module()
    column_name = f"{CAMERA_COLUMN_PREFIX}{role}.video_frame_index"
    try:
        column = pq.read_table(parquet_path, columns=[column_name]).column(0)
    except Exception as exc:
        raise CameraSemanticMaterializationError(
            f"Cannot read {column_name} from {parquet_path}: {exc}"
        ) from exc
    return [None if value is None else int(value) for value in column.to_pylist()]


def _declared_video_frame_count(values: list[int | None], *, label: str) -> int:
    present = [value for value in values if value is not None]
    if not present:
        return 0
    if present != list(range(len(present))):
        raise CameraSemanticMaterializationError(
            f"Video frame indices are not contiguous for {label}."
        )
    return len(present)


def _video_probe_frame_count(path: Path) -> int:
    if shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None:
        raise CameraSemanticMaterializationError(
            "Safe video materialization requires ffprobe and ffmpeg."
        )
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0 or probe.stderr.strip():
        raise CameraSemanticMaterializationError(
            f"ffprobe rejected {path}: {probe.stderr.strip()}"
        )
    try:
        streams = json.loads(probe.stdout).get("streams", [])
        frame_count = int(streams[0]["nb_read_frames"])
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CameraSemanticMaterializationError(
            f"ffprobe did not report a decoded frame count for {path}."
        ) from exc
    decode = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "warning",
            "-xerror",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if decode.returncode != 0 or decode.stderr.strip():
        raise CameraSemanticMaterializationError(
            f"Full/tail video decode rejected {path}: {decode.stderr.strip()}"
        )
    return frame_count


def _frame_index_mismatch_count(
    parquet_path: Path,
    resolution: CameraSemanticResolution,
) -> tuple[int, int]:
    if resolution.status == "identity":
        return 0, 0
    left = _index_values(parquet_path, "left_wrist")
    right = _index_values(parquet_path, "right_wrist")
    if len(left) != len(right):
        raise CameraSemanticMaterializationError(
            f"Wrist index columns have different row counts: {parquet_path}"
        )
    mismatches = sum(a != b for a, b in zip(left, right))
    return len(left), mismatches


@dataclass(frozen=True)
class CameraSemanticMaterializationPlan:
    source_root: str
    output_root: str
    registry_path: str
    binding_hash: str
    source_inventory_hash: str
    source_file_count: int
    source_byte_count: int
    source_hardlinked_file_count: int
    tactile_detach_audit: Mapping[str, Any]
    episode_count: int
    frame_count: int
    status_counts: Mapping[str, int]
    remapped_episode_count: int
    remapped_row_count: int
    remapped_mismatched_row_count: int
    episode_resolutions: tuple[Mapping[str, Any], ...]

    @property
    def plan_hash(self) -> str:
        return canonical_content_hash(
            self.to_dict(include_hash=False),
            domain="dexteleop.camera_semantic_materialization_plan.v1",
        )

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "schema": "dexteleop.camera_semantic_materialization_plan.v1",
                "source_mutated": False,
                "output_uses_fresh_inodes": True,
                "frame_gap_policy": "preserve_on_physical_camera",
                "ordinary_loader_output": True,
            }
        )
        if include_hash:
            payload["plan_hash"] = self.plan_hash
        return payload


def plan_camera_semantic_materialization(
    source_root: str | Path,
    output_root: str | Path,
    registry_path: str | Path,
) -> CameraSemanticMaterializationPlan:
    source = Path(source_root).expanduser().resolve(strict=True)
    output = Path(output_root).expanduser().resolve()
    registry = Path(registry_path).expanduser().resolve(strict=True)
    if output == source or source in output.parents or output in source.parents:
        raise CameraSemanticMaterializationError(
            "Source and output dataset trees must be disjoint."
        )
    binding = load_camera_role_binding(registry)
    resolver = DatasetCameraSemanticResolver(source, binding)
    episodes = _episode_records(source)
    files, inventory_hash, byte_count, hardlinked_count = _source_inventory(source)
    tactile_detach_audit = _tactile_detach_audit(source, episodes)
    # A frozen plan must not authorize copying a source whose declared video
    # indices no longer match the fully decodable payload.  This matters most
    # for legacy hardlink lineages, where one truncated inode can appear under
    # several derived dataset names while their Parquet metadata still advertises
    # the original frame count.  Validate before returning an applyable hash so
    # apply never stages a source that is already unrecoverable.
    _validate_dataset_videos(source, episodes)
    status_counts: Counter[str] = Counter()
    frame_count = 0
    remapped_rows = 0
    mismatch_rows = 0
    remapped_episodes = 0
    episode_resolutions: list[dict[str, Any]] = []
    for record in episodes:
        episode_index = int(record["episode_index"])
        resolution = resolver.resolution_for_episode(episode_index)
        _resolution_maps(resolution)
        status_counts[resolution.status] += 1
        parquet_relative = _safe_relative_path(
            source,
            record.get("parquet_path"),
            label=f"episode {episode_index} parquet_path",
        )
        parquet_path = source / parquet_relative
        if not parquet_path.is_file() or parquet_path.is_symlink():
            raise CameraSemanticMaterializationError(
                f"Episode Parquet is missing or unsafe: {parquet_path}"
            )
        rows, mismatches = _frame_index_mismatch_count(parquet_path, resolution)
        episode_resolutions.append(
            {
                "episode_index": episode_index,
                "status": resolution.status,
                "canonical_to_stored_key": resolution.canonical_to_stored_key,
                "source_snapshot_hash": resolution.source_device_snapshot_hash,
                "row_count": rows,
                "left_right_video_index_mismatch_row_count": mismatches,
            }
        )
        frame_count += rows
        if resolution.status != "identity":
            remapped_episodes += 1
            remapped_rows += rows
            mismatch_rows += mismatches
    return CameraSemanticMaterializationPlan(
        source_root=str(source),
        output_root=str(output),
        registry_path=str(registry),
        binding_hash=binding.contract_hash,
        source_inventory_hash=inventory_hash,
        source_file_count=len(files),
        source_byte_count=byte_count,
        source_hardlinked_file_count=hardlinked_count,
        tactile_detach_audit=tactile_detach_audit,
        episode_count=len(episodes),
        frame_count=frame_count,
        status_counts=dict(sorted(status_counts.items())),
        remapped_episode_count=remapped_episodes,
        remapped_row_count=remapped_rows,
        remapped_mismatched_row_count=mismatch_rows,
        episode_resolutions=tuple(episode_resolutions),
    )


def _rewrite_episode_record(
    record: Mapping[str, Any],
    stored_to_canonical: Mapping[str, str],
) -> dict[str, Any]:
    output = dict(record)
    for key in CAMERA_ARTIFACT_MAP_KEYS:
        raw = output.get(key)
        if isinstance(raw, Mapping):
            output[key] = _rewrite_camera_json(raw, stored_to_canonical)
    summary = output.get("summary")
    if isinstance(summary, Mapping):
        summary_copy = dict(summary)
        for key in (
            "video_frame_stats",
            "wrist_depth_frame_stats",
            "wrist_depth_integrity",
        ):
            if key in summary_copy:
                summary_copy[key] = _rewrite_camera_json(
                    summary_copy[key], stored_to_canonical
                )
        output["summary"] = summary_copy
    metadata = output.get("metadata")
    if isinstance(metadata, Mapping):
        metadata_copy = dict(metadata)
        for key in ("wrist_depth_integrity", "video_frame_stats"):
            if key in metadata_copy:
                metadata_copy[key] = _rewrite_camera_json(
                    metadata_copy[key], stored_to_canonical
                )
        output["metadata"] = metadata_copy
    return output


def _rewrite_info(
    info: Mapping[str, Any],
    *,
    swap_wrist_feature_declarations: bool,
) -> dict[str, Any]:
    output = dict(info)
    features = output.get("features")
    if isinstance(features, Mapping):
        # Both canonical wrist feature names normally exist.  Swap their
        # declarations simultaneously so role-specific shapes/codecs follow
        # the physical stream without changing unrelated feature names.
        feature_copy = dict(features)
        left_key = f"{CAMERA_COLUMN_PREFIX}left_wrist"
        right_key = f"{CAMERA_COLUMN_PREFIX}right_wrist"
        if (
            swap_wrist_feature_declarations
            and left_key in feature_copy
            and right_key in feature_copy
        ):
            feature_copy[left_key], feature_copy[right_key] = (
                feature_copy[right_key],
                feature_copy[left_key],
            )
        output["features"] = feature_copy
    return output


def _validate_dataset_videos(
    root: Path,
    episodes: tuple[Mapping[str, Any], ...],
) -> tuple[int, int]:
    video_count = 0
    decoded_frames = 0
    for record in episodes:
        episode_index = int(record["episode_index"])
        parquet_relative = _safe_relative_path(
            root,
            record.get("parquet_path"),
            label=f"episode {episode_index} parquet_path",
        )
        videos = record.get("video_paths")
        if not isinstance(videos, Mapping):
            continue
        for role in WRIST_ROLES:
            raw_path = videos.get(role)
            if not raw_path:
                continue
            relative = _safe_relative_path(
                root, raw_path, label=f"episode {episode_index} {role} video"
            )
            path = root / relative
            declared = _declared_video_frame_count(
                _index_values(root / parquet_relative, role),
                label=f"episode={episode_index} role={role}",
            )
            decoded = _video_probe_frame_count(path)
            if decoded != declared:
                raise CameraSemanticMaterializationError(
                    "Declared/decoded video frame count mismatch: "
                    f"episode={episode_index} role={role} "
                    f"declared={declared} decoded={decoded}"
                )
            video_count += 1
            decoded_frames += decoded
    return video_count, decoded_frames


def materialize_camera_semantics(
    source_root: str | Path,
    output_root: str | Path,
    registry_path: str | Path,
    *,
    expected_plan_hash: str,
) -> dict[str, Any]:
    plan = plan_camera_semantic_materialization(
        source_root, output_root, registry_path
    )
    if not str(expected_plan_hash).strip() or expected_plan_hash != plan.plan_hash:
        raise CameraSemanticMaterializationError(
            "Apply requires the exact current --expected-plan-hash."
        )
    if not bool(plan.tactile_detach_audit["safe_for_camera_materialization"]):
        raise CameraSemanticMaterializationError(
            "Camera materialization requires tactile to be unreferenced or already "
            "retired with verified receipts; current tactile status is "
            f"{plan.tactile_detach_audit['status']}."
        )
    source = Path(plan.source_root)
    output = Path(plan.output_root)
    if output.exists():
        raise CameraSemanticMaterializationError(
            f"Output already exists; overwrite is forbidden: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.camera-semantic-stage-",
            dir=output.parent,
        )
    )
    binding = load_camera_role_binding(plan.registry_path)
    source_resolver = DatasetCameraSemanticResolver(source, binding)
    source_episodes = _episode_records(source)
    resolution_by_episode = {
        int(record["episode_index"]): source_resolver.resolution_for_episode(
            int(record["episode_index"])
        )
        for record in source_episodes
    }
    source_files, _, _, _ = _source_inventory(source)
    parquet_relatives = {
        _safe_relative_path(
            source,
            record.get("parquet_path"),
            label=f"episode {int(record['episode_index'])} parquet_path",
        )
        for record in source_episodes
    }
    copied_files = 0
    copied_bytes = 0
    transformed_camera_files = 0
    try:
        for relative in source_files:
            if relative in parquet_relatives:
                continue
            if relative == Path("meta/devices/realsense.json"):
                continue
            if relative == CAMERA_SEMANTIC_MATERIALIZATION_RELATIVE_PATH:
                continue
            role = _camera_role_in_path(relative)
            if role is not None:
                episode_index = _episode_index_in_path(relative)
                if episode_index is None or episode_index not in resolution_by_episode:
                    raise CameraSemanticMaterializationError(
                        "Camera artifact cannot be bound to a declared episode: "
                        f"{relative}"
                    )
                _, stored_to_canonical = _resolution_maps(
                    resolution_by_episode[episode_index]
                )
                target_relative = _map_camera_artifact_path(
                    relative, stored_to_canonical
                )
                source_path = source / relative
                target_path = staging / target_relative
                if relative.suffix.lower() == ".json":
                    payload = read_json(source_path)
                    if not payload:
                        raise CameraSemanticMaterializationError(
                            f"Camera artifact JSON is empty or invalid: {source_path}"
                        )
                    _write_json_fsync(
                        target_path,
                        _rewrite_camera_json(payload, stored_to_canonical),
                    )
                    copied_bytes += int(target_path.stat().st_size)
                else:
                    copied_bytes += _fresh_copy(source_path, target_path)
                copied_files += 1
                transformed_camera_files += 1
                continue
            copied_bytes += _fresh_copy(source / relative, staging / relative)
            copied_files += 1

        output_episodes: list[dict[str, Any]] = []
        for record in source_episodes:
            episode_index = int(record["episode_index"])
            resolution = resolution_by_episode[episode_index]
            _, stored_to_canonical = _resolution_maps(resolution)
            parquet_relative = _safe_relative_path(
                source,
                record.get("parquet_path"),
                label=f"episode {episode_index} parquet_path",
            )
            _rewrite_parquet(
                source / parquet_relative,
                staging / parquet_relative,
                stored_to_canonical=stored_to_canonical,
            )
            copied_files += 1
            copied_bytes += int((staging / parquet_relative).stat().st_size)
            output_episodes.append(
                _rewrite_episode_record(record, stored_to_canonical)
            )

        output_episode_tuple = tuple(output_episodes)
        _write_jsonl_fsync(
            staging / "meta" / "episodes.jsonl", output_episode_tuple
        )
        info_path = staging / "meta" / "info.json"
        if info_path.is_file():
            statuses = {
                resolution.status for resolution in resolution_by_episode.values()
            }
            info = read_json(info_path)
            features = info.get("features")
            if statuses == {"identity", "swapped"} and isinstance(features, Mapping):
                left_feature = features.get(
                    f"{CAMERA_COLUMN_PREFIX}left_wrist"
                )
                right_feature = features.get(
                    f"{CAMERA_COLUMN_PREFIX}right_wrist"
                )
                if left_feature != right_feature:
                    raise CameraSemanticMaterializationError(
                        "A mixed-semantics organized dataset has different left/right "
                        "feature declarations and cannot expose one canonical global schema."
                    )
            _write_json_fsync(
                info_path,
                _rewrite_info(
                    info,
                    swap_wrist_feature_declarations=statuses == {"swapped"},
                ),
            )
        output_snapshot = _camera_snapshot_from_binding(binding)
        snapshot_path = staging / "meta" / "devices" / "realsense.json"
        _write_json_fsync(snapshot_path, output_snapshot)
        snapshot_resolution = resolve_camera_snapshot(
            output_snapshot,
            binding,
            source_id=output.name,
        )
        receipt: dict[str, Any] = {
            "schema": CAMERA_SEMANTIC_MATERIALIZATION_SCHEMA_V1,
            "complete": True,
            "source_root": str(source),
            "source_inventory_hash": plan.source_inventory_hash,
            "plan_hash": plan.plan_hash,
            "binding_hash": binding.contract_hash,
            "output_snapshot_hash": snapshot_resolution.source_device_snapshot_hash,
            "episode_indices": sorted(resolution_by_episode),
            "source_resolutions": [
                {
                    "episode_index": episode_index,
                    "resolution_hash": resolution_by_episode[
                        episode_index
                    ].semantic_content_hash,
                    "status": resolution_by_episode[episode_index].status,
                    "canonical_to_stored_key": resolution_by_episode[
                        episode_index
                    ].canonical_to_stored_key,
                }
                for episode_index in sorted(resolution_by_episode)
            ],
            "frame_gap_policy": "preserve_on_physical_camera",
            "source_mutated": False,
            "output_uses_fresh_inodes": True,
        }
        receipt["receipt_hash"] = canonical_content_hash(
            receipt,
            domain=CAMERA_SEMANTIC_MATERIALIZATION_HASH_DOMAIN_V1,
        )
        receipt_path = staging / CAMERA_SEMANTIC_MATERIALIZATION_RELATIVE_PATH
        _write_json_fsync(receipt_path, receipt)

        output_resolver = DatasetCameraSemanticResolver(staging, binding)
        for episode_index in sorted(resolution_by_episode):
            if output_resolver.resolution_for_episode(episode_index).status != "identity":
                raise CameraSemanticMaterializationError(
                    f"Output episode {episode_index} is not canonical."
                )
        video_count, decoded_frames = _validate_dataset_videos(
            staging, output_episode_tuple
        )
        output_tactile_audit = _tactile_detach_audit(
            staging, output_episode_tuple
        )
        if output_tactile_audit != plan.tactile_detach_audit:
            raise CameraSemanticMaterializationError(
                "Materialized tactile references/receipts differ from the frozen source audit."
            )
        _, final_inventory_hash, _, _ = _source_inventory(source)
        if final_inventory_hash != plan.source_inventory_hash:
            raise CameraSemanticMaterializationError(
                "Source dataset changed during materialization; staged output was not published."
            )
        os.replace(staging, output)
        _fsync_directory(output.parent)
        return {
            **plan.to_dict(),
            "schema": CAMERA_SEMANTIC_MATERIALIZATION_SCHEMA_V1,
            "applied": True,
            "output_root": str(output),
            "copied_file_count": copied_files,
            "copied_byte_count": copied_bytes,
            "transformed_camera_file_count": transformed_camera_files,
            "validated_video_count": video_count,
            "validated_decoded_video_frame_count": decoded_frames,
            "receipt_path": str(output / CAMERA_SEMANTIC_MATERIALIZATION_RELATIVE_PATH),
            "receipt_hash": receipt["receipt_hash"],
        }
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "CameraSemanticMaterializationError",
    "CameraSemanticMaterializationPlan",
    "materialize_camera_semantics",
    "plan_camera_semantic_materialization",
]

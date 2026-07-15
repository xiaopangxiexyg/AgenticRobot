"""Build auditable, re-indexed LeRobot v2.1-style dataset derivatives.

The existing ``split-episodes`` and ``delete-episodes`` tools operate on one
dataset at a time.  This module complements them with an immutable-source
workflow that can select episodes from multiple roots, merge them into a new
dataset, assign contiguous episode indices, and apply narrowly-scoped schema
migrations.  Large immutable modality files are hard-linked by default; all
metadata and parquet files that may be rewritten are newly written.

The input is a JSON manifest.  See ``manifests/dexteleop_2026_07_sort.json``
for a complete example.  A build is staged beside its destination and only
published once every selected episode has been materialised successfully.
"""

from __future__ import annotations

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

from .io import episode_artifact_paths, read_json, read_jsonl, write_json, write_jsonl
from .selection import parse_episode_selection


_EPISODE_TOKEN_RE = re.compile(r"episode_\d+")
_CHUNK_TOKEN_RE = re.compile(r"chunk-\d+")
_PATH_FIELD_RE = re.compile(r"(?:^|_)(?:path|paths|relative_dir)$")
_DMTAC_SLOT_MIGRATION = "dmtac_slot1_to_slot0"
_DMTAC_SIDES = ("left", "right")
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
    episodes: tuple[PlannedEpisode, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetOrganizationError(ValueError):
    """Raised when an immutable-source derived dataset cannot be built safely."""


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:  # pragma: no cover - dependent on runtime environment.
        raise ModuleNotFoundError(
            "organize-datasets requires pyarrow. Run it in the DexTeleop Python environment."
        ) from exc
    return pa, pq


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
        if "episode_" not in value:
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
) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for artifact_path in episode_artifact_paths(record):
        relative = _safe_relative_path(source_root, artifact_path)
        target = source_root / relative
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
        output_chunk_size = int(raw_dataset.get("chunk_size", 1000))
        if output_chunk_size <= 0:
            raise DatasetOrganizationError(f"Dataset {name} chunk_size must be positive.")
        missing_requested: list[dict[str, Any]] = []
        filtered_by_duration: list[dict[str, Any]] = []
        missing_artifacts: list[dict[str, Any]] = []
        selected: list[tuple[Path, dict[str, Any], tuple[str, ...]]] = []
        minimum_duration_sec = _minimum_duration_sec(manifest, raw_dataset)
        source_fingerprint: dict[str, Any] | None = None
        for raw_source in raw_sources:
            if not isinstance(raw_source, Mapping):
                raise DatasetOrganizationError(f"Dataset {name} has an invalid source entry.")
            source_root = _resolve_manifest_path(path, raw_source.get("dataset_root"), label=f"dataset_root for {name}")
            if output_root == source_root or output_root.is_relative_to(source_root):
                raise DatasetOrganizationError(f"Output {output_root} must not be inside source {source_root}.")
            records_by_index = _record_by_episode_index(source_root)
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
                missing_artifacts.extend(_validate_artifacts(source_root, record))
                selected.append((source_root, record, transforms))
            source_info = read_json(source_root / "meta" / "info.json")
            fingerprint = _source_info_fingerprint(source_info)
            if source_fingerprint is None:
                source_fingerprint = fingerprint
            elif fingerprint != source_fingerprint:
                raise DatasetOrganizationError(
                    f"Dataset {name} has incompatible source info metadata. "
                    "Split by schema or explicitly migrate the source first."
                )
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
                episodes=episodes,
            )
        )
    return plans


def _hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _copy_artifact(
    source_root: Path,
    staging_root: Path,
    artifact_path: str,
    *,
    output_episode_index: int,
    chunk_size: int,
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
        _hardlink_or_copy(source, destination)
        return
    if not source.is_dir():
        raise DatasetOrganizationError(f"Declared artifact disappeared during build: {source}")
    for file_path in source.rglob("*"):
        if not file_path.is_file():
            continue
        rel_child = file_path.relative_to(source)
        target = destination / rel_child
        if file_path.name != "manifest.json":
            _hardlink_or_copy(file_path, target)
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
    for key in ("parquet_path", "video_paths", "tactile_paths", "wrist_depth_mask_paths", "wrist_depth_raw_paths"):
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
        "原始数据集未被移动或修改；大型视频、触觉和深度二进制流以硬链接复用。",
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
        minimum_duration_sec = _minimum_duration_sec(manifest, raw_dataset)
        staging_root = _prepare_output(output_root, overwrite=bool(overwrite))
        try:
            source_entries: list[tuple[Path, dict[str, Any], tuple[str, ...]]] = []
            source_stats_by_key: dict[tuple[Path, int], dict[str, Any]] = {}
            source_infos: dict[Path, dict[str, Any]] = {}
            source_summary: list[dict[str, Any]] = []
            for raw_source in raw_dataset["sources"]:
                source_root = _resolve_manifest_path(manifest_file, raw_source.get("dataset_root"), label="dataset_root")
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
                source_summary.append(
                    {
                        "source_dataset_root": str(source_root),
                        "episode_count": selected_count,
                        "transforms": list(transforms),
                    }
                )
            if len(source_entries) != plan.episode_count:
                raise DatasetOrganizationError("Build selection no longer matches its preflight plan.")
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
                    _copy_artifact(
                        source_root,
                        staging_root,
                        artifact_path,
                        output_episode_index=output_index,
                        chunk_size=chunk_size,
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
            write_json(
                meta_dir / "build_audit.json",
                {
                    "schema": "dexteleop.dataset_organization_audit.v1",
                    "created_at": time.time(),
                    "manifest": str(manifest_file),
                    "plan": plan.to_dict(),
                    "source_summary": source_summary,
                    "artifact_policy": "parquet/metadata rewritten; unchanged video/tactile/wrist-depth files hard-linked with copy fallback",
                },
            )
            shutil.copy2(manifest_file, meta_dir / "sort_manifest.json")
            _copy_source_devices((entry[0] for entry in source_entries), staging_root)
            (staging_root / "README.md").write_text(
                _readme_text(raw_dataset, plan, source_summary=source_summary),
                encoding="utf-8",
            )
            _published_output(staging_root, output_root)
        except BaseException:
            shutil.rmtree(staging_root, ignore_errors=True)
            raise
    return plans


__all__ = [
    "DatasetBuildPlan",
    "DatasetOrganizationError",
    "PlannedEpisode",
    "build_datasets",
    "plan_dataset_build",
]

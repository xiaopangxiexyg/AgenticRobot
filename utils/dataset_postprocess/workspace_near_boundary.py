"""Explicit opt-in UMI table-center -> workspace-near-boundary migration."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.pose_frames import PoseFrameContract

from .tactile_recompress import _active_collection_processes


MIGRATION_ID = "2026-07-19_umi_table_center_to_workspace_near_boundary_x_minus_0p15"
MIGRATION_SCHEMA = "dexteleop.legacy.umi_desk_frame_translation.v1"
EXPECTED_DATASET_SCHEMA = "dexteleop_dataset_v2.3"
EXPECTED_SOURCE_FAMILY = "vive_umi"
EXPECTED_STORAGE_FRAME = "desk_absolute"
DEFAULT_NEW_ORIGIN_IN_OLD_X_M = -0.15

# Each tuple lists X indices inside one serialized pose vector. The quaternion
# slots and all non-pose slots remain byte-semantically unchanged.
POSE_X_INDICES_BY_COLUMN: Mapping[str, tuple[int, ...]] = {
    "action": (0, 7),
    "action.arm.pose": (0, 7),
    "action.left_arm.pose": (0,),
    "action.right_arm.pose": (0,),
    "observation.state": (0, 7),
    "observation.arm.pose": (0, 7),
    "observation.last_action": (0, 7),
    "observation.left_arm.pose": (0,),
    "observation.right_arm.pose": (0,),
    "observation.arm.left.actual_tcp_pose": (0,),
    "observation.arm.right.actual_tcp_pose": (0,),
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        records.append(payload)
    return records


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(record), ensure_ascii=False) + "\n" for record in records
        ),
        encoding="utf-8",
    )


def _translation(x_m: float) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[0, 3] = float(x_m)
    return transform


def _transform_matrix(
    value: Any, *, coordinate_shift_x_m: float, name: str
) -> list[list[float]]:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 transform")
    return (_translation(coordinate_shift_x_m) @ matrix).tolist()


def _migrated_contract(
    original_payload: Mapping[str, Any],
    *,
    new_origin_in_old_x_m: float,
) -> tuple[dict[str, Any], str, str]:
    original = PoseFrameContract.from_dict(original_payload)
    if original.pose_storage_frame != EXPECTED_STORAGE_FRAME:
        raise ValueError(
            f"Expected {EXPECTED_STORAGE_FRAME!r}, got {original.pose_storage_frame!r}"
        )
    if original.source.source_family != EXPECTED_SOURCE_FAMILY:
        raise ValueError(
            f"Expected source_family={EXPECTED_SOURCE_FAMILY!r}, "
            f"got {original.source.source_family!r}"
        )

    coordinate_shift_x_m = -float(new_origin_in_old_x_m)
    payload = deepcopy(original.to_dict())
    source = payload["source"]
    source["source_description"] = (
        "Dual wrist UMI poses migrated from legacy table-center calibration-board "
        "coordinates to the workspace-near-boundary canonical desk."
    )
    source["raw_to_desk_transform"] = _transform_matrix(
        source["raw_to_desk_transform"],
        coordinate_shift_x_m=coordinate_shift_x_m,
        name="raw_to_desk_transform",
    )
    source["raw_to_canonical_desk_transform"] = deepcopy(
        source["raw_to_desk_transform"]
    )
    for legacy_key, canonical_key in (
        (
            "raw_to_desk_transform_by_side",
            "raw_to_canonical_desk_transform_by_side",
        ),
    ):
        transformed_by_side = {
            str(side): _transform_matrix(
                matrix,
                coordinate_shift_x_m=coordinate_shift_x_m,
                name=f"{legacy_key}.{side}",
            )
            for side, matrix in dict(source.get(legacy_key) or {}).items()
        }
        source[legacy_key] = transformed_by_side
        source[canonical_key] = deepcopy(transformed_by_side)

    contract_migration = {
        "schema": MIGRATION_SCHEMA,
        "migration_id": MIGRATION_ID,
        "legacy_frame": "table_center_frame",
        "canonical_frame": "workspace_near_boundary_frame",
        "new_origin_in_legacy_frame_m": [float(new_origin_in_old_x_m), 0.0, 0.0],
        "legacy_coordinates_to_canonical_coordinates_translation_m": [
            coordinate_shift_x_m,
            0.0,
            0.0,
        ],
        "rotation_changed": False,
    }
    source_metadata = dict(source.get("metadata") or {})
    source_metadata.update(
        {
            "desk_definition": "workspace_near_boundary_from_legacy_calibration_board",
            "legacy_calibration_board_frame": "table_center_frame",
            "canonical_desk_urdf_frame": "workspace_near_boundary_frame",
            "canonical_desk_definition": (
                "Left-right centered on the tabletop surface and located at the "
                "operator/robot-near boundary of the valid tabletop workspace along "
                "the tabletop-center-to-operator/robot direction."
            ),
            "legacy_pose_frame_migration": contract_migration,
        }
    )
    source["metadata"] = source_metadata
    contract_metadata = dict(payload.get("metadata") or {})
    contract_metadata["legacy_pose_frame_migration"] = contract_migration
    payload["metadata"] = contract_metadata

    migrated = PoseFrameContract.from_dict(payload)
    return migrated.to_dict(), original.contract_hash, migrated.contract_hash


def _shift_pose_vector(
    value: Any,
    *,
    x_indices: Sequence[int],
    coordinate_shift_x_m: float,
    name: str,
) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a pose vector")
    output = list(value)
    for index in x_indices:
        if index >= len(output):
            raise ValueError(f"{name} is too short for X index {index}: {len(output)}")
        output[index] = float(output[index]) + float(coordinate_shift_x_m)
    return output


def _migrate_rows(
    rows: list[dict[str, Any]],
    *,
    coordinate_shift_x_m: float,
    old_hash: str,
    new_hash: str,
    parquet_path: Path,
) -> None:
    for row_index, row in enumerate(rows):
        row_hash = str(row.get("dexteleop.pose_frame_contract_hash", ""))
        if row_hash != old_hash:
            raise ValueError(
                f"{parquet_path}: row {row_index} contract hash {row_hash!r} "
                f"does not match dataset hash {old_hash!r}"
            )
        for column, x_indices in POSE_X_INDICES_BY_COLUMN.items():
            if column not in row:
                raise ValueError(
                    f"{parquet_path}: missing required pose column {column!r}"
                )
            row[column] = _shift_pose_vector(
                row[column],
                x_indices=x_indices,
                coordinate_shift_x_m=coordinate_shift_x_m,
                name=f"{parquet_path}:row={row_index}:{column}",
            )
        row["dexteleop.pose_frame_contract_hash"] = new_hash


def _episode_parquet_paths(
    dataset_root: Path, episodes: Sequence[Mapping[str, Any]]
) -> list[Path]:
    paths: list[Path] = []
    for record in episodes:
        relative = Path(str(record.get("parquet_path", "")))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe parquet_path: {relative}")
        path = dataset_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise ValueError("episodes.jsonl references a Parquet file more than once")
    return paths


def _backup_root(dataset_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return dataset_root.with_name(
        f"{dataset_root.name}.before_workspace_near_boundary_shift_{stamp}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument(
        "--new-origin-in-old-x-m",
        type=float,
        default=DEFAULT_NEW_ORIGIN_IN_OLD_X_M,
        help=(
            "New workspace-near-boundary origin X expressed in the old table-center "
            "frame. Default: -0.15; stored pose X therefore increases by +0.15."
        ),
    )
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument(
        "--apply-workspace-near-boundary-shift",
        action="store_true",
        help=(
            "Explicitly enable this legacy coordinate migration after creating a "
            "data/meta backup. Without this exact flag the command is read-only."
        ),
    )
    return parser


def migrate_umi_workspace_near_boundary(
    dataset_root: str | Path,
    *,
    new_origin_in_old_x_m: float = DEFAULT_NEW_ORIGIN_IN_OLD_X_M,
    backup_root: str | Path | None = None,
    apply_workspace_near_boundary_shift: bool = False,
) -> dict[str, Any]:
    """Plan or explicitly apply the historical UMI coordinate correction.

    The mutating path is deliberately gated by the long, migration-specific
    ``apply_workspace_near_boundary_shift`` keyword. Ordinary postprocessing
    never calls or enables this function.
    """

    dataset_root = Path(dataset_root).expanduser().resolve()
    info_path = dataset_root / "meta" / "info.json"
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not info_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError(
            "dataset must contain meta/info.json and meta/episodes.jsonl"
        )

    info = _read_json(info_path)
    if str(info.get("dexteleop_dataset_schema")) != EXPECTED_DATASET_SCHEMA:
        raise ValueError("migration only accepts native DexTeleop v2.3 datasets")
    if str(info.get("embodiment")) != "umi_dual_wrist":
        raise ValueError("migration only accepts embodiment=umi_dual_wrist")
    for record in info.get("pose_frame_migrations", []):
        if isinstance(record, Mapping) and record.get("migration_id") == MIGRATION_ID:
            raise RuntimeError(f"migration {MIGRATION_ID} was already applied")

    contract_payload = info.get("pose_frame_contract")
    if not isinstance(contract_payload, Mapping):
        raise ValueError("meta/info.json has no pose_frame_contract object")
    migrated_contract, old_hash, new_hash = _migrated_contract(
        contract_payload,
        new_origin_in_old_x_m=float(new_origin_in_old_x_m),
    )
    if str(info.get("pose_frame_contract_hash")) != old_hash:
        raise ValueError("meta/info.json pose_frame_contract_hash is inconsistent")

    episodes = _read_jsonl(episodes_path)
    parquet_paths = _episode_parquet_paths(dataset_root, episodes)
    coordinate_shift_x_m = -float(new_origin_in_old_x_m)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("this migration requires pyarrow") from exc

    # Dry-run performs the same full row-level preflight as apply. Mutated rows
    # are discarded unless the migration-specific apply gate is true.
    total_rows = 0
    for parquet_path in parquet_paths:
        rows = pq.read_table(parquet_path).to_pylist()
        _migrate_rows(
            rows,
            coordinate_shift_x_m=coordinate_shift_x_m,
            old_hash=old_hash,
            new_hash=new_hash,
            parquet_path=parquet_path,
        )
        total_rows += len(rows)

    result: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "mode": "dry-run",
        "migration_enabled": False,
        "migration_id": MIGRATION_ID,
        "episode_count": len(episodes),
        "frame_count": total_rows,
        "new_origin_in_legacy_frame_m": [float(new_origin_in_old_x_m), 0.0, 0.0],
        "stored_pose_translation_m": [coordinate_shift_x_m, 0.0, 0.0],
        "rotation_changed": False,
        "old_pose_frame_contract_hash": old_hash,
        "new_pose_frame_contract_hash": new_hash,
        "backup_root": None,
    }
    if not apply_workspace_near_boundary_shift:
        return result

    active_collection = _active_collection_processes()
    if active_collection:
        raise RuntimeError(
            "Refusing workspace-near-boundary migration while formal collection "
            "is active: " + "; ".join(active_collection[:3])
        )

    applied_at = datetime.now(timezone.utc).isoformat()
    migration_record = {
        "schema": MIGRATION_SCHEMA,
        "migration_id": MIGRATION_ID,
        "applied_at_utc": applied_at,
        "legacy_frame": "table_center_frame",
        "canonical_frame": "workspace_near_boundary_frame",
        "new_origin_in_legacy_frame_m": [float(new_origin_in_old_x_m), 0.0, 0.0],
        "stored_pose_translation_m": [coordinate_shift_x_m, 0.0, 0.0],
        "rotation_changed": False,
        "old_pose_frame_contract_hash": old_hash,
        "new_pose_frame_contract_hash": new_hash,
        "pose_columns": list(POSE_X_INDICES_BY_COLUMN),
    }

    staging = Path(
        tempfile.mkdtemp(prefix=".umi_workspace_frame_stage_", dir=dataset_root.parent)
    )
    try:
        for parquet_path in parquet_paths:
            relative = parquet_path.relative_to(dataset_root)
            table = pq.read_table(parquet_path)
            rows = table.to_pylist()
            _migrate_rows(
                rows,
                coordinate_shift_x_m=coordinate_shift_x_m,
                old_hash=old_hash,
                new_hash=new_hash,
                parquet_path=parquet_path,
            )
            migrated_table = pa.Table.from_pylist(rows, schema=table.schema)
            output = staging / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            compression = (
                pq.ParquetFile(parquet_path)
                .metadata.row_group(0)
                .column(0)
                .compression.lower()
            )
            pq.write_table(migrated_table, output, compression=compression)

        migrated_info = deepcopy(info)
        migrated_info["pose_frame_contract"] = migrated_contract
        migrated_info["pose_frame_contract_hash"] = new_hash
        migrated_info.setdefault("pose_frame_migrations", []).append(migration_record)
        _write_json(staging / "meta" / "info.json", migrated_info)

        migrated_episodes = deepcopy(episodes)
        for record in migrated_episodes:
            metadata = record.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError(
                    f"episode {record.get('episode_index')} metadata must be an object"
                )
            if str(metadata.get("pose_frame_contract_hash")) != old_hash:
                raise ValueError(
                    f"episode {record.get('episode_index')} contract hash is inconsistent"
                )
            metadata["pose_frame_contract"] = migrated_contract
            metadata["pose_frame_contract_hash"] = new_hash
            metadata.setdefault("pose_frame_migrations", []).append(migration_record)
        _write_jsonl(staging / "meta" / "episodes.jsonl", migrated_episodes)
        _write_json(
            staging / "meta" / "workspace_near_boundary_migration.json",
            migration_record,
        )

        resolved_backup_root = _backup_root(
            dataset_root,
            None if backup_root is None else Path(backup_root).expanduser(),
        )
        if resolved_backup_root.exists():
            raise FileExistsError(
                f"backup target already exists: {resolved_backup_root}"
            )
        resolved_backup_root.mkdir(parents=True)
        shutil.copytree(dataset_root / "data", resolved_backup_root / "data")
        shutil.copytree(dataset_root / "meta", resolved_backup_root / "meta")

        for parquet_path in parquet_paths:
            relative = parquet_path.relative_to(dataset_root)
            os.replace(staging / relative, parquet_path)
        os.replace(staging / "meta" / "info.json", info_path)
        os.replace(staging / "meta" / "episodes.jsonl", episodes_path)
        os.replace(
            staging / "meta" / "workspace_near_boundary_migration.json",
            dataset_root / "meta" / "workspace_near_boundary_migration.json",
        )
        result.update(
            {
                "mode": "applied",
                "migration_enabled": True,
                "backup_root": str(resolved_backup_root),
            }
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = migrate_umi_workspace_near_boundary(
        args.dataset_root,
        new_origin_in_old_x_m=float(args.new_origin_in_old_x_m),
        backup_root=args.backup_root,
        apply_workspace_near_boundary_shift=bool(
            args.apply_workspace_near_boundary_shift
        ),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

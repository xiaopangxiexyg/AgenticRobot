"""Copy-on-write RM65 v2.2 raw-base to v2.3 canonical-desk migration.

This is deliberately a compatibility migration, not a native-v2.3 causal
snapshot reconstruction.  It changes only coordinate-bearing absolute poses,
their frame/layout declarations, and additive validity/provenance columns.
Historical source/audit columns whose names explicitly say ``*_base`` or the
old ``teleop.action_source.*_pose_desk`` remain untouched.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.pose_frames import (
    POSE_FRAME_CONTRACT_METADATA_KEY,
    PoseFrameContract,
    load_pose_frame_contract,
)
from dataset_contracts.pose_training import (
    POSE_TRAINING_DATA_METADATA_KEY,
    PoseTrainingDataContract,
)
from dataset_contracts.v22 import (
    DEXTELEOP_DATASET_SCHEMA_V22,
    PolicyIOLayout,
    policy_io_layout_from_metadata,
)
from dataset_contracts.v23 import (
    DEXTELEOP_DATASET_SCHEMA_V23,
    V23_COMPATIBILITY_KIND_METADATA_KEY,
    V23_COMPATIBILITY_V22_CANONICAL_DESK,
    V23_NATIVE_CAUSAL_SNAPSHOT_METADATA_KEY,
    V23_POSE_ACTION_MODE_METADATA_KEY,
    V23_POSE_STORAGE_FRAME_METADATA_KEY,
    V23_RELATIVE_ACTION_SUPPORTED_METADATA_KEY,
    V23_SOURCE_SCHEMA_METADATA_KEY,
)
from .io import clone_dataset_tree, episode_records, read_json, write_json, write_jsonl
from .tactile_recompress import _active_collection_processes
from .v21_to_v22 import load_v22_layout_json


MIGRATION_SCHEMA = "dexteleop.v22_to_v23_canonical_desk.v1"
ROW_ACTION_FINITE_COLUMN = "training.action_finite"
ROW_LAST_ACTION_FINITE_COLUMN = "training.last_action_finite"
ROW_RELATIVE_LAST_ACTION_FINITE_COLUMN = "training.relative_last_action_finite"
ROW_COMPATIBILITY_KIND_COLUMN = "dexteleop.v23_compatibility_kind"
ROW_POSE_STORAGE_FRAME_COLUMN = "dexteleop.pose_storage_frame"
ROW_POSE_CONTRACT_HASH_COLUMN = "dexteleop.pose_frame_contract_hash"
DEFAULT_TARGET_LAYOUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "teleop_runtime/configs/policy_io_layouts/dual_arm_bimanual_dmtac_gripper_v23_desk.json"
)
_RM65_POSE_FRAME_PROFILE_DIR = (
    Path(__file__).resolve().parents[2]
    / "robot_motion/configs/pose_frames"
)
_RM65_COLLECTION_POSE_FRAME_CONTRACT_PATH = (
    _RM65_POSE_FRAME_PROFILE_DIR
    / "rm65_dual_arm_workspace_near_boundary_v1_collection_desk_absolute.json"
)
_RM65_RUNTIME_POSE_FRAME_CONTRACT_PATH = (
    _RM65_POSE_FRAME_PROFILE_DIR
    / "rm65_dual_arm_workspace_near_boundary_v1_runtime_raw_absolute.json"
)

PACKED_POSE_COLUMNS: Mapping[str, bool] = {
    "action": True,
    "action.arm.pose": True,
    "observation.state": False,
    "observation.arm.pose": False,
    "observation.last_action": True,
}
SIDE_POSE_COLUMNS: Mapping[str, tuple[str, bool]] = {
    "action.{side}_arm.pose": ("action", True),
    "observation.{side}_arm.pose": ("state", False),
    "observation.arm.{side}.actual_tcp_pose": ("state", False),
    "observation.arm.{side}.current_tcp_pose": ("state", True),
}


def _default_rm65_collection_pose_frame_contract(
    active_sides: Sequence[str],
) -> PoseFrameContract | None:
    """Load the checked-in migration profile without importing runtime code."""

    sides = tuple(str(side).strip().lower() for side in active_sides)
    if sides != ("left", "right"):
        return None
    return load_pose_frame_contract(_RM65_COLLECTION_POSE_FRAME_CONTRACT_PATH)


def _default_rm65_runtime_pose_frame_contract() -> PoseFrameContract:
    return load_pose_frame_contract(_RM65_RUNTIME_POSE_FRAME_CONTRACT_PATH)


@dataclass(frozen=True)
class V22ToV23DeskPlan:
    source_root: Path
    output_root: Path
    source_layout_hash: str
    target_layout_hash: str
    pose_frame_contract_hash: str
    source_info_sha256: str
    conversion_id: str
    episode_count: int
    frame_count: int
    invalid_action_rows: int
    invalid_last_action_rows: int
    first_row_invalid_last_action_episodes: int
    parquet_relative_paths: tuple[str, ...]

    def to_dict(self, *, mode: str = "dry-run") -> dict[str, Any]:
        return {
            "operation": V23_COMPATIBILITY_V22_CANONICAL_DESK,
            "schema": MIGRATION_SCHEMA,
            "mode": str(mode),
            "migration_enabled": mode == "applied",
            "source_root": str(self.source_root),
            "output_root": str(self.output_root),
            "source_schema": DEXTELEOP_DATASET_SCHEMA_V22,
            "target_schema": DEXTELEOP_DATASET_SCHEMA_V23,
            "native_v23_causal_snapshot": False,
            "source_layout_hash": self.source_layout_hash,
            "target_layout_hash": self.target_layout_hash,
            "pose_frame_contract_hash": self.pose_frame_contract_hash,
            "source_info_sha256": self.source_info_sha256,
            "conversion_id": self.conversion_id,
            "episode_count": self.episode_count,
            "frame_count": self.frame_count,
            "invalid_action_rows": self.invalid_action_rows,
            "invalid_last_action_rows": self.invalid_last_action_rows,
            "first_row_invalid_last_action_episodes": (
                self.first_row_invalid_last_action_episodes
            ),
            "parquet_count": len(self.parquet_relative_paths),
            "parquet_relative_paths": list(self.parquet_relative_paths),
            "source_mutated": False,
            "relative_action_world_transform_invariant": True,
            "invalid_values_imputed": False,
            "copy_strategy": (
                "metadata_and_parquet_copy; external_artifacts_hardlink_or_copy"
            ),
        }


def _finite_vector(value: Any, *, length: int | None = None) -> bool:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return False
    return bool(
        (length is None or array.size == int(length)) and np.all(np.isfinite(array))
    )


def _same_pose_values(left: Any, right: Any) -> bool:
    try:
        a = np.asarray(left, dtype=float).reshape(-1)
        b = np.asarray(right, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return False
    return bool(a.shape == b.shape and np.allclose(a, b, atol=1e-6, equal_nan=True))


def _validate_row_pose_copies(row: Mapping[str, Any], *, label: str) -> None:
    for side, start in (("left", 0), ("right", 7)):
        action_pose = list(row["action"])[start : start + 7]
        for name, value in {
            "action.arm.pose": list(row["action.arm.pose"])[start : start + 7],
            f"action.{side}_arm.pose": row[f"action.{side}_arm.pose"],
        }.items():
            if not _same_pose_values(action_pose, value):
                raise ValueError(f"{label}:{side} {name} differs from packed action.")
        state_pose = list(row["observation.state"])[start : start + 7]
        for name, value in {
            "observation.arm.pose": list(row["observation.arm.pose"])[start : start + 7],
            f"observation.{side}_arm.pose": row[f"observation.{side}_arm.pose"],
            f"observation.arm.{side}.actual_tcp_pose": row[
                f"observation.arm.{side}.actual_tcp_pose"
            ],
        }.items():
            if not _same_pose_values(state_pose, value):
                raise ValueError(
                    f"{label}:{side} {name} differs from packed observation.state."
                )


def _safe_relative_parquet_path(
    source_root: Path, record: Mapping[str, Any]
) -> Path:
    value = record.get("parquet_path")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Episode {record.get('episode_index')} has no parquet_path.")
    path = Path(value)
    if path.is_absolute():
        try:
            path = path.expanduser().resolve().relative_to(source_root)
        except ValueError as exc:
            raise ValueError(f"Parquet path is outside source root: {value}") from exc
    if not path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe parquet path: {value}")
    resolved = source_root / path
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"Parquet path is missing, non-regular, or symlinked: {resolved}")
    return path


def _validate_layout_transition(source: PolicyIOLayout, target: PolicyIOLayout) -> None:
    if source.action_dim != target.action_dim or source.state_dim != target.state_dim:
        raise ValueError(
            "Target desk layout changes policy dimensions: "
            f"action {source.action_dim}->{target.action_dim}, "
            f"state {source.state_dim}->{target.state_dim}."
        )
    if source.action_discrete_dim != target.action_discrete_dim:
        raise ValueError("Target desk layout changes action discrete dimensions.")
    if source.state_discrete_dim != target.state_discrete_dim:
        raise ValueError("Target desk layout changes state discrete dimensions.")
    for side in ("left", "right"):
        if source.arm_controlled_frame_by_side[side] != target.arm_controlled_frame_by_side[side]:
            raise ValueError(f"Target layout changes {side} controlled frame.")
        if target.arm_reference_frame_by_side[side] != "desk":
            raise ValueError(f"Target layout {side} reference frame must be 'desk'.")
        if source.end_effector(side).to_dict() != target.end_effector(side).to_dict():
            raise ValueError(f"Target layout changes {side} end-effector semantics.")


def _desk_layout_from_source(source: PolicyIOLayout) -> PolicyIOLayout:
    metadata = dict(source.metadata)
    metadata.update(
        {
            "pose_storage_frame": "desk_absolute",
            "compatibility_origin": V23_COMPATIBILITY_V22_CANONICAL_DESK,
        }
    )
    return PolicyIOLayout(
        name=source.name,
        arm_controlled_frame_by_side=source.arm_controlled_frame_by_side,
        arm_reference_frame_by_side={"left": "desk", "right": "desk"},
        end_effectors=source.end_effectors,
        metadata=metadata,
    )


def _transform_pose7(
    value: Any,
    *,
    side: str,
    runtime_contract: PoseFrameContract,
    allow_nonfinite: bool,
    label: str,
) -> tuple[list[Any], bool]:
    try:
        values = list(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be a pose vector.") from exc
    if len(values) != 7:
        raise ValueError(f"{label} must contain 7 values, got {len(values)}.")
    finite = _finite_vector(values, length=7)
    if not finite:
        if allow_nonfinite:
            return values, False
        raise ValueError(f"{label} contains a non-finite required observation pose.")
    transformed = runtime_contract.storage_pose_vector_to_canonical_desk(
        values, side=side
    )
    return transformed.tolist(), True


def _transform_packed(
    value: Any,
    *,
    runtime_contract: PoseFrameContract,
    allow_nonfinite: bool,
    label: str,
) -> tuple[list[Any], bool]:
    try:
        output = list(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be a packed pose vector.") from exc
    if len(output) < 14:
        raise ValueError(f"{label} must contain both 7D arm prefixes.")
    all_finite = True
    for side, start in (("left", 0), ("right", 7)):
        converted, finite = _transform_pose7(
            output[start : start + 7],
            side=side,
            runtime_contract=runtime_contract,
            allow_nonfinite=allow_nonfinite,
            label=f"{label}.{side}",
        )
        output[start : start + 7] = converted
        all_finite = all_finite and finite
    return output, all_finite


def _desk_arm_specs(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    output = deepcopy(dict(value))
    for side in ("left", "right"):
        spec = output.get(side)
        if isinstance(spec, Mapping):
            updated = dict(spec)
            updated["reference_frame"] = "desk"
            output[side] = updated
    return output


def _transform_sample_metadata_json(
    value: Any,
    *,
    runtime_contract: PoseFrameContract,
) -> Any:
    if value is None:
        return None
    try:
        payload = json.loads(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("sample_metadata_json is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("sample_metadata_json must contain an object.")
    record = payload.get("canonical_action_record")
    if isinstance(record, dict):
        action = record.get("action")
        if isinstance(action, dict):
            poses = action.get("arm_pose_by_side")
            if isinstance(poses, dict):
                for side in ("left", "right"):
                    if side in poses:
                        poses[side], _ = _transform_pose7(
                            poses[side],
                            side=side,
                            runtime_contract=runtime_contract,
                            allow_nonfinite=True,
                            label=f"sample_metadata.canonical_action.{side}",
                        )
            frames = action.get("frame_by_effector")
            if isinstance(frames, dict):
                frames.update({"left_arm": "desk", "right_arm": "desk"})
            metadata = action.get("metadata")
            if isinstance(metadata, dict):
                metadata["arm_reference_frame_by_side"] = {
                    "left": "desk",
                    "right": "desk",
                }
                if "arm_action_specs_by_side" in metadata:
                    metadata["arm_action_specs_by_side"] = _desk_arm_specs(
                        metadata["arm_action_specs_by_side"]
                    )
        action_vector = record.get("action_vector")
        if isinstance(action_vector, dict) and "vector" in action_vector:
            action_vector["vector"], _ = _transform_packed(
                action_vector["vector"],
                runtime_contract=runtime_contract,
                allow_nonfinite=True,
                label="sample_metadata.canonical_action_vector",
            )
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _replace_or_append_column(table: Any, name: str, values: Sequence[Any], type_: Any) -> Any:
    import pyarrow as pa

    array = pa.array(values, type=type_)
    index = table.schema.get_field_index(name)
    if index >= 0:
        return table.set_column(index, name, array)
    return table.append_column(name, array)


def _converted_table(
    table: Any,
    *,
    runtime_contract: PoseFrameContract,
    collection_contract: PoseFrameContract,
    target_layout: PolicyIOLayout,
    parquet_label: str,
) -> tuple[Any, dict[str, int]]:
    import pyarrow as pa

    rows = table.to_pylist()
    names = set(table.schema.names)
    required = set(PACKED_POSE_COLUMNS)
    required.update(
        template.format(side=side)
        for template, (_, optional) in SIDE_POSE_COLUMNS.items()
        if not optional
        for side in ("left", "right")
    )
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"{parquet_label} is missing required pose columns: {missing}")
    converted_by_column: dict[str, list[Any]] = {
        name: [] for name in PACKED_POSE_COLUMNS if name in names
    }
    for template in SIDE_POSE_COLUMNS:
        for side in ("left", "right"):
            name = template.format(side=side)
            if name in names:
                converted_by_column[name] = []
    action_finite_values: list[bool] = []
    last_action_finite_values: list[bool] = []
    converted_arm_specs: list[Any] = []
    converted_sample_metadata: list[Any] = []
    for row_index, row in enumerate(rows):
        _validate_row_pose_copies(
            row, label=f"{parquet_label}:row={row_index}"
        )
        for name, allow_nonfinite in PACKED_POSE_COLUMNS.items():
            if name not in names:
                continue
            converted, _ = _transform_packed(
                row[name],
                runtime_contract=runtime_contract,
                allow_nonfinite=allow_nonfinite,
                label=f"{parquet_label}:row={row_index}:{name}",
            )
            converted_by_column[name].append(converted)
        for template, (_, optional) in SIDE_POSE_COLUMNS.items():
            for side in ("left", "right"):
                name = template.format(side=side)
                if name not in names:
                    continue
                converted, _ = _transform_pose7(
                    row[name],
                    side=side,
                    runtime_contract=runtime_contract,
                    allow_nonfinite=optional,
                    label=f"{parquet_label}:row={row_index}:{name}",
                )
                converted_by_column[name].append(converted)
        action_finite_values.append(
            _finite_vector(row.get("action"), length=target_layout.action_dim)
        )
        last_action_finite_values.append(
            _finite_vector(
                row.get("observation.last_action"), length=target_layout.action_dim
            )
        )
        if "arm_action_specs_json" in names:
            try:
                specs = json.loads(str(row["arm_action_specs_json"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{parquet_label}:row={row_index}: invalid arm_action_specs_json"
                ) from exc
            converted_arm_specs.append(
                json.dumps(_desk_arm_specs(specs), ensure_ascii=True, sort_keys=True)
            )
        if "sample_metadata_json" in names:
            converted_sample_metadata.append(
                _transform_sample_metadata_json(
                    row["sample_metadata_json"], runtime_contract=runtime_contract
                )
            )
    output = table
    for name, values in converted_by_column.items():
        output = _replace_or_append_column(
            output, name, values, table.schema.field(name).type
        )
    if "arm_action_specs_json" in names:
        output = _replace_or_append_column(
            output,
            "arm_action_specs_json",
            converted_arm_specs,
            table.schema.field("arm_action_specs_json").type,
        )
    if "sample_metadata_json" in names:
        output = _replace_or_append_column(
            output,
            "sample_metadata_json",
            converted_sample_metadata,
            table.schema.field("sample_metadata_json").type,
        )
    constants = {
        "dexteleop.dataset_schema": DEXTELEOP_DATASET_SCHEMA_V23,
        "policy_io_layout_hash": target_layout.layout_hash,
        ROW_POSE_STORAGE_FRAME_COLUMN: "desk_absolute",
        ROW_POSE_CONTRACT_HASH_COLUMN: collection_contract.contract_hash,
        ROW_COMPATIBILITY_KIND_COLUMN: V23_COMPATIBILITY_V22_CANONICAL_DESK,
    }
    for name, value in constants.items():
        output = _replace_or_append_column(
            output, name, [value] * len(rows), pa.string()
        )
    for name, values in {
        ROW_ACTION_FINITE_COLUMN: action_finite_values,
        ROW_LAST_ACTION_FINITE_COLUMN: last_action_finite_values,
        ROW_RELATIVE_LAST_ACTION_FINITE_COLUMN: last_action_finite_values,
    }.items():
        output = _replace_or_append_column(output, name, values, pa.bool_())
    output = output.replace_schema_metadata(table.schema.metadata)
    return output, {
        "rows": len(rows),
        "invalid_action_rows": sum(not value for value in action_finite_values),
        "invalid_last_action_rows": sum(
            not value for value in last_action_finite_values
        ),
        "first_row_invalid_last_action": int(
            bool(last_action_finite_values) and not last_action_finite_values[0]
        ),
    }


def _contracts_and_layouts(
    info: Mapping[str, Any], target_layout_path: str | Path | None
) -> tuple[PolicyIOLayout, PolicyIOLayout, PoseFrameContract, PoseFrameContract]:
    if str(info.get("dexteleop_dataset_schema")) != DEXTELEOP_DATASET_SCHEMA_V22:
        raise ValueError("Source must be an explicit DexTeleop v2.2 dataset.")
    source_layout = policy_io_layout_from_metadata(info, allow_legacy_v21=False)
    target_layout = (
        _desk_layout_from_source(source_layout)
        if target_layout_path is None or not str(target_layout_path).strip()
        else load_v22_layout_json(target_layout_path)
    )
    _validate_layout_transition(source_layout, target_layout)
    expected_refs = {"left": "left_arm_base", "right": "right_arm_base"}
    if dict(source_layout.arm_reference_frame_by_side) != expected_refs:
        raise ValueError(
            "Source layout is not the supported dual-RM65 base-frame layout: "
            f"{source_layout.arm_reference_frame_by_side!r}"
        )
    declared_refs = dict(info.get("arm_reference_frame_by_side") or {})
    if declared_refs and declared_refs != expected_refs:
        raise ValueError(
            "Dataset arm_reference_frame_by_side is not the supported RM65 station."
        )
    runtime_contract = _default_rm65_runtime_pose_frame_contract()
    collection_contract = (
        _default_rm65_collection_pose_frame_contract(
            ("left", "right")
        )
    )
    if collection_contract is None:
        raise RuntimeError("Checked-in dual-RM65 collection pose contract is unavailable.")
    return source_layout, target_layout, runtime_contract, collection_contract


def plan_v22_to_v23_desk_conversion(
    source_root: str | Path,
    output_root: str | Path,
    *,
    target_layout_path: str | Path | None = None,
) -> V22ToV23DeskPlan:
    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output == source:
        raise ValueError("output_root must differ from source_root.")
    try:
        output.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError("output_root must not be inside source_root.")
    info_path = source / "meta/info.json"
    if not info_path.is_file() or info_path.is_symlink():
        raise ValueError(f"Source must contain regular metadata: {info_path}")
    info_bytes = info_path.read_bytes()
    info = read_json(info_path)
    source_layout, target_layout, runtime_contract, collection_contract = (
        _contracts_and_layouts(info, target_layout_path)
    )
    records = episode_records(source)
    declared_count = info.get("episode_count")
    if declared_count is not None and int(declared_count) != len(records):
        raise ValueError("info.json episode_count disagrees with episodes.jsonl.")
    relative_paths: list[Path] = []
    seen_paths: set[Path] = set()
    seen_episodes: set[int] = set()
    totals = {
        "rows": 0,
        "invalid_action_rows": 0,
        "invalid_last_action_rows": 0,
        "first_row_invalid_last_action": 0,
    }
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("v2.2 desk migration requires pyarrow.") from exc
    for record in records:
        episode_index = int(record["episode_index"])
        if episode_index in seen_episodes:
            raise ValueError(f"Duplicate episode_index: {episode_index}")
        seen_episodes.add(episode_index)
        relative = _safe_relative_parquet_path(source, record)
        if relative in seen_paths:
            raise ValueError(f"Duplicate parquet path: {relative}")
        seen_paths.add(relative)
        relative_paths.append(relative)
        table = pq.read_table(source / relative)
        _, counts = _converted_table(
            table,
            runtime_contract=runtime_contract,
            collection_contract=collection_contract,
            target_layout=target_layout,
            parquet_label=relative.as_posix(),
        )
        for key in totals:
            totals[key] += counts[key]
    declared_frames = info.get("frame_count")
    if declared_frames is not None and int(declared_frames) != totals["rows"]:
        raise ValueError("info.json frame_count disagrees with Parquet rows.")
    source_info_sha256 = "sha256:" + hashlib.sha256(info_bytes).hexdigest()
    conversion_payload = {
        "source_info_sha256": source_info_sha256,
        "source_layout_hash": source_layout.layout_hash,
        "target_layout_hash": target_layout.layout_hash,
        "pose_frame_contract_hash": collection_contract.contract_hash,
        "episode_indices": sorted(seen_episodes),
        "parquet_relative_paths": [path.as_posix() for path in relative_paths],
    }
    conversion_id = "sha256:" + hashlib.sha256(
        json.dumps(conversion_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return V22ToV23DeskPlan(
        source_root=source,
        output_root=output,
        source_layout_hash=source_layout.layout_hash,
        target_layout_hash=target_layout.layout_hash,
        pose_frame_contract_hash=collection_contract.contract_hash,
        source_info_sha256=source_info_sha256,
        conversion_id=conversion_id,
        episode_count=len(records),
        frame_count=totals["rows"],
        invalid_action_rows=totals["invalid_action_rows"],
        invalid_last_action_rows=totals["invalid_last_action_rows"],
        first_row_invalid_last_action_episodes=totals[
            "first_row_invalid_last_action"
        ],
        parquet_relative_paths=tuple(path.as_posix() for path in relative_paths),
    )


def _desk_metadata(
    payload: Mapping[str, Any],
    *,
    target_layout: PolicyIOLayout,
    collection_contract: PoseFrameContract,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    output = deepcopy(dict(payload))
    output.update(target_layout.binding_metadata())
    output.update(
        {
            "dexteleop_dataset_schema": DEXTELEOP_DATASET_SCHEMA_V23,
            POSE_FRAME_CONTRACT_METADATA_KEY: collection_contract.to_dict(),
            "pose_frame_contract_hash": collection_contract.contract_hash,
            V23_POSE_STORAGE_FRAME_METADATA_KEY: "desk_absolute",
            V23_POSE_ACTION_MODE_METADATA_KEY: "absolute",
            V23_RELATIVE_ACTION_SUPPORTED_METADATA_KEY: True,
            V23_SOURCE_SCHEMA_METADATA_KEY: DEXTELEOP_DATASET_SCHEMA_V22,
            V23_COMPATIBILITY_KIND_METADATA_KEY: V23_COMPATIBILITY_V22_CANONICAL_DESK,
            V23_NATIVE_CAUSAL_SNAPSHOT_METADATA_KEY: False,
            POSE_TRAINING_DATA_METADATA_KEY: (
                PoseTrainingDataContract.v22_canonical_desk().to_dict()
            ),
            "arm_reference_frame_by_side": {"left": "desk", "right": "desk"},
        }
    )
    if "arm_action_specs_by_side" in output:
        output["arm_action_specs_by_side"] = _desk_arm_specs(
            output["arm_action_specs_by_side"]
        )
    history = output.get("postprocess_history")
    output["postprocess_history"] = [
        *(list(history) if isinstance(history, list) else []),
        dict(receipt),
    ]
    return output


def convert_v22_dataset_to_v23_desk(
    source_root: str | Path,
    output_root: str | Path,
    *,
    target_layout_path: str | Path | None = None,
    hardlink_external: bool = True,
) -> V22ToV23DeskPlan:
    """Create a new canonical-desk compatibility dataset; never mutate source."""

    plan = plan_v22_to_v23_desk_conversion(
        source_root, output_root, target_layout_path=target_layout_path
    )
    if plan.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output: {plan.output_root}")
    active_collection = _active_collection_processes()
    if active_collection:
        raise RuntimeError(
            "Refusing bulk dataset conversion while formal collection is active: "
            + "; ".join(active_collection[:3])
        )
    info = read_json(plan.source_root / "meta/info.json")
    _, target_layout, runtime_contract, collection_contract = _contracts_and_layouts(
        info, target_layout_path
    )
    applied_at = datetime.now(timezone.utc).isoformat()
    receipt = {
        **plan.to_dict(mode="applied"),
        "applied_at_utc": applied_at,
        "pose_columns": [
            *PACKED_POSE_COLUMNS,
            *(template for template in SIDE_POSE_COLUMNS),
            "sample_metadata_json.canonical_action_record",
        ],
        "historical_audit_columns_preserved": [
            "teleop.action_source.*_wrist_pose_base",
            "teleop.action_source.*_wrist_pose_desk",
            "observation.wrist_camera.*.pose_base",
            "observation.arm.*.actual_hardware_pose_6d",
        ],
    }
    staging = plan.output_root.parent / (
        f".{plan.output_root.name}.v23-desk-staging-{uuid.uuid4().hex}"
    )
    clone_dataset_tree(
        plan.source_root,
        staging,
        copy_prefixes=("meta",),
        skip_prefixes=("data",),
        hardlink_external=bool(hardlink_external),
    )
    try:
        import pyarrow.parquet as pq

        for relative_text in plan.parquet_relative_paths:
            relative = Path(relative_text)
            source_path = plan.source_root / relative
            target_path = staging / relative
            table = pq.read_table(source_path)
            converted, _ = _converted_table(
                table,
                runtime_contract=runtime_contract,
                collection_contract=collection_contract,
                target_layout=target_layout,
                parquet_label=relative.as_posix(),
            )
            target_path.parent.mkdir(parents=True, exist_ok=True)
            compression = "zstd"
            metadata = pq.ParquetFile(source_path).metadata
            if metadata.num_row_groups and metadata.num_columns:
                compression = metadata.row_group(0).column(0).compression.lower()
            pq.write_table(converted, target_path, compression=compression)

        output_info = _desk_metadata(
            info,
            target_layout=target_layout,
            collection_contract=collection_contract,
            receipt=receipt,
        )
        output_info["dataset_name"] = plan.output_root.name
        output_info["dataset_dir_name"] = plan.output_root.name
        write_json(staging / "meta/info.json", output_info)
        converted_episodes = []
        compact_episode_receipt = {
            "schema": MIGRATION_SCHEMA,
            "operation": V23_COMPATIBILITY_V22_CANONICAL_DESK,
            "conversion_id": plan.conversion_id,
            "source_schema": DEXTELEOP_DATASET_SCHEMA_V22,
            "target_schema": DEXTELEOP_DATASET_SCHEMA_V23,
            "native_v23_causal_snapshot": False,
            "source_layout_hash": plan.source_layout_hash,
            "target_layout_hash": plan.target_layout_hash,
            "pose_frame_contract_hash": plan.pose_frame_contract_hash,
            "applied_at_utc": applied_at,
        }
        for record in episode_records(plan.source_root):
            output_record = deepcopy(record)
            metadata = output_record.get("metadata")
            output_record["metadata"] = _desk_metadata(
                metadata if isinstance(metadata, Mapping) else {},
                target_layout=target_layout,
                collection_contract=collection_contract,
                receipt=compact_episode_receipt,
            )
            converted_episodes.append(output_record)
        write_jsonl(staging / "meta/episodes.jsonl", converted_episodes)
        write_json(staging / "meta/v22_to_v23_desk_conversion.json", receipt)
        staging.rename(plan.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return plan


__all__ = [
    "DEFAULT_TARGET_LAYOUT_PATH",
    "MIGRATION_SCHEMA",
    "ROW_ACTION_FINITE_COLUMN",
    "ROW_LAST_ACTION_FINITE_COLUMN",
    "ROW_RELATIVE_LAST_ACTION_FINITE_COLUMN",
    "V22ToV23DeskPlan",
    "convert_v22_dataset_to_v23_desk",
    "plan_v22_to_v23_desk_conversion",
]

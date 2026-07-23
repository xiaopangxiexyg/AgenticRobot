"""Helpers for binding a merged native-v2.3 dataset to canonical desk storage."""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.pose_frames import PoseFrameContract, PoseSourceProfile


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(value), ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def _merged_contract(source_contracts: Mapping[str, Mapping[str, Any]]) -> PoseFrameContract:
    parsed = [PoseFrameContract.from_dict(value) for value in source_contracts.values()]
    if not parsed or any(contract.pose_storage_frame != "desk_absolute" for contract in parsed):
        raise ValueError("All merged v2.3 sources must already store desk_absolute poses")
    active_sides = parsed[0].source.active_sides
    canonical_frame = parsed[0].canonical_absolute_frame
    if any(contract.source.active_sides != active_sides for contract in parsed):
        raise ValueError("Merged pose contracts have different active sides")
    if any(contract.canonical_absolute_frame != canonical_frame for contract in parsed):
        raise ValueError("Merged pose contracts have different canonical desk frames")
    hashes = tuple(sorted(source_contracts))
    source = PoseSourceProfile(
        source_id="vive_umi_merged_workspace_near_boundary_v1",
        source_family="vive_umi",
        source_description=(
            "Merged UMI trajectories already stored in the canonical "
            "workspace-near-boundary desk frame."
        ),
        topology="dual_wrist",
        active_sides=active_sides,
        raw_absolute_frame_by_side={
            side: "workspace_near_boundary_frame" for side in active_sides
        },
        raw_to_desk_transform=np.eye(4),
        desk_frame=canonical_frame,
        metadata={
            "desk_definition": "workspace_near_boundary_frame",
            "derived_canonical_desk_storage": True,
            "source_pose_frame_contract_hashes": list(hashes),
        },
    )
    return PoseFrameContract(
        source=source,
        pose_storage_frame="desk_absolute",
        canonical_absolute_frame=canonical_frame,
        metadata={
            "merge_semantics": (
                "pose values unchanged; provenance contracts rebound to one "
                "canonical desk storage contract"
            ),
            "source_pose_frame_contract_hashes": list(hashes),
        },
    )


def rebind_merged_v23_canonical_desk_contract_in_place(
    dataset_root: str | Path,
) -> dict[str, Any]:
    """Rebind a newly merged v2.3 dataset without changing any pose values."""

    root = Path(dataset_root).expanduser().resolve()
    info_path = root / "meta/info.json"
    episodes_path = root / "meta/episodes.jsonl"
    info = _read_json(info_path)
    if info.get("dexteleop_dataset_schema") != "dexteleop_dataset_v2.3":
        raise ValueError("Expected a native DexTeleop v2.3 dataset")
    episodes = _read_jsonl(episodes_path)

    source_contracts: dict[str, dict[str, Any]] = {}
    for record in episodes:
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"episode {record.get('episode_index')} has no metadata")
        payload = metadata.get("pose_frame_contract")
        old_hash = str(metadata.get("pose_frame_contract_hash", ""))
        if not isinstance(payload, Mapping):
            raise ValueError(f"episode {record.get('episode_index')} has no pose contract")
        contract = PoseFrameContract.from_dict(payload)
        if contract.contract_hash != old_hash:
            raise ValueError(f"episode {record.get('episode_index')} pose contract hash mismatch")
        source_contracts.setdefault(old_hash, contract.to_dict())

    merged = _merged_contract(source_contracts)
    merged_payload = merged.to_dict()
    merged_hash = merged.contract_hash

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("v2.3 merge rebinding requires pyarrow") from exc

    staging = Path(tempfile.mkdtemp(prefix=".v23_merge_rebind_", dir=root.parent))
    try:
        for record in episodes:
            parquet_path = root / str(record["parquet_path"])
            table = pq.read_table(parquet_path)
            old_hash = str(record["metadata"]["pose_frame_contract_hash"])
            hash_column = "dexteleop.pose_frame_contract_hash"
            hash_index = table.schema.get_field_index(hash_column)
            if hash_index < 0:
                raise ValueError(f"{parquet_path} has no pose contract hash column")
            values = table.column(hash_index).to_pylist()
            if any(str(value) != old_hash for value in values):
                raise ValueError(f"{parquet_path} row pose contract hash mismatch")
            table = table.set_column(
                hash_index,
                hash_column,
                pa.array(
                    [merged_hash] * table.num_rows,
                    type=table.schema.field(hash_index).type,
                ),
            )
            source_column = "dexteleop.pose_source_id"
            source_index = table.schema.get_field_index(source_column)
            if source_index < 0:
                raise ValueError(f"{parquet_path} has no pose source id column")
            table = table.set_column(
                source_index,
                source_column,
                pa.array(
                    [merged.source_id] * table.num_rows,
                    type=table.schema.field(source_index).type,
                ),
            )
            output = staging / record["parquet_path"]
            output.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, output, compression="zstd")

        migrated_episodes = deepcopy(episodes)
        for record in migrated_episodes:
            metadata = record["metadata"]
            metadata["source_pose_frame_contract"] = metadata["pose_frame_contract"]
            metadata["source_pose_frame_contract_hash"] = metadata["pose_frame_contract_hash"]
            metadata["pose_frame_contract"] = merged_payload
            metadata["pose_frame_contract_hash"] = merged_hash
        migrated_info = deepcopy(info)
        migrated_info["pose_frame_contract"] = merged_payload
        migrated_info["pose_frame_contract_hash"] = merged_hash
        migrated_info["source_pose_frame_contracts_by_hash"] = source_contracts
        migrated_info.setdefault("postprocess_history", []).append(
            {
                "tool": (
                    "utils.dataset_postprocess.v23_merge."
                    "rebind_merged_v23_canonical_desk_contract_in_place"
                ),
                "pose_values_changed": False,
                "source_pose_frame_contract_hashes": sorted(source_contracts),
                "merged_pose_frame_contract_hash": merged_hash,
            }
        )
        _write_json(staging / "meta/info.json", migrated_info)
        _write_jsonl(staging / "meta/episodes.jsonl", migrated_episodes)
        _write_json(
            staging / "meta/source_pose_frame_contracts.json",
            {
                "schema": "dexteleop.merged_pose_frame_provenance.v1",
                "pose_values_changed": False,
                "merged_pose_frame_contract": merged_payload,
                "merged_pose_frame_contract_hash": merged_hash,
                "source_pose_frame_contracts_by_hash": source_contracts,
            },
        )

        for record in episodes:
            os.replace(staging / record["parquet_path"], root / record["parquet_path"])
        os.replace(staging / "meta/info.json", info_path)
        os.replace(staging / "meta/episodes.jsonl", episodes_path)
        os.replace(
            staging / "meta/source_pose_frame_contracts.json",
            root / "meta/source_pose_frame_contracts.json",
        )
    finally:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)

    return {
        "dataset_root": str(root),
        "episode_count": len(episodes),
        "pose_values_changed": False,
        "source_pose_frame_contract_hashes": sorted(source_contracts),
        "merged_pose_frame_contract_hash": merged_hash,
    }


__all__ = ["rebind_merged_v23_canonical_desk_contract_in_place"]

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dataset_contracts.pose_frames import PoseFrameContract, PoseSourceProfile
from utils.dataset_postprocess.workspace_near_boundary import (
    MIGRATION_ID,
    migrate_umi_workspace_near_boundary,
)


pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _pose(x: float) -> list[float]:
    return [x, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]


def _dataset(root: Path) -> tuple[PoseFrameContract, dict[str, object]]:
    source = PoseSourceProfile(
        source_id="vive_umi_dual_wrist_v1",
        source_family="vive_umi",
        source_description="legacy board-defined desk",
        topology="dual_wrist",
        active_sides=("left", "right"),
        raw_absolute_frame_by_side={"left": "vive_world", "right": "vive_world"},
        raw_to_desk_transform=np.eye(4),
        metadata={"desk_definition": "calibration_board_frame"},
    )
    contract = PoseFrameContract(source=source, pose_storage_frame="desk_absolute")
    left = _pose(-0.3)
    right = _pose(-0.4)
    packed = left + right + [0.1, 0.2]
    row: dict[str, object] = {
        "action": packed,
        "action.arm.pose": left + right,
        "action.left_arm.pose": left,
        "action.right_arm.pose": right,
        "observation.state": packed,
        "observation.arm.pose": left + right,
        "observation.last_action": packed,
        "observation.left_arm.pose": left,
        "observation.right_arm.pose": right,
        "observation.arm.left.actual_tcp_pose": left,
        "observation.arm.right.actual_tcp_pose": right,
        "dexteleop.pose_frame_contract_hash": contract.contract_hash,
    }
    parquet_path = root / "data/chunk-000/episode_000000.parquet"
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([row]), parquet_path, compression="zstd")
    episode = {
        "episode_index": 0,
        "parquet_path": "data/chunk-000/episode_000000.parquet",
        "metadata": {
            "pose_frame_contract": contract.to_dict(),
            "pose_frame_contract_hash": contract.contract_hash,
        },
    }
    _write_json(
        root / "meta/info.json",
        {
            "dexteleop_dataset_schema": "dexteleop_dataset_v2.3",
            "embodiment": "umi_dual_wrist",
            "pose_storage_frame": "desk_absolute",
            "pose_frame_contract": contract.to_dict(),
            "pose_frame_contract_hash": contract.contract_hash,
            "episode_count": 1,
        },
    )
    (root / "meta/episodes.jsonl").write_text(json.dumps(episode) + "\n")
    return contract, row


def test_workspace_shift_is_dry_run_until_migration_specific_gate(tmp_path: Path) -> None:
    root = tmp_path / "umi"
    original_contract, original_row = _dataset(root)
    info_before = (root / "meta/info.json").read_bytes()
    parquet_before = (root / "data/chunk-000/episode_000000.parquet").read_bytes()

    plan = migrate_umi_workspace_near_boundary(root)

    assert plan["mode"] == "dry-run"
    assert plan["migration_enabled"] is False
    assert plan["stored_pose_translation_m"] == [0.15, 0.0, 0.0]
    assert (root / "meta/info.json").read_bytes() == info_before
    assert (root / "data/chunk-000/episode_000000.parquet").read_bytes() == parquet_before

    result = migrate_umi_workspace_near_boundary(
        root,
        apply_workspace_near_boundary_shift=True,
    )

    assert result["mode"] == "applied"
    assert result["migration_enabled"] is True
    assert Path(str(result["backup_root"])).is_dir()
    info = json.loads((root / "meta/info.json").read_text())
    assert info["pose_frame_contract_hash"] != original_contract.contract_hash
    assert info["pose_frame_migrations"][-1]["migration_id"] == MIGRATION_ID
    migrated_row = pq.read_table(
        root / "data/chunk-000/episode_000000.parquet"
    ).to_pylist()[0]
    for column in (
        "action",
        "action.arm.pose",
        "observation.state",
        "observation.arm.pose",
        "observation.last_action",
    ):
        assert migrated_row[column][0] == pytest.approx(original_row[column][0] + 0.15)
        assert migrated_row[column][7] == pytest.approx(original_row[column][7] + 0.15)
        assert migrated_row[column][1:7] == original_row[column][1:7]
    assert migrated_row["action.left_arm.pose"][0] == pytest.approx(-0.15)
    assert migrated_row["action.right_arm.pose"][0] == pytest.approx(-0.25)

    with pytest.raises(RuntimeError, match="already applied"):
        migrate_umi_workspace_near_boundary(
            root,
            apply_workspace_near_boundary_shift=True,
        )

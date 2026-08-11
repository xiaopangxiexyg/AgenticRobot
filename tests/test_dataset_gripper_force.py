from __future__ import annotations

import json

import pytest

from dataset_contracts.gripper_force import GripperForceDatasetContract
from dataset_interface.gripper_force import (
    bind_gripper_force_contract,
    read_gripper_force_contract,
)


pytestmark = pytest.mark.integration


def _populate(root) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "frame_count": 1}) + "\n",
        encoding="utf-8",
    )
    (meta / "info.json").write_text(
        json.dumps({"episode_count": 1, "frame_count": 1}),
        encoding="utf-8",
    )


def test_blank_dataset_can_bind_gripper_force_profile(tmp_path) -> None:
    root = tmp_path / "dataset"
    contract = GripperForceDatasetContract(
        force_by_side=(("left", 0.31), ("right", 0.64)),
        profile_name="profile.json",
        task="firm_grasp",
    )

    bind_gripper_force_contract(root, contract)

    assert read_gripper_force_contract(root) == contract


def test_populated_dataset_rejects_new_or_changed_force_profile(tmp_path) -> None:
    root = tmp_path / "dataset"
    first = GripperForceDatasetContract(force_by_side=(("left", 0.31),))
    second = GripperForceDatasetContract(force_by_side=(("left", 0.42),))
    bind_gripper_force_contract(root, first)
    _populate(root)

    with pytest.raises(ValueError, match="new/empty dataset"):
        bind_gripper_force_contract(root, first)
    with pytest.raises(ValueError, match="new/empty dataset|different"):
        bind_gripper_force_contract(root, second)

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dataset_contracts.pose_frames import (
    CANONICAL_DESK_FRAME,
    DESK_ABSOLUTE_FRAME,
    POSE_STORAGE_FRAME_DESK,
    POSE_STORAGE_FRAME_RAW,
    PoseFrameContract,
    PoseFrameContractError,
    PoseSourceProfile,
    legacy_rm65_source_id,
    load_pose_frame_contract,
    load_pose_source_profile,
    pose_frame_contract_from_metadata,
)
from dataset_contracts.v23 import (
    DEXTELEOP_DATASET_SCHEMA_V23,
    resolve_v23_dataset_contract,
    v22_to_v23_compatibility_metadata,
)
from dataset_contracts.v22 import PolicyIOLayout


def _translation(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [x, y, z]
    return pose


def test_legacy_source_id_captures_topology() -> None:
    assert legacy_rm65_source_id(("left", "right")) == "rm65_real_dual_arm_v1"
    assert legacy_rm65_source_id(("left",)) == "rm65_real_single_left_v1"
    assert legacy_rm65_source_id(("right",)) == "rm65_real_single_right_v1"


def test_shared_source_transform_is_default_and_side_override_wins() -> None:
    source = PoseSourceProfile(
        source_id="fixture_v1",
        source_family="test",
        source_description="fixture",
        topology="dual_arm",
        raw_to_desk_transform=_translation(1.0),
        raw_to_desk_transform_by_side={"right": _translation(2.0)},
    )
    contract = PoseFrameContract(source=source, pose_storage_frame=POSE_STORAGE_FRAME_RAW)
    raw_pose = _translation(0.5)
    assert contract.storage_pose_to_desk(raw_pose, side="left")[0, 3] == pytest.approx(1.5)
    assert contract.storage_pose_to_desk(raw_pose, side="right")[0, 3] == pytest.approx(2.5)
    assert contract.desk_pose_to_storage(
        contract.storage_pose_to_desk(raw_pose, side="right"), side="right"
    ) == pytest.approx(raw_pose)


def test_per_side_raw_frames_map_into_one_canonical_desk() -> None:
    source = PoseSourceProfile(
        source_id="rm65_real_dual_arm_fixture_v1",
        source_family="real_robot",
        source_description="fixture with distinct controller base frames",
        topology="dual_arm",
        raw_absolute_frame_by_side={
            "left": "left_rm65_controller_base",
            "right": "right_rm65_controller_base",
        },
        raw_to_desk_transform_by_side={
            "left": _translation(1.0),
            "right": _translation(-2.0),
        },
    )
    contract = PoseFrameContract.v23_forced_raw(source)

    assert source.raw_frame("left") == "left_rm65_controller_base"
    assert source.raw_frame("right") == "right_rm65_controller_base"
    assert source.canonical_desk_frame == CANONICAL_DESK_FRAME
    assert contract.storage_pose_to_canonical_desk(
        _translation(0.25), side="left"
    )[0, 3] == pytest.approx(1.25)
    assert contract.storage_pose_to_canonical_desk(
        _translation(0.25), side="right"
    )[0, 3] == pytest.approx(-1.75)


def test_canonical_names_serialize_while_legacy_names_stay_compatible() -> None:
    source = PoseSourceProfile(
        source_id="fixture_v1",
        source_family="test",
        source_description="fixture",
        topology="single_arm",
        active_sides=("left",),
        raw_to_desk_transform=_translation(0.4),
    )
    payload = source.to_dict()
    assert payload["canonical_desk_frame"] == payload["desk_frame"] == "desk"
    assert payload["raw_to_canonical_desk_transform"] == payload["raw_to_desk_transform"]

    canonical_only = dict(payload)
    canonical_only.pop("desk_frame")
    canonical_only.pop("raw_to_desk_transform")
    canonical_only.pop("raw_to_desk_transform_by_side")
    restored = PoseSourceProfile.from_dict(canonical_only)
    assert restored.effective_raw_to_canonical_desk_transform("left") == pytest.approx(
        _translation(0.4)
    )

    conflicting = dict(payload)
    conflicting["raw_to_desk_transform"] = _translation(9.0).tolist()
    with pytest.raises(PoseFrameContractError, match="Conflicting pose contract aliases"):
        PoseSourceProfile.from_dict(conflicting)


def test_contract_rejects_source_canonical_frame_disagreement() -> None:
    source = PoseSourceProfile(
        source_id="fixture_v1",
        source_family="test",
        source_description="fixture",
        topology="single_arm",
        active_sides=("left",),
        desk_frame="task_table",
    )
    with pytest.raises(PoseFrameContractError, match="disagrees"):
        PoseFrameContract(source=source, canonical_absolute_frame="desk")


def test_v23_desk_storage_is_identity_at_frame_boundary() -> None:
    source = PoseSourceProfile(
        source_id="fixture_v1",
        source_family="test",
        source_description="fixture",
        topology="dual_arm",
    )
    contract = PoseFrameContract.v23_default_desk(source)
    pose = _translation(0.2, -0.1, 0.4)
    assert contract.pose_storage_frame == POSE_STORAGE_FRAME_DESK
    assert contract.canonical_absolute_frame == DESK_ABSOLUTE_FRAME
    assert contract.storage_pose_to_desk(pose, side="left") == pytest.approx(pose)


def test_legacy_v22_contract_is_raw_and_explicitly_marked() -> None:
    contract = PoseFrameContract.legacy_v22_raw(active_sides=("left", "right"))
    assert contract.pose_storage_frame == POSE_STORAGE_FRAME_RAW
    assert contract.source.source_id == "rm65_real_dual_arm_v1"
    assert contract.source.metadata["compatibility_source"] == "v2.2_or_earlier"


def test_contract_round_trip_and_legacy_metadata_fallback() -> None:
    contract = PoseFrameContract.legacy_v22_raw(active_sides=("left",))
    restored = PoseFrameContract.from_dict(contract.to_dict())
    assert restored.to_dict() == contract.to_dict()

    fallback = pose_frame_contract_from_metadata({}, active_sides=("right",))
    assert fallback.source.source_id == "rm65_real_single_right_v1"


def test_explicit_source_and_contract_json_loaders(tmp_path) -> None:
    contract = PoseFrameContract.legacy_v22_raw(active_sides=("left", "right"))
    source_path = tmp_path / "source.json"
    source_path.write_text(
        json.dumps({"pose_source_contract": contract.source.to_dict()}),
        encoding="utf-8",
    )
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps({"pose_frame_contract": contract.to_dict()}),
        encoding="utf-8",
    )
    assert load_pose_source_profile(source_path).to_dict() == contract.source.to_dict()
    assert load_pose_frame_contract(contract_path).to_dict() == contract.to_dict()


def test_explicit_contract_loader_reports_missing_file_as_contract_error(tmp_path) -> None:
    missing = tmp_path / "missing_pose_contract.json"
    with pytest.raises(PoseFrameContractError, match="Cannot read pose frame contract"):
        load_pose_frame_contract(missing)


def test_default_rm65_runtime_identity_pose_contract_is_loadable() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "robot_motion/configs/pose_frames/rm65_real_dual_arm_v1_identity.json"
    )
    contract = load_pose_frame_contract(path)
    assert contract.pose_storage_frame == POSE_STORAGE_FRAME_RAW
    assert contract.source.source_id == "rm65_real_dual_arm_v1"
    np.testing.assert_allclose(
        contract.source.effective_raw_to_canonical_desk_transform("left"),
        np.eye(4),
    )


def test_invalid_source_transform_is_rejected() -> None:
    bad = np.eye(4, dtype=float)
    bad[0, 0] = 2.0
    with pytest.raises(PoseFrameContractError, match="orthonormal"):
        PoseSourceProfile(
            source_id="bad",
            source_family="test",
            source_description="bad",
            topology="single_arm",
            active_sides=("left",),
            raw_to_desk_transform=bad,
        )


def test_v22_metadata_resolves_to_marked_raw_v23_compatibility_view() -> None:
    layout = PolicyIOLayout()
    metadata = {
        "dexteleop_dataset_schema": "dexteleop_dataset_v2.2",
        "active_sides": ["left", "right"],
        **layout.binding_metadata(),
    }
    contract = resolve_v23_dataset_contract(metadata)
    assert contract.pose_frames.source.source_id == "rm65_real_dual_arm_v1"
    assert contract.compatibility_mode is True
    assert contract.pose_storage_frame == POSE_STORAGE_FRAME_RAW


def test_v23_requires_explicit_pose_contract_and_compatibility_patch_is_copy_only() -> None:
    with pytest.raises(PoseFrameContractError, match="pose_frame_contract"):
        resolve_v23_dataset_contract(
            {"dexteleop_dataset_schema": DEXTELEOP_DATASET_SCHEMA_V23},
        )

    layout = PolicyIOLayout()
    legacy = {
        "dexteleop_dataset_schema": "dexteleop_dataset_v2.2",
        **layout.binding_metadata(),
    }
    legacy_before = dict(legacy)
    patched = v22_to_v23_compatibility_metadata(legacy)
    assert legacy == legacy_before
    assert patched["dexteleop_dataset_schema"] == DEXTELEOP_DATASET_SCHEMA_V23
    assert patched["pose_storage_frame"] == POSE_STORAGE_FRAME_RAW


def test_v23_rejects_conflicting_top_level_storage_frame() -> None:
    source = PoseSourceProfile(
        source_id="fixture_v23",
        source_family="test",
        source_description="fixture",
        topology="dual_arm",
    )
    contract = PoseFrameContract.v23_default_desk(source)
    layout = PolicyIOLayout()
    metadata = {
        "dexteleop_dataset_schema": DEXTELEOP_DATASET_SCHEMA_V23,
        **layout.binding_metadata(),
        "pose_frame_contract": contract.to_dict(),
        "pose_storage_frame": POSE_STORAGE_FRAME_RAW,
    }
    with pytest.raises(PoseFrameContractError, match="disagrees"):
        resolve_v23_dataset_contract(metadata)

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from dataset_contracts.pose_frames import (
    POSE_STORAGE_FRAME_DESK,
    POSE_STORAGE_FRAME_RAW,
    load_pose_frame_contract,
    load_pose_source_profile,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
POSE_CONFIG_DIR = REPO_ROOT / "robot_motion" / "configs" / "pose_frames"
SOURCE_PATH = POSE_CONFIG_DIR / "rm65_dual_arm_table_center_v1_source.json"
COLLECTION_CONTRACT_PATH = (
    POSE_CONFIG_DIR
    / "rm65_dual_arm_table_center_v1_collection_desk_absolute.json"
)
RUNTIME_CONTRACT_PATH = (
    POSE_CONFIG_DIR / "rm65_dual_arm_table_center_v1_runtime_raw_absolute.json"
)
REFERENCE_URDF_PATH = (
    REPO_ROOT
    / "assets"
    / "rm65_inspire_gripper_v3"
    / "rm65_dual_arm_wrist_cameras_v3_head_camera.urdf"
)


def _origin_matrix(element: ET.Element) -> np.ndarray:
    origin = element.find("origin")
    xyz = np.zeros(3, dtype=float)
    rpy = np.zeros(3, dtype=float)
    if origin is not None:
        xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    pose[:3, 3] = xyz
    return pose


def _joint(root: ET.Element, name: str) -> ET.Element:
    joint = root.find(f"joint[@name='{name}']")
    assert joint is not None, name
    assert joint.get("type") == "fixed", name
    return joint


def test_table_center_collection_and_runtime_contracts_share_one_source_profile():
    source = load_pose_source_profile(SOURCE_PATH)
    collection = load_pose_frame_contract(COLLECTION_CONTRACT_PATH)
    runtime = load_pose_frame_contract(RUNTIME_CONTRACT_PATH)

    assert collection.pose_storage_frame == POSE_STORAGE_FRAME_DESK
    assert runtime.pose_storage_frame == POSE_STORAGE_FRAME_RAW
    assert collection.source.to_dict() == source.to_dict()
    assert runtime.source.to_dict() == source.to_dict()
    assert source.canonical_desk_frame == "desk"
    assert source.metadata["canonical_frame_urdf_link"] == "table_center_frame"
    assert source.metadata["hardware_measurement_status"] == "not_measured"


def test_table_center_source_extrinsics_are_derived_from_reference_urdf():
    source = load_pose_source_profile(SOURCE_PATH)
    root = ET.parse(REFERENCE_URDF_PATH).getroot()

    table_joint = _joint(root, "table_joint")
    table_center_joint = _joint(root, "table_to_table_center_frame")
    assert table_center_joint.find("parent").get("link") == "table"
    assert table_center_joint.find("child").get("link") == "table_center_frame"

    world_from_desk = _origin_matrix(table_joint) @ _origin_matrix(table_center_joint)
    np.testing.assert_allclose(world_from_desk[:3, 3], [-0.412, 0.0, 0.01])
    np.testing.assert_allclose(world_from_desk[:3, :3], np.eye(3), atol=1e-12)

    for side in ("left", "right"):
        world_from_base = _origin_matrix(_joint(root, f"{side}_arm"))
        expected_desk_from_base = np.linalg.inv(world_from_desk) @ world_from_base
        np.testing.assert_allclose(
            source.effective_raw_to_canonical_desk_transform(side),
            expected_desk_from_base,
            atol=1e-12,
        )


def test_runtime_raw_and_collection_desk_contracts_have_identical_pose_mapping():
    collection = load_pose_frame_contract(COLLECTION_CONTRACT_PATH)
    runtime = load_pose_frame_contract(RUNTIME_CONTRACT_PATH)
    raw_pose = np.eye(4, dtype=float)
    raw_pose[:3, 3] = [0.25, -0.1, 0.4]

    for side in ("left", "right"):
        collection_desk = (
            collection.source.effective_raw_to_canonical_desk_transform(side)
            @ raw_pose
        )
        runtime_desk = runtime.storage_pose_to_canonical_desk(raw_pose, side=side)
        np.testing.assert_allclose(runtime_desk, collection_desk, atol=1e-12)
        np.testing.assert_allclose(
            runtime.canonical_desk_pose_to_storage(runtime_desk, side=side),
            raw_pose,
            atol=1e-12,
        )

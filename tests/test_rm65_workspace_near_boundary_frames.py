from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = REPO_ROOT / "assets"
EXPECTED_WORKSPACE_NEAR_BOUNDARY_WORLD_XYZ = np.array([-0.472, 0.0, 0.01])
EXPECTED_WORKSPACE_NEAR_BOUNDARY_RPY = np.array([0.0, 0.0, np.pi])
EXPECTED_CAMERA_POLE_ORIGINS = {
    "camera_pole_joint": np.array([0.025, 0.0, 0.5]),
    "left_camera_pole_joint": np.array([0.025, -0.478, 0.5]),
    "right_camera_pole_joint": np.array([0.025, 0.439, 0.5]),
}
REQUIRED_ACTIVE_STATION_URDF = (
    ASSET_ROOT
    / "rm65_inspire_gripper_v3"
    / "rm65_dual_arm_wrist_cameras_v3_head_camera_4b2_cad_with_legacy_housing.urdf"
)


def _active_rm65_table_urdfs() -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(ASSET_ROOT.rglob("*.urdf"))
        if "backup" not in path.parts
        and not path.name.endswith("_temp_visualization.urdf")
        and "rm65" in path.name
        and ET.parse(path).getroot().find("link[@name='table']") is not None
    )


def _active_rm65_camera_pole_urdfs() -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(ASSET_ROOT.rglob("*.urdf"))
        if "backup" not in path.parts
        and "rm65" in path.name
        and ET.parse(path).getroot().find("joint[@name='camera_pole_joint']")
        is not None
    )


def _xyz(element: ET.Element) -> np.ndarray:
    origin = element.find("origin")
    assert origin is not None
    return np.fromstring(origin.attrib["xyz"], sep=" ")


def _rpy(element: ET.Element) -> np.ndarray:
    origin = element.find("origin")
    assert origin is not None
    return np.fromstring(origin.attrib["rpy"], sep=" ")


def test_all_active_rm65_table_urdfs_define_geometry_free_workspace_near_boundary_frame():
    urdf_paths = _active_rm65_table_urdfs()
    assert REQUIRED_ACTIVE_STATION_URDF in urdf_paths

    for urdf_path in urdf_paths:
        root = ET.parse(urdf_path).getroot()
        frame = root.find("link[@name='workspace_near_boundary_frame']")
        assert frame is not None, urdf_path
        assert len(frame) == 0, urdf_path

        joint = root.find("joint[@name='table_to_workspace_near_boundary_frame']")
        assert joint is not None, urdf_path
        assert joint.attrib["type"] == "fixed", urdf_path
        assert joint.find("parent").attrib["link"] == "table", urdf_path
        assert joint.find("child").attrib["link"] == "workspace_near_boundary_frame", urdf_path
        np.testing.assert_allclose(_xyz(joint), [0.0, 0.0, 0.035])
        np.testing.assert_allclose(_rpy(joint), EXPECTED_WORKSPACE_NEAR_BOUNDARY_RPY)

        legacy_frame = root.find("link[@name='table_center_frame']")
        assert legacy_frame is not None, urdf_path
        legacy_joint = root.find(
            "joint[@name='workspace_near_boundary_to_legacy_table_center_frame']"
        )
        assert legacy_joint is not None, urdf_path
        assert legacy_joint.find("parent").attrib["link"] == "workspace_near_boundary_frame"
        assert legacy_joint.find("child").attrib["link"] == "table_center_frame"
        np.testing.assert_allclose(_xyz(legacy_joint), [0.0, 0.0, 0.0])
        np.testing.assert_allclose(_rpy(legacy_joint), [0.0, 0.0, 0.0])

        table_joint = root.find("joint[@name='table_joint']")
        assert table_joint is not None, urdf_path
        np.testing.assert_allclose(
            _xyz(table_joint) + _xyz(joint),
            EXPECTED_WORKSPACE_NEAR_BOUNDARY_WORLD_XYZ,
            err_msg=str(urdf_path),
        )


def test_v1_tables_match_v3_station_geometry():
    v1_paths = (
        ASSET_ROOT / "rm65_inspire_gripper_v1" / "rm65_dual_arm_gripper_v1.urdf",
        ASSET_ROOT
        / "rm65_inspire_gripper_v1"
        / "rm65_dual_arm_gripper_v1_head_camera.urdf",
    )

    for urdf_path in v1_paths:
        root = ET.parse(urdf_path).getroot()
        table = root.find("link[@name='table']")
        assert table is not None
        visual_box = table.find("visual/geometry/box")
        collision_box = table.find("collision/geometry/box")
        assert visual_box is not None
        assert collision_box is not None
        assert visual_box.attrib["size"] == "0.824 1.3 0.05"
        assert collision_box.attrib["size"] == "0.824 1.3 0.05"

        table_joint = root.find("joint[@name='table_joint']")
        assert table_joint is not None
        np.testing.assert_allclose(_xyz(table_joint), [-0.472, 0.0, -0.025])


def test_workspace_near_boundary_visualization_urdf_uses_the_same_desk_axes():
    urdf_path = (
        ASSET_ROOT
        / "rm65_inspire_gripper_v3"
        / "rm65_dual_arm_wrist_cameras_v3_head_camera_workspace_near_boundary_temp_visualization.urdf"
    )
    root = ET.parse(urdf_path).getroot()
    joint = root.find("joint[@name='table_to_workspace_near_boundary_frame']")
    assert joint is not None
    np.testing.assert_allclose(_rpy(joint), EXPECTED_WORKSPACE_NEAR_BOUNDARY_RPY)
    table_joint = root.find("joint[@name='table_joint']")
    assert table_joint is not None
    np.testing.assert_allclose(_xyz(table_joint), [-0.472, 0.0, -0.025])


def test_all_active_rm65_camera_poles_use_the_shifted_station_origins():
    urdf_paths = _active_rm65_camera_pole_urdfs()
    assert REQUIRED_ACTIVE_STATION_URDF in urdf_paths

    for urdf_path in urdf_paths:
        root = ET.parse(urdf_path).getroot()
        for joint_name, expected_xyz in EXPECTED_CAMERA_POLE_ORIGINS.items():
            joint = root.find(f"joint[@name='{joint_name}']")
            assert joint is not None, (urdf_path, joint_name)
            np.testing.assert_allclose(_xyz(joint), expected_xyz, err_msg=str(urdf_path))

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = REPO_ROOT / "assets"
EXPECTED_TABLE_CENTER_WORLD_XYZ = np.array([-0.412, 0.0, 0.01])


def _active_rm65_table_urdfs() -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(ASSET_ROOT.rglob("*.urdf"))
        if "backup" not in path.parts
        and not path.name.endswith("_temp_visualization.urdf")
        and "rm65" in path.name
        and ET.parse(path).getroot().find("link[@name='table']") is not None
    )


def _xyz(element: ET.Element) -> np.ndarray:
    origin = element.find("origin")
    assert origin is not None
    return np.fromstring(origin.attrib["xyz"], sep=" ")


def test_all_active_rm65_table_urdfs_define_geometry_free_table_center_frame():
    urdf_paths = _active_rm65_table_urdfs()
    assert len(urdf_paths) == 12

    for urdf_path in urdf_paths:
        root = ET.parse(urdf_path).getroot()
        frame = root.find("link[@name='table_center_frame']")
        assert frame is not None, urdf_path
        assert len(frame) == 0, urdf_path

        joint = root.find("joint[@name='table_to_table_center_frame']")
        assert joint is not None, urdf_path
        assert joint.attrib["type"] == "fixed", urdf_path
        assert joint.find("parent").attrib["link"] == "table", urdf_path
        assert joint.find("child").attrib["link"] == "table_center_frame", urdf_path
        np.testing.assert_allclose(_xyz(joint), [0.0, 0.0, 0.035])

        table_joint = root.find("joint[@name='table_joint']")
        assert table_joint is not None, urdf_path
        np.testing.assert_allclose(
            _xyz(table_joint) + _xyz(joint),
            EXPECTED_TABLE_CENTER_WORLD_XYZ,
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
        np.testing.assert_allclose(_xyz(table_joint), [-0.412, 0.0, -0.025])

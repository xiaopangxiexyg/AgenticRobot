from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from sim.mujoco_real2sim.mjcf import convert_urdf_to_mjcf_xml
from sim.mujoco_real2sim.scene import MujocoScene
from sim.mujoco_real2sim.urdf import discover_camera_mounts, parse_urdf


URDF = Path("assets/rm65_inspire_gripper_v1/rm65_dual_arm_gripper_v1_head_camera.urdf")


def test_discovers_registry_cameras_from_explicit_urdf() -> None:
    model = parse_urdf(URDF)

    cameras = discover_camera_mounts(model)

    assert [(camera.name, camera.link_name) for camera in cameras] == [
        ("front", "head_camera_frame"),
        ("left_wrist", "left_camera_rgb"),
        ("right_wrist", "right_camera_rgb"),
    ]


def test_mjcf_conversion_promotes_camera_links_to_mujoco_cameras() -> None:
    model = parse_urdf(URDF)
    cameras = discover_camera_mounts(model)

    xml_text = convert_urdf_to_mjcf_xml(model, camera_mounts=cameras)
    root = ET.fromstring(xml_text)
    camera_names = {element.attrib["name"] for element in root.findall(".//camera")}
    joint_names = {element.attrib["name"] for element in root.findall(".//joint")}

    assert {"front", "left_wrist", "right_wrist"} <= camera_names
    assert {"left_arm_joint1", "right_arm_joint6"} <= joint_names


def test_mujoco_scene_loads_arms_and_cameras() -> None:
    pytest.importorskip("mujoco")

    scene = MujocoScene.from_urdf(URDF, camera_names=("all",))

    assert set(scene.arm_specs) == {"left", "right"}
    assert set(scene.camera_names) == {"front", "left_wrist", "right_wrist"}
    assert scene.get_arm_qpos("right").shape == (6,)


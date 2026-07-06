from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from sim.mujoco_real2sim.backend import MuJoCoGripperBackend, MuJoCoGripperBackendConfig
from sim.mujoco_real2sim.config import GripperSpec
from sim.mujoco_real2sim.controllers import GripperController
from sim.mujoco_real2sim.mjcf import convert_urdf_to_mjcf_xml
from sim.mujoco_real2sim.scene import MujocoScene
from sim.mujoco_real2sim.urdf import discover_camera_mounts, parse_urdf
from teleop_runtime.schemas import EffectorCommand


URDF = Path("assets/rm65_inspire_gripper_v2/rm65_dual_arm_wrist_cameras_v2_head_camera.urdf")


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


def test_mjcf_conversion_preserves_prismatic_joint_limits(tmp_path: Path) -> None:
    urdf_path = tmp_path / "slider.urdf"
    urdf_path.write_text(
        """\
<robot name="slider">
  <link name="base"/>
  <link name="finger"/>
  <joint name="finger_slide" type="prismatic">
    <parent link="base"/>
    <child link="finger"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.035" upper="0" effort="20" velocity="0.07"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )

    xml_text = convert_urdf_to_mjcf_xml(parse_urdf(urdf_path), camera_mounts=())

    root = ET.fromstring(xml_text)
    joint = root.find(".//joint[@name='finger_slide']")
    assert joint is not None
    assert joint.attrib["type"] == "slide"
    assert joint.attrib["range"] == "-0.035 0"
    assert joint.attrib["limited"] == "true"


def test_mujoco_scene_loads_arms_and_cameras() -> None:
    pytest.importorskip("mujoco")

    scene = MujocoScene.from_urdf(URDF, camera_names=("all",))

    assert set(scene.arm_specs) == {"left", "right"}
    assert set(scene.camera_names) == {"front", "left_wrist", "right_wrist"}
    assert scene.get_arm_qpos("right").shape == (6,)


def test_gripper_spec_maps_normalized_close_to_joint_qpos() -> None:
    spec = GripperSpec(
        name="parallel",
        joint_names=("left", "right"),
        open_qpos=(0.0, 0.0),
        closed_qpos=(-0.035, -0.035),
    )

    assert spec.qpos_from_close(0.0).tolist() == [0.0, 0.0]
    assert spec.qpos_from_close(1.0).tolist() == [-0.035, -0.035]
    assert spec.close_from_qpos((-0.0175, -0.0175)) == pytest.approx(0.5)


def test_mujoco_scene_infers_and_controls_parallel_gripper(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    urdf_path = tmp_path / "parallel_gripper.urdf"
    urdf_path.write_text(
        """\
<robot name="parallel_gripper">
  <link name="base"/>
  <link name="left_finger">
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
    </inertial>
  </link>
  <link name="right_finger">
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
    </inertial>
  </link>
  <joint name="left_finger_joint" type="prismatic">
    <parent link="base"/>
    <child link="left_finger"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.035" upper="0" effort="20" velocity="0.07"/>
  </joint>
  <joint name="right_finger_joint" type="prismatic">
    <parent link="base"/>
    <child link="right_finger"/>
    <axis xyz="-1 0 0"/>
    <limit lower="-0.035" upper="0" effort="20" velocity="0.07"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )

    scene = MujocoScene.from_urdf(urdf_path, camera_names=())

    assert set(scene.gripper_specs) == {"parallel_gripper"}
    assert scene.gripper_close_by_name() == {"parallel_gripper": pytest.approx(0.0)}

    scene.set_gripper_close(None, 1.0)
    assert scene.get_gripper_qpos("parallel_gripper").tolist() == pytest.approx([-0.035, -0.035])
    assert scene.get_gripper_close("parallel_gripper") == pytest.approx(1.0)

    result = GripperController(scene).apply({"parallel_gripper": 0.25})
    assert result["parallel_gripper"] == pytest.approx(0.25)
    assert scene.get_gripper_qpos("parallel_gripper").tolist() == pytest.approx(
        [-0.00875, -0.00875]
    )


def test_mujoco_gripper_backend_accepts_scalar_and_hand_vector(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    urdf_path = tmp_path / "parallel_gripper.urdf"
    urdf_path.write_text(
        """\
<robot name="parallel_gripper">
  <link name="base"/>
  <link name="left_finger"/>
  <link name="right_finger"/>
  <joint name="left_finger_joint" type="prismatic">
    <parent link="base"/>
    <child link="left_finger"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.035" upper="0" effort="20" velocity="0.07"/>
  </joint>
  <joint name="right_finger_joint" type="prismatic">
    <parent link="base"/>
    <child link="right_finger"/>
    <axis xyz="-1 0 0"/>
    <limit lower="-0.035" upper="0" effort="20" velocity="0.07"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    scene = MujocoScene.from_urdf(urdf_path, camera_names=())
    backend = MuJoCoGripperBackend(
        scene,
        MuJoCoGripperBackendConfig(
            gripper_name="parallel_gripper",
            effector_name="right_hand",
            command_dim=1,
        ),
    )

    backend.send_command(
        EffectorCommand(
            effector_name="right_hand",
            side="right",
            command_type="gripper_scalar",
            value=1.0,
        )
    )
    assert scene.get_gripper_close("parallel_gripper") == pytest.approx(1.0)

    backend.send_command(
        EffectorCommand(
            effector_name="right_hand",
            side="right",
            command_type="joint_vector",
            value=[0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        )
    )
    assert scene.get_gripper_close("parallel_gripper") == pytest.approx(0.5)

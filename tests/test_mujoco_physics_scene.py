from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from sim.contracts import FreeBodyObjectSpec, GroundSpec, SceneSpec, SimulationContract
from sim.mujoco.mjcf import convert_urdf_to_mjcf_xml
from sim.mujoco.scene import MujocoScene
from sim.mujoco.urdf import parse_urdf


CONTRACT = Path("sim/configs/rm65_bimanual_physics_v1.json")


def _isolated_table_world(tmp_path: Path) -> Path:
    path = tmp_path / "isolated_table_world.urdf"
    path.write_text(
        """\
<robot name="isolated_table_world">
  <link name="world"/>
  <link name="table">
    <visual><origin xyz="0 0 0.01"/><geometry><box size="0.6 0.6 0.05"/></geometry></visual>
    <collision><origin xyz="0 0 0.01"/><geometry><box size="0.6 0.6 0.05"/></geometry></collision>
  </link>
  <joint name="table_joint" type="fixed">
    <parent link="world"/><child link="table"/><origin xyz="0 0 -0.025"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    return path


def _drop_scene(urdf_path: Path) -> SceneSpec:
    return SceneSpec(
        profile_name="isolated_cube_drop_v1",
        urdf_path=str(urdf_path),
        camera_registry_path="unused.json",
        gravity_mps2=(0.0, 0.0, -9.81),
        ground=GroundSpec(enabled=True, z_m=-0.05),
        objects=(
            FreeBodyObjectSpec(
                name="drop_cube",
                size_m=(0.04, 0.04, 0.04),
                initial_position_m=(0.0, 0.0, 0.12),
                mass_kg=0.05,
            ),
        ),
    )


def test_scene_composition_adds_ground_and_explicit_inertial_free_body(
    tmp_path: Path,
) -> None:
    urdf_path = _isolated_table_world(tmp_path)
    root = ET.fromstring(
        convert_urdf_to_mjcf_xml(
            parse_urdf(urdf_path),
            camera_mounts=(),
            timestep=0.001,
            scene_spec=_drop_scene(urdf_path),
        )
    )

    option = root.find("./option")
    body = root.find(".//body[@name='drop_cube']")
    assert option is not None and option.attrib["gravity"] == "0 0 -9.81"
    assert root.find(".//geom[@name='ground_visual']") is not None
    assert root.find(".//geom[@name='ground_collision']") is not None
    assert root.find(".//geom[@name='table_collision_0']") is not None
    assert body is not None
    assert body.find("./freejoint[@name='drop_cube_freejoint']") is not None
    assert body.find("./inertial") is not None
    assert body.find("./geom[@name='drop_cube_visual']").attrib["contype"] == "0"
    assert body.find("./geom[@name='drop_cube_collision']").attrib["contype"] == "8"


def test_isolated_free_body_falls_and_settles_reproducibly(tmp_path: Path) -> None:
    mujoco = pytest.importorskip("mujoco")
    urdf_path = _isolated_table_world(tmp_path)
    xml_text = convert_urdf_to_mjcf_xml(
        parse_urdf(urdf_path),
        camera_mounts=(),
        timestep=0.001,
        scene_spec=_drop_scene(urdf_path),
    )

    def rollout() -> tuple[np.ndarray, np.ndarray, int]:
        model = mujoco.MjModel.from_xml_string(xml_text)
        data = mujoco.MjData(model)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drop_cube")
        for _ in range(2000):
            mujoco.mj_step(model, data)
        return data.xpos[body_id].copy(), data.qvel.copy(), int(data.ncon)

    position_a, qvel_a, contacts_a = rollout()
    position_b, qvel_b, contacts_b = rollout()

    np.testing.assert_allclose(position_a, position_b, atol=1e-12)
    np.testing.assert_allclose(qvel_a, qvel_b, atol=1e-12)
    # The table top is world z=0.01 and the cube half-height is 0.02.
    assert position_a[2] == pytest.approx(0.03, abs=0.004)
    assert np.linalg.norm(qvel_a) < 0.05
    assert contacts_a >= 1 and contacts_b >= 1


def test_rm65_physics_profile_has_one_urdf_table_and_no_initial_contact() -> None:
    mujoco = pytest.importorskip("mujoco")
    contract = SimulationContract.load(CONTRACT)
    urdf_path = Path(contract.scene.urdf_path)
    if not urdf_path.exists():
        pytest.skip("Local RM65 assets are not available.")
    xml_text = convert_urdf_to_mjcf_xml(
        parse_urdf(urdf_path),
        camera_mounts=(),
        timestep=contract.timing.physics_timestep_sec,
        scene_spec=contract.scene,
    )
    root = ET.fromstring(xml_text)

    assert len(root.findall(".//geom[@name='table_collision_0']")) == 1
    assert root.find(".//geom[@name='table_collision_0']").attrib["contype"] == "1"
    assert root.find(".//geom[@name='left_arm_link3_collision_0']").attrib["contype"] == "2"
    assert root.find(".//geom[@name='left_gripper_left_finger_collision_0']").attrib[
        "contype"
    ] == "4"
    assert root.find(".//geom[@name='debug_cube_collision']").attrib["contype"] == "8"
    excludes = {
        tuple(sorted((item.attrib["body1"], item.attrib["body2"])))
        for item in root.findall("./contact/exclude")
    }
    assert ("left_arm_link2", "left_arm_link3") in excludes
    assert ("left_arm_link3", "right_arm_link3") not in excludes
    model = mujoco.MjModel.from_xml_string(xml_text)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon == 0
    assert model.opt.gravity.tolist() == [0.0, 0.0, -9.81]

    scene = MujocoScene.from_contract(contract)
    assert scene.conversion.scene_profile_name == contract.scene.profile_name
    assert scene.model.opt.timestep == pytest.approx(contract.timing.physics_timestep_sec)
    assert set(scene.camera_names) == set(contract.observation.image_keys)

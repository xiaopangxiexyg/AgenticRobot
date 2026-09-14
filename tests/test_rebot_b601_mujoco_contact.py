from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from robot_motion.rebot_b601_rs import (
    B601_RS_END_EFFECTOR_FRAME,
    B601_RS_URDF_RELATIVE_PATH,
    B601_RS_WORLD_ALIGNED_EE_WRENCH_FRAME,
)
from sim.mujoco.mjcf import MJCFConversion
from sim.mujoco.rebot_b601_rs import (
    B601_RS_CONTACT_SCENE_PROFILE_NAME,
    B601_RS_CONTACT_SENSOR_NAME,
    b601_contact_scene_spec,
    build_b601_mjcf_xml,
    build_b601_mujoco_scene,
    read_b601_contact_observation,
)
from sim.mujoco.scene import MujocoScene, SceneLoadResult

REPO_ROOT = Path(__file__).resolve().parents[1]
REBOT_SOURCE_ROOT = REPO_ROOT / "hardware" / "reBotArm_control_py"
pytestmark = [pytest.mark.optional, pytest.mark.contract]


def _require_b601_assets() -> None:
    if not (REBOT_SOURCE_ROOT / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("initialize the pinned reBotArm_control_py gitlink")


def _synthetic_contact_scene(
    *,
    contact: bool = True,
    include_sensor: bool = True,
) -> MujocoScene:
    mujoco = pytest.importorskip("mujoco")
    obstacle_x_m = 0.035 if contact else 0.1
    sensor_xml = (
        f"""
  <sensor>
    <contact name="{B601_RS_CONTACT_SENSOR_NAME}"
             subtree2="{B601_RS_END_EFFECTOR_FRAME}"
             num="1" data="found force torque pos" reduce="netforce"/>
  </sensor>"""
        if include_sensor
        else ""
    )
    xml = f"""
<mujoco model="b601_contact_contract_test">
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="obstacle" pos="{obstacle_x_m} 0.01 0.1">
      <geom name="obstacle_geom" type="box" size="0.02 0.02 0.02"/>
    </body>
    <body name="{B601_RS_END_EFFECTOR_FRAME}" pos="0 0 0.1">
      <freejoint name="gripper_free"/>
      <geom name="tool_geom" type="box" size="0.02 0.02 0.02" mass="1"/>
    </body>
  </worldbody>
{sensor_xml}
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    conversion = MJCFConversion(
        urdf_path=Path("/synthetic/b601_contact_contract_test.urdf"),
        mjcf_path=Path("/synthetic/b601_contact_contract_test.xml"),
        camera_mounts=(),
        joint_names=("gripper_free",),
        body_names=("obstacle", B601_RS_END_EFFECTOR_FRAME),
        scene_profile_name="synthetic_b601_contact_contract_test",
    )
    return MujocoScene(
        mujoco=mujoco,
        model=model,
        data=mujoco.MjData(model),
        load_result=SceneLoadResult(
            conversion=conversion,
            cameras=(),
            arms={},
            grippers={},
        ),
    )


def test_contact_scene_spec_is_explicit_and_reuses_generic_scene_contract() -> None:
    spec = b601_contact_scene_spec()

    assert spec.profile_name == B601_RS_CONTACT_SCENE_PROFILE_NAME
    assert spec.backend_role == "b601_contact_observation_diagnostic"
    assert spec.gravity_mps2 == (0.0, 0.0, -9.81)
    assert spec.collision.enabled
    assert spec.collision.self_collision_enabled
    assert spec.objects == ()


def test_contact_observation_requires_a_collision_enabled_matching_scene() -> None:
    missing_source = Path("/does/not/need/to/exist")
    with pytest.raises(ValueError, match="collision-enabled scene_spec"):
        build_b601_mjcf_xml(
            missing_source,
            enable_contact_observation=True,
        )
    spec = b601_contact_scene_spec()
    with pytest.raises(ValueError, match="exactly match"):
        build_b601_mjcf_xml(
            missing_source,
            gravity_mps2=(0.0, 0.0, 0.0),
            scene_spec=spec,
            enable_contact_observation=True,
        )
    with pytest.raises(TypeError, match="must be a bool"):
        build_b601_mjcf_xml(
            missing_source,
            gravity_mps2=spec.gravity_mps2,
            scene_spec=spec,
            enable_contact_observation=1,  # type: ignore[arg-type]
        )


def test_opt_in_b601_mjcf_contains_native_net_contact_sensor() -> None:
    pytest.importorskip("mujoco")
    _require_b601_assets()
    spec = b601_contact_scene_spec()
    root = ET.fromstring(
        build_b601_mjcf_xml(
            REBOT_SOURCE_ROOT,
            gravity_mps2=spec.gravity_mps2,
            actuation_mode="effort",
            scene_spec=spec,
            enable_contact_observation=True,
        )
    )

    sensor = root.find(f"./sensor/contact[@name='{B601_RS_CONTACT_SENSOR_NAME}']")
    assert sensor is not None
    assert sensor.attrib == {
        "name": B601_RS_CONTACT_SENSOR_NAME,
        "subtree2": B601_RS_END_EFFECTOR_FRAME,
        "num": "1",
        "data": "found force torque pos",
        "reduce": "netforce",
    }
    assert root.find("./worldbody/geom[@name='ground_collision']") is not None
    assert root.findall(".//geom[@group='3']")


def test_compiled_b601_contact_scene_preserves_scene_metadata() -> None:
    pytest.importorskip("mujoco")
    _require_b601_assets()
    spec = b601_contact_scene_spec()
    scene = build_b601_mujoco_scene(
        REBOT_SOURCE_ROOT,
        gravity_mps2=spec.gravity_mps2,
        actuation_mode="effort",
        scene_spec=spec,
        enable_contact_observation=True,
    )

    assert scene.scene_spec is spec
    assert scene.conversion.scene_profile_name == spec.profile_name
    assert scene.model.nsensor == 1
    observation = read_b601_contact_observation(scene)
    assert observation.wrench.frame_id == B601_RS_WORLD_ALIGNED_EE_WRENCH_FRAME
    assert not observation.force_calibrated


def test_no_contact_returns_exact_zero_without_a_fake_application_point() -> None:
    observation = read_b601_contact_observation(
        _synthetic_contact_scene(contact=False)
    )

    assert not observation.in_contact
    assert observation.physics_time_sec == 0.0
    assert observation.matched_contact_count == 0
    assert observation.application_point_world_m is None
    np.testing.assert_array_equal(observation.wrench.as_vector(), np.zeros(6))


def test_contact_force_acts_on_subtree2_and_moment_is_shifted_to_ee_origin() -> None:
    scene = _synthetic_contact_scene(contact=True)
    sensor_id = scene.mujoco.mj_name2id(
        scene.model,
        scene.mujoco.mjtObj.mjOBJ_SENSOR,
        B601_RS_CONTACT_SENSOR_NAME,
    )
    address = int(scene.model.sensor_adr[sensor_id])
    raw = np.asarray(scene.data.sensordata[address : address + 10], dtype=float)
    observation = read_b601_contact_observation(scene)

    assert observation.in_contact
    assert observation.physics_time_sec == pytest.approx(scene.data.time)
    assert observation.matched_contact_count == int(raw[0]) == 4
    # The obstacle is on the +X side; subtree2 receives the reaction toward -X.
    assert observation.wrench.force_n[0] < 0.0
    np.testing.assert_allclose(observation.wrench.force_n, raw[1:4])
    ee_position = scene.body_pose(B601_RS_END_EFFECTOR_FRAME)[:3, 3]
    expected_torque = raw[4:7] + np.cross(raw[7:10] - ee_position, raw[1:4])
    np.testing.assert_allclose(observation.wrench.torque_nm, expected_torque)
    np.testing.assert_allclose(observation.application_point_world_m, raw[7:10])


def test_contact_observer_fails_closed_on_missing_or_nonfinite_sensor_data() -> None:
    with pytest.raises(ValueError, match="has no contact wrench sensor"):
        read_b601_contact_observation(
            _synthetic_contact_scene(contact=False, include_sensor=False)
        )

    scene = _synthetic_contact_scene(contact=False)
    sensor_id = scene.mujoco.mj_name2id(
        scene.model,
        scene.mujoco.mjtObj.mjOBJ_SENSOR,
        B601_RS_CONTACT_SENSOR_NAME,
    )
    address = int(scene.model.sensor_adr[sensor_id])
    scene.data.sensordata[address + 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        read_b601_contact_observation(scene)

    scene_without_sensor = _synthetic_contact_scene(contact=False)
    scene_without_sensor.model.sensor_dim[0] = 9
    with pytest.raises(ValueError, match="dimension mismatch"):
        read_b601_contact_observation(scene_without_sensor)

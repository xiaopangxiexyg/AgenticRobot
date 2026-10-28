from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sim.contracts import FreeBodyObjectSpec, GroundSpec, SceneSpec, SimulationContract
from sim.contracts.scene import SCENE_SPEC_VERSION_V2
from sim.mujoco.actuation import dmtac_gripper_profile_with_object_contacts
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


def test_visual_color_override_does_not_change_physics(tmp_path: Path) -> None:
    model = parse_urdf(_isolated_table_world(tmp_path))
    spec = _drop_scene(model.path)
    plain = ET.fromstring(convert_urdf_to_mjcf_xml(model, camera_mounts=(), scene_spec=spec))
    colored = ET.fromstring(convert_urdf_to_mjcf_xml(model, camera_mounts=(),
        scene_spec=replace(spec, visual_rgba_by_link={"table": (0.5, 0.5, 0.5, 1)})))
    visual = colored.find(".//geom[@name='table_visual_0']")
    assert visual.get("rgba") == "0.5 0.5 0.5 1"
    visual.set("rgba", plain.find(".//geom[@name='table_visual_0']").get("rgba"))
    assert ET.tostring(colored) == ET.tostring(plain)
    with pytest.raises(ValueError, match="unknown URDF links"):
        convert_urdf_to_mjcf_xml(model, camera_mounts=(),
            scene_spec=replace(spec, visual_rgba_by_link={"missing": (0.5,) * 4}))


@pytest.mark.integration
def test_dmtac_task_has_four_gray_sensor_visuals_without_recoloring_collision() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    task = CubeLiftTaskConfig.load(
        "sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json")
    contract = task.base_contract()
    model = parse_urdf(contract.scene.urdf_path)
    plain_spec = replace(contract.scene, visual_rgba_by_link={})
    converted = []
    for spec in (plain_spec, contract.scene):
        converted.append(ET.fromstring(convert_urdf_to_mjcf_xml(model,
            camera_mounts=(), timestep=.001, scene_spec=spec,
            gripper_actuation_profile=task.gripper_actuation_profile())))
    plain, colored = converted
    assert len(contract.scene.visual_rgba_by_link) == 4
    for link in contract.scene.visual_rgba_by_link:
        geoms = colored.findall(f".//body[@name='{link}']/geom")
        visuals = [geom for geom in geoms if geom.get("group") == "1"]
        assert visuals
        for visual in visuals:
            assert visual.get("rgba") == "0.5 0.5 0.5 1"
            assert visual.get("contype") == visual.get("conaffinity") == "0"
            visual.set("rgba", plain.find(f".//geom[@name='{visual.get('name')}']").get("rgba"))
    assert ET.tostring(colored) == ET.tostring(plain)


@pytest.mark.integration
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


def test_object_visual_mesh_binding_leaves_primitive_collision_unchanged(
    tmp_path: Path,
) -> None:
    urdf_path = _isolated_table_world(tmp_path)
    mesh_path = tmp_path / "visual_only.obj"
    mesh_path.write_text(
        "\n".join(
            (
                "v 0.00 0.00 0.00",
                "v 0.02 0.00 0.00",
                "v 0.00 0.02 0.00",
                "v 0.00 0.00 0.02",
                "f 1 3 2",
                "f 1 2 4",
                "f 2 3 4",
                "f 3 1 4",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    root = ET.fromstring(
        convert_urdf_to_mjcf_xml(
            parse_urdf(urdf_path),
            camera_mounts=(),
            scene_spec=_drop_scene(urdf_path),
            object_visual_mesh_paths={"drop_cube": mesh_path},
        )
    )

    visual = root.find(".//geom[@name='drop_cube_visual']")
    collision = root.find(".//geom[@name='drop_cube_collision']")
    assert visual is not None and visual.get("type") == "mesh"
    assert visual.get("contype") == visual.get("conaffinity") == "0"
    assert visual.get("density") == "0"
    assert collision is not None and collision.get("type") == "box"
    assert collision.get("size") == "0.02 0.02 0.02"
    assert collision.get("contype") == "8"
    mesh_name = visual.get("mesh")
    assert mesh_name
    mesh_asset = root.find(f"./asset/mesh[@name='{mesh_name}']")
    assert mesh_asset is not None
    assert mesh_asset.get("file") == str(mesh_path.resolve())
    assert mesh_asset.get("inertia") == "shell"
    mujoco = pytest.importorskip("mujoco")
    scene = MujocoScene.from_urdf(
        urdf_path,
        camera_names=(),
        scene_spec=_drop_scene(urdf_path),
        object_visual_mesh_paths={"drop_cube": mesh_path},
    )
    model = scene.model
    visual_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "drop_cube_visual"
    )
    collision_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "drop_cube_collision"
    )
    assert model.geom_type[visual_id] == mujoco.mjtGeom.mjGEOM_MESH
    assert model.geom_type[collision_id] == mujoco.mjtGeom.mjGEOM_BOX

    with pytest.raises(ValueError, match="unknown scene objects"):
        convert_urdf_to_mjcf_xml(
            parse_urdf(urdf_path),
            camera_mounts=(),
            scene_spec=_drop_scene(urdf_path),
            object_visual_mesh_paths={"not_in_scene": mesh_path},
        )


def test_visual_mesh_binding_keeps_inertial_and_native_motion_identical(tmp_path: Path) -> None:
    mujoco = pytest.importorskip("mujoco")
    urdf = _isolated_table_world(tmp_path)
    mesh = tmp_path / "visual.obj"
    mesh.write_text("v 0 0 0\nv .02 0 0\nv 0 .02 0\nv 0 0 .02\n"
                    "f 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n")
    spec = _drop_scene(urdf)
    plain = convert_urdf_to_mjcf_xml(parse_urdf(urdf), camera_mounts=(), timestep=.001, scene_spec=spec)
    visual = convert_urdf_to_mjcf_xml(parse_urdf(urdf), camera_mounts=(), timestep=.001,
                                      scene_spec=spec, object_visual_mesh_paths={"drop_cube": mesh})
    first, second = ET.fromstring(plain), ET.fromstring(visual)
    for selector in (".//body[@name='drop_cube']/inertial", ".//geom[@name='drop_cube_collision']"):
        assert ET.tostring(first.find(selector)) == ET.tostring(second.find(selector))
    models = [mujoco.MjModel.from_xml_string(xml) for xml in (plain, visual)]
    data = [mujoco.MjData(model) for model in models]
    for _ in range(200):
        for model, state in zip(models, data):
            mujoco.mj_step(model, state)
    np.testing.assert_allclose(data[0].qpos, data[1].qpos, atol=1e-12, rtol=0)
    np.testing.assert_allclose(data[0].qvel, data[1].qvel, atol=1e-12, rtol=0)
    assert data[0].ncon == data[1].ncon


def test_visual_mesh_binding_rejects_missing_scene_and_unusable_paths(tmp_path: Path) -> None:
    urdf = _isolated_table_world(tmp_path)
    model, spec = parse_urdf(urdf), _drop_scene(urdf)
    path = tmp_path / "mesh.txt"
    path.write_text("not a mesh")
    with pytest.raises(ValueError, match="require a SceneSpec"):
        convert_urdf_to_mjcf_xml(model, camera_mounts=(), object_visual_mesh_paths={"drop_cube": path})
    with pytest.raises(ValueError, match="collision model"):
        convert_urdf_to_mjcf_xml(model, camera_mounts=(),
            scene_spec=replace(spec, collision=replace(spec.collision, enabled=False)),
            object_visual_mesh_paths={"drop_cube": path})
    with pytest.raises(ValueError, match=".obj or .stl"):
        convert_urdf_to_mjcf_xml(model, camera_mounts=(), scene_spec=spec,
            object_visual_mesh_paths={"drop_cube": path})
    with pytest.raises(FileNotFoundError, match="Object visual mesh not found"):
        convert_urdf_to_mjcf_xml(model, camera_mounts=(), scene_spec=spec,
            object_visual_mesh_paths={"drop_cube": tmp_path / "missing.obj"})


@pytest.mark.optional
@pytest.mark.parametrize("shape", ["box", "sphere"])
def test_isolated_free_body_falls_and_settles_reproducibly(tmp_path: Path, shape) -> None:
    mujoco = pytest.importorskip("mujoco")
    urdf_path = _isolated_table_world(tmp_path)
    spec = _drop_scene(urdf_path)
    if shape == "sphere":
        spec = replace(spec, schema_version=SCENE_SPEC_VERSION_V2,
            objects=(replace(spec.objects[0], shape="sphere", inertia_diagonal_kg_m2=None),))
    xml_text = convert_urdf_to_mjcf_xml(
        parse_urdf(urdf_path),
        camera_mounts=(),
        timestep=0.001,
        scene_spec=spec,
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
    # The table top is world z=0.01; both bodies have vertical radius/half-height 0.02.
    assert position_a[2] == pytest.approx(0.03, abs=0.004)
    assert np.linalg.norm(qvel_a) < 0.05
    assert contacts_a >= 1 and contacts_b >= 1


@pytest.mark.optional
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
    assert root.find(
        ".//geom[@name='left_eg2_4b2_cad_l_1_collision_0']"
    ).attrib[
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
    assert set(scene.gripper_specs) == {
        "left_eg2_4b2_cad",
        "right_eg2_4b2_cad",
    }
    for spec in scene.gripper_specs.values():
        assert spec.open_qpos == pytest.approx((0.0,) * 6)
        assert spec.closed_qpos == pytest.approx((0.864,) * 6)


@pytest.mark.parametrize(
    ("shape", "dimensions", "mujoco_size"),
    [
        ("box", (0.07, 0.05, 0.04), (0.035, 0.025, 0.02)),
        ("cylinder", (0.05, 0.05, 0.10), (0.025, 0.05)),
        ("sphere", (0.048, 0.048, 0.048), (0.024,)),
        ("ellipsoid", (0.06, 0.045, 0.035), (0.03, 0.0225, 0.0175)),
    ],
)
def test_scene_v2_maps_full_object_extents_to_mujoco_primitive_sizes(
    tmp_path: Path, shape, dimensions, mujoco_size
) -> None:
    urdf_path = _isolated_table_world(tmp_path)
    spec = FreeBodyObjectSpec(name="test_object", shape=shape, size_m=dimensions)
    scene = SceneSpec(
        schema_version=SCENE_SPEC_VERSION_V2,
        profile_name="primitive_size_test_v2",
        urdf_path=str(urdf_path),
        camera_registry_path="unused.json",
        objects=(spec,),
    )
    root = ET.fromstring(
        convert_urdf_to_mjcf_xml(
            parse_urdf(urdf_path), camera_mounts=(), scene_spec=scene
        )
    )

    for role in ("visual", "collision"):
        geom = root.find(f".//geom[@name='test_object_{role}']")
        assert geom is not None
        assert geom.get("type") == shape
        np.testing.assert_allclose(
            [float(value) for value in geom.get("size").split()], mujoco_size
        )
    inertial = root.find(".//body[@name='test_object']/inertial")
    assert inertial is not None
    assert all(value > 0 for value in spec.inertia_diagonal_kg_m2)


@pytest.mark.optional
def test_scene_v2_primitive_shapes_compile_with_mujoco(tmp_path: Path) -> None:
    mujoco = pytest.importorskip("mujoco")
    urdf_path = _isolated_table_world(tmp_path)
    objects = tuple(
        FreeBodyObjectSpec(
            name=name,
            shape=shape,
            size_m=size,
            initial_position_m=(0.0, offset, 0.5),
        )
        for name, shape, size, offset in (
            ("box", "box", (0.07, 0.05, 0.04), -0.15),
            ("can", "cylinder", (0.05, 0.05, 0.10), -0.05),
            ("ball", "sphere", (0.048, 0.048, 0.048), 0.05),
            ("fruit", "ellipsoid", (0.06, 0.045, 0.035), 0.15),
        )
    )
    scene = SceneSpec(
        schema_version=SCENE_SPEC_VERSION_V2,
        profile_name="primitive_compile_test_v2",
        urdf_path=str(urdf_path),
        camera_registry_path="unused.json",
        objects=objects,
    )
    xml_text = convert_urdf_to_mjcf_xml(
        parse_urdf(urdf_path), camera_mounts=(), scene_spec=scene
    )

    model = mujoco.MjModel.from_xml_string(xml_text)

    for name in ("box", "can", "ball", "fruit"):
        geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_collision"
        )
        assert geom_id >= 0


@pytest.mark.integration
def test_scene_v2_sphere_uses_existing_dmtac_assembly_without_contact_changes() -> None:
    import mujoco
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    task = CubeLiftTaskConfig.load("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json")
    baseline = task.base_contract()
    target = replace(baseline.scene.objects[0], shape="sphere", size_m=(.05, .05, .05),
                     inertia_diagonal_kg_m2=None)
    candidate = replace(baseline, scene=replace(baseline.scene,
        schema_version=SCENE_SPEC_VERSION_V2, objects=(target,)))
    models = [MujocoScene.from_contract(contract,
        gripper_actuation_profile=task.gripper_actuation_profile()).model
        for contract in (baseline, candidate)]
    old, new = models
    assert old.ngeom == new.ngeom and old.nbody == new.nbody
    body = int(new.body(target.name).id)
    geom = int(new.geom(target.name+"_collision").id)
    assert new.geom_type[geom] == mujoco.mjtGeom.mjGEOM_SPHERE
    assert new.geom_size[geom, 0] == pytest.approx(.025)
    np.testing.assert_allclose(new.body_inertia[body], [.4*target.mass_kg*.025**2]*3)
    keep_geoms = np.asarray(new.geom_bodyid != body)
    for name in ("geom_type", "geom_size", "geom_pos", "geom_quat"):
        np.testing.assert_array_equal(getattr(old, name)[keep_geoms], getattr(new, name)[keep_geoms])
    for name in ("geom_contype", "geom_conaffinity", "geom_friction", "geom_solref", "geom_solimp",
                 "pair_geom1", "pair_geom2", "pair_friction", "pair_solref", "pair_solimp",
                 "actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "eq_type", "eq_data"):
        np.testing.assert_array_equal(getattr(old, name), getattr(new, name))
    assert old.opt.timestep == new.opt.timestep
    np.testing.assert_array_equal(old.opt.gravity, new.opt.gravity)


@pytest.mark.optional
def test_scene_v2_primitives_compile_with_the_dmtac_robot_model() -> None:
    mujoco = pytest.importorskip("mujoco")
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    task = CubeLiftTaskConfig.load(
        "sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json"
    )
    base = task.base_contract()
    urdf_path = Path(base.scene.urdf_path)
    if not urdf_path.exists():
        pytest.skip("DMTAC robot assets are not available.")
    objects = tuple(
        FreeBodyObjectSpec(
            name=name,
            shape=shape,
            size_m=size,
            initial_position_m=(-0.272, y, z),
            mass_kg=mass,
            rgba=rgba,
        )
        for name, shape, size, y, z, mass, rgba in (
            ("debug_cube", "box", (0.07, 0.07, 0.07), 0.0, 0.046, 0.08, (0.08, 0.24, 0.9, 1.0)),
            ("red_can", "cylinder", (0.05, 0.05, 0.10), -0.15, 0.062, 0.10, (0.9, 0.12, 0.08, 1.0)),
            ("green_ball", "sphere", (0.048, 0.048, 0.048), 0.15, 0.036, 0.05, (0.12, 0.72, 0.22, 1.0)),
            ("orange_fruit", "ellipsoid", (0.06, 0.045, 0.035), 0.30, 0.0295, 0.06, (1.0, 0.48, 0.06, 1.0)),
        )
    )
    scene = replace(
        base.scene,
        schema_version=SCENE_SPEC_VERSION_V2,
        profile_name="dmtac_graspable_layout_review_v2",
        objects=objects,
    )
    gripper_profile = dmtac_gripper_profile_with_object_contacts(
        task.gripper_actuation_profile(),
        tuple(f"{obj.name}_collision" for obj in objects),
    )
    xml_text = convert_urdf_to_mjcf_xml(
        parse_urdf(urdf_path),
        camera_mounts=(),
        scene_spec=scene,
        gripper_actuation_profile=gripper_profile,
    )

    model = mujoco.MjModel.from_xml_string(xml_text)
    data = mujoco.MjData(model)
    for side, qpos in (("left", task.left_home_qpos), ("right", task.right_home_qpos)):
        for index, value in enumerate(qpos, start=1):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_arm_joint{index}"
            )
            data.qpos[model.jnt_qposadr[joint_id]] = value
    for gripper, joint_names in gripper_profile.joint_names_by_gripper.items():
        for joint_name, value in zip(
            joint_names, gripper_profile.open_qpos_by_gripper[gripper]
        ):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            data.qpos[model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(model, data)

    # The exact tabletop height and baseline cube clearance come from the
    # existing CubeLift task; added objects use the same 1-2 mm settling gap.
    for obj in objects:
        clearance = obj.initial_position_m[2] - obj.size_m[2] / 2 - task.table_top_z_m
        assert clearance == pytest.approx(0.001 if obj.name == "debug_cube" else 0.002)
    for index, left in enumerate(objects):
        for right in objects[index + 1 :]:
            xy_delta = np.abs(
                np.asarray(left.initial_position_m[:2])
                - np.asarray(right.initial_position_m[:2])
            )
            half_size_sum = (
                np.asarray(left.size_m[:2]) + np.asarray(right.size_m[:2])
            ) / 2.0
            aabb_clearance = np.linalg.norm(np.maximum(xy_delta - half_size_sum, 0.0))
            assert aabb_clearance > 0.08
    assert data.ncon == 0

    expected_types = {
        "debug_cube": mujoco.mjtGeom.mjGEOM_BOX,
        "red_can": mujoco.mjtGeom.mjGEOM_CYLINDER,
        "green_ball": mujoco.mjtGeom.mjGEOM_SPHERE,
        "orange_fruit": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
    }
    for name, expected_type in expected_types.items():
        geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_collision"
        )
        assert geom_id >= 0
        assert model.geom_type[geom_id] == expected_type


@pytest.mark.integration
def test_named_object_contact_adapter_compiles_without_changing_actuator_parameters() -> None:
    import mujoco
    from sim.mujoco.actuation import dmtac_gripper_profile_with_object_contacts
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    task = CubeLiftTaskConfig.load("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json")
    baseline = task.base_contract()
    original = baseline.scene.objects[0]
    first = replace(original, name="blue_cube")
    second = replace(original, name="red_cube", initial_position_m=(.4, -.4, .2))
    contract = replace(baseline, scene=replace(baseline.scene,
        schema_version=SCENE_SPEC_VERSION_V2, objects=(first, second)))
    profile = task.gripper_actuation_profile()
    adapted = dmtac_gripper_profile_with_object_contacts(
        profile, ("blue_cube_collision", "red_cube_collision"))
    old = MujocoScene.from_contract(baseline, gripper_actuation_profile=profile).model
    scene = MujocoScene.from_contract(contract, gripper_actuation_profile=adapted)
    new = scene.model
    expected = {(object_name, f"{side}_gripper_{finger}_dm_tac_w2m_sensor_collision_0")
                for object_name in ("blue_cube_collision", "red_cube_collision")
                for side in ("left", "right") for finger in ("l", "r")}
    actual = set()
    for first_id, second_id in zip(new.pair_geom1, new.pair_geom2):
        names = tuple(mujoco.mj_id2name(new, mujoco.mjtObj.mjOBJ_GEOM, int(i))
                      for i in (first_id, second_id))
        actual.add(names)
    assert actual == expected
    for name in ("blue_cube", "red_cube"):
        geom, body = int(new.geom(name + "_collision").id), int(new.body(name).id)
        assert int(new.geom_bodyid[geom]) == body and int(new.geom_contype[geom]) == 8
    for _, sensor_name in expected:
        assert int(new.geom_contype[int(new.geom(sensor_name).id)]) == 4
    for field in ("actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "eq_type", "eq_data"):
        np.testing.assert_array_equal(getattr(old, field), getattr(new, field))
    assert all(tuple(row) == pytest.approx(tuple(old.pair_friction[0])) for row in new.pair_friction)
    from sim.mujoco.deploy.grasp_contact import bind_task_grasp_contact
    for name in ("blue_cube", "red_cube"):
        binding = bind_task_grasp_contact(scene, adapted, name)
        assert binding.sensor_geom_names == (
            "right_gripper_l_dm_tac_w2m_sensor_collision_0",
            "right_gripper_r_dm_tac_w2m_sensor_collision_0")
        assert len(binding.pair_clearances(scene)) == 2


@pytest.mark.integration
@pytest.mark.parametrize("object_name,error,reason", [
    ("missing_collision", KeyError, "unknown geom pair"),
    ("debug_cube_visual", ValueError, "visual-only geoms"),
    ("table_collision_0", ValueError, "object collision"),
    ("left_arm_link3_collision_0", ValueError, "object collision"),
])
def test_named_object_contact_adapter_rejects_missing_or_visual_geoms(object_name, error, reason):
    from sim.mujoco.actuation import dmtac_gripper_profile_with_object_contacts
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    task = CubeLiftTaskConfig.load("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json")
    profile = dmtac_gripper_profile_with_object_contacts(task.gripper_actuation_profile(), (object_name,))
    with pytest.raises(error, match=reason):
        MujocoScene.from_contract(task.base_contract(), gripper_actuation_profile=profile)


@pytest.mark.integration
@pytest.mark.parametrize("problem", ["wrong_group", "wrong_owner", "same_palm_other_sensor", "same_finger_subbody"])
def test_named_object_contact_adapter_rejects_sensor_identity_mismatch(problem):
    from sim.mujoco.modeling.gripper_actuators import add_gripper_position_actuators
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    task = CubeLiftTaskConfig.load("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json")
    contract = task.base_contract()
    root = ET.fromstring(convert_urdf_to_mjcf_xml(
        parse_urdf(contract.scene.urdf_path), camera_mounts=(), scene_spec=contract.scene))
    sensor = root.find(".//geom[@name='right_gripper_l_dm_tac_w2m_sensor_collision_0']")
    assert sensor is not None
    if problem == "wrong_group":
        sensor.set("contype", "1")
    else:
        parents = {child: parent for parent in root.iter() for child in parent}
        parents[sensor].remove(sensor)
        destination = {
            "wrong_owner": "table",
            "same_palm_other_sensor": "right_gripper_r_dm_tac_w2m_sensor",
            "same_finger_subbody": "right_gripper_l_dm_tac_w2m_sensing_frame",
        }[problem]
        body = root.find(f".//body[@name='{destination}']")
        assert body is not None
        body.append(sensor)
    base_profile = task.gripper_actuation_profile()
    profile = dmtac_gripper_profile_with_object_contacts(
        base_profile, ("debug_cube_collision",), replaced_object_collision_geom_names=())
    assert set(base_profile.contact_tuning.geom_pairs).issubset(profile.contact_tuning.geom_pairs)
    with pytest.raises(ValueError, match="configured gripper"):
        add_gripper_position_actuators(root, profile)


@pytest.mark.integration
def test_object_contact_metadata_iterator_cannot_bypass_scene_validation():
    from sim.mujoco.modeling.gripper_actuators import add_gripper_position_actuators
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    task = CubeLiftTaskConfig.load("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json")
    contract = task.base_contract()
    root = ET.fromstring(convert_urdf_to_mjcf_xml(
        parse_urdf(contract.scene.urdf_path), camera_mounts=(), scene_spec=contract.scene))
    original = task.gripper_actuation_profile()
    tuning = replace(original.contact_tuning, object_contact_geom_names=iter(["table_collision_0"]))
    assert tuning.object_contact_geom_names == ("table_collision_0",)
    with pytest.raises(ValueError, match="object collision"):
        add_gripper_position_actuators(root, replace(original, contact_tuning=tuning))


@pytest.mark.integration
@pytest.mark.parametrize("mismatch", [
    "object", "legacy", "friction", "solref", "solimp",
    "friction_unset", "solref_unset", "solimp_unset",
])
def test_object_contact_binding_rejects_compiled_profile_mismatch(mismatch):
    from sim.mujoco.deploy.grasp_contact import bind_task_grasp_contact
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    task = CubeLiftTaskConfig.load("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json")
    baseline = task.base_contract()
    original = baseline.scene.objects[0]
    objects = (original, replace(original, name="blue_cube"),
               replace(original, name="red_cube", initial_position_m=(.4, -.4, .2)))
    contract = replace(baseline, scene=replace(baseline.scene,
        schema_version=SCENE_SPEC_VERSION_V2, objects=objects))
    base_profile = task.gripper_actuation_profile()
    blue_profile = dmtac_gripper_profile_with_object_contacts(base_profile, ("blue_cube_collision",))
    scene = MujocoScene.from_contract(contract, gripper_actuation_profile=blue_profile)
    assert len(bind_task_grasp_contact(scene, blue_profile, "blue_cube").pair_clearances(scene)) == 2
    if mismatch == "object":
        supplied = dmtac_gripper_profile_with_object_contacts(base_profile, ("red_cube_collision",))
        target = "red_cube"
    elif mismatch == "legacy":
        supplied, target = base_profile, original.name
    else:
        field = mismatch.removesuffix("_unset")
        altered = None if mismatch.endswith("_unset") else {
            "friction": (.123, .02, .003), "solref": (.004, 1.),
            "solimp": (.9, .95, .002)}[field]
        supplied = replace(blue_profile, contact_tuning=replace(
            blue_profile.contact_tuning, **{field: altered}))
        target = "blue_cube"
    with pytest.raises(ValueError, match="compiled contact"):
        bind_task_grasp_contact(scene, supplied, target)


@pytest.mark.integration
def test_object_contact_binding_accepts_matching_default_contact_tuning():
    from sim.mujoco.deploy.grasp_contact import bind_task_grasp_contact
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    task = CubeLiftTaskConfig.load("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json")
    base_profile = task.gripper_actuation_profile()
    profile = dmtac_gripper_profile_with_object_contacts(
        base_profile, ("debug_cube_collision",), replaced_object_collision_geom_names=())
    assert set(base_profile.contact_tuning.geom_pairs).issubset(profile.contact_tuning.geom_pairs)
    profile = replace(profile, contact_tuning=replace(
        profile.contact_tuning, friction=None, solref=None, solimp=None))
    scene = MujocoScene.from_contract(task.base_contract(), gripper_actuation_profile=profile)
    assert len(bind_task_grasp_contact(scene, profile, "debug_cube").pair_clearances(scene)) == 2
    altered = replace(profile, contact_tuning=replace(profile.contact_tuning, friction=(1.2, .05, .005)))
    with pytest.raises(ValueError, match="compiled contact"):
        bind_task_grasp_contact(scene, altered, "debug_cube")


@pytest.mark.parametrize("complete", [False, True])
def test_object_contact_binding_requires_all_body_collision_geoms(complete):
    from types import SimpleNamespace
    import mujoco
    from sim.mujoco.actuation.gripper_profile import _object_contact_pair_name
    from sim.mujoco.deploy.grasp_contact import bind_task_grasp_contact

    left = "right_gripper_l_dm_tac_w2m_sensor_collision_0"
    right = "right_gripper_r_dm_tac_w2m_sensor_collision_0"
    pairs = [("payload_0", left), ("payload_0", right)]
    if complete:
        pairs.extend([("payload_1", left), ("payload_1", right)])
    tuning = SimpleNamespace(geom_pairs=pairs,
                             object_contact_geom_names=("payload_0", "payload_1") if complete else ("payload_0",),
                             friction=None, solref=None, solimp=None)
    pair_xml = "".join(f'<pair name="{_object_contact_pair_name(obj, sensor, tuning)}" '
                       f'geom1="{obj}" geom2="{sensor}"/>' for obj, sensor in pairs)
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <body name="payload"><freejoint/>
        <geom name="payload_0" type="box" size=".02 .02 .02" mass=".05" contype="8"/>
        <geom name="payload_1" type="box" size=".01 .01 .01" mass=".01" contype="8"/>
      </body>
      <body name="palm"><body name="knuckle"><joint name="master" type="hinge"/>
        <geom type="sphere" size=".01" mass=".01" contype="0"/></body>
        <body name="right_gripper_l_dm_tac_w2m_sensor"><geom name="{left}" type="sphere" size=".01" contype="4"/></body>
        <body name="right_gripper_r_dm_tac_w2m_sensor"><geom name="{right}" type="sphere" size=".01" contype="4"/></body>
      </body></worldbody><contact>{pair_xml}</contact></mujoco>''')
    scene = SimpleNamespace(model=model, collision_projector=lambda: SimpleNamespace(
        policy=SimpleNamespace(hard_penetration_m=.001)))
    profile = SimpleNamespace(side_by_gripper={"hand": "right"},
                              master_joint_by_gripper={"hand": "master"}, contact_tuning=tuning)
    if complete:
        assert len(bind_task_grasp_contact(scene, profile, "payload").pair_clearances(scene)) == 4
    else:
        with pytest.raises(ValueError, match="complete"):
            bind_task_grasp_contact(scene, profile, "payload")

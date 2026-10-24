from __future__ import annotations

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from sim.mujoco.backend import MuJoCoGripperBackend, MuJoCoGripperBackendConfig
from sim.mujoco.config import CameraMount, GripperSpec
from sim.mujoco.controllers import GripperController
from sim.mujoco.mjcf import convert_urdf_to_mjcf_xml
from sim.mujoco.modeling import geometry as modeling_geometry
from sim.mujoco.modeling.geometry import MeshAssetRegistry
from sim.mujoco.modeling.obj_materials import obj_material_mesh_parts
from sim.mujoco.run_control import build_arg_parser, run
from sim.mujoco.scene import MujocoScene
from sim.mujoco.urdf import URDFModel, discover_camera_mounts, parse_urdf
from teleop_runtime.schemas import EffectorCommand


URDF = Path(
    "assets/rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera.urdf"
)
CAD_SINGLE_ARM_URDF = Path(
    "assets/rm65_inspire_gripper_v3/rm65_arm_wrist_camera_4b2_cad_with_legacy_housing.urdf"
)
CAD_DUAL_ARM_URDF = Path(
    "assets/rm65_inspire_gripper_v3/"
    "rm65_dual_arm_wrist_cameras_v3_head_camera_4b2_cad_with_legacy_housing.urdf"
)


@pytest.fixture(scope="module")
def canonical_urdf_model() -> URDFModel:
    """Parse the immutable canonical URDF once for this test module."""

    return parse_urdf(URDF)


@pytest.fixture(scope="module")
def canonical_camera_mounts(canonical_urdf_model: URDFModel) -> tuple[CameraMount, ...]:
    """Resolve the canonical camera registry once for all XML assertions."""

    return discover_camera_mounts(canonical_urdf_model)


@pytest.fixture(scope="module")
def canonical_mjcf_xml(
    canonical_urdf_model: URDFModel,
    canonical_camera_mounts: tuple[CameraMount, ...],
) -> str:
    """Convert the immutable canonical URDF once for pure XML assertions.

    The conversion is intentionally kept inside the test module rather than
    cached in production: the converter accepts mutable model and scene
    inputs, while these assertions only need one stable canonical output.
    The dump-mjcf CLI test below still exercises the real writer path.
    """

    return convert_urdf_to_mjcf_xml(
        canonical_urdf_model,
        camera_mounts=canonical_camera_mounts,
    )


def test_canonical_mujoco_package_replaces_legacy_namespace() -> None:
    legacy_namespace = ".".join(("sim", "mujoco_" + "real2sim"))

    assert importlib.util.find_spec("sim.mujoco") is not None
    assert importlib.util.find_spec(legacy_namespace) is None


@pytest.mark.integration
def test_discovers_registry_cameras_from_explicit_urdf(
    canonical_camera_mounts: tuple[CameraMount, ...],
) -> None:
    assert [(camera.name, camera.link_name) for camera in canonical_camera_mounts] == [
        ("front", "head_camera_frame"),
        ("left_wrist", "left_camera_rgb"),
        ("right_wrist", "right_camera_rgb"),
    ]


@pytest.mark.integration
def test_mjcf_conversion_promotes_camera_links_to_mujoco_cameras(
    canonical_mjcf_xml: str,
) -> None:
    root = ET.fromstring(canonical_mjcf_xml)
    camera_names = {element.attrib["name"] for element in root.findall(".//camera")}
    joint_names = {element.attrib["name"] for element in root.findall(".//joint")}

    assert {"front", "left_wrist", "right_wrist"} <= camera_names
    assert {"left_arm_joint1", "right_arm_joint6"} <= joint_names


@pytest.mark.integration
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


@pytest.mark.integration
def test_mjcf_conversion_splits_obj_mtl_visual_meshes_by_diffuse_color(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obj_path = tmp_path / "colored.obj"
    mtl_path = tmp_path / "colored.mtl"
    urdf_path = tmp_path / "colored.urdf"
    mtl_path.write_text(
        """\
newmtl red
Kd 1 0 0
d 1

newmtl blue
Kd 0 0 1
d 0.5
""",
        encoding="utf-8",
    )
    obj_path.write_text(
        """\
mtllib colored.mtl
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
usemtl red
f 1 2 3
usemtl blue
f 1 3 4
""",
        encoding="utf-8",
    )
    urdf_path.write_text(
        """\
<robot name="colored">
  <link name="base">
    <visual><geometry><mesh filename="colored.obj"/></geometry></visual>
    <visual><origin xyz="0.1 0 0"/><geometry><mesh filename="colored.obj"/></geometry></visual>
  </link>
</robot>
""",
        encoding="utf-8",
    )

    resolved_sources = []

    def resolve_parts(path):
        resolved_sources.append(path)
        return obj_material_mesh_parts(path)

    monkeypatch.setattr(modeling_geometry, "obj_material_mesh_parts", resolve_parts)
    model = parse_urdf(urdf_path)
    root = ET.fromstring(convert_urdf_to_mjcf_xml(model, camera_mounts=()))
    red = root.find(".//geom[@name='base_visual_0__red']")
    blue = root.find(".//geom[@name='base_visual_0__blue']")
    assert red is not None
    assert blue is not None
    assert red.attrib["rgba"] == "1 0 0 1"
    assert blue.attrib["rgba"] == "0 0 1 0.5"
    mesh_paths = [element.attrib["file"] for element in root.findall("./asset/mesh")]
    assert len(mesh_paths) == 2
    assert all("dexteleop_mujoco_obj_materials_v1" in path for path in mesh_paths)
    assert resolved_sources == [obj_path]
    repeated_red = root.find(".//geom[@name='base_visual_1__red']")
    repeated_blue = root.find(".//geom[@name='base_visual_1__blue']")
    assert repeated_red is not None and repeated_blue is not None
    assert repeated_red.attrib["rgba"] == red.attrib["rgba"]
    assert repeated_blue.attrib["rgba"] == blue.attrib["rgba"]
    assert repeated_red.attrib["pos"] == repeated_blue.attrib["pos"] == "0.1 0 0"

    asset = ET.Element("asset")
    registry = MeshAssetRegistry(asset, model)
    mesh_geometry = model.links["base"].visuals[0].geometry
    first_parts = registry.visual_geom_parts(mesh_geometry)
    second_parts = registry.visual_geom_parts(mesh_geometry)
    assert first_parts == second_parts
    assert first_parts[0].attributes is not second_parts[0].attributes
    assert resolved_sources == [obj_path, obj_path]
    collision = registry.geom_attributes(mesh_geometry)
    collision_mesh = asset.find(f"mesh[@name='{collision['mesh']}']")
    assert collision_mesh is not None
    assert collision_mesh.attrib["file"] == str(obj_path)
    assert "inertia" not in collision_mesh.attrib
    obj_path.unlink()
    with pytest.raises(FileNotFoundError, match="URDF mesh not found"):
        registry.visual_geom_parts(mesh_geometry)


@pytest.mark.integration
@pytest.mark.parametrize("changed_input", ("obj", "mtl", "missing_mtl"))
def test_new_mjcf_conversion_rereads_visual_material_inputs(
    tmp_path: Path,
    changed_input: str,
) -> None:
    obj_path = tmp_path / "part.obj"
    mtl_path = tmp_path / "part.mtl"
    obj_text = (
        "mtllib part.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "usemtl red\nf 1 2 3\n"
    )
    mtl_text = "newmtl red\nKd 1 0 0\nnewmtl green\nKd 0 1 0\n"
    obj_path.write_text(obj_text, encoding="utf-8")
    if changed_input != "missing_mtl":
        mtl_path.write_text(mtl_text, encoding="utf-8")
    urdf_path = tmp_path / "part.urdf"
    urdf_path.write_text(
        '<robot name="part"><link name="base">'
        '<visual><geometry><mesh filename="part.obj"/></geometry></visual>'
        '<visual><geometry><mesh filename="part.obj"/></geometry></visual>'
        '</link></robot>',
        encoding="utf-8",
    )
    model = parse_urdf(urdf_path)
    first_xml = convert_urdf_to_mjcf_xml(model, camera_mounts=())

    if changed_input == "obj":
        obj_path.write_text(obj_text.replace("usemtl red", "usemtl green"), encoding="utf-8")
    else:
        mtl_path.write_text(mtl_text.replace("Kd 1 0 0", "Kd 0 1 0"), encoding="utf-8")

    second_xml = convert_urdf_to_mjcf_xml(model, camera_mounts=())
    assert first_xml != second_xml
    assert [
        geom.attrib["rgba"] for geom in ET.fromstring(second_xml).findall(".//geom")
    ] == ["0 1 0 1", "0 1 0 1"]


@pytest.mark.integration
def test_obj_mtl_texture_mapping_falls_back_without_partial_color_conversion(tmp_path: Path) -> None:
    obj_path = tmp_path / "textured.obj"
    (tmp_path / "textured.mtl").write_text(
        """\
newmtl textured
Kd 1 0 0
map_Kd texture.png
""",
        encoding="utf-8",
    )
    obj_path.write_text(
        """\
mtllib textured.mtl
v 0 0 0
v 1 0 0
v 0 1 0
usemtl textured
f 1 2 3
""",
        encoding="utf-8",
    )

    assert obj_material_mesh_parts(obj_path) == ()


@pytest.mark.integration
def test_mjcf_conversion_preserves_v3_rm65_base_link_obj_mtl_colors(
    canonical_mjcf_xml: str,
) -> None:
    root = ET.fromstring(canonical_mjcf_xml)
    geoms = root.findall(".//geom[@name]")
    base_visuals = [
        geom
        for geom in geoms
        if geom.attrib["name"].startswith("left_arm_base_link_visual_0__")
    ]

    assert len(base_visuals) == 5
    assert {geom.attrib["rgba"] for geom in base_visuals} == {
        "0.0274509804 0.235294118 1 1",
        "0.250980392 0.250980392 0.250980392 1",
        "0.988235294 0.737254902 0.517647059 1",
        "1 0.776470588 0.152941176 1",
        "1 1 1 1",
    }
    assert root.find(".//geom[@name='left_arm_base_link_visual_0']") is None


@pytest.mark.integration
def test_mjcf_conversion_hides_proxy_visuals_by_default(tmp_path: Path) -> None:
    urdf_path = tmp_path / "proxy.urdf"
    urdf_path.write_text(
        """\
<robot name="proxy">
  <link name="base">
    <visual>
      <geometry><box size="0.1 0.1 0.1"/></geometry>
    </visual>
  </link>
  <link name="left_collision_proxy">
    <visual>
      <geometry><box size="1 1 1"/></geometry>
      <material name="proxy"><color rgba="0 0 1 0.25"/></material>
    </visual>
  </link>
  <link name="camera_pole">
    <visual>
      <geometry><box size="0.03 0.03 0.8"/></geometry>
      <material name="pole"><color rgba="0.92 0.45 0.15 0.35"/></material>
    </visual>
  </link>
  <joint name="proxy_fixed" type="fixed">
    <parent link="base"/>
    <child link="left_collision_proxy"/>
  </joint>
  <joint name="pole_fixed" type="fixed">
    <parent link="base"/>
    <child link="camera_pole"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    model = parse_urdf(urdf_path)

    root = ET.fromstring(convert_urdf_to_mjcf_xml(model, camera_mounts=()))
    debug_root = ET.fromstring(
        convert_urdf_to_mjcf_xml(model, camera_mounts=(), debug_proxy_visuals=True)
    )

    geom_names = {geom.attrib["name"] for geom in root.findall(".//geom")}
    debug_geoms = {geom.attrib["name"]: geom for geom in debug_root.findall(".//geom")}
    assert "base_visual_0" in geom_names
    assert "left_collision_proxy_visual_0" not in geom_names
    assert "camera_pole_visual_0" not in geom_names
    assert debug_geoms["left_collision_proxy_visual_0"].attrib["group"] == "4"
    assert debug_geoms["camera_pole_visual_0"].attrib["group"] == "4"


@pytest.mark.integration
def test_mjcf_conversion_keeps_render_camera_on_fovy_projection(tmp_path: Path) -> None:
    urdf_path = tmp_path / "camera.urdf"
    urdf_path.write_text(
        """\
<robot name="camera">
  <link name="camera_rgb"/>
</robot>
""",
        encoding="utf-8",
    )

    xml_text = convert_urdf_to_mjcf_xml(
        parse_urdf(urdf_path),
        camera_mounts=(
            CameraMount(
                name="camera_rgb",
                link_name="camera_rgb",
                width=640,
                height=480,
                fovy_deg=62.75,
                intrinsics={
                    "width": 640,
                    "height": 480,
                    "fx": 393.8,
                    "fy": 393.6,
                    "ppx": 321.1,
                    "ppy": 233.7,
                },
            ),
        ),
    )

    camera = ET.fromstring(xml_text).find(".//camera[@name='camera_rgb']")
    assert camera is not None
    assert camera.attrib["fovy"] == "62.75"
    assert "focalpixel" not in camera.attrib
    assert "principalpixel" not in camera.attrib
    assert "resolution" not in camera.attrib
    assert "sensorsize" not in camera.attrib


@pytest.mark.integration
def test_dump_mjcf_only_does_not_require_mujoco_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This assertion covers the CLI's no-runtime plumbing.  The canonical
    # asset conversion is exercised by ``canonical_mjcf_xml`` above; using it
    # here as well would repeat its mesh/material walk without adding coverage
    # for the dump-only branch.
    urdf_path = tmp_path / "dump_only.urdf"
    urdf_path.write_text(
        """\
<robot name="dump_only">
  <link name="base"/>
  <link name="right_arm"/>
  <joint name="right_arm_joint6" type="revolute">
    <parent link="base"/>
    <child link="right_arm"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    mjcf_out = tmp_path / "scene.xml"

    def fail_scene_load(*args, **kwargs):
        raise AssertionError("dump-mjcf-only should not construct MujocoScene")

    monkeypatch.setattr(MujocoScene, "from_urdf", fail_scene_load)
    args = build_arg_parser().parse_args(
        [
            "--urdf",
            str(urdf_path),
            "--dump-mjcf-only",
            "--mjcf-out",
            str(mjcf_out),
        ]
    )

    run(args)

    root = ET.parse(mjcf_out).getroot()
    assert root.tag == "mujoco"
    assert root.find(".//joint[@name='right_arm_joint6']") is not None


@pytest.mark.optional
def test_mujoco_scene_loads_arms_and_cameras() -> None:
    pytest.importorskip("mujoco")

    scene = MujocoScene.from_urdf(URDF, camera_names=("all",))

    assert set(scene.arm_specs) == {"left", "right"}
    assert set(scene.camera_names) == {"front", "left_wrist", "right_wrist"}
    assert scene.get_arm_qpos("right").shape == (6,)


@pytest.mark.optional
def test_mujoco_scene_loads_single_arm_cad_gripper() -> None:
    pytest.importorskip("mujoco")
    if not CAD_SINGLE_ARM_URDF.exists():
        pytest.skip(f"optional CAD asset is unavailable: {CAD_SINGLE_ARM_URDF}")

    scene = MujocoScene.from_urdf(CAD_SINGLE_ARM_URDF, camera_names=())

    assert set(scene.arm_specs) == {"right"}
    assert set(scene.gripper_specs) == {"eg2_4b2_cad"}
    spec = scene.gripper_specs["eg2_4b2_cad"]
    assert spec.side == "right"
    assert spec.open_qpos == pytest.approx((0.0,) * 6)
    assert spec.closed_qpos == pytest.approx((0.864,) * 6)

    scene.set_gripper_close("eg2_4b2_cad", 1.0)
    assert scene.get_gripper_qpos("eg2_4b2_cad").tolist() == pytest.approx([0.864] * 6)
    scene.set_gripper_close("eg2_4b2_cad", 0.0)
    assert scene.get_gripper_qpos("eg2_4b2_cad").tolist() == pytest.approx([0.0] * 6)


@pytest.mark.optional
def test_mujoco_scene_loads_dual_arm_cad_grippers() -> None:
    pytest.importorskip("mujoco")
    if not CAD_DUAL_ARM_URDF.exists():
        pytest.skip(f"optional CAD asset is unavailable: {CAD_DUAL_ARM_URDF}")

    scene = MujocoScene.from_urdf(CAD_DUAL_ARM_URDF, camera_names=())

    assert set(scene.arm_specs) == {"left", "right"}
    assert set(scene.gripper_specs) == {
        "left_eg2_4b2_cad",
        "right_eg2_4b2_cad",
    }
    assert scene._kinematic_qpos_indices.size == 24
    assert len(set(scene._kinematic_qpos_indices.tolist())) == 24
    for name, side in (
        ("left_eg2_4b2_cad", "left"),
        ("right_eg2_4b2_cad", "right"),
    ):
        spec = scene.gripper_specs[name]
        assert spec.side == side
        assert spec.open_qpos == pytest.approx((0.0,) * 6)
        assert spec.closed_qpos == pytest.approx((0.864,) * 6)
        scene.set_gripper_close(name, 1.0)
        assert scene.get_gripper_qpos(name).tolist() == pytest.approx([0.864] * 6)
        scene.set_gripper_close(name, 0.0)
        assert scene.get_gripper_qpos(name).tolist() == pytest.approx([0.0] * 6)


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


@pytest.mark.optional
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

    scene.step_kinematic_direct_qpos_debug(
        {},
        gripper_close_by_name={"parallel_gripper": 0.75},
    )
    assert scene.get_gripper_close("parallel_gripper") == pytest.approx(0.75)


@pytest.mark.optional
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

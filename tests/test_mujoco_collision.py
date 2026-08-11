from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from sim.contracts.scene import CollisionSpec, GroundSpec, SceneSpec
from sim.mujoco.mjcf import convert_urdf_to_mjcf_xml
from sim.mujoco.modeling.collision import (
    CollisionFilter,
    ENVIRONMENT_GROUP,
    GRIPPER_GROUP,
    OBJECT_GROUP,
    ROBOT_GROUP,
    excluded_body_pairs,
)
from sim.mujoco.urdf import parse_urdf


def _write_collision_urdf(path: Path) -> Path:
    path.write_text(
        """\
<robot name="collision_fixture">
  <link name="base">
    <visual><geometry><box size="0.2 0.2 0.2"/></geometry></visual>
    <collision name="base_contact">
      <origin xyz="0 0 0.01" rpy="0 0 0"/>
      <geometry><box size="0.1 0.1 0.1"/></geometry>
    </collision>
  </link>
  <link name="finger">
    <collision><geometry><sphere radius="0.02"/></geometry></collision>
  </link>
  <joint name="finger_joint" type="revolute">
    <parent link="base"/><child link="finger"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="1"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    return path


def _scene_for(urdf_path: Path) -> SceneSpec:
    return SceneSpec(
        profile_name="collision_test_v1",
        urdf_path=str(urdf_path),
        camera_registry_path="unused.json",
        ground=GroundSpec(enabled=False),
        objects=(),
    )


@pytest.mark.integration
def test_urdf_parser_keeps_visual_and_collision_geometry_separate(tmp_path: Path) -> None:
    model = parse_urdf(_write_collision_urdf(tmp_path / "fixture.urdf"))

    assert len(model.links["base"].visuals) == 1
    assert len(model.links["base"].collisions) == 1
    assert model.links["base"].collisions[0].name == "base_contact"
    assert model.links["base"].collisions[0].geometry.values == (0.1, 0.1, 0.1)
    assert len(model.links["finger"].visuals) == 0
    assert model.links["finger"].collisions[0].geometry.kind == "sphere"


@pytest.mark.integration
def test_physics_conversion_emits_invisible_contact_geom_beside_visual(tmp_path: Path) -> None:
    urdf_path = _write_collision_urdf(tmp_path / "fixture.urdf")
    model = parse_urdf(urdf_path)

    legacy = ET.fromstring(convert_urdf_to_mjcf_xml(model, camera_mounts=()))
    physics = ET.fromstring(
        convert_urdf_to_mjcf_xml(
            model,
            camera_mounts=(),
            scene_spec=_scene_for(urdf_path),
        )
    )

    assert legacy.find(".//geom[@name='base_visual_0']") is not None
    assert legacy.find(".//geom[@name='base_collision_0']") is None

    visual = physics.find(".//geom[@name='base_visual_0']")
    collision = physics.find(".//geom[@name='base_collision_0']")
    assert visual is not None and collision is not None
    assert visual.attrib["contype"] == "0"
    assert visual.attrib["conaffinity"] == "0"
    assert visual.attrib["size"] == "0.1 0.1 0.1"
    assert collision.attrib["rgba"] == "0 0 0 0"
    assert collision.attrib["contype"] != "0"
    assert collision.attrib["conaffinity"] != "0"
    assert collision.attrib["size"] == "0.05 0.05 0.05"


def test_collision_group_allow_matrix_and_masks_are_symmetric() -> None:
    collision_filter = CollisionFilter()

    assert collision_filter.allows(ENVIRONMENT_GROUP, ENVIRONMENT_GROUP) is False
    for first in (ENVIRONMENT_GROUP, ROBOT_GROUP, GRIPPER_GROUP, OBJECT_GROUP):
        for second in (ENVIRONMENT_GROUP, ROBOT_GROUP, GRIPPER_GROUP, OBJECT_GROUP):
            assert collision_filter.allows(first, second) == collision_filter.allows(
                second, first
            )
    assert collision_filter.masks_for_group(ENVIRONMENT_GROUP) == (1, 14)
    assert collision_filter.masks_for_group(ROBOT_GROUP) == (2, 15)
    assert collision_filter.masks_for_group(GRIPPER_GROUP) == (4, 15)
    assert collision_filter.masks_for_group(OBJECT_GROUP) == (8, 15)


@pytest.mark.integration
def test_parent_child_and_explicit_deny_pairs_compile_to_contact_excludes(
    tmp_path: Path,
) -> None:
    urdf_path = _write_collision_urdf(tmp_path / "fixture.urdf")
    model = parse_urdf(urdf_path)
    spec = CollisionSpec(excluded_body_pairs=(("finger", "base"),))

    pairs = excluded_body_pairs(model, spec)
    root = ET.fromstring(
        convert_urdf_to_mjcf_xml(
            model,
            camera_mounts=(),
            scene_spec=SceneSpec(
                profile_name="collision_test_v1",
                urdf_path=str(urdf_path),
                camera_registry_path="unused.json",
                ground=GroundSpec(enabled=False),
                objects=(),
                collision=spec,
            ),
        )
    )

    assert pairs == (("base", "finger"),)
    excludes = {
        tuple(sorted((item.attrib["body1"], item.attrib["body2"])))
        for item in root.findall("./contact/exclude")
    }
    assert excludes == {("base", "finger")}

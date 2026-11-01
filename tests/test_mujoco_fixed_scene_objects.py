"""Fixed scene objects (handover S0): contract, MJCF and the declarative suite.

A fixed object is welded to the world: no free joint, an arbitrary orientation
(an inclined board, a wedge), its pose never changes. Free objects keep their
exact serialization, so existing contracts and their hashes are unchanged.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from project_paths import resolve_path
from sim.contracts import FreeBodyObjectSpec
from sim.mujoco.modeling.collision import CollisionFilter
from sim.mujoco.modeling.objects import add_free_body_object
from sim.mujoco.scene_suite import build_case

pytestmark = [pytest.mark.unit]

PLACE_SUITE = "sim/configs/scene_suites/rm65_dmtac_place_v1.json"


def test_free_objects_keep_their_exact_serialization():
    spec = FreeBodyObjectSpec(name="debug_cube")
    payload = spec.to_dict()
    assert "fixed" not in payload
    assert FreeBodyObjectSpec.from_dict(payload) == spec
    assert FreeBodyObjectSpec.from_dict(payload).fixed is False


def test_a_fixed_object_round_trips_and_validates():
    spec = FreeBodyObjectSpec(name="green_board", size_m=(0.12, 0.10, 0.008), fixed=True,
                              initial_quaternion_wxyz=(math.cos(.1), 0., math.sin(.1), 0.))
    payload = spec.to_dict()
    assert payload["fixed"] is True
    assert FreeBodyObjectSpec.from_dict(payload) == spec
    with pytest.raises(TypeError, match="fixed"):
        FreeBodyObjectSpec(name="board", fixed="yes")


def _body(spec):
    world = ET.Element("worldbody")
    add_free_body_object(world, spec, CollisionFilter())
    return world.find("body")


def test_a_fixed_object_has_no_joint_or_inertia_and_the_same_geoms():
    free = _body(FreeBodyObjectSpec(name="board", size_m=(0.12, 0.10, 0.008)))
    fixed = _body(FreeBodyObjectSpec(name="board", size_m=(0.12, 0.10, 0.008), fixed=True))
    assert free.find("freejoint") is not None and free.find("inertial") is not None
    assert fixed.find("freejoint") is None and fixed.find("inertial") is None
    assert [g.attrib for g in fixed.findall("geom")] == [g.attrib for g in free.findall("geom")]


def _slope_suite(tmp_path, *, pitch_deg=10., extra=None):
    suite = json.loads(resolve_path(PLACE_SUITE).read_text())
    suite["base_task"] = str(resolve_path(suite["base_task"]))
    suite["prototypes"]["green_board"] = {"shape": "box", "size_m": [0.12, 0.10, 0.008],
                                          "mass_kg": 0.2, "rgba": [0.12, 0.62, 0.25, 1]}
    objects = [{"name": "blue_cube", "prototype": "blue_small_cube", "xy_m": [-0.285, -0.02]},
               {"name": "green_board", "prototype": "green_board", "fixed": True,
                "xyz_m": [-0.38, 0.03, 0.03], "rpy_deg": [0., pitch_deg, 20.]}]
    objects += extra or []
    suite["cases"] = [{"id": "slope", "focus": "blue_cube", "objects": objects,
                       "tasks": [{"id": "place", "phase": "payload_pending",
                                  "instruction": "把蓝色小方块放到绿色斜板上，并稳定释放。",
                                  "evaluation": {"kind": "stable_placement", "target": "blue_cube",
                                                 "destination": "green_board",
                                                 "requires_payload_validation": True}}]}]
    path = tmp_path / "slope_suite.json"
    path.write_text(json.dumps(suite))
    return path


def _rotation(quat_wxyz):
    w, x, y, z = quat_wxyz
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


@pytest.mark.parametrize("pitch_deg", [10., 20.])
@pytest.mark.parametrize("variant", [0, 1])
def test_suite_builds_a_fixed_inclined_board_with_the_declared_normal(tmp_path, pitch_deg, variant):
    contract, task, *_ = build_case("slope", variant=variant,
                                    suite_path=_slope_suite(tmp_path, pitch_deg=pitch_deg))
    objects = {o.name: o for o in contract.scene.objects}
    board = objects["green_board"]
    assert board.fixed is True
    offset = json.loads(resolve_path(PLACE_SUITE).read_text())["variants_xy_m"][variant]
    assert board.initial_position_m == pytest.approx((-0.38 + offset[0], 0.03 + offset[1], 0.03))
    normal = _rotation(board.initial_quaternion_wxyz)[:, 2]
    assert math.degrees(math.acos(normal[2])) == pytest.approx(pitch_deg)
    # R = Rz(yaw) Ry(pitch): the normal leans along the yawed board x axis.
    yaw = math.radians(20.)
    assert normal[:2] / np.linalg.norm(normal[:2]) == pytest.approx(
        [math.cos(yaw), math.sin(yaw)])
    cube = objects[task["object_body_name"]]
    assert cube.fixed is False and cube.initial_quaternion_wxyz == pytest.approx((1., 0., 0., 0.))
    assert cube.initial_position_m[2] - cube.size_m[2] / 2. == pytest.approx(
        task["table_top_z_m"] + 0.001, abs=1e-6)


@pytest.mark.parametrize("edit, match", [
    (lambda objects: objects[1].update(xy_m=[0., 0.]), "xyz_m only"),
    (lambda objects: objects[1].update(support="table"), "xyz_m only"),
    (lambda objects: objects[0].update(fixed=True, xyz_m=[0, 0, .03]), "cannot be the focus"),
])
def test_fixed_objects_are_declared_by_pose_and_never_the_focus(tmp_path, edit, match):
    path = _slope_suite(tmp_path)
    suite = json.loads(path.read_text())
    edit(suite["cases"][0]["objects"])
    path.write_text(json.dumps(suite))
    with pytest.raises(ValueError, match=match):
        build_case("slope", suite_path=path)


def test_nothing_is_stacked_on_a_tilted_fixture(tmp_path):
    extra = [{"name": "yellow_cube", "prototype": "yellow_small_cube", "xy_m": [-0.38, 0.03],
              "support": "green_board"}]
    with pytest.raises(ValueError, match="axis-aligned"):
        build_case("slope", suite_path=_slope_suite(tmp_path, extra=extra))

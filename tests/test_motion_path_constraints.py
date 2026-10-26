from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_motion.path_constraints import ToolAxisCone


def _pose(rotation):
    return [0., 0., 0., *rotation.as_quat().tolist()]


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_axis_cone_is_not_a_full_orientation_constraint(axis):
    cone = ToolAxisCone.from_pose(_pose(Rotation.identity()), tool_axis=axis,
                                 max_deviation_rad=math.radians(10.))
    assert cone.accepts(_pose(Rotation.from_euler(axis, 175., degrees=True)))
    other = "y" if axis == "x" else "x"
    assert not cone.accepts(_pose(Rotation.from_euler(other, 11., degrees=True)))
    assert cone.accepts(_pose(Rotation.from_euler(other, 10., degrees=True)))


def test_axis_cone_common_frame_rotation_and_quaternion_sign_are_invariant():
    initial = Rotation.from_euler("xyz", [35., -22., 61.], degrees=True)
    goal = initial * Rotation.from_euler("x", 7., degrees=True)
    cone = ToolAxisCone.from_pose(_pose(initial), tool_axis="z", max_deviation_rad=.2)
    frame = Rotation.from_euler("xyz", [-13., 53., 19.], degrees=True)
    rotated = ToolAxisCone.from_pose(_pose(frame * initial), tool_axis="z", max_deviation_rad=.2)
    assert rotated.deviation_rad(_pose(frame * goal)) == pytest.approx(cone.deviation_rad(_pose(goal)))
    negative = _pose(goal)
    negative[3:] = (-np.asarray(negative[3:])).tolist()
    assert cone.deviation_rad(negative) == pytest.approx(math.radians(7.))


@pytest.mark.parametrize("value", [True, 0., -1., float("nan"), float("inf"), math.pi + .001])
def test_axis_cone_rejects_invalid_bounds(value):
    with pytest.raises(ValueError):
        ToolAxisCone.from_pose([0., 0., 0., 0., 0., 0., 1.], tool_axis="z",
                              max_deviation_rad=value)


@pytest.mark.parametrize("vector", [(0., 0., 0.), (0., 0., 2.), (True, 0., 0.), (float("nan"), 0., 1.)])
def test_axis_cone_rejects_invalid_reference_axis(vector):
    with pytest.raises(ValueError):
        ToolAxisCone("z", vector, .1)


def test_axis_cone_identity_binds_axis_bound_and_start_orientation():
    original = ToolAxisCone("z", (0., 0., 1.), .2)
    assert original.semantic_content_hash == ToolAxisCone("z", (0., 0., 1.), .2).semantic_content_hash
    assert len({original.semantic_content_hash,
                ToolAxisCone("y", (0., 0., 1.), .2).semantic_content_hash,
                ToolAxisCone("z", (0., 0., 1.), .1).semantic_content_hash,
                ToolAxisCone("z", (0., 1., 0.), .2).semantic_content_hash}) == 4

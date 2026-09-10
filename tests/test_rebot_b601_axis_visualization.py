from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from sim.mujoco.rebot_b601_rs.axis_demo import (
    AxisDirectionPhase,
    axis_direction_phases,
    build_axis_visualization_xml,
    direction_frame_pose,
)

pytestmark = [pytest.mark.contract, pytest.mark.optional]
SOURCE = Path(__file__).resolve().parents[1] / "hardware/reBotArm_control_py"


def test_axis_direction_order_matches_foreground_capture_protocol() -> None:
    phases = axis_direction_phases()

    assert [phase.name for phase in phases] == [
        "TRANSLATION_X_POSITIVE",
        "TRANSLATION_X_NEGATIVE",
        "TRANSLATION_Y_POSITIVE",
        "TRANSLATION_Y_NEGATIVE",
        "TRANSLATION_Z_POSITIVE",
        "TRANSLATION_Z_NEGATIVE",
        "ROTATION_X_POSITIVE",
        "ROTATION_X_NEGATIVE",
        "ROTATION_Y_POSITIVE",
        "ROTATION_Y_NEGATIVE",
        "ROTATION_Z_POSITIVE",
        "ROTATION_Z_NEGATIVE",
    ]


@pytest.mark.parametrize("axis", list("xyz"))
@pytest.mark.parametrize("direction", [1, -1])
def test_translation_moves_in_signed_world_axis_and_returns(
    axis: str,
    direction: int,
) -> None:
    anchor = np.asarray((0.2, -0.1, 0.4))
    phase = AxisDirectionPhase("translation", axis, direction)
    expected = anchor.copy()
    expected["xyz".index(axis)] += direction * 0.03

    start_position, start_quaternion = direction_frame_pose(
        phase,
        phase_fraction=0.0,
        anchor_position_m=anchor,
        translation_m=0.03,
        rotation_rad=math.radians(8.0),
    )
    peak_position, peak_quaternion = direction_frame_pose(
        phase,
        phase_fraction=0.5,
        anchor_position_m=anchor,
        translation_m=0.03,
        rotation_rad=math.radians(8.0),
    )
    end_position, end_quaternion = direction_frame_pose(
        phase,
        phase_fraction=1.0,
        anchor_position_m=anchor,
        translation_m=0.03,
        rotation_rad=math.radians(8.0),
    )

    np.testing.assert_allclose(start_position, anchor, atol=1e-12)
    np.testing.assert_allclose(peak_position, expected, atol=1e-12)
    np.testing.assert_allclose(end_position, anchor, atol=1e-12)
    for quaternion in (start_quaternion, peak_quaternion, end_quaternion):
        np.testing.assert_allclose(np.linalg.norm(quaternion), 1.0, atol=1e-12)
    np.testing.assert_allclose(start_quaternion, end_quaternion, atol=1e-12)


@pytest.mark.parametrize("axis", list("xyz"))
@pytest.mark.parametrize("direction", [1, -1])
def test_rotation_uses_signed_world_axis_and_returns(
    axis: str,
    direction: int,
) -> None:
    phase = AxisDirectionPhase("rotation", axis, direction)
    anchor = np.asarray((0.2, -0.1, 0.4))

    peak_position, peak_quaternion = direction_frame_pose(
        phase,
        phase_fraction=0.5,
        anchor_position_m=anchor,
        translation_m=0.03,
        rotation_rad=math.radians(8.0),
    )
    end_position, end_quaternion = direction_frame_pose(
        phase,
        phase_fraction=1.0,
        anchor_position_m=anchor,
        translation_m=0.03,
        rotation_rad=math.radians(8.0),
    )
    expected = np.zeros(3)
    expected["xyz".index(axis)] = direction * math.radians(8.0)
    # scipy uses xyzw; the visualizer boundary stores MuJoCo wxyz.
    peak_xyzw = peak_quaternion[[1, 2, 3, 0]]

    np.testing.assert_allclose(peak_position, anchor, atol=1e-12)
    np.testing.assert_allclose(
        _canonical_rotvec(peak_xyzw),
        expected,
        atol=1e-12,
    )
    np.testing.assert_allclose(end_position, anchor, atol=1e-12)
    np.testing.assert_allclose(end_quaternion, (1.0, 0.0, 0.0, 0.0), atol=1e-12)


def _canonical_rotvec(quaternion_xyzw: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(quaternion_xyzw).as_rotvec()


def test_visualization_xml_adds_noncolliding_static_and_moving_frames() -> None:
    pytest.importorskip("mujoco")
    if not SOURCE.is_dir():
        pytest.skip("B601 source asset is not linked in this worktree")
    xml_text = build_axis_visualization_xml(
        SOURCE,
        anchor_position_m=(0.3, 0.0, 0.25),
    )
    root = ET.fromstring(xml_text)
    anchor = root.find(".//body[@name='b601_axis_anchor_frame']")
    moving = root.find(".//body[@name='b601_axis_moving_frame']")

    assert anchor is not None
    assert moving is not None
    assert moving.attrib["mocap"] == "true"
    assert len(anchor.findall("geom")) == 6
    assert len(moving.findall("geom")) == 6
    for geom in (*anchor.findall("geom"), *moving.findall("geom")):
        assert geom.attrib["contype"] == "0"
        assert geom.attrib["conaffinity"] == "0"
        assert geom.attrib["group"] == "0"

    import mujoco

    model = mujoco.MjModel.from_xml_string(xml_text)
    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "b601_axis_moving_frame",
    )
    assert model.body_mocapid[body_id] >= 0

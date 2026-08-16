from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_motion.coordinate_frames import CoordinateFrameSpec
from robot_motion.coordinate_profiles import STATION_RETARGET_FRAME_CONTRACT_V1
from robot_motion.primitives import (
    LEVEL_TCP_ORIENTATION_BY_HEADING,
    LEVEL_TCP_ORIENTATION_PRIMITIVES,
    LevelTCPHeading,
    LevelTCPOrientationPrimitive,
    level_tcp_orientation,
)


def _desk_frame() -> CoordinateFrameSpec:
    return CoordinateFrameSpec(
        frame_id="desk",
        frame_version="test",
        semantic_parameters={
            "axis_contract": STATION_RETARGET_FRAME_CONTRACT_V1,
        },
    )


def _arbitrary_anchor() -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = Rotation.from_euler(
        "xyz",
        (19.0, -41.0, 78.0),
        degrees=True,
    ).as_matrix()
    pose[:3, 3] = (0.38, -0.17, 0.46)
    return pose


def test_catalog_contains_the_three_requested_absolute_level_headings() -> None:
    expected_yaw = {
        LevelTCPHeading.CENTER_FORWARD: 0.0,
        LevelTCPHeading.FORWARD_LEFT_30: 30.0,
        LevelTCPHeading.FORWARD_RIGHT_30: -30.0,
    }
    assert len(LEVEL_TCP_ORIENTATION_PRIMITIVES) == 3
    assert set(LEVEL_TCP_ORIENTATION_BY_HEADING) == set(expected_yaw)
    assert (
        len({item.semantic_content_hash for item in LEVEL_TCP_ORIENTATION_PRIMITIVES})
        == 3
    )
    for heading, yaw in expected_yaw.items():
        primitive = level_tcp_orientation(heading)
        assert primitive is LEVEL_TCP_ORIENTATION_BY_HEADING[heading]
        assert primitive.heading_yaw_deg == yaw
        assert primitive.controlled_frame_id == "tool_tcp"
        description = primitive.description.lower()
        assert "horizontal" in description
        assert "fixed absolute" in description
        assert "not a relative rotation" in description
        payload = primitive.to_payload()
        assert payload["arm_binding"] == (
            "caller_selects_left_arm_or_right_arm"
        )


@pytest.mark.parametrize(
    ("heading", "yaw_deg"),
    (
        (LevelTCPHeading.CENTER_FORWARD, 0.0),
        (LevelTCPHeading.FORWARD_LEFT_30, 30.0),
        (LevelTCPHeading.FORWARD_RIGHT_30, -30.0),
    ),
)
def test_absolute_heading_preserves_position_and_is_strictly_level(
    heading: LevelTCPHeading,
    yaw_deg: float,
) -> None:
    anchor = _arbitrary_anchor()
    original = anchor.copy()
    result = level_tcp_orientation(heading).apply(
        anchor,
        reference_frame_spec=_desk_frame(),
    )

    np.testing.assert_array_equal(anchor, original)
    np.testing.assert_array_equal(result.target_pose[:3, 3], anchor[:3, 3])
    yaw = math.radians(yaw_deg)
    expected_forward = (math.cos(yaw), math.sin(yaw), 0.0)
    expected_left = (-math.sin(yaw), math.cos(yaw), 0.0)
    np.testing.assert_allclose(
        result.target_pose[:3, 2],
        expected_forward,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.target_pose[:3, 1],
        expected_left,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.target_pose[:3, 0],
        (0.0, 0.0, -1.0),
        rtol=0.0,
        atol=1e-12,
    )
    assert result.anchor_pose.flags.writeable is False
    assert result.target_pose.flags.writeable is False


def test_center_forward_is_absolute_not_an_increment_from_anchor() -> None:
    first = _arbitrary_anchor()
    second = _arbitrary_anchor()
    second[:3, :3] = Rotation.from_euler(
        "xyz",
        (-53.0, 12.0, -104.0),
        degrees=True,
    ).as_matrix()

    first_target = level_tcp_orientation("center_forward").apply(
        first,
        reference_frame_spec=_desk_frame(),
    )
    second_target = level_tcp_orientation("center_forward").apply(
        second,
        reference_frame_spec=_desk_frame(),
    )

    np.testing.assert_allclose(
        first_target.target_pose[:3, :3],
        second_target.target_pose[:3, :3],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        first_target.target_pose[:3, 2],
        (1.0, 0.0, 0.0),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("component_id", ("left_arm", "right_arm"))
def test_same_primitive_applies_to_either_arm_pose(component_id: str) -> None:
    anchor = _arbitrary_anchor()
    result = level_tcp_orientation("forward_right_30").apply(
        anchor,
        reference_frame_spec=_desk_frame(),
    )

    assert component_id in {"left_arm", "right_arm"}
    np.testing.assert_array_equal(result.target_pose[:3, 3], anchor[:3, 3])
    assert result.controlled_frame_id == "tool_tcp"


def test_heading_semantics_and_reference_frame_fail_closed() -> None:
    with pytest.raises(ValueError, match="Unsupported level TCP heading"):
        level_tcp_orientation("backward")
    with pytest.raises(ValueError, match="heading_yaw_deg"):
        LevelTCPOrientationPrimitive(
            primitive_id="forged-left",
            heading=LevelTCPHeading.FORWARD_LEFT_30,
            heading_yaw_deg=-30.0,
        )
    camera_frame = CoordinateFrameSpec(
        frame_id="camera",
        frame_version="test",
        semantic_parameters={"axis_contract": "camera_axes_v1"},
    )
    with pytest.raises(ValueError, match="axis_contract"):
        level_tcp_orientation("center_forward").apply(
            np.eye(4),
            reference_frame_spec=camera_frame,
        )


def test_orientation_target_rejects_translation_or_hash_forgery() -> None:
    valid = level_tcp_orientation("forward_left_30").apply(
        _arbitrary_anchor(),
        reference_frame_spec=_desk_frame(),
    )
    payload = valid.to_payload(include_hash=False)
    payload["primitive_hash"] = "forged"
    with pytest.raises(ValueError, match="primitive_hash"):
        type(valid)(**payload)

    payload = valid.to_payload(include_hash=False)
    payload["target_pose"] = np.asarray(payload["target_pose"], dtype=float)
    payload["target_pose"][0, 3] += 0.01
    with pytest.raises(ValueError, match="preserve translation"):
        type(valid)(**payload)


def test_orientation_primitive_respects_robot_motion_dependency_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "robot_motion/primitives/tcp_orientation.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imports = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    imports.update(
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert imports.isdisjoint(
        {
            "control_composition",
            "control_station",
            "hardware",
            "sim",
            "teleop_runtime",
        }
    )

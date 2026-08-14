from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_motion.coordinate_frames import CoordinateFrameSpec
from robot_motion.coordinate_profiles import STATION_RETARGET_FRAME_CONTRACT_V1
from robot_motion.primitives import (
    CARDINAL_TCP_SHORT_STEP_BY_DIRECTION,
    CARDINAL_TCP_SHORT_STEP_PRIMITIVES,
    SHORT_STEP_DISTANCE_M,
    CardinalTCPPoseDirection,
    TCPPoseDisplacementPrimitive,
    cardinal_tcp_short_step,
)

EXPECTED_DELTAS = {
    CardinalTCPPoseDirection.FORWARD: (0.03, 0.0, 0.0),
    CardinalTCPPoseDirection.BACKWARD: (-0.03, 0.0, 0.0),
    CardinalTCPPoseDirection.LEFT: (0.0, 0.03, 0.0),
    CardinalTCPPoseDirection.RIGHT: (0.0, -0.03, 0.0),
    CardinalTCPPoseDirection.UP: (0.0, 0.0, 0.03),
    CardinalTCPPoseDirection.DOWN: (0.0, 0.0, -0.03),
}


def _reference_frame() -> CoordinateFrameSpec:
    return CoordinateFrameSpec(
        frame_id="station_world",
        frame_version="test",
        semantic_parameters={
            "axis_contract": STATION_RETARGET_FRAME_CONTRACT_V1,
        },
    )


def _rotated_anchor() -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_euler(
        "xyz",
        (31.0, -17.0, 83.0),
        degrees=True,
    ).as_matrix()
    pose[:3, 3] = (0.41, -0.23, 0.57)
    return pose


def test_short_step_catalog_contains_exactly_six_cardinal_3cm_primitives() -> None:
    assert SHORT_STEP_DISTANCE_M == 0.03
    assert len(CARDINAL_TCP_SHORT_STEP_PRIMITIVES) == 6
    assert set(CARDINAL_TCP_SHORT_STEP_BY_DIRECTION) == set(EXPECTED_DELTAS)
    assert len(
        {primitive.primitive_id for primitive in CARDINAL_TCP_SHORT_STEP_PRIMITIVES}
    ) == 6
    assert len(
        {
            primitive.semantic_content_hash
            for primitive in CARDINAL_TCP_SHORT_STEP_PRIMITIVES
        }
    ) == 6

    for direction, expected_delta in EXPECTED_DELTAS.items():
        primitive = cardinal_tcp_short_step(direction)
        assert primitive is CARDINAL_TCP_SHORT_STEP_BY_DIRECTION[direction]
        assert primitive.distance_m == 0.03
        assert primitive.controlled_frame_id == "tool_tcp"
        assert (
            primitive.reference_axis_contract
            == STATION_RETARGET_FRAME_CONTRACT_V1
        )
        np.testing.assert_array_equal(
            primitive.delta_translation_reference_m,
            expected_delta,
        )


@pytest.mark.parametrize("direction", tuple(CardinalTCPPoseDirection))
def test_short_step_adds_translation_in_reference_axes_and_preserves_orientation(
    direction: CardinalTCPPoseDirection,
) -> None:
    anchor = _rotated_anchor()
    original = anchor.copy()

    result = cardinal_tcp_short_step(direction).apply(
        anchor,
        reference_frame_spec=_reference_frame(),
    )

    np.testing.assert_array_equal(anchor, original)
    np.testing.assert_allclose(
        result.target_pose[:3, 3] - anchor[:3, 3],
        EXPECTED_DELTAS[direction],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(result.target_pose[:3, :3], anchor[:3, :3])
    assert result.reference_frame_id == "station_world"
    assert (
        result.reference_frame_spec_hash
        == _reference_frame().semantic_content_hash
    )
    assert result.anchor_pose.flags.writeable is False
    assert result.target_pose.flags.writeable is False
    assert result.delta_translation_reference_m.flags.writeable is False


def test_forward_is_reference_positive_x_not_rotated_tool_positive_x() -> None:
    anchor = np.eye(4, dtype=np.float64)
    anchor[:3, :3] = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()

    result = cardinal_tcp_short_step("forward").apply(
        anchor,
        reference_frame_spec=_reference_frame(),
    )

    np.testing.assert_allclose(
        result.target_pose[:3, 3],
        (0.03, 0.0, 0.0),
        rtol=0.0,
        atol=1e-12,
    )
    assert not np.allclose(result.target_pose[:3, 3], (0.0, 0.03, 0.0))


def test_primitive_and_result_hashes_are_deterministic_and_frame_bound() -> None:
    first = TCPPoseDisplacementPrimitive(
        primitive_id="test.forward",
        direction="forward",
        distance_m=0.03,
    )
    second = TCPPoseDisplacementPrimitive(
        primitive_id="test.forward",
        direction=CardinalTCPPoseDirection.FORWARD,
        distance_m=0.03,
    )
    assert first.semantic_content_hash == second.semantic_content_hash

    anchor = np.eye(4, dtype=np.float64)
    first_result = first.apply(anchor, reference_frame_spec=_reference_frame())
    second_result = second.apply(anchor, reference_frame_spec=_reference_frame())
    assert first_result.semantic_content_hash == second_result.semantic_content_hash

    other_frame = CoordinateFrameSpec(
        frame_id="other_world",
        frame_version="test",
        semantic_parameters={
            "axis_contract": STATION_RETARGET_FRAME_CONTRACT_V1,
        },
    )
    other_result = second.apply(anchor, reference_frame_spec=other_frame)
    assert other_result.semantic_content_hash != first_result.semantic_content_hash


@pytest.mark.parametrize(
    ("anchor", "message"),
    (
        (np.eye(3), "finite 4x4 pose"),
        (np.full((4, 4), np.nan), "finite 4x4 pose"),
        (
            np.asarray(
                (
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 2.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                )
            ),
            "rotation must be orthonormal",
        ),
    ),
)
def test_apply_rejects_invalid_anchor_poses(anchor: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        cardinal_tcp_short_step("up").apply(
            anchor,
            reference_frame_spec=_reference_frame(),
        )


def test_lookup_and_frame_binding_fail_closed() -> None:
    with pytest.raises(ValueError, match="Unsupported cardinal TCP direction"):
        cardinal_tcp_short_step("diagonal")
    with pytest.raises(TypeError, match="reference_frame_spec"):
        cardinal_tcp_short_step("left").apply(
            np.eye(4),
            reference_frame_spec="station_world",  # type: ignore[arg-type]
        )
    incompatible_frame = CoordinateFrameSpec(
        frame_id="camera",
        frame_version="test",
        semantic_parameters={"axis_contract": "camera_axes_v1"},
    )
    with pytest.raises(ValueError, match="axis_contract"):
        cardinal_tcp_short_step("forward").apply(
            np.eye(4),
            reference_frame_spec=incompatible_frame,
        )


def test_displacement_target_rejects_a_forged_primitive_hash() -> None:
    primitive = cardinal_tcp_short_step("down")
    valid = primitive.apply(
        np.eye(4),
        reference_frame_spec=_reference_frame(),
    )

    payload = valid.to_payload(include_hash=False)
    payload["primitive_hash"] = "forged"
    with pytest.raises(ValueError, match="primitive_hash"):
        type(valid)(**payload)


def test_primitive_module_respects_robot_motion_dependency_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "robot_motion/primitives/tcp_displacement.py"
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

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_motion.coordinate_frames import CoordinateFrameSpec
from teleop_runtime.action.vive_residual import (
    build_vive_human_residual_coordinate_binding,
)
from teleop_runtime.calibration import TeleopCalibration


def _transform(
    translation: tuple[float, float, float],
    *,
    rotation_xyz_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = Rotation.from_euler(
        "xyz",
        rotation_xyz_deg,
        degrees=True,
    ).as_matrix()
    value[:3, 3] = translation
    return value


def _calibration() -> TeleopCalibration:
    return TeleopCalibration(
        T_W_O=_transform(
            (0.2, 0.1, 0.0),
            rotation_xyz_deg=(0.0, 0.0, 10.0),
        ),
        T_O_D=_transform(
            (0.0, 0.3, 0.0),
            rotation_xyz_deg=(0.0, 0.0, -20.0),
        ),
        tracker_offsets={
            "left_wrist": _transform((0.0, 0.0, 0.1)),
            "right_wrist": _transform((0.0, 0.0, 0.2)),
        },
    )


def test_vive_binding_matches_existing_calibration_math_with_explicit_bridge() -> None:
    calibration = _calibration()
    bridge = _transform(
        (1.0, 2.0, 0.0),
        rotation_xyz_deg=(0.0, 0.0, 30.0),
    )
    binding = build_vive_human_residual_coordinate_binding(
        calibration,
        coordinate_profile_id="vive-profile",
        coordinate_profile_hash="vive-profile-hash",
        canonical_reference_frame=CoordinateFrameSpec(
            "task/table",
            semantic_parameters={"definition": "task-table-v1"},
        ),
        matrix_canonical_vive_retarget=bridge,
        tracker_role_by_component={
            "left_arm": "left_wrist",
            "right_arm": "right_wrist",
        },
        source_tracker_frame_by_component={
            "left_arm": CoordinateFrameSpec("vive/left_tracker"),
            "right_arm": CoordinateFrameSpec("vive/right_tracker"),
        },
        target_controlled_frame_by_component={
            "left_arm": CoordinateFrameSpec("left_arm/tool_tcp"),
            "right_arm": CoordinateFrameSpec("right_arm/tool_tcp"),
        },
        vive_world_frame=CoordinateFrameSpec("custom/vive_world"),
        vive_operator_frame=CoordinateFrameSpec("custom/operator"),
        vive_retarget_frame=CoordinateFrameSpec("custom/vive_retarget"),
    )
    raw_left_tracker = _transform(
        (0.4, -0.1, 0.5),
        rotation_xyz_deg=(5.0, 15.0, 25.0),
    )

    graph_result, _ = binding.frame_graph.transform_pose_between_frames(
        raw_left_tracker,
        source_reference_frame_id="custom/vive_world",
        source_controlled_frame_id="vive/left_tracker",
        target_reference_frame_id="task/table",
        target_controlled_frame_id="left_arm/tool_tcp",
    )
    existing_calibration_result = (
        bridge
        @ calibration.world_to_vive_retarget(
            raw_left_tracker,
            tracker_role="left_wrist",
        )
    )

    np.testing.assert_allclose(graph_result, existing_calibration_result)
    assert binding.canonical_reference_frame_id == "task/table"
    assert dict(binding.target_controlled_frame_by_component) == {
        "left_arm": "left_arm/tool_tcp",
        "right_arm": "right_arm/tool_tcp",
    }
    assert binding.binding_id.startswith(
        "vive-human-residual-coordinate-binding:"
    )


def test_vive_retarget_is_not_implicitly_treated_as_canonical() -> None:
    calibration = _calibration()

    with pytest.raises(ValueError, match="explicit identity bridge"):
        build_vive_human_residual_coordinate_binding(
            calibration,
            coordinate_profile_id="vive-profile",
            coordinate_profile_hash="vive-profile-hash",
            canonical_reference_frame=CoordinateFrameSpec("shared-frame"),
            matrix_canonical_vive_retarget=_transform((0.1, 0.0, 0.0)),
            tracker_role_by_component={"left_arm": "left_wrist"},
            source_tracker_frame_by_component={
                "left_arm": CoordinateFrameSpec("left_tracker"),
            },
            target_controlled_frame_by_component={
                "left_arm": CoordinateFrameSpec("left_tool"),
            },
            vive_retarget_frame=CoordinateFrameSpec("shared-frame"),
        )


def test_vive_binding_requires_every_declared_tracker_offset() -> None:
    calibration = _calibration()

    with pytest.raises(ValueError, match="missing tracker offsets"):
        build_vive_human_residual_coordinate_binding(
            calibration,
            coordinate_profile_id="vive-profile",
            coordinate_profile_hash="vive-profile-hash",
            canonical_reference_frame=CoordinateFrameSpec("canonical"),
            matrix_canonical_vive_retarget=np.eye(4),
            tracker_role_by_component={"arm": "unknown_wrist"},
            source_tracker_frame_by_component={
                "arm": CoordinateFrameSpec("tracker"),
            },
            target_controlled_frame_by_component={
                "arm": CoordinateFrameSpec("tool"),
            },
        )


def test_vive_binding_requires_explicit_calibration_chain() -> None:
    with pytest.raises(ValueError, match="requires explicit T_W_O and T_O_D"):
        build_vive_human_residual_coordinate_binding(
            TeleopCalibration(
                tracker_offsets={"wrist": np.eye(4)},
            ),
            coordinate_profile_id="vive-profile",
            coordinate_profile_hash="vive-profile-hash",
            canonical_reference_frame=CoordinateFrameSpec("canonical"),
            matrix_canonical_vive_retarget=np.eye(4),
            tracker_role_by_component={"arm": "wrist"},
            source_tracker_frame_by_component={
                "arm": CoordinateFrameSpec("tracker"),
            },
            target_controlled_frame_by_component={
                "arm": CoordinateFrameSpec("tool"),
            },
        )

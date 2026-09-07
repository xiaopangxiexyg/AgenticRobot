"""Inactive RM65 dual-arm cell-frame coordinate contract.

The cell frame uses the reference URDF root origin and human-operation axes:
``+X`` points toward the table, ``+Y`` points toward the left arm, and ``+Z``
points upward.  This module only defines an independent coordinate-frame graph
and pure absolute-pose adapters.  It does not register the frame with any
runtime graph or change motion execution.

All transform names follow ``T_TARGET_SOURCE``.  Twist and wrench conversion
is intentionally outside this contract because those representations require
adjoint and dual-adjoint semantics at their owning boundaries.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
    RigidFrameTransformSpec,
)
from robot_motion.coordinate_profiles import STATION_RETARGET_FRAME_CONTRACT_V1

RM65_DUAL_ARM_CELL_FRAME_ID_V1: Final = "rm65_dual_arm_cell"
RM65_DUAL_ARM_CELL_AXIS_CONTRACT_V1: Final = (
    "rm65_dual_arm_cell_x_table_y_left_arm_z_up_v1"
)
RM65_DUAL_ARM_URDF_WORLD_FRAME_ID_V1: Final = "rm65_dual_arm_urdf_world"
RM65_DUAL_ARM_CANONICAL_STATION_FRAME_ID_V1: Final = "canonical_station"

RM65_DUAL_ARM_REFERENCE_URDF_V4: Final = (
    "assets/rm65_inspire_gripper_v4/"
    "rm65_dual_arm_wrist_cameras_v4_head_camera_4b2_cad_"
    "with_legacy_housing.urdf"
)
RM65_DUAL_ARM_REFERENCE_URDF_V4_SHA256: Final = (
    "0a91ac161dd3aa3ca5bbf3eb2fa13e647b9eb559dfb54ce115c21bb0b3dcbb34"
)

# The cell and URDF-world origins coincide.  Their X and Y axes point in
# opposite directions, which is an exact right-handed pi rotation about +Z.
T_URDF_WORLD_CELL_V1: Final = (
    (-1.0, 0.0, 0.0, 0.0),
    (0.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)

# The legacy canonical Station origin is the tabletop near-boundary landmark:
# +0.472 m along Cell +X and +0.010 m along Cell +Z from the Cell origin.
T_CELL_CANONICAL_STATION_V1: Final = (
    (1.0, 0.0, 0.0, 0.472),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.010),
    (0.0, 0.0, 0.0, 1.0),
)
T_CANONICAL_STATION_CELL_V1: Final = (
    (1.0, 0.0, 0.0, -0.472),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, -0.010),
    (0.0, 0.0, 0.0, 1.0),
)


def build_rm65_dual_arm_cell_frame_overlay_v1() -> CoordinateFrameGraph:
    """Build the independent, non-runtime RM65 Cell coordinate overlay.

    The ``canonical_station`` specification deliberately matches the existing
    active Station specification exactly.  Building this overlay has no side
    effects and does not modify any real, simulation, teleoperation, Gateway,
    safety, IK, or dispatch graph.
    """

    cell_frame = CoordinateFrameSpec(
        frame_id=RM65_DUAL_ARM_CELL_FRAME_ID_V1,
        frame_version="1",
        semantic_parameters={
            "frame_family": "robot_cell",
            "embodiment": "rm65_dual_arm",
            "origin_definition": (
                "reference_urdf_root_world_at_nominal_arm_mount_midpoint"
            ),
            "axis_contract": RM65_DUAL_ARM_CELL_AXIS_CONTRACT_V1,
            "axes": {
                "x": "+X points from the dual-arm midpoint toward the table",
                "y": "+Y points from the dual-arm midpoint toward the left arm",
                "z": "+Z points vertically upward",
            },
            "handedness": "right",
            "reference_urdf": RM65_DUAL_ARM_REFERENCE_URDF_V4,
            "reference_urdf_sha256": RM65_DUAL_ARM_REFERENCE_URDF_V4_SHA256,
        },
    )
    urdf_world_frame = CoordinateFrameSpec(
        frame_id=RM65_DUAL_ARM_URDF_WORLD_FRAME_ID_V1,
        frame_version="1",
        semantic_parameters={
            "frame_family": "urdf_world",
            "root_link": "world",
            "axis_contract": "reference_urdf_native_xyz_v1",
            "reference_urdf": RM65_DUAL_ARM_REFERENCE_URDF_V4,
            "reference_urdf_sha256": RM65_DUAL_ARM_REFERENCE_URDF_V4_SHA256,
        },
    )
    canonical_station_frame = CoordinateFrameSpec(
        frame_id=RM65_DUAL_ARM_CANONICAL_STATION_FRAME_ID_V1,
        frame_version="1",
        semantic_parameters={
            "axis_contract": STATION_RETARGET_FRAME_CONTRACT_V1,
        },
    )
    return CoordinateFrameGraph(
        graph_id="rm65-dual-arm-cell-frame-overlay-v1",
        graph_version="1",
        frames=(cell_frame, urdf_world_frame, canonical_station_frame),
        transforms=(
            RigidFrameTransformSpec(
                transform_id="rm65-cell-to-reference-urdf-world-v1",
                transform_version="1",
                source_frame_id=cell_frame.frame_id,
                target_frame_id=urdf_world_frame.frame_id,
                matrix_target_source=T_URDF_WORLD_CELL_V1,
                semantic_parameters={
                    "basis": "exact_axis_definition_at_shared_urdf_root_origin",
                    "reference_urdf_sha256": (RM65_DUAL_ARM_REFERENCE_URDF_V4_SHA256),
                },
            ),
            RigidFrameTransformSpec(
                transform_id="canonical-station-to-rm65-cell-v1",
                transform_version="1",
                source_frame_id=canonical_station_frame.frame_id,
                target_frame_id=cell_frame.frame_id,
                matrix_target_source=T_CELL_CANONICAL_STATION_V1,
                semantic_parameters={
                    "basis": "reference_urdf_workspace_near_boundary_fixed_chain",
                    "reference_urdf_sha256": (RM65_DUAL_ARM_REFERENCE_URDF_V4_SHA256),
                },
            ),
        ),
    )


def canonical_station_pose_to_rm65_dual_arm_cell(
    pose_canonical_station_controlled: object,
) -> np.ndarray:
    """Convert an absolute ``T_canonical_station_controlled`` pose to Cell."""

    transformed, _ = build_rm65_dual_arm_cell_frame_overlay_v1().transform_pose_matrix(
        pose_canonical_station_controlled,
        source_reference_frame_id=RM65_DUAL_ARM_CANONICAL_STATION_FRAME_ID_V1,
        target_reference_frame_id=RM65_DUAL_ARM_CELL_FRAME_ID_V1,
    )
    return transformed


def rm65_dual_arm_cell_pose_to_canonical_station(
    pose_cell_controlled: object,
) -> np.ndarray:
    """Convert an absolute ``T_cell_controlled`` pose to canonical Station."""

    transformed, _ = build_rm65_dual_arm_cell_frame_overlay_v1().transform_pose_matrix(
        pose_cell_controlled,
        source_reference_frame_id=RM65_DUAL_ARM_CELL_FRAME_ID_V1,
        target_reference_frame_id=RM65_DUAL_ARM_CANONICAL_STATION_FRAME_ID_V1,
    )
    return transformed


__all__ = [
    "RM65_DUAL_ARM_CANONICAL_STATION_FRAME_ID_V1",
    "RM65_DUAL_ARM_CELL_AXIS_CONTRACT_V1",
    "RM65_DUAL_ARM_CELL_FRAME_ID_V1",
    "RM65_DUAL_ARM_REFERENCE_URDF_V4",
    "RM65_DUAL_ARM_REFERENCE_URDF_V4_SHA256",
    "RM65_DUAL_ARM_URDF_WORLD_FRAME_ID_V1",
    "T_CANONICAL_STATION_CELL_V1",
    "T_CELL_CANONICAL_STATION_V1",
    "T_URDF_WORLD_CELL_V1",
    "build_rm65_dual_arm_cell_frame_overlay_v1",
    "canonical_station_pose_to_rm65_dual_arm_cell",
    "rm65_dual_arm_cell_pose_to_canonical_station",
]

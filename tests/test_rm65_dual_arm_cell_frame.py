from __future__ import annotations

import hashlib
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from agent_gateway.profiles.real.perception_frames import (
    build_real_perception_station_frame_binding,
)
from robot_motion import rm65
from robot_motion.coordinate_frames import CoordinateFrameGraph
from robot_motion.rm65.cell_frame import (
    RM65_DUAL_ARM_CANONICAL_STATION_FRAME_ID_V1,
    RM65_DUAL_ARM_CELL_AXIS_CONTRACT_V1,
    RM65_DUAL_ARM_CELL_FRAME_ID_V1,
    RM65_DUAL_ARM_REFERENCE_URDF_V4,
    RM65_DUAL_ARM_REFERENCE_URDF_V4_SHA256,
    RM65_DUAL_ARM_URDF_WORLD_FRAME_ID_V1,
    T_CANONICAL_STATION_CELL_V1,
    T_CELL_CANONICAL_STATION_V1,
    T_URDF_WORLD_CELL_V1,
    build_rm65_dual_arm_cell_frame_overlay_v1,
    canonical_station_pose_to_rm65_dual_arm_cell,
    rm65_dual_arm_cell_pose_to_canonical_station,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pose(
    translation: tuple[float, float, float],
    rotation_xyz_deg: tuple[float, float, float],
) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_euler(
        "xyz",
        rotation_xyz_deg,
        degrees=True,
    ).as_matrix()
    result[:3, 3] = translation
    return result


def _point(transform: object, xyz: tuple[float, float, float]) -> np.ndarray:
    homogeneous = np.array((*xyz, 1.0), dtype=np.float64)
    return np.asarray(transform, dtype=np.float64) @ homogeneous


def _active_real_binding():
    world_from_station = np.array(
        (
            (-1.0, 0.0, 0.0, -0.472),
            (0.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.010),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    return build_real_perception_station_frame_binding(
        collision_scene_profile="rm65-dual-arm-workspace-near-boundary-v1",
        scene_mount_hash="cell-frame-regression-scene-mounts",
        scene_world_from_canonical_station=world_from_station,
    )


def test_cell_contract_is_available_through_the_rm65_public_api() -> None:
    expected_exports = {
        "RM65_DUAL_ARM_CELL_AXIS_CONTRACT_V1",
        "RM65_DUAL_ARM_CELL_FRAME_ID_V1",
        "T_CELL_CANONICAL_STATION_V1",
        "build_rm65_dual_arm_cell_frame_overlay_v1",
        "canonical_station_pose_to_rm65_dual_arm_cell",
        "rm65_dual_arm_cell_pose_to_canonical_station",
    }

    assert expected_exports <= set(rm65.__all__)
    assert all(hasattr(rm65, name) for name in expected_exports)


def test_overlay_is_independent_and_contains_only_reference_frames() -> None:
    graph = build_rm65_dual_arm_cell_frame_overlay_v1()

    assert isinstance(graph, CoordinateFrameGraph)
    assert graph.graph_id == "rm65-dual-arm-cell-frame-overlay-v1"
    assert set(graph.frame_by_id) == {
        RM65_DUAL_ARM_CELL_FRAME_ID_V1,
        RM65_DUAL_ARM_URDF_WORLD_FRAME_ID_V1,
        RM65_DUAL_ARM_CANONICAL_STATION_FRAME_ID_V1,
    }
    assert set(graph.transform_by_id) == {
        "canonical-station-to-rm65-cell-v1",
        "rm65-cell-to-reference-urdf-world-v1",
    }
    assert all(
        runtime_name not in graph.frame_by_id
        for runtime_name in (
            "dual_arm_solver_world",
            "mujoco_world",
            "teleop_world",
            "gateway_world",
        )
    )
    assert (
        graph.frame_by_id[RM65_DUAL_ARM_CELL_FRAME_ID_V1].semantic_parameters[
            "axis_contract"
        ]
        == RM65_DUAL_ARM_CELL_AXIS_CONTRACT_V1
    )


def test_cell_axes_are_exact_right_handed_human_semantic_axes() -> None:
    graph = build_rm65_dual_arm_cell_frame_overlay_v1()
    urdf_world_from_cell = graph.resolve(
        RM65_DUAL_ARM_CELL_FRAME_ID_V1,
        RM65_DUAL_ARM_URDF_WORLD_FRAME_ID_V1,
    ).matrix_target_source

    np.testing.assert_array_equal(
        urdf_world_from_cell,
        np.asarray(T_URDF_WORLD_CELL_V1),
    )
    rotation = urdf_world_from_cell[:3, :3]
    np.testing.assert_array_equal(rotation @ (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    np.testing.assert_array_equal(rotation @ (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))
    np.testing.assert_array_equal(rotation @ (0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
    np.testing.assert_array_equal(rotation.T @ rotation, np.eye(3))
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=0.0)


def test_reference_urdf_landmarks_have_expected_cell_semantics() -> None:
    graph = build_rm65_dual_arm_cell_frame_overlay_v1()
    cell_from_urdf_world = graph.resolve(
        RM65_DUAL_ARM_URDF_WORLD_FRAME_ID_V1,
        RM65_DUAL_ARM_CELL_FRAME_ID_V1,
    ).matrix_target_source

    # The reference URDF mounts the physical left arm at native -Y and the
    # physical right arm at native +Y.
    left_base_cell = _point(cell_from_urdf_world, (0.0, -0.2625, 0.0))
    right_base_cell = _point(cell_from_urdf_world, (0.0, 0.2625, 0.0))
    table_landmark_cell = _point(
        cell_from_urdf_world,
        (-0.472, 0.0, 0.010),
    )

    np.testing.assert_allclose(left_base_cell, (0.0, 0.2625, 0.0, 1.0))
    np.testing.assert_allclose(right_base_cell, (0.0, -0.2625, 0.0, 1.0))
    np.testing.assert_allclose(table_landmark_cell, (0.472, 0.0, 0.010, 1.0))
    assert left_base_cell[1] > 0.0
    assert right_base_cell[1] < 0.0
    assert table_landmark_cell[0] > 0.0


def test_canonical_station_bridge_has_same_axes_and_exact_inverse() -> None:
    graph = build_rm65_dual_arm_cell_frame_overlay_v1()
    cell_from_station = graph.resolve(
        RM65_DUAL_ARM_CANONICAL_STATION_FRAME_ID_V1,
        RM65_DUAL_ARM_CELL_FRAME_ID_V1,
    ).matrix_target_source
    station_from_cell = graph.resolve(
        RM65_DUAL_ARM_CELL_FRAME_ID_V1,
        RM65_DUAL_ARM_CANONICAL_STATION_FRAME_ID_V1,
    ).matrix_target_source

    np.testing.assert_array_equal(
        cell_from_station,
        np.asarray(T_CELL_CANONICAL_STATION_V1),
    )
    np.testing.assert_array_equal(
        station_from_cell,
        np.asarray(T_CANONICAL_STATION_CELL_V1),
    )
    np.testing.assert_array_equal(cell_from_station[:3, :3], np.eye(3))
    np.testing.assert_array_equal(station_from_cell @ cell_from_station, np.eye(4))

    for direction in np.eye(3):
        np.testing.assert_array_equal(
            cell_from_station[:3, :3] @ direction,
            direction,
        )


def test_absolute_pose_adapter_round_trips_random_se3_poses() -> None:
    rng = np.random.default_rng(20260907)

    for _ in range(100):
        station_pose = np.eye(4, dtype=np.float64)
        station_pose[:3, :3] = Rotation.random(random_state=rng).as_matrix()
        station_pose[:3, 3] = rng.uniform(-1.0, 1.0, size=3)

        cell_pose = canonical_station_pose_to_rm65_dual_arm_cell(station_pose)
        restored = rm65_dual_arm_cell_pose_to_canonical_station(cell_pose)

        np.testing.assert_allclose(restored, station_pose, rtol=0.0, atol=1e-12)


def test_relative_pose_is_invariant_under_station_to_cell_reexpression() -> None:
    station_from_anchor = _pose((0.31, -0.18, 0.42), (17.0, -31.0, 44.0))
    station_from_target = _pose((-0.07, 0.26, 0.63), (-23.0, 19.0, 71.0))

    cell_from_anchor = canonical_station_pose_to_rm65_dual_arm_cell(station_from_anchor)
    cell_from_target = canonical_station_pose_to_rm65_dual_arm_cell(station_from_target)

    np.testing.assert_allclose(
        np.linalg.inv(cell_from_anchor) @ cell_from_target,
        np.linalg.inv(station_from_anchor) @ station_from_target,
        rtol=0.0,
        atol=1e-12,
    )


def test_existing_execution_target_is_unchanged_after_explicit_bridge() -> None:
    executor_from_station = _pose((0.8, -0.2, 0.1), (11.0, 29.0, -37.0))
    station_from_tool = _pose((0.27, 0.16, 0.39), (-43.0, 13.0, 61.0))
    old_executor_target = executor_from_station @ station_from_tool

    cell_from_tool = canonical_station_pose_to_rm65_dual_arm_cell(station_from_tool)
    bridged_station_from_tool = rm65_dual_arm_cell_pose_to_canonical_station(
        cell_from_tool
    )
    new_executor_target = executor_from_station @ bridged_station_from_tool

    np.testing.assert_allclose(
        new_executor_target,
        old_executor_target,
        rtol=0.0,
        atol=1e-12,
    )


def test_building_overlay_does_not_mutate_active_real_frame_contract() -> None:
    before = _active_real_binding()
    before_binding_hash = before.semantic_content_hash
    before_graph_hash = before.frame_graph.semantic_content_hash
    before_payload = before.frame_graph.to_payload()

    overlay = build_rm65_dual_arm_cell_frame_overlay_v1()
    after = _active_real_binding()

    assert after.semantic_content_hash == before_binding_hash
    assert after.frame_graph.semantic_content_hash == before_graph_hash
    assert after.frame_graph.to_payload() == before_payload
    assert (
        overlay.frame_by_id[
            RM65_DUAL_ARM_CANONICAL_STATION_FRAME_ID_V1
        ].semantic_content_hash
        == before.reference_frame_spec_hash
    )
    assert overlay is not before.frame_graph


def test_reference_urdf_identity_and_fixed_landmarks_when_asset_is_available() -> None:
    reference_urdf = REPO_ROOT / RM65_DUAL_ARM_REFERENCE_URDF_V4
    if not reference_urdf.is_file():
        pytest.skip("reference v4 URDF is not materialized in this worktree")

    assert hashlib.sha256(reference_urdf.read_bytes()).hexdigest() == (
        RM65_DUAL_ARM_REFERENCE_URDF_V4_SHA256
    )
    root = ET.parse(reference_urdf).getroot()
    root_links = {link.get("name") for link in root.findall("link")} - {
        child.get("link")
        for joint in root.findall("joint")
        if (child := joint.find("child")) is not None
    }
    assert root_links == {"world"}

    expected_origins = {
        "left_arm": (0.0, -0.2625, 0.0),
        "right_arm": (0.0, 0.2625, 0.0),
        "table_joint": (-0.472, 0.0, -0.025),
    }
    for joint_name, expected_xyz in expected_origins.items():
        joint = root.find(f"joint[@name='{joint_name}']")
        assert joint is not None, joint_name
        origin = joint.find("origin")
        assert origin is not None, joint_name
        np.testing.assert_allclose(
            np.fromstring(origin.get("xyz", ""), sep=" "),
            expected_xyz,
            rtol=0.0,
            atol=1e-12,
        )

from __future__ import annotations

import numpy as np
import pytest

from robot_motion.rm65.obstacle_collision import RM65ObstacleCollisionOracle
from robot_motion.scene_obstacles import AxisAlignedBoxObstacle, ObstacleSceneSnapshot


pin = pytest.importorskip("pinocchio")
hppfcl = pytest.importorskip("hppfcl")


def _simple_prismatic_collision_model():
    model = pin.Model()
    joint_id = model.addJoint(0, pin.JointModelPX(), pin.SE3.Identity(), "moving_x")
    model.appendBodyToJoint(joint_id, pin.Inertia.Random(), pin.SE3.Identity())
    collision_model = pin.GeometryModel()
    moving = pin.GeometryObject(
        "arm_link_collision",
        joint_id,
        0,
        pin.SE3.Identity(),
        hppfcl.Sphere(0.05),
    )
    collision_model.addGeometryObject(moving)
    return model, collision_model


def _scene(*, center_x: float = 0.5, clearance: float = 0.0):
    return ObstacleSceneSnapshot(
        reference_frame_id="world",
        reference_frame_spec_hash="world-v1",
        revision=4,
        minimum_clearance_m=clearance,
        obstacles=(
            AxisAlignedBoxObstacle(
                obstacle_id="cube",
                center_xyz_m=(center_x, 0.0, 0.0),
                size_xyz_m=(0.2, 0.2, 0.2),
            ),
        ),
    )


def test_oracle_uses_link_geometry_and_does_not_mutate_live_collision_model() -> None:
    model, live_collision_model = _simple_prismatic_collision_model()
    initial_geometry_count = len(live_collision_model.geometryObjects)
    initial_pair_count = len(live_collision_model.collisionPairs)

    oracle = RM65ObstacleCollisionOracle(
        model=model,
        base_collision_model=live_collision_model,
        reference_full_q=np.zeros(model.nq),
        qpos_indices_by_component={"arm": (0,)},
        scene=_scene(),
        expected_reference_frame_id="world",
        expected_reference_frame_spec_hash="world-v1",
    )

    clear = oracle.check({"arm": np.array([0.0])})
    blocked = oracle.check({"arm": np.array([0.5])})
    assert clear.collision_check_available is True
    assert clear.safe is True
    assert blocked.collision_check_available is True
    assert blocked.safe is False
    assert blocked.metadata["collision_scope"] == "full_pinocchio_collision_model"
    assert blocked.metadata["closest_obstacle_id"] == "cube"
    assert oracle.collision_model_is_independent is True
    assert len(live_collision_model.geometryObjects) == initial_geometry_count
    assert len(live_collision_model.collisionPairs) == initial_pair_count


def test_clearance_margin_can_block_a_geometrically_separated_link() -> None:
    model, collision_model = _simple_prismatic_collision_model()
    oracle = RM65ObstacleCollisionOracle(
        model=model,
        base_collision_model=collision_model,
        reference_full_q=np.zeros(model.nq),
        qpos_indices_by_component={"arm": (0,)},
        scene=_scene(center_x=0.2, clearance=0.06),
    )
    check = oracle.check({"arm": np.array([0.0])})
    assert check.collision_check_available is True
    assert check.minimum_distance_m == pytest.approx(0.05, abs=1e-6)
    assert check.safe is False


def test_frame_contract_mismatch_is_rejected_before_geometry_construction() -> None:
    model, collision_model = _simple_prismatic_collision_model()
    with pytest.raises(ValueError, match="reference_frame_id"):
        RM65ObstacleCollisionOracle(
            model=model,
            base_collision_model=collision_model,
            reference_full_q=np.zeros(model.nq),
            qpos_indices_by_component={"arm": (0,)},
            scene=_scene(),
            expected_reference_frame_id="desk",
        )


def test_invalid_query_and_missing_moving_geometry_fail_closed() -> None:
    model, collision_model = _simple_prismatic_collision_model()
    oracle = RM65ObstacleCollisionOracle(
        model=model,
        base_collision_model=collision_model,
        reference_full_q=np.zeros(model.nq),
        qpos_indices_by_component={"arm": (0,)},
        scene=_scene(),
    )
    invalid = oracle.check({"wrong": np.array([0.0])})
    assert invalid.collision_check_available is False
    assert invalid.safe is False
    assert invalid.reason == "invalid_collision_query"

    static_collision_model = pin.GeometryModel()
    static_collision_model.addGeometryObject(
        pin.GeometryObject(
            "static",
            0,
            0,
            pin.SE3.Identity(),
            hppfcl.Sphere(0.05),
        )
    )
    static_oracle = RM65ObstacleCollisionOracle(
        model=model,
        base_collision_model=static_collision_model,
        reference_full_q=np.zeros(model.nq),
        qpos_indices_by_component={"arm": (0,)},
        scene=_scene(),
    )
    unavailable = static_oracle.check({"arm": np.array([0.0])})
    assert unavailable.collision_check_available is False
    assert unavailable.reason == "collision_model_has_no_moving_geometry"

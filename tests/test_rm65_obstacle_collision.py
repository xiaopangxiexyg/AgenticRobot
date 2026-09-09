from __future__ import annotations

import numpy as np
import pytest

from robot_motion.rm65.obstacle_collision import (
    AttachedBoxCollisionProxy,
    RM65ObstacleCollisionOracle,
)
from robot_motion.scene_obstacles import (
    AxisAlignedBoxObstacle,
    ObstacleSceneSnapshot,
)

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
    repeated_clear = oracle.check({"arm": np.array([0.0])})
    blocked = oracle.check({"arm": np.array([0.5])})
    assert repeated_clear is clear
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


def test_early_exit_query_preserves_clearance_decision_and_distance() -> None:
    model, collision_model = _simple_prismatic_collision_model()
    kwargs = {
        "model": model,
        "base_collision_model": collision_model,
        "reference_full_q": np.zeros(model.nq),
        "qpos_indices_by_component": {"arm": (0,)},
        "scene": _scene(center_x=0.5, clearance=0.06),
    }
    distance_oracle = RM65ObstacleCollisionOracle(**kwargs)
    early_exit_oracle = RM65ObstacleCollisionOracle(
        **kwargs,
        stop_at_first_clearance_violation=True,
    )

    for qpos in (np.array([0.0]), np.array([0.3])):
        distance_check = distance_oracle.check({"arm": qpos})
        early_exit_check = early_exit_oracle.check({"arm": qpos})
        assert early_exit_check.safe is distance_check.safe
        assert early_exit_check.minimum_distance_m == pytest.approx(
            distance_check.minimum_distance_m,
            abs=1e-9,
        )
        assert (
            early_exit_check.metadata["query_mode"]
            == "first_clearance_violation"
        )


def test_attached_box_proxy_tracks_parent_frame_without_mutating_live_model() -> None:
    model, collision_model = _simple_prismatic_collision_model()
    joint_id = model.getJointId("moving_x")
    frame_id = model.addFrame(
        pin.Frame(
            "camera_mount",
            joint_id,
            0,
            pin.SE3.Identity(),
            pin.FrameType.OP_FRAME,
        )
    )
    initial_geometry_count = len(collision_model.geometryObjects)
    proxy = AttachedBoxCollisionProxy(
        geometry_name="camera_proxy",
        parent_frame_name=model.frames[frame_id].name,
        center_xyz_m=(0.0, 0.0, 0.0),
        size_xyz_m=(0.1, 0.1, 0.1),
    )
    oracle = RM65ObstacleCollisionOracle(
        model=model,
        base_collision_model=collision_model,
        reference_full_q=np.zeros(model.nq),
        qpos_indices_by_component={"arm": (0,)},
        scene=_scene(center_x=0.5),
        attached_box_proxies=(proxy,),
    )

    clear = oracle.check({"arm": np.array([0.0])})
    blocked = oracle.check({"arm": np.array([0.5])})

    assert clear.safe is True
    assert blocked.safe is False
    assert blocked.metadata["attached_proxy_geometry_count"] == 1
    assert blocked.metadata["base_geometry_count"] == initial_geometry_count + 1
    assert len(collision_model.geometryObjects) == initial_geometry_count


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


def test_from_solver_requires_reviewed_frame_identity() -> None:
    model, collision_model = _simple_prismatic_collision_model()

    class _Solver:
        current_q = np.zeros(model.nq)
        robot = type(
            "Robot",
            (),
            {"model": model, "collision_model": collision_model},
        )()

    with pytest.raises(ValueError, match="reviewed frame contract"):
        RM65ObstacleCollisionOracle.from_solver(_Solver(), scene=_scene())
    with pytest.raises(ValueError, match="reviewed frame contract"):
        RM65ObstacleCollisionOracle.from_solver(
            _Solver(),
            scene=_scene(),
            expected_reference_frame_id="world",
        )
    with pytest.raises(ValueError, match="reference_frame_id"):
        RM65ObstacleCollisionOracle.from_solver(
            _Solver(),
            scene=_scene(),
            expected_reference_frame_id="desk",
            expected_reference_frame_spec_hash="world-v1",
            qpos_indices_by_component={"arm": (0,)},
        )


def test_real_rm65_scene_profile_oracle_clones_live_model() -> None:
    from robot_motion.rm65 import build_rm65_dual_arm_collision_ik_solver
    from robot_motion.rm65.assets import RM65_ASSET_DIR

    if not RM65_ASSET_DIR.exists():
        pytest.skip("reviewed local RM65 motion-model assets are unavailable")

    bundle = build_rm65_dual_arm_collision_ik_solver()
    live_model = bundle.solver.robot.collision_model
    geometry_count = len(live_model.geometryObjects)
    pair_count = len(live_model.collisionPairs)
    scene = ObstacleSceneSnapshot(
        reference_frame_id="rm65_scene_world",
        reference_frame_spec_hash="test-reviewed-rm65-scene-world-v1",
        revision=1,
        obstacles=(
            AxisAlignedBoxObstacle(
                obstacle_id="far_cube",
                center_xyz_m=(10.0, 10.0, 10.0),
                size_xyz_m=(0.1, 0.1, 0.1),
            ),
        ),
    )
    oracle = RM65ObstacleCollisionOracle.from_solver(
        bundle.solver,
        scene=scene,
        solution_layout=bundle.scene_profile.solution_layout,
        expected_reference_frame_id="rm65_scene_world",
        expected_reference_frame_spec_hash="test-reviewed-rm65-scene-world-v1",
    )
    qpos = np.asarray(bundle.solver.current_q, dtype=float)
    check = oracle.check({"left": qpos[:6], "right": qpos[6:12]})

    assert oracle.scene_hash == scene.semantic_content_hash
    assert check.collision_check_available is True
    assert check.metadata["collision_model_is_independent"] is True
    assert check.metadata["base_geometry_count"] == geometry_count
    assert check.metadata["obstacle_geometry_count"] == 1
    assert check.metadata["collision_pair_count"] > pair_count
    assert len(live_model.geometryObjects) == geometry_count
    assert len(live_model.collisionPairs) == pair_count

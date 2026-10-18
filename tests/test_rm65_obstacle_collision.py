from __future__ import annotations

import threading

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


@pytest.mark.parametrize("active_component", ("left", "right"))
def test_static_scene_uses_kinematic_ancestry_and_never_exempts_normal_checks(active_component):
    model = pin.Model()
    collision = pin.GeometryModel()
    indices = {}
    for side, offset in (("left", 0.), ("right", 1.)):
        joint_id = model.addJoint(0, pin.JointModelPX(),
            pin.SE3(np.eye(3), np.array((0., offset, 0.))), side + "_joint")
        model.appendBodyToJoint(joint_id, pin.Inertia.Random(), pin.SE3.Identity())
        # The shape is on a child joint and deliberately has no arm-name prefix.
        child = model.addJoint(joint_id, pin.JointModelRZ(), pin.SE3.Identity(), side + "_child")
        model.appendBodyToJoint(child, pin.Inertia.Random(), pin.SE3.Identity())
        collision.addGeometryObject(pin.GeometryObject("sensor_" + str(child), child,
            0, pin.SE3.Identity(), hppfcl.Sphere(.05)))
        indices[side] = (model.joints[joint_id].idx_q,)
    scene = ObstacleSceneSnapshot(reference_frame_id="world",
        reference_frame_spec_hash="world-v1", revision=1, minimum_clearance_m=.02,
        obstacles=(AxisAlignedBoxObstacle(obstacle_id="slab",
            center_xyz_m=(.2, .5, 0.), size_xyz_m=(.2, 2., .2)),))
    oracle = RM65ObstacleCollisionOracle(model=model, base_collision_model=collision,
        reference_full_q=pin.neutral(model), qpos_indices_by_component=indices, scene=scene)
    inactive = "right" if active_component == "left" else "left"
    pair = {active_component: np.array((.2,)), inactive: np.array((0.,))}
    assert oracle.check_static_scene(pair, active_component=active_component).safe
    assert not oracle.check(pair).safe  # Active-arm collision is still checked.
    for active_q in (-.5, 0., .2):
        pair = {active_component: np.array((active_q,)), inactive: np.array((.04,))}
        result = oracle.check_static_scene(pair, active_component=active_component)
        assert result.collision_check_available
        assert not result.safe
        assert result.minimum_distance_m == pytest.approx(.01, abs=1e-6)
        assert result.metadata["static_pair_count"] == 1
    assert not oracle.check_static_scene(pair, active_component="missing").collision_check_available


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


def test_clearance_only_revalidation_uses_same_pairs_and_threshold() -> None:
    model, collision_model = _simple_prismatic_collision_model()
    oracle = RM65ObstacleCollisionOracle(
        model=model,
        base_collision_model=collision_model,
        reference_full_q=np.zeros(model.nq),
        qpos_indices_by_component={"arm": (0,)},
        scene=_scene(center_x=0.5, clearance=0.02),
    )

    modes = set()
    for x in np.linspace(0.0, 0.8, 81):
        qpos = {"arm": np.array([x])}
        guard = oracle.check_clearance_only(qpos)
        exact = oracle.check(qpos)
        assert guard.safe is exact.safe
        assert guard.collision_check_available is exact.collision_check_available
        assert guard.metadata["collision_pair_count"] == exact.metadata["collision_pair_count"]
        modes.add(guard.metadata["query_mode"])
        if guard.metadata.get("distance_is_lower_bound"):
            assert guard.metadata["query_clearance_m"] == 0.02
            assert guard.minimum_distance_m <= exact.minimum_distance_m + 1e-9
        else:
            assert guard.minimum_distance_m == pytest.approx(exact.minimum_distance_m)
        assert oracle.check_clearance_only(qpos) is guard
    assert modes == {"first_clearance_violation", "all_pair_minimum_distance"}


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


@pytest.mark.parametrize("early_exit", [False, True])
def test_search_guard_preserves_safety_and_refinement_band_without_poisoning_cache(early_exit) -> None:
    model, collision_model = _simple_prismatic_collision_model()
    kwargs = dict(model=model, base_collision_model=collision_model,
                  reference_full_q=np.zeros(model.nq), qpos_indices_by_component={"arm": (0,)},
                  scene=_scene(center_x=0.5, clearance=0.02))
    oracle = RM65ObstacleCollisionOracle(**kwargs, stop_at_first_clearance_violation=early_exit)
    exact = RM65ObstacleCollisionOracle(**kwargs)
    for x in (0.0, 0.31, 0.32, 0.325, 0.33, 0.34, 0.5):
        qpos = {"arm": np.array([x])}
        expected = exact.check(qpos)
        search = oracle.check_for_search(qpos, refinement_distance_m=0.03)
        assert search.collision_check_available == expected.collision_check_available
        assert search.safe == expected.safe
        assert (search.minimum_distance_m <= 0.03) == (expected.minimum_distance_m <= 0.03)
        assert search.metadata["minimum_clearance_m"] == 0.02
        assert search.metadata["search_guard_clearance_m"] == 0.03
        assert search.metadata["collision_pair_count"] == expected.metadata["collision_pair_count"]
        if expected.minimum_distance_m <= 0.03:
            assert search.minimum_distance_m == pytest.approx(expected.minimum_distance_m, abs=1e-9)
            assert search.metadata["query_mode"] == "all_pair_minimum_distance"
        assert oracle.check_for_search(qpos, refinement_distance_m=0.03) is search
        ordinary = oracle.check(qpos)
        assert ordinary.minimum_distance_m == pytest.approx(expected.minimum_distance_m, abs=1e-9)
        assert ordinary is not search
    assert oracle.scene_hash == exact.scene_hash


@pytest.mark.parametrize("margin", [-0.01, float("nan"), float("inf")])
def test_search_guard_rejects_invalid_refinement_distance(margin) -> None:
    model, collision_model = _simple_prismatic_collision_model()
    oracle = RM65ObstacleCollisionOracle(model=model, base_collision_model=collision_model,
        reference_full_q=np.zeros(model.nq), qpos_indices_by_component={"arm": (0,)}, scene=_scene())
    with pytest.raises(ValueError, match="refinement_distance_m"):
        oracle.check_for_search({"arm": np.zeros(1)}, refinement_distance_m=margin)


def test_search_guard_failure_is_not_hidden_by_exact_retry(monkeypatch) -> None:
    from types import SimpleNamespace
    model, collision_model = _simple_prismatic_collision_model()
    oracle = RM65ObstacleCollisionOracle(model=model, base_collision_model=collision_model,
        reference_full_q=np.zeros(model.nq), qpos_indices_by_component={"arm": (0,)}, scene=_scene())
    qpos = {"arm": np.zeros(1)}
    assert oracle.check(qpos).safe
    def fail(*_):
        raise RuntimeError("guard query failed")
    monkeypatch.setattr(oracle, "_pin", SimpleNamespace(computeCollisions=fail))
    unavailable = oracle.check_for_search(qpos, refinement_distance_m=0.03)
    assert not unavailable.safe
    assert not unavailable.collision_check_available
    assert unavailable.reason == "pinocchio_fcl_collision_query_failed"
    invalid = oracle.check_for_search({"wrong": np.zeros(1)}, refinement_distance_m=0.03)
    assert not invalid.collision_check_available
    assert invalid.reason == "invalid_collision_query"


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


def test_observed_voxels_block_search_and_final_checks_and_permit_a_detour():
    from robot_motion.observed_obstacles import ObservedObstacleCloud
    from robot_motion.path_planning import plan_collision_aware_joint_path, JointPathPlannerConfig, JointPathStatus
    model = pin.Model()
    joint = model.addJoint(0, pin.JointModelTranslation(), pin.SE3.Identity(), "proxy_translation")
    collision = pin.GeometryModel()
    collision.addGeometryObject(pin.GeometryObject("hand_proxy", joint, 0,
        pin.SE3.Identity(), hppfcl.Box(.04, .04, .04)))
    oracle = RM65ObstacleCollisionOracle(model=model, base_collision_model=collision,
        reference_full_q=np.zeros(3), qpos_indices_by_component={"arm": (0, 1, 2)},
        scene=_scene(center_x=5, clearance=.02))
    cloud = ObservedObstacleCloud(reference_frame_id="world", evidence={"capture": "unit"},
        points_m=np.array([[0, y, z] for y in np.linspace(-.04, .04, 9) for z in np.linspace(-.04, .04, 9)]))
    fixed_hash = oracle.scene_hash
    oracle.bind_observed_cloud(cloud)
    assert oracle.scene_hash != fixed_hash
    blocked = {"arm": np.zeros(3)}
    for query in (oracle.check, oracle.check_clearance_only,
                  lambda q: oracle.check_for_search(q, refinement_distance_m=.03)):
        result = query(blocked)
        assert not result.safe and result.collision_check_available
        assert result.reason == "observed_obstacle_clearance_blocked"
    start, target = {"arm": np.array([-.2, 0, 0])}, {"arm": np.array([.2, 0, 0])}
    assert oracle.check(start).safe and oracle.check(target).safe
    assert oracle.check_for_search(start, refinement_distance_m=.03).minimum_distance_m > .03
    plan = plan_collision_aware_joint_path(start_qpos_by_component=start,
        target_qpos_by_component=target, movable_components=("arm",), oracle=oracle,
        config=JointPathPlannerConfig(seed=2, edge_resolution_rad=.01),
        joint_lower_by_component={"arm": np.full(3, -.3)},
        joint_upper_by_component={"arm": np.full(3, .3)})
    assert plan.status == JointPathStatus.PATH_FOUND
    assert len(plan.waypoints) > 2
    assert all(oracle.check(q).safe for q in plan.waypoints)
    # Cached queries must not leave subsequent proxy FK stuck at another pose.
    oracle.check(blocked)
    box = oracle.world_obstacle_proxy_boxes(target)[0]
    np.testing.assert_allclose(box[1][:3, 3], target["arm"])
    with pytest.raises(ValueError, match="before"):
        oracle.bind_observed_cloud(cloud)



@pytest.mark.parametrize('early_stop', [False, True])
def test_static_template_has_independent_observed_query_data_and_no_cross_scene_cache(early_stop):
    from robot_motion.observed_obstacles import ObservedObstacleCloud
    model = pin.Model()
    joint = model.addJoint(0, pin.JointModelTranslation(), pin.SE3.Identity(), 'hand')
    geometry = pin.GeometryModel()
    geometry.addGeometryObject(pin.GeometryObject('hand', joint, 0,
        pin.SE3.Identity(), hppfcl.Box(.04, .04, .04)))
    template = RM65ObstacleCollisionOracle(model=model, base_collision_model=geometry,
        reference_full_q=np.zeros(3), qpos_indices_by_component={'arm': (0, 1, 2)},
        scene=_scene(center_x=5, clearance=.02), stop_at_first_clearance_violation=early_stop)
    q = {'arm': np.zeros(3)}
    assert template.check(q).safe
    fixed_hash = template.scene_hash
    requests = []
    for point, safe in (([1., 1., 1.], True), ([0., 0., 0.], False), ([1., 1., 1.], True)):
        cloud = ObservedObstacleCloud(reference_frame_id='world', points_m=np.array([point]), evidence={})
        request = template.for_observed_cloud(cloud)
        assert request._model is template._model
        assert request._collision_model is template._collision_model
        assert request._model_data is not template._model_data
        assert request._collision_data is not template._collision_data
        assert not request._check_cache and not request._search_check_cache
        for check in (request.check, request.check_clearance_only,
                      lambda q: request.check_for_search(q, refinement_distance_m=.03)):
            assert check(q).safe == safe
        assert request.scene_hash != fixed_hash
        requests.append(request)
    assert requests[0]._check_cache is not requests[2]._check_cache
    assert template.scene_hash == fixed_hash and template.check(q).safe
    with pytest.raises(ValueError, match='static oracle'):
        requests[0].for_observed_cloud(cloud)
    with pytest.raises(ValueError, match='reference frames'):
        template.for_observed_cloud(ObservedObstacleCloud(reference_frame_id='other',
            points_m=np.array([[0., 0., 0.]]), evidence={}))



@pytest.mark.parametrize("layout", ["contiguous", "fortran", "strided", "reversed"])
@pytest.mark.parametrize("size", [0.004, 0.1])
def test_observed_voxels_preserve_all_numeric_cells_order_hash_and_input(layout, size):
    from dataset_contracts.canonical_encoding import canonical_content_hash
    from robot_motion.observed_obstacles import ObservedObstacleCloud

    rng = np.random.default_rng(7)
    values = np.concatenate(
        (
            rng.normal(size=(4096, 3)) * 0.02,
            np.array(
                [
                    [0.0, -0.0, 0.0],
                    [-0.0, 0.0, -0.0],
                    [size, -size, np.nextafter(size, 0.0)],
                    [np.nextafter(-size, -np.inf), -size, np.nextafter(-size, 0.0)],
                    [1e30, -1e30, 1e15],
                    [-1e30, 1e30, -1e15],
                ]
            ),
        )
    )
    if layout == "fortran":
        values = np.asfortranarray(values)
    elif layout == "strided":
        padded = np.empty((len(values), 6))
        padded[:, ::2] = values
        values = padded[:, ::2]
    elif layout == "reversed":
        values = values[::-1, ::-1]
    before = values.copy()
    expected = (np.unique(np.floor(values / size), axis=0) + 0.5) * size
    evidence = {"views": ["main", "wrist_left", "wrist_right"]}
    expected_hash = canonical_content_hash(
        {
            "reference_frame_id": "world",
            "points_m": expected,
            "voxel_size_m": size,
            "evidence": evidence,
        },
        domain="dexteleop.observed_obstacle_cloud.v1",
    )

    cloud = ObservedObstacleCloud(
        reference_frame_id="world",
        points_m=values,
        evidence=evidence,
        voxel_size_m=size,
    )

    np.testing.assert_array_equal(cloud.points_m, expected)
    np.testing.assert_array_equal(values, before)
    assert not cloud.points_m.flags.writeable
    assert cloud.semantic_content_hash == expected_hash
    reordered = ObservedObstacleCloud(
        reference_frame_id="world",
        points_m=values[::-1],
        evidence=evidence,
        voxel_size_m=size,
    )
    np.testing.assert_array_equal(reordered.points_m, expected)
    assert reordered.semantic_content_hash == expected_hash
    assert cloud.quantization_margin_m == pytest.approx(np.sqrt(3) * size / 2)


@pytest.mark.parametrize("early_exit", (False, True))
def test_static_scene_query_does_not_overwrite_concurrent_ordinary_query(monkeypatch, early_exit):
    model, collision = pin.Model(), pin.GeometryModel()
    indices = {}
    for side, offset in (("left", 0.), ("right", 1.)):
        joint = model.addJoint(0, pin.JointModelPX(),
            pin.SE3(np.eye(3), np.array((0., offset, 0.))), side + "_joint")
        model.appendBodyToJoint(joint, pin.Inertia.Random(), pin.SE3.Identity())
        collision.addGeometryObject(pin.GeometryObject(side + "_sensor", joint,
            0, pin.SE3.Identity(), hppfcl.Sphere(.05)))
        indices[side] = (model.joints[joint].idx_q,)
    scene = ObstacleSceneSnapshot(reference_frame_id="world",
        reference_frame_spec_hash="world-v1", revision=1, minimum_clearance_m=.02,
        obstacles=(AxisAlignedBoxObstacle(obstacle_id="slab",
            center_xyz_m=(.2, .5, 0.), size_xyz_m=(.2, 2., .2)),))
    oracle = RM65ObstacleCollisionOracle(model=model, base_collision_model=collision,
        reference_full_q=pin.neutral(model), qpos_indices_by_component=indices,
        scene=scene, stop_at_first_clearance_violation=early_exit)
    blocked_pair = {"left": np.array((.2,)), "right": np.array((0.,))}
    static_pair = {"left": np.array((-.5,)), "right": np.array((0.,))}
    computed, release = threading.Event(), threading.Event()
    buffers, placements, result, errors = {}, {}, {}, []

    def observe(native):
        def compute(*arguments):
            returned = native(*arguments)
            label = "ordinary" if threading.current_thread() is worker else "static"
            buffers[label] = (arguments[1], arguments[3])
            if label == "ordinary":
                placements["joint"] = np.stack([pose.translation for pose in arguments[1].oMi])
                placements["geometry"] = np.stack([pose.translation for pose in arguments[3].oMg])
                computed.set()
                if not release.wait(10.):
                    raise RuntimeError("Test did not release the ordinary query.")
            return returned
        return compute

    monkeypatch.setattr(pin, "computeDistances", observe(pin.computeDistances))
    monkeypatch.setattr(pin, "computeCollisions", observe(pin.computeCollisions))

    def run_ordinary():
        try:
            result["ordinary"] = oracle.check(blocked_pair)
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=run_ordinary, name="static-query-isolation-probe")
    worker.start()
    try:
        assert computed.wait(10.), "Ordinary native query did not reach the barrier."
        static_result = oracle.check_static_scene(static_pair, active_component="left")
        assert static_result.collision_check_available and static_result.safe
        joint_after_static = np.stack([pose.translation for pose in oracle._model_data.oMi])
        geometry_after_static = np.stack([pose.translation for pose in oracle._collision_data.oMg])
    finally:
        release.set()
        worker.join(10.)
    assert not worker.is_alive()
    if errors:
        raise errors[0]
    assert result["ordinary"].collision_check_available and not result["ordinary"].safe
    assert oracle.check(blocked_pair) is result["ordinary"]
    assert len(oracle._check_cache) == 1
    assert buffers["static"][0] is not buffers["ordinary"][0]
    assert buffers["static"][1] is not buffers["ordinary"][1]
    np.testing.assert_array_equal(joint_after_static, placements["joint"])
    np.testing.assert_array_equal(geometry_after_static, placements["geometry"])


def test_rigid_endpoint_query_matches_native_shape_distance_without_mutating_qpos_cache():
    model, collision = _simple_prismatic_collision_model()
    oracle = RM65ObstacleCollisionOracle(model=model, base_collision_model=collision,
        reference_full_q=pin.neutral(model), qpos_indices_by_component={"arm": (0,)},
        scene=_scene(clearance=.02))
    current = {"arm": np.array((0.,))}
    original = oracle.check(current)
    for x in (0., .3, .5):
        target = np.eye(4)
        target[0, 3] = x
        advisory = oracle.check_rigid_endpoint_pose(current, active_component="arm",
            current_tool_tcp_world=np.eye(4), target_tool_tcp_world=target)
        ordinary = oracle.check({"arm": np.array((x,))})
        assert advisory.safe == ordinary.safe
        assert advisory.minimum_distance_m == pytest.approx(ordinary.minimum_distance_m, abs=1e-9)
        assert advisory.metadata["admission_authority"] is False
        assert advisory.metadata["queried_pair_count"] == 1
        assert oracle.check(current) is original
    invalid = np.eye(4)
    invalid[0, 0] = 2.
    with pytest.raises(ValueError, match="rigid world poses"):
        oracle.check_rigid_endpoint_pose(current, active_component="arm",
            current_tool_tcp_world=np.eye(4), target_tool_tcp_world=invalid)
    assert not oracle.check_rigid_endpoint_pose(current, active_component="absent",
        current_tool_tcp_world=np.eye(4), target_tool_tcp_world=np.eye(4)).collision_check_available

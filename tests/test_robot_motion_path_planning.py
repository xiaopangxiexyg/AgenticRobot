from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from robot_motion.path_planning import (
    JointConfigurationCheck,
    JointPathPlannerConfig,
    JointPathStatus,
    plan_collision_aware_joint_path,
)
from robot_motion.scene_obstacles import AxisAlignedBoxObstacle, ObstacleSceneSnapshot


class _PlanarBoxOracle:
    def __init__(self, *, scene_hash: str = "scene") -> None:
        self.queries: list[dict[str, np.ndarray]] = []
        self.scene_hash = scene_hash

    def check(self, qpos_by_component):
        query = {key: np.asarray(value, dtype=float).copy() for key, value in qpos_by_component.items()}
        self.queries.append(query)
        x, y = query["left"]
        blocked = abs(x) <= 0.25 and abs(y) <= 0.30
        return JointConfigurationCheck(
            safe=not blocked,
            reason="inside_box" if blocked else "clear",
        )


def _planner_config(**overrides) -> JointPathPlannerConfig:
    values = {
        "seed": 7,
        "max_iterations": 1200,
        "max_nodes": 2400,
        "extension_step_rad": 0.20,
        "edge_resolution_rad": 0.025,
        "sample_padding_rad": 1.0,
        "goal_bias": 0.20,
        "shortcut_attempts": 80,
    }
    values.update(overrides)
    return JointPathPlannerConfig(**values)


def test_cost_aware_shortcut_does_not_replace_a_path_with_a_longer_workspace_edge():
    from robot_motion.path_planning.sampling import _EdgeResult, _shortcut_path
    path = [np.array([0.]), np.array([1.]), np.array([2.])]
    cost = lambda a, b: 10. if abs(float(b[0] - a[0])) > 1.5 else 1.
    result, unavailable, timed_out = _shortcut_path(path,
        lambda *_: _EdgeResult(True, True), np.random.default_rng(0), 80, cost=cost)
    assert len(result) == 3 and unavailable is None and not timed_out
    shortened, _, _ = _shortcut_path(path,
        lambda *_: _EdgeResult(True, True), np.random.default_rng(0), 80,
        cost=lambda a, b: abs(float(b[0] - a[0])))
    assert len(shortened) == 2


@pytest.mark.parametrize("invalid_cost", [float("nan"), float("inf"), -1.])
def test_planner_rejects_invalid_workspace_cost(invalid_cost):
    with pytest.raises(ValueError, match="finite non-negative"):
        plan_collision_aware_joint_path(start_qpos_by_component={"left":[-1.,0.]},
            target_qpos_by_component={"left":[1.,0.]}, movable_components=("left",),
            oracle=_PlanarBoxOracle(), config=_planner_config(),
            edge_cost=lambda *_: invalid_cost)


def test_cost_optimization_keeps_existing_total_deadline():
    clock = [0.]
    def cost(a, b):
        clock[0] += 20.
        return float(np.linalg.norm(a["left"] - b["left"]))
    result = plan_collision_aware_joint_path(start_qpos_by_component={"left":[-1.,0.]},
        target_qpos_by_component={"left":[1.,0.]}, movable_components=("left",),
        oracle=_PlanarBoxOracle(), config=_planner_config(time_budget_sec=10.),
        edge_cost=cost, monotonic=lambda: clock[0])
    assert result.status is JointPathStatus.PATH_SEARCH_EXHAUSTED
    assert result.reason == "time_budget_exhausted"


def test_aabb_scene_is_immutable_canonical_and_order_independent() -> None:
    first = AxisAlignedBoxObstacle("first", (0.0, 0.0, 0.2), (0.4, 0.2, 0.6))
    second = AxisAlignedBoxObstacle("second", (1.0, 2.0, 3.0), (0.1, 0.2, 0.3))
    scene = ObstacleSceneSnapshot(
        reference_frame_id="world",
        reference_frame_spec_hash="frame-hash",
        revision=2,
        obstacles=(second, first),
        minimum_clearance_m=0.015,
    )
    reordered = ObstacleSceneSnapshot(
        reference_frame_id="world",
        reference_frame_spec_hash="frame-hash",
        revision=2,
        obstacles=(first, second),
        minimum_clearance_m=0.015,
    )

    assert tuple(item.obstacle_id for item in scene.obstacles) == ("first", "second")
    assert scene.semantic_content_hash == reordered.semantic_content_hash
    assert first.minimum_xyz_m == pytest.approx((-0.2, -0.1, -0.1))
    assert first.maximum_xyz_m == pytest.approx((0.2, 0.1, 0.5))
    with pytest.raises(FrozenInstanceError):
        scene.revision = 3
    with pytest.raises(ValueError, match="unique"):
        ObstacleSceneSnapshot("world", "frame-hash", 0, (first, first))
    with pytest.raises(ValueError, match="positive"):
        AxisAlignedBoxObstacle("bad", (0.0, 0.0, 0.0), (0.1, 0.0, 0.1))
    with pytest.raises(TypeError, match="AxisAlignedBoxObstacle"):
        ObstacleSceneSnapshot("world", "frame-hash", 0, (object(),))


def test_direct_path_is_preferred_and_receipt_never_authorizes_execution() -> None:
    oracle = _PlanarBoxOracle()
    result = plan_collision_aware_joint_path(
        start_qpos_by_component={"left": [-1.0, 0.7], "right": [4.0, 5.0]},
        target_qpos_by_component={"left": [1.0, 0.7], "right": [4.0, 5.0]},
        movable_components=("left",),
        oracle=oracle,
        config=_planner_config(),
        scene_hash="scene",
    )

    assert result.status is JointPathStatus.PATH_FOUND
    assert result.metadata["direct_path"] is True
    assert len(result.waypoints) == 2
    assert result.execution_certified is False
    assert result.real_motion_authorized is False
    assert result.to_dict()["execution_certified"] is False
    with pytest.raises(ValueError):
        result.waypoints[0]["left"][0] = 0.0


def test_birrt_routes_around_box_and_freezes_inactive_component_everywhere() -> None:
    oracle = _PlanarBoxOracle()
    kwargs = dict(
        start_qpos_by_component={"left": [-1.0, 0.0], "right": [4.0, 5.0]},
        target_qpos_by_component={"left": [1.0, 0.0], "right": [4.0, 5.0]},
        movable_components=("left",),
        oracle=oracle,
        config=_planner_config(),
        joint_lower_by_component={"left": [-1.5, -1.0]},
        joint_upper_by_component={"left": [1.5, 1.0]},
        scene_hash="scene",
    )
    result = plan_collision_aware_joint_path(**kwargs)

    assert result.status is JointPathStatus.PATH_FOUND
    assert result.metadata["direct_path"] is False
    assert result.metadata["shortcut_revalidated"] is True
    assert len(result.waypoints) >= 3
    assert all(np.array_equal(query["right"], [4.0, 5.0]) for query in oracle.queries)
    assert all(np.array_equal(waypoint["right"], [4.0, 5.0]) for waypoint in result.waypoints)
    for left, right in zip(result.waypoints, result.waypoints[1:]):
        delta = right["left"] - left["left"]
        sample_count = max(1, int(np.ceil(np.max(np.abs(delta)) / 0.025)))
        for index in range(sample_count + 1):
            point = left["left"] + delta * (index / sample_count)
            assert not (abs(point[0]) <= 0.25 and abs(point[1]) <= 0.30)

    second_oracle = _PlanarBoxOracle()
    second = plan_collision_aware_joint_path(**{**kwargs, "oracle": second_oracle})
    assert second.status is JointPathStatus.PATH_FOUND
    assert [item["left"].tolist() for item in second.waypoints] == [
        item["left"].tolist() for item in result.waypoints
    ]


@pytest.mark.parametrize(
    ("start", "target", "expected"),
    [
        ([0.0, 0.0], [1.0, 0.0], JointPathStatus.START_BLOCKED),
        ([-1.0, 0.0], [0.0, 0.0], JointPathStatus.TARGET_BLOCKED),
    ],
)
def test_endpoint_collision_status_is_typed(start, target, expected) -> None:
    result = plan_collision_aware_joint_path(
        start_qpos_by_component={"left": start},
        target_qpos_by_component={"left": target},
        movable_components=("left",),
        oracle=_PlanarBoxOracle(),
    )
    assert result.status is expected
    assert result.waypoints == ()


def test_collision_backend_failure_is_not_misreported_as_search_exhaustion() -> None:
    class _UnavailableOnInterior:
        def check(self, qpos_by_component):
            x = float(qpos_by_component["left"][0])
            if -0.05 <= x <= 0.05:
                return JointConfigurationCheck(
                    safe=False,
                    collision_check_available=False,
                    reason="backend_offline",
                )
            return JointConfigurationCheck(safe=True)

    result = plan_collision_aware_joint_path(
        start_qpos_by_component={"left": [-1.0]},
        target_qpos_by_component={"left": [1.0]},
        movable_components=("left",),
        oracle=_UnavailableOnInterior(),
        config=_planner_config(edge_resolution_rad=0.01),
    )
    assert result.status is JointPathStatus.COLLISION_CHECK_UNAVAILABLE
    assert result.waypoints == ()


def test_near_collision_edges_are_refined_without_slowing_clear_edges() -> None:
    class _NarrowCollisionOracle:
        scene_hash = "narrow-collision"

        def __init__(self) -> None:
            self.queries: list[float] = []

        def check(self, qpos_by_component):
            value = float(qpos_by_component["left"][0])
            self.queries.append(value)
            blocked = 0.145 <= value <= 0.155
            return JointConfigurationCheck(
                safe=not blocked,
                minimum_distance_m=0.025,
                reason="narrow_collision" if blocked else "near_clearance",
            )

    oracle = _NarrowCollisionOracle()
    result = plan_collision_aware_joint_path(
        start_qpos_by_component={"left": [0.0]},
        target_qpos_by_component={"left": [0.2]},
        movable_components=("left",),
        oracle=oracle,
        config=_planner_config(
            max_iterations=1,
            max_nodes=2,
            edge_resolution_rad=0.1,
            adaptive_edge_resolution_rad=0.01,
            adaptive_refinement_distance_m=0.03,
        ),
        scene_hash="narrow-collision",
    )

    assert result.status is JointPathStatus.PATH_SEARCH_EXHAUSTED
    assert any(0.145 <= value <= 0.155 for value in oracle.queries)

    clear = _NarrowCollisionOracle()
    clear_result = plan_collision_aware_joint_path(
        start_qpos_by_component={"left": [-0.2]},
        target_qpos_by_component={"left": [-0.1]},
        movable_components=("left",),
        oracle=clear,
        config=_planner_config(
            edge_resolution_rad=0.1,
            adaptive_edge_resolution_rad=0.01,
            adaptive_refinement_distance_m=0.02,
        ),
        scene_hash="narrow-collision",
    )
    assert clear_result.status is JointPathStatus.PATH_FOUND
    assert len(clear.queries) == 2


def test_bounded_search_exhaustion_does_not_claim_kinematic_unreachability() -> None:
    class _SolidWall:
        def check(self, qpos_by_component):
            x = float(qpos_by_component["left"][0])
            return JointConfigurationCheck(safe=abs(x) > 0.1)

    result = plan_collision_aware_joint_path(
        start_qpos_by_component={"left": [-1.0, 0.0]},
        target_qpos_by_component={"left": [1.0, 0.0]},
        movable_components=("left",),
        oracle=_SolidWall(),
        config=_planner_config(max_iterations=60, max_nodes=120),
        joint_lower_by_component={"left": [-1.5, -1.0]},
        joint_upper_by_component={"left": [1.5, 1.0]},
    )
    assert result.status is JointPathStatus.PATH_SEARCH_EXHAUSTED
    assert "unreachable" not in result.reason
    assert result.metadata["search_is_complete"] is False


def test_wall_time_budget_returns_typed_partial_search_result() -> None:
    now = [0.0]

    class _SlowClearOracle:
        def check(self, _qpos_by_component):
            now[0] += 0.1
            return JointConfigurationCheck(safe=True)

    result = plan_collision_aware_joint_path(
        start_qpos_by_component={"left": [0.0]},
        target_qpos_by_component={"left": [1.0]},
        movable_components=("left",),
        oracle=_SlowClearOracle(),
        config=_planner_config(
            edge_resolution_rad=0.01,
            time_budget_sec=0.25,
        ),
        monotonic=lambda: now[0],
    )

    assert result.status is JointPathStatus.PATH_SEARCH_EXHAUSTED
    assert result.reason == "time_budget_exhausted"
    assert result.waypoints == ()
    assert result.metadata["search_termination"] == "time_budget_exhausted"
    assert result.metadata["termination_phase"] == "direct_edge"
    assert result.metadata["time_budget_sec"] == pytest.approx(0.25)


@pytest.mark.parametrize("value", (0.0, -1.0, float("nan"), float("inf")))
def test_wall_time_budget_must_be_positive_and_finite(value: float) -> None:
    with pytest.raises(ValueError, match="time_budget_sec"):
        JointPathPlannerConfig(time_budget_sec=value)


def test_inactive_target_change_is_an_invalid_request() -> None:
    result = plan_collision_aware_joint_path(
        start_qpos_by_component={"left": [0.0], "right": [1.0]},
        target_qpos_by_component={"left": [0.2], "right": [1.1]},
        movable_components=("left",),
        oracle=_PlanarBoxOracle(),
    )
    assert result.status is JointPathStatus.INVALID_REQUEST
    assert "inactive component" in result.reason


def test_component_mapping_order_is_not_part_of_request_semantics() -> None:
    class _AlwaysClear:
        def check(self, _qpos_by_component):
            return JointConfigurationCheck(safe=True)

    result = plan_collision_aware_joint_path(
        start_qpos_by_component={"left": [0.0], "right": [1.0]},
        target_qpos_by_component={"right": [1.0], "left": [0.2]},
        movable_components=("left",),
        oracle=_AlwaysClear(),
    )
    assert result.status is JointPathStatus.PATH_FOUND


def test_scene_bound_oracle_owns_receipt_hash_and_rejects_stale_label() -> None:
    obstacle = AxisAlignedBoxObstacle("box", (0.0, 0.0, 0.0), (0.2, 0.2, 0.2))
    stale_scene = ObstacleSceneSnapshot("world", "world-v1", 1, (obstacle,))
    current_scene = ObstacleSceneSnapshot("world", "world-v1", 2, (obstacle,))
    oracle = _PlanarBoxOracle(scene_hash=current_scene.semantic_content_hash)
    inherited = plan_collision_aware_joint_path(
        start_qpos_by_component={"left": [-1.0, 0.7]},
        target_qpos_by_component={"left": [1.0, 0.7]},
        movable_components=("left",),
        oracle=oracle,
    )
    assert inherited.status is JointPathStatus.PATH_FOUND
    assert inherited.scene_hash == current_scene.semantic_content_hash
    assert inherited.metadata["scene_binding"] == "oracle_bound"

    query_count = len(oracle.queries)
    mismatched = plan_collision_aware_joint_path(
        start_qpos_by_component={"left": [-1.0, 0.7]},
        target_qpos_by_component={"left": [1.0, 0.7]},
        movable_components=("left",),
        oracle=oracle,
        scene_hash=stale_scene.semantic_content_hash,
    )
    assert mismatched.status is JointPathStatus.INVALID_REQUEST
    assert mismatched.scene_hash == current_scene.semantic_content_hash
    assert mismatched.metadata["requested_scene_hash"] == stale_scene.semantic_content_hash
    assert mismatched.metadata["oracle_scene_hash"] == current_scene.semantic_content_hash
    assert len(oracle.queries) == query_count


def test_unbound_oracle_cannot_be_labeled_with_an_unverified_scene_hash() -> None:
    class _UnboundOracle:
        def check(self, _qpos_by_component):
            return JointConfigurationCheck(safe=True)

    rejected = plan_collision_aware_joint_path(
        start_qpos_by_component={"left": [0.0]},
        target_qpos_by_component={"left": [0.2]},
        movable_components=("left",),
        oracle=_UnboundOracle(),
        scene_hash="unverified-label",
    )
    assert rejected.status is JointPathStatus.INVALID_REQUEST
    assert rejected.scene_hash is None
    assert rejected.metadata["scene_binding"] == "unbound_oracle"

    accepted = plan_collision_aware_joint_path(
        start_qpos_by_component={"left": [0.0]},
        target_qpos_by_component={"left": [0.2]},
        movable_components=("left",),
        oracle=_UnboundOracle(),
    )
    assert accepted.status is JointPathStatus.PATH_FOUND
    assert accepted.scene_hash is None
    assert accepted.metadata["scene_binding"] == "unbound_oracle"


def test_shared_goal_forest_finds_other_branch_when_primary_is_disconnected() -> None:
    class _BoxAndWall(_PlanarBoxOracle):
        def check(self, qpos_by_component):
            assessment = super().check(qpos_by_component)
            x = float(qpos_by_component["left"][0])
            return JointConfigurationCheck(safe=assessment.safe and not 0.8 <= x <= 0.9)

    oracle = _BoxAndWall()
    arguments = dict(
        start_qpos_by_component={"left": [-1.0, 0.0], "right": [4.0, 5.0]},
        target_qpos_by_component={"left": [1.2, 0.0], "right": [4.0, 5.0]},
        alternative_target_qpos_by_component=({"left": [0.6, 0.0], "right": [4.0, 5.0]},),
        movable_components=("left",), oracle=oracle, config=_planner_config(),
        joint_lower_by_component={"left": [-1.5, -1.0]},
        joint_upper_by_component={"left": [1.5, 1.0]},
    )
    result = plan_collision_aware_joint_path(**arguments)
    assert result.status is JointPathStatus.PATH_FOUND
    assert result.metadata["direct_path"] is False
    assert result.metadata["shared_goal_search"] is True
    assert result.metadata["selected_target_index"] == 1
    assert result.metadata["target_count"] == 2
    assert result.metadata["shortcut_revalidated"] is True
    np.testing.assert_array_equal(result.target_qpos_by_component["left"], [0.6, 0.0])
    np.testing.assert_array_equal(result.waypoints[-1]["left"], [0.6, 0.0])
    assert all(np.array_equal(query["right"], [4.0, 5.0]) for query in oracle.queries)
    query_keys = [query["left"].tobytes() for query in oracle.queries]
    assert len(query_keys) == len(set(query_keys))
    for left, right in zip(result.waypoints, result.waypoints[1:]):
        count = max(1, int(np.ceil(np.max(np.abs(right["left"] - left["left"])) / 0.025)))
        for alpha in np.linspace(0.0, 1.0, count + 1):
            x, y = left["left"] + alpha * (right["left"] - left["left"])
            assert not (abs(x) <= 0.25 and abs(y) <= 0.30)
            assert not 0.8 <= x <= 0.9
    repeated = plan_collision_aware_joint_path(**{**arguments, "oracle": _BoxAndWall()})
    assert [q["left"].tolist() for q in repeated.waypoints] == [q["left"].tolist() for q in result.waypoints]
    primary_only = plan_collision_aware_joint_path(**{
        **arguments, "oracle": _BoxAndWall(), "alternative_target_qpos_by_component": (),
    })
    assert primary_only.status is JointPathStatus.PATH_SEARCH_EXHAUSTED


@pytest.mark.parametrize("alternative, expected", [
    ([1.0, 0.7], JointPathStatus.PATH_FOUND),
    ([0.1, 0.1], JointPathStatus.TARGET_BLOCKED),
])
def test_goal_set_skips_blocked_endpoint_but_reports_all_blocked(alternative, expected) -> None:
    result = plan_collision_aware_joint_path(
        start_qpos_by_component={"left": [-1.0, 0.7]},
        target_qpos_by_component={"left": [0.0, 0.0]},
        alternative_target_qpos_by_component=({"left": alternative},),
        movable_components=("left",), oracle=_PlanarBoxOracle(), config=_planner_config(),
    )
    assert result.status is expected
    assert len(result.metadata["target_checks"]) == 2
    if expected is JointPathStatus.PATH_FOUND:
        assert result.metadata["selected_target_index"] == 1
        np.testing.assert_array_equal(result.target_qpos_by_component["left"], alternative)
    else:
        assert result.waypoints == ()


@pytest.mark.parametrize("alternative", [
    "not-a-qpos-mapping",
    {"left": [1.0], "right": [4.1, 5.0]},
    {"left": [1.0, 0.7], "right": [4.1, 5.0]},
    {"left": [1.0, 0.7]},
    {"left": [float("nan"), 0.7], "right": [4.0, 5.0]},
    {"left": [1.6, 0.7], "right": [4.0, 5.0]},
])
def test_invalid_goal_set_is_rejected_before_any_collision_query(alternative) -> None:
    oracle = _PlanarBoxOracle()
    result = plan_collision_aware_joint_path(
        start_qpos_by_component={"left": [-1.0, 0.7], "right": [4.0, 5.0]},
        target_qpos_by_component={"left": [1.0, 0.7], "right": [4.0, 5.0]},
        alternative_target_qpos_by_component=(alternative,),
        movable_components=("left",), oracle=oracle,
        joint_lower_by_component={"left": [-1.5, -1.0]},
        joint_upper_by_component={"left": [1.5, 1.0]},
    )
    assert result.status is JointPathStatus.INVALID_REQUEST
    assert result.waypoints == ()
    assert oracle.queries == []


def test_goal_set_backend_failure_cannot_be_hidden_by_an_already_reachable_goal() -> None:
    class _FailingSecondGoal:
        def check(self, qpos_by_component):
            failed = float(qpos_by_component["left"][0]) == 2.0
            return JointConfigurationCheck(safe=not failed, collision_check_available=not failed)

    result = plan_collision_aware_joint_path(
        start_qpos_by_component={"left": [0.0]},
        target_qpos_by_component={"left": [0.0]},
        alternative_target_qpos_by_component=({"left": [2.0]},),
        movable_components=("left",), oracle=_FailingSecondGoal(),
    )
    assert result.status is JointPathStatus.COLLISION_CHECK_UNAVAILABLE
    assert result.waypoints == ()


def test_goal_set_uses_one_wall_time_budget_including_endpoint_checks() -> None:
    now = [0.0]
    calls = [0]

    class _SlowOracle:
        def check(self, _qpos_by_component):
            calls[0] += 1
            now[0] += 0.1
            return JointConfigurationCheck(safe=True)

    result = plan_collision_aware_joint_path(
        start_qpos_by_component={"left": [0.0]},
        target_qpos_by_component={"left": [1.0]},
        alternative_target_qpos_by_component=tuple({"left": [float(i)]} for i in range(2, 10)),
        movable_components=("left",), oracle=_SlowOracle(),
        config=_planner_config(time_budget_sec=0.25), monotonic=lambda: now[0],
    )
    assert result.status is JointPathStatus.PATH_SEARCH_EXHAUSTED
    assert result.reason == "time_budget_exhausted"
    assert result.metadata["termination_phase"] == "target_check"
    assert result.waypoints == ()
    assert calls[0] == 3


def test_goal_roots_count_against_shared_node_budget() -> None:
    oracle = _PlanarBoxOracle()
    result = plan_collision_aware_joint_path(
        start_qpos_by_component={"left": [-1.0, 0.0]},
        target_qpos_by_component={"left": [1.0, 0.0]},
        alternative_target_qpos_by_component=({"left": [1.1, 0.0]},),
        movable_components=("left",), oracle=oracle, config=_planner_config(max_nodes=3),
    )
    assert result.status is JointPathStatus.PATH_SEARCH_EXHAUSTED
    assert result.metadata["search_node_count"] == 3
    assert result.waypoints == ()
    oracle = _PlanarBoxOracle()
    invalid = plan_collision_aware_joint_path(
        start_qpos_by_component={"left": [-1.0, 0.0]},
        target_qpos_by_component={"left": [1.0, 0.0]},
        alternative_target_qpos_by_component=({"left": [1.1, 0.0]},),
        movable_components=("left",), oracle=oracle, config=_planner_config(max_nodes=2),
    )
    assert invalid.status is JointPathStatus.INVALID_REQUEST
    assert oracle.queries == []


def test_expired_budget_never_starts_a_collision_query() -> None:
    clock = iter((0.0, 1.0))
    oracle = _PlanarBoxOracle()
    result = plan_collision_aware_joint_path(
        start_qpos_by_component={"left": [-1.0, 0.7]},
        target_qpos_by_component={"left": [1.0, 0.7]},
        movable_components=("left",), oracle=oracle,
        config=_planner_config(time_budget_sec=0.1), monotonic=lambda: next(clock),
    )
    assert result.reason == "time_budget_exhausted"
    assert result.metadata["termination_phase"] == "start_check"
    assert result.waypoints == ()
    assert oracle.queries == []


def test_shared_motion_quality_preserves_grasp_bounds_and_short_motion():
    from robot_motion.path_planning.motion_quality import (
        DEFAULT_PATH_MOTION_QUALITY_LIMITS,
        sampled_path_motion_quality,
        tcp_path_edge_cost,
    )

    limits = DEFAULT_PATH_MOTION_QUALITY_LIMITS
    q = np.zeros((3, 6))
    xyz = np.array([[0, 0, 0], [.02, 0, 0], [.03, 0, 0]])
    quality = sampled_path_motion_quality(q, xyz)
    assert quality["tcp_detour_ratio"] == pytest.approx(1)
    assert not limits.exceeded(quality)
    # A short near-zero-displacement motion is not rejected on ratio alone.
    assert not limits.exceeded(sampled_path_motion_quality(q, np.array([[0, 0, 0], [.01, 0, 0], [0, 0, 0]])))
    q[1:, 0] = np.deg2rad(219.2)
    assert limits.exceeded(sampled_path_motion_quality(q, xyz))
    q[:] = 0
    assert limits.exceeded(sampled_path_motion_quality(q, np.array([[0, 0, 0], [1, 0, 0], [.1, 0, 0]])))
    start, end = {"right": np.zeros(6)}, {"right": np.ones(6)}

    def pose(pair):
        matrix = np.eye(4)
        matrix[0, 3] = pair["right"][0]
        return matrix

    assert tcp_path_edge_cost(start, end, side="right", pose_for_pair=pose, step_rad=.1) == pytest.approx(1)


@pytest.mark.parametrize("q,xyz", (
    (np.zeros((0, 6)), np.zeros((0, 3))),
    (np.zeros((2, 6)), np.zeros((3, 3))),
    (np.full((2, 6), np.nan), np.zeros((2, 3))),
    (np.zeros((2, 6)), np.full((2, 3), np.inf)),
))
def test_shared_motion_quality_rejects_corrupt_samples(q, xyz):
    from robot_motion.path_planning.motion_quality import sampled_path_motion_quality

    with pytest.raises(ValueError):
        sampled_path_motion_quality(q, xyz)


def test_endpoint_joint_travel_bound_uses_same_unwrapped_proximal_limit():
    from robot_motion.path_planning.motion_quality import PathMotionQualityLimits
    limits = PathMotionQualityLimits()
    start, end = np.zeros(6), np.zeros(6)
    end[0] = np.pi
    assert not limits.endpoint_joint_travel_exceeded(start, end)
    end[0] += 1e-6
    assert limits.endpoint_joint_travel_exceeded(start, end)
    start[0], end[0] = np.deg2rad([-170., 170.])
    assert limits.endpoint_joint_travel_exceeded(start, end)
    start[:] = 0.; end[:] = 0.; end[4] = 2 * np.pi
    assert not limits.endpoint_joint_travel_exceeded(start, end)
    end[:] = 0.; end[1] = np.deg2rad(120.)
    assert not limits.endpoint_joint_travel_exceeded(start, end)
    assert PathMotionQualityLimits(maximum_proximal_joint_travel_rad=np.deg2rad(100.)).endpoint_joint_travel_exceeded(start, end)


def test_endpoint_rejection_implies_existing_final_quality_rejection():
    from robot_motion.path_planning.motion_quality import PathMotionQualityLimits, sampled_path_motion_quality
    limits = PathMotionQualityLimits()
    rng = np.random.default_rng(19)
    for _ in range(64):
        path = rng.uniform(-3., 3., size=(10, 6))
        if limits.endpoint_joint_travel_exceeded(path[0], path[-1]):
            assert limits.exceeded(sampled_path_motion_quality(path, np.zeros((10, 3))))


@pytest.mark.parametrize("start,end", [(np.zeros(2),np.zeros(2)),(np.zeros(6),np.zeros(5)),
                                        (np.zeros(6),np.full(6,np.nan))])
def test_endpoint_quality_corruption_is_not_a_valid_rejection(start,end):
    from robot_motion.path_planning.motion_quality import PathMotionQualityLimits
    with pytest.raises(ValueError):PathMotionQualityLimits().endpoint_joint_travel_exceeded(start,end)

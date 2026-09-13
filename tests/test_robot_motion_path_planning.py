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

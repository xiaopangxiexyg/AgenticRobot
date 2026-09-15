import numpy as np
import pytest

from robot_motion.grasp_visible_scene import ParallelJawSolidGeometry, VisibleGraspScene
from robot_motion.path_planning import JointConfigurationCheck
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import _VisibleGripperPathOracle


def test_visible_gripper_oracle_checks_target_without_replacing_arm_safety():
    class ProductionOracle:
        scene_hash = "production:unchanged"
        def check(self, qpos):
            return JointConfigurationCheck(safe=True, collision_check_available=True)
    class Builder:
        def tool_tcp_world_targets_for_qpos_pair(self, **_):
            return {side: np.eye(4) for side in ("left", "right")}
    scene = VisibleGraspScene(points=np.array([[0.04, 0, -0.02]]), target_mask=np.array([True]),
                             geometry=ParallelJawSolidGeometry(-0.043, 0.05, 0.01, 0.0255, 0.07))
    oracle = _VisibleGripperPathOracle(oracle=ProductionOracle(), visible_scene=scene,
                                      target_builder=Builder(), side="right")
    check = oracle.check({side: np.zeros(6) for side in ("left", "right")})
    assert not check.safe and check.reason == "visible_gripper_path_collision"
    assert check.metadata["target_collision_points"] == 1
    assert oracle.scene_hash != "production:unchanged"


@pytest.mark.parametrize("available", [False, True])
def test_production_unavailable_or_collision_is_never_overridden_by_visible_scene(available):
    rejected = JointConfigurationCheck(safe=False, collision_check_available=available,
                                      reason="production_collision" if available else "production_unavailable")
    class Oracle:
        scene_hash = "production:unchanged"
        def check(self, _):
            return rejected
    class MustNotRead:
        def tool_tcp_world_targets_for_qpos_pair(self, **_):
            raise AssertionError("visible points must not bypass production refusal")
    scene = VisibleGraspScene(points=np.array([[.04, 0, -.02]]), target_mask=np.array([False]),
                             geometry=ParallelJawSolidGeometry(-0.043, 0.05, 0.01, 0.0255, 0.07))
    scene, removed = scene.without_captured_gripper_observations({"right": np.eye(4)})
    assert removed == 1
    oracle = _VisibleGripperPathOracle(oracle=Oracle(), visible_scene=scene,
                                      target_builder=MustNotRead(), side="right")
    assert oracle.check({}) is rejected


@pytest.mark.parametrize("target_obstacle", [False, True])
def test_capture_self_filter_clears_start_but_keeps_later_path_obstacles(target_obstacle):
    class Oracle:
        scene_hash = "production:unchanged"
        def check(self, _):
            return JointConfigurationCheck(safe=True, collision_check_available=True)
    class Builder:
        def tool_tcp_world_targets_for_qpos_pair(self, *, left_qpos, right_qpos):
            pose = np.eye(4)
            pose[0, 3] = right_qpos[0]
            return {"left": np.eye(4), "right": pose}
    scene = VisibleGraspScene(points=np.array([[.04, 0, -.02], [.24, 0, -.02]]),
                             target_mask=np.array([False, target_obstacle]),
                             geometry=ParallelJawSolidGeometry(-.043, .05, .01, .0255, .07))
    start = {side: np.zeros(6) for side in ("left", "right")}
    builder = Builder()
    assert scene.collision_counts(np.eye(4), opening_m=.07) == (0, 1)
    filtered, removed = scene.without_captured_gripper_observations(
        builder.tool_tcp_world_targets_for_qpos_pair(left_qpos=start["left"], right_qpos=start["right"])
    )
    assert removed == 1
    oracle = _VisibleGripperPathOracle(oracle=Oracle(), visible_scene=filtered,
                                      target_builder=builder, side="right")
    assert oracle.check(start).safe
    later = {side: q.copy() for side, q in start.items()}
    later["right"][0] = .2
    check = oracle.check(later)
    assert not check.safe and check.reason == "visible_gripper_path_collision"
    assert check.metadata["target_collision_points"] == int(target_obstacle)
    assert check.metadata["background_collision_points"] == int(not target_obstacle)

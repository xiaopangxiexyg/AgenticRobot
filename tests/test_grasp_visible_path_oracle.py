import numpy as np

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


def test_production_unavailable_or_collision_is_never_overridden_by_visible_scene():
    rejected = JointConfigurationCheck(safe=False, collision_check_available=False,
                                      reason="production_unavailable")
    class Oracle:
        scene_hash = "production:unchanged"
        def check(self, _):
            return rejected
    class MustNotRead:
        def tool_tcp_world_targets_for_qpos_pair(self, **_):
            raise AssertionError("visible points must not bypass production refusal")
    scene = VisibleGraspScene(points=np.empty((0, 3)), target_mask=np.empty(0, dtype=bool),
                             geometry=ParallelJawSolidGeometry(-0.043, 0.05, 0.01, 0.0255, 0.07))
    oracle = _VisibleGripperPathOracle(oracle=Oracle(), visible_scene=scene,
                                      target_builder=MustNotRead(), side="right")
    assert oracle.check({}) is rejected

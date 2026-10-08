from dataclasses import replace

import numpy as np
import pytest

from robot_motion.object_transport import GraspedObjectObservation
from robot_motion.scene_obstacles import AxisAlignedBoxObstacle
from teleop_runtime.skill_deploy.rm65_object_transport import RM65ObjectTransportPlanner
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import RM65GraspTrajectoryPlanningConfig, RM65GraspTrajectoryPlanningError
from teleop_runtime.skill_deploy.rm65_lower_assembly import build_exclusive_rm65_skill_target_builder
from test_rm65_grasp_trajectory import _real_profile, SAFE_QPOS
from teleop_runtime.skill_deploy.rm65_profile import ExclusiveRM65SkillDeploymentProfile, resolve_exclusive_rm65_skill_deployment_profile

pytestmark = [pytest.mark.integration]


def planner_and_observation(side, *, dmtac=False):
    # The existing fixture binds the same reviewed tool calibration as grasp
    # planning. No hardware resource is started by these pure constructors.
    profile = _real_profile(envelope_profile_name=f"agent_grasp_{side}_absolute_standard_v1.json")
    if dmtac:
        profile = resolve_exclusive_rm65_skill_deployment_profile(ExclusiveRM65SkillDeploymentProfile(
            arm_ip_assignment_path=profile.arm_ip_assignment_path,
            tool_profile_path_by_side=profile.tool_profile_path_by_side,
            envelope_profile_path=profile.envelope_profile_path,
            collision_scene_profile="dual_arm_hand_proxy_dmtac_strict"))
    builder = build_exclusive_rm65_skill_target_builder(profile)
    planner = RM65ObjectTransportPlanner(profile=profile, planning_target_builder=builder,
        target_envelope_filter=builder.pre_ik_target_safety_filter,
        config=RM65GraspTrajectoryPlanningConfig(path_planning_time_budget_sec=20))
    tcp = builder.tool_tcp_world_targets_for_qpos_pair(left_qpos=SAFE_QPOS["left"], right_qpos=SAFE_QPOS["right"])[side]
    object_pose = tcp.copy()
    held = GraspedObjectObservation(object_id="test-cube", side=side, reference_frame_id=planner.scene.reference_frame_id,
        scene_revision=planner.scene.revision, measured_at=10., reference_from_object=object_pose,
        reference_from_tool=tcp, qpos_by_side=SAFE_QPOS, box_size_m=(.04, .04, .04), object_from_box=np.eye(4),
        geometry_complete=True, pose_evidence_ref="simulator:pose", grasp_evidence_ref="simulator:contact-follow")
    target_q = {s: q.copy() for s, q in SAFE_QPOS.items()}
    target_q[side][0] += .04
    goal = builder.tool_tcp_world_targets_for_qpos_pair(left_qpos=target_q["left"], right_qpos=target_q["right"])[side]
    return planner, held, goal


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("dmtac", [False, True])
def test_native_rm65_carried_geometry_path_time_parameterization_and_other_arm(side, dmtac):
    planner, held, goal = planner_and_observation(side, dmtac=dmtac)
    original_scene = planner.scene
    plan = planner.plan_object_move(observation=held, object_goal=goal, world_scene=original_scene, speed_scale=.2)
    assert plan.details["object_position_error_m"] <= .003
    assert plan.details["object_orientation_error_deg"] <= 1.
    assert plan.details["dense_revalidation"]["sample_count"] > 0
    other = "left" if side == "right" else "right"
    np.testing.assert_allclose(plan.trajectory.qpos_by_side[other], np.broadcast_to(SAFE_QPOS[other], plan.trajectory.qpos_by_side[other].shape), atol=1e-12)
    assert all(not x.startswith(side+"_hand_proxy_") for x in plan.details["carried_collision_pairs"])
    assert other+"_hand_proxy_0" in plan.details["carried_collision_pairs"]
    assert plan.details["execution_authorized"] is False
    assert len(original_scene.obstacles) == 4
    with pytest.raises(ValueError, match="request-local"):
        planner.plan_object_move(observation=held, object_goal=goal, world_scene=original_scene, speed_scale=.2)


def test_empty_hand_clear_does_not_make_payload_clear():
    planner, held, goal = planner_and_observation("right")
    # Put an identified obstacle at the held object's offset bounding geometry,
    # away from the hand proxies. The start must fail before any path is given.
    offset = np.eye(4)
    offset[:3, 3] = (0, 0, .15)
    held = replace(held, object_from_box=offset)
    center = (held.reference_from_object @ offset)[:3, 3]
    world = replace(planner.scene, obstacles=(*planner.scene.obstacles,
        AxisAlignedBoxObstacle("neighbour", tuple(center), (.025, .025, .025)),
        AxisAlignedBoxObstacle(held.object_id, tuple(held.reference_from_object[:3, 3]), (.04, .04, .04))))
    with pytest.raises(RM65GraspTrajectoryPlanningError):
        planner.plan_object_move(observation=held, object_goal=goal, world_scene=world, speed_scale=.2)
    check = planner.oracle.check(held.qpos_by_side)
    assert not check.safe and check.collision_check_available
    names = check.metadata["closest_geometry_names"]
    assert "held_object::test-cube" in names
    assert {o.obstacle_id for o in planner.scene.obstacles} == {o.obstacle_id for o in world.obstacles}-{held.object_id}


def test_wrong_fk_and_changed_fixture_cannot_be_planned():
    planner, held, goal = planner_and_observation("right")
    wrong = held.reference_from_tool.copy()
    wrong[0, 3] += .01
    with pytest.raises(ValueError, match="FK/tool"):
        planner.plan_object_move(observation=replace(held, reference_from_tool=wrong), object_goal=goal,
                                 world_scene=planner.scene, speed_scale=.2)

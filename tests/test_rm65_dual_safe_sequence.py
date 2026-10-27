"""Native dual-arm planning and MuJoCo playback; no hardware or grasp evidence."""

from dataclasses import replace

import numpy as np
import pytest
from test_mujoco_dmtac_gripper import _timing
from test_rm65_grasp_trajectory import SAFE_QPOS, _real_profile

from robot_motion.pose import pose6_to_matrix, pose_delta_metrics
from robot_motion.runtime import EffectorCommand, TeleopCommand
from sim.contracts import SimulationContract
from sim.mujoco.actuation import (
    cad_bimanual_dmtac_gripper_profile,
    rm65_bimanual_udp_feedback_profile,
)
from sim.mujoco.adapters import (
    MujocoPositionExecutionConfig,
    MujocoPositionExecutionRig,
)
from sim.mujoco.artifacts import ModelArtifactCache
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import (
    RM65GraspJointTrajectoryPlanner,
    RM65GraspTrajectoryPlanningConfig,
)
from teleop_runtime.skill_deploy.rm65_lower_assembly import (
    build_exclusive_rm65_skill_target_builder,
)

pytestmark = [pytest.mark.integration]


def test_native_left_up_right_up_then_sequential_safe_return(record_property):
    profile = _real_profile(envelope_profile_name="agent_grasp_dual_absolute_standard_v1.json",
        collision_scene_profile="dual_arm_hand_proxy_dmtac_grasp")
    builder = build_exclusive_rm65_skill_target_builder(profile)
    planner = RM65GraspJointTrajectoryPlanner(profile=profile, planning_target_builder=builder,
        target_envelope_filter=builder.pre_ik_target_safety_filter,
        config=RM65GraspTrajectoryPlanningConfig(path_planning_time_budget_sec=20))
    current = {arm: q.copy() for arm, q in SAFE_QPOS.items()}
    initial = builder.tool_tcp_world_targets_for_qpos_pair(left_qpos=current["left"], right_qpos=current["right"])
    durations, planned = [], []
    for arm, up in (("left", True), ("right", True), ("right", False), ("left", False)):
        target = initial[arm].copy()
        if up:
            target[2, 3] += .03
        other = "left" if arm == "right" else "right"
        trajectory = planner(qpos_by_side=current, side=arm, target_tool_tcp_world=target,
            speed_scale=.2, require_grasp_approach=False)
        np.testing.assert_allclose(trajectory.qpos_by_side[other],
            np.broadcast_to(current[other], trajectory.qpos_by_side[other].shape), atol=1e-12)
        current = {side: trajectory.qpos_by_side[side][-1].copy() for side in ("left", "right")}
        measured = builder.tool_tcp_world_targets_for_qpos_pair(left_qpos=current["left"], right_qpos=current["right"])
        distance, angle = pose_delta_metrics(measured[arm], target)
        assert distance <= .003 and angle <= 1.
        assert planner.oracle.check(current).safe
        durations.append(float(trajectory.time_from_start_sec[-1]))
        planned.append((arm, target, trajectory))
    for arm in ("left", "right"):
        distance, angle = pose_delta_metrics(measured[arm], initial[arm])
        assert distance <= .003 and angle <= 1.
    record_property("leg_duration_sec", durations)

    contract = SimulationContract.load("sim/configs/rm65_bimanual_physics_v1.json")
    contract = replace(contract, scene=replace(contract.scene,
        urdf_path="assets/realman_dual_arm_inspire_gripper_dmtacm/realman_dual_arm_inspire_gripper_dmtacm.urdf",
        camera_registry_path="robot_motion/configs/rm65_camera_registry_v4.json"))
    arms, hands = rm65_bimanual_udp_feedback_profile(), cad_bimanual_dmtac_gripper_profile()
    scene, _ = ModelArtifactCache().load_scene(contract,
        position_actuation_profile=arms, gripper_actuation_profile=hands)
    scene.set_qpos_by_side(SAFE_QPOS)
    def tcp_from_scene(arm):
        return scene.body_pose(scene.arm_specs[arm].controlled_body) @ pose6_to_matrix(
            profile.tool_tcp_pose_flange_m_rad_by_side[arm])
    for arm in ("left", "right"):
        np.testing.assert_allclose(tcp_from_scene(arm), initial[arm], atol=1e-7)
    rig = MujocoPositionExecutionRig(scene, MujocoPositionExecutionConfig(
        timing=_timing(), actuation_profile=arms, gripper_actuation_profile=hands,
        realtime_worker=False, follow_qpos_alpha=1.0))
    rig.start()
    try:
        for arm, target, trajectory in planned:
            other = "left" if arm == "right" else "right"
            held_pose = tcp_from_scene(other)
            started = float(scene.data.time)
            for index, stamp in enumerate(trajectory.time_from_start_sec):
                command = TeleopCommand(timestamp=started+float(stamp))
                for side in ("left", "right"):
                    command.add_command(EffectorCommand(effector_name=side+"_arm", side=side,
                        command_type="joint_vector", value=trajectory.qpos_by_side[side][index]))
                assert all(receipt.accepted for receipt in rig.send_commands(command).values())
                next_stamp = (float(trajectory.time_from_start_sec[index+1])
                    if index+1 < trajectory.time_from_start_sec.size else float(stamp)+1.)
                while float(scene.data.time) < started+next_stamp:
                    rig.step_physics()
            distance, angle = pose_delta_metrics(tcp_from_scene(arm), target)
            assert distance <= .003 and angle <= 1., (arm, distance, angle)
            distance, angle = pose_delta_metrics(tcp_from_scene(other), held_pose)
            assert distance <= .003 and angle <= 1., ("inactive_arm_drift", other, distance, angle)
        record_property("mujoco_simulated_sec", float(scene.data.time))
    finally:
        rig.close()

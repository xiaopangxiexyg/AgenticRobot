"""Physical parallel-jaw fixture, not an RM65 hardware/grasp acceptance test.

The cube is free (no weld/attachment constraint). Simulator truth supplies pose;
the existing path sampler and C2 retimer drive a synthetic Cartesian gantry.
Native RM65 IK/proxy planning is covered separately in test_rm65_object_transport.
"""
import numpy as np
import pytest

from robot_motion.object_transport import GraspedObjectObservation, verify_rigid_follow
from robot_motion.path_planning import JointConfigurationCheck, JointPathPlannerConfig, plan_collision_aware_joint_path
from robot_motion.position_control.horizon_governor import JointHorizonLimits
from robot_motion.position_control.offline_waypoint_trajectory import build_offline_quintic_waypoint_trajectory

pytestmark = [pytest.mark.integration]

MODEL = """<mujoco>
  <option timestep="0.002" integrator="implicitfast" iterations="100" noslip_iterations="10"/>
  <default><geom friction="2 0.01 0.001" condim="4" solref="0.01 1"/></default>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 .02" rgba=".4 .4 .4 1"/>
    <geom name="wall" type="box" pos=".15 0 .15" size=".015 .08 .15"/>
    <body name="cube" pos="0 0 .025"><freejoint/>
      <geom name="object" type="box" size=".025 .025 .025" mass=".06" rgba="0 .2 1 1"/>
    </body>
    <body name="driver" mocap="true" pos="0 0 .135"/>
    <body name="hand" pos="0 0 .135"><freejoint/>
      <geom name="palm" type="box" size=".07 .02 .015"/>
      <body name="finger_l"><joint name="close_l" type="slide" axis="1 0 0" range="0 .045" damping="1"/>
        <geom name="finger_l_geom" type="box" pos="-.06 0 -.085" size=".008 .015 .035" mass=".03"/>
      </body>
      <body name="finger_r"><joint name="close_r" type="slide" axis="-1 0 0" range="0 .045" damping="1"/>
        <geom name="finger_r_geom" type="box" pos=".06 0 -.085" size=".008 .015 .035" mass=".03"/>
      </body>
    </body>
  </worldbody>
  <equality><weld body1="driver" body2="hand" solref=".004 1"/></equality>
  <actuator><position joint="close_l" kp="500" ctrlrange="0 .045" forcerange="-10 10"/>
    <position joint="close_r" kp="500" ctrlrange="0 .045" forcerange="-10 10"/></actuator>
</mujoco>"""


def test_physical_grasp_relation_object_goal_and_collision_aware_transport():
    import mujoco
    import hppfcl as fcl

    model = mujoco.MjModel.from_xml_string(MODEL)
    data = mujoco.MjData(model)
    object_geom = model.geom("object").id
    wall = model.geom("wall").id
    finger_geoms = {model.geom(name).id for name in ("finger_l_geom", "finger_r_geom")}
    moving_geoms = [object_geom, model.geom("palm").id, *finger_geoms]
    shapes = {g: fcl.Box(*(2*model.geom_size[g])) for g in [wall, *moving_geoms]}

    def clearance(state):
        # Use the same distance backend as the real RM65 planner. MuJoCo owns
        # dynamics/contacts; FCL owns conservative geometric distance queries.
        world = fcl.Transform3f(state.geom_xmat[wall].reshape(3, 3), state.geom_xpos[wall])
        return min(fcl.distance(shapes[g], fcl.Transform3f(state.geom_xmat[g].reshape(3, 3), state.geom_xpos[g]),
            shapes[wall], world, fcl.DistanceRequest(), fcl.DistanceResult()) for g in moving_geoms)

    def contacts():
        return {int(c.geom1 if c.geom2 == object_geom else c.geom2) for c in data.contact
                if object_geom in (c.geom1, c.geom2)}

    def observe():
        poses = {}
        for name in ("cube", "hand"):
            body = model.body(name).id
            transform = np.eye(4)
            transform[:3, :3] = data.xmat[body].reshape(3, 3)
            transform[:3, 3] = data.xpos[body]
            poses[name] = transform
        return GraspedObjectObservation(object_id="blue-cube", side="right", reference_frame_id="sim-world",
            scene_revision=1, measured_at=float(data.time), reference_from_object=poses["cube"],
            reference_from_tool=poses["hand"], qpos_by_side={s: np.zeros(6) for s in ("left", "right")},
            box_size_m=(.05, .05, .05), object_from_box=np.eye(4), geometry_complete=True,
            pose_evidence_ref="mujoco:body-pose", grasp_evidence_ref="mujoco:bilateral-contact")

    # Establish two physical contacts. Closing success alone is not acceptance.
    data.ctrl[:] = .04
    for _ in range(500):
        mujoco.mj_step(model, data)
    assert finger_geoms <= contacts()
    before = observe()
    for index in range(1, 501):
        u = index / 500
        data.mocap_pos[0, 2] = .135 + .08*(10*u**3-15*u**4+6*u**5)
        mujoco.mj_step(model, data)
    assert finger_geoms <= contacts()
    held = observe()
    assert held.reference_from_object[2, 3]-before.reference_from_object[2, 3] > .075
    follow = verify_rigid_follow(before, held)

    goal = held.reference_from_object.copy()
    goal[0, 3] += .30
    tool_goal = held.tool_goal(goal)
    planning_data = mujoco.MjData(model)
    planning_data.qpos[:] = data.qpos
    planning_data.mocap_pos[:] = data.mocap_pos
    object_qpos = int(model.jnt_qposadr[model.body("cube").jntadr[0]])
    hand_qpos = int(model.jnt_qposadr[model.body("hand").jntadr[0]])

    class Oracle:
        def check(self, qpos_by_component):
            tcp = held.reference_from_tool.copy()
            tcp[:2, 3] = qpos_by_component["gantry_xy"]
            planning_data.mocap_pos[0] = tcp[:3, 3]
            planning_data.qpos[hand_qpos:hand_qpos+3] = tcp[:3, 3]
            mujoco.mju_mat2Quat(planning_data.qpos[hand_qpos+3:hand_qpos+7], tcp[:3, :3].reshape(-1))
            obj = tcp @ held.tool_from_object
            planning_data.qpos[object_qpos:object_qpos+3] = obj[:3, 3]
            mujoco.mju_mat2Quat(planning_data.qpos[object_qpos+3:object_qpos+7], obj[:3, :3].reshape(-1))
            mujoco.mj_forward(model, planning_data)
            distance = clearance(planning_data)
            return JointConfigurationCheck(safe=distance >= .02, minimum_distance_m=distance)

    oracle = Oracle()
    assert not oracle.check({"gantry_xy": np.array([.15, 0])}).safe
    planned = plan_collision_aware_joint_path(
        start_qpos_by_component={"gantry_xy": held.reference_from_tool[:2, 3]},
        target_qpos_by_component={"gantry_xy": tool_goal[:2, 3]}, movable_components=("gantry_xy",),
        oracle=oracle, joint_lower_by_component={"gantry_xy": [-.05, -.25]},
        joint_upper_by_component={"gantry_xy": [.35, .25]},
        config=JointPathPlannerConfig(seed=3, extension_step_rad=.035, edge_resolution_rad=.003,
            sample_padding_rad=.2, time_budget_sec=5))
    assert planned.status.value == "path_found", (dict(planned.metadata),
        [(xy, oracle.check({"gantry_xy": xy}).minimum_distance_m) for xy in ([0, 0], [0, .18], [.15, .18], [.3, .18], [.3, 0])])
    points = np.array([w["gantry_xy"] for w in planned.waypoints])
    durations = np.maximum(np.linalg.norm(np.diff(points, axis=0), axis=1)/.04, .8)
    limits = JointHorizonLimits(min_position_rad=np.array([-.05, -.25]), max_position_rad=np.array([.35, .25]),
        max_velocity_rad_s=np.full(2, .10), max_acceleration_rad_s2=np.full(2, .5), max_jerk_rad_s3=np.full(2, 5.))
    trajectory = build_offline_quintic_waypoint_trajectory(np.r_[0., np.cumsum(durations)], points,
        command_hz=500, limits=limits, timing_mode="safe_retime_global")
    for xy in trajectory.q_rad:
        assert oracle.check({"gantry_xy": xy}).safe
    min_clearance = 1.
    for xy in trajectory.q_rad:
        data.mocap_pos[0, :2] = xy
        mujoco.mj_step(model, data)
        min_clearance = min(min_clearance, clearance(data))
        assert all(wall not in (contact.geom1, contact.geom2) for contact in data.contact)
    after = observe()
    assert finger_geoms <= contacts()
    verify_rigid_follow(held, after)
    error = np.linalg.norm(after.reference_from_object[:3, 3]-goal[:3, 3])
    assert error < .003
    assert min_clearance >= .015
    print({"fixture": "synthetic_parallel_jaw_not_rm65", "object_welded": False,
           "lift_m": follow["tool_displacement_m"], "carry_duration_sec": float(trajectory.timestamps[-1]),
           "object_goal_error_m": float(error), "minimum_wall_clearance_m": float(min_clearance)})

"""The held-object adapter cannot manufacture a grasp from a static scene."""

from pathlib import Path

import numpy as np
import pytest


def test_complete_box_geometry_rejects_rigid_child_and_visual_additions():
    mj = pytest.importorskip("mujoco")
    from agent_gateway.profiles.mujoco.held_box import _complete_box_geom

    def model(extra):
        return mj.MjModel.from_xml_string(
            "<mujoco><worldbody><body name='held'>"
            "<freejoint/><geom name='box' type='box' size='.02 .02 .02'/>"
            + extra + "</body></worldbody></mujoco>"
        )

    simple = model("")
    assert _complete_box_geom(simple, mj, simple.body("held").id) == simple.geom("box").id
    duplicate = model("<geom name='same_visual' type='box' size='.02 .02 .02' "
                      "contype='0' conaffinity='0'/>")
    assert (_complete_box_geom(duplicate, mj, duplicate.body("held").id)
            == duplicate.geom("box").id)
    with_child = model("<body name='rigid_tip' pos='.03 0 0'>"
                       "<geom name='tip' type='sphere' size='.005'/></body>")
    with pytest.raises(ValueError, match="one complete target geometry"):
        _complete_box_geom(with_child, mj, with_child.body("held").id)
    with_visual = model("<geom name='visual_tip' type='sphere' size='.005' "
                        "contype='0' conaffinity='0'/>")
    with pytest.raises(ValueError, match="unmatched additional shape"):
        _complete_box_geom(with_visual, mj, with_visual.body("held").id)


@pytest.mark.integration
@pytest.mark.optional
@pytest.mark.slow_sim
def test_held_box_port_rejects_unverified_native_object_without_stepping():
    pytest.importorskip("mujoco")
    from agent_gateway.profiles.mujoco.held_box import MujocoHeldBoxObservationPort
    from agent_gateway.profiles.mujoco.motion_factory import (
        build_mujoco_agent_tcp_pose_gateway_session,
        mujoco_agent_tcp_pose_execution_observation_plan_v1,
    )
    from sim.mujoco.deploy import MujocoSimHILTargetConfig

    task = (Path(__file__).resolve().parents[1]
            / "sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json")
    session = build_mujoco_agent_tcp_pose_gateway_session(
        target_config=MujocoSimHILTargetConfig(
            task_config_path=task, realtime_worker=False,
            runtime_session_id="test:held-box-port",
            control_domain_id="test:held-box-port",
            execution_observation_plan=mujoco_agent_tcp_pose_execution_observation_plan_v1(),
            exposed_pose_sides=("right",), allow_scripted_task_reset=False,
        ),
        allowed_client_ids=("test:held-box-port",),
        enable_grippers=True, enable_grasp_approach=True,
        grasp_contact_target_body="debug_cube",
        enable_object_state=True, enable_object_follow_observation=True,
        enable_object_motion=True,
    )
    try:
        app = session.application
        app.start_observation_ready()
        qpos = app.scene.data.qpos.copy()
        qvel = app.scene.data.qvel.copy()
        sim_time = float(app.scene.data.time)
        port = MujocoHeldBoxObservationPort(
            application=app, observer=session.object_observer,
            object_id="debug_cube", side="right",
            reference_frame_id="mujoco_sim_hil/world",
            current_station_revision=lambda: 0,
        )
        with pytest.raises(ValueError, match="identity or side"):
            MujocoHeldBoxObservationPort(
                application=app, observer=session.object_observer,
                object_id="debug_cube", side="left",
                reference_frame_id="mujoco_sim_hil/world",
                current_station_revision=lambda: 0,
            )
        original_tcp = session.object_observer.world_tcp_from_scene
        def wrong_tcp(scene):
            result = original_tcp(scene).copy()
            result[0, 3] += .01
            return result
        session.object_observer.world_tcp_from_scene = wrong_tcp
        with pytest.raises(ValueError, match="TCP differs"):
            port.capture_held_box(expected_station_revision=0)
        session.object_observer.world_tcp_from_scene = original_tcp
        with pytest.raises(ValueError, match="closed|grasp|lift"):
            port.capture_held_box(expected_station_revision=0)
        assert np.array_equal(qpos, app.scene.data.qpos)
        assert np.array_equal(qvel, app.scene.data.qvel)
        assert sim_time == float(app.scene.data.time)
        closed = session.close()
        assert closed.accepted and app.closed
        with pytest.raises(RuntimeError, match="closed"):
            port.capture_held_box(expected_station_revision=0)
    finally:
        if not session.application.closed:
            closed = session.close()
            assert closed.accepted and session.application.closed

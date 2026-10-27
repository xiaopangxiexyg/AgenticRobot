"""A fixed support borrows the native rig without stepping or another owner."""

from pathlib import Path

import numpy as np
import pytest


@pytest.mark.integration
@pytest.mark.optional
@pytest.mark.slow_sim
def test_fixed_box_support_reads_native_model_with_same_depth_cut_and_closes():
    pytest.importorskip("mujoco")
    from agent_gateway.profiles.mujoco.back_projection import MujocoBackProjectionCapturePort
    from agent_gateway.profiles.mujoco.motion_factory import (
        build_mujoco_agent_tcp_pose_gateway_session,
        mujoco_agent_tcp_pose_execution_observation_plan_v1,
    )
    from agent_gateway.profiles.mujoco.support_box import MujocoFixedBoxSupportPort
    from sim.mujoco.deploy import MujocoSimHILTargetConfig

    root = Path(__file__).resolve().parents[1]
    task = root / "sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json"
    session = build_mujoco_agent_tcp_pose_gateway_session(
        target_config=MujocoSimHILTargetConfig(
            task_config_path=task, realtime_worker=False,
            runtime_session_id="test:fixed-box-support",
            control_domain_id="test:fixed-box-support",
            execution_observation_plan=mujoco_agent_tcp_pose_execution_observation_plan_v1(),
            exposed_pose_sides=("right",), allow_scripted_task_reset=False,
        ),
        allowed_client_ids=("test:fixed-box-support",),
    )
    try:
        app = session.application
        app.start_observation_ready()
        before_qpos = app.scene.data.qpos.copy()
        before_qvel = app.scene.data.qvel.copy()
        before_time = float(app.scene.data.time)
        depth_port = MujocoBackProjectionCapturePort(
            application=app, source_id_by_role={"main": "front"},
            source_frame_id="mujoco_sim_hil/world",
            reference_frame_id="mujoco_sim_hil/world",
            current_station_revision=lambda: 0,
        )
        port = MujocoFixedBoxSupportPort(
            application=app, depth_capture_port=depth_port,
            allowed_support_body_names=(app.task_config.table_body_name,
                                        app.task_config.object_body_name),
            reference_frame_id="mujoco_sim_hil/world",
            depth_role="main",
            current_station_revision=lambda: 0,
        )
        paired, observed = port.capture_fixed_support(
            support_id=app.task_config.table_body_name,
            expected_station_revision=0)
        depth = paired.depth
        assert paired.rgb.source_id == depth.source_id == "front"
        assert paired.rgb.station_revision == depth.station_revision == 0
        assert paired.rgb_jpeg is not None and paired.depth_npy is not None
        assert observed.geometry_complete is True
        assert observed.measured_at == depth.captured_at
        assert observed.scene_revision == depth.station_revision
        assert all(value > 0 for value in observed.box_size_m)
        assert not observed.reference_from_box.flags.writeable
        with pytest.raises(ValueError, match="Moving support"):
            port.capture_fixed_support(
                support_id=app.task_config.object_body_name,
                expected_station_revision=0)
        model = app.scene.model
        moving_child = None
        for body in range(1, model.nbody):
            if int(model.body_jntnum[body]) != 0:
                continue
            ancestor = int(model.body_parentid[body])
            while ancestor:
                if int(model.body_jntnum[ancestor]) != 0:
                    moving_child = model.body(body).name
                    break
                ancestor = int(model.body_parentid[ancestor])
            if moving_child is not None:
                break
        assert moving_child is not None
        nested_port = MujocoFixedBoxSupportPort(
            application=app, depth_capture_port=depth_port,
            allowed_support_body_names=(moving_child,),
            reference_frame_id="mujoco_sim_hil/world",
            depth_role="main",
            current_station_revision=lambda: 0,
        )
        with pytest.raises(ValueError, match="Moving support"):
            nested_port.capture_fixed_support(support_id=moving_child,
                                              expected_station_revision=0)
        with pytest.raises(ValueError, match="Station revision"):
            port.capture_fixed_support(
                support_id=app.task_config.table_body_name,
                expected_station_revision=1)
        assert np.array_equal(before_qpos, app.scene.data.qpos)
        assert np.array_equal(before_qvel, app.scene.data.qvel)
        assert before_time == float(app.scene.data.time)
        closed = session.close()
        assert closed.accepted and app.closed
        with pytest.raises(RuntimeError, match="closed"):
            port.capture_fixed_support(
                support_id=app.task_config.table_body_name,
                expected_station_revision=0)
    finally:
        if not session.application.closed:
            closed = session.close()
            assert closed.accepted and session.application.closed
